"""Prediction service for single and batch predictions."""
import datetime
import logging
import os
import uuid
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.utils.file_utils import ensure_dir, load_dataframe

logger = logging.getLogger(__name__)


class InputValidationError(ValueError):
    """Raised when user-supplied prediction inputs do not satisfy the schema.

    Carries machine-readable field lists plus one human-readable message so the
    API layer can answer with a single, clear 4xx response.
    """

    def __init__(self, missing_fields=None, invalid_fields=None):
        self.missing_fields = list(missing_fields or [])
        self.invalid_fields = dict(invalid_fields or {})
        parts = []
        if self.missing_fields:
            parts.append("missing required input(s): " + ", ".join(self.missing_fields))
        if self.invalid_fields:
            parts.append("invalid input(s): " + "; ".join(
                f"{name} ({reason})" for name, reason in self.invalid_fields.items()))
        self.message = (
            "Prediction input validation failed: " + " | ".join(parts)
            if parts else "Prediction input validation failed"
        )
        super().__init__(self.message)


def _is_unsupplied(value) -> bool:
    """True when a field was left out, nulled or sent as blank text."""
    if value is None or value is pd.NaT:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def validate_prediction_inputs(input_dict: dict, schema_fields: list) -> dict:
    """Check user inputs against the experiment's input schema.

    Rules:
      * every schema field marked ``required`` must actually be supplied
        (absent keys, ``null`` and blank strings all count as not supplied);
      * ``number`` fields must carry a finite number, either typed or as a
        numeric string;
      * values must be scalars;
      * fields not present in the schema (columns the pipeline drops or
        re-derives) may be omitted freely and are passed through untouched.

    Returns a copy of the inputs with numeric strings normalised to numbers so
    the fitted pipeline sees the value the user meant. Nothing is invented for a
    field the user did not fill in — that is the caller's bug to report.

    Raises:
        InputValidationError: if any required field is missing or invalid.
    """
    if not isinstance(input_dict, dict):
        raise InputValidationError(invalid_fields={"<body>": "expected a JSON object of field values"})

    if not schema_fields:
        logger.warning("No input schema available; prediction inputs cannot be validated.")
        return dict(input_dict)

    schema_names = {f.get("name") for f in schema_fields if f.get("name")}
    missing: list[str] = []
    invalid: dict[str, str] = {}
    cleaned: dict[str, Any] = {}

    for field in schema_fields:
        name = field.get("name")
        if not name:
            continue
        field_type = field.get("type") or "number"
        required = field.get("required", True) is not False
        value = input_dict.get(name)

        if name not in input_dict or _is_unsupplied(value):
            if required:
                missing.append(name)
            elif name in input_dict and value is not None:
                cleaned[name] = value
            continue

        if isinstance(value, (list, tuple, dict, set)):
            invalid[name] = "expected a single value, got a list/object"
            continue

        if field_type == "number":
            number = value
            if isinstance(value, str):
                try:
                    number = float(value.strip())
                except (TypeError, ValueError):
                    invalid[name] = f"expected a number, got '{value.strip()[:30]}'"
                    continue
            if isinstance(number, bool):
                number = float(number)
            if isinstance(number, float) and (np.isnan(number) or np.isinf(number)):
                invalid[name] = "expected a finite number"
                continue
            cleaned[name] = number
        else:
            cleaned[name] = value

    # Extra keys are not schema fields (e.g. columns the pipeline drops); leave
    # them for the fitted pipeline to ignore rather than inventing or refusing.
    for key, value in input_dict.items():
        if key not in schema_names:
            cleaned[key] = value

    if missing or invalid:
        raise InputValidationError(missing, invalid)
    return cleaned


class Predictor:
    """Handle single and batch predictions using trained models.

    Column reproduction (feature engineering + preprocessing) is owned by the
    fitted ``PreprocessingEngine`` passed in here, which guarantees the exact
    same feature matrix as training.
    """

    def __init__(self, model, preprocessor, feature_names: list[str],
                 target_column: str, problem_type: str):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = list(feature_names or [])
        self.target_column = target_column
        self.problem_type = problem_type

    # ───────────────────────────── single ─────────────────────────────
    def predict_single(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Make a prediction for a single input."""
        try:
            df = pd.DataFrame([dict(input_dict or {})])
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = np.nan

            X = self._transform(df)
            pred = self.model.predict(X)[0]
            result: dict[str, Any] = {"prediction": self._to_native(pred)}

            if self.problem_type == "classification" and hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(X)[0]
                    classes = self.model.classes_
                    result["probabilities"] = {
                        str(c): round(float(p), 4) for c, p in zip(classes, probs)
                    }
                    result["confidence"] = round(float(np.max(probs)), 4)
                except Exception as exc:
                    logger.warning(f"Probability output unavailable: {exc}")

            return result
        except Exception as e:
            logger.exception(f"Single prediction failed: {e}")
            return {"error": str(e)}

    # ───────────────────────────── batch ──────────────────────────────
    def predict_batch(self, input_path: str, storage_path: str = "storage") -> dict[str, Any]:
        """Make predictions for an uploaded CSV/XLSX file."""
        try:
            df = load_dataframe(input_path)
            input_df = df.copy()

            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = np.nan

            X = self._transform(df)
            preds = self.model.predict(X)
            input_df["prediction"] = [self._to_native(p) for p in preds]

            if self.problem_type == "classification" and hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(X)
                    input_df["confidence"] = [round(float(np.max(p)), 4) for p in probs]
                except Exception as exc:
                    logger.warning(f"Batch probability output unavailable: {exc}")

            out_dir = ensure_dir(os.path.join(storage_path, "predictions"))
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            unique = uuid.uuid4().hex[:8]
            out_path = str(out_dir / f"predictions_{ts}_{unique}.csv")
            input_df.to_csv(out_path, index=False)

            preview = [self._json_safe_record(r)
                       for r in input_df.head(10).to_dict(orient="records")]

            return {
                "predictions_path": out_path,
                "predictions_file": os.path.basename(out_path),
                "predictions_url": f"/predictions/{os.path.basename(out_path)}",
                "num_predictions": len(input_df),
                "preview": preview,
            }
        except Exception as e:
            logger.exception(f"Batch prediction failed: {e}")
            return {"error": str(e)}

    # ──────────────────────────── helpers ─────────────────────────────
    def _transform(self, df: pd.DataFrame):
        """Apply the fitted preprocessing (feature engineering included)."""
        if self.preprocessor is not None:
            return self.preprocessor.transform(df)
        return df.values

    def get_input_schema(self, original_df: Optional[pd.DataFrame] = None) -> list[dict[str, Any]]:
        """Generate input field schema for the prediction form."""
        schema = []
        for feat in self.feature_names:
            info: dict[str, Any] = {
                "name": feat,
                "type": "number",
                "required": True,
                "sample_values": [],
            }
            if original_df is not None and feat in original_df.columns:
                dtype = original_df[feat].dtype
                if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                    info["type"] = "text"
                elif pd.api.types.is_bool_dtype(dtype):
                    info["type"] = "boolean"
                try:
                    info["sample_values"] = [
                        self._to_native(v) for v in original_df[feat].dropna().unique()[:5].tolist()
                    ]
                except Exception:
                    pass
            schema.append(info)
        return schema

    @staticmethod
    def _to_native(val):
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return None if np.isnan(val) else float(val)
        if isinstance(val, (np.bool_,)):
            return bool(val)
        if isinstance(val, float) and np.isnan(val):
            return None
        return val

    @classmethod
    def _json_safe_record(cls, record: dict) -> dict:
        """Replace NaN/NaT with None so the batch preview stays valid JSON."""
        out = {}
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.bool_,)):
                value = bool(value)
            elif isinstance(value, float):
                value = None if pd.isna(value) else float(value)
            elif value is pd.NaT:
                value = None
            out[str(key)] = value
        return out

