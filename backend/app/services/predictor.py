"""Prediction service for single and batch predictions."""
import os
import datetime
import logging
from typing import Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Predictor:
    """Handle single and batch predictions using trained models."""

    def __init__(self, model, preprocessor, feature_names: list[str],
                 target_column: str, problem_type: str):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.target_column = target_column
        self.problem_type = problem_type

    def predict_single(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Make a prediction for a single input."""
        try:
            df = pd.DataFrame([input_dict])
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[self.feature_names]

            if self.preprocessor is not None:
                X = self.preprocessor.transform(df)
            else:
                X = df.values

            pred = self.model.predict(X)[0]
            result: dict[str, Any] = {
                "prediction": self._to_native(pred)
            }

            if self.problem_type == "classification" and hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(X)[0]
                    classes = self.model.classes_
                    result["probabilities"] = {
                        str(c): round(float(p), 4) for c, p in zip(classes, probs)
                    }
                    result["confidence"] = round(float(np.max(probs)), 4)
                except Exception:
                    pass

            return result
        except Exception as e:
            logger.exception(f"Single prediction failed: {e}")
            return {"error": str(e)}

    def predict_batch(self, csv_path: str, storage_path: str = "storage") -> dict[str, Any]:
        """Make predictions for a batch CSV file."""
        try:
            df = pd.read_csv(csv_path)
            input_df = df.copy()

            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = np.nan

            if self.preprocessor is not None:
                X = self.preprocessor.transform(df[self.feature_names])
            else:
                X = df[self.feature_names].values

            preds = self.model.predict(X)
            input_df["prediction"] = [self._to_native(p) for p in preds]

            if self.problem_type == "classification" and hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(X)
                    input_df["confidence"] = [round(float(np.max(p)), 4) for p in probs]
                except Exception:
                    pass

            out_dir = os.path.join(storage_path, "predictions")
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"predictions_{ts}.csv")
            input_df.to_csv(out_path, index=False)

            return {
                "predictions_path": out_path,
                "num_predictions": len(input_df),
                "preview": input_df.head(10).to_dict(orient="records"),
            }
        except Exception as e:
            logger.exception(f"Batch prediction failed: {e}")
            return {"error": str(e)}

    def get_input_schema(self, original_df: pd.DataFrame = None) -> list[dict[str, Any]]:
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
                    info["sample_values"] = original_df[feat].dropna().unique()[:5].tolist()
                    info["sample_values"] = [self._to_native(v) for v in info["sample_values"]]
                except Exception:
                    pass
            schema.append(info)
        return schema

    @staticmethod
    def _to_native(val):
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, (np.bool_,)):
            return bool(val)
        return val
