"""Feature engineering service - conservative, leakage-safe derived features.

Every decision that could see the evaluation data (which interaction pairs to
build, the frequency maps used for high-cardinality encoding, whether a datetime
column parses) is learned in :meth:`FeatureEngineer.fit` on the **training split
only**.  :meth:`FeatureEngineer.transform` is then a pure function of the rows it
is given, so the same object can be reused for the test split and for
single/batch predictions without leaking information.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

from app.services.feature_operations import OPERATION_REGISTRY

logger = logging.getLogger(__name__)

MAX_INTERACTIONS = 5
_MAX_INTERACTION_SOURCES = 5
_MI_SAMPLE_LIMIT = 5000
DATETIME_PARTS = ("year", "month", "day", "dayofweek")


class FeatureEngineer:
    """Conservative feature engineering for tabular data (fit on train only)."""

    def __init__(self):
        self.datetime_columns: list[str] = []
        self.interaction_pairs: list[tuple[str, str]] = []
        self.frequency_maps: dict[str, dict[Any, float]] = {}
        self.numeric_new_features: list[str] = []
        self.transformations: list[dict[str, Any]] = []
        self.fitted = False
        # Registry-approved operation mode (validated LLM suggestions).
        self.operations: list[dict[str, Any]] = []
        self.rejected_operations: list[dict[str, Any]] = []
        self.operation_state: dict[str, dict[str, Any]] = {}
        self.mode = "heuristics"

    # ──────────────────────────────── fit ────────────────────────────────
    def fit(self, df: pd.DataFrame, column_profiles: list[dict],
            target_column: str, max_interactions: int = MAX_INTERACTIONS,
            operations: "list[dict] | None" = None) -> "FeatureEngineer":
        """Learn feature-engineering decisions from the training data only.

        With ``operations`` - the registry-approved output of
        ``validate_operations()`` - the engineer runs in operation mode: ONLY
        those transformations are applied, and their fit-time statistics
        (frequency maps, z-score moments, rare-category sets, log shifts) are
        captured on this frame. Without operations the legacy deterministic
        heuristics run unchanged.
        """
        self.datetime_columns = []
        self.interaction_pairs = []
        self.frequency_maps = {}
        self.numeric_new_features = []
        self.transformations = []
        self.operations = [dict(op) for op in (operations or [])]
        self.operation_state = {}
        self.mode = "operations" if self.operations else "heuristics"

        if target_column not in df.columns:
            logger.warning(f"Target column '{target_column}' missing during feature engineering fit.")
            self.fitted = True
            return self

        if self.operations:
            self._fit_operations(df)
            self.fitted = True
            logger.info(
                f"Feature engineering (operation mode) fitted on training split: "
                f"{len(self.operations)} registry-approved operations -> "
                f"{len(self.numeric_new_features)} new features"
            )
            return self

        # 1. Datetime extraction (only for columns that actually parse on train rows)
        for cp in column_profiles:
            col = cp["name"]
            if cp["inferred_type"] != "datetime" or col == target_column or col not in df.columns:
                continue
            try:
                parsed = pd.to_datetime(df[col].dropna().head(200), errors="coerce")
                if parsed.notna().sum() == 0:
                    continue  # unparseable -> leave the column to preprocessing
            except Exception as exc:
                logger.warning(f"Datetime parsing failed for '{col}': {exc}")
                continue
            self.datetime_columns.append(col)
            parts = [f"{col}_{part}" for part in DATETIME_PARTS]
            self.numeric_new_features.extend(parts)
            self.transformations.append({
                "type": "datetime_extraction",
                "source": col,
                "features": ", ".join(parts),
                "fitted_on": "training split",
            })

        # 2. Numerical interactions from the most informative training columns
        num_cols = [
            cp["name"] for cp in column_profiles
            if cp["inferred_type"] in ("numerical", "boolean")
            and cp["name"] != target_column and cp["name"] in df.columns
        ]
        if 2 <= len(num_cols) <= 10:
            self.interaction_pairs = self._select_interaction_pairs(
                df, num_cols, target_column, max_interactions
            )
            if self.interaction_pairs:
                for c1, c2 in self.interaction_pairs:
                    self.numeric_new_features.append(f"{c1}_x_{c2}")
                self.transformations.append({
                    "type": "interaction",
                    "description": (
                        f"Created {len(self.interaction_pairs)} interaction feature(s) from the "
                        f"most informative numerical training columns"
                    ),
                    "features": [f"{c1}_x_{c2}" for c1, c2 in self.interaction_pairs],
                    "fitted_on": "training split",
                })

        # 3. Frequency encoding for high-cardinality categoricals
        for cp in column_profiles:
            col = cp["name"]
            if (cp["inferred_type"] == "categorical" and cp["unique_count"] > 20
                    and col != target_column and col in df.columns):
                freq = df[col].value_counts(normalize=True, dropna=True)
                if freq.empty:
                    continue
                self.frequency_maps[col] = {str(k): float(v) for k, v in freq.items()}
                new_col = f"{col}_freq"
                self.numeric_new_features.append(new_col)
                self.transformations.append({
                    "type": "frequency_encoding",
                    "source": col,
                    "feature": new_col,
                    "fitted_on": "training split",
                })

        self.fitted = True
        logger.info(
            f"Feature engineering fitted on training split: "
            f"{len(self.datetime_columns)} datetime, {len(self.interaction_pairs)} interactions, "
            f"{len(self.frequency_maps)} frequency maps -> {len(self.numeric_new_features)} new features"
        )
        return self

    def _select_interaction_pairs(self, df: pd.DataFrame, num_cols: list[str],
                                  target_column: str, k: int) -> list[tuple[str, str]]:
        """Pick up to k column pairs using train-only association with the target."""
        target = df[target_column]
        X = df[num_cols].apply(pd.to_numeric, errors="coerce")
        empty = [c for c in X.columns if X[c].isna().all()]
        if empty:
            X = X.drop(columns=empty)
        if X.shape[1] < 2:
            return []
        X = X.fillna(X.median())

        scores: dict[str, float] = {}
        try:
            discrete_target = not (
                pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 20
            )
            if discrete_target:
                from sklearn.feature_selection import mutual_info_classif
                y = pd.factorize(target)[0]
                if len(X) > _MI_SAMPLE_LIMIT:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(len(X), _MI_SAMPLE_LIMIT, replace=False)
                    Xs, ys = X.iloc[idx], y[idx]
                else:
                    Xs, ys = X, y
                mi = mutual_info_classif(Xs, ys, random_state=42)
                scores = {c: float(v) for c, v in zip(X.columns, mi) if np.isfinite(v)}
            else:
                for col in X.columns:
                    corr = X[col].corr(target)
                    if corr is not None and np.isfinite(corr):
                        scores[col] = abs(float(corr))
        except Exception as exc:
            logger.warning(f"Interaction selection failed: {exc}")
            return []

        ranked = [c for c in sorted(scores, key=scores.get, reverse=True)
                  if scores[c] > 0][:_MAX_INTERACTION_SOURCES]
        pairs = [(ranked[i], ranked[j])
                 for i in range(len(ranked)) for j in range(i + 1, len(ranked))]
        return pairs[:k]

    # ───────────────────── registry-approved operations ──────────────────
    @staticmethod
    def _state_key(index: int, op: dict) -> str:
        return f"{index}:{op.get('column')}:{op.get('operation')}"

    def _fit_operations(self, df: pd.DataFrame) -> None:
        """Capture fit-time statistics for each registry-approved operation.

        Everything learned here comes from the training split only; transform()
        merely replays it, so test/prediction rows never influence this state.
        """
        for index, op in enumerate(self.operations):
            column = op.get("column")
            operation = op.get("operation")
            spec = OPERATION_REGISTRY.get(operation)
            if spec is None or column not in df.columns:
                continue
            key = self._state_key(index, op)
            params = op.get("params") or {}

            if spec.category == "numeric":
                series = pd.to_numeric(df[column], errors="coerce")
                if operation == "zscore":
                    finite = series.dropna()
                    mean = float(finite.mean()) if len(finite) else 0.0
                    std = float(finite.std()) if len(finite) else 1.0
                    if not np.isfinite(std) or std <= 0:
                        std = 1.0
                    if not np.isfinite(mean):
                        mean = 0.0
                    self.operation_state[key] = {"mean": mean, "std": std}
                elif operation == "log1p":
                    finite = series.dropna()
                    minimum = float(finite.min()) if len(finite) else 0.0
                    shift = (1.0 - minimum) if minimum <= 0 else 0.0
                    self.operation_state[key] = {"shift": shift}
            elif spec.category == "categorical":
                normalized = df[column].astype("string").fillna("<missing>")
                counts = normalized.value_counts(dropna=False)
                total = float(counts.sum()) or 1.0
                if operation == "frequency_encoding":
                    self.operation_state[key] = {
                        "frequencies": {str(k): float(v) / total
                                        for k, v in counts.items()}
                    }
                elif operation == "rare_category_grouping":
                    threshold = float(params.get("threshold", 0.02))
                    group_label = str(params.get("group_label", "Other"))
                    keep = {str(k) for k, v in counts.items()
                            if float(v) / total >= threshold}
                    self.operation_state[key] = {
                        "keep": keep, "group_label": group_label,
                    }
            # text and datetime operations are row-local: nothing to remember.

            partner = params.get("with_column") if spec.pair_operation else None
            features = list(op.get("features") or [])
            if not features:
                features = [spec.feature_name(column, partner)]
            for feature in features:
                if feature != column and feature not in self.numeric_new_features:
                    self.numeric_new_features.append(feature)
            self.transformations.append({
                "type": "operation",
                "operation": operation,
                "source": column,
                "feature": ", ".join(features) or column,
                "category": spec.category,
                "reason": op.get("reason") or "",
                "fitted_on": "training split",
            })

    def _apply_operation(self, result: pd.DataFrame, index: int, op: dict) -> pd.DataFrame:
        """Replay one approved operation using the state captured at fit time."""
        column = op.get("column")
        operation = op.get("operation")
        spec = OPERATION_REGISTRY.get(operation)
        if spec is None or column not in result.columns:
            return result
        params = op.get("params") or {}
        key = self._state_key(index, op)
        state = self.operation_state.get(key, {})
        if not state and operation in ("zscore", "log1p", "frequency_encoding",
                                       "rare_category_grouping"):
            return result  # never fitted (column missing on the training split)

        try:
            if spec.category == "numeric":
                series = pd.to_numeric(result[column], errors="coerce")
                if operation == "log1p":
                    with np.errstate(invalid="ignore"):
                        result[spec.feature_name(column)] = np.log1p(
                            series + float(state.get("shift", 0.0)))
                elif operation == "sqrt":
                    with np.errstate(invalid="ignore"):
                        result[spec.feature_name(column)] = np.sqrt(series)
                elif operation == "square":
                    result[spec.feature_name(column)] = series * series
                elif operation == "absolute":
                    result[spec.feature_name(column)] = np.abs(series)
                elif operation == "zscore":
                    result[spec.feature_name(column)] = (
                        (series - float(state.get("mean", 0.0)))
                        / float(state.get("std", 1.0))
                    )
                elif operation == "interaction":
                    partner = params.get("with_column")
                    if partner in result.columns:
                        result[spec.feature_name(column, partner)] = (
                            series * pd.to_numeric(result[partner], errors="coerce")
                        )
            elif spec.category == "categorical":
                if operation == "frequency_encoding":
                    frequencies = state.get("frequencies", {})
                    normalized = result[column].astype("string").fillna("<missing>")
                    result[spec.feature_name(column)] = (
                        normalized.map(frequencies).astype(float).fillna(0.0)
                    )
                elif operation == "rare_category_grouping":
                    keep = state.get("keep", set())
                    group_label = state.get("group_label", "Other")
                    normalized = result[column].astype("string").fillna("<missing>")
                    result[column] = normalized.where(
                        normalized.isin(keep), group_label
                    ).astype(object)
            elif spec.category == "text":
                text_col = result[column].astype("string").fillna("")
                if operation == "char_count":
                    result[spec.feature_name(column)] = text_col.str.len().astype(float)
                elif operation == "word_count":
                    result[spec.feature_name(column)] = (
                        text_col.str.split().str.len().fillna(0).astype(float)
                    )
                elif operation == "text_length":
                    result[spec.feature_name(column)] = (
                        text_col.str.replace(r"\s+", "", regex=True).str.len().astype(float)
                    )
            elif spec.category == "datetime":
                parsed = pd.to_datetime(result[column], errors="coerce")
                if operation in ("year", "month", "day", "dayofweek"):
                    result[spec.feature_name(column)] = (
                        getattr(parsed.dt, operation).astype(float)
                    )
        except Exception as exc:
            logger.warning(
                f"Operation '{operation}' on '{column}' failed during transform: {exc}"
            )
        return result

    # ────────────────────────────── transform ────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned transformations to any frame (train/test/single row)."""
        result = df.copy()
        if not self.fitted:
            return result

        if self.operations:
            # Registry mode: apply exactly the approved operations with the
            # statistics captured on the training split. Pure function of the
            # input rows + fitted state, so test/prediction frames are safe.
            for index, op in enumerate(self.operations):
                result = self._apply_operation(result, index, op)
            return result

        # 1. Datetime extraction (raw column is replaced by its calendar parts)
        for col in self.datetime_columns:
            if col not in result.columns:
                continue
            try:
                dt = pd.to_datetime(result[col], errors="coerce")
                for part in DATETIME_PARTS:
                    result[f"{col}_{part}"] = getattr(dt.dt, part).astype(float)
            except Exception as exc:
                logger.warning(f"Datetime extraction failed for '{col}': {exc}")
                for part in DATETIME_PARTS:
                    result[f"{col}_{part}"] = np.nan
            result = result.drop(columns=[col], errors="ignore")

        # 2. Interactions (always materialised so the feature matrix width is stable)
        for c1, c2 in self.interaction_pairs:
            name = f"{c1}_x_{c2}"
            try:
                result[name] = (pd.to_numeric(result[c1], errors="coerce")
                                * pd.to_numeric(result[c2], errors="coerce"))
            except Exception:
                result[name] = np.nan

        # 3. Frequency encoding using the training-split maps (unseen -> 0)
        for col, freq_map in self.frequency_maps.items():
            new_col = f"{col}_freq"
            if col in result.columns:
                result[new_col] = result[col].map(freq_map).astype(float).fillna(0.0)
            else:
                result[new_col] = 0.0

        return result

    def fit_transform(self, df: pd.DataFrame, column_profiles: list[dict],
                      target_column: str) -> pd.DataFrame:
        return self.fit(df, column_profiles, target_column).transform(df)

    # ────────────────────────── backward compatibility ───────────────────
    def apply(self, df: pd.DataFrame, column_profiles: list[dict],
              target_column: str) -> tuple[pd.DataFrame, list[str], list[dict]]:
        """Fit on the supplied frame and transform it (used when rebuilding a saved model)."""
        frame = self.fit_transform(df, column_profiles, target_column)
        return frame, list(self.numeric_new_features), list(self.transformations)

    @property
    def new_features(self) -> list[str]:
        return list(self.numeric_new_features)

    def get_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "mode": self.mode,
            "new_features_count": len(self.numeric_new_features),
            "new_features": list(self.numeric_new_features),
            "transformations": list(self.transformations),
            "fitted_on": "training split only (no leakage from test rows)",
        }
        if self.mode == "operations":
            summary["operations"] = list(self.operations)
            summary["rejected_operations"] = list(self.rejected_operations)
        return summary
