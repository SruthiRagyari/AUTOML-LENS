"""Dataset profiling service - analyzes dataset structure and quality."""
import logging
from typing import Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DatasetProfiler:
    """Comprehensive dataset profiler for tabular data."""

    def profile(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate a complete profile of the dataset."""
        rows, cols = df.shape
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        dup_rows = int(df.duplicated().sum())
        total_missing = int(df.isnull().sum().sum())
        total_cells = rows * cols

        column_profiles = []
        warnings = []
        type_counts = {}

        for col in df.columns:
            series = df[col]
            cp = self._profile_column(series, rows)
            column_profiles.append(cp)

            itype = cp["inferred_type"]
            type_counts[itype] = type_counts.get(itype, 0) + 1

            # Generate warnings
            if cp["missing_percentage"] > 1.0:
                warnings.append(
                    f"'{col}' contains {cp['missing_percentage']:.1f}% missing values."
                )
            if itype == "id_like":
                warnings.append(f"'{col}' appears to be an identifier column.")
            if itype == "constant":
                warnings.append(f"'{col}' is constant (single unique value).")
            if itype == "categorical" and cp["unique_count"] > 50:
                warnings.append(
                    f"'{col}' has high cardinality ({cp['unique_count']} unique values)."
                )
            if cp["missing_percentage"] > 50:
                warnings.append(
                    f"'{col}' has >50% missing data — consider dropping."
                )

        if dup_rows > 0:
            pct = dup_rows / rows * 100
            warnings.append(f"Dataset contains {dup_rows} duplicate rows ({pct:.1f}%).")

        return {
            "rows": rows,
            "columns": cols,
            "memory_usage_mb": round(memory_mb, 2),
            "duplicate_rows": dup_rows,
            "duplicate_percentage": round(dup_rows / max(rows, 1) * 100, 2),
            "total_missing": total_missing,
            "total_missing_percentage": round(
                total_missing / max(total_cells, 1) * 100, 2
            ),
            "column_profiles": column_profiles,
            "warnings": warnings,
            "column_type_summary": type_counts,
        }

    def _profile_column(self, series: pd.Series, n_rows: int) -> dict[str, Any]:
        """Profile a single column."""
        col_name = series.name
        dtype_str = str(series.dtype)
        missing = int(series.isnull().sum())
        missing_pct = round(missing / max(n_rows, 1) * 100, 2)
        non_null = series.dropna()
        unique_count = int(non_null.nunique())

        profile: dict[str, Any] = {
            "name": col_name,
            "dtype": dtype_str,
            "inferred_type": "categorical",
            "unique_count": unique_count,
            "missing_count": missing,
            "missing_percentage": missing_pct,
            "mean": None,
            "median": None,
            "std": None,
            "min_val": None,
            "max_val": None,
            "cardinality": unique_count,
            "sample_values": [],
        }

        # Sample values
        try:
            samples = non_null.unique()[:5].tolist()
            profile["sample_values"] = [
                self._safe_serialize(v) for v in samples
            ]
        except Exception:
            profile["sample_values"] = []

        # Infer type
        profile["inferred_type"] = self._infer_type(
            series, col_name, dtype_str, unique_count, n_rows
        )

        # Stats for numerical
        if profile["inferred_type"] == "numerical" and len(non_null) > 0:
            try:
                num = pd.to_numeric(non_null, errors="coerce").dropna()
                if len(num) > 0:
                    profile["mean"] = round(float(num.mean()), 4)
                    profile["median"] = round(float(num.median()), 4)
                    profile["std"] = round(float(num.std()), 4)
                    profile["min_val"] = self._safe_serialize(num.min())
                    profile["max_val"] = self._safe_serialize(num.max())
            except Exception:
                pass

        return profile

    def _infer_type(
        self, series: pd.Series, name: str, dtype: str,
        unique_count: int, n_rows: int
    ) -> str:
        """Infer the semantic type of a column."""
        name_lower = name.lower()
        non_null = series.dropna()

        # Constant
        if unique_count <= 1:
            return "constant"

        # Boolean, but only when the column really is binary in a way the
        # numeric pipeline can consume: a genuine bool dtype, or numbers that
        # are exactly 0/1.
        #
        # Text such as 'yes'/'no' must NOT be reported as boolean. The
        # preprocessing engine routes boolean columns into the numeric imputer,
        # so a text boolean reached the median imputer as a string and failed
        # with "Cannot use median strategy with non-numeric data". Such a
        # column is an ordinary two-valued category and is now treated as one,
        # which one-hot encodes correctly.
        if series.dtype == bool:
            return "boolean"
        if unique_count <= 3 and len(non_null) > 0:
            try:
                observed = set(pd.unique(non_null.to_numpy()))
                if observed and all(
                    isinstance(v, (int, float, bool)) and not pd.isna(v)
                    and float(v) in (0.0, 1.0)
                    for v in observed
                ):
                    return "boolean"
            except (TypeError, ValueError):
                pass

        # Datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # ID-like
        if unique_count > 0 and n_rows > 0:
            unique_ratio = unique_count / n_rows
            id_patterns = ["_id", "id_", "index", "key", "uuid", "guid"]
            if (name_lower == "id" or any(p in name_lower for p in id_patterns)) and unique_ratio > 0.9:
                return "id_like"
            if name_lower in ("id", "index", "row", "rowid", "row_id") and unique_ratio > 0.95:
                return "id_like"

        # Numerical
        if pd.api.types.is_numeric_dtype(series):
            return "numerical"

        # Text vs categorical for object dtype
        if dtype == "object" or pd.api.types.is_string_dtype(series):
            if len(non_null) > 0:
                avg_len = non_null.astype(str).str.len().mean()
                if avg_len > 50:
                    return "text"
                unique_ratio = unique_count / max(n_rows, 1)
                if avg_len > 20 and unique_ratio > 0.5:
                    return "text"

            # Try parsing as datetime
            try:
                pd.to_datetime(non_null.head(20), format='mixed', dayfirst=False)
                return "datetime"
            except Exception:
                pass

            return "categorical"

        return "categorical"

    def suggest_target(
        self, profile: dict, task_description: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """Suggest target column and problem type."""
        columns = profile["column_profiles"]
        candidates = []
        n_cols = len(columns)

        target_keywords = [
            "target", "label", "class", "output", "y_",
            "price", "salary", "income", "cost", "amount", "value",
            "survived", "churn", "default", "fraud", "species",
            "category", "diagnosis", "outcome", "result", "response",
            "approved", "cancelled", "converted", "purchased", "success",
        ]

        for idx, cp in enumerate(columns):
            name = cp["name"]
            name_lower = name.lower()
            itype = cp["inferred_type"]

            if itype in ("id_like", "constant", "text", "datetime"):
                continue

            score = 0

            # Keyword match (strong signal)
            for kw in target_keywords:
                if kw in name_lower:
                    score += 15
                    break
            # Exact match bonus
            if name_lower in ("churn", "target", "label", "survived", "price", "y", "output"):
                score += 10

            # Last column heuristic — strong in tabular ML
            if idx == n_cols - 1:
                score += 8
            elif idx >= n_cols - 3:
                score += 3

            # Binary classification targets are most common
            if itype in ("boolean", "numerical") and cp["unique_count"] == 2:
                score += 8
            elif itype == "categorical" and 2 <= cp["unique_count"] <= 5:
                score += 6
            elif itype == "categorical" and 5 < cp["unique_count"] <= 20:
                score += 3
            elif itype == "numerical" and cp["unique_count"] > 20:
                score += 2  # Regression target

            # Penalty for high missing
            if cp["missing_percentage"] > 30:
                score -= 10
            elif cp["missing_percentage"] > 10:
                score -= 3

            # Penalty for high cardinality categoricals (unlikely to be target)
            if itype == "categorical" and cp["unique_count"] > 20:
                score -= 8

            if score >= 0:
                candidates.append((name, score, cp))

        if not candidates:
            # Fallback: last column
            last = [cp for cp in columns if cp["inferred_type"] not in ("id_like", "constant")]
            if last:
                best_cp = last[-1]
                return best_cp["name"], self.detect_problem_type(best_cp)
            return None, None

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_name = candidates[0][0]
        best_cp = candidates[0][2]

        return best_name, self.detect_problem_type(best_cp)

    def detect_problem_type(self, column_profile: dict) -> str:
        """Detect classification vs regression from column profile."""
        itype = column_profile["inferred_type"]
        unique = column_profile["unique_count"]

        if itype == "categorical":
            return "classification"
        if itype == "boolean":
            return "classification"
        if itype == "numerical":
            if unique <= 20:
                return "classification"
            return "regression"
        return "classification"

    @staticmethod
    def _safe_serialize(val: Any) -> Any:
        """Convert numpy/pandas types to Python native for JSON serialization."""
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, (np.bool_,)):
            return bool(val)
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            return str(val)
        if isinstance(val, (np.ndarray,)):
            return val.tolist()
        return val




