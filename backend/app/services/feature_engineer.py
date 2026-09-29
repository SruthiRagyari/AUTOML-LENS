"""Feature engineering service - creates derived features conservatively."""
import logging
from typing import Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Conservative feature engineering for tabular data."""

    def __init__(self):
        self.transformations: list[dict[str, str]] = []
        self.new_features: list[str] = []

    def apply(self, df: pd.DataFrame, column_profiles: list[dict],
              target_column: str) -> tuple[pd.DataFrame, list[str], list[dict]]:
        """Apply feature engineering transformations."""
        result_df = df.copy()
        self.transformations = []
        self.new_features = []

        num_cols = [
            cp["name"] for cp in column_profiles
            if cp["inferred_type"] == "numerical" and cp["name"] != target_column
        ]
        dt_cols = [
            cp["name"] for cp in column_profiles
            if cp["inferred_type"] == "datetime" and cp["name"] != target_column
        ]
        cat_cols = [
            cp["name"] for cp in column_profiles
            if cp["inferred_type"] == "categorical" and cp["name"] != target_column
        ]

        # 1. Datetime extraction
        for col in dt_cols:
            if col in result_df.columns:
                result_df = self._extract_datetime(result_df, col)

        # 2. Numerical interactions (limited)
        if 2 <= len(num_cols) <= 10:
            result_df = self._create_interactions(result_df, num_cols, target_column)

        # 3. Frequency encoding for high-cardinality categoricals
        for cp in column_profiles:
            if cp["inferred_type"] == "categorical" and cp["unique_count"] > 20 and cp["name"] != target_column:
                if cp["name"] in result_df.columns:
                    result_df = self._frequency_encode(result_df, cp["name"])

        return result_df, self.new_features, self.transformations

    def _extract_datetime(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        try:
            dt = pd.to_datetime(df[col], errors="coerce")
            for part, accessor in [("year", "year"), ("month", "month"),
                                    ("day", "day"), ("dayofweek", "dayofweek")]:
                new_col = f"{col}_{part}"
                df[new_col] = getattr(dt.dt, accessor).astype(float)
                self.new_features.append(new_col)
            self.transformations.append({
                "type": "datetime_extraction",
                "source": col,
                "features": f"{col}_year, {col}_month, {col}_day, {col}_dayofweek",
            })
        except Exception as e:
            logger.warning(f"Datetime extraction failed for {col}: {e}")
        return df

    def _create_interactions(self, df: pd.DataFrame, num_cols: list[str],
                              target_col: str) -> pd.DataFrame:
        """Create interaction features between top correlated pairs."""
        try:
            cols_in_df = [c for c in num_cols if c in df.columns]
            if len(cols_in_df) < 2 or target_col not in df.columns:
                return df

            # Find top correlated pairs with target
            correlations = {}
            target = df[target_col]
            if not pd.api.types.is_numeric_dtype(target):
                return df

            for col in cols_in_df:
                try:
                    corr = abs(df[col].corr(target))
                    if not np.isnan(corr):
                        correlations[col] = corr
                except Exception:
                    pass

            if len(correlations) < 2:
                return df

            top_cols = sorted(correlations, key=correlations.get, reverse=True)[:5]

            interactions_created = 0
            for i in range(len(top_cols)):
                for j in range(i + 1, len(top_cols)):
                    if interactions_created >= 5:
                        break
                    c1, c2 = top_cols[i], top_cols[j]
                    new_name = f"{c1}_x_{c2}"
                    try:
                        df[new_name] = df[c1] * df[c2]
                        self.new_features.append(new_name)
                        interactions_created += 1
                    except Exception:
                        pass

            if interactions_created > 0:
                self.transformations.append({
                    "type": "interaction",
                    "description": f"Created {interactions_created} interaction features from top correlated numerical columns",
                })
        except Exception as e:
            logger.warning(f"Interaction feature creation failed: {e}")
        return df

    def _frequency_encode(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        try:
            freq = df[col].value_counts(normalize=True)
            new_col = f"{col}_freq"
            df[new_col] = df[col].map(freq).fillna(0)
            self.new_features.append(new_col)
            self.transformations.append({
                "type": "frequency_encoding",
                "source": col,
                "feature": new_col,
            })
        except Exception as e:
            logger.warning(f"Frequency encoding failed for {col}: {e}")
        return df

    def get_summary(self) -> dict[str, Any]:
        return {
            "new_features_count": len(self.new_features),
            "new_features": self.new_features,
            "transformations": self.transformations,
        }
