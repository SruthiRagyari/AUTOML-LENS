"""Preprocessing engine using sklearn pipelines to prevent data leakage."""
import logging
from typing import Any, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class PreprocessingEngine:
    """Builds sklearn preprocessing pipelines from dataset profile."""

    def __init__(self):
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names_out: list[str] = []
        self.dropped_columns: list[str] = []
        self.datetime_features: dict[str, list[str]] = {}
        self.summary: dict[str, Any] = {}
        # Feature-engineering wiring (filled by build_and_fit so transform()
        # reproduces exactly the columns the model was fitted on).
        self.feature_engineer: Optional[Any] = None
        self.numeric_input_columns: list[str] = []
        self.categorical_input_columns: list[str] = []
        self.text_input_columns: list[str] = []

    def build_and_fit(
        self, df: pd.DataFrame, target_column: str,
        column_profiles: list[dict], problem_type: str,
        extra_numeric: Optional[list[str]] = None,
        feature_engineer: Optional[Any] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Build preprocessing pipeline and fit-transform the data.

        ``extra_numeric`` carries engineered numeric columns (interactions,
        frequency encodings, datetime parts) so they are actually fed to the
        model instead of being discarded by ``remainder="drop"``.
        """
        self.feature_engineer = feature_engineer
        # Categorize columns by type
        numerical_cols = []
        categorical_cols = []
        text_cols = []
        datetime_cols = []
        drop_cols = []

        for cp in column_profiles:
            name = cp["name"]
            if name == target_column:
                continue
            itype = cp["inferred_type"]

            if itype in ("id_like", "constant"):
                drop_cols.append(name)
            elif itype == "datetime":
                datetime_cols.append(name)
            elif itype == "text":
                text_cols.append(name)
            elif itype == "numerical":
                numerical_cols.append(name)
            elif itype == "boolean":
                # Treat as numerical (0/1)
                numerical_cols.append(name)
            elif itype == "categorical":
                categorical_cols.append(name)
            else:
                categorical_cols.append(name)

        self.dropped_columns = drop_cols

        # Prepare DataFrame - extract datetime features, drop id/constant
        work_df = df.drop(columns=[target_column] + drop_cols, errors="ignore")

        # Extract datetime features
        new_num_cols = []
        for col in datetime_cols:
            if col in work_df.columns:
                try:
                    dt = pd.to_datetime(work_df[col], errors="coerce")
                    work_df[f"{col}_year"] = dt.dt.year.astype(float)
                    work_df[f"{col}_month"] = dt.dt.month.astype(float)
                    work_df[f"{col}_day"] = dt.dt.day.astype(float)
                    work_df[f"{col}_dayofweek"] = dt.dt.dayofweek.astype(float)
                    new_cols = [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"]
                    new_num_cols.extend(new_cols)
                    self.datetime_features[col] = new_cols
                    work_df = work_df.drop(columns=[col])
                except Exception as e:
                    logger.warning(f"Failed to parse datetime column {col}: {e}")
                    work_df = work_df.drop(columns=[col], errors="ignore")

        numerical_cols = [c for c in numerical_cols if c in work_df.columns] + new_num_cols
        # Engineered numeric features (interactions / frequency encodings / datetime
        # parts) must be part of the numeric transformer or remainder="drop" loses them.
        engineered_cols = [c for c in (extra_numeric or [])
                           if c in work_df.columns and c not in numerical_cols]
        numerical_cols = numerical_cols + engineered_cols
        categorical_cols = [c for c in categorical_cols if c in work_df.columns]
        text_cols = [c for c in text_cols if c in work_df.columns]

        # Convert boolean-like columns
        for col in numerical_cols:
            if col in work_df.columns and work_df[col].dtype == "object":
                mapping = {"true": 1, "false": 0, "yes": 1, "no": 0}
                work_df[col] = work_df[col].astype(str).str.lower().map(mapping).fillna(0).astype(float)

        # Build transformers
        transformers = []

        if numerical_cols:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("numerical", num_pipeline, numerical_cols))

        if categorical_cols:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=20,
                )),
            ])
            transformers.append(("categorical", cat_pipeline, categorical_cols))

        if text_cols:
            for tc in text_cols[:3]:  # Limit to 3 text columns
                text_pipeline = Pipeline([
                    ("tfidf", TfidfVectorizer(
                        max_features=100,
                        stop_words="english",
                    )),
                ])
                transformers.append((f"text_{tc}", text_pipeline, tc))

        if not transformers:
            # Edge case: no features
            logger.warning("No features to preprocess!")
            self.feature_names_out = []
            return np.array([]).reshape(len(df), 0), []

        # Remember exactly which columns the transformer needs so transform()
        # can backfill anything missing from partial (prediction) inputs.
        self.numeric_input_columns = list(numerical_cols)
        self.categorical_input_columns = list(categorical_cols)
        self.text_input_columns = list(text_cols)

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )

        # Fit and transform
        X = self.preprocessor.fit_transform(work_df)

        # Get feature names
        try:
            self.feature_names_out = list(self.preprocessor.get_feature_names_out())
        except Exception:
            self.feature_names_out = [f"feature_{i}" for i in range(X.shape[1])]

        # Build summary
        self.summary = {
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "text_columns": text_cols,
            "datetime_columns": list(self.datetime_features.keys()),
            "dropped_columns": drop_cols,
            "datetime_extracted_features": self.datetime_features,
            "total_features_in": len(numerical_cols) + len(categorical_cols) + len(text_cols),
            "engineered_columns": engineered_cols,
            "total_features_out": X.shape[1],
            "numerical_strategy": "median imputation + standard scaling",
            "categorical_strategy": "mode imputation + one-hot encoding (max 20 categories)",
            "text_strategy": "TF-IDF (max 100 features)" if text_cols else "N/A",
        }

        logger.info(
            f"Preprocessing: {len(numerical_cols)} numerical, "
            f"{len(categorical_cols)} categorical, {len(text_cols)} text -> "
            f"{X.shape[1]} features"
        )
        return X, self.feature_names_out

    def transform(self, df: pd.DataFrame, target_column: str = None) -> np.ndarray:
        """Transform new data using fitted preprocessor."""
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not fitted. Call build_and_fit first.")

        if self.feature_engineer is not None:
            # Reproduce the engineered columns learned on the training split.
            df = self.feature_engineer.transform(df)

        cols_to_drop = ([target_column] if target_column else []) + self.dropped_columns
        work_df = df.drop(columns=cols_to_drop, errors="ignore")

        # Extract datetime features
        for col, new_cols in self.datetime_features.items():
            if col in work_df.columns:
                try:
                    dt = pd.to_datetime(work_df[col], errors="coerce")
                    work_df[f"{col}_year"] = dt.dt.year.astype(float)
                    work_df[f"{col}_month"] = dt.dt.month.astype(float)
                    work_df[f"{col}_day"] = dt.dt.day.astype(float)
                    work_df[f"{col}_dayofweek"] = dt.dt.dayofweek.astype(float)
                    work_df = work_df.drop(columns=[col])
                except Exception:
                    work_df = work_df.drop(columns=[col], errors="ignore")

        work_df = self._backfill_expected_columns(work_df)
        return self.preprocessor.transform(work_df)

    def _backfill_expected_columns(self, work_df: pd.DataFrame) -> pd.DataFrame:
        """Add any expected column missing from a partial input frame.

        Guarantees the transformer always receives the columns it was fitted on
        (numeric -> NaN for median imputation, categorical -> None for mode
        imputation, text -> empty string for TF-IDF).
        """
        for col in self.numeric_input_columns:
            if col not in work_df.columns:
                work_df[col] = np.nan
        for col in self.categorical_input_columns:
            if col not in work_df.columns:
                work_df[col] = None
        for col in self.text_input_columns:
            if col not in work_df.columns:
                work_df[col] = ""
        return work_df

    def get_summary(self) -> dict[str, Any]:
        return self.summary

