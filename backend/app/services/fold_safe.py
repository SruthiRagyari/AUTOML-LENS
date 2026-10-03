"""Fold-safe feature engineering + preprocessing for leakage-safe cross-validation.

Cross-validation must refit **every learned transformation** on the fold's own
training rows. Fitting ``FeatureEngineer``/``PreprocessingEngine`` once on the
whole training split and then running CV on the resulting matrix leaks
information between folds: the scaler, the imputer, the one-hot encoder, the
frequency maps and the operation statistics (z-score moments, rare-category
sets, log shifts) would all have been estimated from rows that later appear in
the validation fold.

This module provides the sklearn-compatible glue that makes per-fold fitting
possible. :class:`FoldSafeFeaturePreprocessor` learns its state in ``fit`` from
exactly the rows it is handed, so sklearn's own splitters (``cross_val_score``,
Optuna trials) refit it independently for every fold. The final model is then
refit on the complete training portion via :func:`build_final_pipeline`, which
wraps the already-fitted state without relearning anything from the holdout.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from app.services.feature_engineer import FeatureEngineer, MAX_INTERACTIONS
from app.services.preprocessor import PreprocessingEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSpec:
    """Immutable description of the feature pipeline to build for each fold.

    Carries only configuration (column metadata, target name, approved
    operations). No fitted statistic lives here, so an identical spec can be
    safely reused for every CV fold, every Optuna trial and the final refit -
    each of which learns its own state from its own rows.
    """

    column_profiles: Sequence[dict]
    target_column: str
    problem_type: str
    operations: Optional[Sequence[dict]] = None
    max_interactions: int = MAX_INTERACTIONS


def _as_frame(X: Any) -> pd.DataFrame:
    """Feature engineering needs named columns, so a DataFrame is required."""
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError(
        "FoldSafeFeaturePreprocessor expects a pandas DataFrame (got "
        f"{type(X).__name__}); feature engineering and named-column "
        "preprocessing cannot run on a bare array."
    )


class FoldSafeFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Fit feature engineering + preprocessing on exactly the rows received.

    Because sklearn clones and refits pipeline steps for every CV fold, all
    learned state - frequency maps, scaling, imputation, categorical encoding,
    interaction selection and registry-approved operation statistics - is
    derived from fold-training rows only. ``transform`` is a pure function of
    the rows it is given plus the fitted state, so it is also safe for the
    validation fold, the holdout and single/batch prediction.
    """

    def __init__(self, column_profiles=None, target_column=None,
                 problem_type="classification", operations=None,
                 max_interactions: int = MAX_INTERACTIONS):
        # sklearn convention: store constructor params unchanged.
        self.column_profiles = column_profiles
        self.target_column = target_column
        self.problem_type = problem_type
        self.operations = operations
        self.max_interactions = max_interactions

    def fit(self, X, y=None):
        frame = _as_frame(X)
        self.feature_engineer_ = FeatureEngineer()
        self.feature_engineer_.fit(
            frame, self.column_profiles, self.target_column,
            max_interactions=self.max_interactions, operations=self.operations,
        )
        engineered = self.feature_engineer_.transform(frame)
        # build_and_fit learns the ColumnTransformer state from the engineered
        # fold-training frame only.
        self.engine_ = PreprocessingEngine()
        self.engine_.build_and_fit(
            engineered, self.target_column, self.column_profiles,
            self.problem_type,
            extra_numeric=self.feature_engineer_.new_features,
            feature_engineer=self.feature_engineer_,
        )
        # Real evidence that this fit saw a single fold, not the whole split.
        self.fitted_rows_ = int(len(frame))
        self.feature_names_out_ = list(self.engine_.feature_names_out)
        self._record_fit(frame)
        return self

    def transform(self, X):
        frame = _as_frame(X)
        return self.engine_.transform(frame, target_column=self.target_column)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.engine_.feature_names_out), dtype=object)

    # -- test hook ----------------------------------------------------------
    def _record_fit(self, frame: pd.DataFrame) -> None:
        """Hook so tests can observe what each fold actually fitted on."""
        return None


def build_fold_safe_pipeline(estimator: Any, spec: FeatureSpec) -> Pipeline:
    """Compose ``fit-per-fold`` feature processing with a fresh estimator.

    The returned :class:`sklearn.pipeline.Pipeline` is what every CV split
    receives: sklearn refits the ``features`` step (and the estimator) on each
    fold's training rows, so validation rows never influence any learned state.
    """
    operations = list(spec.operations) if spec.operations else None
    return Pipeline([
        ("features", FoldSafeFeaturePreprocessor(
            column_profiles=spec.column_profiles,
            target_column=spec.target_column,
            problem_type=spec.problem_type,
            operations=operations,
            max_interactions=spec.max_interactions,
        )),
        ("model", estimator),
    ])


class FittedFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Expose an already-fitted FE/preprocessing pair as a single pipeline step.

    Used for the *final* pipeline only: the state it wraps was learned on the
    complete training portion, so it must never be refit (that would be the one
    fit allowed to see all training rows).
    """

    def __init__(self, feature_engineer=None, preprocessing_engine=None,
                 target_column=None):
        self.feature_engineer = feature_engineer
        self.preprocessing_engine = preprocessing_engine
        self.target_column = target_column

    def fit(self, X, y=None):
        # Nothing to learn: the wrapped state is already fitted.
        return self

    def transform(self, X):
        return self.preprocessing_engine.transform(
            _as_frame(X), target_column=self.target_column
        )

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.preprocessing_engine.feature_names_out),
                          dtype=object)


def build_final_pipeline(model: Any, feature_engineer: Any,
                         preprocessing_engine: Any,
                         target_column: str) -> Pipeline:
    """Wrap the complete-training-portion state into one persistable pipeline.

    This is the artifact persisted after selection: feature engineering,
    preprocessing and the model fitted on the complete training portion, with
    the holdout never having influenced any of them.
    """
    return Pipeline([
        ("features", FittedFeaturePreprocessor(
            feature_engineer=feature_engineer,
            preprocessing_engine=preprocessing_engine,
            target_column=target_column,
        )),
        ("model", model),
    ])

