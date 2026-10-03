"""Research-Grade Model Fusion and Ensemble Optimization Subsystem.

Leakage-Safety & Holdout-Isolation Protocol
-------------------------------------------
1. Out-of-Fold (OOF) Prediction Generation:
   All ensemble candidate evaluations, stacking meta-model training, and
   weight optimizations use out-of-fold predictions computed strictly
   on the training split (X_train, y_train) using K-Fold or Stratified K-Fold.
   The final holdout (X_test, y_test) is NEVER accessed during ensemble weight
   optimization, stacking, or selection.

2. Ensemble Strategies Supported:
   - Classification:
     * Soft Voting (uniform probability averaging across calibrated members)
     * Weighted Ensemble (constrained numerical optimization on OOF probabilities)
     * Stacking (regularized meta-learner trained on OOF probability vectors)
   - Regression:
     * Uniform Averaging (equal weights across candidate regressors)
     * Weighted Averaging (constrained optimization minimizing OOF error / maximizing R2)
     * Stacking (regularized linear meta-model trained on OOF predictions)

3. Selection Evidence:
   Ensemble candidates produce cross-validated scores derived exclusively from
   the OOF predictions on the training split. They compete on an equal footing
   with individual candidate models via TrainingManager.select_best_model.

4. Final Refit & Evaluation:
   The selected winner (whether individual or ensemble) is refit on the full
   training portion and evaluated exactly ONCE on the untouched holdout.
"""
from __future__ import annotations

import copy
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from app.services.evaluator import get_metric_spec, Evaluator, is_higher_better
from app.services.trainer import TrainingResult

logger = logging.getLogger(__name__)


# =========================================================================
# 1. Scikit-Learn Compatible Ensemble Estimators
# =========================================================================

class EnsembleModel(BaseEstimator):
    """Unified scikit-learn compatible estimator for fused models.

    Works seamlessly with Predictor, single/batch prediction, joblib serialization,
    and permutation feature importance.
    """

    def __init__(
        self,
        problem_type: str = "classification",
        strategy: str = "weighted_average",
        member_names: Optional[List[str]] = None,
        member_models: Optional[List[Any]] = None,
        weights: Optional[Union[List[float], np.ndarray, Dict[str, float]]] = None,
        meta_learner: Optional[Any] = None,
        classes: Optional[Union[List[Any], np.ndarray]] = None,
    ):
        self.problem_type = problem_type
        self.strategy = strategy
        self.member_names = list(member_names or [])
        self.member_models = list(member_models or [])
        self.meta_learner = meta_learner
        self.classes_ = np.asarray(classes) if classes is not None else None

        # Normalize weights into a 1D numpy array aligned with member_names
        if weights is None or len(self.member_names) == 0:
            n = max(len(self.member_names), 1)
            self.weights_ = np.full(n, 1.0 / n, dtype=np.float64)
        elif isinstance(weights, dict):
            w_arr = np.array([float(weights.get(name, 0.0)) for name in self.member_names], dtype=np.float64)
            s = np.sum(w_arr)
            self.weights_ = w_arr / s if s > 0 else np.full(len(self.member_names), 1.0 / len(self.member_names))
        else:
            w_arr = np.asarray(weights, dtype=np.float64)
            s = np.sum(w_arr)
            self.weights_ = w_arr / s if s > 0 else np.full(len(w_arr), 1.0 / len(w_arr))

    @property
    def weights_dict(self) -> Dict[str, float]:
        """Human-readable dictionary mapping member name to normalized weight."""
        return {
            name: round(float(self.weights_[i]), 6)
            for i, name in enumerate(self.member_names)
            if i < len(self.weights_)
        }

    def fit(self, X: Any, y: Any) -> EnsembleModel:
        """Fit all member models (and meta-learner if stacking) on the complete training portion."""
        if self.problem_type == "classification":
            self.classes_ = np.unique(y)

        # Refit each member model if it is an unfitted estimator
        for model in self.member_models:
            if hasattr(model, "fit"):
                model.fit(X, y)

        if self.strategy == "stacking" and self.meta_learner is not None:
            # Generate predictions on X to fit the meta-learner
            meta_features = self._construct_meta_features(X)
            self.meta_learner.fit(meta_features, y)

        return self

    def _align_probabilities(self, model: Any, X: Any) -> np.ndarray:
        """Extract predict_proba and align columns with self.classes_."""
        proba = model.predict_proba(X)
        if self.classes_ is None:
            return proba
        model_classes = getattr(model, "classes_", None)
        if model_classes is None or len(model_classes) != len(self.classes_):
            return proba
        if np.array_equal(model_classes, self.classes_):
            return proba

        # Align columns to match canonical self.classes_ order
        aligned = np.zeros((proba.shape[0], len(self.classes_)), dtype=np.float64)
        for target_idx, target_cls in enumerate(self.classes_):
            source_idx = np.where(model_classes == target_cls)[0]
            if len(source_idx) > 0:
                aligned[:, target_idx] = proba[:, source_idx[0]]
        return aligned

    def _construct_meta_features(self, X: Any) -> np.ndarray:
        """Stack member predictions horizontally for meta-learner input."""
        features_list = []
        for model in self.member_models:
            if self.problem_type == "classification" and hasattr(model, "predict_proba"):
                probs = self._align_probabilities(model, X)
                features_list.append(probs)
            else:
                preds = np.asarray(model.predict(X)).reshape(-1, 1)
                features_list.append(preds)
        return np.hstack(features_list)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities via soft voting or stacking."""
        if self.problem_type != "classification":
            raise AttributeError("predict_proba is only available for classification models.")

        if self.strategy == "stacking" and self.meta_learner is not None:
            meta_X = self._construct_meta_features(X)
            if hasattr(self.meta_learner, "predict_proba"):
                return self.meta_learner.predict_proba(meta_X)
            else:
                # If meta-learner doesn't have predict_proba, use decision function or one-hot
                preds = self.meta_learner.predict(meta_X)
                out = np.zeros((len(preds), len(self.classes_)), dtype=np.float64)
                for i, p in enumerate(preds):
                    idx = np.where(self.classes_ == p)[0]
                    if len(idx) > 0:
                        out[i, idx[0]] = 1.0
                return out

        # Soft voting or weighted probability fusion
        weighted_proba = None
        for i, model in enumerate(self.member_models):
            if hasattr(model, "predict_proba"):
                p = self._align_probabilities(model, X)
            else:
                # Fallback: one-hot from hard predict
                hard = model.predict(X)
                p = np.zeros((len(hard), len(self.classes_)), dtype=np.float64)
                for row_idx, val in enumerate(hard):
                    c_idx = np.where(self.classes_ == val)[0]
                    if len(c_idx) > 0:
                        p[row_idx, c_idx[0]] = 1.0

            w = self.weights_[i] if i < len(self.weights_) else 1.0 / len(self.member_models)
            if weighted_proba is None:
                weighted_proba = w * p
            else:
                weighted_proba += w * p

        # Renormalize rows to sum strictly to 1.0
        row_sums = weighted_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return weighted_proba / row_sums

    def predict(self, X: Any) -> np.ndarray:
        """Generate final predictions."""
        if self.problem_type == "classification":
            probs = self.predict_proba(X)
            best_class_indices = np.argmax(probs, axis=1)
            if self.classes_ is not None and len(self.classes_) > 0:
                return self.classes_[best_class_indices]
            return best_class_indices

        # Regression
        if self.strategy == "stacking" and self.meta_learner is not None:
            meta_X = self._construct_meta_features(X)
            return np.asarray(self.meta_learner.predict(meta_X))

        # Weighted regression prediction
        preds_weighted = None
        for i, model in enumerate(self.member_models):
            y_pred = np.asarray(model.predict(X), dtype=np.float64)
            w = self.weights_[i] if i < len(self.weights_) else 1.0 / len(self.member_models)
            if preds_weighted is None:
                preds_weighted = w * y_pred
            else:
                preds_weighted += w * y_pred
        return preds_weighted

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Aggregated feature importances if member models support them."""
        valid_importances = []
        valid_weights = []
        for i, model in enumerate(self.member_models):
            # Check for standard tree feature_importances_ or pipeline estimator
            m = model
            if hasattr(m, "named_steps") and "model" in m.named_steps:
                m = m.named_steps["model"]
            if hasattr(m, "feature_importances_"):
                valid_importances.append(np.asarray(m.feature_importances_, dtype=np.float64))
                valid_weights.append(self.weights_[i] if i < len(self.weights_) else 1.0)
            elif hasattr(m, "coef_"):
                c = np.abs(np.asarray(m.coef_, dtype=np.float64))
                if c.ndim > 1:
                    c = c.mean(axis=0)
                valid_importances.append(c)
                valid_weights.append(self.weights_[i] if i < len(self.weights_) else 1.0)

        if not valid_importances:
            return None

        # Check dimension consistency
        dim = valid_importances[0].shape[0]
        if not all(imp.shape[0] == dim for imp in valid_importances):
            return None

        # Weighted combination
        weights_norm = np.array(valid_weights, dtype=np.float64)
        s = np.sum(weights_norm)
        if s > 0:
            weights_norm /= s

        agg = np.zeros(dim, dtype=np.float64)
        for w, imp in zip(weights_norm, valid_importances):
            agg += w * imp
        return agg


# =========================================================================
# 2. Out-Of-Fold (OOF) Prediction Generator
# =========================================================================

def generate_oof_predictions(
    candidate_results: List[TrainingResult],
    model_definitions: List[Any],
    X_train: Any,
    y_train: Any,
    problem_type: str,
    n_folds: int = 5,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Generate out-of-fold (OOF) predictions strictly on the training split.

    Returns:
        oof_dict: {model_name: np.ndarray (N x C for clf probs, N for reg)}
        cv_splits: list of (train_idx, val_idx)
        classes: canonical classes array for classification (or None for reg)
    """
    n_samples = len(y_train)
    y_arr = np.asarray(y_train)

    if problem_type == "classification":
        classes = np.unique(y_arr)
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        classes = None
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits = list(splitter.split(X_train, y_train))
    oof_dict: Dict[str, np.ndarray] = {}

    # Map model_name to its ModelDefinition
    defs_by_name = {mdef.name: mdef for mdef in model_definitions}

    for res in candidate_results:
        if res.status != "COMPLETED" or res.model_name not in defs_by_name:
            continue

        mdef = defs_by_name[res.model_name]
        params = res.best_params or {}

        # Initialize OOF array
        if problem_type == "classification":
            oof_arr = np.zeros((n_samples, len(classes)), dtype=np.float64)
        else:
            oof_arr = np.zeros(n_samples, dtype=np.float64)

        success = True
        for train_idx, val_idx in splits:
            if isinstance(X_train, pd.DataFrame):
                X_f_tr, X_f_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
            else:
                X_f_tr, X_f_va = X_train[train_idx], X_train[val_idx]

            y_f_tr = y_arr[train_idx]

            try:
                # Instantiate clean clone with identical hyperparameters
                fold_model = mdef.create_model(**params)
                fold_model.fit(X_f_tr, y_f_tr)

                if problem_type == "classification":
                    if hasattr(fold_model, "predict_proba"):
                        proba = fold_model.predict_proba(X_f_va)
                        fold_classes = getattr(fold_model, "classes_", classes)
                        # Align columns with canonical classes
                        if np.array_equal(fold_classes, classes):
                            oof_arr[val_idx] = proba
                        else:
                            for c_i, cls in enumerate(classes):
                                pos = np.where(fold_classes == cls)[0]
                                if len(pos) > 0:
                                    oof_arr[val_idx, c_i] = proba[:, pos[0]]
                    else:
                        # Hard prediction fallback
                        preds = fold_model.predict(X_f_va)
                        for r_i, p_val in enumerate(preds):
                            c_i = np.where(classes == p_val)[0]
                            if len(c_i) > 0:
                                oof_arr[val_idx[r_i], c_i[0]] = 1.0
                else:
                    preds = fold_model.predict(X_f_va)
                    oof_arr[val_idx] = preds
            except Exception as e:
                logger.warning(f"OOF generation failed for model {res.model_name} on fold: {e}")
                success = False
                break

        if success:
            oof_dict[res.model_name] = oof_arr

    return oof_dict, splits, classes


# =========================================================================
# 3. Ensemble Weight Optimization & Scoring
# =========================================================================

def _score_predictions(
    y_true: np.ndarray,
    preds_or_probs: np.ndarray,
    problem_type: str,
    metric_name: str,
    classes: Optional[np.ndarray] = None,
) -> float:
    """Evaluate predictions against y_true using the specified metric."""
    spec = get_metric_spec(metric_name)

    if problem_type == "classification":
        if preds_or_probs.ndim == 2:
            pred_classes = classes[np.argmax(preds_or_probs, axis=1)] if classes is not None else np.argmax(preds_or_probs, axis=1)
            probs = preds_or_probs
        else:
            pred_classes = preds_or_probs
            probs = None

        if spec.scoring == "accuracy":
            return float(accuracy_score(y_true, pred_classes))
        elif spec.scoring in ("f1_weighted", "f1"):
            return float(f1_score(y_true, pred_classes, average="weighted", zero_division=0))
        elif spec.scoring == "roc_auc" and probs is not None:
            try:
                if len(classes) == 2:
                    return float(roc_auc_score(y_true, probs[:, 1]))
                else:
                    return float(roc_auc_score(y_true, probs, multi_class="ovr", average="weighted"))
            except Exception:
                return float(f1_score(y_true, pred_classes, average="weighted", zero_division=0))
        else:
            # Fallback to Evaluator
            res = Evaluator.evaluate_classification(y_true, pred_classes, probs)
            val = res.get(metric_name)
            return float(val) if val is not None else float(accuracy_score(y_true, pred_classes))

    # Regression
    preds = preds_or_probs
    if spec.scoring in ("neg_mean_squared_error", "mse"):
        # sklearn negates mse so higher is better
        return float(-mean_squared_error(y_true, preds))
    elif spec.scoring in ("neg_root_mean_squared_error", "rmse"):
        return float(-np.sqrt(mean_squared_error(y_true, preds)))
    elif spec.scoring in ("neg_mean_absolute_error", "mae"):
        return float(-mean_absolute_error(y_true, preds))
    elif spec.scoring == "r2":
        return float(r2_score(y_true, preds))
    else:
        res = Evaluator.evaluate_regression(y_true, preds)
        val = res.get(metric_name)
        return float(val) if val is not None else float(r2_score(y_true, preds))


def optimize_ensemble_weights(
    oof_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    problem_type: str,
    metric_name: str,
    classes: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """Optimize ensemble weights strictly on OOF predictions using SLSQP simplex optimization.

    Returns:
        weights: np.ndarray of shape (M,) summing to 1.0, >= 0.
        best_score: The optimal metric score achieved on OOF predictions.
    """
    model_names = list(oof_dict.keys())
    m = len(model_names)
    if m == 1:
        score = _score_predictions(y_true, oof_dict[model_names[0]], problem_type, metric_name, classes)
        return np.array([1.0], dtype=np.float64), score

    # Check individual model scores to find the strongest baseline starting point
    individual_scores = []
    for name in model_names:
        s = _score_predictions(y_true, oof_dict[name], problem_type, metric_name, classes)
        individual_scores.append(s)

    best_single_idx = int(np.argmax(individual_scores))
    uniform_weights = np.full(m, 1.0 / m, dtype=np.float64)

    # Objective function to MINIMIZE (sklearn scoring is 'higher is better', so negate)
    def loss_func(weights_raw):
        w = np.clip(weights_raw, 0, None)
        s = np.sum(w)
        if s > 0:
            w /= s
        else:
            w = uniform_weights

        if problem_type == "classification":
            fused = np.zeros_like(oof_dict[model_names[0]])
            for i, name in enumerate(model_names):
                fused += w[i] * oof_dict[name]
        else:
            fused = np.zeros_like(oof_dict[model_names[0]])
            for i, name in enumerate(model_names):
                fused += w[i] * oof_dict[name]

        score = _score_predictions(y_true, fused, problem_type, metric_name, classes)
        return -score  # Minimize negative score

    # Bounds: w_i in [0, 1]
    bounds = [(0.0, 1.0) for _ in range(m)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    best_w = uniform_weights
    best_loss = loss_func(uniform_weights)

    # Try starting from: 1) uniform weights, 2) best single model
    initial_guesses = [
        uniform_weights,
    ]
    single_start = np.zeros(m, dtype=np.float64)
    single_start[best_single_idx] = 1.0
    initial_guesses.append(single_start)

    for init_w in initial_guesses:
        try:
            res = minimize(
                loss_func,
                init_w,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 100, "ftol": 1e-6},
            )
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_w = res.x
        except Exception as e:
            logger.debug(f"SLSQP trial optimization failed: {e}")

    # Ensure clean simplex projection
    best_w = np.clip(best_w, 0.0, None)
    s = np.sum(best_w)
    if s > 0:
        best_w /= s
    else:
        best_w = uniform_weights

    opt_score = -loss_func(best_w)
    return best_w, float(opt_score)


# =========================================================================
# 4. Ensemble Candidate Construction & CV Evaluation
# =========================================================================

def build_ensemble_candidates(
    candidate_results: List[TrainingResult],
    model_definitions: List[Any],
    X_train: Any,
    y_train: Any,
    problem_type: str,
    primary_metric: str,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[TrainingResult, EnsembleModel]]:
    """Build and evaluate genuine ensemble candidates using OOF predictions on training data.

    Returns:
        List of (TrainingResult, EnsembleModel) pairs for candidates that beat or compete with baselines.
    """
    valid_results = [r for r in candidate_results if r.status == "COMPLETED" and r.trained_model is not None]
    if len(valid_results) < 2:
        logger.info("Fewer than 2 trained candidate models; model fusion skipped.")
        return []

    y_arr = np.asarray(y_train)

    # 1. Generate Out-of-Fold predictions strictly on training split
    oof_dict, splits, classes = generate_oof_predictions(
        candidate_results=valid_results,
        model_definitions=model_definitions,
        X_train=X_train,
        y_train=y_train,
        problem_type=problem_type,
        n_folds=n_folds,
        seed=seed,
    )

    if len(oof_dict) < 2:
        logger.info(f"OOF predictions succeeded for only {len(oof_dict)} models; fusion skipped.")
        return []

    member_names = list(oof_dict.keys())
    member_models = [next(r.trained_model for r in valid_results if r.model_name == name) for name in member_names]

    ensemble_candidates: List[Tuple[TrainingResult, EnsembleModel]] = []

    # ---------------------------------------------------------------------
    # Strategy A: Soft Voting / Uniform Average
    # ---------------------------------------------------------------------
    strategy_a_name = "ensemble_soft_voting" if problem_type == "classification" else "ensemble_uniform_avg"
    strategy_a_display = "Ensemble (Soft Voting)" if problem_type == "classification" else "Ensemble (Uniform Average)"
    uniform_weights = np.full(len(member_names), 1.0 / len(member_names), dtype=np.float64)

    # Compute fold CV scores for Strategy A
    fold_scores_a = []
    for train_idx, val_idx in splits:
        y_val = y_arr[val_idx]
        if problem_type == "classification":
            fused_val = np.zeros_like(oof_dict[member_names[0]][val_idx])
            for i, name in enumerate(member_names):
                fused_val += uniform_weights[i] * oof_dict[name][val_idx]
        else:
            fused_val = np.zeros_like(oof_dict[member_names[0]][val_idx])
            for i, name in enumerate(member_names):
                fused_val += uniform_weights[i] * oof_dict[name][val_idx]

        f_score = _score_predictions(y_val, fused_val, problem_type, primary_metric, classes)
        fold_scores_a.append(round(float(f_score), 6))

    mean_cv_a = float(np.mean(fold_scores_a))
    model_a = EnsembleModel(
        problem_type=problem_type,
        strategy="soft_voting" if problem_type == "classification" else "uniform_average",
        member_names=member_names,
        member_models=member_models,
        weights=uniform_weights,
        classes=classes,
    )

    provenance_a = {
        "is_ensemble": True,
        "strategy": model_a.strategy,
        "members": member_names,
        "weights": model_a.weights_dict,
        "best_cv_score": mean_cv_a,
        "metric": primary_metric,
        "n_folds_used": n_folds,
        "seed": seed,
        "status": "COMPLETED",
        "evidence_source": "out_of_fold_training_predictions",
        "holdout_used_for_selection": False,
    }

    res_a = TrainingResult(
        model_name=strategy_a_name,
        display_name=strategy_a_display,
        status="COMPLETED",
        cv_scores=fold_scores_a,
        optimization=provenance_a,
        trained_model=model_a,
    )
    ensemble_candidates.append((res_a, model_a))

    # ---------------------------------------------------------------------
    # Strategy B: Weighted Ensemble (Optimization on OOF)
    # ---------------------------------------------------------------------
    strategy_b_name = "ensemble_weighted"
    strategy_b_display = "Ensemble (Optimized Weights)"
    opt_weights, opt_score = optimize_ensemble_weights(
        oof_dict=oof_dict,
        y_true=y_arr,
        problem_type=problem_type,
        metric_name=primary_metric,
        classes=classes,
        seed=seed,
    )

    # Compute fold CV scores for Strategy B
    fold_scores_b = []
    for train_idx, val_idx in splits:
        y_val = y_arr[val_idx]
        if problem_type == "classification":
            fused_val = np.zeros_like(oof_dict[member_names[0]][val_idx])
            for i, name in enumerate(member_names):
                fused_val += opt_weights[i] * oof_dict[name][val_idx]
        else:
            fused_val = np.zeros_like(oof_dict[member_names[0]][val_idx])
            for i, name in enumerate(member_names):
                fused_val += opt_weights[i] * oof_dict[name][val_idx]

        f_score = _score_predictions(y_val, fused_val, problem_type, primary_metric, classes)
        fold_scores_b.append(round(float(f_score), 6))

    mean_cv_b = float(np.mean(fold_scores_b))
    model_b = EnsembleModel(
        problem_type=problem_type,
        strategy="weighted_average",
        member_names=member_names,
        member_models=member_models,
        weights=opt_weights,
        classes=classes,
    )

    provenance_b = {
        "is_ensemble": True,
        "strategy": "weighted_average",
        "members": member_names,
        "weights": model_b.weights_dict,
        "best_cv_score": mean_cv_b,
        "metric": primary_metric,
        "n_folds_used": n_folds,
        "seed": seed,
        "status": "COMPLETED",
        "evidence_source": "out_of_fold_training_predictions",
        "holdout_used_for_selection": False,
    }

    res_b = TrainingResult(
        model_name=strategy_b_name,
        display_name=strategy_b_display,
        status="COMPLETED",
        cv_scores=fold_scores_b,
        optimization=provenance_b,
        trained_model=model_b,
    )
    ensemble_candidates.append((res_b, model_b))

    # ---------------------------------------------------------------------
    # Strategy C: Stacking (Regularized Meta-Learner on OOF)
    # ---------------------------------------------------------------------
    try:
        # Build OOF meta-features matrix
        oof_meta_features = []
        for name in member_names:
            preds = oof_dict[name]
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            oof_meta_features.append(preds)
        X_meta = np.hstack(oof_meta_features)

        if problem_type == "classification":
            # RidgeClassifier or LogisticRegression with cross-validation on OOF
            meta_clf = LogisticRegression(C=1.0, max_iter=200, random_state=seed)
        else:
            meta_clf = Ridge(alpha=1.0, random_state=seed)

        # Cross-validate the meta-learner on the OOF training data
        fold_scores_c = []
        for train_idx, val_idx in splits:
            X_m_tr, X_m_va = X_meta[train_idx], X_meta[val_idx]
            y_m_tr, y_m_va = y_arr[train_idx], y_arr[val_idx]
            meta_fold = clone(meta_clf)
            meta_fold.fit(X_m_tr, y_m_tr)

            if problem_type == "classification" and hasattr(meta_fold, "predict_proba"):
                fused_val = meta_fold.predict_proba(X_m_va)
            else:
                fused_val = meta_fold.predict(X_m_va)

            f_score = _score_predictions(y_m_va, fused_val, problem_type, primary_metric, classes)
            fold_scores_c.append(round(float(f_score), 6))

        mean_cv_c = float(np.mean(fold_scores_c))

        # Fit meta-learner on full OOF matrix
        meta_clf.fit(X_meta, y_arr)

        model_c = EnsembleModel(
            problem_type=problem_type,
            strategy="stacking",
            member_names=member_names,
            member_models=member_models,
            meta_learner=meta_clf,
            classes=classes,
        )

        provenance_c = {
            "is_ensemble": True,
            "strategy": "stacking",
            "meta_learner": type(meta_clf).__name__,
            "members": member_names,
            "weights": {name: round(1.0 / len(member_names), 4) for name in member_names},
            "best_cv_score": mean_cv_c,
            "metric": primary_metric,
            "n_folds_used": n_folds,
            "seed": seed,
            "status": "COMPLETED",
            "evidence_source": "out_of_fold_training_predictions",
            "holdout_used_for_selection": False,
        }

        res_c = TrainingResult(
            model_name="ensemble_stacking",
            display_name="Ensemble (Stacking)",
            status="COMPLETED",
            cv_scores=fold_scores_c,
            optimization=provenance_c,
            trained_model=model_c,
        )
        ensemble_candidates.append((res_c, model_c))
    except Exception as e:
        logger.warning(f"Stacking candidate construction failed: {e}")

    return ensemble_candidates
