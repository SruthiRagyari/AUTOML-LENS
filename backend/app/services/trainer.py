"""Training manager - orchestrates the full ML pipeline."""
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from app.services.evaluator import Evaluator, get_scoring_string, is_higher_better
from app.services.optimizer import OptunaOptimizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model_name: str
    display_name: str
    status: str = "QUEUED"
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    optimized_metrics: dict[str, Any] = field(default_factory=dict)
    best_params: dict[str, Any] = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)
    training_time: float = 0.0
    prediction_time: float = 0.0
    optimization_history: list[dict] = field(default_factory=list)
    trained_model: Any = None


class TrainingManager:
    """Manages model training, optimization, and evaluation."""

    def __init__(self, X_train, X_test, y_train, y_test,
                 model_definitions: list, problem_type: str,
                 primary_metric: str, n_folds: int = 5,
                 n_trials: int = 20, mode: str = "automl"):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_definitions = model_definitions
        self.problem_type = problem_type
        self.primary_metric = primary_metric
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.mode = mode

    def _get_cv(self):
        if self.problem_type == "classification":
            return StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        return KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

    def _evaluate(self, model, X, y):
        y_pred = model.predict(X)
        y_proba = None
        if self.problem_type == "classification" and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X)
            except Exception:
                pass
        if self.problem_type == "classification":
            return Evaluator.evaluate_classification(y, y_pred, y_proba)
        return Evaluator.evaluate_regression(y, y_pred)

    def train_all(
        self, progress_callback: Optional[Callable] = None
    ) -> list[TrainingResult]:
        """Train all models with baseline + optional optimization."""
        results = []
        total = len(self.model_definitions)

        for idx, mdef in enumerate(self.model_definitions):
            pct = (idx / total) * 100
            if progress_callback:
                progress_callback(mdef.display_name, "RUNNING", pct)

            result = TrainingResult(
                model_name=mdef.name, display_name=mdef.display_name
            )

            try:
                # 1. Baseline training
                model = mdef.create_model()
                t0 = time.time()
                model.fit(self.X_train, self.y_train)
                train_time = time.time() - t0

                t0 = time.time()
                baseline_metrics = self._evaluate(model, self.X_test, self.y_test)
                pred_time = time.time() - t0

                result.baseline_metrics = baseline_metrics
                result.training_time = round(train_time, 4)
                result.prediction_time = round(pred_time, 4)
                best_model = model

                # 2. Optuna optimization (if not baseline mode)
                if self.mode != "baseline" and mdef.search_space is not None:
                    if progress_callback:
                        progress_callback(mdef.display_name, "OPTIMIZING", pct + (100 / total * 0.3))

                    optimizer = OptunaOptimizer(
                        X_train=self.X_train, y_train=self.y_train,
                        model_definition=mdef, problem_type=self.problem_type,
                        metric_name=self.primary_metric,
                        n_trials=self.n_trials, n_folds=self.n_folds,
                    )
                    opt_result = optimizer.optimize()
                    result.best_params = opt_result.best_params
                    result.optimization_history = opt_result.trials_data

                    # 3. Retrain with best params
                    if opt_result.best_params:
                        opt_model = mdef.create_model(**opt_result.best_params)
                        t0 = time.time()
                        opt_model.fit(self.X_train, self.y_train)
                        result.training_time += round(time.time() - t0, 4)

                        t0 = time.time()
                        opt_metrics = self._evaluate(opt_model, self.X_test, self.y_test)
                        result.prediction_time += round(time.time() - t0, 4)
                        result.optimized_metrics = opt_metrics
                        best_model = opt_model

                # 4. Cross-validation on best model
                scoring = get_scoring_string(self.primary_metric)
                cv = self._get_cv()
                try:
                    cv_scores = cross_val_score(
                        best_model, self.X_train, self.y_train,
                        cv=cv, scoring=scoring, n_jobs=1
                    )
                    result.cv_scores = [round(float(s), 6) for s in cv_scores]
                except Exception as e:
                    logger.warning(f"CV scoring failed for {mdef.name}: {e}")

                result.trained_model = best_model
                result.status = "COMPLETED"

                if progress_callback:
                    progress_callback(mdef.display_name, "COMPLETED", pct + (100 / total))

            except Exception as e:
                logger.exception(f"Training failed for {mdef.name}: {e}")
                result.status = "FAILED"
                if progress_callback:
                    progress_callback(mdef.display_name, "FAILED", pct + (100 / total))

            results.append(result)

        return results

    @staticmethod
    def select_best_model(
        results: list[TrainingResult], metric: str
    ) -> Optional[TrainingResult]:
        """Select the best model based on actual metric values."""
        completed = [r for r in results if r.status == "COMPLETED"]
        if not completed:
            return None

        higher = is_higher_better(metric)

        def get_score(r: TrainingResult) -> float:
            metrics = r.optimized_metrics if r.optimized_metrics else r.baseline_metrics
            val = metrics.get(metric)
            if val is None:
                # Try common alternatives
                for alt in [metric, f"{metric}_weighted", metric.replace("neg_", "")]:
                    val = metrics.get(alt)
                    if val is not None:
                        break
            if val is None:
                return float("-inf") if higher else float("inf")
            return float(val)

        completed.sort(key=get_score, reverse=higher)
        return completed[0]
