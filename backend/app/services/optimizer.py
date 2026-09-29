"""Optuna hyperparameter optimization engine."""
import time
import logging
from dataclasses import dataclass, field
from typing import Any
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from app.services.evaluator import get_scoring_string, is_higher_better

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationResult:
    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    trials_data: list[dict[str, Any]] = field(default_factory=list)
    optimization_time: float = 0.0


class OptunaOptimizer:
    """Hyperparameter optimizer using Optuna with cross-validation."""

    def __init__(self, X_train, y_train, model_definition, problem_type: str,
                 metric_name: str, n_trials: int = 20, n_folds: int = 5):
        self.X_train = X_train
        self.y_train = y_train
        self.model_definition = model_definition
        self.problem_type = problem_type
        self.metric_name = metric_name
        self.n_trials = n_trials
        self.n_folds = n_folds

    def _get_cv(self):
        if self.problem_type == "classification":
            return StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        return KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

    def _get_direction(self) -> str:
        scoring = get_scoring_string(self.metric_name)
        # sklearn neg_* scoring: higher is better (less negative = better)
        # For standard metrics like accuracy, f1: higher is better
        # So Optuna should always maximize the sklearn scoring output
        return "maximize"

    def optimize(self) -> OptimizationResult:
        cv = self._get_cv()
        scoring = get_scoring_string(self.metric_name)
        trials_data = []

        def objective(trial):
            t0 = time.time()
            try:
                params = self.model_definition.search_space(trial)
                model = self.model_definition.create_model(**params)
                scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=cv, scoring=scoring, n_jobs=1
                )
                score = float(scores.mean())
                trials_data.append({
                    "trial_number": trial.number,
                    "params": params,
                    "score": score,
                    "duration": round(time.time() - t0, 3),
                    "status": "COMPLETE",
                })
                return score
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed for {self.model_definition.name}: {e}")
                trials_data.append({
                    "trial_number": trial.number,
                    "params": {},
                    "score": None,
                    "duration": round(time.time() - t0, 3),
                    "status": "FAIL",
                })
                raise optuna.TrialPruned()

        t_start = time.time()
        study = optuna.create_study(direction=self._get_direction())
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        opt_time = time.time() - t_start

        best_params = study.best_params if study.best_trial else {}
        best_score = study.best_value if study.best_trial else 0.0

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=self.n_trials,
            trials_data=trials_data,
            optimization_time=round(opt_time, 3),
        )
