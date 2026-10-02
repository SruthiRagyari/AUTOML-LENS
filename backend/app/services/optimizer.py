"""Optuna hyperparameter optimization engine.

Design notes
------------
* **No holdout access.** The optimizer only ever sees the matrices it is given
  (``X_train``/``y_train``). The caller splits train/test *before* calling in,
  so CV folds are drawn exclusively from training rows and the holdout can
  never influence tuning.
* **Explicit direction.** The study direction is derived from
  :func:`app.services.evaluator.get_metric_spec` rather than hardcoded, making
  ``metric -> sklearn scoring -> Optuna direction`` a single testable contract.
  sklearn negates loss scorers, so the study maximizes the negated value while
  the reported metric keeps its natural "lower is better" meaning.
* **Honest failure.** A trial that raises is recorded with its real error and
  pruned. If *every* trial fails, ``best_score`` stays ``None`` and
  ``best_params`` stays empty - never a fabricated 0.0.
* **Deterministic.** The TPE sampler is seeded, so re-running an experiment
  explores the same hyperparameters.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from app.services.evaluator import get_metric_spec

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _fire(cb: Optional[Callable], trial_number: int) -> None:
    """Invoke a progress callback; never let it break the search."""
    if cb is None:
        return
    try:
        cb(trial_number)
    except Exception:  # pragma: no cover - defensive
        logger.warning("trial progress callback failed", exc_info=True)


def _fire2(cb: Optional[Callable[[int, bool], None]], trial_number: int,
           ok: bool) -> None:
    if cb is None:
        return
    try:
        cb(trial_number, ok)
    except Exception:  # pragma: no cover - defensive
        logger.warning("trial done callback failed", exc_info=True)

# Fixed seed so a re-run of the same experiment replays the same search.
# Recorded in the provenance block for reproducibility.
OPTUNA_SEED = 42

# Status values persisted per trained model.
STATUS_COMPLETED = "COMPLETED"   # every requested trial finished
STATUS_PARTIAL = "PARTIAL"       # at least one trial succeeded, some failed
STATUS_FAILED = "FAILED"         # no trial produced a score
STATUS_SKIPPED = "SKIPPED"       # optimization intentionally not run


@dataclass
class OptimizationResult:
    best_params: dict[str, Any]
    best_score: Optional[float]
    n_trials: int
    trials_data: list[dict[str, Any]] = field(default_factory=list)
    optimization_time: float = 0.0
    # ---- explicit provenance (persisted alongside the model result) -------
    metric: str = ""
    scoring: str = ""
    direction: str = "maximize"          # direction of the Optuna study
    raw_direction: str = "maximize"      # direction of the user-facing metric
    n_trials_requested: int = 0
    n_trials_completed: int = 0
    n_trials_failed: int = 0
    n_folds_requested: int = 0
    n_folds_used: int = 0
    cv_strategy: str = ""
    seed: int = OPTUNA_SEED
    status: str = STATUS_COMPLETED
    error: Optional[str] = None
    fold_note: Optional[str] = None

    def to_provenance(self) -> dict[str, Any]:
        """JSON-safe provenance block persisted with the experiment."""
        return {
            "metric": self.metric,
            "scoring": self.scoring,
            "direction": self.direction,
            "raw_direction": self.raw_direction,
            "n_trials_requested": self.n_trials_requested,
            "n_trials_completed": self.n_trials_completed,
            "n_trials_failed": self.n_trials_failed,
            "n_folds_requested": self.n_folds_requested,
            "n_folds_used": self.n_folds_used,
            "cv_strategy": self.cv_strategy,
            "seed": self.seed,
            # None when every trial failed - never substituted with 0.
            "best_cv_score": self.best_score,
            "best_params": self.best_params,
            "status": self.status,
            # Real message from the failure, or None. Never invented.
            "error": self.error,
            "fold_note": self.fold_note,
            "optimization_time": self.optimization_time,
        }



class OptunaOptimizer:
    """Hyperparameter optimizer using Optuna with cross-validation.

    ``X_train``/``y_train`` must be the training split only; the holdout set is
    never passed in, so tuning cannot leak test information.
    """

    def __init__(self, X_train, y_train, model_definition, problem_type: str,
                 metric_name: str, n_trials: int = 20, n_folds: int = 5,
                 seed: int = OPTUNA_SEED):
        self.X_train = X_train
        self.y_train = y_train
        self.model_definition = model_definition
        self.problem_type = problem_type
        self.metric_name = metric_name
        self.n_trials = max(1, int(n_trials))
        self.n_folds = int(n_folds)
        self.seed = int(seed)
        # Resolved once in __init__ so a bad metric fails fast and loudly.
        self.spec = get_metric_spec(metric_name)

    # -- CV -----------------------------------------------------------------
    def _resolve_folds(self) -> tuple[int, Optional[str]]:
        """Clamp the fold count to what the training split actually supports.

        StratifiedKFold needs at least as many folds as the rarest class has
        samples. Rather than letting the study crash, clamp and record a note.
        """
        n_samples = len(self.y_train)
        requested = self.n_folds
        usable = max(2, requested)
        note = None
        if self.problem_type == "classification":
            classes, counts = np.unique(np.asarray(self.y_train), return_counts=True)
            if len(classes) > 1:
                usable = min(usable, int(counts.min()))
        usable = min(usable, n_samples)
        if usable < 2:
            usable = 2
            note = (f"only {n_samples} training row(s) available; "
                    f"{requested}-fold CV is not possible, used 2 folds")
        elif usable != requested:
            note = (f"requested {requested} folds but the training split supports "
                    f"{usable}; clamped to {usable}")
        return usable, note

    def _get_cv(self, n_splits: Optional[int] = None):
        if self.problem_type == "classification":
            return StratifiedKFold(n_splits=n_splits or self.n_folds, shuffle=True,
                                   random_state=self.seed)
        return KFold(n_splits=n_splits or self.n_folds, shuffle=True,
                     random_state=self.seed)

    def _get_direction(self) -> str:
        """Study direction, derived from the metric -> scoring mapping."""
        return self.spec.direction
    # -- main ---------------------------------------------------------------
    def optimize(
        self,
        trial_callback: Optional[Callable[[int], None]] = None,
        trial_done_callback: Optional[Callable[[int, bool], None]] = None,
    ) -> OptimizationResult:
        """Run the study.

        ``trial_callback(trial_number)`` fires when Optuna genuinely starts a
        trial and ``trial_done_callback(trial_number, ok)`` when it finishes, so
        a progress display can name the trial actually running. Both are
        best-effort: a raising callback must never break the search, so failures
        are swallowed after being logged.
        """
        scoring = self.spec.scoring
        n_folds_used, fold_note = self._resolve_folds()
        cv = self._get_cv(n_folds_used)
        cv_strategy = (
            "StratifiedKFold" if self.problem_type == "classification" else "KFold"
        )
        trials_data: list[dict[str, Any]] = []
        trial_errors: list[str] = []

        def objective(trial):
            t0 = time.time()
            _fire(trial_callback, trial.number)
            ok = False
            try:
                params = self.model_definition.search_space(trial)
                model = self.model_definition.create_model(**params)
                scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=cv, scoring=scoring, n_jobs=1
                )
                score = float(scores.mean())
                if not np.isfinite(score):
                    raise ValueError(f"non-finite CV score: {score}")
                trials_data.append({
                    "trial_number": trial.number,
                    "params": params,
                    "score": score,
                    "duration": round(time.time() - t0, 3),
                    "status": "COMPLETE",
                    "error": None,
                })
                ok = True
                return score
            except Exception as e:
                message = f"{type(e).__name__}: {e}"
                logger.warning(
                    f"Trial {trial.number} failed for "
                    f"{self.model_definition.name}: {message}"
                )
                trial_errors.append(message)
                trials_data.append({
                    "trial_number": trial.number,
                    "params": {},
                    "score": None,          # never fabricate a score
                    "duration": round(time.time() - t0, 3),
                    "status": "FAIL",
                    "error": message,       # the real error, verbatim
                })
                raise optuna.TrialPruned()
            finally:
                _fire2(trial_done_callback, trial.number, ok)

        t_start = time.time()
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction=self._get_direction(), sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        opt_time = time.time() - t_start

        # Accessing study.best_trial raises ValueError when no trial completed,
        # so inspect the trial states explicitly instead.
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        n_completed = len(completed)
        n_failed = len(study.trials) - n_completed
        best_params = study.best_params if completed else {}
        # None (not a fabricated 0.0) when no trial completed successfully.
        best_score: Optional[float] = float(study.best_value) if completed else None

        if n_completed == 0:
            status = STATUS_FAILED
            # Real first error, or an honest "no trial completed".
            error = trial_errors[0] if trial_errors else "no trial completed"
        elif n_failed > 0:
            status = STATUS_PARTIAL
            # Trials failed but a real best value exists; note why it is partial.
            error = f"{n_failed} of {self.n_trials} trials failed: {trial_errors[0]}" \
                if trial_errors else None
        else:
            status = STATUS_COMPLETED
            error = None

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=self.n_trials,
            trials_data=trials_data,
            optimization_time=round(opt_time, 3),
            metric=self.spec.metric,
            scoring=scoring,
            direction=self.spec.direction,
            raw_direction=self.spec.raw_direction,
            n_trials_requested=self.n_trials,
            n_trials_completed=n_completed,
            n_trials_failed=n_failed,
            n_folds_requested=self.n_folds,
            n_folds_used=n_folds_used,
            cv_strategy=cv_strategy,
            seed=self.seed,
            status=status,
            error=error,
            fold_note=fold_note,
        )


def skipped_optimization(metric_name: str, n_folds: int, reason: str,
                         seed: int = OPTUNA_SEED) -> OptimizationResult:
    """Provenance for a run where optimization was deliberately not executed.

    Used by ``mode='baseline'`` so the persisted record distinguishes
    "not optimized" from "optimized and failed".
    """
    spec = get_metric_spec(metric_name)
    return OptimizationResult(
        best_params={},
        best_score=None,
        n_trials=0,
        trials_data=[],
        optimization_time=0.0,
        metric=spec.metric,
        scoring=spec.scoring,
        direction=spec.direction,
        raw_direction=spec.raw_direction,
        n_trials_requested=0,
        n_trials_completed=0,
        n_trials_failed=0,
        n_folds_requested=n_folds,
        n_folds_used=0,
        cv_strategy="",
        seed=seed,
        status=STATUS_SKIPPED,
        error=None,
        fold_note=reason,
    )
