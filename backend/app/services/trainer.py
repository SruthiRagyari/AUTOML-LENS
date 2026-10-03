"""Training manager - orchestrates the full ML pipeline.

CV methodology (leakage-safe)
-----------------------------
When a raw training frame + :class:`FeatureSpec` are supplied, every CV fold
and every Optuna trial wraps its candidate model in a fold-safe pipeline that
fits feature engineering + preprocessing on that fold's training rows only.
When only pre-transformed matrices are supplied (legacy callers/tests), CV
operates on the matrices as before and the public API is unchanged.
"""
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from app.services.evaluator import (
    Evaluator, get_metric_spec,
)
from app.services.fold_safe import (
    FeatureSpec, build_final_pipeline, build_fold_safe_pipeline,
)
from app.services.optimizer import (
    OptunaOptimizer, OptimizationResult, OPTUNA_SEED,
    skipped_optimization,
)
from app.services.training_progress import (
    TrainingProgress, PHASE_TRAINING, PHASE_OPTIMIZING, PHASE_REFINING,
    PHASE_SCORING, MODEL_COMPLETED, MODEL_FAILED,
)

logger = logging.getLogger(__name__)

# -- Deterministic tie-breaking -------------------------------------------
# Two CV scores are treated as TIED when they agree to within this tolerance.
# It encodes only the numerical resolution of the CV arithmetic (float64
# accumulation over n folds). It is NOT a claim that differences below it are
# statistically insignificant: no confidence interval or significance test is
# computed here, so a difference larger than the tolerance is simply ranked
# first, and a difference inside it is declared indistinguishable rather than
# resolved by pretending to more precision than the CV scores support.
TIE_REL_TOL = 1e-9
TIE_ABS_TOL = 1e-12

TIE_BREAK_RULE = (
    "1) best CV score according to metric direction (the ranking score is "
    "always the sklearn scorer value, which is 'higher is better' because "
    "neg_* losses are pre-negated, so a lower raw loss scores higher); "
    "2) on a tie, lowest CV fold standard deviation "
    "(most stable across folds); 3) still tied, most CV folds evaluated "
    "(most evidence); 4) still tied, lexicographically smallest model_name. "
    "All evidence comes from the training split; holdout metrics are never "
    "consulted, not even to break a tie."
)



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
    # Real Optuna provenance for THIS model: metric/scoring/direction, trial
    # counts, folds, best CV score, status and the real error when it failed.
    optimization: dict = field(default_factory=dict)
    trained_model: Any = None
    error_message: Optional[str] = None


class TrainingManager:
    """Manages model training, optimization, and evaluation."""

    def __init__(self, X_train, X_test, y_train, y_test,
                 model_definitions: list, problem_type: str,
                 primary_metric: str, n_folds: int = 5,
                 n_trials: int = 20, mode: str = "automl",
                 optimization_seed: int = OPTUNA_SEED,
                 fast_demo: bool = False,
                 feature_spec: Optional[FeatureSpec] = None,
                 X_train_raw: Optional[pd.DataFrame] = None,
                 X_test_raw: Optional[pd.DataFrame] = None):
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
        # Deterministic Optuna sampler seed, recorded in the provenance block.
        self.optimization_seed = optimization_seed
        # fast_demo shrinks trials/folds upstream; flag it so the persisted
        # record never implies a full-budget search was run.
        self.fast_demo = fast_demo
        # metric -> sklearn scoring -> direction, resolved once and reused.
        self.metric_spec = get_metric_spec(primary_metric)
        # Optional leakage-safe inputs: when both a FeatureSpec and the RAW
        # training frame (with the target column) are supplied, CV, Optuna and
        # the final refit all operate on fold-safe pipelines that refit
        # feature engineering + preprocessing per fold-training split.
        # Legacy matrix-only callers are unaffected.
        self.feature_spec = feature_spec
        self.X_train_raw = X_train_raw
        self.X_test_raw = X_test_raw
        # Filled during train_all(): the fitted fold-safe pipeline of the
        # selected winner (final refit on the complete training portion).
        self.final_pipeline_: Any = None

    @property
    def fold_safe_cv(self) -> bool:
        """True when CV/Optuna refit preprocessing inside every fold."""
        return self.feature_spec is not None and self.X_train_raw is not None

    def _candidate(self, model):
        """Wrap a model in a fold-safe pipeline when the raw frame is known."""
        if self.fold_safe_cv:
            return build_fold_safe_pipeline(model, self.feature_spec)
        return model

    def _cv_X_y(self):
        """CV inputs: raw frame in fold-safe mode, matrices otherwise."""
        if self.fold_safe_cv:
            return self._frame_with_target(self.X_train_raw, self.y_train), self.y_train
        return self.X_train, self.y_train

    def _fit_frame(self) -> Optional[pd.DataFrame]:
        """Frame the fold-safe ``features`` step can fit on (target included)."""
        if not self.fold_safe_cv:
            return None
        return self._frame_with_target(self.X_train_raw, self.y_train)

    def _test_frame(self):
        """Frame the fitted fold-safe pipeline can score/predict on."""
        if self.fold_safe_cv:
            return self._frame_with_target(self.X_test_raw, self.y_test)
        return self.X_test

    @staticmethod
    def _frame_with_target(X_raw, y) -> pd.DataFrame:
        """Return a DataFrame whose rows pair features with the target column.

        Experiments pass the raw frames with the target already present; in that
        case they are returned untouched. Matrix-only callers instead pass the
        target name through the FeatureSpec, so the series is (re)attached
        positionally to keep per-fold fitting possible.
        """
        if isinstance(X_raw, pd.DataFrame):
            return X_raw
        raise TypeError(
            "Fold-safe training requires pandas DataFrames with the target "
            f"column present (got {type(X_raw).__name__})."
        )

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
        self,
        progress_callback: Optional[Callable] = None,
        progress: Optional[TrainingProgress] = None,
    ) -> list[TrainingResult]:
        """Train all models with baseline + optional optimization.

        ``progress`` is a :class:`TrainingProgress` tracker updated with real
        state as each stage actually begins (which model, which phase, which
        Optuna trial). It carries no percentage and no ETA on purpose.

        ``progress_callback`` is the legacy ``(name, status, pct)`` hook. It is
        still invoked, but the percentage it receives is the exact fraction of
        models finished (``n_finished / n_total``) rather than an interpolation
        of assumed per-stage cost.
        """
        results = []
        total = len(self.model_definitions)
        n_finished = 0

        def _legacy(name: str, status: str) -> None:
            if progress_callback:
                progress_callback(name, status, (n_finished / total) * 100 if total else 0.0)

        for idx, mdef in enumerate(self.model_definitions):
            if progress is not None:
                progress.model_started(mdef.name, mdef.display_name, PHASE_TRAINING)
            _legacy(mdef.display_name, "RUNNING")

            result = TrainingResult(
                model_name=mdef.name, display_name=mdef.display_name
            )

            try:
                # 1. Baseline training (training portion only; holdout untouched).
                # Fold-safe mode fits a full FE+preprocessing+model pipeline on
                # the raw training frame; legacy mode fits the matrix directly.
                # Legacy callers may pass numpy matrices; the fold-safe path needs
                # the raw frame, so target columns are (re)attached by position.
                import pandas as _pd  # local import: hot path, keeps header light
                raw_train_df = self._fit_frame()
                model = mdef.create_model()
                baseline_candidate = self._candidate(model)
                t0 = time.time()
                if self.fold_safe_cv:
                    baseline_candidate.fit(raw_train_df, self.y_train)
                    base_test_X, base_test_y = self._test_frame(), self.y_test
                else:
                    baseline_candidate.fit(self.X_train, self.y_train)
                    base_test_X, base_test_y = self.X_test, self.y_test
                train_time = time.time() - t0

                t0 = time.time()
                baseline_metrics = self._evaluate(
                    baseline_candidate, base_test_X, base_test_y)
                pred_time = time.time() - t0

                result.baseline_metrics = baseline_metrics
                result.training_time = round(train_time, 4)
                result.prediction_time = round(pred_time, 4)
                best_model = model
                best_candidate = baseline_candidate

                # 2. Optuna optimization (if not baseline mode)
                if self.mode != "baseline" and mdef.search_space is not None:
                    if progress is not None:
                        progress.model_phase(mdef.name, PHASE_OPTIMIZING)
                    _legacy(mdef.display_name, "OPTIMIZING")

                    # Optuna only ever sees the training portion: CV folds come
                    # from the training split, so the holdout cannot influence
                    # hyperparameter tuning. Fold-safe mode passes the raw frame
                    # + FeatureSpec so every trial refits FE+preprocessing per
                    # fold; legacy mode passes the pre-transformed matrix.
                    try:
                        cv_X, cv_y = self._cv_X_y()
                        optimizer = OptunaOptimizer(
                            X_train=cv_X, y_train=cv_y,
                            model_definition=mdef, problem_type=self.problem_type,
                            metric_name=self.primary_metric,
                            n_trials=self.n_trials, n_folds=self.n_folds,
                            seed=self.optimization_seed,
                            feature_spec=self.feature_spec if self.fold_safe_cv else None,
                        )
                        opt_result = optimizer.optimize(
                            trial_callback=(
                                (lambda n: progress.trial_started(mdef.name, n))
                                if progress is not None else None
                            ),
                            trial_done_callback=(
                                (lambda n, ok: progress.trial_finished(
                                    mdef.name, ok=ok))
                                if progress is not None else None
                            ),
                        )
                    except Exception as e:
                        # One model's search blowing up must not abort the run.
                        # Record the real failure and keep the baseline model.
                        message = f"{type(e).__name__}: {e}"
                        logger.exception(
                            f"Optimization failed for {mdef.name}, continuing "
                            f"with baseline: {message}"
                        )
                        opt_result = skipped_optimization(
                            self.primary_metric, self.n_folds,
                            reason=f"optimization raised: {message}",
                            seed=self.optimization_seed,
                        )
                        opt_result.status = "FAILED"
                        opt_result.error = message

                    result.best_params = opt_result.best_params
                    result.optimization_history = opt_result.trials_data
                    result.optimization = opt_result.to_provenance()
                    result.optimization["fast_demo"] = self.fast_demo

                    # 3. Retrain with best params (training portion only)
                    if opt_result.best_params:
                        if progress is not None:
                            progress.model_phase(mdef.name, PHASE_REFINING)
                        opt_model = mdef.create_model(**opt_result.best_params)
                        opt_candidate = self._candidate(opt_model)
                        t0 = time.time()
                        if self.fold_safe_cv:
                            opt_candidate.fit(self._fit_frame(), self.y_train)
                            opt_test_X = self._test_frame()
                        else:
                            opt_candidate.fit(self.X_train, self.y_train)
                            opt_test_X = self.X_test
                        result.training_time += round(time.time() - t0, 4)

                        t0 = time.time()
                        opt_metrics = self._evaluate(opt_candidate, opt_test_X, self.y_test)
                        result.prediction_time += round(time.time() - t0, 4)
                        result.optimized_metrics = opt_metrics
                        best_model = opt_model
                        best_candidate = opt_candidate

                else:
                    # No search ran (baseline mode, or a model with no search
                    # space). Record that explicitly so the persisted trace
                    # never implies an optimization happened.
                    result.optimization = skipped_optimization(
                        self.primary_metric, self.n_folds,
                        reason=(
                            f"mode={self.mode}" if self.mode == "baseline"
                            else "model has no hyperparameter search space"
                        ),
                        seed=self.optimization_seed,
                    ).to_provenance()
                    result.optimization["fast_demo"] = self.fast_demo

                # 4. Cross-validation on the best model (training portion only).
                # Fold-safe mode: feature engineering + preprocessing are refit
                # inside every fold via the candidate pipeline, so no learned
                # state is shared between fold-train and fold-validation.
                # The scoring string comes from the explicit metric spec.
                scoring = self.metric_spec.scoring
                cv = self._get_cv()
                if progress is not None:
                    progress.model_phase(mdef.name, PHASE_SCORING)
                try:
                    cv_X, cv_y = self._cv_X_y()
                    cv_scores = cross_val_score(
                        best_candidate, cv_X, cv_y,
                        cv=cv, scoring=scoring, n_jobs=1
                    )
                    result.cv_scores = [round(float(s), 6) for s in cv_scores]
                except Exception as e:
                    logger.warning(f"CV scoring failed for {mdef.name}: {e}")

                result.trained_model = best_candidate
                result.status = "COMPLETED"
                n_finished += 1
                if progress is not None:
                    progress.model_finished(mdef.name, MODEL_COMPLETED)
                _legacy(mdef.display_name, "COMPLETED")

            except Exception as e:
                logger.exception(f"Training failed for {mdef.name}: {e}")
                result.status = "FAILED"
                result.error_message = f"{type(e).__name__}: {e}"
                n_finished += 1
                # Keep any real optimization provenance already produced; only
                # synthesize a failure record when none exists yet.
                if not result.optimization:
                    result.optimization = skipped_optimization(
                        self.primary_metric, self.n_folds,
                        reason=f"training failed: {result.error_message}",
                        seed=self.optimization_seed,
                    ).to_provenance()
                    result.optimization["status"] = "FAILED"
                    result.optimization["error"] = result.error_message
                    result.optimization["fast_demo"] = self.fast_demo
                if progress is not None:
                    progress.model_finished(
                        mdef.name, MODEL_FAILED, error=result.error_message
                    )
                _legacy(mdef.display_name, "FAILED")

            results.append(result)

        return results

    @staticmethod
    def build_selection_provenance(
        results: list[TrainingResult], metric: str, selected: Optional[TrainingResult]
    ) -> dict:
        """Persistable record of HOW the winner was chosen.

        Makes explicit that selection used cross-validation on the training
        split, and keeps the final holdout score clearly separate so the two
        can never be confused in a report.
        """
        spec = get_metric_spec(metric)
        cv_score = (
            TrainingManager._selection_cv_score(selected)
            if selected is not None else None
        )
        # Recomputed (deterministically) so the persisted tie-break record can
        # never drift from the selection that actually happened.
        _, tie_break = TrainingManager._select_with_provenance(results, metric)
        return {
            "tie_break": tie_break,
            "selection_metric": metric,
            "selection_scoring": spec.scoring,
            # Direction of the CV evidence used for ranking (always maximize on
            # the sklearn scoring value; neg_* losses are pre-negated).
            "selection_direction": spec.direction,
            # Direction of the raw reported metric, for human-readable reports.
            "selection_raw_direction": spec.raw_direction,
            "evidence_source": "cross_validation_on_training_split",
            "selected_model": selected.model_name if selected else None,
            "selected_model_display_name": selected.display_name if selected else None,
            # CV score that decided the winner - training data only.
            "selected_model_cv_score": cv_score,
            "selected_model_cv_folds": (
                (selected.optimization or {}).get("n_folds_used")
                if selected else None
            ),
            # The holdout metrics of the winner, evaluated ONCE after selection.
            # Never used to choose the model.
            "final_holdout_metrics": (
                (selected.optimized_metrics or selected.baseline_metrics)
                if selected else None
            ),
            "holdout_used_for_selection": False,
            "candidates_considered": [
                {
                    "model_name": r.model_name,
                    "status": r.status,
                    "cv_score": TrainingManager._selection_cv_score(r),
                    # Real CV dispersion, used only to break a tie.
                    "cv_fold_std": TrainingManager._cv_stability(r),
                    "cv_fold_count": TrainingManager._cv_fold_count(r),
                    "selected": selected is not None and r.model_name == selected.model_name,
                }
                for r in results
            ],
            "note": (
                "Selection ranked models by cross-validation on the training "
                "split only. The holdout set was not used to choose the winner."
            ),
        }

    @staticmethod
    def _select_with_provenance(
        results: list[TrainingResult], metric: str
    ) -> tuple[Optional[TrainingResult], dict]:
        """Rank by CV score, recording how any tie was resolved.

        Returns ``(winner, tie_break_provenance)``. The provenance states
        whether a tie occurred, which models were tied, their shared CV score,
        and which criterion decided it, so the decision stays auditable.
        """
        base = {
            "rule": TIE_BREAK_RULE,
            "tolerance": {"rel_tol": TIE_REL_TOL, "abs_tol": TIE_ABS_TOL},
            "tolerance_meaning": (
                "Scores within this distance are the same number to the "
                "precision the CV arithmetic provides; no statistical "
                "significance claim is made."
            ),
            "tie_detected": False,
            "tied_models": [],
            "tied_cv_scores": None,
            "resolved_by": None,
            "winner": None,
            "tie_broken_by_holdout": False,
        }
        candidates = [(r, TrainingManager._selection_cv_score(r))
                      for r in results if r.status == "COMPLETED"]
        scored = [(r, s) for r, s in candidates if s is not None]
        if not scored:
            return None, base

        # sklearn scorers are all "higher is better" once losses are negated,
        # so the CV score always maximizes. get_metric_spec encodes this.
        reverse = get_metric_spec(metric).direction == "maximize"
        ordered = sorted(scored, key=lambda pair: pair[1], reverse=reverse)

        # Everything equal to the best score, within the documented tolerance.
        best_score = ordered[0][1]
        tied = [(r, s) for r, s in ordered
                if math.isclose(s, best_score, rel_tol=TIE_REL_TOL,
                                abs_tol=TIE_ABS_TOL)]

        resolved_by = None
        winner = tied[0][0]
        if len(tied) > 1:
            winner, resolved_by = TrainingManager._tie_break(tied)

        provenance = dict(base)
        provenance.update({
            "tie_detected": len(tied) > 1,
            "tied_models": [r.model_name for r, _ in tied],
            "tied_cv_scores": {r.model_name: float(s) for r, s in tied},
            "resolved_by": resolved_by,
            "winner": winner.model_name,
        })
        return winner, provenance

    @staticmethod
    def _cv_stability(r: TrainingResult) -> Optional[float]:
        """Standard deviation of the model's CV fold scores.

        Used only to break a tie between models sharing the same CV mean, where
        a tighter spread genuinely indicates a less fold-dependent result.
        Returns ``None`` when fewer than two folds were scored, so a model with
        no spread information is ranked last instead of being given a fake 0.0.
        """
        vals = [float(v) for v in (r.cv_scores or [])
                if isinstance(v, (int, float))]
        if len(vals) < 2:
            return None
        return float(np.std(vals))

    @staticmethod
    def _cv_fold_count(r: TrainingResult) -> int:
        return len(r.cv_scores or [])

    @staticmethod
    def _tie_break(
        tied: list[tuple[TrainingResult, float]]
    ) -> tuple[TrainingResult, Optional[str]]:
        """Apply the deterministic cascade to models sharing the best CV score.

        Order: lowest fold standard deviation -> most folds evaluated ->
        lexicographically smallest model_name. Returns the winner and the name
        of the criterion that actually decided it.
        """
        # Criterion 2: most stable across CV folds (lowest spread).
        usable = [(TrainingManager._cv_stability(r), r) for r, _ in tied]
        usable = [(s, r) for s, r in usable if s is not None]
        resolved_by = None
        if usable:
            best_std = min(s for s, _ in usable)
            finalists = [r for s, r in usable
                         if math.isclose(s, best_std, rel_tol=TIE_REL_TOL,
                                         abs_tol=TIE_ABS_TOL)]
            resolved_by = "cv_fold_std"
        else:
            # No tied model has >= 2 folds, so no spread exists to compare.
            finalists = [r for r, _ in tied]

        if len(finalists) > 1:
            # Criterion 3: most folds evaluated (most evidence).
            most = max(TrainingManager._cv_fold_count(r) for r in finalists)
            if any(TrainingManager._cv_fold_count(r) != most for r in finalists):
                finalists = [r for r in finalists
                             if TrainingManager._cv_fold_count(r) == most]
                resolved_by = "cv_fold_count"
            elif resolved_by is None:
                resolved_by = "cv_fold_count"

        if len(finalists) > 1:
            # Criterion 4: fully deterministic, independent of input order.
            finalists.sort(key=lambda r: r.model_name)
            resolved_by = "model_name"

        return finalists[0], resolved_by

    @staticmethod
    def select_best_model(
        results: list[TrainingResult], metric: str
    ) -> Optional[TrainingResult]:
        """Select the winning model using TRAINING/CV evidence only.

        The untouched holdout must never decide which model wins: choosing a
        model by its test score leaks the holdout into selection and inflates
        the reported result. Selection therefore ranks models by a comparable
        cross-validation score computed on the training split.

        Evidence priority per model:
          1. ``optimization.best_cv_score`` - the Optuna CV mean for the tuned
             model (training data only).
          2. ``cv_scores`` mean - CV of the final refit model (training data only).
          3. no CV evidence -> the model cannot be selected on; it is skipped.

        Models with no CV score are never given a fabricated value. Equal
        scores are resolved by :data:`TIE_BREAK_RULE`, which is deterministic
        and never consults the holdout.
        """
        winner, _ = TrainingManager._select_with_provenance(results, metric)
        return winner

    @staticmethod
    def _selection_cv_score(r: TrainingResult) -> Optional[float]:
        """Comparable CV evidence for one model, or None when unavailable."""
        opt = r.optimization or {}
        best_cv = opt.get("best_cv_score")
        if isinstance(best_cv, (int, float)):
            return float(best_cv)
        if r.cv_scores:
            return float(sum(r.cv_scores) / len(r.cv_scores))
        return None
