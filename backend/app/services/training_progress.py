"""Real training progress tracking.

Design constraints
------------------
Everything reported here is a fact about work that has actually started or
finished. In particular this module deliberately provides **no**:

* percentage complete - the work is not linearly divisible (one SVM fit on a
  large frame can outlast every other model combined), so any percentage would
  be an invented number;
* ETA / seconds remaining - predicting the future would require a cost model
  that is never validated;
* synthetic chatter - a stage is only reported once the work it describes has
  genuinely begun.

What is reported instead: the model currently running, the stage it is in, the
Optuna trial number actually being evaluated, how many models finished and how
many failed, and measured elapsed seconds.

The tracker is thread-safe because training runs in a worker thread while the
API keeps answering requests (see ``app.api.experiments.train_experiment``).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Experiment-level lifecycle.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Stage the runner is currently in. Reported verbatim; never interpolated.
PHASE_PREPARING = "preparing"       # reading data, splitting, preprocessing
PHASE_TRAINING = "training"         # baseline fit
PHASE_OPTIMIZING = "optimizing"     # Optuna study running
PHASE_REFINING = "refining"         # refit with the best parameters
PHASE_SCORING = "scoring"           # cross-validation of the refit model
PHASE_SELECTING = "selecting"       # picking the winner from CV evidence
PHASE_PERSISTING = "persisting"     # writing artifacts
PHASE_DONE = "done"

# Per-model states.
MODEL_QUEUED = "queued"
MODEL_RUNNING = "running"
MODEL_COMPLETED = "completed"
MODEL_FAILED = "failed"


@dataclass
class ModelProgress:
    """Per-model state. Every field is observed, never estimated."""

    model_name: str
    display_name: str
    index: int = 0
    status: str = MODEL_QUEUED
    phase: Optional[str] = None
    error: Optional[str] = None
    trials_completed: int = 0
    trials_failed: int = 0
    trial_current: Optional[int] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        elapsed = None
        if self.started_at is not None:
            end = self.finished_at if self.finished_at is not None else time.time()
            elapsed = round(end - self.started_at, 2)
        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "index": self.index,
            "status": self.status,
            "phase": self.phase,
            "error": self.error,
            "trials_completed": self.trials_completed,
            "trials_failed": self.trials_failed,
            "trial_current": self.trial_current,
            "elapsed_seconds": elapsed,
        }


class TrainingProgress:
    """Thread-safe, honest progress for a single training run.

    ``persist`` is an optional callback invoked on coarse transitions (phase
    change, model start/finish) so state also reaches the database. It is
    isolated: a failing persistence hook must never abort a training run.
    """

    def __init__(
        self,
        experiment_id: int,
        model_names: Optional[list] = None,
        n_trials: Optional[int] = None,
        persist: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_trials = n_trials
        self.status = STATUS_QUEUED
        self.phase: Optional[str] = None
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.current_model: Optional[str] = None
        self.models: dict = {}
        self._lock = threading.Lock()
        self._persist = persist

        for i, (name, display) in enumerate(model_names or []):
            self.models[name] = ModelProgress(
                model_name=name, display_name=display, index=i
            )

    def _notify(self) -> None:
        """Persist the current snapshot; failures never propagate."""
        if self._persist is None:
            return
        try:
            self._persist(self.snapshot())
        except Exception:  # pragma: no cover - defensive
            logger.warning("Progress persistence failed", exc_info=True)

    def start(self, phase: str = PHASE_PREPARING) -> None:
        with self._lock:
            self.status = STATUS_RUNNING
            self.phase = phase
            self.started_at = time.time()
        self._notify()

    def set_phase(self, phase: str) -> None:
        """Announce a stage that has genuinely begun."""
        with self._lock:
            self.phase = phase
        self._notify()

    def finish(self, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            self.status = status
            self.error = error
            self.finished_at = time.time()
            if status in (STATUS_COMPLETED, STATUS_FAILED):
                self.phase = PHASE_DONE
            self.current_model = None
        self._notify()

    def model_started(self, model_name: str, display_name: str,
                      phase: str = PHASE_TRAINING) -> None:
        with self._lock:
            model = self.models.get(model_name)
            if model is None:
                model = ModelProgress(
                    model_name=model_name, display_name=display_name,
                    index=len(self.models),
                )
                self.models[model_name] = model
            model.status = MODEL_RUNNING
            model.phase = phase
            model.started_at = time.time()
            model.error = None
            self.current_model = model_name
            self.phase = phase
        self._notify()

    def model_phase(self, model_name: str, phase: str) -> None:
        with self._lock:
            model = self.models.get(model_name)
            if model is not None:
                model.phase = phase
            self.phase = phase
        self._notify()

    def trial_started(self, model_name: str, trial_number: int) -> None:
        """Called with the trial Optuna has genuinely begun evaluating."""
        with self._lock:
            model = self.models.get(model_name)
            if model is not None:
                model.trial_current = trial_number
                model.phase = PHASE_OPTIMIZING
            self.phase = PHASE_OPTIMIZING
        self._notify()

    def trial_finished(self, model_name: str, *, ok: bool) -> None:
        with self._lock:
            model = self.models.get(model_name)
            if model is not None:
                if ok:
                    model.trials_completed += 1
                else:
                    model.trials_failed += 1
                model.trial_current = None

    def model_finished(self, model_name: str, status: str,
                       error: Optional[str] = None) -> None:
        with self._lock:
            model = self.models.get(model_name)
            if model is not None:
                model.status = status
                model.phase = None
                model.error = error
                model.finished_at = time.time()
                model.trial_current = None
            if self.current_model == model_name:
                self.current_model = None
        self._notify()

    def snapshot(self) -> dict:
        """Current real state. Contains no percentage and no ETA."""
        with self._lock:
            models = [m.to_dict() for m in
                      sorted(self.models.values(), key=lambda m: m.index)]
            completed = sum(1 for m in models if m["status"] == MODEL_COMPLETED)
            failed = sum(1 for m in models if m["status"] == MODEL_FAILED)
            total = len(models)
            running = next(
                (m for m in models if m["status"] == MODEL_RUNNING), None
            )
            elapsed = None
            if self.started_at is not None:
                end = self.finished_at if self.finished_at is not None else time.time()
                elapsed = round(end - self.started_at, 2)

            return {
                "experiment_id": self.experiment_id,
                "status": self.status,
                "phase": self.phase,
                "current_model": self.current_model,
                "current_model_display_name": running["display_name"] if running else None,
                "current_model_index": running["index"] if running else None,
                "current_trial": running["trial_current"] if running else None,
                "n_trials_per_model": self.n_trials,
                "models_total": total,
                "models_completed": completed,
                "models_failed": failed,
                "models_remaining": max(0, total - completed - failed),
                "elapsed_seconds": elapsed,
                "error": self.error,
                "models": models,
                "notes": (
                    "Counts, current model and elapsed time are measured. No "
                    "percentage or ETA is reported because neither can be known "
                    "without inventing numbers."
                ),
            }


# Process-wide registry so a status endpoint can read a run started elsewhere.
_REGISTRY: dict = {}
_REGISTRY_LOCK = threading.Lock()


def register(progress: TrainingProgress) -> None:
    with _REGISTRY_LOCK:
        _REGISTRY[progress.experiment_id] = progress


def unregister(experiment_id: int) -> None:
    with _REGISTRY_LOCK:
        _REGISTRY.pop(experiment_id, None)


def get(experiment_id: int):
    """Return the live tracker for a run in this process, if any."""
    with _REGISTRY_LOCK:
        return _REGISTRY.get(experiment_id)
