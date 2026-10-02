"""Training progress must reflect real backend state and nothing else.

The tracker deliberately reports no percentage and no ETA: the work is not
linearly divisible, so both would have to be invented. These tests pin that
contract, plus the per-model state machine and the fact that a real failure
is preserved verbatim.
"""
import threading
import time

from app.services.training_progress import (
    TrainingProgress, MODEL_QUEUED, MODEL_RUNNING, MODEL_COMPLETED,
    MODEL_FAILED, PHASE_TRAINING, PHASE_OPTIMIZING, PHASE_SCORING,
    STATUS_QUEUED, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED,
)
import app.services.training_progress as tp


def make(models=(("a", "Model A"), ("b", "Model B")), **kw):
    return TrainingProgress(1, list(models), n_trials=4, **kw)


FORBIDDEN = ("percent", "pct", "eta", "remaining_seconds", "estimated",
             "estimate")


class TestNoFabricatedNumbers:
    def test_snapshot_has_no_percentage_or_eta(self):
        p = make()
        p.start()
        p.model_started("a", "Model A")
        snap = p.snapshot()
        for key in snap:
            assert not any(bad in key.lower() for bad in FORBIDDEN), key

    def test_per_model_rows_have_no_fabricated_fields(self):
        p = make()
        p.start()
        p.model_started("a", "Model A")
        for model in p.snapshot()["models"]:
            for key in model:
                assert not any(bad in key.lower() for bad in FORBIDDEN), key

    def test_counts_are_exact_not_estimated(self):
        p = make()
        p.start()
        p.model_finished("a", MODEL_COMPLETED)
        snap = p.snapshot()
        assert snap["models_total"] == 2
        assert snap["models_completed"] == 1
        assert snap["models_failed"] == 0
        assert snap["models_remaining"] == 1


class TestLifecycleStates:
    def test_starts_queued_then_running(self):
        p = make()
        assert p.snapshot()["status"] == STATUS_QUEUED
        p.start()
        assert p.snapshot()["status"] == STATUS_RUNNING

    def test_queued_models_are_reported_before_they_start(self):
        p = make()
        p.start()
        assert [m["status"] for m in p.snapshot()["models"]] == \
            [MODEL_QUEUED] * 2

    def test_completed_and_failed_terminal_states(self):
        p = make()
        p.start()
        p.model_started("a", "Model A")
        p.model_finished("a", MODEL_COMPLETED)
        p.model_started("b", "Model B")
        p.model_finished("b", MODEL_FAILED, error="ValueError: boom")
        p.finish(STATUS_COMPLETED)
        snap = p.snapshot()
        assert snap["status"] == STATUS_COMPLETED
        assert snap["models_completed"] == 1
        assert snap["models_failed"] == 1
        assert snap["models_remaining"] == 0
        assert snap["phase"] == "done"


class TestCurrentModelIsReal:
    def test_reports_the_model_actually_running(self):
        p = make()
        p.start()
        p.model_started("b", "Model B")
        snap = p.snapshot()
        assert snap["current_model"] == "b"
        assert snap["current_model_display_name"] == "Model B"
        assert snap["current_model_index"] == 1

    def test_current_model_clears_when_that_model_finishes(self):
        p = make()
        p.start()
        p.model_started("a", "Model A")
        p.model_finished("a", MODEL_COMPLETED)
        assert p.snapshot()["current_model"] is None

    def test_phases_are_reported_when_they_genuinely_begin(self):
        p = make()
        p.start()
        p.model_started("a", "Model A")
        assert p.snapshot()["phase"] == PHASE_TRAINING
        p.model_phase("a", PHASE_SCORING)
        assert p.snapshot()["phase"] == PHASE_SCORING
        assert p.snapshot()["models"][0]["phase"] == PHASE_SCORING


class TestTrialReporting:
    def test_current_trial_is_the_one_optuna_started(self):
        p = make()
        p.start()
        p.model_started("a", "Model A")
        p.model_phase("a", PHASE_OPTIMIZING)
        p.trial_started("a", 2)
        snap = p.snapshot()
        assert snap["current_trial"] == 2
        assert snap["phase"] == PHASE_OPTIMIZING

    def test_trial_counts_track_success_and_failure(self):
        p = make()
        p.start()
        p.model_started("a", "Model A")
        p.trial_started("a", 0)
        p.trial_finished("a", ok=True)
        p.trial_started("a", 1)
        p.trial_finished("a", ok=False)
        model = p.snapshot()["models"][0]
        assert model["trials_completed"] == 1
        assert model["trials_failed"] == 1
        assert model["trial_current"] is None

    def test_n_trials_per_model_is_reported(self):
        assert make().snapshot()["n_trials_per_model"] == 4


class TestErrorsArePreserved:
    def test_model_error_is_kept_verbatim(self):
        p = make()
        p.start()
        p.model_started("a", "Model A")
        p.model_finished("a", MODEL_FAILED,
                         error="MemoryError: could not allocate")
        model = p.snapshot()["models"][0]
        assert model["status"] == MODEL_FAILED
        assert model["error"] == "MemoryError: could not allocate"

    def test_run_level_failure_records_the_error(self):
        p = make()
        p.start()
        p.finish(STATUS_FAILED, error="Training failed: bad column")
        snap = p.snapshot()
        assert snap["status"] == STATUS_FAILED
        assert snap["error"] == "Training failed: bad column"


class TestElapsedIsMeasured:
    def test_elapsed_is_none_before_start(self):
        assert make().snapshot()["elapsed_seconds"] is None

    def test_elapsed_grows_then_freezes_on_finish(self):
        p = make()
        p.start()
        time.sleep(0.05)
        assert p.snapshot()["elapsed_seconds"] > 0
        p.finish(STATUS_COMPLETED)
        first = p.snapshot()["elapsed_seconds"]
        time.sleep(0.05)
        assert p.snapshot()["elapsed_seconds"] == first


class TestPersistenceHook:
    def test_snapshots_are_pushed_to_the_persist_callback(self):
        seen = []
        p = make(persist=seen.append)
        p.start()
        p.model_started("a", "Model A")
        assert len(seen) >= 2
        assert seen[-1]["current_model"] == "a"

    def test_a_failing_persist_hook_never_breaks_training(self):
        def boom(_snapshot):
            raise RuntimeError("db gone")

        p = make(persist=boom)
        p.start()          # must not raise
        p.model_started("a", "Model A")
        assert p.snapshot()["status"] == STATUS_RUNNING


class TestThreadSafety:
    def test_concurrent_updates_keep_counts_consistent(self):
        p = make(models=tuple((f"m{i}", f"M{i}") for i in range(8)))
        p.start()

        def worker(i):
            name = f"m{i}"
            p.model_started(name, f"M{i}")
            for t in range(4):
                p.trial_started(name, t)
                p.trial_finished(name, ok=True)
            p.model_finished(name, MODEL_COMPLETED)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = p.snapshot()
        assert snap["models_completed"] == 8
        assert snap["models_failed"] == 0
        assert snap["current_model"] is None
        assert sum(m["trials_completed"] for m in snap["models"]) == 32


class TestRegistry:
    def test_register_get_unregister(self):
        p = make()
        tp.register(p)
        assert tp.get(1) is p
        tp.unregister(1)
        assert tp.get(1) is None

    def test_get_returns_none_for_unknown_experiment(self):
        assert tp.get(999999) is None
