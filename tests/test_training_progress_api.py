"""API-level guarantees for real training progress.

Two things must hold:

1. ``GET /{id}/progress`` reports genuine backend state - the model actually
   running, real counts, real errors - and never a fabricated percentage/ETA.
2. Training must not block the FastAPI event loop. Long synchronous model
   fitting used to stall the whole server, so a progress poll could not even be
   answered while training was in flight. A poll issued concurrently with a
   running experiment must now be served promptly.
"""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.core import database as database_module
from app.core.config import settings
from app.api import experiments as experiments_api

CSV = (
    "age,city,hours,income\n"
    "23,rj,40,0\n24,rj,42,0\n25,sp,38,0\n26,sp,45,1\n"
    "27,rj,41,1\n28,rj,44,0\n29,sp,39,1\n30,sp,46,0\n"
    "31,rj,43,1\n32,rj,47,0\n33,sp,37,1\n34,sp,48,0\n"
    "35,rj,40,1\n36,rj,49,0\n37,sp,36,1\n38,sp,50,0\n"
    "39,rj,42,1\n40,rj,51,0\n41,sp,35,1\n42,sp,52,0\n"
    "43,rj,44,1\n44,rj,53,0\n45,sp,34,1\n46,sp,54,0\n"
    "47,rj,41,1\n48,rj,55,0\n"
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings.DATABASE_URL = f"sqlite:///{tmp_path}/t.db"
    settings.STORAGE_PATH = str(tmp_path)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    database_module._engine = None
    database_module._SessionLocal = None
    experiments_api._experiment_cache.clear()
    from app.main import app
    with TestClient(app) as c:
        yield c
    database_module._engine = None
    database_module._SessionLocal = None
    experiments_api._experiment_cache.clear()


def _experiment(client, name="prog"):
    up = client.post("/api/datasets/upload",
                      files={"file": (f"{name}.csv", CSV.encode(), "text/csv")})
    ds = up.json()
    exp = client.post("/api/experiments", json={
        "dataset_id": ds["id"], "name": name, "target_column": "income",
        "problem_type": "classification", "primary_metric": "f1_weighted",
        "n_folds": 2, "n_trials": 2, "mode": "automl",
    }).json()
    return exp["id"]


FORBIDDEN = ("percent", "pct", "eta", "remaining_seconds", "estimate")


class TestProgressEndpoint:
    def test_unknown_experiment_is_404(self, client):
        assert client.get("/api/experiments/424242/progress").status_code == 404

    def test_never_trained_reports_no_snapshot(self, client):
        eid = _experiment(client, "fresh")
        body = client.get(f"/api/experiments/{eid}/progress").json()
        assert body["experiment_id"] == eid
        assert body["progress"] is None
        assert body["source"] == "none"

    def test_after_training_progress_holds_real_counts(self, client):
        eid = _experiment(client, "trained")
        client.post(f"/api/experiments/{eid}/train")
        body = client.get(f"/api/experiments/{eid}/progress").json()

        assert body["experiment_status"] == "completed"
        assert body["source"] == "persisted"   # run finished, tracker released
        prog = body["progress"]
        assert prog["status"] == "completed"
        assert prog["models_total"] > 0
        assert prog["models_completed"] + prog["models_failed"] == \
            prog["models_total"]
        assert prog["models_remaining"] == 0
        assert prog["elapsed_seconds"] is not None
        allowed = {"queued", "running", "completed", "failed"}
        assert {m["status"] for m in prog["models"]} <= allowed

    def test_no_fabricated_percentage_or_eta_anywhere(self, client):
        eid = _experiment(client, "nofake")
        client.post(f"/api/experiments/{eid}/train")
        prog = client.get(f"/api/experiments/{eid}/progress").json()["progress"]
        for key in prog:
            assert not any(bad in key.lower() for bad in FORBIDDEN), key
        for model in prog["models"]:
            for key in model:
                assert not any(bad in key.lower() for bad in FORBIDDEN), key

    def test_progress_survives_the_request(self, client):
        eid = _experiment(client, "persist")
        client.post(f"/api/experiments/{eid}/train")
        first = client.get(f"/api/experiments/{eid}/progress").json()
        second = client.get(f"/api/experiments/{eid}/progress").json()
        assert first["progress"] == second["progress"]


class TestEventLoopIsNotBlocked:
    def test_progress_answers_while_training_runs(self, client):
        """A progress poll issued during training must return promptly.

        With the old synchronous implementation the event loop was blocked for
        the whole run, so this request could not be served until training was
        already over - the app looked frozen with no way to observe anything.
        """
        eid = _experiment(client, "concurrent")
        outcome = {}

        def run():
            try:
                outcome["resp"] = client.post(f"/api/experiments/{eid}/train")
            except Exception as exc:            # pragma: no cover
                outcome["exc"] = exc

        trainer_thread = threading.Thread(target=run, daemon=True)
        trainer_thread.start()

        saw_training = False
        latencies = []
        deadline = time.time() + 120
        while time.time() < deadline:
            t0 = time.time()
            resp = client.get(f"/api/experiments/{eid}/progress")
            latencies.append(time.time() - t0)
            assert resp.status_code == 200
            if resp.json()["experiment_status"] == "training":
                saw_training = True
            if not trainer_thread.is_alive():
                break
            time.sleep(0.02)

        trainer_thread.join(timeout=180)

        assert "exc" not in outcome, outcome.get("exc")
        assert outcome.get("resp") is not None
        assert outcome["resp"].status_code == 200, outcome["resp"].text
        assert saw_training, "never observed status=training while it ran"
        # Every poll was served while training was still in flight. If the loop
        # had been blocked, the first poll would have taken the whole run.
        assert max(latencies) < 10.0, (
            f"progress endpoint was starved by training: {latencies}"
        )

    def test_progress_during_run_shows_the_live_tracker(self, client):
        """While running, the endpoint serves the in-process tracker."""
        eid = _experiment(client, "live")
        seen = []

        def run():
            client.post(f"/api/experiments/{eid}/train")

        t = threading.Thread(target=run, daemon=True)
        t.start()
        deadline = time.time() + 120
        while time.time() < deadline and t.is_alive():
            body = client.get(f"/api/experiments/{eid}/progress").json()
            if body["source"] == "live" and body["progress"]:
                seen.append(body["progress"])
            time.sleep(0.02)
        t.join(timeout=180)

        assert seen, "never observed a live progress snapshot during training"
        # Counts only ever move forward as models genuinely finish.
        completed = [s["models_completed"] for s in seen]
        assert completed == sorted(completed)
        assert max(completed) >= 1

    def test_training_still_returns_real_results(self, client):
        """Offloading to a thread must not change the payload."""
        eid = _experiment(client, "payload")
        body = client.post(f"/api/experiments/{eid}/train").json()
        assert body["status"] == "completed"
        assert body["models"]
        assert body["best_model"] is not None
        assert body["selection"]["holdout_used_for_selection"] is False
        assert body["selection"]["tie_break"]["winner"] == \
            body["selection"]["selected_model"]
