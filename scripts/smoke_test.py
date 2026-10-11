"""Local end-to-end smoke test for the AutoML-Lens backend (no API keys).

Exercises the real FastAPI application in-process (``TestClient``) against a
temporary database and storage directory:

    1. ``GET  /api/health``
    2. ``POST /api/chat``  (deterministic fallback assistant)
    3. For both bundled demo datasets (``demo_data/*.csv``):
         ``POST /api/datasets/upload``
         ``POST /api/experiments``               (create)
         ``POST /api/experiments/{id}/analyze``  (deterministic fallback plan)
         ``POST /api/experiments/{id}/train``    (fast_demo=true, seed=42)
         ``GET  /api/experiments/{id}/status``   (polled until terminal)
         ``GET  /api/experiments/{id}/results``
    4. Verifies the live ``backend/automl_lens.db`` and ``backend/storage`` are
       untouched (read-only fingerprint taken before and after the run).

Usage (from the repository root):
    python scripts/smoke_test.py
"""
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

# Pin the keyless path before the application settings are imported, mirroring
# scripts/run_tests_keyless.py (an empty-string env value cannot be expressed
# through the Windows shell).
os.environ["GEMINI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["LLM_PROVIDER"] = "fallback"

from app.core.config import settings  # noqa: E402
from app.core import database as database_module  # noqa: E402
from app.api import experiments as experiments_api  # noqa: E402

# ---------------------------------------------------------------------------
# Isolation: temporary database + storage, deterministic fallback provider.
# ---------------------------------------------------------------------------
_tmp_root = Path(tempfile.mkdtemp(prefix="automl_lens_smoke_"))
settings.DATABASE_URL = f"sqlite:///{(_tmp_root / 'smoke.db').as_posix()}"
settings.STORAGE_PATH = str(_tmp_root / "storage")
settings.LLM_PROVIDER = "fallback"
settings.GEMINI_API_KEY = ""
settings.OPENAI_API_KEY = ""
database_module._engine = None
database_module._SessionLocal = None
experiments_api._experiment_cache.clear()

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

PROD_DB = REPO_ROOT / "backend" / "automl_lens.db"
PROD_STORAGE = REPO_ROOT / "backend" / "storage"

DEMO_DATASETS = [
    ("classification.csv", "Churn"),
    ("regression.csv", "Price"),
]

_t0 = time.perf_counter()


def step(msg: str) -> None:
    print(f"[{time.perf_counter() - _t0:8.2f}s] {msg}", flush=True)


def production_fingerprint() -> dict:
    """Read-only fingerprint of the live DB row counts and storage files."""
    counts = {"experiments": None, "datasets": None}
    if PROD_DB.exists():
        conn = sqlite3.connect(PROD_DB)
        try:
            cur = conn.cursor()
            counts["experiments"] = cur.execute(
                "SELECT COUNT(*) FROM experiments").fetchone()[0]
            counts["datasets"] = cur.execute(
                "SELECT COUNT(*) FROM datasets").fetchone()[0]
        finally:
            conn.close()
    files = sorted(
        os.path.join(root, name)
        for root, _, names in os.walk(PROD_STORAGE)
        for name in names
    )
    return {"counts": counts, "files": files}


def check(condition: bool, label: str) -> None:
    if not condition:
        print(f"SMOKE TEST FAILED at: {label}", flush=True)
        sys.exit(1)
    step(f"OK  {label}")


def main() -> None:
    step(f"Temporary workspace: {_tmp_root}")
    baseline = production_fingerprint()
    step(f"Production baseline: db_counts={baseline['counts']} "
         f"storage_files={len(baseline['files'])}")

    with TestClient(app) as client:
        # 1. Health ---------------------------------------------------------
        r = client.get("/api/health")
        check(r.status_code == 200, "GET /api/health -> 200")
        body = r.json()
        check(body.get("status") == "healthy", "health status == healthy")
        step(f"    health: {body}")

        # 2. Chat (fallback assistant) --------------------------------------
        r = client.post("/api/chat", json={"message": "What is AutoML-Lens?"})
        check(r.status_code == 200, "POST /api/chat -> 200")
        chat = r.json()
        check(bool(chat.get("response")), "chat returned a response")
        step(f"    chat provider: {chat.get('provider')}")

        # 3. Full pipeline for each demo dataset ----------------------------
        for csv_name, target in DEMO_DATASETS:
            csv_path = REPO_ROOT / "demo_data" / csv_name
            step(f"--- {csv_name} (target={target}) ---")

            with open(csv_path, "rb") as fh:
                r = client.post(
                    "/api/datasets/upload",
                    files={"file": (csv_name, fh, "text/csv")},
                )
            check(r.status_code == 200, f"upload {csv_name} -> 200")
            ds = r.json()
            check(ds["rows"] > 0 and ds["columns"] > 1,
                  f"{csv_name}: {ds['rows']} rows x {ds['columns']} columns")
            step(f"    dataset id={ds['id']}")

            r = client.post("/api/experiments", json={
                "dataset_id": ds["id"],
                "name": f"smoke-{csv_name}",
                "target_column": target,
                "problem_type": "auto",
                "n_folds": 3,
                "n_trials": 3,
            })
            check(r.status_code == 200, f"create experiment for {csv_name} -> 200")
            exp = r.json()
            exp_id = exp["id"]
            step(f"    experiment id={exp_id}")

            r = client.post(f"/api/experiments/{exp_id}/analyze")
            check(r.status_code == 200, f"analyze {csv_name} -> 200")
            analysis = r.json()
            provider = (analysis.get("provider_used")
                        or analysis.get("result", {}).get("provider_used"))
            step(f"    analysis provider: {provider}")
            r = client.get(f"/api/experiments/{exp_id}/status")
            check(r.json()["status"] == "analyzed",
                  "status after analyze == analyzed")

            t_train = time.perf_counter()
            r = client.post(f"/api/experiments/{exp_id}/train",
                            params={"fast_demo": "true", "seed": 42})
            check(r.status_code == 200, f"train {csv_name} -> 200")
            train = r.json()
            check(train["status"] == "completed", f"train {csv_name} completed")
            step(f"    trained models: {len(train['models'])} in "
                 f"{time.perf_counter() - t_train:.2f}s")

            # Poll status until terminal (fast_demo training is synchronous,
            # so this normally observes the terminal state immediately).
            terminal = None
            deadline = time.perf_counter() + 600
            polls = 0
            while time.perf_counter() < deadline:
                polls += 1
                r = client.get(f"/api/experiments/{exp_id}/status")
                terminal = r.json()["status"]
                if terminal in ("completed", "failed"):
                    break
                time.sleep(2)
            check(terminal == "completed",
                  f"poll /status -> completed ({polls} poll(s))")

            r = client.get(f"/api/experiments/{exp_id}/results")
            check(r.status_code == 200, f"results {csv_name} -> 200")
            results = r.json()
            check(bool(results.get("best_model_name")),
                  f"best model selected: {results['best_model_name']} "
                  f"(holdout score: {results.get('best_score')})")

    # 4. Production state unchanged -----------------------------------------
    after = production_fingerprint()
    check(after == baseline,
          "production backend/automl_lens.db and backend/storage untouched")

    step("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

