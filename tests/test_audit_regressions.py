"""Regression tests for the C1-C10 findings in ``docs/AUDIT_REPORT.md``.

Every test names the audit issue it guards and asserts the *fixed* behaviour, so
the suite fails while an issue is still open and passes once it is closed.

The API tests run against a throwaway SQLite database and storage directory, so
the developer's ``automl_lens.db`` / ``backend/storage`` are never touched.
"""
import asyncio
import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"
TARGET = "target"


# ───────────────────────────── sample data ────────────────────────────
def sample_frame() -> pd.DataFrame:
    """Deterministic, learnable classification dataset (numeric + categorical)."""
    rng = np.random.RandomState(7)
    n = 100
    num_a = rng.randint(0, 90, n).astype(float)
    num_b = rng.uniform(0, 10, n)
    cat_c = rng.choice(["low", "mid", "high"], n)
    high = np.array([c == "high" for c in cat_c])
    target = ((num_a > 45) ^ (num_b > 5) ^ high).astype(int)
    return pd.DataFrame({
        "num_a": num_a, "num_b": num_b, "cat_c": cat_c, TARGET: target,
    })


def engineering_frame() -> pd.DataFrame:
    """Frame that exercises the feature-engineering branches that matter here.

    Two informative numerics (interaction candidates) plus a 35-level
    categorical (frequency encoding), so ``remainder="drop"`` has real
    engineered columns to lose.
    """
    rng = np.random.RandomState(11)
    n = 240
    num_a = rng.uniform(0, 50, n)
    num_b = rng.uniform(-10, 10, n)
    city = [f"city_{i % 35:02d}" for i in range(n)]
    target = ((num_a > 25) ^ (num_b > 0)).astype(int)
    return pd.DataFrame({
        "num_a": num_a, "num_b": num_b, "city": city, TARGET: target,
    })


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def xlsx_bytes(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


# ───────────────────────────── fixtures ───────────────────────────────
@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """App bound to a temporary database/storage with the LLM fallback pinned."""
    from app.core.config import settings
    from app.core import database as database_module
    from app.api import experiments as experiments_api

    tmp_dir = tmp_path_factory.mktemp("audit_regressions")
    saved = {
        "DATABASE_URL": settings.DATABASE_URL,
        "STORAGE_PATH": settings.STORAGE_PATH,
        "LLM_PROVIDER": settings.LLM_PROVIDER,
        "GEMINI_API_KEY": settings.GEMINI_API_KEY,
        "OPENAI_API_KEY": settings.OPENAI_API_KEY,
    }
    settings.DATABASE_URL = f"sqlite:///{(tmp_dir / 'test.db').as_posix()}"
    settings.STORAGE_PATH = str(tmp_dir)
    settings.LLM_PROVIDER = "fallback"
    settings.GEMINI_API_KEY = ""
    settings.OPENAI_API_KEY = ""
    database_module._engine = None
    database_module._SessionLocal = None
    experiments_api._experiment_cache.clear()

    from app.main import app
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        for key, value in saved.items():
            setattr(settings, key, value)
        database_module._engine = None
        database_module._SessionLocal = None
        experiments_api._experiment_cache.clear()


def upload_csv(client: TestClient, frame: pd.DataFrame, name: str) -> int:
    resp = client.post(
        "/api/datasets/upload",
        files={"file": (f"{name}.csv", io.BytesIO(csv_bytes(frame)), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def create_experiment(client: TestClient, dataset_id: int, *, name: str,
                      problem_type: str = "classification",
                      primary_metric="f1_weighted") -> int:
    created = client.post("/api/experiments", json={
        "dataset_id": dataset_id,
        "target_column": TARGET,
        "problem_type": problem_type,
        "primary_metric": primary_metric,
        "mode": "baseline",
        "name": name,
    })
    assert created.status_code == 200, created.text
    return created.json()["id"]


@pytest.fixture(scope="module")
def trained(client):
    """upload -> create -> analyze -> train, once for the whole module."""
    dataset_id = upload_csv(client, sample_frame(), "audit_trained")
    exp_id = create_experiment(client, dataset_id, name="audit_trained")
    assert client.post(f"/api/experiments/{exp_id}/analyze").status_code == 200
    resp = client.post(f"/api/experiments/{exp_id}/train?fast_demo=true")
    assert resp.status_code == 200, resp.text
    return dataset_id, exp_id


def prediction_payload(client: TestClient, exp_id: int) -> dict:
    """A payload that satisfies the experiment's own input schema."""
    fields = client.get(f"/api/experiments/{exp_id}/input-schema").json()["fields"]
    return {
        f["name"]: (1 if f["type"] == "number" else "low")
        for f in fields
    }


# ═════════════ C6 — /train must not block the event loop ═══════════════
def test_c6_training_is_not_run_inline_on_the_event_loop():
    """C6: the whole training body ran on the event loop, so ``/api/health``
    (and every other request) stalled until Optuna finished."""
    from app.api import experiments as experiments_api

    source = Path(experiments_api.__file__).read_text(encoding="utf-8")
    assert "to_thread" in source, "training body is still executed inline"

    # The endpoint itself must stay async so the worker thread is awaited.
    assert asyncio.iscoroutinefunction(experiments_api.train_experiment)


def test_c6_health_responds_while_training_is_in_flight(client):
    """C6: a slow in-flight /train must not freeze /api/health.

    Latency is the discriminator: if the training body runs on the event loop,
    every probe is queued behind it and each one measures the whole training
    run. A healthy server answers each probe in milliseconds.
    """
    dataset_id = upload_csv(client, engineering_frame(), "c6_health")
    exp_id = create_experiment(client, dataset_id, name="c6_health")

    client.get("/api/health")  # warm up, so connection setup is not counted
    observed = []
    latencies = []

    async def probe_health():
        await asyncio.sleep(0)  # let the training request start
        for _ in range(40):
            started = time.perf_counter()
            response = await asyncio.to_thread(client.get, "/api/health")
            latencies.append(time.perf_counter() - started)
            observed.append(response.status_code)
            await asyncio.sleep(0.02)

    async def run_both():
        await asyncio.gather(
            asyncio.to_thread(
                client.post, f"/api/experiments/{exp_id}/train?fast_demo=true"
            ),
            probe_health(),
        )

    asyncio.run(run_both())

    assert len(observed) >= 5, f"only {len(observed)} health probes completed"
    assert all(code == 200 for code in observed), observed
    median = sorted(latencies)[len(latencies) // 2]
    assert median < 1.0, (
        f"median /api/health latency was {median:.2f}s while training ran - "
        "the event loop is blocked"
    )


# ══════════ C7 — configuration must not depend on the CWD ══════════════
def test_c7_env_file_is_anchored_to_the_backend_package():
    """C7: ``load_dotenv(<repo_root>/.env)`` pointed at a file that does not
    exist, so keys were silently ignored."""
    from app.core.config import BACKEND_DIR, ENV_FILE, settings

    assert BACKEND_DIR.name == "backend"
    assert ENV_FILE == BACKEND_DIR / ".env"
    assert settings.model_config.get("env_file")


def test_c7_database_url_and_storage_are_resolved_against_backend():
    """C7: ``DATABASE_URL`` was ignored and storage was CWD-relative."""
    from app.core.config import BACKEND_DIR, settings

    # The module-scoped API fixture rebinds these to a temp dir; restore the
    # shipped relative defaults so this test checks the resolution rules.
    saved_url, saved_storage = settings.DATABASE_URL, settings.STORAGE_PATH
    settings.DATABASE_URL = "sqlite:///./automl_lens.db"
    settings.STORAGE_PATH = "./storage"
    try:
        resolved = settings.resolved_database_url
        db_path = Path(resolved[len("sqlite:///"):])
        assert db_path.is_absolute()
        assert db_path.parent == BACKEND_DIR
        assert settings.storage_path == BACKEND_DIR / "storage"
    finally:
        settings.DATABASE_URL, settings.STORAGE_PATH = saved_url, saved_storage


def test_c7_launcher_writes_variable_names_the_app_reads():
    """C7: ``run_project.bat`` wrote ``STORAGE_BASE_PATH``, a name nothing reads."""
    script = (REPO_ROOT / "run_project.bat").read_text(encoding="utf-8", errors="replace")
    assert "STORAGE_BASE_PATH" not in script
    assert "STORAGE_PATH=" in script


# ═════ C8 — the in-memory cache must not pin the full dataset ═════════
def test_c8_experiment_cache_does_not_hold_the_full_dataframe():
    """C8: every completed experiment kept ``df_original`` alive in a
    process-wide dict, so RAM grew with each run."""
    from app.api import experiments as experiments_api

    source = Path(experiments_api.__file__).read_text(encoding="utf-8")
    assert '"df_original"' not in source


def test_c8_experiment_cache_is_bounded():
    """C8: the cache had no size limit at all."""
    from app.api import experiments as experiments_api

    cache = experiments_api._experiment_cache
    assert hasattr(cache, "maxsize"), "cache is still an unbounded dict"


# ═════ C9 — chart/table must use the experiment's primary metric ═══════
def test_c9_model_comparison_uses_the_resolved_primary_metric(client, trained):
    """C9: the comparison chart took "the first number in the metrics dict",
    so an ``f1_weighted`` run could be charted with accuracy."""
    _, exp_id = trained
    results = client.get(f"/api/experiments/{exp_id}/results").json()
    primary = results["primary_metric"]

    completed = [m for m in results["models"] if m["status"] == "COMPLETED"]
    assert completed, "no model completed - cannot check the chart contract"
    for model in completed:
        metrics = model["optimized_metrics"] or model["baseline_metrics"]
        assert model["holdout_primary_key"] == primary
        assert model["holdout_primary"] == pytest.approx(float(metrics[primary]))


def test_c9_frontend_chart_binds_to_the_primary_metric():
    """C9: the chart data still read ``Object.values(...)[0]``."""
    jsx = (FRONTEND_SRC / "pages" / "Experiment.jsx").read_text(encoding="utf-8")
    chart_block = jsx[jsx.index("const comparisonData"):jsx.index("const expl =")]
    assert "holdout_primary" in chart_block
    assert "Object.values(" not in chart_block


# ═════ C10 — prediction history, unique batch file, report metrics ═════
def test_c10_prediction_rows_are_persisted(client, trained):
    """C10: predictions were never written to the ``predictions`` table."""
    from app.core.database import Prediction, get_session_factory

    _, exp_id = trained
    payload = prediction_payload(client, exp_id)
    resp = client.post(f"/api/experiments/{exp_id}/predict", json=payload)
    assert resp.status_code == 200, resp.text

    db = get_session_factory()()
    try:
        rows = db.query(Prediction).filter(Prediction.experiment_id == exp_id).all()
    finally:
        db.close()
    assert rows, "no Prediction row was stored for a single prediction"
    assert rows[-1].prediction_type == "single"
    assert rows[-1].input_json and rows[-1].result_json


def test_c10_batch_input_filename_is_unique_and_history_recorded(client, trained):
    """C10: the batch input path was a fixed ``batch_input_{exp_id}.csv``."""
    from app.core.database import Prediction, get_session_factory

    _, exp_id = trained
    frame = pd.DataFrame([prediction_payload(client, exp_id)])

    for _ in range(2):
        resp = client.post(
            f"/api/experiments/{exp_id}/batch-predict",
            files={"file": ("batch.csv", io.BytesIO(csv_bytes(frame)), "text/csv")},
        )
        assert resp.status_code == 200, resp.text

    db = get_session_factory()()
    try:
        rows = (db.query(Prediction)
                .filter(Prediction.experiment_id == exp_id,
                        Prediction.prediction_type == "batch")
                .all())
    finally:
        db.close()
    assert len(rows) >= 2, "batch predictions were not recorded"
    paths = [r.batch_file_path for r in rows]
    assert len(set(paths)) == len(paths), f"batch input path reused: {paths}"


def test_c10_report_carries_the_primary_metric(client, trained):
    """C10: the report payload passed ``{"metrics": {}}``, so the best-model
    metrics section had nothing labelled to show."""
    _, exp_id = trained
    body = client.get(f"/api/experiments/{exp_id}/report").json()
    assert body["report_url"]
    html = Path(body["report_path"]).read_text(encoding="utf-8")
    primary = client.get(f"/api/experiments/{exp_id}/results").json()["primary_metric"]
    assert primary in html


def test_c10_report_generator_does_not_guess_the_primary_metric():
    """C10: the report took ``list(metrics.values())[0]`` as the test score."""
    from app.services.reporter import ReportGenerator

    html = ReportGenerator().generate_html_report({
        "experiment_id": 999,
        "target": TARGET,
        "problem_type": "classification",
        "primary_metric": "f1_weighted",
        "models_results": [{
            "display_name": "Toy", "model_name": "toy", "status": "COMPLETED",
            "cv_scores": [0.5, 0.6],
            # f1_weighted is deliberately NOT the first key in the dict.
            "optimized_metrics": {"accuracy": 0.11, "f1_weighted": 0.99},
        }],
        "best_model": {"display_name": "Toy", "model_name": "toy",
                       "optimized_metrics": {"accuracy": 0.11,
                                             "f1_weighted": 0.99}},
        "metrics": {"primary_metric": "f1_weighted"},
    })
    assert "f1_weighted" in html
    assert "0.9900" in html, "the primary metric value is missing from the report"
    assert "0.1100" in html

# ══════════════════════════ C1 — .xlsx ingestion ═══════════════════════
def test_c1_xlsx_is_read_not_merely_accepted(client):
    """C1: .xlsx uploaded fine, but every downstream reader used ``read_csv``,
    so columns came back empty and analysis/training broke silently."""
    frame = sample_frame()
    resp = client.post(
        "/api/datasets/upload",
        files={"file": ("c1.xlsx", io.BytesIO(xlsx_bytes(frame)),
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet")},
    )
    assert resp.status_code == 200, resp.text
    dataset_id = resp.json()["id"]

    columns = client.get(f"/api/datasets/{dataset_id}/columns")
    assert columns.status_code == 200, columns.text
    assert sorted(columns.json()["columns"]) == sorted(frame.columns)

    profiled = client.get(f"/api/datasets/{dataset_id}/profile")
    assert profiled.status_code == 200, profiled.text
    body = profiled.json()
    assert body["rows"] == len(frame)
    assert len(body["column_profiles"]) == len(frame.columns)


# ══════════════ C2 — /train must resolve an "auto" problem type ════════
def test_c2_train_detects_problem_type_when_auto(client):
    """C2: with ``problem_type`` falsy, /train selected models for task
    ``None``, so the registry returned nothing and the run "completed" with an
    empty model list."""
    dataset_id = upload_csv(client, sample_frame(), "c2_auto")
    exp_id = create_experiment(client, dataset_id, name="c2_auto",
                               problem_type="auto", primary_metric=None)
    created = client.get(f"/api/experiments/{exp_id}").json()
    assert not created["problem_type"] or created["problem_type"] == "auto"

    resp = client.post(f"/api/experiments/{exp_id}/train?fast_demo=true")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["problem_type"] in ("classification", "regression")
    assert body["primary_metric"], "a detected problem type needs a default metric"
    assert body["models"], "an auto problem type must still train models"


# ══════════ C3 — /analyze must not downgrade a finished experiment ═════
def test_c3_analyze_does_not_downgrade_completed(client, trained):
    """C3: /analyze flipped a completed experiment back to ``analyzed``, and the
    detail payload hid the stored analysis so the UI re-called the LLM."""
    _, exp_id = trained
    before = client.get(f"/api/experiments/{exp_id}").json()
    assert before["status"] == "completed"
    assert before["has_analysis"] is True
    assert before["has_results"] is True
    assert before["llm_analysis"], "stored analysis must be exposed for hydration"

    assert client.post(f"/api/experiments/{exp_id}/analyze").status_code == 200
    after = client.get(f"/api/experiments/{exp_id}").json()
    assert after["status"] == "completed", "analyze downgraded a completed run"


def test_c3_frontend_hydrates_instead_of_recalling_the_llm():
    """C3: the experiment page re-ran /analyze on every load instead of reading
    the stored analysis (``has_analysis``)."""
    jsx = (FRONTEND_SRC / "pages" / "Experiment.jsx").read_text(encoding="utf-8")
    assert "has_analysis" in jsx
    assert "llm_analysis" in jsx


# ══════ C4 — engineered columns must reach the model (remainder=drop) ══
def test_c4_engineered_columns_reach_the_model_matrix():
    """C4: engineered interactions / frequency encodings were silently dropped
    by the ColumnTransformer's ``remainder="drop"``."""
    from app.services.profiler import DatasetProfiler
    from app.services.feature_engineer import FeatureEngineer
    from app.services.preprocessor import PreprocessingEngine

    frame = engineering_frame()
    profile = DatasetProfiler().profile(frame)
    engineer = FeatureEngineer().fit(frame, profile["column_profiles"], TARGET)
    assert engineer.new_features, "this frame should trigger feature engineering"

    engine = PreprocessingEngine()
    X, names = engine.build_and_fit(
        engineer.transform(frame), TARGET, profile["column_profiles"],
        "classification", extra_numeric=engineer.new_features,
        feature_engineer=engineer,
    )
    for column in engineer.new_features:
        assert column in engine.summary["engineered_columns"], column
        assert any(column in name for name in names), f"{column} was dropped"
    assert engine.summary["total_features_out"] == X.shape[1]

    # Prediction-time transform must reproduce the training width exactly.
    X_new = engine.transform(frame.drop(columns=[TARGET]))
    assert X_new.shape[1] == X.shape[1]


# ═══════════ C5 — provider label / confidence are not fabricated ═══════
def test_c5_frontend_does_not_invent_confidence_or_provider():
    """C5: the AI panel defaulted confidence to 0.8 and labelled the
    deterministic fallback as if it were a real LLM."""
    jsx = (FRONTEND_SRC / "pages" / "Experiment.jsx").read_text(encoding="utf-8")
    assert "|| 0.8" not in jsx
    assert "provider_used" in jsx
    assert "not reported" in jsx


def test_c5_analysis_payload_carries_provider_provenance(client):
    """C5: provenance must travel in the payload the UI actually consumes."""
    dataset_id = upload_csv(client, sample_frame(), "c5_provenance")
    exp_id = create_experiment(client, dataset_id, name="c5_provenance")
    body = client.post(f"/api/experiments/{exp_id}/analyze").json()
    assert body["provider_used"]
    assert body["is_fallback"] is True
    assert body["result"]["provider_used"] == body["provider_used"]
    assert body["result"]["is_fallback"] is True

