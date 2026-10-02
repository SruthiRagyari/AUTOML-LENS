"""End-to-end API tests for LLM-assisted feature engineering.

Covers: /analyze surfaces honest provider provenance and operation slots;
/train re-validates stored suggestions, executes only registry-approved
operations, feeds the engineered features to the model, and predictions keep
working afterwards. Uses the deterministic fallback provider (no API key)."""
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

TARGET = "target"


def sample_csv() -> bytes:
    rng = np.random.RandomState(7)
    n = 100
    num_a = rng.randint(1, 90, n).astype(float)
    num_b = rng.uniform(0, 10, n)
    cat_c = rng.choice(["low", "mid", "high"], n)
    high = np.array([c == "high" for c in cat_c])
    target = ((num_a > 45) ^ (num_b > 5) ^ high).astype(int)
    return pd.DataFrame({
        "num_a": num_a, "num_b": num_b, "cat_c": cat_c, TARGET: target,
    }).to_csv(index=False).encode("utf-8")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app.core.config import settings
    from app.core import database as database_module
    from app.api import experiments as experiments_api

    tmp_dir = tmp_path_factory.mktemp("llm_analysis_api")
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


def create_and_analyze(client, name):
    upload = client.post(
        "/api/datasets/upload",
        files={"file": (f"{name}.csv", io.BytesIO(sample_csv()), "text/csv")},
    )
    assert upload.status_code == 200, upload.text
    dataset_id = upload.json()["id"]
    created = client.post("/api/experiments", json={
        "dataset_id": dataset_id, "target_column": TARGET,
        "problem_type": "classification", "primary_metric": "f1_weighted",
        "mode": "baseline", "name": name,
    })
    assert created.status_code == 200, created.text
    exp_id = created.json()["id"]
    analyzed = client.post(f"/api/experiments/{exp_id}/analyze")
    assert analyzed.status_code == 200, analyzed.text
    return dataset_id, exp_id, analyzed.json()


def test_analyze_reports_honest_provenance_and_operation_slots(client):
    _, exp_id, analysis = create_and_analyze(client, "llm_analysis_exp")
    assert analysis["provider_used"] == "Fallback (Deterministic)"
    assert analysis["is_fallback"] is True
    result = analysis["result"]
    assert result["provider_used"] == "Fallback (Deterministic)"
    assert result["is_fallback"] is True
    assert result["operation_source"] == "deterministic_defaults"
    assert result["suggested_operations"] == []
    assert result["problem_understanding"]
    assert isinstance(result["rejected_operations"], list)
    # legacy free-text panel still travels alongside the new structured fields
    assert result["reasoning"]
    assert result["candidate_models"]


def test_train_without_operations_uses_deterministic_heuristics(client):
    _, exp_id, _ = create_and_analyze(client, "llm_train_exp")
    trained = client.post(f"/api/experiments/{exp_id}/train?fast_demo=true")
    assert trained.status_code == 200, trained.text
    body = trained.json()
    assert body["provider_used"] == "Fallback (Deterministic)"
    assert body["suggested_operations"] == []
    assert body["feature_engineering"]["mode"] == "heuristics"
    results = client.get(f"/api/experiments/{exp_id}/results").json()
    assert results["feature_engineering_summary"]["mode"] == "heuristics"


def test_train_executes_only_registry_validated_operations(client):
    _, exp_id, _ = create_and_analyze(client, "llm_ops_exp")

    # Inject suggestions the way a chatty provider would leave them in the
    # stored analysis: one alias, two valid ops, one aimed at the target and
    # one unknown. /train must re-validate and execute only the safe ones.
    from app.core.database import get_session_factory, Experiment
    db = get_session_factory()()
    try:
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        payload = json.loads(exp.llm_analysis_json)
        payload["result"]["operation_source"] = "provider"
        payload["result"]["suggested_operations"] = [
            {"column": "num_a", "operation": "log", "reason": "skewed"},
            {"column": "cat_c", "operation": "frequency_encoding"},
            {"column": TARGET, "operation": "zscore"},
            {"column": "num_a", "operation": "launch_missiles"},
        ]
        exp.llm_analysis_json = json.dumps(payload)
        db.commit()
    finally:
        db.close()

    trained = client.post(f"/api/experiments/{exp_id}/train?fast_demo=true")
    assert trained.status_code == 200, trained.text
    body = trained.json()

    accepted = [(o["column"], o["operation"]) for o in body["suggested_operations"]]
    assert ("num_a", "log1p") in accepted          # alias normalized
    assert ("cat_c", "frequency_encoding") in accepted
    assert all(o["column"] != TARGET for o in body["suggested_operations"])
    reasons = " | ".join(r.get("reason", "") for r in body["rejected_operations"])
    assert "target column" in reasons
    assert "not in the allowed operation registry" in reasons

    fe_summary = body["feature_engineering"]
    assert fe_summary["mode"] == "operations"
    assert "num_a_log1p" in fe_summary["new_features"]
    assert "cat_c_freq" in fe_summary["new_features"]
    assert fe_summary["rejected_operations"], "rejections must be surfaced"

    # engineered columns actually reached the model matrix
    results = client.get(f"/api/experiments/{exp_id}/results").json()
    engineered = results["preprocessing_summary"]["engineered_columns"]
    assert any("log1p" in c for c in engineered), engineered
    assert any("freq" in c for c in engineered), engineered

    # prediction still works after operation-mode training (cached FE+preprocessor)
    resp = client.post(f"/api/experiments/{exp_id}/predict",
                       json={"num_a": 10, "num_b": 3.5, "cat_c": "low"})
    assert resp.status_code == 200, resp.text
    assert "prediction" in resp.json()

