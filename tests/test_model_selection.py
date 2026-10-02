"""Tests for LLM-assisted, registry-validated model selection.

Covers: structured recommendations accepted only when they name real registry
models compatible with the problem type; unknown / incompatible / over-budget
names rejected with reasons; deterministic fallback labelling; recommendation
flow into training; persisted research trace; full classification and
regression runs. No LLM API key is required - the real Gemini test lives in
test_real_gemini.py.
"""
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


def classification_csv() -> bytes:
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


def regression_csv() -> bytes:
    rng = np.random.RandomState(11)
    n = 120
    x1 = rng.normal(0, 1, n)
    x2 = rng.uniform(0, 5, n)
    grp = rng.choice(["a", "b", "c"], n)
    y = (3 * x1 + 1.5 * x2
         + np.array([0.5 if g == "c" else 0.0 for g in grp])
         + rng.normal(0, 0.3, n))
    return pd.DataFrame({"x1": x1, "x2": x2, "grp": grp, TARGET: y
                         }).to_csv(index=False).encode("utf-8")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app.core.config import settings
    from app.core import database as database_module
    from app.api import experiments as experiments_api

    tmp_dir = tmp_path_factory.mktemp("model_selection")
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


def create_and_analyze(client, name, csv_bytes=None, problem_type="classification",
                       metric="f1_weighted"):
    upload = client.post(
        "/api/datasets/upload",
        files={"file": (f"{name}.csv", io.BytesIO(csv_bytes or classification_csv()),
                        "text/csv")},
    )
    assert upload.status_code == 200, upload.text
    dataset_id = upload.json()["id"]
    created = client.post("/api/experiments", json={
        "dataset_id": dataset_id, "target_column": TARGET,
        "problem_type": problem_type, "primary_metric": metric,
        "mode": "baseline", "name": name,
    })
    assert created.status_code == 200, created.text
    exp_id = created.json()["id"]
    analyzed = client.post(f"/api/experiments/{exp_id}/analyze")
    assert analyzed.status_code == 200, analyzed.text
    return dataset_id, exp_id, analyzed.json()


def inject_recommendations(exp_id, recommendations, *, rejected=(), source="provider",
                           provider="Test Provider", is_fallback=False):
    """Store provider-style recommendations the way a real analysis would."""
    from app.core.database import get_session_factory, Experiment
    db = get_session_factory()()
    try:
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        payload = json.loads(exp.llm_analysis_json)
        payload["result"]["model_recommendations"] = list(recommendations)
        payload["result"]["rejected_model_recommendations"] = list(rejected)
        payload["result"]["model_selection_source"] = source
        payload["result"]["provider_used"] = provider
        payload["result"]["is_fallback"] = is_fallback
        payload["provider_used"] = provider
        payload["is_fallback"] = is_fallback
        exp.llm_analysis_json = json.dumps(payload)
        db.commit()
    finally:
        db.close()


def ctx(problem="classification"):
    return {"problem_type": problem, "target_column": TARGET,
            "columns": ["a", "b", TARGET]}


# ── 1. valid LLM model recommendation ────────────────────────────────────────
def test_valid_llm_model_recommendation_accepted():
    from app.llm.gemini import GeminiProvider
    provider = GeminiProvider(api_key="")
    result = provider.validate_analysis({
        "problem_type": "classification", "target_column": TARGET,
        "reasoning": "looks separable",
        "model_recommendations": [
            {"model_id": "random_forest_clf", "reason": "captures interactions",
             "suitability": "high", "strengths": "robust", "limitations": "memory"},
            "logistic_regression",  # bare-string quirk tolerated
        ],
    }, ctx())
    ids = [r["model_id"] for r in result.model_recommendations]
    assert ids == ["random_forest_clf", "logistic_regression"]
    assert result.model_recommendations[0]["display_name"].startswith("Random Forest")
    assert result.model_selection_source == "provider"
    assert result.rejected_model_recommendations == []


# ── 2. unknown model rejection ───────────────────────────────────────────────
def test_unknown_model_rejected():
    from app.llm.gemini import GeminiProvider
    provider = GeminiProvider(api_key="")
    result = provider.validate_analysis({
        "problem_type": "classification", "target_column": TARGET,
        "reasoning": "r",
        "model_recommendations": [
            {"model_id": "xgboost_forest_2099", "reason": "sounds good"},
        ],
    }, ctx())
    assert result.model_recommendations == []
    assert len(result.rejected_model_recommendations) == 1
    rejected = result.rejected_model_recommendations[0]
    assert rejected["model_id"] == "xgboost_forest_2099"
    assert "not in the model registry" in rejected["reason"]
    assert result.model_selection_source == "none"


# ── 3. incompatible model rejection ──────────────────────────────────────────
def test_incompatible_model_rejected():
    from app.llm.gemini import GeminiProvider
    provider = GeminiProvider(api_key="")
    result = provider.validate_analysis({
        "problem_type": "classification", "target_column": TARGET,
        "reasoning": "r",
        "model_recommendations": [{"model_id": "svr", "reason": "powerful"}],
        "candidate_models": ["svr", "knn_clf", "ridge"],
    }, ctx())
    assert result.model_recommendations == []
    assert "incompatible with classification" in \
        result.rejected_model_recommendations[0]["reason"]
    # legacy string list gets the same task-compatibility filter
    assert result.candidate_models == ["knn_clf"]


# ── 5. recommendation -> registry validation (unit) ─────────────────────────
def test_recommendation_registry_validation():
    from app.services.model_registry import (
        ModelRegistry, MAX_MODEL_RECOMMENDATIONS,
        get_model_catalog, validate_model_recommendations,
    )
    reg = ModelRegistry()
    recs = [
        {"model_id": "Random-Forest-Clf"},        # normalized alias
        {"model_id": "deepthought_70b"},          # unknown
        {"model_id": "svr"},                      # wrong task
        "logistic_regression",
        {"model_id": "logistic_regression"},      # duplicate
        {"model_id": 42},                         # no usable id
        {"model_id": "knn_clf"},
    ]
    accepted, rejected = validate_model_recommendations(recs, "classification")
    assert [r["model_id"] for r in accepted] == \
        ["random_forest_clf", "logistic_regression", "knn_clf"]
    reasons = " ".join(r["reason"] for r in rejected)
    assert "not in the model registry" in reasons
    assert "incompatible with classification" in reasons
    assert "duplicate recommendation" in reasons
    assert "no model_id provided" in reasons

    # budget cap: only registry task models, never more than the cap
    every_model = [{"model_id": m} for m in reg.get_all_model_names("classification")]
    capped, overflow = validate_model_recommendations(every_model, "classification")
    assert len(capped) == MAX_MODEL_RECOMMENDATIONS
    assert len(overflow) == len(every_model) - MAX_MODEL_RECOMMENDATIONS
    assert "budget cap" in overflow[0]["reason"]

    # prompt catalog is generated from the same registry
    catalog = get_model_catalog("regression")
    assert {c["name"] for c in catalog} == set(reg.get_all_model_names("regression"))
    assert all(c["task"] == "regression" for c in catalog)


# ── 4. deterministic fallback ────────────────────────────────────────────────
def test_deterministic_fallback_model_selection(client):
    _, exp_id, analysis = create_and_analyze(client, "ms_fallback")
    assert analysis["is_fallback"] is True
    result = analysis["result"]
    assert result["model_selection_source"] == "deterministic_defaults"
    recs = result["model_recommendations"]
    assert recs, "deterministic selection must still propose models"
    from app.services.model_registry import ModelRegistry
    reg = ModelRegistry()
    for rec in recs:
        model = reg.get_model(rec["model_id"])
        assert model is not None, rec
        assert model.task == "classification", rec
        assert rec["reason"], "deterministic picks must explain themselves"
    assert result["rejected_model_recommendations"] == []


# ── 6. recommendation -> training integration ────────────────────────────────
def test_recommendations_drive_training_candidates(client):
    _, exp_id, _ = create_and_analyze(client, "ms_train")
    inject_recommendations(exp_id, [
        {"model_id": "logistic_regression", "reason": "linear baseline"},
        {"model_id": "random_forest_clf", "reason": "non-linear interactions"},
        {"model_id": "xgboost_2099", "reason": "hallucinated name"},
        {"model_id": "svr", "reason": "wrong task"},
    ])

    trained = client.post(f"/api/experiments/{exp_id}/train")
    assert trained.status_code == 200, trained.text
    trace = trained.json()["model_selection"]

    assert trace["source"] == "provider"
    assert trace["provider_used"] == "Test Provider"
    assert trace["is_fallback"] is False
    assert [r["model_id"] for r in trace["recommended"]] == \
        ["logistic_regression", "random_forest_clf"]
    reasons = {r["model_id"]: r["reason"] for r in trace["rejected"]}
    assert "not in the model registry" in reasons["xgboost_2099"]
    assert "incompatible with classification" in reasons["svr"]

    # only the validated recommendations were scheduled and trained
    assert set(trace["models_trained"]) == {"logistic_regression", "random_forest_clf"}
    assert trace["final_model"] in trace["models_trained"]
    assert trace["metric"] and trace["final_score"] is not None
    assert "No A/B comparison" in trace["comparison_note"]

    results = client.get(f"/api/experiments/{exp_id}/results").json()
    assert {m["model_name"] for m in results["models"]} == \
        {"logistic_regression", "random_forest_clf"}
    assert results["model_selection"]["source"] == "provider"


# ── 7. persisted recommendation provenance ───────────────────────────────────
def test_recommendation_provenance_is_persisted(client):
    _, exp_id, _ = create_and_analyze(client, "ms_provenance")
    inject_recommendations(exp_id, [
        {"model_id": "random_forest_clf", "reason": "robust default"},
        {"model_id": "not_a_real_model", "reason": "made up"},
    ])

    trained = client.post(f"/api/experiments/{exp_id}/train?fast_demo=true")
    assert trained.status_code == 200, trained.text

    from app.core.database import get_session_factory, Experiment
    db = get_session_factory()()
    try:
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        trace = json.loads(exp.model_selection_json)
        analysis = json.loads(exp.llm_analysis_json)["result"]
    finally:
        db.close()

    # analysis-level provenance survives the round trip
    assert analysis["provider_used"] == "Test Provider"
    assert analysis["is_fallback"] is False
    assert analysis["model_selection_source"] == "provider"
    assert [r["model_id"] for r in analysis["model_recommendations"]] == \
        ["random_forest_clf", "not_a_real_model"]

    # train-time trace: recommendations, refusals, trained models, winner, metric
    assert trace["provider_used"] == "Test Provider"
    assert trace["is_fallback"] is False
    assert trace["source"] == "fast_demo"
    assert [r["model_id"] for r in trace["recommended"]] == ["random_forest_clf"]
    assert any("not in the model registry" in r["reason"] for r in trace["rejected"])
    assert trace["models_trained"], "trace must list models actually trained"
    assert trace["final_model"] in trace["models_trained"]
    assert trace["metric"] and trace["final_score"] is not None

    # and it is served back on /results and the experiment read
    results = client.get(f"/api/experiments/{exp_id}/results").json()
    assert results["model_selection"]["final_model"] == trace["final_model"]
    fetched = client.get(f"/api/experiments/{exp_id}").json()
    assert fetched["model_selection"]["provider_used"] == "Test Provider"


# ── 8. real end-to-end classification ────────────────────────────────────────
def test_end_to_end_classification(client):
    _, exp_id, analysis = create_and_analyze(client, "ms_e2e_clf")
    assert analysis["result"]["model_recommendations"], \
        "fallback must propose structured recommendations"
    trained = client.post(f"/api/experiments/{exp_id}/train?fast_demo=true")
    assert trained.status_code == 200, trained.text
    body = trained.json()
    assert body["best_model"] is not None
    results = client.get(f"/api/experiments/{exp_id}/results").json()
    trace = results["model_selection"]
    assert trace["models_trained"]
    assert trace["final_model"] in trace["models_trained"]
    assert trace["metric"] and trace["final_score"] is not None
    resp = client.post(f"/api/experiments/{exp_id}/predict",
                       json={"num_a": 10, "num_b": 3.5, "cat_c": "low"})
    assert resp.status_code == 200, resp.text
    assert "prediction" in resp.json()


# ── 9. real end-to-end regression ────────────────────────────────────────────
def test_end_to_end_regression(client):
    _, exp_id, analysis = create_and_analyze(
        client, "ms_e2e_reg", csv_bytes=regression_csv(),
        problem_type="regression", metric="r2",
    )
    assert analysis["result"]["model_selection_source"] == "deterministic_defaults"
    trained = client.post(f"/api/experiments/{exp_id}/train?fast_demo=true")
    assert trained.status_code == 200, trained.text
    body = trained.json()
    assert body["best_model"] is not None
    results = client.get(f"/api/experiments/{exp_id}/results").json()
    trace = results["model_selection"]
    assert trace["models_trained"]
    assert trace["final_model"] in trace["models_trained"]
    assert trace["metric"] and trace["final_score"] is not None
    reg_ids = {r["model_id"] for r in trace["recommended"]}
    from app.services.model_registry import ModelRegistry
    reg = ModelRegistry()
    assert reg_ids and all(reg.get_model(m).task == "regression" for m in reg_ids)
    resp = client.post(f"/api/experiments/{exp_id}/predict",
                       json={"x1": 0.5, "x2": 2.0, "grp": "c"})
    assert resp.status_code == 200, resp.text
    assert "prediction" in resp.json()
