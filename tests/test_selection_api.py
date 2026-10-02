"""API-level proof that the holdout never decides the winning model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app.core.config import settings
    from app.core import database as database_module
    from app.api import experiments as experiments_api

    tmp_dir = tmp_path_factory.mktemp("selection_api")
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


def _clf_csv(rows=160, seed=9):
    rng = np.random.RandomState(seed)
    a = rng.normal(size=rows)
    b = rng.normal(size=rows)
    c = rng.choice(["p", "q", "r"], rows)
    y = ((a * 1.4 + b > 0.2) ^ (c == "p")).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c, "target": y})


def _train(client, name, problem_type="classification",
           metric="f1_weighted", csv=None):
    data = (csv if csv is not None else _clf_csv()).to_csv(index=False).encode()
    up = client.post("/api/datasets/upload",
                     files={"file": (f"{name}.csv", data, "text/csv")})
    assert up.status_code == 200, up.text
    exp = client.post("/api/experiments", json={
        "dataset_id": up.json()["id"], "name": name,
        "target_column": "target", "problem_type": problem_type,
        "primary_metric": metric, "n_folds": 3, "n_trials": 4, "mode": "automl",
    })
    assert exp.status_code == 200, exp.text
    exp_id = exp.json()["id"]
    trained = client.post(f"/api/experiments/{exp_id}/train")
    assert trained.status_code == 200, trained.text
    return exp_id, trained.json()


def test_selection_provenance_is_published_over_api(client):
    _, body = _train(client, "sel_prov")
    sel = body["selection"]
    assert sel is not None
    assert sel["selection_metric"] == "f1_weighted"
    assert sel["selection_scoring"] == "f1_weighted"
    assert sel["selection_direction"] == "maximize"
    assert sel["evidence_source"] == "cross_validation_on_training_split"
    assert sel["holdout_used_for_selection"] is False
    assert sel["selected_model_cv_score"] == pytest.approx(
        body["selection_cv_score"])
    assert sel["final_holdout_metrics"] is not None
    assert len(sel["candidates_considered"]) == len(body["models"])


def test_winner_is_the_cv_argmax_not_the_holdout_argmax(client):
    """The decisive assertion: the winner must be the CV maximum."""
    _, body = _train(client, "sel_argmax")
    cands = body["selection"]["candidates_considered"]
    scored = [c for c in cands if c["cv_score"] is not None]
    assert scored, "no candidate had CV evidence"
    cv_winner = max(scored, key=lambda c: c["cv_score"])
    selected = next(c for c in cands if c["selected"])
    assert selected["model_name"] == cv_winner["model_name"]
    assert selected["cv_score"] == cv_winner["cv_score"]


def test_persisted_results_separate_cv_score_from_holdout_score(client):
    exp_id, body = _train(client, "sel_persist")
    sel = body["selection"]
    assert sel["selected_model_cv_score"] is not None
    assert sel["final_holdout_metrics"] is not None
    assert "f1_weighted" in sel["final_holdout_metrics"]
    # best_score remains the final HOLDOUT number of the chosen winner.
    winner = next(m for m in body["models"] if m["is_best"])
    holdout = winner.get("optimized_metrics") or winner.get("baseline_metrics")
    assert body["best_score"] == pytest.approx(holdout["f1_weighted"])
    assert body["selection_cv_score"] == pytest.approx(
        sel["selected_model_cv_score"])

    # Everything survives a refresh.
    results = client.get(f"/api/experiments/{exp_id}/results").json()
    assert results["selection"] is not None
    assert results["selection"]["selected_model"] == sel["selected_model"]
    assert results["selection"]["holdout_used_for_selection"] is False
    assert results["selection"]["selected_model_cv_score"] == pytest.approx(
        sel["selected_model_cv_score"])
    trace_sel = (results.get("model_selection") or {}).get("selection")
    assert trace_sel is not None
    assert trace_sel["evidence_source"] == "cross_validation_on_training_split"


def test_regression_selection_over_api(client):
    rng = np.random.RandomState(12)
    n = 160
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = x1 * 3.0 - x2 * 1.5 + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    _, body = _train(client, "sel_reg", problem_type="regression",
                     metric="neg_root_mean_squared_error", csv=df)
    sel = body["selection"]
    assert sel["selection_metric"] == "neg_root_mean_squared_error"
    assert sel["selection_scoring"] == "neg_root_mean_squared_error"
    # Negated RMSE: selection direction is maximize.
    assert sel["selection_direction"] == "maximize"
    assert sel["selection_raw_direction"] == "minimize"
    assert sel["holdout_used_for_selection"] is False
    scored = [c for c in sel["candidates_considered"] if c["cv_score"] is not None]
    assert scored
    # CV scores are genuinely negative for this metric.
    assert all(c["cv_score"] < 0 for c in scored)
    assert sel["selected_model_cv_score"] == max(c["cv_score"] for c in scored)


def test_failed_candidate_does_not_break_the_api_experiment(client, monkeypatch):
    """Inject a model that always fails; the run must still finish and select."""
    from app.services.model_registry import ModelRegistry

    original_get_models = ModelRegistry.get_models

    class _BrokenDef:
        name = "broken_model"
        display_name = "Broken Model"
        task = "classification"
        search_space = staticmethod(lambda trial: {})
        default_params: dict = {}

        def create_model(self, **p):
            raise RuntimeError("cannot construct broken model")

    def _patched(self, task, model_names=None):
        defs = original_get_models(self, task, model_names)
        return [_BrokenDef(), *defs][:4]

    monkeypatch.setattr(ModelRegistry, "get_models", _patched)
    _, body = _train(client, "sel_broken")
    assert body["status"] == "completed"
    assert body["best_model_name"], "a winner was still selected"
    statuses = {m["model_name"]: m["status"] for m in body["models"]}
    assert statuses.get("broken_model") == "FAILED"
    assert body["selection"]["selected_model"] is not None
    failed = [c for c in body["selection"]["candidates_considered"]
              if c["status"] == "FAILED"]
    assert any(c["model_name"] == "broken_model" for c in failed)
    # The broken model never got fabricated CV evidence.
    broken_cand = next(c for c in body["selection"]["candidates_considered"]
                       if c["model_name"] == "broken_model")
    assert broken_cand["cv_score"] is None

