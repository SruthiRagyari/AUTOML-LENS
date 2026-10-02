"""API-level test: optimization provenance is persisted and served back."""
import io
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """App bound to a temporary database/storage with the LLM fallback pinned."""
    from app.core.config import settings
    from app.core import database as database_module
    from app.api import experiments as experiments_api

    tmp_dir = tmp_path_factory.mktemp("optimization_api")
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


def _csv(rows=140, seed=5):
    rng = np.random.RandomState(seed)
    a = rng.normal(size=rows)
    b = rng.normal(size=rows)
    c = rng.choice(["x", "y", "z"], rows)
    y = ((a + b > 0) ^ (c == "x")).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c, "target": y})


def _upload_and_train(client, name, problem_type="classification",
                      metric="f1_weighted", fast_demo=True):
    up = client.post("/api/datasets/upload",
                     files={"file": (f"{name}.csv", _csv().to_csv(index=False).encode(),
                                     "text/csv")})
    assert up.status_code == 200, up.text
    ds_id = up.json()["id"]
    exp = client.post("/api/experiments", json={
        "dataset_id": ds_id, "name": name, "target_column": "target",
        "problem_type": problem_type, "primary_metric": metric,
        "n_folds": 5, "n_trials": 8, "mode": "automl",
    })
    assert exp.status_code == 200, exp.text
    exp_id = exp.json()["id"]
    suffix = "?fast_demo=true" if fast_demo else ""
    trained = client.post(f"/api/experiments/{exp_id}/train{suffix}")
    assert trained.status_code == 200, trained.text
    return exp_id, trained.json()


def test_optimization_budget_is_reported_honestly(client):
    """fast_demo shrinks the budget and the API must say so, not hide it."""
    _, body = _upload_and_train(client, "opt_budget", fast_demo=True)
    budget = body["optimization_budget"]
    assert budget["configured_n_trials"] == 8
    assert budget["n_trials_used"] == 5          # fast demo override
    assert budget["configured_n_folds"] == 5
    assert budget["n_folds_used"] == 3
    assert budget["fast_demo"] is True
    assert budget["budget_reduced"] is True      # labelled honestly
    assert budget["seed"] == 42
    spec = budget["metric_spec"]
    assert spec["metric"] == "f1_weighted"
    assert spec["scoring"] == "f1_weighted"
    assert spec["direction"] == "maximize"


def test_each_model_carries_real_optimization_provenance(client):
    _, body = _upload_and_train(client, "opt_prov", fast_demo=True)
    models = body["models"]
    assert models
    for m in models:
        o = m.get("optimization")
        assert o, f"{m['model_name']} has no optimization provenance"
        assert o["metric"] == "f1_weighted"
        assert o["scoring"] == "f1_weighted"
        assert o["direction"] == "maximize"
        assert o["raw_direction"] == "maximize"
        assert o["n_trials_requested"] == 5
        assert o["n_trials_completed"] + o["n_trials_failed"] == 5
        assert o["n_folds_requested"] == 3
        assert o["n_folds_used"] == 3
        assert o["cv_strategy"] == "StratifiedKFold"
        assert o["seed"] == 42
        assert o["fast_demo"] is True
        assert o["status"] in ("COMPLETED", "PARTIAL", "FAILED", "SKIPPED")
        # Real trials recorded, or an honest zero with a real reason.
        assert len(m["optimization_history"]) == o["n_trials_requested"]
        if o["status"] == "COMPLETED":
            assert o["best_cv_score"] is not None
            assert isinstance(o["best_cv_score"], float)
        # best_params must match what the optimizer reported.
        assert m["best_params"] == o["best_params"]


def test_optimization_provenance_is_persisted(client):
    exp_id, _ = _upload_and_train(client, "opt_persist", fast_demo=True)
    res = client.get(f"/api/experiments/{exp_id}/results")
    assert res.status_code == 200, res.text
    body = res.json()
    for m in body["models"]:
        assert m.get("optimization"), "provenance lost on /results"
        assert m["optimization"]["metric"] == "f1_weighted"
    # The budget survives a refresh too.
    budget = (body.get("model_selection") or {}).get("optimization_budget")
    assert budget and budget["seed"] == 42
    assert budget["metric_spec"]["direction"] == "maximize"

    # And it is in the per-model DB rows.
    rows = client.get(f"/api/experiments/{exp_id}/models").json()
    assert rows and all(r.get("optimization") for r in rows)


def test_regression_provenance_over_api(client):
    _, body = _upload_and_train(
        client, "opt_reg", problem_type="regression",
        metric="neg_root_mean_squared_error", fast_demo=True,
    )
    # This dataset's target is binary, so regression metrics are what we asked
    # for; the point is the recorded direction contract, not the score.
    for m in body["models"]:
        o = m.get("optimization") or {}
        if not o:
            continue
        assert o["metric"] == "neg_root_mean_squared_error"
        assert o["scoring"] == "neg_root_mean_squared_error"
        assert o["direction"] == "maximize"       # study maximizes negated RMSE
        assert o["raw_direction"] == "minimize"   # reported RMSE lower is better
        assert o["negated"] if "negated" in o else True
