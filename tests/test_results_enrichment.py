"""Results payload must expose the real primary metric per model.

The frontend used to pick "the first number in the metrics dict" to display as
the model's score. For a classifier that silently showed *accuracy* even when
the experiment had selected on F1; for a regressor it happened to show
whichever metric happened to be written first.

These tests pin that ``/results`` now reports the primary metric explicitly,
with the key actually used, so the UI never has to guess.
"""
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

    tmp_dir = tmp_path_factory.mktemp("results_enrich")
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


def _train(client, name, problem_type, metric, df):
    up = client.post("/api/datasets/upload",
                     files={"file": (f"{name}.csv",
                                     df.to_csv(index=False).encode(),
                                     "text/csv")})
    assert up.status_code == 200, up.text
    exp = client.post("/api/experiments", json={
        "dataset_id": up.json()["id"], "name": name,
        "target_column": "target", "problem_type": problem_type,
        "primary_metric": metric, "n_folds": 3, "n_trials": 3,
        "mode": "automl",
    })
    assert exp.status_code == 200, exp.text
    exp_id = exp.json()["id"]
    trained = client.post(f"/api/experiments/{exp_id}/train")
    assert trained.status_code == 200, trained.text
    return exp_id


def _clf(rows=150, seed=11):
    rng = np.random.RandomState(seed)
    a = rng.normal(size=rows)
    b = rng.normal(size=rows)
    y = ((a * 1.5 + b) > 0.15).astype(int)
    return pd.DataFrame({"a": a, "b": b, "target": y})


class TestClassificationPrimaryMetric:
    @pytest.fixture(scope="class")
    def exp_id(self, client):
        return _train(client, "enrich_clf", "classification",
                      "f1_weighted", _clf())

    def test_reported_key_is_the_selected_metric_not_the_first_number(
            self, client, exp_id):
        r = client.get(f"/api/experiments/{exp_id}/results").json()
        assert r["primary_metric"] == "f1_weighted"
        checked = 0
        for m in r["models"]:
            if m["status"] != "COMPLETED":
                continue
            met = m["optimized_metrics"] or m["baseline_metrics"]
            # accuracy IS present and IS written first in the dict; the payload
            # must still name f1_weighted.
            assert m["holdout_primary_key"] == "f1_weighted"
            assert m["holdout_primary"] == pytest.approx(met["f1_weighted"])
            checked += 1
        assert checked > 0

    def test_selection_evidence_is_attached_per_model(self, client, exp_id):
        r = client.get(f"/api/experiments/{exp_id}/results").json()
        cv = {c["model_name"]: c for c in r["selection"]["candidates_considered"]}
        for m in r["models"]:
            cand = cv.get(m["model_name"])
            assert cand is not None
            assert m["selection_score"] == pytest.approx(cand["cv_score"])
            assert m["selection_cv_folds"] == cand["cv_fold_count"]
            if cand["cv_fold_std"] is not None:
                assert m["selection_cv_std"] == pytest.approx(
                    cand["cv_fold_std"])

    def test_holdout_was_not_used_for_selection(self, client, exp_id):
        r = client.get(f"/api/experiments/{exp_id}/results").json()
        assert r["selection"]["holdout_used_for_selection"] is False


class TestRegressionPrimaryMetricUsesAlias:
    @pytest.fixture(scope="class")
    def exp_id(self, client):
        return _train(client, "enrich_reg", "regression",
                      "neg_root_mean_squared_error", _reg())

    def test_negated_loss_resolves_to_the_real_rmse_key(self, client, exp_id):
        r = client.get(f"/api/experiments/{exp_id}/results").json()
        assert r["primary_metric"] == "neg_root_mean_squared_error"
        checked = 0
        for m in r["models"]:
            if m["status"] != "COMPLETED":
                continue
            met = m["optimized_metrics"] or m["baseline_metrics"]
            assert m["holdout_primary_key"] == "rmse", \
                "neg_root_mean_squared_error must resolve to the stored rmse key"
            assert m["holdout_primary"] == pytest.approx(met["rmse"])
            assert m["holdout_primary"] >= 0, \
                "RMSE is stored positive; the UI must not show a negative error"
            checked += 1
        assert checked > 0


class TestFailedModelsAreNotPresentedAsSuccessful:
    def test_failed_model_carries_no_cv_evidence_and_no_metrics(self, client):
        exp_id = _train(client, "fail_case", "classification",
                        "f1_weighted", _clf(seed=5))
        r = client.get(f"/api/experiments/{exp_id}/results").json()
        for m in r["models"]:
            if m["status"] == "FAILED":
                assert m["selection_score"] is None
                assert not m.get("optimized_metrics")
                assert m["is_best"] is not True
def _reg(rows=150, seed=12):
    rng = np.random.RandomState(seed)
    a = rng.normal(size=rows)
    b = rng.normal(size=rows)
    y = a * 3.0 - b * 1.5 + rng.normal(scale=0.2, size=rows)
    return pd.DataFrame({"a": a, "b": b, "target": y})