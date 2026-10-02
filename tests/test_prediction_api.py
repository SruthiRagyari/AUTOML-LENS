"""API tests for prediction input validation, input-schema reliability and
train/response consistency.

The whole app is exercised through ``TestClient`` against a throwaway SQLite
database and storage directory, so nothing here touches the developer's
``automl_lens.db`` or ``backend/storage``.
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.services.predictor import InputValidationError, validate_prediction_inputs

TARGET = "target"
REQUIRED_FIELDS = ["num_a", "num_b", "cat_c"]


# ───────────────────────────── helpers ────────────────────────────────
def sample_csv() -> bytes:
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
    }).to_csv(index=False).encode("utf-8")


def upload_dataset(client: TestClient, name: str) -> int:
    filename = name if name.lower().endswith((".csv", ".xlsx")) else f"{name}.csv"
    resp = client.post(
        "/api/datasets/upload",
        files={"file": (filename, io.BytesIO(sample_csv()), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def train_experiment(client: TestClient, name: str):
    """upload -> create -> analyze -> train (never calls the profile endpoint)."""
    dataset_id = upload_dataset(client, name)
    created = client.post("/api/experiments", json={
        "dataset_id": dataset_id,
        "target_column": TARGET,
        "problem_type": "classification",
        "primary_metric": "f1_weighted",
        "mode": "baseline",
        "name": name,
    })
    assert created.status_code == 200, created.text
    exp_id = created.json()["id"]

    analyzed = client.post(f"/api/experiments/{exp_id}/analyze")
    assert analyzed.status_code == 200, analyzed.text

    # Same call shape the frontend uses: fast_demo is a query parameter.
    trained = client.post(f"/api/experiments/{exp_id}/train?fast_demo=true")
    assert trained.status_code == 200, trained.text
    return dataset_id, exp_id, trained.json()


def valid_input() -> dict:
    return {"num_a": 10, "num_b": 3.5, "cat_c": "low"}


# ───────────────────────────── fixtures ───────────────────────────────
@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """App bound to a temporary database/storage with the LLM fallback pinned."""
    from app.core.config import settings
    from app.core import database as database_module
    from app.api import experiments as experiments_api

    tmp_dir = tmp_path_factory.mktemp("prediction_api")
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


@pytest.fixture(scope="module")
def trained(client):
    """One shared trained experiment for the prediction tests."""
    return train_experiment(client, "prediction_api_exp")


# ──────────────────── 1. prediction input validation ──────────────────
def test_predict_rejects_empty_payload(client, trained):
    _, exp_id, _ = trained
    resp = client.post(f"/api/experiments/{exp_id}/predict", json={})
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert "missing required input" in detail
    for field in REQUIRED_FIELDS:
        assert field in detail


def test_predict_rejects_partially_missing_payload(client, trained):
    _, exp_id, _ = trained
    resp = client.post(f"/api/experiments/{exp_id}/predict",
                       json={"num_a": 10, "cat_c": "low"})
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert "num_b" in detail
    assert "num_a" not in detail and "cat_c" not in detail


def test_predict_treats_blank_and_null_as_missing(client, trained):
    _, exp_id, _ = trained
    resp = client.post(f"/api/experiments/{exp_id}/predict",
                       json={"num_a": "   ", "num_b": None, "cat_c": "mid"})
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert "num_a" in detail and "num_b" in detail


def test_predict_rejects_non_numeric_value_for_numeric_field(client, trained):
    _, exp_id, _ = trained
    payload = valid_input()
    payload["num_a"] = "not-a-number"
    resp = client.post(f"/api/experiments/{exp_id}/predict", json=payload)
    assert resp.status_code == 422, resp.text
    assert "num_a" in resp.json()["detail"]


def test_predict_accepts_complete_payload(client, trained):
    _, exp_id, _ = trained
    resp = client.post(f"/api/experiments/{exp_id}/predict", json=valid_input())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "prediction" in body
    assert body.get("model_name")


def test_predict_accepts_numeric_strings_from_html_forms(client, trained):
    """The browser sends number inputs as strings — they must still work."""
    _, exp_id, _ = trained
    resp = client.post(f"/api/experiments/{exp_id}/predict",
                       json={"num_a": "0", "num_b": "5.5", "cat_c": "high"})
    assert resp.status_code == 200, resp.text
    assert "prediction" in resp.json()


# ─────────────────── 2. input-schema reliability ──────────────────────
def test_input_schema_works_without_calling_profile(client):
    """A freshly trained experiment answers input-schema with no profile call."""
    dataset_id, exp_id, _ = train_experiment(client, "no_profile_exp")

    dataset_before = client.get(f"/api/datasets/{dataset_id}").json()
    assert dataset_before["has_profile"] is False, "test must not profile first"

    resp = client.get(f"/api/experiments/{exp_id}/input-schema")
    assert resp.status_code == 200, resp.text
    fields = resp.json()["fields"]
    names = [f["name"] for f in fields]
    assert sorted(names) == sorted(REQUIRED_FIELDS)
    assert TARGET not in names
    assert all(f["required"] is True for f in fields)
    assert all(f["type"] in ("number", "text") for f in fields)

    # The profile is cached now, so the datasets endpoint agrees as well.
    assert client.get(f"/api/datasets/{dataset_id}").json()["has_profile"] is True


def test_input_schema_drives_prediction_requirements(client, trained):
    """Whatever the schema declares required is what predict demands."""
    _, exp_id, _ = trained
    fields = client.get(f"/api/experiments/{exp_id}/input-schema").json()["fields"]
    payload = {f["name"]: v for f, v in zip(fields, [1, 2, "mid"])}
    dropped = payload.pop(fields[0]["name"])
    resp = client.post(f"/api/experiments/{exp_id}/predict", json=payload)
    assert resp.status_code == 422, resp.text
    assert fields[0]["name"] in resp.json()["detail"]
    assert dropped is not None


# ─────────────────── 3. train response consistency ────────────────────
def test_train_response_reports_real_best_model(client, trained):
    _, exp_id, body = trained
    winner = next(m for m in body["models"] if m["is_best"])
    metrics = winner["optimized_metrics"] or winner["baseline_metrics"]

    assert body["best_model_name"] == winner["display_name"]
    # The score must be the winner's real value for the metric named back to us.
    assert body["best_metric"] in metrics
    assert body["best_score"] == pytest.approx(float(metrics[body["best_metric"]]))
    assert body["best_metric"] == "f1_weighted"

    assert body["best_model"]["model_name"] == winner["model_name"]
    assert body["best_model"]["display_name"] == winner["display_name"]
    assert body["best_model"]["score"] == pytest.approx(body["best_score"])

    experiment = client.get(f"/api/experiments/{exp_id}").json()
    assert experiment["best_model_name"] == body["best_model_name"]
    assert experiment["best_score"] == pytest.approx(body["best_score"])


def test_results_response_agrees_with_train_response(client, trained):
    _, exp_id, body = trained
    results = client.get(f"/api/experiments/{exp_id}/results").json()
    assert results["best_model_name"] == body["best_model_name"]
    assert results["best_score"] == pytest.approx(body["best_score"])
    assert results["best_model"]["model_name"] == body["best_model"]["model_name"]
    assert results["best_model"]["score"] == pytest.approx(body["best_score"])


# ──────── 4. validator behaviour (optional fields / bad values) ───────
def test_validator_allows_optional_fields_to_be_omitted():
    schema = [
        {"name": "num_a", "type": "number", "required": True},
        {"name": "extra", "type": "text", "required": False},
    ]
    cleaned = validate_prediction_inputs({"num_a": 1}, schema)
    assert cleaned == {"num_a": 1}

    with pytest.raises(InputValidationError) as err:
        validate_prediction_inputs({}, schema)
    assert err.value.missing_fields == ["num_a"]


def test_validator_rejects_invalid_values_and_keeps_extra_keys():
    schema = [{"name": "num_a", "type": "number", "required": True}]
    with pytest.raises(InputValidationError) as err:
        validate_prediction_inputs({"num_a": ["1", "2"]}, schema)
    assert "num_a" in err.value.invalid_fields

    cleaned = validate_prediction_inputs({"num_a": "2.5", "ignored": "x"}, schema)
    assert cleaned["num_a"] == 2.5
    assert cleaned["ignored"] == "x"


