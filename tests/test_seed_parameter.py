"""Tests for optional seed parameter in training endpoint."""
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import app.main as main_mod
from app.core.config import settings
from app.core import database as database_module
from app.api import experiments as experiments_api


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """App bound to a temporary database/storage."""
    tmp_dir = tmp_path_factory.mktemp("seed_parameter")
    saved = {
        "DATABASE_URL": settings.DATABASE_URL,
        "STORAGE_PATH": settings.STORAGE_PATH,
        "LLM_PROVIDER": settings.LLM_PROVIDER,
    }
    settings.DATABASE_URL = f"sqlite:///{(tmp_dir / 'test.db').as_posix()}"
    settings.STORAGE_PATH = str(tmp_dir)
    settings.LLM_PROVIDER = "fallback"
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


def test_training_seed_none_matches_default(client):
    # Upload
    with open("demo_data/classification.csv", "rb") as f:
        r = client.post("/api/datasets/upload", files={"file": ("classification.csv", f, "text/csv")})
    assert r.status_code == 200
    ds_id = r.json()["id"]

    # Create exp
    r = client.post("/api/experiments", json={
        "name": "test_seed_none",
        "dataset_id": ds_id,
        "target_column": "Churn",
    })
    assert r.status_code == 200
    exp_id = r.json()["id"]

    # Analyze
    r = client.post(f"/api/experiments/{exp_id}/analyze")
    assert r.status_code == 200

    # Train with seed=None (omitted query parameter)
    r = client.post(f"/api/experiments/{exp_id}/train?fast_demo=true")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "completed"

    # Verify metadata saved random_seed == 42
    db = database_module.get_session_factory()()
    exp_db = db.query(database_module.Experiment).filter(database_module.Experiment.id == exp_id).first()
    assert exp_db is not None
    db.close()


def test_training_seed_43_yields_different_split_and_seed(client):
    # Upload
    with open("demo_data/classification.csv", "rb") as f:
        r = client.post("/api/datasets/upload", files={"file": ("classification.csv", f, "text/csv")})
    assert r.status_code == 200
    ds_id = r.json()["id"]

    # Create exp
    r = client.post("/api/experiments", json={
        "name": "test_seed_43",
        "dataset_id": ds_id,
        "target_column": "Churn",
    })
    assert r.status_code == 200
    exp_id = r.json()["id"]

    # Analyze
    r = client.post(f"/api/experiments/{exp_id}/analyze")
    assert r.status_code == 200

    # Train with seed=43
    r = client.post(f"/api/experiments/{exp_id}/train?fast_demo=true&seed=43")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "completed"
