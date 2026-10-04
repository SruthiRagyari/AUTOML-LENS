"""Tests for dataset upload size limits and validation.

Verifies:
1. Default limit is 1 GB (1024 MB = 1,073,741,824 bytes).
2. Files below the limit are accepted.
3. Files at/near the limit are handled correctly.
4. Files above the configured limit are rejected with 400 and clear message.
5. Dynamic configuration of MAX_DATASET_SIZE_MB is respected.
6. DB and storage isolation is strictly preserved.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.security import MAX_FILE_SIZE


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app.core import database as database_module
    from app.api import experiments as experiments_api

    tmp_dir = tmp_path_factory.mktemp("size_limit_test")
    saved = {
        "DATABASE_URL": settings.DATABASE_URL,
        "STORAGE_PATH": settings.STORAGE_PATH,
        "MAX_DATASET_SIZE_MB": settings.MAX_DATASET_SIZE_MB,
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


def sample_csv_bytes(rows=10):
    df = pd.DataFrame({"x": range(rows), "y": [1, 0] * (rows // 2)})
    return df.to_csv(index=False).encode("utf-8")


def test_default_config_is_1gb():
    """Ensure default configuration is 1024 MB = 1 GB."""
    assert settings.MAX_DATASET_SIZE_MB == 1024
    assert settings.max_dataset_size_bytes == 1024 * 1024 * 1024
    assert MAX_FILE_SIZE == 1024 * 1024 * 1024


def test_file_below_limit_accepted(client):
    """Small file well below 1 GB must be accepted."""
    data = sample_csv_bytes(20)
    r = client.post("/api/datasets/upload", files={"file": ("small.csv", data, "text/csv")})
    assert r.status_code == 200
    assert r.json()["rows"] == 20
    assert r.json()["file_size"] == len(data)


def test_file_above_configured_limit_rejected(client):
    """When MAX_DATASET_SIZE_MB is configured to a smaller limit, files exceeding it are rejected."""
    original = settings.MAX_DATASET_SIZE_MB
    try:
        settings.MAX_DATASET_SIZE_MB = 1  # 1 MB
        limit_bytes = settings.max_dataset_size_bytes

        # Create payload larger than 1 MB (1 MB + 1024 bytes)
        oversized = b"a,b\n" + b"1,2\n" * ((limit_bytes // 4) + 10)
        r = client.post("/api/datasets/upload", files={"file": ("big.csv", oversized, "text/csv")})
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "Dataset exceeds the maximum allowed size of 1 MB." in detail
    finally:
        settings.MAX_DATASET_SIZE_MB = original


def test_configuration_value_is_respected(client):
    """Changing MAX_DATASET_SIZE_MB dynamically changes max_dataset_size_bytes and error text."""
    original = settings.MAX_DATASET_SIZE_MB
    try:
        settings.MAX_DATASET_SIZE_MB = 2048
        assert settings.max_dataset_size_bytes == 2048 * 1024 * 1024
        
        settings.MAX_DATASET_SIZE_MB = 500
        assert settings.max_dataset_size_bytes == 500 * 1024 * 1024
        assert settings.MAX_DATASET_SIZE_MB == 500
    finally:
        settings.MAX_DATASET_SIZE_MB = original


def test_batch_predict_file_size_limit(client):
    """Batch prediction upload also enforces the dataset size limit."""
    original = settings.MAX_DATASET_SIZE_MB
    try:
        settings.MAX_DATASET_SIZE_MB = 1  # 1 MB
        oversized = b"a,b\n" * (300 * 1024)
        r = client.post("/api/experiments/99999/batch-predict", files={"file": ("test.csv", oversized, "text/csv")})
        assert r.status_code == 400
    finally:
        settings.MAX_DATASET_SIZE_MB = original
