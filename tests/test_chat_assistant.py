"""Chat regression tests: no secret/config dumps, real project answers, history."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app.core.config import settings
    from app.core import database as database_module
    from app.api import experiments as experiments_api

    tmp_dir = tmp_path_factory.mktemp("chat_regressions")
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


def _chat(client, message, mode="general", history=None):
    resp = client.post("/api/chat", json={
        "message": message,
        "experiment_id": None,
        "context_mode": mode,
        "history": history or [],
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_greeting_has_no_config_dump(client):
    body = _chat(client, "hii")
    text = body["response"].lower()
    assert body["is_fallback"] is True
    assert "backend/.env" not in body["response"]
    assert "gemini_api_key" not in text
    assert "llm_provider" not in text
    assert "deterministic fallback mode" not in body["response"]
    assert "hi" in text


def test_general_ml_question_answered_without_machine_hi_bug(client):
    body = _chat(client, "What is machine learning?")
    assert "machine learning" in body["response"].lower()
    assert "backend/.env" not in body["response"]


def test_photosynthesis_answered(client):
    body = _chat(client, "What is photosynthesis?")
    assert "photosynthesis" in body["response"].lower()


def test_python_decorators_resolves_to_decorators(client):
    body = _chat(client, "Explain Python decorators.")
    assert "decorator" in body["response"].lower()


def test_history_followup_resolves_topic(client):
    body = _chat(client, "Give me an example.", history=[
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning learns patterns."},
    ])
    assert "machine learning" in body["response"].lower()


def test_project_target_uses_real_context(client):
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    up = client.post("/api/datasets/upload", files={"file": ("chat.csv", df.to_csv(index=False), "text/csv")})
    assert up.status_code == 200, up.text
    exp = client.post("/api/experiments", json={
        "dataset_id": up.json()["id"], "target_column": "target",
        "problem_type": "classification", "primary_metric": "f1_weighted",
    })
    assert exp.status_code == 200, exp.text
    exp_id = exp.json()["id"]
    resp = client.post("/api/chat", json={
        "message": "What is my target?",
        "experiment_id": exp_id,
        "context_mode": "project",
        "history": [],
    })
    assert resp.status_code == 200, resp.text
    assert "target" in resp.json()["response"].lower()


def test_project_best_model_honest_when_untrained(client):
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    up = client.post("/api/datasets/upload", files={"file": ("chat2.csv", df.to_csv(index=False), "text/csv")})
    exp = client.post("/api/experiments", json={
        "dataset_id": up.json()["id"], "target_column": "target",
        "problem_type": "classification", "primary_metric": "f1_weighted",
    })
    exp_id = exp.json()["id"]
    resp = client.post("/api/chat", json={
        "message": "Why was this model selected?",
        "experiment_id": exp_id,
        "context_mode": "project",
        "history": [],
    })
    assert resp.status_code == 200, resp.text
    # Honest empty state, never a fabricated winner.
    assert "don't have" in resp.json()["response"].lower()
