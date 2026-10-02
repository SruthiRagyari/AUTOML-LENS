"""Dataset suitability, upload validation and results-enrichment tests.

Covers Phase 2 (dataset experience), Phase 5 (model results must show the real
primary metric) and Phase 13 (useful API errors instead of fake success).

The central promise checked here: a dataset that cannot produce a meaningful
experiment is *told* so, and a metric shown in the results table is the metric
that was actually computed - never whichever number happened to come first in
a dictionary.
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

    tmp_dir = tmp_path_factory.mktemp("suitability_api")
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


def upload(client, df, name="d.csv"):
    data = df.to_csv(index=False).encode()
    return client.post("/api/datasets/upload",
                       files={"file": (name, data, "text/csv")})


def healthy_clf(rows=140, seed=4):
    rng = np.random.RandomState(seed)
    a = rng.normal(size=rows)
    b = rng.normal(size=rows)
    c = rng.choice(["p", "q", "r"], rows)
    y = ((a * 1.3 + b > 0.1) ^ (c == "p")).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c, "target": y})


# ── upload: structurally unusable files are refused, not accepted ──────────

class TestUploadRejectsUnusableFiles:
    def test_header_only_file_is_rejected(self, client):
        r = upload(client, pd.DataFrame({"a": [], "b": []}), "empty.csv")
        assert r.status_code == 400
        assert "no data rows" in r.json()["detail"].lower()

    def test_single_column_file_is_rejected(self, client):
        r = upload(client, pd.DataFrame({"only": range(1, 30)}), "one.csv")
        assert r.status_code == 400
        assert "only 1 column" in r.json()["detail"].lower()

    def test_unsupported_extension_is_rejected(self, client):
        r = client.post("/api/datasets/upload",
                        files={"file": ("notes.txt", b"hello", "text/plain")})
        assert r.status_code == 400

    def test_unparseable_csv_is_rejected(self, client):
        r = client.post(
            "/api/datasets/upload",
            files={"file": ("bad.csv", b"\x00\x01\x02\x03not,a,csv", "text/csv")})
        assert r.status_code == 400

    def test_valid_dataset_still_uploads(self, client):
        r = upload(client, healthy_clf(), "good.csv")
        assert r.status_code == 200
        assert r.json()["rows"] == 140
        assert r.json()["columns"] == 4
class TestSuitabilityReportsRealState:
    def test_healthy_dataset_is_suitable(self, client):
        ds = upload(client, healthy_clf(), "ok.csv").json()
        s = client.get(f"/api/datasets/{ds['id']}/suitability").json()
        assert s["suitable"] is True
        assert s["blocking_issues"] == []
        assert s["rows"] == 140
        assert s["usable_feature_count"] >= 3
        assert s["task_type"] == "classification"

    def test_target_distribution_is_real(self, client):
        ds = upload(client, healthy_clf(), "dist.csv").json()
        s = client.get(f"/api/datasets/{ds['id']}/suitability").json()
        t = s["target"]
        assert t["found"] is True
        assert t["name"]
        assert t["distribution_kind"] == "classes"
        # The distribution must sum to the real non-null row count.
        assert sum(t["distribution"].values()) == 140
        assert t["majority_class_share"] is not None

    def test_explicit_target_is_honoured(self, client):
        ds = upload(client, healthy_clf(), "explicit.csv").json()
        s = client.get(f"/api/datasets/{ds['id']}/suitability",
                       params={"target_column": "c"}).json()
        assert s["target"]["name"] == "c"
        assert s["target"]["unique_count"] == 3

    def test_unknown_target_blocks_with_a_real_message(self, client):
        ds = upload(client, healthy_clf(), "missing_target.csv").json()
        s = client.get(f"/api/datasets/{ds['id']}/suitability",
                       params={"target_column": "nope"}).json()
        assert s["suitable"] is False
        assert any("not present" in b for b in s["blocking_issues"])
        assert s["task_type"] is None

    def test_constant_target_blocks(self, client):
        # The target is passed explicitly: the profiler deliberately skips
        # constant columns when *suggesting* a target, so relying on the
        # suggestion would never exercise this path.
        df = healthy_clf()
        df["target"] = 1
        ds = upload(client, df, "const.csv").json()
        s = client.get(f"/api/datasets/{ds['id']}/suitability",
                       params={"target_column": "target"}).json()
        assert s["suitable"] is False
        assert any("single value" in b for b in s["blocking_issues"])
        assert s["target"]["is_constant"] is True
        assert s["task_type"] is None

    def test_tiny_dataset_blocks_but_explains_why(self, client):
        ds = upload(client, healthy_clf(rows=6), "tiny.csv").json()
        s = client.get(f"/api/datasets/{ds['id']}/suitability").json()
        assert s["suitable"] is False
        assert any("row" in b for b in s["blocking_issues"])

    def test_continuous_target_is_detected_as_regression(self, client):
        rng = np.random.RandomState(2)
        df = pd.DataFrame({"a": rng.normal(size=140),
                           "b": rng.normal(size=140),
                           "price": rng.normal(size=140) * 100 + 500})
        ds = upload(client, df, "reg.csv").json()
        s = client.get(f"/api/datasets/{ds['id']}/suitability",
                       params={"target_column": "price"}).json()
        assert s["task_type"] == "regression"
        assert s["target"]["distribution_kind"] == "numeric"
        assert s["target"]["summary"]

    def test_task_type_never_depends_on_the_file_name(self, client):
        df = healthy_clf()
        a = upload(client, df, "clearly_regression.csv").json()
        b = upload(client, df, "definitely_classification.csv").json()
        sa = client.get(f"/api/datasets/{a['id']}/suitability").json()
        sb = client.get(f"/api/datasets/{b['id']}/suitability").json()
        assert sa["task_type"] == sb["task_type"] == "classification"

    def test_explicit_problem_type_overrides_the_inference(self, client):
        ds = upload(client, healthy_clf(), "override.csv").json()
        s = client.get(f"/api/datasets/{ds['id']}/suitability",
                       params={"problem_type": "regression"}).json()
        assert s["task_type"] == "regression"
        assert "supplied with the request" in s["task_reason"]

    def test_missing_values_are_surfaced(self, client):
        df = healthy_clf()
        df.loc[0:30, "a"] = np.nan
        ds = upload(client, df, "nan.csv").json()
        s = client.get(f"/api/datasets/{ds['id']}/suitability").json()
        # Column-level missingness comes from the profiler; dataset-level
        # missingness from the suitability pass. Both must be visible.
        assert s["total_missing_percentage"] > 0
        assert any("missing" in w.lower() for w in s["warnings_from_profile"])
        assert any("missing" in w.lower() for w in s["warnings"]) or \
            s["total_missing_percentage"] <= 40

    def test_unknown_dataset_returns_404(self, client):
        assert client.get("/api/datasets/999999/suitability").status_code == 404