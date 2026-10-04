"""Regression tests for the Target Column dropdown in the New Experiment flow.

Symptom (user-reported): after uploading a dataset on ``/dashboard?new=1`` the
"Dataset Readiness" panel rendered correctly, but the *Target Column*
``<select>`` had no options at all. Because the ``<select required>`` had a
``value`` (the suggested target) with no matching ``<option>``, the browser
blocked submission with "Select an item in the list" and the experiment could
never be launched.

Root cause: ``Dashboard.onDrop`` read ``column_names`` off the **profile**
response. ``column_names`` is only returned by the **upload** response; the
profile response exposes the columns as ``column_profiles``
(``[{"name": ..., ...}]``). The list therefore arrived ``undefined`` -> ``[]``
and the dropdown rendered nothing.

These tests pin the contract the creation flow depends on: after upload and
profiling, the *real* dataset columns must be reachable -- under
``column_profiles`` (the key the UI now derives from) and, for consistency with
the upload response, under ``column_names``. Nothing here hard-codes a dataset
or a specific column name.
"""
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_JSX = REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"
XLSX_MIME = ("application/vnd.openxmlformats-officedocument."
             "spreadsheetml.sheet")


# ───────────────────────────── sample data ────────────────────────────
def classification_frame() -> pd.DataFrame:
    """Categorical target, so the label can only be chosen if it is listed."""
    rng = np.random.RandomState(3)
    n = 120
    return pd.DataFrame({
        "age": rng.randint(18, 70, n),
        "income": rng.uniform(20_000, 150_000, n).round(2),
        "city": rng.choice(["Oslo", "Bergen", "Tromso"], n),
        "churn": rng.choice(["yes", "no"], n),
    })


def regression_frame() -> pd.DataFrame:
    """Continuous target with an unambiguous regression label."""
    rng = np.random.RandomState(5)
    n = 140
    rooms = rng.randint(1, 8, n)
    area = rng.uniform(30, 250, n).round(2)
    return pd.DataFrame({
        "area_sqm": area,
        "rooms": rooms,
        "price": (area * 3100 + rooms * 9000 + rng.normal(0, 5000, n)).round(2),
    })


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def xlsx_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="data")
    return buffer.getvalue()


def _serialize(filename: str, frame: pd.DataFrame) -> bytes:
    return xlsx_bytes(frame) if filename.endswith(".xlsx") else csv_bytes(frame)


# ───────────────────────────── fixtures ───────────────────────────────
@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """App bound to a temporary database/storage with the LLM fallback pinned.

    Mirrors the throwaway-app fixture used elsewhere: the developer's
    ``automl_lens.db`` and ``backend/storage`` are never touched.
    """
    from app.core.config import settings
    from app.core import database as database_module
    from app.api import experiments as experiments_api

    tmp_dir = tmp_path_factory.mktemp("target_column_flow")
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


def upload(client: TestClient, filename: str, payload: bytes,
           mime: str = "text/csv") -> int:
    resp = client.post("/api/datasets/upload",
                       files={"file": (filename, io.BytesIO(payload), mime)})
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def dropdown_options(client: TestClient, dataset_id: int):
    """The column names the New Experiment UI derives its options from."""
    profile = client.get(f"/api/datasets/{dataset_id}/profile")
    assert profile.status_code == 200, profile.text
    body = profile.json()
    profiles = body.get("column_profiles") or []
    return [cp["name"] for cp in profiles], body


# ───────────────────────── coverage of the contract ───────────────────
def test_profile_column_profiles_is_the_dropdown_source(client):
    """The creation UI derives options from ``column_profiles[].name``."""
    frame = classification_frame()
    dataset_id = upload(client, "clf.csv", csv_bytes(frame))

    names, _ = dropdown_options(client, dataset_id)
    assert names, "profile returned no column profiles -> empty dropdown"
    assert names == list(frame.columns)
    assert set(names) == set(frame.columns)


def test_profile_also_exposes_column_names(client):
    """``column_names`` must agree with the upload response for the same data.

    Reading this key off the profile response is what returned ``undefined``
    and left the dropdown empty; the field now exists and matches the columns.
    """
    frame = classification_frame()
    dataset_id = upload(client, "clf_names.csv", csv_bytes(frame))

    _, body = dropdown_options(client, dataset_id)
    assert body.get("column_names") == list(frame.columns)

    # The upload response and the profile response must not disagree.
    upload_resp = client.post(
        "/api/datasets/upload",
        files={"file": ("clf_names2.csv", io.BytesIO(csv_bytes(frame)),
                        "text/csv")},
    )
    assert upload_resp.json()["column_names"] == body["column_names"]


def test_columns_endpoint_matches_the_profile(client):
    """The dedicated /columns fallback returns the same ordered names."""
    frame = classification_frame()
    dataset_id = upload(client, "clf_cols.csv", csv_bytes(frame))

    names, _ = dropdown_options(client, dataset_id)
    columns = client.get(f"/api/datasets/{dataset_id}/columns")
    assert columns.status_code == 200, columns.text
    assert columns.json()["columns"] == names


def test_dashboard_reads_column_profiles_and_not_a_missing_key():
    """Source guard: the upload handler must not read ``column_names`` off the
    profile response (the exact bug), must derive options from
    ``column_profiles``, and must keep the /columns fallback."""
    source = DASHBOARD_JSX.read_text(encoding="utf-8")
    assert "profRes.data.column_names" not in source, (
        "Dashboard still reads column_names from the profile response")
    assert "column_profiles" in source, (
        "Dashboard no longer derives options from column_profiles")
    assert "columnNamesFromProfile" in source, (
        "Dashboard lost the profile -> option-list derivation")
    assert "api.getColumns" in source, (
        "Dashboard lost the /columns fallback for the option list")


# ─────────────── the full create flow, per file type ──────────────────
@pytest.mark.parametrize(
    "filename,mime,builder,expected_target",
    [
        ("iris_like.csv", "text/csv", classification_frame, "churn"),
        ("houses_reg.csv", "text/csv", regression_frame, "price"),
        ("sheet.xlsx", XLSX_MIME, classification_frame, "churn"),
    ],
)
def test_every_dataset_column_is_selectable(client, filename, mime, builder,
                                            expected_target):
    """Drop-in coverage: CSV classification, CSV regression and XLSX.

    The Target Column options must be exactly the uploaded dataset's columns,
    every time, for every supported file type -- no special-casing.
    """
    frame = builder()
    dataset_id = upload(client, filename, _serialize(filename, frame), mime)

    names, _ = dropdown_options(client, dataset_id)
    assert set(names) == set(frame.columns), (
        f"{filename}: dropdown options {names} != columns {list(frame.columns)}")

    # The suggested target must itself be a real, selectable column.
    suitability = client.get(f"/api/datasets/{dataset_id}/suitability")
    assert suitability.status_code == 200, suitability.text
    suit = suitability.json()
    assert suit["suggested_target"] in names
    assert suit["target_column"] in names

    # And the flow accepts that exact column as the experiment target.
    created = client.post("/api/experiments", json={
        "dataset_id": dataset_id,
        "target_column": expected_target,
        "problem_type": "auto",
        "primary_metric": None,
        "mode": "baseline",
        "name": f"target-flow::{filename}",
    })
    assert created.status_code == 200, created.text
    exp_id = created.json()["id"]
    stored = client.get(f"/api/experiments/{exp_id}").json()
    assert stored["target_column"] == expected_target


def test_creation_flow_accepts_each_real_column_as_target(client):
    """The user must be able to pick *any* real column, not just the label."""
    frame = regression_frame()
    dataset_id = upload(client, "anycol.csv", csv_bytes(frame))
    names, _ = dropdown_options(client, dataset_id)

    for column in names:
        created = client.post("/api/experiments", json={
            "dataset_id": dataset_id,
            "target_column": column,
            "problem_type": "auto",
            "primary_metric": None,
            "mode": "baseline",
            "name": f"anycol::{column}",
        })
        assert created.status_code == 200, f"{column}: {created.text}"
