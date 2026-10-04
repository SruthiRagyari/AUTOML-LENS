"""pytest configuration."""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# ---------------------------------------------------------------------------
# Production (live) state baseline.
#
# A few isolation tests assert that the test suite never writes to the live
# ``backend/automl_lens.db`` or ``backend/storage``. Those paths are a running,
# user-mutable application store -- a user can legitimately upload a dataset
# from the UI between test runs -- so the expected state is captured once here,
# before any test executes, instead of being hard-coded as a magic number.
# ---------------------------------------------------------------------------
PRODUCTION_DB_PATH = "backend/automl_lens.db"
PRODUCTION_STORAGE_PATH = "backend/storage"


def snapshot_production_state():
    """Read-only fingerprint of the live DB row counts and storage files."""
    counts = {"experiments": None, "datasets": None}
    if os.path.exists(PRODUCTION_DB_PATH):
        conn = sqlite3.connect(PRODUCTION_DB_PATH)
        try:
            cur = conn.cursor()
            counts["experiments"] = cur.execute(
                "SELECT COUNT(*) FROM experiments"
            ).fetchone()[0]
            counts["datasets"] = cur.execute(
                "SELECT COUNT(*) FROM datasets"
            ).fetchone()[0]
        finally:
            conn.close()
    files = sorted(
        os.path.join(root, name)
        for root, _, names in os.walk(PRODUCTION_STORAGE_PATH)
        for name in names
    )
    return {"counts": counts, "files": files}


# Captured at import time (pytest imports conftest before running any test).
PRODUCTION_BASELINE = snapshot_production_state()


@pytest.fixture(scope="session")
def production_baseline():
    """The live DB/storage fingerprint taken before the suite started."""
    return PRODUCTION_BASELINE
