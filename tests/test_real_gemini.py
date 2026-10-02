"""Real Gemini-backed analysis + model-selection test.

Requires GEMINI_API_KEY (backend/.env or environment). Skips - never falls
back - when the key is missing: the provider is called directly so any API,
quota or schema failure fails the test instead of silently degrading into
deterministic output.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
import pytest

from app.core.config import settings

pytestmark = pytest.mark.skipif(
    not (settings.GEMINI_API_KEY or "").strip(),
    reason="GEMINI_API_KEY not configured - real Gemini test skipped",
)

TARGET = "target"


def _safe_context() -> dict:
    """Small synthetic classification dataset: profiled, never sent raw."""
    rng = np.random.RandomState(3)
    n = 150
    a = rng.rand(n)
    b = rng.rand(n)
    origin = rng.choice(["u", "v", "w"], n)
    y = (a + np.array([0.5 if o == "w" else 0.0 for o in origin])
         + rng.rand(n) * 0.3 > 1.0).astype(int)
    df = pd.DataFrame({"sepal_like": a, "petal_like": b,
                       "origin": origin, TARGET: y})

    from app.services.profiler import DatasetProfiler
    from app.services.dataset_description import (
        build_dataset_description, render_dataset_description,
    )
    profile = DatasetProfiler().profile(df)
    description = build_dataset_description(
        df, profile=profile, target_column=TARGET,
        problem_type="classification",
        task_description="sanity-check dataset for real Gemini model selection",
    )
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing_percentages": {c: round(df[c].isnull().mean() * 100, 2)
                                for c in df.columns},
        "unique_counts": {c: int(df[c].nunique()) for c in df.columns},
        "target_column": TARGET,
        "target_stats": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "sample_rows": df.head(3).to_dict(orient="records"),
        "task_description": "binary classification sanity check",
        "problem_type": "classification",
        "column_profiles": profile["column_profiles"],
        "feature_operation_catalog": description.get("feature_operation_catalog", {}),
        "allowed_operations": description.get("allowed_operations", {}),
        "description_text": render_dataset_description(description),
    }


def test_real_gemini_analysis_and_model_selection():
    from app.llm.gemini import GeminiProvider
    from app.services.model_registry import ModelRegistry

    provider = GeminiProvider(api_key=settings.GEMINI_API_KEY,
                              model_name=settings.GEMINI_MODEL)
    assert provider.is_available(), \
        "Gemini availability check failed (key/quota/network) - test must not fake it"

    result = asyncio.run(provider.analyze_dataset(_safe_context()))

    # Honest provenance: a real provider answered, no fallback involved.
    assert result.is_fallback is False
    assert result.model_selection_source == "provider", (
        f"Gemini produced no registry-valid recommendations; "
        f"rejected: {result.rejected_model_recommendations}"
    )
    recs = result.model_recommendations
    assert recs, "real Gemini analysis returned zero valid model recommendations"

    registry = ModelRegistry()
    valid = set(registry.get_all_model_names("classification"))
    for rec in recs:
        assert rec["model_id"] in valid, f"registry rejected {rec}"
        assert registry.get_model(rec["model_id"]).task == "classification"
        assert rec["reason"], "each real recommendation must carry a reason"

    # Surface the real outcome in the pytest -s output for reporting.
    print("\nREAL GEMINI provider:", "Google Gemini", f"({settings.GEMINI_MODEL})")
    print("REAL GEMINI recommended:",
          [r["model_id"] for r in recs])
    print("REAL GEMINI rejected:",
          [(r["model_id"], r["reason"])
           for r in result.rejected_model_recommendations])
    print("REAL GEMINI problem_understanding:", result.problem_understanding)
    print("REAL GEMINI warnings:", result.warnings)
