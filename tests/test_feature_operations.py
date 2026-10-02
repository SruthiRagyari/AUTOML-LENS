"""Unit tests for the controlled operation registry, the strict LLM analysis
schema and the compact dataset description (safety rails for LLM feature
engineering: no code execution, no target leakage, bounded proposals)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
import pytest

from app.llm.base import InvalidAnalysisResponse
from app.llm.fallback import FallbackLLMProvider
from app.services.dataset_description import (
    build_dataset_description,
    render_dataset_description,
)
from app.services.feature_operations import (
    MAX_OPERATIONS,
    normalize_operation_name,
    validate_operations,
)
from app.services.profiler import DatasetProfiler

TARGET = "target"


def make_df(n=120, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "income": rng.uniform(1000, 9000, n).round(2),
        "age": rng.randint(18, 80, n),
        "city": rng.choice(["a", "b", "c", "zzz"], n, p=[0.45, 0.4, 0.1, 0.05]),
        "note": rng.choice(["hello world", "hi there", "good day"], n),
        "constant": "same",
        TARGET: rng.choice([0, 1], n),
    })


@pytest.fixture(scope="module")
def df():
    return make_df()


@pytest.fixture(scope="module")
def column_profiles(df):
    return DatasetProfiler().profile(df)["column_profiles"]


def context(df, column_profiles, **overrides):
    ctx = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing_percentages": {},
        "unique_counts": {c: int(df[c].nunique()) for c in df.columns},
        "target_column": TARGET,
        "column_profiles": column_profiles,
    }
    ctx.update(overrides)
    return ctx


# ─── 1. registry alias normalization ────────────────────────────────────
def test_alias_normalization_maps_common_spellings():
    assert normalize_operation_name("log") == "log1p"
    assert normalize_operation_name("z_score") == "zscore"
    assert normalize_operation_name("standardize") == "zscore"
    assert normalize_operation_name("character_count") == "char_count"
    assert normalize_operation_name("frequency") == "frequency_encoding"
    assert normalize_operation_name("apply_log1p") == "log1p"
    assert normalize_operation_name("run_dance") is None
    assert normalize_operation_name(42) is None


def test_validate_operations_normalizes_aliases(column_profiles):
    accepted, rejected = validate_operations(
        [{"column": "income", "operation": "log"}], column_profiles, TARGET)
    assert rejected == []
    assert accepted[0]["operation"] == "log1p"
    assert accepted[0]["features"] == ["income_log1p"]


# ─── 2. rejection rules ─────────────────────────────────────────────────
def test_target_unknown_and_mismatched_operations_are_rejected(column_profiles):
    proposals = [
        {"column": TARGET, "operation": "zscore"},
        {"column": "no_such_column", "operation": "zscore"},
        {"column": "city", "operation": "sqrt"},
        {"column": "age", "operation": "teleport"},
        {"column": "constant", "operation": "square"},
        {"column": "income", "operation": "frequency_encoding"},
        {"column": "income", "operation": "log1p"},
    ]
    accepted, rejected = validate_operations(proposals, column_profiles, TARGET)
    assert [(o["column"], o["operation"]) for o in accepted] == [("income", "log1p")]
    reasons = " | ".join(r["reason"] for r in rejected)
    assert "target column" in reasons
    assert "does not exist" in reasons
    assert "not allowed" in reasons
    assert "not in the allowed operation registry" in reasons
    assert "supports no operations" in reasons


def test_pair_operation_cannot_reference_target(column_profiles):
    accepted, rejected = validate_operations(
        [{"column": "income", "operation": "interaction",
          "params": {"with_column": TARGET}}],
        column_profiles, TARGET)
    assert accepted == []
    assert "target" in rejected[0]["reason"]


def test_duplicate_after_alias_normalization_is_rejected(column_profiles):
    accepted, rejected = validate_operations(
        [{"column": "income", "operation": "log"},
         {"column": "income", "operation": "log1p"}],
        column_profiles, TARGET)
    assert len(accepted) == 1
    assert any("duplicate" in r["reason"] for r in rejected)


def test_operation_budget_caps_accepted(column_profiles):
    proposals = [{"column": "income", "operation": op}
                 for op in ("log1p", "sqrt", "square", "absolute", "zscore")]
    accepted, rejected = validate_operations(
        proposals, column_profiles, TARGET, max_operations=2)
    assert len(accepted) == 2
    assert any("budget" in r["reason"] for r in rejected)
    assert MAX_OPERATIONS >= 2


def test_param_defaults_applied_when_params_omitted(column_profiles):
    # Regression: the defaults path used to crash instead of applying defaults.
    accepted, rejected = validate_operations(
        [{"column": "city", "operation": "rare_category_grouping"}],
        column_profiles, TARGET)
    assert rejected == []
    assert accepted[0]["params"]["threshold"] == pytest.approx(0.02)
    assert accepted[0]["params"]["group_label"] == "Other"


def test_pair_operation_requires_with_column(column_profiles):
    accepted, rejected = validate_operations(
        [{"column": "income", "operation": "interaction"}],
        column_profiles, TARGET)
    assert accepted == []
    assert "with_column" in rejected[0]["reason"]


# ─── 3. structural schema validation ────────────────────────────────────
def test_missing_required_fields_raise_invalid_analysis_response():
    provider = FallbackLLMProvider()
    with pytest.raises(InvalidAnalysisResponse) as exc:
        provider.validate_analysis({"problem_type": "classification"})
    assert "reasoning" in str(exc.value)
    assert "target_column" in str(exc.value)


def test_wrong_container_types_fail_strictly():
    provider = FallbackLLMProvider()
    base = {"problem_type": "classification", "target_column": TARGET, "reasoning": "r"}
    with pytest.raises(InvalidAnalysisResponse):
        provider.validate_analysis({**base, "suggested_operations": "log1p"})
    with pytest.raises(InvalidAnalysisResponse):
        provider.validate_analysis({**base, "problem_understanding": ["not a string"]})


def test_non_object_response_rejected():
    with pytest.raises(InvalidAnalysisResponse):
        FallbackLLMProvider().validate_analysis(["not", "an", "object"])


def test_quirky_list_scalars_are_coerced_not_fatal():
    result = FallbackLLMProvider().validate_analysis({
        "problem_type": "classification",
        "target_column": TARGET,
        "reasoning": "r",
        "warnings": "one warning as a string",
        "candidate_models": ["random_forest_clf", 7, None],
        "confidence": "not-a-number",
    })
    assert result.warnings[0] == "one warning as a string"
    # the int was coerced to str, then dropped by model-name validation
    assert result.candidate_models == ["random_forest_clf"]
    assert result.confidence is None
    assert any("confidence" in w for w in result.warnings)


# ─── 4. context-aware analysis validation ───────────────────────────────
def test_validate_analysis_splits_accepted_and_rejected(df, column_profiles):
    result = FallbackLLMProvider().validate_analysis({
        "problem_type": "classification",
        "target_column": TARGET,
        "reasoning": "because",
        "suggested_operations": [
            {"column": "income", "operation": "log"},
            {"column": TARGET, "operation": "zscore"},
            {"column": "city", "operation": "teleport"},
        ],
        "useful_feature_candidates": ["income", "ghost_column"],
    }, context(df, column_profiles))
    assert [(o["column"], o["operation"])
            for o in result.suggested_operations] == [("income", "log1p")]
    assert len(result.rejected_operations) == 2
    assert result.operation_source == "provider"
    assert result.useful_feature_candidates == ["income"]
    assert any(fe.get("operation") == "log1p" for fe in result.feature_engineering)
    assert any("refused by the registry" in w for w in result.warnings)


def test_without_operations_source_is_deterministic_defaults(df, column_profiles):
    result = FallbackLLMProvider().validate_analysis(
        {"problem_type": "regression", "target_column": TARGET, "reasoning": "r"},
        context(df, column_profiles))
    assert result.operation_source == "deterministic_defaults"


def test_configured_target_wins_over_provider_claim(df, column_profiles):
    result = FallbackLLMProvider().validate_analysis(
        {"problem_type": "classification", "target_column": "income",
         "reasoning": "r"},
        context(df, column_profiles))
    assert result.target_column == TARGET
    assert any("using configured target" in w for w in result.warnings)


# ─── 5. compact dataset description ─────────────────────────────────────
def test_description_summary_and_catalog_exclude_target(df, column_profiles):
    desc = build_dataset_description(
        df, profile={"column_profiles": column_profiles},
        target_column=TARGET, problem_type="classification")
    assert isinstance(desc["target"], dict)
    assert desc["target"]["column"] == TARGET
    assert TARGET not in desc["feature_operation_catalog"]
    assert "log1p" in desc["feature_operation_catalog"]["income"]
    rendered = render_dataset_description(desc)
    assert json.loads(rendered)["target"]["column"] == TARGET


def test_description_shrinks_to_char_budget(df, column_profiles):
    full = build_dataset_description(
        df, profile={"column_profiles": column_profiles},
        target_column=TARGET, problem_type="classification", max_chars=100_000)
    full_len = len(render_dataset_description(full))
    budget = max(full_len - 400, 2800)
    small = build_dataset_description(
        df, profile={"column_profiles": column_profiles},
        target_column=TARGET, problem_type="classification", max_chars=budget)
    assert len(render_dataset_description(small)) <= budget
    assert small["truncation"]["characters"] <= budget
    truncation = small["truncation"]
    assert (truncation["stats_dropped"] or truncation["sample_values_dropped"]
            or truncation["columns_omitted"] > 0)


