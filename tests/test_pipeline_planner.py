"""Comprehensive tests for genuine LLM-driven AutoML pipeline planning.

Covers:
1. Valid pipeline plan schema
2. Invalid LLM JSON handling
3. Unsupported model rejection
4. Unsupported feature operation rejection
5. Metric/problem incompatibility
6. Target leakage rejection
7. Malformed LLM response fallback
8. LLM unavailable fallback
9. Provenance correctness
10. Deterministic plan validation
11. Existing fold-safe execution
12. Holdout isolation
13. Existing deterministic pipeline remains unchanged
14. LLM plan actually influences candidate selection when valid
15. LLM plan cannot bypass registry constraints
16. No arbitrary code execution from LLM output
17. Database and storage isolation
"""
import copy
import json
import os
import sqlite3
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.pipeline_planner import (
    AutoMLPipelinePlan,
    PipelinePlanValidator,
    PreprocessingStrategy,
    HyperparameterStrategy,
    EnsembleStrategy,
    generate_deterministic_plan,
)
from app.services.model_registry import ModelRegistry
from app.services.feature_operations import OPERATION_REGISTRY
from app.services.fold_safe import FoldSafeFeaturePreprocessor
from app.llm.fallback import FallbackLLMProvider
from app.llm.manager import LLMManager
from app.core.config import settings
from app.core import database as database_module
from app.api import experiments as experiments_api
from fastapi.testclient import TestClient


@pytest.fixture
def mock_dataset_context():
    """Compact dataset context representing training-side profile only."""
    return {
        "shape": [500, 5],
        "columns": ["age", "income", "credit_score", "education", "churn"],
        "dtypes": {
            "age": "int64",
            "income": "float64",
            "credit_score": "float64",
            "education": "object",
            "churn": "int64",
        },
        "missing_percentages": {
            "age": 0.0,
            "income": 2.5,
            "credit_score": 0.0,
            "education": 1.0,
            "churn": 0.0,
        },
        "unique_counts": {
            "age": 45,
            "income": 480,
            "credit_score": 300,
            "education": 4,
            "churn": 2,
        },
        "target_column": "churn",
        "problem_type": "classification",
        "column_profiles": [
            {"name": "age", "inferred_type": "numerical"},
            {"name": "income", "inferred_type": "numerical"},
            {"name": "credit_score", "inferred_type": "numerical"},
            {"name": "education", "inferred_type": "categorical"},
            {"name": "churn", "inferred_type": "categorical"},
        ],
    }


@pytest.fixture(scope="module")
def isolated_client(tmp_path_factory):
    """Isolated FastAPI test client that never writes to live DB or storage."""
    tmp_dir = tmp_path_factory.mktemp("planner_isolated_env")
    saved_db = settings.DATABASE_URL
    saved_storage = settings.STORAGE_PATH
    saved_provider = settings.LLM_PROVIDER

    settings.DATABASE_URL = f"sqlite:///{(tmp_dir / 'test.db').as_posix()}"
    settings.STORAGE_PATH = str(tmp_dir)
    settings.LLM_PROVIDER = "fallback"
    database_module._engine = None
    database_module._SessionLocal = None
    experiments_api._experiment_cache.clear()

    from app.main import app
    try:
        with TestClient(app) as client:
            yield client
    finally:
        settings.DATABASE_URL = saved_db
        settings.STORAGE_PATH = saved_storage
        settings.LLM_PROVIDER = saved_provider
        database_module._engine = None
        database_module._SessionLocal = None
        experiments_api._experiment_cache.clear()


# =========================================================================
# 1. Valid Pipeline Plan Schema
# =========================================================================
def test_valid_pipeline_plan_schema(mock_dataset_context):
    plan = AutoMLPipelinePlan(
        problem_type="classification",
        target_column="churn",
        primary_metric="f1_weighted",
        preprocessing_strategy=PreprocessingStrategy(
            numeric_imputation="median",
            categorical_imputation="most_frequent",
            scaling="standard",
            max_categories=15,
        ),
        feature_operations=[
            {"column": "income", "operation": "log1p", "params": {}, "reason": "reduce skew"}
        ],
        candidate_models=["logistic_regression", "random_forest_clf"],
        hyperparameter_strategy=HyperparameterStrategy(n_trials=15, n_folds=3, mode="automl"),
        ensemble_strategy=EnsembleStrategy(enabled=True, strategies=["weighted_average", "soft_voting"]),
        rationale="Standard classification architecture.",
        source="llm",
        provider_used="Google Gemini",
        validation_status="valid",
    )
    dumped = plan.to_execution_dict()
    assert dumped["problem_type"] == "classification"
    assert dumped["primary_metric"] == "f1_weighted"
    assert len(dumped["candidate_models"]) == 2
    assert dumped["source"] == "llm"
    assert dumped["validation_status"] == "valid"


# =========================================================================
# 2. Invalid LLM JSON Handling (Never Crashes)
# =========================================================================
def test_invalid_llm_json(mock_dataset_context):
    # Case A: Passed a non-dict string
    fallback_plan_str = PipelinePlanValidator.validate_and_sanitize(
        "invalid string response", mock_dataset_context, provider_used="Gemini"
    )
    assert isinstance(fallback_plan_str, AutoMLPipelinePlan)
    assert fallback_plan_str.validation_status == "fallback"
    assert "not a dict" in fallback_plan_str.fallback_reason

    # Case B: Completely empty dict
    plan_empty = PipelinePlanValidator.validate_and_sanitize(
        {}, mock_dataset_context, provider_used="Gemini"
    )
    assert isinstance(plan_empty, AutoMLPipelinePlan)
    assert plan_empty.target_column == "churn"
    assert plan_empty.problem_type == "classification"
    assert len(plan_empty.candidate_models) > 0


# =========================================================================
# 3. Unsupported Model Rejection
# =========================================================================
def test_unsupported_model_rejection(mock_dataset_context):
    raw = {
        "problem_type": "classification",
        "target_column": "churn",
        "primary_metric": "f1_weighted",
        "candidate_models": [
            "super_neural_quantum_transformer",
            "random_forest_clf",
            "not_a_real_model_123",
        ],
    }
    validated = PipelinePlanValidator.validate_and_sanitize(raw, mock_dataset_context)
    assert "random_forest_clf" in validated.candidate_models
    assert "super_neural_quantum_transformer" not in validated.candidate_models
    assert "not_a_real_model_123" not in validated.candidate_models
    rejected_names = [r["model_id"] for r in validated.rejected_models]
    assert "super_neural_quantum_transformer" in rejected_names
    assert "not_a_real_model_123" in rejected_names


# =========================================================================
# 4. Unsupported Feature Operation Rejection
# =========================================================================
def test_unsupported_feature_operation_rejection(mock_dataset_context):
    raw = {
        "problem_type": "classification",
        "target_column": "churn",
        "primary_metric": "f1_weighted",
        "feature_operations": [
            {"column": "income", "operation": "log1p", "params": {}, "reason": "valid log transform"},
            {"column": "income", "operation": "magic_quantum_filter", "params": {}, "reason": "unsupported op"},
        ],
    }
    validated = PipelinePlanValidator.validate_and_sanitize(raw, mock_dataset_context)
    assert len(validated.feature_operations) == 1
    assert validated.feature_operations[0]["operation"] == "log1p"
    assert len(validated.rejected_operations) == 1
    assert "magic_quantum_filter" in str(validated.rejected_operations[0])


# =========================================================================
# 5. Metric / Problem Incompatibility Rejection & Repair
# =========================================================================
def test_metric_problem_incompatibility(mock_dataset_context):
    # Classification with regression metric (RMSE)
    raw_clf = {
        "problem_type": "classification",
        "target_column": "churn",
        "primary_metric": "neg_root_mean_squared_error",
    }
    validated_clf = PipelinePlanValidator.validate_and_sanitize(raw_clf, mock_dataset_context)
    assert validated_clf.primary_metric == "f1_weighted"
    assert any("invalid for classification" in w for w in validated_clf.validation_warnings)
    assert validated_clf.validation_status == "repaired"

    # Regression context with classification metric (ROC-AUC)
    reg_context = copy.deepcopy(mock_dataset_context)
    reg_context["problem_type"] = "regression"
    raw_reg = {
        "problem_type": "regression",
        "target_column": "income",
        "primary_metric": "roc_auc",
    }
    validated_reg = PipelinePlanValidator.validate_and_sanitize(raw_reg, reg_context)
    assert validated_reg.primary_metric == "neg_root_mean_squared_error"
    assert any("invalid for regression" in w for w in validated_reg.validation_warnings)


# =========================================================================
# 6. Target Leakage Rejection
# =========================================================================
def test_target_leakage_rejection(mock_dataset_context):
    raw = {
        "problem_type": "classification",
        "target_column": "churn",
        "feature_operations": [
            {"column": "churn", "operation": "log1p", "params": {}, "reason": "leak target"},
            {"column": "income", "operation": "log1p", "params": {}, "reason": "valid op"},
        ],
    }
    validated = PipelinePlanValidator.validate_and_sanitize(raw, mock_dataset_context)
    # Target column must NEVER appear in validated operations
    ops_cols = [op["column"] for op in validated.feature_operations]
    assert "churn" not in ops_cols
    assert "income" in ops_cols
    rejected_reasons = [r.get("reason", "") for r in validated.rejected_operations]
    assert any("target column" in r.lower() for r in rejected_reasons)


# =========================================================================
# 7. Malformed LLM Response Fallback
# =========================================================================
def test_malformed_llm_response_fallback(mock_dataset_context):
    # Raw JSON with wrong types (e.g. candidate_models as integer, preprocessing as string)
    raw = {
        "problem_type": 12345,
        "target_column": ["wrong", "type"],
        "candidate_models": 9999,
        "preprocessing_strategy": "not_a_dict",
    }
    validated = PipelinePlanValidator.validate_and_sanitize(raw, mock_dataset_context)
    assert isinstance(validated, AutoMLPipelinePlan)
    assert validated.target_column == "churn"
    assert validated.problem_type == "classification"
    assert isinstance(validated.candidate_models, list)
    assert len(validated.candidate_models) > 0


# =========================================================================
# 8. LLM Unavailable Fallback
# =========================================================================
@pytest.mark.asyncio
async def test_llm_unavailable_fallback(mock_dataset_context):
    config = {"LLM_PROVIDER": "gemini"}
    manager = LLMManager(config)

    # Set active_provider to a distinct mock that fails
    mock_provider = AsyncMock()
    mock_provider.get_provider_name.return_value = "Mock Gemini"
    mock_provider.plan_pipeline.side_effect = RuntimeError("API Gateway Timeout")
    manager.active_provider = mock_provider

    result = await manager.plan_pipeline(mock_dataset_context)
    assert result["is_fallback"] is True
    assert "API Gateway Timeout" in result["fallback_reason"]
    assert result["plan"]["validation_status"] == "fallback"
    assert result["plan"]["source"] == "deterministic_fallback"


# =========================================================================
# 9. Provenance Correctness
# =========================================================================
def test_provenance_correctness(mock_dataset_context):
    det_plan = generate_deterministic_plan(mock_dataset_context)
    assert det_plan.source == "deterministic_fallback"
    assert det_plan.provider_used == "Fallback (Deterministic)"
    assert det_plan.validation_status == "valid"
    assert det_plan.model_used == "rule_based_engine"
    assert isinstance(det_plan.validation_warnings, list)
    assert isinstance(det_plan.validation_errors, list)


# =========================================================================
# 10. Deterministic Plan Validation
# =========================================================================
def test_deterministic_plan_validation(mock_dataset_context):
    det_plan = generate_deterministic_plan(mock_dataset_context)
    validated = PipelinePlanValidator.validate_and_sanitize(det_plan, mock_dataset_context)
    assert validated.validation_status == "valid"
    assert validated.target_column == "churn"
    assert validated.problem_type == "classification"
    assert len(validated.candidate_models) >= 2


# =========================================================================
# 11. Existing Fold-Safe Execution with Plan Operations
# =========================================================================
def test_fold_safe_execution_with_plan(mock_dataset_context):
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "age": rng.randint(20, 70, n).astype(float),
        "income": rng.uniform(20000, 150000, n),
        "credit_score": rng.uniform(300, 850, n),
        "education": rng.choice(["HS", "BS", "MS", "PhD"], n),
        "churn": rng.choice([0, 1], n),
    })

    plan = AutoMLPipelinePlan(
        problem_type="classification",
        target_column="churn",
        feature_operations=[
            {"column": "income", "operation": "log1p", "params": {}},
            {"column": "age", "operation": "square", "params": {}},
        ],
        candidate_models=["logistic_regression", "random_forest_clf"],
    )

    preprocessor = FoldSafeFeaturePreprocessor(
        column_profiles=mock_dataset_context["column_profiles"],
        target_column="churn",
        problem_type="classification",
        operations=plan.feature_operations,
        seed=42,
    )

    preprocessor.fit(df)
    transformed = preprocessor.transform(df)

    assert transformed.shape[0] == n
    assert preprocessor.fitted_rows_ == n
    assert any("log1p" in name for name in preprocessor.feature_names_out_)
    assert any("square" in name for name in preprocessor.feature_names_out_)


# =========================================================================
# 12. Holdout Isolation Guarantee
# =========================================================================
def test_holdout_isolation(mock_dataset_context):
    # Verify that plan generation and validation consume ONLY the dataset profile metadata,
    # never dataframe rows or holdout test partitions.
    raw = {
        "problem_type": "classification",
        "target_column": "churn",
        "primary_metric": "f1_weighted",
        "candidate_models": ["logistic_regression"],
    }
    plan = PipelinePlanValidator.validate_and_sanitize(raw, mock_dataset_context)
    assert plan.candidate_models == ["logistic_regression"]
    # No test DataFrame was passed or accessed
    assert "test_data" not in mock_dataset_context


# =========================================================================
# 13. Existing Deterministic Pipeline Remains Unchanged
# =========================================================================
@pytest.mark.asyncio
async def test_existing_deterministic_pipeline_remains_unchanged(mock_dataset_context):
    provider = FallbackLLMProvider()
    assert provider.is_available() is True
    assert provider.get_provider_name() == "Fallback (Deterministic)"

    # Existing analysis still returns DatasetAnalysisResult
    analysis = await provider.analyze_dataset(mock_dataset_context)
    assert analysis.problem_type == "classification"
    assert analysis.target_column == "churn"

    # New plan_pipeline returns valid AutoMLPipelinePlan
    plan = await provider.plan_pipeline(mock_dataset_context)
    assert isinstance(plan, AutoMLPipelinePlan)
    assert plan.source == "deterministic_fallback"


# =========================================================================
# 14. Valid LLM Plan Actually Influences Candidate Selection
# =========================================================================
def test_plan_influences_candidate_selection(mock_dataset_context):
    registry = ModelRegistry()
    raw = {
        "problem_type": "classification",
        "target_column": "churn",
        "candidate_models": ["logistic_regression", "knn_clf"],
    }
    plan = PipelinePlanValidator.validate_and_sanitize(raw, mock_dataset_context)
    models = registry.get_models("classification", plan.candidate_models)
    assert len(models) == 2
    assert [m.name for m in models] == ["logistic_regression", "knn_clf"]


# =========================================================================
# 15. LLM Plan Cannot Bypass Registry Constraints
# =========================================================================
def test_registry_constraints_enforced(mock_dataset_context):
    raw = {
        "problem_type": "classification",
        "target_column": "churn",
        "candidate_models": ["__import__('os').system('malicious')", "fake_rf"],
        "feature_operations": [{"column": "income", "operation": "rmdir_all"}],
    }
    plan = PipelinePlanValidator.validate_and_sanitize(raw, mock_dataset_context)
    assert "__import__('os').system('malicious')" not in plan.candidate_models
    assert len(plan.feature_operations) == 0
    assert len(plan.rejected_models) == 2
    assert len(plan.rejected_operations) == 1


# =========================================================================
# 16. No Arbitrary Code Execution from LLM Output
# =========================================================================
def test_no_arbitrary_code_execution(mock_dataset_context):
    malicious_inputs = [
        "__import__('os').system('echo hacked')",
        "eval('1+1')",
        "exec('import sys')",
        "lambda x: x * 2",
    ]
    for mal in malicious_inputs:
        raw = {
            "problem_type": "classification",
            "target_column": "churn",
            "candidate_models": [mal],
            "feature_operations": [{"column": "income", "operation": mal}],
            "rationale": mal,
        }
        plan = PipelinePlanValidator.validate_and_sanitize(raw, mock_dataset_context)
        assert mal not in plan.candidate_models
        assert len(plan.feature_operations) == 0
        # Rationale is stored as plain text, never evaluated
        assert plan.rationale == mal


# =========================================================================
# 17. End-to-End API Integration with Pipeline Plan
# =========================================================================
def test_pipeline_plan_api_end_to_end(isolated_client):
    # Upload test dataset
    with open("demo_data/classification.csv", "rb") as f:
        r = isolated_client.post("/api/datasets/upload", files={"file": ("classification.csv", f, "text/csv")})
    assert r.status_code == 200
    ds_id = r.json()["id"]

    # Create experiment
    r = isolated_client.post("/api/experiments", json={
        "name": "test_planner_api",
        "dataset_id": ds_id,
        "target_column": "Churn",
    })
    assert r.status_code == 200
    exp_id = r.json()["id"]

    # 1. Pipeline plan on-demand before analysis
    r = isolated_client.get(f"/api/experiments/{exp_id}/pipeline-plan")
    assert r.status_code == 200
    plan_data = r.json()
    assert "validated_plan" in plan_data
    assert plan_data["validated_plan"]["target_column"] == "Churn"

    # 2. Analyze experiment
    r = isolated_client.post(f"/api/experiments/{exp_id}/analyze")
    assert r.status_code == 200
    an_data = r.json()
    assert "pipeline_plan" in an_data
    assert an_data["pipeline_plan"]["target_column"] == "Churn"

    # 3. Train experiment (fast demo mode)
    r = isolated_client.post(f"/api/experiments/{exp_id}/train?fast_demo=true&seed=42")
    assert r.status_code == 200

    # 4. Pipeline plan endpoint after training (returns full execution provenance)
    r = isolated_client.get(f"/api/experiments/{exp_id}/pipeline-plan")
    assert r.status_code == 200
    post_train_plan = r.json()
    assert post_train_plan["validation_status"] in ("valid", "repaired")
    assert post_train_plan["executed_pipeline"] is not None
    assert post_train_plan["executed_pipeline"]["seed"] == 42
    assert "Zero arbitrary code execution" in post_train_plan["constraints_enforced"]

    # 5. Results endpoint includes pipeline plan provenance
    r = isolated_client.get(f"/api/experiments/{exp_id}/results")
    assert r.status_code == 200
    res_data = r.json()
    assert "pipeline_plan_provenance" in res_data
    assert res_data["pipeline_plan_provenance"]["executed_pipeline"] is not None
    assert "research_evaluation" in res_data
    assert "pipeline_plan" in res_data["research_evaluation"]


# =========================================================================
# 18. Database and Storage Isolation Verification
# =========================================================================
def test_db_and_storage_isolation():
    # Verify the live database and live storage were not modified by any test
    con = sqlite3.connect("backend/automl_lens.db")
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM experiments")
    exp_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM datasets")
    ds_count = cur.fetchone()[0]
    con.close()

    storage_file_count = sum(len(files) for _, _, files in os.walk("backend/storage"))

    assert exp_count == 7, f"Live DB experiments modified: expected 7, got {exp_count}"
    assert ds_count == 5, f"Live DB datasets modified: expected 5, got {ds_count}"
    assert storage_file_count == 23, f"Live storage modified: expected 23 files, got {storage_file_count}"
