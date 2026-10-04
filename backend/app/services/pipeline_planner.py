"""Structured AutoML Pipeline Plan schema, validator, and deterministic generator.

Research Boundary:
This module implements a structured decision layer for LLM-guided AutoML pipeline
planning. The LLM is NEVER permitted to generate or execute arbitrary Python code.
All model selections are strictly constrained by ModelRegistry, all feature operations
by OPERATION_REGISTRY, and all metrics by the Evaluator's MetricSpec contracts.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.services.model_registry import ModelRegistry, validate_model_recommendations
from app.services.feature_operations import validate_operations, OPERATION_REGISTRY
from app.services.evaluator import (
    get_metric_spec,
    SUPPORTED_CLASSIFICATION_METRICS,
    SUPPORTED_REGRESSION_METRICS,
)

logger = logging.getLogger(__name__)

# Registry singletons
_REGISTRY = ModelRegistry()
CLASSIFICATION_MODELS = _REGISTRY.get_all_model_names("classification")
REGRESSION_MODELS = _REGISTRY.get_all_model_names("regression")
ALL_SUPPORTED_MODELS = set(CLASSIFICATION_MODELS + REGRESSION_MODELS)

SUPPORTED_ENSEMBLE_STRATEGIES_CLF = ["weighted_average", "soft_voting", "stacking"]
SUPPORTED_ENSEMBLE_STRATEGIES_REG = ["weighted_average", "uniform_average", "stacking"]


class PreprocessingStrategy(BaseModel):
    """Controlled preprocessing configuration."""
    model_config = ConfigDict(extra="ignore")

    numeric_imputation: str = Field(default="median", description="median or mean")
    categorical_imputation: str = Field(default="most_frequent", description="most_frequent")
    scaling: str = Field(default="standard", description="standard or none")
    max_categories: int = Field(default=20, ge=2, le=50, description="Max categories for one-hot encoding")


class HyperparameterStrategy(BaseModel):
    """Controlled HPO search configuration."""
    model_config = ConfigDict(extra="ignore")

    n_trials: int = Field(default=20, ge=1, le=100)
    n_folds: int = Field(default=5, ge=2, le=10)
    mode: str = Field(default="automl", description="automl or fast")
    sampler_seed: Optional[int] = None


class EnsembleStrategy(BaseModel):
    """Controlled model fusion / ensemble configuration."""
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    strategies: List[str] = Field(default_factory=lambda: ["weighted_average", "soft_voting", "stacking"])
    meta_learner: Optional[str] = None


class AutoMLPipelinePlan(BaseModel):
    """Structured, fully validated specification for an AutoML pipeline execution."""
    model_config = ConfigDict(extra="ignore")

    problem_type: str = Field(..., description="classification or regression")
    target_column: str
    primary_metric: str = "f1_weighted"
    preprocessing_strategy: PreprocessingStrategy = Field(default_factory=PreprocessingStrategy)
    feature_operations: List[Dict[str, Any]] = Field(default_factory=list)
    candidate_models: List[str] = Field(default_factory=list)
    hyperparameter_strategy: HyperparameterStrategy = Field(default_factory=HyperparameterStrategy)
    ensemble_strategy: EnsembleStrategy = Field(default_factory=EnsembleStrategy)
    rationale: str = ""
    constraints: Dict[str, Any] = Field(default_factory=dict)

    # Provenance and audit tracking
    source: str = "llm"  # "llm" | "deterministic_fallback" | "repaired"
    provider_used: str = ""
    model_used: str = ""
    validation_status: str = "valid"  # "valid" | "repaired" | "fallback"
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    rejected_models: List[Dict[str, Any]] = Field(default_factory=list)
    rejected_operations: List[Dict[str, Any]] = Field(default_factory=list)
    fallback_reason: Optional[str] = None

    def to_execution_dict(self) -> Dict[str, Any]:
        """Convert plan to serializable dictionary suitable for storage and APIs."""
        return self.model_dump()


def _extract_profiles_from_context(dataset_context: dict) -> list[dict]:
    """Retrieve or synthesize column profiles for validation."""
    profiles = dataset_context.get("column_profiles")
    if isinstance(profiles, list) and profiles:
        return [p for p in profiles if isinstance(p, dict)]
    synthesized = []
    for name, dtype in (dataset_context.get("dtypes") or {}).items():
        dtype_str = str(dtype).lower()
        if any(token in dtype_str for token in ("int", "float", "double", "decimal", "number")):
            inferred = "numerical"
        elif any(token in dtype_str for token in ("object", "string", "category", "bool")):
            inferred = "categorical"
        else:
            inferred = "unknown"
        synthesized.append({"name": str(name), "inferred_type": inferred})
    return synthesized


def generate_deterministic_plan(dataset_context: dict) -> AutoMLPipelinePlan:
    """Generate a fully valid, reproducible pipeline plan using rule-based heuristics.

    Guaranteed to execute without any external LLM dependencies or API calls.
    """
    shape = dataset_context.get("shape", [0, 0])
    columns = dataset_context.get("columns", [])
    dtypes = dataset_context.get("dtypes", {})
    missing_pct = dataset_context.get("missing_percentages", {})
    unique_counts = dataset_context.get("unique_counts", {})
    target_col = dataset_context.get("target_column") or (columns[-1] if columns else "target")

    # Determine problem type
    target_dtype = str(dtypes.get(target_col, "object"))
    target_unique = unique_counts.get(target_col, 999)

    configured_problem = dataset_context.get("problem_type")
    if configured_problem in ("classification", "regression"):
        problem_type = configured_problem
    elif "float" in target_dtype and target_unique > 20:
        problem_type = "regression"
    elif "int" in target_dtype and target_unique > 20:
        problem_type = "regression"
    else:
        problem_type = "classification"

    n_rows = shape[0] if shape else 0

    # Default candidate models based on problem type and dataset scale
    if problem_type == "classification":
        if n_rows < 1000:
            candidate_models = ["logistic_regression", "random_forest_clf", "gradient_boosting_clf", "knn_clf"]
        else:
            candidate_models = ["logistic_regression", "random_forest_clf", "gradient_boosting_clf", "hist_gradient_boosting_clf"]
        primary_metric = "f1_weighted"
        ensemble_strategies = ["weighted_average", "soft_voting", "stacking"]
    else:
        if n_rows < 1000:
            candidate_models = ["ridge", "random_forest_reg", "gradient_boosting_reg", "knn_reg"]
        else:
            candidate_models = ["ridge", "random_forest_reg", "gradient_boosting_reg", "hist_gradient_boosting_reg"]
        primary_metric = "neg_root_mean_squared_error"
        ensemble_strategies = ["weighted_average", "uniform_average", "stacking"]

    # Preprocessing strategy
    preprocessing_strategy = PreprocessingStrategy(
        numeric_imputation="median",
        categorical_imputation="most_frequent",
        scaling="standard",
        max_categories=20,
    )

    feature_operations: List[Dict[str, Any]] = []

    # Fast demo / budget constraints
    n_folds = 3 if n_rows > 10000 else 5
    n_trials = 10 if n_rows > 10000 else 20
    hpo_strategy = HyperparameterStrategy(n_trials=n_trials, n_folds=n_folds, mode="automl")
    ensemble_strategy = EnsembleStrategy(enabled=True, strategies=ensemble_strategies)

    rationale = (
        f"Deterministic rule-based AutoML pipeline plan for {problem_type} task on '{target_col}' "
        f"({n_rows} rows, {len(columns)} cols). Standardized preprocessing, Optuna tuning ({n_trials} trials, "
        f"{n_folds} folds), and model fusion ensemble."
    )

    return AutoMLPipelinePlan(
        problem_type=problem_type,
        target_column=target_col,
        primary_metric=primary_metric,
        preprocessing_strategy=preprocessing_strategy,
        feature_operations=feature_operations,
        candidate_models=candidate_models,
        hyperparameter_strategy=hpo_strategy,
        ensemble_strategy=ensemble_strategy,
        rationale=rationale,
        source="deterministic_fallback",
        provider_used="Fallback (Deterministic)",
        model_used="rule_based_engine",
        validation_status="valid",
        validation_errors=[],
        validation_warnings=["Generated by deterministic rule-based planner (no LLM key required)."],
    )


class PipelinePlanValidator:
    """Validates and constrains candidate AutoML pipeline plans against system registries.

    Guarantees:
    1. Zero arbitrary code execution: schema-only validation.
    2. Registry adherence: all models validated against ModelRegistry; operations against OPERATION_REGISTRY.
    3. Target leakage prevention: target column can never be transformed.
    4. Task compatibility: metrics and models must match problem_type.
    5. Graceful fallback: malformed inputs are safely repaired or substituted with deterministic defaults.
    """

    @classmethod
    def validate_and_sanitize(
        cls,
        raw_plan: Union[dict, AutoMLPipelinePlan],
        dataset_context: dict,
        provider_used: str = "llm",
        model_used: str = "unknown",
    ) -> AutoMLPipelinePlan:
        """Validate, sanitize, or safely repair a pipeline plan.

        Never crashes: returns a valid AutoMLPipelinePlan under all failure modes.
        """
        if isinstance(raw_plan, AutoMLPipelinePlan):
            plan_dict = raw_plan.model_dump()
        elif isinstance(raw_plan, dict):
            plan_dict = copy.deepcopy(raw_plan)
        else:
            logger.warning(f"Malformed plan input of type {type(raw_plan).__name__}; falling back to deterministic plan.")
            fallback = generate_deterministic_plan(dataset_context)
            fallback.source = "deterministic_fallback"
            fallback.validation_status = "fallback"
            fallback.fallback_reason = f"Plan input was not a dict or AutoMLPipelinePlan (got {type(raw_plan).__name__})"
            return fallback

        warnings: List[str] = []
        errors: List[str] = []
        repaired = False

        columns = [str(c) for c in dataset_context.get("columns", [])]
        column_profiles = _extract_profiles_from_context(dataset_context)
        known_col_names = {p.get("name") for p in column_profiles if isinstance(p.get("name"), str)}
        if not known_col_names:
            known_col_names = set(columns)

        # 1. Target Column Validation
        target = dataset_context.get("target_column")
        plan_target = plan_dict.get("target_column")
        if target:
            if plan_target != target:
                warnings.append(f"Plan target '{plan_target}' mismatched configured target '{target}'; coerced to '{target}'.")
                plan_dict["target_column"] = target
                repaired = True
        elif plan_target and plan_target in known_col_names:
            plan_dict["target_column"] = plan_target
        elif columns:
            plan_dict["target_column"] = columns[-1]
            warnings.append(f"Target column missing; defaulted to last column '{columns[-1]}'.")
            repaired = True
        else:
            plan_dict["target_column"] = "target"

        # 2. Problem Type Validation
        context_problem = dataset_context.get("problem_type")
        plan_problem = str(plan_dict.get("problem_type", "")).lower()
        if plan_problem in ("classification", "regression"):
            if context_problem in ("classification", "regression") and plan_problem != context_problem:
                warnings.append(f"Plan problem_type '{plan_problem}' conflicted with configured '{context_problem}'; using configured.")
                plan_dict["problem_type"] = context_problem
                repaired = True
            else:
                plan_dict["problem_type"] = plan_problem
        else:
            inferred = context_problem if context_problem in ("classification", "regression") else "classification"
            plan_dict["problem_type"] = inferred
            warnings.append(f"Invalid problem_type '{plan_problem}'; defaulted to '{inferred}'.")
            repaired = True

        problem_type = plan_dict["problem_type"]

        # 3. Metric Validation
        metric = str(plan_dict.get("primary_metric", "")).lower()
        valid_clf_metrics = set(SUPPORTED_CLASSIFICATION_METRICS)
        valid_reg_metrics = set(SUPPORTED_REGRESSION_METRICS)

        metric_aliases = {
            "rmse": "neg_root_mean_squared_error",
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "f1": "f1_weighted",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
        }
        if metric in metric_aliases:
            metric = metric_aliases[metric]

        if problem_type == "classification":
            if metric not in valid_clf_metrics:
                warnings.append(f"Metric '{metric}' invalid for classification; defaulted to 'f1_weighted'.")
                plan_dict["primary_metric"] = "f1_weighted"
                repaired = True
            else:
                plan_dict["primary_metric"] = metric
        else:
            if metric not in valid_reg_metrics:
                warnings.append(f"Metric '{metric}' invalid for regression; defaulted to 'neg_root_mean_squared_error'.")
                plan_dict["primary_metric"] = "neg_root_mean_squared_error"
                repaired = True
            else:
                plan_dict["primary_metric"] = metric

        # 4. Candidate Models Validation
        raw_models = plan_dict.get("candidate_models", [])
        if not isinstance(raw_models, list):
            raw_models = [raw_models] if raw_models else []

        accepted_recs, rejected_recs = validate_model_recommendations(
            [{"model_id": m} if isinstance(m, str) else m for m in raw_models],
            problem_type=problem_type,
        )

        valid_model_names = [r["model_id"] for r in accepted_recs]
        if not valid_model_names:
            default_models = (
                ["logistic_regression", "random_forest_clf", "gradient_boosting_clf"]
                if problem_type == "classification"
                else ["ridge", "random_forest_reg", "gradient_boosting_reg"]
            )
            warnings.append(f"No valid models remained after registry validation; defaulted to {default_models}.")
            valid_model_names = default_models
            repaired = True

        plan_dict["candidate_models"] = valid_model_names

        # 5. Feature Operations Validation (Controlled Registry & Leakage Safe)
        raw_ops = plan_dict.get("feature_operations", [])
        if not isinstance(raw_ops, list):
            raw_ops = []

        accepted_ops, rejected_ops = validate_operations(
            raw_ops,
            column_profiles=column_profiles,
            target_column=plan_dict["target_column"],
        )
        plan_dict["feature_operations"] = accepted_ops

        # 6. Preprocessing Strategy Validation
        raw_pre = plan_dict.get("preprocessing_strategy", {})
        if not isinstance(raw_pre, dict):
            raw_pre = {}

        num_imp = raw_pre.get("numeric_imputation", "median")
        if num_imp not in ("median", "mean"):
            num_imp = "median"
            repaired = True

        cat_imp = raw_pre.get("categorical_imputation", "most_frequent")
        if cat_imp != "most_frequent":
            cat_imp = "most_frequent"
            repaired = True

        scaling = raw_pre.get("scaling", "standard")
        if scaling not in ("standard", "none"):
            scaling = "standard"
            repaired = True

        try:
            max_cat = int(raw_pre.get("max_categories", 20))
            max_cat = max(2, min(50, max_cat))
        except (ValueError, TypeError):
            max_cat = 20
            repaired = True

        plan_dict["preprocessing_strategy"] = {
            "numeric_imputation": num_imp,
            "categorical_imputation": cat_imp,
            "scaling": scaling,
            "max_categories": max_cat,
        }

        # 7. Hyperparameter Strategy Validation
        raw_hpo = plan_dict.get("hyperparameter_strategy", {})
        if not isinstance(raw_hpo, dict):
            raw_hpo = {}

        try:
            n_trials = int(raw_hpo.get("n_trials", 20))
            n_trials = max(1, min(100, n_trials))
        except (ValueError, TypeError):
            n_trials = 20
            repaired = True

        try:
            n_folds = int(raw_hpo.get("n_folds", 5))
            n_folds = max(2, min(10, n_folds))
        except (ValueError, TypeError):
            n_folds = 5
            repaired = True

        mode = str(raw_hpo.get("mode", "automl")).lower()
        if mode not in ("automl", "fast"):
            mode = "automl"
            repaired = True

        plan_dict["hyperparameter_strategy"] = {
            "n_trials": n_trials,
            "n_folds": n_folds,
            "mode": mode,
            "sampler_seed": raw_hpo.get("sampler_seed"),
        }

        # 8. Ensemble Strategy Validation
        raw_ens = plan_dict.get("ensemble_strategy", {})
        if not isinstance(raw_ens, dict):
            raw_ens = {}

        ens_enabled = bool(raw_ens.get("enabled", True))
        allowed_strategies = (
            SUPPORTED_ENSEMBLE_STRATEGIES_CLF if problem_type == "classification"
            else SUPPORTED_ENSEMBLE_STRATEGIES_REG
        )
        raw_strategies = raw_ens.get("strategies", allowed_strategies)
        if not isinstance(raw_strategies, list):
            raw_strategies = allowed_strategies

        valid_strategies = [s for s in raw_strategies if s in allowed_strategies]
        if not valid_strategies and ens_enabled:
            valid_strategies = [allowed_strategies[0]]
            warnings.append(f"No valid ensemble strategies provided; defaulted to {valid_strategies}.")
            repaired = True

        plan_dict["ensemble_strategy"] = {
            "enabled": ens_enabled,
            "strategies": valid_strategies,
            "meta_learner": raw_ens.get("meta_learner"),
        }

        # 9. Assembly & Provenance
        status = "repaired" if repaired else "valid"
        source = "repaired" if repaired else ("deterministic_fallback" if "fallback" in provider_used.lower() else "llm")

        try:
            plan = AutoMLPipelinePlan(
                problem_type=plan_dict["problem_type"],
                target_column=plan_dict["target_column"],
                primary_metric=plan_dict["primary_metric"],
                preprocessing_strategy=PreprocessingStrategy(**plan_dict["preprocessing_strategy"]),
                feature_operations=plan_dict["feature_operations"],
                candidate_models=plan_dict["candidate_models"],
                hyperparameter_strategy=HyperparameterStrategy(**plan_dict["hyperparameter_strategy"]),
                ensemble_strategy=EnsembleStrategy(**plan_dict["ensemble_strategy"]),
                rationale=str(plan_dict.get("rationale", "")),
                constraints=plan_dict.get("constraints", {}),
                source=source,
                provider_used=provider_used,
                model_used=model_used,
                validation_status=status,
                validation_errors=errors,
                validation_warnings=warnings,
                rejected_models=rejected_recs,
                rejected_operations=rejected_ops,
                fallback_reason=None,
            )
            return plan
        except ValidationError as exc:
            logger.warning(f"Plan validation error: {exc}; falling back to deterministic plan.")
            fallback = generate_deterministic_plan(dataset_context)
            fallback.validation_status = "fallback"
            fallback.source = "deterministic_fallback"
            fallback.fallback_reason = f"Pydantic validation failed: {exc}"
            return fallback
