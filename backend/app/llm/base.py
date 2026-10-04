"""Abstract base class for LLM providers, the analysis schema and validation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.services.feature_operations import validate_operations
from app.services.model_registry import ModelRegistry, validate_model_recommendations
from app.services.pipeline_planner import AutoMLPipelinePlan

# Supported model names, derived from the live registry so the validation
# whitelist can never drift from what the pipeline can actually train.
_REGISTRY = ModelRegistry()
CLASSIFICATION_MODELS = _REGISTRY.get_all_model_names("classification")
REGRESSION_MODELS = _REGISTRY.get_all_model_names("regression")
ALL_MODELS = set(CLASSIFICATION_MODELS + REGRESSION_MODELS)

CLASSIFICATION_METRICS = [
    "accuracy", "f1_weighted", "precision_weighted",
    "recall_weighted", "balanced_accuracy", "roc_auc"
]
REGRESSION_METRICS = [
    "r2", "neg_mean_squared_error", "neg_mean_absolute_error",
    "neg_root_mean_squared_error"
]

# Structural requirements: absent or blank means the answer is unusable.
REQUIRED_ANALYSIS_FIELDS = ("problem_type", "target_column", "reasoning")
# Strict when present: a wrong container type fails the whole answer instead of
# being silently coerced, so callers notice a provider that ignored the schema.
STRICT_STR_FIELDS = ("problem_understanding",)
STRICT_LIST_FIELDS = ("suggested_operations",)
# List fields tolerated as a single scalar (a common model quirk).
STR_LIST_FIELDS = (
    "candidate_models", "warnings", "useful_feature_candidates",
    "potentially_irrelevant_columns", "leakage_warnings",
    "modelling_considerations", "rejected_operations",
)
DICT_LIST_FIELDS = ("preprocessing", "feature_engineering",
                   "suggested_operations", "model_recommendations")
STRINGIFIED_DICT_LIST_FIELDS = ("preprocessing", "feature_engineering")


class InvalidAnalysisResponse(ValueError):
    """Raised when a provider answer fails structural schema validation.

    ``LLMManager`` treats this like any other provider failure and falls back to
    the deterministic provider, so a malformed answer degrades instead of
    crashing the pipeline.
    """

    def __init__(self, errors: list[str]):
        self.errors = [str(e) for e in errors]
        super().__init__("; ".join(self.errors))


def _pydantic_errors(exc: ValidationError) -> list[str]:
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        messages.append(
            f"{location}: {error.get('msg')}" if location else str(error.get("msg"))
        )
    return messages


def _normalize_raw(raw: dict) -> tuple[dict, list[str]]:
    """Coerce common provider quirks before Pydantic sees them.

    Returns ``(normalized, notes)``. Structural problems raise
    ``InvalidAnalysisResponse``; element-level noise is dropped and reported as
    notes (surfaced later as warnings) so one bad entry cannot fail the answer.
    """
    out: dict[str, Any] = {k: v for k, v in raw.items() if v is not None}
    notes: list[str] = []
    errors: list[str] = []

    for name in REQUIRED_ANALYSIS_FIELDS:
        value = out.get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{name}' must be a non-empty string")
    for name in STRICT_STR_FIELDS:
        if name in out and not isinstance(out[name], str):
            errors.append(f"'{name}' must be a string when present")
    for name in STRICT_LIST_FIELDS:
        if name in out and not isinstance(out[name], list):
            errors.append(f"'{name}' must be a list when present")
    if errors:
        raise InvalidAnalysisResponse(errors)

    for name in STR_LIST_FIELDS:
        value = out.get(name)
        if value is None:
            continue
        if isinstance(value, str):
            out[name] = [value]
        elif isinstance(value, (list, tuple)):
            kept = [v for v in value if isinstance(v, (str, int, float, bool))]
            if len(kept) != len(value):
                notes.append(f"'{name}' contained non-text entries that were dropped")
            out[name] = [str(v) for v in kept]
        else:
            out[name] = []
            notes.append(f"'{name}' was not a list and was cleared")

    # Model recommendations may arrive as one name, one object, or a list
    # mixing names and objects; normalize to a list of objects first.
    mr = out.get("model_recommendations")
    if isinstance(mr, (str, dict)):
        mr = [mr]
    if isinstance(mr, (list, tuple)):
        out["model_recommendations"] = [
            {"model_id": v} if isinstance(v, str) else v for v in mr
        ]
    elif mr is not None:
        out["model_recommendations"] = []
        notes.append("'model_recommendations' was not a list and was cleared")

    for name in DICT_LIST_FIELDS:
        value = out.get(name)
        if value is None:
            continue
        if isinstance(value, dict):
            out[name] = [value]
        elif isinstance(value, (list, tuple)):
            kept = [dict(v) for v in value if isinstance(v, dict)]
            if len(kept) != len(value):
                notes.append(f"'{name}' contained non-object entries that were dropped")
            out[name] = kept
        else:
            out[name] = []
            notes.append(f"'{name}' was not a list and was cleared")

    for name in STRINGIFIED_DICT_LIST_FIELDS:
        out[name] = [{str(k): str(v) for k, v in entry.items()}
                     for entry in out.get(name, [])]

    if out.get("confidence") is not None:
        try:
            out["confidence"] = float(out["confidence"])
        except (TypeError, ValueError):
            out["confidence"] = None
            notes.append("confidence was not a number and was dropped")

    return out, notes

def _profiles_from_context(dataset_context: dict) -> list[dict]:
    """Column profiles for operation validation.

    The pipeline passes real profiler output in ``column_profiles``; if a caller
    only has dtypes, synthesize minimal profiles so type-compatibility checks
    still work instead of being skipped.
    """
    profiles = dataset_context.get("column_profiles")
    if isinstance(profiles, list) and profiles:
        return [p for p in profiles if isinstance(p, dict)]
    synthesized = []
    for name, dtype in (dataset_context.get("dtypes") or {}).items():
        dtype = str(dtype)
        if any(token in dtype for token in ("int", "float", "double", "decimal", "number")):
            inferred = "numerical"
        elif any(token in dtype for token in ("object", "string", "category", "bool")):
            inferred = "categorical"
        else:
            inferred = "unknown"
        synthesized.append({"name": str(name), "inferred_type": inferred})
    return synthesized


class DatasetAnalysisResult(BaseModel):
    """Structured, validated result of a dataset analysis."""
    model_config = ConfigDict(extra="ignore")

    problem_type: str = Field(..., description="classification or regression")
    target_column: str
    reasoning: str
    problem_understanding: Optional[str] = None
    preprocessing: list[dict[str, str]] = []
    feature_engineering: list[dict[str, str]] = []
    candidate_models: list[str] = []
    recommended_metric: str = "f1_weighted"
    optimization_strategy: str = "optuna"
    warnings: list[str] = []
    useful_feature_candidates: list[str] = []
    potentially_irrelevant_columns: list[str] = []
    leakage_warnings: list[str] = []
    modelling_considerations: list[str] = []
    # Registry-validated operations. After validate_analysis() only entries that
    # survived validate_operations() remain here; everything else moves to
    # rejected_operations together with the reason it was refused.
    suggested_operations: list[dict[str, Any]] = []
    rejected_operations: list[dict[str, Any]] = []
    operation_source: str = "none"  # provider | deterministic_defaults | none
    # Registry-validated model recommendations. After validate_analysis() only
    # models that exist in the registry and match the problem type remain here;
    # refused entries move to rejected_model_recommendations with a reason.
    model_recommendations: list[dict[str, Any]] = []
    rejected_model_recommendations: list[dict[str, Any]] = []
    model_selection_source: str = "none"  # provider | deterministic_defaults | none
    # Provenance, filled by the provider/manager so stored analyses are auditable.
    provider_used: str = ""
    is_fallback: bool = False
    # None means "the provider did not report one". Never substitute a default.
    confidence: Optional[float] = None

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def analyze_dataset(self, dataset_context: dict) -> DatasetAnalysisResult:
        """Analyze dataset context and return structured recommendations."""
        ...

    @abstractmethod
    async def explain_results(self, results_context: dict) -> str:
        """Explain ML experiment results in natural language."""
        ...

    @abstractmethod
    async def chat(self, message: str, experiment_context: Optional[dict] = None) -> str:
        """Answer a user question about the experiment."""
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the display name of this provider."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is currently available."""
        ...

    @abstractmethod
    async def plan_pipeline(self, dataset_context: dict) -> AutoMLPipelinePlan:
        """Generate a structured, validated AutoML pipeline plan."""
        ...

    def validate_analysis(self, raw_response: dict,
                          dataset_context: Optional[dict] = None) -> DatasetAnalysisResult:
        """Validate and sanitize a raw provider answer.

        Structural problems raise ``InvalidAnalysisResponse`` so the manager can
        fall back; everything else is corrected in place and reported through
        ``warnings``. When ``dataset_context`` is given, ``suggested_operations``
        is filtered through the controlled operation registry: only
        registry-approved, leakage-safe operations survive, and the target
        column can never be engineered.
        """
        if not isinstance(raw_response, dict):
            raise InvalidAnalysisResponse(
                [f"analysis response must be a JSON object, "
                 f"got {type(raw_response).__name__}"]
            )

        normalized, notes = _normalize_raw(raw_response)
        try:
            result = DatasetAnalysisResult(**normalized)
        except ValidationError as exc:
            raise InvalidAnalysisResponse(_pydantic_errors(exc)) from exc

        # Validate problem_type
        if result.problem_type not in ("classification", "regression"):
            result.problem_type = "classification"

        # Clamp only when the provider actually reported a confidence value
        if result.confidence is not None:
            result.confidence = max(0.0, min(1.0, float(result.confidence)))

        # Validate candidate_models against the live registry for this problem
        # type: names must exist AND be task-compatible, never arbitrary.
        _task_models = set(
            CLASSIFICATION_MODELS if result.problem_type == "classification"
            else REGRESSION_MODELS
        )
        valid_models = [m for m in result.candidate_models if m in _task_models]
        if not valid_models:
            if result.problem_type == "classification":
                valid_models = ["random_forest_clf", "gradient_boosting_clf"]
            else:
                valid_models = ["random_forest_reg", "gradient_boosting_reg"]
        result.candidate_models = valid_models

        # Validate metric
        if result.problem_type == "classification":
            if result.recommended_metric not in CLASSIFICATION_METRICS:
                result.recommended_metric = "f1_weighted"
        else:
            if result.recommended_metric not in REGRESSION_METRICS:
                result.recommended_metric = "neg_root_mean_squared_error"

        if dataset_context is not None:
            self._apply_context_validation(result, dataset_context, notes)

        if notes:
            result.warnings = list(result.warnings) + [f"validation: {n}" for n in notes]
        return result

    # ────────────────────────── context-aware checks ─────────────────────
    def _apply_context_validation(self, result: DatasetAnalysisResult,
                                  dataset_context: dict, notes: list[str]) -> None:
        target = dataset_context.get("target_column")
        target = target if isinstance(target, str) and target else None
        columns = [c for c in (dataset_context.get("columns") or []) if isinstance(c, str)]
        known = set(columns)
        if not known:
            known = {p.get("name") for p in _profiles_from_context(dataset_context)
                     if isinstance(p.get("name"), str)}

        # The configured target wins: the pipeline trains on it regardless of
        # what the model believes the label is.
        if target and result.target_column != target:
            if result.target_column:
                notes.append(
                    f"provider reported target_column '{result.target_column}'; "
                    f"using configured target '{target}'"
                )
            result.target_column = target

        # Drop free-text column references that do not exist in this dataset.
        for field_name in ("useful_feature_candidates", "potentially_irrelevant_columns"):
            names = getattr(result, field_name)
            if known:
                kept = [n for n in names if n in known]
                if len(kept) != len(names):
                    notes.append(
                        f"{field_name} referenced unknown columns that were dropped"
                    )
                setattr(result, field_name, kept)

        # Controlled feature engineering: only registry-approved operations.
        accepted, rejected = validate_operations(
            result.suggested_operations,
            _profiles_from_context(dataset_context),
            target_column=target,
        )
        result.suggested_operations = accepted
        result.rejected_operations = list(result.rejected_operations) + list(rejected)
        result.operation_source = "provider" if accepted else "deterministic_defaults"
        if rejected:
            notes.append(
                f"{len(rejected)} suggested operation(s) refused by the registry"
            )

        if accepted:
            existing = {entry.get("name") for entry in result.feature_engineering}
            for op in accepted:
                feature = (op.get("features") or [op.get("column", "")])[0]
                if feature in existing:
                    continue
                result.feature_engineering.append({
                    "name": feature,
                    "description": op.get("reason")
                    or f"{op['operation']} on '{op['column']}'",
                    "type": op.get("category", "derived"),
                    "operation": op.get("operation", ""),
                    "source": "validated_operation",
                })
                existing.add(feature)

        # Controlled model selection: only registry-validated, task-compatible
        # models survive; every refusal carries a reason for the audit trail.
        context_problem = dataset_context.get("problem_type") or result.problem_type
        accepted_recs, rejected_recs = validate_model_recommendations(
            result.model_recommendations, context_problem
        )
        result.model_recommendations = accepted_recs
        result.rejected_model_recommendations = (
            list(result.rejected_model_recommendations) + list(rejected_recs)
        )
        if accepted_recs:
            result.model_selection_source = (
                "deterministic_defaults" if result.is_fallback else "provider"
            )
        else:
            result.model_selection_source = "none"
        if rejected_recs:
            notes.append(
                f"{len(rejected_recs)} model recommendation(s) refused by the registry"
            )
