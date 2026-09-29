"""Abstract base class for LLM providers and shared data models."""
from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel, Field


# Supported model names for validation
CLASSIFICATION_MODELS = [
    "logistic_regression", "decision_tree_clf", "random_forest_clf",
    "gradient_boosting_clf", "hist_gradient_boosting_clf",
    "knn_clf", "svm_clf", "naive_bayes"
]
REGRESSION_MODELS = [
    "linear_regression", "ridge", "lasso",
    "decision_tree_reg", "random_forest_reg",
    "gradient_boosting_reg", "hist_gradient_boosting_reg",
    "knn_reg", "svr"
]
ALL_MODELS = set(CLASSIFICATION_MODELS + REGRESSION_MODELS)

CLASSIFICATION_METRICS = [
    "accuracy", "f1_weighted", "precision_weighted",
    "recall_weighted", "balanced_accuracy", "roc_auc"
]
REGRESSION_METRICS = [
    "r2", "neg_mean_squared_error", "neg_mean_absolute_error",
    "neg_root_mean_squared_error"
]


class DatasetAnalysisResult(BaseModel):
    """Structured result from LLM dataset analysis."""
    problem_type: str = Field(..., description="classification or regression")
    target_column: str
    reasoning: str
    preprocessing: list[dict[str, str]] = []
    feature_engineering: list[dict[str, str]] = []
    candidate_models: list[str] = []
    recommended_metric: str = "f1_weighted"
    optimization_strategy: str = "optuna"
    warnings: list[str] = []
    confidence: float = 0.8


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

    def validate_analysis(self, raw_response: dict) -> DatasetAnalysisResult:
        """Validate and sanitize raw LLM response against supported options."""
        result = DatasetAnalysisResult(**raw_response)

        # Validate problem_type
        if result.problem_type not in ("classification", "regression"):
            result.problem_type = "classification"

        # Clamp confidence
        result.confidence = max(0.0, min(1.0, result.confidence))

        # Validate candidate_models against supported list
        valid_models = [m for m in result.candidate_models if m in ALL_MODELS]
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

        return result
