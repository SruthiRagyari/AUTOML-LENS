"""Pydantic schemas for API request/response models."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─── Dataset ───────────────────────────────────────────────
class DatasetUploadResponse(BaseModel):
    id: int
    filename: str
    original_filename: str
    rows: int
    columns: int
    file_size: int
    uploaded_at: str


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    inferred_type: str
    unique_count: int
    missing_count: int
    missing_percentage: float
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    cardinality: int = 0
    sample_values: list[Any] = []


class DatasetProfileResponse(BaseModel):
    dataset_id: int
    rows: int
    columns: int
    memory_usage_mb: float
    duplicate_rows: int
    duplicate_percentage: float
    total_missing: int
    total_missing_percentage: float
    column_profiles: list[ColumnProfile]
    warnings: list[str]
    suggested_target: Optional[str] = None
    suggested_problem_type: Optional[str] = None
    column_type_summary: dict[str, int] = {}


# ─── Experiment ────────────────────────────────────────────
class ExperimentCreateRequest(BaseModel):
    dataset_id: int
    name: Optional[str] = None
    target_column: str
    problem_type: str = "auto"  # auto / classification / regression
    primary_metric: Optional[str] = None
    mode: str = "llm_assisted"  # baseline / automl / llm_assisted
    n_folds: int = 5
    n_trials: int = 20
    task_description: Optional[str] = None


class ExperimentResponse(BaseModel):
    id: int
    name: Optional[str]
    dataset_id: int
    target_column: Optional[str]
    problem_type: Optional[str]
    primary_metric: Optional[str]
    mode: str
    n_folds: int
    n_trials: int
    status: str
    llm_provider: Optional[str] = None
    best_model_name: Optional[str] = None
    best_score: Optional[float] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentResponse]
    total: int


# ─── LLM Analysis ─────────────────────────────────────────
class LLMAnalysisResponse(BaseModel):
    provider: str
    problem_type: str
    target_column: str
    reasoning: str
    preprocessing: list[dict]
    feature_engineering: list[dict]
    candidate_models: list[str]
    recommended_metric: str
    optimization_strategy: str
    warnings: list[str]
    confidence: float


# ─── Training ─────────────────────────────────────────────
class TrainRequest(BaseModel):
    model_names: Optional[list[str]] = None  # None = use all/recommended
    fast_demo: bool = False


class TrainingStatusResponse(BaseModel):
    experiment_id: int
    status: str
    progress: float  # 0-100
    current_model: Optional[str] = None
    models_completed: int = 0
    models_total: int = 0
    message: Optional[str] = None


class ModelResult(BaseModel):
    model_name: str
    display_name: str
    status: str
    baseline_metrics: Optional[dict] = None
    optimized_metrics: Optional[dict] = None
    best_params: Optional[dict] = None
    cv_scores: Optional[list[float]] = None
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None
    is_best: bool = False


class TrainingResultsResponse(BaseModel):
    experiment_id: int
    problem_type: str
    primary_metric: str
    mode: str
    models: list[ModelResult]
    best_model: Optional[ModelResult] = None
    llm_explanation: Optional[str] = None
    total_training_time: Optional[float] = None


# ─── Explainability ───────────────────────────────────────
class ExplainabilityResponse(BaseModel):
    method: str
    feature_importance: list[dict]
    top_features: list[str]
    explanation_text: str


# ─── Prediction ───────────────────────────────────────────
class PredictRequest(BaseModel):
    features: dict[str, Any]


class PredictResponse(BaseModel):
    prediction: Any
    confidence: Optional[float] = None
    probabilities: Optional[dict[str, float]] = None
    model_name: str


class BatchPredictResponse(BaseModel):
    num_predictions: int
    predictions_file: str
    preview: list[dict]


# ─── Chat ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    experiment_id: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    provider: str


# ─── Report ───────────────────────────────────────────────
class ReportResponse(BaseModel):
    report_path: str
    report_url: str


# ─── Health ───────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    llm_available: bool


# ─── Model Download Metadata ─────────────────────────────
class ModelMetadata(BaseModel):
    model_name: str
    display_name: str
    target_column: str
    problem_type: str
    features: list[str]
    preprocessing_summary: dict
    best_params: dict
    metrics: dict
    random_seed: int = 42
    timestamp: str
    library_versions: dict


# ─── Input Schema (for prediction form) ──────────────────
class InputFieldSchema(BaseModel):
    name: str
    dtype: str
    required: bool = True
    sample_values: list[Any] = []
