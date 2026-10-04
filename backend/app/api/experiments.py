"""Experiment API routes - the core AutoML pipeline."""
import asyncio
import json
import logging
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.core.database import get_db, Dataset, Experiment, TrainingRun, Prediction
from app.services.profiler import DatasetProfiler
from app.services.preprocessor import PreprocessingEngine
from app.services.feature_engineer import FeatureEngineer
from app.services.dataset_description import (
    build_dataset_description,
    render_dataset_description,
)
from app.services.feature_operations import validate_operations
from app.services.model_registry import (
    ModelRegistry, validate_model_recommendations,
)
from app.services.trainer import TrainingManager
from app.services.training_progress import (
    TrainingProgress, PHASE_PERSISTING, PHASE_SELECTING,
    STATUS_COMPLETED as TP_COMPLETED, STATUS_FAILED as TP_FAILED,
)
from app.services import training_progress
from app.services import training_progress as tp
from app.core.database import get_session_factory
from app.services.optimizer import OPTUNA_SEED
from app.services.evaluator import Evaluator, get_scoring_string, get_metric_spec
from app.services.explainer import Explainer
from app.services.predictor import (
    Predictor, InputValidationError, validate_prediction_inputs,
)
from app.services.reporter import ReportGenerator
from app.models.schemas import ExperimentCreateRequest
from app.utils.file_utils import load_dataframe

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory store for trained objects (in production, use proper persistence).
# Deliberately bounded: a long-lived process must not pin every trained bundle
# forever, and the uploaded dataset is never kept here at all — evicted
# entries are rebuilt from the artifacts already written to disk.
class _ExperimentCache:
    """LRU mapping of experiment id -> trained-object bundle.

    ``maxsize`` is exposed so the bound is visible to operators and tests; the
    least-recently-used entry is dropped on insert once that bound is exceeded.
    Dropping an entry is never a correctness problem: every reader falls back to
    reloading the model, preprocessor and dataset from disk.
    """

    def __init__(self, maxsize: int = 8):
        self.maxsize = maxsize
        self._data: OrderedDict = OrderedDict()

    @staticmethod
    def _key(exp_id):
        return int(exp_id)

    def __setitem__(self, exp_id, value):
        key = self._key(exp_id)
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            _evicted_key, _evicted = self._data.popitem(last=False)
            logger.info(
                f"Experiment cache full ({self.maxsize}); evicted experiment "
                f"{_evicted_key} (reloadable from disk)."
            )

    def __getitem__(self, exp_id):
        return self._data[self._key(exp_id)]

    def __delitem__(self, exp_id):
        del self._data[self._key(exp_id)]

    def __contains__(self, exp_id):
        return self._key(exp_id) in self._data

    def get(self, exp_id, default=None):
        key = self._key(exp_id)
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def keys(self):
        return list(self._data.keys())

    def clear(self):
        self._data.clear()

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"<_ExperimentCache {len(self._data)}/{self.maxsize} entries>"


_experiment_cache = _ExperimentCache()


def _get_llm_manager():
    from app.main import llm_manager
    return llm_manager


def _safe_json(text: Optional[str]):
    """Parse a stored JSON column, returning ``None`` instead of raising."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


# ─────────────────────── input schema helpers ─────────────────────────
# Columns the profiler judges non-informative are never asked of the user.
_NON_INPUT_COLUMN_TYPES = ("id_like", "constant")


def _resolve_dataset_profile(ds: Dataset, db: Session) -> dict:
    """Return a dataset profile, generating and caching it when never profiled.

    ``profile_json`` used to be written only by ``GET /datasets/{id}/profile``, so
    anything depending on the profile silently came back empty until the user
    happened to open the profiling page. Producing it on demand and caching it the
    same way the datasets router does keeps schema lookups reliable without
    changing that endpoint's behaviour.
    """
    if ds.profile_json:
        try:
            profile = json.loads(ds.profile_json)
        except (TypeError, ValueError):
            profile = {}
        if profile.get("column_profiles"):
            return profile

    try:
        df = load_dataframe(ds.file_path)
    except Exception as e:
        raise HTTPException(400, f"Cannot read dataset: {str(e)}")

    profiler = DatasetProfiler()
    profile = profiler.profile(df)
    suggested_target, suggested_type = profiler.suggest_target(profile)
    profile["dataset_id"] = ds.id
    profile["suggested_target"] = suggested_target
    profile["suggested_problem_type"] = suggested_type

    ds.profile_json = json.dumps(profile, default=str)
    db.commit()
    logger.info(f"Generated and cached profile for dataset {ds.id} on demand")
    return profile


def _model_input_columns(exp: Experiment) -> list:
    """Raw input columns the fitted pipeline consumes, read from persisted state.

    ``preprocessing_json`` is stored on the experiment row during training, so this
    survives a restart and never asks the user for engineered columns or for
    columns the pipeline dropped.
    """
    if not exp.preprocessing_json:
        return []
    try:
        summary = json.loads(exp.preprocessing_json)
    except (TypeError, ValueError):
        return []
    columns: list = []
    for key in ("numerical_columns", "categorical_columns", "text_columns", "datetime_columns"):
        columns.extend(summary.get(key) or [])
    return list(dict.fromkeys(columns))


def _build_input_schema(exp: Experiment, ds: Dataset, db: Session) -> list:
    """Build the user-facing prediction schema for a trained experiment."""
    profile = _resolve_dataset_profile(ds, db)
    column_profiles = profile.get("column_profiles", [])
    profiled = {cp["name"]: cp for cp in column_profiles}

    expected = [name for name in _model_input_columns(exp) if name in profiled]
    if not expected:
        expected = [
            cp["name"] for cp in column_profiles
            if cp["name"] != exp.target_column
            and cp.get("inferred_type") not in _NON_INPUT_COLUMN_TYPES
        ]

    try:
        df_orig = load_dataframe(ds.file_path)
    except Exception as e:  # sample values are a convenience, not a requirement
        logger.warning(f"Could not read dataset {ds.id} for schema samples: {e}")
        df_orig = None

    fields = []
    for name in expected:
        inferred = profiled.get(name, {}).get("inferred_type", "")
        field_type = "number" if inferred in ("numerical", "boolean") else "text"
        sample_values = []
        if df_orig is not None and name in df_orig.columns:
            try:
                sample_values = [
                    (int(v) if isinstance(v, bool)
                     else float(v) if hasattr(v, "real") else str(v))
                    for v in df_orig[name].dropna().unique()[:5].tolist()
                ]
            except Exception:
                sample_values = []
        fields.append({
            "name": name,
            "type": field_type,
            "required": True,
            "sample_values": sample_values,
        })
    return fields


def _prediction_schema(exp: Experiment, db: Session, cache: Optional[dict] = None) -> list:
    """Schema used to validate prediction inputs, memoised on the warm cache."""
    if cache is not None and cache.get("input_schema"):
        return cache["input_schema"]
    ds = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found for this experiment")
    schema = _build_input_schema(exp, ds, db)
    if cache is not None:
        cache["input_schema"] = schema
    return schema


# Scoring aliases: the request uses sklearn naming, results store short names.
_METRIC_KEY_ALIASES = {
    "neg_mean_squared_error": "mse",
    "neg_root_mean_squared_error": "rmse",
    "neg_mean_absolute_error": "mae",
}


def _best_score_and_metric(exp: Experiment, metrics: Optional[dict]):
    """Read the primary metric from a result dict and report which key was used.

    When the requested metric is not present for this model, the first numeric
    metric is used — but the key actually used is returned as well, so responses
    can label the score honestly instead of implying it is ``primary_metric``.
    """
    if not metrics:
        return None, None
    metric_key = _METRIC_KEY_ALIASES.get(exp.primary_metric, exp.primary_metric)
    value = metrics.get(metric_key)
    if value is None:
        value, metric_key = next(
            ((v, k) for k, v in metrics.items() if isinstance(v, (int, float))), (None, None))
    return (float(value) if value is not None else None), metric_key


def _persist_progress(exp_id: int, snapshot: dict) -> None:
    """Write a real progress snapshot to the experiment row.

    Called from the training worker thread, so it opens its own short-lived
    session rather than touching the request-scoped one. Every failure is
    swallowed: losing a progress update must never abort a training run. The
    snapshot contains only observed state, so persisting it cannot record
    anything untrue.
    """
    session = None
    try:
        session = get_session_factory()()
        exp = session.query(Experiment).filter(Experiment.id == exp_id).first()
        if exp is None:
            return
        exp.progress_json = json.dumps(snapshot, default=str)
        session.commit()
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning(f"Could not persist training progress: {exc}")
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def _exp_to_progress_response(exp: Experiment, live: Optional[TrainingProgress]) -> dict:
    """Progress payload: the live tracker when this process owns the run,
    otherwise the last persisted snapshot (e.g. after a restart)."""
    snapshot = None
    source = "none"
    if live is not None:
        snapshot = live.snapshot()
        source = "live"
    elif exp.progress_json:
        try:
            snapshot = json.loads(exp.progress_json)
            source = "persisted"
        except (TypeError, ValueError):
            snapshot = None
    return {
        "experiment_id": exp.id,
        "experiment_status": exp.status,
        "source": source,
        "progress": snapshot,
        "error_message": exp.error_message,
    }


@router.post("")
async def create_experiment(req: ExperimentCreateRequest, db: Session = Depends(get_db)):
    """Create a new experiment."""
    ds = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")

    exp = Experiment(
        dataset_id=req.dataset_id,
        name=req.name or f"Experiment on {ds.original_filename}",
        target_column=req.target_column,
        problem_type=req.problem_type if req.problem_type != "auto" else None,
        primary_metric=req.primary_metric,
        mode=req.mode,
        n_folds=req.n_folds,
        n_trials=req.n_trials,
        status="created",
        task_description=req.task_description,
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)

    return _exp_to_dict(exp)


@router.get("")
async def list_experiments(db: Session = Depends(get_db)):
    """List all experiments."""
    exps = db.query(Experiment).order_by(Experiment.created_at.desc()).all()
    return {
        "experiments": [_exp_to_dict(e) for e in exps],
        "total": len(exps),
    }


def _resolve_bench_file(rel_path: str) -> Optional[Path]:
    for base in [Path.cwd(), Path(__file__).resolve().parents[3], Path(__file__).resolve().parents[2]]:
        candidate = base / rel_path
        if candidate.exists():
            return candidate
    return None


@router.get("/benchmarks/summary")
async def get_benchmarks_summary():
    """Retrieve the aggregated multi-seed benchmark results and research evaluation."""
    summary_path = _resolve_bench_file("benchmarks/results/benchmark_summary.json")
    if summary_path and summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading benchmark summary: {e}")
    results_path = _resolve_bench_file("benchmarks/results/benchmark_results.json")
    if results_path and results_path.exists():
        try:
            from app.services.benchmarker import aggregate_benchmark_records
            with open(results_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            return aggregate_benchmark_records(raw_data)
        except Exception as e:
            logger.warning(f"Error aggregating raw benchmark results: {e}")
    return {
        "status": "not_available",
        "message": "No benchmark results found. Run benchmarks to generate empirical multi-seed data.",
        "datasets": {},
    }


@router.get("/benchmarks/report")
async def get_benchmarks_report():
    """Download or view the standalone HTML research benchmark report."""
    from fastapi.responses import HTMLResponse
    report_path = _resolve_bench_file("docs/BENCHMARK_REPORT.html")
    if report_path and report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)
        except Exception as e:
            logger.warning(f"Error reading benchmark report: {e}")
    
    summary = await get_benchmarks_summary()
    if summary.get("datasets"):
        reporter = ReportGenerator()
        html = reporter.generate_benchmark_html_report(summary)
        return HTMLResponse(content=html, status_code=200)

    raise HTTPException(status_code=404, detail="Benchmark report not found. Please run benchmarks first.")


@router.get("/models/catalog")
async def get_models_catalog():
    """Return catalog of all supported models in the registry."""
    registry = ModelRegistry()
    models = []
    for name, m in registry._models.items():
        models.append({
            "name": m.name,
            "display_name": m.display_name,
            "task": m.task,
            "has_search_space": m.search_space is not None,
            "default_params": {k: str(v) for k, v in m.default_params.items()},
            "explainability_method": m.explainability_method,
        })
    return {"models": models, "total": len(models)}


@router.get("/{exp_id}")
async def get_experiment(exp_id: int, db: Session = Depends(get_db)):
    """Get experiment details."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return _exp_to_dict(exp)


@router.delete("/{exp_id}")
async def delete_experiment(exp_id: int, db: Session = Depends(get_db)):
    """Delete an experiment and its associated runs/predictions."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")

    # Delete dependent predictions and training runs
    db.query(Prediction).filter(Prediction.experiment_id == exp_id).delete()
    db.query(TrainingRun).filter(TrainingRun.experiment_id == exp_id).delete()

    # Clean up artifacts if files exist
    if exp.model_path:
        try:
            mp = Path(exp.model_path)
            if mp.exists():
                mp.unlink()
        except Exception as e:
            logger.warning(f"Could not delete model file {exp.model_path}: {e}")
    if exp.preprocessor_path:
        try:
            pp = Path(exp.preprocessor_path)
            if pp.exists():
                pp.unlink()
        except Exception as e:
            logger.warning(f"Could not delete preprocessor file {exp.preprocessor_path}: {e}")

    db.delete(exp)
    db.commit()
    return {"message": f"Experiment {exp_id} deleted successfully", "id": exp_id}


# Statuses whose stored analysis is already the answer for this experiment.
# /analyze must never move one of these back, and must not re-call the provider
# for them unless the caller asks for it explicitly (``?force=true``).
_ANALYSIS_FROZEN_STATUSES = ("training", "completed")


@router.post("/{exp_id}/analyze")
async def analyze_experiment(exp_id: int, force: bool = False,
                             db: Session = Depends(get_db)):
    """Run LLM analysis on the dataset.

    Idempotent for a run that already has a stored analysis and is training or
    completed: the stored payload is returned unchanged. Re-running the analysis
    used to reset a completed experiment's status to ``analyzed`` - which hid its
    results from the UI - and the page re-ran it on every load. Pass
    ``?force=true`` for a deliberate re-run.
    """
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")

    if exp.llm_analysis_json and not force and exp.status in _ANALYSIS_FROZEN_STATUSES:
        stored = _safe_json(exp.llm_analysis_json)
        if isinstance(stored, dict):
            return stored
        logger.warning(
            f"Stored analysis for experiment {exp_id} is unreadable; re-running."
        )

    ds = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
    df = load_dataframe(ds.file_path)

    # Profile if needed
    profiler = DatasetProfiler()
    profile = profiler.profile(df)

    # Auto-detect problem type if needed
    if not exp.problem_type or exp.problem_type == "auto":
        target_cp = next((cp for cp in profile["column_profiles"] if cp["name"] == exp.target_column), None)
        if target_cp:
            exp.problem_type = profiler.detect_problem_type(target_cp)
        else:
            exp.problem_type = "classification"

    # Set default metric if needed
    if not exp.primary_metric:
        exp.primary_metric = "f1_weighted" if exp.problem_type == "classification" else "neg_root_mean_squared_error"

    # Build dataset context for LLM
    target_series = df[exp.target_column]
    target_stats = {}
    if pd.api.types.is_numeric_dtype(target_series):
        target_stats = {
            "mean": float(target_series.mean()),
            "std": float(target_series.std()),
            "min": float(target_series.min()),
            "max": float(target_series.max()),
            "unique": int(target_series.nunique()),
        }
    else:
        vc = target_series.value_counts().head(10)
        target_stats = {str(k): int(v) for k, v in vc.items()}

    # Compact, char-budgeted description drives the prompt. The operation catalog
    # inside it is the only vocabulary a provider may choose feature operations
    # from; validation later enforces it even if the model ignores the instruction.
    description: dict = {}
    description_text = ""
    try:
        description = build_dataset_description(
            df, profile=profile, target_column=exp.target_column,
            problem_type=exp.problem_type,
            task_description=exp.task_description or "",
        )
        description_text = render_dataset_description(description)
    except Exception as exc:
        logger.warning(f"Dataset description build failed, using raw context: {exc}")

    dataset_context = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_percentages": {col: round(df[col].isnull().mean() * 100, 2) for col in df.columns},
        "unique_counts": {col: int(df[col].nunique()) for col in df.columns},
        "target_column": exp.target_column,
        "target_stats": target_stats,
        "sample_rows": df.head(3).to_dict(orient="records"),
        "task_description": exp.task_description or "",
        # Ground truth for model-selection validation: the pipeline
        # trains with the configured problem type, not the provider's.
        "problem_type": exp.problem_type,
        # Profile + registry context for leakage-safe operation validation.
        "column_profiles": profile["column_profiles"],
        "feature_operation_catalog": description.get("feature_operation_catalog", {}),
        "allowed_operations": description.get("allowed_operations", {}),
        "description_text": description_text,
    }

    # Run LLM analysis and structured pipeline planning
    llm = _get_llm_manager()
    analysis = await llm.analyze_dataset(dataset_context)
    try:
        plan_payload = await llm.plan_pipeline(dataset_context)
        analysis["pipeline_plan"] = plan_payload.get("plan")
    except Exception as exc:
        logger.warning(f"Pipeline planning failed during analyze: {exc}")
        from app.services.pipeline_planner import generate_deterministic_plan
        det_plan = generate_deterministic_plan(dataset_context)
        analysis["pipeline_plan"] = det_plan.model_dump()

    exp.llm_analysis_json = json.dumps(analysis, default=str)
    exp.llm_provider = analysis.get("provider_used", "unknown")
    exp.status = "analyzed"
    db.commit()

    return analysis


@router.post("/{exp_id}/train")
async def train_experiment(exp_id: int, fast_demo: bool = False, seed: Optional[int] = None, db: Session = Depends(get_db)):
    """Run the full training pipeline."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")

    ds = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
    df = load_dataframe(ds.file_path)
    effective_seed = OPTUNA_SEED if seed is None else int(seed)

    try:
        exp.status = "training"
        db.commit()

        # Profile
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        # Resolve an "auto"/unspecified task before anything downstream reads it.
        # /analyze always did this; /train did not, so an auto experiment chose
        # models for task ``None``, trained nothing and still reported success.
        if not exp.problem_type or exp.problem_type == "auto":
            target_cp = next(
                (cp for cp in profile["column_profiles"]
                 if cp["name"] == exp.target_column), None)
            exp.problem_type = (
                profiler.detect_problem_type(target_cp)
                if target_cp else "classification"
            )
        # Every metric lookup below assumes a name, so default it here too.
        if not exp.primary_metric:
            exp.primary_metric = (
                "f1_weighted" if exp.problem_type == "classification"
                else "neg_root_mean_squared_error"
            )
        db.commit()

        # Structured AutoML Pipeline Plan resolution
        from app.services.pipeline_planner import (
            AutoMLPipelinePlan, PipelinePlanValidator, generate_deterministic_plan
        )
        plan_dataset_context = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_percentages": {col: round(df[col].isnull().mean() * 100, 2) for col in df.columns},
            "unique_counts": {col: int(df[col].nunique()) for col in df.columns},
            "target_column": exp.target_column,
            "problem_type": exp.problem_type,
            "column_profiles": profile["column_profiles"],
        }

        # Feature operations and plan proposed by the LLM analysis, re-validated against the
        # fresh profile so only registry-approved, leakage-safe entries execute.
        llm_result: dict = {}
        raw_plan = None
        if exp.llm_analysis_json:
            try:
                stored = json.loads(exp.llm_analysis_json)
                llm_result = stored.get("result", {}) if isinstance(stored, dict) else {}
                raw_plan = stored.get("pipeline_plan") if isinstance(stored, dict) else None
            except Exception as exc:
                logger.warning(f"Stored LLM analysis unreadable, ignoring: {exc}")

        if raw_plan:
            pipeline_plan = PipelinePlanValidator.validate_and_sanitize(
                raw_plan, plan_dataset_context,
                provider_used=exp.llm_provider or "unknown",
            )
        else:
            pipeline_plan = generate_deterministic_plan(plan_dataset_context)

        proposed_ops = pipeline_plan.feature_operations or (llm_result.get("suggested_operations") or [])
        rejected_ops = list(pipeline_plan.rejected_operations or []) + list(llm_result.get("rejected_operations") or [])
        accepted_ops, late_rejections = validate_operations(
            proposed_ops, profile["column_profiles"], exp.target_column
        )
        rejected_ops.extend(late_rejections)

        y = df[exp.target_column].copy()

        # Encode target if needed
        label_map = None
        if exp.problem_type == "classification" and y.dtype == "object":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=df.index, name=exp.target_column)
            label_map = {i: c for i, c in enumerate(le.classes_)}

        # Split FIRST: frequency maps, z-score moments, interaction selection and
        # scaling must all be learned from the training rows only.
        train_idx, test_idx = train_test_split(
            df.index, test_size=0.2, random_state=effective_seed,
            stratify=y if exp.problem_type == "classification" else None,
        )
        df_train, df_test = df.loc[train_idx], df.loc[test_idx]

        # Feature engineering (fit on the training split only)
        fe = FeatureEngineer(seed=effective_seed)
        fe.rejected_operations = rejected_ops
        fe.fit(
            df_train, profile["column_profiles"], exp.target_column,
            operations=accepted_ops or None,
        )
        df_eng_train = fe.transform(df_train)

        # Preprocessing (fit on the engineered training split only)
        preprocessor = PreprocessingEngine()
        X_train, feature_names = preprocessor.build_and_fit(
            df_eng_train, exp.target_column, profile["column_profiles"], exp.problem_type,
            extra_numeric=fe.new_features, feature_engineer=fe,
        )
        X_test = preprocessor.transform(df_test, target_column=exp.target_column)
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        # Select models
        registry = ModelRegistry()
        # Fast demo deliberately shrinks the search budget; the values actually
        # used are recorded in the trace so a short run is never presented as a
        # full-budget one.
        n_folds = 3 if fast_demo else exp.n_folds
        n_trials = 5 if fast_demo else exp.n_trials
        optimization_budget = {
            "configured_n_folds": exp.n_folds,
            "configured_n_trials": exp.n_trials,
            "n_folds_used": n_folds,
            "n_trials_used": n_trials,
            "fast_demo": bool(fast_demo),
            "budget_reduced": bool(fast_demo) and (
                n_folds != exp.n_folds or n_trials != exp.n_trials
            ),
            "seed": effective_seed,
            # metric -> scoring -> direction for this experiment, persisted.
            "metric_spec": get_metric_spec(exp.primary_metric).to_dict(),
        }

        # Model selection: validated pipeline_plan first, then LLM recommendations, then defaults.
        model_names = None
        selection_source = "registry_default"
        proposed_recs = list(llm_result.get("model_recommendations") or [])
        rejected_recs = list(pipeline_plan.rejected_models or []) + list(llm_result.get("rejected_model_recommendations") or [])
        accepted_recs, late_rejections = validate_model_recommendations(
            proposed_recs, exp.problem_type
        )
        rejected_recs.extend(late_rejections)

        if fast_demo:
            model_names = registry.get_fast_demo_models(exp.problem_type)
            selection_source = "fast_demo"
        elif accepted_recs:
            model_names = [rec["model_id"] for rec in accepted_recs]
            selection_source = llm_result.get("model_selection_source") or (
                "deterministic_defaults" if llm_result.get("is_fallback")
                else "provider"
            )
        elif pipeline_plan.candidate_models:
            model_names = pipeline_plan.candidate_models
            selection_source = "provider" if pipeline_plan.source == "llm" else "deterministic_defaults"
            accepted_recs = [
                {"model_id": m, "reason": "Selected by validated pipeline plan"}
                for m in pipeline_plan.candidate_models
            ]
        elif llm_result.get("candidate_models"):
            wrapped, cand_rejected = validate_model_recommendations(
                [{"model_id": m} for m in llm_result["candidate_models"]],
                exp.problem_type,
            )
            rejected_recs.extend(cand_rejected)
            if wrapped:
                model_names = [rec["model_id"] for rec in wrapped]
                selection_source = "candidate_models"

        model_defs = registry.get_models(exp.problem_type, model_names)
        if not model_defs:
            model_defs = registry.get_models(exp.problem_type)

        # Train
        trainer = TrainingManager(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            model_definitions=model_defs,
            problem_type=exp.problem_type,
            primary_metric=exp.primary_metric,
            n_folds=n_folds, n_trials=n_trials,
            mode=exp.mode,
            # Deterministic Optuna seed, recorded in the per-model provenance.
            optimization_seed=effective_seed,
            fast_demo=bool(fast_demo),
        )

        # Train. The heavy work (feature engineering, preprocessing, model
        # fitting, Optuna, CV, holdout evaluation) is CPU-bound and blocking,
        # so it is executed in a worker thread. Without this a multi-minute run
        # would stall the asyncio event loop and the whole API - including
        # GET /{id}/progress - would look frozen while the server was busy.
        # The tracker is registered so a status endpoint can read it, and each
        # real transition is also persisted to the experiment row.
        progress = TrainingProgress(
            exp_id,
            model_names=[(m.name, m.display_name) for m in model_defs],
            n_trials=n_trials,
            persist=lambda snap: _persist_progress(exp_id, snap),
        )
        training_progress.register(progress)
        progress.start(tp.PHASE_PREPARING)
        _persist_progress(exp_id, progress.snapshot())

        try:
            results = await asyncio.to_thread(
                trainer.train_all, None, progress
            )
        finally:
            progress.set_phase(tp.PHASE_SELECTING)

        # Build ensemble candidates strictly using OOF predictions on the training split (leakage-safe).
        # Final holdout is untouched during ensemble candidate creation, weight learning, and selection.
        enable_ensemble = (
            pipeline_plan.ensemble_strategy.enabled
            if pipeline_plan and hasattr(pipeline_plan, "ensemble_strategy")
            else True
        )
        ensemble_candidates = []
        if enable_ensemble:
            from app.services.ensemble import build_ensemble_candidates
            ensemble_candidates = build_ensemble_candidates(
                candidate_results=results,
                model_definitions=model_defs,
                X_train=X_train,
                y_train=y_train,
                problem_type=exp.problem_type,
                primary_metric=exp.primary_metric,
                n_folds=n_folds,
                seed=effective_seed,
            )
        ensemble_results = [cand[0] for cand in ensemble_candidates]

        # 1. Select the winning model among registry candidate models (CV evidence ONLY)
        selection_metric = exp.primary_metric
        best = trainer.select_best_model(results, selection_metric)
        selection_provenance = trainer.build_selection_provenance(
            results, selection_metric, best
        )

        # 2. Final holdout evaluation for ensemble candidates:
        # Evaluated ONCE on untouched holdout AFTER selection has completed.
        for ens_res, ens_model in ensemble_candidates:
            t0 = time.time()
            holdout_metrics = trainer._evaluate(ens_model, X_test, y_test)
            ens_res.prediction_time = round(time.time() - t0, 4)
            ens_res.optimized_metrics = holdout_metrics
            ens_res.baseline_metrics = holdout_metrics

        # 3. Model Fusion Selection: compare individual winner against ensemble candidates (CV evidence only)
        all_candidates = results + ensemble_results
        fusion_winner = trainer.select_best_model(all_candidates, selection_metric)
        fusion_provenance = trainer.build_selection_provenance(
            all_candidates, selection_metric, fusion_winner
        )
        is_ensemble_winner = (
            fusion_winner is not None and fusion_winner.model_name in [er.model_name for er in ensemble_results]
        )

        progress.set_phase(PHASE_PERSISTING)

        # Explainability on best model
        explain_result = None
        if best and best.trained_model is not None:
            try:
                explain_result = Explainer.explain(
                    best.trained_model, X_test, feature_names, y_test=y_test
                )
            except Exception as e:
                logger.warning(f"Explainability failed: {e}")

        # Save best model
        model_path = None
        if best and best.trained_model is not None:
            model_path = str(settings.models_path / f"model_{exp_id}.joblib")
            joblib.dump(best.trained_model, model_path)

            if is_ensemble_winner and fusion_winner is not None:
                joblib.dump(fusion_winner.trained_model, str(settings.models_path / f"model_{exp_id}_ensemble.joblib"))

            # Save metadata
            import sklearn
            meta = {
                "model_name": best.model_name,
                "display_name": best.display_name,
                "target_column": exp.target_column,
                "problem_type": exp.problem_type,
                "features": feature_names,
                "preprocessing_summary": preprocessor.get_summary(),
                "best_params": best.best_params,
                "metrics": best.optimized_metrics or best.baseline_metrics,
                "random_seed": effective_seed,
                "timestamp": str(exp.created_at),
                "library_versions": {
                    "scikit-learn": sklearn.__version__,
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                },
            }
            if is_ensemble_winner and fusion_winner is not None:
                meta["ensemble"] = {
                    "is_ensemble": True,
                    "strategy": getattr(fusion_winner.trained_model, "strategy", "weighted_average"),
                    "members": getattr(fusion_winner.trained_model, "member_names", []),
                    "weights": getattr(fusion_winner.trained_model, "weights_dict", {}),
                    "selection_cv_score": (fusion_winner.optimization or {}).get("best_cv_score"),
                    "holdout_used_for_selection": False,
                }
            meta_path = str(settings.models_path / f"model_{exp_id}_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)
            # Persist the fitted preprocessing (it embeds the fitted feature
            # engineer) so predictions reuse the exact training-split statistics
            # instead of refitting on the whole dataset after a restart.
            joblib.dump(
                preprocessor,
                str(settings.models_path / f"model_{exp_id}_preprocessor.joblib"),
            )

        # LLM explanation of results
        llm_explanation = None
        llm = _get_llm_manager()
        if best:
            try:
                # Resolve every model's score through the experiment's own primary
                # metric. Taking "the first number in the metrics dict" used to
                # label an accuracy value as if it were f1_weighted.
                best_score_for_prompt, _ = _best_score_and_metric(
                    exp, best.optimized_metrics or best.baseline_metrics
                )
                per_model_scores = {
                    r.model_name: _best_score_and_metric(
                        exp, r.optimized_metrics or r.baseline_metrics
                    )[0]
                    for r in results
                }
                results_context = {
                    "best_model": best.model_name,
                    "best_display_name": best.display_name,
                    "best_score": best_score_for_prompt,
                    "metric_name": exp.primary_metric,
                    "problem_type": exp.problem_type,
                    "best_params": best.best_params,
                    "n_trials": n_trials,
                    "all_results": [
                        {
                            "model_name": r.model_name,
                            "display_name": r.display_name,
                            "score": per_model_scores.get(r.model_name),
                            "status": r.status,
                        }
                        for r in results
                    ],
                }
                llm_explanation = await llm.explain_results(results_context)
            except Exception as e:
                logger.warning(f"LLM explanation failed: {e}")

        # Save results to DB
        results_data = []
        for r in results:
            rd = {
                "model_name": r.model_name,
                "display_name": r.display_name,
                "status": r.status,
                "baseline_metrics": r.baseline_metrics,
                "optimized_metrics": r.optimized_metrics,
                "best_params": r.best_params,
                "cv_scores": r.cv_scores,
                "training_time": r.training_time,
                "prediction_time": r.prediction_time,
                "optimization_history": r.optimization_history,
                # Real Optuna provenance for this model (metric/scoring/
                # direction, trial counts, folds, best CV score, status, error).
                "optimization": r.optimization,
                "is_best": best is not None and r.model_name == best.model_name,
            }
            results_data.append(rd)

            # Save training run to DB
            tr = TrainingRun(
                experiment_id=exp_id,
                model_name=r.model_name,
                display_name=r.display_name,
                status=r.status.lower(),
                baseline_metrics_json=json.dumps(r.baseline_metrics, default=str),
                optimized_metrics_json=json.dumps(r.optimized_metrics, default=str) if r.optimized_metrics else None,
                best_params_json=json.dumps(r.best_params, default=str) if r.best_params else None,
                cv_scores_json=json.dumps(r.cv_scores) if r.cv_scores else None,
                optimization_history_json=json.dumps(r.optimization_history, default=str) if r.optimization_history else None,
                optimization_json=json.dumps(r.optimization, default=str) if r.optimization else None,
                training_time=r.training_time,
                prediction_time=r.prediction_time,
                model_path=model_path if r.model_name == (best.model_name if best else None) else None,
            )
            db.add(tr)

        exp.results_json = json.dumps(results_data, default=str)
        exp.best_model_name = best.display_name if best else None
        # best_score is the FINAL HOLDOUT score of the already-selected winner.
        # Selection itself happened on CV evidence only (see selection_provenance),
        # so this number is an unbiased estimate rather than a selection artifact.
        exp.best_score, best_metric_key = _best_score_and_metric(
            exp, (best.optimized_metrics or best.baseline_metrics) if best else None
        )
        exp.preprocessing_json = json.dumps(preprocessor.get_summary(), default=str)
        exp.feature_engineering_json = json.dumps(fe.get_summary(), default=str)

        ensemble_data = [
            {
                "model_name": er.model_name,
                "display_name": er.display_name,
                "status": er.status,
                "baseline_metrics": er.baseline_metrics,
                "optimized_metrics": er.optimized_metrics,
                "best_params": er.best_params,
                "cv_scores": er.cv_scores,
                "selection_score": (er.optimization or {}).get("best_cv_score"),
                "holdout_primary": (
                    _best_score_and_metric(exp, er.optimized_metrics)[0]
                    if er.optimized_metrics else None
                ),
                "optimization": er.optimization,
                "is_best": is_ensemble_winner and er.model_name == (fusion_winner.model_name if fusion_winner else ""),
            }
            for er in ensemble_results
        ]

        fusion_summary = {
            "is_ensemble_winner": is_ensemble_winner,
            "selected_system": fusion_winner.model_name if fusion_winner else (best.model_name if best else None),
            "selected_system_display_name": fusion_winner.display_name if fusion_winner else (best.display_name if best else None),
            "individual_winner": best.model_name if best else None,
            "individual_cv_score": selection_provenance.get("selected_model_cv_score"),
            "best_ensemble": fusion_winner.model_name if is_ensemble_winner else (ensemble_results[0].model_name if ensemble_results else None),
            "best_ensemble_cv_score": (fusion_winner.optimization or {}).get("best_cv_score") if is_ensemble_winner else ((ensemble_results[0].optimization or {}).get("best_cv_score") if ensemble_results else None),
            "ensemble_candidates": ensemble_data,
            "provenance": fusion_provenance,
            "holdout_used_for_selection": False,
        }

        executed_pipeline_summary = {
            "target_column": exp.target_column,
            "problem_type": exp.problem_type,
            "primary_metric": exp.primary_metric,
            "candidate_models_trained": [r.model_name for r in results if r.status == "COMPLETED"],
            "feature_operations_applied": accepted_ops,
            "feature_operations_rejected": rejected_ops,
            "n_folds": n_folds,
            "n_trials": n_trials,
            "ensemble_enabled": enable_ensemble,
            "ensemble_strategies": (
                pipeline_plan.ensemble_strategy.strategies if pipeline_plan and hasattr(pipeline_plan, "ensemble_strategy") else []
            ),
            "seed": effective_seed,
            "fast_demo": bool(fast_demo),
        }
        pipeline_plan_provenance = {
            "source": pipeline_plan.source if pipeline_plan else "deterministic_fallback",
            "provider_used": (pipeline_plan.provider_used if pipeline_plan and pipeline_plan.provider_used else exp.llm_provider) or "fallback",
            "model_used": pipeline_plan.model_used if pipeline_plan else "unknown",
            "validation_status": pipeline_plan.validation_status if pipeline_plan else "valid",
            "validation_errors": pipeline_plan.validation_errors if pipeline_plan else [],
            "validation_warnings": pipeline_plan.validation_warnings if pipeline_plan else [],
            "fallback_reason": pipeline_plan.fallback_reason if pipeline_plan else None,
            "raw_recommendation": raw_plan,
            "validated_plan": pipeline_plan.model_dump() if pipeline_plan else None,
            "executed_pipeline": executed_pipeline_summary,
            "constraints_enforced": [
                "Zero arbitrary code execution",
                "ModelRegistry validation",
                "OPERATION_REGISTRY validation",
                "Metric direction contract verification",
                "Holdout isolation guarantee",
                "Fold-safe training-split preprocessing",
            ],
        }

        # Research trace: recommendations, refusals, trained models, winner.
        # Makes no performance claim - no A/B comparison against deterministic
        # selection was ever run.
        model_selection_trace = {
            "source": selection_source,
            # Exactly how much search budget was used, and whether fast demo
            # shrank it. No claim of a full-budget search when it was reduced.
            "optimization_budget": optimization_budget,
            "provider_used": (llm_result or {}).get("provider_used", ""),
            "is_fallback": bool((llm_result or {}).get("is_fallback", True)),
            "recommended": accepted_recs,
            "rejected": rejected_recs,
            "requested_candidates": model_names or [],
            "fast_demo": bool(fast_demo),
            "models_trained": [r.model_name for r in results
                               if r.status == "COMPLETED"],
            "models_failed": [r.model_name for r in results if r.status == "FAILED"],
            "ensemble_candidates": ensemble_data,
            "fusion": fusion_summary,
            "is_ensemble_winner": is_ensemble_winner,
            "final_model": best.model_name if best else None,
            "final_score": exp.best_score,
            "metric": best_metric_key if best else None,
            # How the winner was chosen: CV on training data, holdout untouched.
            "selection": selection_provenance,
            "pipeline_plan_provenance": pipeline_plan_provenance,
            "comparison_note": (
                "No A/B comparison against deterministic-only selection was "
                "performed; no performance-improvement claim is made."
            ),
        }
        exp.model_selection_json = json.dumps(model_selection_trace, default=str)
        if explain_result:
            exp.explainability_json = json.dumps({
                "method_used": explain_result.method_used,
                "feature_importance": explain_result.feature_importance[:20],
                "top_features": explain_result.top_features,
                "explanation_text": explain_result.explanation_text,
            }, default=str)
        exp.status = "completed"
        db.commit()

        # Record the terminal state of the run: real counts, real errors.
        progress.finish(TP_COMPLETED)
        training_progress.unregister(exp_id)

        # Cache experiment objects for predictions
        _experiment_cache[exp_id] = {
            "model": best.trained_model if best else None,
            "preprocessor": preprocessor,
            "feature_names": [cp["name"] for cp in profile["column_profiles"] if cp["name"] != exp.target_column],
            "feature_names_transformed": feature_names,
            "target_column": exp.target_column,
            "problem_type": exp.problem_type,
            "label_map": label_map,
        }

        total_time = sum(r.training_time for r in results)
        return {
            "experiment_id": exp_id,
            "status": "completed",
            "problem_type": exp.problem_type,
            "primary_metric": exp.primary_metric,
            "mode": exp.mode,
            "models": results_data,
            # Flat fields mirror the persisted experiment columns, so /train,
            # /results and GET /experiments/{id} all agree on the winner.
            # best_metric names the metric the score actually comes from.
            "best_model_name": exp.best_model_name,
            # best_score/best_metric describe the FINAL HOLDOUT evaluation of the
            # winner; selection used CV evidence only (see selection_cv_score).
            "best_score": exp.best_score,
            "best_metric": best_metric_key,
            "selection_cv_score": selection_provenance.get("selected_model_cv_score"),
            "best_model": {
                "model_name": best.model_name,
                "display_name": best.display_name,
                "name": best.display_name,
                "score": exp.best_score,
                "metric": best_metric_key,
                "params": best.best_params,
            } if best else None,
            "llm_explanation": llm_explanation,
            "provider_used": (llm_result or {}).get("provider_used", ""),
            "suggested_operations": accepted_ops,
            "rejected_operations": rejected_ops,
            "model_selection": model_selection_trace,
            "ensemble_candidates": ensemble_data,
            "fusion": fusion_summary,
            "is_ensemble_winner": is_ensemble_winner,
            "optimization_budget": optimization_budget,
            # Full, persisted record of how the winner was chosen (CV only).
            "selection": selection_provenance,
            "feature_engineering": json.loads(exp.feature_engineering_json)
            if exp.feature_engineering_json else {},
            "total_training_time": round(total_time, 2),
        }

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        exp.status = "failed"
        exp.error_message = str(e)
        db.commit()
        # Surface the real failure in the progress record too. Guarded because
        # the tracker only exists once the run has actually started.
        try:
            progress.finish(TP_FAILED, error=str(e))
            training_progress.unregister(exp_id)
        except (NameError, UnboundLocalError):
            pass
        raise HTTPException(500, f"Training failed: {str(e)}")


@router.get("/{exp_id}/progress")
async def get_training_progress(exp_id: int, db: Session = Depends(get_db)):
    """Real, observed training progress for an experiment.

    Returns which model is actually running, the stage it is in, the Optuna
    trial being evaluated, completed/failed counts and measured elapsed time.
    Deliberately reports no percentage and no ETA: the work is not linearly
    divisible, so either number would have to be invented.
    """
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return _exp_to_progress_response(exp, training_progress.get(exp_id))


@router.get("/{exp_id}/status")
async def get_status(exp_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return {"experiment_id": exp_id, "status": exp.status}


def _best_model_from_results(exp: Experiment) -> Optional[dict]:
    """Rebuild the best_model payload from persisted results and experiment columns."""
    try:
        models = json.loads(exp.results_json or "[]")
    except (TypeError, ValueError):
        models = []
    winner = next((m for m in models if m.get("is_best")), None)
    if not winner:
        return None
    _score, _metric = _best_score_and_metric(
        exp, winner.get("optimized_metrics") or winner.get("baseline_metrics")
    )
    return {
        "model_name": winner.get("model_name"),
        "display_name": winner.get("display_name"),
        "name": exp.best_model_name or winner.get("display_name"),
        "score": exp.best_score if exp.best_score is not None else _score,
        "metric": _metric,
        "params": winner.get("best_params") or {},
    }


@router.get("/{exp_id}/results")
async def get_results(exp_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")
    if not exp.results_json:
        raise HTTPException(400, "No results available. Run training first.")
    return {
        "experiment_id": exp_id,
        "problem_type": exp.problem_type,
        "primary_metric": exp.primary_metric,
        "mode": exp.mode,
        "models": _enrich_models(json.loads(exp.results_json), exp),
        "best_model_name": exp.best_model_name,
        "best_score": exp.best_score,
        # Selection used CV on the training split; best_score is the holdout
        # evaluation of the already-chosen winner. Both are reported separately.
        "selection": (json.loads(exp.model_selection_json).get("selection")
                      if exp.model_selection_json else None),
        # Same best_model shape as the /train response, rebuilt from persisted
        # state so the frontend banner keeps working after a refresh.
        "best_model": _best_model_from_results(exp),
        # Summaries persisted at train time: preprocessing columns actually used
        # and the feature-engineering mode (operations vs heuristics) including
        # any operation the registry rejected.
        "preprocessing_summary": json.loads(exp.preprocessing_json)
        if exp.preprocessing_json else {},
        "feature_engineering_summary": json.loads(exp.feature_engineering_json)
        if exp.feature_engineering_json else "N/A",
        "model_selection": json.loads(exp.model_selection_json)
        if exp.model_selection_json else None,
        "ensemble_candidates": (
            json.loads(exp.model_selection_json).get("fusion", {}).get("ensemble_candidates", [])
            if exp.model_selection_json else []
        ),
        "fusion": (
            json.loads(exp.model_selection_json).get("fusion")
            if exp.model_selection_json else None
        ),
        "is_ensemble_winner": (
            json.loads(exp.model_selection_json).get("fusion", {}).get("is_ensemble_winner", False)
            if exp.model_selection_json else False
        ),
        "pipeline_plan_provenance": (
            json.loads(exp.model_selection_json).get("pipeline_plan_provenance")
            if exp.model_selection_json else None
        ),
        "research_evaluation": _build_research_evaluation(exp, db),
    }


def _build_research_evaluation(exp: Experiment, db: Session) -> dict:
    """Build structured research evaluation and benchmarking metadata."""
    ds = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first() if exp.dataset_id else None
    ds_profile = json.loads(ds.profile_json) if ds and ds.profile_json else {}
    model_sel = json.loads(exp.model_selection_json) if exp.model_selection_json else {}
    fusion = model_sel.get("fusion") or {}
    selection = model_sel.get("selection") or {}
    budget = model_sel.get("optimization_budget") or {}
    llm_an = json.loads(exp.llm_analysis_json) if exp.llm_analysis_json else {}
    llm_res = llm_an.get("result") or llm_an
    fe_data = json.loads(exp.feature_engineering_json) if exp.feature_engineering_json else {}
    results_list = json.loads(exp.results_json) if exp.results_json else []

    is_ens_winner = bool(fusion.get("is_ensemble_winner", False))
    condition = "llm_assisted" if exp.llm_provider and "fallback" not in exp.llm_provider.lower() else "fallback"

    ds_name = (
        getattr(ds, "original_filename", None)
        or getattr(ds, "filename", None)
        or f"dataset_{exp.dataset_id}"
    ) if ds else f"dataset_{exp.dataset_id}"

    from app.services.benchmarker import get_git_revision, get_file_sha256
    ds_hash = get_file_sha256(ds.file_path) if ds and hasattr(ds, "file_path") and ds.file_path else None
    git_rev = get_git_revision()

    return {
        "dataset_name": ds_name,
        "dataset_hash": ds_hash,
        "dataset_rows": ds_profile.get("rows", getattr(ds, "rows", None)),
        "dataset_columns": ds_profile.get("columns", getattr(ds, "columns", None)),
        "problem_type": exp.problem_type,
        "primary_metric": exp.primary_metric,
        "condition": condition,
        "llm_provider": exp.llm_provider or "fallback",
        "reproducibility": {
            "seed": budget.get("seed", 42),
            "n_folds": budget.get("n_folds_used", exp.n_folds),
            "n_trials": budget.get("n_trials_used", exp.n_trials),
            "holdout_used_for_selection": False,
            "holdout_isolation": "100% Unseen (Evaluated post-selection only)",
            "selection_evidence": "CV on training split only",
            "dataset_hash": ds_hash,
            "git_commit": git_rev,
            "selection_metric": exp.primary_metric,
            "selection_scoring": selection.get("selection_scoring"),
            "selection_direction": selection.get("selection_direction", "maximize"),
            "raw_direction": selection.get("selection_raw_direction"),
        },
        "candidate_models_count": len(results_list),
        "ensemble_candidates_count": len(fusion.get("ensemble_candidates", [])),
        "is_ensemble_winner": is_ens_winner,
        "winner_name": exp.best_model_name,
        "winner_type": "ensemble" if is_ens_winner else "individual",
        "winner_cv_score": selection.get("selected_model_cv_score"),
        "winner_holdout_score": exp.best_score,
        "selection_evidence_source": selection.get("evidence_source", "cross_validation_training_split"),
        "provenance": {
            "dataset_name": ds_name,
            "dataset_hash": ds_hash,
            "problem_type": exp.problem_type,
            "target_column": exp.target_column,
            "seed": budget.get("seed", 42),
            "n_folds": budget.get("n_folds_used", exp.n_folds),
            "n_trials": budget.get("n_trials_used", exp.n_trials),
            "candidate_models": [r.get("model_name") for r in results_list],
            "selected_model": exp.best_model_name,
            "selection_metric": exp.primary_metric,
            "selection_cv_score": selection.get("selected_model_cv_score"),
            "final_holdout_score": exp.best_score,
            "ensemble_winner": is_ens_winner,
            "llm_provider": exp.llm_provider or "fallback",
            "git_commit": git_rev,
            "timestamp": str(exp.updated_at or exp.created_at),
        },
        "llm_advisory": {
            "recommended_models": [m.get("model_id") or m.get("model") for m in llm_res.get("model_recommendations", [])] if isinstance(llm_res.get("model_recommendations"), list) else [],
            "recommended_metric": llm_res.get("recommended_metric"),
            "proposed_operations": fe_data.get("operations_proposed", fe_data.get("proposed_operations", [])),
            "applied_operations": fe_data.get("operations_applied", fe_data.get("operations", [])),
            "rejected_operations": fe_data.get("rejected_operations", []),
            "is_advisory_only": True,
        },
        "pipeline_plan_provenance": model_sel.get("pipeline_plan_provenance"),
        "pipeline_plan": (model_sel.get("pipeline_plan_provenance") or {}).get("validated_plan"),
        "pipeline_source": (model_sel.get("pipeline_plan_provenance") or {}).get("source", "deterministic_fallback"),
    }


@router.get("/{exp_id}/pipeline-plan")
async def get_pipeline_plan(exp_id: int, db: Session = Depends(get_db)):
    """Get the structured AutoML pipeline plan and execution provenance for an experiment."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")

    # 1. If already trained, return execution provenance
    if exp.model_selection_json:
        try:
            ms = json.loads(exp.model_selection_json)
            if ms.get("pipeline_plan_provenance"):
                return ms["pipeline_plan_provenance"]
        except Exception:
            pass

    # 2. If analyzed, return analyzed plan
    if exp.llm_analysis_json:
        try:
            stored = json.loads(exp.llm_analysis_json)
            if isinstance(stored, dict) and "pipeline_plan" in stored:
                plan = stored["pipeline_plan"]
                return {
                    "source": plan.get("source", "llm"),
                    "provider_used": exp.llm_provider or "unknown",
                    "model_used": plan.get("model_used", ""),
                    "validation_status": plan.get("validation_status", "valid"),
                    "validation_errors": plan.get("validation_errors", []),
                    "validation_warnings": plan.get("validation_warnings", []),
                    "validated_plan": plan,
                    "raw_recommendation": plan,
                    "executed_pipeline": None,
                }
        except Exception:
            pass

    # 3. Generate deterministic plan on the fly
    ds = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
    if ds:
        profile = _resolve_dataset_profile(ds, db)
        from app.services.pipeline_planner import generate_deterministic_plan
        ctx = {
            "columns": [cp["name"] for cp in profile.get("column_profiles", [])],
            "column_profiles": profile.get("column_profiles", []),
            "target_column": exp.target_column,
            "problem_type": exp.problem_type,
            "shape": [profile.get("rows", 0), profile.get("columns", 0)],
        }
        plan = generate_deterministic_plan(ctx)
        return {
            "source": "deterministic_fallback",
            "provider_used": "Fallback (Deterministic)",
            "validation_status": "valid",
            "validation_errors": [],
            "validation_warnings": ["Generated on-demand using rule-based planner."],
            "validated_plan": plan.model_dump(),
            "raw_recommendation": None,
            "executed_pipeline": None,
        }
    raise HTTPException(404, "Dataset not found")


def _enrich_models(models: list[dict], exp: Experiment) -> list[dict]:
    """Attach the numbers the results table needs, without touching storage.

    Each model gains:

    ``selection_score`` / ``selection_cv_std`` / ``selection_cv_folds``
        the cross-validation evidence that selection actually ranked on,
        copied from the persisted selection provenance so CV and holdout can
        never be confused in the UI.
    ``holdout_primary`` / ``holdout_primary_key``
        the holdout value of the experiment's primary metric, resolved through
        the existing alias table (``neg_root_mean_squared_error`` ->
        ``rmse``). The key that was used is returned alongside the value, so a
        fallback to some other metric is visible rather than implied.

    The stored ``results_json`` is left exactly as it was; these are read-time
    additions computed from data the backend already holds.
    """
    selection = None
    if exp.model_selection_json:
        try:
            selection = json.loads(exp.model_selection_json).get("selection")
        except (TypeError, ValueError):
            selection = None
    cv_by_model = {}
    if selection:
        for cand in selection.get("candidates_considered", []):
            cv_by_model[cand.get("model_name")] = cand

    enriched = []
    for raw in models:
        model = dict(raw)
        cand = cv_by_model.get(model.get("model_name")) or {}
        model["selection_score"] = cand.get("cv_score")
        model["selection_cv_std"] = cand.get("cv_fold_std")
        model["selection_cv_folds"] = cand.get("cv_fold_count")

        holdout = model.get("optimized_metrics") or model.get("baseline_metrics")
        value, key = _best_score_and_metric(exp, holdout)
        model["holdout_primary"] = value
        model["holdout_primary_key"] = key
        enriched.append(model)
    return enriched


@router.get("/{exp_id}/models")
async def get_models(exp_id: int, db: Session = Depends(get_db)):
    runs = db.query(TrainingRun).filter(TrainingRun.experiment_id == exp_id).all()
    return [
        {
            "model_name": r.model_name,
            "display_name": r.display_name,
            "status": r.status,
            "baseline_metrics": json.loads(r.baseline_metrics_json) if r.baseline_metrics_json else None,
            "optimized_metrics": json.loads(r.optimized_metrics_json) if r.optimized_metrics_json else None,
            "best_params": json.loads(r.best_params_json) if r.best_params_json else None,
            # Real Optuna provenance persisted at train time.
            "optimization": json.loads(r.optimization_json) if r.optimization_json else None,
            "cv_scores": json.loads(r.cv_scores_json) if r.cv_scores_json else None,
            "training_time": r.training_time,
            "prediction_time": r.prediction_time,
        }
        for r in runs
    ]


@router.get("/{exp_id}/explainability")
async def get_explainability(exp_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp or not exp.explainability_json:
        raise HTTPException(404, "Explainability data not found")
    return json.loads(exp.explainability_json)


@router.post("/{exp_id}/predict")
async def predict(exp_id: int, features: dict, db: Session = Depends(get_db)):
    """Make a single prediction."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp or exp.status != "completed":
        raise HTTPException(400, "Experiment not completed")

    # Validate the supplied values against the experiment's input schema before
    # touching the model: anything the user did not provide must be reported,
    # never silently imputed into a confident-looking prediction.
    schema_fields = _prediction_schema(exp, db, _experiment_cache.get(exp_id))
    try:
        cleaned_inputs = validate_prediction_inputs(features, schema_fields)
    except InputValidationError as e:
        logger.warning(f"Rejected prediction input for experiment {exp_id}: {e.message}")
        raise HTTPException(422, e.message)

    cache = _experiment_cache.get(exp_id)
    if not cache or cache["model"] is None:
        # Try to restore from disk
        model_path = settings.models_path / f"model_{exp_id}.joblib"
        meta_path = settings.models_path / f"model_{exp_id}_metadata.json"
        if not model_path.exists():
            raise HTTPException(400, "Model not found on disk. Please re-run training.")
        if not meta_path.exists():
            raise HTTPException(400, "Model metadata not found. Please re-run training.")
        try:
            loaded_model = joblib.load(str(model_path))
            with open(str(meta_path)) as _mf:
                meta = json.load(_mf)
            # Prefer the preprocessor persisted at training time (it embeds the
            # feature engineer fitted on the training split); only fall back to a
            # full refit when that artifact is missing (models from before it).
            _ds = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
            _df = load_dataframe(_ds.file_path)
            _profiler = DatasetProfiler()
            _profile = _profiler.profile(_df)
            _prep_path = settings.models_path / f"model_{exp_id}_preprocessor.joblib"
            _preprocessor = None
            if _prep_path.exists():
                _preprocessor = joblib.load(str(_prep_path))
                _feature_names = list(_preprocessor.feature_names_out)
            if _preprocessor is None:
                _fe = FeatureEngineer()
                _df_eng, _, _ = _fe.apply(_df, _profile["column_profiles"], exp.target_column)
                _preprocessor = PreprocessingEngine()
                _, _feature_names = _preprocessor.build_and_fit(
                    _df_eng, exp.target_column, _profile["column_profiles"], exp.problem_type,
                    extra_numeric=_fe.new_features, feature_engineer=_fe,
                )
            _label_map = None
            _y = _df[exp.target_column]
            if exp.problem_type == "classification" and _y.dtype == "object":
                from sklearn.preprocessing import LabelEncoder
                _le = LabelEncoder()
                _le.fit_transform(_y)
                _label_map = {i: c for i, c in enumerate(_le.classes_)}
            _feature_cols = [cp["name"] for cp in _profile["column_profiles"] if cp["name"] != exp.target_column]
            _experiment_cache[exp_id] = {
                "model": loaded_model,
                "preprocessor": _preprocessor,
                "feature_names": _feature_cols,
                "feature_names_transformed": _feature_names,
                "target_column": exp.target_column,
                "problem_type": exp.problem_type,
                "label_map": _label_map,
            }
            cache = _experiment_cache[exp_id]
            logger.info(f"Restored experiment {exp_id} from disk.")
        except Exception as _e:
            logger.exception(f"Failed to restore model from disk: {_e}")
            raise HTTPException(400, f"Model restore failed: {str(_e)}. Please re-run training.")

    predictor = Predictor(
        model=cache["model"],
        preprocessor=cache["preprocessor"],
        feature_names=cache["feature_names"],
        target_column=cache["target_column"],
        problem_type=cache["problem_type"],
    )
    # Reuse the schema on subsequent predictions without rereading the dataset.
    cache["input_schema"] = schema_fields

    result = predictor.predict_single(cleaned_inputs)
    if "error" in result:
        raise HTTPException(400, result["error"])

    # Map label back if needed
    if cache.get("label_map") and "prediction" in result:
        pred_val = result["prediction"]
        result["prediction"] = cache["label_map"].get(pred_val, pred_val)

    result["model_name"] = exp.best_model_name

    # Keep a durable record of every single prediction: the predictions table
    # exists for this history, and it used to stay permanently empty.
    _record_prediction(db, exp_id, "single", cleaned_inputs, result)
    return result


def _record_prediction(db: Session, exp_id: int, prediction_type: str,
                       inputs: dict, result: dict,
                       batch_file_path: Optional[str] = None) -> None:
    """Persist one prediction run to the ``predictions`` table.

    Best effort by design: a failure to write history must never turn a
    successful prediction into an error, so it is logged and swallowed.
    ``result`` is stored as returned by the predictor (compact: batch responses
    carry a path, a count and a ≤10-row preview, not the full output file).
    """
    try:
        db.add(Prediction(
            experiment_id=exp_id,
            prediction_type=prediction_type,
            input_json=json.dumps(inputs, default=str),
            result_json=json.dumps(result, default=str),
            batch_file_path=batch_file_path,
        ))
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.exception(
            f"Could not record {prediction_type} prediction for experiment {exp_id}"
        )


@router.post("/{exp_id}/batch-predict")
async def batch_predict(exp_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Batch prediction from uploaded CSV."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp or exp.status != "completed":
        raise HTTPException(400, "Experiment not completed")

    cache = _experiment_cache.get(exp_id)
    if not cache or cache["model"] is None:
        # Restore from disk
        exp2 = db.query(Experiment).filter(Experiment.id == exp_id).first()
        model_path = settings.models_path / f"model_{exp_id}.joblib"
        if not model_path.exists():
            raise HTTPException(400, "Model not found on disk. Please re-run training.")
        try:
            loaded_model = joblib.load(str(model_path))
            _ds2 = db.query(Dataset).filter(Dataset.id == exp2.dataset_id).first()
            _df2 = load_dataframe(_ds2.file_path)
            _profiler2 = DatasetProfiler()
            _profile2 = _profiler2.profile(_df2)
            _prep_path2 = settings.models_path / f"model_{exp_id}_preprocessor.joblib"
            _preprocessor2 = None
            if _prep_path2.exists():
                _preprocessor2 = joblib.load(str(_prep_path2))
                _fn2 = list(_preprocessor2.feature_names_out)
            if _preprocessor2 is None:
                _fe2 = FeatureEngineer()
                _df_eng2, _, _ = _fe2.apply(_df2, _profile2["column_profiles"], exp2.target_column)
                _preprocessor2 = PreprocessingEngine()
                _, _fn2 = _preprocessor2.build_and_fit(
                    _df_eng2, exp2.target_column, _profile2["column_profiles"], exp2.problem_type,
                    extra_numeric=_fe2.new_features, feature_engineer=_fe2,
                )
            _fc2 = [cp["name"] for cp in _profile2["column_profiles"] if cp["name"] != exp2.target_column]
            cache = {
                "model": loaded_model, "preprocessor": _preprocessor2,
                "feature_names": _fc2, "feature_names_transformed": _fn2,
                "target_column": exp2.target_column, "problem_type": exp2.problem_type,
                "label_map": None,
            }
            _experiment_cache[exp_id] = cache
        except Exception as _e:
            raise HTTPException(400, f"Model restore failed: {str(_e)}")

    limit_label = (
        f"{settings.MAX_DATASET_SIZE_MB // 1024} GB"
        if settings.MAX_DATASET_SIZE_MB >= 1024
        else f"{settings.MAX_DATASET_SIZE_MB} MB"
    )
    if file.size and file.size > settings.max_dataset_size_bytes:
        raise HTTPException(400, f"Batch file exceeds the maximum allowed size of {limit_label}.")

    # Save the uploaded file under a unique name
    temp_path = settings.predictions_path / (
        f"batch_input_{exp_id}_{uuid.uuid4().hex[:12]}.csv"
    )
    chunk_size = 1024 * 1024
    total_size = 0
    try:
        with open(temp_path, "wb") as f:
            while chunk := await file.read(chunk_size):
                total_size += len(chunk)
                if total_size > settings.max_dataset_size_bytes:
                    raise HTTPException(400, f"Batch file exceeds the maximum allowed size of {limit_label}.")
                f.write(chunk)
    except HTTPException:
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to upload batch file: {str(e)}")

    # Same contract as single prediction: a batch file must carry the columns the
    # model consumes; missing ones are reported instead of being imputed away.
    required_columns = [fld["name"] for fld in _prediction_schema(exp, db, cache)
                        if fld.get("required", True)]
    try:
        file_columns = set(load_dataframe(temp_path, nrows=0).columns)
    except Exception as e:
        raise HTTPException(400, f"Cannot read batch file: {str(e)}")
    missing_columns = [c for c in required_columns if c not in file_columns]
    if missing_columns:
        message = ("Prediction input validation failed: batch file is missing required "
                   "input(s): " + ", ".join(missing_columns))
        logger.warning(f"Rejected batch input for experiment {exp_id}: {message}")
        raise HTTPException(422, message)

    predictor = Predictor(
        model=cache["model"],
        preprocessor=cache["preprocessor"],
        feature_names=cache["feature_names"],
        target_column=cache["target_column"],
        problem_type=cache["problem_type"],
    )

    result = predictor.predict_batch(str(temp_path), str(settings.storage_path))
    if "error" in result:
        raise HTTPException(400, result["error"])

    # Record the batch run too, including the unique input path it consumed, so
    # the predictions table doubles as an audit trail of what was uploaded.
    _record_prediction(
        db, exp_id, "batch",
        {
            "file_name": file.filename,
            "input_path": str(temp_path),
            "columns": sorted(file_columns),
            "num_predictions": result.get("num_predictions"),
        },
        result,
        batch_file_path=str(temp_path),
    )
    return result


@router.get("/{exp_id}/report")
async def generate_report(exp_id: int, db: Session = Depends(get_db)):
    """Generate experiment report."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")

    ds = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
    profile = json.loads(ds.profile_json) if ds.profile_json else {}

    report_data = {
        "experiment_id": exp_id,
        "timestamp": str(exp.created_at),
        "dataset_info": {
            "rows": profile.get("rows", ds.rows),
            "columns": profile.get("columns", ds.columns),
            "memory_usage_mb": profile.get("memory_usage_mb", "N/A"),
            "total_missing": profile.get("total_missing", 0),
            "total_missing_percentage": profile.get("total_missing_percentage", 0),
            "duplicate_rows": profile.get("duplicate_rows", 0),
        },
        "target": exp.target_column,
        "problem_type": exp.problem_type,
        "llm_analysis": json.loads(exp.llm_analysis_json).get("result", {}).get("reasoning", "N/A") if exp.llm_analysis_json else "N/A",
        "preprocessing_summary": json.loads(exp.preprocessing_json) if exp.preprocessing_json else {},
        "feature_engineering_summary": json.loads(exp.feature_engineering_json) if exp.feature_engineering_json else "N/A",
        "models_results": json.loads(exp.results_json) if exp.results_json else [],
        "best_model": next(
            (m for m in json.loads(exp.results_json) if m.get("is_best")) if exp.results_json else [],
            {}
        ),
        "explainability": json.loads(exp.explainability_json) if exp.explainability_json else {},
        # The report labels its score column with this name; without it the
        # generator had to guess which metric it was showing.
        "primary_metric": exp.primary_metric,
        "metrics": {"primary_metric": exp.primary_metric},
        "selection": json.loads(exp.model_selection_json).get("selection") if exp.model_selection_json else None,
        "model_selection": json.loads(exp.model_selection_json) if exp.model_selection_json else None,
        "research_evaluation": _build_research_evaluation(exp, db),
    }

    reporter = ReportGenerator()
    html = reporter.generate_html_report(report_data)
    path = reporter.save_report(html, str(exp_id), str(settings.storage_path))

    exp.report_path = path
    db.commit()

    return {"report_path": path, "report_url": f"/reports/report_{exp_id}.html"}


@router.get("/{exp_id}/download-model")
async def download_model(exp_id: int):
    """Download trained model file."""
    model_path = settings.models_path / f"model_{exp_id}.joblib"
    if not model_path.exists():
        raise HTTPException(404, "Model not found")
    return FileResponse(
        str(model_path),
        media_type="application/octet-stream",
        filename=f"automl_lens_model_{exp_id}.joblib"
    )


@router.get("/{exp_id}/download-metadata")
async def download_metadata(exp_id: int):
    """Download model metadata."""
    meta_path = settings.models_path / f"model_{exp_id}_metadata.json"
    if not meta_path.exists():
        raise HTTPException(404, "Metadata not found")
    return FileResponse(str(meta_path), media_type="application/json",
                        filename=f"model_{exp_id}_metadata.json")


@router.get("/{exp_id}/input-schema")
async def get_input_schema(exp_id: int, db: Session = Depends(get_db)):
    """Get input field schema for prediction form. Works after server restart."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp or exp.status != "completed":
        raise HTTPException(400, "Experiment not completed")
    ds = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found for this experiment")

    # Built from persisted experiment/dataset state: the profile is generated and
    # cached on demand, so this no longer depends on the profile page having been
    # opened first, and the field list matches what the fitted pipeline consumes.
    return {
        "fields": _prediction_schema(exp, db, _experiment_cache.get(exp_id)),
        "target_column": exp.target_column,
    }


def _exp_to_dict(exp: Experiment) -> dict:
    return {
        "id": exp.id,
        "name": exp.name,
        "dataset_id": exp.dataset_id,
        "target_column": exp.target_column,
        "problem_type": exp.problem_type,
        "primary_metric": exp.primary_metric,
        "mode": exp.mode,
        "n_folds": exp.n_folds,
        "n_trials": exp.n_trials,
        "status": exp.status,
        "llm_provider": exp.llm_provider,
        "best_model_name": exp.best_model_name,
        "best_score": exp.best_score,
        "model_selection": json.loads(exp.model_selection_json)
        if exp.model_selection_json else None,
        # Observable state, so the UI can hydrate from what is stored instead of
        # re-running the analysis/LLM on every page load.
        "has_analysis": bool(exp.llm_analysis_json),
        "has_results": bool(exp.results_json),
        "report_path": exp.report_path,
        "llm_analysis": _safe_json(exp.llm_analysis_json),
        "error_message": exp.error_message,
        "created_at": str(exp.created_at),
        "updated_at": str(exp.updated_at),
    }



