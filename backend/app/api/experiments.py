"""Experiment API routes - the core AutoML pipeline."""
import json
import logging
import time
import traceback
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
from app.core.database import get_db, Dataset, Experiment, TrainingRun
from app.services.profiler import DatasetProfiler
from app.services.preprocessor import PreprocessingEngine
from app.services.feature_engineer import FeatureEngineer
from app.services.model_registry import ModelRegistry
from app.services.trainer import TrainingManager
from app.services.evaluator import Evaluator, get_scoring_string
from app.services.explainer import Explainer
from app.services.predictor import Predictor
from app.services.reporter import ReportGenerator
from app.models.schemas import ExperimentCreateRequest

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory store for trained objects (in production, use proper persistence)
_experiment_cache = {}


def _get_llm_manager():
    from app.main import llm_manager
    return llm_manager


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


@router.get("/{exp_id}")
async def get_experiment(exp_id: int, db: Session = Depends(get_db)):
    """Get experiment details."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return _exp_to_dict(exp)


@router.post("/{exp_id}/analyze")
async def analyze_experiment(exp_id: int, db: Session = Depends(get_db)):
    """Run LLM analysis on the dataset."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")

    ds = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
    df = pd.read_csv(ds.file_path)

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
    }

    # Run LLM analysis
    llm = _get_llm_manager()
    analysis = await llm.analyze_dataset(dataset_context)

    exp.llm_analysis_json = json.dumps(analysis, default=str)
    exp.llm_provider = analysis.get("provider_used", "unknown")
    exp.status = "analyzed"
    db.commit()

    return analysis


@router.post("/{exp_id}/train")
async def train_experiment(exp_id: int, fast_demo: bool = False, db: Session = Depends(get_db)):
    """Run the full training pipeline."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")

    ds = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
    df = pd.read_csv(ds.file_path)

    try:
        exp.status = "training"
        db.commit()

        # Profile
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        # Feature engineering
        fe = FeatureEngineer()
        df_eng, new_feats, fe_transforms = fe.apply(df, profile["column_profiles"], exp.target_column)

        # Preprocessing
        preprocessor = PreprocessingEngine()
        y = df_eng[exp.target_column].copy()

        # Encode target if needed
        label_map = None
        if exp.problem_type == "classification" and y.dtype == "object":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=exp.target_column)
            label_map = {i: c for i, c in enumerate(le.classes_)}

        X_transformed, feature_names = preprocessor.build_and_fit(
            df_eng, exp.target_column, profile["column_profiles"], exp.problem_type
        )

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42,
            stratify=y if exp.problem_type == "classification" else None
        )

        # Select models
        registry = ModelRegistry()
        n_folds = 3 if fast_demo else exp.n_folds
        n_trials = 5 if fast_demo else exp.n_trials

        # Get model names from LLM analysis or registry
        model_names = None
        if exp.llm_analysis_json:
            analysis = json.loads(exp.llm_analysis_json)
            result = analysis.get("result", {})
            model_names = result.get("candidate_models", None)

        if fast_demo:
            model_names = registry.get_fast_demo_models(exp.problem_type)

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
        )

        results = trainer.train_all()

        # Select best model
        # For best model selection, use the user-facing metric name
        selection_metric = exp.primary_metric
        if selection_metric in ("neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error"):
            # Map to the stored metric name
            metric_map = {
                "neg_mean_squared_error": "mse",
                "neg_root_mean_squared_error": "rmse",
                "neg_mean_absolute_error": "mae",
            }
            selection_metric = metric_map.get(selection_metric, selection_metric)

        best = trainer.select_best_model(results, selection_metric)

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
                "random_seed": 42,
                "timestamp": str(exp.created_at),
                "library_versions": {
                    "scikit-learn": sklearn.__version__,
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                },
            }
            meta_path = str(settings.models_path / f"model_{exp_id}_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)

        # LLM explanation of results
        llm_explanation = None
        llm = _get_llm_manager()
        if best:
            try:
                results_context = {
                    "best_model": best.model_name,
                    "best_display_name": best.display_name,
                    "best_score": float(list((best.optimized_metrics or best.baseline_metrics).values())[0]) if (best.optimized_metrics or best.baseline_metrics) else None,
                    "metric_name": exp.primary_metric,
                    "problem_type": exp.problem_type,
                    "best_params": best.best_params,
                    "n_trials": n_trials,
                    "all_results": [
                        {
                            "model_name": r.model_name,
                            "display_name": r.display_name,
                            "score": float(list((r.optimized_metrics or r.baseline_metrics).values())[0]) if (r.optimized_metrics or r.baseline_metrics) else None,
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
                training_time=r.training_time,
                prediction_time=r.prediction_time,
                model_path=model_path if r.model_name == (best.model_name if best else None) else None,
            )
            db.add(tr)

        exp.results_json = json.dumps(results_data, default=str)
        exp.best_model_name = best.display_name if best else None
        exp.best_score = float(list((best.optimized_metrics or best.baseline_metrics).values())[0]) if best and (best.optimized_metrics or best.baseline_metrics) else None
        exp.preprocessing_json = json.dumps(preprocessor.get_summary(), default=str)
        exp.feature_engineering_json = json.dumps(fe.get_summary(), default=str)
        if explain_result:
            exp.explainability_json = json.dumps({
                "method_used": explain_result.method_used,
                "feature_importance": explain_result.feature_importance[:20],
                "top_features": explain_result.top_features,
                "explanation_text": explain_result.explanation_text,
            }, default=str)
        exp.status = "completed"
        db.commit()

        # Cache experiment objects for predictions
        _experiment_cache[exp_id] = {
            "model": best.trained_model if best else None,
            "preprocessor": preprocessor,
            "feature_names": [cp["name"] for cp in profile["column_profiles"] if cp["name"] != exp.target_column],
            "feature_names_transformed": feature_names,
            "target_column": exp.target_column,
            "problem_type": exp.problem_type,
            "label_map": label_map,
            "df_original": df,
        }

        total_time = sum(r.training_time for r in results)
        return {
            "experiment_id": exp_id,
            "status": "completed",
            "problem_type": exp.problem_type,
            "primary_metric": exp.primary_metric,
            "mode": exp.mode,
            "models": results_data,
            "best_model": {
                "model_name": best.model_name,
                "display_name": best.display_name,
                "score": exp.best_score,
                "params": best.best_params,
            } if best else None,
            "llm_explanation": llm_explanation,
            "total_training_time": round(total_time, 2),
        }

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        exp.status = "failed"
        exp.error_message = str(e)
        db.commit()
        raise HTTPException(500, f"Training failed: {str(e)}")


@router.get("/{exp_id}/status")
async def get_status(exp_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return {"experiment_id": exp_id, "status": exp.status}


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
        "models": json.loads(exp.results_json),
        "best_model_name": exp.best_model_name,
        "best_score": exp.best_score,
    }


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

    cache = _experiment_cache.get(exp_id)
    if not cache or cache["model"] is None:
        # Try loading model from disk
        model_path = settings.models_path / f"model_{exp_id}.joblib"
        if not model_path.exists():
            raise HTTPException(400, "Model not available. Re-run training.")
        raise HTTPException(400, "Model cache expired. Please re-run training for prediction support.")

    predictor = Predictor(
        model=cache["model"],
        preprocessor=cache["preprocessor"],
        feature_names=cache["feature_names"],
        target_column=cache["target_column"],
        problem_type=cache["problem_type"],
    )

    result = predictor.predict_single(features)
    if "error" in result:
        raise HTTPException(400, result["error"])

    # Map label back if needed
    if cache.get("label_map") and "prediction" in result:
        pred_val = result["prediction"]
        result["prediction"] = cache["label_map"].get(pred_val, pred_val)

    result["model_name"] = exp.best_model_name
    return result


@router.post("/{exp_id}/batch-predict")
async def batch_predict(exp_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Batch prediction from uploaded CSV."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp or exp.status != "completed":
        raise HTTPException(400, "Experiment not completed")

    cache = _experiment_cache.get(exp_id)
    if not cache or cache["model"] is None:
        raise HTTPException(400, "Model cache expired. Please re-run training.")

    # Save uploaded file
    content = await file.read()
    temp_path = settings.predictions_path / f"batch_input_{exp_id}.csv"
    with open(temp_path, "wb") as f:
        f.write(content)

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
        "metrics": {},
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
    """Get input field schema for prediction form."""
    cache = _experiment_cache.get(exp_id)
    if not cache:
        raise HTTPException(400, "Model cache not available. Re-run training.")

    predictor = Predictor(
        model=cache["model"],
        preprocessor=cache["preprocessor"],
        feature_names=cache["feature_names"],
        target_column=cache["target_column"],
        problem_type=cache["problem_type"],
    )
    schema = predictor.get_input_schema(cache.get("df_original"))
    return {"fields": schema}


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
        "error_message": exp.error_message,
        "created_at": str(exp.created_at),
        "updated_at": str(exp.updated_at),
    }
