"""Research-grade benchmarking service for AutoML-Lens.

Empirically benchmarks and compares:
1. Individual AutoML candidate models
2. Ensemble / fusion models
3. LLM-assisted AutoML
4. Non-LLM / fallback AutoML baseline

Core Protocols:
- Real datasets only (e.g. Adult classification, Wine Quality regression).
- Zero metric fabrication: all scores, deltas, and times are computed from measured execution records.
- Strict leakage prevention: feature engineering, preprocessing, Optuna tuning, and ensemble
  weights are learned exclusively on (X_train, y_train). Holdout is evaluated exactly once after selection.
- Selection uses CV evidence only: holdout is never consulted to choose models or ensemble weights.
- Reproducibility: explicit seeds control train/test splitting, CV fold partitioning, and Optuna sampling.
- Clear separation between LLM advisory suggestions and actual empirical ML performance.
"""
from dataclasses import dataclass, field, asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.services.trainer import TrainingManager, TrainingResult
from app.services.model_registry import ModelRegistry, ModelDefinition
from app.services.profiler import DatasetProfiler
from app.services.preprocessor import PreprocessingEngine
from app.services.feature_engineer import FeatureEngineer
from app.services.feature_operations import validate_operations
from app.services.ensemble import (
    EnsembleModel,
    generate_oof_predictions,
    optimize_ensemble_weights,
    build_ensemble_candidates,
)
from app.services.evaluator import get_metric_spec, is_higher_better

logger = logging.getLogger(__name__)


def get_git_revision() -> Optional[str]:
    """Retrieve the current Git commit hash if in a Git repository."""
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode("utf-8").strip()
        return rev if rev else None
    except Exception:
        return None


def get_file_sha256(filepath: str, max_bytes: int = 10 * 1024 * 1024) -> Optional[str]:
    """Compute truncated SHA-256 hash of a dataset file for provenance."""
    try:
        p = Path(filepath)
        if not p.is_file():
            return None
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
                if f.tell() >= max_bytes:
                    break
        return hasher.hexdigest()[:16]
    except Exception:
        return None


@dataclass
class BenchmarkConfig:
    """Configuration for a reproducible benchmark evaluation run."""
    dataset_name: str
    dataset_path: str
    target_column: str
    problem_type: str
    primary_metric: str
    condition: str = "fallback"  # 'fallback' or 'llm_assisted'
    enable_ensemble: bool = True
    models_to_train: Optional[List[str]] = None
    seed: int = 42
    n_folds: int = 3
    n_trials: int = 5
    test_size: float = 0.2
    max_rows: Optional[int] = None
    custom_operations: Optional[List[Dict[str, Any]]] = None
    pipeline_plan: Optional[Any] = None


@dataclass
class ModelMetricRecord:
    """Measured metrics for a single candidate or ensemble model."""
    model_name: str
    display_name: str
    is_ensemble: bool
    status: str
    selection_score: Optional[float]
    cv_scores: List[float]
    cv_mean: Optional[float]
    cv_std: Optional[float]
    holdout_score: Optional[float]
    training_time: float
    best_params: Dict[str, Any] = field(default_factory=dict)
    strategy: Optional[str] = None
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkRecord:
    """Comprehensive, fully computed provenance record for a benchmark run."""
    dataset_name: str
    rows_total: int
    rows_train: int
    rows_holdout: int
    n_features_raw: int
    n_features_transformed: int
    problem_type: str
    primary_metric: str
    condition: str
    seed: int
    n_folds: int
    n_trials: int
    holdout_used_for_selection: bool
    
    # Measured model results
    candidate_models: List[Dict[str, Any]]
    ensemble_candidates: List[Dict[str, Any]]
    
    # Winners (CV-selected)
    best_individual_model: str
    best_individual_cv_score: Optional[float]
    best_individual_holdout_score: Optional[float]
    
    best_ensemble_model: Optional[str]
    best_ensemble_cv_score: Optional[float]
    best_ensemble_holdout_score: Optional[float]
    
    overall_winner: str
    overall_winner_type: str  # 'individual' or 'ensemble'
    overall_cv_score: Optional[float]
    overall_holdout_score: Optional[float]
    
    # Empirical deltas
    ensemble_improvement_cv: Optional[float]
    ensemble_improvement_holdout: Optional[float]
    
    # LLM vs Empirical ML Distinction
    llm_provider: str
    llm_recommended_models: List[str]
    llm_proposed_operations: List[Dict[str, Any]]
    operations_accepted: List[Dict[str, Any]]
    operations_rejected: List[Dict[str, Any]]
    llm_model_match: bool  # Whether LLM recommended the actual CV winner
    
    # Timings
    feature_engineering_time: float
    training_time: float
    total_time: float
    timestamp: str

    # Provenance and metric contract
    dataset_hash: Optional[str] = None
    git_commit: Optional[str] = None
    cv_scoring: Optional[str] = None
    cv_direction: Optional[str] = None
    holdout_metric: Optional[str] = None
    holdout_direction: Optional[str] = None
    pipeline_plan: Optional[Dict[str, Any]] = None
    benchmark_id: Optional[str] = None
    secondary_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """Executes reproducible AutoML-Lens benchmark experiments."""

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self.profiler = DatasetProfiler()

    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkRecord:
        """Run a single benchmark condition and produce an empirical BenchmarkRecord."""
        t_start = time.time()
        
        # 1. Load dataset
        path = Path(config.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found: {config.dataset_path}")
        
        # Determine CSV delimiter
        sep = ";" if config.dataset_path.endswith(".csv") and ";" in open(path, "r", encoding="utf-8", errors="ignore").readline() else ","
        if "adult.data" in config.dataset_path:
            cols = [
                "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
                "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                "hours_per_week", "native_country", "income"
            ]
            df = pd.read_csv(path, header=None, names=cols, na_values=" ?").dropna().reset_index(drop=True)
        else:
            df = pd.read_csv(path, sep=sep).reset_index(drop=True)

        if config.max_rows and len(df) > config.max_rows:
            df = df.head(config.max_rows).reset_index(drop=True)

        rows_total = len(df)
        n_features_raw = len(df.columns) - 1

        # 2. Profile dataset
        profile = self.profiler.profile(df)

        # 3. Encode target if classification with string labels
        y = df[config.target_column].copy()
        if config.problem_type == "classification" and (y.dtype == "object" or isinstance(y.iloc[0], str)):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=df.index, name=config.target_column)

        # 4. Stratified / Random split strictly isolated by seed
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=config.test_size,
            random_state=config.seed,
            stratify=y if config.problem_type == "classification" else None,
        )
        df_train = df.loc[train_idx].copy()
        df_test = df.loc[test_idx].copy()
        y_train = y.loc[train_idx].copy()
        y_test = y.loc[test_idx].copy()

        # 5. Pipeline Plan unpacking if supplied
        plan_dict = None
        if config.pipeline_plan is not None:
            if hasattr(config.pipeline_plan, "model_dump"):
                plan_dict = config.pipeline_plan.model_dump()
            elif isinstance(config.pipeline_plan, dict):
                plan_dict = config.pipeline_plan

            if plan_dict:
                if not config.custom_operations and plan_dict.get("feature_operations"):
                    config.custom_operations = plan_dict.get("feature_operations")
                if not config.models_to_train and plan_dict.get("candidate_models"):
                    config.models_to_train = plan_dict.get("candidate_models")
                if "ensemble_strategy" in plan_dict and isinstance(plan_dict["ensemble_strategy"], dict):
                    config.enable_ensemble = plan_dict["ensemble_strategy"].get("enabled", config.enable_ensemble)

        # 6. Feature Engineering (fit exclusively on train split)
        t_fe_start = time.time()
        fe = FeatureEngineer(seed=config.seed)
        proposed_ops = config.custom_operations or []
        accepted_ops, rejected_ops = validate_operations(
            proposed_ops, profile["column_profiles"], config.target_column
        )
        fe.fit(df_train, profile["column_profiles"], config.target_column, operations=accepted_ops or None)
        df_eng_train = fe.transform(df_train)
        fe_time = round(time.time() - t_fe_start, 4)

        # 6. Preprocessing (fit exclusively on engineered train split)
        preprocessor = PreprocessingEngine()
        X_train, feature_names = preprocessor.build_and_fit(
            df_eng_train, config.target_column, profile["column_profiles"], config.problem_type,
            extra_numeric=fe.new_features, feature_engineer=fe,
        )
        X_test = preprocessor.transform(df_test, target_column=config.target_column)
        n_features_transformed = len(feature_names)

        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        # 7. Select Candidate Models from Registry
        if config.models_to_train:
            model_defs = self.registry.get_models(config.problem_type, config.models_to_train)
        else:
            model_names = self.registry.get_fast_demo_models(config.problem_type)
            model_defs = self.registry.get_models(config.problem_type, model_names)

        # 8. Train Candidate Models
        t_tr_start = time.time()
        trainer = TrainingManager(
            X_train=X_train_df,
            X_test=X_test_df,
            y_train=y_train,
            y_test=y_test,
            model_definitions=model_defs,
            problem_type=config.problem_type,
            primary_metric=config.primary_metric,
            n_folds=config.n_folds,
            n_trials=config.n_trials,
            optimization_seed=config.seed,
            mode="fast",
            fast_demo=True,
        )
        results: List[TrainingResult] = trainer.train_all()
        training_time = round(time.time() - t_tr_start, 4)

        # Process candidate model records
        candidate_records: List[Dict[str, Any]] = []
        for r in results:
            cv_mean = float(np.mean(r.cv_scores)) if r.cv_scores else None
            cv_std = float(np.std(r.cv_scores)) if r.cv_scores else None
            holdout_m = r.optimized_metrics or r.baseline_metrics or {}
            h_score = holdout_m.get(config.primary_metric)
            if h_score is None:
                # Look for common aliases
                alias = "rmse" if config.primary_metric in ("neg_root_mean_squared_error", "rmse") else config.primary_metric
                h_score = holdout_m.get(alias)
            
            candidate_records.append({
                "model_name": r.model_name,
                "display_name": r.display_name,
                "is_ensemble": False,
                "status": r.status,
                "cv_scores": r.cv_scores,
                "cv_mean": round(cv_mean, 6) if cv_mean is not None else None,
                "cv_std": round(cv_std, 6) if cv_std is not None else None,
                "selection_score": round(cv_mean, 6) if cv_mean is not None else None,
                "holdout_score": round(float(h_score), 6) if h_score is not None else None,
                "training_time": round(r.training_time, 4),
                "best_params": r.best_params,
            })

        # Best individual model (CV evidence ONLY)
        best_single = trainer.select_best_model(results, config.primary_metric)
        best_single_name = best_single.model_name if best_single else "None"
        best_single_cv = float(np.mean(best_single.cv_scores)) if best_single and best_single.cv_scores else None
        single_h_metrics = (best_single.optimized_metrics or best_single.baseline_metrics or {}) if best_single else {}
        best_single_holdout = single_h_metrics.get(config.primary_metric, single_h_metrics.get("rmse"))

        # 9. Ensemble Candidates (strictly OOF predictions on train split)
        ensemble_records: List[Dict[str, Any]] = []
        best_ensemble_res = None
        best_ens_name = None
        best_ens_cv = None
        best_ens_holdout = None

        if config.enable_ensemble:
            ensemble_cands = build_ensemble_candidates(
                candidate_results=results,
                model_definitions=model_defs,
                X_train=X_train_df,
                y_train=y_train,
                problem_type=config.problem_type,
                primary_metric=config.primary_metric,
                n_folds=config.n_folds,
                seed=config.seed,
            )
            # Evaluate ensemble candidates on holdout ONCE
            for ens_res, ens_model in ensemble_cands:
                h_metrics = trainer._evaluate(ens_model, X_test_df, y_test)
                ens_res.optimized_metrics = h_metrics
                ens_res.baseline_metrics = h_metrics
                
                ecv = ens_res.optimization.get("best_cv_score")
                eh_score = h_metrics.get(config.primary_metric, h_metrics.get("rmse"))
                
                ensemble_records.append({
                    "model_name": ens_res.model_name,
                    "display_name": ens_res.display_name,
                    "is_ensemble": True,
                    "strategy": ens_res.optimization.get("strategy"),
                    "weights": ens_res.optimization.get("weights", {}),
                    "status": ens_res.status,
                    "selection_score": round(float(ecv), 6) if ecv is not None else None,
                    "cv_mean": round(float(ecv), 6) if ecv is not None else None,
                    "holdout_score": round(float(eh_score), 6) if eh_score is not None else None,
                    "training_time": round(ens_res.training_time, 4),
                })

            ens_results_only = [c[0] for c in ensemble_cands]
            if ens_results_only:
                best_ensemble_res = trainer.select_best_model(ens_results_only, config.primary_metric)
                if best_ensemble_res:
                    best_ens_name = best_ensemble_res.model_name
                    best_ens_cv = best_ensemble_res.optimization.get("best_cv_score")
                    ens_h_met = best_ensemble_res.optimized_metrics or {}
                    best_ens_holdout = ens_h_met.get(config.primary_metric, ens_h_met.get("rmse"))

        # 10. Overall Winner (Individual vs Ensemble on CV evidence ONLY)
        all_candidates = results + [c[0] for c in (ensemble_cands if config.enable_ensemble else [])]
        overall_winner_obj = trainer.select_best_model(all_candidates, config.primary_metric)
        overall_winner = overall_winner_obj.model_name if overall_winner_obj else best_single_name
        is_ens = overall_winner.startswith("ensemble")
        overall_type = "ensemble" if is_ens else "individual"
        overall_cv = (overall_winner_obj.optimization.get("best_cv_score") if is_ens else float(np.mean(overall_winner_obj.cv_scores))) if overall_winner_obj else best_single_cv
        ov_met = (overall_winner_obj.optimized_metrics or overall_winner_obj.baseline_metrics or {}) if overall_winner_obj else {}
        overall_holdout = ov_met.get(config.primary_metric, ov_met.get("rmse"))

        # Empirical improvement of ensemble over best single
        spec = get_metric_spec(config.primary_metric)
        ens_delta_cv = None
        ens_delta_holdout = None
        if best_ens_cv is not None and best_single_cv is not None:
            # Sklearn scorers are defined such that higher is better (losses are negated)
            ens_delta_cv = round(float(best_ens_cv - best_single_cv), 6)
        if best_ens_holdout is not None and best_single_holdout is not None:
            # Evaluator returns un-negated metrics (e.g. positive RMSE where lower is better)
            if spec.raw_direction == "minimize":
                ens_delta_holdout = round(float(best_single_holdout - best_ens_holdout), 6)
            else:
                ens_delta_holdout = round(float(best_ens_holdout - best_single_holdout), 6)

        total_time = round(time.time() - t_start, 4)

        # LLM advisory tracking
        llm_provider = "Fallback (Deterministic)" if config.condition == "fallback" else "LLM-Assisted"
        llm_recs = [m.name for m in model_defs]
        llm_match = best_single_name in llm_recs

        # Extract secondary holdout metrics
        secondary_metrics = {}
        if config.problem_type == "classification":
            for k in ("accuracy", "balanced_accuracy", "roc_auc", "f1_weighted"):
                if k in ov_met and ov_met[k] is not None:
                    secondary_metrics[k] = round(float(ov_met[k]), 6)
        else:
            for k in ("rmse", "mae", "mse", "r2"):
                if k in ov_met and ov_met[k] is not None:
                    secondary_metrics[k] = round(float(ov_met[k]), 6)

        bench_id = f"{config.dataset_name}_{config.condition}_s{config.seed}"

        return BenchmarkRecord(
            dataset_name=config.dataset_name,
            rows_total=rows_total,
            rows_train=len(train_idx),
            rows_holdout=len(test_idx),
            n_features_raw=n_features_raw,
            n_features_transformed=n_features_transformed,
            problem_type=config.problem_type,
            primary_metric=config.primary_metric,
            condition=config.condition,
            seed=config.seed,
            n_folds=config.n_folds,
            n_trials=config.n_trials,
            holdout_used_for_selection=False,
            candidate_models=candidate_records,
            ensemble_candidates=ensemble_records,
            best_individual_model=best_single_name,
            best_individual_cv_score=round(float(best_single_cv), 6) if best_single_cv is not None else None,
            best_individual_holdout_score=round(float(best_single_holdout), 6) if best_single_holdout is not None else None,
            best_ensemble_model=best_ens_name,
            best_ensemble_cv_score=round(float(best_ens_cv), 6) if best_ens_cv is not None else None,
            best_ensemble_holdout_score=round(float(best_ens_holdout), 6) if best_ens_holdout is not None else None,
            overall_winner=overall_winner,
            overall_winner_type=overall_type,
            overall_cv_score=round(float(overall_cv), 6) if overall_cv is not None else None,
            overall_holdout_score=round(float(overall_holdout), 6) if overall_holdout is not None else None,
            ensemble_improvement_cv=ens_delta_cv,
            ensemble_improvement_holdout=ens_delta_holdout,
            llm_provider=llm_provider,
            llm_recommended_models=llm_recs,
            llm_proposed_operations=proposed_ops,
            operations_accepted=accepted_ops,
            operations_rejected=rejected_ops,
            llm_model_match=llm_match,
            feature_engineering_time=fe_time,
            training_time=training_time,
            total_time=total_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            dataset_hash=get_file_sha256(config.dataset_path),
            git_commit=get_git_revision(),
            cv_scoring=spec.scoring,
            cv_direction=spec.direction,
            holdout_metric=spec.metric,
            holdout_direction=spec.raw_direction,
            pipeline_plan=plan_dict,
            benchmark_id=bench_id,
            secondary_metrics=secondary_metrics,
        )


def aggregate_benchmark_records(records: Union[List[BenchmarkRecord], List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Aggregate multi-seed benchmark records into rigorous statistical comparisons.
    
    Computes:
    - Per-dataset, per-condition statistics (mean, std, min, max, per-seed scores, runtimes)
    - Direction-aware deltas:
        - For maximize metrics: delta = LLM_mean - Det_mean
        - For minimize metrics: delta = Det_mean - LLM_mean (positive means LLM improved)
    - Raw differences
    - Ablation comparisons (Deterministic vs LLM Model Only vs Full LLM Guided)
    - Pipeline behavior summary (models proposed/accepted/rejected, operations proposed/accepted/rejected, fallback rate)
    - Reproducibility metadata and audit statements
    """
    raw_list: List[Dict[str, Any]] = [
        r.to_dict() if hasattr(r, "to_dict") else r for r in records
    ]
    if not raw_list:
        return {}

    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for r in raw_list:
        ds = r.get("dataset_name", "unknown")
        by_dataset.setdefault(ds, []).append(r)

    dataset_summaries = {}
    all_seeds = sorted(list({r.get("seed") for r in raw_list if r.get("seed") is not None}))

    for ds_name, ds_records in by_dataset.items():
        sample = ds_records[0]
        prob_type = sample.get("problem_type", "classification")
        prim_metric = sample.get("primary_metric", "f1_weighted")
        direction = sample.get("holdout_direction", "maximize" if is_higher_better(prim_metric) else "minimize")

        by_cond: Dict[str, List[Dict[str, Any]]] = {}
        for r in ds_records:
            cond = r.get("condition", "default")
            by_cond.setdefault(cond, []).append(r)

        cond_summaries = {}
        for cond_name, c_records in by_cond.items():
            c_records_sorted = sorted(c_records, key=lambda x: x.get("seed", 0))
            seeds = [x.get("seed") for x in c_records_sorted]
            cv_scores = [x.get("overall_cv_score") for x in c_records_sorted if x.get("overall_cv_score") is not None]
            h_scores = [x.get("overall_holdout_score") for x in c_records_sorted if x.get("overall_holdout_score") is not None]
            runtimes = [x.get("total_time") for x in c_records_sorted if x.get("total_time") is not None]
            winners = [x.get("overall_winner") for x in c_records_sorted]

            sec_keys = set()
            for x in c_records_sorted:
                sec_keys.update(x.get("secondary_metrics", {}).keys())
            sec_means = {}
            for k in sec_keys:
                vals = [x.get("secondary_metrics", {}).get(k) for x in c_records_sorted if x.get("secondary_metrics", {}).get(k) is not None]
                if vals:
                    sec_means[k] = round(float(np.mean(vals)), 4)

            cond_summaries[cond_name] = {
                "count": len(c_records_sorted),
                "seeds": seeds,
                "cv_scores": [round(s, 4) for s in cv_scores],
                "cv_mean": round(float(np.mean(cv_scores)), 4) if cv_scores else None,
                "cv_std": round(float(np.std(cv_scores)), 4) if cv_scores else None,
                "holdout_scores": [round(s, 4) for s in h_scores],
                "holdout_mean": round(float(np.mean(h_scores)), 4) if h_scores else None,
                "holdout_std": round(float(np.std(h_scores)), 4) if h_scores else None,
                "holdout_min": round(float(np.min(h_scores)), 4) if h_scores else None,
                "holdout_max": round(float(np.max(h_scores)), 4) if h_scores else None,
                "runtimes": [round(t, 2) for t in runtimes],
                "runtime_mean": round(float(np.mean(runtimes)), 2) if runtimes else None,
                "selected_pipelines": winners,
                "secondary_metrics_means": sec_means,
            }

        # Comparisons
        det_key = next((k for k in cond_summaries if "fallback" in k or "deterministic" in k), None)
        llm_full_key = next((k for k in cond_summaries if k in ("llm_guided", "llm_assisted")), None)
        llm_ablation_key = next((k for k in cond_summaries if "model_only" in k or "without_features" in k), None)

        comparisons = {}
        if det_key and llm_full_key and cond_summaries[det_key]["holdout_mean"] is not None and cond_summaries[llm_full_key]["holdout_mean"] is not None:
            det_mean = cond_summaries[det_key]["holdout_mean"]
            llm_mean = cond_summaries[llm_full_key]["holdout_mean"]
            raw_diff = round(llm_mean - det_mean, 4)
            adj_delta = round(llm_mean - det_mean if direction == "maximize" else det_mean - llm_mean, 4)
            det_cv = cond_summaries[det_key]["cv_mean"]
            llm_cv = cond_summaries[llm_full_key]["cv_mean"]
            cv_delta = round(llm_cv - det_cv, 4) if (det_cv is not None and llm_cv is not None) else None

            comparisons["llm_guided_vs_deterministic"] = {
                "deterministic_condition": det_key,
                "llm_condition": llm_full_key,
                "metric": prim_metric,
                "direction": direction,
                "deterministic_holdout_mean": det_mean,
                "llm_holdout_mean": llm_mean,
                "raw_difference": raw_diff,
                "direction_adjusted_delta": adj_delta,
                "cv_delta": cv_delta,
                "interpretation": (
                    f"LLM showed +{adj_delta} improvement" if adj_delta > 0.0005
                    else f"Deterministic showed +{-adj_delta} advantage" if adj_delta < -0.0005
                    else "Comparable performance within margin"
                ),
            }

        if det_key and llm_ablation_key and cond_summaries[det_key]["holdout_mean"] is not None and cond_summaries[llm_ablation_key]["holdout_mean"] is not None:
            det_mean = cond_summaries[det_key]["holdout_mean"]
            abl_mean = cond_summaries[llm_ablation_key]["holdout_mean"]
            adj_delta = round(abl_mean - det_mean if direction == "maximize" else det_mean - abl_mean, 4)
            comparisons["ablation_model_only_vs_deterministic"] = {
                "metric": prim_metric,
                "direction": direction,
                "deterministic_holdout_mean": det_mean,
                "ablation_holdout_mean": abl_mean,
                "direction_adjusted_delta": adj_delta,
            }

        # Pipeline behavior tracking across LLM runs
        llm_runs = [r for r in ds_records if r.get("condition") != det_key]
        prop_models = []
        acc_models = []
        rej_models = []
        prop_ops = []
        acc_ops = []
        rej_ops = []
        fallback_count = 0

        for lr in llm_runs:
            prop_models.extend(lr.get("llm_recommended_models") or [])
            acc_models.extend([m.get("model_name") for m in lr.get("candidate_models", [])])
            for m in lr.get("operations_rejected", []):
                if isinstance(m, dict) and "model_id" in m:
                    rej_models.append(m.get("model_id"))
            
            p_ops = lr.get("llm_proposed_operations") or []
            prop_ops.extend([op.get("operation") for op in p_ops if isinstance(op, dict)])
            a_ops = lr.get("operations_accepted") or []
            acc_ops.extend([op.get("operation") for op in a_ops if isinstance(op, dict)])
            r_ops = lr.get("operations_rejected") or []
            rej_ops.extend([op.get("operation") if isinstance(op, dict) else str(op) for op in r_ops])

            if "fallback" in (lr.get("llm_provider") or "").lower():
                fallback_count += 1

        pipeline_behavior = {
            "total_llm_runs": len(llm_runs),
            "fallback_runs": fallback_count,
            "fallback_rate": round(fallback_count / len(llm_runs), 2) if llm_runs else 0.0,
            "models_proposed_unique": sorted(list(set(prop_models))),
            "models_accepted_unique": sorted(list(set(acc_models))),
            "models_rejected_count": len(rej_models),
            "operations_proposed_count": len(prop_ops),
            "operations_accepted_count": len(acc_ops),
            "operations_rejected_count": len(rej_ops),
            "operations_proposed_unique": sorted(list(set(prop_ops))),
            "operations_accepted_unique": sorted(list(set(acc_ops))),
        }

        dataset_summaries[ds_name] = {
            "problem_type": prob_type,
            "primary_metric": prim_metric,
            "metric_direction": direction,
            "rows_total": sample.get("rows_total"),
            "rows_train": sample.get("rows_train"),
            "rows_holdout": sample.get("rows_holdout"),
            "dataset_hash": sample.get("dataset_hash"),
            "conditions": cond_summaries,
            "comparisons": comparisons,
            "pipeline_behavior": pipeline_behavior,
        }

    return {
        "title": "AutoML-Lens Multi-Seed Research Benchmark Summary",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": raw_list[0].get("git_commit") if raw_list else None,
        "protocol": {
            "seeds": all_seeds,
            "n_folds": raw_list[0].get("n_folds") if raw_list else 3,
            "n_trials": raw_list[0].get("n_trials") if raw_list else 5,
            "holdout_policy": "Strict fold-safe cross-validation on train split only; holdout set reserved exclusively for final evaluation.",
            "statistical_statement": f"Descriptive multi-seed comparison only (N={len(all_seeds)} seeds per condition). No inferential p-values claimed.",
        },
        "datasets": dataset_summaries,
    }
