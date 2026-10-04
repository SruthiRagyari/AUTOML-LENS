"""Research-grade benchmark runner script for AutoML-Lens.

Runs reproducible multi-seed benchmarks on real datasets:
1. Adult Census Income (Classification): benchmarks/data/adult/adult.data
2. Wine Quality Red (Regression): benchmarks/data/wine+quality/winequality-red.csv

Evaluates 3 experimental conditions across identical seeds [42, 123, 456]:
- Condition A: Deterministic Baseline ('deterministic')
- Condition B: LLM Model Only Ablation ('llm_model_only')
- Condition C: Full LLM-Guided Pipeline Planning ('llm_guided')

Outputs:
- benchmarks/results/benchmark_results.json (raw per-seed execution records)
- benchmarks/results/benchmark_summary.json (statistical aggregation & pipeline behavior)
- docs/BENCHMARK_REPORT.md (research report markdown)
- docs/BENCHMARK_REPORT.html (standalone scientific HTML report)
"""
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import List, Dict, Any

# Ensure backend is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.services.benchmarker import (
    BenchmarkRunner, BenchmarkConfig, BenchmarkRecord, aggregate_benchmark_records
)
from app.services.pipeline_planner import (
    generate_deterministic_plan, PipelinePlanValidator, AutoMLPipelinePlan
)
from app.services.reporter import ReportGenerator
from app.core.config import settings
from app.llm.manager import LLMManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_benchmarks")

SEEDS = [42, 123, 456]


def get_llm_plan_for_dataset(context: dict) -> AutoMLPipelinePlan:
    """Obtain a structured AutoML pipeline plan via LLM Manager with automatic fallback."""
    llm_cfg = {
        "LLM_PROVIDER": "gemini" if settings.GEMINI_API_KEY else "fallback",
        "GEMINI_API_KEY": settings.GEMINI_API_KEY,
        "GEMINI_MODEL": settings.GEMINI_MODEL,
    }
    manager = LLMManager(llm_cfg)
    try:
        res = asyncio.run(manager.plan_pipeline(context))
        plan_dict = res.get("plan")
        if plan_dict:
            return PipelinePlanValidator.validate_and_sanitize(
                plan_dict, context, provider_used=res.get("provider_used", "gemini")
            )
    except Exception as exc:
        logger.warning(f"LLM planning failed, falling back to deterministic: {exc}")
    
    return generate_deterministic_plan(context)


def run_full_suite() -> List[BenchmarkRecord]:
    runner = BenchmarkRunner()
    records: List[BenchmarkRecord] = []

    # Datasets Configuration
    datasets = [
        {
            "name": "adult_census_income",
            "path": "benchmarks/data/adult/adult.data",
            "target": "income",
            "problem_type": "classification",
            "primary_metric": "f1_weighted",
            "max_rows": 1000,
            "sample_cols": [
                "age", "workclass", "education", "marital_status", "occupation",
                "relationship", "race", "sex", "capital_gain", "capital_loss",
                "hours_per_week", "native_country"
            ],
            "num_cols": ["age", "capital_gain", "capital_loss", "hours_per_week"],
            "cat_cols": ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex"],
        },
        {
            "name": "winequality_red",
            "path": "benchmarks/data/wine+quality/winequality-red.csv",
            "target": "quality",
            "problem_type": "regression",
            "primary_metric": "rmse",
            "max_rows": 800,
            "sample_cols": [
                "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                "pH", "sulphates", "alcohol"
            ],
            "num_cols": [
                "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                "pH", "sulphates", "alcohol"
            ],
            "cat_cols": [],
        },
    ]

    for ds in datasets:
        path = ds["path"]
        if not os.path.exists(path):
            logger.warning(f"Dataset path does not exist: {path}, skipping.")
            continue

        logger.info(f"\n=======================================================")
        logger.info(f"BENCHMARKING DATASET: {ds['name']} ({ds['problem_type']})")
        logger.info(f"=======================================================")

        # Build compact context for pipeline planners
        ctx = {
            "columns": ds["sample_cols"] + [ds["target"]],
            "target_column": ds["target"],
            "problem_type": ds["problem_type"],
            "shape": [ds["max_rows"], len(ds["sample_cols"]) + 1],
            "column_profiles": [
                {"name": c, "inferred_type": "numerical" if c in ds["num_cols"] else "categorical"}
                for c in ds["sample_cols"] + [ds["target"]]
            ],
        }

        # 1. Generate Deterministic Plan
        det_plan = generate_deterministic_plan(ctx)
        logger.info(f"Deterministic plan: models={det_plan.candidate_models}, ops={len(det_plan.feature_operations)}")

        # 2. Generate LLM Plan
        llm_plan = get_llm_plan_for_dataset(ctx)
        logger.info(f"LLM plan ({llm_plan.source}): models={llm_plan.candidate_models}, ops={len(llm_plan.feature_operations)}")

        # 3. Create Ablation Plan (LLM model selection without feature operations)
        ablation_plan = llm_plan.model_copy(deep=True)
        ablation_plan.feature_operations = []

        # Iterate across seeds for all 3 conditions under identical split and CV controls
        for seed in SEEDS:
            logger.info(f"\n--- Running Seed: {seed} ---")

            # Condition A: Deterministic Baseline
            cfg_det = BenchmarkConfig(
                dataset_name=ds["name"],
                dataset_path=path,
                target_column=ds["target"],
                problem_type=ds["problem_type"],
                primary_metric=ds["primary_metric"],
                condition="deterministic",
                enable_ensemble=det_plan.ensemble_strategy.enabled,
                models_to_train=det_plan.candidate_models,
                seed=seed,
                n_folds=3,
                n_trials=0,
                max_rows=ds["max_rows"],
                pipeline_plan=det_plan,
            )
            rec_det = runner.run_benchmark(cfg_det)
            records.append(rec_det)
            logger.info(f"[{ds['name']} | Seed {seed} | Deterministic] Winner: {rec_det.overall_winner} | Holdout: {rec_det.overall_holdout_score:.4f} | Time: {rec_det.total_time:.2f}s")

            # Condition B: LLM Model Selection Only (Ablation)
            cfg_abl = BenchmarkConfig(
                dataset_name=ds["name"],
                dataset_path=path,
                target_column=ds["target"],
                problem_type=ds["problem_type"],
                primary_metric=ds["primary_metric"],
                condition="llm_model_only",
                enable_ensemble=ablation_plan.ensemble_strategy.enabled,
                models_to_train=ablation_plan.candidate_models,
                seed=seed,
                n_folds=3,
                n_trials=0,
                max_rows=ds["max_rows"],
                pipeline_plan=ablation_plan,
            )
            rec_abl = runner.run_benchmark(cfg_abl)
            records.append(rec_abl)
            logger.info(f"[{ds['name']} | Seed {seed} | LLM Model Only] Winner: {rec_abl.overall_winner} | Holdout: {rec_abl.overall_holdout_score:.4f} | Time: {rec_abl.total_time:.2f}s")

            # Condition C: Full LLM-Guided Pipeline Planning
            cfg_llm = BenchmarkConfig(
                dataset_name=ds["name"],
                dataset_path=path,
                target_column=ds["target"],
                problem_type=ds["problem_type"],
                primary_metric=ds["primary_metric"],
                condition="llm_guided",
                enable_ensemble=llm_plan.ensemble_strategy.enabled,
                models_to_train=llm_plan.candidate_models,
                custom_operations=llm_plan.feature_operations,
                seed=seed,
                n_folds=3,
                n_trials=0,
                max_rows=ds["max_rows"],
                pipeline_plan=llm_plan,
            )
            rec_llm = runner.run_benchmark(cfg_llm)
            records.append(rec_llm)
            logger.info(f"[{ds['name']} | Seed {seed} | Full LLM-Guided] Winner: {rec_llm.overall_winner} | Holdout: {rec_llm.overall_holdout_score:.4f} | Time: {rec_llm.total_time:.2f}s")

    return records


def save_and_report_results(records: List[BenchmarkRecord]) -> tuple[str, str, str, str]:
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_results.json"
    summary_path = out_dir / "benchmark_summary.json"
    doc_path = Path("docs/BENCHMARK_REPORT.md")
    html_path = Path("docs/BENCHMARK_REPORT.html")
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save raw per-run records
    raw_data = [r.to_dict() for r in records]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)
    logger.info(f"Saved {len(records)} benchmark records to {json_path}")

    # 2. Compute statistical summary
    summary = aggregate_benchmark_records(records)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved benchmark summary to {summary_path}")

    # 3. Generate HTML report
    reporter = ReportGenerator()
    html_content = reporter.generate_benchmark_html_report(summary)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Saved HTML benchmark report to {html_path}")

    # 4. Generate Markdown Report from strictly measured data
    md_lines = [
        "# AutoML-Lens Empirical Research Benchmark Report",
        "",
        f"> **Generated:** {summary.get('generated_at')}",
        f"> **Git Revision:** `{summary.get('git_commit')}`",
        f"> **Protocol:** Strict fold-safe CV on training split; holdout set reserved exclusively for final evaluation.",
        f"> **Statistical Scope:** {summary.get('protocol', {}).get('statistical_statement', '')}",
        "",
        "---",
        "",
        "## 1. Executive Summary & Research Findings",
        "",
        "- **Empirical Reality:** Real performance measurements on real datasets (UCI Adult Census & UCI Wine Quality Red) across 3 reproducible seeds (`[42, 123, 456]`).",
        "- **Holdout Discipline:** Model/pipeline selection was performed using training-side cross-validation only; the holdout set was reserved exclusively for final evaluation.",
        "- **Ablation Findings:** Evaluated deterministic baseline, LLM model selection alone, and full LLM-guided pipeline planning.",
        "",
        "---",
        "",
    ]

    for ds_name, ds_data in summary.get("datasets", {}).items():
        prob = ds_data.get("problem_type")
        metric = ds_data.get("primary_metric")
        dir_str = ds_data.get("metric_direction")
        conditions = ds_data.get("conditions", {})
        comp = ds_data.get("comparisons", {}).get("llm_guided_vs_deterministic", {})
        behavior = ds_data.get("pipeline_behavior", {})

        md_lines.extend([
            f"## 2. Dataset: `{ds_name}` ({prob.upper()})",
            "",
            f"- **Primary Metric:** `{metric}` (Direction: `{dir_str}`)",
            f"- **Total Samples:** {ds_data.get('rows_total')} ({ds_data.get('rows_train')} train / {ds_data.get('rows_holdout')} holdout)",
            f"- **Dataset SHA256:** `{ds_data.get('dataset_hash')}`",
            "",
            "### Multi-Seed Aggregate Results (Descriptive Statistics)",
            "",
            "| Condition | Seeds | CV Score (Mean ± Std) | Holdout Score (Mean ± Std) | Holdout [Min, Max] | Mean Runtime |",
            "|---|---|---|---|---|---|",
        ])

        for c_name, c in conditions.items():
            cv_s = f"{c.get('cv_mean'):.4f} ± {c.get('cv_std'):.4f}" if c.get('cv_mean') is not None else "N/A"
            h_s = f"**{c.get('holdout_mean'):.4f} ± {c.get('holdout_std'):.4f}**" if c.get('holdout_mean') is not None else "N/A"
            min_max = f"[{c.get('holdout_min'):.4f}, {c.get('holdout_max'):.4f}]" if c.get('holdout_min') is not None else "N/A"
            rt = f"{c.get('runtime_mean'):.2f}s" if c.get('runtime_mean') is not None else "N/A"
            md_lines.append(f"| `{c_name}` | {len(c.get('seeds', []))} | {cv_s} | {h_s} | {min_max} | {rt} |")

        if comp:
            adj_d = comp.get("direction_adjusted_delta", 0.0)
            raw_d = comp.get("raw_difference", 0.0)
            interp = comp.get("interpretation", "")
            md_lines.extend([
                "",
                f"> **Comparison vs Deterministic Baseline:**",
                f"> - Deterministic Mean: **{comp.get('deterministic_holdout_mean')}**",
                f"> - LLM-Guided Mean: **{comp.get('llm_holdout_mean')}**",
                f"> - **Direction-Adjusted $\\Delta$:** **{adj_d:+.4f}** ({interp})",
                f"> - Raw Difference: `{raw_d:+.4f}`",
                "",
            ])

        md_lines.extend([
            "### Per-Seed Execution Records",
            "",
            "| Seed | Condition | CV Score | Holdout Score | Selected Winning Pipeline | Wall Time |",
            "|---|---|---|---|---|---|",
        ])

        for c_name, c in conditions.items():
            seeds = c.get("seeds", [])
            cvs = c.get("cv_scores", [])
            houts = c.get("holdout_scores", [])
            times = c.get("runtimes", [])
            pipes = c.get("selected_pipelines", [])
            for i, s in enumerate(seeds):
                cv_v = f"{cvs[i]:.4f}" if i < len(cvs) else "N/A"
                h_v = f"{houts[i]:.4f}" if i < len(houts) else "N/A"
                t_v = f"{times[i]:.2f}s" if i < len(times) else "N/A"
                p_v = f"`{pipes[i]}`" if i < len(pipes) else "N/A"
                md_lines.append(f"| `{s}` | `{c_name}` | {cv_v} | {h_v} | {p_v} | {t_v} |")

        md_lines.extend([
            "",
            "### Pipeline Behavior & Registry Validation Analysis",
            "",
            f"- **Candidate Models Proposed:** `{', '.join(behavior.get('models_proposed_unique', [])) or 'None'}`",
            f"- **Candidate Models Validated:** `{', '.join(behavior.get('models_accepted_unique', [])) or 'None'}`",
            f"- **Feature Operations Proposed:** {behavior.get('operations_proposed_count', 0)} ({', '.join(behavior.get('operations_proposed_unique', [])) or 'None'})",
            f"- **Feature Operations Accepted:** {behavior.get('operations_accepted_count', 0)}",
            f"- **Feature Operations Rejected:** {behavior.get('operations_rejected_count', 0)}",
            f"- **Fallback Rate:** {behavior.get('fallback_rate', 0.0):.0%} ({behavior.get('fallback_runs', 0)} of {behavior.get('total_llm_runs', 0)} runs)",
            "",
            "---",
            "",
        ])

    md_lines.extend([
        "## 3. Scientific Integrity & Limitations",
        "",
        "1. **Holdout Isolation Guarantee:** At no point did the LLM, feature engineering, Optuna HPO, or ensemble stacking consult holdout test rows. Holdout evaluation was executed exactly once post-selection.",
        "2. **Safety & Zero Arbitrary Code:** The LLM functioned strictly as a structured decision layer constrained by `ModelRegistry` and `OPERATION_REGISTRY`. No dynamically generated Python code was executed.",
        "3. **Claim Boundary:** This research makes descriptive multi-seed comparisons. It does not claim general superiority or statistical significance without larger sample sizes.",
        "",
        "---",
        "*Report automatically generated from measured benchmark records by AutoML-Lens.*",
    ])

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Saved Markdown benchmark report to {doc_path}")

    return str(json_path), str(summary_path), str(doc_path), str(html_path)


if __name__ == "__main__":
    records = run_full_suite()
    j_p, s_p, d_p, h_p = save_and_report_results(records)
    print(f"\n=======================================================")
    print(f"BENCHMARK COMPLETE")
    print(f"Raw Records: {j_p}")
    print(f"Summary JSON: {s_p}")
    print(f"Markdown Report: {d_p}")
    print(f"HTML Report: {h_p}")
    print(f"=======================================================")

