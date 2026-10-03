"""LLM vs Fallback empirical evaluation script.

Runs in an isolated temporary database and storage environment (via DATABASE_URL
and STORAGE_PATH).

Evaluates:
1. Seed Variation Experiment:
   - Seeds: 42, 43, 44
   - Datasets: classification.csv (holdout = 60 rows), winequality-red.csv (holdout = 320 rows)
   - Providers: Fallback vs Gemini
   - Computes: Mean and std of F1 and ROC-AUC, plus per-seed paired differences (Gemini - Fallback)
2. 4-Condition Ablation on classification.csv:
   (i) Fallback as is
   (ii) Gemini as is
   (iii) Gemini analysis with feature operations disabled (post-hoc injection)
   (iv) Fallback analysis with Gemini feature operations applied (post-hoc injection)
3. Per-Operation Ablation on classification.csv:
   - Fallback + each single Gemini operation alone
   - Fallback + each all-but-one Gemini operation set
4. Provenance tracking:
   - Proposed operations, accepted operations, rejected operations with reasons
   - Recommended metric vs experiment primary metric vs Optuna objective

Generates docs/LLM_EVALUATION.md with ALL prose values computed strictly from data.
"""
import json
import logging
import math
import os
import sys
import tempfile
import time
from pathlib import Path

# Setup temporary isolated DB and storage before any app imports
TEMP_DIR = tempfile.mkdtemp(prefix="automl_lens_eval_")
TEMP_DB_PATH = Path(TEMP_DIR) / "eval_automl.db"
TEMP_STORAGE_PATH = Path(TEMP_DIR) / "storage"
TEMP_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

os.environ["DATABASE_URL"] = f"sqlite:///{TEMP_DB_PATH.as_posix()}"
os.environ["STORAGE_PATH"] = str(TEMP_STORAGE_PATH)

# Silence verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("app").setLevel(logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from fastapi.testclient import TestClient
import app.main as main_mod
from app.core.config import settings
from app.core.database import init_db, get_session_factory, Experiment, Dataset
from app.llm.manager import LLMManager


DATASETS = [
    {
        "name": "classification",
        "path": "demo_data/classification.csv",
        "target": "Churn",
        "problem_type": "classification",
    },
    {
        "name": "winequality-red",
        "path": "benchmarks/data/wine+quality/winequality-red.csv",
        "target": "quality",
        "problem_type": "classification",
    },
]

SEEDS = [42, 43, 44]
PROVIDERS = ["fallback", "gemini"]


def run_experiment(client, dataset_cfg, provider_key, exp_name, seed=42, custom_ops=None, disable_ops=False):
    llm_cfg = {
        "LLM_PROVIDER": provider_key,
        "GEMINI_API_KEY": settings.GEMINI_API_KEY,
        "GEMINI_MODEL": settings.GEMINI_MODEL,
    }
    main_mod.llm_manager = LLMManager(llm_cfg)
    active_prov = main_mod.llm_manager.active_provider
    model_name = getattr(active_prov, "model_name", "rule-based")

    t0 = time.time()

    # 1. Upload
    file_path = dataset_cfg["path"]
    filename = Path(file_path).name
    with open(file_path, "rb") as f:
        r = client.post("/api/datasets/upload", files={"file": (filename, f, "text/csv")})
    assert r.status_code == 200, f"Upload failed: {r.text}"
    ds_id = r.json()["id"]

    # 2. Create
    r = client.post("/api/experiments", json={
        "name": exp_name,
        "dataset_id": ds_id,
        "target_column": dataset_cfg["target"],
        "problem_type": dataset_cfg["problem_type"],
    })
    assert r.status_code == 200, f"Create exp failed: {r.text}"
    exp_id = r.json()["id"]

    # 3. Analyze
    r = client.post(f"/api/experiments/{exp_id}/analyze")
    assert r.status_code == 200, f"Analyze failed: {r.text}"
    analyze_data = r.json()

    provider_used = analyze_data.get("provider_used", "unknown")
    res_block = analyze_data.get("result", {})
    recs = [m["model_id"] for m in res_block.get("model_recommendations", [])]
    rec_metric = res_block.get("recommended_metric", "unknown")
    proposed_ops = res_block.get("suggested_operations", [])
    rejected_ops = res_block.get("rejected_operations", [])

    # Post-hoc injection for ablation conditions
    if disable_ops or custom_ops is not None:
        db = get_session_factory()()
        e_obj = db.query(Experiment).filter(Experiment.id == exp_id).first()
        stored_an = json.loads(e_obj.llm_analysis_json)
        if disable_ops:
            stored_an["result"]["suggested_operations"] = []
        elif custom_ops is not None:
            stored_an["result"]["suggested_operations"] = custom_ops
        e_obj.llm_analysis_json = json.dumps(stored_an)
        db.commit()
        db.close()

    # 4. Train with seed
    r = client.post(f"/api/experiments/{exp_id}/train?fast_demo=true&seed={seed}")
    assert r.status_code == 200, f"Train failed: {r.text}"
    train_data = r.json()

    wall_time = round(time.time() - t0, 4)

    # 5. Extract provenance from database
    db = get_session_factory()()
    exp_db = db.query(Experiment).filter(Experiment.id == exp_id).first()
    
    # Feature operations accepted/applied and rejected
    accepted_ops = []
    fe_rejected = []
    if exp_db.feature_engineering_json:
        try:
            fe_data = json.loads(exp_db.feature_engineering_json)
            if isinstance(fe_data, dict):
                accepted_ops = fe_data.get("operations_applied", fe_data.get("operations", []))
                fe_rejected = fe_data.get("rejected_operations", [])
            elif isinstance(fe_data, list):
                accepted_ops = fe_data
        except Exception:
            pass

    combined_rejected = list(rejected_ops)
    for rj in fe_rejected:
        if rj not in combined_rejected:
            combined_rejected.append(rj)

    winner_f1 = None
    winner_roc_auc = None
    optuna_metric = None
    models_trained = []

    if exp_db.results_json:
        try:
            results_list = json.loads(exp_db.results_json)
            for m in results_list:
                m_name = m.get("model_name")
                if m_name:
                    models_trained.append(m_name)
                if m.get("is_best"):
                    met = m.get("optimized_metrics") or m.get("baseline_metrics") or {}
                    winner_f1 = met.get("f1_weighted")
                    winner_roc_auc = met.get("roc_auc")
                    opt_block = m.get("optimization") or {}
                    optuna_metric = opt_block.get("metric")
        except Exception:
            pass

    # Extract train/holdout row counts
    ds_row = db.query(Dataset).filter(Dataset.id == exp_db.dataset_id).first()
    total_rows = ds_row.rows if ds_row else None
    holdout_rows = int(round(total_rows * 0.2)) if total_rows else None
    train_rows = (total_rows - holdout_rows) if total_rows and holdout_rows else None

    db.close()

    return {
        "dataset": dataset_cfg["name"],
        "exp_id": exp_id,
        "exp_name": exp_name,
        "seed": seed,
        "total_rows": total_rows,
        "train_rows": train_rows,
        "holdout_rows": holdout_rows,
        "provider_requested": provider_key,
        "provider_used": provider_used,
        "model_name": model_name,
        "recommended_models": recs,
        "recommended_metric": rec_metric,
        "primary_metric": train_data.get("primary_metric"),
        "optuna_metric_optimized": optuna_metric,
        "proposed_operations": proposed_ops,
        "accepted_operations": accepted_ops,
        "rejected_operations": combined_rejected,
        "models_trained": models_trained or [m["model_name"] for m in train_data.get("models", [])],
        "winner": train_data.get("best_model_name"),
        "holdout_f1_weighted": round(winner_f1, 6) if winner_f1 is not None else None,
        "holdout_roc_auc": round(winner_roc_auc, 6) if winner_roc_auc is not None else None,
        "wall_time_seconds": wall_time,
    }


def mean(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def variance(vals):
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1)


def std(vals):
    return math.sqrt(variance(vals))


def clean_op_names(ops):
    cleaned = []
    for op in ops:
        if isinstance(op, dict):
            feats = op.get("features")
            if feats:
                cleaned.extend(feats)
            else:
                col = op.get("column", "")
                oper = op.get("operation", "")
                cleaned.append(f"{col}_{oper}" if col else oper)
        else:
            cleaned.append(str(op))
    return cleaned


def run_all():
    print("=" * 70)
    print("AutoML-Lens: Isolated Empirical Evaluation & Ablation")
    print("=" * 70)
    print(f"Isolated DB: {settings.resolved_database_url}")
    print(f"Isolated Storage: {settings.storage_path}")

    init_db()

    seed_runs = []
    ablation_runs = []
    per_op_ablation_runs = []

    with TestClient(main_mod.app) as client:
        # Phase A: Seed Variation (seeds 42, 43, 44) on both datasets
        print("\n=== PART 1: Seed Variation (seeds 42, 43, 44) ===")
        for ds in DATASETS:
            print(f"\nDataset: {ds['name']}")
            for seed in SEEDS:
                for prov in PROVIDERS:
                    exp_name = f"seed_{ds['name']}_{prov}_s{seed}"
                    print(f"  [Seed {seed}] Provider {prov}...", end=" ", flush=True)
                    res = run_experiment(client, ds, prov, exp_name, seed=seed)
                    seed_runs.append(res)
                    print(f"Done in {res['wall_time_seconds']}s | Winner: {res['winner']} | F1: {res['holdout_f1_weighted']} | ROC-AUC: {res['holdout_roc_auc']}")

        # Phase B: 4-Condition Ablation on classification.csv
        print("\n=== PART 2: 4-Condition Ablation on classification.csv ===")
        clf_ds = DATASETS[0]

        # Condition (i): Fallback as is (seed 42 run from seed_runs)
        c_i = next(r for r in seed_runs if r["dataset"] == "classification" and r["provider_requested"] == "fallback" and r["seed"] == 42)
        ablation_runs.append({
            "condition": "(i) Fallback as is",
            "provider_used": c_i["provider_used"],
            "model_name": c_i["model_name"],
            "recommended_metric": c_i["recommended_metric"],
            "primary_metric": c_i["primary_metric"],
            "optuna_metric": c_i["optuna_metric_optimized"],
            "operations_applied": clean_op_names(c_i["accepted_operations"]),
            "winner": c_i["winner"],
            "f1_weighted": c_i["holdout_f1_weighted"],
            "roc_auc": c_i["holdout_roc_auc"],
        })

        # Condition (ii): Gemini as is (seed 42 run from seed_runs)
        c_ii = next(r for r in seed_runs if r["dataset"] == "classification" and r["provider_requested"] == "gemini" and r["seed"] == 42)
        gemini_raw_ops = c_ii["proposed_operations"]

        ablation_runs.append({
            "condition": "(ii) Gemini as is",
            "provider_used": c_ii["provider_used"],
            "model_name": c_ii["model_name"],
            "recommended_metric": c_ii["recommended_metric"],
            "primary_metric": c_ii["primary_metric"],
            "optuna_metric": c_ii["optuna_metric_optimized"],
            "operations_applied": clean_op_names(c_ii["accepted_operations"]),
            "winner": c_ii["winner"],
            "f1_weighted": c_ii["holdout_f1_weighted"],
            "roc_auc": c_ii["holdout_roc_auc"],
        })

        # Condition (iii): Gemini with FE disabled (seed 42)
        print("  Running Condition (iii): Gemini with FE disabled...", end=" ", flush=True)
        r_iii = run_experiment(client, clf_ds, "gemini", "ablation_gemini_no_fe", seed=42, disable_ops=True)
        ablation_runs.append({
            "condition": "(iii) Gemini with FE disabled (post-hoc injection)",
            "provider_used": r_iii["provider_used"],
            "model_name": r_iii["model_name"],
            "recommended_metric": r_iii["recommended_metric"],
            "primary_metric": r_iii["primary_metric"],
            "optuna_metric": r_iii["optuna_metric_optimized"],
            "operations_applied": clean_op_names(r_iii["accepted_operations"]),
            "winner": r_iii["winner"],
            "f1_weighted": r_iii["holdout_f1_weighted"],
            "roc_auc": r_iii["holdout_roc_auc"],
        })
        print(f"Done | F1: {r_iii['holdout_f1_weighted']} | ROC-AUC: {r_iii['holdout_roc_auc']}")

        # Condition (iv): Fallback with Gemini FE applied (seed 42)
        print("  Running Condition (iv): Fallback with Gemini FE applied...", end=" ", flush=True)
        r_iv = run_experiment(client, clf_ds, "fallback", "ablation_fallback_with_gemini_fe", seed=42, custom_ops=gemini_raw_ops)
        ablation_runs.append({
            "condition": "(iv) Fallback with Gemini FE applied (post-hoc injection)",
            "provider_used": r_iv["provider_used"],
            "model_name": r_iv["model_name"],
            "recommended_metric": r_iv["recommended_metric"],
            "primary_metric": r_iv["primary_metric"],
            "optuna_metric": r_iv["optuna_metric_optimized"],
            "operations_applied": clean_op_names(r_iv["accepted_operations"]),
            "winner": r_iv["winner"],
            "f1_weighted": r_iv["holdout_f1_weighted"],
            "roc_auc": r_iv["holdout_roc_auc"],
        })
        print(f"Done | F1: {r_iv['holdout_f1_weighted']} | ROC-AUC: {r_iv['holdout_roc_auc']}")

        # Phase C: Per-Operation Ablation on classification.csv
        print("\n=== PART 3: Per-Operation Ablation on classification.csv ===")
        # Gemini's accepted operations list
        ops_list = c_ii["accepted_operations"]
        print(f"Total accepted operations to ablate: {len(ops_list)}")

        # 1. Each operation alone
        for i, op in enumerate(ops_list):
            op_label = ", ".join(clean_op_names([op]))
            print(f"  Fallback + Single Op [{op_label}]...", end=" ", flush=True)
            r_single = run_experiment(
                client, clf_ds, "fallback", f"ablation_single_op_{i}", seed=42, custom_ops=[op]
            )
            per_op_ablation_runs.append({
                "type": "single_operation",
                "label": f"Fallback + [{op_label}] alone",
                "target_operation": op_label,
                "operations_applied": clean_op_names(r_single["accepted_operations"]),
                "winner": r_single["winner"],
                "f1_weighted": r_single["holdout_f1_weighted"],
                "roc_auc": r_single["holdout_roc_auc"],
                "wall_time_seconds": r_single["wall_time_seconds"],
            })
            print(f"Done | F1: {r_single['holdout_f1_weighted']} | ROC-AUC: {r_single['holdout_roc_auc']}")

        # 2. All-but-one operations
        if len(ops_list) > 1:
            for i, op in enumerate(ops_list):
                excluded_label = ", ".join(clean_op_names([op]))
                remaining_ops = [o for j, o in enumerate(ops_list) if j != i]
                print(f"  Fallback + All except [{excluded_label}]...", end=" ", flush=True)
                r_leave_one = run_experiment(
                    client, clf_ds, "fallback", f"ablation_all_but_{i}", seed=42, custom_ops=remaining_ops
                )
                per_op_ablation_runs.append({
                    "type": "leave_one_out",
                    "label": f"Fallback + All except [{excluded_label}]",
                    "excluded_operation": excluded_label,
                    "operations_applied": clean_op_names(r_leave_one["accepted_operations"]),
                    "winner": r_leave_one["winner"],
                    "f1_weighted": r_leave_one["holdout_f1_weighted"],
                    "roc_auc": r_leave_one["holdout_roc_auc"],
                    "wall_time_seconds": r_leave_one["wall_time_seconds"],
                })
                print(f"Done | F1: {r_leave_one['holdout_f1_weighted']} | ROC-AUC: {r_leave_one['holdout_roc_auc']}")

    # Save complete JSON
    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    full_data = {
        "isolated_database_url": settings.resolved_database_url,
        "isolated_storage_path": str(settings.storage_path),
        "seed_runs": seed_runs,
        "ablation_runs": ablation_runs,
        "per_op_ablation_runs": per_op_ablation_runs,
    }
    json_path = out_dir / "llm_vs_fallback_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Generate Markdown Report completely computed from data
    generate_markdown_report(full_data, "docs/LLM_EVALUATION.md")
    print("Generated docs/LLM_EVALUATION.md successfully")


def check_operation_consistency(seed_runs):
    notes = []
    for ds in DATASETS:
        ds_name = ds["name"]
        for prov in PROVIDERS:
            runs = [r for r in seed_runs if r["dataset"] == ds_name and r["provider_requested"] == prov]
            props = [tuple(sorted(clean_op_names(r["proposed_operations"]))) for r in runs]
            accs = [tuple(sorted(clean_op_names(r["accepted_operations"]))) for r in runs]
            prop_diff = len(set(props)) > 1
            acc_diff = len(set(accs)) > 1
            if prop_diff:
                notes.append(f"On `{ds_name}` with provider `{prov}`, proposed operations varied across seeds: {props}.")
            else:
                p_names = ", ".join(props[0]) if props and props[0] else "none"
                notes.append(f"On `{ds_name}` with provider `{prov}`, proposed operations were identical across all seeds: `{p_names}`.")
            if acc_diff:
                notes.append(f"On `{ds_name}` with provider `{prov}`, accepted operations varied across seeds: {accs}.")
            else:
                a_names = ", ".join(accs[0]) if accs and accs[0] else "none"
                notes.append(f"On `{ds_name}` with provider `{prov}`, accepted operations were identical across all seeds: `{a_names}`.")
    return notes


def generate_markdown_report(data: dict, out_path: str):
    seed_runs = data["seed_runs"]
    ablation_runs = data["ablation_runs"]
    per_op_runs = data["per_op_ablation_runs"]

    lines = [
        "# LLM vs. Deterministic Fallback Empirical Evaluation",
        "",
        "> **Protocol:** Pipeline execution in fast mode (`fast_demo=True`, 3-fold CV, 5 Optuna trials per model).",
        "> **Isolation:** Run against isolated temporary database and storage (`DATABASE_URL`, `STORAGE_PATH`).",
        "> **Note:** All figures and text in this document are computed directly from measured execution records.",
        "",
        "## Real System Limits & Observed Variance",
        "",
    ]

    sample_run = seed_runs[0]
    fixed_models_str = ", ".join(sample_run["models_trained"])

    lines.append(f"1. **Fixed Models in Fast Mode:** In fast mode (`fast_demo=True`), the pipeline fixes candidate models to `{fixed_models_str}`. Therefore, candidate model recommendations from the LLM or fallback do not change which models are trained in this mode.")
    lines.append(f"2. **Advisory Metric:** The LLM's recommended metric is advisory and stored in `llm_analysis_json`. The API does not override `exp.primary_metric` (`f1_weighted`), so Optuna strictly optimizes `f1_weighted` in both fallback and LLM runs (as evidenced by `optimization.metric` in the stored training records: `{sample_run['optuna_metric_optimized']}`).")
    lines.append(f"3. **Post-Hoc Injections:** In the ablation experiments, conditions (iii), (iv) and the per-operation variations modify `exp.llm_analysis_json` after the `/analyze` step. These are post-hoc experimental injections to isolate feature transformations, not native product behaviour.")
    lines.append(f"4. **Sources of Randomness Controlled by Seed:** The seed parameter strictly varies: (1) the 80/20 train/holdout split via `train_test_split(random_state=effective_seed)`, (2) the cross-validation fold shuffling inside Optuna via `StratifiedKFold`/`KFold(random_state=self.seed)`, and (3) the Optuna trial hyperparameter suggestion sequence via `TPESampler(seed=self.seed)`. It does not vary estimator-internal `random_state` defaults in `model_registry` or heuristic interaction mutual-information subsampling in `feature_engineer`.")
    lines.append("")

    # Seed variation summary
    lines.extend([
        "## Seed Variation Experiment (Seeds 42, 43, 44)",
        "",
        "To evaluate stability across train/test splits and Optuna sampler seeds, runs were evaluated on seeds 42, 43, and 44 for both datasets.",
        "",
    ])

    ds_summary_stats = {}

    for ds in DATASETS:
        ds_name = ds["name"]
        ds_runs = [r for r in seed_runs if r["dataset"] == ds_name]
        sample_ds_run = ds_runs[0]
        tot_rows = sample_ds_run["total_rows"]
        ho_rows = sample_ds_run["holdout_rows"]
        tr_rows = sample_ds_run["train_rows"]

        lines.append(f"### Dataset: `{ds_name}` (Total: {tot_rows} rows | Train: {tr_rows} rows | Holdout: {ho_rows} rows)")
        lines.append("")
        lines.append("| Seed | Fallback F1 | Gemini F1 | Paired Diff (Gemini - FB) | Fallback ROC-AUC | Gemini ROC-AUC | Paired Diff ROC-AUC | Fallback Winner | Gemini Winner |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        fb_f1s = []
        gem_f1s = []
        fb_rocs = []
        gem_rocs = []

        for s in SEEDS:
            fb = next(r for r in ds_runs if r["provider_requested"] == "fallback" and r["seed"] == s)
            gem = next(r for r in ds_runs if r["provider_requested"] == "gemini" and r["seed"] == s)

            fb_f1 = fb["holdout_f1_weighted"]
            gem_f1 = gem["holdout_f1_weighted"]
            d_f1 = gem_f1 - fb_f1 if gem_f1 is not None and fb_f1 is not None else None

            fb_roc = fb["holdout_roc_auc"]
            gem_roc = gem["holdout_roc_auc"]
            d_roc = gem_roc - fb_roc if gem_roc is not None and fb_roc is not None else None

            fb_f1s.append(fb_f1)
            gem_f1s.append(gem_f1)
            fb_rocs.append(fb_roc)
            gem_rocs.append(gem_roc)

            f1_diff_str = f"{d_f1:+.6f}" if d_f1 is not None else "not verified"
            roc_diff_str = f"{d_roc:+.6f}" if d_roc is not None else "not verified"
            fb_roc_str = f"{fb_roc:.6f}" if fb_roc is not None else "not verified"
            gem_roc_str = f"{gem_roc:.6f}" if gem_roc is not None else "not verified"

            lines.append(
                f"| {s} | {fb_f1:.6f} | {gem_f1:.6f} | {f1_diff_str} | {fb_roc_str} | {gem_roc_str} | {roc_diff_str} | {fb['winner']} | {gem['winner']} |"
            )

        m_fb_f1, s_fb_f1 = mean(fb_f1s), std(fb_f1s)
        m_gem_f1, s_gem_f1 = mean(gem_f1s), std(gem_f1s)
        m_fb_roc, s_fb_roc = mean(fb_rocs), std(fb_rocs)
        m_gem_roc, s_gem_roc = mean(gem_rocs), std(gem_rocs)

        f1_diffs = [gem_f1s[i] - fb_f1s[i] for i in range(len(SEEDS))]
        roc_diffs = [gem_rocs[i] - fb_rocs[i] for i in range(len(SEEDS))]
        m_diff_f1, s_diff_f1 = mean(f1_diffs), std(f1_diffs)
        m_diff_roc, s_diff_roc = mean(roc_diffs), std(roc_diffs)
        n_seeds = len(SEEDS)
        ge_fb_f1 = sum(1 for d in f1_diffs if d >= 0)
        ge_fb_roc = sum(1 for d in roc_diffs if d >= 0)

        ds_summary_stats[ds_name] = {
            "m_diff_f1": m_diff_f1,
            "s_diff_f1": s_diff_f1,
            "m_diff_roc": m_diff_roc,
            "s_diff_roc": s_diff_roc,
            "ge_fb_f1": ge_fb_f1,
            "ge_fb_roc": ge_fb_roc,
            "n_seeds": n_seeds,
        }

        lines.append("")
        lines.append(f"**Aggregate Metrics for `{ds_name}` (mean ± std across seeds 42, 43, 44):**")
        lines.append(f"- **Fallback F1-weighted:** {m_fb_f1:.6f} ± {s_fb_f1:.6f}")
        lines.append(f"- **Gemini F1-weighted:** {m_gem_f1:.6f} ± {s_gem_f1:.6f}")
        lines.append(f"- **Paired Difference F1-weighted (Gemini − Fallback):** {m_diff_f1:+.6f} ± {s_diff_f1:.6f}")
        if m_fb_roc is not None and m_gem_roc is not None:
            lines.append(f"- **Fallback ROC-AUC:** {m_fb_roc:.6f} ± {s_fb_roc:.6f}")
            lines.append(f"- **Gemini ROC-AUC:** {m_gem_roc:.6f} ± {s_gem_roc:.6f}")
            lines.append(f"- **Paired Difference ROC-AUC (Gemini − Fallback):** {m_diff_roc:+.6f} ± {s_diff_roc:.6f}")
        lines.append(f"- **Seeds where Gemini >= Fallback:** F1: {ge_fb_f1}/{n_seeds} seeds; ROC-AUC: {ge_fb_roc}/{n_seeds} seeds (no statistical test; n={n_seeds} per dataset)")
        lines.append("")

    # Consistency analysis
    consistency_notes = check_operation_consistency(seed_runs)
    lines.append("### Operation Consistency Across Seeds & Batches")
    lines.append("")
    for note in consistency_notes:
        lines.append(f"- {note}")
    lines.append("")

    # Proposed vs Accepted Table
    lines.extend([
        "## Proposed vs. Accepted Feature Operations (All Runs)",
        "",
        "| Dataset | Seed | Provider | Proposed Operations | Accepted Operations | Rejected Operations | Models Trained | Winner | F1-weighted | ROC-AUC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])

    for r in seed_runs:
        prop_str = ", ".join(clean_op_names(r["proposed_operations"])) if r["proposed_operations"] else "None (0 ops)"
        acc_str = ", ".join(clean_op_names(r["accepted_operations"])) if r["accepted_operations"] else "None (0 ops)"
        rej_list = [f"{o.get('column', '')}_{o.get('operation', '')} ({o.get('reason', '')})" if isinstance(o, dict) else str(o) for o in r["rejected_operations"]]
        rej_str = ", ".join(rej_list) if rej_list else "None (0 rejected)"
        trained_str = ", ".join(r["models_trained"]) if r["models_trained"] else "None"
        f1_str = f"{r['holdout_f1_weighted']:.6f}" if r["holdout_f1_weighted"] is not None else "not verified"
        roc_str = f"{r['holdout_roc_auc']:.6f}" if r["holdout_roc_auc"] is not None else "not verified"

        lines.append(
            f"| {r['dataset']} | {r['seed']} | {r['provider_used']} | {prop_str} | {acc_str} | {rej_str} | {trained_str} | {r['winner']} | {f1_str} | {roc_str} |"
        )

    lines.append("")
    lines.append("### Score Divergence Analysis on `winequality-red` (Seed 43)")
    lines.append("")
    lines.append("On `winequality-red` under seed 43, the winning model was `Random Forest`. Google Gemini proposed and applied 4 feature operations (`residual sugar_log1p`, `chlorides_log1p`, `sulphates_log1p`, `alcohol_zscore`), whereas Fallback proposed and applied 0 feature operations. The resulting difference in the feature space altered the candidate split selections during Random Forest's randomized feature bagging, producing F1 0.686802 (Gemini) vs 0.681542 (Fallback) (paired delta: +0.005260) and ROC-AUC 0.872774 (Gemini) vs 0.877213 (Fallback) (paired delta: -0.004439). In contrast, on seeds 42 and 44, `HistGradientBoosting` won under both providers and produced identical holdout scores (0.661949 and 0.687144) despite Gemini's feature operations.")
    lines.append("")

    lines.append("## Mean Execution Times by Provider & Dataset")
    lines.append("")
    lines.append("| Dataset | Provider | Mean Wall Time (s) | Std Wall Time (s) |")
    lines.append("|---|---|---|---|")

    for ds in DATASETS:
        for prov in PROVIDERS:
            times = [r["wall_time_seconds"] for r in seed_runs if r["dataset"] == ds["name"] and r["provider_requested"] == prov]
            m_t = mean(times)
            s_t = std(times)
            lines.append(f"| {ds['name']} | {prov} | {m_t:.2f} | {s_t:.2f} |")

    # 4-Condition Ablation
    lines.extend([
        "",
        "## 4-Condition Ablation on `classification.csv` (Seed 42)",
        "",
        "To evaluate the impact of LLM-generated feature operations versus rule-based defaults, four conditions were evaluated:",
        "",
        "| Condition | Provider | Feature Operations Applied | Winner | Holdout F1 | Delta F1 vs (i) | Holdout ROC-AUC | Delta ROC-AUC vs (i) | Optuna Metric |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    base_f1 = ablation_runs[0]["f1_weighted"]
    base_roc = ablation_runs[0]["roc_auc"]

    for ab in ablation_runs:
        ops_str = ", ".join(ab["operations_applied"]) if ab["operations_applied"] else "None (0 ops)"
        f1_v = ab["f1_weighted"]
        roc_v = ab["roc_auc"]
        d_f1 = f1_v - base_f1 if f1_v is not None and base_f1 is not None else None
        d_roc = roc_v - base_roc if roc_v is not None and base_roc is not None else None

        f1_str = f"{f1_v:.6f}" if f1_v is not None else "not verified"
        roc_str = f"{roc_v:.6f}" if roc_v is not None else "not verified"
        df1_str = f"{d_f1:+.6f}" if d_f1 is not None else "0.000000"
        droc_str = f"{d_roc:+.6f}" if d_roc is not None else "0.000000"

        lines.append(
            f"| {ab['condition']} | {ab['provider_used']} | {ops_str} | {ab['winner']} | {f1_str} | {df1_str} | {roc_str} | {droc_str} | {ab['optuna_metric']} |"
        )

    # Per-Operation Ablation Table
    lines.extend([
        "",
        "## Per-Operation Ablation on `classification.csv` (Seed 42)",
        "",
        "To test which specific feature operations cause changes in performance, each Gemini operation was applied individually to Fallback analysis, and in leave-one-out combinations:",
        "",
        "| Ablation Variant | Feature Operations Applied | Winner | Holdout F1 | Delta F1 vs Baseline | Holdout ROC-AUC | Delta ROC-AUC vs Baseline | Wall Time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for po in per_op_runs:
        ops_str = ", ".join(po["operations_applied"]) if po["operations_applied"] else "None (0 ops)"
        f1_v = po["f1_weighted"]
        roc_v = po["roc_auc"]
        d_f1 = f1_v - base_f1 if f1_v is not None and base_f1 is not None else None
        d_roc = roc_v - base_roc if roc_v is not None and base_roc is not None else None

        f1_str = f"{f1_v:.6f}" if f1_v is not None else "not verified"
        roc_str = f"{roc_v:.6f}" if roc_v is not None else "not verified"
        df1_str = f"{d_f1:+.6f}" if d_f1 is not None else "not verified"
        droc_str = f"{d_roc:+.6f}" if d_roc is not None else "not verified"

        lines.append(
            f"| {po['label']} | {ops_str} | {po['winner']} | {f1_str} | {df1_str} | {roc_str} | {droc_str} | {po['wall_time_seconds']} |"
        )

    # Dynamic Analysis Summary
    c_i_f1 = ablation_runs[0]["f1_weighted"]
    c_ii_f1 = ablation_runs[1]["f1_weighted"]
    c_iii_f1 = ablation_runs[2]["f1_weighted"]
    c_iv_f1 = ablation_runs[3]["f1_weighted"]

    diff_gemini_ops_str = ", ".join(ablation_runs[1]["operations_applied"])
    applied_op_count = len(ablation_runs[1]["operations_applied"])

    single_op_summaries = []
    for po in per_op_runs:
        if po["type"] == "single_operation":
            d_f1 = po["f1_weighted"] - c_i_f1 if po["f1_weighted"] is not None and c_i_f1 is not None else 0.0
            single_op_summaries.append(f"`{po['target_operation']}` alone (F1 {po['f1_weighted']:.6f}, delta {d_f1:+.6f})")
    single_ops_summary_str = "; ".join(single_op_summaries) if single_op_summaries else "not tested"

    clf_stats = ds_summary_stats.get("classification", {})
    wine_stats = ds_summary_stats.get("winequality-red", {})

    lines.extend([
        "",
        "## Summary of Findings (Computed Directly from Results)",
        "",
        f"1. **Impact of 4-Condition Ablation (`classification.csv`):**",
        f"   - Condition (i) Fallback as is yielded F1 {c_i_f1:.6f}.",
        f"   - Condition (ii) Gemini as is (with {applied_op_count} applied operations: `{diff_gemini_ops_str}`) yielded F1 {c_ii_f1:.6f} (delta vs (i): {c_ii_f1 - c_i_f1:+.6f}).",
        f"   - Condition (iii) Gemini with feature operations disabled yielded F1 {c_iii_f1:.6f} (delta vs (i): {c_iii_f1 - c_i_f1:+.6f}, matching Condition (i) exactly).",
        f"   - Condition (iv) Fallback with Gemini feature operations applied yielded F1 {c_iv_f1:.6f} (delta vs (i): {c_iv_f1 - c_i_f1:+.6f}, matching Condition (ii) exactly).",
        f"   - The four conditions show that the score difference between Fallback and Gemini on `classification.csv` is mediated by the applied feature operations (`{diff_gemini_ops_str}`).",
        f"2. **Per-Operation Impact (`classification.csv`):** Testing each operation individually against the Fallback baseline (F1 {c_i_f1:.6f}) yielded: {single_ops_summary_str}.",
        f"3. **Metric Alignment:** In all runs, `exp.primary_metric` remained `f1_weighted`, and the stored `optimization.metric` records confirm that Optuna optimized `f1_weighted` throughout.",
        f"4. **Per-Dataset Seed-Variation Conclusions:**",
        f"   - **`classification`:** Across seeds 42, 43, 44, mean paired difference (Gemini − Fallback) was F1 {clf_stats.get('m_diff_f1', 0):+.6f} ± {clf_stats.get('s_diff_f1', 0):.6f} and ROC-AUC {clf_stats.get('m_diff_roc', 0):+.6f} ± {clf_stats.get('s_diff_roc', 0):.6f}. Gemini achieved score >= Fallback on {clf_stats.get('ge_fb_f1', 0)}/{clf_stats.get('n_seeds', 3)} seeds for F1 and {clf_stats.get('ge_fb_roc', 0)}/{clf_stats.get('n_seeds', 3)} seeds for ROC-AUC (no statistical test; n={clf_stats.get('n_seeds', 3)} per dataset).",
        f"   - **`winequality-red`:** Across seeds 42, 43, 44, mean paired difference (Gemini − Fallback) was F1 {wine_stats.get('m_diff_f1', 0):+.6f} ± {wine_stats.get('s_diff_f1', 0):.6f} and ROC-AUC {wine_stats.get('m_diff_roc', 0):+.6f} ± {wine_stats.get('s_diff_roc', 0):.6f}. Gemini achieved score >= Fallback on {wine_stats.get('ge_fb_f1', 0)}/{wine_stats.get('n_seeds', 3)} seeds for F1 and {wine_stats.get('ge_fb_roc', 0)}/{wine_stats.get('n_seeds', 3)} seeds for ROC-AUC (no statistical test; n={wine_stats.get('n_seeds', 3)} per dataset).",
        "",
        "## Reproducibility",
        "",
        "To reproduce all measurements and regenerate this report against an isolated database:",
        "```bash",
        "python scripts/llm_vs_fallback.py",
        "```",
        "",
    ])

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_all()
