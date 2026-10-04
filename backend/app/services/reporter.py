"""Report generation service - produces professional HTML reports."""
import os
import datetime
from typing import Any


class ReportGenerator:
    """Generate comprehensive HTML experiment reports."""

    # The stored primary metric may be the sklearn scoring name while the
    # evaluator reports the short display name; same alias table as the API.
    _PRIMARY_METRIC_ALIASES = {"neg_root_mean_squared_error": "rmse"}

    @classmethod
    def _primary_metric_key(cls, payload: dict) -> str:
        """Name the metric this report should present as *the* score.

        Read from the payload the API passes in (``primary_metric`` or
        ``metrics.primary_metric``), falling back to the winner's own declared
        metric. Returns ``None`` when nothing names a metric, in which case the
        report keeps the old "first numeric value" behaviour but labels the key.
        """
        metrics = payload.get("metrics") or {}
        candidate = payload.get("primary_metric") or metrics.get("primary_metric")
        if not isinstance(candidate, str) or not candidate:
            candidate = (payload.get("best_model") or {}).get("metric")
        if not isinstance(candidate, str) or not candidate:
            return None
        return cls._PRIMARY_METRIC_ALIASES.get(candidate, candidate)

    @staticmethod
    def _metric_value(metrics, primary_key):
        """Return ``(value, key_shown)`` for a model's metrics.

        The primary metric is preferred; when a model did not report it, the key
        actually used is returned as well, so the number can be labelled
        honestly instead of implying it is the primary metric.
        """
        if not metrics:
            return None, None
        value = metrics.get(primary_key) if primary_key else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), primary_key
        for key, other in metrics.items():
            if isinstance(other, (int, float)) and not isinstance(other, bool):
                return float(other), key
        return None, None

    @staticmethod
    def _format_metric(value, key, primary_key) -> str:
        if value is None:
            return "N/A"
        text = f"{value:.4f}"
        if key and primary_key and key != primary_key:
            text += f' <span style="font-size:11px;color:#888">({key})</span>'
        return text

    def generate_html_report(self, data: dict[str, Any]) -> str:
        exp_id = data.get("experiment_id", "N/A")
        ts = data.get("timestamp", datetime.datetime.now().isoformat())
        ds = data.get("dataset_info", {})
        target = data.get("target", "N/A")
        ptype = data.get("problem_type", "N/A")
        llm = data.get("llm_analysis", "Fallback mode used.")
        preproc = data.get("preprocessing_summary", {})
        fe = data.get("feature_engineering_summary", "Standard feature processing.")
        models = data.get("models_results", [])
        best = data.get("best_model", {})
        expl = data.get("explainability", {})
        metrics = data.get("metrics", {})

        # Build the models table. The score column shows the experiment's PRIMARY
        # metric; a model that did not report it shows the key actually used in
        # parentheses, instead of a number that looks like the primary metric.
        primary_key = self._primary_metric_key(data)
        score_header = f"Score ({primary_key})" if primary_key else "Score"
        model_rows = ""
        for m in models:
            name = m.get("display_name", m.get("model_name", ""))
            cv = m.get("cv_scores", [])
            cv_mean = f"{sum(cv)/len(cv):.4f}" if cv else "N/A"
            met = m.get("optimized_metrics") or m.get("baseline_metrics", {})
            value, key_used = self._metric_value(met, primary_key)
            primary = self._format_metric(value, key_used, primary_key)
            tt = m.get("training_time", 0) or 0
            status = m.get("status", "N/A")
            is_best = " &#9733;" if m.get("model_name") == best.get("model_name") else ""
            model_rows += f"<tr><td>{name}{is_best}</td><td>{cv_mean}</td><td>{primary}</td><td>{tt:.2f}s</td><td>{status}</td></tr>"

        # Preprocessing summary
        preproc_html = ""
        if isinstance(preproc, dict):
            for k, v in preproc.items():
                preproc_html += f"<li><strong>{k}:</strong> {v}</li>"
        else:
            preproc_html = f"<li>{preproc}</li>"

        # Explainability
        top_feats = expl.get("top_features", [])
        feat_html = "".join(f"<li>{f}</li>" for f in top_feats[:10])

        # Best model metrics
        best_met = best.get("optimized_metrics") or best.get("baseline_metrics", {})
        met_html = ""
        for k, v in best_met.items():
            label = f"{k} (primary)" if k == primary_key else k
            if isinstance(v, (int, float)):
                met_html += f"<tr><td>{label}</td><td>{v:.4f}</td></tr>"
            elif k not in ("confusion_matrix", "classification_report", "class_distribution", "residuals_summary"):
                met_html += f"<tr><td>{k}</td><td>{v}</td></tr>"

        # Model Fusion & Ensemble Optimization
        selection = data.get("selection") or {}
        model_selection = data.get("model_selection") or {}
        ensemble_candidates = [m for m in models if m.get("optimization", {}).get("is_ensemble") or "ensemble" in m.get("model_name", "")]
        if not ensemble_candidates:
            raw_cands = data.get("ensemble_candidates") or (model_selection.get("fusion") or {}).get("ensemble_candidates") or (data.get("fusion") or {}).get("ensemble_candidates") or []
            ensemble_candidates = raw_cands
        is_ensemble_winner = (
            best.get("optimization", {}).get("is_ensemble")
            or "ensemble" in best.get("model_name", "")
            or bool((model_selection.get("fusion") or {}).get("is_ensemble_winner"))
            or bool((data.get("fusion") or {}).get("is_ensemble_winner"))
        )

        fusion_html = ""
        if ensemble_candidates:
            ens_rows = ""
            for em in ensemble_candidates:
                ename = em.get("display_name", em.get("model_name", ""))
                eopt = em.get("optimization", {})
                estrat = eopt.get("strategy", "N/A")
                ecv = em.get("cv_scores", [])
                ecv_mean = f"{sum(ecv)/len(ecv):.4f}" if ecv else "N/A"
                emet = em.get("optimized_metrics") or em.get("baseline_metrics", {})
                evalue, ekey = self._metric_value(emet, primary_key)
                eprimary = self._format_metric(evalue, ekey, primary_key)
                eweights = eopt.get("weights", {})
                w_str = ", ".join(f"<strong>{k}</strong>: {v*100:.1f}%" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in eweights.items()) if isinstance(eweights, dict) else "N/A"
                is_win = " &#9733;" if em.get("model_name") == best.get("model_name") else ""
                ens_rows += f"<tr><td>{ename}{is_win}</td><td><code>{estrat}</code></td><td>{ecv_mean}</td><td>{eprimary}</td><td style='font-size:12px'>{w_str}</td></tr>"

            fusion_html = f"""
<div class="section">
<h2>6. Model Fusion &amp; Ensemble Optimization</h2>
<p>Model fusion combines predictions from real trained candidate models using out-of-fold (OOF) cross-validation evidence on the training split.</p>
<table>
<thead><tr><th>Ensemble Candidate</th><th>Strategy</th><th>OOF CV Score</th><th>{score_header}</th><th>Member Weights</th></tr></thead>
<tbody>{ens_rows}</tbody>
</table>
<div class="highlight" style="margin-top:14px;font-size:13px">
<strong>Leakage Safeguards:</strong>
<ul>
<li>Ensemble weights were optimized strictly on out-of-fold training predictions (training split only).</li>
<li>Candidate selection ranked individual models and ensemble candidates using CV evidence only.</li>
<li><strong>Holdout used for selection:</strong> <code>False</code> (final holdout was evaluated once after freezing the winning architecture).</li>
</ul>
</div>
</div>
"""

        # Research Benchmarking & Evaluation Summary
        research_eval = data.get("research_evaluation") or (data.get("model_selection") or {}).get("research_evaluation")
        benchmark_html = ""
        if research_eval:
            ds_name = research_eval.get("dataset_name", "N/A")
            ptype = research_eval.get("problem_type", "N/A")
            pmetric = research_eval.get("primary_metric", "N/A")
            cond = research_eval.get("condition", "fallback")
            prov = research_eval.get("llm_provider", "fallback")
            repro = research_eval.get("reproducibility", {})
            seed_val = repro.get("seed", "N/A")
            n_folds_val = repro.get("n_folds", "N/A")
            n_trials_val = repro.get("n_trials", "N/A")
            git_rev_val = repro.get("git_commit") or "N/A"
            hash_val = repro.get("dataset_hash") or "N/A"
            w_name = research_eval.get("winner_name", "N/A")
            w_type = research_eval.get("winner_type", "individual")
            w_cv = research_eval.get("winner_cv_score")
            w_holdout = research_eval.get("winner_holdout_score")
            llm_ad = research_eval.get("llm_advisory", {})
            recs = llm_ad.get("recommended_models", [])
            recs_str = ", ".join(recs) if recs else "None"

            benchmark_html = f"""
<div class="section">
<h2>8. Research Benchmarking &amp; Empirical Evaluation</h2>
<p>Empirical benchmark summary contrasting candidate models, ensemble fusion, and LLM advisory recommendations under controlled experimental conditions.</p>
<table>
<thead><tr><th>Dimension</th><th>Experimental Specification / Value</th></tr></thead>
<tbody>
<tr><td><strong>Dataset &amp; Task</strong></td><td><code>{ds_name}</code> &middot; {ptype} (metric: <code>{pmetric}</code>)</td></tr>
<tr><td><strong>Experimental Condition</strong></td><td><code>{cond}</code> (provider: {prov})</td></tr>
<tr><td><strong>Reproducibility Settings</strong></td><td>Seed: <code>{seed_val}</code> &middot; Folds: <code>{n_folds_val}</code> &middot; Trials: <code>{n_trials_val}</code> &middot; Git: <code>{git_rev_val}</code></td></tr>
<tr><td><strong>Dataset Checksum</strong></td><td>SHA-256: <code>{hash_val}</code></td></tr>
<tr><td><strong>Selection Protocol</strong></td><td>Strictly CV evidence on training split &middot; <code>holdout_used_for_selection: False</code></td></tr>
<tr><td><strong>Winning Architecture</strong></td><td><strong>{w_name}</strong> ({w_type.capitalize()})</td></tr>
<tr><td><strong>Selection Evidence (CV)</strong></td><td>{f'{w_cv:.4f}' if isinstance(w_cv, (int, float)) else 'N/A'}</td></tr>
<tr><td><strong>Holdout Evaluation</strong></td><td>{f'{w_holdout:.4f}' if isinstance(w_holdout, (int, float)) else 'N/A'}</td></tr>
<tr><td><strong>LLM Advisory vs Reality</strong></td><td>Recommended models: <em>{recs_str}</em> &middot; Advisory only (winner chosen by CV evidence)</td></tr>
</tbody>
</table>
</div>
"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoML-Lens Report #{exp_id}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;color:#1a1a2e;background:#f8f9fc;line-height:1.7;padding:40px 20px}}
.container{{max-width:1100px;margin:0 auto}}
.header{{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);color:#fff;padding:40px;border-radius:12px;margin-bottom:30px}}
.header h1{{font-size:28px;margin-bottom:8px}}
.header .meta{{opacity:0.8;font-size:14px}}
.section{{background:#fff;border-radius:10px;padding:28px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,0.06)}}
.section h2{{color:#302b63;border-bottom:2px solid #e8e8f0;padding-bottom:10px;margin-bottom:18px;font-size:20px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{border:1px solid #e8e8f0;padding:10px 14px;text-align:left}}
th{{background:#f4f4fa;font-weight:600;color:#302b63}}
tr:nth-child(even){{background:#fafafe}}
.highlight{{background:#f0f4ff;border-left:4px solid #4361ee;padding:16px;border-radius:6px;margin:16px 0}}
.badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600}}
.badge-clf{{background:#e8f5e9;color:#2e7d32}}
.badge-reg{{background:#e3f2fd;color:#1565c0}}
ul{{padding-left:20px}}
li{{margin-bottom:6px}}
.footer{{text-align:center;color:#999;font-size:13px;margin-top:40px;padding-top:20px;border-top:1px solid #eee}}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>&#128300; AutoML-Lens Experiment Report</h1>
<div class="meta">Experiment #{exp_id} &middot; Generated {ts}</div>
</div>

<div class="section">
<h2>1. Dataset Overview</h2>
<table><tr><th>Property</th><th>Value</th></tr>
<tr><td>Rows</td><td>{ds.get('rows','N/A')}</td></tr>
<tr><td>Columns</td><td>{ds.get('columns','N/A')}</td></tr>
<tr><td>Memory</td><td>{ds.get('memory_usage_mb','N/A')} MB</td></tr>
<tr><td>Missing Values</td><td>{ds.get('total_missing','N/A')} ({ds.get('total_missing_percentage','N/A')}%)</td></tr>
<tr><td>Duplicates</td><td>{ds.get('duplicate_rows','N/A')}</td></tr>
</table>
</div>

<div class="section">
<h2>2. Target &amp; Problem Type</h2>
<p><strong>Target Column:</strong> {target}</p>
<p><strong>Problem Type:</strong> <span class="badge {'badge-clf' if ptype=='classification' else 'badge-reg'}">{ptype}</span></p>
</div>

<div class="section">
<h2>3. AI Analysis</h2>
<div class="highlight">{llm if isinstance(llm, str) else str(llm)}</div>
</div>

<div class="section">
<h2>4. Preprocessing</h2>
<ul>{preproc_html}</ul>
</div>

<div class="section">
<h2>5. Candidate Models Evaluated</h2>
<table>
<thead><tr><th>Model</th><th>CV Score</th><th>{score_header}</th><th>Time</th><th>Status</th></tr></thead>
<tbody>{model_rows}</tbody>
</table>
<p style="font-size:12px;color:#888;margin-top:8px">&#9733; = Best Model</p>
</div>

{fusion_html}

<div class="section">
<h2>7. Best Model Details</h2>
<p><strong>Model:</strong> {best.get('display_name','N/A')}</p>
<p><strong>Parameters:</strong> {best.get('best_params',{})}</p>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{met_html}</tbody></table>
</div>

<div class="section">
<h2>8. Explainability</h2>
<p><strong>Method:</strong> {expl.get('method_used','N/A')}</p>
<p>{expl.get('explanation_text','')}</p>
<h3 style="margin-top:14px">Top Features</h3>
<ol>{feat_html}</ol>
</div>

{benchmark_html}

<div class="section">
<h2>10. Limitations &amp; Conclusion</h2>
<p>This automated analysis is based on the provided dataset. Results should be validated on independent test data before any production deployment. External factors and dataset biases may influence real-world performance.</p>
<p><strong>Conclusion:</strong> The <em>{best.get('display_name','')}</em> model was selected as the optimal predictor based on evaluation metrics. Further validation and domain expert review are recommended.</p>
</div>

<div class="footer">
<p>Generated by <strong>AutoML-Lens</strong> &mdash; LLM-Powered Automated Machine Learning Framework</p>
</div>
</div>
</body>
</html>"""

    def save_report(self, html: str, experiment_id: str, storage_path: str = "storage") -> str:
        out_dir = os.path.join(storage_path, "reports")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"report_{experiment_id}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    def generate_benchmark_html_report(self, summary: dict[str, Any]) -> str:
        """Generate a standalone, scientific HTML research benchmark report from stored data."""
        title = summary.get("title", "AutoML-Lens Multi-Seed Research Benchmark Report")
        gen_time = summary.get("generated_at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        git_commit = summary.get("git_commit", "N/A")
        proto = summary.get("protocol", {})
        seeds_str = ", ".join(str(s) for s in proto.get("seeds", []))
        datasets = summary.get("datasets", {})

        dataset_sections = ""
        for ds_name, ds_data in datasets.items():
            prob = ds_data.get("problem_type", "classification")
            prim_metric = ds_data.get("primary_metric", "metric")
            direction = ds_data.get("metric_direction", "maximize")
            hash_val = ds_data.get("dataset_hash", "N/A")
            rows = f"{ds_data.get('rows_total', 0)} ({ds_data.get('rows_train', 0)} train / {ds_data.get('rows_holdout', 0)} holdout)"
            conditions = ds_data.get("conditions", {})
            comparisons = ds_data.get("comparisons", {})
            behavior = ds_data.get("pipeline_behavior", {})

            # Aggregate Table Rows
            table_rows = ""
            for cond_name, c in conditions.items():
                c_label = "Deterministic Baseline" if "fallback" in cond_name or "deterministic" in cond_name else "LLM Model Only (Ablation)" if "model_only" in cond_name else "Full LLM-Guided"
                cv_fmt = f"{c.get('cv_mean', 0.0):.4f} &plusmn; {c.get('cv_std', 0.0):.4f}" if c.get("cv_mean") is not None else "N/A"
                h_fmt = f"{c.get('holdout_mean', 0.0):.4f} &plusmn; {c.get('holdout_std', 0.0):.4f}" if c.get("holdout_mean") is not None else "N/A"
                min_max = f"[{c.get('holdout_min', 0.0):.4f}, {c.get('holdout_max', 0.0):.4f}]" if c.get("holdout_min") is not None else "N/A"
                time_fmt = f"{c.get('runtime_mean', 0.0):.2f}s" if c.get("runtime_mean") is not None else "N/A"
                table_rows += f"<tr><td><strong>{c_label}</strong> (<code>{cond_name}</code>)</td><td>{cv_fmt}</td><td><strong>{h_fmt}</strong></td><td>{min_max}</td><td>{time_fmt}</td></tr>"

            # Per-Seed Table Rows
            per_seed_rows = ""
            for cond_name, c in conditions.items():
                c_label = "Deterministic" if "fallback" in cond_name or "deterministic" in cond_name else "LLM Model Only" if "model_only" in cond_name else "Full LLM"
                seeds = c.get("seeds", [])
                cvs = c.get("cv_scores", [])
                houts = c.get("holdout_scores", [])
                times = c.get("runtimes", [])
                pipes = c.get("selected_pipelines", [])
                for i, s in enumerate(seeds):
                    cv_val = f"{cvs[i]:.4f}" if i < len(cvs) else "N/A"
                    h_val = f"{houts[i]:.4f}" if i < len(houts) else "N/A"
                    t_val = f"{times[i]:.2f}s" if i < len(times) else "N/A"
                    p_val = pipes[i] if i < len(pipes) else "N/A"
                    per_seed_rows += f"<tr><td><code>{s}</code></td><td>{c_label}</td><td>{cv_val}</td><td>{h_val}</td><td><code>{p_val}</code></td><td>{t_val}</td></tr>"

            # Comparison & Delta Box
            comp_box = ""
            llm_comp = comparisons.get("llm_guided_vs_deterministic", {})
            if llm_comp:
                adj_d = llm_comp.get("direction_adjusted_delta", 0.0)
                raw_d = llm_comp.get("raw_difference", 0.0)
                interp = llm_comp.get("interpretation", "")
                delta_color = "#81c784" if adj_d > 0 else "#e57373" if adj_d < 0 else "#90caf9"
                comp_box = f"""
                <div style="background:#1a2332;border:1px solid #2e4368;border-radius:6px;padding:14px;margin:16px 0;">
                    <h4 style="margin:0 0 8px 0;color:#90caf9;">Empirical Comparison (Direction: {direction})</h4>
                    <div style="display:flex;gap:20px;flex-wrap:wrap;">
                        <div>Deterministic Mean: <strong>{llm_comp.get('deterministic_holdout_mean', 'N/A')}</strong></div>
                        <div>LLM-Guided Mean: <strong>{llm_comp.get('llm_holdout_mean', 'N/A')}</strong></div>
                        <div style="color:{delta_color};font-weight:600;">Direction-Adjusted &Delta;: {adj_d:+.4f} ({interp})</div>
                        <div style="color:#aaa;font-size:12px;">Raw Difference: {raw_d:+.4f}</div>
                    </div>
                </div>
                """

            # Pipeline Behavior
            behavior_html = f"""
            <div style="background:#1f1f1f;border:1px solid #333;border-radius:6px;padding:14px;margin-top:14px;">
                <h4 style="margin:0 0 8px 0;color:#e0e0e0;">Pipeline Behavior &amp; Registry Validation</h4>
                <ul style="margin:0;padding-left:18px;font-size:13px;line-height:1.6;">
                    <li><strong>Candidate Models Proposed by LLM:</strong> <code>{', '.join(behavior.get('models_proposed_unique', [])) or 'None'}</code></li>
                    <li><strong>Candidate Models Accepted into Registry:</strong> <code>{', '.join(behavior.get('models_accepted_unique', [])) or 'None'}</code></li>
                    <li><strong>Feature Operations Proposed:</strong> {behavior.get('operations_proposed_count', 0)} (<code>{', '.join(behavior.get('operations_proposed_unique', [])) or 'None'}</code>)</li>
                    <li><strong>Feature Operations Accepted:</strong> {behavior.get('operations_accepted_count', 0)}</li>
                    <li><strong>Fallback Events:</strong> {behavior.get('fallback_runs', 0)} / {behavior.get('total_llm_runs', 0)} (Rate: {behavior.get('fallback_rate', 0.0):.0%})</li>
                </ul>
            </div>
            """

            dataset_sections += f"""
            <div class="section">
                <h2>Dataset: <code>{ds_name}</code> ({prob.upper()})</h2>
                <p><strong>Target:</strong> Primary Metric: <code>{prim_metric}</code> ({direction}) &middot; Rows: {rows} &middot; SHA256: <code>{hash_val}</code></p>
                {comp_box}
                <h3>Aggregate Benchmark Matrix (Multi-Seed Descriptive Statistics)</h3>
                <table>
                    <thead><tr><th>Condition</th><th>CV Score (Mean &plusmn; Std)</th><th>Holdout Score (Mean &plusmn; Std)</th><th>Holdout Range [Min, Max]</th><th>Runtime</th></tr></thead>
                    <tbody>{table_rows}</tbody>
                </table>
                <h3 style="margin-top:20px;">Per-Seed Execution Records</h3>
                <table>
                    <thead><tr><th>Seed</th><th>Condition</th><th>CV Score</th><th>Holdout Score</th><th>Selected Winning Pipeline</th><th>Runtime</th></tr></thead>
                    <tbody>{per_seed_rows}</tbody>
                </table>
                {behavior_html}
            </div>
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f0f0f; color: #e0e0e0; margin: 0; padding: 24px; }}
.container {{ max-width: 1040px; margin: 0 auto; }}
.header {{ border-bottom: 2px solid #2a2a2a; padding-bottom: 16px; margin-bottom: 24px; }}
h1 {{ margin: 0 0 6px 0; font-size: 24px; color: #fff; }}
.meta {{ font-size: 13px; color: #888; }}
.section {{ background: #161616; border: 1px solid #282828; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
h2 {{ margin-top: 0; font-size: 18px; color: #64b5f6; border-bottom: 1px solid #282828; padding-bottom: 8px; }}
h3 {{ font-size: 15px; margin: 16px 0 8px 0; color: #e0e0e0; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #262626; }}
th {{ background: #1c1c1c; color: #aaa; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
tr:hover {{ background: #1a1a1a; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; background: #222; padding: 2px 6px; border-radius: 4px; font-size: 12px; color: #ce93d8; }}
.footer {{ text-align: center; font-size: 12px; color: #666; margin-top: 40px; border-top: 1px solid #222; padding-top: 20px; }}
.alert {{ background: #1a2332; border-left: 4px solid #64b5f6; padding: 12px 16px; border-radius: 4px; margin: 16px 0; font-size: 13px; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
    <h1>{title}</h1>
    <div class="meta">Generated: {gen_time} &middot; Git Revision: <code>{git_commit}</code> &middot; Evaluated Seeds: <code>{seeds_str}</code></div>
</div>

<div class="section">
    <h2>1. Research Objective &amp; Methodology</h2>
    <p>This report documents the rigorous empirical comparison between <strong>Deterministic Rule-Based AutoML</strong> and <strong>LLM-Guided Structured AutoML Pipeline Planning</strong>.</p>
    <div class="alert">
        <strong>Holdout Discipline:</strong> Model/pipeline selection was performed using training-side cross-validation only; the holdout set was reserved exclusively for final evaluation.
    </div>
    <p><strong>Evaluation Protocol:</strong> {proto.get('holdout_policy', '')} <em>{proto.get('statistical_statement', '')}</em></p>
</div>

{dataset_sections}

<div class="section">
    <h2>3. Limitations &amp; Scientific Conclusion</h2>
    <ul style="font-size:13px;line-height:1.6;padding-left:18px;">
        <li><strong>Sample Scope:</strong> Evaluated on UCI Adult Census Income (binary classification) and UCI Wine Quality Red (regression) across 3 reproducible seeds.</li>
        <li><strong>Registry Boundary:</strong> All candidate models and feature operations were constrained to pre-validated registries; zero arbitrary code was executed.</li>
        <li><strong>Claim Boundary:</strong> Does not claim reinforcement learning, autonomous code generation, or state-of-the-art general intelligence. Results represent descriptive multi-seed comparison only.</li>
    </ul>
    <p style="margin-top:14px;"><strong>Conclusion:</strong> Structured LLM-guided pipeline planning operates safely within registry constraints and can automatically formulate model ensembles, feature operations, and hyperparameter strategies. Measured holdout performance must be interpreted relative to the deterministic baseline under identical seed and fold protocols.</p>
</div>

<div class="footer">
    <p>Generated by <strong>AutoML-Lens</strong> &mdash; Structured LLM-Guided Automated Machine Learning</p>
</div>
</div>
</body>
</html>"""

