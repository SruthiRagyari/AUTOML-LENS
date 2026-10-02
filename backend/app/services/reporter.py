"""Report generation service - produces professional HTML reports."""
import os
import datetime
from typing import Any


class ReportGenerator:
    """Generate comprehensive HTML experiment reports."""

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

        # Build models table
        model_rows = ""
        for m in models:
            name = m.get("display_name", m.get("model_name", ""))
            cv = m.get("cv_scores", [])
            cv_mean = f"{sum(cv)/len(cv):.4f}" if cv else "N/A"
            met = m.get("optimized_metrics") or m.get("baseline_metrics", {})
            primary = list(met.values())[0] if met else "N/A"
            if isinstance(primary, (int, float)):
                primary = f"{primary:.4f}"
            tt = m.get("training_time", 0)
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
            if isinstance(v, (int, float)):
                met_html += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
            elif k not in ("confusion_matrix", "classification_report", "class_distribution", "residuals_summary"):
                met_html += f"<tr><td>{k}</td><td>{v}</td></tr>"

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
<h2>5. Models Trained</h2>
<table>
<thead><tr><th>Model</th><th>CV Score</th><th>Test Score</th><th>Time</th><th>Status</th></tr></thead>
<tbody>{model_rows}</tbody>
</table>
<p style="font-size:12px;color:#888;margin-top:8px">&#9733; = Best Model</p>
</div>

<div class="section">
<h2>6. Best Model Details</h2>
<p><strong>Model:</strong> {best.get('display_name','N/A')}</p>
<p><strong>Parameters:</strong> {best.get('best_params',{})}</p>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{met_html}</tbody></table>
</div>

<div class="section">
<h2>7. Explainability</h2>
<p><strong>Method:</strong> {expl.get('method_used','N/A')}</p>
<p>{expl.get('explanation_text','')}</p>
<h3 style="margin-top:14px">Top Features</h3>
<ol>{feat_html}</ol>
</div>

<div class="section">
<h2>8. Limitations &amp; Conclusion</h2>
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
