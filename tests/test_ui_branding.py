"""Regression guards for the removal of academic/project branding from the public UI.

AutoML-Lens is presented as a standalone software product, so the user-facing
frontend must not describe itself as a final-year / college / research project,
must not carry the "RESEARCH" wordmark badge beside the logo, and must not
advertise internal development status (test counts, "freeze verified" milestones).

The tests read the frontend source, so removed strings cannot creep back in. They
deliberately assert the *fixed* behaviour on both sides: the academic phrasing is
gone AND the legitimate technical content around it - methodology, model registry,
benchmarks, metrics and reproducibility - is still there.

Scope note: ``research_evaluation`` (a backend response key) and the inert
``research-section`` CSS hook are internal, never rendered, and are deliberately
left alone; renaming the key would change the public API contract.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"
INDEX_HTML = REPO_ROOT / "frontend" / "index.html"

SOURCE_FILES = sorted(
    [p for p in FRONTEND_SRC.rglob("*") if p.suffix in {".jsx", ".js", ".css"}]
    + ([INDEX_HTML] if INDEX_HTML.exists() else [])
)


def read(rel):
    """Read a frontend source file relative to ``frontend/src``."""
    return (FRONTEND_SRC / rel).read_text(encoding="utf-8")


# Branding / internal-status text that must never reach a user's screen.
BANNED_PATTERNS = {
    "final-year project": r"final[-\s]?year",
    "research project": r"research[-\s]project",
    "research-grade": r"research[-\s]?grade",
    "academic/student/college project": r"(?:academic|student|college)\s+project",
    "RESEARCH wordmark badge": r">\s*RESEARCH\s*<",
    "freeze / freeze verified": r"\bfreeze\b",
    "test-count status text": r"tests?\s+(?:passing|verified)",
}


# ═══════════════════════ branding must stay removed ═══════════════════════
def test_no_academic_branding_anywhere_in_frontend():
    """No academic/project/internal-status branding in any frontend source file."""
    hits = []
    for path in SOURCE_FILES:
        text = path.read_text(encoding="utf-8")
        for label, pattern in BANNED_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                rel = path.relative_to(REPO_ROOT)
                hits.append(f"{rel}: {label}: {match.group(0)!r}")
    assert not hits, "academic branding reintroduced:\n" + "\n".join(hits)


def test_header_shows_only_the_product_name():
    """Both headers show AUTOML-LENS and carry no badge (navbar and landing)."""
    for page in ("components/Layout/Navbar.jsx", "pages/Landing.jsx"):
        jsx = read(page)
        assert "netflix-brand-logo" in jsx, page
        assert ">AUTOML-LENS</span>" in jsx, page
        assert "netflix-brand-tag" not in jsx, page


def test_footer_carries_no_project_or_dev_status_text():
    """The footer keeps the product name and drops the project/status block."""
    jsx = read("components/Layout/Layout.jsx")
    assert ">AUTOML-LENS</span>" in jsx
    assert "footer-right" not in jsx
    assert "LLM-Guided Structured AutoML Platform" in jsx


def test_document_title_is_product_only():
    """The browser tab title carries no academic framing."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    title = re.search(r"<title>(.*?)</title>", html, re.DOTALL)
    assert title, "index.html has no <title>"
    assert "AutoML-Lens" in title.group(1)
    for pattern in BANNED_PATTERNS.values():
        assert not re.search(pattern, title.group(1), re.IGNORECASE)


# ═══════════════════════ technical content preserved ═══════════════════════
def test_reports_page_presents_benchmark_artifacts():
    """Benchmark reporting stays; only the academic framing was reworded."""
    jsx = read("pages/Reports.jsx")
    assert "Reports & Benchmarks" in jsx
    assert "Standalone Multi-Seed Benchmark Evaluation Report" in jsx
    assert "Multi-Seed Benchmark Highlights" in jsx


def test_methodology_page_keeps_technical_documentation():
    """Methodology / algorithms / evaluation / citations are NOT academic clutter."""
    jsx = read("pages/Methodology.jsx")
    for needle in ("Pipeline Architecture", "Model Registry",
                   "Evaluation Methodology", "References"):
        assert needle in jsx, needle


def test_about_page_still_documents_product_and_limits():
    jsx = read("pages/About.jsx")
    for needle in ("Product Overview", "Technology Stack", "Current Limitations"):
        assert needle in jsx, needle


def test_experiment_page_keeps_benchmark_and_reproducibility_facts():
    """The evaluation section still reports benchmarks and reproducibility."""
    jsx = read("pages/Experiment.jsx")
    for needle in ("BenchmarkEvaluationSection",
                   "Multi-Seed Empirical Benchmark Evaluation",
                   "Open Benchmark Report (HTML)",
                   "Fold"):
        assert needle in jsx, needle


def test_research_evaluation_api_contract_is_still_consumed():
    """Internal field name is load-bearing: renaming it breaks the API contract."""
    jsx = read("pages/Experiment.jsx")
    assert "research_evaluation" in jsx