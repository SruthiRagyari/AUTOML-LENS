"""Check README.md and docs/FINAL_RESULTS.md against benchmark_summary.json.

Compares parsed JSON values (means, population stds, per-seed scores,
row counts, n_trials, seeds) with the docs, and forbids stale numbers
and overclaiming wording outside docs/archive/.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "benchmarks" / "results" / "benchmark_summary.json").read_text(encoding="utf-8"))

adult = summary["datasets"]["adult_census_income"]
wine = summary["datasets"]["winequality_red"]
proto = summary["protocol"]

failures = []


def expect_doc_contains(path, text, label, variants):
    if not any(v in text for v in variants):
        failures.append(f"{path} MISSING {label} (tried {variants!r})")


def check_doc_numbers(rel):
    text = (ROOT / rel).read_text(encoding="utf-8")
    for ds_label, conds in [
        ("adult", adult["conditions"]),
        ("wine", wine["conditions"]),
    ]:
        for cond_name in ["deterministic", "llm_model_only", "llm_guided"]:
            c = conds[cond_name]
            for score in c["holdout_scores"]:
                expect_doc_contains(rel, text, f"{ds_label}/{cond_name} per-seed {score}",
                                    [f"{score}", f"{score:.4f}"])
            expect_doc_contains(rel, text, f"{ds_label}/{cond_name} mean {c['holdout_mean']}",
                                [f"{c['holdout_mean']}", f"{c['holdout_mean']:.4f}"])
            expect_doc_contains(rel, text, f"{ds_label}/{cond_name} std {c['holdout_std']}",
                                [f"{c['holdout_std']}", f"{c['holdout_std']:.4f}"])
    # row counts from JSON: accept plain and comma-formatted variants
    # (docs write "1,000 rows"; JSON stores 1000)
    for ds, label in [(adult, "adult"), (wine, "wine")]:
        for key in ["rows_total", "rows_train", "rows_holdout"]:
            n = ds[key]
            variants = [str(n)]
            if n >= 1000:
                variants.append(f"{n:,}")
            expect_doc_contains(rel, text, f"{label} {key}={n}", variants)
    # train/holdout splits as written in docs
    expect_doc_contains(rel, text, "adult 800/200 split",
                        ["800 train / 200 holdout", "800/200", "800 train", "200 holdout"])
    expect_doc_contains(rel, text, "wine 640/160 split",
                        ["640 train / 160 holdout", "640/160", "640 train", "160 holdout"])
    # protocol
    expect_doc_contains(rel, text, "seeds [42, 123, 456]", ["[42, 123, 456]", "42, 123, 456"])
    expect_doc_contains(rel, text, "n_trials = 0", ["n_trials=0", "n_trials = 0", "n_trials`) | **0**"])
    if "population std (ddof=0)" not in text.lower():
        failures.append(f"{rel} MISSING population std (ddof=0) convention")
    return text


for rel in ["README.md", "docs/FINAL_RESULTS.md"]:
    check_doc_numbers(rel)

stale = ["0.0187", "0.0101", "0.0307", "0.0298", "0.7961", "0.8335",
         "2,000", "2000 rows", "N=2,000", "Subsampled to 2,000"]
forbidden = ["LLM win", "Det win", "LLM advantage", "Deterministic advantage",
             "LLM improvement", "Deterministic baseline advantage",
             "LLM improvement driven", "+0.0072$ improvement", "+0.0072 improvement"]

# BENCHMARK_REPORT.md/.html are generated artifacts: check them for stale
# numbers but wording rules apply to the hand-written docs below.
WORDING_SCOPE = ["README.md", "docs/FINAL_RESULTS.md",
                 "docs/PROJECT_DOCUMENTATION.md", "docs/RESEARCH_PAPER_DRAFT.md",
                 "docs/PRESENTATION_OUTLINE.md", "docs/VIVA_QUESTIONS.md",
                 "docs/DEMO_SCRIPT.md", "docs/DEMO.md",
                 "docs/ARCHITECTURE_DIAGRAM.md", "docs/LLM_EVALUATION.md",
                 "docs/DEPLOYMENT.md"]

texts = {}
texts["README.md"] = (ROOT / "README.md").read_text(encoding="utf-8")
for md in sorted((ROOT / "docs").glob("*.md")):
    texts[f"docs/{md.name}"] = md.read_text(encoding="utf-8")

for rel, text in texts.items():
    for s in stale:
        if s in text:
            failures.append(f"{rel} CONTAINS STALE {s!r}")
    if rel in WORDING_SCOPE:
        low = text.lower()
        for s in forbidden:
            idx = low.find(s.lower())
            while idx != -1:
                # allow negated "not ... improvement" phrasing (Task-2 compliant)
                window_start = max(0, idx - 60)
                window = low[window_start:idx]
                if "not" in window or "n't" in window or "cannot" in window or "never" in window:
                    idx = low.find(s.lower(), idx + 1)
                    continue
                failures.append(f"{rel} CONTAINS FORBIDDEN {s!r}")
                break

# verify JSON itself matches ground truth headline numbers
assert adult["conditions"]["deterministic"]["holdout_scores"] == [0.7937, 0.8379, 0.8127]
assert adult["conditions"]["deterministic"]["holdout_mean"] == 0.8148
assert adult["conditions"]["deterministic"]["holdout_std"] == 0.0181
assert adult["conditions"]["llm_guided"]["holdout_scores"] == [0.798, 0.8379, 0.83]
assert adult["conditions"]["llm_guided"]["holdout_mean"] == 0.822
assert adult["conditions"]["llm_guided"]["holdout_std"] == 0.0173
assert wine["conditions"]["deterministic"]["holdout_scores"] == [0.5461, 0.5289, 0.5881]
assert wine["conditions"]["deterministic"]["holdout_mean"] == 0.5544
assert wine["conditions"]["deterministic"]["holdout_std"] == 0.0249
assert wine["conditions"]["llm_guided"]["holdout_scores"] == [0.5461, 0.5285, 0.5921]
assert wine["conditions"]["llm_guided"]["holdout_mean"] == 0.5556
assert wine["conditions"]["llm_guided"]["holdout_std"] == 0.0268
assert adult["rows_total"] == 1000 and adult["rows_train"] == 800 and adult["rows_holdout"] == 200
assert wine["rows_total"] == 800 and wine["rows_train"] == 640 and wine["rows_holdout"] == 160
assert proto["n_trials"] == 0 and proto["seeds"] == [42, 123, 456] and proto["n_folds"] == 3

if failures:
    print("FAIL:")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("PASS: docs consistency check — JSON-parsed headline numbers present, no stale/forbidden strings.")
