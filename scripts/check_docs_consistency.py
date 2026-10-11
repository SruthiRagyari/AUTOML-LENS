"""Check README.md, docs/*.md, and docs/BENCHMARK_REPORT.html against benchmark_summary.json.

Compares parsed JSON values (means, population stds, per-seed scores,
row counts, n_trials, seeds) with the docs, and forbids stale numbers
and overclaiming wording outside docs/archive/.

Task-2 wording rules (no negation exceptions):
  * Forbidden exact phrases (case-insensitive): "LLM win", "Det win",
    "LLM advantage", "Deterministic advantage", "LLM improvement",
    "deterministic baseline advantage".
  * "improvement" is forbidden on any line that also contains "llm" and
    either "0.0072" or "adult".
  * Literal row-count strings "800 train / 200 holdout",
    "640 train / 160 holdout", "1,000" must appear in README.md and
    docs/FINAL_RESULTS.md; the Wine total must appear as "800 rows" or
    "800 (640 train / 160 holdout".

Usage: python scripts/check_docs_consistency.py [ROOT]
ROOT defaults to the repository root; pass a scratch root to prove the
checker FAILS on a copy containing violations.
"""
import json
import pathlib
import sys

ROOT = (pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1
        else pathlib.Path(__file__).resolve().parents[1])
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
    # Literal row-count strings required by Task 2 (no loose variants):
    # Adult total must appear as "1,000" and its split as
    # "800 train / 200 holdout"; Wine split as "640 train / 160 holdout",
    # and the Wine total as "800 rows" or "800 (640 train / 160 holdout".
    expect_doc_contains(rel, text, "adult total literal '1,000'", ["1,000"])
    expect_doc_contains(rel, text, "adult split literal '800 train / 200 holdout'",
                        ["800 train / 200 holdout"])
    expect_doc_contains(rel, text, "wine split literal '640 train / 160 holdout'",
                        ["640 train / 160 holdout"])
    expect_doc_contains(rel, text, "wine total '800 rows' or '800 (640 train / 160 holdout'",
                        ["800 rows", "800 (640 train / 160 holdout"])
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

# Wording scope: README.md, every docs/*.md outside docs/archive/, and the
# generated docs/BENCHMARK_REPORT.html (its wording was neutralized by hand
# along with the interpretation strings in benchmark_summary.json).
texts = {}
texts["README.md"] = (ROOT / "README.md").read_text(encoding="utf-8")
for md in sorted((ROOT / "docs").glob("*.md")):
    texts[f"docs/{md.name}"] = md.read_text(encoding="utf-8")
html_path = ROOT / "docs" / "BENCHMARK_REPORT.html"
if html_path.exists():
    texts["docs/BENCHMARK_REPORT.html"] = html_path.read_text(encoding="utf-8")

# Forbidden exact phrases (case-insensitive). No negation exceptions: these
# phrasings overclaim regardless of surrounding words, so docs must avoid
# them entirely and use neutral "difference" wording instead.
FORBIDDEN_PHRASES = ["llm win", "det win", "llm advantage",
                     "deterministic advantage", "llm improvement",
                     "deterministic baseline advantage"]

for rel, text in texts.items():
    for s in stale:
        if s in text:
            failures.append(f"{rel} CONTAINS STALE {s!r}")
    low = text.lower()
    for s in FORBIDDEN_PHRASES:
        if s in low:
            failures.append(f"{rel} CONTAINS FORBIDDEN PHRASE {s!r}")
    # "improvement" is banned on any line tying it to the LLM delta claim;
    # no negation exception (rewrite as neutral "difference" instead).
    for lineno, line in enumerate(text.splitlines(), 1):
        lline = line.lower()
        if ("improvement" in lline and "llm" in lline
                and ("0.0072" in lline or "adult" in lline)):
            failures.append(
                f"{rel}:{lineno} CONTAINS 'improvement' ON LLM DELTA LINE: "
                f"{line.strip()[:120]}")

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
