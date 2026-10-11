"""Detect measured-number drift between main and the working copy.

For each tracked benchmark document, extract every number from the version
on main (``git show main:<path>``) and from the working copy, then print any
number that appears in one but not the other. A clean review means no
measured figure changed while wording was edited.

Usage (from the repository root):
    python scripts/check_number_drift.py
"""
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "docs/BENCHMARK_REPORT.md",
    "docs/BENCHMARK_REPORT.html",
    "docs/LLM_EVALUATION.md",
]

# Integers, decimals and scientific notation (sign kept: -0.0012 != 0.0012).
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def extract(text: str) -> set:
    return set(NUMBER_RE.findall(text))


def from_main(path: str) -> str:
    out = subprocess.run(
        ["git", "show", f"main:{path}"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    )
    return out.stdout.decode("utf-8", errors="replace")


def main() -> int:
    drifted = 0
    for path in FILES:
        old = extract(from_main(path))
        working = extract(
            (REPO_ROOT / path).read_text(encoding="utf-8", errors="replace"))
        only_main = sorted(old - working)
        only_work = sorted(working - old)
        if not only_main and not only_work:
            print(f"OK    {path}: {len(old)} distinct number(s), no drift")
            continue
        drifted += 1
        print(f"DRIFT {path}:")
        for n in only_main:
            print(f"  only in main        : {n}")
        for n in only_work:
            print(f"  only in working copy: {n}")
    if drifted:
        print(f"FAIL: number drift in {drifted} file(s)")
        return 1
    print("PASS: no number differs between main and the working copy")
    return 0


if __name__ == "__main__":
    sys.exit(main())