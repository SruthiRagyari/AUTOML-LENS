"""Check public-facing wording in README.md and docs/ for placeholder text,
overclaim/hype phrases, and academic-branding phrases that were deliberately
removed from the public surface (see commit 4e597e8).

docs/archive/ is excluded: it holds historical documents kept for reference.

Usage (from the repository root):
    python scripts/check_public_wording.py
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (pattern, human label) — matched case-insensitively.
FORBIDDEN = [
    # Placeholder / unfinished-work markers
    (r"\bTODO\b", "placeholder: TODO"),
    (r"\bFIXME\b", "placeholder: FIXME"),
    (r"\bTBD\b", "placeholder: TBD"),
    (r"lorem ipsum", "placeholder text"),
    (r"placeholder text", "placeholder text"),
    (r"\bcoming soon\b", "unfinished-status wording"),
    (r"\bwork in progress\b", "unfinished-status wording"),
    (r"not yet implemented", "unfinished-status wording"),
    # Overclaim / hype
    (r"world[- ]class", "hype: world-class"),
    (r"revolutionary", "hype: revolutionary"),
    (r"unparalleled", "hype: unparalleled"),
    (r"unbeatable", "hype: unbeatable"),
    (r"best[- ]in[- ]class", "hype: best-in-class"),
    (r"100\s*%\s*accura", "hype: 100% accurate"),
    (r"perfect accuracy", "hype: perfect accuracy"),
    (r"guaranteed (?:to |accuracy|results)", "hype: guaranteed"),
    (r"(?:dramatically|vastly|always) outperforms", "hype: outperforms claim"),
    (r"\bnever fails\b", "hype: never fails"),
    # Academic/student branding removed from the public surface
    (r"academic project", "branding: academic project"),
    (r"student project", "branding: student project"),
    (r"college project", "branding: college project"),
    (r"\bcoursework\b", "branding: coursework"),
    (r"course project", "branding: course project"),
    (r"Azure for Students", "branding: Azure for Students"),
    (r"\bsemester\b", "branding: semester"),
]


def collect_files():
    yield REPO_ROOT / "README.md"
    for path in sorted((REPO_ROOT / "docs").rglob("*")):
        if path.suffix.lower() not in (".md", ".html", ".txt", ".json"):
            continue
        if "archive" in path.relative_to(REPO_ROOT).parts:
            continue
        yield path


def main() -> int:
    hits = []
    scanned = 0
    for path in collect_files():
        if not path.is_file():
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(REPO_ROOT).as_posix()
        for pattern, label in FORBIDDEN:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                line_no = text.count("\n", 0, match.start()) + 1
                hits.append(f"{rel}:{line_no}: [{label}] {match.group(0)!r}")

    if hits:
        print(f"FAIL: {len(hits)} forbidden wording occurrence(s) "
              f"in {scanned} file(s) scanned:")
        for item in hits:
            print(f"  {item}")
        return 1
    print(f"PASS: public wording clean — {len(FORBIDDEN)} pattern(s), "
          f"{scanned} file(s) scanned, 0 hits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
