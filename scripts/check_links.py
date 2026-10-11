"""Check that every relative link in the public docs resolves (with
GitHub-accurate, case-sensitive matching) and that archived docs are not
referenced from their old paths.

Scope: README.md, every *.md / *.html under docs/ (docs/archive excluded from
link extraction but its files may be linked TO via their new archive path).

Usage (from the repository root):
    python scripts/check_links.py
"""
import os
import re
import sys
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

INLINE_MD = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_REF = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']")


def exists_case_sensitive(rel: str) -> bool:
    """Resolve rel (POSIX-style, relative to repo root) requiring exact case."""
    current = REPO_ROOT
    for part in rel.split("/"):
        if not part or part == ".":
            continue
        try:
            entries = os.listdir(current)
        except (NotADirectoryError, FileNotFoundError):
            return False
        if part not in entries:
            return False
        current = current / part
    return current.is_file() or current.is_dir()


def collect_targets(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = HTML_REF if path.suffix.lower() == ".html" else INLINE_MD
    for match in pattern.finditer(text):
        line_no = text.count("\n", 0, match.start()) + 1
        yield line_no, match.group(1).strip()


def main() -> int:
    files = [REPO_ROOT / "README.md"]
    files += sorted((REPO_ROOT / "docs").rglob("*.md"))
    files += sorted((REPO_ROOT / "docs").rglob("*.html"))

    checked = 0
    broken = []
    for path in files:
        if "docs" in path.parts and "archive" in path.parts:
            continue  # archived docs are read via the README archive entries
        rel_file = path.relative_to(REPO_ROOT).as_posix()
        for line_no, raw in collect_targets(path):
            target = raw
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            if not target or target.startswith(("#", "mailto:", "tel:",
                                                "data:", "http://",
                                                "https://", "//")):
                continue
            target = urllib.parse.unquote(target.split("#", 1)[0])
            if not target:
                continue
            checked += 1
            if target.startswith("/"):
                resolved = target.lstrip("/")
            else:
                base = path.parent.relative_to(REPO_ROOT).as_posix()
                resolved = urllib.parse.urljoin(base + "/", target)
            if not exists_case_sensitive(resolved):
                broken.append(f"{rel_file}:{line_no}: {raw}")

    if broken:
        print(f"FAIL: {len(broken)} broken link(s) out of {checked} checked:")
        for item in broken:
            print(f"  {item}")
        return 1
    print(f"PASS: {checked} relative link(s) across {len(files)} file(s) "
          f"all resolve (case-sensitive).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
