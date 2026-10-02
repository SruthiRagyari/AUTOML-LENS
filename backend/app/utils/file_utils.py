"""File utility functions, including the shared tabular-data loader."""
import csv
import os
from pathlib import Path

import pandas as pd

EXCEL_EXTENSIONS = {".xlsx", ".xlsm", ".xls"}
CSV_EXTENSIONS = {".csv", ".txt", ".tsv", ".data"}

# Encodings tried in order when reading delimited text files. Real-world
# exports are frequently cp1252/latin-1 rather than utf-8.
_CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

# Delimiters considered when a delimited file does not look comma-separated.
# Restricted to real CSV field separators so that prose containing ordinary
# punctuation can never be mistaken for a delimiter.
_CANDIDATE_DELIMITERS = (",", ";", "\t", "|")

# How much text to sample when guessing the delimiter.
_SNIFF_SAMPLE_BYTES = 64 * 1024


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if needed."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_size(path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def safe_path_join(base: str, *parts: str) -> str:
    """Safely join path components preventing traversal."""
    result = os.path.normpath(os.path.join(base, *parts))
    if not result.startswith(os.path.normpath(base)):
        raise ValueError("Path traversal detected")
    return result


def is_excel_path(path: str | Path) -> bool:
    """True when the path points at an Excel workbook."""
    return Path(path).suffix.lower() in EXCEL_EXTENSIONS


def read_excel(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read an Excel workbook, raising an actionable error when the engine is missing."""
    try:
        return pd.read_excel(path, **kwargs)
    except ImportError as exc:  # optional engine (xlrd for legacy .xls) not installed
        raise RuntimeError(
            f"Cannot read '{Path(path).name}': legacy .xls workbooks require the "
            f"'xlrd' package (pip install xlrd==2.0.1). Details: {exc}"
        ) from exc


def _read_sample(path: str | Path, encoding: str) -> str:
    """Read a bounded text sample used only for delimiter guessing."""
    with open(path, "r", encoding=encoding, errors="replace") as fh:
        return fh.read(_SNIFF_SAMPLE_BYTES)


def detect_delimiter(path: str | Path, encoding: str = "utf-8-sig") -> str | None:
    """Return the delimiter that best splits this text file, or ``None``.

    A candidate is accepted only when it splits every sampled line into the
    *same* number of fields, and that number is greater than one. Parsing with
    each candidate (rather than counting characters) is what makes this safe:
    commas used as decimal marks or inside prose make the field count ragged,
    so such a file correctly reports no delimiter and keeps its columns intact.
    """
    try:
        sample = _read_sample(path, encoding)
    except OSError:
        return None
    if not sample.strip():
        return None

    # Sample raw lines; each is then parsed with every candidate delimiter.
    lines = [ln for ln in sample.splitlines() if ln.strip()][:20]
    if not lines:
        return None

    best: str | None = None
    best_fields = 1
    for candidate in _CANDIDATE_DELIMITERS:
        try:
            parsed = [next(csv.reader([line], delimiter=candidate))
                      for line in lines]
        except (csv.Error, StopIteration):
            continue
        widths = {len(fields) for fields in parsed}
        # Every line must split into the same number of fields, and a real
        # delimited dataset has more than one field per line.
        if len(widths) != 1:
            continue
        width = widths.pop()
        if width > best_fields:
            best, best_fields = candidate, width
    return best


def load_dataframe(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a tabular dataset (CSV/TSV/XLSX/XLS) with the parser matching the file.

    Excel workbooks are read with openpyxl (.xlsx/.xlsm) or xlrd (.xls).
    Delimited text is read as CSV, retrying common encodings so that a
    non-UTF-8 export does not fail the upload.

    A non-comma delimiter is detected automatically. ``pd.read_csv`` does not
    raise for such a file - it silently returns every line as a single column -
    so the result is inspected and re-read when it collapses to one column.
    """
    path = Path(path)
    if is_excel_path(path):
        return read_excel(path, **kwargs)

    last_error: Exception | None = None
    for encoding in _CSV_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=encoding, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
            continue

        # One column from a delimited file means the separator was wrong, not
        # that the dataset has a single field. Retry with the detected one.
        if df.shape[1] == 1:
            delimiter = detect_delimiter(path, encoding)
            if delimiter and delimiter != ",":
                try:
                    return pd.read_csv(
                        path, encoding=encoding, sep=delimiter, **kwargs
                    )
                except Exception:
                    pass
        return df

    # Last resort: let pandas sniff the delimiter (semicolon / tab exports).
    try:
        return pd.read_csv(path, encoding="latin-1", sep=None, engine="python", **kwargs)
    except Exception:
        if last_error is not None:
            raise last_error
        raise
