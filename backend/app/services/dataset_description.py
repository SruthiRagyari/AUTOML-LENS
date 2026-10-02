"""Compact, structured dataset description for LLM prompts.

The profiler already produces rich per-column stats; feeding that JSON blob
verbatim to a model wastes tokens and still hides the few facts that matter for
feature engineering. This module reshapes the profiler output plus light column
statistics into a small dictionary that is:

* structured (stable keys, native JSON types, no numpy objects),
* truncated (max columns, max sample values, max characters),
* paired with the operation catalog the model is allowed to choose from.

Nothing here reads the target's relationship to other columns: the description
must describe the *inputs*, not leak label information.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.services.feature_operations import (
    build_operation_catalog,
    operation_catalog_descriptions,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_COLUMNS = 40
DEFAULT_MAX_SAMPLE_VALUES = 3
DEFAULT_MAX_CHARS = 9000
_MAX_EXAMPLE_LEN = 48
_MAX_TARGET_LEVELS = 8


def _native(value: Any) -> Any:
    """Convert numpy/pandas scalars into JSON-serialisable Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, (np.ndarray,)):
        return [_native(v) for v in value.tolist()]
    if value is pd.NaT:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _shorten(value: Any, limit: int = _MAX_EXAMPLE_LEN) -> Any:
    value = _native(value)
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "..."
    return value


def _round(value: Any, digits: int = 4) -> Optional[float]:
    value = _native(value)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _numeric_stats(series: pd.Series) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    try:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
    except Exception:
        return stats
    if numeric.empty:
        return stats
    stats = {
        "mean": _round(numeric.mean()),
        "std": _round(numeric.std()),
        "min": _round(numeric.min()),
        "max": _round(numeric.max()),
        "median": _round(numeric.median()),
    }
    try:
        stats["skew"] = _round(float(numeric.skew()), 3)
    except Exception:
        pass
    return stats


def _column_entry(
    df: pd.DataFrame,
    profile: dict,
    *,
    with_samples: bool = True,
    with_stats: bool = True,
    max_sample_values: int = DEFAULT_MAX_SAMPLE_VALUES,
) -> dict[str, Any]:
    """One compact, JSON-safe column description."""
    name = profile.get("name")
    entry: dict[str, Any] = {
        "name": name,
        "dtype": profile.get("dtype"),
        "inferred_type": profile.get("inferred_type"),
        "unique_count": _native(profile.get("unique_count")),
        "missing_percentage": _round(profile.get("missing_percentage"), 2),
    }
    if with_samples:
        samples: list[Any] = []
        if name in df.columns:
            try:
                samples = [_shorten(v) for v in df[name].dropna().unique()[:max_sample_values]]
            except Exception:
                samples = []
        if not samples:
            samples = [_shorten(v) for v in (profile.get("sample_values") or [])[:max_sample_values]]
        entry["sample_values"] = samples
    if with_stats and entry.get("inferred_type") == "numerical" and name in df.columns:
        stats = _numeric_stats(df[name])
        if stats:
            entry["stats"] = stats
    return entry


def _target_summary(
    df: pd.DataFrame,
    target_column: str,
    target_profile: Optional[dict],
    problem_type: Optional[str],
) -> dict[str, Any]:
    """Describe the label itself (class balance / spread) for problem understanding."""
    summary: dict[str, Any] = {
        "column": target_column,
        "problem_type": problem_type,
        "inferred_type": (target_profile or {}).get("inferred_type"),
        "unique_count": _native((target_profile or {}).get("unique_count")),
        "missing_percentage": _round((target_profile or {}).get("missing_percentage"), 2),
    }
    if target_column not in df.columns:
        summary["note"] = "target column not present in the supplied frame"
        return summary
    series = df[target_column]
    if problem_type == "regression" or (target_profile or {}).get("inferred_type") == "numerical":
        summary["stats"] = _numeric_stats(series)
    try:
        counts = series.value_counts(dropna=False).head(_MAX_TARGET_LEVELS)
        summary["distribution"] = {str(_native(k)): int(v) for k, v in counts.items()}
    except Exception:
        pass
    return summary

def build_dataset_description(
    df: pd.DataFrame,
    profile: Optional[dict] = None,
    target_column: Optional[str] = None,
    problem_type: Optional[str] = None,
    task_description: Optional[str] = None,
    *,
    max_columns: int = DEFAULT_MAX_COLUMNS,
    max_sample_values: int = DEFAULT_MAX_SAMPLE_VALUES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """Build the structured description that provider prompts are generated from.

    Result keys: ``dataset``, ``target``, ``columns``,
    ``feature_operation_catalog``, ``allowed_operations``, ``task_description``
    and ``truncation``. ``feature_operation_catalog`` maps every candidate input
    column to the only operation names a model may choose for it - the model is
    never asked to invent an operation, and never offered the target column.
    """
    from app.services.profiler import DatasetProfiler  # local import: no cycle

    rows, cols = (tuple(df.shape) if hasattr(df, "shape") else (0, 0))
    if profile is None:
        profile = DatasetProfiler().profile(df) if rows else {
            "column_profiles": [], "warnings": [], "column_type_summary": {},
        }
    column_profiles = [p for p in (profile.get("column_profiles") or []) if isinstance(p, dict)]

    description: dict[str, Any] = {
        "dataset": {
            "rows": int(rows),
            "columns": int(cols),
            "duplicate_rows": _native(profile.get("duplicate_rows")),
            "total_missing_percentage": _round(profile.get("total_missing_percentage"), 2),
            "column_type_summary": profile.get("column_type_summary") or {},
            "memory_usage_mb": _round(profile.get("memory_usage_mb"), 3),
        },
        "target": None,
        "columns": [],
        "feature_operation_catalog": build_operation_catalog(column_profiles, target_column),
        "allowed_operations": operation_catalog_descriptions(),
        "task_description": (task_description or "")[:400] or None,
        "truncation": {"columns_omitted": 0, "sample_values_dropped": False,
                       "stats_dropped": False, "characters": 0},
    }

    if target_column:
        target_profile = next((p for p in column_profiles if p.get("name") == target_column), None)
        description["target"] = _target_summary(df, target_column, target_profile, problem_type)

    ordered = column_profiles[:max(0, int(max_columns))]
    description["truncation"]["columns_omitted"] = max(0, len(column_profiles) - len(ordered))
    description["columns"] = [
        _column_entry(df, p, max_sample_values=max_sample_values) for p in ordered
    ]

    _apply_char_budget(description, max_chars)
    description["truncation"]["characters"] = len(_dumps(description))
    return description


def _dumps(description: dict) -> str:
    return json.dumps(description, ensure_ascii=False, separators=(",", ":"), default=str)


def _apply_char_budget(description: dict, max_chars: int) -> None:
    """Shrink the description in place until it fits ``max_chars`` of JSON."""
    budget = max(1200, int(max_chars))
    truncation = description["truncation"]
    if len(_dumps(description)) <= budget:
        return

    # 1) numeric stats are the least load-bearing, drop them first.
    for entry in description["columns"]:
        entry.pop("stats", None)
    truncation["stats_dropped"] = True
    if len(_dumps(description)) <= budget:
        return

    # 2) then the example values.
    for entry in description["columns"]:
        entry.pop("sample_values", None)
    truncation["sample_values_dropped"] = True
    if len(_dumps(description)) <= budget:
        return

    # 3) finally drop trailing columns and record how many were cut.
    kept: list[dict] = []
    for entry in description["columns"]:
        kept.append(entry)
        if len(_dumps({**description, "columns": kept})) > budget:
            kept.pop()
            break
    dropped = len(description["columns"]) - len(kept)
    description["columns"] = kept
    truncation["columns_omitted"] = int(truncation["columns_omitted"]) + int(dropped)


def render_dataset_description(description: dict) -> str:
    """Render the description as deterministic prompt text (nothing is invented)."""
    return json.dumps(description, ensure_ascii=False, separators=(",", ":"), default=str)

