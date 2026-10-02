"""Dataset suitability assessment.

Turns a real dataframe plus its profiler output into an explicit, auditable
answer to "can AutoML reasonably run on this file, and is the chosen target
usable?".

Everything here is measured from the data. Nothing is assumed, defaulted or
invented: when a value cannot be computed it is reported as ``None`` and the
reason is stated in ``task_reason`` / ``blocking_issues``.

The result is split into two levels on purpose:

``blocking_issues``
    The experiment cannot produce a meaningful model at all (no rows, no
    features, a target that is missing or has a single constant value). The
    UI uses these to stop the user before they waste a run.

``warnings``
    Usable, but worth knowing (small dataset, heavy missingness, extreme class
    imbalance, high-cardinality columns).
"""
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A class distribution longer than this stops being readable in the UI, so it
# is truncated and the truncation is reported rather than silently dropped.
MAX_DISTRIBUTION_CLASSES = 20

# Below this a stratified CV split with the default fold count is not meaningful.
MIN_ROWS_FOR_TRAINING = 10
SMALL_DATASET_WARNING_ROWS = 100


def _jsonable(value: Any) -> Any:
    """Convert numpy / pandas scalars so the payload survives JSON encoding."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


def assess_suitability(
    df: pd.DataFrame,
    column_profiles: list[dict],
    target_column: Optional[str] = None,
    problem_type: Optional[str] = None,
) -> dict[str, Any]:
    """Assess whether ``df`` can drive an AutoML run.

    ``column_profiles`` must be the profiler output for the same dataframe.
    ``target_column`` may be ``None``; pass whatever the profiler suggested, or
    nothing at all to be told what is still missing.
    """
    rows, cols = df.shape
    blocking: list[str] = []
    warnings: list[str] = []

    # ── structural checks ────────────────────────────────────────────────
    if rows == 0:
        blocking.append("The file has a header but no data rows.")
    if cols == 0:
        blocking.append("The file has no columns.")
    elif cols < 2:
        blocking.append(
            f"Only {cols} column found. AutoML needs at least one feature "
            "column in addition to the target."
        )

    # A feature column is usable when it is not empty, not constant and not a
    # pure identifier - the same rule the preprocessing engine applies.
    usable_features = [
        cp["name"] for cp in column_profiles
        if cp["inferred_type"] not in ("id_like", "constant")
    ]
    if cols >= 2 and not usable_features:
        blocking.append(
            "No usable feature columns: every remaining column is constant or "
            "looks like an identifier."
        )
    if rows and rows < MIN_ROWS_FOR_TRAINING:
        blocking.append(
            f"Only {rows} data row(s). Cross-validation needs at least "
            f"{MIN_ROWS_FOR_TRAINING} rows to form folds."
        )
    elif rows and rows < SMALL_DATASET_WARNING_ROWS:
        warnings.append(
            f"Only {rows} rows: results will be noisy and holdout metrics "
            "unstable. Treat them as indicative only."
        )

    total_missing_pct = 0.0
    if rows and cols:
        total_missing_pct = float(
            df.isnull().to_numpy().sum() / (rows * cols) * 100)
    if total_missing_pct > 40:
        warnings.append(
            f"{total_missing_pct:.1f}% of all cells are missing - most columns "
            "may be unusable."
        )

    target_info = _empty_target_info(target_column)

    if not target_column:
        blocking.append(
            "No target column selected. Pick a column to predict before "
            "starting an experiment."
        )
    elif target_column not in df.columns:
        blocking.append(
            f"Target column '{target_column}' is not present in the file."
        )
    else:
        _assess_target(df, column_profiles, target_column, target_info,
                       blocking, warnings)

    # ── task type ────────────────────────────────────────────────────────
    task_type = None
    task_reason = "Not determined: no usable target."
    if target_info["found"] and not blocking:
        if problem_type in ("classification", "regression"):
            task_type = problem_type
            task_reason = f"Task type supplied with the request: {problem_type}."
        else:
            task_type = _infer_task_type(target_info)
            task_reason = (
                "Derived from the target column profile: "
                f"{task_type} (inferred type "
                f"{target_info.get('inferred_type')}, "
                f"{target_info.get('unique_count')} distinct value(s))."
            )

    return {
        "rows": rows,
        "columns": cols,
        "usable_feature_count": len(usable_features),
        "total_missing_percentage": round(total_missing_pct, 2),
        "suitable": not blocking,
        "blocking_issues": blocking,
        "warnings": warnings,
        "target": target_info,
        "task_type": task_type,
        "task_reason": task_reason,
    }


def _empty_target_info(target_column: Optional[str]) -> dict[str, Any]:
    """A target block with every field explicitly unknown (never defaulted)."""
    return {
        "name": target_column,
        "found": False,
        "inferred_type": None,
        "dtype": None,
        "unique_count": None,
        "missing_count": None,
        "missing_percentage": None,
        "is_constant": None,
        "class_count": None,
        "distribution": None,
        "distribution_truncated": False,
        "min_class_share": None,
        "majority_class_share": None,
        "summary": None,
        "distribution_kind": None,
    }


def _assess_target(df, column_profiles, target_column, target_info,
                   blocking, warnings) -> None:
    """Fill ``target_info`` from the real column and record any problems."""
    series = df[target_column]
    cp = next((c for c in column_profiles if c["name"] == target_column), {})
    non_null = series.dropna()
    unique_count = int(non_null.nunique())

    target_info.update({
        "found": True,
        "inferred_type": cp.get("inferred_type"),
        "dtype": str(series.dtype),
        "unique_count": unique_count,
        "missing_count": int(series.isnull().sum()),
        "missing_percentage": round(
            float(series.isnull().sum() / max(len(df), 1) * 100), 2),
        "is_constant": unique_count <= 1,
    })

    if len(non_null) == 0:
        blocking.append(f"Target column '{target_column}' is entirely empty.")
        return
    if unique_count <= 1:
        blocking.append(
            f"Target column '{target_column}' has a single value "
            f"({_jsonable(non_null.iloc[0])!r}). There is nothing to learn."
        )
        return

    rows = len(df)
    # Only low-cardinality targets have a class balance worth reporting; a
    # continuous target gets descriptive statistics instead.
    if unique_count <= max(2, min(50, int(rows * 0.5))):
        counts = non_null.value_counts()
        target_info["distribution_kind"] = "classes"
        target_info["class_count"] = int(len(counts))
        truncated = len(counts) > MAX_DISTRIBUTION_CLASSES
        top = counts.head(MAX_DISTRIBUTION_CLASSES) if truncated else counts
        target_info["distribution"] = {
            str(_jsonable(k)): int(v) for k, v in top.items()
        }
        target_info["distribution_truncated"] = bool(truncated)
        total = float(counts.sum()) or 1.0
        shares = counts / total
        target_info["majority_class_share"] = round(float(shares.max()), 4)
        target_info["min_class_share"] = round(float(shares.min()), 4)
        target_info["summary"] = (
            f"{len(counts)} distinct value(s); the most common covers "
            f"{shares.max() * 100:.1f}% of rows."
        )
        rare = [str(_jsonable(k)) for k, v in counts.items() if v < 2]
        if rare:
            warnings.append(
                f"Target '{target_column}' has {len(rare)} value(s) with fewer "
                f"than 2 rows (e.g. {rare[:3]}); stratified folds cannot be "
                "built for those classes."
            )
        if float(shares.min()) < 0.05:
            warnings.append(
                f"Target '{target_column}' is imbalanced: the rarest class "
                f"covers only {float(shares.min()) * 100:.1f}% of rows."
            )
    else:
        target_info["distribution_kind"] = "numeric"
        target_info["summary"] = (
            f"Continuous target: min {_jsonable(non_null.min())}, "
            f"median {_jsonable(non_null.median())}, "
            f"max {_jsonable(non_null.max())}"
        )

    if target_info["missing_percentage"]:
        warnings.append(
            f"Target column '{target_column}' has "
            f"{target_info['missing_percentage']:.1f}% missing values; those "
            "rows are dropped before training."
        )
    if cp.get("inferred_type") in ("id_like", "constant"):
        warnings.append(
            f"'{target_column}' looks like an identifier or a constant, which "
            "rarely makes a useful prediction target."
        )


def _infer_task_type(target_info: dict[str, Any]) -> str:
    """Classification vs regression, decided from the real target profile.

    Mirrors :meth:`DatasetProfiler.detect_problem_type` so the profiler's
    suggestion and this endpoint never disagree.
    """
    itype = target_info.get("inferred_type")
    unique = target_info.get("unique_count") or 0
    if itype in ("categorical", "boolean"):
        return "classification"
    if itype == "numerical":
        return "classification" if unique <= 20 else "regression"
    return "classification" if unique <= 20 else "regression"
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value