"""Input validation utilities."""
import re
from typing import Any


def sanitize_string(value: str, max_length: int = 255) -> str:
    """Sanitize string input."""
    if not isinstance(value, str):
        return str(value)[:max_length]
    value = re.sub(r'[<>{}|\\^`]', '', value)
    return value.strip()[:max_length]


def validate_column_name(name: str, available: list[str]) -> bool:
    """Validate column name exists in dataset."""
    return name in available


def validate_metric(metric: str, problem_type: str) -> str:
    """Validate and return appropriate metric."""
    clf_metrics = {"accuracy", "f1_weighted", "precision_weighted",
                   "recall_weighted", "balanced_accuracy", "roc_auc"}
    reg_metrics = {"r2", "neg_mean_squared_error", "neg_mean_absolute_error",
                   "neg_root_mean_squared_error", "mse", "rmse", "mae"}

    if problem_type == "classification":
        return metric if metric in clf_metrics else "f1_weighted"
    return metric if metric in reg_metrics else "neg_root_mean_squared_error"
