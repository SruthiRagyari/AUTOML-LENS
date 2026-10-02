"""Model evaluation with comprehensive metrics."""
import logging
from dataclasses import dataclass
import numpy as np
from typing import Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)

logger = logging.getLogger(__name__)

SUPPORTED_CLASSIFICATION_METRICS = [
    "accuracy", "f1_weighted", "precision_weighted",
    "recall_weighted", "balanced_accuracy", "roc_auc",
]
SUPPORTED_REGRESSION_METRICS = [
    "r2", "neg_mean_squared_error", "neg_mean_absolute_error",
    "neg_root_mean_squared_error",
]

# Map user-facing metric names to sklearn scoring strings
METRIC_TO_SCORING = {
    "accuracy": "accuracy",
    "f1_weighted": "f1_weighted",
    "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted",
    "balanced_accuracy": "balanced_accuracy",
    "roc_auc": "roc_auc",
    "r2": "r2",
    "mse": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "neg_mean_squared_error": "neg_mean_squared_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
    "neg_root_mean_squared_error": "neg_root_mean_squared_error",
}


# Metrics where a LARGER raw value is WORSE (errors / losses). Every other
# metric in the registry (accuracy, f1_weighted, r2, ...) is a score where a
# larger value is better.
LOWER_IS_BETTER_METRICS = frozenset({
    "mse", "rmse", "mae", "mape",
    "mean_squared_error", "mean_absolute_error", "root_mean_squared_error",
    "mean_squared_log_error", "neg_mean_squared_log_error",
    "neg_mean_squared_error", "neg_mean_absolute_error",
    "neg_root_mean_squared_error",
})


@dataclass(frozen=True)
class MetricSpec:
    """Explicit, persisted metric -> sklearn scoring -> direction contract.

    metric         user-facing metric name stored on the experiment.
    scoring        sklearn scoring string actually passed to cross_val_score.
    direction      direction Optuna must use on the value `scoring` returns.
                   sklearn negates every loss scorer (the ``neg_*`` family), so
                   the returned value is always "higher is better" and the
                   study direction is always ``maximize``. This is *derived*
                   from `negated`, never assumed, so a future raw (un-negated)
                   loss scorer cannot silently invert the study.
    raw_direction  direction of the un-negated metric the user sees in the
                   evaluation table (``minimize`` for rmse/mae/mse).
    negated        whether sklearn already flipped the sign for us.
    """

    metric: str
    scoring: str
    direction: str
    raw_direction: str
    negated: bool

    @property
    def is_higher_better(self) -> bool:
        return self.direction == "maximize"

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "scoring": self.scoring,
            "direction": self.direction,
            "raw_direction": self.raw_direction,
            "negated": self.negated,
        }


def is_higher_better(metric_name: str) -> bool:
    """Return True if higher RAW metric values are better."""
    return metric_name not in LOWER_IS_BETTER_METRICS


def get_scoring_string(metric_name: str) -> str:
    """Convert metric name to sklearn scoring string."""
    return METRIC_TO_SCORING.get(metric_name, metric_name)


def get_metric_spec(metric_name: str) -> MetricSpec:
    """Resolve metric -> sklearn scoring -> Optuna direction, explicitly.

    Single source of truth for "which direction do we optimize", shared by the
    optimizer, the trainer and the persisted provenance so the study direction
    can never drift from the metric that gets reported to the user.
    """
    scoring = get_scoring_string(metric_name)
    # A `neg_*` scorer already returns a sign-flipped value, so a bigger number
    # is better no matter what the underlying loss is. Any other
    # lower-is-better metric would arrive as a RAW loss and must be minimized.
    negated = scoring.startswith("neg_")
    direction = "maximize" if negated or is_higher_better(metric_name) else "minimize"
    return MetricSpec(
        metric=metric_name,
        scoring=scoring,
        direction=direction,
        raw_direction="maximize" if is_higher_better(metric_name) else "minimize",
        negated=negated,
    )


class Evaluator:
    """Comprehensive model evaluator."""

    @staticmethod
    def evaluate_classification(
        y_true, y_pred, y_proba=None
    ) -> dict[str, Any]:
        """Evaluate classification model with all relevant metrics."""
        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }

        # ROC-AUC
        if y_proba is not None:
            try:
                classes = np.unique(y_true)
                if len(classes) == 2:
                    if y_proba.ndim == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                    else:
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                    metrics["pr_auc"] = float(average_precision_score(y_true, y_proba[:, 1] if y_proba.ndim == 2 else y_proba))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    ))
            except Exception as e:
                logger.warning(f"ROC-AUC calculation failed: {e}")
                metrics["roc_auc"] = None

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Classification report
        try:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics["classification_report"] = report
        except Exception:
            pass

        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        metrics["class_distribution"] = {
            str(k): int(v) for k, v in zip(unique, counts)
        }

        return metrics

    @staticmethod
    def evaluate_regression(y_true, y_pred) -> dict[str, Any]:
        """Evaluate regression model with all relevant metrics."""
        metrics = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        }

        # MAPE
        try:
            mask = np.abs(y_true) > 1e-10
            if mask.sum() > 0:
                mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
                metrics["mape"] = mape
        except Exception:
            metrics["mape"] = None

        # Residuals summary
        residuals = np.array(y_true) - np.array(y_pred)
        metrics["residuals_summary"] = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
        }

        return metrics

    @staticmethod
    def get_metric_value(metrics: dict, metric_name: str) -> Optional[float]:
        """Extract a specific metric value."""
        return metrics.get(metric_name)
