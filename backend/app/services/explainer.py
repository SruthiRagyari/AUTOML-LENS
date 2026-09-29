"""Model explainability using SHAP, feature importance, and permutation importance."""
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


@dataclass
class ExplainabilityResult:
    method_used: str
    feature_importance: list[dict[str, Any]]
    top_features: list[str]
    explanation_text: str
    shap_values: Optional[np.ndarray] = None


class Explainer:
    """Generate model explanations using multiple methods."""

    @staticmethod
    def explain(model, X_test, feature_names: list[str],
                y_test=None, method: str = "auto") -> ExplainabilityResult:
        """Explain model predictions."""
        if isinstance(X_test, pd.DataFrame):
            X_arr = X_test.values
        else:
            X_arr = X_test

        # Align feature names with data
        if len(feature_names) != X_arr.shape[1]:
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

        if method == "auto":
            method = Explainer._select_method(model)

        result = None

        if method == "shap":
            result = Explainer._shap_explain(model, X_arr, feature_names)

        if result is None and hasattr(model, "feature_importances_"):
            result = Explainer._feature_importance(model, feature_names)

        if result is None and hasattr(model, "coef_"):
            result = Explainer._coefficients(model, feature_names)

        if result is None and y_test is not None:
            result = Explainer._permutation(model, X_arr, y_test, feature_names)

        if result is None:
            # Last resort: try feature importance
            if hasattr(model, "feature_importances_"):
                result = Explainer._feature_importance(model, feature_names)
            else:
                result = ExplainabilityResult(
                    method_used="none",
                    feature_importance=[],
                    top_features=[],
                    explanation_text="Explainability not available for this model type.",
                )

        return result

    @staticmethod
    def _select_method(model) -> str:
        model_name = type(model).__name__.lower()
        tree_models = [
            "randomforest", "gradientboosting", "histgradientboosting",
            "decisiontree", "xgb", "lgbm",
        ]
        if any(t in model_name for t in tree_models):
            return "shap"
        if hasattr(model, "coef_"):
            return "coefficients"
        if hasattr(model, "feature_importances_"):
            return "feature_importance"
        return "permutation"

    @staticmethod
    def _shap_explain(model, X, feature_names) -> Optional[ExplainabilityResult]:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:min(200, len(X))])

            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values

            mean_abs = np.abs(sv).mean(axis=0)
            fi = sorted(
                [{"feature": n, "importance": round(float(v), 6)}
                 for n, v in zip(feature_names, mean_abs)],
                key=lambda x: abs(x["importance"]), reverse=True
            )
            top = [x["feature"] for x in fi[:10]]
            text = Explainer._make_text(fi[:5])

            return ExplainabilityResult("shap", fi, top, text, sv)
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")
            return None

    @staticmethod
    def _feature_importance(model, feature_names) -> ExplainabilityResult:
        imp = model.feature_importances_
        fi = sorted(
            [{"feature": n, "importance": round(float(v), 6)}
             for n, v in zip(feature_names, imp)],
            key=lambda x: abs(x["importance"]), reverse=True
        )
        top = [x["feature"] for x in fi[:10]]
        return ExplainabilityResult("feature_importance", fi, top, Explainer._make_text(fi[:5]))

    @staticmethod
    def _coefficients(model, feature_names) -> ExplainabilityResult:
        coefs = model.coef_
        if coefs.ndim > 1:
            coefs = coefs[0]
        fi = sorted(
            [{"feature": n, "importance": round(float(v), 6)}
             for n, v in zip(feature_names, coefs)],
            key=lambda x: abs(x["importance"]), reverse=True
        )
        top = [x["feature"] for x in fi[:10]]
        return ExplainabilityResult("coefficients", fi, top, Explainer._make_text(fi[:5]))

    @staticmethod
    def _permutation(model, X, y, feature_names) -> ExplainabilityResult:
        res = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=1)
        fi = sorted(
            [{"feature": n, "importance": round(float(v), 6)}
             for n, v in zip(feature_names, res.importances_mean)],
            key=lambda x: abs(x["importance"]), reverse=True
        )
        top = [x["feature"] for x in fi[:10]]
        return ExplainabilityResult("permutation", fi, top, Explainer._make_text(fi[:5]))

    @staticmethod
    def _make_text(top_features: list[dict]) -> str:
        if not top_features:
            return "No features available for explanation."
        names = [f"'{f['feature']}'" for f in top_features]
        if len(names) > 1:
            feat_str = ", ".join(names[:-1]) + f", and {names[-1]}"
        else:
            feat_str = names[0]
        return (
            f"The features {feat_str} had a strong influence on the model's predictions. "
            f"These attributes are the primary drivers of the model's decision-making process."
        )
