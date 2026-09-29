"""Central model registry with hyperparameter search spaces for Optuna."""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB


@dataclass
class ModelDefinition:
    """Definition of an ML model with its metadata and search space."""
    name: str
    display_name: str
    task: str  # classification or regression
    model_class: type
    default_params: dict[str, Any] = field(default_factory=dict)
    search_space: Optional[Callable] = None
    supported_metrics: list[str] = field(default_factory=list)
    explainability_method: str = "permutation"

    def create_model(self, **override_params) -> Any:
        params = {**self.default_params, **override_params}
        return self.model_class(**params)


# ── Search Space Functions ──────────────────────────────────
def _lr_space(trial):
    return {
        "C": trial.suggest_float("C", 0.01, 100.0, log=True),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"]),
    }

def _dt_clf_space(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }

def _rf_clf_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
    }

def _gb_clf_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }

def _hgb_clf_space(trial):
    return {
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
    }

def _knn_space(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
    }

def _svc_space(trial):
    return {
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
    }

def _nb_space(trial):
    return {
        "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True),
    }

def _linreg_space(trial):
    return {
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
    }

def _ridge_space(trial):
    return {
        "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
    }

def _lasso_space(trial):
    return {
        "alpha": trial.suggest_float("alpha", 0.001, 10.0, log=True),
    }

def _dt_reg_space(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }

def _rf_reg_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
    }

def _gb_reg_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }

def _hgb_reg_space(trial):
    return {
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
    }

def _knn_reg_space(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
    }

def _svr_space(trial):
    return {
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
    }


class ModelRegistry:
    """Central registry of all supported ML models."""

    def __init__(self):
        self._models: dict[str, ModelDefinition] = {}
        self._register_all()

    def _register_all(self):
        # ── Classification ──
        self._register(ModelDefinition(
            name="logistic_regression", display_name="Logistic Regression",
            task="classification", model_class=LogisticRegression,
            default_params={"max_iter": 1000, "random_state": 42},
            search_space=_lr_space, explainability_method="coefficients",
        ))
        self._register(ModelDefinition(
            name="decision_tree_clf", display_name="Decision Tree",
            task="classification", model_class=DecisionTreeClassifier,
            default_params={"random_state": 42},
            search_space=_dt_clf_space, explainability_method="feature_importance",
        ))
        self._register(ModelDefinition(
            name="random_forest_clf", display_name="Random Forest",
            task="classification", model_class=RandomForestClassifier,
            default_params={"random_state": 42, "n_jobs": -1, "n_estimators": 100},
            search_space=_rf_clf_space, explainability_method="shap",
        ))
        self._register(ModelDefinition(
            name="gradient_boosting_clf", display_name="Gradient Boosting",
            task="classification", model_class=GradientBoostingClassifier,
            default_params={"random_state": 42},
            search_space=_gb_clf_space, explainability_method="shap",
        ))
        self._register(ModelDefinition(
            name="hist_gradient_boosting_clf", display_name="HistGradientBoosting",
            task="classification", model_class=HistGradientBoostingClassifier,
            default_params={"random_state": 42},
            search_space=_hgb_clf_space, explainability_method="shap",
        ))
        self._register(ModelDefinition(
            name="knn_clf", display_name="K-Nearest Neighbors",
            task="classification", model_class=KNeighborsClassifier,
            default_params={},
            search_space=_knn_space, explainability_method="permutation",
        ))
        self._register(ModelDefinition(
            name="svm_clf", display_name="Support Vector Machine",
            task="classification", model_class=SVC,
            default_params={"random_state": 42, "probability": True},
            search_space=_svc_space, explainability_method="permutation",
        ))
        self._register(ModelDefinition(
            name="naive_bayes", display_name="Naive Bayes",
            task="classification", model_class=GaussianNB,
            default_params={},
            search_space=_nb_space, explainability_method="permutation",
        ))

        # ── Regression ──
        self._register(ModelDefinition(
            name="linear_regression", display_name="Linear Regression",
            task="regression", model_class=LinearRegression,
            default_params={},
            search_space=_linreg_space, explainability_method="coefficients",
        ))
        self._register(ModelDefinition(
            name="ridge", display_name="Ridge Regression",
            task="regression", model_class=Ridge,
            default_params={"random_state": 42},
            search_space=_ridge_space, explainability_method="coefficients",
        ))
        self._register(ModelDefinition(
            name="lasso", display_name="Lasso Regression",
            task="regression", model_class=Lasso,
            default_params={"random_state": 42},
            search_space=_lasso_space, explainability_method="coefficients",
        ))
        self._register(ModelDefinition(
            name="decision_tree_reg", display_name="Decision Tree Regressor",
            task="regression", model_class=DecisionTreeRegressor,
            default_params={"random_state": 42},
            search_space=_dt_reg_space, explainability_method="feature_importance",
        ))
        self._register(ModelDefinition(
            name="random_forest_reg", display_name="Random Forest Regressor",
            task="regression", model_class=RandomForestRegressor,
            default_params={"random_state": 42, "n_jobs": -1, "n_estimators": 100},
            search_space=_rf_reg_space, explainability_method="shap",
        ))
        self._register(ModelDefinition(
            name="gradient_boosting_reg", display_name="Gradient Boosting Regressor",
            task="regression", model_class=GradientBoostingRegressor,
            default_params={"random_state": 42},
            search_space=_gb_reg_space, explainability_method="shap",
        ))
        self._register(ModelDefinition(
            name="hist_gradient_boosting_reg", display_name="HistGradientBoosting Regressor",
            task="regression", model_class=HistGradientBoostingRegressor,
            default_params={"random_state": 42},
            search_space=_hgb_reg_space, explainability_method="shap",
        ))
        self._register(ModelDefinition(
            name="knn_reg", display_name="K-Nearest Neighbors Regressor",
            task="regression", model_class=KNeighborsRegressor,
            default_params={},
            search_space=_knn_reg_space, explainability_method="permutation",
        ))
        self._register(ModelDefinition(
            name="svr", display_name="Support Vector Regressor",
            task="regression", model_class=SVR,
            default_params={},
            search_space=_svr_space, explainability_method="permutation",
        ))

    def _register(self, model_def: ModelDefinition):
        self._models[model_def.name] = model_def

    def get_models(self, task: str, model_names: list[str] | None = None) -> list[ModelDefinition]:
        if model_names:
            return [self._models[n] for n in model_names if n in self._models and self._models[n].task == task]
        return [m for m in self._models.values() if m.task == task]

    def get_model(self, name: str) -> Optional[ModelDefinition]:
        return self._models.get(name)

    def get_all_model_names(self, task: str) -> list[str]:
        return [m.name for m in self._models.values() if m.task == task]

    def get_fast_demo_models(self, task: str) -> list[str]:
        if task == "classification":
            return ["logistic_regression", "random_forest_clf", "hist_gradient_boosting_clf"]
        return ["ridge", "random_forest_reg", "hist_gradient_boosting_reg"]
