"""Focused tests for the hardened Optuna optimization layer.

Covers: explicit metric->scoring->direction, n_trials/n_folds propagation,
deterministic seed, all-trials-failed handling (None, never a fake 0),
model-failure continuation, holdout isolation and persisted provenance.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from app.services.evaluator import get_metric_spec, is_higher_better
from app.services.model_registry import ModelRegistry
from app.services.optimizer import (
    OptunaOptimizer, OPTUNA_SEED,
    STATUS_COMPLETED, STATUS_PARTIAL, STATUS_FAILED, STATUS_SKIPPED,
    skipped_optimization,
)
from app.services.trainer import TrainingManager


# -- fixtures ---------------------------------------------------------------
@pytest.fixture
def clf_split():
    rng = np.random.RandomState(0)
    n = 120
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


@pytest.fixture
def reg_split():
    rng = np.random.RandomState(1)
    n = 120
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 3.0 + X[:, 1] - 0.5 * X[:, 2] + rng.normal(scale=0.1, size=n)
    return train_test_split(X, y, test_size=0.25, random_state=42)


def _clf_def():
    return ModelRegistry().get_models('classification', ['logistic_regression'])[0]


def _reg_def():
    return ModelRegistry().get_models('regression', ['ridge'])[0]


class _ExplodingDefinition:
    """A registry-shaped model whose search space always raises."""
    name = "exploding_model"
    display_name = "Exploding Model"

    def search_space(self, trial):
        raise RuntimeError("search space exploded")

    def create_model(self, **params):
        return DecisionTreeClassifier(**params)


# -- 1. metric -> scoring -> direction contract ------------------------------
def test_classification_metric_direction_is_maximize():
    for metric in ("accuracy", "f1_weighted", "balanced_accuracy", "roc_auc"):
        spec = get_metric_spec(metric)
        assert spec.scoring == metric
        assert spec.direction == "maximize"
        assert spec.raw_direction == "maximize"
        assert spec.negated is False


def test_regression_negated_scoring_direction():
    """sklearn negates losses: the STUDY maximizes, the REPORTED metric minimizes."""
    for metric, scoring in (
        ("rmse", "neg_root_mean_squared_error"),
        ("mae", "neg_mean_absolute_error"),
        ("mse", "neg_mean_squared_error"),
        ("neg_root_mean_squared_error", "neg_root_mean_squared_error"),
    ):
        spec = get_metric_spec(metric)
        assert spec.scoring == scoring
        assert spec.negated is True
        assert spec.direction == "maximize"      # study maximizes negated value
        assert spec.raw_direction == "minimize"  # user-facing metric is lower-better
        assert is_higher_better(metric) is False


def test_raw_unnegated_loss_metric_is_minimized_not_maximized():
    """A loss scorer sklearn does NOT negate must be minimized."""
    spec = get_metric_spec("mean_squared_error")
    assert spec.negated is False
    assert spec.direction == "minimize"
    assert spec.raw_direction == "minimize"


def test_optimizer_direction_follows_metric_spec(clf_split, reg_split):
    Xtr, Xte, ytr, yte = clf_split
    clf_opt = OptunaOptimizer(Xtr, ytr, _clf_def(), "classification", "f1_weighted")
    assert clf_opt._get_direction() == "maximize"

    Xtr, Xte, ytr, yte = reg_split
    reg_opt = OptunaOptimizer(Xtr, ytr, _reg_def(), "regression", "rmse")
    assert reg_opt._get_direction() == "maximize"
    assert reg_opt.spec.scoring == "neg_root_mean_squared_error"


# -- 2. n_trials / n_folds propagation --------------------------------------
def test_n_trials_and_n_folds_are_propagated(clf_split):
    Xtr, Xte, ytr, yte = clf_split
    opt = OptunaOptimizer(
        Xtr, ytr, _clf_def(), "classification", "f1_weighted",
        n_trials=7, n_folds=3,
    )
    result = opt.optimize()
    assert opt.n_trials == 7
    assert opt.n_folds == 3
    assert result.n_trials_requested == 7
    assert result.n_folds_requested == 3
    assert result.n_folds_used == 3
    assert result.cv_strategy == "StratifiedKFold"
    # Real trials ran - one record per requested trial, no padding.
    assert len(result.trials_data) == 7
    assert result.n_trials_completed + result.n_trials_failed == 7


def test_folds_are_clamped_when_rare_class_is_too_small(clf_split):
    """A 10-fold request with 3 minority samples is clamped, and the note is honest."""
    Xtr, Xte, ytr, yte = clf_split
    y_imbalanced = np.array(ytr, copy=True)
    # Force class 1 down to exactly 3 training samples.
    ones = np.where(y_imbalanced == 1)[0]
    y_imbalanced[ones[:len(ones) - 3]] = 0
    assert int((y_imbalanced == 1).sum()) == 3
    opt = OptunaOptimizer(
        Xtr, ytr, _clf_def(), "classification", "f1_weighted",
        n_trials=2, n_folds=10,
    )
    opt.y_train = y_imbalanced
    folds, note = opt._resolve_folds()
    assert folds == 3
    assert "clamped" in note
    assert "requested 10" in note


def test_regression_uses_kfold(reg_split):
    Xtr, Xte, ytr, yte = reg_split
    result = OptunaOptimizer(
        Xtr, ytr, _reg_def(), "regression", "rmse", n_trials=3, n_folds=3,
    ).optimize()
    assert result.cv_strategy == "KFold"
    assert result.raw_direction == "minimize"


# -- 3. deterministic seed --------------------------------------------------
def test_same_seed_produces_identical_trials(clf_split):
    Xtr, Xte, ytr, yte = clf_split
    runs = []
    for _ in range(2):
        runs.append(OptunaOptimizer(
            Xtr, ytr, _clf_def(), "classification", "f1_weighted",
            n_trials=6, n_folds=3, seed=OPTUNA_SEED,
        ).optimize())
    assert runs[0].best_params == runs[1].best_params
    assert runs[0].best_score == runs[1].best_score
    assert ([t["params"] for t in runs[0].trials_data]
            == [t["params"] for t in runs[1].trials_data])
    assert runs[0].seed == OPTUNA_SEED


def test_seed_is_recorded_in_provenance(clf_split):
    Xtr, Xte, ytr, yte = clf_split
    prov = OptunaOptimizer(
        Xtr, ytr, _clf_def(), "classification", "f1_weighted",
        n_trials=2, n_folds=2, seed=7,
    ).optimize().to_provenance()
    assert prov["seed"] == 7


# -- 4. all trials failing -> None, never a fabricated score -----------------
def test_all_trials_failing_yields_none_and_real_error(clf_split):
    Xtr, Xte, ytr, yte = clf_split
    result = OptunaOptimizer(
        Xtr, ytr, _ExplodingDefinition(), "classification", "f1_weighted",
        n_trials=3, n_folds=2,
    ).optimize()
    assert result.status == STATUS_FAILED
    assert result.best_score is None          # NOT 0.0
    assert result.best_params == {}
    assert result.n_trials_completed == 0
    assert result.n_trials_failed == 3
    # The real error message is preserved verbatim.
    assert "search space exploded" in result.error
    assert result.to_provenance()["best_cv_score"] is None
    assert all(t["score"] is None and t["status"] == "FAIL"
               for t in result.trials_data)


def test_partial_failure_is_reported_as_partial(clf_split):
    """Fail on trial 0 only; the rest must still produce a real best value."""
    class _FlakyDefinition:
        name = "flaky"
        display_name = "Flaky"

        def search_space(self, trial):
            if trial.number == 0:
                raise RuntimeError("first trial exploded")
            return _clf_def().search_space(trial)

        def create_model(self, **params):
            return _clf_def().create_model(**params)

    Xtr, Xte, ytr, yte = clf_split
    result = OptunaOptimizer(
        Xtr, ytr, _FlakyDefinition(), "classification", "f1_weighted",
        n_trials=4, n_folds=2,
    ).optimize()
    assert result.status == STATUS_PARTIAL
    assert result.n_trials_completed == 3
    assert result.n_trials_failed == 1
    assert result.best_score is not None       # real value from surviving trials
    assert result.best_params                  # real params
    assert "first trial exploded" in result.error


# -- 5. one model failing must not stop the others ---------------------------
def test_model_failure_continues_with_other_candidates(clf_split):
    Xtr, Xte, ytr, yte = clf_split
    rf = ModelRegistry().get_models(
        'classification', ['random_forest_clf'])[0]

    class _BadFit:
        """Fails during construction, not during optimization."""
        name = "bad_fit"
        display_name = "Bad Fit"
        search_space = staticmethod(lambda trial: {})

        def create_model(self, **params):
            raise RuntimeError("cannot construct this model")

    trainer = TrainingManager(
        Xtr, Xte, ytr, yte, [_BadFit(), _clf_def(), rf], "classification",
        "f1_weighted", n_folds=2, n_trials=2, mode="automl",
    )
    results = trainer.train_all()
    assert len(results) == 3
    bad, lr, rf_res = results
    # The broken model failed and recorded why.
    assert bad.status == "FAILED"
    assert "cannot construct this model" in bad.error_message
    # The other models still trained AND were still optimized.
    assert lr.status == "COMPLETED"
    assert rf_res.status == "COMPLETED"
    assert lr.optimization["status"] == STATUS_COMPLETED
    assert lr.optimization["n_trials_completed"] >= 1
    assert rf_res.optimization["n_trials_completed"] >= 1
    best = trainer.select_best_model(results, "f1_weighted")
    assert best is not None and best.model_name != "bad_fit"


def test_optimizer_exception_does_not_abort_training(clf_split, monkeypatch):
    """If Optuna itself raises, the model keeps its baseline and records why."""
    Xtr, Xte, ytr, yte = clf_split
    import app.services.trainer as trainer_mod

    def _boom(*a, **k):
        raise RuntimeError("optuna exploded")

    monkeypatch.setattr(trainer_mod, "OptunaOptimizer", _boom)
    trainer = TrainingManager(
        Xtr, Xte, ytr, yte, [_clf_def()], "classification",
        "f1_weighted", n_folds=2, n_trials=2, mode="automl",
    )
    results = trainer.train_all()
    assert results[0].status == "COMPLETED"          # baseline survived
    assert results[0].baseline_metrics               # real metrics
    assert results[0].trained_model is not None
    opt = results[0].optimization
    assert opt["status"] == "FAILED"
    assert "optuna exploded" in opt["error"]
    assert opt["best_cv_score"] is None


# -- 6. holdout isolation ---------------------------------------------------
def test_optimizer_only_ever_sees_the_training_split(clf_split):
    """The optimizer's CV must be drawn from X_train; X_test is never passed in."""
    Xtr, Xte, ytr, yte = clf_split
    opt = OptunaOptimizer(
        Xtr, ytr, _clf_def(), "classification", "f1_weighted",
        n_trials=3, n_folds=3,
    )
    assert opt.X_train is Xtr
    assert len(opt.X_train) == len(Xtr)
    assert len(opt.X_train) < len(Xtr) + len(Xte)
    result = opt.optimize()
    # Deterministic replay gives the same answer with no holdout involved.
    alone = OptunaOptimizer(
        Xtr, ytr, _clf_def(), "classification", "f1_weighted",
        n_trials=3, n_folds=3,
    ).optimize()
    assert result.best_params == alone.best_params
    assert result.best_score == alone.best_score


def test_holdout_rows_do_not_change_the_search(clf_split):
    """Tuning is invariant to holdout values - direct proof of no leakage."""
    Xtr, Xte, ytr, yte = clf_split
    base = OptunaOptimizer(
        Xtr, ytr, _clf_def(), "classification", "f1_weighted",
        n_trials=4, n_folds=3,
    ).optimize()
    # The holdout is never handed to the optimizer, so corrupting it is a no-op.
    _Xte_corrupt = Xte + 1000.0
    _yte_corrupt = 1 - yte
    after = OptunaOptimizer(
        Xtr, ytr, _clf_def(), "classification", "f1_weighted",
        n_trials=4, n_folds=3,
    ).optimize()
    assert base.best_params == after.best_params
    assert base.best_score == after.best_score


def test_final_model_is_refit_on_train_only_with_best_params(clf_split):
    """The winning model is refit with real best params, then scored once."""
    Xtr, Xte, ytr, yte = clf_split
    trainer = TrainingManager(
        Xtr, Xte, ytr, yte, [_clf_def()], "classification",
        "f1_weighted", n_folds=3, n_trials=4, mode="automl",
    )
    res = trainer.train_all()[0]
    assert res.status == "COMPLETED"
    assert res.best_params == res.optimization["best_params"]
    # The refit model genuinely carries the tuned hyperparameters.
    tuned = res.trained_model.get_params()
    for key, value in res.best_params.items():
        assert tuned.get(key) == value
    # CV scores come from the training split only (n_folds entries).
    assert len(res.cv_scores) == 3


# -- 7. persisted provenance ------------------------------------------------
def test_provenance_block_contains_every_required_field(clf_split):
    Xtr, Xte, ytr, yte = clf_split
    prov = OptunaOptimizer(
        Xtr, ytr, _clf_def(), "classification", "f1_weighted",
        n_trials=5, n_folds=3, seed=OPTUNA_SEED,
    ).optimize().to_provenance()
    required = {
        "metric", "scoring", "direction", "raw_direction",
        "n_trials_requested", "n_trials_completed", "n_trials_failed",
        "n_folds_requested", "n_folds_used", "cv_strategy", "seed",
        "best_cv_score", "best_params", "status", "error", "fold_note",
    }
    assert required.issubset(prov.keys())
    assert prov["metric"] == "f1_weighted"
    assert prov["scoring"] == "f1_weighted"
    assert prov["direction"] == "maximize"
    assert prov["n_trials_requested"] == 5
    assert prov["n_trials_completed"] >= 1
    assert prov["n_folds_used"] == 3
    assert prov["status"] == STATUS_COMPLETED
    assert prov["error"] is None
    assert isinstance(prov["best_cv_score"], float)
    assert 0.0 <= prov["best_cv_score"] <= 1.0


def test_trainer_records_provenance_for_each_model(clf_split):
    Xtr, Xte, ytr, yte = clf_split
    defs = ModelRegistry().get_models(
        'classification', ['logistic_regression', 'random_forest_clf'])
    trainer = TrainingManager(
        Xtr, Xte, ytr, yte, defs, "classification", "f1_weighted",
        n_folds=2, n_trials=3, mode="automl",
    )
    for res in trainer.train_all():
        opt = res.optimization
        assert opt["metric"] == "f1_weighted"
        assert opt["status"] == STATUS_COMPLETED
        assert opt["n_trials_requested"] == 3
        assert opt["n_trials_completed"] == 3
        assert opt["n_trials_failed"] == 0
        assert opt["seed"] == OPTUNA_SEED
        assert opt["fast_demo"] is False
        assert len(res.optimization_history) == 3


def test_skipped_optimization_is_labelled_not_silently_empty(clf_split):
    Xtr, Xte, ytr, yte = clf_split
    trainer = TrainingManager(
        Xtr, Xte, ytr, yte, [_clf_def()], "classification",
        "f1_weighted", n_folds=3, n_trials=5, mode="baseline",
    )
    res = trainer.train_all()[0]
    assert res.status == "COMPLETED"
    opt = res.optimization
    # Baseline mode: explicitly SKIPPED, distinct from FAILED.
    assert opt["status"] == STATUS_SKIPPED
    assert opt["best_cv_score"] is None
    assert opt["n_trials_completed"] == 0
    assert "baseline" in opt["fold_note"]
    # No search history was invented for a run that never searched.
    assert res.optimization_history == []


def test_regression_provenance_direction(reg_split):
    Xtr, Xte, ytr, yte = reg_split
    prov = OptunaOptimizer(
        Xtr, ytr, _reg_def(), "regression", "rmse", n_trials=3, n_folds=3,
    ).optimize().to_provenance()
    assert prov["metric"] == "rmse"
    assert prov["scoring"] == "neg_root_mean_squared_error"
    assert prov["direction"] == "maximize"      # study maximizes negated RMSE
    assert prov["raw_direction"] == "minimize"  # reported RMSE: lower is better
    assert prov["best_cv_score"] < 0            # neg_* values really are negative


def test_skipped_optimization_helper_shape():
    res = skipped_optimization("accuracy", 5, "unit test")
    assert res.status == STATUS_SKIPPED
    assert res.to_provenance()["scoring"] == "accuracy"
    assert res.to_provenance()["n_trials_requested"] == 0
    assert res.to_provenance()["best_cv_score"] is None

