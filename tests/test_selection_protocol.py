"""Focused tests for the corrected selection / evaluation protocol.

The winner must be chosen from TRAINING/CV evidence only; the holdout is used
exactly once, afterwards, for the final unbiased evaluation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from app.services.model_registry import ModelRegistry
from app.services.trainer import TrainingManager, TrainingResult


# -- fixtures ---------------------------------------------------------------
@pytest.fixture
def clf_split():
    rng = np.random.RandomState(3)
    n = 140
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] * 1.5 + X[:, 1] - 0.5 * X[:, 2] > 0).astype(int)
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


@pytest.fixture
def reg_split():
    rng = np.random.RandomState(4)
    n = 140
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 4.0 - X[:, 1] * 2.0 + rng.normal(scale=0.2, size=n)
    return train_test_split(X, y, test_size=0.25, random_state=42)


def _result(name, *, cv=None, holdout=None, status="COMPLETED"):
    """Build a TrainingResult with CV evidence and a DELIBERATELY conflicting
    holdout score, so any test that ranks by holdout picks the wrong model."""
    r = TrainingResult(model_name=name, display_name=name.title())
    r.status = status
    r.optimization = {"best_cv_score": cv} if cv is not None else {}
    r.cv_scores = []
    r.optimized_metrics = holdout or {}
    r.baseline_metrics = holdout or {}
    return r


# -- 1. selection must not look at holdout scores ---------------------------
def test_selection_ignores_holdout_scores_entirely():
    """Two models: better CV, worse holdout. CV evidence must win."""
    good_cv = _result("good_cv", cv=0.90, holdout={"f1_weighted": 0.10})
    bad_cv = _result("bad_cv", cv=0.30, holdout={"f1_weighted": 0.99})
    best = TrainingManager.select_best_model([good_cv, bad_cv], "f1_weighted")
    assert best is not None
    assert best.model_name == "good_cv", "holdout score leaked into selection"


def test_selection_outcome_is_invariant_to_holdout_values():
    """Same pair, flipped holdout values -> selection must NOT change."""
    a = _result("a", cv=0.80, holdout={"f1_weighted": 0.05})
    b = _result("b", cv=0.60, holdout={"f1_weighted": 0.95})
    first = TrainingManager.select_best_model([a, b], "f1_weighted").model_name
    a.optimized_metrics = {"f1_weighted": 0.99}
    b.optimized_metrics = {"f1_weighted": 0.01}
    second = TrainingManager.select_best_model([a, b], "f1_weighted").model_name
    assert first == second == "a"


def test_selection_falls_back_to_cv_scores_when_no_optuna_evidence():
    a = _result("a", cv=None, holdout={"f1_weighted": 0.99})
    a.cv_scores = [0.70, 0.72, 0.74]
    b = _result("b", cv=None, holdout={"f1_weighted": 0.10})
    b.cv_scores = [0.30, 0.31, 0.32]
    best = TrainingManager.select_best_model([a, b], "f1_weighted")
    assert best.model_name == "a"
    assert best.optimized_metrics["f1_weighted"] == 0.99  # misleading, ignored


def test_model_without_cv_evidence_is_never_selected():
    """No CV evidence -> not selectable; never given a fabricated score."""
    no_cv = _result("no_cv", cv=None, holdout={"f1_weighted": 1.0})
    with_cv = _result("with_cv", cv=0.10, holdout={"f1_weighted": 0.2})
    best = TrainingManager.select_best_model([no_cv, with_cv], "f1_weighted")
    assert best.model_name == "with_cv"
    assert TrainingManager._selection_cv_score(no_cv) is None
    assert TrainingManager._selection_cv_score(with_cv) == 0.10


def test_all_failing_models_yield_no_selection():
    results = [_result("a", status="FAILED"), _result("b", status="FAILED")]
    assert TrainingManager.select_best_model(results, "f1_weighted") is None


# -- 2. regression direction on neg_* CV scores -----------------------------
def test_regression_selection_maximizes_negated_rmse():
    """neg_* CV scores are negative; the LESS negative (better RMSE) must win,
    even though numerically it is smaller."""
    better = _result("better", cv=-1.0, holdout={"rmse": 999.0})
    worse = _result("worse", cv=-50.0, holdout={"rmse": 1.0})
    best = TrainingManager.select_best_model(
        [worse, better], "neg_root_mean_squared_error")
    assert best.model_name == "better"
    assert best.optimization["best_cv_score"] == -1.0


def test_regression_selection_uses_raw_loss_direction():
    """A raw (un-negated) loss must be MINIMIZED."""
    better = _result("better", cv=1.0, holdout={"mean_squared_error": 999.0})
    worse = _result("worse", cv=50.0, holdout={"mean_squared_error": 1.0})
    best = TrainingManager.select_best_model(
        [better, worse], "mean_squared_error")
    assert best.model_name == "better"


def test_classification_and_regression_both_select_correctly(clf_split, reg_split):
    Xtr, Xte, ytr, yte = clf_split
    reg = ModelRegistry()
    clf_defs = reg.get_models('classification',
                              ['logistic_regression', 'random_forest_clf'])
    t = TrainingManager(Xtr, Xte, ytr, yte, clf_defs, "classification",
                        "f1_weighted", n_folds=3, n_trials=3, mode="automl")
    results = t.train_all()
    best = t.select_best_model(results, "f1_weighted")
    assert best is not None
    scores = [r.optimization["best_cv_score"] for r in results]
    assert best.optimization["best_cv_score"] == max(scores)

    Xtr, Xte, ytr, yte = reg_split
    reg_defs = reg.get_models('regression', ['ridge', 'random_forest_reg'])
    t2 = TrainingManager(Xtr, Xte, ytr, yte, reg_defs, "regression",
                         "neg_root_mean_squared_error", n_folds=3, n_trials=3,
                         mode="automl")
    results2 = t2.train_all()
    best2 = t2.select_best_model(results2, "neg_root_mean_squared_error")
    scores2 = [r.optimization["best_cv_score"] for r in results2]
    assert best2.optimization["best_cv_score"] == max(scores2)


# -- 3. failed candidates do not break the experiment -----------------------
def test_failed_candidates_are_excluded_but_survivors_still_selected(clf_split):
    Xtr, Xte, ytr, yte = clf_split
    reg = ModelRegistry()
    good = reg.get_models('classification', ['logistic_regression'])[0]

    broken_result = TrainingResult(model_name="broken", display_name="Broken")
    broken_result.status = "FAILED"
    broken_result.error_message = "RuntimeError: cannot build"
    broken_result.optimization = {
        "status": "FAILED", "error": "RuntimeError: cannot build"}

    alive = TrainingManager(
        Xtr, Xte, ytr, yte, [good], "classification", "f1_weighted",
        n_folds=2, n_trials=2, mode="automl").train_all()

    best = TrainingManager.select_best_model([broken_result, *alive], "f1_weighted")
    assert best is not None and best.model_name == good.name

    prov = TrainingManager.build_selection_provenance(
        [broken_result, *alive], "f1_weighted", best)
    assert prov["selected_model"] == good.name
    # The failure is recorded, not hidden.
    broken_entry = next(c for c in prov["candidates_considered"]
                        if c["model_name"] == "broken")
    assert broken_entry["status"] == "FAILED"
    assert broken_entry["selected"] is False


# -- 4. selection provenance ------------------------------------------------
def test_selection_provenance_has_all_required_fields():
    winner = _result("winner", cv=0.88, holdout={"f1_weighted": 0.71})
    loser = _result("loser", cv=0.55, holdout={"f1_weighted": 0.69})
    prov = TrainingManager.build_selection_provenance(
        [winner, loser], "f1_weighted", winner)

    assert prov["selection_metric"] == "f1_weighted"
    assert prov["selection_scoring"] == "f1_weighted"
    assert prov["selection_direction"] == "maximize"
    assert prov["selected_model"] == "winner"
    assert prov["selected_model_cv_score"] == 0.88
    # The holdout score is kept separate and clearly labelled.
    assert prov["final_holdout_metrics"]["f1_weighted"] == 0.71
    assert prov["holdout_used_for_selection"] is False
    assert prov["evidence_source"] == "cross_validation_on_training_split"
    # Both candidates are listed with their CV evidence.
    assert len(prov["candidates_considered"]) == 2
    assert sum(c["selected"] for c in prov["candidates_considered"]) == 1


def test_selection_provenance_reports_null_when_nothing_selected():
    prov = TrainingManager.build_selection_provenance(
        [_result("a", status="FAILED")], "f1_weighted", None)
    assert prov["selected_model"] is None
    assert prov["selected_model_cv_score"] is None
    assert prov["final_holdout_metrics"] is None


def test_cv_score_and_holdout_score_are_never_conflated():
    """The two numbers must stay distinct all the way into provenance."""
    winner = _result("winner", cv=0.42, holdout={"f1_weighted": 0.99})
    prov = TrainingManager.build_selection_provenance(
        [winner], "f1_weighted", winner)
    assert prov["selected_model_cv_score"] == 0.42            # decided it
    assert prov["final_holdout_metrics"]["f1_weighted"] == 0.99  # reported it
    assert prov["selected_model_cv_score"] != \
        prov["final_holdout_metrics"]["f1_weighted"]


def test_regression_selection_provenance_direction():
    winner = _result("winner", cv=-2.5, holdout={"rmse": 3.1})
    prov = TrainingManager.build_selection_provenance(
        [winner], "neg_root_mean_squared_error", winner)
    assert prov["selection_scoring"] == "neg_root_mean_squared_error"
    assert prov["selection_direction"] == "maximize"     # negated value
    assert prov["selection_raw_direction"] == "minimize"  # reported RMSE
    assert prov["selected_model_cv_score"] == -2.5

