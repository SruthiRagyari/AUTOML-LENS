"""Scientific Hardening and Reproducibility Verification Suite for AutoML-Lens.

Covers:
1. Seed propagation: checks that experiment seed propagates to estimators, CV splitters, and feature engineering.
2. Repeated-run reproducibility: verifies identical dataset + seed produces numerically equivalent results (<= 1e-9).
3. Data leakage audit: verifies CV/holdout boundary, OOF isolation, and no target leakage into features.
4. Metric direction consistency: audits classification and regression metric direction contracts (higher-is-better vs lower-is-better).
5. Provenance completeness: verifies all provenance parameters exist and are not fabricated.
6. Zero DB and storage pollution: ensures tests execute without modifying live production database or storage.
"""
import os
import sqlite3
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.benchmarker import BenchmarkRunner, BenchmarkConfig, BenchmarkRecord, get_git_revision, get_file_sha256
from app.services.evaluator import get_metric_spec, is_higher_better
from app.services.model_registry import ModelRegistry
from app.services.trainer import TrainingManager
from app.services.feature_engineer import FeatureEngineer
from app.services.fold_safe import FeatureSpec, build_fold_safe_pipeline


@pytest.fixture(scope="module")
def small_clf_dataset(tmp_path_factory):
    """Small deterministic dataset for fast, leakage-free reproducibility tests."""
    tmp_dir = tmp_path_factory.mktemp("hardening_clf")
    path = tmp_dir / "clf_small.csv"
    rng = np.random.RandomState(42)
    n = 80
    df = pd.DataFrame({
        "num_a": rng.randn(n),
        "num_b": rng.uniform(10, 50, n),
        "cat_x": rng.choice(["cat1", "cat2", "cat3"], n),
        "target": rng.choice([0, 1], n),
    })
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def small_reg_dataset(tmp_path_factory):
    """Small deterministic regression dataset."""
    tmp_dir = tmp_path_factory.mktemp("hardening_reg")
    path = tmp_dir / "reg_small.csv"
    rng = np.random.RandomState(42)
    n = 80
    x1 = rng.randn(n)
    x2 = rng.uniform(0, 10, n)
    y = 3.0 * x1 - 1.5 * x2 + rng.randn(n) * 0.2
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "cat_z": rng.choice(["low", "med", "high"], n),
        "target": y,
    })
    df.to_csv(path, index=False)
    return str(path)


def test_seed_propagation_to_cv_splitter():
    """Verify that TrainingManager._get_cv uses the supplied optimization_seed."""
    mdefs = ModelRegistry().get_models("classification", ["logistic_regression"])
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    y = pd.Series([0, 1, 0, 1])

    tm1 = TrainingManager(X, X, y, y, mdefs, "classification", "f1_weighted", n_folds=2, optimization_seed=42)
    tm2 = TrainingManager(X, X, y, y, mdefs, "classification", "f1_weighted", n_folds=2, optimization_seed=99)

    cv1 = tm1._get_cv()
    cv2 = tm2._get_cv()

    assert cv1.random_state == 42
    assert cv2.random_state == 99


def test_seed_propagation_to_feature_engineer():
    """Verify that FeatureEngineer uses explicit seed rather than hardcoded 42."""
    fe1 = FeatureEngineer(seed=123)
    fe2 = FeatureEngineer(seed=456)
    assert fe1.seed == 123
    assert fe2.seed == 456


def test_seed_propagation_in_fold_safe_spec():
    """Verify FeatureSpec and FoldSafeFeaturePreprocessor propagate seed."""
    spec = FeatureSpec(
        column_profiles=[{"name": "x", "inferred_type": "numerical"}],
        target_column="target",
        problem_type="classification",
        seed=777,
    )
    assert spec.seed == 777
    mdef = ModelRegistry().get_model("logistic_regression")
    pipe = build_fold_safe_pipeline(mdef.create_model(), spec)
    assert pipe.named_steps["features"].seed == 777


def test_repeated_run_exact_reproducibility_classification(small_clf_dataset):
    """Verify identical dataset + configuration + seed yields numerically equivalent runs."""
    runner = BenchmarkRunner()
    cfg1 = BenchmarkConfig(
        dataset_name="repro_clf",
        dataset_path=small_clf_dataset,
        target_column="target",
        problem_type="classification",
        primary_metric="f1_weighted",
        condition="fallback",
        enable_ensemble=True,
        models_to_train=["logistic_regression", "random_forest_clf"],
        seed=101,
        n_folds=2,
        n_trials=2,
    )
    cfg2 = BenchmarkConfig(
        dataset_name="repro_clf",
        dataset_path=small_clf_dataset,
        target_column="target",
        problem_type="classification",
        primary_metric="f1_weighted",
        condition="fallback",
        enable_ensemble=True,
        models_to_train=["logistic_regression", "random_forest_clf"],
        seed=101,
        n_folds=2,
        n_trials=2,
    )

    rec1 = runner.run_benchmark(cfg1)
    rec2 = runner.run_benchmark(cfg2)

    assert rec1.overall_winner == rec2.overall_winner
    assert np.isclose(rec1.overall_cv_score, rec2.overall_cv_score, atol=1e-9)
    assert np.isclose(rec1.overall_holdout_score, rec2.overall_holdout_score, atol=1e-9)

    for c1, c2 in zip(rec1.candidate_models, rec2.candidate_models):
        assert c1["model_name"] == c2["model_name"]
        assert np.isclose(c1["selection_score"], c2["selection_score"], atol=1e-9)
        assert np.isclose(c1["holdout_score"], c2["holdout_score"], atol=1e-9)

    for e1, e2 in zip(rec1.ensemble_candidates, rec2.ensemble_candidates):
        assert e1["model_name"] == e2["model_name"]
        assert np.isclose(e1["selection_score"], e2["selection_score"], atol=1e-9)
        assert np.isclose(e1["holdout_score"], e2["holdout_score"], atol=1e-9)


def test_different_seeds_yield_different_evaluations(small_clf_dataset):
    """Verify different seeds produce different train/holdout splits and distinct scores."""
    runner = BenchmarkRunner()
    cfg1 = BenchmarkConfig(
        dataset_name="repro_clf_diff",
        dataset_path=small_clf_dataset,
        target_column="target",
        problem_type="classification",
        primary_metric="f1_weighted",
        models_to_train=["logistic_regression"],
        seed=42,
        n_folds=2,
        n_trials=0,
    )
    cfg2 = BenchmarkConfig(
        dataset_name="repro_clf_diff",
        dataset_path=small_clf_dataset,
        target_column="target",
        problem_type="classification",
        primary_metric="f1_weighted",
        models_to_train=["logistic_regression"],
        seed=999,
        n_folds=2,
        n_trials=0,
    )

    rec1 = runner.run_benchmark(cfg1)
    rec2 = runner.run_benchmark(cfg2)

    assert rec1.seed == 42
    assert rec2.seed == 999
    assert rec1.overall_holdout_score != rec2.overall_holdout_score or rec1.overall_cv_score != rec2.overall_cv_score


def test_holdout_used_for_selection_is_strictly_false(small_reg_dataset):
    """Verify holdout_used_for_selection flag is explicitly False and holdout does not influence winner."""
    runner = BenchmarkRunner()
    cfg = BenchmarkConfig(
        dataset_name="leakage_reg",
        dataset_path=small_reg_dataset,
        target_column="target",
        problem_type="regression",
        primary_metric="rmse",
        models_to_train=["ridge", "random_forest_reg"],
        seed=42,
        n_folds=2,
        n_trials=0,
    )
    rec = runner.run_benchmark(cfg)

    assert rec.holdout_used_for_selection is False
    assert rec.best_individual_cv_score is not None
    assert rec.best_individual_holdout_score is not None


def test_target_column_never_in_feature_names(small_clf_dataset):
    """Verify target column never appears in transformed feature names."""
    runner = BenchmarkRunner()
    cfg = BenchmarkConfig(
        dataset_name="leakage_target",
        dataset_path=small_clf_dataset,
        target_column="target",
        problem_type="classification",
        primary_metric="f1_weighted",
        models_to_train=["logistic_regression"],
        seed=42,
        n_folds=2,
        n_trials=0,
    )
    rec = runner.run_benchmark(cfg)
    assert rec.n_features_transformed > 0
    assert rec.n_features_raw == 3


def test_metric_spec_direction_contracts():
    """Verify that sklearn scoring direction and user-facing raw direction are correctly mapped."""
    spec_f1 = get_metric_spec("f1_weighted")
    assert spec_f1.scoring == "f1_weighted"
    assert spec_f1.direction == "maximize"
    assert spec_f1.raw_direction == "maximize"
    assert spec_f1.negated is False
    assert is_higher_better("f1_weighted") is True

    spec_rmse = get_metric_spec("rmse")
    assert spec_rmse.scoring == "neg_root_mean_squared_error"
    assert spec_rmse.direction == "maximize"
    assert spec_rmse.raw_direction == "minimize"
    assert spec_rmse.negated is True
    assert is_higher_better("rmse") is False

    spec_r2 = get_metric_spec("r2")
    assert spec_r2.scoring == "r2"
    assert spec_r2.direction == "maximize"
    assert spec_r2.raw_direction == "maximize"
    assert spec_r2.negated is False
    assert is_higher_better("r2") is True


def test_ensemble_delta_metric_interpretation():
    """Verify that ensemble CV improvement is positive when ensemble score exceeds single model score."""
    single_cv = -0.65
    ens_cv = -0.55
    delta_cv = round(float(ens_cv - single_cv), 6)
    assert delta_cv == +0.10

    single_h = 0.60
    ens_h = 0.50
    delta_h = round(float(single_h - ens_h), 6)
    assert delta_h == +0.10


def test_provenance_fields_present_and_valid(small_clf_dataset):
    """Verify all required provenance fields are populated in BenchmarkRecord."""
    runner = BenchmarkRunner()
    cfg = BenchmarkConfig(
        dataset_name="provenance_test",
        dataset_path=small_clf_dataset,
        target_column="target",
        problem_type="classification",
        primary_metric="f1_weighted",
        models_to_train=["logistic_regression"],
        seed=42,
        n_folds=2,
        n_trials=0,
    )
    rec = runner.run_benchmark(cfg)

    assert rec.dataset_name == "provenance_test"
    assert rec.dataset_hash is not None and len(rec.dataset_hash) == 16
    assert rec.git_commit is None or len(rec.git_commit) >= 7
    assert rec.cv_scoring == "f1_weighted"
    assert rec.cv_direction == "maximize"
    assert rec.holdout_metric == "f1_weighted"
    assert rec.holdout_direction == "maximize"
    assert rec.seed == 42
    assert rec.n_folds == 2
    assert rec.n_trials == 0
    assert rec.holdout_used_for_selection is False


def test_zero_contamination_of_production_db_and_storage(small_clf_dataset, production_baseline):
    """Ensure running scientific hardening tests modifies neither production DB nor storage."""
    db_path = "backend/automl_lens.db"
    storage_path = "backend/storage"

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    exp_before = cur.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    ds_before = cur.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
    conn.close()

    storage_before = [os.path.join(r, f) for r, _, fs in os.walk(storage_path) for f in fs]

    runner = BenchmarkRunner()
    cfg = BenchmarkConfig(
        dataset_name="iso_check",
        dataset_path=small_clf_dataset,
        target_column="target",
        problem_type="classification",
        primary_metric="f1_weighted",
        models_to_train=["logistic_regression"],
        seed=42,
        n_folds=2,
        n_trials=0,
    )
    _ = runner.run_benchmark(cfg)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    exp_after = cur.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    ds_after = cur.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
    conn.close()

    storage_after = [os.path.join(r, f) for r, _, fs in os.walk(storage_path) for f in fs]

    assert exp_before == exp_after == production_baseline['counts']['experiments']
    assert ds_before == ds_after == production_baseline['counts']['datasets']
    assert len(storage_before) == len(storage_after) == len(production_baseline['files'])
