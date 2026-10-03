"""Automated tests for Research-Grade Benchmarking & Evaluation Subsystem.

Covers:
- Benchmark isolation and zero production DB/storage contamination
- Metric correctness (classification and regression)
- Strict CV/holdout separation (holdout never used for selection)
- Reproducibility (deterministic seeds yield identical results)
- LLM vs Fallback condition recording
- Absence of fabricated metrics
"""
import os
import sqlite3
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.benchmarker import BenchmarkRunner, BenchmarkConfig, BenchmarkRecord
from app.services.model_registry import ModelRegistry


@pytest.fixture(scope="module")
def sample_classification_csv(tmp_path_factory):
    """Creates a small, deterministic classification dataset in a temp folder."""
    tmp_dir = tmp_path_factory.mktemp("bench_clf_data")
    path = tmp_dir / "clf_data.csv"
    rng = np.random.RandomState(42)
    n = 120
    df = pd.DataFrame({
        "num1": rng.randn(n),
        "num2": rng.uniform(0, 10, n),
        "cat1": rng.choice(["A", "B", "C"], n),
        "target": rng.choice([0, 1], n),
    })
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def sample_regression_csv(tmp_path_factory):
    """Creates a small, deterministic regression dataset in a temp folder."""
    tmp_dir = tmp_path_factory.mktemp("bench_reg_data")
    path = tmp_dir / "reg_data.csv"
    rng = np.random.RandomState(42)
    n = 120
    x1 = rng.randn(n)
    x2 = rng.uniform(10, 50, n)
    y = 2.5 * x1 + 0.1 * x2 + rng.randn(n) * 0.5
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "cat_feature": rng.choice(["low", "medium", "high"], n),
        "target": y,
    })
    df.to_csv(path, index=False)
    return str(path)


def test_metric_correctness_classification(sample_classification_csv):
    """Verify classification benchmark produces valid metrics within [0, 1]."""
    runner = BenchmarkRunner()
    cfg = BenchmarkConfig(
        dataset_name="synth_clf",
        dataset_path=sample_classification_csv,
        target_column="target",
        problem_type="classification",
        primary_metric="f1_weighted",
        condition="fallback",
        enable_ensemble=True,
        models_to_train=["logistic_regression", "random_forest_clf"],
        seed=42,
        n_folds=3,
        n_trials=0,
    )
    rec = runner.run_benchmark(cfg)

    assert rec.problem_type == "classification"
    assert rec.primary_metric == "f1_weighted"
    assert 0.0 <= rec.overall_cv_score <= 1.0
    assert 0.0 <= rec.overall_holdout_score <= 1.0
    assert len(rec.candidate_models) == 2
    assert len(rec.ensemble_candidates) >= 1

    for cm in rec.candidate_models:
        assert cm["selection_score"] is not None
        assert 0.0 <= cm["selection_score"] <= 1.0
        assert cm["holdout_score"] is not None
        assert 0.0 <= cm["holdout_score"] <= 1.0


def test_metric_correctness_regression(sample_regression_csv):
    """Verify regression benchmark produces valid finite RMSE and proper ranking."""
    runner = BenchmarkRunner()
    cfg = BenchmarkConfig(
        dataset_name="synth_reg",
        dataset_path=sample_regression_csv,
        target_column="target",
        problem_type="regression",
        primary_metric="rmse",
        condition="fallback",
        enable_ensemble=True,
        models_to_train=["ridge", "random_forest_reg"],
        seed=42,
        n_folds=3,
        n_trials=0,
    )
    rec = runner.run_benchmark(cfg)

    assert rec.problem_type == "regression"
    assert rec.primary_metric == "rmse"
    assert np.isfinite(rec.overall_cv_score)
    assert np.isfinite(rec.overall_holdout_score)
    assert rec.overall_holdout_score >= 0.0  # RMSE must be non-negative


def test_cv_holdout_strict_separation(sample_classification_csv):
    """Verify holdout data is never used for selection and holdout_used_for_selection is False."""
    runner = BenchmarkRunner()
    cfg = BenchmarkConfig(
        dataset_name="synth_clf",
        dataset_path=sample_classification_csv,
        target_column="target",
        problem_type="classification",
        primary_metric="f1_weighted",
        condition="fallback",
        enable_ensemble=True,
        models_to_train=["logistic_regression", "random_forest_clf"],
        seed=42,
        n_folds=3,
        n_trials=0,
    )
    rec = runner.run_benchmark(cfg)

    assert rec.holdout_used_for_selection is False

    # Winner must have the best CV score (accounting for deterministic tie breaking)
    candidate_scores = {c["model_name"]: c["selection_score"] for c in rec.candidate_models}
    ensemble_scores = {e["model_name"]: e["selection_score"] for e in rec.ensemble_candidates}
    all_cv_scores = {**candidate_scores, **ensemble_scores}

    top_cv_score = max(all_cv_scores.values())
    assert np.isclose(rec.overall_cv_score, top_cv_score, atol=1e-5)
    assert rec.overall_winner in all_cv_scores


def test_benchmark_reproducibility(sample_classification_csv):
    """Verify identical seeds produce numerically identical CV and holdout evaluations."""
    runner = BenchmarkRunner()
    cfg1 = BenchmarkConfig(
        dataset_name="synth_clf",
        dataset_path=sample_classification_csv,
        target_column="target",
        problem_type="classification",
        primary_metric="f1_weighted",
        seed=100,
        n_folds=3,
        n_trials=0,
        models_to_train=["logistic_regression", "random_forest_clf"],
    )
    cfg2 = BenchmarkConfig(
        dataset_name="synth_clf",
        dataset_path=sample_classification_csv,
        target_column="target",
        problem_type="classification",
        primary_metric="f1_weighted",
        seed=100,
        n_folds=3,
        n_trials=0,
        models_to_train=["logistic_regression", "random_forest_clf"],
    )

    rec1 = runner.run_benchmark(cfg1)
    rec2 = runner.run_benchmark(cfg2)

    assert rec1.overall_winner == rec2.overall_winner
    assert np.isclose(rec1.overall_cv_score, rec2.overall_cv_score)
    assert np.isclose(rec1.overall_holdout_score, rec2.overall_holdout_score)
    for c1, c2 in zip(rec1.candidate_models, rec2.candidate_models):
        assert c1["model_name"] == c2["model_name"]
        assert np.isclose(c1["selection_score"], c2["selection_score"])
        assert np.isclose(c1["holdout_score"], c2["holdout_score"])


def test_llm_vs_fallback_recording(sample_classification_csv):
    """Verify benchmark records distinguish LLM advisory operations from ML performance."""
    runner = BenchmarkRunner()
    custom_ops = [{"operation": "log1p", "column": "num2", "new_column_name": "num2_log1p"}]
    
    cfg = BenchmarkConfig(
        dataset_name="synth_clf",
        dataset_path=sample_classification_csv,
        target_column="target",
        problem_type="classification",
        primary_metric="f1_weighted",
        condition="llm_assisted",
        seed=42,
        n_folds=3,
        n_trials=0,
        custom_operations=custom_ops,
        models_to_train=["logistic_regression", "random_forest_clf"],
    )
    rec = runner.run_benchmark(cfg)

    assert rec.condition == "llm_assisted"
    assert rec.llm_proposed_operations == custom_ops
    assert len(rec.operations_accepted) == 1
    assert rec.operations_accepted[0]["operation"] == "log1p"
    assert rec.llm_recommended_models == ["logistic_regression", "random_forest_clf"]


def test_no_fabricated_metrics(sample_classification_csv):
    """Verify there are no placeholder strings, NaNs, or invented values."""
    runner = BenchmarkRunner()
    cfg = BenchmarkConfig(
        dataset_name="synth_clf",
        dataset_path=sample_classification_csv,
        target_column="target",
        problem_type="classification",
        primary_metric="f1_weighted",
        condition="fallback",
        models_to_train=["logistic_regression", "random_forest_clf"],
        seed=42,
        n_folds=3,
        n_trials=0,
    )
    rec = runner.run_benchmark(cfg)

    assert isinstance(rec.rows_total, int) and rec.rows_total > 0
    assert isinstance(rec.rows_train, int) and rec.rows_train > 0
    assert isinstance(rec.rows_holdout, int) and rec.rows_holdout > 0
    assert isinstance(rec.overall_cv_score, float) and not np.isnan(rec.overall_cv_score)
    assert isinstance(rec.overall_holdout_score, float) and not np.isnan(rec.overall_holdout_score)
    assert isinstance(rec.total_time, float) and rec.total_time > 0.0


def test_benchmark_isolation_and_no_production_contamination(sample_classification_csv):
    """Verify running benchmarks causes ZERO modifications to backend/automl_lens.db or storage."""
    db_path = "backend/automl_lens.db"
    storage_path = "backend/storage"

    # Pre-check
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    exp_before = c.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    ds_before = c.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
    conn.close()

    storage_before = [os.path.join(r, f) for r, _, files in os.walk(storage_path) for f in files]

    # Run benchmark
    runner = BenchmarkRunner()
    cfg = BenchmarkConfig(
        dataset_name="synth_clf",
        dataset_path=sample_classification_csv,
        target_column="target",
        problem_type="classification",
        primary_metric="f1_weighted",
        condition="fallback",
        models_to_train=["logistic_regression"],
        seed=42,
        n_folds=3,
        n_trials=0,
    )
    _ = runner.run_benchmark(cfg)

    # Post-check
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    exp_after = c.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    ds_after = c.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
    conn.close()

    storage_after = [os.path.join(r, f) for r, _, files in os.walk(storage_path) for f in files]

    assert exp_before == exp_after == 7
    assert ds_before == ds_after == 5
    assert len(storage_before) == len(storage_after) == 23
