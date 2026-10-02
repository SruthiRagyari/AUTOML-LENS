"""FeatureEngineer operation mode: fit-time statistics come from the training
split only, replay is safe for test/prediction frames, and the engineered
features actually reach the model matrix."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
import pytest

from app.services.feature_engineer import FeatureEngineer
from app.services.feature_operations import validate_operations
from app.services.preprocessor import PreprocessingEngine
from app.services.profiler import DatasetProfiler

TARGET = "target"


def make_df(n=120, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "income": rng.uniform(1000, 9000, n).round(2),
        "age": rng.randint(18, 80, n),
        "city": rng.choice(["a", "b", "c", "zzz"], n, p=[.45, .4, .1, .05]),
        # Free text long enough for the profiler to infer type "text" (>50 chars).
        "note": [f"comment number {i} says something quite lengthy about item {i} "
                 f"with plenty of characters here" for i in range(n)],
        "when": pd.date_range("2024-01-01", periods=n, freq="D"),
        "constant": "same",
        TARGET: rng.choice([0, 1], n),
    })


def _setup():
    df = make_df()
    profiles = DatasetProfiler().profile(df)["column_profiles"]
    ops, rejected = validate_operations([
        {"column": "income", "operation": "log"},
        {"column": "income", "operation": "interaction",
         "params": {"with_column": "age"}},
        {"column": "age", "operation": "zscore"},
        {"column": "city", "operation": "frequency_encoding"},
        {"column": "city", "operation": "rare_category_grouping",
         "params": {"threshold": 0.1, "group_label": "Other"}},
        {"column": "note", "operation": "word_count"},
        {"column": "when", "operation": "month"},
    ], profiles, TARGET)
    assert rejected == [], rejected
    return df, profiles, ops


def test_operation_mode_creates_features_on_both_splits():
    df, profiles, ops = _setup()
    train, test = df.iloc[:90], df.iloc[90:]
    fe = FeatureEngineer()
    fe.fit(train, profiles, TARGET, operations=ops)
    assert fe.mode == "operations"
    eng_train = fe.transform(train)
    eng_test = fe.transform(test)
    assert fe.new_features, "operation mode must register its new features"
    for col in fe.new_features:
        assert col in eng_train.columns, col
        assert col in eng_test.columns, col
    # the label is untouched; sources may be rewritten only by in-place ops
    assert eng_train[TARGET].equals(train[TARGET])
    assert "income" in eng_train.columns and "age" in eng_train.columns
    # replay is a pure function of rows + fitted state (idempotent)
    again = fe.transform(test)
    pd.testing.assert_frame_equal(eng_test, again)


def test_fit_statistics_come_from_training_split_only():
    df, profiles, ops = _setup()
    train, test = df.iloc[:90], df.iloc[90:]
    fe = FeatureEngineer()
    fe.fit(train, profiles, TARGET, operations=ops)
    eng_test = fe.transform(test)

    z_state = [v for k, v in fe.operation_state.items() if k.endswith(":age:zscore")][0]
    expected = (test["age"] - z_state["mean"]) / z_state["std"]
    assert np.allclose(eng_test["age_zscore"], expected)
    # moments equal the TRAINING frame, not the test frame
    assert z_state["mean"] == pytest.approx(float(train["age"].mean()))
    assert z_state["std"] == pytest.approx(float(train["age"].std()))

    freq_state = [v for k, v in fe.operation_state.items()
                  if k.endswith(":city:frequency_encoding")][0]
    train_share = train["city"].value_counts(normalize=True)
    for category, share in freq_state["frequencies"].items():
        assert share == pytest.approx(float(train_share[category]))

    rare_state = [v for k, v in fe.operation_state.items()
                  if k.endswith(":city:rare_category_grouping")][0]
    assert rare_state["group_label"] == "Other"
    # categories below the train-share threshold are excluded from the keep set
    rare = {c for c, p in train_share.items() if p < 0.1}
    assert not (rare & set(rare_state["keep"]))

    # a category never seen in training maps to frequency 0
    unseen = fe.transform(test.assign(city="never_seen_category"))
    assert (unseen["city_freq"] == 0.0).all()


def test_heuristics_mode_unchanged_without_operations():
    df, profiles, _ = _setup()
    fe = FeatureEngineer()
    fe.fit(df, profiles, TARGET)
    assert fe.mode == "heuristics"
    assert fe.get_summary()["mode"] == "heuristics"
    assert "operations" not in fe.get_summary()
    assert fe.transform(df).shape[0] == len(df)


def test_engineered_features_reach_the_model_matrix():
    df, profiles, ops = _setup()
    train, test = df.iloc[:90], df.iloc[90:]
    fe = FeatureEngineer()
    fe.fit(train, profiles, TARGET, operations=ops)
    eng_train = fe.transform(train)

    prep = PreprocessingEngine()
    X_train, names = prep.build_and_fit(
        eng_train, TARGET, profiles, "classification",
        extra_numeric=fe.new_features, feature_engineer=fe)
    # transform() re-applies the fitted feature engineer to the raw test frame
    X_test = prep.transform(test, target_column=TARGET)

    assert X_train.shape[0] == 90 and X_test.shape[0] == 30
    assert X_train.shape[1] == X_test.shape[1], (X_train.shape, X_test.shape)
    assert any("log1p" in n for n in names), names
    assert any("freq" in n for n in names), names
    assert any("_x_" in n for n in names), names
    summary = fe.get_summary()
    assert summary["mode"] == "operations"
    assert summary["operations"] and summary["fitted_on"].startswith("training")
