"""Regression tests for the two-valued text column bug.

The profiler used to report ANY column with at most three distinct values as
``boolean``. The preprocessing engine routes boolean columns into the numeric
imputer, so a plain ``yes``/``no`` feature reached the median imputer as a
string and training died with:

    ValueError: Cannot use median strategy with non-numeric data:
    could not convert string to float: 'no'

That broke real uploads (found on UCI Bank Marketing) but not the synthetic
tests, which used numeric targets only. These tests pin the corrected rule:
boolean means a genuine bool column or numbers that are exactly 0/1.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
import pytest

from app.services.profiler import DatasetProfiler
from app.services.preprocessor import PreprocessingEngine
from app.services.feature_engineer import FeatureEngineer


def types_of(df, target):
    prof = DatasetProfiler().profile(df)
    return {c["name"]: c["inferred_type"] for c in prof["column_profiles"]}, prof


class TestBooleanInferenceIsNarrow:
    def test_real_bool_dtype_is_boolean(self):
        df = pd.DataFrame({"flag": [True, False, True, False],
                           "x": [1, 2, 3, 4]})
        types, _ = types_of(df, "x")
        assert types["flag"] == "boolean"

    def test_numeric_zero_one_is_boolean(self):
        df = pd.DataFrame({"flag": [0.0, 1.0, 1.0, 0.0], "x": [1, 2, 3, 4]})
        types, _ = types_of(df, "x")
        assert types["flag"] == "boolean"

    @pytest.mark.parametrize("column,values", [
        ("yes_no", ["yes", "no", "yes", "no"]),
        ("male_female", ["m", "f", "m", "f"]),
        ("pass_fail", ["pass", "fail", "pass", "fail"]),
        ("up_down", ["up", "down", "up", "down"]),
        ("str_zero_one", ["0", "1", "0", "1"]),
        ("three_way", ["a", "b", "c", "a"]),
    ])
    def test_text_two_or_three_valued_columns_are_categorical(self, column, values):
        df = pd.DataFrame({column: values, "x": [1, 2, 3, 4]})
        types, _ = types_of(df, "x")
        assert types[column] == "categorical", (
            f"{column} must not be boolean: the numeric imputer would choke")

    def test_integer_zero_one_stays_numerical(self):
        """int 0/1 is safe numerically; it need not be labelled boolean."""
        df = pd.DataFrame({"flag": [0, 1, 0, 1], "x": [1, 2, 3, 4]})
        types, _ = types_of(df, "x")
        assert types["flag"] in ("numerical", "boolean")


class TestTextBooleanDoesNotBreakPreprocessing:
    def test_full_pipeline_accepts_yes_no_feature_column(self):
        """The exact shape that used to raise during training."""
        rng = np.random.RandomState(3)
        n = 120
        df = pd.DataFrame({
            "age": rng.normal(size=n),
            "outcome": rng.choice(["yes", "no"], n),
            "grade": rng.choice(["A", "B", "C"], n),
            "duration": rng.normal(size=n) * 10 + 100,
        })
        prof = DatasetProfiler().profile(df)
        fe = FeatureEngineer()
        result = fe.apply(df, prof["column_profiles"], "duration")
        eng = result[0] if isinstance(result, tuple) else result
        prep = PreprocessingEngine()
        X, names = prep.build_and_fit(
            eng, "duration", prof["column_profiles"], "regression",
            extra_numeric=fe.new_features, feature_engineer=fe)
        assert X.shape[0] == n
        assert X.shape[1] > 0
        assert np.isfinite(X).all(), "no NaN/inf may reach the model"
        # The text column really was one-hot encoded, not silently dropped.
        assert any(n.startswith("categorical__outcome") for n in names)

    def test_transform_round_trip_on_unseen_rows(self):
        rng = np.random.RandomState(4)
        n = 100
        df = pd.DataFrame({
            "flag": rng.choice(["yes", "no"], n),
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        })
        prof = DatasetProfiler().profile(df)
        fe = FeatureEngineer()
        result = fe.apply(df, prof["column_profiles"], "y")
        eng = result[0] if isinstance(result, tuple) else result
        prep = PreprocessingEngine()
        prep.build_and_fit(eng, "y", prof["column_profiles"], "regression",
                           extra_numeric=fe.new_features, feature_engineer=fe)
        X2 = prep.transform(df.iloc[[0, 1]], target_column="y")
        assert X2.shape[0] == 2
        assert np.isfinite(X2).all()


class TestBooleanTargetStillDetectedAsClassification:
    def test_yes_no_target_is_classification(self):
        n = 120
        df = pd.DataFrame({
            "x": np.arange(n, dtype=float),
            "y": ["yes"] * 60 + ["no"] * 60,
        })
        prof = DatasetProfiler().profile(df)
        target, ptype = DatasetProfiler().suggest_target(prof)
        assert ptype == "classification"
        assert target == "y"

    def test_numeric_zero_one_target_is_classification(self):
        n = 120
        df = pd.DataFrame({
            "x": np.arange(n, dtype=float),
            "y": [0] * 60 + [1] * 60,
        })
        prof = DatasetProfiler().profile(df)
        _target, ptype = DatasetProfiler().suggest_target(prof)
        assert ptype == "classification"