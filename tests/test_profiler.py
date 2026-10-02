"""Tests for the dataset profiler."""
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.profiler import DatasetProfiler


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'CustomerID': range(1000, 1000 + n),
        'Age': np.random.randint(18, 70, n),
        'Income': np.random.normal(50000, 15000, n),
        'City': np.random.choice(['NYC', 'LA', 'Chicago'], n),
        'Churn': np.random.choice([0, 1], n),
    })


def test_profile_basic(sample_df):
    p = DatasetProfiler()
    profile = p.profile(sample_df)
    assert profile['rows'] == 100
    assert profile['columns'] == 5
    assert 'column_profiles' in profile
    assert len(profile['column_profiles']) == 5


def test_column_type_inference(sample_df):
    p = DatasetProfiler()
    profile = p.profile(sample_df)
    cp_dict = {cp['name']: cp for cp in profile['column_profiles']}
    assert cp_dict['CustomerID']['inferred_type'] in ('id_like', 'numerical')  # high unique ratio
    assert cp_dict['Age']['inferred_type'] == 'numerical'
    assert cp_dict['City']['inferred_type'] == 'categorical'
    assert cp_dict['Churn']['inferred_type'] in ('numerical', 'boolean')


def test_target_suggestion(sample_df):
    p = DatasetProfiler()
    profile = p.profile(sample_df)
    target, ptype = p.suggest_target(profile)
    assert target == 'Churn'
    assert ptype == 'classification'


def test_missing_values():
    df = pd.DataFrame({
        'A': [1.0, None, 3.0, None, 5.0],
        'B': ['x', 'y', None, 'y', 'z'],
        'C': [1, 0, 1, 0, 1],
    })
    p = DatasetProfiler()
    profile = p.profile(df)
    cp_dict = {cp['name']: cp for cp in profile['column_profiles']}
    assert cp_dict['A']['missing_percentage'] == 40.0
    assert cp_dict['B']['missing_percentage'] == 20.0
    assert profile['total_missing'] == 3


def test_numerical_stats(sample_df):
    p = DatasetProfiler()
    profile = p.profile(sample_df)
    income_cp = next(cp for cp in profile['column_profiles'] if cp['name'] == 'Income')
    assert income_cp['mean'] is not None
    assert income_cp['std'] is not None
    assert income_cp['min_val'] is not None

