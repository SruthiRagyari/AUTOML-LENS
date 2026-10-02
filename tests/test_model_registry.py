"""Tests for the model registry."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.model_registry import ModelRegistry


def test_registry_loads():
    r = ModelRegistry()
    assert len(r.get_models('classification')) == 8
    assert len(r.get_models('regression')) == 9


def test_get_model():
    r = ModelRegistry()
    m = r.get_model('random_forest_clf')
    assert m is not None
    assert m.task == 'classification'
    assert m.display_name == 'Random Forest'


def test_create_model():
    r = ModelRegistry()
    m = r.get_model('logistic_regression')
    model = m.create_model()
    assert model is not None


def test_fast_demo_models():
    r = ModelRegistry()
    clf_models = r.get_fast_demo_models('classification')
    reg_models = r.get_fast_demo_models('regression')
    assert len(clf_models) == 3
    assert len(reg_models) == 3


def test_search_space():
    import optuna
    r = ModelRegistry()
    m = r.get_model('random_forest_clf')
    study = optuna.create_study()
    trial = study.ask()
    params = m.search_space(trial)
    assert 'n_estimators' in params
    assert 'max_depth' in params
