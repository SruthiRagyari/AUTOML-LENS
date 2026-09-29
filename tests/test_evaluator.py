"""Tests for the evaluator."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.evaluator import Evaluator


def test_classification_metrics():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    metrics = Evaluator.evaluate_classification(y_true, y_pred)
    assert 'accuracy' in metrics
    assert 'f1_weighted' in metrics
    assert 'confusion_matrix' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    assert len(metrics['confusion_matrix']) == 2


def test_regression_metrics():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    metrics = Evaluator.evaluate_regression(y_true, y_pred)
    assert 'mae' in metrics
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert metrics['rmse'] == pytest.approx(np.sqrt(metrics['mse']), rel=1e-4)


def test_roc_auc_binary():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0])
    y_proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.6, 0.4]])
    metrics = Evaluator.evaluate_classification(y_true, y_pred, y_proba)
    assert 'roc_auc' in metrics
    assert metrics['roc_auc'] is not None


def test_perfect_classifier():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    metrics = Evaluator.evaluate_classification(y_true, y_pred)
    assert metrics['accuracy'] == 1.0
    assert metrics['f1_weighted'] == 1.0
