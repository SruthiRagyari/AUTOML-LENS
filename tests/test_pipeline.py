"""Integration test — full training pipeline."""
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from sklearn.model_selection import train_test_split
from app.services.profiler import DatasetProfiler
from app.services.preprocessor import PreprocessingEngine
from app.services.model_registry import ModelRegistry
from app.services.trainer import TrainingManager
from app.services.explainer import Explainer


@pytest.fixture
def clf_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'Age': np.random.randint(18, 70, n),
        'Income': np.random.normal(50000, 15000, n),
        'CreditScore': np.random.randint(300, 850, n),
        'City': np.random.choice(['NYC', 'LA', 'Chicago'], n),
        'Churn': np.random.choice([0, 1], n),
    })
    return df


def test_full_classification_pipeline(clf_data):
    """End-to-end classification pipeline test."""
    # Profile
    profiler = DatasetProfiler()
    profile = profiler.profile(clf_data)
    target, ptype = profiler.suggest_target(profile)
    assert target == 'Churn'
    assert ptype == 'classification'

    # Preprocess
    y = clf_data['Churn']
    preprocessor = PreprocessingEngine()
    X, feature_names = preprocessor.build_and_fit(clf_data, 'Churn', profile['column_profiles'], 'classification')
    assert X.shape[0] == len(clf_data)
    assert X.shape[1] > 0

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train (baseline only for speed)
    registry = ModelRegistry()
    fast_models = registry.get_models('classification', ['logistic_regression', 'random_forest_clf'])
    trainer = TrainingManager(
        X_train, X_test, y_train, y_test,
        fast_models, 'classification', 'f1_weighted',
        n_folds=3, n_trials=3, mode='baseline'
    )
    results = trainer.train_all()
    assert len(results) == 2
    assert all(r.status == 'COMPLETED' for r in results)

    # Select best
    best = trainer.select_best_model(results, 'f1_weighted')
    assert best is not None
    assert best.trained_model is not None

    # Explainability
    expl = Explainer.explain(best.trained_model, X_test, feature_names, y_test)
    assert expl.method_used in ('feature_importance', 'coefficients', 'shap', 'permutation')
    assert len(expl.feature_importance) > 0
    assert len(expl.top_features) > 0
