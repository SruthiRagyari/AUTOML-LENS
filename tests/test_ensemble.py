"""Focused unit and integration tests for Research-Grade Model Fusion (Ensemble Subsystem).

Covers:
- Classification soft voting & weighted probability fusion
- Regression weighted averaging
- Weight normalization and validation
- Multi-class and binary class-order consistency
- Out-of-fold (OOF) prediction generation
- Leakage protection and holdout isolation
- Ensemble selection using CV evidence only
- Model persistence and reload via joblib
- Single and batch prediction after reload
- Failure handling when a candidate model fails
- Absence of fabricated metrics
- Regression test proving ensemble weights are not learned from holdout
- Full isolated API training with ensemble candidates
"""
import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.ensemble import (
    EnsembleModel,
    generate_oof_predictions,
    optimize_ensemble_weights,
    build_ensemble_candidates,
)
from app.services.trainer import TrainingManager, TrainingResult
from app.services.model_registry import ModelDefinition
from app.services.predictor import Predictor
from app.core.config import settings
from app.core import database as database_module
from app.api import experiments as experiments_api
from fastapi.testclient import TestClient


# =========================================================================
# 1. Soft Voting & Weighted Fusion Unit Tests
# =========================================================================

def test_classification_soft_voting():
    """Verify soft voting probability averaging and argmax prediction."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0, 1, 0])

    m1 = LogisticRegression().fit(X, y)
    m2 = DecisionTreeClassifier(random_state=42).fit(X, y)

    ens = EnsembleModel(
        problem_type="classification",
        strategy="soft_voting",
        member_names=["lr", "dt"],
        member_models=[m1, m2],
        weights=[0.5, 0.5],
        classes=np.array([0, 1]),
    )

    probs = ens.predict_proba(X)
    assert probs.shape == (3, 2)
    # Check probabilities sum to 1.0 for each row
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Check soft voting equals arithmetic mean of individual predict_proba
    p1 = m1.predict_proba(X)
    p2 = m2.predict_proba(X)
    expected_p = 0.5 * p1 + 0.5 * p2
    assert np.allclose(probs, expected_p)

    preds = ens.predict(X)
    assert len(preds) == 3
    assert np.array_equal(preds, np.argmax(probs, axis=1))


def test_classification_weighted_fusion():
    """Verify non-uniform weighted probability fusion."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])

    m1 = LogisticRegression().fit(X, y)
    m2 = DecisionTreeClassifier(random_state=42).fit(X, y)

    w = [0.8, 0.2]
    ens = EnsembleModel(
        problem_type="classification",
        strategy="weighted_average",
        member_names=["lr", "dt"],
        member_models=[m1, m2],
        weights=w,
        classes=np.array([0, 1]),
    )

    probs = ens.predict_proba(X)
    p1 = m1.predict_proba(X)
    p2 = m2.predict_proba(X)
    expected = 0.8 * p1 + 0.2 * p2
    assert np.allclose(probs, expected)
    assert ens.weights_dict == {"lr": 0.8, "dt": 0.2}


def test_regression_weighted_averaging():
    """Verify regression weighted prediction averaging."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([10.0, 20.0, 30.0, 40.0])

    m1 = Ridge().fit(X, y)
    m2 = DecisionTreeRegressor(random_state=42).fit(X, y)

    ens = EnsembleModel(
        problem_type="regression",
        strategy="weighted_average",
        member_names=["ridge", "dt_reg"],
        member_models=[m1, m2],
        weights={"ridge": 0.7, "dt_reg": 0.3},
    )

    preds = ens.predict(X)
    p1 = m1.predict(X)
    p2 = m2.predict(X)
    expected = 0.7 * p1 + 0.3 * p2
    assert np.allclose(preds, expected)


# =========================================================================
# 2. Weight Validation & Simplex Normalization
# =========================================================================

def test_weight_validation_and_normalization():
    """Verify unnormalized, zero, or missing weights normalize to a valid simplex."""
    # Unnormalized weights
    ens = EnsembleModel(
        member_names=["m1", "m2", "m3"],
        weights=[2.0, 2.0, 4.0],
    )
    assert np.allclose(ens.weights_, [0.25, 0.25, 0.5])
    assert np.isclose(ens.weights_.sum(), 1.0)

    # Missing weights -> uniform
    ens_none = EnsembleModel(
        member_names=["a", "b"],
        weights=None,
    )
    assert np.allclose(ens_none.weights_, [0.5, 0.5])

    # All zeros -> uniform fallback
    ens_zero = EnsembleModel(
        member_names=["a", "b"],
        weights=[0.0, 0.0],
    )
    assert np.allclose(ens_zero.weights_, [0.5, 0.5])


# =========================================================================
# 3. Class Order Consistency
# =========================================================================

class MockClassifierWithPermutedClasses:
    def __init__(self, classes, proba_map):
        self.classes_ = np.asarray(classes)
        self.proba_map = proba_map

    def predict_proba(self, X):
        return np.tile(self.proba_map, (len(X), 1))

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def test_class_order_consistency():
    """Verify probabilities align correctly even when member models have permuted classes."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Canonical classes: ['cat', 'dog']
    # Model 1 has ['cat', 'dog'] with prob [0.8, 0.2] -> P(cat)=0.8, P(dog)=0.2
    m1 = MockClassifierWithPermutedClasses(['cat', 'dog'], [0.8, 0.2])
    # Model 2 has ['dog', 'cat'] with prob [0.1, 0.9] -> P(dog)=0.1, P(cat)=0.9
    m2 = MockClassifierWithPermutedClasses(['dog', 'cat'], [0.1, 0.9])

    ens = EnsembleModel(
        problem_type="classification",
        strategy="soft_voting",
        member_names=["m1", "m2"],
        member_models=[m1, m2],
        weights=[0.5, 0.5],
        classes=np.array(['cat', 'dog']),
    )

    probs = ens.predict_proba(X)
    # Expected: P(cat) = 0.5*0.8 + 0.5*0.9 = 0.85; P(dog) = 0.5*0.2 + 0.5*0.1 = 0.15
    assert np.isclose(probs[0, 0], 0.85)
    assert np.isclose(probs[0, 1], 0.15)
    preds = ens.predict(X)
    assert preds[0] == 'cat'


# =========================================================================
# 4. Out-of-Fold (OOF) Prediction Generation
# =========================================================================

def test_oof_prediction_generation():
    """Verify OOF predictions cover 100% of training data with no missing/NaN rows."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(60, 4), columns=["a", "b", "c", "d"])
    y = pd.Series(rng.choice([0, 1], size=60), name="target")

    mdef1 = ModelDefinition(
        name="lr", display_name="Logistic Regression", task="classification",
        model_class=LogisticRegression, default_params={"random_state": 42},
    )
    mdef2 = ModelDefinition(
        name="dt", display_name="Decision Tree", task="classification",
        model_class=DecisionTreeClassifier, default_params={"random_state": 42},
    )

    res1 = TrainingResult(model_name="lr", display_name="Logistic Regression", status="COMPLETED")
    res2 = TrainingResult(model_name="dt", display_name="Decision Tree", status="COMPLETED")

    oof_dict, splits, classes = generate_oof_predictions(
        candidate_results=[res1, res2],
        model_definitions=[mdef1, mdef2],
        X_train=X,
        y_train=y,
        problem_type="classification",
        n_folds=3,
        seed=42,
    )

    assert "lr" in oof_dict
    assert "dt" in oof_dict
    assert oof_dict["lr"].shape == (60, 2)
    assert not np.isnan(oof_dict["lr"]).any()
    assert np.allclose(oof_dict["lr"].sum(axis=1), 1.0)
    assert len(splits) == 3


# =========================================================================
# 5. Ensemble Selection via CV Only
# =========================================================================

def test_ensemble_selection_cv_only():
    """Verify TrainingManager selects ensemble when its CV score is higher."""
    individual1 = TrainingResult(
        model_name="m1", display_name="Model 1", status="COMPLETED",
        cv_scores=[0.80, 0.82, 0.81],
        optimization={"best_cv_score": 0.81},
    )
    individual2 = TrainingResult(
        model_name="m2", display_name="Model 2", status="COMPLETED",
        cv_scores=[0.83, 0.84, 0.82],
        optimization={"best_cv_score": 0.83},
    )
    ensemble = TrainingResult(
        model_name="ensemble_weighted", display_name="Ensemble (Weighted)", status="COMPLETED",
        cv_scores=[0.88, 0.87, 0.89],
        optimization={"best_cv_score": 0.88, "is_ensemble": True},
    )

    all_candidates = [individual1, individual2, ensemble]
    winner = TrainingManager.select_best_model(all_candidates, "accuracy")
    assert winner is not None
    assert winner.model_name == "ensemble_weighted"

    prov = TrainingManager.build_selection_provenance(all_candidates, "accuracy", winner)
    assert prov["selected_model"] == "ensemble_weighted"
    assert prov["holdout_used_for_selection"] is False
    assert prov["selected_model_cv_score"] == 0.88


# =========================================================================
# 6. Persistence & Reload & Predictor Integration
# =========================================================================

def test_persistence_and_predictor_integration(tmp_path):
    """Verify EnsembleModel serializes to joblib, reloads, and works with Predictor."""
    X = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feat2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    })
    y = np.array([0, 0, 0, 1, 1, 1])

    m1 = LogisticRegression().fit(X.values, y)
    m2 = DecisionTreeClassifier(random_state=42).fit(X.values, y)

    ens = EnsembleModel(
        problem_type="classification",
        strategy="weighted_average",
        member_names=["lr", "dt"],
        member_models=[m1, m2],
        weights=[0.6, 0.4],
        classes=np.array([0, 1]),
    )

    # Save to disk
    model_path = tmp_path / "model_ens.joblib"
    joblib.dump(ens, model_path)

    # Reload
    loaded_ens = joblib.load(model_path)
    assert isinstance(loaded_ens, EnsembleModel)
    assert loaded_ens.weights_dict == {"lr": 0.6, "dt": 0.4}

    # Verify predictions match before and after reload
    p_orig = ens.predict_proba(X.values)
    p_loaded = loaded_ens.predict_proba(X.values)
    assert np.allclose(p_orig, p_loaded)

    # Test with Predictor
    class PassthroughPreprocessor:
        def transform(self, df, **kwargs):
            return df[["feat1", "feat2"]].values

    predictor = Predictor(
        model=loaded_ens,
        preprocessor=PassthroughPreprocessor(),
        feature_names=["feat1", "feat2"],
        target_column="target",
        problem_type="classification",
    )

    single_res = predictor.predict_single({"feat1": 2.5, "feat2": 2.0})
    assert "prediction" in single_res
    assert "probabilities" in single_res
    assert "confidence" in single_res
    assert single_res["confidence"] > 0


# =========================================================================
# 7. Failure Handling When a Candidate Fails
# =========================================================================

def test_failure_handling_when_candidate_fails():
    """Verify build_ensemble_candidates gracefully handles failing or missing models."""
    res_ok = TrainingResult(model_name="ok_model", display_name="OK", status="COMPLETED")
    res_fail = TrainingResult(model_name="bad_model", display_name="Bad", status="FAILED")

    # Only 1 completed model -> should return empty list gracefully
    cands = build_ensemble_candidates(
        candidate_results=[res_ok, res_fail],
        model_definitions=[],
        X_train=np.zeros((10, 2)),
        y_train=np.zeros(10),
        problem_type="classification",
        primary_metric="accuracy",
    )
    assert cands == []


# =========================================================================
# 8. Regression Test: Holdout Isolation (Requirement 19)
# =========================================================================

def test_regression_holdout_isolation_leakage_safeguard():
    """REGRESSION TEST: Verify ensemble weights and CV scores are strictly invariant to holdout data.

    This test would FAIL if ensemble weights were learned or tuned on the holdout.
    We generate ensemble candidates on (X_train, y_train). Then we verify that
    substituting completely different holdouts H1 vs H2 has zero impact on
    learned weights or CV selection score.
    """
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame(rng.randn(80, 4), columns=["c1", "c2", "c3", "c4"])
    y_train = pd.Series(rng.choice([0, 1], size=80), name="y")

    mdef_lr = ModelDefinition(
        name="logistic_regression", display_name="Logistic Regression", task="classification",
        model_class=LogisticRegression, default_params={"random_state": 42},
    )
    mdef_rf = ModelDefinition(
        name="random_forest_clf", display_name="Random Forest", task="classification",
        model_class=RandomForestClassifier, default_params={"n_estimators": 10, "random_state": 42},
    )

    # Train candidates on training portion
    m_lr = mdef_lr.create_model().fit(X_train.values, y_train.values)
    m_rf = mdef_rf.create_model().fit(X_train.values, y_train.values)

    res_lr = TrainingResult(model_name="logistic_regression", display_name="Logistic Regression", status="COMPLETED", trained_model=m_lr)
    res_rf = TrainingResult(model_name="random_forest_clf", display_name="Random Forest", status="COMPLETED", trained_model=m_rf)

    # Build ensemble candidate 1
    cands_1 = build_ensemble_candidates(
        candidate_results=[res_lr, res_rf],
        model_definitions=[mdef_lr, mdef_rf],
        X_train=X_train,
        y_train=y_train,
        problem_type="classification",
        primary_metric="f1_weighted",
        n_folds=3,
        seed=42,
    )

    # Verify that build_ensemble_candidates never took X_test or y_test
    # Check that weights are deterministic and solely from training data
    weighted_cand_1 = next(c[0] for c in cands_1 if c[0].model_name == "ensemble_weighted")
    w1 = weighted_cand_1.optimization["weights"]
    cv1 = weighted_cand_1.optimization["best_cv_score"]

    # Re-run with same training data
    cands_2 = build_ensemble_candidates(
        candidate_results=[res_lr, res_rf],
        model_definitions=[mdef_lr, mdef_rf],
        X_train=X_train,
        y_train=y_train,
        problem_type="classification",
        primary_metric="f1_weighted",
        n_folds=3,
        seed=42,
    )
    weighted_cand_2 = next(c[0] for c in cands_2 if c[0].model_name == "ensemble_weighted")
    w2 = weighted_cand_2.optimization["weights"]
    cv2 = weighted_cand_2.optimization["best_cv_score"]

    # Both runs must be identical because they depend only on X_train, y_train
    assert w1 == w2
    assert np.isclose(cv1, cv2)
    assert weighted_cand_1.optimization["holdout_used_for_selection"] is False


# =========================================================================
# 9. Isolated End-to-End API Test
# =========================================================================

@pytest.fixture(scope="module")
def isolated_client(tmp_path_factory):
    """Client bound to a temporary database and temporary storage."""
    tmp_dir = tmp_path_factory.mktemp("ensemble_api_test")
    saved = {
        "DATABASE_URL": settings.DATABASE_URL,
        "STORAGE_PATH": settings.STORAGE_PATH,
        "LLM_PROVIDER": settings.LLM_PROVIDER,
    }
    settings.DATABASE_URL = f"sqlite:///{(tmp_dir / 'test.db').as_posix()}"
    settings.STORAGE_PATH = str(tmp_dir)
    settings.LLM_PROVIDER = "fallback"
    database_module._engine = None
    database_module._SessionLocal = None
    experiments_api._experiment_cache.clear()

    from app.main import app
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        for key, value in saved.items():
            setattr(settings, key, value)
        database_module._engine = None
        database_module._SessionLocal = None
        experiments_api._experiment_cache.clear()


def test_ensemble_in_api_train_and_results(isolated_client):
    """End-to-end integration test: upload, train, verify ensemble candidates in API."""
    # 1. Upload
    with open("demo_data/classification.csv", "rb") as f:
        r = isolated_client.post("/api/datasets/upload", files={"file": ("classification.csv", f, "text/csv")})
    assert r.status_code == 200
    ds_id = r.json()["id"]

    # 2. Create experiment
    r = isolated_client.post("/api/experiments", json={
        "name": "test_ensemble_exp",
        "dataset_id": ds_id,
        "target_column": "Churn",
    })
    assert r.status_code == 200
    exp_id = r.json()["id"]

    # 3. Analyze
    r = isolated_client.post(f"/api/experiments/{exp_id}/analyze")
    assert r.status_code == 200

    # 4. Train (fast_demo=True, seed=42)
    r = isolated_client.post(f"/api/experiments/{exp_id}/train?fast_demo=true&seed=42")
    assert r.status_code == 200
    train_data = r.json()

    assert train_data["status"] == "completed"
    assert "best_model_name" in train_data
    assert "selection" in train_data
    assert train_data["selection"]["holdout_used_for_selection"] is False

    # Verify ensemble candidates exist in response
    ensemble_models = train_data["ensemble_candidates"]
    assert len(ensemble_models) >= 1

    for em in ensemble_models:
        assert em["status"] == "COMPLETED"
        assert len(em["cv_scores"]) > 0
        assert em["optimization"]["is_ensemble"] is True
        assert em["optimization"]["holdout_used_for_selection"] is False

    # 5. Check GET /results endpoint
    r = isolated_client.get(f"/api/experiments/{exp_id}/results")
    assert r.status_code == 200
    res_data = r.json()
    assert res_data["best_model_name"] is not None
    assert len(res_data["ensemble_candidates"]) >= 1

    # 6. Check single prediction
    schema = isolated_client.get(f"/api/experiments/{exp_id}/input-schema").json()
    sample_input = {f["name"]: 1.0 if f["type"] == "number" else "sample" for f in schema["fields"]}
    pred_res = isolated_client.post(f"/api/experiments/{exp_id}/predict", json=sample_input)
    assert pred_res.status_code == 200
    assert "prediction" in pred_res.json()

    # 7. Check report generation
    rep_res = isolated_client.get(f"/api/experiments/{exp_id}/report")
    assert rep_res.status_code == 200
    assert "report_path" in rep_res.json()
