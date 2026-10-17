"""Unit and integration tests for model inference, schema enforcement, and risk scoring."""

import os
import pytest
from src.predict import HeartDiseasePredictor


@pytest.fixture(scope="module")
def predictor():
    """Fixture providing an initialized HeartDiseasePredictor instance."""
    return HeartDiseasePredictor()


def test_predictor_initialization(predictor):
    """Verify that predictor loads model and metadata."""
    assert predictor.pipeline is not None
    assert isinstance(predictor.metadata, dict)
    assert "best_model" in predictor.metadata


def test_predict_patient_healthy(predictor):
    """Verify inference for a low-risk clinical profile."""
    low_risk_patient = {
        "age": 35,
        "sex": 0,
        "cp": 2,          # Atypical angina
        "trestbps": 115,  # Normal BP
        "chol": 180,      # Healthy cholesterol
        "fbs": 0,
        "restecg": 0,
        "thalach": 175,   # High peak HR
        "exang": 0,       # No exercise angina
        "oldpeak": 0.0,   # No ST depression
        "slope": 1,       # Upsloping ST
        "ca": 0.0,        # No calcified vessels
        "thal": 3.0,      # Normal thallium
    }

    result = predictor.predict_patient(low_risk_patient)
    assert isinstance(result, dict)
    assert result["prediction_class"] in [0, 1]
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_category"] in [
        "Low Clinical Likelihood",
        "Moderate / Borderline Clinical Likelihood",
        "Elevated Clinical Likelihood",
    ]
    assert "disclaimer" in result
    assert "contributing_factors" in result
    assert result["risk_score"] < 0.5


def test_predict_patient_high_risk(predictor):
    """Verify inference for a high-risk clinical profile."""
    high_risk_patient = {
        "age": 67,
        "sex": 1,
        "cp": 4,          # Asymptomatic
        "trestbps": 160,  # Hypertension
        "chol": 286,      # High cholesterol
        "fbs": 1,
        "restecg": 2,     # LVH
        "thalach": 108,   # Low peak HR
        "exang": 1,       # Exercise angina
        "oldpeak": 2.6,   # Severe ST depression
        "slope": 2,       # Flat ST
        "ca": 3.0,        # 3 major vessels
        "thal": 7.0,      # Reversible defect
    }

    result = predictor.predict_patient(high_risk_patient)
    assert result["prediction_class"] == 1
    assert result["risk_score"] > 0.70
    assert result["risk_category"] == "Elevated Clinical Likelihood"
    assert len(result["contributing_factors"]) > 0


def test_feature_ordering_enforcement(predictor):
    """Verify that scrambled dictionary key order does not affect model output."""
    patient_ordered = {
        "age": 58,
        "sex": 1,
        "cp": 3,
        "trestbps": 140,
        "chol": 211,
        "fbs": 1,
        "restecg": 0,
        "thalach": 165,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 2,
        "ca": 0.0,
        "thal": 3.0,
    }

    # Reverse key order
    keys_reversed = list(reversed(list(patient_ordered.keys())))
    patient_scrambled = {k: patient_ordered[k] for k in keys_reversed}

    result_ordered = predictor.predict_patient(patient_ordered)
    result_scrambled = predictor.predict_patient(patient_scrambled)

    assert result_ordered["prediction_class"] == result_scrambled["prediction_class"]
    assert result_ordered["risk_score"] == pytest.approx(result_scrambled["risk_score"], abs=1e-5)


def test_invalid_input_raises_value_error(predictor):
    """Verify that submitting an out-of-bounds record raises ValueError."""
    bad_patient = {
        "age": 12,        # Below minimum (18)
        "sex": 1,
        "cp": 1,
        "trestbps": 120,
        "chol": 200,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 0.0,
        "thal": 3.0,
    }
    with pytest.raises(ValueError, match="below valid biological minimum"):
        predictor.predict_patient(bad_patient)


def test_missing_model_file_raises_error():
    """Verify descriptive FileNotFoundError when model path does not exist."""
    fake_path = "non_existent_dir/ghost_model.joblib"
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        HeartDiseasePredictor(model_path=fake_path)
