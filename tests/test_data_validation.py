"""Unit tests for dataset loading, schema verification, and input validation."""

import os
import pytest
import numpy as np
import pandas as pd

from src.data_loader import (
    DEFAULT_DATA_PATH,
    EXPECTED_FEATURE_COLUMNS,
    load_heart_disease_data,
    prepare_features_and_target,
    split_data,
    validate_raw_dataframe,
)
from src.preprocessing import validate_input_record


def test_dataset_file_exists():
    """Verify that the standard dataset exists on disk."""
    assert os.path.exists(DEFAULT_DATA_PATH), f"Expected dataset at {DEFAULT_DATA_PATH}"


def test_load_heart_disease_data_schema():
    """Verify that dataset loads with the required dimensions and feature set."""
    df = load_heart_disease_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 303
    for col in EXPECTED_FEATURE_COLUMNS:
        assert col in df.columns, f"Feature column '{col}' missing from loaded dataset."
    assert "target" in df.columns
    assert set(df["target"].unique()).issubset({0, 1})


def test_validate_raw_dataframe_missing_column():
    """Verify that dropping a critical column raises ValueError."""
    df = load_heart_disease_data()
    df_tampered = df.drop(columns=["chol"])
    with pytest.raises(ValueError, match="missing required feature columns"):
        validate_raw_dataframe(df_tampered)


def test_validate_raw_dataframe_empty():
    """Verify that passing an empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        validate_raw_dataframe(empty_df)


def test_prepare_features_and_target():
    """Verify feature matrix X and target y extraction."""
    df = load_heart_disease_data()
    X, y = prepare_features_and_target(df)
    assert X.shape == (303, 13)
    assert len(y) == 303
    assert list(X.columns) == EXPECTED_FEATURE_COLUMNS
    assert y.name in ["target", "num"]


def test_stratified_split_preserves_class_balance():
    """Verify train_test_split maintains proportional class distribution."""
    df = load_heart_disease_data()
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    overall_positive_rate = y.mean()
    train_positive_rate = y_train.mean()
    test_positive_rate = y_test.mean()

    # Proportions should match within 1.5% tolerance
    assert abs(train_positive_rate - overall_positive_rate) < 0.015
    assert abs(test_positive_rate - overall_positive_rate) < 0.015
    assert len(X_train) == 242
    assert len(X_test) == 61


def test_validate_input_record_valid():
    """Verify that a biologically valid patient record passes validation."""
    valid_patient = {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 0.0,
        "thal": 3.0,
    }
    is_valid, errors = validate_input_record(valid_patient)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_input_record_missing_keys():
    """Verify that omitting required features is caught."""
    incomplete_patient = {
        "age": 55,
        "sex": 1,
        # missing cp, trestbps, etc.
    }
    is_valid, errors = validate_input_record(incomplete_patient)
    assert is_valid is False
    assert any("Missing required clinical parameter" in err for err in errors)


def test_validate_input_record_out_of_bounds():
    """Verify that out-of-range clinical values are caught."""
    invalid_patient = {
        "age": 150,       # Out of bounds (> 100)
        "sex": 5,         # Invalid binary
        "cp": 9,          # Invalid category
        "trestbps": 350,  # Exceeds max
        "chol": 40,       # Below min
        "fbs": 2,         # Invalid binary
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": -2.0,  # Negative ST depression invalid
        "slope": 2,
        "ca": 5.0,        # Exceeds max vessels (3)
        "thal": 99.0,     # Unsupported category
    }
    is_valid, errors = validate_input_record(invalid_patient)
    assert is_valid is False
    assert len(errors) >= 5
