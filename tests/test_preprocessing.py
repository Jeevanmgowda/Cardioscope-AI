"""Unit tests for feature preprocessing, pipeline transformations, and leakage prevention."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.data_loader import load_heart_disease_data, prepare_features_and_target, split_data
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
    get_feature_names_out,
)


def test_build_preprocessor_structure():
    """Verify ColumnTransformer is assembled with correct sub-transformers."""
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)

    transformer_names = [t[0] for t in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names
    assert "bin" in transformer_names
    assert "ord" in transformer_names


def test_preprocessor_handles_missing_values():
    """Verify preprocessor imputes missing values in ca, thal, and numerical columns."""
    df = load_heart_disease_data()
    X, _ = prepare_features_and_target(df)

    # Artificially inject NaNs to test robust imputation
    X_with_nans = X.copy()
    X_with_nans.loc[0, "chol"] = np.nan
    X_with_nans.loc[1, "oldpeak"] = np.nan
    X_with_nans.loc[2, "ca"] = np.nan
    X_with_nans.loc[3, "thal"] = np.nan

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X_with_nans)

    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any(), "Transformed feature matrix contains NaNs after preprocessing!"


def test_get_feature_names_out():
    """Verify transformed feature names extraction."""
    df = load_heart_disease_data()
    X, _ = prepare_features_and_target(df)

    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    feature_names = get_feature_names_out(preprocessor)
    assert isinstance(feature_names, list)
    assert len(feature_names) > len(X.columns), "One-hot encoding should expand feature count."
    # Ensure numericals are present
    for num_col in NUMERICAL_FEATURES:
        assert any(num_col in name for name in feature_names)


def test_leakage_safety_train_test_independence():
    """Verify that preprocessing scaler means reflect only training data."""
    df = load_heart_disease_data()
    X, y = prepare_features_and_target(df)
    X_train, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # Extract fitted mean of the numerical scaler
    fitted_means = preprocessor.named_transformers_["num"].named_steps["scaler"].mean_
    actual_train_means = X_train[NUMERICAL_FEATURES].mean().values

    # Preprocessor scaler must match training set means
    np.testing.assert_allclose(fitted_means, actual_train_means, rtol=1e-3)

    # Transforming test set should succeed without altering fitted statistics
    transformed_test = preprocessor.transform(X_test)
    assert transformed_test.shape[0] == len(X_test)
    assert not np.isnan(transformed_test).any()
