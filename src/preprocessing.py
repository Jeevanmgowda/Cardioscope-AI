"""Preprocessing pipeline and transformation module for Heart Disease Risk Prediction.

Constructs leakage-safe Scikit-Learn ColumnTransformers and Pipelines,
handles categorical encoding, median/mode imputation, and feature scaling.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature subsets by clinical measurement type and mathematical representation
NUMERICAL_FEATURES: List[str] = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES: List[str] = [
    "cp",
    "restecg",
    "slope",
    "thal",
]

BINARY_FEATURES: List[str] = [
    "sex",
    "fbs",
    "exang",
]

ORDINAL_FEATURES: List[str] = [
    "ca",
]

# Clinical valid physiological boundaries for validation
FEATURE_RANGES: Dict[str, Dict[str, Union[float, int, List[Union[int, float]]]]] = {
    "age": {"min": 18, "max": 100, "label": "Age (years)", "unit": "years"},
    "sex": {"allowed": [0, 1], "label": "Sex (0=Female, 1=Male)"},
    "cp": {"allowed": [1, 2, 3, 4], "label": "Chest Pain Type (1-4)"},
    "trestbps": {"min": 80, "max": 240, "label": "Resting Blood Pressure", "unit": "mm Hg"},
    "chol": {"min": 100, "max": 600, "label": "Serum Cholesterol", "unit": "mg/dL"},
    "fbs": {"allowed": [0, 1], "label": "Fasting Blood Sugar > 120 mg/dL (0=No, 1=Yes)"},
    "restecg": {"allowed": [0, 1, 2], "label": "Resting ECG Results (0, 1, 2)"},
    "thalach": {"min": 60, "max": 220, "label": "Maximum Heart Rate Achieved", "unit": "bpm"},
    "exang": {"allowed": [0, 1], "label": "Exercise Induced Angina (0=No, 1=Yes)"},
    "oldpeak": {"min": 0.0, "max": 10.0, "label": "ST Depression (Oldpeak)", "unit": "mm"},
    "slope": {"allowed": [1, 2, 3], "label": "Peak Exercise ST Slope (1, 2, 3)"},
    "ca": {"min": 0, "max": 3, "allowed": [0, 1, 2, 3, 0.0, 1.0, 2.0, 3.0], "label": "Major Vessels Colored (0-3)"},
    "thal": {"allowed": [3.0, 6.0, 7.0, 3, 6, 7], "label": "Thalassemia (3=Normal, 6=Fixed, 7=Reversible)"},
}


def build_preprocessor() -> ColumnTransformer:
    """Construct an end-to-end ColumnTransformer.

    Applies:
    - Median imputation + StandardScaler to continuous numerical features.
    - Most frequent imputation + OneHotEncoder to categorical features.
    - Most frequent imputation to binary and ordinal features.

    Returns:
        ColumnTransformer: Preprocessing transformer ready to fit on training data.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("bin", binary_pipeline, BINARY_FEATURES),
            ("ord", ordinal_pipeline, ORDINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names_out(fitted_preprocessor: ColumnTransformer) -> List[str]:
    """Retrieve output feature names from a fitted ColumnTransformer.

    Args:
        fitted_preprocessor: Fitted ColumnTransformer instance.

    Returns:
        List[str]: Names of all transformed features.
    """
    try:
        names = fitted_preprocessor.get_feature_names_out()
        return list(names)
    except Exception:
        # Fallback manual reconstruction if get_feature_names_out encounters version differences
        names = list(NUMERICAL_FEATURES)
        try:
            cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
            for col, cats in zip(CATEGORICAL_FEATURES, cat_encoder.categories_):
                for cat in cats:
                    names.append(f"{col}_{cat}")
        except Exception:
            pass
        names.extend(BINARY_FEATURES)
        names.extend(ORDINAL_FEATURES)
        return names


def validate_input_record(record: Dict[str, Union[int, float]]) -> Tuple[bool, List[str]]:
    """Validate a single patient input dictionary against schema and biological bounds.

    Args:
        record: Mapping of feature names to patient values.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_error_messages)
    """
    errors: List[str] = []

    # Check for missing required features
    all_required = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES
    for feat in all_required:
        if feat not in record:
            errors.append(f"Missing required clinical parameter: '{feat}'")
        elif record[feat] is None or (isinstance(record[feat], float) and np.isnan(record[feat])):
            errors.append(f"Value for '{feat}' cannot be null/empty.")

    if errors:
        return False, errors

    # Check value bounds and allowed categories
    for feat, rules in FEATURE_RANGES.items():
        val = record.get(feat)
        if val is None:
            continue

        try:
            numeric_val = float(val)
        except (ValueError, TypeError):
            errors.append(f"Field '{feat}' must be numeric, received: {val}")
            continue

        if "min" in rules and numeric_val < rules["min"]:
            errors.append(
                f"Value for '{feat}' ({numeric_val}) is below valid biological minimum of {rules['min']}."
            )
        if "max" in rules and numeric_val > rules["max"]:
            errors.append(
                f"Value for '{feat}' ({numeric_val}) exceeds valid biological maximum of {rules['max']}."
            )
        if "allowed" in rules and numeric_val not in rules["allowed"]:
            errors.append(
                f"Value for '{feat}' ({numeric_val}) is not among allowed categories: {rules['allowed']}."
            )

    return len(errors) == 0, errors
