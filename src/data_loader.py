"""Data loading, validation, and splitting module for Heart Disease Risk Prediction.

Handles safe dataset retrieval from local files or UCI repository fallbacks,
verifies data schema integrity, and generates stratified train/test partitions.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "heart_disease.csv")
UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/45/data.csv"

EXPECTED_FEATURE_COLUMNS: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET_COLUMNS = ["num", "target"]


def load_heart_disease_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load the UCI Heart Disease dataset.

    Attempts to load from the specified filepath or default local path.
    Falls back to fetching directly from the official UCI repository if the
    local file is absent.

    Args:
        filepath: Optional path to a CSV file.

    Returns:
        pd.DataFrame: Cleanly loaded DataFrame with standardized columns.

    Raises:
        FileNotFoundError: If neither the file nor remote endpoint is accessible.
        ValueError: If loaded data fails structural schema validation.
    """
    path_to_try = filepath or DEFAULT_DATA_PATH

    if os.path.exists(path_to_try):
        df = pd.read_csv(path_to_try)
    else:
        try:
            df = pd.read_csv(UCI_DATA_URL)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not load data from local path '{path_to_try}' or UCI URL '{UCI_DATA_URL}'."
            ) from exc

    # Ensure binary target exists
    if "target" not in df.columns:
        if "num" in df.columns:
            df["target"] = (df["num"] > 0).astype(int)
        else:
            raise ValueError("Dataset does not contain 'num' or 'target' column.")

    validate_raw_dataframe(df)
    return df


def validate_raw_dataframe(df: pd.DataFrame) -> bool:
    """Validate that the dataset satisfies structural and schema expectations.

    Args:
        df: Input DataFrame.

    Returns:
        bool: True if validation succeeds.

    Raises:
        ValueError: If required columns are missing or the dataset is empty.
    """
    if df.empty:
        raise ValueError("Loaded dataset is empty (0 rows).")

    missing_features = [col for col in EXPECTED_FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing required feature columns: {missing_features}")

    if "target" not in df.columns and "num" not in df.columns:
        raise ValueError("Dataset missing target column ('target' or 'num').")

    return True


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract the feature matrix X and binary classification target y.

    Args:
        df: Cleaned or validated DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y)
    """
    validate_raw_dataframe(df)

    X = df[EXPECTED_FEATURE_COLUMNS].copy()
    if "target" in df.columns:
        y = df["target"].astype(int).copy()
    else:
        y = (df["num"] > 0).astype(int).copy()

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train/test split.

    Preserves target class distribution in both splits to ensure reliable
    and leakage-free model evaluation.

    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Proportion of dataset allocated to the test set (default: 0.2).
        random_state: Random seed for deterministic reproducibility.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Union[int, float, dict]]:
    """Compute key demographic and clinical summary statistics.

    Args:
        df: Heart disease DataFrame.

    Returns:
        Dict containing counts, class balances, and missing value indicators.
    """
    total_records = len(df)
    target_series = df["target"] if "target" in df.columns else (df["num"] > 0).astype(int)
    disease_cases = int(target_series.sum())
    healthy_cases = total_records - disease_cases

    missing_summary = df[EXPECTED_FEATURE_COLUMNS].isnull().sum().to_dict()

    return {
        "total_records": total_records,
        "healthy_cases": healthy_cases,
        "healthy_percentage": round((healthy_cases / total_records) * 100, 2),
        "disease_cases": disease_cases,
        "disease_percentage": round((disease_cases / total_records) * 100, 2),
        "missing_values": {k: v for k, v in missing_summary.items() if v > 0},
        "feature_count": len(EXPECTED_FEATURE_COLUMNS),
    }


if __name__ == "__main__":
    data = load_heart_disease_data()
    summary = get_dataset_summary(data)
    print("Dataset Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    X_mat, y_vec = prepare_features_and_target(data)
    X_tr, X_te, y_tr, y_te = split_data(X_mat, y_vec)
    print(f"Train split shape: X={X_tr.shape}, y={y_tr.shape}")
    print(f"Test split shape:  X={X_te.shape}, y={y_te.shape}")
