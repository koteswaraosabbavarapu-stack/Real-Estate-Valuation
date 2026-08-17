"""
Data Loader Module for Real Estate Valuation System.
Provides reusable functions to load and validate train and test datasets.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd


# Base directory relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"


def load_train_data(filepath: str | Path = TRAIN_PATH) -> pd.DataFrame:
    """
    Load the training dataset containing property features and SalePrice.

    Args:
        filepath: Path to train.csv file.

    Returns:
        pd.DataFrame: Loaded training dataset.

    Raises:
        FileNotFoundError: If train.csv does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Training data file not found at: {path}")

    df = pd.read_csv(path)
    if "SalePrice" not in df.columns:
        raise ValueError(f"Target column 'SalePrice' not found in training dataset at {path}")

    return df


def load_test_data(filepath: str | Path = TEST_PATH) -> pd.DataFrame:
    """
    Load the test dataset containing property features without SalePrice.

    Args:
        filepath: Path to test.csv file.

    Returns:
        pd.DataFrame: Loaded test dataset.

    Raises:
        FileNotFoundError: If test.csv does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Test data file not found at: {path}")

    df = pd.read_csv(path)
    return df


def validate_dataset(df: pd.DataFrame, is_train: bool = True) -> Dict[str, Any]:
    """
    Validate the structure, dimensions, target column, duplicates, and missing values.

    Args:
        df: DataFrame to validate.
        is_train: True if validating train dataset, False for test.

    Returns:
        Dict containing validation summary metrics.
    """
    num_rows, num_cols = df.shape
    num_duplicates = int(df.duplicated().sum())
    has_target = "SalePrice" in df.columns
    missing_counts = df.isnull().sum()
    cols_with_missing = int((missing_counts > 0).sum())
    total_missing_values = int(missing_counts.sum())

    validation_result = {
        "is_train": is_train,
        "rows": num_rows,
        "columns": num_cols,
        "duplicate_rows": num_duplicates,
        "has_sale_price": has_target,
        "columns_with_missing_values": cols_with_missing,
        "total_missing_cells": total_missing_values,
        "id_column_present": "Id" in df.columns,
    }

    if is_train and not has_target:
        raise ValueError("Validation Failed: Target column 'SalePrice' missing in training set.")

    return validation_result


def get_data_splits(filepath: str | Path = TRAIN_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load training data and separate into features (X) and target (y).

    Args:
        filepath: Path to train.csv file.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features X and Target y (SalePrice).
    """
    df = load_train_data(filepath)
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]
    return X, y


if __name__ == "__main__":
    print("Testing data_loader module...")
    train_df = load_train_data()
    test_df = load_test_data()

    train_val = validate_dataset(train_df, is_train=True)
    test_val = validate_dataset(test_df, is_train=False)

    print("Train Validation Summary:", train_val)
    print("Test Validation Summary:", test_val)
    print("Data Loader Module tested successfully!")
