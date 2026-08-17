"""
Preprocessing Module for Real Estate Valuation System.
Builds scikit-learn ColumnTransformer and full pipeline architectures.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.feature_engineering import HouseFeatureEngineer


# Categorical features where 'NA' means absence of that amenity
NONE_CAT_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType"
]


class DatasetCategoricalCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans domain-specific categorical features:
    Replaces NA with 'None' for amenities where NA denotes absence of the feature.
    """

    def __init__(self, none_cols: List[str] = None):
        self.none_cols = none_cols or NONE_CAT_COLS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.none_cols:
            if col in df.columns:
                df[col] = df[col].fillna("None").astype(str)
        return df


def get_feature_column_names(df_engineered: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical column names from an engineered DataFrame.
    """
    cols_to_exclude = ["Id", "SalePrice"]
    available_cols = [c for c in df_engineered.columns if c not in cols_to_exclude]

    num_cols = df_engineered[available_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_engineered[available_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Construct ColumnTransformer for numerical and categorical features.

    Numerical Pipeline:
    - Median Imputation
    - StandardScaler

    Categorical Pipeline:
    - Missing Imputation (constant 'Missing')
    - OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop"
    )

    return preprocessor


def build_full_pipeline(regressor, num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    """
    Build complete end-to-end Pipeline:
    Raw Data -> Categorical Cleaning -> Feature Engineering -> Preprocessor (Imputation/Encoding/Scaling) -> Regressor
    """
    preprocessor = build_preprocessor(num_cols, cat_cols)

    pipeline = Pipeline([
        ("cleaner", DatasetCategoricalCleaner()),
        ("feature_engineer", HouseFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    return pipeline


if __name__ == "__main__":
    from src.data_loader import load_train_data
    from sklearn.linear_model import Ridge

    print("Testing Preprocessing Module...")
    raw_df = load_train_data()
    X = raw_df.drop(columns=["SalePrice"])
    y = raw_df["SalePrice"]

    # Test feature extraction after engineering
    fe = HouseFeatureEngineer()
    cleaner = DatasetCategoricalCleaner()
    df_transformed = fe.transform(cleaner.transform(X))
    num_cols, cat_cols = get_feature_column_names(df_transformed)

    print(f"Identified {len(num_cols)} numerical and {len(cat_cols)} categorical features.")

    pipe = build_full_pipeline(Ridge(), num_cols, cat_cols)
    pipe.fit(X, y)
    preds = pipe.predict(X.head())
    print("Sample pipeline predictions on first 5 properties:")
    for i, p in enumerate(preds):
        print(f"  Property {i+1}: Actual ${y.iloc[i]:,.2f} | Predicted ${p:,.2f}")
    print("Preprocessing Module tested successfully!")
