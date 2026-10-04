"""
Preprocessing Module for Indian Real Estate Valuation System.
Builds scikit-learn ColumnTransformer and full pipeline architectures.
Handles numerical scaling, missing value imputation, categorical one-hot encoding for City,
and target encoding for micro-market Location without data leakage.
"""

import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, TargetEncoder
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import IndianHouseFeatureEngineer, ALL_AMENITY_COLS


def get_feature_column_names(df_engineered: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Identify numerical, city, and location column names from an engineered DataFrame.
    """
    exclude_cols = ["Price", "Price_per_sqft", "Price_Lakhs", "Id"]
    available_cols = [c for c in df_engineered.columns if c not in exclude_cols]

    city_cols = ["City"] if "City" in available_cols else []
    loc_cols = ["Location"] if "Location" in available_cols else []
    
    num_cols = [
        c for c in available_cols
        if c not in ["City", "Location"] and (
            pd.api.types.is_numeric_dtype(df_engineered[c]) or c in ALL_AMENITY_COLS or
            c in ["Area", "No. of Bedrooms", "Resale", "Area_per_BHK", "BHK_x_Area", "Furnishing_Score", "Recreation_Score", "Security_Score", "Total_Amenities"]
        )
    ]

    return num_cols, city_cols, loc_cols


def build_preprocessor(num_cols: List[str], city_cols: List[str], loc_cols: List[str]) -> ColumnTransformer:
    """
    Construct ColumnTransformer for numerical, city, and location features.

    Numerical Pipeline:
    - Median Imputation
    - RobustScaler (handles real estate skewed distributions and outliers)

    City Pipeline:
    - Imputer (constant 'Unknown')
    - OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    Location Pipeline:
    - Imputer (constant 'Unknown')
    - TargetEncoder(target_type='continuous', smooth='auto') for 1,700+ distinct micro-markets
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    city_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    loc_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("target_enc", TargetEncoder(target_type="continuous", smooth="auto")),
    ])

    transformers = [
        ("num", num_pipeline, num_cols),
    ]

    if city_cols:
        transformers.append(("city", city_pipeline, city_cols))
    if loc_cols:
        transformers.append(("loc", loc_pipeline, loc_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


def build_full_pipeline(regressor, num_cols: List[str], city_cols: List[str], loc_cols: List[str]) -> Pipeline:
    """
    Build complete end-to-end Pipeline:
    Raw Data -> Feature Engineering -> Preprocessor (Imputation / Scaling / Encoding) -> Regressor
    """
    preprocessor = build_preprocessor(num_cols, city_cols, loc_cols)

    pipeline = Pipeline([
        ("feature_engineer", IndianHouseFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    return pipeline


if __name__ == "__main__":
    from src.data_loader import load_clean_housing_data
    from sklearn.linear_model import Ridge

    print("Testing Preprocessing Module...")
    clean_df = load_clean_housing_data()
    X = clean_df.drop(columns=["Price", "Price_per_sqft", "Price_Lakhs"], errors="ignore")
    y = np.log1p(clean_df["Price"])

    fe = IndianHouseFeatureEngineer()
    X_fe = fe.transform(X)
    num_cols, city_cols, loc_cols = get_feature_column_names(X_fe)
    print(f"Features: {len(num_cols)} Numerical, {len(city_cols)} City, {len(loc_cols)} Location")

    pipe = build_full_pipeline(Ridge(alpha=10.0), num_cols, city_cols, loc_cols)
    pipe.fit(X, y)
    preds = pipe.predict(X.head())
    print("Sample pipeline predicted log1p prices:", preds[:3])
    print("Converted to INR (Rupees):", np.expm1(preds[:3]))
    print("Preprocessing Module test completed successfully!")
