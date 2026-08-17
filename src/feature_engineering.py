"""
Feature Engineering Module for Real Estate Valuation System.
Implements domain-specific engineered features as a scikit-learn compatible Transformer.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for Engineering Domain-Specific Features:
    - TotalSF: Total square footage across basement, 1st floor, and 2nd floor.
    - TotalBathrooms: Combined full and half baths (including basement).
    - TotalPorchSF: Sum of deck, open porch, enclosed porch, 3-season, and screen porch area.
    - HouseAge: Age of property at time of sale.
    - RemodAge: Years since remodel at time of sale.
    - GarageAge: Age of garage at time of sale.
    - HasPool, HasGarage, HasBsmt, HasFireplace: Indicator flags for key luxury amenities.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create copy to prevent modifying original dataframe
        df = X.copy()

        # Fill missing values for calculation columns safely with 0
        def safe_get(col, default=0.0):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(default)
            return default

        total_bsmt_sf = safe_get("TotalBsmtSF")
        first_flr_sf = safe_get("1stFlrSF")
        second_flr_sf = safe_get("2ndFlrSF")
        
        # 1. Total Square Footage
        df["TotalSF"] = total_bsmt_sf + first_flr_sf + second_flr_sf

        # 2. Total Bathrooms
        full_bath = safe_get("FullBath")
        half_bath = safe_get("HalfBath")
        bsmt_full_bath = safe_get("BsmtFullBath")
        bsmt_half_bath = safe_get("BsmtHalfBath")
        df["TotalBathrooms"] = full_bath + (0.5 * half_bath) + bsmt_full_bath + (0.5 * bsmt_half_bath)

        # 3. Total Porch & Outdoor Living Area
        wood_deck_sf = safe_get("WoodDeckSF")
        open_porch_sf = safe_get("OpenPorchSF")
        enclosed_porch = safe_get("EnclosedPorch")
        three_ssn_porch = safe_get("3SsnPorch")
        screen_porch = safe_get("ScreenPorch")
        df["TotalPorchSF"] = wood_deck_sf + open_porch_sf + enclosed_porch + three_ssn_porch + screen_porch

        # 4. Property Age Calculations
        yr_sold = safe_get("YrSold", default=2010)
        year_built = safe_get("YearBuilt", default=1970)
        year_remod = safe_get("YearRemodAdd", default=1970)
        garage_yr = safe_get("GarageYrBlt", default=year_built)

        df["HouseAge"] = np.maximum(0, yr_sold - year_built)
        df["RemodAge"] = np.maximum(0, yr_sold - year_remod)
        df["GarageAge"] = np.maximum(0, yr_sold - garage_yr)

        # 5. Amenity indicator features
        pool_area = safe_get("PoolArea")
        garage_area = safe_get("GarageArea")
        fireplaces = safe_get("Fireplaces")

        df["HasPool"] = (pool_area > 0).astype(int)
        df["HasGarage"] = (garage_area > 0).astype(int)
        df["HasBsmt"] = (total_bsmt_sf > 0).astype(int)
        df["HasFireplace"] = (fireplaces > 0).astype(int)

        # Drop identifier if present
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])

        return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to apply HouseFeatureEngineer to a DataFrame directly.
    """
    fe = HouseFeatureEngineer()
    return fe.transform(df)


if __name__ == "__main__":
    from data_loader import load_train_data
    raw_df = load_train_data()
    print("Testing Feature Engineering Module...")
    engineered = engineer_features(raw_df)
    print(f"Original columns: {raw_df.shape[1]}, Engineered columns: {engineered.shape[1]}")
    print("Engineered features preview:")
    preview_cols = ["TotalSF", "TotalBathrooms", "TotalPorchSF", "HouseAge", "RemodAge", "HasGarage", "HasFireplace"]
    print(engineered[preview_cols].head())
    print("Feature Engineering Module tested successfully!")
