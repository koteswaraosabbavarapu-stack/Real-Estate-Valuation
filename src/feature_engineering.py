"""
Feature Engineering Module for Indian Real Estate Valuation System.
Implements domain-specific engineered features as a scikit-learn compatible Transformer.
Avoids data leakage: strictly engineered from property attributes without referencing target price.
"""

import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Default amenity lists
FURNISHING_ITEMS = [
    "AC", "BED", "TV", "DiningTable", "Sofa", "Wardrobe", "Refrigerator", "Microwave", "WashingMachine"
]

RECREATION_ITEMS = [
    "Gymnasium", "SwimmingPool", "ClubHouse", "SportsFacility", "IndoorGames",
    "JoggingTrack", "LandscapedGardens", "Children'splayarea", "GolfCourse"
]

SECURITY_ITEMS = [
    "24X7Security", "Intercom", "PowerBackup", "MaintenanceStaff", "LiftAvailable"
]

ALL_AMENITY_COLS = [
    "MaintenanceStaff", "Gymnasium", "SwimmingPool", "LandscapedGardens",
    "JoggingTrack", "RainWaterHarvesting", "IndoorGames", "ShoppingMall",
    "Intercom", "SportsFacility", "ATM", "ClubHouse", "School", "24X7Security",
    "PowerBackup", "CarParking", "StaffQuarter", "Cafeteria", "MultipurposeRoom",
    "Hospital", "WashingMachine", "Gasconnection", "AC", "Wifi",
    "Children'splayarea", "LiftAvailable", "BED", "VaastuCompliant",
    "Microwave", "GolfCourse", "TV", "DiningTable", "Sofa", "Wardrobe",
    "Refrigerator"
]


class IndianHouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for Engineering Domain-Specific Features
    in Indian Residential Real Estate:

    - Area_per_BHK: Built-up square footage per bedroom (spaciousness indicator).
    - BHK_x_Area: Non-linear interaction between bedroom count and total floor area.
    - Furnishing_Score: Count of premium home appliances and furniture items.
    - Security_Score: Composite index of security, power, and building maintenance.
    - Recreation_Score: Composite index of club, fitness, sports, and green landscape.
    - Total_Amenities: Total active amenities count across 35 residential features.
    - Clean Location & City: Normalized text representations.
    """

    def __init__(
        self,
        furnishing_items: Optional[List[str]] = None,
        recreation_items: Optional[List[str]] = None,
        security_items: Optional[List[str]] = None,
        amenity_cols: Optional[List[str]] = None
    ):
        self.furnishing_items = furnishing_items or FURNISHING_ITEMS
        self.recreation_items = recreation_items or RECREATION_ITEMS
        self.security_items = security_items or SECURITY_ITEMS
        self.amenity_cols = amenity_cols or ALL_AMENITY_COLS

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Helper to extract numeric series safely
        def safe_numeric(col_name: str, default_val: float = 0.0) -> pd.Series:
            if col_name in df.columns:
                return pd.to_numeric(df[col_name], errors="coerce").fillna(default_val)
            return pd.Series(default_val, index=df.index)

        # 1. Clean amenity columns (replace 9 or NaN with 0)
        for col in self.amenity_cols:
            if col in df.columns:
                df[col] = df[col].replace(9, 0).fillna(0).astype(int)
            else:
                df[col] = 0

        area = safe_numeric("Area", default_val=1000.0)
        bhk = safe_numeric("No. of Bedrooms", default_val=2.0)
        resale = safe_numeric("Resale", default_val=0.0)

        # Ensure valid lower bounds
        area = np.maximum(area, 200.0)
        bhk = np.maximum(bhk, 1.0)

        # 2. Area per bedroom (spaciousness ratio)
        df["Area_per_BHK"] = area / bhk

        # 3. BHK and Area interaction
        df["BHK_x_Area"] = bhk * area

        # 4. Domain score features
        f_cols = [c for c in self.furnishing_items if c in df.columns]
        df["Furnishing_Score"] = df[f_cols].sum(axis=1) if f_cols else 0

        r_cols = [c for c in self.recreation_items if c in df.columns]
        df["Recreation_Score"] = df[r_cols].sum(axis=1) if r_cols else 0

        s_cols = [c for c in self.security_items if c in df.columns]
        df["Security_Score"] = df[s_cols].sum(axis=1) if s_cols else 0

        a_cols = [c for c in self.amenity_cols if c in df.columns]
        df["Total_Amenities"] = df[a_cols].sum(axis=1) if a_cols else 0

        # 5. Clean categorical location and city text
        if "Location" in df.columns:
            df["Location"] = df["Location"].astype(str).str.strip()
        else:
            df["Location"] = "Unknown"

        if "City" in df.columns:
            df["City"] = df["City"].astype(str).str.strip()
        else:
            df["City"] = "Unknown"

        # Drop ID or unneeded target leakage columns if present
        for col in ["Id", "Price", "Price_per_sqft", "Price_Lakhs"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to apply IndianHouseFeatureEngineer to a DataFrame directly.
    """
    fe = IndianHouseFeatureEngineer()
    return fe.transform(df)


if __name__ == "__main__":
    from src.data_loader import load_clean_housing_data
    
    print("Testing Indian House Feature Engineering Module...")
    df = load_clean_housing_data()
    X = df.drop(columns=["Price", "Price_per_sqft", "Price_Lakhs"], errors="ignore")
    fe = IndianHouseFeatureEngineer()
    X_trans = fe.transform(X)
    print(f"Original shape: {X.shape}, Transformed shape: {X_trans.shape}")
    print("Engineered feature columns preview:")
    preview_cols = ["City", "Location", "Area", "No. of Bedrooms", "Area_per_BHK", "BHK_x_Area", "Furnishing_Score", "Security_Score", "Total_Amenities"]
    print(X_trans[preview_cols].head())
    print("Feature Engineering test completed successfully!")
