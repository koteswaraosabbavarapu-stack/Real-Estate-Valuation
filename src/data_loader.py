"""
Data Loader Module for Indian Real Estate Valuation System.
Downloads, caches, loads, and validates the 6 Major Metropolitan Cities Indian residential housing dataset.
Covers Mumbai, Bangalore, Delhi, Chennai, Hyderabad, and Kolkata.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import urllib.request
import pandas as pd
import numpy as np

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
COMBINED_DATA_PATH = RAW_DATA_DIR / "indian_metro_housing.csv"

# Public GitHub mirror repository for the 6 Indian metro cities housing datasets
# Original Source: Housing Prices in Metropolitan Areas of India (Kaggle / Open Dataset)
CITY_DATASET_URLS = {
    "Bangalore": "https://raw.githubusercontent.com/anish2105/House-Price-Prediction/master/Bangalore.csv",
    "Mumbai": "https://raw.githubusercontent.com/anish2105/House-Price-Prediction/master/Mumbai.csv",
    "Delhi": "https://raw.githubusercontent.com/anish2105/House-Price-Prediction/master/Delhi.csv",
    "Chennai": "https://raw.githubusercontent.com/anish2105/House-Price-Prediction/master/Chennai.csv",
    "Hyderabad": "https://raw.githubusercontent.com/anish2105/House-Price-Prediction/master/Hyderabad.csv",
    "Kolkata": "https://raw.githubusercontent.com/anish2105/House-Price-Prediction/master/Kolkata.csv"
}

# Amenity column names present in the 6-metro dataset
AMENITY_COLUMNS = [
    "MaintenanceStaff", "Gymnasium", "SwimmingPool", "LandscapedGardens",
    "JoggingTrack", "RainWaterHarvesting", "IndoorGames", "ShoppingMall",
    "Intercom", "SportsFacility", "ATM", "ClubHouse", "School", "24X7Security",
    "PowerBackup", "CarParking", "StaffQuarter", "Cafeteria", "MultipurposeRoom",
    "Hospital", "WashingMachine", "Gasconnection", "AC", "Wifi",
    "Children'splayarea", "LiftAvailable", "BED", "VaastuCompliant",
    "Microwave", "GolfCourse", "TV", "DiningTable", "Sofa", "Wardrobe",
    "Refrigerator"
]


def download_city_datasets_if_needed(force_download: bool = False) -> None:
    """
    Download raw city CSV files from the public repository and cache locally.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download individual city CSVs
    for city, url in CITY_DATASET_URLS.items():
        city_csv_path = RAW_DATA_DIR / f"{city}.csv"
        if not city_csv_path.exists() or force_download:
            print(f"Downloading dataset for {city} from {url}...")
            urllib.request.urlretrieve(url, city_csv_path)
            print(f"  -> Saved {city}.csv ({city_csv_path.stat().st_size / 1024:.1f} KB)")

    # Create combined dataset if not exists
    if not COMBINED_DATA_PATH.exists() or force_download:
        print("Combining all 6 metropolitan datasets into single dataset...")
        dfs = []
        for city in CITY_DATASET_URLS.keys():
            city_path = RAW_DATA_DIR / f"{city}.csv"
            df_city = pd.read_csv(city_path)
            df_city["City"] = city
            dfs.append(df_city)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(COMBINED_DATA_PATH, index=False)
        print(f"Saved combined Indian real-estate dataset: {COMBINED_DATA_PATH} ({len(combined_df)} records)")


def load_raw_dataset(filepath: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Load raw Indian housing dataset. Automatically downloads if not found.
    """
    path = Path(filepath) if filepath else COMBINED_DATA_PATH
    if not path.exists():
        download_city_datasets_if_needed()
    
    df = pd.read_csv(path)
    return df


def load_clean_housing_data(filepath: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Load and clean the Indian housing dataset for ML training and EDA.
    
    Cleaning steps:
    1. Standardize amenity indicators: Replace 9 (not specified/missing) with 0.
    2. Drop duplicate property listings.
    3. Filter out unrealistic non-residential data-entry anomalies (Area < 250 sqft or > 10,000 sqft, Price/sqft < ₹1,500 or > ₹55,000).
    4. Clean and strip location names.
    5. Add Price_in_Lakhs and Price_per_sqft for analytics.
    """
    df = load_raw_dataset(filepath)
    
    # Clean amenity columns (9 -> 0)
    amenity_cols = [c for c in AMENITY_COLUMNS if c in df.columns]
    for c in amenity_cols:
        df[c] = df[c].replace(9, 0).fillna(0).astype(int)
    
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Clean location strings
    if "Location" in df.columns:
        df["Location"] = df["Location"].astype(str).str.strip()
    
    # Standardize BHK / bedrooms column name if present
    if "No. of Bedrooms" in df.columns:
        df["No. of Bedrooms"] = pd.to_numeric(df["No. of Bedrooms"], errors="coerce").fillna(2).astype(int)
    
    # Compute derived market metrics
    df["Price_per_sqft"] = df["Price"] / df["Area"]
    df["Price_Lakhs"] = df["Price"] / 100000.0
    
    # Filter realistic residential price/sqft and area boundaries to remove extreme data entry typos
    valid_mask = (
        (df["Price_per_sqft"] >= 1800) &
        (df["Price_per_sqft"] <= 48000) &
        (df["Area"] >= 300) &
        (df["Area"] <= 8500) &
        (df["Price"] >= 600000) &
        (df["No. of Bedrooms"] >= 1) &
        (df["No. of Bedrooms"] <= 8)
    )
    df_clean = df[valid_mask].copy().reset_index(drop=True)
    
    return df_clean


def get_data_splits(test_size: float = 0.20, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load cleaned dataset and split into train and test sets.
    """
    from sklearn.model_selection import train_test_split
    
    df = load_clean_housing_data()
    X = df.drop(columns=["Price", "Price_per_sqft", "Price_Lakhs"], errors="ignore")
    y = df["Price"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=X["City"]
    )
    return X_train, X_test, y_train, y_test


def validate_dataset(df: pd.DataFrame, is_train: bool = True) -> Dict[str, Any]:
    """
    Validate dataset structure, columns, missing values, and data integrity.
    """
    required_cols = ["Price", "Area", "Location", "No. of Bedrooms", "Resale", "City"] if is_train else ["Area", "Location", "No. of Bedrooms", "Resale", "City"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    summary = {
        "is_train": is_train,
        "rows": len(df),
        "columns": len(df.columns),
        "cities": df["City"].unique().tolist() if "City" in df.columns else [],
        "distinct_localities": df["Location"].nunique() if "Location" in df.columns else 0,
        "missing_required_columns": missing_cols,
        "is_valid": len(missing_cols) == 0
    }
    return summary


if __name__ == "__main__":
    print("Testing data_loader module for Indian Real Estate...")
    download_city_datasets_if_needed()
    raw_df = load_raw_dataset()
    print(f"Raw dataset shape: {raw_df.shape}")
    clean_df = load_clean_housing_data()
    print(f"Cleaned dataset shape: {clean_df.shape}")
    print("\nSummary by City in Cleaned Data:")
    print(clean_df.groupby("City").agg(
        Count=("Price", "count"),
        Median_Price_Lakhs=("Price_Lakhs", "median"),
        Mean_Area_SqFt=("Area", "mean"),
        Median_Rate_SqFt=("Price_per_sqft", "median")
    ))
    print("\nValidation:", validate_dataset(clean_df))
    print("data_loader test completed successfully!")
