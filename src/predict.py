"""
Prediction Module for Real Estate Valuation System.
Loads the trained ML pipeline and generates price predictions for single or multiple properties.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Union, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

_cached_model = None
_default_row = None


def get_default_property_template() -> Dict[str, Any]:
    """
    Get a complete baseline property template using median and mode values from train.csv.
    Ensures that when a user provides partial inputs via UI or API, missing features have valid defaults.
    """
    global _default_row
    if _default_row is None:
        if TRAIN_DATA_PATH.exists():
            df = pd.read_csv(TRAIN_DATA_PATH).drop(columns=["SalePrice", "Id"], errors="ignore")
            defaults = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    defaults[col] = float(df[col].median())
                else:
                    mode_val = df[col].mode()
                    defaults[col] = str(mode_val.iloc[0]) if not mode_val.empty else "None"
            _default_row = defaults
        else:
            _default_row = {}
    return _default_row.copy()


def load_valuation_model():
    """
    Load the saved trained model pipeline from models/house_price_model.pkl.
    """
    global _cached_model
    if _cached_model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Please run src/train.py first.")
        _cached_model = joblib.load(MODEL_PATH)
    return _cached_model


def predict_house_price(property_data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]) -> Union[float, List[float]]:
    """
    Estimate the market valuation / sale price of given residential property data.

    Args:
        property_data: Dict of feature key-values, list of dicts, or pandas DataFrame.

    Returns:
        float or List[float]: Predicted SalePrice in USD ($).
    """
    model = load_valuation_model()
    template = get_default_property_template()

    # Case 1: Single Property Dict
    if isinstance(property_data, dict):
        full_record = template.copy()
        full_record.update(property_data)
        input_df = pd.DataFrame([full_record])
        prediction = model.predict(input_df)[0]
        return float(round(prediction, 2))

    # Case 2: List of property dicts
    elif isinstance(property_data, list):
        rows = []
        for item in property_data:
            full_record = template.copy()
            full_record.update(item)
            rows.append(full_record)
        input_df = pd.DataFrame(rows)
        predictions = model.predict(input_df)
        return [float(round(p, 2)) for p in predictions]

    # Case 3: Pandas DataFrame
    elif isinstance(property_data, pd.DataFrame):
        input_df = property_data.copy()
        # Fill any missing columns from template
        for col, val in template.items():
            if col not in input_df.columns:
                input_df[col] = val
        predictions = model.predict(input_df)
        if len(predictions) == 1:
            return float(round(predictions[0], 2))
        return [float(round(p, 2)) for p in predictions]

    else:
        raise TypeError("property_data must be a dict, list of dicts, or pandas DataFrame")


if __name__ == "__main__":
    print("Testing Prediction Module...")
    
    # Sample Test Property
    sample_house = {
        "OverallQual": 7,
        "GrLivArea": 1710,
        "TotalBsmtSF": 856,
        "1stFlrSF": 856,
        "2ndFlrSF": 854,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "TotRmsAbvGrd": 8,
        "YearBuilt": 2003,
        "YearRemodAdd": 2003,
        "GarageCars": 2,
        "GarageArea": 548,
        "Neighborhood": "CollgCr",
        "LotArea": 8450,
        "Fireplaces": 0,
    }

    predicted_val = predict_house_price(sample_house)
    print(f"Sample Property Valuation:")
    print(f"  Neighborhood:    {sample_house['Neighborhood']}")
    print(f"  Overall Quality: {sample_house['OverallQual']}/10")
    print(f"  Living Area:     {sample_house['GrLivArea']} sq ft")
    print(f"  Bedrooms:        {sample_house['BedroomAbvGr']}")
    print(f"  Bathrooms:       {sample_house['FullBath']} Full, {sample_house['HalfBath']} Half")
    print(f"  Year Built:      {sample_house['YearBuilt']}")
    print(f"  --> Estimated Market Value: ${predicted_val:,.2f}")
    print("\nPrediction Module tested successfully!")
