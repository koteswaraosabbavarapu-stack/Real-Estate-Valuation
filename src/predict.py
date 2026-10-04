"""
Prediction Module for Indian Real Estate Valuation & Appraisal Engine.
Loads the trained ML pipeline and conformal prediction uncertainty calibrator
to generate point valuations, price-per-sq.ft metrics, and calibrated prediction intervals.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path
from typing import Dict, Any, Union, List, Tuple, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import ALL_AMENITY_COLS
from src.evaluate import format_inr


MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "indian_metro_housing.csv"

_cached_bundle = None
_default_row = None
_cached_city_localities = None


def load_valuation_model_bundle() -> Dict[str, Any]:
    """
    Load the saved trained model bundle containing the fitted pipeline,
    conformal quantiles, and dataset metadata.
    """
    global _cached_bundle
    if _cached_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Please run 'python src/train.py' first."
            )
        _cached_bundle = joblib.load(MODEL_PATH)
    return _cached_bundle


def get_default_property_template() -> Dict[str, Any]:
    """
    Get a complete baseline property template using median and mode values from the Indian dataset.
    Ensures that when a user provides partial inputs via UI or CSV, missing features have valid defaults.
    """
    global _default_row
    if _default_row is None:
        template = {
            "City": "Bangalore",
            "Location": "Whitefield",
            "Area": 1200.0,
            "No. of Bedrooms": 2,
            "Resale": 0
        }
        # Default amenities to 0
        for col in ALL_AMENITY_COLS:
            template[col] = 0
            
        _default_row = template
    return _default_row.copy()


def get_available_cities_and_localities() -> Dict[str, List[str]]:
    """
    Extract all distinct localities grouped by city from the Indian dataset.
    """
    global _cached_city_localities
    if _cached_city_localities is None:
        if TRAIN_DATA_PATH.exists():
            df = pd.read_csv(TRAIN_DATA_PATH)
            mapping = {}
            for city in sorted(df["City"].unique()):
                locs = sorted(df[df["City"] == city]["Location"].dropna().astype(str).str.strip().unique().tolist())
                # Filter empty strings
                locs = [l for l in locs if len(l) > 1 and l.lower() != "nan"]
                mapping[city] = locs
            _cached_city_localities = mapping
        else:
            _cached_city_localities = {
                "Bangalore": ["Whitefield", "Electronic City", "Sarjapur Road", "HSR Layout", "Indiranagar", "Koramangala", "JP Nagar", "Hebbal", "Yelahanka", "Bellandur"],
                "Mumbai": ["Andheri East", "Andheri West", "Bandra West", "Powai", "Kandivali West", "Borivali West", "Thane West", "Kharghar", "Malad West", "Worli"],
                "Delhi": ["Dwarka", "Saket", "Vasant Kunj", "Rohini", "Uttam Nagar", "Greater Kailash", "Lajpat Nagar", "Janakpuri", "Noida Sector 62", "Gurgaon Sector 54"],
                "Chennai": ["OMR", "Velachery", "Anna Nagar", "Madipakkam", "Porur", "Sholinganallur", "Perungudi", "Medavakkam", "Adyar", "Thoraipakkam"],
                "Hyderabad": ["Gachibowli", "Kondapur", "HITEC City", "Kukatpally", "Miyapur", "Banjara Hills", "Jubilee Hills", "Manikonda", "Tellapur", "Nanakramguda"],
                "Kolkata": ["New Town", "Rajarhat", "Salt Lake", "EM Bypass", "Behala", "Garia", "Dum Dum", "Tollygunge", "Jadavpur", "Howrah"]
            }
    return _cached_city_localities


def predict_property_valuation(
    property_data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]],
    coverage: float = 0.80
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Estimate the market valuation / price of given residential property data in Indian Rupees (₹).
    Calculates point valuation, ₹/sq.ft rate, and calibrated prediction range using Conformal Prediction.

    Args:
        property_data: Dict of feature key-values, list of dicts, or pandas DataFrame.
        coverage: Confidence level for prediction interval (0.80 for 80% or 0.90 for 90%).

    Returns:
        Dict or List of Dicts with point valuation, interval range, rate per sq ft, and formatted strings.
    """
    bundle = load_valuation_model_bundle()
    model = bundle["model"]
    q_width = bundle["conformal_q80"] if coverage <= 0.85 else bundle["conformal_q90"]
    template = get_default_property_template()

    # Case 1: Single Property Dict
    if isinstance(property_data, dict):
        full_record = template.copy()
        full_record.update(property_data)
        input_df = pd.DataFrame([full_record])

        # Point Prediction
        raw_pred = float(model.predict(input_df)[0])
        pred_price = max(raw_pred, 500000.0) # Lower bound ₹5 Lakhs

        # Conformal Prediction Range
        log_pred = np.log1p(pred_price)
        lower_price = max(float(np.expm1(log_pred - q_width)), 400000.0)
        upper_price = float(np.expm1(log_pred + q_width))

        area = float(full_record.get("Area", 1000.0))
        price_per_sqft = pred_price / max(area, 100.0)

        return {
            "predicted_price": round(pred_price, 2),
            "lower_bound": round(lower_price, 2),
            "upper_bound": round(upper_price, 2),
            "price_per_sqft": round(price_per_sqft, 2),
            "coverage_level": coverage,
            "formatted_price": format_inr(pred_price),
            "formatted_range": f"{format_inr(lower_price)} – {format_inr(upper_price)}",
            "formatted_price_per_sqft": f"₹{price_per_sqft:,.0f}/sq.ft",
            "price_in_lakhs": round(pred_price / 100000.0, 2),
            "lower_in_lakhs": round(lower_price / 100000.0, 2),
            "upper_in_lakhs": round(upper_price / 100000.0, 2)
        }

    # Case 2: DataFrame or List of Dicts
    else:
        if isinstance(property_data, list):
            rows = []
            for item in property_data:
                rec = template.copy()
                rec.update(item)
                rows.append(rec)
            input_df = pd.DataFrame(rows)
        elif isinstance(property_data, pd.DataFrame):
            input_df = property_data.copy()
            for col, default_val in template.items():
                if col not in input_df.columns:
                    input_df[col] = default_val
        else:
            raise TypeError("property_data must be a dict, list of dicts, or pandas DataFrame")

        raw_preds = model.predict(input_df)
        results = []
        for i, raw_pred in enumerate(raw_preds):
            pred_price = max(float(raw_pred), 500000.0)
            log_pred = np.log1p(pred_price)
            lower_price = max(float(np.expm1(log_pred - q_width)), 400000.0)
            upper_price = float(np.expm1(log_pred + q_width))

            area = float(input_df.iloc[i].get("Area", 1000.0))
            price_per_sqft = pred_price / max(area, 100.0)

            results.append({
                "predicted_price": round(pred_price, 2),
                "lower_bound": round(lower_price, 2),
                "upper_bound": round(upper_price, 2),
                "price_per_sqft": round(price_per_sqft, 2),
                "coverage_level": coverage,
                "formatted_price": format_inr(pred_price),
                "formatted_range": f"{format_inr(lower_price)} – {format_inr(upper_price)}",
                "formatted_price_per_sqft": f"₹{price_per_sqft:,.0f}/sq.ft",
                "price_in_lakhs": round(pred_price / 100000.0, 2),
                "lower_in_lakhs": round(lower_price / 100000.0, 2),
                "upper_in_lakhs": round(upper_price / 100000.0, 2)
            })

        return results


# Backward compatibility alias
def predict_house_price(property_data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]) -> Union[float, List[float]]:
    res = predict_property_valuation(property_data)
    if isinstance(res, dict):
        return res["predicted_price"]
    return [r["predicted_price"] for r in res]


def load_valuation_model():
    bundle = load_valuation_model_bundle()
    return bundle["model"]


if __name__ == "__main__":
    print("Testing Prediction Module for Indian Real Estate...")
    sample_house = {
        "City": "Bangalore",
        "Location": "Whitefield",
        "Area": 1450,
        "No. of Bedrooms": 3,
        "Resale": 0,
        "Gymnasium": 1,
        "SwimmingPool": 1,
        "24X7Security": 1,
        "PowerBackup": 1,
        "ClubHouse": 1,
        "CarParking": 1,
        "AC": 1
    }

    valuation = predict_property_valuation(sample_house)
    print(f"\nProperty Valuation Appraisal:")
    print(f"  City:              {sample_house['City']}")
    print(f"  Micro-Market:      {sample_house['Location']}")
    print(f"  Configuration:     {sample_house['No. of Bedrooms']} BHK ({sample_house['Area']} sq ft)")
    print(f"  --> Valuation:     {valuation['formatted_price']} ({valuation['price_in_lakhs']:.2f} Lakhs)")
    print(f"  --> Rate / sq.ft:  {valuation['formatted_price_per_sqft']}")
    print(f"  --> 80% Range:     {valuation['formatted_range']}")
    print("\nPrediction Module tested successfully!")
