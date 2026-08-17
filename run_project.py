"""
Master Orchestration Script for Real Estate Valuation & Automated Appraisal Engine.
Runs end-to-end: Data Loading -> EDA -> Feature Engineering & Preprocessing -> Training -> Evaluation -> Prediction Demo.
"""

import sys
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_train_data, load_test_data, validate_dataset
from src.eda_runner import run_eda
from src.train import train_and_evaluate_all
from src.predict import predict_house_price


def run_full_pipeline():
    print("=" * 65)
    print("  REAL ESTATE VALUATION & AUTOMATED APPRAISAL ENGINE")
    print("             END-TO-END PIPELINE EXECUTION")
    print("=" * 65)

    # Step 1: Data Loading & Validation
    print("\n[STEP 1/5] Loading and Validating Raw Datasets...")
    train_df = load_train_data()
    test_df = load_test_data()
    train_val = validate_dataset(train_df, is_train=True)
    test_val = validate_dataset(test_df, is_train=False)
    print(f"  --> Train Set: {train_val['rows']} rows, {train_val['columns']} columns (Target 'SalePrice' Present: {train_val['has_sale_price']})")
    print(f"  --> Test Set:  {test_val['rows']} rows, {test_val['columns']} columns (Target Present: {test_val['has_sale_price']})")

    # Step 2: Exploratory Data Analysis & Plotting
    print("\n[STEP 2/5] Running Exploratory Data Analysis (EDA)...")
    run_eda()

    # Step 3: Model Training, Evaluation & Selection
    print("\n[STEP 3/5] Training Regression Models & Evaluating Metrics...")
    results_df, best_model = train_and_evaluate_all()

    # Step 4: Verification of Inference System
    print("\n[STEP 4/5] Testing Single Property Live Prediction...")
    sample_property = {
        "Neighborhood": "CollgCr",
        "OverallQual": 8,
        "GrLivArea": 2100,
        "TotalBsmtSF": 1100,
        "1stFlrSF": 1100,
        "2ndFlrSF": 1000,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "TotRmsAbvGrd": 8,
        "YearBuilt": 2008,
        "YearRemodAdd": 2009,
        "GarageCars": 2,
        "GarageArea": 600,
        "LotArea": 10500,
        "Fireplaces": 1,
    }
    est_price = predict_house_price(sample_property)
    print(f"  --> Test Property in '{sample_property['Neighborhood']}' (Quality: {sample_property['OverallQual']}/10, Area: {sample_property['GrLivArea']} sqft)")
    print(f"  --> Automated Model Appraisal: ${est_price:,.2f}")

    # Step 5: Summary
    print("\n[STEP 5/5] Pipeline Run Completed Successfully!")
    print("=" * 65)
    print("  Artifacts Generated:")
    print("  - Models:     models/house_price_model.pkl")
    print("  - Metadata:   models/model_metadata.json")
    print("  - Results:    outputs/results/model_comparison.csv")
    print("  - Plots:      outputs/plots/ (10 figures generated)")
    print("  - Web App:    app.py")
    print("=" * 65)
    print("\nTo launch the interactive web application, run:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    run_full_pipeline()
