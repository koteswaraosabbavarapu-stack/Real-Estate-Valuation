"""
Master Orchestration Script for Indian Real Estate Valuation & Automated Appraisal Engine.
Runs end-to-end:
1. Data Acquisition & Validation (6 Major Indian Metropolitan Cities)
2. Exploratory Data Analysis (EDA) & Plot Generation
3. Feature Engineering & ColumnTransformer Preprocessing Pipeline
4. Multi-Model 5-Fold Cross-Validation & Test Holdout Evaluation (Ridge, RF, GBR, XGBoost)
5. Conformal Prediction Uncertainty Calibration (80% & 90% Prediction Intervals)
6. Model Serialization & Live Indian Property Appraisal Verification
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import download_city_datasets_if_needed, load_clean_housing_data, validate_dataset
from src.eda_runner import run_eda
from src.train import train_and_evaluate_all
from src.predict import predict_property_valuation


def run_full_pipeline():
    print("=" * 70)
    print("      BHARATPROP AI — INDIAN REAL ESTATE VALUATION ENGINE")
    print("                 END-TO-END PIPELINE EXECUTION")
    print("=" * 70)

    # Step 1: Data Acquisition & Validation
    print("\n[STEP 1/5] Acquiring and Validating Indian Metropolitan Housing Dataset...")
    download_city_datasets_if_needed()
    clean_df = load_clean_housing_data()
    val_summary = validate_dataset(clean_df, is_train=True)
    print(f"  --> Validated Dataset: {val_summary['rows']:,} properties across {len(val_summary['cities'])} Metros")
    print(f"  --> Covered Cities:    {', '.join(val_summary['cities'])}")
    print(f"  --> Micro-Markets:     {val_summary['distinct_localities']:,} distinct localities")

    # Step 2: Exploratory Data Analysis & Plotting
    print("\n[STEP 2/5] Running Exploratory Data Analysis (EDA) & Generating Plots...")
    run_eda()

    # Step 3: Model Training, Cross-Validation & Conformal Calibration
    print("\n[STEP 3/5] Training Regression Models, Cross-Validating & Calibrating Intervals...")
    results_df, best_model = train_and_evaluate_all()

    # Step 4: Verification of Inference System
    print("\n[STEP 4/5] Testing Live Indian Property Automated Valuation...")
    sample_property = {
        "City": "Bangalore",
        "Location": "Whitefield",
        "Area": 1650,
        "No. of Bedrooms": 3,
        "Resale": 0,
        "Gymnasium": 1,
        "SwimmingPool": 1,
        "ClubHouse": 1,
        "24X7Security": 1,
        "PowerBackup": 1,
        "CarParking": 1,
        "LiftAvailable": 1,
        "AC": 1,
        "VaastuCompliant": 1
    }
    appraisal = predict_property_valuation(sample_property, coverage=0.80)
    print(f"  --> Test Property: {sample_property['City']} — {sample_property['Location']} ({sample_property['No. of Bedrooms']} BHK, {sample_property['Area']} sqft)")
    print(f"  --> Point Valuation:      {appraisal['formatted_price']} (₹{appraisal['price_in_lakhs']:.2f} Lakhs)")
    print(f"  --> Estimated Rate:       {appraisal['formatted_price_per_sqft']}")
    print(f"  --> 80% Prediction Range: {appraisal['formatted_range']}")

    # Step 5: Summary & Run Instructions
    print("\n[STEP 5/5] Complete Indian PropTech Pipeline Completed Successfully!")
    print("=" * 70)
    print("  Artifacts Generated:")
    print("  - Trained Model Bundle: models/house_price_model.pkl")
    print("  - System Metadata:      models/model_metadata.json")
    print("  - Comparison Leaderboard: outputs/results/model_comparison.csv")
    print("  - Visualization Suite:  outputs/plots/ (10 figures generated)")
    print("  - Interactive Web App:  app.py")
    print("=" * 70)
    print("\nTo launch the interactive Streamlit web application, run:")
    print("  streamlit run app.py\n")


if __name__ == "__main__":
    run_full_pipeline()
