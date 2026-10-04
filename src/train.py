"""
Model Training & Selection Pipeline for Indian Real Estate Valuation System.
Trains and compares Ridge Regression, Random Forest, Gradient Boosting, and XGBoost models
using 5-Fold Cross-Validation and holdout test evaluation on genuine Indian residential data.
Implements Conformal Prediction for calibrated uncertainty quantification.
Saves the best model artifact, comparison table, metadata, and evaluation plots.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from src.data_loader import load_clean_housing_data
from src.feature_engineering import IndianHouseFeatureEngineer, ALL_AMENITY_COLS
from src.preprocessing import get_feature_column_names, build_full_pipeline
from src.evaluate import calculate_metrics, plot_actual_vs_predicted, plot_residuals, format_inr


MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
PLOTS_DIR = OUTPUTS_DIR / "plots"


def get_candidate_models() -> Dict[str, Any]:
    """
    Define regression model candidates with tuned hyperparameters for Indian real estate data.
    """
    models = {
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150, max_depth=14, min_samples_split=6, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=175, learning_rate=0.08, max_depth=5, subsample=0.85, random_state=42
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=220, learning_rate=0.06, max_depth=6, subsample=0.85,
            colsample_bytree=0.85, random_state=42, n_jobs=-1
        )
    }
    return models


def train_and_evaluate_all():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("   INDIAN REAL ESTATE - MODEL TRAINING & SELECTION")
    print("==================================================")

    # 1. Load Data
    clean_df = load_clean_housing_data()
    X = clean_df.drop(columns=["Price", "Price_per_sqft", "Price_Lakhs"], errors="ignore")
    y = clean_df["Price"]

    # 2. Train-Validation-Test Split (70% Train, 15% Calibration, 15% Test)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=X["City"]
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, random_state=42, stratify=X_train_full["City"]
    )

    print(f"Dataset Split: {len(X_train)} Train (70%), {len(X_calib)} Calibration (15%), {len(X_test)} Test (15%)")
    print(f"Metropolitan Cities: {', '.join(clean_df['City'].unique())}")
    print(f"Distinct Micro-Markets: {clean_df['Location'].nunique()}")

    # 3. Discover Feature Columns on Training Data
    fe = IndianHouseFeatureEngineer()
    X_train_fe = fe.transform(X_train)
    num_cols, city_cols, loc_cols = get_feature_column_names(X_train_fe)
    print(f"Features: {len(num_cols)} Numerical, {len(city_cols)} City, {len(loc_cols)} Location")

    # 4. Train and Evaluate each Candidate Model
    candidate_models = get_candidate_models()
    results_list = []
    trained_models = {}
    test_predictions = {}
    calib_predictions = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, base_regressor in candidate_models.items():
        print(f"\nEvaluating [{name}] with 5-Fold Cross-Validation & Log-Target Transform...")
        pipeline = build_full_pipeline(base_regressor, num_cols, city_cols, loc_cols)

        model = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )

        # Cross validation on log target for stability
        cv_log_r2_scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_cv_tr, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_tr, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_cv_tr, y_cv_tr)
            y_cv_pred = model.predict(X_cv_val)
            cv_log_r2 = calculate_metrics(y_cv_val, y_cv_pred)["R2_log"]
            cv_log_r2_scores.append(cv_log_r2)

        mean_cv_r2 = float(np.mean(cv_log_r2_scores))

        # Fit on full training set
        model.fit(X_train, y_train)

        # Predictions on Holdout Test Set
        y_test_pred = model.predict(X_test)
        y_test_pred = np.clip(y_test_pred, a_min=500000.0, a_max=None)

        # Predictions on Calibration Set
        y_calib_pred = model.predict(X_calib)
        y_calib_pred = np.clip(y_calib_pred, a_min=500000.0, a_max=None)

        metrics = calculate_metrics(y_test, y_test_pred)
        metrics["Model"] = name
        metrics["CV_R2_Log"] = round(mean_cv_r2, 4)
        results_list.append(metrics)

        trained_models[name] = model
        test_predictions[name] = y_test_pred
        calib_predictions[name] = y_calib_pred

        print(f"  --> Test MAE: {format_inr(metrics['MAE'])} ({metrics['MAE_Lakhs']:.2f} L) | "
              f"RMSE: {format_inr(metrics['RMSE'])} ({metrics['RMSE_Lakhs']:.2f} L) | "
              f"R²: {metrics['R2']:.4f} | R²(Log): {metrics['R2_log']:.4f} | "
              f"CV R²(Log): {mean_cv_r2:.4f} | MAPE: {metrics['MAPE']:.2f}%")

    # 5. Compile Comparison Table
    results_df = pd.DataFrame(results_list)[
        ["Model", "MAE", "RMSE", "R2", "R2_log", "CV_R2_Log", "MAPE", "MAE_Lakhs", "RMSE_Lakhs"]
    ]
    results_df = results_df.sort_values(by="MAE", ascending=True).reset_index(drop=True)
    comparison_csv_path = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(comparison_csv_path, index=False)

    print("\n==================================================")
    print("         MODEL COMPARISON SUMMARY TABLE")
    print("==================================================")
    display_df = results_df[["Model", "MAE_Lakhs", "RMSE_Lakhs", "R2", "R2_log", "CV_R2_Log", "MAPE"]].copy()
    display_df.columns = ["Model", "MAE (₹ L)", "RMSE (₹ L)", "R² (Rupees)", "R² (Log)", "5-Fold CV R²", "MAPE (%)"]
    print(display_df.to_string(index=False))
    print(f"\nSaved comparison table to: {comparison_csv_path}")

    # 6. Select Best Model
    best_model_name = results_df.iloc[0]["Model"]
    best_mae = results_df.iloc[0]["MAE"]
    best_rmse = results_df.iloc[0]["RMSE"]
    best_r2 = results_df.iloc[0]["R2"]
    best_r2_log = results_df.iloc[0]["R2_log"]
    best_mape = results_df.iloc[0]["MAPE"]
    best_pipeline = trained_models[best_model_name]
    best_test_preds = test_predictions[best_model_name]
    best_calib_preds = calib_predictions[best_model_name]

    print("\n==================================================")
    print(f" BEST MODEL SELECTED: {best_model_name}")
    print(f" Test MAE:      {format_inr(best_mae)} ({best_mae/100000:.2f} Lakhs)")
    print(f" Test RMSE:     {format_inr(best_rmse)} ({best_rmse/100000:.2f} Lakhs)")
    print(f" Test R² Score: {best_r2:.4f} (Log R²: {best_r2_log:.4f})")
    print(f" Test MAPE:     {best_mape:.2f}%")
    print("==================================================")

    # 7. Conformal Prediction Calibration for Uncertainty Quantification
    # Non-conformity scores: |log(y) - log(y_hat)| on independent calibration set
    log_y_calib = np.log1p(y_calib.values)
    log_y_calib_pred = np.log1p(best_calib_preds)
    calib_residuals = np.abs(log_y_calib - log_y_calib_pred)

    # Compute conformal quantiles for 80% and 90% prediction intervals
    alpha_80 = 0.20
    q_level_80 = np.ceil((len(X_calib) + 1) * (1 - alpha_80)) / len(X_calib)
    conformal_q80 = float(np.quantile(calib_residuals, min(q_level_80, 1.0)))

    alpha_90 = 0.10
    q_level_90 = np.ceil((len(X_calib) + 1) * (1 - alpha_90)) / len(X_calib)
    conformal_q90 = float(np.quantile(calib_residuals, min(q_level_90, 1.0)))

    # Test empirical coverage on test set
    log_test_pred = np.log1p(best_test_preds)
    test_low_80 = np.expm1(log_test_pred - conformal_q80)
    test_high_80 = np.expm1(log_test_pred + conformal_q80)
    empirical_coverage_80 = float(np.mean((y_test.values >= test_low_80) & (y_test.values <= test_high_80)) * 100.0)

    print(f"\nConformal Uncertainty Calibration (Independently Calibrated on {len(X_calib)} properties):")
    print(f"  - 80% Prediction Interval Log-Width: ±{conformal_q80:.4f} (Test Empirical Coverage: {empirical_coverage_80:.2f}%)")
    print(f"  - 90% Prediction Interval Log-Width: ±{conformal_q90:.4f}")

    # 8. Generate Evaluation Visualizations
    plot_actual_vs_predicted(
        y_test.values, best_test_preds, best_model_name, PLOTS_DIR / "08_actual_vs_predicted.png"
    )
    plot_residuals(
        y_test.values, best_test_preds, best_model_name, PLOTS_DIR / "09_residuals_plot.png"
    )

    # Comparison Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.barplot(x="Model", y="MAE_Lakhs", hue="Model", data=results_df, palette="Greens_r", legend=False, ax=axes[0])
    axes[0].set_title("Test MAE (₹ in Lakhs) — Lower is Better", fontweight="bold")
    axes[0].set_ylabel("MAE (₹ in Lakhs)")
    axes[0].tick_params(axis="x", rotation=25)

    sns.barplot(x="Model", y="RMSE_Lakhs", hue="Model", data=results_df, palette="Blues_r", legend=False, ax=axes[1])
    axes[1].set_title("Test RMSE (₹ in Lakhs) — Lower is Better", fontweight="bold")
    axes[1].set_ylabel("RMSE (₹ in Lakhs)")
    axes[1].tick_params(axis="x", rotation=25)

    sns.barplot(x="Model", y="R2_log", hue="Model", data=results_df, palette="Purples_r", legend=False, ax=axes[2])
    axes[2].set_title("Log-Scale R² Score — Higher is Better", fontweight="bold")
    axes[2].set_ylabel("R² (log-scale)")
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    comparison_plot_path = PLOTS_DIR / "10_model_comparison_bar.png"
    fig.savefig(comparison_plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {comparison_plot_path.name}")

    # 9. Save Best Model Bundle
    model_bundle = {
        "model": best_pipeline,
        "model_name": best_model_name,
        "conformal_q80": conformal_q80,
        "conformal_q90": conformal_q90,
        "empirical_coverage_80": empirical_coverage_80,
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cities": clean_df["City"].unique().tolist(),
        "localities_count": clean_df["Location"].nunique(),
        "amenities": ALL_AMENITY_COLS
    }
    model_save_path = MODELS_DIR / "house_price_model.pkl"
    joblib.dump(model_bundle, model_save_path)
    print(f"\nSaved best model bundle with Conformal Calibrator to: {model_save_path}")

    # 10. Save Metadata JSON
    metadata = {
        "project": "BharatProp AI - Indian Real Estate Valuation & Automated Appraisal Engine",
        "best_model_name": best_model_name,
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": "Housing Prices in Metropolitan Areas of India (6 Major Cities)",
        "geographic_coverage": clean_df["City"].unique().tolist(),
        "total_properties_trained": len(X_train),
        "calibration_properties": len(X_calib),
        "test_properties": len(X_test),
        "distinct_localities": clean_df["Location"].nunique(),
        "test_mae_rupees": best_mae,
        "test_mae_lakhs": round(best_mae / 100000.0, 2),
        "test_rmse_rupees": best_rmse,
        "test_rmse_lakhs": round(best_rmse / 100000.0, 2),
        "test_r2_rupees": best_r2,
        "test_r2_log": best_r2_log,
        "test_mape_percent": best_mape,
        "conformal_interval_80_q": conformal_q80,
        "conformal_interval_90_q": conformal_q90,
        "conformal_empirical_coverage_80": empirical_coverage_80,
        "numerical_features": num_cols,
        "categorical_features": ["City", "Location"],
        "amenities_count": len(ALL_AMENITY_COLS)
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to: {metadata_path}")

    # 11. Verification Reload and Prediction
    print("\nVerifying saved model reload...")
    loaded_bundle = joblib.load(model_save_path)
    loaded_model = loaded_bundle["model"]
    test_sample = X_test.iloc[:3]
    sample_preds = loaded_model.predict(test_sample)

    print("Verification Predictions on 3 Validation Samples:")
    for idx in range(3):
        act = y_test.iloc[idx]
        pred = sample_preds[idx]
        loc = test_sample.iloc[idx]["Location"]
        city = test_sample.iloc[idx]["City"]
        bhk = test_sample.iloc[idx]["No. of Bedrooms"]
        area = test_sample.iloc[idx]["Area"]
        print(f"  Property {idx+1} ({city}, {loc} | {bhk} BHK, {area} sqft):")
        print(f"    Actual:    {format_inr(act)} ({act/100000:.2f} L)")
        print(f"    Predicted: {format_inr(pred)} ({pred/100000:.2f} L)")
        print(f"    Error:     {format_inr(abs(act - pred))} ({abs(act - pred)/100000:.2f} L)")

    print("\nTraining and evaluation pipeline completed successfully!")
    return results_df, best_model_name


if __name__ == "__main__":
    train_and_evaluate_all()
