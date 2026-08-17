"""
Model Training & Selection Pipeline for Real Estate Valuation System.
Trains Linear Regression, Random Forest, Gradient Boosting, and XGBoost models,
evaluates performance, selects the best model, and saves the final artifact.
"""

import sys
import json
from pathlib import Path

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
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from src.data_loader import load_train_data
from src.feature_engineering import HouseFeatureEngineer
from src.preprocessing import DatasetCategoricalCleaner, get_feature_column_names, build_full_pipeline
from src.evaluate import calculate_metrics, plot_actual_vs_predicted, plot_residuals


MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
PLOTS_DIR = OUTPUTS_DIR / "plots"


def get_models_dict():
    """
    Define regression model candidates with tuned hyperparameters.
    """
    models = {
        "Linear Regression (Ridge)": Ridge(alpha=10.0, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=4, subsample=0.85, random_state=42
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=4, subsample=0.85, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
    }
    return models


def train_and_evaluate_all():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("      REAL ESTATE VALUATION - MODEL TRAINING")
    print("==================================================")

    # 1. Load Data
    raw_df = load_train_data()
    X = raw_df.drop(columns=["SalePrice"])
    y = raw_df["SalePrice"]

    # 2. Train-Validation Split (80% Train, 20% Validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Dataset Split: {len(X_train)} Train samples (80%), {len(X_val)} Validation samples (20%)")

    # 3. Discover Feature Columns on Training Data
    cleaner = DatasetCategoricalCleaner()
    fe = HouseFeatureEngineer()
    X_train_transformed = fe.transform(cleaner.transform(X_train))
    num_cols, cat_cols = get_feature_column_names(X_train_transformed)
    print(f"Features: {len(num_cols)} Numerical, {len(cat_cols)} Categorical")

    # 4. Train and Evaluate each Candidate Model
    candidate_models = get_models_dict()
    results_list = []
    trained_pipelines = {}
    val_predictions = {}

    for name, base_model in candidate_models.items():
        print(f"\nTraining [{name}]...")
        # Build full pipeline with Preprocessing -> Feature Engineering -> Regressor
        pipeline = build_full_pipeline(base_model, num_cols, cat_cols)

        # Wrap in TransformedTargetRegressor for log1p transformation of target SalePrice
        model = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics = calculate_metrics(y_val, y_pred)
        metrics["Model"] = name
        results_list.append(metrics)
        trained_pipelines[name] = model
        val_predictions[name] = y_pred

        print(f"  --> RMSE: ${metrics['RMSE']:,.2f} | MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f}")

    # 5. Compile Comparison Table
    results_df = pd.DataFrame(results_list)[["Model", "RMSE", "MAE", "R2"]]
    results_df = results_df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    comparison_csv_path = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(comparison_csv_path, index=False)
    print("\n==================================================")
    print("             MODEL COMPARISON RESULTS")
    print("==================================================")
    print(results_df.to_string(index=False))
    print(f"\nSaved comparison results to: {comparison_csv_path}")

    # 6. Select Best Model
    best_model_name = results_df.iloc[0]["Model"]
    best_rmse = results_df.iloc[0]["RMSE"]
    best_mae = results_df.iloc[0]["MAE"]
    best_r2 = results_df.iloc[0]["R2"]
    best_pipeline = trained_pipelines[best_model_name]
    best_preds = val_predictions[best_model_name]

    print("\n==================================================")
    print(f" BEST MODEL SELECTED: {best_model_name}")
    print(f" Validation RMSE: ${best_rmse:,.2f}")
    print(f" Validation MAE:  ${best_mae:,.2f}")
    print(f" Validation R²:   {best_r2:.4f}")
    print("==================================================")

    # 7. Generate Evaluation Visualizations
    plot_actual_vs_predicted(
        y_val.values, best_preds, best_model_name, PLOTS_DIR / "08_actual_vs_predicted.png"
    )
    plot_residuals(
        y_val.values, best_preds, best_model_name, PLOTS_DIR / "09_residuals_plot.png"
    )

    # Comparison Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.barplot(x="Model", y="RMSE", hue="Model", data=results_df, palette="Blues_r", legend=False, ax=axes[0])
    axes[0].set_title("Validation RMSE ($) - Lower is Better", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=35)

    sns.barplot(x="Model", y="MAE", hue="Model", data=results_df, palette="Greens_r", legend=False, ax=axes[1])
    axes[1].set_title("Validation MAE ($) - Lower is Better", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=35)

    sns.barplot(x="Model", y="R2", hue="Model", data=results_df, palette="Purples_r", legend=False, ax=axes[2])
    axes[2].set_title("Validation R² Score - Higher is Better", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    comparison_plot_path = PLOTS_DIR / "10_model_comparison_bar.png"
    fig.savefig(comparison_plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {comparison_plot_path.name}")

    # 8. Save Best Model Pipeline
    model_save_path = MODELS_DIR / "house_price_model.pkl"
    joblib.dump(best_pipeline, model_save_path)
    print(f"\nSaved best model pipeline to: {model_save_path}")

    # 9. Save Metadata JSON
    metadata = {
        "best_model_name": best_model_name,
        "validation_rmse": best_rmse,
        "validation_mae": best_mae,
        "validation_r2": best_r2,
        "num_features": len(num_cols) + len(cat_cols),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "target": "SalePrice",
        "train_size": len(X_train),
        "val_size": len(X_val),
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to: {metadata_path}")

    # 10. Verify reloading saved model
    print("\nVerifying saved model reload...")
    loaded_model = joblib.load(model_save_path)
    test_sample = X_val.iloc[:3]
    sample_preds = loaded_model.predict(test_sample)
    print("Verification Predictions on 3 Validation Samples:")
    for idx, (act, pred) in enumerate(zip(y_val.iloc[:3], sample_preds)):
        print(f"  Property {idx+1}: Actual ${act:,.2f} | Predicted ${pred:,.2f} (Error: ${abs(act - pred):,.2f})")

    print("\nTraining and evaluation pipeline completed successfully!")
    return results_df, best_model_name


if __name__ == "__main__":
    train_and_evaluate_all()
