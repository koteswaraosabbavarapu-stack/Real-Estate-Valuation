"""
Model Evaluation Module for Real Estate Valuation System.
Calculates regression performance metrics (RMSE, MAE, R²) and generates evaluation plots.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> Dict[str, float]:
    """
    Calculate RMSE, MAE, and R² scores on actual dollar scale.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted values.

    Returns:
        Dict with RMSE, MAE, R2 scores.
    """
    y_t = np.array(y_true).flatten()
    y_p = np.array(y_pred).flatten()

    # Prevent negative values if any
    y_p = np.clip(y_p, a_min=10000.0, a_max=None)

    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4)
    }


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_path: Path):
    """
    Generate and save Actual vs Predicted scatter plot with ideal identity line.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color="#1f77b4", edgecolor="w", s=50, ax=ax)

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction (y = x)")

    metrics = calculate_metrics(y_true, y_pred)
    ax.set_title(f"Actual vs Predicted SalePrice - {model_name}\n"
                 f"RMSE: ${metrics['RMSE']:,.2f} | MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Actual SalePrice ($)", fontsize=11)
    ax.set_ylabel("Predicted SalePrice ($)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path.name}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_path: Path):
    """
    Generate and save Residuals vs Predicted plot.
    """
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color="#e65100", edgecolor="w", s=50, ax=ax)
    ax.axhline(0, color="black", linestyle="--", lw=1.5)

    ax.set_title(f"Residual Plot - {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted SalePrice ($)", fontsize=11)
    ax.set_ylabel("Residuals (Actual - Predicted) ($)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path.name}")
