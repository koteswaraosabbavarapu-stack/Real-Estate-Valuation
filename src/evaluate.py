"""
Model Evaluation Module for Indian Real Estate Valuation System.
Calculates regression performance metrics (RMSE, MAE, R², MAPE) in Indian Rupees (₹)
and generates evaluation plots.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Calculate RMSE, MAE, R², and MAPE scores on the actual Indian Rupee (₹) scale
    as well as log-scale R².

    Args:
        y_true: Ground truth target prices in INR.
        y_pred: Predicted target prices in INR.

    Returns:
        Dict with RMSE, MAE, R2, R2_log, MAPE scores.
    """
    y_t = np.array(y_true, dtype=float).flatten()
    y_p = np.array(y_pred, dtype=float).flatten()

    # Clip predictions to sensible positive lower bound (₹5 Lakhs)
    y_p = np.clip(y_p, a_min=500000.0, a_max=None)

    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    mape = float(np.mean(np.abs((y_t - y_p) / y_t)) * 100.0)
    
    # Log-scale R² (measures proportional accuracy across wide price tiers)
    r2_log = float(r2_score(np.log1p(y_t), np.log1p(y_p)))

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4),
        "R2_log": round(r2_log, 4),
        "MAPE": round(mape, 2),
        "MAE_Lakhs": round(mae / 100000.0, 2),
        "RMSE_Lakhs": round(rmse / 100000.0, 2)
    }


def format_inr(amount: float) -> str:
    """Format numerical amount into standard Indian currency string (Lakhs / Crores)."""
    if amount >= 10000000:
        return f"₹{amount / 10000000:.2f} Cr"
    elif amount >= 100000:
        return f"₹{amount / 100000:.2f} L"
    else:
        return f"₹{amount:,.0f}"


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path
):
    """
    Generate and save Actual vs Predicted scatter plot in Lakhs with ideal identity line (y = x).
    """
    y_t_lakhs = np.array(y_true) / 100000.0
    y_p_lakhs = np.array(y_pred) / 100000.0

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=y_t_lakhs,
        y=y_p_lakhs,
        alpha=0.4,
        color="#1f77b4",
        edgecolor="w",
        s=40,
        ax=ax
    )

    max_val = min(max(y_t_lakhs.max(), y_p_lakhs.max()), 1000.0) # Up to ₹10 Crore for clear visibility
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect Prediction (y = x)")

    metrics = calculate_metrics(y_true, y_pred)
    ax.set_title(
        f"Actual vs Predicted Valuation — {model_name}\n"
        f"MAE: {format_inr(metrics['MAE'])} | RMSE: {format_inr(metrics['RMSE'])} | R² (log): {metrics['R2_log']:.4f}",
        fontsize=11,
        fontweight="bold"
    )
    ax.set_xlabel("Actual Price (₹ in Lakhs)", fontsize=11)
    ax.set_ylabel("Predicted Price (₹ in Lakhs)", fontsize=11)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path.name}")


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path
):
    """
    Generate and save Residuals vs Predicted plot in Lakhs.
    """
    y_t_lakhs = np.array(y_true) / 100000.0
    y_p_lakhs = np.array(y_pred) / 100000.0
    residuals = y_t_lakhs - y_p_lakhs

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        x=y_p_lakhs,
        y=residuals,
        alpha=0.4,
        color="#e65100",
        edgecolor="w",
        s=40,
        ax=ax
    )
    ax.axhline(0, color="black", linestyle="--", lw=1.5)

    ax.set_title(f"Residual Plot (Actual - Predicted) — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Price (₹ in Lakhs)", fontsize=11)
    ax.set_ylabel("Residual Error (₹ in Lakhs)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path.name}")
