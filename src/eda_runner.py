"""
Exploratory Data Analysis (EDA) Runner
Generates statistical insights and saves visualization plots to outputs/plots/
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to avoid GUI deallocator warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
DATA_DIR = BASE_DIR / "data" / "raw"


def run_eda():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train.csv"
    df = pd.read_csv(train_path)

    print("==================================================")
    print("       EXPLORATORY DATA ANALYSIS (EDA)")
    print("==================================================")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"Numerical Features: {len(num_cols)} (including Id & SalePrice)")
    print(f"Categorical Features: {len(cat_cols)}")
    print(f"Duplicate Rows: {df.duplicated().sum()}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"Count": missing, "Percentage": missing_pct})
    missing_df = missing_df[missing_df["Count"] > 0].sort_values(by="Count", ascending=False)
    print(f"\nTop 10 Features with Missing Values:\n{missing_df.head(10)}")

    # Target variable statistics
    sp = df["SalePrice"]
    print("\n--- Target Variable (SalePrice) Statistics ---")
    print(f"Mean:        ${sp.mean():,.2f}")
    print(f"Median:      ${sp.median():,.2f}")
    print(f"Std Dev:     ${sp.std():,.2f}")
    print(f"Min:         ${sp.min():,.2f}")
    print(f"Max:         ${sp.max():,.2f}")
    print(f"Skewness:    {sp.skew():.4f}")
    print(f"Log1p Skew:  {np.log1p(sp).skew():.4f}")

    # Plot 1: SalePrice Distribution (Original & Log1p)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["SalePrice"], kde=True, ax=axes[0], color="#2b5c8f", bins=40)
    axes[0].set_title(f"Original SalePrice Distribution (Skew: {sp.skew():.2f})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("SalePrice ($)")
    axes[0].set_ylabel("Frequency")

    sns.histplot(np.log1p(df["SalePrice"]), kde=True, ax=axes[1], color="#2e8b57", bins=40)
    axes[1].set_title(f"Log-Transformed log1p(SalePrice) (Skew: {np.log1p(sp).skew():.2f})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("log1p(SalePrice)")
    axes[1].set_ylabel("Frequency")

    fig.tight_layout()
    plot1_path = PLOTS_DIR / "01_saleprice_distribution.png"
    fig.savefig(plot1_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot1_path.name}")

    # Plot 2: SalePrice Boxplot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df["SalePrice"], color="#5c6bc0", ax=ax, flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})
    ax.set_title("SalePrice Boxplot - Outlier Detection", fontsize=12, fontweight="bold")
    ax.set_xlabel("SalePrice ($)")
    plot2_path = PLOTS_DIR / "02_saleprice_boxplot.png"
    fig.savefig(plot2_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot2_path.name}")

    # Plot 3: Overall Quality vs SalePrice
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="OverallQual", y="SalePrice", hue="OverallQual", data=df, palette="viridis", legend=False, ax=ax)
    ax.set_title("Overall Quality (1-10) vs SalePrice", fontsize=13, fontweight="bold")
    ax.set_xlabel("Overall Quality Rating")
    ax.set_ylabel("SalePrice ($)")
    plot3_path = PLOTS_DIR / "03_overallqual_vs_saleprice.png"
    fig.savefig(plot3_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot3_path.name}")

    # Plot 4: GrLivArea vs SalePrice
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.regplot(x="GrLivArea", y="SalePrice", data=df,
                scatter_kws={"alpha": 0.5, "color": "#1f77b4"},
                line_kws={"color": "#d62728", "linewidth": 2}, ax=ax)
    ax.set_title("Above Ground Living Area (GrLivArea) vs SalePrice", fontsize=13, fontweight="bold")
    ax.set_xlabel("Above Ground Living Area (sq ft)")
    ax.set_ylabel("SalePrice ($)")
    plot4_path = PLOTS_DIR / "04_grlivarea_vs_saleprice.png"
    fig.savefig(plot4_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot4_path.name}")

    # Plot 5: YearBuilt vs SalePrice
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="YearBuilt", y="SalePrice", data=df, hue="OverallQual", palette="magma", alpha=0.7, ax=ax)
    ax.set_title("Year Built vs SalePrice (Colored by Overall Quality)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year Built")
    ax.set_ylabel("SalePrice ($)")
    plot5_path = PLOTS_DIR / "05_yearbuilt_vs_saleprice.png"
    fig.savefig(plot5_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot5_path.name}")

    # Plot 6: Top Correlations with SalePrice Heatmap
    numeric_df = df[num_cols].drop(columns=["Id"], errors="ignore")
    corr_matrix = numeric_df.corr()
    top_corr_features = corr_matrix["SalePrice"].abs().sort_values(ascending=False).head(12).index
    top_corr = numeric_df[top_corr_features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(top_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True,
                linewidths=0.5, annot_kws={"size": 9}, ax=ax)
    ax.set_title("Top 12 Correlated Features with SalePrice", fontsize=13, fontweight="bold")
    plot6_path = PLOTS_DIR / "06_correlation_heatmap.png"
    fig.savefig(plot6_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot6_path.name}")

    # Plot 7: Missing Values Percentage
    if len(missing_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=missing_df["Percentage"].head(15), y=missing_df.head(15).index, hue=missing_df.head(15).index, palette="mako", legend=False, ax=ax)
        ax.set_title("Top 15 Features with Missing Values (%)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Missing Data Percentage (%)")
        ax.set_ylabel("Feature Name")
        plot7_path = PLOTS_DIR / "07_missing_values_bar.png"
        fig.savefig(plot7_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {plot7_path.name}")

    print("\nEDA Completed successfully! All plots saved in outputs/plots/")


if __name__ == "__main__":
    run_eda()
