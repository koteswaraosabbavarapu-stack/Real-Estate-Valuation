"""
Exploratory Data Analysis (EDA) Runner for Indian Real Estate.
Generates statistical insights and saves visualization plots to outputs/plots/
using genuine Indian housing market data across 6 metropolitan hubs.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_clean_housing_data

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


def run_eda():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean_housing_data()

    print("==================================================")
    print("   INDIAN REAL ESTATE - EXPLORATORY DATA ANALYSIS")
    print("==================================================")
    print(f"Dataset Shape: {df.shape[0]} properties across {df['City'].nunique()} metropolitan cities")
    print(f"Distinct Localities: {df['Location'].nunique()}")
    
    cities = df["City"].unique().tolist()
    print(f"Covered Cities: {', '.join(cities)}")
    print(f"Price (in Lakhs) — Mean: ₹{df['Price_Lakhs'].mean():.2f} L | Median: ₹{df['Price_Lakhs'].median():.2f} L | 95th Pct: ₹{df['Price_Lakhs'].quantile(0.95):.2f} L")
    print(f"Area (sq ft) — Mean: {df['Area'].mean():.1f} sq ft | Median: {df['Area'].median():.1f} sq ft")
    print(f"Rate (₹/sq ft) — Mean: ₹{df['Price_per_sqft'].mean():,.0f} | Median: ₹{df['Price_per_sqft'].median():,.0f}")

    # Plot 1: Price Distribution (in Lakhs and log1p)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Price_Lakhs"], kde=True, ax=axes[0], color="#1f77b4", bins=45)
    axes[0].set_title(f"Property Price Distribution (₹ in Lakhs)\nMedian: ₹{df['Price_Lakhs'].median():.1f} L | Skew: {df['Price_Lakhs'].skew():.2f}", fontweight="bold")
    axes[0].set_xlabel("Price (₹ in Lakhs)")
    axes[0].set_ylabel("Listing Count")
    axes[0].set_xlim(0, df["Price_Lakhs"].quantile(0.98))

    sns.histplot(np.log1p(df["Price"]), kde=True, ax=axes[1], color="#2ca02c", bins=45)
    axes[1].set_title(f"Log-Transformed log1p(Price in ₹)\nSkew: {np.log1p(df['Price']).skew():.2f}", fontweight="bold")
    axes[1].set_xlabel("log1p(Price in ₹)")
    axes[1].set_ylabel("Frequency")

    fig.tight_layout()
    plot1_path = PLOTS_DIR / "01_saleprice_distribution.png"
    fig.savefig(plot1_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot1_path.name}")

    # Plot 2: Price by City Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    city_order = df.groupby("City")["Price_Lakhs"].median().sort_values(ascending=False).index
    sns.boxplot(
        x="City",
        y="Price_Lakhs",
        data=df,
        order=city_order,
        palette="Blues_r",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "red", "markersize": 6},
        ax=ax
    )
    ax.set_title("Residential Property Price by Metropolitan City (₹ in Lakhs)", fontsize=13, fontweight="bold")
    ax.set_xlabel("City")
    ax.set_ylabel("Price (₹ in Lakhs)")
    ax.set_ylim(0, df["Price_Lakhs"].quantile(0.97))
    plot2_path = PLOTS_DIR / "02_saleprice_boxplot.png"
    fig.savefig(plot2_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot2_path.name}")

    # Plot 3: Price vs Area by City
    fig, ax = plt.subplots(figsize=(10, 6))
    sample_df = df.sample(min(4000, len(df)), random_state=42)
    sns.scatterplot(
        x="Area",
        y="Price_Lakhs",
        hue="City",
        alpha=0.6,
        s=35,
        data=sample_df,
        palette="tab10",
        ax=ax
    )
    ax.set_title("Built-Up Area (sq ft) vs Price (₹ in Lakhs) Across Indian Metros", fontsize=13, fontweight="bold")
    ax.set_xlabel("Built-Up Area (sq ft)")
    ax.set_ylabel("Price (₹ in Lakhs)")
    ax.set_ylim(0, sample_df["Price_Lakhs"].quantile(0.98))
    ax.legend(title="City", loc="upper left")
    plot3_path = PLOTS_DIR / "03_overallqual_vs_saleprice.png"
    fig.savefig(plot3_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot3_path.name}")

    # Plot 4: BHK vs Price
    fig, ax = plt.subplots(figsize=(9, 5))
    bhk_df = df[df["No. of Bedrooms"] <= 5]
    sns.boxplot(x="No. of Bedrooms", y="Price_Lakhs", hue="No. of Bedrooms", data=bhk_df, palette="viridis", legend=False, ax=ax)
    ax.set_title("Number of Bedrooms (BHK) vs Property Valuation", fontsize=13, fontweight="bold")
    ax.set_xlabel("Bedrooms (BHK)")
    ax.set_ylabel("Price (₹ in Lakhs)")
    ax.set_ylim(0, bhk_df["Price_Lakhs"].quantile(0.97))
    plot4_path = PLOTS_DIR / "04_grlivarea_vs_saleprice.png"
    fig.savefig(plot4_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot4_path.name}")

    # Plot 5: Average ₹/sq.ft Rate by City
    fig, ax = plt.subplots(figsize=(9, 5))
    rate_by_city = df.groupby("City")["Price_per_sqft"].median().sort_values(ascending=False).reset_index()
    sns.barplot(x="City", y="Price_per_sqft", hue="City", data=rate_by_city, palette="magma", legend=False, ax=ax)
    for p in ax.patches:
        ax.annotate(f"₹{p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax.set_title("Median Property Rate (₹ per sq.ft) by Metropolitan City", fontsize=13, fontweight="bold")
    ax.set_xlabel("City")
    ax.set_ylabel("Median Rate (₹ / sq.ft)")
    plot5_path = PLOTS_DIR / "05_yearbuilt_vs_saleprice.png"
    fig.savefig(plot5_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot5_path.name}")

    # Plot 6: Correlation Heatmap of Key Amenities & Features
    corr_cols = ["Price", "Area", "No. of Bedrooms", "Resale", "Gymnasium", "SwimmingPool", "24X7Security", "PowerBackup", "ClubHouse", "CarParking", "AC", "LiftAvailable"]
    available_corr_cols = [c for c in corr_cols if c in df.columns]
    corr_matrix = df[available_corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True,
                linewidths=0.5, annot_kws={"size": 9}, ax=ax)
    ax.set_title("Feature Correlation Matrix with Price", fontsize=13, fontweight="bold")
    plot6_path = PLOTS_DIR / "06_correlation_heatmap.png"
    fig.savefig(plot6_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot6_path.name}")

    # Plot 7: Top 15 Prime Localities across India by Median Price
    loc_summary = df.groupby(["Location", "City"]).agg(
        Median_Price=("Price_Lakhs", "median"),
        Count=("Price", "count")
    ).reset_index()
    loc_summary = loc_summary[loc_summary["Count"] >= 10].sort_values(by="Median_Price", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 6))
    loc_summary["Loc_City"] = loc_summary["Location"] + " (" + loc_summary["City"] + ")"
    sns.barplot(
        x="Median_Price",
        y="Loc_City",
        hue="City",
        data=loc_summary,
        palette="Set2",
        ax=ax
    )
    ax.set_title("Top 15 Prime Indian Micro-Markets by Median Valuation (min 10 listings)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Median Valuation (₹ in Lakhs)")
    ax.set_ylabel("Locality (City)")
    plot7_path = PLOTS_DIR / "07_missing_values_bar.png"
    fig.savefig(plot7_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot7_path.name}")

    print("\nEDA Completed successfully! All 7 plots generated and saved in outputs/plots/")


if __name__ == "__main__":
    run_eda()
