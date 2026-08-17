"""
Real Estate Valuation & Automated Appraisal Engine
Premium Streamlit Web Application
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.predict import predict_house_price, get_default_property_template, load_valuation_model
from src.data_loader import load_train_data


# -------------------------------------------------------------
# PAGE CONFIGURATION & INLINE ASSETS
# -------------------------------------------------------------
st.set_page_config(
    page_title="PropValuate AI | Real Estate Valuation Engine",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Navbar Banner */
    .top-hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.8rem;
        color: #ffffff;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .hero-tagline {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* Custom Navigation Pills */
    .stRadio > div[role="radiogroup"] {
        display: flex;
        gap: 0.5rem;
        background: #f1f5f9;
        padding: 0.4rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .stRadio > div[role="radiogroup"] > label {
        flex: 1;
        text-align: center;
        background: transparent;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    /* Valuation Highlight Card */
    .appraisal-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white !important;
        text-align: center;
        box-shadow: 0 12px 28px -6px rgba(37, 99, 235, 0.35);
        margin-bottom: 1.5rem;
    }
    .appraisal-label {
        text-transform: uppercase;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        opacity: 0.9;
        color: #e0e7ff;
    }
    .appraisal-amount {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.4rem 0;
        letter-spacing: -1px;
        color: #ffffff;
    }
    .appraisal-range {
        font-size: 0.95rem;
        color: #dbeafe;
        font-weight: 500;
    }

    /* Quick Info Cards */
    .stat-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .stat-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.2rem;
    }

    /* Preset Buttons */
    .preset-pill {
        display: inline-block;
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #334155;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    /* Section Subheadings */
    .section-head {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_metadata():
    meta_path = PROJECT_ROOT / "models" / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_training_df():
    try:
        return load_train_data()
    except Exception:
        return pd.DataFrame()


metadata = get_metadata()
train_df = get_training_df()


# -------------------------------------------------------------
# PRESET PROPERTY PROFILES (For Fast Demo & Testing)
# -------------------------------------------------------------
PROPERTY_PRESETS = {
    "🏡 Suburban Family Home": {
        "Neighborhood": "CollgCr", "BldgType": "1Fam", "HouseStyle": "2Story", "MSZoning": "RL",
        "GrLivArea": 1850, "1stFlrSF": 950, "2ndFlrSF": 900, "TotalBsmtSF": 920,
        "LotArea": 9600, "LotFrontage": 75, "OverallQual": 7, "OverallCond": 5,
        "YearBuilt": 2004, "YearRemodAdd": 2005, "BedroomAbvGr": 3, "FullBath": 2, "HalfBath": 1,
        "TotRmsAbvGrd": 7, "GarageCars": 2, "GarageArea": 520, "Fireplaces": 1, "WoodDeckSF": 140
    },
    "🏰 Luxury Executive Estate": {
        "Neighborhood": "NridgHt", "BldgType": "1Fam", "HouseStyle": "2Story", "MSZoning": "RL",
        "GrLivArea": 3200, "1stFlrSF": 1650, "2ndFlrSF": 1550, "TotalBsmtSF": 1600,
        "LotArea": 15000, "LotFrontage": 105, "OverallQual": 9, "OverallCond": 5,
        "YearBuilt": 2010, "YearRemodAdd": 2012, "BedroomAbvGr": 4, "FullBath": 3, "HalfBath": 1,
        "TotRmsAbvGrd": 10, "GarageCars": 3, "GarageArea": 840, "Fireplaces": 2, "WoodDeckSF": 280
    },
    "🏘️ Urban Starter Townhouse": {
        "Neighborhood": "Somerst", "BldgType": "TwnhsE", "HouseStyle": "2Story", "MSZoning": "FV",
        "GrLivArea": 1350, "1stFlrSF": 680, "2ndFlrSF": 670, "TotalBsmtSF": 680,
        "LotArea": 3200, "LotFrontage": 40, "OverallQual": 7, "OverallCond": 5,
        "YearBuilt": 2008, "YearRemodAdd": 2008, "BedroomAbvGr": 2, "FullBath": 2, "HalfBath": 1,
        "TotRmsAbvGrd": 5, "GarageCars": 2, "GarageArea": 440, "Fireplaces": 0, "WoodDeckSF": 0
    },
    "🏚️ Historic Ames Bungalow": {
        "Neighborhood": "OldTown", "BldgType": "1Fam", "HouseStyle": "1.5Fin", "MSZoning": "RM",
        "GrLivArea": 1200, "1stFlrSF": 800, "2ndFlrSF": 400, "TotalBsmtSF": 750,
        "LotArea": 6500, "LotFrontage": 55, "OverallQual": 5, "OverallCond": 6,
        "YearBuilt": 1935, "YearRemodAdd": 1985, "BedroomAbvGr": 3, "FullBath": 1, "HalfBath": 0,
        "TotRmsAbvGrd": 6, "GarageCars": 1, "GarageArea": 260, "Fireplaces": 0, "WoodDeckSF": 40
    }
}


# -------------------------------------------------------------
# SIDEBAR ENHANCEMENTS
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏡 **PropValuate AI**")
    st.caption("Automated Real Estate Appraisal Engine")
    st.markdown("---")
    
    st.markdown("#### ⚡ Quick Presets")
    st.write("Load realistic sample homes instantly:")
    
    selected_preset = st.selectbox(
        "Choose Property Profile:",
        ["None (Custom Form)"] + list(PROPERTY_PRESETS.keys()),
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### 🎯 Active ML Model")
    st.markdown(f"**Model:** `{metadata.get('best_model_name', 'Ridge Regression')}`")
    st.markdown(f"**R² Explained Variance:** `{metadata.get('validation_r2', 0.9226):.4f}`")
    st.markdown(f"**Validation RMSE:** `±${metadata.get('validation_rmse', 24366.32):,.2f}`")
    st.markdown(f"**Validation MAE:** `±${metadata.get('validation_mae', 16023.18):,.2f}`")
    
    st.markdown("---")
    st.markdown("#### 📖 Project Quick Links")
    st.markdown("- [College Project Report](file:///docs/PROJECT_REPORT.md)")
    st.markdown("- [Viva Voce Q&A Notes](file:///docs/VIVA_NOTES.md)")
    st.markdown("- [Project Documentation](file:///README.md)")


# -------------------------------------------------------------
# TOP HERO BANNER & MODERN TAB NAVIGATION
# -------------------------------------------------------------
st.markdown("""
<div class="top-hero-banner">
    <div class="hero-title">🏡 Real Estate Valuation & Automated Appraisal Engine</div>
    <div class="hero-tagline">Advanced Regression System trained on the Ames Housing Dataset with 92.26% Explained Price Variance</div>
</div>
""", unsafe_allow_html=True)

# Primary Navigation Tabs
nav_tab1, nav_tab2, nav_tab3, nav_tab4 = st.tabs([
    "🏠 Property Appraisal",
    "📊 Model Performance & Benchmarks",
    "📈 Market Analytics & EDA",
    "📂 Batch Property Valuation"
])


# -------------------------------------------------------------
# TAB 1: PROPERTY APPRAISAL ENGINE
# -------------------------------------------------------------
with nav_tab1:
    # Determine default values based on preset selection
    preset_data = PROPERTY_PRESETS.get(selected_preset, {})

    st.markdown("<div class='section-head'>⚙️ Property Configuration & Specifications</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    neighborhood_options = [
        "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NWAmes", 
        "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes", "SawyerW", "IDOTRR", 
        "MeadowV", "Edwards", "Timber", "Gilbert", "StoneBr", "ClearCr", "NPkVill", 
        "Blmngtn", "BrDale", "SWISU", "Blueste"
    ]
    bldg_options = ["1Fam (Single-family Detached)", "2fmCon (Two-family)", "Duplex", "TwnhsE (Townhouse End)", "Twnhs (Townhouse Inside)"]
    style_options = ["1Story", "2Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Unf", "2.5Fin"]
    zoning_options = ["RL (Residential Low Density)", "RM (Residential Medium Density)", "FV (Floating Village)", "RH (Residential High)", "C (Commercial)"]

    with col_left:
        subtab_loc, subtab_dim, subtab_qual, subtab_amen = st.tabs([
            "📍 Location & Architecture", "📐 Floor Area & Lot", "⭐ Quality & Construction", "🚗 Rooms & Garage"
        ])

        with subtab_loc:
            c1, c2 = st.columns(2)
            with c1:
                cur_neigh = preset_data.get("Neighborhood", "CollgCr")
                neigh_idx = neighborhood_options.index(cur_neigh) if cur_neigh in neighborhood_options else 0
                neighborhood = st.selectbox("Neighborhood / Sector", sorted(neighborhood_options), index=neigh_idx)
                
                cur_bldg = preset_data.get("BldgType", "1Fam")
                b_idx = [i for i, opt in enumerate(bldg_options) if opt.startswith(cur_bldg)]
                bldg_type = st.selectbox("Building Type", bldg_options, index=b_idx[0] if b_idx else 0)
                bldg_val = bldg_type.split()[0]
            with c2:
                cur_style = preset_data.get("HouseStyle", "2Story")
                s_idx = style_options.index(cur_style) if cur_style in style_options else 1
                house_style = st.selectbox("House Style / Stories", style_options, index=s_idx)
                
                cur_zone = preset_data.get("MSZoning", "RL")
                z_idx = [i for i, opt in enumerate(zoning_options) if opt.startswith(cur_zone)]
                ms_zoning = st.selectbox("Zoning Classification", zoning_options, index=z_idx[0] if z_idx else 0)
                zoning_val = ms_zoning.split()[0]

        with subtab_dim:
            c1, c2 = st.columns(2)
            with c1:
                gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=6000, value=int(preset_data.get("GrLivArea", 1750)), step=25)
                first_flr_sf = st.number_input("1st Floor Area (sq ft)", min_value=300, max_value=4000, value=int(preset_data.get("1stFlrSF", 950)), step=25)
                second_flr_sf = st.number_input("2nd Floor Area (sq ft)", min_value=0, max_value=3000, value=int(preset_data.get("2ndFlrSF", 800)), step=25)
            with c2:
                total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=4000, value=int(preset_data.get("TotalBsmtSF", 900)), step=25)
                lot_area = st.number_input("Lot Parcel Area (sq ft)", min_value=1000, max_value=100000, value=int(preset_data.get("LotArea", 9500)), step=100)
                lot_frontage = st.number_input("Lot Frontage (linear ft)", min_value=20, max_value=300, value=int(preset_data.get("LotFrontage", 70)), step=5)

        with subtab_qual:
            c1, c2 = st.columns(2)
            with c1:
                overall_qual = st.slider("Overall Finish Quality (1-10)", min_value=1, max_value=10, value=int(preset_data.get("OverallQual", 7)), help="1: Very Poor, 5: Average, 10: Masterpiece")
                overall_cond = st.slider("Overall Maintenance Condition (1-9)", min_value=1, max_value=9, value=int(preset_data.get("OverallCond", 5)), help="1: Poor, 5: Normal, 9: Pristine")
            with c2:
                year_built = st.number_input("Year Built", min_value=1870, max_value=2024, value=int(preset_data.get("YearBuilt", 2005)), step=1)
                year_remod = st.number_input("Year Remodeled / Modernized", min_value=1950, max_value=2024, value=int(preset_data.get("YearRemodAdd", 2006)), step=1)

        with subtab_amen:
            c1, c2 = st.columns(2)
            with c1:
                bedrooms = st.number_input("Bedrooms Above Ground", min_value=0, max_value=8, value=int(preset_data.get("BedroomAbvGr", 3)), step=1)
                full_bath = st.number_input("Full Bathrooms", min_value=0, max_value=4, value=int(preset_data.get("FullBath", 2)), step=1)
                half_bath = st.number_input("Half Bathrooms", min_value=0, max_value=3, value=int(preset_data.get("HalfBath", 1)), step=1)
                tot_rms = st.number_input("Total Rooms Above Ground", min_value=2, max_value=15, value=int(preset_data.get("TotRmsAbvGrd", 7)), step=1)
            with c2:
                garage_cars = st.number_input("Garage Car Capacity", min_value=0, max_value=4, value=int(preset_data.get("GarageCars", 2)), step=1)
                garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=int(preset_data.get("GarageArea", 500)), step=25)
                fireplaces = st.number_input("Number of Fireplaces", min_value=0, max_value=4, value=int(preset_data.get("Fireplaces", 1)), step=1)
                wood_deck_sf = st.number_input("Wood Deck Area (sq ft)", min_value=0, max_value=1000, value=int(preset_data.get("WoodDeckSF", 120)), step=10)

        property_input = {
            "Neighborhood": neighborhood,
            "BldgType": bldg_val,
            "HouseStyle": house_style,
            "MSZoning": zoning_val,
            "GrLivArea": gr_liv_area,
            "1stFlrSF": first_flr_sf,
            "2ndFlrSF": second_flr_sf,
            "TotalBsmtSF": total_bsmt_sf,
            "LotArea": lot_area,
            "LotFrontage": lot_frontage,
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "BedroomAbvGr": bedrooms,
            "FullBath": full_bath,
            "HalfBath": half_bath,
            "TotRmsAbvGrd": tot_rms,
            "GarageCars": garage_cars,
            "GarageArea": garage_area,
            "Fireplaces": fireplaces,
            "WoodDeckSF": wood_deck_sf,
            "YrSold": 2010,
        }

        st.markdown("<br>", unsafe_allow_html=True)
        recompute = st.button("🚀 Calculate Real-Time Appraisal", type="primary", use_container_width=True)

    with col_right:
        st.markdown("<div class='section-head'>💵 Market Valuation Report</div>", unsafe_allow_html=True)
        
        # Live Prediction Calculation
        estimated_price = predict_house_price(property_input)
        rmse_val = metadata.get("validation_rmse", 24366.32)
        lower_bound = max(15000, estimated_price - rmse_val)
        upper_bound = estimated_price + rmse_val
        price_per_sqft = estimated_price / max(1, gr_liv_area)

        # Tier Classification
        if estimated_price > 350000:
            tier_badge = "👑 Luxury Tier"
        elif estimated_price > 200000:
            tier_badge = "🌟 Premium Upper-Mid Tier"
        elif estimated_price > 130000:
            tier_badge = "🏡 Standard Residential Tier"
        else:
            tier_badge = "🏷️ Affordable Entry Tier"

        st.markdown(f"""
        <div class='appraisal-card'>
            <div class='appraisal-label'>{tier_badge} • Automated Fair Market Value</div>
            <div class='appraisal-amount'>${estimated_price:,.2f}</div>
            <div class='appraisal-range'>Statistical Range: <b>${lower_bound:,.0f} — ${upper_bound:,.0f}</b> (±${rmse_val:,.0f} RMSE)</div>
        </div>
        """, unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-label'>Price / Sq Ft</div>
                <div class='stat-val'>${price_per_sqft:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-label'>Total Indoor SF</div>
                <div class='stat-val'>{gr_liv_area + total_bsmt_sf:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-label'>House Age</div>
                <div class='stat-val'>{max(0, 2026 - year_built)} yrs</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Appraisal Factsheet")
        st.markdown(f"""
        - **Neighborhood Cluster:** `{neighborhood}`
        - **Quality & Condition:** Quality `{overall_qual}/10` • Condition `{overall_cond}/9`
        - **Room Allocation:** `{bedrooms} Bedrooms` • `{full_bath + 0.5*half_bath} Baths` • `{tot_rms} Total Rooms`
        - **Garage & Parking:** `{garage_cars} Cars` ({garage_area} sq ft)
        - **Outdoor Living:** `{wood_deck_sf} sq ft` Deck Area
        """)


# -------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE & BENCHMARKS
# -------------------------------------------------------------
with nav_tab2:
    st.markdown("<div class='section-head'>📊 Model Evaluation & Benchmarks</div>", unsafe_allow_html=True)
    st.markdown("Comparison across all 4 machine learning regression models evaluated on the identical 20% holdout validation dataset:")

    csv_path = PROJECT_ROOT / "outputs" / "results" / "model_comparison.csv"
    if csv_path.exists():
        comp_df = pd.read_csv(csv_path)
        st.dataframe(
            comp_df.style.format({
                "RMSE": "${:,.2f}",
                "MAE": "${:,.2f}",
                "R2": "{:.4f}"
            }).highlight_min(subset=["RMSE", "MAE"], color="#dcfce7")
              .highlight_max(subset=["R2"], color="#dcfce7"),
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Visual Validation & Residuals")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p1 = PROJECT_ROOT / "outputs" / "plots" / "08_actual_vs_predicted.png"
        if p1.exists():
            st.image(str(p1), caption="Actual vs Predicted SalePrice (R² = 0.9226)", use_container_width=True)
    with col_p2:
        p2 = PROJECT_ROOT / "outputs" / "plots" / "09_residuals_plot.png"
        if p2.exists():
            st.image(str(p2), caption="Residuals Distribution (Homoscedastic Error)", use_container_width=True)

    p3 = PROJECT_ROOT / "outputs" / "plots" / "10_model_comparison_bar.png"
    if p3.exists():
        st.image(str(p3), caption="RMSE, MAE, and R² Metric Comparison Across All Algorithms", use_container_width=True)


# -------------------------------------------------------------
# TAB 3: MARKET EXPLORATORY ANALYTICS
# -------------------------------------------------------------
with nav_tab3:
    st.markdown("<div class='section-head'>📈 Real Estate Exploratory Data Analytics (EDA)</div>", unsafe_allow_html=True)
    st.markdown("Key valuation patterns, correlation coefficients, and skewness distributions identified during EDA:")

    c1, c2 = st.columns(2)
    with c1:
        p1 = PROJECT_ROOT / "outputs" / "plots" / "01_saleprice_distribution.png"
        if p1.exists():
            st.image(str(p1), caption="Target SalePrice Distribution: Skewness Corrected via log1p", use_container_width=True)
    with c2:
        p2 = PROJECT_ROOT / "outputs" / "plots" / "03_overallqual_vs_saleprice.png"
        if p2.exists():
            st.image(str(p2), caption="Strongest Valuation Driver: Overall Quality (r = 0.79)", use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        p3 = PROJECT_ROOT / "outputs" / "plots" / "04_grlivarea_vs_saleprice.png"
        if p3.exists():
            st.image(str(p3), caption="Living Area (GrLivArea) vs SalePrice (r = 0.71)", use_container_width=True)
    with c4:
        p4 = PROJECT_ROOT / "outputs" / "plots" / "06_correlation_heatmap.png"
        if p4.exists():
            st.image(str(p4), caption="Top 12 Correlated Features Heatmap", use_container_width=True)


# -------------------------------------------------------------
# TAB 4: BATCH PROPERTY VALUATION
# -------------------------------------------------------------
with nav_tab4:
    st.markdown("<div class='section-head'>📂 Bulk Property Valuation & Export</div>", unsafe_allow_html=True)
    st.markdown("Upload a CSV file with property records or appraise the sample test dataset in bulk:")

    uploaded_file = st.file_uploader("Upload CSV Spreadsheet", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(batch_df)} properties for automated appraisal.")
        
        if st.button("🚀 Run Batch Valuation on Uploaded File", type="primary"):
            with st.spinner("Calculating property valuations..."):
                preds = predict_house_price(batch_df)
                result_df = batch_df.copy()
                result_df["Estimated_SalePrice"] = preds
                result_df["Estimated_Price_Formatted"] = [f"${p:,.2f}" for p in preds]
                st.dataframe(result_df.head(25), use_container_width=True)

                csv_data = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Appraised Properties CSV",
                    data=csv_data,
                    file_name="bulk_property_valuations.csv",
                    mime="text/csv",
                )
    else:
        st.info("💡 You can also run an appraisal on pre-loaded sample properties from the dataset below:")
        sample_path = PROJECT_ROOT / "data" / "raw" / "test.csv"
        if sample_path.exists():
            sample_df = pd.read_csv(sample_path).head(10)
            st.dataframe(sample_df[["Id", "MSSubClass", "MSZoning", "LotArea", "Neighborhood", "OverallQual", "YearBuilt", "GrLivArea"]], use_container_width=True)
            if st.button("Appraise Sample Properties Above", type="primary"):
                sample_preds = predict_house_price(sample_df)
                sample_res = sample_df[["Id", "Neighborhood", "OverallQual", "YearBuilt", "GrLivArea"]].copy()
                sample_res["Estimated_SalePrice"] = [f"${p:,.2f}" for p in sample_preds]
                st.dataframe(sample_res, use_container_width=True)
