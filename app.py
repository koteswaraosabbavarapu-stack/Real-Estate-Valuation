"""
Real Estate Valuation & Automated Appraisal Engine
Streamlit Web Application
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.predict import predict_house_price, get_default_property_template, load_valuation_model
from src.data_loader import load_train_data


# Page Configuration
st.set_page_config(
    page_title="Real Estate Valuation Engine",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    /* Main Layout Styling */
    .main-header {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4a5568;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .price-display {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        color: #ffffff !important;
        border-radius: 12px;
        padding: 1.8rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .price-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
        letter-spacing: -0.5px;
    }
    .badge {
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
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


# Sidebar Information
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.markdown("### 🏡 Real Estate Engine")
    st.markdown("Automated Appraisal System powered by supervised Machine Learning regression on the Ames Housing Dataset.")
    st.markdown("---")
    
    st.markdown("#### 🏆 Best Model Metrics")
    st.markdown(f"**Algorithm:** {metadata.get('best_model_name', 'Linear Regression (Ridge)')}")
    st.markdown(f"**Validation R²:** `{metadata.get('validation_r2', 0.9226):.4f}`")
    st.markdown(f"**Validation RMSE:** `${metadata.get('validation_rmse', 24366.32):,.2f}`")
    st.markdown(f"**Validation MAE:** `${metadata.get('validation_mae', 16023.18):,.2f}`")
    st.markdown("---")
    
    st.markdown("#### 📌 Navigation")
    app_mode = st.radio(
        "Choose Mode:",
        ["🏠 Property Appraisal", "📊 Model Performance", "📈 Market Exploratory Analytics", "📂 Batch Valuation"],
        index=0
    )


# -------------------------------------------------------------
# MODE 1: SINGLE PROPERTY APPRAISAL
# -------------------------------------------------------------
if app_mode == "🏠 Property Appraisal":
    st.markdown("<h1 class='main-header'>🏡 Real Estate Valuation & Automated Appraisal Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Configure property characteristics below to generate an instantaneous, data-driven market price appraisal.</p>", unsafe_allow_html=True)

    # Input Form Layout
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("### 📋 Property Specifications")
        
        tab_loc, tab_size, tab_qual, tab_ext = st.tabs([
            "📍 Location & Type", "📐 Area & Dimensions", "⭐ Quality & Age", "🚗 Amenities & Garage"
        ])

        neighborhood_options = [
            "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NWAmes", 
            "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes", "SawyerW", "IDOTRR", 
            "MeadowV", "Edwards", "Timber", "Gilbert", "StoneBr", "ClearCr", "NPkVill", 
            "Blmngtn", "BrDale", "SWISU", "Blueste"
        ]

        with tab_loc:
            c1, c2 = st.columns(2)
            with c1:
                neighborhood = st.selectbox("Neighborhood / Location", sorted(neighborhood_options), index=neighborhood_options.index("CollgCr"))
                bldg_type = st.selectbox("Building Type", ["1Fam (Single-family Detached)", "2fmCon (Two-family Conversion)", "Duplex", "TwnhsE (Townhouse End)", "Twnhs (Townhouse Inside)"], index=0)
                bldg_val = bldg_type.split()[0]
            with c2:
                house_style = st.selectbox("House Style / Architecture", ["1Story", "2Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Unf", "2.5Fin"], index=1)
                ms_zoning = st.selectbox("Zoning Classification", ["RL (Residential Low Density)", "RM (Residential Medium Density)", "FV (Floating Village)", "RH (Residential High Density)", "C (Commercial)"], index=0)
                zoning_val = ms_zoning.split()[0]

        with tab_size:
            c1, c2 = st.columns(2)
            with c1:
                gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=6000, value=1750, step=25)
                first_flr_sf = st.number_input("1st Floor Area (sq ft)", min_value=300, max_value=4000, value=950, step=25)
                second_flr_sf = st.number_input("2nd Floor Area (sq ft)", min_value=0, max_value=3000, value=800, step=25)
            with c2:
                total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=4000, value=900, step=25)
                lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=100000, value=9500, step=100)
                lot_frontage = st.number_input("Lot Frontage (Linear feet)", min_value=20, max_value=300, value=70, step=5)

        with tab_qual:
            c1, c2 = st.columns(2)
            with c1:
                overall_qual = st.slider("Overall Material & Finish Quality (1 - 10)", min_value=1, max_value=10, value=7, help="1: Very Poor, 5: Average, 10: Very Excellent")
                overall_cond = st.slider("Overall Condition (1 - 9)", min_value=1, max_value=9, value=5, help="1: Very Poor, 5: Average, 9: Excellent")
            with c2:
                year_built = st.number_input("Year Built", min_value=1870, max_value=2024, value=2005, step=1)
                year_remod = st.number_input("Year Remodeled / Additions", min_value=1950, max_value=2024, value=2006, step=1)

        with tab_ext:
            c1, c2 = st.columns(2)
            with c1:
                bedrooms = st.number_input("Bedrooms Above Ground", min_value=0, max_value=8, value=3, step=1)
                full_bath = st.number_input("Full Bathrooms", min_value=0, max_value=4, value=2, step=1)
                half_bath = st.number_input("Half Bathrooms", min_value=0, max_value=3, value=1, step=1)
                tot_rms = st.number_input("Total Rooms Above Ground", min_value=2, max_value=15, value=7, step=1)
            with c2:
                garage_cars = st.number_input("Garage Car Capacity", min_value=0, max_value=4, value=2, step=1)
                garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=500, step=25)
                fireplaces = st.number_input("Number of Fireplaces", min_value=0, max_value=4, value=1, step=1)
                wood_deck_sf = st.number_input("Wood Deck Area (sq ft)", min_value=0, max_value=1000, value=120, step=10)

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

        appraise_btn = st.button("🚀 Calculate Estimated Valuation", type="primary", use_container_width=True)

    with col_right:
        st.markdown("### 💵 Appraisal Result")
        
        # Calculate prediction
        estimated_price = predict_house_price(property_input)
        rmse_val = metadata.get("validation_rmse", 24366.32)
        lower_bound = max(15000, estimated_price - rmse_val)
        upper_bound = estimated_price + rmse_val
        price_per_sqft = estimated_price / max(1, gr_liv_area)

        st.markdown(f"""
        <div class='price-display'>
            <div style='text-transform: uppercase; font-size: 0.85rem; letter-spacing: 1px;'>Estimated Fair Market Value</div>
            <div class='price-value'>${estimated_price:,.2f}</div>
            <div style='opacity: 0.9; font-size: 0.95rem;'>Expected Range: <b>${lower_bound:,.0f} - ${upper_bound:,.0f}</b></div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Price / Sq Ft", f"${price_per_sqft:.2f}/sqft")
        with m2:
            st.metric("Total Sq Footage", f"{gr_liv_area + total_bsmt_sf:,.0f} sq ft")

        st.markdown("#### 🔍 Property Valuation Breakdown")
        st.markdown(f"""
        - **Neighborhood Base:** `{neighborhood}`
        - **Quality Index:** `{overall_qual}/10` (Condition: `{overall_cond}/9`)
        - **Effective Age:** `{2026 - year_built} years` (Remodel: `{2026 - year_remod} years ago`)
        - **Bathrooms:** `{full_bath + 0.5*half_bath} Baths` | **Garage:** `{garage_cars} Car Capacity`
        - **Model Confidence:** `±${rmse_val:,.2f}` RMSE on hold-out validation
        """)


# -------------------------------------------------------------
# MODE 2: MODEL PERFORMANCE & ARCHITECTURE
# -------------------------------------------------------------
elif app_mode == "📊 Model Performance":
    st.markdown("<h1 class='main-header'>📊 Machine Learning Model Benchmarks</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Detailed evaluation metrics and comparison across trained regression algorithms.</p>", unsafe_allow_html=True)

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

    st.markdown("### 📈 Evaluation Plots")
    c1, c2 = st.columns(2)
    with c1:
        p1 = PROJECT_ROOT / "outputs" / "plots" / "08_actual_vs_predicted.png"
        if p1.exists():
            st.image(str(p1), caption="Actual vs Predicted SalePrice (Validation Set)", use_container_width=True)
    with c2:
        p2 = PROJECT_ROOT / "outputs" / "plots" / "09_residuals_plot.png"
        if p2.exists():
            st.image(str(p2), caption="Residuals vs Predicted Distribution", use_container_width=True)

    p3 = PROJECT_ROOT / "outputs" / "plots" / "10_model_comparison_bar.png"
    if p3.exists():
        st.image(str(p3), caption="Model Performance Metric Comparison", use_container_width=True)


# -------------------------------------------------------------
# MODE 3: MARKET EXPLORATORY ANALYTICS
# -------------------------------------------------------------
elif app_mode == "📈 Market Exploratory Analytics":
    st.markdown("<h1 class='main-header'>📈 Real Estate Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Exploratory patterns and price dynamics discovered in the Ames Housing dataset.</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        p1 = PROJECT_ROOT / "outputs" / "plots" / "01_saleprice_distribution.png"
        if p1.exists():
            st.image(str(p1), caption="Target Variable Distribution & Log Transformation", use_container_width=True)
    with c2:
        p2 = PROJECT_ROOT / "outputs" / "plots" / "03_overallqual_vs_saleprice.png"
        if p2.exists():
            st.image(str(p2), caption="Overall Quality Rating vs SalePrice", use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        p3 = PROJECT_ROOT / "outputs" / "plots" / "04_grlivarea_vs_saleprice.png"
        if p3.exists():
            st.image(str(p3), caption="Living Area (GrLivArea) vs SalePrice", use_container_width=True)
    with c4:
        p4 = PROJECT_ROOT / "outputs" / "plots" / "06_correlation_heatmap.png"
        if p4.exists():
            st.image(str(p4), caption="Top Correlation Matrix with SalePrice", use_container_width=True)


# -------------------------------------------------------------
# MODE 4: BATCH PROPERTY VALUATION
# -------------------------------------------------------------
elif app_mode == "📂 Batch Valuation":
    st.markdown("<h1 class='main-header'>📂 Batch Property Valuation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Upload a CSV containing multiple properties to generate instant automated appraisals for each property.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Properties CSV", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(batch_df)}** properties for appraisal.")
        
        if st.button("🚀 Run Batch Valuation", type="primary"):
            with st.spinner("Generating valuations..."):
                preds = predict_house_price(batch_df)
                result_df = batch_df.copy()
                result_df["Estimated_SalePrice"] = preds
                st.success("Valuation complete!")
                st.dataframe(result_df.head(20), use_container_width=True)

                csv_data = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Valuations CSV",
                    data=csv_data,
                    file_name="property_valuations.csv",
                    mime="text/csv",
                )
    else:
        st.info("💡 You can upload a test dataset CSV or use sample records from the dataset.")
        sample_path = PROJECT_ROOT / "data" / "raw" / "test.csv"
        if sample_path.exists():
            st.markdown("#### Sample Test Records Available:")
            sample_df = pd.read_csv(sample_path).head(5)
            st.dataframe(sample_df)
            if st.button("Appraise First 5 Sample Test Properties"):
                sample_preds = predict_house_price(sample_df)
                sample_res = sample_df[["Id", "MSSubClass", "MSZoning", "LotArea", "Neighborhood", "OverallQual", "YearBuilt"]].copy()
                sample_res["Estimated_SalePrice"] = [f"${p:,.2f}" for p in sample_preds]
                st.dataframe(sample_res, use_container_width=True)
