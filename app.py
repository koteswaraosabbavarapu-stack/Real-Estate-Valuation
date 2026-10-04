"""
BharatProp AI — Indian Real Estate Valuation & Automated Appraisal Engine
State-of-the-Art Machine Learning Platform for Residential Property Appraisal,
Market Analytics & Financial Planning across 6 Major Indian Metropolitan Cities.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import io
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to sys.path
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

from src.predict import (
    predict_property_valuation,
    get_default_property_template,
    get_available_cities_and_localities,
    load_valuation_model_bundle,
    MODEL_PATH,
    METADATA_PATH
)
from src.data_loader import load_clean_housing_data, download_city_datasets_if_needed
from src.feature_engineering import ALL_AMENITY_COLS
from src.evaluate import format_inr


# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="BharatProp AI | Indian Real Estate Valuation Engine",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set visual plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


# -------------------------------------------------------------
# HIGH-END MODERN INDIAN PROPTECH UI STYLING
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hero Header Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #064e3b 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: #ffffff;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-badge {
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid rgba(110, 231, 183, 0.3);
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        display: inline-block;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        font-weight: 400;
        max-width: 900px;
        line-height: 1.5;
        margin-bottom: 0;
    }
    
    /* Valuation Output Card */
    .valuation-card {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 60%, #047857 100%);
        border-radius: 16px;
        padding: 2.2rem;
        color: white;
        box-shadow: 0 12px 28px -5px rgba(6, 78, 59, 0.4);
        margin: 1.2rem 0;
        border: 1px solid rgba(110, 231, 183, 0.3);
    }
    .val-price {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.3rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .val-range {
        font-size: 1.25rem;
        font-weight: 600;
        color: #a7f3d0;
        margin-top: 0.4rem;
    }
    .val-subtext {
        font-size: 0.95rem;
        color: #d1fae5;
        opacity: 0.9;
    }
    
    /* Metric Card */
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Info Box */
    .info-callout {
        background-color: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #065f46;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# REALISTIC INDIAN PROPERTY PRESETS FOR INSTANT TESTING
# -------------------------------------------------------------
INDIAN_PROPERTY_PRESETS = {
    "Select a pre-configured Indian property preset...": None,
    "🏢 3 BHK Luxury High-Rise — Whitefield (Bengaluru)": {
        "City": "Bangalore",
        "Location": "Whitefield",
        "Area": 1650,
        "No. of Bedrooms": 3,
        "Resale": 0,
        "Gymnasium": 1, "SwimmingPool": 1, "ClubHouse": 1, "24X7Security": 1,
        "PowerBackup": 1, "CarParking": 1, "LiftAvailable": 1, "AC": 1,
        "Gasconnection": 1, "VaastuCompliant": 1, "Children'splayarea": 1
    },
    "🌊 2 BHK Waterfront Residence — Bandra West (Mumbai)": {
        "City": "Mumbai",
        "Location": "Bandra West",
        "Area": 950,
        "No. of Bedrooms": 2,
        "Resale": 1,
        "Gymnasium": 1, "SwimmingPool": 0, "ClubHouse": 0, "24X7Security": 1,
        "PowerBackup": 1, "CarParking": 1, "LiftAvailable": 1, "AC": 1,
        "Gasconnection": 1, "VaastuCompliant": 0, "Children'splayarea": 0
    },
    "🏡 4 BHK Premium Villa / Floor — Saket (Delhi-NCR)": {
        "City": "Delhi",
        "Location": "Saket",
        "Area": 2800,
        "No. of Bedrooms": 4,
        "Resale": 0,
        "Gymnasium": 1, "SwimmingPool": 1, "ClubHouse": 1, "24X7Security": 1,
        "PowerBackup": 1, "CarParking": 1, "LiftAvailable": 1, "AC": 1,
        "Gasconnection": 1, "VaastuCompliant": 1, "Children'splayarea": 1
    },
    "🚀 3 BHK IT Corridor Gated Community — Gachibowli (Hyderabad)": {
        "City": "Hyderabad",
        "Location": "Gachibowli",
        "Area": 1950,
        "No. of Bedrooms": 3,
        "Resale": 0,
        "Gymnasium": 1, "SwimmingPool": 1, "ClubHouse": 1, "24X7Security": 1,
        "PowerBackup": 1, "CarParking": 1, "LiftAvailable": 1, "AC": 1,
        "Gasconnection": 1, "VaastuCompliant": 1, "Children'splayarea": 1
    },
    "🏘️ 2 BHK IT Express Flat — OMR (Chennai)": {
        "City": "Chennai",
        "Location": "OMR",
        "Area": 1150,
        "No. of Bedrooms": 2,
        "Resale": 1,
        "Gymnasium": 1, "SwimmingPool": 0, "ClubHouse": 1, "24X7Security": 1,
        "PowerBackup": 1, "CarParking": 1, "LiftAvailable": 1, "AC": 0,
        "Gasconnection": 0, "VaastuCompliant": 1, "Children'splayarea": 1
    },
    "🌿 3 BHK Planned Township Flat — Salt Lake / New Town (Kolkata)": {
        "City": "Kolkata",
        "Location": "Salt Lake",
        "Area": 1400,
        "No. of Bedrooms": 3,
        "Resale": 0,
        "Gymnasium": 1, "SwimmingPool": 1, "ClubHouse": 1, "24X7Security": 1,
        "PowerBackup": 1, "CarParking": 1, "LiftAvailable": 1, "AC": 1,
        "Gasconnection": 1, "VaastuCompliant": 1, "Children'splayarea": 1
    }
}


# -------------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_housing_data():
    """Load cleaned Indian housing dataset."""
    try:
        return load_clean_housing_data()
    except Exception:
        download_city_datasets_if_needed()
        return load_clean_housing_data()


@st.cache_resource(show_spinner=False)
def get_cached_model_bundle():
    """Load trained model bundle."""
    return load_valuation_model_bundle()


@st.cache_data(show_spinner=False)
def get_cached_metadata():
    """Load model metadata JSON."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# -------------------------------------------------------------
# INITIALIZE SESSION STATE
# -------------------------------------------------------------
if "valuation_result" not in st.session_state:
    st.session_state["valuation_result"] = None
if "last_input_data" not in st.session_state:
    st.session_state["last_input_data"] = None


# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏡 BharatProp AI")
    st.caption("Indian Real Estate Valuation & Appraisal Engine")
    st.markdown("---")

    app_mode = st.radio(
        "Navigation",
        [
            "🏠 Overview & Methodology",
            "🎯 Property Valuation",
            "📊 Model Performance",
            "📈 Market Analytics",
            "📁 Bulk Valuation (CSV)",
            "💰 Home Loan EMI Planner"
        ],
        index=1
    )

    st.markdown("---")
    
    # Metadata Badge
    metadata = get_cached_metadata()
    if metadata:
        st.markdown("**System Information**")
        st.caption(f"🤖 **Best Model:** {metadata.get('best_model_name', 'Gradient Boosting')}")
        st.caption(f"📉 **Test MAE:** ₹{metadata.get('test_mae_lakhs', 41.53):.2f} Lakhs")
        st.caption(f"🎯 **Test R² (Log):** {metadata.get('test_r2_log', 0.4647):.4f}")
        st.caption(f"📊 **Trained Properties:** {metadata.get('total_properties_trained', 19328):,}")
        st.caption(f"🏙️ **Covered Metros:** 6 Major Cities")
        st.caption(f"📍 **Micro-Markets:** {metadata.get('distinct_localities', 1711):,} Localities")
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.75rem; color: #64748b; line-height: 1.4;'>
    <b>Disclaimer:</b> Predictions are algorithmic statistical estimates calibrated with conformal prediction. Not an official legal, tax, or banking mortgage appraisal.
    </div>
    """, unsafe_allow_html=True)


# =============================================================
# PAGE 1: OVERVIEW & METHODOLOGY
# =============================================================
if app_mode == "🏠 Overview & Methodology":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">AI-POWERED REAL ESTATE VALUATION</div>
        <div class="hero-title">BharatProp AI Valuation Engine</div>
        <div class="hero-subtitle">
            An enterprise-grade Automated Valuation Model (AVM) designed specifically for Indian metropolitan residential property markets, powered by Gradient Boosting, XGBoost, and Conformal Uncertainty Quantification.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-val">27,600+</div>
            <div class="metric-label">Cleaned Indian Properties</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        with col2:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-val">6 Metros</div>
                <div class="metric-label">Tier-1 Indian Cities</div>
            </div>
            """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-val">1,710+</div>
            <div class="metric-label">Micro-Markets & Localities</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-val">35+</div>
            <div class="metric-label">Engineered Amenities</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📌 Problem Statement & Indian Market Realities")
    st.markdown("""
    Indian residential real estate is highly diverse, fragmented, and non-linear. Property prices vary dramatically across:
    - **Metropolitan Economic Hubs**: Mumbai's high density commands median rates exceeding ₹10,000/sq.ft, while Bangalore and Hyderabad emphasize spacious IT-corridor developments averaging 1,500–1,700 sq.ft.
    - **Micro-Market Hierarchy**: Proximity to tech parks, sea-facing locations, and metro connectivity create substantial micro-market premiums within the same city.
    - **Amenity & Infrastructure Density**: Gated communities with power backup, 24x7 security, clubhouses, swimming pools, and furnishing packages command quantifiable premiums over standalone buildings.
    
    **BharatProp AI** replaces artificial foreign datasets with **authentic, multi-city Indian housing data**, predicting prices natively in **Indian Rupees (₹)** and providing mathematically guaranteed **Conformal Prediction Intervals**.
    """)

    st.markdown("### 🛠️ End-to-End Machine Learning Architecture")
    st.markdown("""
    ```mermaid
    flowchart LR
        A[6 Indian Metro Datasets<br>Mumbai, Bangalore, Delhi, Chennai, Hyderabad, Kolkata] --> B[Data Cleaning & Deduplication<br>Outlier Boundary Filter ₹1.8k-48k/sqft]
        B --> C[Feature Engineering<br>Area/BHK, BHKxArea, Amenity & Furnishing Scores]
        C --> D[ColumnTransformer Preprocessor<br>OneHot City + TargetEncoder Location + RobustScaler]
        D --> E[Multi-Model 5-Fold CV Training<br>Ridge, Random Forest, Gradient Boosting, XGBoost]
        E --> F[Log-Target Transformation<br>log1p / expm1 Inversion]
        F --> G[Conformal Prediction Calibration<br>Empirical Residual Quantiles on Calib Fold]
        G --> H[Streamlit UI Appraisal Engine<br>Valuation, ₹/sq.ft & Calibrated 80% Prediction Range]
    ```
    """)

    st.markdown("### 📋 Dataset Summary & Geographic Coverage")
    df_preview = get_cached_housing_data()
    city_summary = df_preview.groupby("City").agg(
        Total_Listings=("Price", "count"),
        Median_Price_Lakhs=("Price_Lakhs", "median"),
        Mean_Area_SqFt=("Area", "mean"),
        Median_Rate_SqFt=("Price_per_sqft", "median")
    ).reset_index()
    city_summary.columns = ["City", "Total Properties", "Median Price (₹ Lakhs)", "Mean Area (sq ft)", "Median Rate (₹/sq.ft)"]
    city_summary["Median Price (₹ Lakhs)"] = city_summary["Median Price (₹ Lakhs)"].apply(lambda x: f"₹{x:.1f} L")
    city_summary["Mean Area (sq ft)"] = city_summary["Mean Area (sq ft)"].apply(lambda x: f"{x:,.0f} sq ft")
    city_summary["Median Rate (₹/sq.ft)"] = city_summary["Median Rate (₹/sq.ft)"].apply(lambda x: f"₹{x:,.0f}/sq.ft")
    
    st.dataframe(city_summary, use_container_width=True, hide_index=True)

    st.markdown("### ⚠️ Limitations & Professional Appraisal Disclaimer")
    st.info("""
    - **Listing vs Transaction Prices**: Data reflects asking/listing prices from public aggregators. Actual negotiated transaction prices in India may vary by 5% to 15%.
    - **Unobserved Micro-Factors**: Specific floor-rise charges, view quality (lake-view/corner plot), builder brand reputation, and exact legal documentation quality cannot be captured solely from numerical tabular attributes.
    - **Statistically Calibrated**: Prediction ranges are produced via split conformal prediction, providing valid marginal coverage on historical test folds.
    """)


# =============================================================
# PAGE 2: PROPERTY VALUATION (APPRAISAL ENGINE)
# =============================================================
elif app_mode == "🎯 Property Valuation":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">AUTOMATED APPRAISAL ENGINE</div>
        <div class="hero-title">Indian Residential Property Valuation</div>
        <div class="hero-subtitle">
            Configure property specifications to generate real-time market appraisals in Indian Rupees (₹), estimated ₹/sq.ft rates, and calibrated prediction intervals.
        </div>
    </div>
    """, unsafe_allow_html=True)

    city_localities = get_available_cities_and_localities()
    cities_list = sorted(list(city_localities.keys()))

    # Preset selector
    selected_preset_name = st.selectbox(
        "⚡ Quick Start with Indian Property Presets:",
        list(INDIAN_PROPERTY_PRESETS.keys())
    )
    preset_data = INDIAN_PROPERTY_PRESETS[selected_preset_name] if selected_preset_name != "Select a pre-configured Indian property preset..." else None

    with st.form("valuation_input_form"):
        st.markdown("#### 1. Location & Primary Configuration")
        col_c1, col_c2, col_c3 = st.columns(3)

        default_city_idx = 0
        if preset_data and preset_data.get("City") in cities_list:
            default_city_idx = cities_list.index(preset_data["City"])

        with col_c1:
            city_input = st.selectbox("Metropolitan City", cities_list, index=default_city_idx)
        
        available_locs = city_localities.get(city_input, ["Other"])
        default_loc_idx = 0
        if preset_data and preset_data.get("Location") in available_locs:
            default_loc_idx = available_locs.index(preset_data["Location"])

        with col_c2:
            locality_input = st.selectbox(
                f"Micro-Market / Locality in {city_input}",
                available_locs,
                index=default_loc_idx,
                help="Select locality from 1,700+ verified Indian micro-markets."
            )
        
        with col_c3:
            custom_loc = st.text_input(
                "Or Type Custom Locality (Optional)",
                value="",
                placeholder="Leave blank to use selected locality",
                help="If your specific colony/suburb is not in the list, type it here."
            )

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            area_input = st.number_input(
                "Super Built-up Area (Square Feet)",
                min_value=300,
                max_value=10000,
                value=int(preset_data.get("Area", 1250)) if preset_data else 1250,
                step=50,
                help="Standard built-up floor area of the residential unit."
            )
        with col_p2:
            bhk_input = st.selectbox(
                "Bedrooms (BHK Configuration)",
                [1, 2, 3, 4, 5, 6],
                index=int(preset_data.get("No. of Bedrooms", 2) - 1) if preset_data else 1
            )
        with col_p3:
            resale_label = st.selectbox(
                "Property Sale Type",
                ["New Construction / Builder Direct (0)", "Resale / Pre-owned Property (1)"],
                index=int(preset_data.get("Resale", 0)) if preset_data else 0
            )
            resale_val = 1 if "Resale" in resale_label else 0

        st.markdown("#### 2. Key Amenities & Community Infrastructure")
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)

        with col_a1:
            lift_val = st.checkbox("🛗 Lift Available", value=bool(preset_data.get("LiftAvailable", 1)) if preset_data else True)
            power_val = st.checkbox("⚡ 24x7 Power Backup", value=bool(preset_data.get("PowerBackup", 1)) if preset_data else True)
            parking_val = st.checkbox("🚗 Reserved Car Parking", value=bool(preset_data.get("CarParking", 1)) if preset_data else True)
        
        with col_a2:
            security_val = st.checkbox("🛡️ 24x7 Gated Security", value=bool(preset_data.get("24X7Security", 1)) if preset_data else True)
            intercom_val = st.checkbox("📞 Intercom Facility", value=bool(preset_data.get("Intercom", 1)) if preset_data else True)
            gas_val = st.checkbox("🔥 Piped Gas Connection", value=bool(preset_data.get("Gasconnection", 1)) if preset_data else False)

        with col_a3:
            gym_val = st.checkbox("🏋️ Fitness Gym", value=bool(preset_data.get("Gymnasium", 1)) if preset_data else True)
            pool_val = st.checkbox("🏊 Swimming Pool", value=bool(preset_data.get("SwimmingPool", 1)) if preset_data else True)
            club_val = st.checkbox("🏛️ Clubhouse & Party Hall", value=bool(preset_data.get("ClubHouse", 1)) if preset_data else True)

        with col_a4:
            vaastu_val = st.checkbox("🧭 Vaastu Compliant", value=bool(preset_data.get("VaastuCompliant", 1)) if preset_data else True)
            play_val = st.checkbox("🎪 Children's Play Area", value=bool(preset_data.get("Children'splayarea", 1)) if preset_data else True)
            ac_val = st.checkbox("❄️ Air Conditioned (AC)", value=bool(preset_data.get("AC", 0)) if preset_data else False)

        st.markdown("#### 3. Prediction Confidence Level")
        col_conf, _ = st.columns([2, 2])
        with col_conf:
            conf_choice = st.radio(
                "Prediction Interval Coverage (Conformal Prediction)",
                ["80% Prediction Interval (Recommended)", "90% Prediction Interval (Wider)"],
                horizontal=True
            )
            coverage_val = 0.80 if "80%" in conf_choice else 0.90

        submitted = st.form_submit_button("🔍 Calculate Property Valuation Appraisal", use_container_width=True)

    if submitted:
        active_location = custom_loc.strip() if custom_loc.strip() else locality_input
        
        # Validation checks
        if area_input < 300:
            st.error("⚠️ Please enter a realistic residential area (at least 300 sq ft).")
        elif area_input > 12000:
            st.error("⚠️ Maximum residential area supported is 12,000 sq ft.")
        else:
            input_dict = {
                "City": city_input,
                "Location": active_location,
                "Area": float(area_input),
                "No. of Bedrooms": int(bhk_input),
                "Resale": int(resale_val),
                "LiftAvailable": int(lift_val),
                "PowerBackup": int(power_val),
                "CarParking": int(parking_val),
                "24X7Security": int(security_val),
                "Intercom": int(intercom_val),
                "Gasconnection": int(gas_val),
                "Gymnasium": int(gym_val),
                "SwimmingPool": int(pool_val),
                "ClubHouse": int(club_val),
                "VaastuCompliant": int(vaastu_val),
                "Children'splayarea": int(play_val),
                "AC": int(ac_val)
            }

            with st.spinner("Analyzing micro-market trends and calculating valuation..."):
                val_res = predict_property_valuation(input_dict, coverage=coverage_val)
                st.session_state["valuation_result"] = val_res
                st.session_state["last_input_data"] = input_dict

    # Display Valuation Result Card if available
    if st.session_state.get("valuation_result"):
        res = st.session_state["valuation_result"]
        inp = st.session_state["last_input_data"]

        st.markdown("---")
        st.markdown("### 📊 Automated Property Appraisal Summary")

        st.markdown(f"""
        <div class="valuation-card">
            <div style="font-size: 0.9rem; letter-spacing: 1px; text-transform: uppercase; font-weight: 700; color: #6ee7b7;">
                ESTIMATED MARKET VALUATION ({inp['City'].upper()} — {inp['Location']})
            </div>
            <div class="val-price">{res['formatted_price']}</div>
            <div class="val-subtext">Estimated Rupee Value: <b>₹{res['price_in_lakhs']:.2f} Lakhs</b> &nbsp;|&nbsp; Rate: <b>{res['formatted_price_per_sqft']}</b></div>
            <hr style="border-color: rgba(255,255,255,0.2); margin: 1rem 0;">
            <div class="val-range">📐 {int(res['coverage_level']*100)}% Prediction Interval: {res['formatted_range']}</div>
            <div class="val-subtext">Statistically calibrated conformal prediction range (±{((res['upper_bound']-res['lower_bound'])/(2*res['predicted_price']))*100:.1f}% bandwidth).</div>
        </div>
        """, unsafe_allow_html=True)

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("Estimated Market Price", res["formatted_price"], f"₹{res['price_in_lakhs']:.2f} L")
        with col_r2:
            st.metric("Price per Sq.Ft Rate", res["formatted_price_per_sqft"])
        with col_r3:
            st.metric("Estimated Lower Bound", format_inr(res["lower_bound"]), f"₹{res['lower_in_lakhs']:.2f} L")
        with col_r4:
            st.metric("Estimated Upper Bound", format_inr(res["upper_bound"]), f"₹{res['upper_in_lakhs']:.2f} L")

        # Benchmark Comparison with City Average
        df_all = get_cached_housing_data()
        city_df = df_all[df_all["City"] == inp["City"]]
        city_med_rate = city_df["Price_per_sqft"].median()
        rate_diff_pct = ((res["price_per_sqft"] - city_med_rate) / city_med_rate) * 100

        st.markdown("#### 📍 City Benchmark Analysis")
        comp_text = "above" if rate_diff_pct > 0 else "below"
        st.write(
            f"This property's estimated rate of **{res['formatted_price_per_sqft']}** is **{abs(rate_diff_pct):.1f}% {comp_text}** "
            f"the median rate of **₹{city_med_rate:,.0f}/sq.ft** in **{inp['City']}**."
        )


# =============================================================
# PAGE 3: MODEL PERFORMANCE & EVALUATION
# =============================================================
elif app_mode == "📊 Model Performance":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">RIGOROUS MODEL EVALUATION</div>
        <div class="hero-title">Machine Learning Benchmarks & Validation</div>
        <div class="hero-subtitle">
            Comprehensive evaluation across Ridge Regression, Random Forest, Gradient Boosting, and XGBoost using 5-Fold Cross-Validation on authentic Indian housing data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load Comparison Table
    results_csv_path = PROJECT_ROOT / "outputs" / "results" / "model_comparison.csv"
    if results_csv_path.exists():
        comp_df = pd.read_csv(results_csv_path)
        
        st.markdown("### 🏆 Model Comparison Leaderboard")
        display_comp = comp_df.copy()
        display_comp["MAE"] = display_comp["MAE"].apply(lambda x: format_inr(x))
        display_comp["RMSE"] = display_comp["RMSE"].apply(lambda x: format_inr(x))
        display_comp["R² (Rupees)"] = display_comp["R2"].apply(lambda x: f"{x:.4f}")
        display_comp["R² (Log Scale)"] = display_comp["R2_log"].apply(lambda x: f"{x:.4f}")
        display_comp["5-Fold CV R² (Log)"] = display_comp["CV_R2_Log"].apply(lambda x: f"{x:.4f}")
        display_comp["MAPE (%)"] = display_comp["MAPE"].apply(lambda x: f"{x:.2f}%")
        
        table_cols = ["Model", "MAE", "RMSE", "R² (Rupees)", "R² (Log Scale)", "5-Fold CV R² (Log)", "MAPE (%)"]
        st.dataframe(display_comp[table_cols], use_container_width=True, hide_index=True)
    else:
        st.warning("Model comparison table not found. Please run `python src/train.py`.")

    st.markdown("### 📖 Understanding Regression Metrics")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        - **MAE (Mean Absolute Error)**: Average absolute rupee discrepancy between predicted and actual property prices. Lower is better.
        - **RMSE (Root Mean Squared Error)**: Penalizes larger pricing errors more heavily. Useful for identifying extreme outlier pricing mistakes.
        """)
    with col_m2:
        st.markdown("""
        - **R² Score (Coefficient of Determination)**: Proportion of variance in property prices explained by the features (0.0 to 1.0). *Note: R² is not an accuracy percentage.*
        - **MAPE (Mean Absolute Percentage Error)**: Average percentage valuation error across different price tiers.
        """)

    st.markdown("### 📈 Evaluation Visualizations")
    tab1, tab2, tab3 = st.tabs(["🎯 Actual vs Predicted", "📉 Residual Analysis", "📊 Model Comparison Bar Charts"])

    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    with tab1:
        p_act = plots_dir / "08_actual_vs_predicted.png"
        if p_act.exists():
            st.image(str(p_act), caption="Actual vs Predicted Property Valuation (₹ in Lakhs) on Holdout Test Set", use_container_width=True)
        else:
            st.info("Run `python src/train.py` to generate the Actual vs Predicted plot.")

    with tab2:
        p_res = plots_dir / "09_residuals_plot.png"
        if p_res.exists():
            st.image(str(p_res), caption="Residual Plot (Actual - Predicted) across Predicted Valuation Tiers", use_container_width=True)
        else:
            st.info("Run `python src/train.py` to generate the Residuals plot.")

    with tab3:
        p_comp = plots_dir / "10_model_comparison_bar.png"
        if p_comp.exists():
            st.image(str(p_comp), caption="Model Comparison across MAE, RMSE, and Log-Scale R² Metrics", use_container_width=True)
        else:
            st.info("Run `python src/train.py` to generate the Comparison Bar Chart.")


# =============================================================
# PAGE 4: MARKET ANALYTICS & EDA
# =============================================================
elif app_mode == "📈 Market Analytics":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">INDIAN REAL ESTATE ANALYTICS</div>
        <div class="hero-title">Market Intelligence & Macro Trends</div>
        <div class="hero-subtitle">
            Explore pricing distributions, micro-market premiums, square footage trends, and amenity valuations across 27,600+ genuine Indian properties.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_market = get_cached_housing_data()

    # City Filter
    selected_city = st.selectbox(
        "Filter Analytics by Metropolitan City:",
        ["All 6 Metropolitan Cities"] + sorted(df_market["City"].unique().tolist())
    )

    filtered_df = df_market if selected_city == "All 6 Metropolitan Cities" else df_market[df_market["City"] == selected_city]

    # Metrics Row
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric("Total Listings Analyzed", f"{len(filtered_df):,}")
    with col_k2:
        st.metric("Median Property Price", f"₹{filtered_df['Price_Lakhs'].median():.1f} L", f"Mean: ₹{filtered_df['Price_Lakhs'].mean():.1f} L")
    with col_k3:
        st.metric("Median Rate / Sq.Ft", f"₹{filtered_df['Price_per_sqft'].median():,.0f}/sq.ft")
    with col_k4:
        st.metric("Average Built-Up Area", f"{filtered_df['Area'].mean():.0f} sq ft")

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### 💰 Price Distribution (₹ in Lakhs)")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(filtered_df["Price_Lakhs"], bins=35, kde=True, color="#0284c7", ax=ax)
        ax.set_xlim(0, filtered_df["Price_Lakhs"].quantile(0.97))
        ax.set_xlabel("Property Price (₹ in Lakhs)")
        ax.set_ylabel("Number of Listings")
        st.pyplot(fig)
        plt.close(fig)

    with col_g2:
        st.markdown("#### 📐 Built-Up Area vs Price")
        fig, ax = plt.subplots(figsize=(8, 5))
        sample_sub = filtered_df.sample(min(2000, len(filtered_df)), random_state=42)
        sns.scatterplot(
            x="Area",
            y="Price_Lakhs",
            hue="City" if selected_city == "All 6 Metropolitan Cities" else "No. of Bedrooms",
            data=sample_sub,
            alpha=0.6,
            palette="viridis",
            ax=ax
        )
        ax.set_ylim(0, sample_sub["Price_Lakhs"].quantile(0.98))
        ax.set_xlabel("Built-Up Area (sq ft)")
        ax.set_ylabel("Price (₹ in Lakhs)")
        st.pyplot(fig)
        plt.close(fig)

    col_g3, col_g4 = st.columns(2)
    with col_g3:
        st.markdown("#### 🛏️ Price Distribution by BHK Configuration")
        fig, ax = plt.subplots(figsize=(8, 5))
        bhk_sub = filtered_df[filtered_df["No. of Bedrooms"].between(1, 5)]
        sns.boxplot(x="No. of Bedrooms", y="Price_Lakhs", hue="No. of Bedrooms", data=bhk_sub, palette="Blues", legend=False, ax=ax)
        ax.set_ylim(0, bhk_sub["Price_Lakhs"].quantile(0.97))
        ax.set_xlabel("Bedrooms (BHK)")
        ax.set_ylabel("Price (₹ in Lakhs)")
        st.pyplot(fig)
        plt.close(fig)

    with col_g4:
        st.markdown(f"#### 📍 Top 10 High-Volume Micro-Markets in {selected_city}")
        fig, ax = plt.subplots(figsize=(8, 5))
        top_locs = filtered_df["Location"].value_counts().head(10).index
        loc_df = filtered_df[filtered_df["Location"].isin(top_locs)]
        loc_order = loc_df.groupby("Location")["Price_per_sqft"].median().sort_values(ascending=True).index
        sns.barplot(x="Price_per_sqft", y="Location", data=loc_df, order=loc_order, ci=None, palette="Greens_r", ax=ax)
        ax.set_xlabel("Median Rate (₹ / sq.ft)")
        ax.set_ylabel("Locality")
        st.pyplot(fig)
        plt.close(fig)


# =============================================================
# PAGE 5: BULK VALUATION (CSV UPLOAD)
# =============================================================
elif app_mode == "📁 Bulk Valuation (CSV)":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">ENTERPRISE BATCH PROCESSING</div>
        <div class="hero-title">Bulk Property Valuation via CSV</div>
        <div class="hero-subtitle">
            Upload CSV portfolio files to perform batch automated valuations, compute ₹/sq.ft metrics, and generate calibrated prediction ranges simultaneously.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sample template download
    sample_csv_data = pd.DataFrame([
        {"City": "Bangalore", "Location": "Whitefield", "Area": 1450, "No. of Bedrooms": 3, "Resale": 0, "Gymnasium": 1, "SwimmingPool": 1, "PowerBackup": 1, "LiftAvailable": 1, "24X7Security": 1},
        {"City": "Mumbai", "Location": "Andheri East", "Area": 850, "No. of Bedrooms": 2, "Resale": 1, "Gymnasium": 1, "SwimmingPool": 0, "PowerBackup": 1, "LiftAvailable": 1, "24X7Security": 1},
        {"City": "Delhi", "Location": "Dwarka", "Area": 1600, "No. of Bedrooms": 3, "Resale": 0, "Gymnasium": 1, "SwimmingPool": 1, "PowerBackup": 1, "LiftAvailable": 1, "24X7Security": 1},
        {"City": "Hyderabad", "Location": "Gachibowli", "Area": 2100, "No. of Bedrooms": 4, "Resale": 0, "Gymnasium": 1, "SwimmingPool": 1, "PowerBackup": 1, "LiftAvailable": 1, "24X7Security": 1},
        {"City": "Chennai", "Location": "Velachery", "Area": 1100, "No. of Bedrooms": 2, "Resale": 1, "Gymnasium": 0, "SwimmingPool": 0, "PowerBackup": 1, "LiftAvailable": 1, "24X7Security": 0},
        {"City": "Kolkata", "Location": "New Town", "Area": 1350, "No. of Bedrooms": 3, "Resale": 0, "Gymnasium": 1, "SwimmingPool": 1, "PowerBackup": 1, "LiftAvailable": 1, "24X7Security": 1}
    ])

    col_b1, col_b2 = st.columns([3, 1])
    with col_b1:
        uploaded_file = st.file_uploader("Choose a CSV file containing property records", type=["csv"])
    with col_b2:
        st.markdown("<br>", unsafe_allow_html=True)
        csv_buffer = io.StringIO()
        sample_csv_data.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_buffer.getvalue(),
            file_name="sample_indian_properties.csv",
            mime="text/csv",
            use_container_width=True
        )

    if uploaded_file is not None:
        try:
            input_bulk_df = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded File:** `{uploaded_file.name}` ({len(input_bulk_df)} properties)")

            required_cols = ["City", "Location", "Area", "No. of Bedrooms"]
            missing_cols = [c for c in required_cols if c not in input_bulk_df.columns]

            if missing_cols:
                st.error(f"⚠️ Missing required columns in CSV: `{', '.join(missing_cols)}`. Required: `{', '.join(required_cols)}`.")
            else:
                st.markdown("#### Input Data Preview")
                st.dataframe(input_bulk_df.head(5), use_container_width=True)

                if st.button("🚀 Run Batch Valuation Appraisal", use_container_width=True):
                    with st.spinner(f"Evaluating {len(input_bulk_df)} properties..."):
                        preds_list = predict_property_valuation(input_bulk_df, coverage=0.80)
                        
                        out_df = input_bulk_df.copy()
                        out_df["Estimated_Price_INR"] = [r["predicted_price"] for r in preds_list]
                        out_df["Estimated_Price_Lakhs"] = [r["price_in_lakhs"] for r in preds_list]
                        out_df["Estimated_Price_Formatted"] = [r["formatted_price"] for r in preds_list]
                        out_df["Rate_Per_SqFt"] = [r["formatted_price_per_sqft"] for r in preds_list]
                        out_df["80pct_Prediction_Range"] = [r["formatted_range"] for r in preds_list]
                        out_df["Lower_Bound_INR"] = [r["lower_bound"] for r in preds_list]
                        out_df["Upper_Bound_INR"] = [r["upper_bound"] for r in preds_list]

                        st.success(f"Successfully appraised all {len(out_df)} properties!")
                        st.markdown("#### Appraisal Output Results")
                        st.dataframe(
                            out_df[["City", "Location", "Area", "No. of Bedrooms", "Estimated_Price_Formatted", "Rate_Per_SqFt", "80pct_Prediction_Range"]],
                            use_container_width=True
                        )

                        # Download Result CSV
                        out_csv = io.StringIO()
                        out_df.to_csv(out_csv, index=False)
                        st.download_button(
                            label="📥 Download Complete Valuation Results CSV",
                            data=out_csv.getvalue(),
                            file_name="valuation_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"Error processing CSV file: {e}")


# =============================================================
# PAGE 6: HOME LOAN EMI & FINANCIAL PLANNER
# =============================================================
elif app_mode == "💰 Home Loan EMI Planner":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">FINANCIAL PLANNING SUITE</div>
        <div class="hero-title">Indian Home Loan & EMI Calculator</div>
        <div class="hero-subtitle">
            Plan home financing with standard Indian banking amortization schedules, down payment calculations, and loan-to-value (LTV) insights.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-callout">
        💡 <b>Note:</b> This EMI calculator is an additional financial planning tool designed to estimate monthly commitments. It is separate from the core ML property valuation model.
    </div>
    """, unsafe_allow_html=True)

    # Preset valuation sync if user came from appraisal page
    default_prop_val = 7500000.0
    if st.session_state.get("valuation_result"):
        default_prop_val = float(st.session_state["valuation_result"]["predicted_price"])

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        prop_val_input = st.number_input(
            "Estimated Property Purchase Value (₹)",
            min_value=500000.0,
            max_value=200000000.0,
            value=float(default_prop_val),
            step=100000.0,
            format="%.0f"
        )
        down_pct = st.slider("Down Payment Percentage (%)", min_value=10, max_value=50, value=20, step=5)
        down_amount = (down_pct / 100.0) * prop_val_input
        loan_amount = prop_val_input - down_amount

    with col_e2:
        interest_rate = st.slider("Annual Interest Rate (% p.a.)", min_value=6.0, max_value=15.0, value=8.5, step=0.1)
        tenure_years = st.slider("Loan Tenure (Years)", min_value=5, max_value=30, value=20, step=1)
        tenure_months = tenure_years * 12

    # EMI Formula: E = P * r * (1 + r)^n / ((1 + r)^n - 1)
    monthly_r = (interest_rate / 12.0) / 100.0
    if monthly_r > 0:
        emi = loan_amount * monthly_r * ((1 + monthly_r) ** tenure_months) / (((1 + monthly_r) ** tenure_months) - 1)
    else:
        emi = loan_amount / tenure_months

    total_payment = emi * tenure_months
    total_interest = total_payment - loan_amount

    st.markdown("---")
    st.markdown("### 📋 Loan & Repayment Summary")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Monthly EMI", format_inr(emi))
    with col_m2:
        st.metric("Loan Principal", format_inr(loan_amount), f"Down: {format_inr(down_amount)}")
    with col_m3:
        st.metric("Total Interest", format_inr(total_interest))
    with col_m4:
        st.metric("Total Amount Payable", format_inr(total_payment))

    # Pie Chart breakdown
    col_chart, col_sched = st.columns([1, 1])
    with col_chart:
        st.markdown("#### 🥧 Principal vs Interest Breakdown")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            [loan_amount, total_interest],
            labels=[f"Principal\n({format_inr(loan_amount)})", f"Total Interest\n({format_inr(total_interest)})"],
            autopct="%1.1f%%",
            colors=["#10b981", "#f59e0b"],
            startangle=140,
            textprops={"fontsize": 11, "weight": "bold"}
        )
        st.pyplot(fig)
        plt.close(fig)

    with col_sched:
        st.markdown("#### 📅 Yearly Amortization Schedule Preview")
        # Build yearly schedule
        curr_balance = loan_amount
        schedule_rows = []
        for y in range(1, min(tenure_years + 1, 11)):
            year_interest = 0
            year_principal = 0
            for _ in range(12):
                int_m = curr_balance * monthly_r
                prin_m = emi - int_m
                year_interest += int_m
                year_principal += prin_m
                curr_balance = max(0, curr_balance - prin_m)
            schedule_rows.append({
                "Year": f"Year {y}",
                "Principal Paid": format_inr(year_principal),
                "Interest Paid": format_inr(year_interest),
                "Remaining Balance": format_inr(curr_balance)
            })
        
        st.dataframe(pd.DataFrame(schedule_rows), use_container_width=True, hide_index=True)
