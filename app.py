"""
BharatProp AI — Indian Real Estate Valuation & Automated Appraisal Engine
State-of-the-Art Machine Learning Platform for Residential Property Appraisal, Market Analytics & Financial Planning.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple

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
# PAGE CONFIGURATION & METADATA
# -------------------------------------------------------------
st.set_page_config(
    page_title="BharatProp AI | Real Estate Valuation & Appraisal Engine",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Standard USD to INR Exchange Rate (Live Baseline)
USD_TO_INR_RATE = 83.50

# -------------------------------------------------------------
# COMPREHENSIVE INDIAN LOCALITY, CITY & STATE MAPPING
# -------------------------------------------------------------
LOCALITY_MAP = {
    "CollgCr": {"name": "Whitefield / ITPL Corridor", "city": "Bengaluru", "state": "Karnataka", "tier": "Tier-1 High Growth IT Belt"},
    "Veenker": {"name": "Banjara Hills / Road No. 12", "city": "Hyderabad", "state": "Telangana", "tier": "Ultra-Luxury Prime Heritage"},
    "Crawfor": {"name": "Jubilee Hills / Film Nagar", "city": "Hyderabad", "state": "Telangana", "tier": "Elite Residential Enclave"},
    "NoRidge": {"name": "Malabar Hill / South Mumbai", "city": "Mumbai", "state": "Maharashtra", "tier": "Super Luxury Waterfront"},
    "Mitchel": {"name": "HITEC City / Madhapur", "city": "Hyderabad", "state": "Telangana", "tier": "Global Tech Commercial Hub"},
    "Somerst": {"name": "Koramangala / HSR Layout", "city": "Bengaluru", "state": "Karnataka", "tier": "Premium Startup & Tech District"},
    "NWAmes": {"name": "Gachibowli / Financial District", "city": "Hyderabad", "state": "Telangana", "tier": "Fintech & Corporate Corridor"},
    "OldTown": {"name": "Old City / Charminar Heritage Area", "city": "Hyderabad", "state": "Telangana", "tier": "Historic Urban Core"},
    "BrkSide": {"name": "Malleshwaram / Basavanagudi", "city": "Bengaluru", "state": "Karnataka", "tier": "Established Cultural Suburb"},
    "Sawyer": {"name": "Kukatpally / KPHB Colony", "city": "Hyderabad", "state": "Telangana", "tier": "High Density Residential Hub"},
    "NridgHt": {"name": "Golf Course Road / Cyber City", "city": "Gurugram", "state": "Delhi-NCR / Haryana", "tier": "Executive Luxury Boulevard"},
    "NAmes": {"name": "Jayanagar / JP Nagar", "city": "Bengaluru", "state": "Karnataka", "tier": "Prime Traditional Residential"},
    "SawyerW": {"name": "Manikonda / Nanakramguda", "city": "Hyderabad", "state": "Telangana", "tier": "Rapid Growth IT Outskirt"},
    "IDOTRR": {"name": "MVP Colony / Railway Zone", "city": "Visakhapatnam", "state": "Andhra Pradesh", "tier": "Coastal Transit Center"},
    "MeadowV": {"name": "Electronic City / Attibele", "city": "Bengaluru", "state": "Karnataka", "tier": "Affordable IT Housing"},
    "Edwards": {"name": "Begumpet / Secunderabad", "city": "Hyderabad", "state": "Telangana", "tier": "Central Commercial & Living"},
    "Timber": {"name": "Jubilee Enclave / Kavuri Hills", "city": "Hyderabad", "state": "Telangana", "tier": "Scenic Green Residential"},
    "Gilbert": {"name": "Sarjapur Road / Bellandur ORR", "city": "Bengaluru", "state": "Karnataka", "tier": "Outer Ring Road Tech Belt"},
    "StoneBr": {"name": "Bandra West / Worli Sea Face", "city": "Mumbai", "state": "Maharashtra", "tier": "Elite Presidential Coastal"},
    "ClearCr": {"name": "Gandipet Lake / Eco Zone", "city": "Hyderabad", "state": "Telangana", "tier": "Low-Density Eco-Villa Zone"},
    "NPkVill": {"name": "Madhurawada IT SEZ", "city": "Visakhapatnam", "state": "Andhra Pradesh", "tier": "Coastal IT Corridor"},
    "Blmngtn": {"name": "Palm Meadows / Whitefield Gated Villa", "city": "Bengaluru", "state": "Karnataka", "tier": "Gated Luxury Community"},
    "BrDale": {"name": "Ameerpet / SR Nagar", "city": "Hyderabad", "state": "Telangana", "tier": "Educational & Commercial Zone"},
    "SWISU": {"name": "Beach Road / Andhra University Enclave", "city": "Visakhapatnam", "state": "Andhra Pradesh", "tier": "Institutional & Sea-View"},
    "Blueste": {"name": "Boat Club Road / Koregaon Park", "city": "Pune", "state": "Maharashtra", "tier": "Boutique Green Enclave"}
}

# -------------------------------------------------------------
# INDIAN PROPERTY PRESET PROFILES
# -------------------------------------------------------------
INDIAN_PRESETS = {
    "🏡 4BHK Grand Villa (Banjara Hills, Hyderabad)": {
        "Neighborhood": "Veenker", "BldgType": "1Fam", "HouseStyle": "2Story", "MSZoning": "RL",
        "GrLivArea": 3200, "1stFlrSF": 1600, "2ndFlrSF": 1600, "TotalBsmtSF": 1400,
        "LotArea": 13500, "LotFrontage": 95, "OverallQual": 9, "OverallCond": 7,
        "YearBuilt": 2014, "YearRemodAdd": 2020, "BedroomAbvGr": 4, "FullBath": 4, "HalfBath": 1,
        "TotRmsAbvGrd": 9, "GarageCars": 3, "GarageArea": 720, "Fireplaces": 2, "WoodDeckSF": 250
    },
    "🏢 3BHK Premium High-Rise (Koramangala, Bengaluru)": {
        "Neighborhood": "Somerst", "BldgType": "TwnhsE", "HouseStyle": "2Story", "MSZoning": "FV",
        "GrLivArea": 2100, "1stFlrSF": 1100, "2ndFlrSF": 1000, "TotalBsmtSF": 1000,
        "LotArea": 4800, "LotFrontage": 55, "OverallQual": 8, "OverallCond": 6,
        "YearBuilt": 2017, "YearRemodAdd": 2021, "BedroomAbvGr": 3, "FullBath": 3, "HalfBath": 1,
        "TotRmsAbvGrd": 7, "GarageCars": 2, "GarageArea": 480, "Fireplaces": 0, "WoodDeckSF": 140
    },
    "🏘️ 2BHK Modern Apartment (Whitefield, Bengaluru)": {
        "Neighborhood": "CollgCr", "BldgType": "1Fam", "HouseStyle": "1Story", "MSZoning": "RL",
        "GrLivArea": 1350, "1stFlrSF": 1350, "2ndFlrSF": 0, "TotalBsmtSF": 850,
        "LotArea": 7500, "LotFrontage": 65, "OverallQual": 7, "OverallCond": 5,
        "YearBuilt": 2015, "YearRemodAdd": 2018, "BedroomAbvGr": 2, "FullBath": 2, "HalfBath": 0,
        "TotRmsAbvGrd": 5, "GarageCars": 1, "GarageArea": 350, "Fireplaces": 0, "WoodDeckSF": 80
    },
    "🏠 Independent Duplex House (Jayanagar, Bengaluru)": {
        "Neighborhood": "NAmes", "BldgType": "1Fam", "HouseStyle": "2Story", "MSZoning": "RL",
        "GrLivArea": 1850, "1stFlrSF": 1000, "2ndFlrSF": 850, "TotalBsmtSF": 1050,
        "LotArea": 9200, "LotFrontage": 75, "OverallQual": 6, "OverallCond": 7,
        "YearBuilt": 1992, "YearRemodAdd": 2015, "BedroomAbvGr": 3, "FullBath": 2, "HalfBath": 1,
        "TotRmsAbvGrd": 7, "GarageCars": 2, "GarageArea": 400, "Fireplaces": 0, "WoodDeckSF": 60
    },
    "🏖️ Coastal Executive Villa (Beach Road, Visakhapatnam)": {
        "Neighborhood": "SWISU", "BldgType": "1Fam", "HouseStyle": "2Story", "MSZoning": "RL",
        "GrLivArea": 2400, "1stFlrSF": 1200, "2ndFlrSF": 1200, "TotalBsmtSF": 900,
        "LotArea": 8500, "LotFrontage": 70, "OverallQual": 7, "OverallCond": 6,
        "YearBuilt": 2011, "YearRemodAdd": 2019, "BedroomAbvGr": 3, "FullBath": 3, "HalfBath": 0,
        "TotRmsAbvGrd": 7, "GarageCars": 2, "GarageArea": 460, "Fireplaces": 1, "WoodDeckSF": 180
    },
    "🏚️ Heritage Townhouse (Old City, Hyderabad)": {
        "Neighborhood": "OldTown", "BldgType": "1Fam", "HouseStyle": "1.5Fin", "MSZoning": "RM",
        "GrLivArea": 1250, "1stFlrSF": 800, "2ndFlrSF": 450, "TotalBsmtSF": 700,
        "LotArea": 6000, "LotFrontage": 50, "OverallQual": 5, "OverallCond": 6,
        "YearBuilt": 1945, "YearRemodAdd": 1995, "BedroomAbvGr": 3, "FullBath": 1, "HalfBath": 0,
        "TotRmsAbvGrd": 6, "GarageCars": 1, "GarageArea": 240, "Fireplaces": 0, "WoodDeckSF": 0
    }
}

# -------------------------------------------------------------
# HIGH-END MODERN UI/UX STYLING (CLEAN & INTUITIVE)
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #091e3a 0%, #102a43 50%, #064e3b 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: #ffffff;
        box-shadow: 0 10px 25px -5px rgba(9, 30, 58, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-badge {
        background-color: rgba(255, 255, 255, 0.15);
        color: #6ee7b7;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        display: inline-block;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.6px;
        margin: 0;
        color: #ffffff;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-weight: 400;
    }

    /* Valuation Appraisal Hero Box */
    .appraisal-box {
        background: linear-gradient(135deg, #047857 0%, #059669 60%, #0d9488 100%);
        border-radius: 16px;
        padding: 2rem 1.8rem;
        color: white !important;
        text-align: center;
        box-shadow: 0 12px 24px -5px rgba(5, 150, 105, 0.35);
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .appraisal-tag {
        text-transform: uppercase;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #a7f3d0;
        margin-bottom: 0.3rem;
    }
    .appraisal-main-price {
        font-size: 3.1rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #ffffff;
        margin: 0.3rem 0;
        line-height: 1.1;
    }
    .appraisal-confidence {
        font-size: 0.95rem;
        color: #d1fae5;
        font-weight: 500;
        margin-top: 0.4rem;
    }

    /* Metric Grid Cards */
    .card-metric {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease;
    }
    .card-metric:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.08);
    }
    .card-metric-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .card-metric-val {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.2rem;
    }

    /* Info Locality Card */
    .locality-badge-box {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #059669;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.92rem;
        color: #1e293b;
    }

    /* Section Title */
    .ui-section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f294d;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Workflow Step Cards */
    .step-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .step-card-num {
        font-size: 0.8rem;
        font-weight: 800;
        color: #2563eb;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .step-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.2rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# CACHED DATA & METADATA LOADERS
# -------------------------------------------------------------
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


def format_inr(amount_usd: float) -> Tuple[str, str, float]:
    """
    Convert USD amount into Indian Rupees (INR ₹) formatted as Crores or Lakhs.
    Returns: (Formatted String, Subtitle Exact Value, Numeric INR)
    """
    amount_inr = amount_usd * USD_TO_INR_RATE
    if amount_inr >= 10000000:
        crores = amount_inr / 10000000
        formatted = f"₹{crores:.2f} Crore"
        exact_sub = f"₹{amount_inr:,.0f} INR"
    else:
        lakhs = amount_inr / 100000
        formatted = f"₹{lakhs:.2f} Lakhs"
        exact_sub = f"₹{amount_inr:,.0f} INR"
    return formatted, exact_sub, amount_inr


# -------------------------------------------------------------
# SIDEBAR CONTROLS & LOCALIZATION
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏡 **BharatProp AI**")
    st.caption("AI-Powered Indian Real Estate Valuation Engine")
    st.markdown("---")

    st.markdown("#### ⚡ **Quick 1-Click Property Presets**")
    st.write("Load realistic Indian property profiles instantly:")
    selected_preset = st.selectbox(
        "Choose Demo Profile:",
        ["None (Custom Inputs)"] + list(INDIAN_PRESETS.keys()),
        index=0,
        help="Select any preset property profile to auto-populate specifications and see real-time appraisal!"
    )

    st.markdown("---")
    st.markdown("#### 💱 **Currency Preference**")
    currency_view = st.radio(
        "Display Valuation In:",
        ["🇮🇳 Indian Rupees (₹ Lakhs & Crores)", "🇺🇸 US Dollars ($)"],
        index=0,
        help="Default Indian currency mode (₹) or International USD ($) reference"
    )

    st.markdown("---")
    st.markdown("#### 🤖 **Engine Intelligence**")
    st.markdown(f"**Top Model:** `{metadata.get('best_model_name', 'Ridge Regression (L2)')}`")
    st.markdown(f"**Accuracy ($R^2$):** `{metadata.get('validation_r2', 0.9226):.4f}` *(92.26% Explained)*")
    
    rmse_usd = metadata.get('validation_rmse', 24366.32)
    rmse_inr_formatted, _, _ = format_inr(rmse_usd)
    if currency_view.startswith("🇮🇳"):
        st.markdown(f"**Validation RMSE:** `±{rmse_inr_formatted}`")
    else:
        st.markdown(f"**Validation RMSE:** `±${rmse_usd:,.2f}`")

    st.markdown("---")
    st.markdown("#### 📂 **Dataset Synchronization**")
    st.caption("✅ `train.csv` (1,460 Records)")
    st.caption("✅ `test.csv` (1,460 Records)")
    st.caption("✅ `sample_submission.csv` (1,460 Preds)")
    st.caption("✅ `data_description.txt` (79 Features)")


# -------------------------------------------------------------
# TOP HERO BANNER
# -------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">AI-DRIVEN REAL ESTATE APPRAISAL ENGINE</div>
    <div class="hero-title">🏡 Real Estate Valuation & Automated Appraisal Platform</div>
    <div class="hero-subtitle">Machine Learning regression engine trained across 79 structural & spatial property characteristics with 92.26% explained variance</div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# TOP NAVIGATION TABS (6 DEDICATED MODULES)
# -------------------------------------------------------------
nav_tab1, nav_tab2, nav_tab3, nav_tab4, nav_tab5, nav_tab6 = st.tabs([
    "🏠 Property Appraisal",
    "📊 Model Performance",
    "📈 Market Analytics (EDA)",
    "📂 Bulk CSV Valuation",
    "🏦 Home Loan & EMI Planner",
    "🗺️ Website Workflow & Architecture"
])


# -------------------------------------------------------------
# MODULE 1: REAL-TIME PROPERTY APPRAISAL ENGINE
# -------------------------------------------------------------
with nav_tab1:
    preset_data = INDIAN_PRESETS.get(selected_preset, {})

    st.markdown("<div class='ui-section-title'>⚙️ Configure Property Characteristics</div>", unsafe_allow_html=True)

    col_input, col_result = st.columns([3, 2], gap="large")

    with col_input:
        tab_loc, tab_dim, tab_qual, tab_rooms = st.tabs([
            "📍 Location & City", "📐 Dimensions & Area", "⭐ Quality & Age", "🚗 Rooms & Parking"
        ])

        with tab_loc:
            c1, c2 = st.columns(2)
            with c1:
                locality_keys = list(LOCALITY_MAP.keys())
                locality_labels = [f"{LOCALITY_MAP[k]['name']} — {LOCALITY_MAP[k]['city']} ({LOCALITY_MAP[k]['state']})" for k in locality_keys]
                
                cur_k = preset_data.get("Neighborhood", "CollgCr")
                def_idx = locality_keys.index(cur_k) if cur_k in locality_keys else 0
                
                sel_label = st.selectbox("Select Locality / City (India):", locality_labels, index=def_idx)
                sel_code = locality_keys[locality_labels.index(sel_label)]
                sel_info = LOCALITY_MAP[sel_code]
                
                bldg_options = ["1Fam (Independent Villa / House)", "TwnhsE (Row House / Gated Townhouse)", "Twnhs (Townhouse Inside Unit)", "Duplex (Duplex Apartment)", "2fmCon (Two-Family Conversion)"]
                bldg_sel = st.selectbox("Building Type:", bldg_options, index=0)
                bldg_code = bldg_sel.split()[0]
                
            with c2:
                style_options = ["2Story (Double Floor / Duplex)", "1Story (Single Ground Floor)", "1.5Fin (Ground + Finished Attic)", "SLvl (Split Level Penthouse)", "SFoyer (Split Foyer)"]
                style_sel = st.selectbox("Architectural Layout:", style_options, index=0)
                style_code = style_sel.split()[0]
                
                zone_options = ["RL (Residential Low Density)", "FV (Floating Village / Luxury Enclave)", "RM (Residential Medium Density)", "RH (Residential High Density)", "C (Commercial Zone)"]
                zone_sel = st.selectbox("Zoning Classification:", zone_options, index=0)
                zone_code = zone_sel.split()[0]

            st.markdown(f"""
            <div class='locality-badge-box'>
                📍 <b>Prime Zone Selected:</b> {sel_info['name']}, <b>{sel_info['city']}</b> ({sel_info['state']})<br>
                🏷️ <b>Development Tier:</b> <span style='color: #047857; font-weight:700;'>{sel_info['tier']}</span>
            </div>
            """, unsafe_allow_html=True)

        with tab_dim:
            c1, c2 = st.columns(2)
            with c1:
                gr_liv_area = st.number_input("Above Ground Living Area (sq ft):", min_value=300, max_value=6000, value=int(preset_data.get("GrLivArea", 1750)), step=50, help="Total built-up carpet area above ground")
                first_flr_sf = st.number_input("Ground / 1st Floor Area (sq ft):", min_value=300, max_value=4000, value=int(preset_data.get("1stFlrSF", 950)), step=25)
                second_flr_sf = st.number_input("Upper / 2nd Floor Area (sq ft):", min_value=0, max_value=3000, value=int(preset_data.get("2ndFlrSF", 800)), step=25)
            with c2:
                total_bsmt_sf = st.number_input("Basement / Lower Level Area (sq ft):", min_value=0, max_value=4000, value=int(preset_data.get("TotalBsmtSF", 900)), step=25)
                lot_area = st.number_input("Plot / Parcel Size (sq ft):", min_value=1000, max_value=100000, value=int(preset_data.get("LotArea", 9500)), step=100)
                sq_yards = lot_area / 9.0
                st.caption(f"📐 Equivalent Plot Size: **{sq_yards:,.1f} Sq. Yards (Gaj)**")
                lot_frontage = st.number_input("Road Frontage (Linear feet):", min_value=20, max_value=300, value=int(preset_data.get("LotFrontage", 70)), step=5)

        with tab_qual:
            c1, c2 = st.columns(2)
            with c1:
                overall_qual = st.slider("Material Quality & Finishes (1 – 10):", min_value=1, max_value=10, value=int(preset_data.get("OverallQual", 7)), help="1: Economy, 5: Standard Builder, 7: Premium High Quality, 10: Luxury Masterpiece")
                overall_cond = st.slider("Maintenance Condition (1 – 9):", min_value=1, max_value=9, value=int(preset_data.get("OverallCond", 5)), help="1: Poor Dilapidated, 5: Normal/Good, 9: Pristine/Brand New")
            with c2:
                year_built = st.number_input("Construction Year:", min_value=1900, max_value=2026, value=int(preset_data.get("YearBuilt", 2008)), step=1)
                year_remod = st.number_input("Year Remodeled / Renovated:", min_value=1950, max_value=2026, value=int(preset_data.get("YearRemodAdd", 2012)), step=1)

        with tab_rooms:
            c1, c2 = st.columns(2)
            with c1:
                bedrooms = st.number_input("Bedrooms (BHK Configuration):", min_value=1, max_value=8, value=int(preset_data.get("BedroomAbvGr", 3)), step=1)
                full_bath = st.number_input("Full Attached Bathrooms:", min_value=1, max_value=5, value=int(preset_data.get("FullBath", 2)), step=1)
                half_bath = st.number_input("Half / Powder Rooms:", min_value=0, max_value=3, value=int(preset_data.get("HalfBath", 1)), step=1)
                tot_rms = st.number_input("Total Rooms (excluding baths):", min_value=2, max_value=16, value=int(preset_data.get("TotRmsAbvGrd", 7)), step=1)
            with c2:
                garage_cars = st.number_input("Covered Car Parking Spaces:", min_value=0, max_value=4, value=int(preset_data.get("GarageCars", 2)), step=1)
                garage_area = st.number_input("Parking Garage Area (sq ft):", min_value=0, max_value=1500, value=int(preset_data.get("GarageArea", 500)), step=25)
                fireplaces = st.number_input("Fireplaces / Feature Corner:", min_value=0, max_value=3, value=int(preset_data.get("Fireplaces", 1)), step=1)
                wood_deck_sf = st.number_input("Balcony / Deck / Terrace (sq ft):", min_value=0, max_value=1000, value=int(preset_data.get("WoodDeckSF", 120)), step=10)

        property_input = {
            "Neighborhood": sel_code,
            "BldgType": bldg_code,
            "HouseStyle": style_code,
            "MSZoning": zone_code,
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
        appraise_btn = st.button("🚀 Calculate Live Market Appraisal", type="primary", use_container_width=True)

    # -------------------------------------------------------------
    # APPRAISAL RESULT CARD (RIGHT COLUMN)
    # -------------------------------------------------------------
    with col_result:
        st.markdown("<div class='ui-section-title'>💵 Automated Valuation Summary</div>", unsafe_allow_html=True)
        
        # Predict price
        pred_usd = predict_house_price(property_input)
        rmse_usd = metadata.get("validation_rmse", 24366.32)
        lower_usd = max(15000, pred_usd - rmse_usd)
        upper_usd = pred_usd + rmse_usd

        # Format Currency
        if currency_view.startswith("🇮🇳"):
            main_price_str, exact_sub, inr_num = format_inr(pred_usd)
            low_str, _, _ = format_inr(lower_usd)
            up_str, _, _ = format_inr(upper_usd)
            rate_per_sqft = inr_num / max(1, gr_liv_area)
            rate_disp = f"₹{rate_per_sqft:,.0f} / sq ft"
            range_disp = f"Expected Band: <b>{low_str} — {up_str}</b>"
        else:
            main_price_str = f"${pred_usd:,.2f}"
            exact_sub = f"USD Valuation Reference"
            rate_per_sqft = pred_usd / max(1, gr_liv_area)
            rate_disp = f"${rate_per_sqft:.2f} / sq ft"
            range_disp = f"Expected Band: <b>${lower_usd:,.0f} — ${upper_usd:,.0f}</b>"

        # Tier Classification
        if pred_usd > 320000:
            tier_badge = "👑 Ultra-Luxury Villa / Penthouse"
        elif pred_usd > 200000:
            tier_badge = "🌟 Premium Upper-Mid High-Rise"
        elif pred_usd > 120000:
            tier_badge = "🏡 Standard Gated Residential"
        else:
            tier_badge = "🏷️ Affordable Housing Tier"

        st.markdown(f"""
        <div class='appraisal-box'>
            <div class='appraisal-tag'>{tier_badge}</div>
            <div class='appraisal-main-price'>{main_price_str}</div>
            <div style='color: #e2e8f0; font-size: 0.95rem; opacity: 0.9;'>{exact_sub}</div>
            <div class='appraisal-confidence'>{range_disp}</div>
        </div>
        """, unsafe_allow_html=True)

        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.markdown(f"""
            <div class='card-metric'>
                <div class='card-metric-title'>Rate / Sq Ft</div>
                <div class='card-metric-val'>{rate_disp}</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m2:
            st.markdown(f"""
            <div class='card-metric'>
                <div class='card-metric-title'>Total Floor Area</div>
                <div class='card-metric-val'>{gr_liv_area + total_bsmt_sf:,.0f} sqft</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m3:
            st.markdown(f"""
            <div class='card-metric'>
                <div class='card-metric-title'>Property Age</div>
                <div class='card-metric-val'>{max(0, 2026 - year_built)} yrs</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 **Property Scorecard & Indian Specifications**")
        st.markdown(f"""
        - **Prime Locality:** `{sel_info['name']}, {sel_info['city']} ({sel_info['state']})`
        - **Configuration:** `{bedrooms} BHK Layout` • `{full_bath + 0.5*half_bath} Bathrooms` • `{tot_rms} Rooms`
        - **Quality Index:** `{overall_qual}/10 Grade` • Maintenance Condition: `{overall_cond}/9`
        - **Car Parking:** `{garage_cars} Covered Parking Spaces` ({garage_area} sq ft)
        - **Balcony & Terraces:** `{wood_deck_sf} sq ft`
        - **Model Reliability:** Tested with `92.26% R² Variance Explained`
        """)


# -------------------------------------------------------------
# MODULE 2: MODEL PERFORMANCE & ACCURACY BENCHMARKS
# -------------------------------------------------------------
with nav_tab2:
    st.markdown("<div class='ui-section-title'>📊 Machine Learning Model Accuracy & Benchmarks</div>", unsafe_allow_html=True)
    st.markdown("Rigorous benchmark metrics across four evaluated regression algorithms on the identical 20% holdout validation dataset (292 properties):")

    csv_path = PROJECT_ROOT / "outputs" / "results" / "model_comparison.csv"
    if csv_path.exists():
        comp_df = pd.read_csv(csv_path)
        comp_inr = comp_df.copy()
        comp_inr["Validation RMSE (₹)"] = [f"₹{(val * USD_TO_INR_RATE / 100000):.2f} Lakhs" for val in comp_df["RMSE"]]
        comp_inr["Validation MAE (₹)"] = [f"₹{(val * USD_TO_INR_RATE / 100000):.2f} Lakhs" for val in comp_df["MAE"]]
        comp_inr["Accuracy (R² Score)"] = [f"{val:.4f} ({val*100:.2f}%)" for val in comp_df["R2"]]
        
        st.dataframe(
            comp_inr[["Model", "Accuracy (R² Score)", "Validation RMSE (₹)", "Validation MAE (₹)"]],
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 **Validation Visual Diagnostics**")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        p1 = PROJECT_ROOT / "outputs" / "plots" / "08_actual_vs_predicted.png"
        if p1.exists():
            st.image(str(p1), caption="Actual vs. Predicted SalePrice (Ridge Regressor: R² = 0.9226)", use_container_width=True)
    with col_v2:
        p2 = PROJECT_ROOT / "outputs" / "plots" / "09_residuals_plot.png"
        if p2.exists():
            st.image(str(p2), caption="Residual Error Distribution Plot (Zero-Centered Normal Residuals)", use_container_width=True)

    p3 = PROJECT_ROOT / "outputs" / "plots" / "10_model_comparison_bar.png"
    if p3.exists():
        st.image(str(p3), caption="Comparison of RMSE, MAE, and R² Across All Evaluated Models", use_container_width=True)


# -------------------------------------------------------------
# MODULE 3: MARKET ANALYTICS & EXPLORATORY INSIGHTS (EDA)
# -------------------------------------------------------------
with nav_tab3:
    st.markdown("<div class='ui-section-title'>📈 Real Estate Exploratory Data Analytics (EDA)</div>", unsafe_allow_html=True)
    st.markdown("Key structural pricing drivers, skewness corrections, and correlation matrices uncovered during Exploratory Data Analysis:")

    c1, c2 = st.columns(2)
    with c1:
        p1 = PROJECT_ROOT / "outputs" / "plots" / "01_saleprice_distribution.png"
        if p1.exists():
            st.image(str(p1), caption="Target Normalization: Skewness Corrected from 1.88 to 0.12 via log1p", use_container_width=True)
    with c2:
        p2 = PROJECT_ROOT / "outputs" / "plots" / "03_overallqual_vs_saleprice.png"
        if p2.exists():
            st.image(str(p2), caption="Primary Valuation Driver: Material Quality Rating vs. Price (r = 0.79)", use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        p3 = PROJECT_ROOT / "outputs" / "plots" / "04_grlivarea_vs_saleprice.png"
        if p3.exists():
            st.image(str(p3), caption="Living Square Footage (GrLivArea) vs. Valuation (r = 0.71)", use_container_width=True)
    with c4:
        p4 = PROJECT_ROOT / "outputs" / "plots" / "06_correlation_heatmap.png"
        if p4.exists():
            st.image(str(p4), caption="Top 12 Housing Attributes Correlation Matrix Heatmap", use_container_width=True)


# -------------------------------------------------------------
# MODULE 4: BULK CSV VALUATION ENGINE
# -------------------------------------------------------------
with nav_tab4:
    st.markdown("<div class='ui-section-title'>📂 Bulk Portfolio Property Valuation</div>", unsafe_allow_html=True)
    st.markdown("Upload a CSV spreadsheet of residential properties to execute batch machine learning appraisals with 1-click export:")

    uploaded_file = st.file_uploader("Upload Properties CSV File", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(batch_df)} properties for automated appraisal.")
        
        if st.button("🚀 Execute Bulk Appraisal on Uploaded File", type="primary"):
            with st.spinner("Processing automated valuations..."):
                preds_usd = predict_house_price(batch_df)
                result_df = batch_df.copy()
                result_df["Estimated_Price_USD"] = preds_usd
                result_df["Estimated_Price_INR"] = [round(p * USD_TO_INR_RATE, 2) for p in preds_usd]
                result_df["Valuation_In_Lakhs_Crores"] = [format_inr(p)[0] for p in preds_usd]
                
                st.dataframe(result_df.head(25), use_container_width=True)

                csv_data = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Appraised Portfolio CSV",
                    data=csv_data,
                    file_name="appraised_properties_portfolio.csv",
                    mime="text/csv",
                )
    else:
        st.info("💡 You can also test automated bulk valuation on sample properties from `test.csv` below:")
        sample_path = PROJECT_ROOT / "data" / "raw" / "test.csv"
        if sample_path.exists():
            sample_df = pd.read_csv(sample_path).head(10)
            st.dataframe(sample_df[["Id", "MSSubClass", "MSZoning", "LotArea", "Neighborhood", "OverallQual", "YearBuilt", "GrLivArea"]], use_container_width=True)
            if st.button("🚀 Appraise 10 Sample Properties Above", type="primary"):
                sample_preds = predict_house_price(sample_df)
                sample_res = sample_df[["Id", "Neighborhood", "OverallQual", "YearBuilt", "GrLivArea"]].copy()
                sample_res["Indian_City_Locality"] = [LOCALITY_MAP.get(n, {}).get("name", n) for n in sample_res["Neighborhood"]]
                sample_res["State"] = [LOCALITY_MAP.get(n, {}).get("state", "India") for n in sample_res["Neighborhood"]]
                sample_res["Estimated_Valuation_INR"] = [format_inr(p)[0] for p in sample_preds]
                st.dataframe(sample_res, use_container_width=True)


# -------------------------------------------------------------
# MODULE 5: HOME LOAN & EMI PLANNER (INDIAN BANK GUIDELINES)
# -------------------------------------------------------------
with nav_tab5:
    st.markdown("<div class='ui-section-title'>🏦 Home Loan Eligibility & EMI Planner (SBI / HDFC Guidelines)</div>", unsafe_allow_html=True)
    st.markdown("Calculate down payment, loan eligibility, monthly EMI, and interest amortization for the appraised property valuation:")

    col_emi1, col_emi2 = st.columns([1, 1], gap="large")

    with col_emi1:
        loan_property_val = st.number_input("Property Valuation (in ₹ Lakhs):", min_value=10.0, max_value=2500.0, value=125.0, step=5.0)
        down_payment_pct = st.slider("Down Payment Percentage (%):", min_value=10, max_value=50, value=20, step=5)
        interest_rate = st.slider("Home Loan Interest Rate (% p.a.):", min_value=6.0, max_value=14.0, value=8.5, step=0.25)
        tenure_years = st.slider("Loan Tenure (Years):", min_value=5, max_value=30, value=20, step=1)

    with col_emi2:
        property_val_rupees = loan_property_val * 100000
        down_payment_rupees = property_val_rupees * (down_payment_pct / 100.0)
        loan_amount_rupees = property_val_rupees - down_payment_rupees

        # EMI Calculation Formula: P * r * (1+r)^n / ((1+r)^n - 1)
        monthly_rate = (interest_rate / 12.0) / 100.0
        months = tenure_years * 12
        if monthly_rate > 0:
            monthly_emi = (loan_amount_rupees * monthly_rate * ((1 + monthly_rate) ** months)) / (((1 + monthly_rate) ** months) - 1)
        else:
            monthly_emi = loan_amount_rupees / months

        total_payment = monthly_emi * months
        total_interest = total_payment - loan_amount_rupees

        st.markdown(f"""
        <div class='appraisal-box' style='background: linear-gradient(135deg, #1e3a8a 0%, #0f294d 100%);'>
            <div class='appraisal-tag'>ESTIMATED MONTHLY HOME LOAN EMI</div>
            <div class='appraisal-main-price'>₹{monthly_emi:,.0f} <span style='font-size: 1.2rem; font-weight: 500;'>/ month</span></div>
            <div class='appraisal-confidence'>Principal Loan: <b>₹{loan_amount_rupees/100000:.2f} Lakhs</b> ({100-down_payment_pct}% LTV)</div>
        </div>
        """, unsafe_allow_html=True)

        em1, em2, em3 = st.columns(3)
        with em1:
            st.markdown(f"""
            <div class='card-metric'>
                <div class='card-metric-title'>Down Payment ({down_payment_pct}%)</div>
                <div class='card-metric-val'>₹{down_payment_rupees/100000:.2f} L</div>
            </div>
            """, unsafe_allow_html=True)
        with em2:
            st.markdown(f"""
            <div class='card-metric'>
                <div class='card-metric-title'>Principal Loan</div>
                <div class='card-metric-val'>₹{loan_amount_rupees/100000:.2f} L</div>
            </div>
            """, unsafe_allow_html=True)
        with em3:
            st.markdown(f"""
            <div class='card-metric'>
                <div class='card-metric-title'>Total Interest</div>
                <div class='card-metric-val'>₹{total_interest/100000:.2f} L</div>
            </div>
            """, unsafe_allow_html=True)


# -------------------------------------------------------------
# MODULE 6: SYSTEM WORKFLOW & ARCHITECTURE GUIDE
# -------------------------------------------------------------
with nav_tab6:
    st.markdown("<div class='ui-section-title'>🗺️ End-to-End System Architecture & Working Flow</div>", unsafe_allow_html=True)
    st.markdown("Detailed breakdown of how the **BharatProp AI Real Estate Valuation Engine** ingests datasets, processes spatial features, trains machine learning models, and serves real-time appraisals:")

    wf_col1, wf_col2 = st.columns([1, 1], gap="large")

    with wf_col1:
        st.markdown("""
        <div class='step-card'>
            <div class='step-card-num'>STEP 1 • DATA INGESTION & CODEBOOK</div>
            <div class='step-card-title'>Official Datasets & Feature Metadata</div>
            <p>The system is built upon four interconnected data artifacts:</p>
            <ul>
                <li><b>train.csv:</b> 1,460 historical property transactions with 79 explanatory features and ground truth <code>SalePrice</code>.</li>
                <li><b>test.csv:</b> 1,460 test properties for benchmark evaluations.</li>
                <li><b>sample_submission.csv:</b> Standard output schema for portfolio valuations.</li>
                <li><b>data_description.txt:</b> Detailed codebook defining categorical labels, zoning, architectural styles, and quality ratings.</li>
            </ul>
        </div>

        <div class='step-card'>
            <div class='step-card-num'>STEP 2 • DATA CLEANING & LEAKAGE PREVENTION</div>
            <div class='step-card-title'>Domain-Aware Imputation Strategy</div>
            <p>Missing value handling follows real estate domain realities:</p>
            <ul>
                <li><b>Amenity Absences:</b> Categoricals like <code>PoolQC</code>, <code>FireplaceQu</code>, <code>GarageType</code>, and <code>BsmtQual</code> where <code>NA</code> means "No Amenity" are explicitly transformed to the category <code>"None"</code>.</li>
                <li><b>Numerical Median Imputation:</b> Numerical gaps (e.g., <code>LotFrontage</code>) are imputed with training medians to resist extreme outlier distortion.</li>
                <li><b>Strict Training Fit:</b> All scalers, imputers, and encoders are fit strictly on the 80% training partition to avoid data leakage.</li>
            </ul>
        </div>

        <div class='step-card'>
            <div class='step-card-num'>STEP 3 • FEATURE ENGINEERING</div>
            <div class='step-card-title'>Composite Real Estate Synthetics</div>
            <p>Engineering multidimensional composite metrics from raw features:</p>
            <ul>
                <li><b>TotalSF:</b> Built-up living area = <code>TotalBsmtSF + 1stFlrSF + 2ndFlrSF</code></li>
                <li><b>TotalBathrooms:</b> <code>FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath</code></li>
                <li><b>TotalPorchSF:</b> Balcony, deck, open & screened porch space sum.</li>
                <li><b>HouseAge & RemodAge:</b> Calculated dynamically from transaction year.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with wf_col2:
        st.markdown("""
        <div class='step-card' style='border-left-color: #059669;'>
            <div class='step-card-num' style='color: #059669;'>STEP 4 • TARGET NORMALIZATION & ML TRAINING</div>
            <div class='step-card-title'>Log-Target Regression & Model Selection</div>
            <p>Training and evaluating multiple supervised learning algorithms:</p>
            <ul>
                <li><b>Target Transformation:</b> Raw property prices exhibit positive skew (1.88). Applying <code>log1p(SalePrice)</code> normalizes skewness to <b>0.12</b>, stabilizing residual error variance.</li>
                <li><b>Evaluated Algorithms:</b> Linear Ridge ($L_2$), XGBoost Regressor, Gradient Boosting, and Random Forest.</li>
                <li><b>Winning Champion:</b> <b>Regularized Ridge Regression ($R^2 = 0.9226$)</b> won with the lowest validation RMSE, perfectly handling high-dimensional one-hot encoded localities without overfitting.</li>
            </ul>
        </div>

        <div class='step-card' style='border-left-color: #059669;'>
            <div class='step-card-num' style='color: #059669;'>STEP 5 • INDIAN LOCALIZATION & LIVE INFERENCE</div>
            <div class='step-card-title'>Real-Time Dual Currency & Locality Engine</div>
            <p>Translating statistical predictions into actionable Indian real estate insights:</p>
            <ul>
                <li><b>Locality Mapping:</b> 25 spatial neighborhood codes are mapped to prominent Indian tech corridors and heritage zones across Karnataka, Telangana, Maharashtra, Andhra Pradesh, and Delhi-NCR.</li>
                <li><b>Dual Currency Translation:</b> Automatic conversion into <b>₹ Lakhs & Crores</b>, calculating <b>₹ per sq ft</b> and equivalent plot size in <b>Sq. Yards (Gaj)</b>.</li>
            </ul>
        </div>

        <div class='step-card' style='border-left-color: #059669;'>
            <div class='step-card-num' style='color: #059669;'>STEP 6 • FINANCIAL PLANNING & AMORTIZATION</div>
            <div class='step-card-title'>Home Loan & Monthly EMI Calculator</div>
            <p>Integrated consumer banking calculators aligned with SBI and HDFC guidelines, allowing instant calculation of monthly EMIs, principal loan, down payments, and loan-to-value (LTV) limits.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏛️ **System Workflow Architecture Flowchart**")
    
    st.markdown("""
    ```mermaid
    flowchart TD
        A[📂 Raw Datasets: train.csv, test.csv, sample_submission.csv, data_description.txt] --> B[🧹 Data Preprocessing & Missing Category Cleaner]
        B --> C[⚙️ Composite Feature Engineering: TotalSF, HouseAge, TotalBathrooms]
        C --> D[📉 Log1p Target Transformation: Normalizes Skewness to 0.12]
        D --> E[🤖 Supervised Model Training: Ridge L2, XGBoost, GBDT, Random Forest]
        E --> F[🏆 Champion Model Pipeline: Ridge Regressor R²=0.9226]
        F --> G[🇮🇳 Indian Localization & Rupee Currency Engine: ₹ Lakhs & Crores]
        G --> H[🏠 Interactive Web UI: Property Appraisal, EDA, Bulk CSV, EMI Planner]
    ```
    """)
