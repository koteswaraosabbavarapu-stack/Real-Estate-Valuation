"""
Generate a professional, high-impact 14-slide PowerPoint presentation (.pptx)
for the BharatProp AI Indian Real Estate Valuation project.
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PPTX_PATH = PROJECT_ROOT / "BharatProp_AI_Presentation.pptx"

# Color Palette: Corporate PropTech Theme
COLOR_BG_DARK = RGBColor(15, 23, 42)       # Slate 900
COLOR_PRIMARY = RGBColor(6, 78, 59)        # Emerald 900
COLOR_ACCENT = RGBColor(16, 185, 129)      # Emerald 500
COLOR_ACCENT_BLUE = RGBColor(2, 132, 199)  # Sky 600
COLOR_TEXT_DARK = RGBColor(15, 23, 42)     # Slate 900
COLOR_TEXT_MUTED = RGBColor(100, 116, 139) # Slate 500
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_CARD_BG = RGBColor(248, 250, 252)    # Slate 50
COLOR_CARD_BORDER = RGBColor(226, 232, 240) # Slate 200

def create_presentation():
    prs = Presentation()
    # 16:9 Widescreen slides
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # Blank layout

    def add_header(slide, title_text, category_text="BHARATPROP AI • INDIAN REAL ESTATE AVM"):
        # Header banner
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_c = cat_box.text_frame
        tf_c.word_wrap = True
        p_c = tf_c.paragraphs[0]
        p_c.text = category_text.upper()
        p_c.font.size = Pt(11)
        p_c.font.bold = True
        p_c.font.color.rgb = COLOR_ACCENT

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.7))
        tf_t = title_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(24)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_TEXT_DARK

    def add_card(slide, left, top, width, height, title="", bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)
        
        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.2), width - Inches(0.5), Inches(0.45))
            tf = tb.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(15)
            p.font.bold = True
            p.font.color.rgb = COLOR_PRIMARY
        return shape

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (Dark Hero Theme)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_BG_DARK
    bg1.line.fill.background()

    # Title Box
    tbox = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.0))
    tf1 = tbox.text_frame
    tf1.word_wrap = True
    
    p_badge = tf1.paragraphs[0]
    p_badge.text = "PROPECH • MACHINE LEARNING • CONFORMAL PREDICTION"
    p_badge.font.size = Pt(13)
    p_badge.font.bold = True
    p_badge.font.color.rgb = COLOR_ACCENT

    p_title = tf1.add_paragraph()
    p_title.text = "AI-Based Indian Real Estate Price Prediction\n& Automated Property Valuation System"
    p_title.font.size = Pt(32)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_WHITE
    p_title.space_before = Pt(14)
    p_title.space_after = Pt(14)

    p_sub = tf1.add_paragraph()
    p_sub.text = "BharatProp AI — Automated Valuation Model (AVM) across 6 Major Metropolitan Cities\nPowered by XGBoost, 5-Fold Cross Validation, and Inductive Split Conformal Uncertainty Estimation."
    p_sub.font.size = Pt(16)
    p_sub.font.color.rgb = RGBColor(203, 213, 225) # Slate 300

    p_meta = tf1.add_paragraph()
    p_meta.text = "\nLive App: real-estate-valuation-kotesh.streamlit.app  •  Author: Koteswarao Sabbavarapu"
    p_meta.font.size = Pt(13)
    p_meta.font.color.rgb = COLOR_ACCENT
    p_meta.font.bold = True

    # =========================================================================
    # SLIDE 2: EXECUTIVE SUMMARY & PROJECT VISION
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "Executive Summary & Project Vision")

    # 3 Stat Cards
    c1 = add_card(slide2, Inches(0.8), Inches(1.6), Inches(3.6), Inches(2.2))
    tb1 = slide2.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(3.2), Inches(1.8))
    tf = tb1.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "AUTHENTIC INDIAN DATA"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = COLOR_ACCENT_BLUE
    p2 = tf.add_paragraph()
    p2.text = "Replaced foreign datasets with 27,614 verified residential properties across Mumbai, Bangalore, Delhi-NCR, Hyderabad, Chennai & Kolkata."
    p2.font.size = Pt(13); p2.font.color.rgb = COLOR_TEXT_DARK; p2.space_before = Pt(8)

    c2 = add_card(slide2, Inches(4.8), Inches(1.6), Inches(3.6), Inches(2.2))
    tb2 = slide2.shapes.add_textbox(Inches(5.0), Inches(1.8), Inches(3.2), Inches(1.8))
    tf = tb2.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "PRODUCTION ML PIPELINE"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = COLOR_PRIMARY
    p2 = tf.add_paragraph()
    p2.text = "Trained Ridge, Random Forest, GBR & XGBoost with log-target transforms and continuous micro-market target encoding. Best Test MAE: ₹41.89 L."
    p2.font.size = Pt(13); p2.font.color.rgb = COLOR_TEXT_DARK; p2.space_before = Pt(8)

    c3 = add_card(slide2, Inches(8.8), Inches(1.6), Inches(3.7), Inches(2.2))
    tb3 = slide2.shapes.add_textbox(Inches(9.0), Inches(1.8), Inches(3.3), Inches(1.8))
    tf = tb3.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "CONFORMAL UNCERTAINTY"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = COLOR_ACCENT
    p2 = tf.add_paragraph()
    p2.text = "Eliminated naive ±RMSE errors. Built split conformal prediction intervals achieving exact 79.75% empirical coverage for nominal 80% confidence."
    p2.font.size = Pt(13); p2.font.color.rgb = COLOR_TEXT_DARK; p2.space_before = Pt(8)

    # Bottom Summary Card
    add_card(slide2, Inches(0.8), Inches(4.2), Inches(11.7), Inches(2.7), "🎯 Core Mission & Objectives")
    tbb = slide2.shapes.add_textbox(Inches(1.0), Inches(4.8), Inches(11.3), Inches(1.9))
    tfb = tbb.text_frame
    tfb.word_wrap = True
    bullet_points_s2 = [
        "Eliminate Pricing Information Asymmetry: Provide transparent, algorithmic price discovery for Indian homebuyers, sellers, and lenders.",
        "100% Native Indian Rupee (₹) Denomination: Predict real property valuations in Lakhs & Crores with ₹/sq.ft benchmarking.",
        "Interactive PropTech Web Platform: Deliver a full-featured Streamlit UI featuring single appraisal, market analytics, bulk CSV processing, and EMI planning."
    ]
    for i, bp in enumerate(bullet_points_s2):
        p = tfb.paragraphs[0] if i == 0 else tfb.add_paragraph()
        p.text = f"•  {bp}"
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_after = Pt(10)

    # =========================================================================
    # SLIDE 3: PROBLEM STATEMENT & INDIAN MARKET REALITIES
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Problem Statement: Challenges in Indian Real Estate")

    c_left = add_card(slide3, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3), "❌ Traditional Valuation Pitfalls")
    tbl = slide3.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.5))
    tfl = tbl.text_frame
    tfl.word_wrap = True
    pts_l = [
        "Subjective Broker Quotations: Manual appraisals rely on word-of-mouth broker inquiries, leading to 15%–30% quote inflation.",
        "Extreme Price Skewness: Indian residential properties span from ₹20 Lakhs to ₹80+ Crores; linear models fail due to heavy right-tail variance.",
        "The 'Ames / Foreign Data Fallacy': Legacy ML tutorials map US variables (snow porches, basements, Iowa zoning) to Indian cities, creating completely synthetic results."
    ]
    for i, pt in enumerate(pts_l):
        p = tfl.paragraphs[0] if i == 0 else tfl.add_paragraph()
        p.text = f"•  {pt}"
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_after = Pt(12)

    c_right = add_card(slide3, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), "✅ The BharatProp AI Solution")
    tbr = slide3.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.5))
    tfr = tbr.text_frame
    tfr.word_wrap = True
    pts_r = [
        "Empirical Multi-City Ground Truth: Trained on 27,614 verified listings from Mumbai, Bangalore, Delhi, Hyderabad, Chennai, and Kolkata.",
        "Domain-Specific Feature Engineering: Captures super built-up area, BHK configurations, 35 gated community amenities, and vaastu compliance.",
        "Continuous Target Encoding: Learns micro-market price shrinkage for 1,711 localities without data leakage.",
        "Calibrated Uncertainty Bounds: Delivers defensible, mathematically validated prediction ranges."
    ]
    for i, pt in enumerate(pts_r):
        p = tfr.paragraphs[0] if i == 0 else tfr.add_paragraph()
        p.text = f"•  {pt}"
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_after = Pt(12)

    # =========================================================================
    # SLIDE 4: AUTHENTIC INDIAN DATASET OVERVIEW
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Dataset Overview: 6 Metropolitan Economic Hubs")

    # Table of Cities
    table_shape = slide4.shapes.add_table(7, 6, Inches(0.8), Inches(1.6), Inches(11.7), Inches(3.2))
    table = table_shape.table
    table.columns[0].width = Inches(2.0)
    table.columns[1].width = Inches(2.2)
    table.columns[2].width = Inches(1.8)
    table.columns[3].width = Inches(1.9)
    table.columns[4].width = Inches(1.9)
    table.columns[5].width = Inches(1.9)

    headers = ["Metropolitan City", "State / Region", "Cleaned Properties", "Median Price (₹)", "Mean Area (sqft)", "Median Rate (₹/sqft)"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_BG_DARK
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = COLOR_WHITE; p.alignment = PP_ALIGN.CENTER

    city_data = [
        ["Mumbai", "Maharashtra", "6,499", "₹92.00 L", "995 sq ft", "₹10,086/sq.ft"],
        ["Delhi-NCR", "Delhi / NCR", "3,851", "₹75.00 L", "1,282 sq ft", "₹6,945/sq.ft"],
        ["Bangalore", "Karnataka", "5,364", "₹73.61 L", "1,492 sq ft", "₹5,400/sq.ft"],
        ["Hyderabad", "Telangana", "1,993", "₹78.75 L", "1,667 sq ft", "₹5,112/sq.ft"],
        ["Chennai", "Tamil Nadu", "4,183", "₹59.25 L", "1,221 sq ft", "₹5,475/sq.ft"],
        ["Kolkata", "West Bengal", "5,724", "₹51.25 L", "1,206 sq ft", "₹4,555/sq.ft"],
    ]

    for i, row in enumerate(city_data):
        for j, val in enumerate(row):
            cell = table.cell(i+1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_BG if i % 2 == 0 else COLOR_WHITE
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(12); p.font.color.rgb = COLOR_TEXT_DARK
            if j >= 2: p.alignment = PP_ALIGN.CENTER

    # Dataset key highlights below
    add_card(slide4, Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.8), "📋 Dataset Specifications & Features")
    tb4 = slide4.shapes.add_textbox(Inches(1.0), Inches(5.6), Inches(11.3), Inches(1.2))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    p = tf4.paragraphs[0]
    p.text = "• Total Verified Properties: 27,614 | Distinct Micro-Markets: 1,711 | Target Variable: Price in Indian Rupees (₹)\n• 35 Binary Amenities: Gymnasium, Swimming Pool, 24x7 Security, Power Backup, Lift, Piped Gas, Vaastu, Club House, AC, etc.\n• Source: Kaggle & Open Indian Housing Repository | License: Open Access CC0 for academic & production research."
    p.font.size = Pt(12); p.font.color.rgb = COLOR_TEXT_DARK

    # =========================================================================
    # SLIDE 5: END-TO-END MACHINE LEARNING ARCHITECTURE
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "End-to-End Machine Learning Architecture")

    steps = [
        ("1. Data Ingestion", "Automated multi-city CSV download, deduplication & outlier bounds (₹1.8k–48k/sqft)."),
        ("2. Feature Engineering", "Calculate Area/BHK ratio, BHKxArea interaction, Furnishing, Security & Amenity counts."),
        ("3. Preprocessing", "ColumnTransformer with OneHot (City), continuous TargetEncoder (Location), RobustScaler."),
        ("4. Log-Target Regressor", "TransformedTargetRegressor wraps models with log1p forward and expm1 inverse transforms."),
        ("5. Multi-Model CV", "5-Fold Cross Validation comparing Ridge, Random Forest, Gradient Boosting & XGBoost."),
        ("6. Conformal Calibration", "Empirical non-conformity quantile calculation on calibration fold (N = 4,143 properties).")
    ]

    for idx, (st_title, st_desc) in enumerate(steps):
        col_i = idx % 3
        row_i = idx // 3
        l = Inches(0.8 + col_i * 4.0)
        t = Inches(1.6 + row_i * 2.7)
        add_card(slide5, l, t, Inches(3.7), Inches(2.4), st_title)
        tb = slide5.shapes.add_textbox(l + Inches(0.2), t + Inches(0.7), Inches(3.3), Inches(1.5))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = st_desc
        p.font.size = Pt(13); p.font.color.rgb = COLOR_TEXT_DARK

    # =========================================================================
    # SLIDE 6: DOMAIN FEATURE ENGINEERING & PREPROCESSING
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Domain Feature Engineering & Leakage Prevention")

    add_card(slide6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3), "🧠 Engineered Domain Features")
    tb_fe = slide6.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.5))
    tf_fe = tb_fe.text_frame
    tf_fe.word_wrap = True
    fe_items = [
        "Area_per_BHK: Built-up square footage divided by room count (indicates spatial roominess and luxury tier).",
        "BHK_x_Area Interaction: Captures non-linear price scaling between spaciousness and bedroom density.",
        "Furnishing_Score (0–9): Sum of verified home appliances (AC, TV, Refrigerator, Washing Machine, Microwave, etc.).",
        "Security_Score (0–5): Composite index of 24x7 security, intercom, power backup, maintenance staff, and lifts.",
        "Recreation_Score (0–9): Gated community lifestyle index (gym, pool, clubhouse, sports, gardens, golf course)."
    ]
    for i, it in enumerate(fe_items):
        p = tf_fe.paragraphs[0] if i == 0 else tf_fe.add_paragraph()
        p.text = f"•  {it}"
        p.font.size = Pt(12.5); p.font.color.rgb = COLOR_TEXT_DARK; p.space_after = Pt(10)

    add_card(slide6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), "🛡️ ColumnTransformer & Zero Leakage")
    tb_pp = slide6.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.5))
    tf_pp = tb_pp.text_frame
    tf_pp.word_wrap = True
    pp_items = [
        "Micro-Market Target Encoding: TargetEncoder(target_type='continuous', smooth='auto') fitted strictly on training folds to encode 1,711 localities.",
        "City One-Hot Encoding: Unbiased binary categorical encoding across all 6 metropolitan hubs.",
        "RobustScaler: Centers features using interquartile range (IQR) to prevent luxury mansions from distorting gradients.",
        "No Target Leakage Guarantee: Derived price metrics (e.g. ₹/sq.ft) are calculated strictly after prediction and NEVER fed as input features."
    ]
    for i, it in enumerate(pp_items):
        p = tf_pp.paragraphs[0] if i == 0 else tf_pp.add_paragraph()
        p.text = f"•  {it}"
        p.font.size = Pt(12.5); p.font.color.rgb = COLOR_TEXT_DARK; p.space_after = Pt(10)

    # =========================================================================
    # SLIDE 7: REGRESSION ALGORITHMS & TRAINING
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Regression Models & Log-Target Optimization")

    models_info = [
        ("🥇 XGBoost Regressor", "Extreme gradient boosted decision trees with depth pruning (max_depth=6), colsample subsampling (0.85), learning rate 0.06, and 220 estimators. Best generalizer."),
        ("🥈 Gradient Boosting", "Sequential additive decision trees with shrinkage rate 0.08 and 175 estimators. Strong baseline with lowest test RMSE."),
        ("🥉 Random Forest", "Bagging ensemble of 150 de-correlated trees (max_depth=14) reducing variance across diverse property clusters."),
        ("4️⃣ Ridge Regression", "L2-regularized linear model (alpha=10.0) establishing linear baseline across encoded location priors.")
    ]

    for idx, (m_name, m_desc) in enumerate(models_info):
        col_i = idx % 2
        row_i = idx // 2
        l = Inches(0.8 + col_i * 6.0)
        t = Inches(1.6 + row_i * 2.2)
        add_card(slide7, l, t, Inches(5.7), Inches(2.0), m_name)
        tb = slide7.shapes.add_textbox(l + Inches(0.25), t + Inches(0.65), Inches(5.2), Inches(1.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = m_desc
        p.font.size = Pt(13); p.font.color.rgb = COLOR_TEXT_DARK

    add_card(slide7, Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.8), "💡 The Log-Target Transformation Principle")
    tb_log = slide7.shapes.add_textbox(Inches(1.0), Inches(5.6), Inches(11.3), Inches(1.2))
    tf_log = tb_log.text_frame
    tf_log.word_wrap = True
    p = tf_log.paragraphs[0]
    p.text = "• Property prices are heavily right-skewed: fitting models on raw rupees leads squared errors to over-focus on multi-Crore luxury estates.\n• Wrapping models in TransformedTargetRegressor(func=log1p, inverse_func=expm1) minimizes relative proportional percentage errors across all housing tiers."
    p.font.size = Pt(12.5); p.font.color.rgb = COLOR_TEXT_DARK

    # =========================================================================
    # SLIDE 8: MODEL COMPARISON & BENCHMARK RESULTS
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Model Benchmarking & Evaluation Leaderboard")

    # Comparison Table
    table_shape8 = slide8.shapes.add_table(5, 7, Inches(0.8), Inches(1.6), Inches(11.7), Inches(2.6))
    t8 = table_shape8.table
    t8.columns[0].width = Inches(2.8)
    t8.columns[1].width = Inches(1.5)
    t8.columns[2].width = Inches(1.5)
    t8.columns[3].width = Inches(1.5)
    t8.columns[4].width = Inches(1.5)
    t8.columns[5].width = Inches(1.5)
    t8.columns[6].width = Inches(1.4)

    h8 = ["Model Architecture", "Test MAE", "Test RMSE", "R² (Rupees)", "R² (Log)", "5-Fold CV R²", "MAPE (%)"]
    for j, h in enumerate(h8):
        cell = t8.cell(0, j)
        cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = COLOR_BG_DARK
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = COLOR_WHITE; p.alignment = PP_ALIGN.CENTER

    res_rows = [
        ["🥇 XGBoost Regressor", "₹41.89 L", "₹98.04 L", "0.3921", "0.4664", "0.4788", "41.71%"],
        ["🥈 Gradient Boosting", "₹42.02 L", "₹96.64 L", "0.4093", "0.4655", "0.4786", "42.18%"],
        ["🥉 Random Forest", "₹42.29 L", "₹99.96 L", "0.3680", "0.4499", "0.4643", "42.69%"],
        ["4️⃣ Ridge Regression", "₹44.31 L", "₹102.01 L", "0.3419", "0.4260", "0.4418", "44.81%"]
    ]

    for i, row in enumerate(res_rows):
        for j, val in enumerate(row):
            cell = t8.cell(i+1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(236, 253, 245) if i == 0 else (COLOR_CARD_BG if i % 2 == 1 else COLOR_WHITE)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(12); p.font.color.rgb = COLOR_PRIMARY if i == 0 else COLOR_TEXT_DARK
            if j >= 1: p.alignment = PP_ALIGN.CENTER
            if i == 0: p.font.bold = True

    add_card(slide8, Inches(0.8), Inches(4.5), Inches(11.7), Inches(2.4), "📊 Key Diagnostic Takeaways")
    tb_dt = slide8.shapes.add_textbox(Inches(1.0), Inches(5.0), Inches(11.3), Inches(1.7))
    tf_dt = tb_dt.text_frame
    tf_dt.word_wrap = True
    dts = [
        "Holdout Test Generalization: All metrics evaluated on N = 4,143 independent properties (15% split) unseen during training.",
        "XGBoost Superiority: XGBoost achieved the lowest Test MAE (₹41.89 Lakhs) and lowest MAPE (41.71%) with consistent 5-fold CV stability (0.4788).",
        "Understanding R²: R² is the coefficient of determination (explaining ~47% of logarithmic variance across 1,711 micro-markets); it is NOT a percentage accuracy."
    ]
    for i, dt in enumerate(dts):
        p = tf_dt.paragraphs[0] if i == 0 else tf_dt.add_paragraph()
        p.text = f"•  {dt}"
        p.font.size = Pt(12.5); p.font.color.rgb = COLOR_TEXT_DARK; p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 9: CALIBRATED PREDICTION UNCERTAINTY (CONFORMAL PREDICTION)
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "Prediction Uncertainty: Split Conformal Prediction")

    add_card(slide9, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3), "❌ The Flaw of Naive ±RMSE")
    tb_cr = slide9.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.5))
    tf_cr = tb_cr.text_frame
    tf_cr.word_wrap = True
    flaws = [
        "Falsely Assumes Constant Normal Errors: Real estate errors scale with property value (a ₹2 Cr house has wider absolute dollar error than a ₹30 L flat).",
        "Produces Negative Valuations: For affordable properties, subtracting a fixed global RMSE can produce mathematically impossible negative lower bounds.",
        "Zero Statistical Guarantee: ±RMSE does not guarantee 80% or 95% coverage on real-world test distributions."
    ]
    for i, fl in enumerate(flaws):
        p = tf_cr.paragraphs[0] if i == 0 else tf_cr.add_paragraph()
        p.text = f"•  {fl}"
        p.font.size = Pt(13); p.font.color.rgb = COLOR_TEXT_DARK; p.space_after = Pt(12)

    add_card(slide9, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), "✅ The Conformal Prediction Method")
    tb_cp = slide9.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.5))
    tf_cp = tb_cp.text_frame
    tf_cp.word_wrap = True
    cpts = [
        "Inductive Split Calibration: Non-conformity scores s_i = |log(y_i) - log(y_hat_i)| calculated on N = 4,143 validation properties.",
        "Exact 80% Empirical Coverage: Test set empirical coverage on unseen properties is 79.75% (matching 80% nominal target).",
        "Multiplicative Prediction Ranges: Yields intuitive asymmetric ranges [y_hat * exp(-q), y_hat * exp(+q)] preventing negative prices.",
        "Honest UI Labeling: Presented explicitly as a 'Prediction Range / Interval', NOT a confidence interval."
    ]
    for i, cp in enumerate(cpts):
        p = tf_cp.paragraphs[0] if i == 0 else tf_cp.add_paragraph()
        p.text = f"•  {cp}"
        p.font.size = Pt(13); p.font.color.rgb = COLOR_TEXT_DARK; p.space_after = Pt(12)

    # =========================================================================
    # SLIDE 10: INTERACTIVE WEB APPLICATION (STREAMLIT)
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Interactive Web Application (Streamlit)")

    modules_ui = [
        ("🏠 1. Overview & Methodology", "Project background, Indian market dynamics, pipeline architecture diagram, and data summaries."),
        ("🎯 2. Property Valuation", "Real-time appraisal with dynamic city/locality dropdowns, preset selectors, ₹/sq.ft rates & 80% conformal range."),
        ("📊 3. Model Performance", "Leaderboard table, diagnostic plots (Actual vs Pred, Residuals, Bar Charts) & metric explanations."),
        ("📈 4. Market Analytics", "Interactive multi-city pricing histograms, Area vs Price scatter plots, BHK boxplots, and top locality rankings."),
        ("📁 5. Bulk CSV Valuation", "Batch portfolio appraisal via CSV upload with schema validation and downloadable result CSVs."),
        ("💰 6. Home Loan EMI Planner", "Down payment slider, LTV calculation, interest vs principal pie chart, and 10-year amortization schedule.")
    ]

    for idx, (m_title, m_desc) in enumerate(modules_ui):
        col_i = idx % 3
        row_i = idx // 3
        l = Inches(0.8 + col_i * 4.0)
        t = Inches(1.6 + row_i * 2.7)
        add_card(slide10, l, t, Inches(3.7), Inches(2.4), m_title)
        tb = slide10.shapes.add_textbox(l + Inches(0.2), t + Inches(0.7), Inches(3.3), Inches(1.5))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = m_desc
        p.font.size = Pt(13); p.font.color.rgb = COLOR_TEXT_DARK

    # =========================================================================
    # SLIDE 11: REAL-WORLD BUSINESS USE CASES & INDUSTRY IMPACT
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "Real-World Applications & Industry Impact")

    use_cases = [
        ("👨‍👩‍👧 Homebuyers & Sellers", "Instant fair-market benchmark protecting buyers from broker quote inflation and providing negotiation leverage."),
        ("🏦 Banks & Housing Finance (HFCs)", "Instant collateral pre-screening for home loan underwriting to verify statutory 75%–80% Loan-to-Value (LTV) limits."),
        ("📱 PropTech Portals (e.g. 99acres)", "Powers automated 'Instant Price Estimate' badges and guides sellers on optimal listing pricing."),
        ("📊 Real Estate Funds & REITs", "Bulk portfolio stress-testing via CSV batch evaluation during quarterly asset audits and acquisitions."),
        ("🏗️ Real Estate Developers", "Data-driven pricing for new residential launches by quantifying specific amenity valuation premiums.")
    ]

    for idx, (u_title, u_desc) in enumerate(use_cases):
        t = Inches(1.6 + idx * 1.05)
        add_card(slide11, Inches(0.8), t, Inches(11.7), Inches(0.95))
        tb = slide11.shapes.add_textbox(Inches(1.0), t + Inches(0.12), Inches(11.3), Inches(0.75))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f"{u_title}: "
        p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = COLOR_PRIMARY
        p_desc = p.add_run()
        p_desc.text = u_desc
        p_desc.font.size = Pt(13); p_desc.font.bold = False; p_desc.font.color.rgb = COLOR_TEXT_DARK

    # =========================================================================
    # SLIDE 12: LIMITATIONS, RISK FACTORS & DISCLAIMER
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    add_header(slide12, "Project Limitations & Professional Disclaimer")

    add_card(slide12, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3), "⚠️ Technical & Market Limitations")
    tb_lim = slide12.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.5))
    tf_lim = tb_lim.text_frame
    tf_lim.word_wrap = True
    lims = [
        "Asking vs Transaction Price Gap: Data reflects asking/listing prices; actual negotiated transaction prices may be 5%–15% lower.",
        "Unobserved Micro-Location Factors: Tabular models cannot observe floor-rise premiums, corner plots, ocean views, or builder brand prestige.",
        "Market Dynamics & Inflation: Historical listing data does not automatically factor sudden interest rate or municipal tax changes."
    ]
    for i, lm in enumerate(lims):
        p = tf_lim.paragraphs[0] if i == 0 else tf_lim.add_paragraph()
        p.text = f"•  {lm}"
        p.font.size = Pt(13); p.font.color.rgb = COLOR_TEXT_DARK; p.space_after = Pt(12)

    add_card(slide12, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), "⚖️ Official Professional Disclaimer")
    tb_disc = slide12.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.5))
    tf_disc = tb_disc.text_frame
    tf_disc.word_wrap = True
    p = tf_disc.paragraphs[0]
    p.text = "STATISTICAL ESTIMATE ONLY\n\n"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = RGBColor(185, 28, 28)
    p_d = tf_disc.add_paragraph()
    p_d.text = "The valuations and prediction intervals generated by BharatProp AI are automated statistical approximations intended solely for informational, research, and educational purposes.\n\nThey do NOT constitute certified legal appraisals under RICS / Government guidelines, title search opinions, banking mortgage approvals, or tax advice. Users should consult licensed property evaluators prior to executing transactions."
    p_d.font.size = Pt(12.5); p_d.font.color.rgb = COLOR_TEXT_DARK

    # =========================================================================
    # SLIDE 13: FUTURE ROADMAP & TECHNICAL INNOVATIONS
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    add_header(slide13, "Future Innovations & Product Roadmap")

    roadmap_items = [
        ("🛰️ Geospatial & Satellite Intelligence", "Integrate OpenStreetMap and GIS coordinates for distance-to-metro, proximity to IT corridors, and green space ratios."),
        ("📷 Computer Vision Image Valuation", "Train multi-modal CNN/Vision Transformers on property floor plans and interior photos to estimate furnishing wear & tear."),
        ("📜 RERA & Land Registry Integration", "Connect with state RERA and Sub-Registrar transaction portals to blend asking prices with actual registered circle rates."),
        ("🌐 RESTful Microservice API", "Deploy a FastAPI backend to integrate real-time property valuation endpoints into external fintech and banking apps.")
    ]

    for idx, (r_title, r_desc) in enumerate(roadmap_items):
        t = Inches(1.6 + idx * 1.3)
        add_card(slide13, Inches(0.8), t, Inches(11.7), Inches(1.15), r_title)
        tb = slide13.shapes.add_textbox(Inches(1.0), t + Inches(0.55), Inches(11.3), Inches(0.55))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = r_desc
        p.font.size = Pt(13); p.font.color.rgb = COLOR_TEXT_DARK

    # =========================================================================
    # SLIDE 14: CONCLUSION & PROJECT LINKS (Dark Hero Theme)
    # =========================================================================
    slide14 = prs.slides.add_slide(blank_layout)
    bg14 = slide14.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg14.fill.solid()
    bg14.fill.fore_color.rgb = COLOR_BG_DARK
    bg14.line.fill.background()

    tb14 = slide14.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(4.5))
    tf14 = tb14.text_frame
    tf14.word_wrap = True

    p = tf14.paragraphs[0]
    p.text = "THANK YOU & QUESTIONS"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = COLOR_ACCENT

    p_t = tf14.add_paragraph()
    p_t.text = "BharatProp AI: Indian PropTech Innovation"
    p_t.font.size = Pt(30); p_t.font.bold = True; p_t.font.color.rgb = COLOR_WHITE; p_t.space_before = Pt(10)

    p_body = tf14.add_paragraph()
    p_body.text = "A complete, transparent, and mathematically rigorous Automated Valuation Model engineered for the Indian residential housing market."
    p_body.font.size = Pt(16); p_body.font.color.rgb = RGBColor(203, 213, 225); p_body.space_before = Pt(10); p_body.space_after = Pt(20)

    p_links = tf14.add_paragraph()
    p_links.text = "🌐 Live Streamlit Application:  https://real-estate-valuation-kotesh.streamlit.app/\n💻 GitHub Source Repository:    https://github.com/koteswaraosabbavarapu-stack/Real-Estate-Valuation\n👤 Project Author:             Koteswarao Sabbavarapu"
    p_links.font.size = Pt(14); p_links.font.color.rgb = COLOR_ACCENT; p_links.font.bold = True

    prs.save(str(OUTPUT_PPTX_PATH))
    print(f"Presentation generated successfully: {OUTPUT_PPTX_PATH}")

if __name__ == "__main__":
    create_presentation()
