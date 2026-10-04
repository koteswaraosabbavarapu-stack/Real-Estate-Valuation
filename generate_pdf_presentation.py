"""
Script to generate an elegant, high-impact 14-slide PDF presentation
for the BharatProp AI Indian Real Estate Valuation system using ReportLab.
"""

import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "BharatProp_AI_Presentation.pdf"

# Theme Colors
C_DARK_BG = colors.HexColor("#0f172a")       # Slate 900
C_PRIMARY = colors.HexColor("#064e3b")       # Emerald 900
C_ACCENT_GREEN = colors.HexColor("#10b981")  # Emerald 500
C_ACCENT_BLUE = colors.HexColor("#0284c7")   # Sky 600
C_TEXT_DARK = colors.HexColor("#0f172a")     # Slate 900
C_TEXT_MUTED = colors.HexColor("#64748b")    # Slate 500
C_WHITE = colors.HexColor("#ffffff")
C_CARD_BG = colors.HexColor("#f8fafc")       # Slate 50
C_CARD_BORDER = colors.HexColor("#cbd5e1")   # Slate 300
C_HIGHLIGHT_BG = colors.HexColor("#ecfdf5")  # Emerald 50

PAGE_WIDTH, PAGE_HEIGHT = landscape(letter) # 11 x 8.5 inches


class SlideCanvas(canvas.Canvas):
    """Custom canvas that draws slide borders, headers, and footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_slide_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_slide_decorations(self, total_pages):
        page_num = self._pageNumber
        # Slide 1 and Slide 14 are full dark background slides
        if page_num == 1 or page_num == total_pages:
            self.setFillColor(C_DARK_BG)
            self.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
            # Decorative accent bar
            self.setFillColor(C_ACCENT_GREEN)
            self.rect(0.5 * inch, PAGE_HEIGHT - 0.4 * inch, PAGE_WIDTH - 1.0 * inch, 0.05 * inch, fill=1, stroke=0)
        else:
            # Light slide top header line
            self.setFillColor(C_PRIMARY)
            self.rect(0.5 * inch, PAGE_HEIGHT - 0.35 * inch, PAGE_WIDTH - 1.0 * inch, 0.03 * inch, fill=1, stroke=0)
            
            # Bottom footer line
            self.setFillColor(C_CARD_BORDER)
            self.rect(0.5 * inch, 0.45 * inch, PAGE_WIDTH - 1.0 * inch, 0.01 * inch, fill=1, stroke=0)
            
            # Footer text
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(C_TEXT_MUTED)
            self.drawString(0.5 * inch, 0.28 * inch, "BHARATPROP AI — INDIAN REAL ESTATE AUTOMATED VALUATION SYSTEM (AVM)")
            self.drawRightString(PAGE_WIDTH - 0.5 * inch, 0.28 * inch, f"Slide {page_num} of {total_pages}")


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=landscape(letter),
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.55 * inch
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    style_dark_badge = ParagraphStyle(
        "DarkBadge",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        textColor=C_ACCENT_GREEN,
        spaceAfter=12
    )
    style_dark_title = ParagraphStyle(
        "DarkTitle",
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=C_WHITE,
        spaceAfter=14
    )
    style_dark_sub = ParagraphStyle(
        "DarkSub",
        fontName="Helvetica",
        fontSize=12,
        leading=17,
        textColor=colors.HexColor("#cbd5e1"),
        spaceAfter=20
    )
    style_dark_meta = ParagraphStyle(
        "DarkMeta",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=C_ACCENT_GREEN
    )

    style_header_cat = ParagraphStyle(
        "HeaderCat",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=C_ACCENT_GREEN,
        spaceAfter=2
    )
    style_header_title = ParagraphStyle(
        "HeaderTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=C_TEXT_DARK,
        spaceAfter=12
    )

    style_card_title = ParagraphStyle(
        "CardTitle",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=C_PRIMARY,
        spaceAfter=6
    )
    style_card_body = ParagraphStyle(
        "CardBody",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=C_TEXT_DARK,
        spaceAfter=6
    )
    style_bullet = ParagraphStyle(
        "BulletPoint",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=C_TEXT_DARK,
        leftIndent=12,
        firstLineIndent=-10,
        spaceAfter=5
    )

    story = []

    def make_header(title, cat="BHARATPROP AI • INDIAN REAL ESTATE AVM"):
        return [
            Paragraph(cat.upper(), style_header_cat),
            Paragraph(title, style_header_title)
        ]

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    story.append(Spacer(1, 0.8 * inch))
    story.append(Paragraph("PROPTECH • MACHINE LEARNING • CONFORMAL PREDICTION", style_dark_badge))
    story.append(Paragraph("AI-Based Indian Real Estate Price Prediction<br/>& Automated Property Valuation System", style_dark_title))
    story.append(Paragraph(
        "<b>BharatProp AI</b> — Production Automated Valuation Model (AVM) across 6 Major Metropolitan Cities.<br/>"
        "Powered by XGBoost, 5-Fold Cross-Validation, and Inductive Split Conformal Uncertainty Quantification.",
        style_dark_sub
    ))
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph(
        "🌐 Live Deployed App: <u>real-estate-valuation-kotesh.streamlit.app</u> &nbsp;|&nbsp; "
        "💻 Author: Koteswarao Sabbavarapu",
        style_dark_meta
    ))
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 2: EXECUTIVE SUMMARY & PROJECT VISION
    # =========================================================================
    story.extend(make_header("Executive Summary & Project Vision"))
    
    # 3 Summary Cards in Table
    card1 = [
        Paragraph("AUTHENTIC INDIAN DATA", style_card_title),
        Paragraph("Trained on <b>27,614 verified residential properties</b> across Mumbai, Bangalore, Delhi-NCR, Hyderabad, Chennai & Kolkata.", style_card_body)
    ]
    card2 = [
        Paragraph("PRODUCTION ML PIPELINE", style_card_title),
        Paragraph("Trained Ridge, Random Forest, GBR & XGBoost with log-target transformation. Best Test MAE: <b>₹41.89 Lakhs</b>.", style_card_body)
    ]
    card3 = [
        Paragraph("CONFORMAL UNCERTAINTY", style_card_title),
        Paragraph("Eliminated naive ±RMSE errors. Built split conformal prediction intervals with <b>79.75% empirical test coverage</b>.", style_card_body)
    ]

    t_cards = Table([[card1, card2, card3]], colWidths=[3.2 * inch, 3.2 * inch, 3.2 * inch])
    t_cards.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CARD_BG),
        ("BOX", (0,0), (-1,-1), 1, C_CARD_BORDER),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(t_cards)
    story.append(Spacer(1, 14))

    mission_box = [
        Paragraph("🎯 Core Objectives & Technical Deliverables", style_card_title),
        Paragraph("• <b>Eliminate Pricing Asymmetry:</b> Provide transparent algorithmic price discovery for homebuyers, sellers, and mortgage lenders.", style_bullet),
        Paragraph("• <b>Native Indian Rupee (₹) Denomination:</b> Predict real property values in Lakhs and Crores with ₹/sq.ft micro-market rates.", style_bullet),
        Paragraph("• <b>Interactive PropTech Web Platform:</b> Deploy a full-featured Streamlit UI with single appraisal, market analytics, bulk CSV, and EMI calculator.", style_bullet)
    ]
    t_mission = Table([[mission_box]], colWidths=[9.8 * inch])
    t_mission.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_HIGHLIGHT_BG),
        ("BOX", (0,0), (-1,-1), 1, C_ACCENT_GREEN),
        ("PADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(t_mission)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 3: PROBLEM STATEMENT & INDIAN MARKET REALITIES
    # =========================================================================
    story.extend(make_header("Problem Statement: Challenges in Indian Real Estate"))

    left_box = [
        Paragraph("❌ Traditional Valuation Pitfalls", style_card_title),
        Paragraph("• <b>Subjective Broker Quotes:</b> Manual appraisals rely on word-of-mouth broker inquiries, leading to 15%–30% quote inflation.", style_bullet),
        Paragraph("• <b>Extreme Price Skewness:</b> Indian residential properties range from ₹20 Lakhs to ₹80+ Crores; linear models break down on heavy right-tail variance.", style_bullet),
        Paragraph("• <b>The 'Ames / Foreign Data Fallacy':</b> Legacy ML tutorials map US variables (snow porches, basements, Iowa zoning) to Indian cities, creating artificial synthetic results.", style_bullet)
    ]

    right_box = [
        Paragraph("✅ The BharatProp AI Solution", style_card_title),
        Paragraph("• <b>Empirical Multi-City Ground Truth:</b> Trained on 27,614 verified listings across Mumbai, Bangalore, Delhi, Hyderabad, Chennai, and Kolkata.", style_bullet),
        Paragraph("• <b>Domain Feature Engineering:</b> Super built-up area, BHK configurations, 35 gated community amenities, and vaastu compliance.", style_bullet),
        Paragraph("• <b>Continuous Target Encoding:</b> Learns micro-market price shrinkage for 1,711 localities without data leakage.", style_bullet),
        Paragraph("• <b>Calibrated Uncertainty Bounds:</b> Delivers defensible, mathematically validated prediction ranges.", style_bullet)
    ]

    t_prob = Table([[left_box, right_box]], colWidths=[4.85 * inch, 4.85 * inch])
    t_prob.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), colors.HexColor("#fff1f2")), # Red tint
        ("BOX", (0,0), (0,0), 1, colors.HexColor("#fecdd3")),
        ("BACKGROUND", (1,0), (1,0), C_HIGHLIGHT_BG),              # Green tint
        ("BOX", (1,0), (1,0), 1, C_ACCENT_GREEN),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("PADDING", (0,0), (-1,-1), 12),
    ]))
    story.append(t_prob)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 4: AUTHENTIC INDIAN DATASET OVERVIEW
    # =========================================================================
    story.extend(make_header("Dataset Overview: 6 Metropolitan Economic Hubs"))

    city_table_data = [
        [Paragraph("<b>Metropolitan City</b>", style_card_body),
         Paragraph("<b>State / Region</b>", style_card_body),
         Paragraph("<b>Cleaned Properties</b>", style_card_body),
         Paragraph("<b>Median Price (₹)</b>", style_card_body),
         Paragraph("<b>Mean Area</b>", style_card_body),
         Paragraph("<b>Median Rate</b>", style_card_body)],
        ["Mumbai", "Maharashtra", "6,499", "₹92.00 L", "995 sq ft", "₹10,086/sq.ft"],
        ["Delhi-NCR", "Delhi / NCR", "3,851", "₹75.00 L", "1,282 sq ft", "₹6,945/sq.ft"],
        ["Bangalore", "Karnataka", "5,364", "₹73.61 L", "1,492 sq ft", "₹5,400/sq.ft"],
        ["Hyderabad", "Telangana", "1,993", "₹78.75 L", "1,667 sq ft", "₹5,112/sq.ft"],
        ["Chennai", "Tamil Nadu", "4,183", "₹59.25 L", "1,221 sq ft", "₹5,475/sq.ft"],
        ["Kolkata", "West Bengal", "5,724", "₹51.25 L", "1,206 sq ft", "₹4,555/sq.ft"],
    ]

    t_city = Table(city_table_data, colWidths=[1.8 * inch, 1.8 * inch, 1.5 * inch, 1.5 * inch, 1.6 * inch, 1.6 * inch])
    t_city.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_DARK_BG),
        ("TEXTCOLOR", (0,0), (-1,0), C_WHITE),
        ("ALIGN", (2,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.5, C_CARD_BORDER),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_WHITE, C_CARD_BG]),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t_city)
    story.append(Spacer(1, 10))

    spec_box = [
        Paragraph("📋 Dataset Specifications & Features", style_card_title),
        Paragraph("• <b>Total Verified Properties:</b> 27,614 | <b>Distinct Micro-Markets:</b> 1,711 | <b>Target:</b> Price in Indian Rupees (₹)", style_bullet),
        Paragraph("• <b>35 Binary Amenities:</b> Gymnasium, Swimming Pool, 24x7 Security, Power Backup, Lift, Piped Gas, Vaastu, Club House, AC, etc.", style_bullet),
        Paragraph("• <b>Source:</b> Kaggle & Open Indian Housing Repository | <b>License:</b> Open Access CC0 for research & commercial prototyping.", style_bullet)
    ]
    t_spec = Table([[spec_box]], colWidths=[9.8 * inch])
    t_spec.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CARD_BG),
        ("BOX", (0,0), (-1,-1), 1, C_CARD_BORDER),
        ("PADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(t_spec)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 5: END-TO-END MACHINE LEARNING ARCHITECTURE
    # =========================================================================
    story.extend(make_header("End-to-End Machine Learning Architecture"))

    arch_row1 = [
        [Paragraph("1. Data Ingestion", style_card_title), Paragraph("Multi-city CSV download, deduplication & realistic boundary filters (₹1.8k–48k/sqft).", style_card_body)],
        [Paragraph("2. Feature Engineering", style_card_title), Paragraph("Calculate Area/BHK ratio, BHKxArea interaction, Furnishing, Security & Amenity scores.", style_card_body)],
        [Paragraph("3. Preprocessing", style_card_title), Paragraph("ColumnTransformer with OneHot (City), continuous TargetEncoder (Location), RobustScaler.", style_card_body)]
    ]
    arch_row2 = [
        [Paragraph("4. Log-Target Regressor", style_card_title), Paragraph("TransformedTargetRegressor with log1p forward and expm1 inverse transforms.", style_card_body)],
        [Paragraph("5. Multi-Model 5-Fold CV", style_card_title), Paragraph("5-Fold Cross Validation comparing Ridge, Random Forest, Gradient Boosting & XGBoost.", style_card_body)],
        [Paragraph("6. Conformal Calibration", style_card_title), Paragraph("Empirical non-conformity quantile calculation on calibration fold (N = 4,143 properties).", style_card_body)]
    ]

    t_arch1 = Table([[arch_row1[0], arch_row1[1], arch_row1[2]]], colWidths=[3.2 * inch, 3.2 * inch, 3.2 * inch])
    t_arch1.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CARD_BG),
        ("BOX", (0,0), (-1,-1), 1, C_CARD_BORDER),
        ("PADDING", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    t_arch2 = Table([[arch_row2[0], arch_row2[1], arch_row2[2]]], colWidths=[3.2 * inch, 3.2 * inch, 3.2 * inch])
    t_arch2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CARD_BG),
        ("BOX", (0,0), (-1,-1), 1, C_CARD_BORDER),
        ("PADDING", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(t_arch1)
    story.append(Spacer(1, 10))
    story.append(t_arch2)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 6: DOMAIN FEATURE ENGINEERING & PREPROCESSING
    # =========================================================================
    story.extend(make_header("Domain Feature Engineering & Zero-Leakage Preprocessing"))

    fe_col = [
        Paragraph("🧠 Engineered Domain Features", style_card_title),
        Paragraph("• <b>Area_per_BHK:</b> Built-up square footage divided by room count (indicates spaciousness and luxury tier).", style_bullet),
        Paragraph("• <b>BHK_x_Area Interaction:</b> Captures non-linear price scaling between spaciousness and bedroom density.", style_bullet),
        Paragraph("• <b>Furnishing_Score (0–9):</b> Sum of verified home appliances (AC, TV, Refrigerator, Washing Machine, Microwave).", style_bullet),
        Paragraph("• <b>Security_Score (0–5):</b> Composite index of 24x7 security, intercom, power backup, maintenance, and lifts.", style_bullet),
        Paragraph("• <b>Recreation_Score (0–9):</b> Gated community lifestyle index (gym, pool, clubhouse, sports, green gardens).", style_bullet)
    ]

    pp_col = [
        Paragraph("🛡️ ColumnTransformer & Zero Leakage", style_card_title),
        Paragraph("• <b>Micro-Market Target Encoding:</b> TargetEncoder(target_type='continuous', smooth='auto') fitted strictly on training folds for 1,711 localities.", style_bullet),
        Paragraph("• <b>City One-Hot Encoding:</b> Unbiased binary categorical encoding across all 6 metropolitan hubs.", style_bullet),
        Paragraph("• <b>RobustScaler:</b> Centers features using IQR to prevent multi-Crore outliers from distorting gradients.", style_bullet),
        Paragraph("• <b>No Target Leakage:</b> Derived price metrics (₹/sq.ft) are computed strictly after prediction.", style_bullet)
    ]

    t_fe = Table([[fe_col, pp_col]], colWidths=[4.85 * inch, 4.85 * inch])
    t_fe.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CARD_BG),
        ("BOX", (0,0), (-1,-1), 1, C_CARD_BORDER),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("PADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(t_fe)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 7: REGRESSION ALGORITHMS & TRAINING
    # =========================================================================
    story.extend(make_header("Regression Models & Log-Target Optimization"))

    m1 = [Paragraph("🥇 XGBoost Regressor", style_card_title), Paragraph("Extreme gradient boosted decision trees with depth pruning (max_depth=6), colsample subsampling (0.85), learning rate 0.06, and 220 estimators. Best generalizer.", style_card_body)]
    m2 = [Paragraph("🥈 Gradient Boosting", style_card_title), Paragraph("Sequential additive decision trees with shrinkage rate 0.08 and 175 estimators. Strong baseline with lowest test RMSE.", style_card_body)]
    m3 = [Paragraph("🥉 Random Forest", style_card_title), Paragraph("Bagging ensemble of 150 de-correlated trees (max_depth=14) reducing variance across diverse property clusters.", style_card_body)]
    m4 = [Paragraph("4️⃣ Ridge Regression", style_card_title), Paragraph("L2-regularized linear model (alpha=10.0) establishing linear baseline across encoded location priors.", style_card_body)]

    t_mod1 = Table([[m1, m2]], colWidths=[4.85 * inch, 4.85 * inch])
    t_mod1.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CARD_BG),
        ("BOX", (0,0), (-1,-1), 1, C_CARD_BORDER),
        ("PADDING", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    t_mod2 = Table([[m3, m4]], colWidths=[4.85 * inch, 4.85 * inch])
    t_mod2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CARD_BG),
        ("BOX", (0,0), (-1,-1), 1, C_CARD_BORDER),
        ("PADDING", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(t_mod1)
    story.append(Spacer(1, 8))
    story.append(t_mod2)
    story.append(Spacer(1, 8))

    log_box = [
        Paragraph("💡 The Log-Target Transformation Principle", style_card_title),
        Paragraph("• Property prices are heavily right-skewed: fitting models on raw rupees leads squared errors to over-focus on multi-Crore luxury estates.<br/>"
                  "• Wrapping models in TransformedTargetRegressor(func=log1p, inverse_func=expm1) minimizes relative proportional percentage errors across all housing tiers.", style_card_body)
    ]
    t_log = Table([[log_box]], colWidths=[9.8 * inch])
    t_log.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_HIGHLIGHT_BG),
        ("BOX", (0,0), (-1,-1), 1, C_ACCENT_GREEN),
        ("PADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(t_log)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 8: MODEL COMPARISON & BENCHMARK RESULTS
    # =========================================================================
    story.extend(make_header("Model Benchmarking & Evaluation Leaderboard"))

    eval_table_data = [
        [Paragraph("<b>Model Architecture</b>", style_card_body),
         Paragraph("<b>Test MAE</b>", style_card_body),
         Paragraph("<b>Test RMSE</b>", style_card_body),
         Paragraph("<b>R² (Rupees)</b>", style_card_body),
         Paragraph("<b>R² (Log)</b>", style_card_body),
         Paragraph("<b>5-Fold CV R²</b>", style_card_body),
         Paragraph("<b>MAPE (%)</b>", style_card_body)],
        ["🥇 XGBoost Regressor", "₹41.89 L", "₹98.04 L", "0.3921", "0.4664", "0.4788", "41.71%"],
        ["🥈 Gradient Boosting", "₹42.02 L", "₹96.64 L", "0.4093", "0.4655", "0.4786", "42.18%"],
        ["🥉 Random Forest", "₹42.29 L", "₹99.96 L", "0.3680", "0.4499", "0.4643", "42.69%"],
        ["4️⃣ Ridge Regression", "₹44.31 L", "₹102.01 L", "0.3419", "0.4260", "0.4418", "44.81%"]
    ]

    t_eval = Table(eval_table_data, colWidths=[2.5 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.3 * inch, 1.2 * inch])
    t_eval.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_DARK_BG),
        ("TEXTCOLOR", (0,0), (-1,0), C_WHITE),
        ("BACKGROUND", (0,1), (-1,1), C_HIGHLIGHT_BG),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.5, C_CARD_BORDER),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 10))

    eval_diag_box = [
        Paragraph("📊 Key Evaluation Takeaways", style_card_title),
        Paragraph("• <b>Holdout Test Generalization:</b> All metrics evaluated on N = 4,143 independent properties (15% split) unseen during training.", style_bullet),
        Paragraph("• <b>XGBoost Superiority:</b> XGBoost achieved the lowest Test MAE (₹41.89 Lakhs) and lowest MAPE (41.71%) with consistent 5-fold CV stability (0.4788).", style_bullet),
        Paragraph("• <b>Understanding R²:</b> R² is the coefficient of determination (explaining ~47% of logarithmic variance across 1,711 micro-markets); it is NOT a percentage accuracy.", style_bullet)
    ]
    t_ed = Table([[eval_diag_box]], colWidths=[9.8 * inch])
    t_ed.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CARD_BG),
        ("BOX", (0,0), (-1,-1), 1, C_CARD_BORDER),
        ("PADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(t_ed)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 9: CALIBRATED PREDICTION UNCERTAINTY (CONFORMAL PREDICTION)
    # =========================================================================
    story.extend(make_header("Prediction Uncertainty: Split Conformal Prediction"))

    flaw_box = [
        Paragraph("❌ The Flaw of Naive ±RMSE", style_card_title),
        Paragraph("• <b>Falsely Assumes Constant Normal Errors:</b> Real estate errors scale with property value (a ₹2 Cr house has wider absolute dollar error than a ₹30 L flat).", style_bullet),
        Paragraph("• <b>Produces Negative Valuations:</b> For affordable properties, subtracting a fixed global RMSE can produce mathematically impossible negative lower bounds.", style_bullet),
        Paragraph("• <b>Zero Statistical Guarantee:</b> ±RMSE does not guarantee 80% or 95% coverage on real-world test distributions.", style_bullet)
    ]

    cp_box = [
        Paragraph("✅ The Conformal Prediction Method", style_card_title),
        Paragraph("• <b>Inductive Split Calibration:</b> Non-conformity scores s_i = |log(y_i) - log(y_hat_i)| calculated on N = 4,143 validation properties.", style_bullet),
        Paragraph("• <b>Exact 80% Empirical Coverage:</b> Test set empirical coverage on unseen properties is 79.75% (matching 80% nominal target).", style_bullet),
        Paragraph("• <b>Multiplicative Prediction Ranges:</b> Yields intuitive asymmetric ranges [y_hat * exp(-q), y_hat * exp(+q)] preventing negative prices.", style_bullet),
        Paragraph("• <b>Honest UI Labeling:</b> Presented explicitly as a 'Prediction Range / Interval', NOT a confidence interval.", style_bullet)
    ]

    t_cp = Table([[flaw_box, cp_box]], colWidths=[4.85 * inch, 4.85 * inch])
    t_cp.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), colors.HexColor("#fff1f2")),
        ("BOX", (0,0), (0,0), 1, colors.HexColor("#fecdd3")),
        ("BACKGROUND", (1,0), (1,0), C_HIGHLIGHT_BG),
        ("BOX", (1,0), (1,0), 1, C_ACCENT_GREEN),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("PADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(t_cp)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 10: INTERACTIVE WEB APPLICATION (STREAMLIT)
    # =========================================================================
    story.extend(make_header("Interactive Web Application (Streamlit)"))

    app_r1 = [
        [Paragraph("🏠 1. Overview & Methodology", style_card_title), Paragraph("Project background, Indian market dynamics, pipeline architecture diagram, and data summaries.", style_card_body)],
        [Paragraph("🎯 2. Property Valuation", style_card_title), Paragraph("Real-time appraisal with dynamic city/locality dropdowns, preset selectors, ₹/sq.ft rates & 80% conformal range.", style_card_body)],
        [Paragraph("📊 3. Model Performance", style_card_title), Paragraph("Leaderboard table, diagnostic plots (Actual vs Pred, Residuals, Bar Charts) & metric explanations.", style_card_body)]
    ]
    app_r2 = [
        [Paragraph("📈 4. Market Analytics", style_card_title), Paragraph("Interactive multi-city pricing histograms, Area vs Price scatter plots, BHK boxplots, and top locality rankings.", style_card_body)],
        [Paragraph("📁 5. Bulk CSV Valuation", style_card_title), Paragraph("Batch portfolio appraisal via CSV upload with schema validation and downloadable result CSVs.", style_card_body)],
        [Paragraph("💰 6. Home Loan EMI Planner", style_card_title), Paragraph("Down payment slider, LTV calculation, interest vs principal pie chart, and 10-year amortization schedule.", style_card_body)]
    ]

    t_app1 = Table([[app_r1[0], app_r1[1], app_r1[2]]], colWidths=[3.2 * inch, 3.2 * inch, 3.2 * inch])
    t_app1.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CARD_BG),
        ("BOX", (0,0), (-1,-1), 1, C_CARD_BORDER),
        ("PADDING", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    t_app2 = Table([[app_r2[0], app_r2[1], app_r2[2]]], colWidths=[3.2 * inch, 3.2 * inch, 3.2 * inch])
    t_app2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CARD_BG),
        ("BOX", (0,0), (-1,-1), 1, C_CARD_BORDER),
        ("PADDING", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(t_app1)
    story.append(Spacer(1, 10))
    story.append(t_app2)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 11: REAL-WORLD BUSINESS USE CASES & INDUSTRY IMPACT
    # =========================================================================
    story.extend(make_header("Real-World Applications & Industry Impact"))

    uc_rows = [
        [Paragraph("👨‍👩‍👧 Homebuyers & Sellers", style_card_title), Paragraph("Instant fair-market benchmark protecting buyers from broker quote inflation and providing negotiation leverage.", style_card_body)],
        [Paragraph("🏦 Banks & Housing Finance (HFCs)", style_card_title), Paragraph("Instant collateral pre-screening for home loan underwriting to verify statutory 75%–80% Loan-to-Value (LTV) limits.", style_card_body)],
        [Paragraph("📱 PropTech Portals (e.g. 99acres)", style_card_title), Paragraph("Powers automated 'Instant Price Estimate' badges and guides sellers on optimal listing pricing.", style_card_body)],
        [Paragraph("📊 Real Estate Funds & REITs", style_card_title), Paragraph("Bulk portfolio stress-testing via CSV batch evaluation during quarterly asset audits and acquisitions.", style_card_body)],
        [Paragraph("🏗️ Real Estate Developers", style_card_title), Paragraph("Data-driven pricing for new residential launches by quantifying specific amenity valuation premiums.", style_card_body)]
    ]

    t_uc = Table(uc_rows, colWidths=[3.2 * inch, 6.6 * inch])
    t_uc.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CARD_BG),
        ("GRID", (0,0), (-1,-1), 0.5, C_CARD_BORDER),
        ("PADDING", (0,0), (-1,-1), 6),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t_uc)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 12: LIMITATIONS, RISK FACTORS & DISCLAIMER
    # =========================================================================
    story.extend(make_header("Project Limitations & Professional Disclaimer"))

    lim_box = [
        Paragraph("⚠️ Technical & Market Limitations", style_card_title),
        Paragraph("• <b>Asking vs Transaction Price Gap:</b> Data reflects asking/listing prices; actual negotiated transaction prices may be 5%–15% lower.", style_bullet),
        Paragraph("• <b>Unobserved Micro-Location Factors:</b> Tabular models cannot observe floor-rise premiums, corner plots, ocean views, or builder brand prestige.", style_bullet),
        Paragraph("• <b>Market Dynamics & Inflation:</b> Historical listing data does not automatically factor sudden interest rate or municipal tax changes.", style_bullet)
    ]

    disc_box = [
        Paragraph("⚖️ Official Professional Disclaimer", style_card_title),
        Paragraph("<b>STATISTICAL ESTIMATE ONLY:</b><br/>"
                  "The valuations and prediction intervals generated by BharatProp AI are automated statistical approximations intended solely for informational, research, and educational purposes.<br/><br/>"
                  "They do NOT constitute certified legal appraisals under RICS / Government guidelines, title search opinions, banking mortgage approvals, or tax advice. Users should consult licensed property evaluators prior to executing transactions.", style_card_body)
    ]

    t_ld = Table([[lim_box, disc_box]], colWidths=[4.85 * inch, 4.85 * inch])
    t_ld.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), C_CARD_BG),
        ("BOX", (0,0), (0,0), 1, C_CARD_BORDER),
        ("BACKGROUND", (1,0), (1,0), colors.HexColor("#fef2f2")),
        ("BOX", (1,0), (1,0), 1, colors.HexColor("#fecaca")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("PADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(t_ld)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 13: FUTURE ROADMAP & TECHNICAL INNOVATIONS
    # =========================================================================
    story.extend(make_header("Future Innovations & Product Roadmap"))

    road_rows = [
        [Paragraph("🛰️ Geospatial & Satellite Intelligence", style_card_title), Paragraph("Integrate OpenStreetMap and GIS coordinates for distance-to-metro, proximity to IT corridors, and green space ratios.", style_card_body)],
        [Paragraph("📷 Computer Vision Image Valuation", style_card_title), Paragraph("Train multi-modal CNN/Vision Transformers on floor plans and interior photos to estimate furnishing quality and wear.", style_card_body)],
        [Paragraph("📜 RERA & Land Registry Integration", style_card_title), Paragraph("Connect with state RERA and Sub-Registrar transaction portals to blend asking prices with actual registered circle rates.", style_card_body)],
        [Paragraph("🌐 RESTful Microservice API", style_card_title), Paragraph("Deploy a FastAPI backend to integrate real-time property valuation endpoints into external fintech and banking apps.", style_card_body)]
    ]

    t_road = Table(road_rows, colWidths=[3.2 * inch, 6.6 * inch])
    t_road.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CARD_BG),
        ("GRID", (0,0), (-1,-1), 0.5, C_CARD_BORDER),
        ("PADDING", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t_road)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 14: CONCLUSION & PROJECT LINKS
    # =========================================================================
    story.append(Spacer(1, 0.8 * inch))
    story.append(Paragraph("THANK YOU & QUESTIONS", style_dark_badge))
    story.append(Paragraph("BharatProp AI: Indian PropTech Innovation", style_dark_title))
    story.append(Paragraph(
        "A complete, transparent, and mathematically rigorous Automated Valuation Model engineered for the Indian residential housing market.",
        style_dark_sub
    ))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "🌐 <b>Live Web Application:</b> <u>https://real-estate-valuation-kotesh.streamlit.app/</u><br/>"
        "💻 <b>GitHub Repository:</b> <u>https://github.com/koteswaraosabbavarapu-stack/Real-Estate-Valuation</u><br/>"
        "👤 <b>Project Author:</b> Koteswarao Sabbavarapu",
        style_dark_meta
    ))

    # Build Document with SlideCanvas
    doc.build(story, canvasmaker=SlideCanvas)
    print(f"PDF generated successfully: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
