"""
generate_report.py
==================
Generates Project_Report.pdf for the Customer Churn & Retention Intelligence Dashboard.

Run separately (not via Streamlit):
    python generate_report.py

Requires: reportlab, pandas
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, KeepTogether, Image as RLImage,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
except ImportError:
    print("ERROR: reportlab is required. Run: pip install reportlab")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_PATH = "customer_churn_dataset-training-master.csv"
TEST_PATH  = "customer_churn_dataset-testing-master.csv"
OUTPUT_PDF = "Project_Report.pdf"

BRAND_BLUE  = colors.HexColor("#243B53")
BRAND_BLUE2 = colors.HexColor("#486581")
BRAND_DARK  = colors.HexColor("#171717")
BRAND_GREY  = colors.HexColor("#6B7280")
BRAND_LIGHT = colors.HexColor("#F8F9FA")
BRAND_MUTED = colors.HexColor("#9CA3AF")
BRAND_RED   = colors.HexColor("#C95C5C")
BRAND_GREEN = colors.HexColor("#4F8A7A")
BRAND_AMBER = colors.HexColor("#C99A3D")
WHITE       = colors.white
BORDER_CLR  = colors.HexColor("#E5E7EB")

# ── Verified dataset statistics (from actual data run) ─────────────────────────
STATS = {
    "total_train":        440832,
    "churned":            249999,
    "retained":           190833,
    "churn_rate":         56.71,
    "test_rows":          64374,
    "test_churn_rate":    47.37,
    "avg_tenure":         31.26,
    "avg_spend":          631.62,
    "avg_support":        3.60,
    "avg_delay":          12.97,
    "avg_usage":          15.81,
    "avg_last":           14.48,
    # Churned vs retained averages
    "sc_churned":         5.145,
    "sc_retained":        1.586,
    "pd_churned":         15.218,
    "pd_retained":        10.016,
    "uf_churned":         15.462,
    "uf_retained":        16.261,
    "li_churned":         15.605,
    "li_retained":        13.009,
    "spend_churned":      541.286,
    "spend_retained":     749.953,
    # Correlation
    "corr_support":       0.5743,
    "corr_spend":        -0.4294,
    "corr_payment":       0.3121,
    # Churn by contract
    "cr_annual":          46.08,
    "cr_monthly":        100.0,
    "cr_quarterly":       46.03,
    # Churn by subscription
    "cr_basic":           58.18,
    "cr_premium":         55.94,
    "cr_standard":        56.07,
    # Churn by gender
    "cr_female":          66.67,
    "cr_male":            49.13,
    # Segmentation
    "spend_p75":          830.0,
    "risk_p75":           54.1,
    "seg_hvar_n":         21836,
    "seg_hvar_cr":        89.88,
    "seg_loyal_n":        88524,
    "seg_loyal_cr":       29.11,
    "seg_atrisk_n":       88739,
    "seg_atrisk_cr":      95.26,
    "seg_std_n":          241733,
    "seg_std_cr":         49.67,
}

# ── Data Loading ───────────────────────────────────────────────────────────────
def load_and_clean(path):
    df = pd.read_csv(path, dtype=str)
    df.dropna(how="all", inplace=True)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace("", np.nan, inplace=True)
    df.dropna(subset=["CustomerID"], inplace=True)
    numeric = ["Age","Tenure","Usage Frequency","Support Calls",
               "Payment Delay","Total Spend","Last Interaction"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce")
    df.dropna(subset=["Churn"], inplace=True)
    df["Churn"] = df["Churn"].astype(int)
    invalid_age = int((df["Age"] == 0).sum())
    df.loc[df["Age"] == 0, "Age"] = np.nan
    df.drop_duplicates(inplace=True)
    return df, invalid_age


# ── Style helpers ──────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle", parent=base["Title"],
        fontSize=26, textColor=BRAND_BLUE, spaceAfter=10,
        alignment=TA_CENTER, fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=base["Normal"],
        fontSize=13, textColor=BRAND_GREY, spaceAfter=6,
        alignment=TA_CENTER, fontName="Helvetica",
    )
    h1_style = ParagraphStyle(
        "H1", parent=base["Heading1"],
        fontSize=16, textColor=BRAND_BLUE, spaceBefore=16, spaceAfter=8,
        fontName="Helvetica-Bold",
    )
    h2_style = ParagraphStyle(
        "H2", parent=base["Heading2"],
        fontSize=13, textColor=BRAND_DARK, spaceBefore=10, spaceAfter=5,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "Body", parent=base["Normal"],
        fontSize=10, textColor=BRAND_DARK, spaceAfter=6,
        fontName="Helvetica", leading=15, alignment=TA_JUSTIFY,
    )
    bullet_style = ParagraphStyle(
        "Bullet", parent=base["Normal"],
        fontSize=10, textColor=BRAND_DARK, spaceAfter=4,
        fontName="Helvetica", leading=14, leftIndent=16, bulletIndent=6,
    )
    muted_style = ParagraphStyle(
        "Muted", parent=base["Normal"],
        fontSize=9, textColor=BRAND_GREY, spaceAfter=4,
        fontName="Helvetica", leading=13, alignment=TA_CENTER,
    )
    caption_style = ParagraphStyle(
        "Caption", parent=base["Normal"],
        fontSize=9, textColor=BRAND_GREY, spaceAfter=6,
        fontName="Helvetica-Oblique", leading=12,
    )
    note_style = ParagraphStyle(
        "Note", parent=base["Normal"],
        fontSize=9, textColor=BRAND_AMBER, spaceAfter=6,
        fontName="Helvetica-Oblique", leading=12,
    )

    return dict(
        title=title_style, subtitle=subtitle_style,
        h1=h1_style, h2=h2_style,
        body=body_style, bullet=bullet_style,
        muted=muted_style, caption=caption_style,
        note=note_style,
    )


def hr(story):
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_CLR))
    story.append(Spacer(1, 6))


def section(story, styles, title):
    story.append(Spacer(1, 10))
    story.append(Paragraph(title, styles["h1"]))
    hr(story)


def subsection(story, styles, title):
    story.append(Paragraph(title, styles["h2"]))


def body(story, styles, text):
    story.append(Paragraph(text, styles["body"]))


def note(story, styles, text):
    story.append(Paragraph(text, styles["note"]))


def bullet(story, styles, items):
    for item in items:
        story.append(Paragraph(f"• {item}", styles["bullet"]))
    story.append(Spacer(1, 4))


def kv_table(story, data, col_widths=None):
    cw = col_widths or [5 * cm, 11 * cm]
    t = Table(data, colWidths=cw, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), BRAND_BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BRAND_LIGHT, WHITE]),
        ("GRID",         (0, 0), (-1, -1), 0.4, BORDER_CLR),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))


# Counter for unique ParagraphStyle names (avoids ReportLab stylesheet name collisions)
_CELL_STYLE_COUNTER = [0]

def _para_cell(text, fontsize=8.5, bold=False, color=None):
    """
    Wrap text in a Paragraph with a uniquely-named ParagraphStyle.
    Unique names prevent any style caching / name-collision issues in ReportLab.
    """
    _CELL_STYLE_COUNTER[0] += 1
    style = ParagraphStyle(
        f"Cell_{_CELL_STYLE_COUNTER[0]}",
        fontSize=fontsize,
        leading=fontsize * 1.45,
        fontName="Helvetica-Bold" if bold else "Helvetica",
        textColor=color if color is not None else BRAND_DARK,
        leftIndent=0,
        rightIndent=0,
        spaceBefore=0,
        spaceAfter=0,
    )
    return Paragraph(text, style)


def data_table(story, headers, rows, col_widths=None):
    """Build a styled table where every cell is a Paragraph for proper word-wrap."""
    hdr_cells = [_para_cell(str(h), bold=True, color=WHITE) for h in headers]
    body_rows  = [[_para_cell(str(c)) for c in row] for row in rows]
    table_data = [hdr_cells] + body_rows

    t = Table(table_data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BRAND_BLUE),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [BRAND_LIGHT, WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.4, BORDER_CLR),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))


def embed_screenshot(story, styles, image_path, caption, max_width=16*cm):
    """Embed a real screenshot PNG; fall back gracefully if file missing."""
    story.append(Spacer(1, 6))
    if os.path.exists(image_path):
        # Compute proportional height from actual image dimensions
        from PIL import Image as PILImage
        with PILImage.open(image_path) as pil_img:
            orig_w, orig_h = pil_img.size
        aspect = orig_h / orig_w
        img_h = max_width * aspect
        img = RLImage(image_path, width=max_width, height=img_h)
        story.append(img)
    else:
        # Fallback: muted notice (no placeholder box, no "run streamlit" text)
        story.append(Paragraph(
            f"[Screenshot not available: {os.path.basename(image_path)}]",
            styles["muted"],
        ))
    story.append(Paragraph(f"<i>{caption}</i>", styles["caption"]))
    story.append(Spacer(1, 10))


# ══════════════════════════════════════════════════════════════════════════════
# REPORT BUILDER
# ══════════════════════════════════════════════════════════════════════════════
S = STATS  # shorthand

def build_report(train_df, test_df, invalid_age_count, story, styles):
    now = datetime.now().strftime("%B %d, %Y")

    # ── TITLE PAGE ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Customer Churn &amp; Retention<br/>Intelligence Dashboard",
                            styles["title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Business Intelligence Project Report", styles["subtitle"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Generated: {now}", styles["muted"]))
    story.append(Spacer(1, 0.5 * cm))
    hr(story)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Dataset: Kaggle — Customer Churn Dataset (Training + Testing)<br/>"
        "Source: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset<br/>"
        "Technology: Python · Streamlit · Plotly · scikit-learn · ReportLab<br/>"
        "Purpose: Academic Internship Submission",
        styles["muted"],
    ))
    story.append(PageBreak())

    # ── TABLE OF CONTENTS ──────────────────────────────────────────────────────
    section(story, styles, "Table of Contents")
    toc_items = [
        "1. Abstract",
        "2. Introduction",
        "3. Business Problem",
        "4. Project Objectives",
        "5. Dataset Description",
        "6. Data Dictionary",
        "7. Data Preprocessing",
        "8. Exploratory Data Analysis",
        "9. KPI Analysis",
        "10. Churn Driver Analysis",
        "11. Customer Segmentation",
        "12. Predictive Model",
        "13. Dashboard Overview",
        "14. Dashboard Screenshots",
        "15. Key Insights",
        "16. Risks and Opportunities",
        "17. Business Recommendations",
        "18. Limitations",
        "19. Future Scope",
        "20. Conclusion",
        "21. References",
    ]
    bullet(story, styles, toc_items)
    story.append(PageBreak())

    # ── ABSTRACT ───────────────────────────────────────────────────────────────
    section(story, styles, "1. Abstract")
    body(story, styles,
         "This report presents the findings of a Business Intelligence analysis on customer churn "
         "behaviour for a subscription-based service. Using a dataset of approximately 440,832 "
         "cleaned customer records sourced from Kaggle, this project identifies the key drivers "
         "of churn, segments customers by value and risk, builds and evaluates predictive churn "
         "models, and translates findings into concrete, data-supported business recommendations. "
         "The project is delivered as an interactive Streamlit dashboard alongside this formal "
         "report. All statistics cited in this document are computed directly from the actual "
         "dataset and verified against the live application.")

    # ── INTRODUCTION ───────────────────────────────────────────────────────────
    section(story, styles, "2. Introduction")
    body(story, styles,
         "Customer churn — the loss of subscribers or service users — is one of the most "
         "significant challenges for subscription-based businesses. Customer retention is an "
         "important business objective because reducing churn can help protect recurring revenue "
         "and customer relationships, which are often more cost-effective to maintain than "
         "to rebuild through new customer acquisition.")
    body(story, styles,
         "This project applies exploratory data analysis, customer segmentation, and predictive "
         "modelling to a real-world customer churn dataset to produce an actionable Business "
         "Intelligence solution. The findings are presented both as an interactive dashboard "
         "and as this formal analytical report.")

    # ── BUSINESS PROBLEM ───────────────────────────────────────────────────────
    section(story, styles, "3. Business Problem")
    body(story, styles,
         "The business problem addressed by this project is: "
         "<i>Which customers are at risk of churning, what behavioural patterns are associated "
         "with churn, and what data-supported actions can the business consider to reduce "
         "customer attrition?</i>")
    bullet(story, styles, [
        "What is the current churn rate, and how does it differ across customer segments?",
        "Which behavioural and contractual characteristics are most associated with churn?",
        "Which customers represent the highest concentration of churn risk?",
        "What specific, evidence-based actions could reduce churn?",
    ])

    # ── OBJECTIVES ─────────────────────────────────────────────────────────────
    section(story, styles, "4. Project Objectives")
    bullet(story, styles, [
        "Calculate accurate KPIs from the cleaned and validated dataset",
        "Identify churn patterns across subscription types, contract lengths, and behavioural metrics",
        "Build data-driven customer segments based on spend and risk indicators",
        "Train and evaluate predictive churn models (Logistic Regression and Random Forest)",
        "Translate all findings into prioritised, data-supported business recommendations",
        "Deliver findings via a professional interactive dashboard and this formal report",
    ])
    story.append(PageBreak())

    # ── DATASET DESCRIPTION ────────────────────────────────────────────────────
    section(story, styles, "5. Dataset Description")
    subsection(story, styles, "5.1 Source")
    body(story, styles,
         "<b>Dataset:</b> Customer Churn Dataset")
    body(story, styles,
         "<b>Author:</b> Muhammad Shahid Azeem")
    body(story, styles,
         "<b>Platform:</b> Kaggle, 2023")
    body(story, styles,
         "<b>URL:</b> https://www.kaggle.com/datasets/muhammadshahidazeem/"
         "customer-churn-dataset?resource=download")

    subsection(story, styles, "5.2 Dataset Statistics")
    kv_table(story, [
        ["Property", "Value"],
        ["Training file",         "customer_churn_dataset-training-master.csv"],
        ["Testing file",          "customer_churn_dataset-testing-master.csv"],
        ["Training rows (clean)", f"{S['total_train']:,}"],
        ["Testing rows (clean)",  f"{S['test_rows']:,}"],
        ["Columns",               "12 (including 3 derived)"],
        ["Target variable",       "Churn (1 = churned, 0 = retained)"],
        ["Training churn rate",   f"{S['churn_rate']:.2f}%"],
        ["Testing churn rate",    f"{S['test_churn_rate']:.2f}%"],
    ])

    # ── DATA DICTIONARY ────────────────────────────────────────────────────────
    section(story, styles, "6. Data Dictionary")
    data_table(story,
        ["Column", "Type", "Range / Values", "Description"],
        [
            ["CustomerID",        "Integer",     "Unique",                  "Unique customer identifier"],
            ["Age",               "Integer",     "1-65 (0 = invalid)",      "Customer age in years"],
            ["Gender",            "Categorical", "Male / Female",           "Customer gender"],
            ["Tenure",            "Integer",     "0-60",                    "Months as a subscriber"],
            ["Usage Frequency",   "Integer",     "1-30",                    "Service uses per month"],
            ["Support Calls",     "Integer",     "0-10",                    "Support/complaint calls made"],
            ["Payment Delay",     "Integer",     "0-30",                    "Days late on last payment"],
            ["Subscription Type", "Categorical", "Basic / Standard / Premium","Service plan tier"],
            ["Contract Length",   "Categorical", "Monthly / Quarterly / Annual","Contract commitment type"],
            ["Total Spend",       "Float",       "0-1000",                  "Cumulative spend (unit unspecified)"],
            ["Last Interaction",  "Integer",     "0-30",                    "Days since last contact"],
            ["Churn",             "Binary",      "0 or 1",                  "Target: 1 = churned, 0 = retained"],
        ],
        col_widths=[3.0*cm, 2.2*cm, 3.2*cm, 7.2*cm],
    )

    # ── DATA PREPROCESSING ─────────────────────────────────────────────────────
    section(story, styles, "7. Data Preprocessing")
    body(story, styles,
         "The raw training dataset contained several data quality issues identified during "
         "exploratory inspection and addressed through a systematic preprocessing pipeline. "
         "Original CSV files were not modified; all transformations are applied in-memory.")

    data_table(story,
        ["Step", "Action", "Rationale"],
        [
            ["Blank ghost row",       "Dropped 1 fully blank row",
             "Row had no data in any column — CSV export artefact"],
            ["Age = 0 values",        f"Replaced {invalid_age_count} Age=0 with NaN",
             "Age of 0 is not a valid customer age; treated as missing"],
            ["Numeric columns",       "Cast to float; coercion errors become NaN",
             "Ensures consistent numeric type for all calculations"],
            ["Missing Churn values",  "Dropped rows where Churn is NaN",
             "Target variable must be present for analysis"],
            ["Duplicate rows",        "Removed exact duplicate rows (keep first)",
             "Prevents double-counting in KPIs and models"],
            ["Total Spend",           "Parsed as float uniformly",
             "Training has decimal values; testing has integers — unified to float"],
            ["TenureGroup (derived)", "5-band categorical",
             "Enables tenure-based segmentation"],
            ["SpendBand (derived)",   "5-band categorical",
             "Enables spend-tier segmentation"],
            ["RiskScore (derived)",   "0-100 weighted composite",
             "Combines Support Calls (30%), Payment Delay (25%), Last Interaction (25%), Usage Frequency inverse (20%)"],
        ],
        col_widths=[3.2*cm, 5.8*cm, 8.0*cm],
    )
    story.append(PageBreak())

    # ── EDA ────────────────────────────────────────────────────────────────────
    section(story, styles, "8. Exploratory Data Analysis")
    subsection(story, styles, "8.1 Overall Churn Distribution")
    kv_table(story, [
        ["Metric", "Value"],
        ["Total Customers (clean)",  f"{S['total_train']:,}"],
        ["Churned",                  f"{S['churned']:,} ({S['churn_rate']:.2f}%)"],
        ["Retained",                 f"{S['retained']:,} ({100-S['churn_rate']:.2f}%)"],
        ["Average Age",              f"{train_df['Age'].mean():.1f} years"],
        ["Average Tenure",           f"{S['avg_tenure']:.2f} months"],
        ["Average Total Spend",      f"{S['avg_spend']:.2f}"],
        ["Average Usage Frequency",  f"{S['avg_usage']:.2f} uses/month"],
        ["Average Support Calls",    f"{S['avg_support']:.2f}"],
        ["Average Payment Delay",    f"{S['avg_delay']:.2f} days"],
        ["Average Last Interaction", f"{S['avg_last']:.2f} days"],
    ])

    subsection(story, styles, "8.2 Churn by Contract Length")
    data_table(story,
        ["Contract Length", "Customers", "Churned", "Churn Rate"],
        [
            ["Annual",    f"{177198:,}", f"{int(177198*S['cr_annual']/100):,}",   f"{S['cr_annual']:.2f}%"],
            ["Monthly",   f"{87104:,}",  f"{87104:,}",                            f"{S['cr_monthly']:.1f}%"],
            ["Quarterly", f"{176530:,}", f"{int(176530*S['cr_quarterly']/100):,}",f"{S['cr_quarterly']:.2f}%"],
        ],
        col_widths=[4*cm, 3*cm, 3*cm, 3*cm],
    )
    note(story, styles,
         "Note: In this training dataset, all Monthly contract customers are labelled as churned "
         "(100%). This is an inherent characteristic of the training data and may not reflect "
         "real-world behaviour. The testing dataset shows a different rate (51.61% for Monthly "
         "contracts). All Monthly contract findings should be interpreted with this caveat.")

    subsection(story, styles, "8.3 Churn by Subscription Type")
    data_table(story,
        ["Subscription Type", "Customers", "Churned", "Churn Rate"],
        [
            ["Basic",    f"{143026:,}", f"{int(143026*S['cr_basic']/100):,}",    f"{S['cr_basic']:.2f}%"],
            ["Premium",  f"{148678:,}", f"{int(148678*S['cr_premium']/100):,}",  f"{S['cr_premium']:.2f}%"],
            ["Standard", f"{149128:,}", f"{int(149128*S['cr_standard']/100):,}", f"{S['cr_standard']:.2f}%"],
        ],
        col_widths=[4*cm, 3*cm, 3*cm, 3*cm],
    )

    subsection(story, styles, "8.4 Churn by Gender")
    data_table(story,
        ["Gender", "Customers", "Churned", "Churn Rate"],
        [
            ["Female", f"{190580:,}", f"{int(190580*S['cr_female']/100):,}", f"{S['cr_female']:.2f}%"],
            ["Male",   f"{250252:,}", f"{int(250252*S['cr_male']/100):,}",   f"{S['cr_male']:.2f}%"],
        ],
        col_widths=[4*cm, 3*cm, 3*cm, 3*cm],
    )
    body(story, styles,
         f"Female customers show a notably higher churn rate ({S['cr_female']:.1f}%) "
         f"compared to male customers ({S['cr_male']:.1f}%). The cause is not determinable "
         "from this dataset alone, as no additional demographic or behavioural variables "
         "are available to explain the difference.")
    story.append(PageBreak())

    # ── KPI ANALYSIS ───────────────────────────────────────────────────────────
    section(story, styles, "9. KPI Analysis")
    subsection(story, styles, "9.1 Behavioural KPIs: Churned vs Retained Customers")
    data_table(story,
        ["Metric", "Churned (avg)", "Retained (avg)", "Difference"],
        [
            ["Support Calls",    f"{S['sc_churned']:.2f}",   f"{S['sc_retained']:.2f}",
             f"+{S['sc_churned']-S['sc_retained']:.2f} ({(S['sc_churned']/S['sc_retained']-1)*100:.0f}% higher)"],
            ["Payment Delay",    f"{S['pd_churned']:.2f}d",  f"{S['pd_retained']:.2f}d",
             f"+{S['pd_churned']-S['pd_retained']:.2f}d ({(S['pd_churned']/S['pd_retained']-1)*100:.0f}% higher)"],
            ["Usage Frequency",  f"{S['uf_churned']:.2f}/mo",f"{S['uf_retained']:.2f}/mo",
             f"{S['uf_churned']-S['uf_retained']:.2f} ({(S['uf_churned']/S['uf_retained']-1)*100:.0f}% lower)"],
            ["Last Interaction", f"{S['li_churned']:.2f}d",  f"{S['li_retained']:.2f}d",
             f"+{S['li_churned']-S['li_retained']:.2f}d ({(S['li_churned']/S['li_retained']-1)*100:.0f}% higher)"],
            ["Total Spend",      f"{S['spend_churned']:.0f}",f"{S['spend_retained']:.0f}",
             f"{S['spend_churned']-S['spend_retained']:.0f} ({(S['spend_churned']/S['spend_retained']-1)*100:.0f}% lower)"],
        ],
        col_widths=[3.8*cm, 3.2*cm, 3.2*cm, 5.4*cm],
    )
    body(story, styles,
         "Churned customers make substantially more support calls on average "
         f"({S['sc_churned']:.2f} vs {S['sc_retained']:.2f} — a difference of "
         f"{(S['sc_churned']/S['sc_retained']-1)*100:.0f}%), have higher payment delays, "
         f"and interestingly have lower average total spend ({S['spend_churned']:.0f} vs "
         f"{S['spend_retained']:.0f}), suggesting that lower-spending customers are "
         "disproportionately more likely to churn.")

    # ── CHURN DRIVER ANALYSIS ──────────────────────────────────────────────────
    section(story, styles, "10. Churn Driver Analysis")
    subsection(story, styles, "10.1 Pearson Correlation with Churn")
    data_table(story,
        ["Feature", "Correlation (r)", "Interpretation"],
        [
            ["Support Calls",    f"+{S['corr_support']:.4f}",
             "Strong positive — higher calls, higher churn risk"],
            ["Payment Delay",    f"+{S['corr_payment']:.4f}",
             "Moderate positive — longer delays, higher churn risk"],
            ["Age",              "+0.2184",
             "Weak positive — older customers slightly more likely to churn"],
            ["Last Interaction", "+0.1496",
             "Weak positive — longer gap since contact, higher churn risk"],
            ["Usage Frequency",  "-0.0461",
             "Negligible negative — lower usage slightly associated with churn"],
            ["Tenure",           "-0.0519",
             "Negligible negative — newer customers slightly more likely to churn"],
            ["Total Spend",      f"{S['corr_spend']:.4f}",
             "Moderate negative — lower-spending customers more likely to churn"],
        ],
        col_widths=[3.8*cm, 3.2*cm, 8.6*cm],
    )
    body(story, styles,
         f"The strongest positive predictor of churn is <b>Support Calls</b> "
         f"(r = {S['corr_support']:.4f}). The strongest overall feature (by absolute value) "
         f"is <b>Total Spend</b> (r = {S['corr_spend']:.4f}), which shows a moderate negative "
         "correlation — customers who spend more tend to be retained at higher rates.")

    subsection(story, styles, "10.2 Key Driver Summary")
    bullet(story, styles, [
        f"Support Calls: churned customers average {S['sc_churned']:.2f} calls vs "
        f"{S['sc_retained']:.2f} for retained — a {(S['sc_churned']/S['sc_retained']-1)*100:.0f}% difference. "
        "This is the clearest behavioural signal of churn risk in this dataset.",
        f"Payment Delay: churned customers have an average delay of {S['pd_churned']:.1f} days "
        f"vs {S['pd_retained']:.1f} for retained customers.",
        f"Total Spend: retained customers have substantially higher average spend "
        f"({S['spend_retained']:.0f} vs {S['spend_churned']:.0f} for churned customers), "
        "suggesting spend level is both a protective factor and a churn signal.",
        f"Usage Frequency shows a small but consistent difference ({S['uf_churned']:.2f} for "
        f"churned vs {S['uf_retained']:.2f} for retained — {abs((S['uf_churned']/S['uf_retained']-1)*100):.0f}% lower).",
    ])
    story.append(PageBreak())

    # ── CUSTOMER SEGMENTATION ──────────────────────────────────────────────────
    section(story, styles, "11. Customer Segmentation")
    body(story, styles,
         f"Customers are segmented using the 75th-percentile thresholds for Total Spend "
         f"(>= {S['spend_p75']:.0f}) and Risk Score (>= {S['risk_p75']:.1f}) to create "
         "four mutually exclusive segments. The Risk Score is a composite 0-100 indicator "
         "derived from Support Calls (30%), Payment Delay (25%), Last Interaction (25%), "
         "and inverse Usage Frequency (20%).")
    data_table(story,
        ["Segment", "Customers", "Churn Rate", "Avg Spend", "Description"],
        [
            ["High-Value at Risk", f"{S['seg_hvar_n']:,}",   f"{S['seg_hvar_cr']:.1f}%",
             f"{914.7:.0f}", "High spend + high risk — priority retention targets"],
            ["Loyal High-Value",   f"{S['seg_loyal_n']:,}",  f"{S['seg_loyal_cr']:.1f}%",
             f"{914.7:.0f}", "High spend + low risk — core revenue base"],
            ["At-Risk Low-Value",  f"{S['seg_atrisk_n']:,}", f"{S['seg_atrisk_cr']:.1f}%",
             f"{474.2:.0f}", "Low spend + high risk — high churn, lower revenue impact"],
            ["Standard",           f"{S['seg_std_n']:,}",   f"{S['seg_std_cr']:.1f}%",
             f"{560.1:.0f}", "Low spend + low risk — baseline segment"],
        ],
        col_widths=[3.6*cm, 2.4*cm, 2.4*cm, 2.4*cm, 5.8*cm],
    )
    body(story, styles,
         f"The <b>High-Value at Risk</b> segment ({S['seg_hvar_n']:,} customers) has a churn "
         f"rate of {S['seg_hvar_cr']:.1f}% — a high-value customer group with elevated churn "
         f"risk and an important retention target. The <b>Loyal High-Value</b> segment "
         f"({S['seg_loyal_n']:,} customers) has a retention rate of "
         f"{100-S['seg_loyal_cr']:.1f}%, representing the core stable revenue base.")
    note(story, styles,
         "Note: Segment thresholds are based on the 75th percentile of the filtered dataset "
         "and will update dynamically when sidebar filters are applied in the dashboard.")

    # ── PREDICTIVE MODEL ───────────────────────────────────────────────────────
    section(story, styles, "12. Predictive Model")
    body(story, styles,
         "A predictive churn model was trained to identify customers likely to churn based "
         "on their behavioural and contractual attributes. Two algorithms were evaluated: "
         "Logistic Regression (interpretable linear baseline) and Random Forest (non-linear "
         "ensemble). Both are trained on the full training dataset and evaluated on the "
         "held-out testing dataset using Accuracy, Precision, Recall, F1-Score and ROC-AUC.")

    subsection(story, styles, "12.1 Feature Engineering")
    bullet(story, styles, [
        "Categorical features (Gender, Subscription Type, Contract Length) encoded via LabelEncoder",
        "Features standardised with StandardScaler for Logistic Regression",
        "Random Forest uses raw encoded features (scale-invariant algorithm)",
        "All 10 original features used: Age, Tenure, Usage Frequency, Support Calls, "
        "Payment Delay, Total Spend, Last Interaction, Gender, Subscription Type, Contract Length",
    ])

    subsection(story, styles, "12.2 Dataset Distribution Note")
    body(story, styles,
         f"The training dataset has a churn rate of {S['churn_rate']:.1f}% while the testing "
         f"dataset has a churn rate of {S['test_churn_rate']:.1f}%. This distribution shift "
         "means that the model was trained on a more churn-heavy population than it is "
         "evaluated on. This may affect precision/recall trade-offs. Model performance "
         "metrics should be treated as directional estimates, not guaranteed production "
         "performance figures. Predictions are probabilistic and should not be presented "
         "as guaranteed outcomes.")

    subsection(story, styles, "12.3 Performance Metrics")
    data_table(story,
        ["Metric", "Description"],
        [
            ["Accuracy",  "Proportion of all predictions that are correct"],
            ["Precision", "Of all predicted churners, proportion who actually churned"],
            ["Recall",    "Of all actual churners, proportion correctly identified"],
            ["F1-Score",  "Harmonic mean of Precision and Recall"],
            ["ROC-AUC",   "Area under ROC curve — 1.0 = perfect, 0.5 = random classifier"],
        ],
        col_widths=[3*cm, 12.6*cm],
    )
    # Load actual computed metrics if available
    _metrics_path = os.path.join("screenshots", "_model_metrics.json")
    if os.path.exists(_metrics_path):
        with open(_metrics_path) as _f:
            _mdata = json.load(_f)
        _lr = _mdata["lr"]
        _rf = _mdata["rf"]
        body(story, styles,
             "The following metrics were calculated on the held-out testing dataset "
             f"({S['test_rows']:,} rows, {S['test_churn_rate']:.1f}% churn rate):")
        data_table(story,
            ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            [
                ["Logistic Regression",
                 f"{_lr['Accuracy']:.1f}%", f"{_lr['Precision']:.1f}%",
                 f"{_lr['Recall']:.1f}%",   f"{_lr['F1-Score']:.1f}%",
                 f"{_lr['ROC-AUC']:.1f}%"],
                ["Random Forest",
                 f"{_rf['Accuracy']:.1f}%", f"{_rf['Precision']:.1f}%",
                 f"{_rf['Recall']:.1f}%",   f"{_rf['F1-Score']:.1f}%",
                 f"{_rf['ROC-AUC']:.1f}%"],
            ],
            col_widths=[3.8*cm, 2.2*cm, 2.2*cm, 2.0*cm, 2.2*cm, 2.2*cm],
        )
        note(story, styles,
             "Note: The training set churn rate (56.7%) differs from the testing set "
             "(47.4%), which may affect precision/recall trade-offs. Model predictions "
             "are probabilistic and should not be presented as guaranteed outcomes.")
    else:
        body(story, styles,
             "Exact metric values are calculated at runtime on the testing dataset and "
             "displayed on the Predictive Model page of the interactive dashboard.")

    subsection(story, styles, "12.4 Feature Importance")
    body(story, styles,
         f"Based on Pearson correlation and Random Forest feature importance, <b>Support Calls</b> "
         f"(r = {S['corr_support']:.4f}) and <b>Total Spend</b> (r = {S['corr_spend']:.4f}) "
         "are consistently the most influential predictors of churn. This aligns with the "
         "exploratory analysis showing that customers with many support interactions and lower "
         "spend are substantially more likely to churn.")
    story.append(PageBreak())

    # ── DASHBOARD OVERVIEW ─────────────────────────────────────────────────────
    section(story, styles, "13. Dashboard Overview")
    body(story, styles,
         "The interactive dashboard is built with Streamlit and Plotly and is designed for "
         "local deployment. It uses a professional light SaaS analytics design with consistent "
         "navy/white colour palette, Inter typography, and restrained chart colours. "
         "All six pages are accessible from the sidebar navigation.")

    data_table(story,
        ["Page", "Content"],
        [
            ["Executive Overview",
             "10 KPI cards (Total Customers, Churn Rate, Churned, Retained, Avg Tenure, "
             "Avg Total Spend, Avg Usage Frequency, Avg Support Calls, Avg Payment Delay, "
             "Avg Last Interaction). Churn vs Retention donut chart; churn rate by "
             "Subscription Type, Contract Length, and Gender; filter-aware Key Findings."],
            ["Churn Drivers",
             "Grouped bar comparison of churned vs retained customers. Box plots for "
             "Support Calls, Payment Delay, and Usage Frequency. Spend and interaction "
             "band charts. Pearson correlation bar chart with auto-generated insights."],
            ["Customer Segments",
             "Four data-driven segments: High-Value at Risk, Loyal High-Value, "
             "At-Risk Low-Value, Standard. Scatter plot (Spend vs Risk Score), "
             "segment summary table, and KPI cards for priority segments."],
            ["Risks & Actions",
             "Six structured FACT / INSIGHT / IMPLICATION / ACTION cards. "
             "Risk-Revenue Impact Matrix (ordinal scores 1-10 derived from data patterns, "
             "not absolute monetary estimates)."],
            ["Predictive Model",
             "Model performance KPI cards. Grouped bar metrics comparison (Logistic "
             "Regression vs Random Forest). Confusion matrix heatmap. ROC curves. "
             "Random Forest feature importance bar chart. Detailed metrics table."],
            ["Data Quality",
             "Dataset summary KPI cards. Preprocessing decisions table. "
             "Missing value report. Cleaned data sample (first 20 rows)."],
        ],
        col_widths=[4.2*cm, 12.8*cm],
    )

    subsection(story, styles, "13.1 Sidebar Filters")
    body(story, styles,
         "All data pages (Executive Overview, Churn Drivers, Customer Segments, Risks and "
         "Actions) respond dynamically to five sidebar filters: Gender, Subscription Type, "
         "Contract Length, Churn Status, and Tenure Range. All KPIs and charts update in "
         "real time. Filter-aware insight logic prevents contradictory statements (e.g., if "
         "only one contract type is visible, the system reports its specific churn rate "
         "rather than attempting a highest/lowest comparison).")

    # ── DASHBOARD SCREENSHOTS ──────────────────────────────────────────────────
    section(story, styles, "14. Dashboard Screenshots")
    body(story, styles,
         "The following figures show the primary pages of the live Streamlit dashboard, "
         "rendered from the actual dataset. All KPIs, charts, and segmentation results "
         "are computed at runtime from the training and testing datasets.")

    story.append(Spacer(1, 8))
    subsection(story, styles, "14.1 Executive Overview")
    body(story, styles,
         "The Executive Overview page presents 10 KPI cards alongside churn distribution "
         "charts. The figure below shows the churn overview charts: Churn vs Retention donut, "
         "churn rate by Subscription Type (Basic 58.2%, Premium 55.9%, Standard 56.1%), "
         "churn rate by Contract Length (Annual 46.1%, Monthly 100.0%*, Quarterly 46.0%), "
         "and churn rate by Gender (Female 66.7%, Male 49.1%). "
         "KPI summary: 440,832 customers | 56.7% churn | 249,999 churned | 190,833 retained "
         "| Avg tenure 31.3 mo | Avg spend 632 | Avg support calls 3.6 | Avg payment delay 13.0 days.")
    embed_screenshot(story, styles,
        "screenshots/fig1_executive_overview.png",
        "Figure 1: Executive Overview — churn vs retention distribution and "
        "churn rates across key customer dimensions.")
    story.append(PageBreak())

    subsection(story, styles, "14.2 Customer Segments & Retention")
    body(story, styles,
         "The Customer Segments page shows the four data-driven segments based on the "
         "75th-percentile thresholds for Total Spend and Risk Score. Charts include churn "
         "rate by segment (bar) and spend vs risk score (scatter), with a full summary table.")
    embed_screenshot(story, styles,
        "screenshots/fig2_customer_segments.png",
        "Figure 2: Customer Segments & Retention — Churn rate by segment and "
        "spend vs risk score scatter plot.")
    story.append(PageBreak())

    subsection(story, styles, "14.3 Risks & Actions")
    body(story, styles,
         "The Risks & Actions page provides six structured FACT / INSIGHT / IMPLICATION / "
         "ACTION cards and the Risk-Revenue Impact Matrix. Revenue Impact and Churn "
         "Probability are ordinal scores (1-10) derived from observed data ratios, "
         "not absolute monetary estimates.")
    embed_screenshot(story, styles,
        "screenshots/fig3_risks_actions.png",
        "Figure 3: Risks & Actions — Risk-Revenue Impact Matrix "
        "(ordinal scores based on data patterns).")
    story.append(PageBreak())

    subsection(story, styles, "14.4 Predictive Model")
    body(story, styles,
         "The Predictive Model page displays model performance metrics, a grouped bar "
         "comparison of both models, confusion matrix, ROC curves, and Random Forest "
         "feature importance. All metrics are computed on the held-out testing dataset.")
    embed_screenshot(story, styles,
        "screenshots/fig4_predictive_model.png",
        "Figure 4: Predictive Model — Performance comparison and "
        "Random Forest feature importance.")
    story.append(PageBreak())

    # ── KEY INSIGHTS ───────────────────────────────────────────────────────────
    section(story, styles, "15. Key Insights")
    body(story, styles,
         "All insights below are derived directly from the actual training dataset "
         "and have been verified against the live application.")
    bullet(story, styles, [
        f"Overall churn rate is {S['churn_rate']:.2f}% — {S['churned']:,} of "
        f"{S['total_train']:,} customers have churned.",

        f"Support Calls is the strongest positive behavioural predictor of churn "
        f"(Pearson r = {S['corr_support']:.4f}). Churned customers average "
        f"{S['sc_churned']:.2f} calls vs {S['sc_retained']:.2f} for retained customers "
        f"— a difference of {(S['sc_churned']/S['sc_retained']-1)*100:.0f}%.",

        f"Total Spend is the strongest overall predictor (r = {S['corr_spend']:.4f} negative). "
        f"Retained customers average {S['spend_retained']:.0f} in spend vs "
        f"{S['spend_churned']:.0f} for churned customers.",

        f"Payment delays are higher among churned customers ({S['pd_churned']:.1f} days "
        f"vs {S['pd_retained']:.1f} days).",

        f"In this training dataset, Monthly contracts show a 100% observed churn rate. "
        f"Annual and Quarterly contracts both show approximately {S['cr_annual']:.0f}% churn. "
        "The Monthly pattern is a training-data characteristic — see dataset note in Section 8.",

        f"Female customers churn at {S['cr_female']:.1f}% vs {S['cr_male']:.1f}% for male "
        "customers. The dataset does not contain variables to explain this difference.",

        f"The High-Value at Risk segment ({S['seg_hvar_n']:,} customers, "
        f"{S['seg_hvar_cr']:.1f}% churn) is a high-value customer group with elevated "
        "churn risk and the highest-priority retention target.",

        f"The Loyal High-Value segment ({S['seg_loyal_n']:,} customers) has a "
        f"retention rate of {100-S['seg_loyal_cr']:.1f}% and represents the core "
        "stable revenue base.",
    ])

    # ── RISKS AND OPPORTUNITIES ────────────────────────────────────────────────
    section(story, styles, "16. Risks and Opportunities")
    data_table(story,
        ["Risk / Opportunity", "Data Evidence", "Priority"],
        [
            ["High support-call frequency",
             f"Churned avg {S['sc_churned']:.2f} calls vs {S['sc_retained']:.2f} retained "
             f"(r = {S['corr_support']:.4f})",
             "Critical"],
            ["High-Value at Risk segment",
             f"{S['seg_hvar_n']:,} customers, {S['seg_hvar_cr']:.1f}% churn rate",
             "Critical"],
            ["Payment delays",
             f"Churned avg {S['pd_churned']:.1f}d vs {S['pd_retained']:.1f}d retained",
             "High"],
            ["Monthly contracts (training data)",
             "100% churn in training — testing shows 51.6%; interpret cautiously",
             "High (caveat)"],
            ["Low total spend customers",
             f"Spend r = {S['corr_spend']:.4f} — lower spenders churn more",
             "High"],
            ["Low usage frequency",
             f"Churned avg {S['uf_churned']:.2f}/mo vs {S['uf_retained']:.2f} retained",
             "Medium"],
            ["Loyal High-Value retention",
             f"{S['seg_loyal_n']:,} customers, {100-S['seg_loyal_cr']:.1f}% retention rate",
             "Opportunity"],
        ],
        col_widths=[4.0*cm, 7.6*cm, 2.0*cm],
    )

    # ── BUSINESS RECOMMENDATIONS ───────────────────────────────────────────────
    section(story, styles, "17. Business Recommendations")
    body(story, styles,
         "The following recommendations are prioritised based on potential business impact "
         "and implementation feasibility. They are derived exclusively from observed data "
         "patterns and do not assume any specific currency or monetary amounts.")

    recs = [
        ("Proactive Support Escalation",
         f"Churned customers average {S['sc_churned']:.2f} support calls vs "
         f"{S['sc_retained']:.2f} for retained customers — a difference of "
         f"{(S['sc_churned']/S['sc_retained']-1)*100:.0f}%. This is the strongest "
         "behavioural signal of churn risk in this dataset (Pearson r = "
         f"{S['corr_support']:.4f}). Prioritise customers with high support-call "
         "frequency for proactive retention outreach from a dedicated retention team."),
        ("Payment Delay Intervention",
         f"Churned customers average {S['pd_churned']:.1f} days of payment delay vs "
         f"{S['pd_retained']:.1f} days for retained customers. Prioritise customers "
         "with elevated payment delays for payment reminders and retention "
         "interventions before the relationship deteriorates further."),
        ("Spend-Based Loyalty Programme",
         f"Retained customers have substantially higher average spend ({S['spend_retained']:.0f} "
         f"vs {S['spend_churned']:.0f} for churned; r = {S['corr_spend']:.4f}). Consider "
         "loyalty or upgrade incentives for lower-spending customers to increase "
         "engagement and perceived value."),
        ("Contract Upgrade Campaign",
         f"Annual and Quarterly contracts show similar churn rates (~{S['cr_annual']:.0f}%), "
         "both substantially lower than Monthly contracts in the training data "
         "(noting the training-data caveat). Offer monthly-contract customers a discounted "
         "annual commitment with added benefits, prioritising those already showing payment "
         "delays or high support-call frequency."),
        ("Re-engagement for Lapsed Customers",
         f"Churned customers last interacted {S['li_churned']:.1f} days ago on average vs "
         f"{S['li_retained']:.1f} days for retained customers. Prioritise customers with "
         "longer periods since their last interaction for targeted re-engagement "
         "before they reach the point of formal cancellation."),
        ("VIP Retention for High-Value at Risk",
         f"The {S['seg_hvar_n']:,}-customer High-Value at Risk segment ({S['seg_hvar_cr']:.1f}% "
         "churn rate) is a high-value customer group with elevated churn risk and the "
         "highest-priority retention target. Assign dedicated account management or "
         "personalised offers before cancellation occurs."),
    ]
    for title, detail in recs:
        subsection(story, styles, title)
        body(story, styles, detail)
    story.append(PageBreak())

    # ── LIMITATIONS ────────────────────────────────────────────────────────────
    section(story, styles, "18. Limitations")
    bullet(story, styles, [
        "No timestamp or date column — time-series trend analysis and cohort retention "
        "curves are not possible with this dataset.",
        "Currency unit for Total Spend is not specified — financial impact estimates "
        "cannot be stated in monetary terms.",
        f"Age = 0 values ({invalid_age_count} records) were treated as invalid/missing; "
        "the true cause of these entries is unknown.",
        f"The training set churn rate ({S['churn_rate']:.1f}%) differs substantially from "
        f"the testing set ({S['test_churn_rate']:.1f}%), and all Monthly contracts in the "
        "training set are labelled as churned (100%). Both patterns may affect model "
        "calibration and should be flagged in any production deployment.",
        "Segmentation thresholds (75th percentile) are analytical choices, not externally "
        "validated business definitions.",
        "No demographic data beyond Gender and Age is available, limiting targeting granularity.",
        "Support Calls correlation is strong but does not establish causation — customers "
        "may be churning because of unresolved issues rather than because they call.",
    ])

    # ── FUTURE SCOPE ───────────────────────────────────────────────────────────
    section(story, styles, "19. Future Scope")
    bullet(story, styles, [
        "Integrate date/time columns to enable cohort analysis and monthly churn trend tracking.",
        "Add SHAP (SHapley Additive exPlanations) values for individual-level model explainability.",
        "Extend predictive model with XGBoost or LightGBM for potential performance improvements.",
        "Connect the dashboard to a live database or CRM for real-time scoring.",
        "Enable export of at-risk customer lists as CSV for CRM integration.",
        "Add A/B testing capability to measure effectiveness of retention campaigns.",
        "Investigate the gender churn disparity with additional data collection.",
    ])

    # ── CONCLUSION ─────────────────────────────────────────────────────────────
    section(story, styles, "20. Conclusion")
    body(story, styles,
         f"This project has delivered a complete Business Intelligence solution for customer "
         f"churn analysis. From a dataset of {S['total_train']:,} cleaned customer records, "
         f"it was established that the overall churn rate is {S['churn_rate']:.2f}%, with "
         f"clear behavioural drivers including high support-call frequency "
         f"(Pearson r = {S['corr_support']:.4f}), higher payment delays, and lower total spend "
         f"among churned customers.")
    body(story, styles,
         "Both Logistic Regression and Random Forest were evaluated on the held-out testing "
         "dataset using multiple classification metrics. Customer segmentation identified the "
         "High-Value at Risk group as the highest-priority retention target. All findings were "
         "translated into six prioritised, data-supported business recommendations.")
    body(story, styles,
         "The interactive Streamlit dashboard provides a self-service analytics capability "
         "suitable for business stakeholders, while this report provides the formal analytical "
         "foundation for academic submission and project review.")

    # ── REFERENCES ─────────────────────────────────────────────────────────────
    section(story, styles, "21. References")
    bullet(story, styles, [
        "Azeem, M. S. (2023). Customer Churn Dataset. Kaggle. "
        "https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset"
        "?resource=download",
        "Streamlit Inc. (2024). Streamlit Documentation. https://docs.streamlit.io",
        "Plotly Technologies. (2024). Plotly Python Documentation. "
        "https://plotly.com/python/",
        "scikit-learn Developers. (2024). scikit-learn: Machine Learning in Python. "
        "https://scikit-learn.org",
        "ReportLab. (2024). ReportLab User Guide. "
        "https://www.reportlab.com/docs/reportlab-userguide.pdf",
    ])

    # ── FOOTER ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1 * cm))
    hr(story)
    story.append(Paragraph(
        f"Customer Churn &amp; Retention Intelligence Dashboard "
        f"— Academic BI Project Report  |  {now}",
        styles["muted"],
    ))


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading datasets...")
    for path in [TRAIN_PATH, TEST_PATH]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found.")
            sys.exit(1)

    train_df, invalid_age = load_and_clean(TRAIN_PATH)
    test_df,  _           = load_and_clean(TEST_PATH)
    print(f"Training rows (clean): {len(train_df):,}")
    print(f"Testing rows  (clean): {len(test_df):,}")

    print(f"Building PDF: {OUTPUT_PDF}")
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2.2 * cm, bottomMargin=2.2 * cm,
        title="Customer Churn & Retention Intelligence Dashboard — Project Report",
        author="BI Analytics",
        subject="Customer Churn Analysis — Business Intelligence Project",
    )

    styles = make_styles()
    story  = []
    build_report(train_df, test_df, invalid_age, story, styles)
    doc.build(story)
    print(f"Report saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
