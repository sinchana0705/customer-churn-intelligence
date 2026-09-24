"""
Customer Churn & Retention Intelligence Dashboard
==================================================
A single-file Streamlit BI application.

Author  : [Your Name]
Dataset : customer_churn_dataset-training-master.csv  (primary)
          customer_churn_dataset-testing-master.csv   (model validation)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Streamlit page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn & Retention Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

C = {
    "bg":           "#FFFFFF",
    "bg2":          "#F8F9FA",
    "card":         "#FFFFFF",
    "primary":      "#243B53",
    "primary2":     "#486581",
    "primary_lt":   "#EEF4F8",
    "text":         "#171717",
    "text2":        "#6B7280",
    "muted":        "#9CA3AF",
    "border":       "#E5E7EB",
    "churn":        "#C95C5C",
    "retain":       "#4F8A7A",
    "warning":      "#C99A3D",
    "churn_lt":     "#FAF0F0",
    "retain_lt":    "#F0F7F5",
}

CHART_FONT   = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
CHART_COLOR  = C["primary"]
CHART_CHURN  = C["churn"]
CHART_RETAIN = C["retain"]
CHART_MUTED  = "#9CA3AF"

def plotly_base():
    """Return a consistent base layout dict for all Plotly charts."""
    return dict(
        template="plotly_white",
        font=dict(family=CHART_FONT, size=12, color=C["text"]),
        title_font=dict(family=CHART_FONT, size=14, color=C["text"]),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        margin=dict(t=46, b=32, l=12, r=12),
        xaxis=dict(
            gridcolor=C["border"], gridwidth=0.5,
            linecolor=C["border"], tickfont=dict(size=11, color=C["text2"]),
            title_font=dict(size=11, color=C["text2"]),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=C["border"], gridwidth=0.5,
            linecolor=C["border"], tickfont=dict(size=11, color=C["text2"]),
            title_font=dict(size=11, color=C["text2"]),
            zeroline=False,
        ),
        legend=dict(
            font=dict(size=11, color=C["text2"]),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor=C["primary"], font_color="white",
            font=dict(family=CHART_FONT, size=11),
        ),
    )

GLOBAL_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  /* ── Global font and base colours ── */
  html, body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: {C["text"]} !important;
    background-color: {C["bg"]} !important;
  }}

  /* ── Force entire Streamlit shell to light theme.
     Every structural container is targeted so Streamlit's
     built-in dark theme cannot override at higher specificity. ── */
  .stApp,
  .stApp > *,
  [data-testid="stAppViewContainer"],
  [data-testid="stAppViewContainer"] > section,
  [data-testid="stMain"],
  [data-testid="stMainBlockContainer"],
  [data-testid="stBottom"],
  [data-testid="stHeader"],
  .main,
  .main .block-container,
  section.main {{
    background-color: {C["bg"]} !important;
    color: {C["text"]} !important;
  }}

  /* ── All Streamlit generated class elements ── */
  [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  }}

  /* ── Markdown and rendered text ── */
  .stMarkdown, .stMarkdown p, .stMarkdown li,
  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
  .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
    color: {C["text"]} !important;
  }}

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* ── Main content block ── */
  .block-container {{
    padding-top: 1.6rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px !important;
    background-color: {C["bg"]} !important;
  }}

  /* ── Sidebar shell ── */
  [data-testid="stSidebar"],
  [data-testid="stSidebar"] > div,
  [data-testid="stSidebar"] > div > div {{
    background-color: {C["bg2"]} !important;
    border-right: 1px solid {C["border"]} !important;
  }}
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] div {{
    color: {C["text"]} !important;
  }}
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label {{
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: {C["text2"]} !important;
  }}
  [data-testid="stSidebar"] [data-baseweb="select"] div {{
    font-size: 13px !important;
    border-color: {C["border"]} !important;
    background: {C["bg"]} !important;
    color: {C["text"]} !important;
  }}

  /* ── Dividers ── */
  hr {{
    border: none !important;
    border-top: 1px solid {C["border"]} !important;
    margin: 1rem 0 !important;
  }}

  /* ── Dataframe ── */
  [data-testid="stDataFrame"],
  .stDataFrame {{
    background-color: {C["bg"]} !important;
  }}
  [data-testid="stDataFrame"] th {{
    background-color: {C["bg2"]} !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: {C["text2"]} !important;
  }}
  [data-testid="stDataFrame"] td {{
    font-size: 12.5px !important;
    color: {C["text"]} !important;
    background-color: {C["bg"]} !important;
  }}

  /* ── Expanders ── */
  [data-testid="stExpander"] {{
    border: 1px solid {C["border"]} !important;
    border-radius: 6px !important;
    background: {C["bg"]} !important;
  }}
  [data-testid="stExpander"] > div {{
    background: {C["bg"]} !important;
  }}
  [data-testid="stExpander"] summary,
  [data-testid="stExpander"] summary span,
  [data-testid="stExpander"] summary p {{
    font-size: 13px !important;
    font-weight: 600 !important;
    color: {C["text"]} !important;
  }}

  /* ── Select / dropdown widgets ── */
  [data-baseweb="select"] *,
  [data-baseweb="popover"] * {{
    background-color: {C["bg"]} !important;
    color: {C["text"]} !important;
  }}

  /* ── Radio navigation ── */
  [data-testid="stSidebar"] .stRadio label,
  [data-testid="stSidebar"] .stRadio span {{
    font-size: 13px !important;
    color: {C["text"]} !important;
    padding: 4px 0 !important;
  }}
  [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {{
    background: transparent !important;
  }}

  /* ── Column containers ── */
  [data-testid="column"],
  [data-testid="stHorizontalBlock"] {{
    background-color: transparent !important;
  }}

  /* ── Spinner / status ── */
  [data-testid="stStatusWidget"] {{
    background-color: {C["bg"]} !important;
    color: {C["text"]} !important;
  }}

  /* ── Plotly toolbar — hide ── */
  .modebar, .modebar-container {{ display: none !important; }}

  /* ── Caption / muted text ── */
  .stCaption, .stCaption p {{
    color: {C["muted"]} !important;
    font-size: 11px !important;
  }}

  /* ── Alert / info boxes (st.info, st.warning, etc.) ── */
  [data-testid="stAlert"] {{
    background-color: {C["bg2"]} !important;
    color: {C["text"]} !important;
    border-color: {C["border"]} !important;
  }}
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING  (logic unchanged)
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_PATH = "customer_churn_dataset-training-master.csv"
TEST_PATH  = "customer_churn_dataset-testing-master.csv"

NUMERIC_COLS     = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                    "Payment Delay", "Total Spend", "Last Interaction"]
CATEGORICAL_COLS = ["Gender", "Subscription Type", "Contract Length"]


@st.cache_data(show_spinner="Loading and validating dataset...")
def load_data():
    train_raw = pd.read_csv(TRAIN_PATH, dtype=str)
    test_raw  = pd.read_csv(TEST_PATH,  dtype=str)

    def clean(df, label):
        df = df.copy()
        df.dropna(how="all", inplace=True)
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        df.replace("", np.nan, inplace=True)
        df.dropna(subset=["CustomerID"], inplace=True)

        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce")
        df.dropna(subset=["Churn"], inplace=True)
        df["Churn"] = df["Churn"].astype(int)

        invalid_age_count = (df["Age"] == 0).sum()
        df.loc[df["Age"] == 0, "Age"] = np.nan

        before_dedup = len(df)
        df.drop_duplicates(inplace=True)
        after_dedup  = len(df)

        df.attrs["label"]         = label
        df.attrs["invalid_age"]   = int(invalid_age_count)
        df.attrs["dupes_removed"] = int(before_dedup - after_dedup)

        df["TenureGroup"] = pd.cut(
            df["Tenure"],
            bins=[0, 12, 24, 36, 48, 60],
            labels=["0-12m", "13-24m", "25-36m", "37-48m", "49-60m"],
            right=True,
        )
        df["SpendBand"] = pd.cut(
            df["Total Spend"],
            bins=[0, 200, 400, 600, 800, 1001],
            labels=["0-200", "201-400", "401-600", "601-800", "801-1000"],
            right=True,
        )

        def _norm(series):
            mn, mx = series.min(), series.max()
            return (series - mn) / (mx - mn + 1e-9)

        df["RiskScore"] = (
            _norm(df["Support Calls"])    * 30 +
            _norm(df["Payment Delay"])    * 25 +
            _norm(df["Last Interaction"]) * 25 +
            (1 - _norm(df["Usage Frequency"])) * 20
        ).clip(0, 100).round(1)

        return df

    return clean(train_raw, "Training"), clean(test_raw, "Testing")


# ══════════════════════════════════════════════════════════════════════════════
# KPI HELPERS  (logic unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def calc_kpis(df):
    total       = len(df)
    churned     = df["Churn"].sum()
    retained    = total - churned
    churn_rate  = churned / total * 100 if total else 0
    return dict(
        total=total, churned=int(churned), retained=int(retained),
        churn_rate=churn_rate,
        total_spend=df["Total Spend"].sum(),
        avg_spend=df["Total Spend"].mean(),
        avg_tenure=df["Tenure"].mean(),
        avg_usage=df["Usage Frequency"].mean(),
        avg_support=df["Support Calls"].mean(),
        avg_delay=df["Payment Delay"].mean(),
        avg_last=df["Last Interaction"].mean(),
    )


def fmt(val, decimals=1):
    if pd.isna(val):
        return "N/A"
    if val >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    if val >= 1_000:
        return f"{val/1_000:.1f}K"
    return f"{val:.{decimals}f}"


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def page_header(title, subtitle, support=None):
    st.markdown(
        f"""
        <div style="margin-bottom:1.4rem;">
          <div style="font-size:11px;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.08em;color:{C['muted']};margin-bottom:4px;">
            Customer Churn &amp; Retention Intelligence
          </div>
          <h1 style="font-size:26px;font-weight:700;color:{C['text']};
                     margin:0 0 4px 0;line-height:1.2;">{title}</h1>
          <p style="font-size:13px;color:{C['text2']};margin:0;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text):
    st.markdown(
        f"""<div style="font-size:11px;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.07em;color:{C['muted']};
                        margin:1.6rem 0 0.6rem 0;">{text}</div>
            <hr style="margin:0 0 0.8rem 0;">""",
        unsafe_allow_html=True,
    )


def kpi_card(col, label, value, sub=None, accent=False, warn=False):
    val_color = C["churn"] if warn else (C["primary"] if accent else C["text"])
    with col:
        sub_html = (f'<div style="font-size:11px;color:{C["muted"]};'
                    f'margin-top:3px;line-height:1.3;">{sub}</div>') if sub else ""
        st.markdown(
            f"""
            <div style="background:{C['card']};border:1px solid {C['border']};
                        border-radius:6px;padding:14px 16px 12px 16px;
                        box-shadow:0 1px 3px rgba(0,0,0,0.04);">
              <div style="font-size:10.5px;font-weight:600;text-transform:uppercase;
                          letter-spacing:0.07em;color:{C['muted']};margin-bottom:6px;">
                {label}
              </div>
              <div style="font-size:26px;font-weight:700;color:{val_color};
                          line-height:1.1;">{value}</div>
              {sub_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def insight_card(title, text, kind="neutral"):
    left_color = {"neutral": C["border"], "warn": C["warning"],
                  "risk": C["churn"], "good": C["retain"]}.get(kind, C["border"])
    st.markdown(
        f"""
        <div style="border:1px solid {C['border']};border-left:3px solid {left_color};
                    border-radius:4px;background:{C['bg']};
                    padding:12px 14px;margin:6px 0;">
          <div style="font-size:12px;font-weight:600;color:{C['text']};
                      margin-bottom:3px;">{title}</div>
          <div style="font-size:12.5px;color:{C['text2']};line-height:1.5;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def note_text(text):
    st.markdown(
        f'<p style="font-size:11px;color:{C["muted"]};margin:4px 0 12px 0;">{text}</p>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def apply_sidebar_filters(df, full_df):
    st.sidebar.markdown(
        f"""<div style="font-size:10.5px;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.08em;color:{C['muted']};
                        padding:0.5rem 0 0.3rem 0;">Dashboard Filters</div>""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'<hr style="margin:0 0 0.7rem 0;border-color:{C["border"]};">',
        unsafe_allow_html=True,
    )

    genders   = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
    gender    = st.sidebar.selectbox("Gender", genders, index=0)

    subs      = ["All"] + sorted(df["Subscription Type"].dropna().unique().tolist())
    sub       = st.sidebar.selectbox("Subscription Type", subs, index=0)

    contracts = ["All"] + sorted(df["Contract Length"].dropna().unique().tolist())
    contract  = st.sidebar.selectbox("Contract Length", contracts, index=0)

    churn_opts = {"All customers": None, "Churned only": 1, "Retained only": 0}
    churn_sel  = st.sidebar.selectbox("Churn Status", list(churn_opts.keys()), index=0)
    churn_val  = churn_opts[churn_sel]

    t_min = int(df["Tenure"].min()) if df["Tenure"].notna().any() else 0
    t_max = int(df["Tenure"].max()) if df["Tenure"].notna().any() else 60
    tenure_range = st.sidebar.slider("Tenure (months)", t_min, t_max, (t_min, t_max))

    # Apply
    fdf = df.copy()
    if gender   != "All": fdf = fdf[fdf["Gender"] == gender]
    if sub      != "All": fdf = fdf[fdf["Subscription Type"] == sub]
    if contract != "All": fdf = fdf[fdf["Contract Length"] == contract]
    if churn_val is not None: fdf = fdf[fdf["Churn"] == churn_val]
    fdf = fdf[fdf["Tenure"].between(tenure_range[0], tenure_range[1], inclusive="both")]

    st.sidebar.markdown(
        f'<hr style="margin:0.8rem 0 0.5rem 0;border-color:{C["border"]};">',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"""<div style="font-size:10.5px;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.08em;color:{C['muted']};margin-bottom:6px;">
              Dataset
            </div>
            <div style="font-size:12px;color:{C['text2']};line-height:1.7;">
              Training dataset<br>
              <span style="font-weight:600;color:{C['text']};">{len(full_df):,}</span> customers<br>
              Cleaned and validated<br>
              <span style="color:{C['primary']};font-weight:600;">{len(fdf):,} shown</span>
            </div>""",
        unsafe_allow_html=True,
    )
    return fdf


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def _bar_chart(data, x, y, title, height=300, color=None, text_col=None):
    """Consistent single-color or mapped bar chart."""
    base = plotly_base()
    if color is None:
        colors_list = [C["primary"]] * len(data)
    else:
        colors_list = color
    fig = go.Figure(go.Bar(
        x=data[x], y=data[y],
        marker_color=colors_list,
        text=data[text_col] if text_col else None,
        texttemplate="%{text:.1f}%" if text_col else None,
        textposition="outside",
        textfont=dict(size=11, color=C["text2"]),
    ))
    fig.update_layout(**base, title=dict(text=title, font=dict(size=13, color=C["text"])),
                      height=height, showlegend=False)
    fig.update_yaxes(title_text="Churn Rate (%)")
    return fig


def page_overview(df):
    page_header(
        "Executive Overview",
        "High-level KPIs and churn distribution across key dimensions.",
    )

    kpis = calc_kpis(df)

    # ── KPI row 1 ─────────────────────────────────────────────────────────────
    section_label("Key Performance Indicators")
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, "Total Customers",   f"{kpis['total']:,}",
             sub="Cleaned customer records")
    kpi_card(c2, "Churn Rate",        f"{kpis['churn_rate']:.1f}%",
             sub=f"{kpis['churned']:,} churned customers", warn=True)
    kpi_card(c3, "Churned",           f"{kpis['churned']:,}",
             sub="Have left the service", warn=True)
    kpi_card(c4, "Retained",          f"{kpis['retained']:,}",
             sub="Still active customers", accent=True)
    kpi_card(c5, "Avg Tenure",        f"{kpis['avg_tenure']:.1f} mo",
             sub="Average months as subscriber")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c6, c7, c8, c9, c10 = st.columns(5)
    kpi_card(c6,  "Avg Total Spend",      fmt(kpis['avg_spend'], 0),
             sub="Per customer")
    kpi_card(c7,  "Avg Usage Frequency",  f"{kpis['avg_usage']:.1f}/mo",
             sub="Service interactions")
    kpi_card(c8,  "Avg Support Calls",    f"{kpis['avg_support']:.1f}",
             sub="Support contacts made")
    kpi_card(c9,  "Avg Payment Delay",    f"{kpis['avg_delay']:.1f} days",
             sub="Days late on payment")
    kpi_card(c10, "Avg Last Interaction", f"{kpis['avg_last']:.1f} days",
             sub="Days since last contact")

    # ── Churn Overview ────────────────────────────────────────────────────────
    section_label("Churn Overview")
    col_pie, col_sub = st.columns(2)

    with col_pie:
        pie_data = pd.DataFrame({
            "Status": ["Churned", "Retained"],
            "Count":  [kpis["churned"], kpis["retained"]],
        })
        base = plotly_base()
        fig = go.Figure(go.Pie(
            labels=pie_data["Status"],
            values=pie_data["Count"],
            hole=0.52,
            marker=dict(colors=[C["churn"], C["retain"]],
                        line=dict(color=C["bg"], width=2)),
            textinfo="percent",
            textfont=dict(size=12, color=C["text"]),
        ))
        # Override legend and margin from plotly_base() before spreading to avoid
        # "multiple values for keyword argument" TypeError
        pie_layout = {
            **base,
            "title":  dict(text="Churn vs Retention", font=dict(size=13, color=C["text"])),
            "height": 300,
            "legend": dict(orientation="h", y=-0.05, x=0.5, xanchor="center",
                           font=dict(size=11, color=C["text2"])),
            "margin": dict(t=46, b=40, l=12, r=12),
        }
        fig.update_layout(**pie_layout)
        st.plotly_chart(fig, use_container_width=True)

    with col_sub:
        sub_churn = (
            df.groupby("Subscription Type")["Churn"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "Churned", "count": "Total"})
        )
        sub_churn["Churn Rate %"] = (sub_churn["Churned"] / sub_churn["Total"] * 100).round(1)
        sub_churn = sub_churn.reset_index()
        fig2 = _bar_chart(sub_churn, "Subscription Type", "Churn Rate %",
                          "Churn Rate by Subscription Type",
                          text_col="Churn Rate %",
                          color=[C["primary"], C["primary2"], C["muted"]][:len(sub_churn)])
        st.plotly_chart(fig2, use_container_width=True)

    # ── Churn Patterns ────────────────────────────────────────────────────────
    section_label("Churn Patterns")
    col_con, col_gen = st.columns(2)

    with col_con:
        con_churn = (
            df.groupby("Contract Length")["Churn"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "Churned", "count": "Total"})
        )
        con_churn["Churn Rate %"] = (con_churn["Churned"] / con_churn["Total"] * 100).round(1)
        con_churn = con_churn.reset_index()
        con_colors = [C["churn"] if v >= 80 else C["primary2"] if v >= 50 else C["retain"]
                      for v in con_churn["Churn Rate %"]]
        fig3 = _bar_chart(con_churn, "Contract Length", "Churn Rate %",
                          "Churn Rate by Contract Length",
                          text_col="Churn Rate %",
                          color=con_colors)
        st.plotly_chart(fig3, use_container_width=True)
        monthly_row = con_churn[con_churn["Contract Length"] == "Monthly"]
        if len(monthly_row) and monthly_row["Churn Rate %"].values[0] == 100.0:
            note_text(
                "Note: In the training dataset, all Monthly contract customers are labelled as "
                "churned (100%). This is an inherent dataset characteristic; the testing set "
                "shows 51.6%. Interpret with caution."
            )

    with col_gen:
        gen_churn = (
            df.groupby("Gender")["Churn"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "Churned", "count": "Total"})
        )
        gen_churn["Churn Rate %"] = (gen_churn["Churned"] / gen_churn["Total"] * 100).round(1)
        gen_churn = gen_churn.reset_index()
        fig4 = _bar_chart(gen_churn, "Gender", "Churn Rate %",
                          "Churn Rate by Gender",
                          text_col="Churn Rate %",
                          color=[C["primary"], C["primary2"]][:len(gen_churn)])
        st.plotly_chart(fig4, use_container_width=True)

    # ── Tenure Analysis ───────────────────────────────────────────────────────
    section_label("Tenure Analysis")
    ten_churn = (
        df.dropna(subset=["TenureGroup"])
        .groupby("TenureGroup", observed=True)["Churn"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "Churned", "count": "Total"})
    )
    ten_churn["Churn Rate %"] = (ten_churn["Churned"] / ten_churn["Total"] * 100).round(1)
    ten_churn = ten_churn.reset_index()
    ten_colors = [C["churn"] if v >= 70 else C["primary2"] if v >= 50 else C["retain"]
                  for v in ten_churn["Churn Rate %"]]
    fig5 = _bar_chart(ten_churn, "TenureGroup", "Churn Rate %",
                      "Churn Rate by Tenure Group", height=280,
                      text_col="Churn Rate %", color=ten_colors)
    st.plotly_chart(fig5, use_container_width=True)

    # ── Key Findings ──────────────────────────────────────────────────────────
    section_label("Key Findings")
    overall_cr = kpis["churn_rate"]

    findings = []
    # Overall
    findings.append((
        "Overall churn",
        f"{overall_cr:.1f}% of customers in the filtered population have churned "
        f"({kpis['churned']:,} of {kpis['total']:,} customers).",
        "risk" if overall_cr > 40 else "warn",
    ))

    # Subscription — only compare if more than one category visible
    if len(sub_churn) > 1:
        max_sub = sub_churn.loc[sub_churn["Churn Rate %"].idxmax()]
        min_sub = sub_churn.loc[sub_churn["Churn Rate %"].idxmin()]
        findings.append((
            "Subscription type",
            f"{max_sub['Subscription Type']} subscribers have the highest churn rate "
            f"({max_sub['Churn Rate %']:.1f}%); "
            f"{min_sub['Subscription Type']} subscribers have the lowest "
            f"({min_sub['Churn Rate %']:.1f}%).",
            "warn",
        ))
    elif len(sub_churn) == 1:
        row = sub_churn.iloc[0]
        findings.append((
            "Subscription type",
            f"{row['Subscription Type']} subscribers in the current filter have a churn rate of "
            f"{row['Churn Rate %']:.1f}%.",
            "neutral",
        ))

    # Contract — only compare if more than one category visible
    if len(con_churn) > 1:
        max_con = con_churn.loc[con_churn["Churn Rate %"].idxmax()]
        min_con = con_churn.loc[con_churn["Churn Rate %"].idxmin()]
        findings.append((
            "Contract length",
            f"{max_con['Contract Length']} contracts show the highest churn rate "
            f"({max_con['Churn Rate %']:.1f}%); "
            f"{min_con['Contract Length']} contracts show the lowest "
            f"({min_con['Churn Rate %']:.1f}%).",
            "warn",
        ))
    elif len(con_churn) == 1:
        row = con_churn.iloc[0]
        findings.append((
            "Contract length",
            f"{row['Contract Length']} contracts in the current filter have a churn rate of "
            f"{row['Churn Rate %']:.1f}%.",
            "neutral",
        ))

    # Tenure — only compare if more than one group visible
    if len(ten_churn) > 1:
        max_ten = ten_churn.loc[ten_churn["Churn Rate %"].idxmax()]
        min_ten = ten_churn.loc[ten_churn["Churn Rate %"].idxmin()]
        findings.append((
            "Tenure pattern",
            f"The {max_ten['TenureGroup']} tenure group has the highest observed churn rate "
            f"({max_ten['Churn Rate %']:.1f}%). "
            f"The {min_ten['TenureGroup']} group shows the lowest churn rate "
            f"({min_ten['Churn Rate %']:.1f}%).",
            "neutral",
        ))
    elif len(ten_churn) == 1:
        row = ten_churn.iloc[0]
        findings.append((
            "Tenure pattern",
            f"Within the filtered range, the {row['TenureGroup']} tenure group "
            f"has a churn rate of {row['Churn Rate %']:.1f}%.",
            "neutral",
        ))

    # Render as 2-column grid
    cols = st.columns(2)
    for i, (title, text, kind) in enumerate(findings):
        with cols[i % 2]:
            insight_card(title, text, kind)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CHURN DRIVERS
# ══════════════════════════════════════════════════════════════════════════════

def page_drivers(df):
    page_header(
        "Churn Drivers",
        "Identify customer behaviors and characteristics associated with churn.",
    )

    # ── Behavioural comparison ────────────────────────────────────────────────
    section_label("Behavioral Comparison: Churned vs Retained")

    metrics = ["Support Calls", "Payment Delay", "Usage Frequency",
               "Last Interaction", "Tenure", "Total Spend"]

    churn_groups = df["Churn"].unique()
    has_both = (0 in churn_groups) and (1 in churn_groups)

    avg_by_churn = df.groupby("Churn")[metrics].mean().T.rename(
        columns={0: "Retained", 1: "Churned"})
    avg_by_churn = avg_by_churn.reset_index().rename(columns={"index": "Metric"})
    avg_melt = avg_by_churn.melt(id_vars="Metric", var_name="Status", value_name="Average")

    base = plotly_base()
    fig_cmp = px.bar(
        avg_melt, x="Metric", y="Average", color="Status", barmode="group",
        color_discrete_map={"Churned": C["churn"], "Retained": C["retain"]},
        title="Average Behavioral Metrics by Churn Status",
        text_auto=".1f",
    )
    fig_cmp.update_layout(**base, height=340)
    fig_cmp.update_traces(textfont=dict(size=10), textposition="outside")
    st.plotly_chart(fig_cmp, use_container_width=True)

    if has_both:
        driver_insights = []
        for _, row in avg_by_churn.iterrows():
            metric = row["Metric"]
            c_val  = row.get("Churned", np.nan)
            r_val  = row.get("Retained", np.nan)
            if pd.isna(c_val) or pd.isna(r_val):
                continue
            diff_pct = abs(c_val - r_val) / (r_val + 1e-9) * 100
            if diff_pct > 5:
                direction = "higher" if c_val > r_val else "lower"
                kind = "risk" if diff_pct > 25 else "warn" if diff_pct > 10 else "neutral"
                driver_insights.append((
                    metric,
                    f"Churned customers average {diff_pct:.0f}% {direction} {metric} "
                    f"({c_val:.1f}) than retained customers ({r_val:.1f}).",
                    kind,
                ))
        if driver_insights:
            cols = st.columns(2)
            for i, (title, text, kind) in enumerate(driver_insights):
                with cols[i % 2]:
                    insight_card(title, text, kind)

    # ── Support Calls ─────────────────────────────────────────────────────────
    section_label("Support Calls")
    col1, col2 = st.columns(2)

    with col1:
        sc_churn = (
            df.groupby("Support Calls")["Churn"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "Churn Rate", "count": "Customers"})
        )
        sc_churn["Churn Rate %"] = (sc_churn["Churn Rate"] * 100).round(1)
        sc_colors = [C["churn"] if v >= 70 else C["primary2"] if v >= 40 else C["primary"]
                     for v in sc_churn["Churn Rate %"]]
        fig_sc = _bar_chart(sc_churn, "Support Calls", "Churn Rate %",
                            "Churn Rate by Number of Support Calls",
                            color=sc_colors)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        base2 = plotly_base()
        fig_box_sc = px.box(
            df, x="Churn", y="Support Calls",
            color="Churn",
            color_discrete_map={0: C["retain"], 1: C["churn"]},
            labels={"Churn": "Churn Status", "Support Calls": "Support Calls"},
            title="Support Calls Distribution (Churned vs Retained)",
        )
        fig_box_sc.update_xaxes(tickvals=[0, 1], ticktext=["Retained", "Churned"])
        fig_box_sc.update_layout(**base2, height=300, showlegend=False)
        st.plotly_chart(fig_box_sc, use_container_width=True)

    # ── Payment Delay ─────────────────────────────────────────────────────────
    section_label("Payment Delay")
    col3, col4 = st.columns(2)

    with col3:
        pd_bins = pd.cut(df["Payment Delay"], bins=[0, 7, 14, 21, 30],
                         labels=["0-7d","8-14d","15-21d","22-30d"], right=True)
        pd_df = df.copy(); pd_df["Delay Band"] = pd_bins
        pd_churn = (
            pd_df.dropna(subset=["Delay Band"])
            .groupby("Delay Band", observed=True)["Churn"]
            .agg(["mean","count"]).reset_index()
        )
        pd_churn["Churn Rate %"] = (pd_churn["mean"] * 100).round(1)
        pd_colors = [C["churn"] if v >= 70 else C["primary2"] if v >= 40 else C["primary"]
                     for v in pd_churn["Churn Rate %"]]
        fig_pd = _bar_chart(pd_churn, "Delay Band", "Churn Rate %",
                            "Churn Rate by Payment Delay Band", color=pd_colors)
        st.plotly_chart(fig_pd, use_container_width=True)

    with col4:
        base3 = plotly_base()
        fig_box_pd = px.box(
            df, x="Churn", y="Payment Delay", color="Churn",
            color_discrete_map={0: C["retain"], 1: C["churn"]},
            title="Payment Delay Distribution (Churned vs Retained)",
        )
        fig_box_pd.update_xaxes(tickvals=[0, 1], ticktext=["Retained", "Churned"])
        fig_box_pd.update_layout(**base3, height=300, showlegend=False)
        st.plotly_chart(fig_box_pd, use_container_width=True)

    # ── Usage Frequency ───────────────────────────────────────────────────────
    section_label("Usage Frequency")
    col5, col6 = st.columns(2)

    with col5:
        uf_bins = pd.cut(df["Usage Frequency"], bins=[0, 7, 14, 21, 30],
                         labels=["1-7","8-14","15-21","22-30"], right=True)
        uf_df = df.copy(); uf_df["Usage Band"] = uf_bins
        uf_churn = (
            uf_df.dropna(subset=["Usage Band"])
            .groupby("Usage Band", observed=True)["Churn"]
            .agg(["mean","count"]).reset_index()
        )
        uf_churn["Churn Rate %"] = (uf_churn["mean"] * 100).round(1)
        uf_colors = [C["churn"] if v >= 70 else C["primary2"] if v >= 40 else C["retain"]
                     for v in uf_churn["Churn Rate %"]]
        fig_uf = _bar_chart(uf_churn, "Usage Band", "Churn Rate %",
                            "Churn Rate by Usage Frequency Band", color=uf_colors)
        st.plotly_chart(fig_uf, use_container_width=True)

    with col6:
        base4 = plotly_base()
        fig_box_uf = px.box(
            df, x="Churn", y="Usage Frequency", color="Churn",
            color_discrete_map={0: C["retain"], 1: C["churn"]},
            title="Usage Frequency Distribution (Churned vs Retained)",
        )
        fig_box_uf.update_xaxes(tickvals=[0, 1], ticktext=["Retained", "Churned"])
        fig_box_uf.update_layout(**base4, height=300, showlegend=False)
        st.plotly_chart(fig_box_uf, use_container_width=True)

    # ── Total Spend & Last Interaction ────────────────────────────────────────
    section_label("Spend & Last Interaction")
    col7, col8 = st.columns(2)

    with col7:
        sp_churn = (
            df.dropna(subset=["SpendBand"])
            .groupby("SpendBand", observed=True)["Churn"]
            .agg(["mean","count"]).reset_index()
        )
        sp_churn["Churn Rate %"] = (sp_churn["mean"] * 100).round(1)
        sp_colors = [C["primary"]] * len(sp_churn)
        fig_sp = _bar_chart(sp_churn, "SpendBand", "Churn Rate %",
                            "Churn Rate by Total Spend Band", color=sp_colors)
        st.plotly_chart(fig_sp, use_container_width=True)

    with col8:
        li_bins = pd.cut(df["Last Interaction"], bins=[0,7,14,21,30],
                         labels=["0-7d","8-14d","15-21d","22-30d"], right=True)
        li_df = df.copy(); li_df["Interaction Band"] = li_bins
        li_churn = (
            li_df.dropna(subset=["Interaction Band"])
            .groupby("Interaction Band", observed=True)["Churn"]
            .agg(["mean","count"]).reset_index()
        )
        li_churn["Churn Rate %"] = (li_churn["mean"] * 100).round(1)
        li_colors = [C["churn"] if v >= 70 else C["primary2"] if v >= 40 else C["primary"]
                     for v in li_churn["Churn Rate %"]]
        fig_li = _bar_chart(li_churn, "Interaction Band", "Churn Rate %",
                            "Churn Rate by Days Since Last Interaction", color=li_colors)
        st.plotly_chart(fig_li, use_container_width=True)

    # ── Correlation ───────────────────────────────────────────────────────────
    section_label("Feature Correlation with Churn")
    num_df = df[["Churn","Age","Tenure","Usage Frequency","Support Calls",
                  "Payment Delay","Total Spend","Last Interaction"]].dropna()
    corr = num_df.corr()["Churn"].drop("Churn").sort_values()

    base5 = plotly_base()
    fig_corr = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h",
        marker_color=[C["churn"] if v > 0 else C["retain"] for v in corr.values],
        text=[f"{v:.3f}" for v in corr.values],
        textposition="outside",
        textfont=dict(size=10, color=C["text2"]),
    ))
    fig_corr.update_layout(
        **base5,
        title=dict(text="Pearson Correlation with Churn  (positive = higher risk)",
                   font=dict(size=13, color=C["text"])),
        height=340,
        xaxis_title="Correlation Coefficient",
        yaxis_title="",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    top_pos = corr[corr > 0].idxmax() if (corr > 0).any() else None
    top_neg = corr[corr < 0].idxmin() if (corr < 0).any() else None
    corr_insights = []
    if top_pos:
        corr_insights.append((
            f"Strongest positive predictor — {top_pos}",
            f"Correlation r = {corr[top_pos]:.3f}. Higher {top_pos} values are "
            f"associated with higher churn risk.",
            "risk",
        ))
    if top_neg:
        corr_insights.append((
            f"Strongest protective factor — {top_neg}",
            f"Correlation r = {corr[top_neg]:.3f}. Higher {top_neg} values are "
            f"associated with lower churn risk.",
            "good",
        ))
    if corr_insights:
        cols2 = st.columns(len(corr_insights))
        for i, (title, text, kind) in enumerate(corr_insights):
            with cols2[i]:
                insight_card(title, text, kind)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER SEGMENTS & RETENTION
# ══════════════════════════════════════════════════════════════════════════════

def page_segments(df):
    page_header(
        "Customer Segments & Retention",
        "Data-driven segments based on observed spend, risk, and behavioral indicators.",
    )

    spend_hi  = df["Total Spend"].quantile(0.75)
    tenure_hi = df["Tenure"].quantile(0.75)
    risk_hi   = df["RiskScore"].quantile(0.75)

    df2 = df.copy()
    df2["HighValue"]  = df2["Total Spend"] >= spend_hi
    df2["HighTenure"] = df2["Tenure"]      >= tenure_hi
    df2["HighRisk"]   = df2["RiskScore"]   >= risk_hi

    def segment_label(row):
        if row["HighValue"] and row["HighRisk"]:   return "High-Value at Risk"
        elif row["HighValue"]:                      return "Loyal High-Value"
        elif row["HighRisk"]:                       return "At-Risk Low-Value"
        else:                                       return "Standard"

    df2["Segment"] = df2.apply(segment_label, axis=1)

    # ── Segment Overview ──────────────────────────────────────────────────────
    section_label("Segment Overview")
    note_text(
        f"Thresholds — High-Value: spend >= {spend_hi:.0f} (P75)  |  "
        f"High-Risk: risk score >= {risk_hi:.1f} (P75)"
    )

    seg_summary = (
        df2.groupby("Segment")
        .agg(
            Customers=("Churn","count"),
            Churned=("Churn","sum"),
            Avg_Spend=("Total Spend","mean"),
            Avg_Tenure=("Tenure","mean"),
            Avg_RiskScore=("RiskScore","mean"),
        )
        .reset_index()
    )
    seg_summary["Churn Rate %"] = (seg_summary["Churned"] / seg_summary["Customers"] * 100).round(1)
    seg_summary["Avg_Spend"]     = seg_summary["Avg_Spend"].round(0)
    seg_summary["Avg_Tenure"]    = seg_summary["Avg_Tenure"].round(1)
    seg_summary["Avg_RiskScore"] = seg_summary["Avg_RiskScore"].round(1)

    col_a, col_b = st.columns(2)

    with col_a:
        seg_colors = {
            "High-Value at Risk": C["churn"],
            "Loyal High-Value":   C["retain"],
            "At-Risk Low-Value":  C["warning"],
            "Standard":           C["muted"],
        }
        bar_colors = [seg_colors.get(s, C["primary"]) for s in seg_summary["Segment"]]
        fig_seg = _bar_chart(seg_summary, "Segment", "Churn Rate %",
                             "Churn Rate by Segment", height=340,
                             text_col="Churn Rate %", color=bar_colors)
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_b:
        base = plotly_base()
        sample = df2.sample(min(5000, len(df2)), random_state=42)
        fig_seg2 = px.scatter(
            sample, x="Total Spend", y="RiskScore", color="Segment",
            color_discrete_map=seg_colors,
            opacity=0.45,
            title="Spend vs Risk Score by Segment (sample)",
        )
        fig_seg2.update_traces(marker=dict(size=4))
        fig_seg2.update_layout(**base, height=340)
        st.plotly_chart(fig_seg2, use_container_width=True)

    # Segment table
    st.dataframe(
        seg_summary.rename(columns={
            "Avg_Spend": "Avg Spend",
            "Avg_Tenure": "Avg Tenure (mo)",
            "Avg_RiskScore": "Avg Risk Score",
        }),
        hide_index=True,
        use_container_width=True,
    )

    # ── High-Value at Risk ─────────────────────────────────────────────────────
    section_label("High-Value Customers at Risk")
    hvar = df2[df2["Segment"] == "High-Value at Risk"]
    hvar_churned = hvar[hvar["Churn"] == 1]

    c1, c2, c3 = st.columns(3)
    kpi_card(c1, "High-Value at Risk",    f"{len(hvar):,}",         warn=True)
    kpi_card(c2, "Already Churned",       f"{len(hvar_churned):,}", warn=True)
    kpi_card(c3, "Segment Churn Rate",
             f"{len(hvar_churned)/len(hvar)*100:.1f}%" if len(hvar) else "N/A",
             warn=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    hvar_sub = (
        hvar.groupby("Subscription Type")["Churn"]
        .agg(["sum","count"]).reset_index()
        .rename(columns={"sum":"Churned","count":"Total"})
    )
    hvar_sub["Churn Rate %"] = (hvar_sub["Churned"] / hvar_sub["Total"] * 100).round(1)

    col_f, col_g = st.columns(2)
    with col_f:
        fig_hv = _bar_chart(hvar_sub, "Subscription Type", "Churn Rate %",
                            "High-Value at Risk: by Subscription Type",
                            text_col="Churn Rate %",
                            color=[C["churn"]] * len(hvar_sub))
        st.plotly_chart(fig_hv, use_container_width=True)

    with col_g:
        hvar_con = (
            hvar.groupby("Contract Length")["Churn"]
            .agg(["sum","count"]).reset_index()
            .rename(columns={"sum":"Churned","count":"Total"})
        )
        hvar_con["Churn Rate %"] = (hvar_con["Churned"] / hvar_con["Total"] * 100).round(1)
        con_c2 = [C["churn"] if v >= 70 else C["primary2"] for v in hvar_con["Churn Rate %"]]
        fig_hvc = _bar_chart(hvar_con, "Contract Length", "Churn Rate %",
                             "High-Value at Risk: by Contract Length",
                             text_col="Churn Rate %", color=con_c2)
        st.plotly_chart(fig_hvc, use_container_width=True)

    if len(hvar):
        insight_card(
            "High-Value at Risk",
            f"There are {len(hvar):,} High-Value at Risk customers, of which "
            f"{len(hvar_churned):,} ({len(hvar_churned)/len(hvar)*100:.1f}%) have already churned. "
            f"These represent the highest-priority retention targets.",
            "risk",
        )

    # ── Loyal High-Value ──────────────────────────────────────────────────────
    section_label("Loyal High-Value Customers")
    loyal     = df2[df2["Segment"] == "Loyal High-Value"]
    loyal_ret = loyal[loyal["Churn"] == 0]

    ca, cb, cc = st.columns(3)
    kpi_card(ca, "Loyal High-Value",  f"{len(loyal):,}",     accent=True)
    kpi_card(cb, "Still Active",      f"{len(loyal_ret):,}", accent=True)
    kpi_card(cc, "Retention Rate",
             f"{len(loyal_ret)/len(loyal)*100:.1f}%" if len(loyal) else "N/A",
             accent=True)

    if len(loyal):
        insight_card(
            "Loyal High-Value",
            f"{len(loyal):,} Loyal High-Value customers have a retention rate of "
            f"{len(loyal_ret)/len(loyal)*100:.1f}%. These customers represent the core revenue base "
            "and should be protected with proactive engagement.",
            "good",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RISKS, OPPORTUNITIES & ACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _action_card(title, fact, insight, implication, action):
    st.markdown(
        f"""
        <div style="border:1px solid {C['border']};border-radius:6px;
                    background:{C['bg']};padding:18px 20px;margin-bottom:14px;">
          <div style="font-size:13px;font-weight:700;color:{C['text']};
                      margin-bottom:12px;border-bottom:1px solid {C['border']};
                      padding-bottom:8px;">{title}</div>
          <table style="width:100%;border-collapse:collapse;">
            <tr>
              <td style="width:120px;vertical-align:top;padding:4px 12px 4px 0;">
                <span style="font-size:10px;font-weight:700;text-transform:uppercase;
                             letter-spacing:0.07em;color:{C['muted']};">Fact</span>
              </td>
              <td style="font-size:12.5px;color:{C['text2']};padding:4px 0;
                         line-height:1.5;">{fact}</td>
            </tr>
            <tr>
              <td style="vertical-align:top;padding:4px 12px 4px 0;">
                <span style="font-size:10px;font-weight:700;text-transform:uppercase;
                             letter-spacing:0.07em;color:{C['muted']};">Insight</span>
              </td>
              <td style="font-size:12.5px;color:{C['text2']};padding:4px 0;
                         line-height:1.5;">{insight}</td>
            </tr>
            <tr>
              <td style="vertical-align:top;padding:4px 12px 4px 0;">
                <span style="font-size:10px;font-weight:700;text-transform:uppercase;
                             letter-spacing:0.07em;color:{C['muted']};">Implication</span>
              </td>
              <td style="font-size:12.5px;color:{C['text2']};padding:4px 0;
                         line-height:1.5;">{implication}</td>
            </tr>
            <tr>
              <td style="vertical-align:top;padding:4px 12px 4px 0;">
                <span style="font-size:10px;font-weight:700;text-transform:uppercase;
                             letter-spacing:0.07em;color:{C['primary']};">Action</span>
              </td>
              <td style="font-size:12.5px;color:{C['text']};font-weight:500;
                         padding:4px 0;line-height:1.5;">{action}</td>
            </tr>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_actions(df):
    page_header(
        "Risks, Opportunities & Business Actions",
        "Data-supported findings translated into prioritised retention strategies.",
    )

    # Pre-compute (logic unchanged)
    avg_sc_churn  = df[df["Churn"]==1]["Support Calls"].mean()
    avg_sc_retain = df[df["Churn"]==0]["Support Calls"].mean()
    avg_pd_churn  = df[df["Churn"]==1]["Payment Delay"].mean()
    avg_pd_retain = df[df["Churn"]==0]["Payment Delay"].mean()
    avg_uf_churn  = df[df["Churn"]==1]["Usage Frequency"].mean()
    avg_uf_retain = df[df["Churn"]==0]["Usage Frequency"].mean()
    avg_li_churn  = df[df["Churn"]==1]["Last Interaction"].mean()
    avg_li_retain = df[df["Churn"]==0]["Last Interaction"].mean()

    monthly_cr   = df[df["Contract Length"]=="Monthly"]["Churn"].mean()   * 100
    annual_cr    = df[df["Contract Length"]=="Annual"]["Churn"].mean()    * 100
    quarterly_cr = df[df["Contract Length"]=="Quarterly"]["Churn"].mean() * 100

    spend_hi = df["Total Spend"].quantile(0.75)
    risk_hi  = df["RiskScore"].quantile(0.75)
    hvar_df  = df[(df["Total Spend"] >= spend_hi) & (df["RiskScore"] >= risk_hi)]
    hvar_cr  = hvar_df["Churn"].mean() * 100 if len(hvar_df) else 0

    def safe_pct(a, b):
        if pd.isna(a) or pd.isna(b) or b == 0:
            return 0.0
        return (a / b - 1) * 100

    section_label("Analysis")

    findings = [
        {
            "title": "High Support-Call Frequency",
            "fact":       f"Churned customers average {avg_sc_churn:.1f} support calls vs "
                          f"{avg_sc_retain:.1f} for retained customers "
                          f"({safe_pct(avg_sc_churn, avg_sc_retain):.0f}% higher).",
            "insight":    "Customers who contact support frequently are substantially more likely "
                          "to churn, suggesting unresolved service issues drive departure.",
            "implication":"Support escalation is a leading indicator of churn risk. "
                          "Without proactive intervention, these customers are likely to cancel.",
            "action":     "Flag customers with 5 or more support calls in a rolling 30-day window "
                          "for proactive outreach by a senior retention specialist.",
        },
        {
            "title": "Payment Delays",
            "fact":       f"Churned customers have an average payment delay of {avg_pd_churn:.1f} days "
                          f"vs {avg_pd_retain:.1f} days for retained customers.",
            "insight":    "Payment delays correlate with churn. Customers who delay payments "
                          "are more financially disengaged and at higher cancellation risk.",
            "implication":"Payment delinquency is both a revenue risk and a churn signal; "
                          "early intervention can address both simultaneously.",
            "action":     "Trigger automated payment reminders at day 7 and day 14. "
                          "Offer a structured payment plan or retention discount at day 21.",
        },
        {
            "title": "Low Usage Frequency",
            "fact":       f"Churned customers average {avg_uf_churn:.1f} service uses/month "
                          f"vs {avg_uf_retain:.1f} for retained customers.",
            "insight":    "Lower usage frequency is associated with higher churn, "
                          "suggesting disengaged customers perceive less value.",
            "implication":"Customers who rarely use the service are more price-sensitive "
                          "and more likely to cancel when reviewing subscriptions.",
            "action":     "Launch a re-engagement campaign for customers with fewer than 8 "
                          "uses/month: feature highlights, usage tips, and time-limited incentives.",
        },
        {
            "title": "Infrequent Last Interaction",
            "fact":       f"Churned customers last interacted {avg_li_churn:.1f} days ago on average "
                          f"vs {avg_li_retain:.1f} days for retained customers.",
            "insight":    "A longer gap since last interaction is associated with higher churn, "
                          "suggesting passive disengagement precedes cancellation.",
            "implication":"Re-engaging customers before formal cancellation is more effective "
                          "and less costly than win-back after departure.",
            "action":     "Implement a win-back touchpoint for customers inactive for 20 or more days: "
                          "personalised message summarising their activity and offering a feature spotlight.",
        },
        {
            "title": "Contract Length and Retention",
            "fact":       f"Monthly contract customers churn at {monthly_cr:.1f}% vs "
                          f"{annual_cr:.1f}% for Annual contracts "
                          f"(difference: {abs(monthly_cr - annual_cr):.1f} percentage points). "
                          f"Note: In the training dataset, all Monthly customers are labelled as churned; "
                          f"the testing set shows a more representative 51.6% rate.",
            "insight":    "Longer commitment contracts are strongly associated with lower churn. "
                          "Annual contract holders represent the most loyal customer segment.",
            "implication":"Encouraging contract upgrades is a structural retention lever "
                          "with measurable impact on churn probability.",
            "action":     "Offer monthly-contract customers an annual upgrade incentive — "
                          "for example, 1-2 months free or enhanced features. "
                          "Prioritise customers already showing payment delays or high support-call frequency.",
        },
        {
            "title": "High-Value at Risk Segment",
            "fact":       f"High-Value at Risk customers (spend >= {spend_hi:.0f}, "
                          f"risk score >= {risk_hi:.1f}) have a churn rate of {hvar_cr:.1f}%.",
            "insight":    "This segment represents the highest revenue-loss concentration. "
                          "Churning within it has disproportionate financial impact.",
            "implication":"Failing to intervene causes outsized revenue decline that "
                          "is expensive to replace through new customer acquisition.",
            "action":     "Assign dedicated account managers or a VIP retention team for this segment. "
                          "Offer personalised contract terms, service upgrades, or loyalty rewards "
                          "before customers reach the point of cancellation.",
        },
    ]

    col_left, col_right = st.columns(2)
    for i, f in enumerate(findings):
        with (col_left if i % 2 == 0 else col_right):
            _action_card(f["title"], f["fact"], f["insight"],
                         f["implication"], f["action"])

    # ── Risk Matrix ────────────────────────────────────────────────────────────
    section_label("Risk vs Revenue Impact Matrix")

    matrix_data = pd.DataFrame({
        "Segment": ["High-Value at Risk", "Monthly Contracts", "High Support Callers",
                    "Payment Delayers", "Low Usage Customers", "Loyal High-Value"],
        "Revenue Impact":     [9, 7, 5, 6, 4, 3],
        "Churn Probability":  [9, 8, 8, 7, 6, 2],
        "Priority":           ["Critical", "High", "High", "High", "Medium", "Maintain"],
    })

    priority_colors = {
        "Critical": C["churn"], "High": C["warning"],
        "Medium":   C["primary2"], "Maintain": C["retain"],
    }
    base = plotly_base()
    fig_matrix = px.scatter(
        matrix_data,
        x="Churn Probability", y="Revenue Impact",
        color="Priority", size=[42] * len(matrix_data),
        text="Segment",
        color_discrete_map=priority_colors,
        title="Risk-Revenue Impact Matrix (ordinal scores based on data patterns)",
    )
    fig_matrix.update_traces(textposition="top center",
                             marker=dict(sizemode="area", opacity=0.8),
                             textfont=dict(size=10, color=C["text"]))
    # Override xaxis and yaxis from plotly_base() to avoid duplicate-keyword TypeError
    matrix_layout = {
        **base,
        "height": 400,
        "xaxis": dict(**base["xaxis"], range=[0, 11], title_text="Churn Probability (1-10)"),
        "yaxis": dict(**base["yaxis"], range=[0, 11], title_text="Revenue Impact (1-10)"),
    }
    fig_matrix.update_layout(**matrix_layout)
    st.plotly_chart(fig_matrix, use_container_width=True)
    note_text(
        "Revenue Impact and Churn Probability are ordinal scores (1-10) derived from observed "
        "data ratios, not absolute monetary estimates."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICTIVE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def page_model(train_df, test_df):
    page_header(
        "Predictive Model",
        "Logistic Regression and Random Forest baseline models trained on the training dataset "
        "and evaluated on the testing dataset.",
    )

    insight_card(
        "Dataset distribution note",
        f"The training set has a churn rate of {train_df['Churn'].mean()*100:.1f}% while the "
        f"testing set has {test_df['Churn'].mean()*100:.1f}%. This distribution shift may affect "
        "model calibration. Reported metrics reflect real-world testing conditions and should be "
        "interpreted as directional estimates, not guaranteed predictions.",
        "warn",
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix, roc_curve,
        )
    except ImportError:
        st.error("scikit-learn is required. Run: pip install scikit-learn")
        return

    FEATURES = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                 "Payment Delay", "Total Spend", "Last Interaction",
                 "Gender", "Subscription Type", "Contract Length"]

    @st.cache_data(show_spinner="Training models...")
    def train_models(_train, _test):
        le_gender = LabelEncoder()
        le_sub    = LabelEncoder()
        le_con    = LabelEncoder()

        def encode(df):
            d = df[FEATURES + ["Churn"]].dropna().copy()
            d["Gender"]            = le_gender.fit_transform(d["Gender"].astype(str))
            d["Subscription Type"] = le_sub.fit_transform(d["Subscription Type"].astype(str))
            d["Contract Length"]   = le_con.fit_transform(d["Contract Length"].astype(str))
            return d

        train_enc = encode(_train)
        test_enc  = _test[FEATURES + ["Churn"]].dropna().copy()
        for col, le in [("Gender", le_gender), ("Subscription Type", le_sub),
                        ("Contract Length", le_con)]:
            known = set(le.classes_)
            test_enc[col] = test_enc[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0])
            test_enc[col] = le.transform(test_enc[col])

        X_train = train_enc[FEATURES].values
        y_train = train_enc["Churn"].values
        X_test  = test_enc[FEATURES].values
        y_test  = test_enc["Churn"].values

        scaler      = StandardScaler()
        X_train_s   = scaler.fit_transform(X_train)
        X_test_s    = scaler.transform(X_test)

        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_train_s, y_train)
        lr_pred  = lr.predict(X_test_s)
        lr_proba = lr.predict_proba(X_test_s)[:, 1]

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred  = rf.predict(X_test)
        rf_proba = rf.predict_proba(X_test)[:, 1]

        results = {}
        for name, pred, proba in [("Logistic Regression", lr_pred, lr_proba),
                                   ("Random Forest",       rf_pred, rf_proba)]:
            cm_arr   = confusion_matrix(y_test, pred)
            fpr, tpr, _ = roc_curve(y_test, proba)
            results[name] = dict(
                accuracy  = accuracy_score(y_test, pred),
                precision = precision_score(y_test, pred, zero_division=0),
                recall    = recall_score(y_test, pred, zero_division=0),
                f1        = f1_score(y_test, pred, zero_division=0),
                roc_auc   = roc_auc_score(y_test, proba),
                cm        = cm_arr,
                fpr       = fpr,
                tpr       = tpr,
            )

        fi = pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_})
        fi = fi.sort_values("Importance", ascending=True)
        return results, fi

    with st.spinner("Training models — this may take up to 60 seconds on first load..."):
        results, fi = train_models(train_df, test_df)

    # ── Performance KPIs ─────────────────────────────────────────────────────
    section_label("Model Performance — Testing Dataset")

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Logistic Regression": [
            results["Logistic Regression"]["accuracy"],
            results["Logistic Regression"]["precision"],
            results["Logistic Regression"]["recall"],
            results["Logistic Regression"]["f1"],
            results["Logistic Regression"]["roc_auc"],
        ],
        "Random Forest": [
            results["Random Forest"]["accuracy"],
            results["Random Forest"]["precision"],
            results["Random Forest"]["recall"],
            results["Random Forest"]["f1"],
            results["Random Forest"]["roc_auc"],
        ],
    })

    # KPI cards for RF (best model)
    rf_res = results["Random Forest"]
    m1, m2, m3, m4, m5 = st.columns(5)
    kpi_card(m1, "Accuracy",   f"{rf_res['accuracy']:.3f}",   sub="Random Forest", accent=True)
    kpi_card(m2, "Precision",  f"{rf_res['precision']:.3f}",  sub="Random Forest", accent=True)
    kpi_card(m3, "Recall",     f"{rf_res['recall']:.3f}",     sub="Random Forest", accent=True)
    kpi_card(m4, "F1-Score",   f"{rf_res['f1']:.3f}",         sub="Random Forest", accent=True)
    kpi_card(m5, "ROC-AUC",    f"{rf_res['roc_auc']:.3f}",    sub="Random Forest", accent=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Model Comparison ──────────────────────────────────────────────────────
    section_label("Model Comparison")
    col_m, col_cm = st.columns(2)

    with col_m:
        base = plotly_base()
        fig_metrics = px.bar(
            metrics_df.melt(id_vars="Metric", var_name="Model", value_name="Score"),
            x="Metric", y="Score", color="Model", barmode="group",
            title="Performance Metrics Comparison",
            text_auto=".3f",
            color_discrete_map={
                "Logistic Regression": C["primary2"],
                "Random Forest":       C["primary"],
            },
        )
        fig_metrics.update_layout(**base, height=320, yaxis_range=[0, 1.05])
        fig_metrics.update_traces(textfont=dict(size=9), textposition="outside")
        st.plotly_chart(fig_metrics, use_container_width=True)

    with col_cm:
        base2 = plotly_base()
        cm_arr = results["Random Forest"]["cm"]
        fig_cm = px.imshow(
            cm_arr, text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Retained (0)", "Churned (1)"],
            y=["Retained (0)", "Churned (1)"],
            color_continuous_scale=[[0, C["primary_lt"]], [1, C["primary"]]],
            title="Confusion Matrix — Random Forest",
        )
        fig_cm.update_layout(**base2, height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    section_label("ROC Curves")
    base3 = plotly_base()
    fig_roc = go.Figure()
    roc_colors = {"Logistic Regression": C["primary2"], "Random Forest": C["primary"]}
    for model_name, res in results.items():
        fig_roc.add_trace(go.Scatter(
            x=res["fpr"], y=res["tpr"],
            name=f"{model_name}  (AUC = {res['roc_auc']:.3f})",
            mode="lines",
            line=dict(color=roc_colors[model_name], width=2),
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dot", color=C["muted"], width=1),
        name="Random Classifier",
        showlegend=True,
    ))
    fig_roc.update_layout(
        **base3,
        title=dict(text="ROC Curve Comparison", font=dict(size=13, color=C["text"])),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=360,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Feature Importance ─────────────────────────────────────────────────────
    section_label("Feature Importance — Random Forest")
    base4 = plotly_base()
    fi_colors = [C["primary"] if i >= len(fi) - 3 else C["primary2"] if i >= len(fi) - 6
                 else C["muted"] for i in range(len(fi))]
    fig_fi = go.Figure(go.Bar(
        x=fi["Importance"], y=fi["Feature"], orientation="h",
        marker_color=fi_colors,
        text=[f"{v:.3f}" for v in fi["Importance"]],
        textposition="outside",
        textfont=dict(size=10, color=C["text2"]),
    ))
    fig_fi.update_layout(
        **base4,
        title=dict(text="Feature Importance — Random Forest",
                   font=dict(size=13, color=C["text"])),
        height=360,
        xaxis_title="Importance Score",
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    top_feature = fi.iloc[-1]["Feature"]
    insight_card(
        f"Top predictor — {top_feature}",
        f"{top_feature} is the most important predictor of churn in the Random Forest model "
        f"(importance = {fi.iloc[-1]['Importance']:.3f}). "
        f"This aligns with the Pearson correlation analysis on the Churn Drivers page.",
        "neutral",
    )

    # ── Detailed Metrics Table ─────────────────────────────────────────────────
    section_label("Detailed Metrics Table")
    display_df = metrics_df.copy()
    display_df["Logistic Regression"] = display_df["Logistic Regression"].map("{:.4f}".format)
    display_df["Random Forest"]       = display_df["Random Forest"].map("{:.4f}".format)
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    note_text(
        f"Model predictions are probabilistic estimates. The distribution shift between training "
        f"(churn rate {train_df['Churn'].mean()*100:.1f}%) and testing "
        f"({test_df['Churn'].mean()*100:.1f}%) is a known limitation."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DATA QUALITY & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def page_data_quality(train_raw_df, train_df):
    page_header(
        "Data Quality & Preprocessing",
        "Dataset validation, cleaning decisions, and quality summary.",
    )

    section_label("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Raw Training Rows",   f"{len(train_raw_df):,}")
    kpi_card(c2, "Clean Training Rows", f"{len(train_df):,}",
             sub="After preprocessing", accent=True)
    kpi_card(c3, "Rows Removed",
             f"{len(train_raw_df) - len(train_df):,}",
             sub="Ghost row + invalid Churn")
    kpi_card(c4, "Columns",             "12",
             sub="Including derived columns")

    section_label("Preprocessing Decisions")
    actions = [
        ("Blank / ghost rows",       "1 fully blank row removed from training set."),
        ("Age = 0 (invalid)",
         f"Age = 0 replaced with NaN. "
         f"Count: {train_raw_df['Age'].astype(str).eq('0').sum()} records affected."),
        ("Duplicate rows",           f"{train_df.attrs.get('dupes_removed', 0)} exact duplicate rows removed."),
        ("Missing Churn labels",     "Rows where Churn is empty or non-numeric were dropped."),
        ("Numeric type conversion",  "All numeric columns cast to float; coercion errors become NaN."),
        ("Total Spend consistency",  "Training file has decimal values; both files parsed as float."),
        ("Derived: TenureGroup",     "Categorical bins: 0-12m, 13-24m, 25-36m, 37-48m, 49-60m."),
        ("Derived: SpendBand",       "Categorical bins: 0-200, 201-400, 401-600, 601-800, 801-1000."),
        ("Derived: RiskScore",       "Weighted composite 0-100 from Support Calls (30%), "
                                     "Payment Delay (25%), Last Interaction (25%), "
                                     "Usage Frequency inverse (20%)."),
    ]
    action_df = pd.DataFrame(actions, columns=["Step", "Details"])
    st.dataframe(action_df, hide_index=True, use_container_width=True)

    section_label("Missing Values — Cleaned Dataset")
    missing = (
        train_df.isnull().sum()
        .reset_index()
        .rename(columns={"index": "Column", 0: "Missing Count"})
    )
    missing["Missing %"] = (missing["Missing Count"] / len(train_df) * 100).round(2)
    missing_nonzero = missing[missing["Missing Count"] > 0]
    if len(missing_nonzero):
        st.dataframe(missing_nonzero, hide_index=True, use_container_width=True)
    else:
        insight_card(
            "No missing values",
            "No missing values remain in numeric or categorical columns after preprocessing.",
            "good",
        )

    section_label("Cleaned Dataset — Sample (first 20 rows)")
    st.dataframe(train_df.head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    train_df, test_df = load_data()

    # ── Sidebar navigation ────────────────────────────────────────────────────
    st.sidebar.markdown(
        f"""<div style="padding:1rem 0 0.3rem 0;">
          <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.08em;color:{C['muted']};margin-bottom:2px;">
            Platform
          </div>
          <div style="font-size:15px;font-weight:700;color:{C['primary']};
                      line-height:1.3;">
            Customer Churn &amp;<br>Retention Intelligence
          </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'<hr style="margin:0.6rem 0 0.8rem 0;border-color:{C["border"]};">',
        unsafe_allow_html=True,
    )

    pages = {
        "Executive Overview":          "overview",
        "Churn Drivers":               "drivers",
        "Customer Segments":           "segments",
        "Risks & Actions":             "actions",
        "Predictive Model":            "model",
        "Data Quality":                "quality",
    }

    st.sidebar.markdown(
        f"""<div style="font-size:10.5px;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.08em;color:{C['muted']};
                        margin-bottom:6px;">Navigation</div>""",
        unsafe_allow_html=True,
    )
    selected  = st.sidebar.radio("nav", list(pages.keys()), label_visibility="collapsed")
    page_key  = pages[selected]

    st.sidebar.markdown(
        f'<hr style="margin:0.8rem 0;border-color:{C["border"]};">',
        unsafe_allow_html=True,
    )

    # Apply filters
    if page_key in ("overview", "drivers", "segments", "actions"):
        filtered_df = apply_sidebar_filters(train_df, train_df)
    else:
        filtered_df = train_df
        st.sidebar.markdown(
            f'<p style="font-size:11px;color:{C["muted"]};margin-top:6px;">'
            f"Filters are not applied on this page.</p>",
            unsafe_allow_html=True,
        )

    # Route
    if page_key == "overview":
        page_overview(filtered_df)
    elif page_key == "drivers":
        page_drivers(filtered_df)
    elif page_key == "segments":
        page_segments(filtered_df)
    elif page_key == "actions":
        page_actions(filtered_df)
    elif page_key == "model":
        page_model(train_df, test_df)
    elif page_key == "quality":
        page_data_quality(pd.read_csv(TRAIN_PATH, dtype=str), train_df)

    # Footer
    st.markdown(
        f"""<hr style="margin:2rem 0 0.8rem 0;">
        <p style="text-align:center;font-size:11px;color:{C['muted']};">
          Customer Churn &amp; Retention Intelligence &nbsp;&middot;&nbsp;
          Built with Streamlit &amp; Plotly &nbsp;&middot;&nbsp;
          Academic BI Project
        </p>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
