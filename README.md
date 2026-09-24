# Customer Churn & Retention Intelligence Dashboard

## Overview

A complete, professional Business Intelligence project that analyses customer churn behaviour and translates data patterns into actionable retention strategies. Built as an interactive Streamlit dashboard with a clean SaaS analytics design, backed by Logistic Regression and Random Forest predictive models.

The project follows the analytical hierarchy:
**KPIs → Trends → Drivers → Risks & Opportunities → Business Actions**

---

## Business Problem

Customer churn — the loss of subscribers or service users — is one of the most significant challenges for subscription-based businesses. Customer retention is an important business objective because reducing churn can help protect recurring revenue and customer relationships, which are often more cost-effective to maintain than to rebuild through new customer acquisition.

This project addresses the question:
> *Which customers are at risk of churning, what behavioural patterns are associated with churn, and what data-supported actions can the business consider to reduce customer attrition?*

---

## Objectives

- Calculate accurate KPIs from a validated 440,832-customer dataset
- Identify churn patterns across subscription types, contract lengths, and behavioural metrics
- Build data-driven customer segments (High-Value at Risk, Loyal High-Value, At-Risk, Standard)
- Train and evaluate predictive churn models (Logistic Regression and Random Forest)
- Translate all findings into prioritised, data-supported business recommendations
- Deliver findings via a professional interactive dashboard

---

## Dataset

**Dataset Source:** Kaggle — Customer Churn Dataset

**URL:** [https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?resource=download](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?resource=download)

**Citation:** Azeem, M. S. (2023). *Customer Churn Dataset*. Kaggle.

| Property | Value |
|---|---|
| Primary dataset | `customer_churn_dataset-training-master.csv` |
| Validation dataset | `customer_churn_dataset-testing-master.csv` |
| Training rows (clean) | 440,832 |
| Testing rows | 64,374 |
| Columns | 12 |
| Target variable | `Churn` (1 = churned, 0 = retained) |
| Training churn rate | 56.71% |
| Testing churn rate | 47.37% |

---

## Dataset Features

| Column | Type | Description |
|---|---|---|
| `CustomerID` | Integer | Unique customer identifier |
| `Age` | Integer | Customer age in years (Age=0 treated as invalid) |
| `Gender` | Categorical | Male / Female |
| `Tenure` | Integer | Months as a subscriber (0–60) |
| `Usage Frequency` | Integer | Service interactions per month (1–30) |
| `Support Calls` | Integer | Support/complaint calls made (0–10) |
| `Payment Delay` | Integer | Days late on last payment (0–30) |
| `Subscription Type` | Categorical | Basic / Standard / Premium |
| `Contract Length` | Categorical | Monthly / Quarterly / Annual |
| `Total Spend` | Float | Cumulative spend, 0–1000 (unit not specified) |
| `Last Interaction` | Integer | Days since last customer contact (0–30) |
| `Churn` | Binary | **Target** — 1 = churned, 0 = retained |

---

## Data Preprocessing

| Step | Action |
|---|---|
| Blank ghost rows | 1 fully blank row dropped from training set |
| Age = 0 values | Replaced with NaN (invalid; treated as missing) |
| Numeric columns | Cast to float; coercion errors → NaN |
| Missing Churn labels | Rows with missing/non-numeric Churn dropped |
| Duplicate rows | Exact duplicates removed |
| Total Spend format | Parsed as float consistently across both files |
| `TenureGroup` (derived) | 5-band categorical: 0-12m, 13-24m, 25-36m, 37-48m, 49-60m |
| `SpendBand` (derived) | 5-band categorical: 0-200, 201-400, 401-600, 601-800, 801-1000 |
| `RiskScore` (derived) | Weighted composite 0-100 from Support Calls (30%), Payment Delay (25%), Last Interaction (25%), inverse Usage Frequency (20%) |

**Note on Monthly contracts:** In the training dataset, all Monthly contract customers are labelled as churned (100%). This is an inherent characteristic of the training data. The testing dataset shows 51.6% churn for Monthly contracts. Interpret Monthly contract statistics with this caveat.

---

## Dashboard Pages

The interactive dashboard contains six pages, all accessible from the sidebar navigation.

| Page | Description |
|---|---|
| **Executive Overview** | 10 KPI cards, churn distribution donut chart, churn by subscription type / contract length / gender / tenure group, filter-aware Key Findings |
| **Churn Drivers** | Grouped bar comparison (churned vs retained), box plots, payment delay / usage / spend / interaction band charts, Pearson correlation chart with auto-generated insights |
| **Customer Segments** | Four data-driven segments, scatter plot, segment summary table, KPI cards for High-Value at Risk and Loyal High-Value |
| **Risks & Actions** | Six FACT → INSIGHT → IMPLICATION → ACTION structured cards, Risk vs Revenue Impact priority matrix |
| **Predictive Model** | Model KPI cards, performance comparison chart, confusion matrix, ROC curves, feature importance |
| **Data Quality** | Preprocessing decisions table, missing value report, cleaned data sample |

### Sidebar Filters

All data pages update dynamically when these filters are changed:
- **Gender** (All / Female / Male)
- **Subscription Type** (All / Basic / Standard / Premium)
- **Contract Length** (All / Annual / Monthly / Quarterly)
- **Churn Status** (All customers / Churned only / Retained only)
- **Tenure** (slider, full range by default)

All filters default to "All" / full range so the initial view represents the complete dataset.

---

## Key Insights

All insights are derived from the actual dataset:

1. **56.71% overall churn rate** — 249,999 of 440,832 customers have churned
2. **Support Calls is the strongest positive predictor** (Pearson r = 0.5743). Churned customers average 5.15 support calls vs 1.59 for retained — a 224% difference
3. **Total Spend is the strongest overall predictor** (r = −0.4294). Retained customers average 750 in spend vs 541 for churned customers
4. **Payment delays are higher** among churned customers (15.2 days vs 10.0 days)
5. **Monthly contracts** show 100% churn in the training dataset — this is a training-data characteristic, not a universal finding
6. **Female customers** churn at 66.67% vs 49.13% for male customers; the cause is not determinable from this dataset
7. **High-Value at Risk segment** (21,836 customers): 89.88% churn rate — highest-priority retention target
8. **Loyal High-Value segment** (88,524 customers): 70.89% retention rate — core revenue base

---

## Predictive Model

Two models are trained and evaluated:

| Model | Algorithm | Preprocessing |
|---|---|---|
| Logistic Regression | Linear baseline | StandardScaler normalisation |
| Random Forest | 100 trees, max depth 10 | Raw encoded features |

**Features used:** Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction, Gender (encoded), Subscription Type (encoded), Contract Length (encoded)

**Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC — all computed on the testing dataset at runtime.

**Limitation:** Training churn rate (56.71%) differs from testing (47.37%). This distribution shift may affect model calibration. Predictions are probabilistic estimates, not guaranteed outcomes.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.9+ | Core language |
| Streamlit | Interactive dashboard framework |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computation |
| Plotly | Interactive charts |
| scikit-learn | Predictive models |
| ReportLab | PDF report generation |

---

## Project Structure

```
.
├── app.py                                         # Main Streamlit dashboard (single file)
├── requirements.txt                               # Python dependencies
├── README.md                                      # This file
├── generate_report.py                             # PDF report generator script
├── Project_Report.pdf                             # Generated project report
├── .streamlit/
│   └── config.toml                                # Light theme configuration
├── customer_churn_dataset-training-master.csv     # Primary dataset
└── customer_churn_dataset-testing-master.csv      # Model validation dataset
```

**Submission deliverables:**
1. `app.py`
2. `requirements.txt`
3. `README.md`
4. `Project_Report.pdf`
5. GitHub repository link

---

## Installation

**Requirements:** Python 3.9 or higher

```bash
pip install -r requirements.txt
```

---

## How to Run the Application

1. Ensure the two CSV files are in the same directory as `app.py`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```
4. The dashboard opens automatically at `http://localhost:8501`

To regenerate the PDF report:
```bash
python generate_report.py
```

---

## Project Limitations

- No timestamp column — time-series trend analysis is not possible with this dataset
- Currency unit for `Total Spend` is not specified
- `Age = 0` values (small count) treated as missing; root cause unknown
- Training/testing churn distribution shift (56.71% vs 47.37%) affects model calibration
- Monthly contract 100% churn in training is a dataset-specific pattern, not a universal finding
- Segmentation thresholds (75th percentile) are analytical choices, not externally validated business rules

---

## Future Scope

- Add time-series and cohort analysis if date columns become available
- SHAP values for individual-level model explainability
- XGBoost / LightGBM for improved predictive performance
- Live database / CRM integration for real-time customer scoring
- CSV export of at-risk customer lists for CRM import
- A/B testing framework to measure retention campaign effectiveness

---

*Customer Churn & Retention Intelligence Dashboard — Academic BI Project*
