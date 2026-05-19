# E-Commerce Customer Behavior Analysis & Intelligence Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview

A **production-grade, end-to-end data analytics project** that transforms raw Indian e-commerce transaction data into actionable business intelligence.

The project covers the complete data analytics lifecycle:
- **Data Engineering** — generation, cleaning, validation, and feature engineering
- **Statistical Analysis** — RFM modelling, cohort analysis, customer lifetime value
- **Machine Learning** — K-Means customer clustering with PCA visualisation
- **Business Intelligence** — automated KPI generation and strategic recommendations
- **Interactive Dashboard** — 5-page Streamlit app with real-time filtering

This project demonstrates skills directly relevant to Data Analyst and Data Science roles: end-to-end pipeline design, customer analytics, stakeholder reporting, and dashboard delivery.

---

## Business Problem & Objectives

An Indian e-commerce company wants to understand:

1. **Who are our most valuable customers** and what drives their behaviour?
2. **Why are customers churning** and which cohorts retain best?
3. **Which products and categories** drive the most revenue and returns?
4. **Where should we invest** — which segments, regions, and channels perform best?

**Objectives:**
- Segment customers using RFM analysis and machine learning clustering
- Measure cohort retention and identify churn patterns
- Auto-generate KPIs and business recommendations from data
- Deliver an interactive dashboard for non-technical stakeholders

---

## Dataset Description

**10,000 synthetic but realistic transactions** spanning January 2022 — December 2024.

| Column | Type | Description |
|--------|------|-------------|
| order_id | string | Unique order identifier |
| customer_id | string | Customer identifier (60% repeat customers) |
| customer_name | string | Full name |
| customer_email | string | Email address |
| customer_age | int | Age 18–70 |
| customer_gender | string | Male / Female / Other |
| customer_city | string | 25 Indian cities |
| customer_state | string | Indian state |
| registration_date | datetime | Account creation date |
| order_date | datetime | Order placement date |
| order_status | string | Completed / Returned / Cancelled / Pending |
| product_id | string | Product identifier |
| product_name | string | Brand + sub-category + model |
| category | string | Electronics / Fashion / Home & Kitchen / Sports / Beauty / Books / Toys |
| sub_category | string | E.g., Smartphones, Footwear, Cookware |
| brand | string | E.g., Samsung, Nike, Philips |
| unit_price | float | Price in INR |
| quantity | int | 1–10 units |
| discount_percent | float | 0–50% |
| shipping_cost | float | 0–200 INR |
| payment_method | string | UPI / Credit Card / COD / Debit Card / Wallet / Net Banking |
| device_type | string | Mobile / Desktop / Tablet |
| rating | float | 1–5 (nullable for non-completed orders) |
| review_text | string | Short review (nullable) |
| is_returned | bool | Whether order was returned |
| delivery_days | int | 2–15 days |

**Dataset characteristics:**
- Seasonal trends: 2x higher sales volume in Oct–Dec (festive season)
- 60% repeat customers enabling meaningful cohort analysis
- Realistic Indian city/state distribution across 25 cities
- 15% missing ratings, 30% missing reviews (real-world quality)

---

## Key Analyses Performed

- **Exploratory Data Analysis** — distributions, correlations, time-series, geographic breakdown
- **RFM Analysis** — Recency / Frequency / Monetary scoring with quintile binning
- **Customer Segmentation** — 7 named RFM segments (Champions → Lost Customers)
- **K-Means Clustering** — 4 behavioural clusters with PCA visualisation
- **Cohort Retention Analysis** — 12-month retention heatmap per monthly cohort
- **Churn Rate Tracking** — monthly customer churn with trend analysis
- **Customer Lifetime Value** — CLV computation per customer
- **Revenue Trend Analysis** — YoY growth, seasonal patterns, category performance
- **Return Rate Analysis** — by category, payment method, and product
- **Geographic Intelligence** — revenue and orders by state and city

---

## Key Insights Found

1. **Q4 drives 2x revenue** — Oct–Dec festive season consistently doubles baseline monthly revenue; inventory and ad spend should be pre-positioned by September
2. **60%+ of revenue from repeat customers** — retention efforts yield higher ROI than acquisition in this dataset
3. **Champions segment (~15% of customers) generates ~40% of revenue** — loyalty programme investment is high-priority
4. **Mobile accounts for 60% of orders** — mobile checkout optimisation directly impacts conversion rate
5. **UPI is the #1 payment method (30%)** — UPI cashback incentives can further reduce COD dependency
6. **Fast delivery (<5 days) correlates with 0.4+ higher avg ratings** — logistics SLA improvements drive NPS
7. **At Risk customers had high historical value** — a 30-day win-back campaign with personalised discounts can recover significant revenue

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | Python 3.10+, Pandas 2.0, NumPy |
| Machine Learning | scikit-learn (KMeans, StandardScaler, PCA) |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit 1.32+ |
| Configuration | YAML (PyYAML) |
| Testing | Pytest |
| Notebooks | Jupyter, nbformat |
| Templating | Jinja2 |

---

## Project Structure

```
ecommerce-behavior-analysis/
│
├── data/
│   ├── raw/
│   │   └── ecommerce_data.csv          ← 10,000-row raw dataset
│   └── processed/
│       └── cleaned_data.csv            ← Cleaned + feature-engineered data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       ← EDA: distributions, trends, geography
│   ├── 02_customer_segmentation.ipynb  ← RFM analysis + KMeans clustering
│   ├── 03_cohort_analysis.ipynb        ← Retention heatmap + churn analysis
│   └── 04_business_insights.ipynb      ← Executive summary + recommendations
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  ← DataLoader class with schema validation
│   ├── data_cleaner.py                 ← DataCleaner class (6 cleaning steps)
│   ├── feature_engineering.py          ← FeatureEngineer: RFM, CLV, metrics
│   ├── customer_segmentation.py        ← CustomerSegmentation: RFM + KMeans
│   ├── cohort_analysis.py              ← CohortAnalysis: retention + churn
│   └── insights_generator.py          ← InsightsGenerator: KPIs + text report
│
├── dashboard/
│   └── app.py                          ← 5-page Streamlit dashboard
│
├── visuals/                            ← Auto-generated PNG charts
├── reports/                            ← HTML / text reports
│
├── tests/
│   └── test_data_cleaner.py            ← 20 pytest unit tests (all passing)
│
├── data/
│   └── generate_data.py                ← Synthetic dataset generator
│
├── config/
│   └── config.yaml                     ← All settings, paths, parameters
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Clone & Set Up Environment

```bash
git clone <your-repo-url>
cd ecommerce-behavior-analysis
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python data/generate_data.py
```

Output: `data/raw/ecommerce_data.csv` (10,000 rows)

### 3. Run Data Cleaning Pipeline

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
import yaml; from pathlib import Path
loader = DataLoader()
df = loader.load_raw_data()
cleaner = DataCleaner()
df_clean = cleaner.clean(df)
df_clean.to_csv('data/processed/cleaned_data.csv', index=False)
print('Done:', df_clean.shape)
"
```

### 4. Run Insights Generator

```bash
python src/insights_generator.py
```

### 5. Launch Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501` in your browser.

### 6. Open Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Open notebooks in order: 01 → 02 → 03 → 04

### 7. Run Unit Tests

```bash
pytest tests/ -v
```

Expected: **20 passed** in ~2 seconds.

---

## Dashboard Preview

The dashboard has 5 pages accessible via tabs:

| Page | Contents |
|------|----------|
| Executive Overview | Revenue trend, category breakdown, top products, state map, payment & device analysis |
| Customer Analysis | RFM segments, CLV bubble chart, top 20 customers table, age/gender distribution |
| Cohort & Retention | 12-month retention heatmap, monthly churn rate, revenue per cohort |
| Product Intelligence | Category matrix, sub-category treemap, discount impact, return rate analysis |
| Business Insights | Auto-generated KPIs, strategic recommendations, data download buttons |

All pages respond dynamically to the sidebar filters:
- Date range picker
- State multiselect
- Category multiselect
- Order status filter
- Device type filter

---

## Business Recommendations

1. **Festive Season Strategy**: Allocate 40-50% of annual marketing budget to Q4 (Oct–Dec). Pre-position high-demand Electronics inventory by mid-September to avoid stockouts during peak demand.

2. **Customer Retention Programme**: The 15% Champions segment drives ~40% of revenue. Implement a tiered loyalty programme (Silver/Gold/Platinum) with escalating benefits including free express shipping, early sale access, and dedicated support.

3. **Win-Back Campaign for At-Risk Customers**: Customers with R-Score ≤ 2 but high historical monetary value should receive a personalised win-back email with a time-limited 20% discount within 30 days of their last order.

4. **Mobile-First Checkout**: With 60% of orders from mobile devices, every 100ms reduction in mobile page load time can increase conversion rate by 1%. Prioritise mobile checkout optimisation and consider a native app for high-frequency buyers.

5. **Return Rate Reduction**: High-return categories (especially Fashion and Electronics) should be addressed with: mandatory video reviews, AI-powered size recommendation tools, and a "certified refurbished" programme for returned Electronics to recapture lost margin.

---

## Future Improvements

- [ ] Real-time data ingestion via Apache Kafka or AWS Kinesis
- [ ] A/B test analysis module for campaign measurement
- [ ] Price elasticity modelling per category
- [ ] Customer churn prediction model (XGBoost/LightGBM)
- [ ] Next-best-action recommendation engine
- [ ] Automated weekly email digest with top KPIs
- [ ] Geographic choropleth map of India with drill-down by district
- [ ] Multi-language support for regional e-commerce markets

---

## Author

**Shiva Bhukya**
Data Analyst | Python | SQL | Tableau | Machine Learning

- Email: bhukyashiva086@gmail.com
- [LinkedIn](https://linkedin.com/in/your-profile)
- [GitHub](https://github.com/your-username)

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*Built as a portfolio project demonstrating end-to-end data analytics skills including data engineering, statistical analysis, machine learning, and business intelligence dashboard delivery.*
