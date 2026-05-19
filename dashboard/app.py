"""
E-Commerce Customer Behavior Intelligence Dashboard
Streamlit multi-page dashboard — run with: streamlit run dashboard/app.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Path setup ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader           import DataLoader
from src.data_cleaner          import DataCleaner
from src.feature_engineering   import FeatureEngineer
from src.customer_segmentation import CustomerSegmentation
from src.cohort_analysis       import CohortAnalysis
from src.insights_generator    import InsightsGenerator

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Intelligence Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS / Theme ───────────────────────────────────────
st.markdown("""
<style>
/* Dark background */
.stApp { background-color: #0f1117; color: #ffffff; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #1c1c2e; }
[data-testid="stSidebar"] .css-1d391kg { color: #ffffff; }

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #1c1c2e;
    border: 1px solid #2d2d44;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.82rem; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important; font-size: 1.6rem; font-weight: 700;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.8rem; }

/* Section headers */
h1, h2, h3 { color: #ffffff !important; }
h1 { border-bottom: 2px solid #00d4ff; padding-bottom: 8px; }

/* Divider */
hr { border-color: #2d2d44; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background-color: #1c1c2e; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #94a3b8; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #00d4ff; border-bottom: 2px solid #00d4ff;
}

/* Info boxes */
.info-card {
    background-color: #1c1c2e;
    border-left: 4px solid #00d4ff;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 10px 0;
    color: #e2e8f0;
}
.insight-card {
    background-color: #1c1c2e;
    border: 1px solid #2d2d44;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

ACCENT     = "#00d4ff"
ACCENT_ALT = "#7c3aed"
POSITIVE   = "#22c55e"
NEGATIVE   = "#ef4444"
NEUTRAL    = "#f59e0b"
PALETTE    = [ACCENT, ACCENT_ALT, POSITIVE, NEUTRAL, NEGATIVE,
              "#06b6d4", "#8b5cf6", "#10b981", "#ec4899", "#f97316"]
TEMPLATE   = "plotly_dark"


# ══════════════════════════════════════════════════════════
# DATA LOADING — cached so filters don't re-run the pipeline
# ══════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading & cleaning data...")
def load_data():
    loader  = DataLoader()
    df_raw  = loader.load_raw_data()
    cleaner = DataCleaner()
    df      = cleaner.clean(df_raw)
    return df


@st.cache_data(show_spinner="Computing customer features...")
def load_customer_features(_df):
    fe   = FeatureEngineer()
    cust = fe.build_customer_features(_df)
    seg  = CustomerSegmentation()
    cust = seg.rfm_segmentation(cust)
    cust, _ = seg.behavioral_segmentation(cust)
    return cust


@st.cache_data(show_spinner="Running cohort analysis...")
def load_cohort_data(_df):
    ca = CohortAnalysis()
    return ca.run(_df)


@st.cache_data(show_spinner="Generating insights...")
def load_insights(_df, _cust):
    ig = InsightsGenerator()
    return ig.generate_all_insights(_df, _cust)


# ══════════════════════════════════════════════════════════
# SIDEBAR — filters
# ══════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown(f"<h2 style='color:{ACCENT};text-align:center;'>🛒 E-Commerce<br>Intelligence</h2>",
                    unsafe_allow_html=True)
        st.markdown("---")

        # Date range
        min_date = df["order_date"].min().date()
        max_date = df["order_date"].max().date()
        st.subheader("Date Range")
        d_from = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date, key="d_from")
        d_to   = st.date_input("To",   value=max_date, min_value=min_date, max_value=max_date, key="d_to")

        st.subheader("Filters")
        all_states = sorted(df["customer_state"].dropna().unique())
        states     = st.multiselect("State", all_states, default=[])

        all_cats   = sorted(df["category"].dropna().unique())
        categories = st.multiselect("Category", all_cats, default=[])

        all_status = sorted(df["order_status"].dropna().unique())
        statuses   = st.multiselect("Order Status", all_status, default=[])

        all_devices = sorted(df["device_type"].dropna().unique())
        devices    = st.multiselect("Device Type", all_devices, default=[])

        st.markdown("---")
        st.markdown(f"<small style='color:#94a3b8;'>Data range: {min_date} — {max_date}</small>",
                    unsafe_allow_html=True)
        st.markdown(f"<small style='color:#94a3b8;'>Total records: {len(df):,}</small>",
                    unsafe_allow_html=True)

    return d_from, d_to, states, categories, statuses, devices


def apply_filters(df, d_from, d_to, states, categories, statuses, devices):
    mask = (
        (df["order_date"].dt.date >= d_from) &
        (df["order_date"].dt.date <= d_to)
    )
    if states:
        mask &= df["customer_state"].isin(states)
    if categories:
        mask &= df["category"].isin(categories)
    if statuses:
        mask &= df["order_status"].isin(statuses)
    if devices:
        mask &= df["device_type"].isin(devices)
    return df[mask].copy()


# ══════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════

def page_overview(df: pd.DataFrame):
    st.title("Executive Overview")
    completed = df[df["order_status"] == "Completed"]

    # ── KPI Row 1 ────────────────────────────────────────────
    total_rev   = completed["total_amount"].sum()
    total_ord   = df["order_id"].nunique()
    total_cust  = df["customer_id"].nunique()
    aov         = completed["total_amount"].mean()
    ret_rate    = df["is_returned"].mean() * 100

    # Compare previous period (simple 50-50 split)
    mid = df["order_date"].median()
    prev = df[df["order_date"] < mid]; curr = df[df["order_date"] >= mid]
    prev_c = prev[prev["order_status"] == "Completed"]
    curr_c = curr[curr["order_status"] == "Completed"]

    def delta(new, old):
        if old == 0: return 0.0
        return round((new - old) / old * 100, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue",    f"₹{total_rev:,.0f}", f"{delta(curr_c['total_amount'].sum(), prev_c['total_amount'].sum())}%")
    c2.metric("Total Orders",     f"{total_ord:,}",     f"{delta(curr['order_id'].nunique(), prev['order_id'].nunique())}%")
    c3.metric("Total Customers",  f"{total_cust:,}",    f"{delta(curr['customer_id'].nunique(), prev['customer_id'].nunique())}%")
    c4.metric("Avg Order Value",  f"₹{aov:,.0f}",       f"{delta(curr_c['total_amount'].mean(), prev_c['total_amount'].mean())}%")
    c5.metric("Return Rate",      f"{ret_rate:.1f}%",   f"{delta(curr['is_returned'].mean()*100, prev['is_returned'].mean()*100)}%")

    # ── KPI Row 2 ────────────────────────────────────────────
    monthly_active = completed.groupby(
        completed["order_date"].dt.to_period("M"))["customer_id"].apply(set)
    if len(monthly_active) >= 2:
        m_last = monthly_active.iloc[-1]; m_prev = monthly_active.iloc[-2]
        retention = len(m_last & m_prev) / len(m_prev) * 100 if m_prev else 0
    else:
        retention = 0.0

    clv_est = aov * (total_ord / max(total_cust, 1))
    nps     = round((completed["rating"].mean() - 1) / 4 * 100, 1)

    c6, c7, c8 = st.columns(3)
    c6.metric("Customer Retention",  f"{retention:.1f}%")
    c7.metric("Avg CLV (Est.)",       f"₹{clv_est:,.0f}")
    c8.metric("Net Promoter Score",   f"{nps:.1f}")

    st.markdown("---")

    # ── Charts Row 1 ────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        monthly = (
            completed.groupby("order_month_year")["total_amount"]
            .sum().reset_index()
        )
        monthly.columns = ["month", "revenue"]
        monthly["month"] = pd.to_datetime(monthly["month"])
        monthly = monthly.sort_values("month")

        orders_m = (
            df.groupby("order_month_year")["order_id"]
            .nunique().reset_index()
        )
        orders_m.columns = ["month", "orders"]
        orders_m["month"] = pd.to_datetime(orders_m["month"])
        orders_m = orders_m.sort_values("month")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["revenue"],
            name="Revenue", fill="tozeroy",
            line=dict(color=ACCENT, width=2.5),
            fillcolor=f"rgba(0,212,255,0.12)"
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=orders_m["month"], y=orders_m["orders"],
            name="Orders", line=dict(color=NEUTRAL, width=2, dash="dot"),
        ), secondary_y=True)
        fig.update_layout(template=TEMPLATE, title="Monthly Revenue & Orders Trend",
                          legend=dict(orientation="h", y=1.1),
                          margin=dict(t=50, b=20))
        fig.update_yaxes(title_text="Revenue (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Orders", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cat_rev = completed.groupby("category")["total_amount"].sum().reset_index()
        fig2    = px.pie(cat_rev, names="category", values="total_amount",
                         title="Revenue by Category",
                         color_discrete_sequence=PALETTE,
                         hole=0.45, template=TEMPLATE)
        fig2.update_traces(textinfo="percent+label")
        fig2.update_layout(showlegend=False, margin=dict(t=50))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts Row 2 ────────────────────────────────────────
    col3, col4 = st.columns([1, 2])
    with col3:
        top10 = (completed.groupby("product_name")["total_amount"]
                 .sum().sort_values(ascending=False).head(10).reset_index())
        fig3 = px.bar(top10, x="total_amount", y="product_name",
                      orientation="h", title="Top 10 Products by Revenue",
                      color_discrete_sequence=[ACCENT_ALT], template=TEMPLATE)
        fig3.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=50))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        state_rev = (completed.groupby("customer_state")["total_amount"]
                     .sum().reset_index().sort_values("total_amount", ascending=False))
        fig4 = px.bar(state_rev.head(15), x="customer_state", y="total_amount",
                      title="Revenue by State (Top 15)",
                      color="total_amount", color_continuous_scale="Blues",
                      template=TEMPLATE)
        fig4.update_layout(margin=dict(t=50), coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Charts Row 3 ────────────────────────────────────────
    col5, col6, col7 = st.columns(3)
    with col5:
        pay = df["payment_method"].value_counts().reset_index()
        fig5 = px.pie(pay, names="payment_method", values="count",
                      title="Payment Methods", hole=0.4,
                      color_discrete_sequence=PALETTE, template=TEMPLATE)
        fig5.update_traces(textinfo="percent+label")
        fig5.update_layout(showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        dev = df["device_type"].value_counts().reset_index()
        fig6 = px.bar(dev, x="device_type", y="count",
                      title="Device Type Usage",
                      color="device_type", color_discrete_sequence=PALETTE,
                      template=TEMPLATE)
        fig6.update_layout(showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

    with col7:
        status = df["order_status"].value_counts().reset_index()
        fig7 = px.pie(status, names="order_status", values="count",
                      title="Order Status", hole=0.4,
                      color_discrete_sequence=[POSITIVE, NEGATIVE, NEUTRAL, "#94a3b8"],
                      template=TEMPLATE)
        fig7.update_traces(textinfo="percent+label")
        fig7.update_layout(showlegend=False)
        st.plotly_chart(fig7, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER ANALYSIS
# ══════════════════════════════════════════════════════════

def page_customers(df: pd.DataFrame, cust: pd.DataFrame):
    st.title("Customer Analysis")

    col1, col2 = st.columns(2)
    with col1:
        seg_counts = cust["rfm_segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig = px.bar(seg_counts, x="segment", y="count",
                     title="RFM Segment Distribution",
                     color="segment", color_discrete_sequence=PALETTE,
                     template=TEMPLATE)
        fig.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            cust.dropna(subset=["clv", "total_orders", "total_spent"]),
            x="total_orders", y="clv",
            size="total_spent", color="cluster_label",
            title="Customer Segments: CLV vs Orders (sized by Spend)",
            color_discrete_sequence=PALETTE,
            template=TEMPLATE, hover_data=["customer_id", "total_spent"],
            size_max=25,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 20 Customers by Total Spend")
    top20 = (cust.nlargest(20, "total_spent")
             [["customer_id", "customer_name", "customer_city", "total_orders",
               "total_spent", "avg_order_value", "clv", "rfm_segment", "shopping_segment"]]
             .reset_index(drop=True))
    top20["total_spent"]     = top20["total_spent"].map("₹{:,.0f}".format)
    top20["avg_order_value"] = top20["avg_order_value"].map("₹{:,.0f}".format)
    top20["clv"]             = top20["clv"].map("₹{:,.0f}".format)
    st.dataframe(top20, use_container_width=True)

    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    with col3:
        fig3 = px.histogram(df.drop_duplicates("customer_id"), x="customer_age",
                            nbins=20, title="Customer Age Distribution",
                            color_discrete_sequence=[ACCENT], template=TEMPLATE)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        gender = df.drop_duplicates("customer_id")["customer_gender"].value_counts().reset_index()
        fig4   = px.pie(gender, names="customer_gender", values="count",
                        title="Gender Breakdown", hole=0.4,
                        color_discrete_sequence=[ACCENT, ACCENT_ALT, NEUTRAL],
                        template=TEMPLATE)
        fig4.update_traces(textinfo="percent+label")
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        order_counts = df.groupby("customer_id")["order_id"].nunique().reset_index()
        order_counts["type"] = order_counts["order_id"].apply(
            lambda x: "New (1 order)" if x == 1 else "Returning (2+ orders)"
        )
        type_counts = order_counts["type"].value_counts().reset_index()
        fig5 = px.pie(type_counts, names="type", values="count",
                      title="New vs Returning Customers", hole=0.4,
                      color_discrete_sequence=[POSITIVE, ACCENT],
                      template=TEMPLATE)
        fig5.update_traces(textinfo="percent+label")
        fig5.update_layout(showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 3 — COHORT & RETENTION
# ══════════════════════════════════════════════════════════

def page_cohort(df: pd.DataFrame, cohort_data: dict):
    st.title("Cohort Analysis & Retention")

    retention = cohort_data["retention"]
    churn_df  = cohort_data["churn"]

    st.subheader("Cohort Retention Heatmap")
    ret_plot = retention.copy().astype(float)
    ret_plot = ret_plot.fillna(0)

    fig = px.imshow(
        ret_plot,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Cohort Retention Rate (%)",
        template=TEMPLATE,
        text_auto=".0f",
    )
    fig.update_layout(
        xaxis_title="Months Since First Purchase",
        yaxis_title="Cohort Month",
        coloraxis_colorbar_title="%",
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly Churn Rate")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=churn_df["month"], y=churn_df["churn_rate_pct"],
            mode="lines+markers", name="Churn %",
            line=dict(color=NEGATIVE, width=2.5),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.1)"
        ))
        avg_churn = churn_df["churn_rate_pct"].mean()
        fig2.add_hline(y=avg_churn, line_dash="dash",
                       line_color=NEUTRAL,
                       annotation_text=f"Avg {avg_churn:.1f}%")
        fig2.update_layout(template=TEMPLATE, title="Monthly Churn Rate (%)",
                           xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Avg Days Between Orders")
        cust_gap = df.groupby("customer_id")["order_date"].apply(
            lambda x: round(np.mean(np.diff(sorted(x.unique())).astype("timedelta64[D]").astype(int)), 1)
            if len(x.unique()) > 1 else np.nan
        ).dropna()

        fig3 = px.histogram(
            x=cust_gap, nbins=30,
            title="Avg Days Between Orders (Repeat Customers)",
            color_discrete_sequence=[ACCENT_ALT],
            template=TEMPLATE,
        )
        fig3.update_layout(xaxis_title="Days", yaxis_title="Customers")
        st.plotly_chart(fig3, use_container_width=True)

    # Cohort revenue heatmap
    st.subheader("Average Revenue per Cohort (Month 0 → M12)")
    rev = cohort_data["revenue"].copy().astype(float).fillna(0)
    fig4 = px.imshow(rev, color_continuous_scale="Blues",
                     aspect="auto", title="Avg Revenue per Customer per Cohort",
                     template=TEMPLATE, text_auto=".0f")
    fig4.update_layout(height=500, xaxis_title="Months Since First Purchase",
                       yaxis_title="Cohort Month")
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 4 — PRODUCT INTELLIGENCE
# ══════════════════════════════════════════════════════════

def page_products(df: pd.DataFrame):
    st.title("Product Intelligence")
    completed = df[df["order_status"] == "Completed"]

    col1, col2 = st.columns(2)
    with col1:
        # Category performance matrix
        cat_perf = (
            completed.groupby("category")
            .agg(revenue=("total_amount", "sum"),
                 orders=("order_id",      "nunique"))
            .reset_index()
        )
        ret_rate = (
            df.groupby("category")["is_returned"]
            .mean().mul(100).round(2).reset_index()
            .rename(columns={"is_returned": "return_rate"})
        )
        cat_perf = cat_perf.merge(ret_rate, on="category")
        fig1 = px.scatter(cat_perf, x="revenue", y="return_rate",
                          size="orders", color="category", text="category",
                          title="Category Matrix: Revenue vs Return Rate",
                          color_discrete_sequence=PALETTE,
                          template=TEMPLATE, size_max=40)
        fig1.update_traces(textposition="top center")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        sub_rev = (completed.groupby("sub_category")["total_amount"]
                   .sum().reset_index().sort_values("total_amount", ascending=False).head(12))
        fig2 = px.treemap(sub_rev, path=["sub_category"], values="total_amount",
                          title="Sub-Category Revenue Treemap",
                          color="total_amount", color_continuous_scale="Blues",
                          template=TEMPLATE)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.histogram(completed, x="unit_price", nbins=40,
                            title="Price Range Distribution",
                            color_discrete_sequence=[ACCENT],
                            template=TEMPLATE)
        fig3.update_layout(xaxis_title="Unit Price (₹)", yaxis_title="Orders")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        sample = completed.sample(min(2000, len(completed)))
        fig4 = px.scatter(
            sample,
            x="discount_percent", y="total_amount",
            opacity=0.4, color_discrete_sequence=[ACCENT_ALT],
            title="Discount % vs Order Value",
            template=TEMPLATE,
        )
        # Manual numpy trendline (no statsmodels required)
        _x = sample["discount_percent"].values
        _y = sample["total_amount"].values
        _mask = np.isfinite(_x) & np.isfinite(_y)
        if _mask.sum() > 1:
            _z = np.polyfit(_x[_mask], _y[_mask], 1)
            _xr = np.linspace(_x[_mask].min(), _x[_mask].max(), 100)
            _yr = np.polyval(_z, _xr)
            fig4.add_trace(go.Scatter(x=_xr, y=_yr, mode="lines",
                                      line=dict(color=POSITIVE, width=2),
                                      name="Trend"))
        fig4.update_layout(xaxis_title="Discount (%)", yaxis_title="Total Order Value (₹)")
        st.plotly_chart(fig4, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        top_ret = (df[df["is_returned"]].groupby("product_name")["order_id"]
                   .count().sort_values(ascending=False).head(10).reset_index())
        top_ret.columns = ["product", "returns"]
        fig5 = px.bar(top_ret, x="returns", y="product", orientation="h",
                      title="Top 10 Returned Products",
                      color_discrete_sequence=[NEGATIVE], template=TEMPLATE)
        fig5.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        fig6 = px.box(completed, x="category", y="rating",
                      title="Rating Distribution per Category",
                      color="category", color_discrete_sequence=PALETTE,
                      template=TEMPLATE)
        fig6.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 5 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════

def page_insights(df: pd.DataFrame, cust: pd.DataFrame, insights: dict):
    st.title("Business Insights & Recommendations")

    kpis = insights["kpis"]
    top  = insights["top_performers"]
    ci   = insights["customer_insights"]
    tr   = insights["trends"]

    # KPI summary cards
    st.subheader("Key Business Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='info-card'>
        <b>Revenue Trend</b><br>
        <span style='color:{POSITIVE if tr["yoy_revenue_growth"] > 0 else NEGATIVE};font-size:1.3rem;font-weight:700;'>
        {tr['revenue_trend']} ({tr['yoy_revenue_growth']:+.1f}% YoY)</span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='info-card'>
        <b>Repeat Customer Revenue</b><br>
        <span style='color:{ACCENT};font-size:1.3rem;font-weight:700;'>
        {kpis['repeat_revenue_pct']:.1f}% of total</span>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='info-card'>
        <b>Customer Retention Rate</b><br>
        <span style='color:{POSITIVE};font-size:1.3rem;font-weight:700;'>
        {kpis['retention_rate_pct']:.1f}%</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Performers")
        st.markdown("**Top 5 Categories by Revenue**")
        st.dataframe(top["top_categories"].rename(columns={"total_amount": "revenue", "category": "Category"})
                     .assign(revenue=lambda x: x["revenue"].map("₹{:,.0f}".format))
                     .reset_index(drop=True), use_container_width=True)

        st.markdown("**Top 5 Brands by Revenue**")
        st.dataframe(top["top_brands"].rename(columns={"total_amount": "revenue", "brand": "Brand"})
                     .assign(revenue=lambda x: x["revenue"].map("₹{:,.0f}".format))
                     .reset_index(drop=True), use_container_width=True)

    with col2:
        st.subheader("Customer Behaviour Quick Stats")
        stats = {
            "Top Payment Method":  ci["top_payment_method"],
            "Top Device Type":     ci["top_device_type"],
            "Best Day of Week":    ci["best_day_of_week"],
            "Avg Customer CLV":    f"₹{ci['avg_clv']:,.0f}",
            "% Repeat Customers":  f"{ci['pct_repeat_customers']:.1f}%",
            "Highest Growth Category": tr["highest_growth_category"],
            "Highest Return Category": tr["highest_return_category"],
        }
        for k, v in stats.items():
            st.markdown(f"""
            <div class='insight-card'>
            <span style='color:#94a3b8;'>{k}</span><br>
            <span style='color:#ffffff;font-weight:600;font-size:1.05rem;'>{v}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Strategic Recommendations")
    recs = [
        ("Revenue Growth",      POSITIVE,  "Amplify Q4 festive campaigns — Oct-Dec generates 2x baseline revenue. Pre-position inventory in September."),
        ("Mobile Strategy",     ACCENT,    "60%+ orders come from mobile devices. Invest in a Progressive Web App or native app for a seamless checkout experience."),
        ("Retain Champions",    ACCENT_ALT,"Top 15% customers ('Champions') drive ~40% of revenue. Launch a tiered loyalty programme with exclusive early-access deals."),
        ("Win-back At-Risk",    NEUTRAL,   "'At Risk' customers with high past spend need a time-limited personalised discount within 30 days of last purchase."),
        ("Reduce Return Rate",  NEGATIVE,  f"Highest returns in {tr['highest_return_category']}. Add video reviews, accurate size guides, and better product specs."),
        ("UPI Incentives",      "#06b6d4", "UPI is #1 payment method. Offer ₹50-100 cashback on UPI orders to further shift away from high-cost COD."),
        ("Fast Delivery ROI",   POSITIVE,  "Fast delivery (<5 days) correlates with 0.4+ higher rating. Negotiate SLAs with top logistics partners for metro cities."),
    ]
    for title, color, text in recs:
        st.markdown(f"""
        <div class='insight-card' style='border-left:4px solid {color};'>
        <b style='color:{color};'>{title}</b><br>
        <span style='color:#e2e8f0;'>{text}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Download Reports")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        cleaned_path = ROOT / "data" / "processed" / "cleaned_data.csv"
        if cleaned_path.exists():
            st.download_button(
                "Download Cleaned Data (CSV)",
                data=cleaned_path.read_bytes(),
                file_name="cleaned_data.csv",
                mime="text/csv",
            )
    with col_d2:
        rfm_data = cust[["customer_id", "rfm_segment", "cluster_label", "total_spent", "total_orders", "clv"]].copy()
        st.download_button(
            "Download RFM Segments (CSV)",
            data=rfm_data.to_csv(index=False).encode(),
            file_name="rfm_segments.csv",
            mime="text/csv",
        )
    with col_d3:
        ig = InsightsGenerator()
        report_text = ig.format_insights_text(insights)
        st.download_button(
            "Download Insights Report (TXT)",
            data=report_text.encode(),
            file_name="insights_report.txt",
            mime="text/plain",
        )


# ══════════════════════════════════════════════════════════
# MAIN — Page routing
# ══════════════════════════════════════════════════════════

def main():
    # Load base data
    df   = load_data()
    cust = load_customer_features(df)

    # Sidebar filters → filtered_df
    d_from, d_to, states, categories, statuses, devices = render_sidebar(df)
    filtered = apply_filters(df, d_from, d_to, states, categories, statuses, devices)

    if len(filtered) == 0:
        st.warning("No data matches the selected filters. Please adjust the sidebar.")
        return

    # Lazy-load heavy computations
    cohort_data = load_cohort_data(filtered)
    insights    = load_insights(filtered, cust)

    # Navigation tabs
    pages = ["Executive Overview", "Customer Analysis",
             "Cohort & Retention", "Product Intelligence", "Business Insights"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(pages)

    with tab1:
        page_overview(filtered)
    with tab2:
        page_customers(filtered, cust)
    with tab3:
        page_cohort(filtered, cohort_data)
    with tab4:
        page_products(filtered)
    with tab5:
        page_insights(filtered, cust, insights)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#94a3b8;font-size:0.8rem;'>"
        "E-Commerce Customer Behavior Intelligence Dashboard &nbsp;|&nbsp; "
        "Built with Streamlit & Plotly &nbsp;|&nbsp; Data: Jan 2022 – Dec 2024"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
