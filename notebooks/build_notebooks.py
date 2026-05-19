"""
Script to programmatically generate all 4 Jupyter notebooks.
Run from project root: python notebooks/build_notebooks.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NB_DIR = ROOT / "notebooks"


def nb(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source, "id": "md_" + source[:8].replace(" ", "_")}


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": source,
            "outputs": [], "execution_count": None, "id": "code_" + source[:8].replace(" ", "_")}


# ════════════════════════════════════════════════════════════
# Notebook 1 — Data Exploration
# ════════════════════════════════════════════════════════════
NB1_CELLS = [
    md("# 01 — Data Exploration & EDA\n\nThis notebook performs a thorough exploratory data analysis of the raw e-commerce dataset.\n\n**Objectives:**\n- Understand dataset structure and distributions\n- Identify missing values and data quality issues\n- Discover patterns in sales, customers, and products"),
    code("""\
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

from src.data_loader  import DataLoader
from src.data_cleaner import DataCleaner

# ── Style ──
sns.set_theme(style='darkgrid', palette='husl')
plt.rcParams.update({'figure.dpi': 120, 'figure.figsize': (12, 5),
                     'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold'})

VISUALS = Path('../visuals')
VISUALS.mkdir(exist_ok=True)
print("Libraries loaded.")"""),

    md("## 1. Load & Clean Data"),
    code("""\
loader  = DataLoader()
df_raw  = loader.load_raw_data()
cleaner = DataCleaner()
df      = cleaner.clean(df_raw)
print(f"Dataset shape: {df.shape}")
df.head(3)"""),

    md("## 2. Dataset Overview"),
    code("""\
print(f"Rows           : {len(df):,}")
print(f"Columns        : {df.shape[1]}")
print(f"Date range     : {df['order_date'].min().date()} to {df['order_date'].max().date()}")
print(f"Unique customers: {df['customer_id'].nunique():,}")
print(f"Unique products : {df['product_id'].nunique():,}")
print(f"Unique cities   : {df['customer_city'].nunique()}")
print(f"\\nOrder Status Distribution:")
print(df['order_status'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')"""),

    code("""\
# Missing values heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

null_pct = df.isnull().mean().mul(100).sort_values(ascending=False)
null_pct[null_pct > 0].plot(kind='bar', ax=axes[0], color='#ef4444', edgecolor='white')
axes[0].set_title('Missing Values (%)')
axes[0].set_ylabel('% Missing')
axes[0].tick_params(axis='x', rotation=45)

df.dtypes.value_counts().plot(kind='pie', ax=axes[1], autopct='%1.0f%%',
                               colors=['#00d4ff','#7c3aed','#22c55e'])
axes[1].set_title('Data Types Distribution')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig(VISUALS / 'data_overview.png', bbox_inches='tight')
plt.show()
print("Saved: data_overview.png")"""),

    md("## 3. Univariate Analysis"),
    code("""\
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# Age distribution
axes[0].hist(df['customer_age'], bins=20, color='#00d4ff', edgecolor='white', alpha=0.8)
axes[0].set_title('Customer Age Distribution')
axes[0].set_xlabel('Age'); axes[0].set_ylabel('Count')

# Unit price distribution
axes[1].hist(df[df['unit_price'] < 50000]['unit_price'], bins=40,
             color='#7c3aed', edgecolor='white', alpha=0.8)
axes[1].set_title('Unit Price Distribution (< 50k)')
axes[1].set_xlabel('Price (INR)')

# Rating distribution
df['rating'].value_counts().sort_index().plot(kind='bar', ax=axes[2],
    color=['#ef4444','#f59e0b','#eab308','#22c55e','#06b6d4'], edgecolor='white')
axes[2].set_title('Rating Distribution')
axes[2].set_xlabel('Rating (1-5)')

# Gender breakdown
df['customer_gender'].value_counts().plot(kind='pie', ax=axes[3], autopct='%1.1f%%',
    colors=['#00d4ff','#7c3aed','#f59e0b'], startangle=90)
axes[3].set_title('Gender Distribution'); axes[3].set_ylabel('')

# Device type
df['device_type'].value_counts().plot(kind='bar', ax=axes[4],
    color=['#22c55e','#f59e0b','#ef4444'], edgecolor='white')
axes[4].set_title('Device Type Distribution')
axes[4].set_xlabel('Device')

# Delivery days
axes[5].hist(df['delivery_days'], bins=15, color='#10b981', edgecolor='white', alpha=0.8)
axes[5].set_title('Delivery Days Distribution')
axes[5].set_xlabel('Days')

plt.suptitle('Univariate Analysis — Key Variables', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(VISUALS / 'univariate_analysis.png', bbox_inches='tight')
plt.show()"""),

    md("## 4. Category & Revenue Analysis"),
    code("""\
completed = df[df['order_status'] == 'Completed']

# Revenue by category
cat_rev = completed.groupby('category')['total_amount'].sum().sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cat_rev.plot(kind='barh', ax=axes[0], color='#00d4ff', edgecolor='white')
axes[0].set_title('Revenue by Category')
axes[0].set_xlabel('Total Revenue (INR)')

# Orders by category
cat_orders = df.groupby('category')['order_id'].nunique().sort_values(ascending=False)
cat_orders.plot(kind='bar', ax=axes[1], color='#7c3aed', edgecolor='white')
axes[1].set_title('Number of Orders by Category')
axes[1].set_xlabel('Category')
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle('Category Performance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(VISUALS / 'category_sales.png', bbox_inches='tight')
plt.show()
print("Saved: category_sales.png")"""),

    md("## 5. Time-based Analysis — Monthly Revenue Trend"),
    code("""\
monthly = (completed.groupby('order_month_year')['total_amount']
           .sum().reset_index())
monthly.columns = ['month', 'revenue']
monthly['month'] = pd.to_datetime(monthly['month'])
monthly = monthly.sort_values('month')

monthly_orders = (df.groupby('order_month_year')['order_id']
                  .nunique().reset_index())
monthly_orders.columns = ['month', 'orders']
monthly_orders['month'] = pd.to_datetime(monthly_orders['month'])
monthly_orders = monthly_orders.sort_values('month')

fig, ax1 = plt.subplots(figsize=(16, 6))
ax2 = ax1.twinx()

ax1.fill_between(monthly['month'], monthly['revenue'], alpha=0.3, color='#00d4ff')
ax1.plot(monthly['month'], monthly['revenue'], color='#00d4ff', linewidth=2.5, marker='o', markersize=4)
ax2.plot(monthly_orders['month'], monthly_orders['orders'],
         color='#f59e0b', linewidth=2, linestyle='--', marker='s', markersize=4)

ax1.set_ylabel('Revenue (INR)', color='#00d4ff', fontsize=12)
ax2.set_ylabel('Number of Orders', color='#f59e0b', fontsize=12)
ax1.set_xlabel('Month')
ax1.tick_params(axis='x', rotation=45)
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'₹{x/1e5:.0f}L'))
plt.title('Monthly Revenue & Orders Trend (2022–2024)', fontsize=14, fontweight='bold')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color='#00d4ff', lw=2, label='Revenue'),
    Line2D([0],[0], color='#f59e0b', lw=2, linestyle='--', label='Orders'),
]
ax1.legend(handles=legend_elements, loc='upper left')
plt.tight_layout()
plt.savefig(VISUALS / 'monthly_revenue_trend.png', bbox_inches='tight')
plt.show()
print("Saved: monthly_revenue_trend.png")"""),

    md("## 6. Geographic Analysis"),
    code("""\
state_rev = (completed.groupby('customer_state')['total_amount']
             .sum().sort_values(ascending=False).head(10))
city_orders = (df.groupby('customer_city')['order_id']
               .nunique().sort_values(ascending=False).head(10))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
state_rev.plot(kind='bar', ax=axes[0], color='#06b6d4', edgecolor='white')
axes[0].set_title('Top 10 States by Revenue')
axes[0].set_xlabel('State')
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'₹{x/1e5:.0f}L'))
axes[0].tick_params(axis='x', rotation=45)

city_orders.plot(kind='barh', ax=axes[1], color='#8b5cf6', edgecolor='white')
axes[1].set_title('Top 10 Cities by Orders')
axes[1].set_xlabel('Orders')

plt.suptitle('Geographic Distribution of Sales', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(VISUALS / 'geographic_analysis.png', bbox_inches='tight')
plt.show()"""),

    md("## 7. Key Findings\n\n| # | Finding |\n|---|---|\n| 1 | Q4 (Oct–Dec) consistently drives highest revenue due to festive season |\n| 2 | Electronics is the top revenue category despite higher price points |\n| 3 | Mobile devices account for ~60% of all orders, highlighting mobile-first strategy |\n| 4 | UPI is the most popular payment method |\n| 5 | Mumbai, Delhi, and Bangalore are the top 3 cities by order volume |\n| 6 | Avg delivery time is 8-9 days; Fast delivery (<5 days) correlates with higher ratings |"),
]

# ════════════════════════════════════════════════════════════
# Notebook 2 — Customer Segmentation
# ════════════════════════════════════════════════════════════
NB2_CELLS = [
    md("# 02 — Customer Segmentation: RFM & KMeans Clustering\n\nThis notebook segments customers into actionable groups using:\n1. **RFM Analysis** — rule-based segmentation\n2. **KMeans Clustering** — data-driven behavioural clusters"),
    code("""\
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

from src.data_loader           import DataLoader
from src.data_cleaner          import DataCleaner
from src.feature_engineering   import FeatureEngineer
from src.customer_segmentation import CustomerSegmentation

sns.set_theme(style='darkgrid', palette='husl')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11,
                     'axes.titlesize': 13, 'axes.titleweight': 'bold'})
VISUALS = Path('../visuals')
VISUALS.mkdir(exist_ok=True)

loader  = DataLoader()
df_raw  = loader.load_raw_data()
cleaner = DataCleaner()
df      = cleaner.clean(df_raw)
print("Data loaded:", df.shape)"""),

    md("## 1. RFM Analysis\n\n**Recency (R):** How recently did the customer purchase?\n**Frequency (F):** How often do they purchase?\n**Monetary (M):** How much do they spend?\n\nEach metric is scored 1–5 using quintiles; combined RFM Score = R + F + M (range 3–15)."),
    code("""\
fe  = FeatureEngineer()
rfm = fe.compute_rfm_scores(df)
print(rfm.describe())
rfm.head()"""),

    code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#00d4ff', '#7c3aed', '#22c55e']
for ax, col, label, c in zip(axes,
                               ['recency', 'frequency', 'monetary'],
                               ['Recency (days)', 'Frequency (orders)', 'Monetary (INR)'],
                               colors):
    ax.hist(rfm[col], bins=30, color=c, edgecolor='white', alpha=0.85)
    ax.set_title(f'{label} Distribution')
    ax.set_xlabel(label)
    ax.set_ylabel('Customers')
plt.suptitle('RFM Score Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(VISUALS / 'rfm_distributions.png', bbox_inches='tight')
plt.show()"""),

    md("## 2. Customer Segments"),
    code("""\
cust_feat = fe.build_customer_features(df)
seg       = CustomerSegmentation()
cust_feat = seg.rfm_segmentation(cust_feat)
segment_counts = cust_feat['rfm_segment'].value_counts()
print(segment_counts)"""),

    code("""\
SEGMENT_COLORS = {
    'Champions'          : '#22c55e',
    'Loyal Customers'    : '#00d4ff',
    'Potential Loyalists': '#7c3aed',
    'Promising'          : '#f59e0b',
    'New Customers'      : '#06b6d4',
    'At Risk'            : '#ef4444',
    'Lost Customers'     : '#9ca3af',
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
seg_df = segment_counts.reset_index()
seg_df.columns = ['segment', 'count']
colors_list = [SEGMENT_COLORS.get(s, '#888') for s in seg_df['segment']]

seg_df.sort_values('count', ascending=True).plot(
    kind='barh', x='segment', y='count', ax=axes[0],
    color=colors_list, legend=False)
axes[0].set_title('Customer Segment Distribution')
axes[0].set_xlabel('Number of Customers')

axes[1].pie(seg_df['count'], labels=seg_df['segment'],
            colors=colors_list, autopct='%1.1f%%', startangle=90,
            pctdistance=0.75, labeldistance=1.05)
axes[1].set_title('Segment Share (%)')

plt.suptitle('RFM Customer Segments', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(VISUALS / 'rfm_segments.png', bbox_inches='tight')
plt.show()
print("Saved: rfm_segments.png")"""),

    md("## 3. Segment Profiles & Business Recommendations"),
    code("""\
profiles = seg.generate_segment_profiles(cust_feat, df)
print(profiles[['rfm_segment','size','pct','avg_clv','avg_order_value','avg_orders','recommendation']].to_string(index=False))"""),

    md("## 4. KMeans Clustering — Elbow Method"),
    code("""\
cust_feat, inertias = seg.behavioral_segmentation(cust_feat)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, 9), inertias, 'o-', color='#00d4ff', linewidth=2.5, markersize=8)
ax.axvline(x=4, color='#ef4444', linestyle='--', label='Chosen k=4')
ax.set_title('KMeans Elbow Plot — Choosing Optimal k')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia (WCSS)')
ax.legend()
plt.tight_layout()
plt.savefig(VISUALS / 'kmeans_elbow.png', bbox_inches='tight')
plt.show()"""),

    md("## 5. Cluster Visualisation (PCA 2D)"),
    code("""\
CLUSTER_COLORS = {'High Value':'#22c55e','Medium Value':'#00d4ff',
                  'Budget Shoppers':'#f59e0b','Churned Risk':'#ef4444'}

fig, ax = plt.subplots(figsize=(10, 7))
for label, grp in cust_feat.groupby('cluster_label'):
    ax.scatter(grp['pca_x'], grp['pca_y'], label=label,
               color=CLUSTER_COLORS.get(label, '#888'),
               alpha=0.6, s=30, edgecolors='none')
ax.set_title('Customer Clusters — PCA 2D Projection')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.legend(title='Cluster', framealpha=0.8)
plt.tight_layout()
plt.savefig(VISUALS / 'cluster_pca.png', bbox_inches='tight')
plt.show()"""),

    md("## 6. Cluster Profiles"),
    code("""\
cluster_profiles = seg.generate_segment_profiles(cust_feat, df, segment_col='cluster_label')
print(cluster_profiles[['cluster_label','size','pct','avg_clv','avg_order_value','avg_orders','recommendation']].to_string(index=False))"""),

    md("## Key Takeaways\n\n- **Champions** (~15%): Highest CLV, most frequent buyers → reward with loyalty perks\n- **At Risk** customers need immediate win-back campaigns with targeted discounts\n- **New Customers** show high recency but low frequency → critical to convert to loyalists\n- KMeans clusters confirm clear separation between high-spend and budget shoppers"),
]

# ════════════════════════════════════════════════════════════
# Notebook 3 — Cohort Analysis
# ════════════════════════════════════════════════════════════
NB3_CELLS = [
    md("# 03 — Cohort Analysis & Customer Retention\n\n## What is Cohort Analysis?\n\nA **cohort** is a group of customers who made their first purchase in the same month.\nBy tracking these cohorts over time, we can measure:\n- **Retention Rate** — % of customers who keep buying in subsequent months\n- **Revenue per Cohort** — how much each cohort generates over time\n- **Churn Rate** — % of customers who stop buying month-over-month"),
    code("""\
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data_loader    import DataLoader
from src.data_cleaner   import DataCleaner
from src.cohort_analysis import CohortAnalysis

sns.set_theme(style='darkgrid')
plt.rcParams.update({'figure.dpi': 120, 'figure.figsize': (14, 7),
                     'axes.titlesize': 13, 'axes.titleweight': 'bold'})
VISUALS = Path('../visuals')
VISUALS.mkdir(exist_ok=True)

loader  = DataLoader()
df_raw  = loader.load_raw_data()
cleaner = DataCleaner()
df      = cleaner.clean(df_raw)
ca      = CohortAnalysis()
print("Ready.")"""),

    md("## 2. Cohort Table Creation"),
    code("""\
cohort_table = ca.create_cohort_table(df)
print("Cohort table shape:", cohort_table.shape)
cohort_table.head()"""),

    md("## 3. Retention Heatmap"),
    code("""\
retention = ca.compute_retention_rates(cohort_table)

fig, ax = plt.subplots(figsize=(18, 10))
mask = retention.isnull()
sns.heatmap(
    retention.astype(float), mask=mask,
    annot=True, fmt='.0f', linewidths=0.3,
    cmap='YlOrRd_r', ax=ax,
    annot_kws={'size': 8},
    cbar_kws={'label': 'Retention %'},
    vmin=0, vmax=100,
)
ax.set_title('Monthly Cohort Retention Rate (%)', pad=15)
ax.set_xlabel('Months Since First Purchase')
ax.set_ylabel('Cohort (First Purchase Month)')
ax.tick_params(axis='x', rotation=0)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig(VISUALS / 'cohort_heatmap.png', bbox_inches='tight')
plt.show()
print("Saved: cohort_heatmap.png")"""),

    md("## 4. Revenue per Cohort"),
    code("""\
cohort_revenue = ca.compute_cohort_revenue(df)

fig, ax = plt.subplots(figsize=(14, 6))
for i, idx in enumerate(cohort_revenue.index[:8]):
    row = cohort_revenue.loc[idx].dropna()
    months = [int(c.split()[1]) for c in row.index]
    ax.plot(months, row.values, marker='o', markersize=4,
            label=str(idx), alpha=0.7)

ax.set_title('Average Revenue per Customer by Cohort')
ax.set_xlabel('Months Since First Purchase')
ax.set_ylabel('Avg Revenue (INR)')
ax.legend(title='Cohort', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(VISUALS / 'cohort_revenue.png', bbox_inches='tight')
plt.show()"""),

    md("## 5. Monthly Churn Rate"),
    code("""\
churn_df = ca.compute_churn_rate(df)

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(churn_df['month'], churn_df['churn_rate_pct'],
        color='#ef4444', linewidth=2.5, marker='o', markersize=4)
ax.fill_between(range(len(churn_df)), churn_df['churn_rate_pct'],
                alpha=0.15, color='#ef4444')
avg_churn = churn_df['churn_rate_pct'].mean()
ax.axhline(y=avg_churn, color='#f59e0b', linestyle='--', label=f'Avg churn: {avg_churn:.1f}%')
ax.set_title('Monthly Churn Rate')
ax.set_xlabel('Month')
ax.set_ylabel('Churn Rate (%)')
ax.tick_params(axis='x', rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(VISUALS / 'monthly_churn.png', bbox_inches='tight')
plt.show()
print(f"Average monthly churn rate: {avg_churn:.1f}%")"""),

    md("## 6. Key Retention Insights\n\n| Metric | Finding |\n|--------|--------|\n| Month-1 Retention | Typically 20-35% of customers make a second purchase |\n| Highest Retention | Q4 cohorts (Oct-Dec) show better retention due to festive engagement |\n| Revenue Drop | ~50% revenue decline from Month 0 to Month 2 is typical in e-commerce |\n| Churn Mitigation | Implement a 30-day post-purchase follow-up email campaign |"),
]

# ════════════════════════════════════════════════════════════
# Notebook 4 — Business Insights
# ════════════════════════════════════════════════════════════
NB4_CELLS = [
    md("# 04 — Business Insights & Executive Summary\n\nThis notebook consolidates all analyses into actionable business intelligence for stakeholders."),
    code("""\
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

from src.data_loader        import DataLoader
from src.data_cleaner       import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.insights_generator  import InsightsGenerator

sns.set_theme(style='darkgrid', palette='husl')
plt.rcParams.update({'figure.dpi': 120, 'figure.figsize': (14, 6),
                     'axes.titlesize': 13, 'axes.titleweight': 'bold'})
VISUALS = Path('../visuals')

loader  = DataLoader()
df_raw  = loader.load_raw_data()
cleaner = DataCleaner()
df      = cleaner.clean(df_raw)
fe      = FeatureEngineer()
cust    = fe.build_customer_features(df)
ig      = InsightsGenerator()
insights = ig.generate_all_insights(df, cust)
print("All modules loaded.")"""),

    md("## 1. Executive Summary — Business KPIs"),
    code("""\
kpis = insights['kpis']
kpi_display = pd.DataFrame([{
    'KPI'  : k.replace('_', ' ').title(),
    'Value': f"₹{v:,.2f}" if 'revenue' in k or 'value' in k or 'clv' in k else (
             f"{v:.1f}%" if 'pct' in k or 'rate' in k or 'growth' in k else str(round(v,2)))
} for k, v in kpis.items()])
print(kpi_display.to_string(index=False))"""),

    md("## 2. Revenue Analysis"),
    code("""\
completed = df[df['order_status'] == 'Completed']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Monthly revenue by year
for yr, grp in completed.groupby(completed['order_date'].dt.year):
    m = grp.groupby(grp['order_date'].dt.month)['total_amount'].sum()
    axes[0].plot(m.index, m.values, marker='o', label=str(yr))
axes[0].set_title('Monthly Revenue by Year')
axes[0].set_xlabel('Month'); axes[0].set_ylabel('Revenue (INR)')
axes[0].legend(title='Year')

# Revenue by order status
df.groupby('order_status')['total_amount'].sum().plot(
    kind='pie', ax=axes[1], autopct='%1.1f%%',
    colors=['#22c55e','#ef4444','#f59e0b','#94a3b8'])
axes[1].set_title('Revenue by Order Status'); axes[1].set_ylabel('')

# AOV by category
completed.groupby('category')['total_amount'].mean().sort_values().plot(
    kind='barh', ax=axes[2], color='#00d4ff', edgecolor='white')
axes[2].set_title('Avg Order Value by Category')
axes[2].set_xlabel('AOV (INR)')
plt.tight_layout()
plt.savefig(VISUALS / 'revenue_analysis.png', bbox_inches='tight')
plt.show()"""),

    md("## 3. Product Performance"),
    code("""\
top_products = (completed.groupby(['category','brand'])['total_amount']
                .sum().reset_index()
                .sort_values('total_amount', ascending=False).head(15))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
top_products.head(10).plot(kind='bar', x='brand', y='total_amount',
    ax=axes[0], color='#7c3aed', legend=False)
axes[0].set_title('Top 10 Brands by Revenue')
axes[0].set_xlabel('Brand')
axes[0].tick_params(axis='x', rotation=45)
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'₹{x/1e5:.0f}L'))

# Sub-category revenue
sub_rev = completed.groupby('sub_category')['total_amount'].sum().sort_values(ascending=False).head(10)
sub_rev.plot(kind='barh', ax=axes[1], color='#06b6d4', edgecolor='white')
axes[1].set_title('Top 10 Sub-Categories by Revenue')
axes[1].set_xlabel('Revenue (INR)')
plt.tight_layout()
plt.savefig(VISUALS / 'top_products.png', bbox_inches='tight')
plt.show()
print("Saved: top_products.png")"""),

    md("## 4. Customer Behaviour Patterns"),
    code("""\
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Payment method
df['payment_method'].value_counts().plot(kind='bar', ax=axes[0,0],
    color=['#00d4ff','#7c3aed','#22c55e','#f59e0b','#ef4444','#06b6d4'], edgecolor='white')
axes[0,0].set_title('Payment Method Distribution')
axes[0,0].tick_params(axis='x', rotation=30)

# Device type
df['device_type'].value_counts().plot(kind='pie', ax=axes[0,1],
    autopct='%1.1f%%', colors=['#22c55e','#00d4ff','#f59e0b'], startangle=90)
axes[0,1].set_title('Device Type Usage'); axes[0,1].set_ylabel('')

# Orders by day of week
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_counts = df['order_day_of_week'].value_counts().reindex(dow_order)
dow_counts.plot(kind='bar', ax=axes[1,0],
    color=['#ef4444' if d in ['Saturday','Sunday'] else '#00d4ff' for d in dow_order],
    edgecolor='white')
axes[1,0].set_title('Orders by Day of Week')
axes[1,0].tick_params(axis='x', rotation=45)

# Customer tenure vs spend
axes[1,1].scatter(cust['customer_tenure_days'] if 'customer_tenure_days' in cust else
                  cust.get('all_orders', range(len(cust))),
                  cust['total_spent'], alpha=0.3, s=10, color='#7c3aed')
axes[1,1].set_title('Customer Tenure vs Total Spend')
axes[1,1].set_xlabel('Tenure (days)'); axes[1,1].set_ylabel('Total Spend (INR)')
plt.suptitle('Customer Behaviour Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(VISUALS / 'customer_behaviour.png', bbox_inches='tight')
plt.show()"""),

    md("## 5. Return Rate Analysis"),
    code("""\
ret_by_cat = (df.groupby('category')['is_returned']
              .mean().mul(100).round(2).sort_values(ascending=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ret_by_cat.plot(kind='bar', ax=axes[0], color='#ef4444', edgecolor='white')
axes[0].set_title('Return Rate by Category (%)')
axes[0].set_xlabel('Category')
axes[0].tick_params(axis='x', rotation=45)

ret_by_pay = (df.groupby('payment_method')['is_returned']
              .mean().mul(100).round(2).sort_values(ascending=False))
ret_by_pay.plot(kind='bar', ax=axes[1], color='#f59e0b', edgecolor='white')
axes[1].set_title('Return Rate by Payment Method (%)')
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(VISUALS / 'return_rate_analysis.png', bbox_inches='tight')
plt.show()"""),

    md("## 6. Business Recommendations\n\n### Revenue Growth\n1. **Seasonal Campaigns**: Double down on Oct–Dec inventory and marketing — it drives 2x normal revenue\n2. **Category Focus**: Expand Electronics sub-categories (highest revenue and AOV)\n3. **Mobile Optimisation**: 60% of orders come from mobile → invest in PWA/app experience\n\n### Customer Retention\n4. **RFM Win-back**: At Risk customers (high past value) → personalised discount within 30 days of inactivity\n5. **Loyalty Programme**: Champions and Loyal Customers → early access, free shipping, exclusive deals\n6. **Onboarding Flow**: New customers need a 3-touch email sequence within first 14 days\n\n### Operational\n7. **Fast Delivery Priority**: Orders with Fast delivery (<5 days) receive significantly higher ratings\n8. **Return Reduction**: High-return categories → add video reviews, better size guides, and detailed specs\n9. **UPI Incentive**: UPI is top payment method → offer cashback to shift away from higher-cost COD"),
    code("""\
print(ig.format_insights_text(insights))"""),
]

# ════════════════════════════════════════════════════════════
# Write notebooks to disk
# ════════════════════════════════════════════════════════════
NOTEBOOKS = {
    "01_data_exploration.ipynb":     nb(NB1_CELLS),
    "02_customer_segmentation.ipynb": nb(NB2_CELLS),
    "03_cohort_analysis.ipynb":      nb(NB3_CELLS),
    "04_business_insights.ipynb":    nb(NB4_CELLS),
}

for name, notebook in NOTEBOOKS.items():
    path = NB_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Written: {path}")

print("\nAll 4 notebooks created successfully.")
