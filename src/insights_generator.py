"""
InsightsGenerator — auto-generate business KPIs, rankings, and trend insights.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


class InsightsGenerator:
    """
    Generates business KPIs, top-performer rankings, and trend insights
    from the cleaned orders DataFrame and optional customer feature table.
    """

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.cfg = _load_config(config_path)
        self._curr = self.cfg["dashboard"]["currency_symbol"]

    # ── Internal helpers ──────────────────────────────────────

    @staticmethod
    def _completed(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["order_status"] == "Completed"].copy()

    @staticmethod
    def _pct_change(new: float, old: float) -> float:
        if old == 0:
            return 0.0
        return round((new - old) / old * 100, 2)

    # ── Business KPIs ────────────────────────────────────────

    def compute_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return dict of top-level KPIs."""
        logger.info("Computing business KPIs...")
        completed = self._completed(df)

        total_revenue   = completed["total_amount"].sum()
        total_orders    = df["order_id"].nunique()
        total_customers = df["customer_id"].nunique()
        avg_order_value = completed["total_amount"].mean()
        return_rate     = df["is_returned"].mean() * 100
        avg_rating      = completed["rating"].mean()
        nps             = round((avg_rating - 1) / 4 * 100, 1)  # normalise 1-5 to 0-100

        # Repeat-customer revenue share
        repeat_custs    = df.groupby("customer_id")["order_id"].nunique()
        repeat_ids      = repeat_custs[repeat_custs > 1].index
        repeat_rev_pct  = (
            completed[completed["customer_id"].isin(repeat_ids)]["total_amount"].sum()
            / total_revenue * 100
        )

        # Month-over-month revenue growth (last two full months)
        df2 = completed.copy()
        df2["ym"] = df2["order_date"].dt.to_period("M")
        monthly   = df2.groupby("ym")["total_amount"].sum().sort_index()
        mom_growth = 0.0
        if len(monthly) >= 2:
            mom_growth = self._pct_change(float(monthly.iloc[-1]), float(monthly.iloc[-2]))

        # Retention rate (customers who bought in both of the last 2 months)
        if len(monthly) >= 2:
            last_m  = monthly.index[-1]
            prev_m  = monthly.index[-2]
            last_custs = set(df2[df2["ym"] == last_m]["customer_id"])
            prev_custs = set(df2[df2["ym"] == prev_m]["customer_id"])
            retention_rate = (
                len(last_custs & prev_custs) / len(prev_custs) * 100
                if prev_custs else 0.0
            )
        else:
            retention_rate = 0.0

        kpis = {
            "total_revenue":       round(total_revenue, 2),
            "total_orders":        int(total_orders),
            "total_customers":     int(total_customers),
            "avg_order_value":     round(avg_order_value, 2),
            "return_rate_pct":     round(return_rate, 2),
            "avg_rating":          round(avg_rating, 2),
            "net_promoter_score":  nps,
            "repeat_revenue_pct":  round(repeat_rev_pct, 2),
            "mom_revenue_growth":  mom_growth,
            "retention_rate_pct":  round(retention_rate, 2),
        }
        logger.info("KPIs computed: revenue=%s%.2f, orders=%d",
                    self._curr, total_revenue, total_orders)
        return kpis

    # ── Top Performers ────────────────────────────────────────

    def compute_top_performers(self, df: pd.DataFrame, n: int = 5) -> Dict[str, pd.DataFrame]:
        """Return top-N rankings for categories, products, cities, brands, months."""
        completed = self._completed(df)

        top_categories = (
            completed.groupby("category")["total_amount"]
            .sum().sort_values(ascending=False).head(n)
            .reset_index().rename(columns={"total_amount": "revenue"})
        )
        top_products = (
            completed.groupby("product_name")["total_amount"]
            .sum().sort_values(ascending=False).head(n)
            .reset_index().rename(columns={"total_amount": "revenue"})
        )
        top_cities = (
            df.groupby("customer_city")["order_id"]
            .nunique().sort_values(ascending=False).head(n)
            .reset_index().rename(columns={"order_id": "orders"})
        )
        top_brands = (
            completed.groupby("brand")["total_amount"]
            .sum().sort_values(ascending=False).head(n)
            .reset_index().rename(columns={"total_amount": "revenue"})
        )
        completed["ym_str"] = completed["order_date"].dt.strftime("%b %Y")
        best_month = (
            completed.groupby("ym_str")["total_amount"]
            .sum().sort_values(ascending=False).head(1)
            .reset_index().rename(columns={"total_amount": "revenue", "ym_str": "month"})
        )
        return {
            "top_categories": top_categories,
            "top_products":   top_products,
            "top_cities":     top_cities,
            "top_brands":     top_brands,
            "best_month":     best_month,
        }

    # ── Customer Insights ────────────────────────────────────

    def compute_customer_insights(
        self, df: pd.DataFrame, cust_df: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Return payment method, device, day-of-week, and CLV insights."""
        completed = self._completed(df)

        top_payment = df["payment_method"].value_counts().idxmax()
        top_device  = df["device_type"].value_counts().idxmax()
        best_day    = df["order_day_of_week"].value_counts().idxmax() if "order_day_of_week" in df else "N/A"

        avg_clv = cust_df["clv"].mean().round(2) if cust_df is not None and "clv" in cust_df else 0.0

        # Repeat customer % (>1 order)
        order_counts   = df.groupby("customer_id")["order_id"].nunique()
        pct_repeat     = (order_counts > 1).mean() * 100

        insights = {
            "top_payment_method":    top_payment,
            "top_device_type":       top_device,
            "best_day_of_week":      best_day,
            "avg_clv":               float(avg_clv),
            "pct_repeat_customers":  round(float(pct_repeat), 2),
        }
        return insights

    # ── Trend Insights ────────────────────────────────────────

    def compute_trend_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Determine revenue trend and YoY category growth."""
        completed = self._completed(df)
        completed["year"] = completed["order_date"].dt.year

        years = sorted(completed["year"].unique())
        yoy_growth = 0.0
        if len(years) >= 2:
            rev_by_year = completed.groupby("year")["total_amount"].sum()
            yoy_growth  = self._pct_change(
                float(rev_by_year.iloc[-1]), float(rev_by_year.iloc[-2])
            )

        trend_label = "Growing" if yoy_growth > 5 else ("Declining" if yoy_growth < -5 else "Stable")

        # Category YoY growth (last two years)
        cat_growth = pd.DataFrame()
        if len(years) >= 2:
            last_y = years[-1]; prev_y = years[-2]
            rev_last = completed[completed["year"] == last_y].groupby("category")["total_amount"].sum()
            rev_prev = completed[completed["year"] == prev_y].groupby("category")["total_amount"].sum()
            cat_growth = ((rev_last - rev_prev) / rev_prev * 100).round(2).reset_index()
            cat_growth.columns = ["category", "yoy_growth_pct"]
            cat_growth = cat_growth.sort_values("yoy_growth_pct", ascending=False)

        # Return rate by category
        ret_rate = (
            df.groupby("category")["is_returned"]
            .mean().mul(100).round(2)
            .reset_index().rename(columns={"is_returned": "return_rate_pct"})
            .sort_values("return_rate_pct", ascending=False)
        )

        return {
            "yoy_revenue_growth":       yoy_growth,
            "revenue_trend":            trend_label,
            "category_yoy_growth":      cat_growth,
            "category_return_rates":    ret_rate,
            "highest_growth_category":  cat_growth.iloc[0]["category"] if not cat_growth.empty else "N/A",
            "highest_return_category":  ret_rate.iloc[0]["category"]   if not ret_rate.empty  else "N/A",
        }

    # ── Full insights bundle ──────────────────────────────────

    def generate_all_insights(
        self, df: pd.DataFrame, cust_df: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Run all insight methods and return a single nested dict."""
        kpis        = self.compute_kpis(df)
        top         = self.compute_top_performers(df)
        cust        = self.compute_customer_insights(df, cust_df)
        trends      = self.compute_trend_insights(df)
        return {
            "kpis":             kpis,
            "top_performers":   top,
            "customer_insights": cust,
            "trends":           trends,
        }

    # ── Formatted text output ────────────────────────────────

    def format_insights_text(self, insights: Dict[str, Any]) -> str:
        """Render insights dictionary as readable plain text."""
        c    = self._curr
        kpis = insights["kpis"]
        top  = insights["top_performers"]
        ci   = insights["customer_insights"]
        tr   = insights["trends"]

        lines = [
            "=" * 60,
            "  E-COMMERCE BUSINESS INSIGHTS REPORT",
            "=" * 60,
            "",
            "BUSINESS KPIs",
            "-" * 40,
            f"  Total Revenue         : {c}{kpis['total_revenue']:>15,.2f}",
            f"  Total Orders          : {kpis['total_orders']:>15,}",
            f"  Total Customers       : {kpis['total_customers']:>15,}",
            f"  Avg Order Value       : {c}{kpis['avg_order_value']:>15,.2f}",
            f"  Return Rate           : {kpis['return_rate_pct']:>14.2f}%",
            f"  Avg Rating            : {kpis['avg_rating']:>15.2f}",
            f"  Net Promoter Score    : {kpis['net_promoter_score']:>15.1f}",
            f"  Repeat Revenue %      : {kpis['repeat_revenue_pct']:>14.2f}%",
            f"  MoM Revenue Growth    : {kpis['mom_revenue_growth']:>14.2f}%",
            f"  Retention Rate        : {kpis['retention_rate_pct']:>14.2f}%",
            "",
            "TOP PERFORMERS",
            "-" * 40,
            "  Top Categories by Revenue:",
        ]
        for _, row in top["top_categories"].iterrows():
            lines.append(f"    {row['category']:<25} {c}{row['revenue']:>12,.2f}")

        lines += ["", "  Top Products by Revenue:"]
        for _, row in top["top_products"].iterrows():
            lines.append(f"    {row['product_name'][:40]:<42} {c}{row['revenue']:>12,.2f}")

        lines += ["", "  Top Cities by Orders:"]
        for _, row in top["top_cities"].iterrows():
            lines.append(f"    {row['customer_city']:<25} {row['orders']:>8,} orders")

        if not top["best_month"].empty:
            bm = top["best_month"].iloc[0]
            lines += ["", f"  Best Month: {bm['month']} — {c}{bm['revenue']:,.2f}"]

        lines += [
            "",
            "CUSTOMER INSIGHTS",
            "-" * 40,
            f"  Top Payment Method : {ci['top_payment_method']}",
            f"  Top Device Type    : {ci['top_device_type']}",
            f"  Best Day of Week   : {ci['best_day_of_week']}",
            f"  Avg Customer CLV   : {c}{ci['avg_clv']:,.2f}",
            f"  % Repeat Customers : {ci['pct_repeat_customers']:.1f}%",
            "",
            "TREND INSIGHTS",
            "-" * 40,
            f"  YoY Revenue Growth  : {tr['yoy_revenue_growth']:.2f}%",
            f"  Revenue Trend       : {tr['revenue_trend']}",
            f"  Highest Growth Cat  : {tr['highest_growth_category']}",
            f"  Highest Return Cat  : {tr['highest_return_category']}",
            "=" * 60,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    from src.data_loader  import DataLoader
    from src.data_cleaner import DataCleaner

    loader  = DataLoader()
    df_raw  = loader.load_raw_data()
    cleaner = DataCleaner()
    df      = cleaner.clean(df_raw)

    ig       = InsightsGenerator()
    insights = ig.generate_all_insights(df)
    print(ig.format_insights_text(insights))
