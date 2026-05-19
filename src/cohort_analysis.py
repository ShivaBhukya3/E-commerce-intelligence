"""
CohortAnalysis — monthly cohort retention, revenue, and churn rate analysis.
"""

import logging
import sys
from pathlib import Path

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


class CohortAnalysis:
    """
    Monthly cohort analysis: retention rates, cohort revenue, and churn.

    All methods work on completed orders only (order_status == 'Completed').
    """

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.cfg        = _load_config(config_path)
        self.cohort_cfg = self.cfg["cohort"]

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _completed(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["order_status"] == "Completed"].copy()

    @staticmethod
    def _add_cohort_month(df: pd.DataFrame) -> pd.DataFrame:
        """Add cohort_month (first-purchase month per customer) and order_period."""
        df = df.copy()
        df["order_period"]    = df["order_date"].dt.to_period("M")
        first_purchase        = df.groupby("customer_id")["order_date"].min().dt.to_period("M")
        df["cohort_month"]    = df["customer_id"].map(first_purchase)
        df["cohort_index"]    = (df["order_period"] - df["cohort_month"]).apply(lambda x: x.n)
        return df

    # ── Retention Table ──────────────────────────────────────

    def create_cohort_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a cohort × month_index count matrix.

        Rows   = cohort month (first-purchase month)
        Columns = 0, 1, 2, … (months since first purchase)

        Returns pivot table of unique customer counts.
        """
        logger.info("Creating cohort retention table...")
        completed = self._completed(df)
        cohort_df = self._add_cohort_month(completed)

        max_idx = self.cohort_cfg["retention_months"]
        cohort_df = cohort_df[cohort_df["cohort_index"] <= max_idx]

        cohort_pivot = (
            cohort_df.groupby(["cohort_month", "cohort_index"])["customer_id"]
            .nunique()
            .reset_index()
            .pivot(index="cohort_month", columns="cohort_index", values="customer_id")
        )
        cohort_pivot.index = cohort_pivot.index.astype(str)
        cohort_pivot.columns = [f"Month {c}" for c in cohort_pivot.columns]

        min_size = self.cohort_cfg["min_cohort_size"]
        cohort_pivot = cohort_pivot[cohort_pivot["Month 0"] >= min_size]

        logger.info("Cohort table: %d cohorts x %d periods", *cohort_pivot.shape)
        return cohort_pivot

    # ── Retention Rates ──────────────────────────────────────

    def compute_retention_rates(self, cohort_table: pd.DataFrame) -> pd.DataFrame:
        """
        Convert absolute counts to % retained relative to cohort size (Month 0).

        Returns a percentage DataFrame suitable for a heatmap.
        """
        logger.info("Computing retention rates...")
        cohort_sizes = cohort_table["Month 0"]
        retention    = cohort_table.divide(cohort_sizes, axis=0).round(4) * 100
        logger.info("Retention rates computed.")
        return retention

    # ── Cohort Revenue ───────────────────────────────────────

    def compute_cohort_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Average revenue per customer per cohort over time.

        Returns pivot: cohort_month × cohort_index → avg revenue.
        """
        logger.info("Computing cohort revenue...")
        completed = self._completed(df)
        cohort_df = self._add_cohort_month(completed)

        max_idx   = self.cohort_cfg["retention_months"]
        cohort_df = cohort_df[cohort_df["cohort_index"] <= max_idx]

        rev_pivot = (
            cohort_df.groupby(["cohort_month", "cohort_index"])["total_amount"]
            .mean()
            .round(2)
            .reset_index()
            .pivot(index="cohort_month", columns="cohort_index", values="total_amount")
        )
        rev_pivot.index   = rev_pivot.index.astype(str)
        rev_pivot.columns = [f"Month {c}" for c in rev_pivot.columns]
        logger.info("Cohort revenue computed.")
        return rev_pivot

    # ── Churn Rate ───────────────────────────────────────────

    def compute_churn_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Monthly churn rate = customers who did NOT purchase in month M
        who did purchase in month M-1, divided by customers in M-1.

        Returns DataFrame with columns: month, active_customers, churned, churn_rate_pct.
        """
        logger.info("Computing monthly churn rate...")
        completed = self._completed(df)
        completed["month"] = completed["order_date"].dt.to_period("M")

        monthly_active = (
            completed.groupby("month")["customer_id"]
            .apply(set)
            .reset_index()
        )
        monthly_active.columns = ["month", "customers"]
        monthly_active = monthly_active.sort_values("month").reset_index(drop=True)

        records = []
        for i in range(1, len(monthly_active)):
            prev_month    = monthly_active.loc[i - 1, "month"]
            curr_month    = monthly_active.loc[i,     "month"]
            prev_custs    = monthly_active.loc[i - 1, "customers"]
            curr_custs    = monthly_active.loc[i,     "customers"]
            churned       = len(prev_custs - curr_custs)
            active_prev   = len(prev_custs)
            churn_rate    = round(churned / active_prev * 100, 2) if active_prev > 0 else 0
            records.append({
                "month":            str(curr_month),
                "active_customers": len(curr_custs),
                "churned":          churned,
                "churn_rate_pct":   churn_rate,
            })

        churn_df = pd.DataFrame(records)
        logger.info("Churn rate computed for %d months.", len(churn_df))
        return churn_df

    # ── Full pipeline ─────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> dict:
        """Run all cohort analyses and return results dict."""
        cohort_table = self.create_cohort_table(df)
        retention    = self.compute_retention_rates(cohort_table)
        revenue      = self.compute_cohort_revenue(df)
        churn        = self.compute_churn_rate(df)
        return {
            "cohort_table": cohort_table,
            "retention":    retention,
            "revenue":      revenue,
            "churn":        churn,
        }


if __name__ == "__main__":
    from src.data_loader  import DataLoader
    from src.data_cleaner import DataCleaner

    loader  = DataLoader()
    df_raw  = loader.load_raw_data()
    cleaner = DataCleaner()
    df      = cleaner.clean(df_raw)

    ca      = CohortAnalysis()
    results = ca.run(df)
    print("Cohort table shape :", results["cohort_table"].shape)
    print("Retention shape    :", results["retention"].shape)
    print("Churn records      :", len(results["churn"]))
