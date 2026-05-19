"""
FeatureEngineer — compute RFM scores, CLV, and customer-level behavioral features.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

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


class FeatureEngineer:
    """
    Computes RFM scores, Customer Lifetime Value, and rich per-customer metrics.

    All methods accept a *completed-orders-only* DataFrame unless noted.
    """

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.cfg      = _load_config(config_path)
        self.feat_cfg = self.cfg["features"]
        # reference date = one day after the latest order
        self._ref_date: pd.Timestamp = None

    # ── Helper ───────────────────────────────────────────────

    def _completed_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to completed (non-cancelled, non-pending) orders only."""
        return df[df["order_status"] == "Completed"].copy()

    def _set_ref_date(self, df: pd.DataFrame) -> None:
        self._ref_date = df["order_date"].max() + pd.Timedelta(days=1)

    # ── RFM Scores ───────────────────────────────────────────

    def compute_rfm_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build per-customer RFM table with quintile scores 1-5.

        Returns DataFrame indexed by customer_id with columns:
            recency, frequency, monetary,
            r_score, f_score, m_score, rfm_score, rfm_segment_raw
        """
        logger.info("Computing RFM scores...")
        completed = self._completed_orders(df)
        self._set_ref_date(completed)

        rfm = (
            completed.groupby("customer_id")
            .agg(
                recency   = ("order_date",    lambda x: (self._ref_date - x.max()).days),
                frequency = ("order_id",       "nunique"),
                monetary  = ("total_amount",   "sum"),
            )
            .reset_index()
        )
        rfm["monetary"] = rfm["monetary"].round(2)

        bins = self.feat_cfg["rfm_quantile_bins"]

        def _safe_qcut_asc(series: pd.Series, n: int) -> pd.Series:
            """qcut ascending 1..n, robust to duplicate bin edges."""
            try:
                return pd.qcut(series, q=n, labels=range(1, n + 1), duplicates="drop").astype(int)
            except ValueError:
                # Fall back to rank-based scoring when duplicates collapse bins
                return pd.qcut(series.rank(method="first"), q=n,
                               labels=range(1, n + 1), duplicates="drop").astype(int)

        def _safe_qcut_desc(series: pd.Series, n: int) -> pd.Series:
            """qcut descending n..1 (lower raw value = higher score)."""
            try:
                return pd.qcut(series, q=n, labels=range(n, 0, -1), duplicates="drop").astype(int)
            except ValueError:
                return pd.qcut(series.rank(method="first"), q=n,
                               labels=range(n, 0, -1), duplicates="drop").astype(int)

        # Recency: lower is better → reverse ranking (score 5 = most recent)
        rfm["r_score"] = _safe_qcut_desc(rfm["recency"],   bins)
        rfm["f_score"] = _safe_qcut_asc(rfm["frequency"],  bins)
        rfm["m_score"] = _safe_qcut_asc(rfm["monetary"],   bins)

        rfm["rfm_score"]       = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
        rfm["rfm_segment_raw"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)

        logger.info("RFM computed for %d customers.", len(rfm))
        return rfm

    # ── Customer Lifetime Value ──────────────────────────────

    def compute_customer_lifetime_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CLV = Avg Order Value x Purchase Frequency x Customer Lifespan (months).

        Returns DataFrame with customer_id and clv columns.
        """
        logger.info("Computing Customer Lifetime Value...")
        completed = self._completed_orders(df)
        lifespan_months = self.feat_cfg["clv_lifespan_months"]

        cust = (
            completed.groupby("customer_id")
            .agg(
                total_revenue = ("total_amount", "sum"),
                total_orders  = ("order_id",     "nunique"),
            )
            .reset_index()
        )
        cust["avg_order_value"]     = (cust["total_revenue"] / cust["total_orders"]).round(2)
        # Annualised purchase frequency (orders per month × 12)
        purchase_freq               = cust["total_orders"] / lifespan_months
        cust["clv"]                 = (cust["avg_order_value"] * purchase_freq * lifespan_months).round(2)

        logger.info("CLV computed. Mean CLV = %.2f", cust["clv"].mean())
        return cust[["customer_id", "avg_order_value", "total_orders", "total_revenue", "clv"]]

    # ── Per-customer metrics ─────────────────────────────────

    def compute_customer_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate rich per-customer statistics from all orders (not just completed).

        Returns one row per customer_id.
        """
        logger.info("Computing customer-level metrics...")

        completed = self._completed_orders(df)

        # Orders & spend (on completed)
        spend = (
            completed.groupby("customer_id")
            .agg(
                total_orders      = ("order_id",     "nunique"),
                total_spent       = ("total_amount", "sum"),
                avg_order_value   = ("total_amount", "mean"),
                avg_rating_given  = ("rating",       "mean"),
                preferred_category   = ("category",  lambda x: x.mode().iloc[0] if not x.empty else np.nan),
                preferred_payment    = ("payment_method", lambda x: x.mode().iloc[0] if not x.empty else np.nan),
                preferred_device     = ("device_type", lambda x: x.mode().iloc[0] if not x.empty else np.nan),
            )
            .reset_index()
        )
        spend["total_spent"]      = spend["total_spent"].round(2)
        spend["avg_order_value"]  = spend["avg_order_value"].round(2)
        spend["avg_rating_given"] = spend["avg_rating_given"].round(2)

        # Return rate (all orders)
        ret = (
            df.groupby("customer_id")
            .agg(
                all_orders    = ("order_id", "nunique"),
                returned_cnt  = ("is_returned", "sum"),
            )
            .reset_index()
        )
        ret["return_rate"] = (ret["returned_cnt"] / ret["all_orders"] * 100).round(2)

        # Registration date for tenure
        reg = df.groupby("customer_id")["registration_date"].first().reset_index()
        last_order = df.groupby("customer_id")["order_date"].max().reset_index()
        last_order.columns = ["customer_id", "last_order_date"]

        metrics = (
            spend
            .merge(ret[["customer_id", "return_rate", "all_orders"]], on="customer_id", how="left")
            .merge(reg, on="customer_id", how="left")
            .merge(last_order, on="customer_id", how="left")
        )

        logger.info("Customer metrics computed for %d customers.", len(metrics))
        return metrics

    # ── Behavioral features ──────────────────────────────────

    def add_behavioral_features(
        self, df: pd.DataFrame, metrics: pd.DataFrame, clv_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Append CLV-based and frequency-based behavioral flags to the metrics table.

        Parameters
        ----------
        df      : full cleaned orders DataFrame
        metrics : output of compute_customer_metrics()
        clv_df  : output of compute_customer_lifetime_value()
        """
        logger.info("Adding behavioral features...")
        feat_cfg = self.feat_cfg

        merged = metrics.merge(
            clv_df[["customer_id", "clv"]], on="customer_id", how="left"
        )

        # High-value customer flag (top 20% by CLV)
        threshold = merged["clv"].quantile(feat_cfg["high_value_percentile"])
        merged["is_high_value"] = merged["clv"] >= threshold

        # Shopping frequency segment
        def _freq_segment(n: int) -> str:
            one_time   = feat_cfg["shopping_segments"]["one_time"]
            occasional = feat_cfg["shopping_segments"]["occasional"]
            regular    = feat_cfg["shopping_segments"]["regular"]
            if n <= one_time:
                return "One-time"
            elif occasional[0] <= n <= occasional[1]:
                return "Occasional"
            elif regular[0] <= n <= regular[1]:
                return "Regular"
            else:
                return "Loyal"

        merged["shopping_segment"] = merged["total_orders"].apply(_freq_segment)

        # Average time between orders (days)
        def _avg_days_between(cid: str) -> float:
            dates = sorted(df[df["customer_id"] == cid]["order_date"].unique())
            if len(dates) < 2:
                return np.nan
            gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            return round(np.mean(gaps), 1)

        logger.info("Computing avg days between orders (may take a moment)...")
        merged["avg_days_between_orders"] = merged["customer_id"].map(
            df.groupby("customer_id")["order_date"].apply(
                lambda x: round(np.mean(np.diff(sorted(x.unique())).astype("timedelta64[D]").astype(int)), 1)
                if len(x.unique()) > 1 else np.nan
            )
        )

        logger.info("Behavioral features added.")
        return merged

    # ── Master customer feature table ────────────────────────

    def build_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full pipeline: returns one comprehensive row per customer with all features.

        Columns: customer_id, recency, frequency, monetary,
                 r_score, f_score, m_score, rfm_score,
                 clv, total_orders, total_spent, avg_order_value,
                 return_rate, avg_rating_given, preferred_category,
                 preferred_payment, is_high_value, shopping_segment,
                 avg_days_between_orders, ...
        """
        rfm     = self.compute_rfm_scores(df)
        clv_df  = self.compute_customer_lifetime_value(df)
        metrics = self.compute_customer_metrics(df)
        final   = self.add_behavioral_features(df, metrics, clv_df)

        final = final.merge(
            rfm[["customer_id", "recency", "frequency", "monetary",
                 "r_score", "f_score", "m_score", "rfm_score", "rfm_segment_raw"]],
            on="customer_id", how="left"
        )

        # Add demographic info from raw
        demo = df.drop_duplicates("customer_id")[
            ["customer_id", "customer_name", "customer_email",
             "customer_age", "customer_gender", "customer_city", "customer_state"]
        ]
        final = final.merge(demo, on="customer_id", how="left")

        logger.info("Customer feature table built: %d customers x %d features",
                    len(final), final.shape[1])
        return final


if __name__ == "__main__":
    from src.data_loader  import DataLoader
    from src.data_cleaner import DataCleaner

    loader  = DataLoader()
    df_raw  = loader.load_raw_data()
    cleaner = DataCleaner()
    df      = cleaner.clean(df_raw)

    fe        = FeatureEngineer()
    cust_feat = fe.build_customer_features(df)
    print(cust_feat.head())
    print(f"Shape: {cust_feat.shape}")
