"""
CustomerSegmentation — RFM rule-based segments + KMeans behavioural clusters.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


# ── RFM segment name mapping ──────────────────────────────────
def _assign_rfm_segment(r: int, f: int, m: int) -> str:
    """Map individual R/F/M scores (1-5) to a named business segment."""
    rfm = f"{r}{f}{m}"
    if r >= 4 and f >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal Customers"
    elif r >= 4 and f <= 1:
        return "New Customers"
    elif r >= 3 and f <= 2:
        return "Promising"
    elif r <= 2 and f >= 3:
        return "At Risk"
    elif r <= 2 and f <= 2 and m <= 2:
        return "Lost Customers"
    else:
        return "Potential Loyalists"


class CustomerSegmentation:
    """
    Segments customers using RFM rules and KMeans clustering.

    Typical usage:
        seg   = CustomerSegmentation()
        rfm   = seg.rfm_segmentation(customer_features_df)
        kmeans = seg.behavioral_segmentation(customer_features_df)
        profiles = seg.generate_segment_profiles(rfm, orders_df)
    """

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.cfg     = _load_config(config_path)
        self.seg_cfg = self.cfg["segmentation"]
        self.scaler  = StandardScaler()
        self.kmeans  = None
        self.pca     = None

    # ── RFM Segmentation ─────────────────────────────────────

    def rfm_segmentation(self, cust_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply business rules to R/F/M scores and add a 'rfm_segment' column.

        Parameters
        ----------
        cust_df : output of FeatureEngineer.build_customer_features()
                  must contain r_score, f_score, m_score columns.

        Returns the same DataFrame with 'rfm_segment' appended.
        """
        logger.info("Applying RFM segmentation rules...")
        df = cust_df.copy()

        df["rfm_segment"] = df.apply(
            lambda row: _assign_rfm_segment(
                int(row.get("r_score", 1)),
                int(row.get("f_score", 1)),
                int(row.get("m_score", 1)),
            ),
            axis=1,
        )

        counts = df["rfm_segment"].value_counts()
        logger.info("RFM segments assigned:\n%s", counts.to_string())
        return df

    # ── KMeans Behavioural Clustering ────────────────────────

    def behavioral_segmentation(
        self, cust_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Cluster customers into k=4 groups based on spending behaviour.

        Features used: total_spent, total_orders, avg_order_value, return_rate.

        Returns
        -------
        cust_df_with_cluster : DataFrame with 'cluster' and 'cluster_label' columns
        inertias             : array of inertia values for elbow plot (k=1..8)
        """
        logger.info("Running KMeans behavioural segmentation...")

        FEATURE_COLS = ["total_spent", "total_orders", "avg_order_value", "return_rate"]
        df = cust_df.copy()
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

        X_scaled = self.scaler.fit_transform(df[FEATURE_COLS])

        # Elbow inertias for k=1..8 (used in notebook plotting)
        inertias = []
        for k in range(1, 9):
            km = KMeans(n_clusters=k, random_state=self.seg_cfg["kmeans_random_state"],
                        max_iter=self.seg_cfg["kmeans_max_iter"], n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        # Final model with k=4
        k = self.seg_cfg["kmeans_clusters"]
        self.kmeans = KMeans(
            n_clusters=k,
            random_state=self.seg_cfg["kmeans_random_state"],
            max_iter=self.seg_cfg["kmeans_max_iter"],
            n_init=10,
        )
        df["cluster"] = self.kmeans.fit_predict(X_scaled)

        # PCA for 2-D visualisation
        self.pca = PCA(n_components=self.seg_cfg["pca_components"])
        pca_coords = self.pca.fit_transform(X_scaled)
        df["pca_x"] = pca_coords[:, 0]
        df["pca_y"] = pca_coords[:, 1]

        # Auto-label clusters by average total_spent (descending)
        cluster_means = df.groupby("cluster")["total_spent"].mean().sort_values(ascending=False)
        label_map = dict(
            zip(cluster_means.index, self.seg_cfg["cluster_labels"])
        )
        df["cluster_label"] = df["cluster"].map(label_map)

        logger.info("Cluster distribution:\n%s", df["cluster_label"].value_counts().to_string())
        return df, np.array(inertias)

    # ── Segment Profiles ─────────────────────────────────────

    def generate_segment_profiles(
        self,
        cust_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        segment_col: str = "rfm_segment",
    ) -> pd.DataFrame:
        """
        Generate summary statistics per segment.

        Returns one row per segment with: size, pct, avg_clv, avg_order_value,
        avg_orders, top_category, recommendation.
        """
        logger.info("Generating segment profiles for column: %s", segment_col)
        total = len(cust_df)

        # Top category per segment via orders join
        seg_cat = (
            orders_df[orders_df["order_status"] == "Completed"]
            .merge(cust_df[["customer_id", segment_col]], on="customer_id", how="left")
            .groupby([segment_col, "category"])
            .size()
            .reset_index(name="cnt")
        )
        top_cat = (
            seg_cat.sort_values("cnt", ascending=False)
            .drop_duplicates(subset=[segment_col])
            .set_index(segment_col)["category"]
        )

        profiles = (
            cust_df.groupby(segment_col)
            .agg(
                size            = (segment_col,      "count"),
                avg_clv         = ("clv",            "mean"),
                avg_order_value = ("avg_order_value","mean"),
                avg_orders      = ("total_orders",   "mean"),
                avg_return_rate = ("return_rate",    "mean"),
            )
            .reset_index()
        )
        profiles["pct"]           = (profiles["size"] / total * 100).round(1)
        profiles["avg_clv"]       = profiles["avg_clv"].round(2)
        profiles["avg_order_value"] = profiles["avg_order_value"].round(2)
        profiles["avg_orders"]    = profiles["avg_orders"].round(1)
        profiles["avg_return_rate"] = profiles["avg_return_rate"].round(1)
        profiles["top_category"]  = profiles[segment_col].map(top_cat)

        RECOMMENDATIONS = {
            "Champions":           "Reward with exclusive offers; ask for referrals.",
            "Loyal Customers":     "Upsell premium products; offer loyalty programme.",
            "Promising":           "Nurture with personalised recommendations.",
            "Potential Loyalists": "Provide discount codes to encourage 2nd/3rd purchase.",
            "New Customers":       "Send onboarding email; highlight top products.",
            "At Risk":             "Win-back campaign with time-limited discount.",
            "Lost Customers":      "Reactivation email with aggressive discount.",
            "High Value":          "Priority customer service; early access to new products.",
            "Medium Value":        "Bundle deals to increase basket size.",
            "Budget Shoppers":     "Promote sale events and seasonal discounts.",
            "Churned Risk":        "Churn-prevention offer; gather feedback.",
        }
        profiles["recommendation"] = profiles[segment_col].map(
            RECOMMENDATIONS
        ).fillna("Engage with targeted campaigns.")

        return profiles.sort_values("avg_clv", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from src.data_loader        import DataLoader
    from src.data_cleaner       import DataCleaner
    from src.feature_engineering import FeatureEngineer

    loader  = DataLoader()
    df_raw  = loader.load_raw_data()
    cleaner = DataCleaner()
    df      = cleaner.clean(df_raw)
    fe      = FeatureEngineer()
    cust    = fe.build_customer_features(df)

    seg  = CustomerSegmentation()
    cust = seg.rfm_segmentation(cust)
    cust, inertias = seg.behavioral_segmentation(cust)
    profiles = seg.generate_segment_profiles(cust, df)
    print(profiles.to_string())
