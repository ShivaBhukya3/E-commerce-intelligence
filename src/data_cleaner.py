"""
DataCleaner — clean raw e-commerce data and add derived columns.
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


class DataCleaner:
    """
    Cleans raw e-commerce DataFrame and engineers base derived columns.

    Usage:
        cleaner = DataCleaner()
        df_clean = cleaner.clean(df_raw)
    """

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.cfg        = _load_config(config_path)
        self.clean_cfg  = self.cfg["cleaning"]
        self._initial_rows: int = 0
        self._report_lines: list = []

    # ── Main Entry Point ─────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline and return a cleaned DataFrame."""
        self._initial_rows = len(df)
        self._report_lines = []

        df = self.fix_data_types(df)
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.remove_outliers(df)
        df = self.add_derived_columns(df)
        df = df.reset_index(drop=True)

        self.generate_cleaning_report(df)
        return df

    # ── Step 1: Fix data types ───────────────────────────────

    def fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast columns to their correct dtypes."""
        logger.info("Fixing data types...")
        df = df.copy()

        for col in ["order_date", "registration_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        for col in ["unit_price", "discount_percent", "shipping_cost"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        for col in ["quantity", "delivery_days", "customer_age"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(0).astype(int)

        if "is_returned" in df.columns:
            df["is_returned"] = df["is_returned"].astype(bool)

        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        self._report_lines.append("fix_data_types: completed")
        logger.info("Data types fixed.")
        return df

    # ── Step 2: Remove duplicates ────────────────────────────

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop fully duplicate rows and duplicate order_ids."""
        before = len(df)
        df = df.drop_duplicates()
        full_dups = before - len(df)

        before = len(df)
        df = df.drop_duplicates(subset=["order_id"], keep="first")
        oid_dups = before - len(df)

        msg = f"remove_duplicates: removed {full_dups} full duplicates, {oid_dups} duplicate order_ids"
        self._report_lines.append(msg)
        logger.info(msg)
        return df

    # ── Step 3: Handle missing values ────────────────────────

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill or drop missing values using business rules."""
        before = len(df)

        # Drop rows where order_id or customer_id is null — unusable
        df = df.dropna(subset=["order_id", "customer_id"])
        dropped = before - len(df)

        # Fill rating with per-category median
        if "rating" in df.columns and "category" in df.columns:
            df["rating"] = df.groupby("category")["rating"].transform(
                lambda s: s.fillna(s.median())
            )
            # If an entire category has no ratings, fall back to global median
            global_median = df["rating"].median()
            df["rating"] = df["rating"].fillna(global_median)

        # Fill missing review text
        fill_text = self.clean_cfg.get("fill_review_text", "No Review")
        if "review_text" in df.columns:
            df["review_text"] = df["review_text"].fillna(fill_text)

        # Fill other minor numeric gaps with sensible defaults
        df["discount_percent"]  = df["discount_percent"].fillna(0.0)
        df["shipping_cost"]     = df["shipping_cost"].fillna(df["shipping_cost"].median())
        df["delivery_days"]     = df["delivery_days"].fillna(df["delivery_days"].median().astype(int))

        msg = (f"handle_missing_values: dropped {dropped} rows (null order/customer id). "
               f"Filled rating, review_text, discount, shipping, delivery nulls.")
        self._report_lines.append(msg)
        logger.info(msg)
        return df

    # ── Step 4: Remove outliers ──────────────────────────────

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with implausible price/quantity values."""
        before = len(df)
        max_price = self.clean_cfg["max_unit_price"]
        min_price = self.clean_cfg["min_unit_price"]
        max_qty   = self.clean_cfg["max_quantity"]

        df = df[df["unit_price"].between(min_price, max_price, inclusive="neither")]
        df = df[df["quantity"] <= max_qty]
        df = df[df["quantity"] >= 1]
        df = df[df["discount_percent"].between(0, 100)]

        removed = before - len(df)
        msg = f"remove_outliers: removed {removed} rows (price/quantity/discount out of range)"
        self._report_lines.append(msg)
        logger.info(msg)
        return df

    # ── Step 5: Add derived columns ──────────────────────────

    def add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new columns from existing data."""
        logger.info("Adding derived columns...")
        df = df.copy()

        # Revenue calculation
        df["total_amount"] = (
            df["unit_price"] * df["quantity"] * (1 - df["discount_percent"] / 100)
            + df["shipping_cost"]
        ).round(2)

        # Simulated profit margin (assumes 60% COGS)
        df["profit_margin"] = ((df["unit_price"] - df["unit_price"] * 0.60) / df["unit_price"] * 100).round(2)

        # Date parts
        df["order_month"]       = df["order_date"].dt.month
        df["order_year"]        = df["order_date"].dt.year
        df["order_quarter"]     = df["order_date"].dt.quarter
        df["order_day_of_week"] = df["order_date"].dt.day_name()
        df["order_month_year"]  = df["order_date"].dt.to_period("M").astype(str)

        # Weekend flag
        df["is_weekend"] = df["order_date"].dt.dayofweek >= 5

        # Customer tenure at time of order
        df["customer_tenure_days"] = (
            df["order_date"] - df["registration_date"]
        ).dt.days.clip(lower=0)

        # Delivery speed classification
        df["delivery_status"] = pd.cut(
            df["delivery_days"],
            bins=[0, 4, 10, 999],
            labels=["Fast", "Normal", "Slow"],
            right=True,
        )

        # Discount bucket
        df["discount_bucket"] = pd.cut(
            df["discount_percent"],
            bins=[-1, 0, 10, 25, 50, 100],
            labels=["No Discount", "Low (1-10%)", "Medium (11-25%)", "High (26-50%)", "Very High (>50%)"],
        )

        # Price tier
        df["price_tier"] = pd.cut(
            df["unit_price"],
            bins=[0, 500, 2000, 10000, 50000, float("inf")],
            labels=["Budget", "Economy", "Mid-range", "Premium", "Luxury"],
        )

        self._report_lines.append("add_derived_columns: total_amount, profit_margin, date parts, delivery_status added")
        logger.info("Derived columns added.")
        return df

    # ── Step 6: Cleaning report ──────────────────────────────

    def generate_cleaning_report(self, df_clean: pd.DataFrame) -> None:
        """Print a before/after cleaning summary."""
        final_rows  = len(df_clean)
        removed_pct = (self._initial_rows - final_rows) / max(self._initial_rows, 1) * 100

        print("\n" + "=" * 60)
        print("  DATA CLEANING REPORT")
        print("=" * 60)
        print(f"  Rows before  : {self._initial_rows:>10,}")
        print(f"  Rows after   : {final_rows:>10,}")
        print(f"  Rows removed : {self._initial_rows - final_rows:>10,} ({removed_pct:.1f}%)")
        print(f"  Columns now  : {df_clean.shape[1]:>10}")
        print("\n  STEPS APPLIED:")
        for line in self._report_lines:
            print(f"    • {line}")

        null_remaining = df_clean.isnull().sum()
        null_remaining = null_remaining[null_remaining > 0]
        print("\n  REMAINING NULLS:")
        if null_remaining.empty:
            print("    None")
        else:
            print(null_remaining.to_string())
        print("=" * 60 + "\n")


if __name__ == "__main__":
    from src.data_loader import DataLoader

    loader  = DataLoader()
    df_raw  = loader.load_raw_data()

    cleaner  = DataCleaner()
    df_clean = cleaner.clean(df_raw)

    out_path = PROJECT_ROOT / _load_config()["paths"]["processed_data"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)
    logger.info("Cleaned data saved to: %s  [%d rows]", out_path, len(df_clean))
