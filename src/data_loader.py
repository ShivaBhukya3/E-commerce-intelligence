"""
DataLoader — loads raw and processed datasets with schema validation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# ── Setup ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "order_id", "customer_id", "customer_name", "customer_email",
    "customer_age", "customer_gender", "customer_city", "customer_state",
    "registration_date", "order_date", "order_status", "product_id",
    "product_name", "category", "sub_category", "brand", "unit_price",
    "quantity", "discount_percent", "shipping_cost", "payment_method",
    "device_type", "rating", "review_text", "is_returned", "delivery_days",
]


def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


class DataLoader:
    """Loads raw and processed e-commerce datasets from disk."""

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.cfg       = _load_config(config_path)
        self.root      = PROJECT_ROOT
        self._raw_df:  Optional[pd.DataFrame] = None
        self._proc_df: Optional[pd.DataFrame] = None

    # ── Public Methods ───────────────────────────────────────

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw CSV dataset, parse dates, cache internally."""
        path = self.root / self.cfg["paths"]["raw_data"]
        logger.info("Loading raw data from: %s", path)
        try:
            df = pd.read_csv(
                path,
                parse_dates=["order_date", "registration_date"],
                low_memory=False,
            )
            self._raw_df = df
            logger.info("Raw data loaded: %s rows x %s cols", *df.shape)
            return df
        except FileNotFoundError:
            logger.error("Raw data file not found: %s", path)
            raise
        except Exception as exc:
            logger.exception("Unexpected error loading raw data: %s", exc)
            raise

    def load_processed_data(self) -> pd.DataFrame:
        """Load cleaned/processed CSV dataset."""
        path = self.root / self.cfg["paths"]["processed_data"]
        logger.info("Loading processed data from: %s", path)
        try:
            df = pd.read_csv(
                path,
                parse_dates=["order_date", "registration_date"],
                low_memory=False,
            )
            self._proc_df = df
            logger.info("Processed data loaded: %s rows x %s cols", *df.shape)
            return df
        except FileNotFoundError:
            logger.error("Processed data file not found. Run data_cleaner first: %s", path)
            raise
        except Exception as exc:
            logger.exception("Unexpected error loading processed data: %s", exc)
            raise

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Check that all required columns are present."""
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            logger.warning("Schema validation FAILED. Missing columns: %s", missing)
            return False
        logger.info("Schema validation PASSED. All %d required columns present.", len(REQUIRED_COLUMNS))
        return True

    def get_data_info(self, df: Optional[pd.DataFrame] = None) -> None:
        """Print comprehensive dataset summary."""
        if df is None:
            df = self._raw_df or self._proc_df
        if df is None:
            logger.warning("No dataframe available. Call load_raw_data() or load_processed_data() first.")
            return

        print("\n" + "=" * 60)
        print("  DATASET OVERVIEW")
        print("=" * 60)
        print(f"  Shape            : {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"  Memory usage     : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print("\n  DATA TYPES:")
        for col, dtype in df.dtypes.items():
            print(f"    {col:<30} {str(dtype)}")

        null_counts = df.isnull().sum()
        null_pct    = (null_counts / len(df) * 100).round(2)
        null_df     = pd.DataFrame({"Null Count": null_counts, "Null %": null_pct})
        null_df     = null_df[null_df["Null Count"] > 0]

        print("\n  MISSING VALUES:")
        if null_df.empty:
            print("    No missing values.")
        else:
            print(null_df.to_string())

        print("\n  NUMERIC SUMMARY:")
        print(df.describe(include="number").round(2).to_string())
        print("=" * 60 + "\n")


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_raw_data()
    loader.validate_schema(df)
    loader.get_data_info(df)
