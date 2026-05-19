"""
Unit tests for DataCleaner.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_cleaner import DataCleaner


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Minimal raw DataFrame that mimics the real dataset structure."""
    return pd.DataFrame({
        "order_id":          ["ORD0000001", "ORD0000002", "ORD0000003", "ORD0000001"],  # last is dup
        "customer_id":       ["CUST001",    "CUST002",    None,          "CUST001"],
        "customer_name":     ["Alice Sharma","Bob Patel", "Carol Gupta", "Alice Sharma"],
        "customer_email":    ["a@g.com",    "b@g.com",    "c@g.com",    "a@g.com"],
        "customer_age":      [25,            32,           45,           25],
        "customer_gender":   ["Female",     "Male",       "Female",     "Female"],
        "customer_city":     ["Mumbai",     "Delhi",      "Pune",       "Mumbai"],
        "customer_state":    ["Maharashtra","Delhi",      "Maharashtra","Maharashtra"],
        "registration_date": ["2021-06-01", "2021-08-15", "2022-01-10", "2021-06-01"],
        "order_date":        ["2022-03-15", "2022-05-20", "2022-07-01", "2022-03-15"],
        "order_status":      ["Completed",  "Returned",   "Completed",  "Completed"],
        "product_id":        ["PROD1001",   "PROD1002",   "PROD1003",   "PROD1001"],
        "product_name":      ["Phone X",    "Shirt Y",    "Pan Z",      "Phone X"],
        "category":          ["Electronics","Fashion",    "Home & Kitchen","Electronics"],
        "sub_category":      ["Smartphones","Men's Clothing","Cookware", "Smartphones"],
        "brand":             ["Samsung",    "Zara",       "Prestige",   "Samsung"],
        "unit_price":        [15000.0,      999.0,        450.0,        15000.0],
        "quantity":          [1,            2,            1,            1],
        "discount_percent":  [10.0,         0.0,          5.0,          10.0],
        "shipping_cost":     [49.0,         0.0,          59.0,         49.0],
        "payment_method":    ["UPI",        "COD",        "Credit Card","UPI"],
        "device_type":       ["Mobile",     "Desktop",    "Mobile",     "Mobile"],
        "rating":            [4.0,          None,         5.0,          4.0],
        "review_text":       ["Great!",     None,         None,         "Great!"],
        "is_returned":       [False,        True,         False,        False],
        "delivery_days":     [4,            8,            6,            4],
    })


@pytest.fixture
def cleaner() -> DataCleaner:
    return DataCleaner()


@pytest.fixture
def cleaned_df(cleaner, sample_raw_df) -> pd.DataFrame:
    return cleaner.clean(sample_raw_df)


# ── Test: remove_duplicates ───────────────────────────────────

class TestRemoveDuplicates:
    def test_duplicate_order_id_removed(self, cleaner, sample_raw_df):
        df = cleaner.fix_data_types(sample_raw_df.copy())
        df = cleaner.remove_duplicates(df)
        # ORD0000001 appears twice → should keep only 1
        assert df["order_id"].duplicated().sum() == 0

    def test_row_count_reduced(self, cleaner, sample_raw_df):
        df = cleaner.fix_data_types(sample_raw_df.copy())
        df = cleaner.remove_duplicates(df)
        assert len(df) < len(sample_raw_df)


# ── Test: handle_missing_values ──────────────────────────────

class TestHandleMissingValues:
    def test_null_customer_id_dropped(self, cleaner, sample_raw_df):
        df = cleaner.fix_data_types(sample_raw_df.copy())
        df = cleaner.remove_duplicates(df)
        df = cleaner.handle_missing_values(df)
        assert df["customer_id"].isnull().sum() == 0

    def test_rating_filled(self, cleaner, sample_raw_df):
        df = cleaner.fix_data_types(sample_raw_df.copy())
        df = cleaner.remove_duplicates(df)
        df = cleaner.handle_missing_values(df)
        assert df["rating"].isnull().sum() == 0

    def test_review_text_filled(self, cleaner, sample_raw_df):
        df = cleaner.fix_data_types(sample_raw_df.copy())
        df = cleaner.remove_duplicates(df)
        df = cleaner.handle_missing_values(df)
        assert df["review_text"].isnull().sum() == 0
        # Default fill value should be present
        assert "No Review" in df["review_text"].values


# ── Test: derived columns exist ──────────────────────────────

class TestDerivedColumnsExist:
    EXPECTED_COLS = [
        "total_amount", "profit_margin", "order_month", "order_year",
        "order_quarter", "order_day_of_week", "order_month_year",
        "is_weekend", "customer_tenure_days", "delivery_status",
        "discount_bucket", "price_tier",
    ]

    def test_all_derived_columns_present(self, cleaned_df):
        for col in self.EXPECTED_COLS:
            assert col in cleaned_df.columns, f"Missing column: {col}"

    def test_delivery_status_valid_values(self, cleaned_df):
        valid = {"Fast", "Normal", "Slow"}
        actual = set(cleaned_df["delivery_status"].dropna().unique())
        assert actual.issubset(valid), f"Unexpected delivery_status values: {actual - valid}"

    def test_order_quarter_range(self, cleaned_df):
        assert cleaned_df["order_quarter"].between(1, 4).all()

    def test_order_month_range(self, cleaned_df):
        assert cleaned_df["order_month"].between(1, 12).all()


# ── Test: total_amount calculation ───────────────────────────

class TestTotalAmountCalculation:
    def test_formula_correctness(self, cleaned_df):
        """total_amount = unit_price * qty * (1 - disc/100) + shipping"""
        row = cleaned_df.iloc[0]
        expected = round(
            row["unit_price"] * row["quantity"] * (1 - row["discount_percent"] / 100)
            + row["shipping_cost"],
            2,
        )
        assert abs(row["total_amount"] - expected) < 0.01, (
            f"total_amount mismatch: got {row['total_amount']}, expected {expected}"
        )

    def test_total_amount_positive(self, cleaned_df):
        assert (cleaned_df["total_amount"] >= 0).all()

    def test_profit_margin_range(self, cleaned_df):
        # Simulated margin at 40% gross → should be ~40
        assert cleaned_df["profit_margin"].between(0, 100).all()


# ── Test: no negative prices ─────────────────────────────────

class TestNoNegativePrices:
    def test_unit_price_positive(self, cleaned_df):
        assert (cleaned_df["unit_price"] > 0).all()

    def test_shipping_cost_non_negative(self, cleaned_df):
        assert (cleaned_df["shipping_cost"] >= 0).all()

    def test_discount_in_valid_range(self, cleaned_df):
        assert cleaned_df["discount_percent"].between(0, 100).all()

    def test_quantity_at_least_one(self, cleaned_df):
        assert (cleaned_df["quantity"] >= 1).all()


# ── Test: data type correctness ───────────────────────────────

class TestDataTypes:
    def test_order_date_is_datetime(self, cleaned_df):
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df["order_date"])

    def test_registration_date_is_datetime(self, cleaned_df):
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df["registration_date"])

    def test_is_returned_is_bool(self, cleaned_df):
        assert cleaned_df["is_returned"].dtype == bool

    def test_customer_tenure_non_negative(self, cleaned_df):
        assert (cleaned_df["customer_tenure_days"] >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
