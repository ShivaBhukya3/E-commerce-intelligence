"""
Generate a realistic 10,000-row Indian E-Commerce dataset.
Run: python data/generate_data.py
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime, timedelta

# ── Resolve project root ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "config" / "config.yaml"

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def date_range_array(start: str, end: str) -> np.ndarray:
    """Return numpy array of daily dates between start and end."""
    return pd.date_range(start=start, end=end, freq="D").to_numpy()


def seasonal_weights(dates: np.ndarray, boost_months: list, multiplier: float) -> np.ndarray:
    """Higher probability weights for seasonal months."""
    months = pd.DatetimeIndex(dates).month
    weights = np.where(np.isin(months, boost_months), multiplier, 1.0)
    return weights / weights.sum()


def generate_dataset(cfg: dict) -> pd.DataFrame:
    gen_cfg   = cfg["data_generation"]
    cat_cfg   = cfg["categories"]
    geo_cfg   = cfg["geography"]["cities_states"]
    ord_cfg   = cfg["orders"]

    rng = random.Random(gen_cfg["random_seed"])
    np.random.seed(gen_cfg["random_seed"])

    N          = gen_cfg["n_rows"]
    N_CUST     = gen_cfg["n_customers"]
    DATE_START = gen_cfg["date_start"]
    DATE_END   = gen_cfg["date_end"]
    BOOST_M    = gen_cfg["seasonal_boost_months"]
    BOOST_X    = gen_cfg["seasonal_multiplier"]
    MISS_RATE  = gen_cfg["missing_rating_pct"]
    MISS_REV   = gen_cfg["missing_review_pct"]

    all_dates = date_range_array(DATE_START, DATE_END)
    s_weights = seasonal_weights(all_dates, BOOST_M, BOOST_X)

    cities  = list(geo_cfg.keys())
    states  = list(geo_cfg.values())
    city_idx = list(range(len(cities)))

    # ── Customer pool ────────────────────────────────────────
    customer_ids    = [f"CUST{str(i).zfill(5)}" for i in range(1, N_CUST + 1)]
    customer_names  = _generate_names(N_CUST, rng)
    customer_emails = [f"{n.lower().replace(' ', '.')}{rng.randint(1, 999)}@{rng.choice(['gmail.com','yahoo.com','hotmail.com','outlook.com'])}"
                       for n in customer_names]
    customer_ages   = np.random.randint(18, 71, size=N_CUST)
    customer_genders = np.random.choice(ord_cfg["genders"], size=N_CUST,
                                        p=ord_cfg["gender_weights"])
    cust_city_idx   = np.random.choice(city_idx, size=N_CUST)
    customer_cities = [cities[i] for i in cust_city_idx]
    customer_states_list = [states[i] for i in cust_city_idx]

    reg_dates = np.random.choice(
        pd.date_range(start="2021-01-01", end=DATE_START, freq="D"),
        size=N_CUST, replace=True
    )
    reg_dates = pd.to_datetime(reg_dates)

    cust_df = pd.DataFrame({
        "customer_id":    customer_ids,
        "customer_name":  customer_names,
        "customer_email": customer_emails,
        "customer_age":   customer_ages,
        "customer_gender": customer_genders,
        "customer_city":  customer_cities,
        "customer_state": customer_states_list,
        "registration_date": reg_dates,
    })

    # ── Order rows ───────────────────────────────────────────
    rows = []
    order_counter = 1

    # Assign orders — 60% repeat so sample with replacement biased to same customers
    # First give each customer at least 1 order, then fill remainder
    base_cust = customer_ids.copy()
    extra_n   = N - N_CUST
    # repeat customers: choose from top 60% of pool
    loyal_pool = customer_ids[:int(0.60 * N_CUST)]
    extra_custs = np.random.choice(loyal_pool, size=extra_n, replace=True).tolist()
    all_custs = base_cust + extra_custs
    np.random.shuffle(all_custs)

    # Pre-sample bulk arrays for speed
    order_dates   = np.random.choice(all_dates, size=N, p=s_weights, replace=True)
    order_statuses= np.random.choice(ord_cfg["statuses"], size=N, p=ord_cfg["status_weights"])
    payment_meths = np.random.choice(ord_cfg["payment_methods"], size=N, p=ord_cfg["payment_weights"])
    device_types  = np.random.choice(ord_cfg["device_types"],  size=N, p=ord_cfg["device_weights"])
    quantities    = np.random.randint(1, 11, size=N)
    discounts     = np.random.choice([0, 5, 10, 15, 20, 25, 30, 40, 50], size=N,
                                     p=[0.30, 0.15, 0.15, 0.10, 0.10, 0.08, 0.06, 0.04, 0.02])
    ship_costs    = np.round(np.random.uniform(0, 200, size=N), 2)
    delivery_days = np.random.randint(2, 16, size=N)

    cat_names = list(cat_cfg.keys())
    cat_weights_arr = [0.30, 0.25, 0.15, 0.10, 0.08, 0.07, 0.05]  # Electronics dominant
    chosen_cats  = np.random.choice(cat_names, size=N, p=cat_weights_arr)

    for i in range(N):
        cid    = all_custs[i]
        crow   = cust_df[cust_df["customer_id"] == cid].iloc[0]
        cat    = chosen_cats[i]
        c_cfg  = cat_cfg[cat]
        sub    = rng.choice(c_cfg["sub_categories"])
        brand  = rng.choice(c_cfg["brands"])
        pid    = f"PROD{rng.randint(1000, 9999)}"
        pname  = _product_name(cat, sub, brand, rng)
        price  = round(rng.uniform(*c_cfg["price_range"]), 2)
        status = order_statuses[i]
        is_ret = (status == "Returned")
        rating = None
        review = None
        if status == "Completed":
            if rng.random() > MISS_RATE:
                rating = int(np.random.choice([1,2,3,4,5], p=[0.05,0.08,0.17,0.35,0.35]))
            if rng.random() > MISS_REV:
                review = rng.choice(REVIEW_TEMPLATES.get(cat, ["Good product", "Nice item", "Worth buying"]))

        rows.append({
            "order_id":          f"ORD{str(order_counter).zfill(7)}",
            "customer_id":       cid,
            "customer_name":     crow["customer_name"],
            "customer_email":    crow["customer_email"],
            "customer_age":      crow["customer_age"],
            "customer_gender":   crow["customer_gender"],
            "customer_city":     crow["customer_city"],
            "customer_state":    crow["customer_state"],
            "registration_date": crow["registration_date"],
            "order_date":        pd.Timestamp(order_dates[i]),
            "order_status":      status,
            "product_id":        pid,
            "product_name":      pname,
            "category":          cat,
            "sub_category":      sub,
            "brand":             brand,
            "unit_price":        price,
            "quantity":          int(quantities[i]),
            "discount_percent":  float(discounts[i]),
            "shipping_cost":     float(ship_costs[i]),
            "payment_method":    payment_meths[i],
            "device_type":       device_types[i],
            "rating":            rating,
            "review_text":       review,
            "is_returned":       is_ret,
            "delivery_days":     int(delivery_days[i]),
        })
        order_counter += 1

    df = pd.DataFrame(rows)
    df["order_date"]        = pd.to_datetime(df["order_date"])
    df["registration_date"] = pd.to_datetime(df["registration_date"])
    df = df.sort_values("order_date").reset_index(drop=True)
    return df


# ── Name & review helpers ────────────────────────────────────

FIRST_NAMES = [
    "Aarav","Aditi","Aisha","Akash","Amara","Amit","Anjali","Ankit","Ananya","Arjun",
    "Ayaan","Bhavna","Chetan","Deepa","Divya","Gaurav","Ishaan","Jaya","Karan","Kavya",
    "Kunal","Lakshmi","Manish","Meera","Mohit","Nandini","Neha","Nikhil","Pooja","Priya",
    "Rahul","Rajesh","Riya","Rohit","Sanya","Saurabh","Shivam","Sneha","Suresh","Tanvi",
    "Umesh","Varun","Vikram","Vinay","Vishal","Yash","Zara","Rajan","Harsha","Pallavi",
    "Arun","Sunita","Vikas","Rekha","Deepak","Swati","Manoj","Shruti","Pankaj","Smita"
]
LAST_NAMES = [
    "Sharma","Patel","Gupta","Singh","Kumar","Verma","Joshi","Mehta","Shah","Nair",
    "Iyer","Reddy","Rao","Jain","Agarwal","Malhotra","Kapoor","Bose","Ghosh","Banerjee",
    "Chatterjee","Das","Mishra","Tiwari","Pandey","Yadav","Chauhan","Saxena","Srivastava","Kulkarni"
]

def _generate_names(n: int, rng: random.Random) -> list:
    return [f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(n)]


def _product_name(category: str, sub: str, brand: str, rng: random.Random) -> str:
    adjectives = ["Pro", "Plus", "Elite", "Max", "Ultra", "Smart", "Premium", "Lite", "Eco", "Classic"]
    models     = ["X1", "S2", "V3", "Z5", "A7", "G9", "T10", "M4", "R6", "K8"]
    return f"{brand} {sub} {rng.choice(adjectives)} {rng.choice(models)}"


REVIEW_TEMPLATES = {
    "Electronics":   ["Excellent performance!", "Great value for money.", "Battery life is amazing.",
                      "Fast delivery, good packaging.", "Works perfectly as described.",
                      "Highly recommend this product.", "Sound quality is superb."],
    "Fashion":       ["Fits perfectly!", "Nice quality fabric.", "Loved the colour.",
                      "Great stitching quality.", "Comfortable and stylish.",
                      "Perfect for casual wear.", "Exactly as shown in picture."],
    "Home & Kitchen":["Very sturdy and durable.", "Easy to assemble.", "Looks great in my kitchen.",
                      "Great quality product.", "Saves a lot of time.",
                      "Perfect size for my home.", "Exactly what I needed."],
    "Sports":        ["Great for workouts!", "Durable and sturdy.", "Lightweight and comfortable.",
                      "Good grip and traction.", "Perfect for outdoor activities.",
                      "High quality material.", "Worth every rupee."],
    "Beauty":        ["Skin feels amazing!", "Nice fragrance.", "Works as promised.",
                      "Gentle on sensitive skin.", "Love the packaging.",
                      "Will definitely buy again.", "Best product in this range."],
    "Books":         ["Very informative read.", "Gripping storyline.", "Great for students.",
                      "Well written and engaging.", "Fast delivery, original copy.",
                      "Changed my perspective.", "Highly recommend to everyone."],
    "Toys":          ["Kids loved it!", "Good quality and safe.", "Great educational toy.",
                      "Perfect gift for children.", "Durable and colorful.",
                      "Hours of entertainment.", "Assembly was easy."],
}


if __name__ == "__main__":
    print("Loading configuration...")
    cfg = load_config(CONFIG_PATH)

    print(f"Generating {cfg['data_generation']['n_rows']:,} rows of e-commerce data...")
    df = generate_dataset(cfg)

    out_path = ROOT / cfg["paths"]["raw_data"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nDataset saved: {out_path}")
    print(f"Shape         : {df.shape}")
    print(f"Date range    : {df['order_date'].min().date()} -> {df['order_date'].max().date()}")
    print(f"Unique customers : {df['customer_id'].nunique():,}")
    print(f"Unique products  : {df['product_id'].nunique():,}")
    print(f"Order statuses:\n{df['order_status'].value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
