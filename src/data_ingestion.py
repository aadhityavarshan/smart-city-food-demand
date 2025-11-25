import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .config import RAW_DATA_DIR

def generate_dummy_data(n_days: int = 120):
    start = datetime.today() - timedelta(days=n_days)
    dates = [start + timedelta(days=i) for i in range(n_days)]

    rng = np.random.default_rng(seed=42)
    temp = rng.normal(loc=20, scale=5, size=n_days)
    is_weekend = [1 if date.weekday() >= 5 else 0 for date in dates]
    promo = rng.integers(0, 2, size=n_days)

    base = 80 + 10 * np.array(is_weekend) + 15 * np.array(promo)
    noise = rng.normal(loc=0, scale=8, size=n_days)
    orders = np.maximum(10, base + noise).astype(int)
    
    return pd.DataFrame({
        "date": dates,
        "temp_c": temp,
        "is_weekend": is_weekend,
        "promo": promo,
        "orders": orders
    })

def save_raw_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dummy_data()
    path = RAW_DATA_DIR / "restaurant_data.csv"
    df.to_csv(path, index=False)
    print(f"Raw data saved to {path}")

if __name__ == "__main__":
    save_raw_data()