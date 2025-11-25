import pandas as pd
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_raw_data():
    return pd.read_csv(RAW_DATA_DIR / "restaurant_data.csv", parse_dates=["date"])

def build_features(df):
    df = df.sort_values("date")
    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["rolling_orders_7d"] = df["orders"].rolling(window=7, min_periods=1).mean()
    return df

def save_processed(df):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / "processed_restaurant_data.csv"
    df.to_csv(path, index=False)
    print(f"Processed data saved to {path}")

if __name__ == "__main__":
    raw = load_raw_data()
    processed = build_features(raw)
    save_processed(processed)