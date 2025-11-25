import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from ..config import PROCESSED_DATA_DIR

def load_processed_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA_DIR / "processed_restaurant_data.csv", parse_dates=["date"])

def train_rf_model():
    df = load_processed_data()

    features = [
        "temp_c",
        "is_weekend",
        "promo",
        "day_of_week",
        "rolling_orders_7d",
        "month"
    ]

    X = df[features]
    y = df["orders"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    base_model = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
    }

    search = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_absolute_error')
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"RandomForest MAE: {mae:.2f}")

    model_path = PROCESSED_DATA_DIR / "rf_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved RF model to {model_path}")

    return best_model, mae

if __name__ == "__main__":
    train_rf_model()