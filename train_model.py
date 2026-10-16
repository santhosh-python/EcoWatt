"""Train and evaluate the next-day EcoWatt electricity model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.prepare_data import prepare_daily_readings

FEATURES = [
    "lag_1", "lag_2", "lag_7", "rolling_7",
    "day_of_week", "month", "day_of_year", "is_weekend",
]


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Create lagged and calendar features without using future readings."""
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").set_index("date")
    frame["lag_1"] = frame["kwh"].shift(1)
    frame["lag_2"] = frame["kwh"].shift(2)
    frame["lag_7"] = frame["kwh"].shift(7)
    frame["rolling_7"] = frame["kwh"].shift(1).rolling(7).mean()
    frame["day_of_week"] = frame.index.dayofweek
    frame["month"] = frame.index.month
    frame["day_of_year"] = frame.index.dayofyear
    frame["is_weekend"] = (frame.index.dayofweek >= 5).astype(int)
    frame["target"] = frame["kwh"].shift(-1)
    return frame.dropna(subset=FEATURES + ["target"])


def train_and_save(daily: pd.DataFrame, model_path: Path, metadata_path: Path) -> dict:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    features = build_features(daily)
    if len(features) < 30:
        raise ValueError("At least 30 usable daily readings are required to train")
    split = min(max(int(len(features) * 0.8), 1), len(features) - 1)
    train, test = features.iloc[:split], features.iloc[split:]
    model = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42,
    )
    model.fit(train[FEATURES], train["target"])
    prediction = model.predict(test[FEATURES])
    baseline = test["lag_1"].to_numpy()
    metrics = {
        "model_mae": float(mean_absolute_error(test["target"], prediction)),
        "model_rmse": float(np.sqrt(mean_squared_error(test["target"], prediction))),
        "baseline_mae": float(mean_absolute_error(test["target"], baseline)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(test["target"], baseline))),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {
        "features": FEATURES, "model": "HistGradientBoostingRegressor",
        "source_rows": int(len(daily)), "usable_rows": int(len(features)),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "last_date": str(daily["date"].iloc[-1]), "metrics": metrics,
        "cleaning": {
            "invalid_numeric_rows": "discarded", "incomplete_days": "discarded",
            "minimum_valid_fraction": 0.95,
            "within_day_missing": "time-interpolated then converted from kW-minutes to kWh",
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/household_power_consumption.zip"))
    parser.add_argument("--model", type=Path, default=Path("models/next_day_model.joblib"))
    parser.add_argument("--metadata", type=Path, default=Path("models/metadata.json"))
    args = parser.parse_args()
    metadata = train_and_save(
        prepare_daily_readings(args.dataset), args.model, args.metadata
    )
    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
