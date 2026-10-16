from pathlib import Path

import pandas as pd

from src.prepare_data import load_minute_readings, prepare_daily_readings


def test_prepare_daily_readings_interpolates_missing_value(tmp_path: Path):
    timestamps = pd.date_range("2024-01-01", periods=1440, freq="min")
    rows = pd.DataFrame(
        {"Date": timestamps.strftime("%d/%m/%Y"), "Time": timestamps.strftime("%H:%M:%S"),
         "Global_active_power": ["1.0"] * 1440}
    )
    rows.loc[100, "Global_active_power"] = "?"
    source = tmp_path / "household_power_consumption.txt"
    rows.to_csv(source, sep=";", index=False)
    daily = prepare_daily_readings(source)
    assert len(daily) == 1
    assert daily.loc[0, "valid_minutes"] == 1439
    assert daily.loc[0, "kwh"] == 24.0


def test_invalid_dates_are_removed(tmp_path: Path):
    source = tmp_path / "readings.txt"
    pd.DataFrame(
        {"Date": ["01/01/2024", "not-a-date"], "Time": ["00:00:00", "00:01:00"],
         "Global_active_power": ["1.0", "2.0"]}
    ).to_csv(source, sep=";", index=False)
    assert len(load_minute_readings(source)) == 1
