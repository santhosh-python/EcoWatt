"""Prepare daily electricity readings from the UCI minute-level dataset.

Rows with invalid numeric readings are discarded. A day is retained when it
contains all expected timestamps and at least 95% valid readings; missing
readings within retained days are time-interpolated and converted from kW
minutes to kWh. Days below that threshold are dropped.
"""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

EXPECTED_MINUTES = 24 * 60
MIN_VALID_FRACTION = 0.95
POWER_COLUMN = "Global_active_power"


def _open_source(path: Path):
    if path.suffix.lower() == ".zip":
        archive = ZipFile(path)
        names = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if not names:
            archive.close()
            raise ValueError(f"No .txt dataset found in {path}")
        return archive, archive.open(names[0])
    return None, path


def load_minute_readings(path: str | Path) -> pd.DataFrame:
    """Load source rows, parse timestamps, and coerce invalid power to NaN."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    archive, handle = _open_source(source)
    try:
        frame = pd.read_csv(
            handle,
            sep=";",
            na_values=["?"],
            usecols=["Date", "Time", POWER_COLUMN],
            dtype={"Date": "string", "Time": "string", POWER_COLUMN: "string"},
        )
    finally:
        if archive is not None:
            archive.close()
    frame["timestamp"] = pd.to_datetime(
        frame["Date"] + " " + frame["Time"], dayfirst=True, errors="coerce"
    )
    frame["power_kw"] = pd.to_numeric(frame[POWER_COLUMN], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).loc[:, ["timestamp", "power_kw"]]
    return frame.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")


def prepare_daily_readings(
    path: str | Path, *, min_valid_fraction: float = MIN_VALID_FRACTION
) -> pd.DataFrame:
    """Convert minute readings to daily kWh readings."""
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be between 0 and 1")
    minute = load_minute_readings(path)
    days: list[dict[str, object]] = []
    for date, group in minute.groupby(minute.index.normalize()):
        expected = pd.date_range(
            date, date + pd.Timedelta(days=1) - pd.Timedelta(minutes=1), freq="min"
        )
        readings = group["power_kw"].reindex(expected)
        valid = int(readings.notna().sum())
        if len(readings) != EXPECTED_MINUTES or valid < EXPECTED_MINUTES * min_valid_fraction:
            continue
        filled = readings.interpolate(method="time").ffill().bfill()
        days.append(
            {"date": date.date().isoformat(), "kwh": float(filled.sum() / 60), "valid_minutes": valid}
        )
    result = pd.DataFrame(days)
    if result.empty:
        raise ValueError("No complete days met the valid-reading threshold")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/daily_energy.csv"))
    args = parser.parse_args()
    output = prepare_daily_readings(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output):,} daily readings to {args.output}")