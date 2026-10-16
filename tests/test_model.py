import pandas as pd
import pytest

from train_model import build_features


def test_features_use_previous_values_and_calendar():
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    frame = build_features(pd.DataFrame({"date": dates, "kwh": range(12)}))
    assert frame.iloc[0]["lag_1"] == 6
    assert frame.iloc[0]["target"] == 8
    assert frame.iloc[0]["rolling_7"] == pytest.approx(3)

