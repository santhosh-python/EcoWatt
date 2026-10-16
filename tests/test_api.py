from datetime import date

from fastapi.testclient import TestClient

import api


def test_health_reports_missing_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(api, "METADATA_PATH", tmp_path / "missing.json")
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_predict_validates_history_length():
    response = TestClient(api.app).post(
        "/predict", json={"history": [1, 2], "last_date": str(date(2024, 1, 1))}
    )
    assert response.status_code == 422


def test_predict_returns_cost_and_carbon(monkeypatch):
    class FakeModel:
        def predict(self, frame):
            assert list(frame.columns) == api.FEATURES
            return [12.5]

    monkeypatch.setattr(api, "_load_artifacts", lambda: (FakeModel(), {"model": "test-model"}))
    response = TestClient(api.app).post(
        "/predict",
        json={"history": [1, 2, 3, 4, 5, 6, 7], "last_date": "2024-01-01"},
    )
    assert response.status_code == 200
    assert response.json()["prediction_kwh"] == 12.5
    assert response.json()["estimated_cost_usd"] == 2.0
