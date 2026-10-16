"""Local FastAPI service for EcoWatt next-day predictions."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
FEATURES = [
    "lag_1", "lag_2", "lag_7", "rolling_7",
    "day_of_week", "month", "day_of_year", "is_weekend",
]
MODEL_PATH = ROOT / "models" / "next_day_model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

app = FastAPI(title="EcoWatt Forecast API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    history: list[float] = Field(..., min_length=7, max_length=366)
    last_date: date


class PredictionResponse(BaseModel):
    date: date
    prediction_kwh: float
    estimated_cost_usd: float
    estimated_carbon_kg: float
    model: str


def _load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _feature_row(history: list[float], last_date: date) -> pd.DataFrame:
    if any(value < 0 or value > 1000 for value in history):
        raise HTTPException(status_code=422, detail="history values must be between 0 and 1000 kWh")
    target_date = last_date + timedelta(days=1)
    values = {
        "lag_1": history[-1], "lag_2": history[-2], "lag_7": history[-7],
        "rolling_7": sum(history[-7:]) / 7,
        "day_of_week": target_date.weekday(), "month": target_date.month,
        "day_of_year": target_date.timetuple().tm_yday,
        "is_weekend": int(target_date.weekday() >= 5),
    }
    return pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)


@app.get("/health")
def health() -> dict[str, object]:
    model, metadata = _load_artifacts()
    return {
        "status": "ok" if model is not None else "model_unavailable",
        "model_loaded": model is not None,
        "model": metadata.get("model") if metadata else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model, metadata = _load_artifacts()
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model is unavailable. Run train_model.py first.")
    try:
        prediction = max(0.0, float(model.predict(_feature_row(request.history, request.last_date))[0]))
    except HTTPException:
        raise
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid prediction input: {error}") from error
    return PredictionResponse(
        date=request.last_date + timedelta(days=1),
        prediction_kwh=round(prediction, 3),
        estimated_cost_usd=round(prediction * 0.16, 2),
        estimated_carbon_kg=round(prediction * 0.4, 2),
        model=metadata["model"],
    )
