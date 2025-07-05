#!/usr/bin/env python3
# src/dashboard/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import numpy as np
from typing import List, Optional

app = FastAPI(title="Electricity Price Forecast API")

# Load models on startup
dashboard_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
model_dir = os.path.join(dashboard_dir, "../models/rf_tuned_models")
models = {}
for fname in os.listdir(model_dir):
    if fname.endswith(".joblib"):
        zone = fname.replace("rf_tuned_", "").replace(".joblib", "")
        models[zone] = joblib.load(os.path.join(model_dir, fname))

# Load feature template
processed_dir = os.path.abspath(os.path.join(dashboard_dir, "../../data/processed"))
feature_cols = pd.read_csv(os.path.join(processed_dir, "features.csv"), nrows=1).columns.tolist()

class ForecastRequest(BaseModel):
    zone: str
    features: List[float]  # must match feature_cols order

class ForecastResponse(BaseModel):
    zone: str
    prediction: float
    model: str

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    zone = req.zone.upper()
    if zone not in models:
        raise HTTPException(status_code=404, detail=f"Zone '{zone}' not found")
    if len(req.features) != len(feature_cols):
        raise HTTPException(status_code=400,
            detail=f"Expected {len(feature_cols)} features, got {len(req.features)}")

    # Prepare input
    x = np.array(req.features, dtype=float).reshape(1, -1)
    # Predict
    model = models[zone]
    pred = model.predict(x)[0]
    return ForecastResponse(zone=zone, prediction=float(pred), model=f"rf_tuned_{zone}")

@app.get("/zones", response_model=List[str])
def get_zones():
    return sorted(models.keys())
