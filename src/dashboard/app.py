#!/usr/bin/env python3
# src/dashboard/app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.exceptions import DataConversionWarning
from datetime import timedelta
from streamlit_autorefresh import st_autorefresh
import datetime
import json


# ─── Auto–refresh & countdown ───────────────────────────────────────────────
# UI-refresh every second (countdown)
st_autorefresh(interval=1_000, key="ui_refresh")
# Data-refresh every hour
st_autorefresh(interval=3_600_000, key="data_refresh")

now = datetime.datetime.now()
next_hour = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
remaining = next_hour - now
mins, secs = divmod(remaining.seconds, 60)

st.markdown(f"""
<style>
  .countdown-text {{
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    margin: 20px 0;
  }}
  .progress-container {{
    width: 100%; height: 10px; background: #e1e8f0; border-radius: 5px; overflow: hidden; margin-bottom: 20px;
  }}
  .progress-bar {{
    background: #1F77B4; height: 100%; width: 0%; animation: progressBar {remaining.seconds}s linear forwards;
  }}
  @keyframes progressBar {{
    from {{ width: 0%; }}
    to   {{ width: 100%; }}
  }}
</style>
<div class="countdown-text">Next update in {mins:02d}:{secs:02d}</div>
<div class="progress-container"><div class="progress-bar"></div></div>
""", unsafe_allow_html=True)

# ─── Suppress sklearn warnings ──────────────────────────────────────────────
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=DataConversionWarning)

# ─── Page config & base CSS ───────────────────────────────────────────────
st.set_page_config(page_title="Electricity Price Forecast", layout="wide")
st.markdown("""
    <style>
    .reportview-container, .main { background-color: #f7f9fc; }
    .css-1d391kg .css-1d391kg { background-color: #003366; }
    .css-1d391kg .css-1d391kg * { color: white !important; }
    [data-testid="metric-container"] {
        background-color: #ffffff; border: 1px solid #dde2e6;
        border-radius: 8px; padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR     = os.path.join(PROJECT_ROOT, "src", "models", "rf_tuned_models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")

FEATURES_PATH = os.path.join(PROCESSED_DIR, "features.csv")
PRICES_PATH   = os.path.join(PROCESSED_DIR, "..", "interim", "clean_prices.csv")
WEATHER_PATH  = os.path.join(RAW_DIR, "weather.csv")
GEOJSON_PATH  = os.path.join(RAW_DIR, "zones.geojson")

# ─── Caching loaders ────────────────────────────────────────────────────────
@st.cache_data
def load_models(model_dir):
    models = {}
    for fname in os.listdir(model_dir):
        if fname.endswith(".joblib") and fname.startswith("rf_tuned_"):
            zone = fname.replace("rf_tuned_", "").replace(".joblib", "")
            models[zone] = joblib.load(os.path.join(model_dir, fname))
    return models

@st.cache_data
def load_data(features_path, prices_path, weather_path):
    df_feat  = pd.read_csv(features_path, index_col=0, parse_dates=[0])
    df_price = pd.read_csv(prices_path, index_col=0, parse_dates=[0])
    df_weather = (
        pd.read_csv(weather_path, index_col=0, parse_dates=[0])
        if os.path.exists(weather_path)
        else None
    )
    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_feat.fillna(method="ffill", inplace=True)
    df_feat.fillna(method="bfill", inplace=True)
    return df_feat, df_price, df_weather

# ─── Load resources ────────────────────────────────────────────────────────
models = load_models(MODEL_DIR)
df_feat, df_price, df_weather = load_data(FEATURES_PATH, PRICES_PATH, WEATHER_PATH)

latest = df_feat.iloc[[-1]]
timestamp = latest.index[0]

# ─── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
zone_labels = {
    "NO1": "NO1 – Oslo area (East)",
    "NO2": "NO2 – Central Norway",
    "NO3": "NO3 – Western Coast",
    "NO4": "NO4 – Eastern Norway (north)",
    "NO5": "NO5 – Northern Norway"
}
label_list = [zone_labels[z] for z in models.keys()]
sel_label = st.sidebar.selectbox("Zone", label_list)
sel_zone = [z for z, lbl in zone_labels.items() if lbl == sel_label][0]
hist_range = st.sidebar.slider("Historical window (days)", 1, 30, value=7)
show_weather = st.sidebar.checkbox("Show weather overlay", value=True)
show_errors  = st.sidebar.checkbox("Show error metrics", value=True)

# ─── Title & timestamp ──────────────────────────────────────────────────────
st.title("⚡ Electricity Price Forecast Dashboard ⚡")
st.markdown(f"**Last updated**: {timestamp.strftime('%Y-%m-%d %H:%M')} (Oslo)")

# ─── Next-hour forecasts & KPI-cards ────────────────────────────────────────
forecasts = []
for zone, model in models.items():
    pred = model.predict(latest.values)[0]
    forecasts.append({"Zone": zone, "Next Hour (NOK/MWh)": float(pred)})
df_fc = pd.DataFrame(forecasts).set_index("Zone")

st.header("Next Hour Forecasts")
cols = st.columns(len(models))
for col, zone in zip(cols, df_fc.index):
    col.metric(label=zone, value=f"{df_fc.loc[zone, 'Next Hour (NOK/MWh)']:.2f}")



# ─── Map point for selected zone ────────────────────────────────────────────
zone_coords = {
    "NO1": (59.91, 10.75),
    "NO2": (63.43, 10.40),
    "NO3": (59.12, 6.04),
    "NO4": (61.19, 8.97),
    "NO5": (69.65, 18.96)
}
lat, lon = zone_coords.get(sel_zone, (0, 0))
st.subheader(f"Zone {sel_zone} Location")
st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=6)

# ─── Historical plot ────────────────────────────────────────────────────────
st.subheader(f"Historical Price vs Prediction for {sel_zone}")
end = timestamp
start = end - timedelta(days=hist_range)
actual = df_price[sel_zone].loc[start:end]
hist_feat = df_feat.loc[start:end].copy()
hist_preds = models[sel_zone].predict(hist_feat.values)
df_plot = pd.DataFrame({"Actual": actual.values, "Predicted": hist_preds}, index=actual.index)
st.line_chart(df_plot)

if show_weather and df_weather is not None:
    wf = df_weather[["temperature", "wind"]].loc[start:end]
    st.line_chart(wf)

if show_errors:
    diff = df_plot["Actual"] - df_plot["Predicted"]
    mae  = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    mape = np.mean(np.abs(diff / (actual.values + 1e-6))) * 100
    st.markdown(f"**MAE:** {mae:.2f}   **RMSE:** {rmse:.2f}   **MAPE:** {mape:.2f}%")

# ─── Feature importances ─────────────────────────────────────────────────────
st.subheader(f"Top 10 Feature Importances for {sel_zone}")
importances = pd.Series(models[sel_zone].feature_importances_, index=df_feat.columns).nlargest(10)
st.bar_chart(importances)

# ─── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Powered by tuned RandomForest models with weather & calendar features.")
st.markdown("Made with ❤️ by Sushant")
