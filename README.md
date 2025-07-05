# ⚡ Electricity-Forecast

> **A reproducible end-to-end pipeline for hourly electricity-price prediction in Norway (zones NO1-NO5) – from raw API calls, through feature engineering & model tuning, to a live Streamlit dashboard.**

---

## 📑 Table of Contents
1. [Project Scope](#project-scope)  
2. [Key Highlights](#key-highlights)  
3. [End-to-End Architecture](#end-to-end-architecture)  
4. [Directory Layout](#directory-layout)  
5. [Workflow in Detail](#workflow-in-detail)  
6. [Local Setup & Quick Start](#local-setup--quick-start)  
7. [Model Benchmarks](#model-benchmarks)  
8. [Streamlit Dashboard](#streamlit-dashboard)  
9. [Scheduled Retraining (CRONS)](#scheduled-retraining-crons)  
10. [CI/CD & Cloud Deploy](#cicd--cloud-deploy)  
11. [Roadmap](#roadmap)  
12. [Contributing](#contributing)  
13. [License](#license)

---

## Project Scope
| Goal | Deliver a pluggable framework that **predicts the next-hour spot price** for each Norwegian bidding zone (NO1–NO5) and publishes the forecast in real time. |
|------|-------------------------------------------------------------------------------------------------------------------------------------------|
| Why? | Intraday price volatility matters for **EV charging**, **industrial load-shifting**, and **home-battery optimisation**. |
| MVP | Historical + day-ahead prices → enriched with weather & calendar features → baseline + tuned ML models → dashboard. |

---

## Key Highlights
| Category | What we did | Why it matters |
|----------|-------------|---------------|
| **Data** | 365 × 24 h of prices (HvaKosterStrømmen) + 2 y of hourly weather (Frost API) | solid history for both training & covariate generation |
| **Features** | time-of-day, weekday, weekend, Norway-holiday flags, rolling means (3/6/24 h), capped pct-changes, rolling weather means | captures seasonality + exogenous effects |
| **Baselines** | Persistence, SARIMA, Random Forest | establishes performance floor & classic statistical reference |
| **Hyper-tuning** | GridSearchCV on RF (n_estimators, max_depth, min_samples_leaf) with TimeSeriesSplit | lowers SMAPE by ≈30 % vs baseline-RF |
| **Metrics** | MAE • RMSE • MAPE • SMAPE | SMAPE chosen for near-zero-price stability |
| **Dashboard** | KPI cards, animated countdown, auto-refresh, choropleth map, CSV export, error metrics, feature importances | one-stop UI for analysts & operations |
| **Ops** | Hourly autorefresh, cron fetch script, weekly retrain script, GitHub → Streamlit Cloud CI/CD | devil-free maintenance |

---

## End-to-End Architecture

```text
            ┌────────────┐
            │ Cron Jobs  │  (cron_fetch.sh  •  retrain_weekly.sh)
            └──────┬─────┘
                   ▼
┌───────────────────────────────────────────────────────────────────┐
│                           Data Layer                            │
│   1. fetch_hvakoster.py ─────────┐                               │
│   2. frost_fetch.py  ────────────┼──►  data/raw/*                │
│                                   │                               │
│   3. preprocess.py (clean)  ──────┘                               │
│          data/interim/clean_prices.csv                            │
│   4. build_features.py  →  data/processed/{features,labels}.csv   │
└───────────────────────────────────────────────────────────────────┘
                   ▼
┌───────────────────────────────────────────────────────────────────┐
│                          Model Layer                             │
│   • train_persistence.py                                         │
│   • grid_search_sarima.py  → best (p,d,q)(P,D,Q)                 │
│   • train_arima.py                                               │
│   • train_rf.py  &  grid_search_rf.py                            │
│   • train_tuned_rf_smape.py  → src/models/rf_tuned_models/*.joblib│
│   • metrics saved under data/processed/*.csv                     │
└───────────────────────────────────────────────────────────────────┘
                   ▼
┌───────────────────────────────────────────────────────────────────┐
│                         Dashboard Layer                          │
│   Streamlit src/dashboard/app.py                                 │
│   - Live forecasts (auto-refresh)                                │
│   - Choropleth (Altair + GeoJSON)                                │
│   - Download, KPIs, feature importances                          │
└───────────────────────────────────────────────────────────────────┘

## Directory Layout
<details>
<summary>Click to expand</summary>

```text
Electricity-forecast/
├── .streamlit/               # Streamlit theme
│   └── config.toml
├── config/
│   └── api_keys.yml          # kept out of VCS with .gitignore
├── data/
│   ├── raw/                  # original downloads
│   ├── interim/              # cleaned but not yet featured
│   └── processed/            # ready for ML
├── notebooks/                # exploratory analysis
├── src/
│   ├── data/                 # ingestion & preprocessing
│   ├── features/             # feature builders
│   ├── models/               # training scripts
│   └── dashboard/            # Streamlit app
├── models/                   # saved artefacts
├── scripts/                  # cron & helper bash scripts
├── requirements.txt
└── README.md



## Workflow in Detail

<details>
<summary>1 ▶ Data Ingestion</summary>

| Script             | Purpose                                                                                  |
|--------------------|------------------------------------------------------------------------------------------|
| `fetch_hvakoster.py` | Loops date-by-date and calls the HvaKosterStrømmen API for NO1–NO5. Saves `all_zones_YYYY-MM-DD.csv` to `data/raw/`. |
| `frost_fetch.py`   | Performs OAuth2 to Frost API, fetches temperature & wind for station SN18700 (Oslo-Blindern) in 90-day chunks. Saves `weather.csv`. |

</details>

<details>
<summary>2 ▶ Preprocessing</summary>

- **Merge & timezone**: Reads all raw CSVs, converts UTC timestamps → `Europe/Oslo`.  
- **Re-index**: Ensures a continuous hourly `DatetimeIndex` without gaps.  
- **Clean missing & outliers**:  
  - Linear interpolation for short gaps.  
  - Drop or clip negative prices.  
- **Output**:  
  - `data/interim/clean_prices.csv`  

</details>

<details>
<summary>3 ▶ Feature Engineering</summary>

| Feature           | Implementation                              | Comment                       |
|-------------------|---------------------------------------------|-------------------------------|
| Calendar hour     | `df.index.hour`                             | Captures intraday pattern     |
| Weekday (0–6)     | `df.index.dayofweek`                        | Weekly seasonality           |
| Weekend flag      | `weekday.isin([5,6])`                       | Binary weekend indicator      |
| Norway holidays   | Custom `pandas.tseries.holiday` calendar    | Flags extra spikes/valleys    |
| Rolling means     | `df[col].rolling(window).mean()` (3h, 6h, 24h) | Smooths short-term noise      |
| Δ% capped         | `pct_change().clip(-100,100)`               | Robust to extreme outliers    |
| Weather features  | Rolling temp & wind means                   | Exogenous covariates          |

- **Labels**: Shift price by –1 hour → `zone_label` columns  
- **Outputs**:  
  - `data/processed/features.csv`  
  - `data/processed/labels.csv`  

</details>

<details>
<summary>4 ▶ Exploratory Data Analysis (EDA)</summary>

- **Hour-of-day heatmap**: Peaks around 08:00 & 18:00  
- **Weekday vs Weekend**: Notably cheaper Sundays  
- **Correlation matrix**:  
  - NO1/NO2/NO5: strong positive correlations  
  - NO4: somewhat decorrelated  
- **Histogram of Δ%**: Fat tails justify capping at ±100%  
- **Notebooks**: All visuals live in `notebooks/02_eda.ipynb`.  

</details>

<details>
<summary>5 ▶ Baseline Models</summary>

| Model            | MAE (kr/MWh) | SMAPE (%) | Comment                  |
|------------------|--------------|-----------|--------------------------|
| Persistence      | 46.2         | 62.1      | Last-observation carried forward |
| SARIMA           | 41.8         | 55.0      | Seasonal ARIMA per zone  |
| RF baseline      | 30.5         | 40.3      | First ML benchmark       |

</details>

<details>
<summary>6 ▶ Hyper-tuning Random Forest</summary>

- **Parameters**:  
  - `n_estimators` ∈ {50, 100, 200}  
  - `max_depth` ∈ {None, 5, 10}  
  - `min_samples_leaf` ∈ {1, 5}  
- **Validation**: `TimeSeriesSplit(n_splits=3)`  
- **Result**:  
  - Best SMAPE ≈ 28% for NO1  
  - ≈30% reduction in SMAPE vs RF baseline  
- **Artifacts**: Models stored in `src/models/rf_tuned_models/`  

</details>

<details>
<summary>7 ▶ Streamlit Dashboard</summary>

- **Auto-refresh**: Uses `streamlit-autorefresh` to reload data hourly.  
- **Animated countdown**: CSS-driven progress bar updating each second.  
- **KPI Cards**: Next-hour forecast per zone (NOK/MWh).  
- **Choropleth map**: Altair + `zones.geojson` (no heavy GeoPandas).  
- **Historical plot**: Actual vs. predicted, with optional weather overlay.  
- **Feature importances**: Top-10 bar chart for the selected zone.  
- **CSV export**: One-click download of all forecasts.  
- **Screenshot**: `docs/dashboard_overview.png`  

</details>

<details>
<summary>8 ▶ Deployment</summary>

| Step             | Description                                                                      |
|------------------|----------------------------------------------------------------------------------|
| **GitHub**       | Push to `main` triggers CI (lint, tests)                                         |
| **Streamlit Cloud** | Reads `requirements.txt` and deploys `src/dashboard/app.py` automatically      |
| **URL**          | `https://<your-username>.streamlit.app`                                          |
| **Secrets**      | Configure Frost & HvaKosterStrømmen keys in Streamlit Cloud’s secrets manager.   |

</details>



Local Setup & Quick Start
bash
Copy
Edit
# 1. clone & install
git clone https://github.com/<you>/Electricity-forecast.git
cd Electricity-forecast
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. fetch & preprocess (first run)
python src/data/fetch_hvakoster.py
python src/data/frost_fetch.py
python src/data/preprocess.py
python src/features/build_features.py

# 3. train tuned RF
python src/models/train_tuned_rf_smape.py

# 4. run dashboard
streamlit run src/dashboard/app.py
> **Tip 🧠** All heavy scripts have `tqdm` progress bars – grab a coffee while weather chunks load.

---

## Model Benchmarks

| Zone | Persistence SMAPE | SARIMA SMAPE | RF Baseline | **RF Tuned** |
|------|-------------------|--------------|-------------|--------------|
| NO1  | 62.8 %            | 56.2 %       | 41.0 %      | **28.4 %**   |
| NO2  | 60.9 %            | 54.8 %       | 45.2 %      | **30.1 %**   |
| NO3  | 58.2 %            | 51.5 %       | 44.7 %      | **31.5 %**   |
| NO4  | 66.3 %            | 60.4 %       | 50.9 %      | **34.8 %**   |
| NO5  | 61.1 %            | 55.7 %       | 43.9 %      | **29.7 %**   |

> **Takeaway:** Tuned Random Forest outperforms all baselines across every zone.

---

## Scheduled Retraining (CRONS)

| Script                       | Frequency                  | What it does                                                                                   |
|------------------------------|----------------------------|-----------------------------------------------------------------------------------------------|
| `scripts/cron_fetch.sh`      | Hourly                     | Fetches latest prices & weather data, appends to `data/raw/`                                  |
| `scripts/retrain_weekly.sh`  | Weekly (Sunday 02:00)      | Rebuilds features, retrains tuned RF, commits & pushes to GitHub (auto-redeploy via cloud)   |

_Add these to your system crontab or configure a scheduled GitHub Actions workflow._

---

## CI/CD & Cloud Deploy

- **GitHub Actions**  
  - Lints code with `flake8`, runs unit tests and starts model retraining on schedule.  
- **Streamlit Community Cloud**  
  - Auto-builds from `requirements.txt` and deploys `src/dashboard/app.py`.  
- **Optional Docker**  
  - Include a `Dockerfile` for Heroku/Render if custom system libraries are needed.

---

## Roadmap

- [ ] Compare with **XGBoost** & **LightGBM**  
- [ ] Implement **probabilistic intervals** (Quantile RF)  
- [ ] Add **notification webhook** for threshold breaches  
- [ ] Store data in **InfluxDB** (time-series database)  
- [ ] Orchestrate via **Kubernetes cronjobs** & track experiments in **MLflow**

---


## Contributing

1. **Fork** the repository  
2. **Create a branch** for your feature or fix:  
   ```bash
   git checkout -b feat/my-feature
3. pre-commit install
4. feat(models): add LightGBM baseline
5. fix(dashboard): correct counting error
6. **Push** your branch and open a **Pull Request**.

## License

MIT © 2025 Sushant  

Feel free to use, fork, or cite – but do drop a ⭐ if you find it useful!  
