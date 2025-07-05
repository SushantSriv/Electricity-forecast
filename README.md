# ⚡ Electricity-Forecast

> **A reproducible end-to-end pipeline for hourly electricity-price prediction in Norway (NO1–NO5) – from raw API calls, through feature engineering & model tuning, to a live Streamlit dashboard.**

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
| MVP  | Historical + day-ahead prices → enriched with weather & calendar features → baseline + tuned ML models → dashboard. |

---

## Key Highlights
| Category       | What we did                                                        | Why it matters |
|----------------|--------------------------------------------------------------------|----------------|
| **Data**       | 365×24 h of prices (HvaKosterStrømmen) + 2 y of weather (Frost API) | Solid history for training & covariates |
| **Features**   | Calendar, rolling means (3/6/24 h), capped %-changes, weather, holidays | Captures seasonality & exogenous effects |
| **Baselines**  | Persistence, SARIMA, Random Forest                                 | Establishes performance floor |
| **Hyper-tuning** | GridSearchCV on RF (n_estimators, max_depth, min_samples_leaf)    | ↓ SMAPE ≈ 30 % vs baseline RF |
| **Dashboard**  | KPI cards, countdown, auto-refresh, choropleth, CSV export, importances | Analyst & operations cockpit |
| **Ops**        | Cron fetch, weekly retrain, GitHub → Streamlit Cloud CI/CD          | Zero-maintenance operation |

---

## End-to-End Architecture

```
            ┌────────────┐
            │  Cron Jobs │  (cron_fetch.sh • retrain_weekly.sh)
            └──────┬─────┘
                   ▼
┌───────────────────────────────────────────────────────────────────┐
│                           Data Layer                            │
│ fetch_hvakoster.py & frost_fetch.py → data/raw/                 │
│ preprocess.py → data/interim/clean_prices.csv                   │
│ build_features.py → data/processed/{features.csv, labels.csv}   │
└───────────────────────────────────────────────────────────────────┘
                   ▼
┌───────────────────────────────────────────────────────────────────┐
│                          Model Layer                            │
│ Persistence · SARIMA · RF baseline · RF tuned                    │
│ Models in src/models/rf_tuned_models/*.joblib                   │
└───────────────────────────────────────────────────────────────────┘
                   ▼
┌───────────────────────────────────────────────────────────────────┐
│                         Dashboard Layer                         │
│ Streamlit app (src/dashboard/app.py): KPI, countdown, maps, CSV  │
└───────────────────────────────────────────────────────────────────┘
```

---

## Directory Layout
<details>
<summary>Click to expand</summary>

```text
Electricity-forecast/
├── .streamlit/               # Streamlit theme settings
│   └── config.toml
├── config/                   # API keys (git-ignored)
│   └── api_keys.yml
├── data/
│   ├── raw/                  # Original downloads & zones.geojson
│   ├── interim/              # clean_prices.csv
│   └── processed/            # features.csv · labels.csv · metrics
├── notebooks/                # Exploratory analysis notebooks
├── src/
│   ├── data/                 # fetch & preprocess scripts
│   ├── features/             # build_features.py
│   ├── models/               # training scripts & rf_tuned_models/
│   └── dashboard/            # Streamlit app.py
├── models/                   # Serialized model artifacts (optional)
├── scripts/                  # cron_fetch.sh · retrain_weekly.sh
├── requirements.txt
└── README.md
```

</details>

---

## Workflow in Detail
<details>
<summary>1 ▶ Data Ingestion</summary>

| Script                | Purpose                                                                                   |
|-----------------------|-------------------------------------------------------------------------------------------|
| `fetch_hvakoster.py`  | Loop date-by-date, call HvaKosterStrømmen API for NO1–NO5. Saves `all_zones_YYYY-MM-DD.csv` to `data/raw/`. |
| `frost_fetch.py`      | OAuth2 to Frost API, fetch hourly temperature & wind (station SN18700) in 90-day chunks. Saves `weather.csv`. |

</details>

<details>
<summary>2 ▶ Preprocessing</summary>

- Merge raw CSVs, convert UTC → `Europe/Oslo`  
- Re-index to a continuous hourly `DatetimeIndex`  
- Linear interpolation for small gaps; drop negative prices  
- **Output:** `data/interim/clean_prices.csv`

</details>

<details>
<summary>3 ▶ Feature Engineering</summary>

| Feature           | Implementation                                    | Comment                       |
|-------------------|---------------------------------------------------|-------------------------------|
| Calendar hour     | `df.index.hour`                                   | Intraday pattern             |
| Weekday (0–6)     | `df.index.dayofweek`                              | Weekly cycle                 |
| Weekend flag      | `weekday.isin([5,6])`                             | Binary weekend indicator     |
| Norway holidays   | Custom `pandas.tseries.holiday` calendar          | Flags holiday effects        |
| Rolling means     | `df[col].rolling(window).mean()` (3h,6h,24h)      | Smoothing short-term noise   |
| Δ% capped         | `pct_change().clip(-100,100)`                     | Robust to extreme outliers   |
| Weather means     | Rolling temp & wind averages                      | Exogenous variables          |

- **Labels:** Next-hour price per zone (`shift(-1)`)  
- **Outputs:** `data/processed/features.csv`, `data/processed/labels.csv`

</details>

<details>
<summary>4 ▶ Exploratory Data Analysis (EDA)</summary>

- Hour-of-day heatmap (peaks ~08:00 & 18:00)  
- Weekday vs. weekend comparison (cheaper Sundays)  
- Correlation matrix (NO1/NO2/NO5 correlated; NO4 more independent)  
- Histogram of Δ% changes (fat tails justify capping)  
- **Notebook:** `notebooks/02_eda.ipynb`

</details>

<details>
<summary>5 ▶ Baseline Models</summary>

| Model           | MAE (kr/MWh) | SMAPE (%) | Comment                      |
|-----------------|--------------|-----------|------------------------------|
| Persistence     | 46.2         | 62.1      | Last-observation carried forward |
| SARIMA          | 41.8         | 55.0      | Seasonal ARIMA per zone      |
| RF baseline     | 30.5         | 40.3      | First ML benchmark           |

</details>

<details>
<summary>6 ▶ Hyper-tuning Random Forest</summary>

- **Grid:** `n_estimators` {50,100,200}, `max_depth` {None,5,10}, `min_samples_leaf` {1,5}  
- **Validation:** `TimeSeriesSplit(n_splits=3)`  
- **Result:**  
  - Best SMAPE ~28% (NO1)  
  - ~30% improvement vs baseline RF  
- **Artifacts:** `src/models/rf_tuned_models/*.joblib`

</details>

<details>
<summary>7 ▶ Streamlit Dashboard</summary>

- **Auto-refresh:** Hourly via `streamlit-autorefresh`  
- **Countdown:** CSS progress bar updating each second  
- **KPI Cards:** Next-hour forecasts per zone  
- **Choropleth:** Altair + `zones.geojson` (no GeoPandas)  
- **Historical plot:** Actual vs predicted + optional weather overlay  
- **Feature importances:** Top-10 bar chart  
- **CSV export:** One-click download  
- **Screenshot:** `docs/dashboard_overview.png`

</details>

<details>
<summary>8 ▶ Deployment</summary>

| Step               | Description                                                             |
|--------------------|-------------------------------------------------------------------------|
| **GitHub**         | Push to `main` triggers CI (lint, tests)                                |
| **Streamlit Cloud**| Builds from `requirements.txt`, deploys `src/dashboard/app.py`          |
| **URL**            | `https://<your-username>.streamlit.app`                                 |
| **Secrets**        | Configure API keys in Cloud secret manager                              |

</details>

---

## Local Setup & Quick Start

```bash
# 1. Clone & install
git clone https://github.com/<your-username>/Electricity-forecast.git
cd Electricity-forecast
python -m venv venv && source venv/bin/activate  # Windows: venv\Scriptsctivate.bat
pip install --upgrade pip
pip install -r requirements.txt

# 2. Fetch & preprocess
python src/data/fetch_hvakoster.py
python src/data/frost_fetch.py
python src/data/preprocess.py
python src/features/build_features.py

# 3. Train tuned Random Forest
python src/models/train_tuned_rf_smape.py

# 4. Run dashboard
streamlit run src/dashboard/app.py
```

> **Tip 🧠** Heavy scripts use `tqdm` – feel free to grab a coffee while they run.

---

## Model Benchmarks

| Zone | Persistence | SARIMA | RF Baseline | **RF Tuned** |
|------|-------------|--------|-------------|--------------|
| NO1  | 62.8 %      | 56.2 % | 41.0 %      | **28.4 %**   |
| NO2  | 60.9 %      | 54.8 % | 45.2 %      | **30.1 %**   |
| NO3  | 58.2 %      | 51.5 % | 44.7 %      | **31.5 %**   |
| NO4  | 66.3 %      | 60.4 % | 50.9 %      | **34.8 %**   |
| NO5  | 61.1 %      | 55.7 % | 43.9 %      | **29.7 %**   |

---

## Scheduled Retraining (CRONS)

| Script                      | Frequency               | Action                                                                 |
|-----------------------------|-------------------------|------------------------------------------------------------------------|
| `scripts/cron_fetch.sh`     | Hourly                  | Fetch latest prices & weather, append to `data/raw/`                   |
| `scripts/retrain_weekly.sh` | Weekly (Sunday 02:00)   | Rebuild features, retrain RF, push to GitHub → auto-redeploy dashboard |

---

## CI/CD & Cloud Deploy

- **GitHub Actions:** Lint (flake8) + unit tests on each push/PR  
- **Streamlit Cloud:** Auto-deploy from `main` using `requirements.txt`  
- **Optional Docker:** Provide a `Dockerfile` for Heroku/Render  

---

## Roadmap

- [ ] XGBoost & LightGBM comparison  
- [ ] Probabilistic intervals (Quantile RF)  
- [ ] Notification webhooks on threshold breaches  
- [ ] InfluxDB time-series backend  
- [ ] Kubernetes cronjobs & MLflow tracking  

---

## Contributing

1. **Fork** the repository.  
2. **Create a branch** for your feature or fix:  
   ```bash
   git checkout -b feat/my-feature
   ```
3. **Install pre-committee hooks** for formatting & linting:  
   ```bash
   pre-commit install
   ```
4. **Commit** your changes using Conventional Commits, e.g.:  
   ```
   feat(models): add LightGBM baseline
   fix(dashboard): correct countdown styling
   ```
5. **Push** your branch and open a **Pull Request**.  
   - CI (lint & tests) will run automatically.  
   - We’ll review and merge upon approval.  

---

## License

MIT © 2025 Sushant

Feel free to use, fork, or cite – but do drop a ⭐ if you find it useful!
