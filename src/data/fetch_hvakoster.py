# src/data/fetch_hvakoster.py

import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import certifi

# Konfigurasjon
BASE_URL = "https://www.hvakosterstrommen.no/api/v1/prices"
ZONES    = ["NO1", "NO2", "NO3", "NO4", "NO5"]
START    = datetime(2024, 7, 1)
END      = datetime(2025, 6, 30)

# Bygg liste over alle datoer i intervallet
dates = [START + timedelta(days=i) for i in range((END - START).days + 1)]

records = []

# Ytre loop med progress bar for dager
for current in tqdm(dates, desc="Downloading days", unit="day"):
    year     = current.year
    date_str = current.strftime("%m-%d")

    # Indre loop med progress bar for soner
    for zone in tqdm(ZONES, desc=f"  Zones on {current.date()}", leave=False, unit="zone"):
        url = f"{BASE_URL}/{year}/{date_str}_{zone}.json"
        resp = requests.get(url, verify=False)
        resp.raise_for_status()
        for entry in resp.json():
            records.append({
                "datetime":  entry["time_start"],
                "zone":      zone,
                "price_NOK": entry["NOK_per_kWh"]
            })

# Bygg DataFrame
df = pd.DataFrame(records)
df["datetime"] = pd.to_datetime(df["datetime"])
df.set_index("datetime", inplace=True)
df.sort_index(inplace=True)

# Pivot til wide-format: én kolonne per sone
df_pivot = df.pivot_table(values="price_NOK", index=df.index, columns="zone")

# Beregn prosentvis fremgang
df_pct_hour = df_pivot.pct_change(periods=1) * 100
df_pct_hour.columns = [f"{col}_pct_hourly" for col in df_pct_hour.columns]

df_pct_daily = df_pivot.pct_change(periods=24) * 100
df_pct_daily.columns = [f"{col}_pct_daily" for col in df_pct_daily.columns]

# Slå sammen originaldata og prosent-kolonner
df_full = pd.concat([df_pivot, df_pct_hour, df_pct_daily], axis=1)

# Eksporter til CSV
output_path = "data/raw/all_zones_hourly_prices_with_pct.csv"
df_full.to_csv(output_path)

print(f"\nDone! Data lagret til {output_path}")
print(df_full.head())
