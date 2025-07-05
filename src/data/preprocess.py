# src/data/preprocess.py

import os
import pandas as pd

# 1. Sørg for at interim-mappen finnes
interim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "interim"))
os.makedirs(interim_dir, exist_ok=True)

# 2. Last inn rådata
raw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "all_zones_hourly_prices_with_pct.csv"))
df = pd.read_csv(raw_path, index_col="datetime")

# 3. Tving indeksen til DatetimeIndex med UTC
df.index = pd.to_datetime(df.index, utc=True)

# 4. Konverter til Oslo-tid
df.index = df.index.tz_convert("Europe/Oslo")

# 5. Reindekser for å fylle alle timer
full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h", tz="Europe/Oslo")
df = df.reindex(full_idx)

# 6. Interpoler tidsbasert og klipp negative verdier til null
df = df.interpolate(method="time").clip(lower=0)

# 7. Skriv ut clean data
out_path = os.path.join(interim_dir, "clean_prices.csv")
df.to_csv(out_path)
print(f"Preprocessing ferdig! Fil lagret til {out_path}")
