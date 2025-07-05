# src/models/grid_search_sarima.py

import os
import itertools
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress convergence warnings for clarity
warnings.filterwarnings("ignore")


def load_series(zone: str, data_path: str) -> pd.Series:
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    series = df[zone].copy()
    # Ensure no timezone
    series.index = pd.to_datetime(series.index, utc=True).tz_convert(None)
    # Ensure hourly frequency
    series = series.asfreq('H')
    return series


def train_test_split(series: pd.Series, test_size: int = 24*30) -> tuple:
    """
    Split the series into train and test sets. Default last 30 days as test.
    """
    train = series.iloc[:-test_size]
    test  = series.iloc[-test_size:]
    return train, test


def grid_search(zone: str, data_path: str, 
                ps=range(0,3), ds=[0,1], qs=range(0,3), 
                Ps=range(0,2), Ds=[1], Qs=range(0,2), m=24):
    """
    Perform grid search over SARIMA orders and seasonal orders for a given zone.
    Returns best config by AIC.
    """
    series = load_series(zone, data_path)
    train, test = train_test_split(series)

    best_aic = np.inf
    best_cfg = None

    for p, d, q in itertools.product(ps, ds, qs):
        for P, D, Q in itertools.product(Ps, Ds, Qs):
            try:
                model = SARIMAX(
                    train,
                    order=(p,d,q),
                    seasonal_order=(P,D,Q,m),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res = model.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_cfg = (p, d, q, P, D, Q, m)
                    print(f"New best AIC={best_aic:.2f} with config {best_cfg}")
            except Exception:
                continue

    return best_cfg, best_aic


def main():
    # Settings
    zones = ["NO1","NO2","NO3","NO4","NO5"]
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "interim", "clean_prices.csv")
    )

    results = {}
    for zone in zones:
        print(f"\nGrid searching for zone {zone}…")
        cfg, aic = grid_search(zone, data_path)
        results[zone] = {"best_config": cfg, "best_aic": aic}

    # Output summary
    df = pd.DataFrame(results).T
    print("\nSummary of best SARIMA configurations:")
    print(df)

    # Save to CSV
    out_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "sarima_grid_search.csv")
    )
    df.to_csv(out_path)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
