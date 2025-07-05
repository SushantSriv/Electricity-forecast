# src/models/train_arima.py

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Zone-specific SARIMA configurations from grid search
SARIMA_CONFIGS = {
    "NO1": {"order": (2, 0, 2), "seasonal_order": (1, 1, 1, 24)},
    "NO2": {"order": (2, 1, 2), "seasonal_order": (1, 1, 1, 24)},
    "NO3": {"order": (2, 0, 2), "seasonal_order": (1, 1, 1, 24)},
    "NO4": {"order": (2, 1, 2), "seasonal_order": (0, 1, 1, 24)},
    "NO5": {"order": (2, 0, 2), "seasonal_order": (1, 1, 1, 24)},
}


def load_series(zone: str, data_path: str) -> pd.Series:
    """
    Load the hourly price series for a given zone from clean data.
    """
    df = pd.read_csv(data_path, index_col=0)
    series = df[zone].copy()
    # Convert index to datetime, coerce to UTC, then remove tz for compatibility
    series.index = pd.to_datetime(series.index, utc=True)
    series.index = series.index.tz_convert(None)
    # Ensure regular hourly frequency on the series
    series = series.asfreq('h')
    return series


def train_test_split(series: pd.Series, test_days: int = 30) -> tuple:
    """
    Split the series into training and testing sets.
    The test set is the last `test_days` days.
    """
    steps = test_days * 24
    train = series.iloc[:-steps]
    test = series.iloc[-steps:]
    return train, test


def fit_sarima(train: pd.Series, order: tuple, seasonal_order: tuple):
    """
    Fit a SARIMA model with the specified order parameters.
    """
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)


def forecast_sarima(model_fit, steps: int, start_time: pd.Timestamp) -> pd.Series:
    """
    Forecast the next `steps` points, starting from `start_time`.
    """
    preds = model_fit.forecast(steps)
    index = pd.date_range(start=start_time, periods=steps, freq='h')
    return pd.Series(preds, index=index)


def evaluate(true: np.ndarray, pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, and MAPE between true and predicted arrays.
    """
    mask = (~np.isnan(true)) & (true != 0)
    mae = mean_absolute_error(true[mask], pred[mask]) if np.any(mask) else np.nan
    rmse = np.sqrt(mean_squared_error(true[mask], pred[mask])) if np.any(mask) else np.nan
    mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100 if np.any(mask) else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def main():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_path = os.path.join(BASE_DIR, "data", "interim", "clean_prices.csv")

    results = []
    for zone, cfg in SARIMA_CONFIGS.items():
        print(f"Processing zone {zone} with SARIMA{cfg['order']}x{cfg['seasonal_order']}")
        series = load_series(zone, data_path)
        train, test = train_test_split(series)
        model_fit = fit_sarima(train, cfg['order'], cfg['seasonal_order'])
        preds = forecast_sarima(model_fit, len(test), test.index[0])
        metrics = evaluate(test.values, preds.values)
        results.append({"zone": zone, **metrics})

    df_results = pd.DataFrame(results).set_index('zone')
    print("\nSARIMA Evaluation Results (with zone-specific configs):")
    print(df_results)

    out_path = os.path.join(BASE_DIR, "data", "processed", "sarima_zone_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_results.to_csv(out_path)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
