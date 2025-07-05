# src/models/train_persistence.py

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data(features_path: str, labels_path: str):
    """
    Load features and labels CSVs into DataFrames.
    Returns X (features) and y (labels) as DataFrames.
    """
    X = pd.read_csv(features_path, index_col=0, parse_dates=True)
    y = pd.read_csv(labels_path, index_col=0, parse_dates=True)
    return X, y


def persistence_predict(X: pd.DataFrame, zones: list) -> pd.DataFrame:
    """
    Predict next-hour price based on previous hour's actual price.
    X must contain the original zone price columns.
    Returns a DataFrame of predictions with same index as X.
    """
    preds = pd.DataFrame(index=X.index)
    for z in zones:
        preds[f"{z}_pred"] = X[z]
    return preds


def evaluate(y_true: pd.DataFrame, y_pred: pd.DataFrame, zones: list) -> dict:
    """
    Compute MAE, RMSE, and MAPE for each zone, avoiding division by zero.
    """
    results = {}
    for z in zones:
        true = y_true[f"{z}_label"].values
        pred = y_pred[f"{z}_pred"].values
        # Mask valid entries: true not NaN and non-zero
        mask = (~np.isnan(true)) & (np.abs(true) > 1e-6)
        mae = mean_absolute_error(true[mask], pred[mask]) if np.any(mask) else np.nan
        rmse = np.sqrt(mean_squared_error(true[mask], pred[mask])) if np.any(mask) else np.nan
        if np.any(mask):
            mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
        else:
            mape = np.nan
        results[z] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    return results


def main():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    features_path = os.path.join(BASE_DIR, "data", "processed", "features.csv")
    labels_path = os.path.join(BASE_DIR, "data", "processed", "labels.csv")

    print("Loading data...")
    X, y = load_data(features_path, labels_path)

    zones = [col.replace("_label", "") for col in y.columns]
    print(f"Zones: {zones}")

    print("Running persistence model...")
    y_pred = persistence_predict(X, zones)

    print("Evaluating...")
    results = evaluate(y, y_pred, zones)
    for z, metrics in results.items():
        print(f"Zone {z}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.2f}%")

if __name__ == "__main__":
    main()
