#!/usr/bin/env python3
# src/models/train_tuned_rf_smape.py

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE) in %.
    """
    eps = 1e-6
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)


def main():
    # Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    proc_dir = os.path.join(BASE_DIR, "data", "processed")
    features_path = os.path.join(proc_dir, "features.csv")
    labels_path = os.path.join(proc_dir, "labels.csv")
    grid_path = os.path.join(proc_dir, "rf_grid_search_zone_results.csv")
    output_csv = os.path.join(proc_dir, "rf_tuned_final_results_smape.csv")

    model_dir = os.path.join(BASE_DIR, "src", "models", "rf_tuned_models")
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    X = pd.read_csv(features_path, index_col=0)
    y = pd.read_csv(labels_path, index_col=0)

    # Split last month as test
    test_size = 24 * 30
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # Clean
    for df in (X_train, X_test):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)

    # Load tuned params
    grid = pd.read_csv(grid_path, index_col="zone")

    results = {}
    for zone, params in grid.iterrows():
        print(f"Training zone {zone} with params: {params.to_dict()}")
        n_est = int(params.n_estimators)
        max_d = None if pd.isna(params.max_depth) else int(params.max_depth)
        min_leaf = int(params.min_samples_leaf)

        rf = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_leaf=min_leaf,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train[f"{zone}_label"])

        preds = rf.predict(X_test)
        true = y_test[f"{zone}_label"].values
        mask = ~np.isnan(true)

        mae = mean_absolute_error(true[mask], preds[mask])
        rmse = np.sqrt(mean_squared_error(true[mask], preds[mask]))
        smape_val = smape(true[mask], preds[mask])

        results[zone] = {"MAE": mae, "RMSE": rmse, "SMAPE_%": smape_val}

        # Save model
        model_file = os.path.join(model_dir, f"rf_tuned_{zone}.joblib")
        joblib.dump(rf, model_file)
        print(f"Saved model for zone {zone} to {model_file}")

    # Write results
    df_final = pd.DataFrame(results).T
    df_final.to_csv(output_csv)
    print(f"SMAPE results saved to {output_csv}")

if __name__ == "__main__":
    main()
