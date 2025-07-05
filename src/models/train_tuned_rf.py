#!/usr/bin/env python3
# src/models/train_tuned_rf.py

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def main():
    # 1) Paths
    BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    features_path = os.path.join(processed_dir, "features.csv")
    labels_path   = os.path.join(processed_dir, "labels.csv")
    grid_path     = os.path.join(processed_dir, "rf_grid_search_zone_results.csv")
    output_path   = os.path.join(processed_dir, "rf_tuned_final_results.csv")
    
    # Folder for serialized models
    model_dir = os.path.join(BASE_DIR, "src", "models", "rf_tuned_models")
    os.makedirs(model_dir, exist_ok=True)

    # 2) Load data
    X = pd.read_csv(features_path, index_col=0)
    y = pd.read_csv(labels_path,   index_col=0)

    # 3) Split off last month as test set
    test_size = 24 * 30  # last 30 days
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # 4) Clean infinities and fill missing
    for df in (X_train, X_test):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)

    # 5) Load tuned hyperparameters
    grid = pd.read_csv(grid_path, index_col="zone")

    # 6) Train, evaluate and serialize per zone
    results = {}
    for zone, params in grid.iterrows():
        print(f"[{zone}] Training with params: n_estimators={params.n_estimators}, "
              f"max_depth={params.max_depth}, min_samples_leaf={params.min_samples_leaf}")

        # Parse params
        n_est = int(params.n_estimators)
        max_d = None if pd.isna(params.max_depth) else int(params.max_depth)
        min_leaf = int(params.min_samples_leaf)

        # Instantiate & train
        rf = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_leaf=min_leaf,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train[f"{zone}_label"])

        # Predict & evaluate
        preds = rf.predict(X_test)
        true  = y_test[f"{zone}_label"].values
        # Avoid divide-by-zero in MAPE
        mask = (~np.isnan(true)) & (np.abs(true) > 1e-6)

        mae  = mean_absolute_error(true[mask], preds[mask])
        rmse = np.sqrt(mean_squared_error(true[mask], preds[mask]))
        mape = (np.mean(np.abs((true[mask] - preds[mask]) / true[mask])) * 100)

        results[zone] = {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}

        # Serialize model
        model_file = os.path.join(model_dir, f"rf_tuned_{zone}.joblib")
        joblib.dump(rf, model_file)
        print(f"[{zone}] Model saved to {model_file}\n")

    # 7) Save final results
    df_final = pd.DataFrame(results).T
    df_final.to_csv(output_path)
    print("All done!")
    print(f"Final evaluation metrics written to {output_path}")

if __name__ == "__main__":
    main()
