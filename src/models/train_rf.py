# src/models/train_rf.py

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    # 1) Paths for features and labels
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    features_path = os.path.join(BASE_DIR, "data", "processed", "features.csv")
    labels_path   = os.path.join(BASE_DIR, "data", "processed", "labels.csv")

    # 2) Load data
    X = pd.read_csv(features_path, index_col=0)
    y = pd.read_csv(labels_path,   index_col=0)

    # 3) Split last month as test set
    test_size = 24 * 30  # 30 days
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # 4) Clean infinite values and fill missing
    X_train = X_train.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    X_test  = X_test.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # 5) Train and evaluate per zone
    zones = [col.replace('_label', '') for col in y_train.columns]
    results = {}
    for zone in zones:
        print(f"Training RF for {zone}...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train[f"{zone}_label"])
        pred = rf.predict(X_test)
        true = y_test[f"{zone}_label"].values
        # mask out NaNs and near-zero true values to avoid divide-by-zero
        mask = (~np.isnan(true)) & (np.abs(true) > 1e-6)
        mae  = mean_absolute_error(true[mask], pred[mask])
        rmse = np.sqrt(mean_squared_error(true[mask], pred[mask]))
        # Use MAPE; consider SMAPE to mitigate near-zero issues
        mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
        results[zone] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    # 6) Reporting
    df_results = pd.DataFrame(results).T
    print("\nRandom Forest Evaluation Results:")
    print(df_results)

    # 7) Save to CSV
    output_path = os.path.join(BASE_DIR, "data", "processed", "rf_zone_results.csv")
    df_results.to_csv(output_path)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
