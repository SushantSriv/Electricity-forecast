#!/usr/bin/env python3
# src/features/build_features.py

import os
import pandas as pd
import numpy as np
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, GoodFriday, EasterMonday


def load_clean_data(path: str) -> pd.DataFrame:
    """
    Load the preprocessed clean prices CSV.
    The first column is the datetime index.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df.index.name = "datetime"
    df.index = pd.to_datetime(df.index, utc=True)
    try:
        df.index = df.index.tz_convert("Europe/Oslo")
    except Exception:
        pass
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df.index.hour
    df["weekday"] = df.index.dayofweek
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df


def add_pct_changes(df: pd.DataFrame, zones: list) -> pd.DataFrame:
    for zone in zones:
        df[f"{zone}_pct_hourly"] = df[zone].pct_change().replace([np.inf, -np.inf], np.nan)
    return df


def add_rolling_means(df: pd.DataFrame, zones: list, windows: list) -> pd.DataFrame:
    for zone in zones:
        for w in windows:
            df[f"{zone}_ma_{w}h"] = df[zone].rolling(window=w, min_periods=1).mean()
    return df


def cap_extreme_pct(df: pd.DataFrame, zones: list, cap: float = 100.0) -> pd.DataFrame:
    for zone in zones:
        col = f"{zone}_pct_hourly"
        if col in df.columns:
            df[col] = df[col].clip(lower=-cap, upper=cap)
    return df


class NorwayHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        GoodFriday,
        EasterMonday,
        Holiday("Labour Day", month=5, day=1, observance=nearest_workday),
        Holiday("Constitution Day", month=5, day=17, observance=nearest_workday),
        # Additional fixed-date holidays
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
        Holiday("Second Christmas Day", month=12, day=26, observance=nearest_workday)
    ]


def add_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    cal = NorwayHolidayCalendar()
    holidays = cal.holidays(start=df.index.min().date(), end=df.index.max().date())
    df["is_holiday"] = df.index.normalize().isin(holidays).astype(int)
    return df


def load_weather_data(path: str) -> pd.DataFrame:
    w = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    w.index = pd.to_datetime(w.index, utc=True)
    try:
        w.index = w.index.tz_convert("Europe/Oslo")
    except Exception:
        pass
    w = w.rename(columns={"temp": "temperature", "wind_speed": "wind"})
    return w


def add_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame, windows: list) -> pd.DataFrame:
    df = df.join(weather_df, how="left")
    df["temperature"] = df["temperature"].interpolate().ffill().bfill()
    df["wind"] = df["wind"].interpolate().ffill().bfill()
    for w in windows:
        df[f"temp_ma_{w}h"] = df["temperature"].rolling(window=w, min_periods=1).mean()
        df[f"wind_ma_{w}h"] = df["wind"].rolling(window=w, min_periods=1).mean()
    return df


def build_features():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_path = os.path.join(BASE_DIR, "data", "interim", "clean_prices.csv")
    weather_path = os.path.join(BASE_DIR, "data", "raw", "weather.csv")
    out_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load price data
    df = load_clean_data(raw_path)
    zones = [c for c in df.columns if c.startswith("NO")]

    # 2) Calendar features
    df = add_calendar_features(df)

    # 3) Percent changes
    df = add_pct_changes(df, zones)

    # 4) Rolling means on price
    df = add_rolling_means(df, zones, windows=[3, 6, 24])

    # 5) Cap extreme % changes
    df = cap_extreme_pct(df, zones, cap=100.0)

    # 6) Holiday flag
    df = add_holiday_feature(df)

    # 7) Weather features
    if os.path.exists(weather_path):
        weather_df = load_weather_data(weather_path)
        df = add_weather_features(df, weather_df, windows=[3, 6, 24])
    else:
        print(f"Warning: {weather_path} ikke funnet. Hopper over weather features.")

    # 8) Prepare next-hour labels
    labels = df[zones].shift(-1).add_suffix("_label")

    # 9) Align and drop last row (no label)
    features = df.iloc[:-1].copy()
    labels = labels.iloc[:-1].copy()

    # 10) Save
    features_path = os.path.join(out_dir, "features.csv")
    labels_path = os.path.join(out_dir, "labels.csv")
    features.to_csv(features_path)
    labels.to_csv(labels_path)

    print(f"Features saved to {features_path}")
    print(f"Labels saved to {labels_path}")


if __name__ == "__main__":
    build_features()
