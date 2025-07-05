#!/usr/bin/env python3
# src/data/frost_fetch.py

import os
import requests
import pandas as pd
from datetime import datetime, timedelta


def get_frost_token() -> str:
    """
    Henter et OAuth2 access token fra Frost API ved bruk av client credentials.
    """
    token_url = "https://frost.met.no/auth/accessToken"
    client_id     = "9dc1990c-2a2d-48e3-9a10-b813663a121d"
    client_secret = "b9132332-c29e-4a21-a4ca-2745546c071a"
    if not client_id or not client_secret:
        raise RuntimeError("Sett FROST_CLIENT_ID og FROST_CLIENT_SECRET som env-vars!")
    resp = requests.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        }
    )
    resp.raise_for_status()
    return resp.json().get("access_token")


def fetch_frost_data_chunk(parameter: str, source: str, start: str, end: str, token: str) -> pd.DataFrame:
    """
    Henter timeobservasjoner for én parameter i en angitt tidsperiode.
    Bruker 'timeresolutions' istedenfor 'resolution'.
    """
    url = "https://frost.met.no/observations/v0.jsonld"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "sources": source,
        "elements": parameter,
        "referencetime": f"{start}/{end}",
        "timeresolutions": "PT1H"
    }
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        print(f"ERROR fetching {parameter}: {resp.status_code} - {resp.text}")
        resp.raise_for_status()
    data = resp.json().get("data", [])
    records = [(obs["referenceTime"], obs["observations"][0]["value"]) for obs in data]
    df = pd.DataFrame(records, columns=["datetime", parameter])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Oslo")
    return df.set_index("datetime")


def main():
    # Sti for output
    BASE_DIR = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_csv = os.path.join(raw_dir, "weather.csv")

    # Hent access token
    token = get_frost_token()

    # Definer periode (siste 2 år)
    end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=365 * 2)
    delta = timedelta(days=90)

    temp_dfs = []
    wind_dfs = []
    cur = start
    while cur < end:
        nxt = min(cur + delta, end)
        s_iso = cur.strftime("%Y-%m-%dT%H:%M:%SZ")
        e_iso = nxt.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"Henter air_temperature fra {s_iso} til {e_iso}")
        temp_dfs.append(fetch_frost_data_chunk("air_temperature", "SN18700", s_iso, e_iso, token))
        print(f"Henter wind_speed     fra {s_iso} til {e_iso}")
        wind_dfs.append(fetch_frost_data_chunk("wind_speed", "SN18700", s_iso, e_iso, token))
        cur = nxt

    df_temp = pd.concat(temp_dfs).sort_index().drop_duplicates()
    df_wind = pd.concat(wind_dfs).sort_index().drop_duplicates()

    # Slå sammen temperatur og vind
    df = pd.concat([
        df_temp.rename(columns={"air_temperature": "temperature"}),
        df_wind.rename(columns={"wind_speed": "wind"})
    ], axis=1)

    df.to_csv(out_csv)
    print(f"Værdata lagret til {out_csv}")


if __name__ == "__main__":
    main()
