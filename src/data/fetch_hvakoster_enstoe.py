import os, requests, pandas as pd, datetime as dt, certifi, xml.etree.ElementTree as ET

BASE_URL  = "https://web-api.tp.entsoe.eu/api"
TOKEN     = "5f1fc58c-d40b-4840-8794-deb783310c3d"

ZONES = {
    "NO1": "10YNO-1--------2",
    "NO2": "10YNO-2--------T",
    "NO3": "10YNO-3--------J",
    "NO4": "10YNO-4--------9",
    "NO5": "10Y1001A1001A48H",
}

START_UTC = dt.datetime(2024, 7, 1, 0, 0, tzinfo=dt.timezone.utc)
END_UTC   = dt.datetime(2025, 6, 30, 23, 0, tzinfo=dt.timezone.utc)

def fmt(ts):                      # ENTSO-E expects UTC, yyyymmddHHMM
    return ts.strftime("%Y%m%d%H%M")

rows = []
for zone, eic in ZONES.items():
    params = {
        "securityToken": TOKEN,
        "documentType":  "A44",
        "in_Domain":     eic,
        "out_Domain":    eic,
        "periodStart":   fmt(START_UTC),
        "periodEnd":     fmt(END_UTC),
    }
    r = requests.get(BASE_URL, params=params, timeout=60, verify=False)
    r.raise_for_status()

    ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0"}
    root = ET.fromstring(r.content)

    for ts in root.findall(".//ns:TimeSeries", ns):
        for p in ts.findall(".//ns:Point", ns):
            position   = int(p.find("ns:position", ns).text)
            price_eur  = float(p.find("ns:price.amount", ns).text)
            # build timestamp: each position is one hour from period start
            period_start = dt.datetime.fromisoformat(
                ts.find(".//ns:timeInterval/ns:start", ns).text.replace("Z", "+00:00")
            )
            stamp = period_start + dt.timedelta(hours=position-1)
            rows.append({"datetime": stamp, "zone": zone, "EUR_per_MWh": price_eur})

df = pd.DataFrame(rows).set_index("datetime").sort_index()
df = df.pivot_table(values="EUR_per_MWh", index=df.index, columns="zone")
df.to_csv("data/raw/all_zones_hourly_prices.csv")
print(df.head())
