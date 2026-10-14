#!/usr/bin/env python3
"""Download hourly London weather (Open-Meteo archive, no API key) for the travel-time model.

Writes data/processed/weather_hourly_london.csv with the columns
scripts/train_travel_time_lfb.py joins on (London local hour):

  time, temp_c, precip_mm, wind_kph, rh_pct, is_rain, is_heavy_rain, is_cold, is_hot

One point (central London) for the whole city: weather is city-wide context,
not trip-local.

Run:  .venv/bin/python scripts/fetch_london_weather.py [--start 2021-01-01] [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "weather_hourly_london.csv"
URL = "https://archive-api.open-meteo.com/v1/archive"
LAT, LON = 51.5074, -0.1278


def fetch(start: str, end: str) -> pd.DataFrame:
    q = urllib.parse.urlencode(
        {
            "latitude": LAT,
            "longitude": LON,
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m",
            "wind_speed_unit": "kmh",
            "timezone": "Europe/London",
        }
    )
    with urllib.request.urlopen(f"{URL}?{q}", timeout=120) as r:
        h = json.load(r)["hourly"]
    w = pd.DataFrame(
        {
            "time": pd.to_datetime(h["time"]),
            "temp_c": h["temperature_2m"],
            "precip_mm": h["precipitation"],
            "wind_kph": h["wind_speed_10m"],
            "rh_pct": h["relative_humidity_2m"],
        }
    )
    w["is_rain"] = (w["precip_mm"] >= 0.1).astype(int)
    w["is_heavy_rain"] = (w["precip_mm"] >= 2.5).astype(int)
    w["is_cold"] = (w["temp_c"] <= 2).astype(int)
    w["is_hot"] = (w["temp_c"] >= 27).astype(int)
    return w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=(dt.date.today() - dt.timedelta(days=7)).isoformat())
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    w = fetch(args.start, args.end).dropna(subset=["temp_c"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(args.out, index=False)
    print(f"{len(w):,} hours {w['time'].min()} → {w['time'].max()} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
