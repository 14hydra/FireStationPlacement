#!/usr/bin/env python3
"""
"A day in the life" replay: every LFB engine movement on one real day, animated.

For each engine mobilisation on the chosen day (LFB mobilisation records):
  mobilised → turnout (TurnoutTimeSeconds) → drives out along the shortest legal road
  route (one-way streets respected) for TravelTimeSeconds → on scene until
  DateAndTimeLeft → drives back to its home station (return drive assumed to take the
  same time as the drive out; LFB leaves DateAndTimeReturned empty).
Routes are the shortest legal routes on the OSM drive graph (firestation.road_distance),
not the exact streets driven. Engines LFB recorded as held up by traffic, roadworks or
traffic calming are flagged. The "road speed" gauge is the median engine speed in that
hour of the day across all 2025 trips (data/processed/lfb_travel_training.csv).

Writes data/figures/data_exploration/station_busyness/live_day.html (deck.gl; open in a browser).

Run:  .venv/bin/python scripts/build_live_day.py [--date 2025-01-05]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "data" / "raw" / "london"
OUT = ROOT / "data" / "figures" / "data_exploration" / "station_busyness" / "live_day.html"
TEMPLATE = Path(__file__).with_name("live_day_template.html")
TRAFFIC_DELAYS = {"Traffic, roadworks, etc": "traffic", "Traffic calming measures": "calming"}


def _time(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    return t.fillna(pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce"))


def _num(x, default: float) -> float:
    v = pd.to_numeric(x, errors="coerce")
    return float(v) if pd.notna(v) else default


def load_day(date: pd.Timestamp) -> pd.DataFrame:
    f = "lfb_mobilisations_2025.csv" if date.year >= 2025 else "lfb_mobilisations_2021_2024.csv"
    m = pd.read_csv(RAW / f, encoding="utf-8-sig", low_memory=False, dtype={"IncidentNumber": str})
    m["mob"] = _time(m["DateAndTimeMobilised"])
    m = m[m["mob"].dt.normalize() == date].copy()
    m["left"] = _time(m["DateAndTimeLeft"])
    return m


def load_incidents(numbers: set[str]) -> pd.DataFrame:
    from pyproj import Transformer

    cols = ["IncidentNumber", "IncidentGroup", "PropertyType", "IncGeo_BoroughName",
            "Easting_m", "Northing_m", "Easting_rounded", "Northing_rounded"]
    parts = [pd.read_csv(RAW / f, usecols=cols, dtype={"IncidentNumber": str}, low_memory=False)
             for f in ["lfb_incidents_2024_onwards.csv", "lfb_incidents_2018_2023.csv"]]
    inc = pd.concat(parts).drop_duplicates("IncidentNumber")
    inc = inc[inc["IncidentNumber"].isin(numbers)].copy()
    e = pd.to_numeric(inc["Easting_m"], errors="coerce").fillna(inc["Easting_rounded"])
    n = pd.to_numeric(inc["Northing_m"], errors="coerce").fillna(inc["Northing_rounded"])
    inc["lon"], inc["lat"] = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True).transform(
        e.to_numpy(), n.to_numpy())
    return inc.set_index("IncidentNumber")


def hourly_speed_kph() -> list[float]:
    d = pd.read_csv(ROOT / "data" / "processed" / "lfb_travel_training.csv",
                    usecols=["cal_year", "hour_of_call", "road_km", "drive_s"])
    d = d[(d["cal_year"] == 2025) & (d["road_km"] >= 1)]
    return (d["road_km"] / d["drive_s"] * 3600).groupby(d["hour_of_call"]).median().round(1).tolist()


class PathFinder:
    """Shortest legal road paths (node coordinates) from / to stations, cached per station."""

    def __init__(self):
        from firestation.road_distance import RoadRouter, load_or_build_graph

        self.r = RoadRouter(load_or_build_graph())
        self.csr_t = self.r.csr.T.tocsr()
        self._out, self._back = {}, {}

    def _pred(self, cache, csr, node):
        from scipy.sparse.csgraph import dijkstra

        if node not in cache:
            cache[node] = dijkstra(csr, directed=True, indices=node, return_predecessors=True)[1]
        return cache[node]

    def _chain(self, pred, start, stop):
        out, n = [start], start
        while n != stop and pred[n] >= 0:
            n = pred[n]
            out.append(n)
        return out

    def route(self, s_lat, s_lon, d_lat, d_lon, back: bool) -> np.ndarray:
        (s,), _ = self.r.snap([s_lat], [s_lon])
        (d,), _ = self.r.snap([d_lat], [d_lon])
        if back:  # incident -> station: tree on the reversed graph rooted at the station
            nodes = self._chain(self._pred(self._back, self.csr_t, s), d, s)
            pts = [(d_lon, d_lat)] + [(self.r.node_lon[n], self.r.node_lat[n]) for n in nodes] + [(s_lon, s_lat)]
        else:
            nodes = self._chain(self._pred(self._out, self.r.csr, s), d, s)[::-1]
            pts = [(s_lon, s_lat)] + [(self.r.node_lon[n], self.r.node_lat[n]) for n in nodes] + [(d_lon, d_lat)]
        return np.array(pts)


def timed(path: np.ndarray, t0: float, t1: float) -> tuple[list, list]:
    seg = np.hypot(np.diff(path[:, 0]) * np.cos(np.radians(51.5)), np.diff(path[:, 1]))
    cum = np.r_[0, np.cumsum(seg)]
    frac = cum / cum[-1] if cum[-1] > 0 else np.linspace(0, 1, len(path))
    ts = t0 + frac * (t1 - t0)
    return np.round(path, 5).tolist(), np.round(ts, 1).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2025-01-05")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    date = pd.Timestamp(args.date)

    m = load_day(date)
    stations = pd.read_csv(RAW / "london_fire_stations.csv").set_index("name")[["lat", "lon"]]
    home = m[m["DeployedFromLocation"] == "Home Station"]
    code_to_station = dict(zip(home["DeployedFromStation_Code"], home["DeployedFromStation_Name"]))
    m["home_station"] = m["Resource_Code"].str[:3].map(code_to_station).fillna(m["DeployedFromStation_Name"])
    m = m[m["DeployedFromStation_Name"].isin(stations.index) & m["home_station"].isin(stations.index)]
    inc = load_incidents(set(m["IncidentNumber"]))
    m = m[m["IncidentNumber"].isin(inc.index)]
    print(f"{date.date()}: {len(m):,} engine mobilisations to {m['IncidentNumber'].nunique():,} incidents")

    pf = PathFinder()
    midnight = date
    trips, engines = [], {}
    for _, r in m.sort_values("mob").iterrows():
        i = inc.loc[r["IncidentNumber"]]
        st = stations.loc[r["DeployedFromStation_Name"]]
        hm = stations.loc[r["home_station"]]
        t_mob = (r["mob"] - midnight).total_seconds()
        turnout = _num(r["TurnoutTimeSeconds"], 60)
        travel = _num(r["TravelTimeSeconds"], 300)
        t_go = t_mob + max(turnout, 0)
        t_arr = t_go + max(travel, 30)
        t_left = (r["left"] - midnight).total_seconds() if pd.notna(r["left"]) else t_arr + 20 * 60
        t_left = max(t_left, t_arr + 60)
        t_home = t_left + max(travel, 30)
        out_p, out_t = timed(pf.route(st.lat, st.lon, i.lat, i.lon, back=False), t_go, t_arr)
        back_p, back_t = timed(pf.route(hm.lat, hm.lon, i.lat, i.lon, back=True), t_left, t_home)
        trips.append({
            "id": r["Resource_Code"], "home": r["home_station"], "from": r["DeployedFromStation_Name"],
            "inc": r["IncidentNumber"], "group": i["IncidentGroup"], "prop": str(i["PropertyType"]).strip(),
            "borough": str(i["IncGeo_BoroughName"]).title().replace(" And ", " and ").replace(" Upon ", " upon "),
            "delay": TRAFFIC_DELAYS.get(r["DelayCode_Description"], ""),
            "t": [round(t_mob), round(t_go), round(t_arr), round(t_left), round(t_home)],
            "travel_s": round(travel), "inc_ll": [round(i.lon, 5), round(i.lat, 5)],
            "out": {"p": out_p, "t": out_t}, "back": {"p": back_p, "t": back_t},
        })
        engines.setdefault(r["home_station"], set()).add(r["Resource_Code"])

    # Engines per station: from the yearly busyness table (pump codes sent from home in the year),
    # never fewer than the engines seen on this day.
    yearly = ROOT / "data" / "figures" / "data_exploration" / "station_busyness" / "station_busyness.csv"
    fleet = pd.read_csv(yearly).set_index("station")["engines"].to_dict() if yearly.exists() else {}
    station_list = [{"name": s, "ll": [round(stations.loc[s, "lon"], 5), round(stations.loc[s, "lat"], 5)],
                     "engines": int(max(fleet.get(s, 1), len(engines.get(s, ()))))} for s in stations.index]

    payload = {
        "date": date.strftime("%A %-d %B %Y"),
        "trips": trips,
        "stations": station_list,
        "speed_by_hour": hourly_speed_kph(),
        "n_incidents": int(m["IncidentNumber"].nunique()),
    }
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"{len(trips):,} trips -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
