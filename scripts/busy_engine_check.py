#!/usr/bin/env python3
"""How much time could covering busy stations save? Measured from LFB's own records.

No travel-time or dispatch model: each engine's busy periods (mobilised to left the scene)
are replayed from the mobilisation records to find, at each call, whether the home engines
of the call's station area were all out. Then:

  check     when the area's station is empty, how often does another station respond?
  penalty   first-engine attendance when empty vs not, within the same 500 m square
            and period of the day (so busy areas and hours are compared with themselves)
  ceiling   share of calls x penalty: the most that perfect cover could recover
  timing    how long the station had been empty when the call came: a covering engine
            needs minutes to get there, so only calls after that can gain
  neighbours whether a neighbouring station had an engine to spare at that moment

Limits: an engine out for other reasons (training, repairs, fire safety visits) looks free, so
"empty" is undercounted; the time to return from the scene is not counted; a ceiling is not
what a real cover rule would achieve, because the moved engine leaves its own area thinner.

Writes data/figures/validation/busy_engines/: summary.json, by_period.csv.

Run:  PYTHONPATH=src .venv/bin/python scripts/busy_engine_check.py [--year 2024]
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

from firestation.facility import haversine_km  # noqa: E402

RAW = ROOT / "data" / "raw" / "london"
OUT = ROOT / "data" / "figures" / "validation" / "busy_engines"
PERIODS = {"Night 00-07": range(0, 7), "Morning 07-10": range(7, 10), "Midday 10-16": range(10, 16),
           "Evening 16-20": range(16, 20), "Late 20-24": range(20, 24)}
NEIGHBOURS = 4
MOB_FILES = {y: "lfb_mobilisations_2025.csv" if y >= 2025 else "lfb_mobilisations_2021_2024.csv" for y in range(2021, 2027)}


def when(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.replace("NULL", np.nan), dayfirst=True, errors="coerce", format="mixed")


def station_sweep(start: np.ndarray, end: np.ndarray, pumps: int):
    """Event times, engines out after each event, and when the station last became empty."""
    t = np.r_[start, end]
    d = np.r_[np.ones(len(start), int), -np.ones(len(end), int)]
    o = np.lexsort((d, t))  # at one instant, ends before starts
    t, d = t[o], d[o]
    out = np.cumsum(d)
    empty_since = np.full(len(t), np.nan)
    since = np.nan
    for i in range(len(t)):
        if out[i] >= pumps and not (i and out[i - 1] >= pumps):
            since = t[i]
        elif out[i] < pumps:
            since = np.nan
        empty_since[i] = since
    return t, out, empty_since


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2024)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cols = ["IncidentNumber", "CalYear", "HourOfCall", "Resource_Code", "DateAndTimeMobilised", "DateAndTimeArrived",
            "DateAndTimeLeft", "DateAndTimeReturned", "AttendanceTimeSeconds", "DeployedFromStation_Name",
            "DeployedFromStation_Code", "DeployedFromLocation", "PumpOrder"]
    m = pd.read_csv(RAW / MOB_FILES[args.year], usecols=cols, low_memory=False, encoding="utf-8-sig")
    m = m[m["CalYear"] == args.year].copy()
    m["t0"], m["t_left"] = when(m["DateAndTimeMobilised"]), when(m["DateAndTimeLeft"])
    m["t_arr"], m["t_ret"] = when(m["DateAndTimeArrived"]), when(m["DateAndTimeReturned"])
    sec = lambda s: (s - pd.Timestamp("2000-01-01")).dt.total_seconds().to_numpy()  # noqa: E731
    fields = {c: round(float(m[c].notna().mean()), 3) for c in ("t0", "t_arr", "t_left", "t_ret")}
    resolution = {"share_t0_with_seconds": round(float((m["t0"].dt.second != 0).mean()), 3)}
    m["busy_s"] = (m["t_left"] - m["t0"]).dt.total_seconds()
    print(f"{args.year}: {len(m):,} mobilisations; fields present {fields}; {resolution}", flush=True)
    print("Deployed from:", m["DeployedFromLocation"].value_counts(normalize=True).round(4).to_dict(), flush=True)

    # Engines in regular use, grouped by home station (first three characters of the code).
    code = m["Resource_Code"].astype(str)
    m = m[code.str.match(r"^[A-Z]\d{3}$")].copy()
    m["home"] = m["Resource_Code"].str[:3]
    use = m.groupby("Resource_Code").size()
    regular = use[use >= 100].index
    pumps = pd.Series(regular.str[:3]).value_counts()
    names = m[m["DeployedFromLocation"] == "Home Station"].groupby("DeployedFromStation_Code")["DeployedFromStation_Name"].agg(
        lambda s: s.value_counts().index[0])
    m = m[m["Resource_Code"].isin(regular)]
    busy = m.dropna(subset=["t0", "t_left"])
    busy = busy[(busy["busy_s"] > 0) & (busy["busy_s"] < 6 * 3600)]
    print(f"Regular engines {len(regular)}, stations {len(pumps)}; busy period median {busy['busy_s'].median() / 60:.1f} min, "
          f"mean {busy['busy_s'].mean() / 60:.1f} min", flush=True)

    sweeps = {}
    for s, g in busy.groupby("home"):
        if s in pumps.index:
            sweeps[s] = station_sweep(sec(g["t0"]), sec(g["t_left"]), int(pumps[s]))

    def state(station_codes: np.ndarray, t: np.ndarray):
        out = np.zeros(len(t))
        since = np.full(len(t), np.nan)
        for s in np.unique(station_codes):
            if s not in sweeps:
                continue
            ev, n, em = sweeps[s]
            i = np.flatnonzero(station_codes == s)
            j = np.searchsorted(ev, t[i], side="left") - 1  # state just before the call
            ok = j >= 0
            out[i[ok]] = n[j[ok]]
            since[i[ok]] = em[j[ok]]
        return out, since

    # Calls: recorded first engine, location and station area.
    inc = pd.read_csv(RAW / ("lfb_incidents_2024_onwards.csv" if args.year >= 2024 else "lfb_incidents_2018_2023.csv"),
                      usecols=["IncidentNumber", "CalYear", "Latitude", "Longitude", "Easting_rounded", "Northing_rounded",
                               "IncidentStationGround", "FirstPumpArriving_AttendanceTime", "FirstPumpArriving_DeployedFromStation"],
                      low_memory=False, encoding="utf-8-sig")
    inc = inc[inc["CalYear"] == args.year].dropna(subset=["FirstPumpArriving_AttendanceTime", "IncidentStationGround"])
    first = m.sort_values("t0").drop_duplicates("IncidentNumber")[["IncidentNumber", "t0", "HourOfCall"]]
    c = inc.merge(first, on="IncidentNumber").dropna(subset=["t0"]).reset_index(drop=True)
    name_to_code = {v.upper(): k for k, v in names.items()}
    c["ground"] = c["IncidentStationGround"].astype(str).str.upper().map(name_to_code)
    c = c.dropna(subset=["ground"]).reset_index(drop=True)
    c["att"] = c["FirstPumpArriving_AttendanceTime"].astype(float)
    t = sec(c["t0"])
    n_out, since = state(c["ground"].to_numpy(), t)
    c["pumps"] = c["ground"].map(pumps).fillna(1)
    c["empty"] = n_out >= c["pumps"]
    c["since_min"] = (t - since) / 60
    c["from_ground"] = c["FirstPumpArriving_DeployedFromStation"].astype(str).str.upper() == c["IncidentStationGround"].astype(str).str.upper()
    print(f"Calls {len(c):,}; station empty at call {c['empty'].mean():.1%}", flush=True)
    print(f"Check: first engine from the area's own station: empty {c.loc[c['empty'], 'from_ground'].mean():.1%}, "
          f"not empty {c.loc[~c['empty'], 'from_ground'].mean():.1%}", flush=True)

    # Penalty within the same 500 m square and period.
    c["sq"] = (c["Easting_rounded"].astype(float) // 500).astype(int).astype(str) + "_" + (c["Northing_rounded"].astype(float) // 500).astype(int).astype(str)
    c["period"] = c["HourOfCall"].map({h: p for p, r in PERIODS.items() for h in r})
    c["over6"] = (c["att"] > 360).astype(float)

    def within(df: pd.DataFrame, col: str, flag: str) -> float:
        g = df.groupby(["sq", "period"])
        x = df[flag].astype(float) - g[flag].transform(lambda s: s.astype(float).mean())
        y = df[col] - g[col].transform("mean")
        return float((x * y).sum() / (x * x).sum())

    def penalty(df, flag="empty", n_boot=200, seed=0):
        est = within(df, "att", flag)
        sq = df["sq"].unique()
        rng = np.random.default_rng(seed)
        parts = {k: v for k, v in df.groupby("sq")}
        b = []
        for _ in range(n_boot):
            pick = rng.choice(sq, len(sq))
            d = pd.concat([parts[k].assign(sq=f"{i}") for i, k in enumerate(pick)], ignore_index=True)
            b.append(within(d, "att", flag))
        return est, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    est, lo, hi = penalty(c)
    p6 = within(c, "over6", "empty")
    share = float(c["empty"].mean())
    print(f"Penalty when empty: +{est:.1f} s [{lo:.1f}, {hi:.1f}]; over-6-min +{p6 * 100:.1f} points", flush=True)
    print(f"Ceiling of perfect cover: {share * est:.1f} s average; {share * p6 * 100:.2f} points of 6-minute calls", flush=True)
    summary = {"doc": __doc__, "year": args.year, "fields_present": fields, "resolution": resolution,
               "deployed_from": m["DeployedFromLocation"].value_counts(normalize=True).round(4).to_dict(),
               "regular_engines": int(len(regular)), "stations": int(len(pumps)), "calls": int(len(c)),
               "busy_median_min": round(float(busy["busy_s"].median() / 60), 1), "busy_mean_min": round(float(busy["busy_s"].mean() / 60), 1),
               "mean_attendance_s": round(float(c["att"].mean()), 1), "share_over_6min": round(float(c["over6"].mean()), 4),
               "share_empty": round(share, 4), "penalty_s": round(est, 1), "penalty_lo": round(lo, 1), "penalty_hi": round(hi, 1),
               "penalty_over6_points": round(p6 * 100, 1), "ceiling_avg_s": round(share * est, 2),
               "ceiling_over6_points": round(share * p6 * 100, 2),
               "first_from_own_station_when_empty": round(float(c.loc[c["empty"], "from_ground"].mean()), 4),
               "first_from_own_station_when_not_empty": round(float(c.loc[~c["empty"], "from_ground"].mean()), 4)}

    # Timing: how long the station had been empty when the call came.
    e = c[c["empty"]]
    timing = {}
    for mins in (0, 3, 5, 8, 10, 15):
        f = c[(~c["empty"]) | (c["since_min"] >= mins)]
        pen = within(f, "att", "empty")
        s_all = float(((c["empty"]) & (c["since_min"] >= mins)).mean())
        timing[mins] = {"share_of_all_calls": round(s_all, 4), "penalty_s": round(pen, 1), "ceiling_s": round(s_all * pen, 2)}
    summary["timing_by_minutes_empty"] = timing
    print("Empty for at least N minutes:", json.dumps(timing), flush=True)

    # Does a neighbour have an engine to spare when the station is empty?
    st = pd.read_csv(RAW / "london_fire_stations.csv")
    st["code"] = st["name"].str.upper().map(name_to_code)
    st = st.dropna(subset=["code"]).reset_index(drop=True)
    D = haversine_km(st["lon"].to_numpy()[:, None], st["lat"].to_numpy()[:, None], st["lon"].to_numpy()[None, :], st["lat"].to_numpy()[None, :])
    near = {st.at[i, "code"]: st["code"].to_numpy()[np.argsort(D[i])[1:NEIGHBOURS + 1]] for i in range(len(st))}
    te = sec(e["t0"])
    free = np.zeros(len(e), bool)
    two = np.zeros(len(e), bool)
    grounds = e["ground"].to_numpy()
    for k in range(NEIGHBOURS):
        nb = np.array([near[g][k] if g in near else "" for g in grounds], dtype=object)
        o, _ = state(nb, te)
        total = pd.Series(nb).map(pumps).fillna(0).to_numpy()
        free |= o < total
        two |= (total - o) >= 2
    summary["when_empty_a_near_station_has_a_free_engine"] = round(float(free.mean()), 4)
    summary["when_empty_a_near_station_has_two_free_engines"] = round(float(two.mean()), 4)
    print(f"When empty: one of {NEIGHBOURS} nearest stations has a free engine {free.mean():.1%}, two free {two.mean():.1%}", flush=True)

    # By period and worst stations.
    rows = []
    for p in PERIODS:
        g = c[c["period"] == p]
        rows.append({"period": p, "calls": len(g), "share_empty": round(float(g["empty"].mean()), 4),
                     "mean_att_empty": round(float(g.loc[g["empty"], "att"].mean()), 1),
                     "mean_att_not_empty": round(float(g.loc[~g["empty"], "att"].mean()), 1),
                     "over6_empty": round(float(g.loc[g["empty"], "over6"].mean()), 3),
                     "over6_not_empty": round(float(g.loc[~g["empty"], "over6"].mean()), 3)})
    bp = pd.DataFrame(rows)
    bp.to_csv(OUT / "by_period.csv", index=False)
    print(bp.to_string(index=False), flush=True)
    by_st = c.groupby("IncidentStationGround").agg(calls=("empty", "size"), share_empty=("empty", "mean"), mean_att=("att", "mean")).sort_values("share_empty", ascending=False)
    by_st.round(3).to_csv(OUT / "by_station.csv")
    summary["share_of_empty_calls_in_top_10_stations"] = round(float((by_st["share_empty"] * by_st["calls"]).nlargest(10).sum() / (by_st["share_empty"] * by_st["calls"]).sum()), 3)
    print(by_st.head(8).round(3).to_string(), flush=True)
    print("Top 10 stations' share of empty calls:", summary["share_of_empty_calls_in_top_10_stations"])
    summary["over6_when_empty"] = round(float(c.loc[c["empty"], "over6"].mean()), 3)
    summary["over6_when_not_empty"] = round(float(c.loc[~c["empty"], "over6"].mean()), 3)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print({k: v for k, v in summary.items() if k != "doc" and not isinstance(v, dict)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
