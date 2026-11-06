#!/usr/bin/env python3
"""Part 2 final test: score the fixed targeted layouts on 2026 calls, once, against docs/prediction_2026.md.

Nothing is chosen here. Target boroughs, layouts, scoring settings and pass marks were all fixed before 2026 was looked at.
Scoring is the same as scripts/targeted_areas.py: demand cells, busy-engine rates, turnout and the Kolesar fit all come from 2024;
only the calls being counted change (2026 instead of 2025). Three travel models: LightGBM, road Kolesar, straight-line Kolesar.
Also refits the drive-time check of scripts/outer_error_check.py on 2026 trips (prediction 3).

  --check-year 2025   runs the same code on 2025 and writes to final_test_2026/check_2025/, to show that it reproduces the
                      published 2025 numbers. Allowed any number of times; it never reads 2026.
  (no flag)           the real test. Refuses to run if docs/prediction_2026.md or this script has uncommitted changes, or if it
                      has already been run (data/figures/placement/final_test_2026/result.json exists).

Run:  PYTHONPATH=src .venv/bin/python scripts/final_test_2026.py --check-year 2025
      PYTHONPATH=src .venv/bin/python scripts/final_test_2026.py          # once only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from firestation.busy_placement import ErlangBusyModel  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid, posture_seconds, weighted_cvar  # noqa: E402
from firestation.fire_station_location import (build_travel_matrix, busy_prob_vector, drop_far_incidents, fit_scorer_kolesar,  # noqa: E402
                                               recorded_station_turnout, snap_to_cells)
from firestation.ml_scorer import cell_boroughs  # noqa: E402
from firestation.station_capacity import default_pump_table  # noqa: E402
from firestation.travel_time_model import attach_weather, load_weather, predict  # noqa: E402

PL = ROOT / "data" / "figures" / "placement"
OUT = PL / "final_test_2026"
PREDICTION = ROOT / "docs" / "prediction_2026.md"
TARGETS = ["HILLINGDON", "HAVERING", "ENFIELD", "BROMLEY", "REDBRIDGE"]  # fixed from recorded 2024 (targeted/targets.csv)
PRIMARY = "rule_b_target_10"
LAYOUTS = {  # name -> saved layout; all fixed before 2026 was used
    "rule_b_target_10": PL / "targeted" / "rule_b" / "layouts" / "target_10.csv",
    "rule_b_target_20": PL / "targeted" / "rule_b" / "layouts" / "target_20.csv",
    "target_10": PL / "targeted" / "layouts" / "target_10.csv",
    "target_20": PL / "targeted" / "layouts" / "target_20.csv",
    "citywide_10": PL / "targeted" / "layouts" / "citywide_10.csv",
    "planner_10": PL / "recommended_wide" / "replace_10_boroughs1s" / "planned_fire_stations.csv",
    "planner_20": PL / "recommended_wide" / "replace_20_boroughs1s" / "planned_fire_stations.csv",
}
SCORERS = ("lgbm", "road", "kolesar")
BOOT = 1000
LAST_DAY = {2025: "2025-12-31", 2026: "2026-07-31"}  # 2026: the months in the download when the prediction was written


def up_to(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[pd.to_datetime(df["date_of_call"], format="mixed") <= pd.Timestamp(LAST_DAY[year])]


def committed_clean(*paths: Path) -> bool:
    for p in paths:
        rel = str(p.relative_to(ROOT))
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
        dirty = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        if not tracked or dirty:
            print(f"{rel}: not committed, or has uncommitted changes")
            return False
    return True


def borough_means(r, w, boroughs, names):
    return np.array([w[boroughs == b] @ r[boroughs == b] / max(w[boroughs == b].sum(), 1e-9) for b in names])


def drive_check(year: int) -> dict:
    """Prediction 3: recorded vs modelled drive time on first-engine trips from the home station, engine not busy, target boroughs."""
    df = pd.read_csv(ROOT / "data" / "processed" / "lfb_travel_training.csv", low_memory=False)
    te = df[(df["cal_year"] == year) & (df["pump_order"] == 1) & (df["deployed_from_location"] == "Home Station") & (df["busy_flag"] == 0)
            & df["borough"].astype(str).str.upper().isin(TARGETS)].reset_index(drop=True)
    del df
    te = up_to(te, year).reset_index(drop=True)
    te = attach_weather(te, load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    bundle = joblib.load(ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib")
    p = np.concatenate([predict(bundle, te.iloc[i:i + 50000]) for i in range(0, len(te), 50000)])
    y = te["drive_s"].to_numpy(float)
    rng = np.random.default_rng(year)
    bs = [np.polyfit(p[i], y[i], 1)[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
    return {"trips": int(len(y)), "slope": round(float(np.polyfit(p, y, 1)[0]), 3), "slope_lo": round(float(np.percentile(bs, 2.5)), 3),
            "slope_hi": round(float(np.percentile(bs, 97.5)), 3), "recorded_minus_model_s": round(float((y - p).mean()), 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-year", type=int, choices=(2025,), help="Dry run on an already-used year; never reads 2026")
    args = ap.parse_args()
    year = args.check_year or 2026
    out = OUT / f"check_{year}" if args.check_year else OUT
    if not args.check_year:
        if (OUT / "result.json").exists():
            print("Already run once; see", OUT / "result.json")
            return 1
        if not committed_clean(PREDICTION, Path(__file__).resolve()):
            print("Commit docs/prediction_2026.md and this script first.")
            return 1
    out.mkdir(parents=True, exist_ok=True)

    spec = get_city("london")
    houses = load_fire_stations(spec)
    incidents = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024, year]), houses)
    demand = incidents[incidents["cal_year"] == 2024].reset_index(drop=True)
    ev = incidents[incidents["cal_year"] == year]
    ev = up_to(ev, year).reset_index(drop=True)
    print(f"{year} calls scored: {len(ev)} (to {LAST_DAY[year]})", flush=True)
    cells = build_demand_grid(demand, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    w = cells["n_incidents"].to_numpy(float)
    boroughs = np.asarray(cell_boroughs(cells, demand)).astype(str)
    names = sorted(set(boroughs) - {""})
    is_t = np.isin(boroughs, TARGETS)
    cidx, _ = snap_to_cells(ev, cells)
    n = np.bincount(cidx, minlength=len(cells)).astype(float)
    N, tN = n.sum(), n[is_t].sum()
    pumps_today = default_pump_table(houses).set_index("station")["n_pumps"].reindex(houses["facilityname"].astype(str)).fillna(1).astype(int).to_numpy()
    tab, new_turn = recorded_station_turnout([2024])
    norm = {k.upper(): v for k, v in tab.items()}
    kmodel, _, _ = fit_scorer_kolesar(demand, houses)

    layouts = {"today": pd.DataFrame({"site_name": houses["facilityname"].astype(str), "lat": houses["lat"], "lon": houses["lon"],
                                      "is_existing": True, "n_pumps": pumps_today})}
    for k, f in LAYOUTS.items():
        layouts[k] = pd.read_csv(f)[["site_name", "lat", "lon", "is_existing", "n_pumps"]]
    key = lambda r: str(r.site_name) if bool(r.is_existing) else f"{float(r.lat):.6f},{float(r.lon):.6f}"  # noqa: E731
    sites, index = [], {}
    for L in layouts.values():
        for r in L.itertuples():
            if key(r) not in index:
                index[key(r)] = len(sites)
                sites.append({"site_name": str(r.site_name), "lat": float(r.lat), "lon": float(r.lon), "is_existing": bool(r.is_existing)})
    sites = pd.DataFrame(sites)
    rows = {k: np.array([index[key(r)] for r in L.itertuples()]) for k, L in layouts.items()}
    pumps = {k: np.maximum(L["n_pumps"].fillna(1).astype(int).to_numpy(), 1) for k, L in layouts.items()}
    turnout = np.array([norm.get(r.site_name.upper(), new_turn) if r.is_existing else new_turn for r in sites.itertuples()])
    measured = busy_prob_vector(sites, demand)
    boot = np.random.default_rng(0).multinomial(int(N), n / N, size=BOOT).astype(float)

    records, brows = [], []
    for sc in SCORERS:
        print(f"Scoring with {sc} on {year} calls ({int(N)} calls, {len(sites)} sites)", flush=True)
        Ts, _ = build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer=sc, kolesar=kmodel, incidents=demand)
        As = Ts + turnout[:, None]
        bp = np.ones(len(sites), int)
        bp[rows["today"]] = pumps["today"]
        m = ErlangBusyModel.calibrate(As, rows["today"], measured, bp, w)

        def resp(k):
            m.pumps = np.ones(len(sites), int)
            m.pumps[rows[k]] = pumps[k]
            return posture_seconds(As, rows[k], m.probs(As, rows[k]))

        q0 = resp("today")
        b0 = borough_means(q0, n, boroughs, names)
        for k in LAYOUTS:
            q = resp(k)
            d = q - q0
            bd = borough_means(q, n, boroughs, names) - b0
            tb = boot[:, is_t] @ d[is_t] / boot[:, is_t].sum(1)
            records.append({"layout": k, "scorer": sc, "moved": int((~layouts[k]["is_existing"].astype(bool)).sum()),
                            "d_target_s": round(float(n[is_t] @ d[is_t] / tN), 2), "target_lo": round(float(np.percentile(tb, 2.5)), 2),
                            "target_hi": round(float(np.percentile(tb, 97.5)), 2), "d_london_s": round(float(n @ d / N), 2),
                            "d_slowest10_s": round(weighted_cvar(q, n) - weighted_cvar(q0, n), 2),
                            "boroughs_worse_1s": int((bd >= 1).sum()), "worst_borough_s": round(float(bd.max()), 1),
                            "worst_borough": names[int(bd.argmax())], "calls_30s_slower_pct": round(float(n[d >= 30].sum() / N * 100), 1)})
            brows += [{"layout": k, "scorer": sc, "borough": b, "target": b in TARGETS, "d_s": round(float(x), 2)} for b, x in zip(names, bd)]
    df = pd.DataFrame(records)
    df.to_csv(out / "scores.csv", index=False)
    pd.DataFrame(brows).to_csv(out / "boroughs.csv", index=False)

    p = df[df["layout"] == PRIMARY].set_index("scorer")
    drive = drive_check(year)
    verdict = {"p1_targets_10s_faster_all_models": bool((p["d_target_s"] <= -10).all()),
               "p2_no_borough_1s_worse_any_model": bool((p["boroughs_worse_1s"] == 0).all()),
               "p3_drive_slope_0.9_to_1.1": bool(0.9 <= drive["slope"] <= 1.1)}
    res = {"year": year, "dry_run": bool(args.check_year), "calls": int(N), "target_calls": int(tN), "primary": PRIMARY,
           "primary_scores": p[["d_target_s", "target_lo", "target_hi", "d_london_s", "boroughs_worse_1s", "worst_borough_s"]].to_dict("index"),
           "drive_check": drive, "verdict": verdict, "all_pass": all(verdict.values()), "rows": records}
    (out / "result.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(df.to_string(index=False))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
