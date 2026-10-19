#!/usr/bin/env python3
"""Score every station layout under three travel-time models, with bootstrap intervals.

The validation rule from the project overview: a layout counts as better than today's
stations only if it wins under **all three** travel-time models, because a layout chosen
with one model always looks good under that same model.

  lgbm     LightGBM decision trees on route street features (the project's model)
  road     Kolesar curve on shortest legal road distance
  kolesar  Kolesar curve on straight-line distance (refit on LFB attendance)

Layouts:
  - today's stations;
  - recommended layouts (scripts/recommended_placement.py; LightGBM-chosen), including any
    borough-floor runs that have finished;
  - distance-based layouts: the same rules (alpha 1, busy engines, slowest-10 % and average
    floors, pumps move with stations) but sites chosen with the road or straight-line
    Kolesar model instead of LightGBM (``--make-distance-layouts``; no street refinement).

Scoring (held-out 2025 calls, each at its 0.9 km demand cell; busy engines with each
layout's own pump counts; turnout as recorded, 73 s for new sites), with two models of
which station responds (``response``):
  busy_blend  nearest station, or the next nearest when it is busy (Erlang loss)
  dispatch    the learned dispatch model (firestation.dispatch): any of the 8 fastest
              open stations, with probabilities learned from LFB's first-engine records
Measures:
  mean_s, slowest10_s (average of the slowest 10 % of calls), within_6min;
  each layout minus today, with a 95 % paired bootstrap interval (resampling 2025 calls).
  wins_all_three: the interval for the average is below 0 under every model.

Writes data/figures/placement/layout_scoring/: scores.csv, summary.json, layouts/ (distance-based).

Run:  PYTHONPATH=src .venv/bin/python scripts/score_layouts.py [--make-distance-layouts]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.busy_placement import ErlangBusyModel  # noqa: E402
from firestation.dispatch import dispatch_response, load_dispatch  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid, posture_seconds, weighted_cvar  # noqa: E402
from firestation.fire_station_location import (  # noqa: E402
    build_travel_matrix,
    busy_prob_vector,
    drop_far_incidents,
    fit_scorer_kolesar,
    recorded_station_turnout,
    run_fire_station_plan,
    snap_to_cells,
)
from firestation.station_capacity import default_pump_table  # noqa: E402

OUT = ROOT / "data" / "figures" / "placement" / "layout_scoring"
REC = ROOT / "data" / "figures" / "placement" / "recommended"
SCORERS = ("lgbm", "road", "kolesar")
SCENARIOS = {"expand_3": {"mode": "expand", "add_stations": 3},
             "replace_10": {"mode": "replace", "replace_r": 10},
             "redesign": {"mode": "redesign"}}


def make_distance_layouts(houses, incidents, scenarios):
    (OUT / "layouts").mkdir(parents=True, exist_ok=True)
    for scorer in ("road", "kolesar"):
        for sc in scenarios:
            path = OUT / "layouts" / f"{scorer}_{sc}.csv"
            if path.exists():
                continue
            kw = dict(SCENARIOS[sc])
            if kw["mode"] == "redesign":
                kw["n_houses"] = len(houses)
            print(f"  choosing {sc} with {scorer}…", flush=True)
            with contextlib.redirect_stdout(io.StringIO()):
                r = run_fire_station_plan(city_id="london", houses=houses, incidents=incidents, scorer=scorer,
                                          demand_years=[2024], eval_years=[2025], max_incident_rows=None,
                                          use_busy=True, keep_slowest=True, keep_average=True,
                                          busy_rounds=4, solver_time_s=600, **kw)
            r.selected.to_csv(path, index=False)


def load_layouts(houses, today_pumps) -> dict[str, pd.DataFrame]:
    today = pd.DataFrame({"site_name": houses["facilityname"].astype(str), "lat": houses["lat"], "lon": houses["lon"],
                          "is_existing": True, "n_pumps": today_pumps})
    out = {"today": today}
    for d in sorted(p for p in REC.iterdir() if p.is_dir() and (p / "planned_fire_stations.csv").exists()):
        out[f"lgbm_{d.name}"] = pd.read_csv(d / "planned_fire_stations.csv")
    for f in sorted((OUT / "layouts").glob("*.csv")):
        out[f.stem] = pd.read_csv(f)
    return out


def site_key(r) -> str:
    return str(r.site_name) if bool(r.is_existing) else f"{float(r.lat):.6f},{float(r.lon):.6f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-distance-layouts", action="store_true")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    spec = get_city("london")
    houses = load_fire_stations(spec)
    incidents = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024, 2025]), houses)
    demand = incidents[incidents["cal_year"] == 2024].reset_index(drop=True)
    ev = incidents[incidents["cal_year"] == 2025].reset_index(drop=True)
    if args.make_distance_layouts:
        print("Distance-based layouts (same rules, sites chosen by road / straight-line Kolesar)…", flush=True)
        make_distance_layouts(houses, incidents, args.scenarios.split(","))

    pumps_today = default_pump_table(houses).set_index("station")["n_pumps"].reindex(
        houses["facilityname"].astype(str)).fillna(1).astype(int).to_numpy()
    layouts = load_layouts(houses, pumps_today)
    print(f"Layouts: {', '.join(layouts)}", flush=True)

    # Union of all sites, today's stations first.
    sites, index = [], {}
    for name, L in layouts.items():
        for r in L.itertuples():
            k = site_key(r)
            if k not in index:
                index[k] = len(sites)
                sites.append({"site_name": str(r.site_name), "lat": float(r.lat), "lon": float(r.lon),
                              "is_existing": bool(r.is_existing)})
    sites = pd.DataFrame(sites)
    rows = {name: np.array([index[site_key(r)] for r in L.itertuples()]) for name, L in layouts.items()}
    pumps = {name: (L["n_pumps"].fillna(1).astype(int).to_numpy() if "n_pumps" in L else np.ones(len(L), int))
             for name, L in layouts.items()}

    cells = build_demand_grid(demand, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    cidx, _ = snap_to_cells(ev, cells)
    counts = np.bincount(cidx, minlength=len(cells)).astype(float)
    n = counts.sum()
    load_w = cells["n_incidents"].to_numpy(float)

    turn_tab, turn_new = recorded_station_turnout([2024])
    norm = {k.upper(): v for k, v in turn_tab.items()}
    turnout = np.array([norm.get(r.site_name.upper(), turn_new) if r.is_existing else turn_new
                        for r in sites.itertuples()])
    measured = busy_prob_vector(sites.assign(site_name=sites["site_name"]), demand)
    kmodel, _, _ = fit_scorer_kolesar(demand, houses)

    dmodel = load_dispatch()
    rng = np.random.default_rng(0)
    boot = rng.multinomial(int(n), counts / n, size=args.boot).astype(float)  # resampled 2025 calls per cell

    records = []
    today_rows = rows["today"]
    for scorer in SCORERS:
        print(f"Travel times: {scorer} ({len(sites)} sites × {len(cells)} cells)…", flush=True)
        T, _ = build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer=scorer,
                                   kolesar=kmodel, incidents=demand)
        A = T + turnout[:, None]
        base_pumps = np.ones(len(sites), int)
        base_pumps[today_rows] = pumps["today"]
        model = ErlangBusyModel.calibrate(A, today_rows, measured, base_pumps, load_w)
        responses = {"busy_blend": {}, "dispatch": {}}
        for name, r in rows.items():
            model.pumps = np.ones(len(sites), int)
            model.pumps[r] = np.maximum(pumps[name], 1)
            responses["busy_blend"][name] = posture_seconds(A, r, model.probs(A, r))
            responses["dispatch"][name] = dispatch_response(A, r, model, dmodel)
        for kind, resp in responses.items():
            r0 = resp["today"]
            b_mean0 = boot @ r0 / n
            b_cvar0 = np.array([weighted_cvar(r0, b) for b in boot])
            for name, r in resp.items():
                mean = float(counts @ r / n)
                cvar = weighted_cvar(r, counts)
                d_mean = boot @ r / n - b_mean0
                d_cvar = np.array([weighted_cvar(r, b) for b in boot]) - b_cvar0
                records.append({
                    "layout": name, "scorer": scorer, "response": kind, "mean_s": round(mean, 2),
                    "slowest10_s": round(cvar, 2),
                    "within_6min": round(float(counts @ (r <= 360) / n), 4),
                    "d_mean_s": round(mean - float(counts @ r0 / n), 2),
                    "d_mean_lo": round(float(np.percentile(d_mean, 2.5)), 2),
                    "d_mean_hi": round(float(np.percentile(d_mean, 97.5)), 2),
                    "d_slowest10_s": round(cvar - weighted_cvar(r0, counts), 2),
                    "d_slowest10_lo": round(float(np.percentile(d_cvar, 2.5)), 2),
                    "d_slowest10_hi": round(float(np.percentile(d_cvar, 97.5)), 2),
                })
    df = pd.DataFrame(records)
    keys = ["layout", "response"]
    nt = df[df["layout"] != "today"]
    wins = nt.groupby(keys).agg(wins_all_three=("d_mean_hi", lambda s: bool((s < 0).all())))
    wins["slowest10_no_worse_all_three"] = nt.groupby(keys)["d_slowest10_s"].apply(lambda s: bool((s <= 0).all()))
    df = df.merge(wins, left_on=keys, right_index=True, how="left")
    df.to_csv(OUT / "scores.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(
        {"doc": __doc__, "n_calls_2025": int(n), "bootstrap": args.boot,
         "rows": df.to_dict(orient="records")}, indent=2, default=str) + "\n")
    pd.set_option("display.width", 220)
    print(df.pivot_table(index="layout", columns=["response", "scorer"], values="d_mean_s").round(1).to_string())
    print(wins.to_string())
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
