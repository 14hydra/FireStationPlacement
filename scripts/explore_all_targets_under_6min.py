#!/usr/bin/env python3
"""Exploratory (not pre-registered): can all 5 target boroughs get under LFB's 6-minute average?

Question asked 2026-11-15, after the Part 2 tests. Nothing here changes Part 2's fixed results.

"Under 6 minutes" for a borough = recorded 2024 average + modelled change < 360 s (the same reading as results.md 5.8).
Two searches, both on 2024 demand with LightGBM and busy engines (targeted_areas.py's set-up and swap search):

  moves_<k>  move up to k stations (5, 10, ..., 40), same engines, with targeted_areas.py's protections: no borough's
             average worse, London average no worse, slowest 10 % no worse. Objective: total seconds the 5 target
             boroughs are over 360 s (plus a tiny pull on their average).
  add_<n>    add n new stations, closing none (greedy, one at a time), until every target borough is under 360 s.
             --engines move: each new station takes the second engine of the nearest two-engine station (London's
             engine count fixed); --engines new: each new station gets a new engine.

Each layout is then scored on 2025 calls under LightGBM, road Kolesar and straight-line Kolesar.
Writes data/figures/placement/all_targets_6min/.

Run:  PYTHONPATH=src .venv/bin/python scripts/explore_all_targets_under_6min.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import targeted_areas as TA  # noqa: E402

from firestation.busy_placement import ErlangBusyModel, PumpPlan  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid, posture_seconds, weighted_cvar  # noqa: E402
from firestation.fire_station_location import (build_house_candidates, build_travel_matrix, busy_prob_vector,  # noqa: E402
                                               drop_far_incidents, fit_scorer_kolesar, recorded_station_turnout,
                                               snap_to_cells)
from firestation.ml_scorer import cell_boroughs  # noqa: E402
from firestation.station_capacity import default_pump_table  # noqa: E402

OUT = ROOT / "data" / "figures" / "placement" / "all_targets_6min"
TARGET_S = 360.0
MOVE_LIMITS = (5, 10, 15, 20, 25, 30, 40)
MAX_ADD = 15


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = get_city("london")
    houses = load_fire_stations(spec)
    incidents = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024, 2025]), houses)
    demand = incidents[incidents["cal_year"] == 2024].reset_index(drop=True)
    ev = incidents[incidents["cal_year"] == 2025].reset_index(drop=True)

    tb = pd.read_csv(ROOT / "data" / "figures" / "placement" / "targeted" / "targets.csv").set_index("b")
    targets = tb.index[tb["target"]].tolist()
    rec = tb["recorded_mean_s"]

    cells = build_demand_grid(demand, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    w = cells["n_incidents"].to_numpy(float)
    boroughs = np.asarray(cell_boroughs(cells, demand)).astype(str)
    names = sorted(set(boroughs) - {""})
    tpos = [names.index(b) for b in targets]
    is_t = np.isin(boroughs, targets)
    cidx, _ = snap_to_cells(ev, cells)
    n25 = np.bincount(cidx, minlength=len(cells)).astype(float)
    pumps_today = default_pump_table(houses).set_index("station")["n_pumps"].reindex(
        houses["facilityname"].astype(str)).fillna(1).astype(int).to_numpy()
    tab, new_turn = recorded_station_turnout([2024])
    norm = {k.upper(): v for k, v in tab.items()}
    kmodel, _, _ = fit_scorer_kolesar(demand, houses)

    pool = build_house_candidates(houses, cells, demand_candidates=300, spread_km=1.5)
    ex_rows = np.flatnonzero(pool["is_existing"].to_numpy())
    pnames = pool["site_name"].astype(str).str.upper().to_numpy()
    turn = np.array([norm.get(n, new_turn) if e else new_turn for n, e in zip(pnames, pool["is_existing"])])
    lat, lon = pool["lat"].to_numpy(float), pool["lon"].to_numpy(float)
    TA.SITE_LATLON = (lat, lon)
    base = np.ones(len(pool), int)
    base[ex_rows] = pumps_today
    measured = busy_prob_vector(pool, demand)

    A = {}
    for sc in TA.SCORERS:
        T, _ = build_travel_matrix(pool, cells, surrogate=TravelTimeSurrogate(), scorer=sc, kolesar=kmodel, incidents=demand)
        A[sc] = T + turn[:, None]

    def busy_model(sc, extra):
        return ErlangBusyModel.calibrate(A[sc], ex_rows, measured, base, w, PumpPlan(base, ex_rows, lat, lon, extra=extra))

    today = list(ex_rows)
    m = busy_model("lgbm", "move")
    r0 = m.response(A["lgbm"], today)
    W = w.sum()
    b0, m0, c0 = TA.borough_means(r0, w, boroughs, names), w @ r0 / W, weighted_cvar(r0, w)

    def over(bm_now, bm_today, weights_rec=rec):
        """Seconds each target borough is over 360 s: recorded 2024 + modelled change."""
        return np.array([weights_rec[b] + bm_now[i] - bm_today[i] - TARGET_S for b, i in zip(targets, tpos)])

    def shortfall(model):
        def f(idx):
            r = model.response(A["lgbm"], idx)
            bm = TA.borough_means(r, w, boroughs, names)
            return float(np.clip(over(bm, b0), 0, None).sum() + 1e-3 * (w[is_t] @ r[is_t]) / w[is_t].sum())
        return f

    def ok(idx):
        r = m.response(A["lgbm"], idx)
        ex = np.clip(TA.borough_means(r, w, boroughs, names) - b0, 0, None).sum() + max(w @ r / W - m0, 0) + max(weighted_cvar(r, w) - c0, 0)
        return ex <= 1e-6

    gap0 = np.clip(over(b0, b0), 0, None)
    wt = w * np.array([gap0[targets.index(b)] if b in targets else 0.0 for b in boroughs])
    print("Over 360 s today:", dict(zip(targets, gap0.round(1))), flush=True)

    layouts = {}
    idx = today
    for k in MOVE_LIMITS:
        t0 = time.time()
        idx, val = TA.search(A["lgbm"], w, wt, m, idx, ex_rows, k, ok, shortfall(m), log=f"moves_{k}")
        layouts[f"moves_{k}"] = (list(idx), "move")
        print(f"    {time.time() - t0:.0f} s, shortfall {val:.2f}", flush=True)

    for extra in ("move", "new"):
        ma = busy_model("lgbm", extra)
        f = shortfall(ma)
        idx = list(today)
        cand = np.setdiff1d(np.arange(len(pool)), ex_rows)
        for n in range(1, MAX_ADD + 1):
            vals = np.array([f(idx + [int(o)]) for o in cand])
            j = int(vals.argmin())
            idx = idx + [int(cand[j])]
            cand = np.delete(cand, j)
            layouts[f"add_{n}_{extra}"] = (list(idx), extra)
            print(f"  add_{n} ({extra} engines): shortfall {vals[j]:.2f}", flush=True)
            if vals[j] < 0.5:
                break

    # Score on 2025 calls, three models.
    rows = []
    for sc in TA.SCORERS:
        for extra in ("move", "new"):
            mm = busy_model(sc, extra)
            q0 = mm.response(A[sc], today)
            bt0 = TA.borough_means(q0, n25, boroughs, names)
            for name, (lay, e) in layouts.items():
                if e != extra:
                    continue
                q = mm.response(A[sc], lay)
                bd = TA.borough_means(q, n25, boroughs, names) - bt0
                after = {b: round(float(rec[b] + bd[i]), 1) for b, i in zip(targets, tpos)}
                nt = [i for i, b in enumerate(names) if b not in targets]
                rows.append({"layout": name, "scorer": sc, "stations": len(lay),
                             "moved": len(set(ex_rows) - set(lay)), "opened": len(set(lay) - set(ex_rows)),
                             **after, "all_under_360": all(v < TARGET_S for v in after.values()),
                             "worst_other_borough_s": round(float(bd[nt].max()), 1),
                             "d_london_s": round(float(n25 @ (q - q0) / n25.sum()), 2)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scores.csv", index=False)
    for name, (lay, e) in layouts.items():
        pool.iloc[lay][["site_name", "lat", "lon", "is_existing"]].to_csv(OUT / f"{name}.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"doc": __doc__, "targets": targets, "over_360_today_2024": dict(zip(targets, gap0.round(2).tolist()))}, indent=2) + "\n")
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
