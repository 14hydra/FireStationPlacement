#!/usr/bin/env python3
"""Exploratory (not pre-registered): count calls over 6 minutes, and search for layouts that cut that count directly.

Asked 2026-11-18 after the eMentor's second meeting. Nothing here changes Part 2's fixed results.

A call's time after a change = its recorded first-engine time + the modelled change for its 0.9 km cell (as in
scripts/beyond_average.py). Set-up and pool are the same as scripts/explore_all_targets_under_6min.py.

Layouts scored:
  moves_10, add_5_move, add_5_new   saved by explore_all_targets_under_6min.py (objective: borough averages under 360 s)
  count_moves_<k>                    move up to k stations; objective: number of 2024 target-borough calls over 360 s
                                     (smoothed), with the same protections: no borough average worse, London average and
                                     slowest 10 % no worse
  count_add_<n>_<engines>            add n stations greedily with that objective; engines "move" (taken from the nearest
                                     two-engine station) or "new"
Searches use 2024 calls; every layout is then scored on 2026 calls (Jan to 31 Jul, the Part 2 test calls) under three models.
Writes data/figures/placement/calls_under_6min/.

Run:  PYTHONPATH=src .venv/bin/python scripts/explore_calls_under_6min.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import final_test_2026 as F  # noqa: E402
import targeted_areas as TA  # noqa: E402

from firestation.busy_placement import ErlangBusyModel, PumpPlan  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid, weighted_cvar  # noqa: E402
from firestation.fire_station_location import (build_house_candidates, build_travel_matrix, busy_prob_vector,  # noqa: E402
                                               drop_far_incidents, fit_scorer_kolesar, recorded_station_turnout,
                                               snap_to_cells)
from firestation.ml_scorer import cell_boroughs  # noqa: E402
from firestation.station_capacity import default_pump_table  # noqa: E402

PREV = ROOT / "data" / "figures" / "placement" / "all_targets_6min"
OUT = ROOT / "data" / "figures" / "placement" / "calls_under_6min"
TARGET_S, TAU = 360.0, 10.0  # smoothing width (s) for the call count during the search
MOVE_LIMITS = (10, 20)
ADD_N = 5


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = get_city("london")
    houses = load_fire_stations(spec)
    inc = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024, 2026]), houses)
    demand = inc[inc["cal_year"] == 2024].reset_index(drop=True)
    ev = F.up_to(inc[inc["cal_year"] == 2026], 2026).reset_index(drop=True)

    cells = build_demand_grid(demand, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    w = cells["n_incidents"].to_numpy(float)
    W = w.sum()
    boroughs = np.asarray(cell_boroughs(cells, demand)).astype(str)
    names = sorted(set(boroughs) - {""})
    targets = F.TARGETS
    c24, _ = snap_to_cells(demand, cells)
    c26, _ = snap_to_cells(ev, cells)
    rec24, rec26 = demand["travel_seconds"].to_numpy(float), ev["travel_seconds"].to_numpy(float)
    t24 = np.isin(boroughs[c24], targets)
    tc, tr = c24[t24], rec24[t24]  # 2024 target-borough calls the search counts
    n26 = np.bincount(c26, minlength=len(cells)).astype(float)

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

    today = [int(i) for i in ex_rows]
    m = busy_model("lgbm", "move")
    r0 = m.response(A["lgbm"], today)
    b0, m0, cv0 = TA.borough_means(r0, w, boroughs, names), w @ r0 / W, weighted_cvar(r0, w)

    def count_fn(model):
        q0 = model.response(A["lgbm"], today)

        def f(idx):
            d = model.response(A["lgbm"], idx) - q0
            return float((1 / (1 + np.exp(-(tr + d[tc] - TARGET_S) / TAU))).sum())
        return f

    def ok_fn(model):
        def ok(idx):
            r = model.response(A["lgbm"], idx)
            ex = np.clip(TA.borough_means(r, w, boroughs, names) - b0, 0, None).sum() + max(w @ r / W - m0, 0) + max(weighted_cvar(r, w) - cv0, 0)
            return ex <= 1e-6
        return ok

    # Search weights: target calls that sit 0 to 2 minutes over 360 s, the ones a modest change can bring under.
    near = (tr > TARGET_S) & (tr <= TARGET_S + 120)
    wt = np.bincount(tc[near], minlength=len(cells)).astype(float)

    def row_of(a, b):  # pool row nearest a saved site (the saved CSVs round coordinates)
        j = int(np.argmin((lat - a) ** 2 + (lon - b) ** 2))
        assert abs(lat[j] - a) + abs(lon[j] - b) < 1e-5
        return j
    layouts = {}
    for name, e in (("moves_10", "move"), ("add_5_move", "move"), ("add_5_new", "new")):
        L = pd.read_csv(PREV / f"{name}.csv")
        layouts[name] = ([row_of(r.lat, r.lon) for r in L.itertuples()], e)

    f = count_fn(m)
    print(f"2024 target calls over 360 s (smoothed) today: {f(today):.0f}", flush=True)
    idx = today
    for k in MOVE_LIMITS:
        t0 = time.time()
        idx, val = TA.search(A["lgbm"], w, wt, m, idx, ex_rows, k, ok_fn(m), f, log=f"count_moves_{k}")
        layouts[f"count_moves_{k}"] = (list(idx), "move")
        print(f"    {time.time() - t0:.0f} s, smoothed count {val:.0f}", flush=True)

    for extra in ("move", "new"):
        ma = busy_model("lgbm", extra)
        fa, oka = count_fn(ma), ok_fn(ma)
        idx = list(today)
        cand = np.setdiff1d(np.arange(len(pool)), ex_rows)
        for n in range(1, ADD_N + 1):
            vals = np.array([fa(idx + [int(o)]) for o in cand])
            for j in np.argsort(vals):  # best addition that keeps the protections
                if oka(idx + [int(cand[j])]):
                    break
            else:
                print(f"  count_add_{n}_{extra}: no addition keeps the protections", flush=True)
                break
            idx = idx + [int(cand[j])]
            print(f"  count_add_{n}_{extra}: smoothed count {vals[j]:.0f}", flush=True)
            cand = np.delete(cand, j)
        layouts[f"count_add_{len(idx) - len(today)}_{extra}"] = (list(idx), extra)

    # Score on 2026 calls (recorded + modelled change), three models.
    groups = {"targets": np.isin(boroughs[c26], targets), "london": np.ones(len(ev), bool),
              **{b.title(): boroughs[c26] == b for b in targets}}
    rows = [{"layout": "today", "scorer": "-", **{f"over6_{g}": int((rec26[mk] > TARGET_S).sum()) for g, mk in groups.items()}}]
    for sc in TA.SCORERS:
        for extra in ("move", "new"):
            mm = busy_model(sc, extra)
            q0 = mm.response(A[sc], today)
            bt0 = TA.borough_means(q0, n26, boroughs, names)
            for name, (lay, e) in layouts.items():
                if e != extra:
                    continue
                q = mm.response(A[sc], lay)
                after = rec26 + (q - q0)[c26]
                bd = TA.borough_means(q, n26, boroughs, names) - bt0
                tpos = [names.index(b) for b in targets]
                nt = [i for i in range(len(names)) if i not in tpos]
                rows.append({"layout": name, "scorer": sc, "moved": len(set(ex_rows) - set(lay)), "opened": len(set(lay) - set(ex_rows)),
                             **{f"over6_{g}": int((after[mk] > TARGET_S).sum()) for g, mk in groups.items()},
                             "d_target_mean_s": round(float(n26[np.isin(boroughs, targets)] @ (q - q0)[np.isin(boroughs, targets)] / n26[np.isin(boroughs, targets)].sum()), 2),
                             "d_london_mean_s": round(float(n26 @ (q - q0) / n26.sum()), 2),
                             "worst_other_borough_s": round(float(bd[nt].max()), 1), "worst_other_borough": names[nt[int(bd[nt].argmax())]]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scores.csv", index=False)
    for name, (lay, _) in layouts.items():
        pool.iloc[lay][["site_name", "lat", "lon", "is_existing"]].to_csv(OUT / f"{name}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
