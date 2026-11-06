#!/usr/bin/env python3
"""Improve the layouts from move10_limit.py under the full busy model, then refine new sites to street junctions.

The exact solver holds busy rates fixed in each round, so its layout need not be the best once they are recalculated. No floors and no cap here. Wider site list.
Also starts Move 10 from the planner's sites nearest to the layout in data/figures/placement/recommended/replace_10, to check the search does not depend on its start.

Writes data/figures/placement/move10_limit/polished.csv.

Run:  PYTHONPATH=src .venv/bin/python scripts/move10_polish.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from firestation.busy_placement import ErlangBusyModel, PumpPlan, extend_busy_model  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid, haversine_km, posture_seconds, weighted_cvar  # noqa: E402
from firestation.fire_station_location import (build_house_candidates, build_travel_matrix, busy_prob_vector, drop_far_incidents,  # noqa: E402
                                               fit_scorer_kolesar, recorded_station_turnout, snap_to_cells)
from firestation.site_refine import refine_sites  # noqa: E402
from firestation.station_capacity import default_pump_table  # noqa: E402
from placement_methods import search  # noqa: E402

OUT = ROOT / "data" / "figures" / "placement" / "move10_limit"


def main() -> int:
    spec = get_city("london")
    houses = load_fire_stations(spec)
    incidents = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024, 2025]), houses)
    demand = incidents[incidents["cal_year"] == 2024].reset_index(drop=True)
    ev = incidents[incidents["cal_year"] == 2025].reset_index(drop=True)
    cells = build_demand_grid(demand, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    w = cells["n_incidents"].to_numpy(float)
    c25, _ = snap_to_cells(ev, cells)
    n25 = np.bincount(c25, minlength=len(cells)).astype(float)
    pumps_today = default_pump_table(houses).set_index("station")["n_pumps"].reindex(houses["facilityname"].astype(str)).fillna(1).astype(int).to_numpy()
    tab, new_turn = recorded_station_turnout([2024])
    norm = {k.upper(): v for k, v in tab.items()}
    kmodel, _, _ = fit_scorer_kolesar(demand, houses)
    sites = build_house_candidates(houses, cells, demand_candidates=300, spread_km=1.5)
    ex = np.flatnonzero(sites["is_existing"].to_numpy())
    names = sites["site_name"].str.upper().to_numpy()
    turnout = np.array([norm.get(x, new_turn) if e else new_turn for x, e in zip(names, sites["is_existing"])])
    T, _ = build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer="lgbm", kolesar=kmodel, incidents=demand)
    A = T + turnout[:, None]
    lat, lon = sites["lat"].to_numpy(float), sites["lon"].to_numpy(float)
    base = np.ones(len(sites), int)
    base[ex] = pumps_today
    measured = busy_prob_vector(sites, demand)
    move = ErlangBusyModel.calibrate(A, ex, measured, base, w, PumpPlan(base, ex, lat, lon, extra="move"))
    add = ErlangBusyModel.calibrate(A, ex, measured, base, w, PumpPlan(base, ex, lat, lon, extra="new", new_site_pumps=2))
    today = ex.tolist()
    r0 = move.response(A, today)
    W = w.sum()

    def load(name):
        d = pd.read_csv(OUT / "layouts" / name)
        out = []
        for _, x in d.iterrows():
            dist = haversine_km(lon, lat, x["lon"], x["lat"])
            out.append(int(dist.argmin()))
        return out

    # The planner's Move 10: closed stations by name, opened sites at their nearest candidates.
    rec = ROOT / "data" / "figures" / "placement" / "recommended" / "replace_10"
    closed = {str(x).upper() for x in pd.read_csv(rec / "closed_fire_stations.csv")["site_name"]}
    op = pd.read_csv(rec / "opened_fire_stations.csv")
    new = np.flatnonzero(~sites["is_existing"].to_numpy())
    planner = [i for i in today if names[i] not in closed] + [int(new[haversine_km(lon[new], lat[new], a, b).argmin()]) for a, b in zip(op["lon"], op["lat"])]

    rows = []

    def report(label, idx, model, Ax=A, cand=sites, m_full=None):
        m = m_full or model
        r = m.response(Ax, idx)
        d = r - r0
        N = n25.sum()
        row = {"plan": label, "moved": len(set(today) - set(int(i) for i in idx)), "avg_2024": round(float(w @ d / W), 2), "avg_2025": round(float(n25 @ d / N), 2),
               "slowest10": round(weighted_cvar(r, n25) - weighted_cvar(r0, n25), 2), "faster_30s_pct": round(float(n25[d <= -30].sum() / N * 100), 1),
               "slower_30s_pct": round(float(n25[d >= 30].sum() / N * 100), 1), "slower_120s_pct": round(float(n25[d >= 120].sum() / N * 100), 2)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "polished.csv", index=False)
        print(row, flush=True)

    def make_rows(df):
        Tn, _ = build_travel_matrix(df, cells, surrogate=TravelTimeSurrogate(), scorer="lgbm", kolesar=kmodel, incidents=demand)
        return Tn + new_turn

    def refined(label, idx, model):
        n0 = len(sites)

        def obj(Ax, i, cand):
            m = extend_busy_model(model, cand["lat"].to_numpy(float)[n0:], cand["lon"].to_numpy(float)[n0:])
            return float(w @ m.response(Ax, i) / W)

        opened = [i for i in idx if i not in set(today)]
        ref = refine_sites(A, sites, cells, list(idx), opened, w, make_rows=make_rows, objective=obj, cap_ok=lambda *a: True, radius_km=1.0)
        c = ref["candidates"]
        m = extend_busy_model(model, c["lat"].to_numpy(float)[n0:], c["lon"].to_numpy(float)[n0:])
        report(label, ref["indices"], model, Ax=ref["A"], m_full=m)

    starts = {"exact free": load("wide_free.csv"), "planner's Move 10": planner, "today": today}
    best, best_v = None, np.inf
    for label, st in starts.items():
        report(f"start: {label}", st, move)
        idx, v = search(A, A, w, move, st, ex, 10)
        report(f"improved from: {label}", idx, move)
        if v < best_v:
            best, best_v = idx, v
    refined("best Move 10 + street refinement", best, move)
    a0 = load("wide_add10.csv")
    report("add 10: exact", a0, add)
    ai, _ = search(A, A, w, add, a0, ex, 0)
    report("add 10: improved", ai, add)
    refined("add 10: improved + street refinement", ai, add)
    print(sites["site_name"].iloc[sorted(set(today) - set(best))].tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
