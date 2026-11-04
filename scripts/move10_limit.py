#!/usr/bin/env python3
"""How large can the saving from moving only 10 stations get, with every protection removed?

Exact solver, busy engines, 2024 demand, scored on 2025 calls by the models. No slowest-10 % floor, no average floor, no borough floor, no 10-minute
cap, no street refinement. Plans:
  free            close any 10, open any 10
  quiet_any       must close the 10 stations with the fewest calls, open any 10
  quiet_central   must close the 10 stations with the fewest calls, open 10 within 8 km of the centre
  outer_central   close any 10 stations over 12 km from the centre, open 10 within 8 km of the centre
  add10           ceiling: keep all stations and add 10 with new engines. Closing stations can only cost time, so with these busy rates Move 10 cannot beat it

Writes data/figures/placement/move10_limit/: results.csv, layouts/.

Run:  PYTHONPATH=src .venv/bin/python scripts/move10_limit.py [--time-limit 900]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.busy_placement import ErlangBusyModel, PumpPlan, solve_busy_exact  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid, haversine_km, posture_seconds, weighted_cvar  # noqa: E402
from firestation.fire_station_location import (build_house_candidates, build_travel_matrix, busy_prob_vector, drop_far_incidents,  # noqa: E402
                                               fit_scorer_kolesar, recorded_station_turnout, snap_to_cells)
from firestation.station_capacity import default_pump_table  # noqa: E402

OUT = ROOT / "data" / "figures" / "placement" / "move10_limit"
POOLS = {"standard": {"demand_candidates": 80, "spread_km": 3.0}, "wide": {"demand_candidates": 300, "spread_km": 1.5}}
CENTRE = (-0.1276, 51.5074)  # Charing Cross
K = 10
OVER6 = (-1.026638838904118, 0.911532189667077)  # from placement_methods.py (2024 recorded calls)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=900.0)
    args = ap.parse_args()
    (OUT / "layouts").mkdir(parents=True, exist_ok=True)
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
    rows = []
    for pool, kw in POOLS.items():
        sites = build_house_candidates(houses, cells, **kw)
        ex = np.flatnonzero(sites["is_existing"].to_numpy())
        ex_set = set(ex.tolist())
        n = len(ex)
        names = sites["site_name"].str.upper().to_numpy()
        turnout = np.array([norm.get(x, new_turn) if e else new_turn for x, e in zip(names, sites["is_existing"])])
        As = {}
        for sc in ("lgbm", "road", "kolesar"):
            T, _ = build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer=sc, kolesar=kmodel, incidents=demand)
            As[sc] = T + turnout[:, None]
        A = As["lgbm"]
        lat, lon = sites["lat"].to_numpy(float), sites["lon"].to_numpy(float)
        base = np.ones(len(sites), int)
        base[ex] = pumps_today
        measured = busy_prob_vector(sites, demand)
        models = {sc: ErlangBusyModel.calibrate(As[sc], ex, measured, base, w, PumpPlan(base, ex, lat, lon, extra="move")) for sc in As}
        add_model = ErlangBusyModel.calibrate(A, ex, measured, base, w, PumpPlan(base, ex, lat, lon, extra="new", new_site_pumps=2))
        today = ex.tolist()
        r0 = {sc: posture_seconds(As[sc], today, models[sc].probs(As[sc], today)) for sc in As}
        km = haversine_km(lon, lat, CENTRE[0], CENTRE[1])
        calls = np.bincount(ex[A[ex].argmin(axis=0)], weights=w, minlength=len(sites))
        quiet = ex[np.argsort(calls[ex])[:K]].tolist()
        print(pool, len(sites), "sites. Fewest calls:", sites["site_name"].iloc[quiet].tolist(), [int(calls[i]) for i in quiet], flush=True)
        inner = [i for i in range(len(sites)) if i not in ex_set and km[i] <= 8]
        new = [i for i in range(len(sites)) if i not in ex_set]
        outer = [i for i in today if km[i] > 12]
        keep_q = [i for i in today if i not in quiet]
        keep_o = [i for i in today if i not in outer]
        plans = {
            "free": dict(k=n, allowed=today + new, min_from=(today, n - K)),
            "quiet_any": dict(k=n, allowed=keep_q + new, min_from=(keep_q, n - K)),
            "quiet_central": dict(k=n, allowed=keep_q + inner, min_from=(keep_q, n - K)),
            "outer_central": dict(k=n, allowed=today + inner, min_from=(today, n - K), fixed_open=keep_o),
            "add10": dict(k=n + K, allowed=today + new, min_from=(today, n)),
        }
        for name, p in plans.items():
            model = add_model if name == "add10" else models["lgbm"]
            k = p.pop("k")
            start = today if name in ("free", "outer_central", "add10") else keep_q + inner[:K]
            res = solve_busy_exact(A, w, k, model, start_idx=start, max_rounds=3, objective="response_time", cap_s=None, time_limit_s=args.time_limit, **p)
            idx = [int(i) for i in res["indices"]]
            r = model.response(A, idx)
            d = r - r0["lgbm"]
            N = n25.sum()
            row = {"pool": pool, "plan": name, "sites": len(sites), "moved": len(ex_set - set(idx)) if name != "add10" else K,
                   "d_mean_lgbm": round(float(n25 @ d / N), 2), "d_slowest10": round(weighted_cvar(r, n25) - weighted_cvar(r0["lgbm"], n25), 2)}
            if name != "add10":
                for sc in ("road", "kolesar"):
                    row[f"d_mean_{sc}"] = round(float(n25 @ (posture_seconds(As[sc], idx, models[sc].probs(As[sc], idx)) - r0[sc]) / N), 2)
            f = lambda x: 1 / (1 + np.exp(-(OVER6[0] + OVER6[1] * (x - 300) / 60)))  # noqa: E731
            row["over6_avoided_per_year"] = round(float(n25 @ (f(r0["lgbm"]) - f(r))))
            row["calls_faster_30s_pct"] = round(float(n25[d <= -30].sum() / N * 100), 1)
            row["calls_slower_30s_pct"] = round(float(n25[d >= 30].sum() / N * 100), 1)
            row["calls_slower_120s_pct"] = round(float(n25[d >= 120].sum() / N * 100), 2)
            row["worst_cell_change_s"] = round(float(d[n25 > 0].max()))
            row["cells_over_10min_pct_calls"] = round(float(n25[r > 600].sum() / N * 100), 2)
            row["cells_over_10min_today_pct_calls"] = round(float(n25[r0["lgbm"] > 600].sum() / N * 100), 2)
            rd = res["rounds"]
            row["optimal_all_rounds"] = all(x.get("optimal_fixed_p", False) for x in rd if "infeasible" not in x)
            row["gap"] = max((x.get("gap") or 0 for x in rd), default=None)
            row["closed"] = "; ".join(sites["site_name"].iloc[sorted(ex_set - set(idx))])
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
            sites.iloc[idx][["site_name", "lat", "lon", "is_existing"]].to_csv(OUT / "layouts" / f"{pool}_{name}.csv", index=False)
            print({a: b for a, b in row.items() if a != "closed"}, "\n  closed:", row["closed"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
