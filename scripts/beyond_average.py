#!/usr/bin/env python3
"""Results beyond the average (eMentor meeting 2): calls over 6 minutes, slowest 10 % / 5 %, percentiles, without outliers.

Run after Part 2. Nothing here changes Part 2's fixed results. Uses the same 2026 calls as scripts/final_test_2026.py
(Jan to 31 Jul 2026; the Aug to Dec one-shot test data is not read), the same fixed layouts and three travel models.

Each call's "after" time = its recorded first-engine time + the modelled change for its 0.9 km cell under the layout
(the same "recorded + modelled change" reading as results.md 5.8). So the "before" numbers are real, and only the change is modelled.
Limitation: every call in a cell gets the same change, so this does not model which individual calls get faster.

Writes data/figures/placement/beyond_average/.
Run:  PYTHONPATH=src .venv/bin/python scripts/beyond_average.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import final_test_2026 as F  # noqa: E402

from firestation.busy_placement import ErlangBusyModel  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid, posture_seconds  # noqa: E402
from firestation.fire_station_location import (build_travel_matrix, busy_prob_vector, drop_far_incidents, fit_scorer_kolesar,  # noqa: E402
                                               recorded_station_turnout, snap_to_cells)
from firestation.ml_scorer import cell_boroughs  # noqa: E402
from firestation.station_capacity import default_pump_table  # noqa: E402

OUT = ROOT / "data" / "figures" / "placement" / "beyond_average"
LAYOUTS = ["rule_b_target_10", "rule_b_target_20", "citywide_10"]
YEAR = 2026


def stats(t: np.ndarray) -> dict:
    s = np.sort(t)
    k10, k5 = max(1, int(round(len(s) * 0.10))), max(1, int(round(len(s) * 0.05)))
    return {"calls": len(s), "over_6min": int((s > 360).sum()), "over_6min_pct": round(float((s > 360).mean() * 100), 2),
            "over_8min": int((s > 480).sum()), "over_10min": int((s > 600).sum()),
            "mean_s": round(float(s.mean()), 1), "median_s": round(float(np.median(s)), 1),
            "p90_s": round(float(np.percentile(s, 90)), 1), "p95_s": round(float(np.percentile(s, 95)), 1),
            "slowest10_mean_s": round(float(s[-k10:].mean()), 1), "slowest5_mean_s": round(float(s[-k5:].mean()), 1),
            "mean_without_slowest5_s": round(float(s[:-k5].mean()), 1)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = get_city("london")
    houses = load_fire_stations(spec)
    inc = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024, YEAR]), houses)
    demand = inc[inc["cal_year"] == 2024].reset_index(drop=True)
    ev = F.up_to(inc[inc["cal_year"] == YEAR], YEAR).reset_index(drop=True)
    cells = build_demand_grid(demand, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    w = cells["n_incidents"].to_numpy(float)
    boroughs = np.asarray(cell_boroughs(cells, demand)).astype(str)
    cidx, _ = snap_to_cells(ev, cells)
    call_b = boroughs[cidx]
    rec = ev["travel_seconds"].to_numpy(float)
    groups = {"5 target boroughs": np.isin(call_b, F.TARGETS), "London": np.ones(len(ev), bool),
              **{b.title(): call_b == b for b in F.TARGETS}}
    print(f"{YEAR} calls (to {F.LAST_DAY[YEAR]}): {len(ev)}, in target boroughs {groups['5 target boroughs'].sum()}", flush=True)

    pumps_today = default_pump_table(houses).set_index("station")["n_pumps"].reindex(houses["facilityname"].astype(str)).fillna(1).astype(int).to_numpy()
    tab, new_turn = recorded_station_turnout([2024])
    norm = {k.upper(): v for k, v in tab.items()}
    kmodel, _, _ = fit_scorer_kolesar(demand, houses)
    layouts = {"today": pd.DataFrame({"site_name": houses["facilityname"].astype(str), "lat": houses["lat"], "lon": houses["lon"],
                                      "is_existing": True, "n_pumps": pumps_today})}
    for k in LAYOUTS:
        layouts[k] = pd.read_csv(F.LAYOUTS[k])[["site_name", "lat", "lon", "is_existing", "n_pumps"]]
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

    out = [{"layout": "today (recorded)", "scorer": "-", "group": g, **stats(rec[m])} for g, m in groups.items()]
    for sc in F.SCORERS:
        T, _ = build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer=sc, kolesar=kmodel, incidents=demand)
        A = T + turnout[:, None]
        bp = np.ones(len(sites), int)
        bp[rows["today"]] = pumps["today"]
        m = ErlangBusyModel.calibrate(A, rows["today"], measured, bp, w)

        def resp(k):
            m.pumps = np.ones(len(sites), int)
            m.pumps[rows[k]] = pumps[k]
            return posture_seconds(A, rows[k], m.probs(A, rows[k]))

        q0 = resp("today")
        for k in LAYOUTS:
            after = rec + (resp(k) - q0)[cidx]
            out += [{"layout": k, "scorer": sc, "group": g, **stats(after[mk])} for g, mk in groups.items()]
        print("done", sc, flush=True)
    df = pd.DataFrame(out)
    df.to_csv(OUT / "stats.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
