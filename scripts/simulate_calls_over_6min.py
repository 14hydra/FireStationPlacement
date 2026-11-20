#!/usr/bin/env python3
"""Exploratory (2026-11-19): recount calls over 6 minutes with the street PDF simulator instead of recorded + modelled change.

An independent check on data/figures/placement/beyond_average/ (results.md 5.9). There, a call's time after a change is its
recorded time + the LightGBM modelled change for its 0.9 km cell. Here every call is simulated from scratch, today and
under each layout, with the street PDF simulator (scripts/explore_street_pdf_model.py, saved as
data/processed/models/street_pdf_sim.joblib):

  responder   the open station with the shortest simulator mean time (+ turnout) to the call's own location; with the
              Erlang busy chance of that station (the same busy model as Part 2) the second-nearest answers instead
  drive       that station's street-route time to the call x a ratio drawn from the simulator's spread for the call's
              group (rush hour x first engine from home x route length)
  turnout     the station's 2024 median turnout (new sites: London median) + a drawn deviation from 2024 records

200 draws per call; today and each layout use the same random numbers (paired), so the change is not swamped by noise.
Same 2026 calls as Part 2 (1 January to 31 July). Nothing in the pipeline is changed.

Writes data/figures/placement/simulated_over_6min/ (summary.json, counts.csv).
Run:  PYTHONPATH=src .venv/bin/python scripts/simulate_calls_over_6min.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import final_test_2026 as F  # noqa: E402

from firestation import ml_scorer  # noqa: E402
from firestation.busy_placement import ErlangBusyModel  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid  # noqa: E402
from firestation.fire_station_location import (build_travel_matrix, busy_prob_vector, drop_far_incidents, fit_scorer_kolesar,  # noqa: E402
                                               recorded_station_turnout, snap_to_cells)
from firestation.ml_scorer import cell_boroughs  # noqa: E402
from firestation.route_features import RouteFeatureRouter  # noqa: E402
from firestation.station_capacity import default_pump_table  # noqa: E402
from firestation.street_speeds import StreetSpeedModel, edge_design  # noqa: E402
from firestation.travel_time_model import DRIVE_CLIP_S  # noqa: E402

OUT = ROOT / "data" / "figures" / "placement" / "simulated_over_6min"
LAYOUTS = ["rule_b_target_10", "rule_b_target_20", "citywide_10"]
K = 200
TARGET_S = 360.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = get_city("london")
    houses = load_fire_stations(spec)
    inc = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024, 2026]), houses)
    demand = inc[inc["cal_year"] == 2024].reset_index(drop=True)
    ev = F.up_to(inc[inc["cal_year"] == 2026], 2026).reset_index(drop=True)
    cells = build_demand_grid(demand, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    w = cells["n_incidents"].to_numpy(float)
    boroughs = np.asarray(cell_boroughs(cells, demand)).astype(str)
    cidx, _ = snap_to_cells(ev, cells)
    call_b = boroughs[cidx]
    rec = ev["travel_seconds"].to_numpy(float)
    groups = {"5 target boroughs": np.isin(call_b, F.TARGETS), "London": np.ones(len(ev), bool),
              **{b.title(): call_b == b for b in F.TARGETS}}

    # Sites and layouts exactly as scripts/final_test_2026.py
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

    # Busy chance per site, same Erlang model as Part 2 (LightGBM cell matrix)
    Tl, _ = build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer="lgbm", kolesar=kmodel, incidents=demand)
    Al = Tl + turnout[:, None]
    bp = np.ones(len(sites), int)
    bp[rows["today"]] = pumps["today"]
    m = ErlangBusyModel.calibrate(Al, rows["today"], busy_prob_vector(sites, demand), bp, w)
    busy = {}
    for k in layouts:
        m.pumps = np.ones(len(sites), int)
        m.pumps[rows[k]] = pumps[k]
        busy[k] = m.probs(Al, rows[k])

    # Simulator mean drive, site x call, along each site's quickest base route with each street's own offset
    sim = joblib.load(ROOT / "data" / "processed" / "models" / "street_pdf_sim.joblib")
    router = ml_scorer._router()
    sm = StreetSpeedModel.load()
    t_base = sm.edge_times(router, edge_design(router))
    t_adj = t_base * (1 + sim["street_offsets"].astype(float))
    e = router.edge_id.tocoo()
    n = router.csr.shape[0]
    g = csr_matrix((t_base[e.data - 1], (e.row, e.col)), shape=(n, n))
    nodes = np.arange(n)
    s_idx, s_off = router.snap(sites["lat"].to_numpy(), sites["lon"].to_numpy())
    d_idx, d_off = router.snap(ev["dest_lat"].to_numpy(), ev["dest_lon"].to_numpy())
    D = np.empty((len(sites), len(ev)), np.float32)
    for i, o in enumerate(s_idx):
        _, pred = dijkstra(g, directed=True, indices=o, return_predecessors=True)
        has = pred >= 0
        vals = np.zeros((n, 1))
        vals[has, 0] = t_adj[np.asarray(router.edge_id[pred[has], nodes[has]]).ravel() - 1]
        S = RouteFeatureRouter._ancestor_sums(pred, vals)[:, 0]
        D[i] = np.clip(sm.coef[0] + sm.coef[1] * (s_off[i] + d_off) + S[d_idx], *DRIVE_CLIP_S)
    print("simulator routes done", flush=True)

    # Paired random numbers
    rng = np.random.default_rng(0)
    U_busy = rng.random((len(ev), K), dtype=np.float32)
    U_ratio = rng.random((len(ev), K), dtype=np.float32)
    tdev = np.sort(sim["turnout_dev_s"])
    T_dev = tdev[rng.integers(0, len(tdev), (len(ev), K))]
    pools = {k: np.sort(v) for k, v in sim["pools"].items()}
    allpool = np.sort(np.concatenate(list(pools.values())))
    nd, dedges = sim["n_dist"], sim["dist_edges"]
    h = pd.to_numeric(ev["hour_of_call"], errors="coerce").fillna(12).to_numpy()
    rush = (((h >= 7) & (h <= 9)) | ((h >= 16) & (h <= 19))).astype(int)

    def simulate(k: str) -> np.ndarray:
        idx = rows[k]
        A = D[idx] + turnout[idx, None]
        two = np.argpartition(A, 1, axis=0)[:2]
        cols = np.arange(A.shape[1])
        swap = A[two[0], cols] > A[two[1], cols]
        first, second = np.where(swap, two[1], two[0]), np.where(swap, two[0], two[1])
        p = busy[k][idx[first]]
        use2 = U_busy < p[:, None]
        site = np.where(use2, idx[second][:, None], idx[first][:, None])
        drive = D[site, cols[:, None]]
        reg = rush[:, None] * 2 * nd + (~use2).astype(int) * nd + np.clip(np.searchsorted(dedges, drive) - 1, 0, nd - 1)
        ratio = np.empty_like(drive)
        for r in np.unique(reg):
            mk = reg == r
            pl = pools.get(int(r), allpool)
            ratio[mk] = pl[np.minimum((U_ratio[mk] * len(pl)).astype(int), len(pl) - 1)]
        return turnout[site] + T_dev + np.clip(drive * ratio, *DRIVE_CLIP_S)

    sims = {k: simulate(k) for k in layouts}
    rows_out, summary = [], {"doc": __doc__, "calls": int(len(ev)), "draws": K}
    for gname, mk in groups.items():
        over0 = (sims["today"][mk] > TARGET_S).sum(0)
        base = {"group": gname, "recorded_over_6min": int((rec[mk] > TARGET_S).sum()),
                "simulated_today_over_6min": round(float(over0.mean()), 0),
                "recorded_mean_s": round(float(rec[mk].mean()), 1), "simulated_today_mean_s": round(float(sims["today"][mk].mean()), 1)}
        for k in LAYOUTS:
            ov = (sims[k][mk] > TARGET_S).sum(0)
            d = ov - over0
            rows_out.append({**base, "layout": k, "simulated_over_6min": round(float(ov.mean()), 0),
                             "change": round(float(d.mean()), 0), "change_lo": int(np.percentile(d, 2.5)), "change_hi": int(np.percentile(d, 97.5)),
                             "change_pct": round(float(d.mean() / over0.mean() * 100), 1),
                             "mean_change_s": round(float((sims[k][mk] - sims["today"][mk]).mean()), 1)})
    df = pd.DataFrame(rows_out)
    df.to_csv(OUT / "counts.csv", index=False)
    summary["rows"] = rows_out
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
