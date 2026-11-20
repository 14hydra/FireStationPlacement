#!/usr/bin/env python3
"""Exploratory (2026-11-20): street PDF simulator as a second, independent check on every threshold result, plus
chance-of-arrival maps.

Extends scripts/simulate_calls_over_6min.py (same simulator, same busy model, same 2026 calls, same paired draws) to
every layout in results.md 5.9:

  rule_b_target_10 (main), rule_b_target_20, citywide_10           pumps as saved with the layout (Part 2)
  count_moves_10, count_moves_20                                    pumps move with their station (PumpPlan)
  add_5_move, count_add_5_move                                      new stations take an engine from elsewhere
  add_5_new, count_add_5_new                                        new stations get a new engine

For each it counts calls over 6, 8 and 10 minutes (5 target boroughs, London, each target borough), today and after,
and sets them beside the recorded + modelled-change numbers (beyond_average/stats.csv, calls_under_6min/scores.csv).

Maps: per 0.9 km cell, the chance the first engine arrives within 6 minutes (call to arrival) = share of the cell's
2026 calls x 200 draws under 360 s; today, main layout, and the change.

Writes data/figures/placement/simulated_thresholds/ (counts.csv, comparison.csv, cell_chance.csv, summary.json,
chance_within_6min_map.png). Caches the site x call simulator times in data/processed/sim_cache/.
Run:  PYTHONPATH=src .venv/bin/python scripts/simulate_threshold_checks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import final_test_2026 as F  # noqa: E402

from firestation import ml_scorer  # noqa: E402
from firestation.busy_placement import ErlangBusyModel, PumpPlan  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid  # noqa: E402
from firestation.fire_station_location import (build_travel_matrix, busy_prob_vector, drop_far_incidents, fit_scorer_kolesar,  # noqa: E402
                                               recorded_station_turnout, snap_to_cells)
from firestation.ml_scorer import cell_boroughs  # noqa: E402
from firestation.route_features import RouteFeatureRouter  # noqa: E402
from firestation.station_capacity import default_pump_table  # noqa: E402
from firestation.street_speeds import StreetSpeedModel, edge_design  # noqa: E402
from firestation.travel_time_model import DRIVE_CLIP_S  # noqa: E402

OUT = ROOT / "data" / "figures" / "placement" / "simulated_thresholds"
CACHE = ROOT / "data" / "processed" / "sim_cache" / "site_call_drive.npz"
CU6 = ROOT / "data" / "figures" / "placement" / "calls_under_6min"
BA = ROOT / "data" / "figures" / "placement" / "beyond_average" / "stats.csv"
SAVED = ["rule_b_target_10", "rule_b_target_20", "citywide_10"]
COUNT = {"count_moves_10": "move", "count_moves_20": "move", "add_5_move": "move", "count_add_5_move": "move",
         "add_5_new": "new", "count_add_5_new": "new"}
LABEL = {"rule_b_target_10": "Main layout, Move 10 (5.8)", "rule_b_target_20": "Rule B targeted, up to 20",
         "citywide_10": "Citywide Move 10", "count_moves_10": "Move 10, aimed at the call count",
         "count_moves_20": "Move 20, aimed at the call count", "add_5_move": "Add 5, engines from elsewhere, averages",
         "count_add_5_move": "Add 5, engines from elsewhere, call count", "add_5_new": "Add 5, new engines, averages",
         "count_add_5_new": "Add 5, new engines, call count"}
K = 200
THRESH = {"over_6min": 360.0, "over_8min": 480.0, "over_10min": 600.0}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
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

    # Sites: today's stations first, then every new site of every layout
    pumps_today = default_pump_table(houses).set_index("station")["n_pumps"].reindex(houses["facilityname"].astype(str)).fillna(1).astype(int).to_numpy()
    tab, new_turn = recorded_station_turnout([2024])
    norm = {k.upper(): v for k, v in tab.items()}
    kmodel, _, _ = fit_scorer_kolesar(demand, houses)
    frames = {"today": pd.DataFrame({"site_name": houses["facilityname"].astype(str), "lat": houses["lat"], "lon": houses["lon"],
                                     "is_existing": True, "n_pumps": pumps_today})}
    for k in SAVED:
        frames[k] = pd.read_csv(F.LAYOUTS[k])[["site_name", "lat", "lon", "is_existing", "n_pumps"]]
    for k in COUNT:
        frames[k] = pd.read_csv(CU6 / f"{k}.csv")[["site_name", "lat", "lon", "is_existing"]]
    key = lambda r: str(r.site_name).upper() if bool(r.is_existing) else f"{float(r.lat):.5f},{float(r.lon):.5f}"  # noqa: E731
    sites, index = [], {}
    for name, L in frames.items():
        for r in L.itertuples():
            if key(r) not in index:
                assert name == "today" or not bool(r.is_existing), f"unknown existing station {r.site_name}"
                index[key(r)] = len(sites)
                sites.append({"site_name": str(r.site_name), "lat": float(r.lat), "lon": float(r.lon), "is_existing": bool(r.is_existing)})
    sites = pd.DataFrame(sites)
    n_ex = len(houses)
    assert sites["is_existing"].iloc[:n_ex].all() and not sites["is_existing"].iloc[n_ex:].any()
    rows = {k: np.array([index[key(r)] for r in L.itertuples()]) for k, L in frames.items()}
    turnout = np.array([norm.get(r.site_name.upper(), new_turn) if r.is_existing else new_turn for r in sites.itertuples()])
    lat, lon = sites["lat"].to_numpy(float), sites["lon"].to_numpy(float)

    base = np.ones(len(sites), int)
    base[:n_ex] = pumps_today
    pumps = {}
    for k, L in frames.items():
        if k in COUNT:
            pumps[k] = PumpPlan(base, np.arange(n_ex), lat, lon, extra=COUNT[k]).assign(rows[k])[0][rows[k]]
        else:
            pumps[k] = np.maximum(L["n_pumps"].fillna(1).astype(int).to_numpy(), 1)
    for k in SAVED:  # sanity: Part 2's saved pumps vs PumpPlan
        pp = PumpPlan(base, np.arange(n_ex), lat, lon, extra="move").assign(rows[k])[0][rows[k]]
        print(f"{k}: saved pumps match PumpPlan on {np.mean(pp == pumps[k]):.0%} of stations", flush=True)

    # Busy chance per site, same Erlang model as Part 2 (LightGBM cell matrix)
    Tl, _ = build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer="lgbm", kolesar=kmodel, incidents=demand)
    Al = Tl + turnout[:, None]
    m = ErlangBusyModel.calibrate(Al, rows["today"], busy_prob_vector(sites, demand), base, w)
    busy = {}
    for k in frames:
        m.pumps = np.ones(len(sites), int)
        m.pumps[rows[k]] = pumps[k]
        busy[k] = m.probs(Al, rows[k])

    # Simulator mean drive, site x call (cached by site coordinates and call count)
    sim = joblib.load(ROOT / "data" / "processed" / "models" / "street_pdf_sim.joblib")
    sm = StreetSpeedModel.load()
    sig = np.round(np.c_[lat, lon], 6)
    D = np.full((len(sites), len(ev)), np.nan, np.float32)
    if CACHE.exists():
        c = np.load(CACHE)
        if int(c["n_calls"]) == len(ev):
            have = {tuple(s): i for i, s in enumerate(c["sig"])}
            for i, s in enumerate(sig):
                if tuple(s) in have:
                    D[i] = c["D"][have[tuple(s)]]
    todo = np.flatnonzero(np.isnan(D[:, 0]))
    print(f"{len(sites)} sites, {len(todo)} to route", flush=True)
    if len(todo):
        router = ml_scorer._router()
        t_base = sm.edge_times(router, edge_design(router))
        t_adj = t_base * (1 + sim["street_offsets"].astype(float))
        e = router.edge_id.tocoo()
        n = router.csr.shape[0]
        g = csr_matrix((t_base[e.data - 1], (e.row, e.col)), shape=(n, n))
        nodes = np.arange(n)
        s_idx, s_off = router.snap(lat, lon)
        d_idx, d_off = router.snap(ev["dest_lat"].to_numpy(), ev["dest_lon"].to_numpy())
        for j, i in enumerate(todo):
            _, pred = dijkstra(g, directed=True, indices=s_idx[i], return_predecessors=True)
            has = pred >= 0
            vals = np.zeros((n, 1))
            vals[has, 0] = t_adj[np.asarray(router.edge_id[pred[has], nodes[has]]).ravel() - 1]
            S = RouteFeatureRouter._ancestor_sums(pred, vals)[:, 0]
            D[i] = np.clip(sm.coef[0] + sm.coef[1] * (s_off[i] + d_off) + S[d_idx], *DRIVE_CLIP_S)
            if j % 20 == 0:
                print(f"  routed {j + 1}/{len(todo)}", flush=True)
        np.savez(CACHE, D=D, sig=sig, n_calls=len(ev))

    # Paired random numbers (identical seed and order to simulate_calls_over_6min.py)
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

    # Per layout: per-draw counts by group and threshold, and per-call chance within 6 minutes
    counts, chance = {}, {}
    for k in frames:
        s = simulate(k)
        counts[k] = {(g, t): (s[mk] > v).sum(0) for g, mk in groups.items() for t, v in THRESH.items()}
        chance[k] = (s <= 360.0).mean(1)
        print(f"simulated {k}", flush=True)

    out = []
    for k in frames:
        if k == "today":
            continue
        for g in groups:
            for t in THRESH:
                c0, c1 = counts["today"][(g, t)], counts[k][(g, t)]
                d = c1 - c0
                out.append({"layout": k, "group": g, "threshold": t, "recorded_today": int((rec[groups[g]] > THRESH[t]).sum()),
                            "simulated_today": round(float(c0.mean())), "simulated_after": round(float(c1.mean())),
                            "change": round(float(d.mean())), "change_lo": int(np.percentile(d, 2.5)),
                            "change_hi": int(np.percentile(d, 97.5)), "change_pct": round(float(d.mean() / c0.mean() * 100), 1)})
    df = pd.DataFrame(out)
    df.to_csv(OUT / "counts.csv", index=False)

    # Side by side with recorded + modelled change (LightGBM / road speeds / Kolesar)
    ba = pd.read_csv(BA)
    sc = pd.read_csv(CU6 / "scores.csv")
    gcol = {"5 target boroughs": "over6_targets", "London": "over6_london", **{b.title(): f"over6_{b.title()}" for b in F.TARGETS}}
    comp = []
    for k in LABEL:
        for g in groups:
            for t in THRESH:
                r = df[(df.layout == k) & (df.group == g) & (df.threshold == t)].iloc[0]
                row = {"layout": LABEL[k], "group": g, "threshold": t, "sim_change": r.change, "sim_lo": r.change_lo, "sim_hi": r.change_hi}
                for scorer in ("lgbm", "road", "kolesar"):
                    v = np.nan
                    if k in SAVED:
                        b = ba[(ba.layout == k) & (ba.scorer == scorer) & (ba.group.str.lower() == g.lower())]
                        b0 = ba[(ba.scorer == "-") & (ba.group.str.lower() == g.lower())]
                        if len(b) and len(b0):
                            v = int(b[t].iloc[0] - b0[t].iloc[0])
                    elif t == "over_6min" and gcol[g] in sc:
                        b = sc[(sc.layout == k) & (sc.scorer == scorer)]
                        v = int(b[gcol[g]].iloc[0] - sc[sc.layout == "today"][gcol[g]].iloc[0])
                    row[scorer] = v
                comp.append(row)
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "comparison.csv", index=False)

    # Chance of arrival within 6 minutes per cell
    ncell = np.bincount(cidx, minlength=len(cells))
    clat = np.bincount(cidx, ev["dest_lat"].to_numpy(float), len(cells)) / np.maximum(ncell, 1)
    clon = np.bincount(cidx, ev["dest_lon"].to_numpy(float), len(cells)) / np.maximum(ncell, 1)
    lat0 = float(pd.to_numeric(demand["dest_lat"], errors="coerce").mean())  # grid of build_demand_grid
    dlat, dlon = 0.9 / 110.574, 0.9 / (111.320 * np.cos(np.radians(lat0)))
    cc = pd.DataFrame({"cell": np.arange(len(cells)), "ix": cells["_ix"].to_numpy(), "iy": cells["_iy"].to_numpy(),
                       "grid_lat": (cells["_iy"].to_numpy() + 0.5) * dlat, "grid_lon": (cells["_ix"].to_numpy() + 0.5) * dlon,
                       "borough": boroughs, "calls_2026": ncell, "lat": clat, "lon": clon,
                       "recorded_within_6min": np.bincount(cidx, (rec <= 360).astype(float), len(cells)) / np.maximum(ncell, 1)})
    for k in frames:
        cc[f"chance_{k}"] = np.bincount(cidx, chance[k], len(cells)) / np.maximum(ncell, 1)
    cc = cc[cc.calls_2026 > 0]
    cc.round(4).to_csv(OUT / "cell_chance.csv", index=False)

    summary = {"doc": __doc__, "calls": int(len(ev)), "draws": K,
               "chance_within_6min": {g: {k: round(float(chance[k][mk].mean()), 4) for k in frames} | {"recorded": round(float((rec[mk] <= 360).mean()), 4)}
                                      for g, mk in groups.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    plot_maps(cc, sites, rows, summary, n_ex)

    pd.set_option("display.width", 250)
    print(comp[comp.threshold == "over_6min"].to_string(index=False))
    print(comp[(comp.threshold != "over_6min") & comp.group.isin(["5 target boroughs", "London"])].to_string(index=False))
    print(json.dumps(summary["chance_within_6min"], indent=1))
    return 0


def plot_maps(cc: pd.DataFrame, sites: pd.DataFrame, rows: dict, summary: dict, n_ex: int) -> None:
    cc = cc[cc.calls_2026 >= 5]
    blues = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    seq = ListedColormap(blues)
    seq_b = [0, .2, .35, .5, .65, .8, .9, 1.0]
    div = ListedColormap(["#a32d2c", "#e66767", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"])
    div_b = [-1, -.05, -.01, .01, .05, .15, 1]
    asp = 1 / np.cos(np.radians(51.5))
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.9), facecolor="#fcfcfb", layout="constrained")
    main = rows["rule_b_target_10"]
    today = set(rows["today"].tolist())
    closed = sorted(today - set(main.tolist()))
    opened = [i for i in main if i >= n_ex]
    s = summary["chance_within_6min"]["5 target boroughs"]
    panels = [("chance_today", seq, seq_b, f"Today (simulated)\n5 target boroughs: {s['today']:.0%} of calls within 6 min"),
              ("chance_rule_b_target_10", seq, seq_b, f"Main layout (Move 10)\n5 target boroughs: {s['rule_b_target_10']:.0%}"),
              (None, div, div_b, "Change in chance (percentage points)\nblue = more likely within 6 min")]
    for ax, (col, cmap, b, title) in zip(axes, panels):
        v = cc[col] if col else cc["chance_rule_b_target_10"] - cc["chance_today"]
        ix, iy = cc.ix.to_numpy() - cc.ix.min(), cc.iy.to_numpy() - cc.iy.min()
        grid = np.full((iy.max() + 1, ix.max() + 1), np.nan)
        grid[iy, ix] = v
        dlo = cc.grid_lon.iloc[0] / (cc.ix.iloc[0] + 0.5)
        dla = cc.grid_lat.iloc[0] / (cc.iy.iloc[0] + 0.5)
        xe = (np.arange(ix.max() + 2) + cc.ix.min()) * dlo
        ye = (np.arange(iy.max() + 2) + cc.iy.min()) * dla
        sc = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid), cmap=cmap, norm=BoundaryNorm(b, cmap.N), edgecolors="#fcfcfb", linewidth=0.15)
        ax.scatter(sites.lon.iloc[list(today)], sites.lat.iloc[list(today)], s=7, c="#3a3a38", linewidths=0, label="Station today")
        if col is None:
            ax.scatter(sites.lon.iloc[closed], sites.lat.iloc[closed], s=46, facecolors="none", edgecolors="#3a3a38", linewidths=1.3, label="Station moved away")
            ax.scatter(sites.lon.iloc[opened], sites.lat.iloc[opened], s=70, marker="*", c="#3a3a38", linewidths=0, label="New site")
            ax.legend(loc="lower left", fontsize=8.5, frameon=False)
        for bname in F.TARGETS:
            t = cc[cc.borough == bname]
            ax.text(t.lon.mean(), t.lat.mean(), bname.title(), ha="center", va="center", fontsize=9, color="#1f1f1d", weight="bold",
                    path_effects=[pe.withStroke(linewidth=3, foreground="#fcfcfb")])
        ax.set_aspect(asp)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, loc="left", color="#1f1f1d")
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02)
        cb.set_ticks(b[1:-1] if col is None else b)
        cb.set_ticklabels([f"{x * 100:+.0f}" for x in b[1:-1]] if col is None else [f"{x:.0%}" for x in b])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors="#5f5e5a")
    fig.suptitle("Chance the first engine arrives within 6 minutes (call to arrival), 2026 calls, street PDF simulator, 200 draws",
                 fontsize=12.5, x=0.01, ha="left", color="#1f1f1d")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    fig.supxlabel("Each square = a 0.9 km cell with 5 or more 2026 calls (1 Jan to 31 Jul). Chance = share of the cell's calls "
             "x draws within 360 s.", fontsize=8.5, color="#5f5e5a", x=0.01, ha="left")
    fig.savefig(OUT / "chance_within_6min_map.png", dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    sys.exit(main())
