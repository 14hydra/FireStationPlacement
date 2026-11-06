#!/usr/bin/env python3
"""Rescore the wider-site planner layouts under three travel models and the land rules.

Layouts: data/figures/placement/recommended_wide/. Chosen on 2024 with LightGBM, scored on 2025 calls at their 0.9 km cells with busy engines and each layout's own
engines, as in score_layouts.py. The layouts are not solved again.

Land rules (placement_analysis.py, OpenStreetMap):
  A  not on excluded land (parks, water, schools, hospitals etc.), road within 100 m, main road within 300 m
  B  A + commercial, industrial, retail or brownfield plot of 2,000 m2+ within 300 m
  C  B + site itself not on residential land
For each rule, every new site that fails moves to the nearest unused candidate site that passes, keeping its engines. Nothing is re-optimised, and replacements get no
street refinement, so these scores are a cautious estimate of what the rule costs.

Writes data/figures/placement/recommended_wide/rescore/: scores.csv, sites.csv, repairs.csv, summary.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/score_wide_layouts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from firestation.busy_placement import ErlangBusyModel  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.dispatch import dispatch_response, load_dispatch  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid, haversine_km, posture_seconds, weighted_cvar  # noqa: E402
from firestation.fire_station_location import (build_house_candidates, build_travel_matrix, busy_prob_vector, drop_far_incidents,  # noqa: E402
                                               fit_scorer_kolesar, recorded_station_turnout, snap_to_cells)
from firestation.ml_scorer import cell_boroughs  # noqa: E402
from firestation.station_capacity import default_pump_table  # noqa: E402
from placement_analysis import classify  # noqa: E402

WIDE = ROOT / "data" / "figures" / "placement" / "recommended_wide"
OUT = WIDE / "rescore"
REC = ROOT / "data" / "figures" / "placement" / "recommended"
SCENARIOS = {"wide_10": WIDE / "replace_10_boroughs1s", "wide_20": WIDE / "replace_20_boroughs1s", "wide_30": WIDE / "replace_30_boroughs1s",
             "wide_46": WIDE / "replace_46_boroughs1s", "wide_redesign": WIDE / "redesign_boroughs1s",
             "standard_10": REC / "replace_10_boroughs1s", "standard_redesign": REC / "redesign_boroughs1s"}
RULES = ("none", "rule_a", "rule_b", "rule_c")
SCORERS = ("lgbm", "road", "kolesar")
BOOT = 1000


def add_rules(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["rule_a"] = ~t.excluded & t.road_100m & t.main_road_300m
    t["rule_b"] = t.rule_a & t.plot_300m
    t["rule_c"] = t.rule_b & ~t.residential
    t["none"] = True
    return t


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = get_city("london")
    houses = load_fire_stations(spec)
    incidents = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024, 2025]), houses)
    demand = incidents[incidents["cal_year"] == 2024].reset_index(drop=True)
    ev = incidents[incidents["cal_year"] == 2025].reset_index(drop=True)
    cells = build_demand_grid(demand, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    cidx, _ = snap_to_cells(ev, cells)
    counts = np.bincount(cidx, minlength=len(cells)).astype(float)
    N = counts.sum()
    load_w = cells["n_incidents"].to_numpy(float)
    boroughs = np.asarray(cell_boroughs(cells, demand)).astype(str)
    pumps_today = default_pump_table(houses).set_index("station")["n_pumps"].reindex(houses["facilityname"].astype(str)).fillna(1).astype(int).to_numpy()

    pool = build_house_candidates(houses, cells, demand_candidates=300, spread_km=1.5)
    pool = pool[~pool["is_existing"].to_numpy(bool)].reset_index(drop=True)
    plans = {n: pd.read_csv(p / "planned_fire_stations.csv") for n, p in SCENARIOS.items()}
    new_pts = pd.concat([p.loc[~p["is_existing"].astype(bool), ["lat", "lon"]] for p in plans.values()]).drop_duplicates().reset_index(drop=True)
    print(f"Classifying {len(pool)} candidates and {len(new_pts)} layout sites…", flush=True)
    both = pd.concat([pool[["lat", "lon"]], new_pts], ignore_index=True)
    tags = add_rules(classify(both["lat"].to_numpy(float), both["lon"].to_numpy(float)))
    pool_t, new_t = tags.iloc[:len(pool)].reset_index(drop=True), tags.iloc[len(pool):].reset_index(drop=True)
    new_t = pd.concat([new_pts, new_t], axis=1)
    key = lambda lat, lon: f"{float(lat):.6f},{float(lon):.6f}"  # noqa: E731
    tag_of = {key(r.lat, r.lon): r for r in new_t.itertuples()}

    today = pd.DataFrame({"site_name": houses["facilityname"].astype(str), "lat": houses["lat"], "lon": houses["lon"], "is_existing": True, "n_pumps": pumps_today})
    layouts = {("today", "none"): today}
    site_rows, repair_rows = [], []
    for name, p in plans.items():
        p = p[["site_name", "lat", "lon", "is_existing", "n_pumps"]].copy()
        p["is_existing"] = p["is_existing"].astype(bool)
        opened = p[~p["is_existing"]]
        row = {"layout": name, "new_sites": len(opened)}
        for c in ("excluded", "road_100m", "main_road_300m", "plot_300m", "residential", "rule_a", "rule_b", "rule_c"):
            row[c] = int(sum(bool(getattr(tag_of[key(r.lat, r.lon)], c)) for r in opened.itertuples()))
        site_rows.append(row)
        for rule in RULES:
            q = p.copy()
            ok = pool_t[rule].to_numpy(bool)
            used = {key(r.lat, r.lon) for r in opened.itertuples() if bool(getattr(tag_of[key(r.lat, r.lon)], rule))}
            km = []
            for i, r in opened.iterrows():
                if bool(getattr(tag_of[key(r.lat, r.lon)], rule)):
                    continue
                d = haversine_km(pool["lon"].to_numpy(float), pool["lat"].to_numpy(float), r.lon, r.lat)
                free = np.array([ok[j] and key(pool.lat[j], pool.lon[j]) not in used for j in range(len(pool))])
                j = int(np.where(free, d, np.inf).argmin())
                q.loc[i, ["lat", "lon"]] = [pool.lat[j], pool.lon[j]]
                q.loc[i, "site_name"] = f"Repaired site {j}"
                used.add(key(pool.lat[j], pool.lon[j]))
                km.append(float(d[j]))
            repair_rows.append({"layout": name, "rule": rule, "repaired": len(km), "median_km": round(float(np.median(km)), 2) if km else 0.0, "max_km": round(max(km), 2) if km else 0.0})
            layouts[(name, rule)] = q
    pd.DataFrame(site_rows).to_csv(OUT / "sites.csv", index=False)
    pd.DataFrame(repair_rows).to_csv(OUT / "repairs.csv", index=False)
    print(pd.DataFrame(site_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(repair_rows).to_string(index=False), flush=True)

    site_k = lambda r: str(r.site_name) if bool(r.is_existing) else key(r.lat, r.lon)  # noqa: E731
    sites, index = [], {}
    for L in layouts.values():
        for r in L.itertuples():
            k = site_k(r)
            if k not in index:
                index[k] = len(sites)
                sites.append({"site_name": str(r.site_name), "lat": float(r.lat), "lon": float(r.lon), "is_existing": bool(r.is_existing)})
    sites = pd.DataFrame(sites)
    rows = {n: np.array([index[site_k(r)] for r in L.itertuples()]) for n, L in layouts.items()}
    pumps = {n: np.maximum(L["n_pumps"].fillna(1).astype(int).to_numpy(), 1) for n, L in layouts.items()}
    tab, new_turn = recorded_station_turnout([2024])
    norm = {k.upper(): v for k, v in tab.items()}
    turnout = np.array([norm.get(r.site_name.upper(), new_turn) if r.is_existing else new_turn for r in sites.itertuples()])
    measured = busy_prob_vector(sites, demand)
    kmodel, _, _ = fit_scorer_kolesar(demand, houses)
    dmodel = load_dispatch()
    rng = np.random.default_rng(0)
    boot = rng.multinomial(int(N), counts / N, size=BOOT).astype(float)
    today_rows = rows[("today", "none")]
    t_b = sorted(set(boroughs))
    b_w = np.array([counts[boroughs == b].sum() for b in t_b])

    records = []
    for sc in SCORERS:
        print(f"Travel times: {sc} ({len(sites)} sites × {len(cells)} cells)…", flush=True)
        T, _ = build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer=sc, kolesar=kmodel, incidents=demand)
        A = T + turnout[:, None]
        base = np.ones(len(sites), int)
        base[today_rows] = pumps[("today", "none")]
        model = ErlangBusyModel.calibrate(A, today_rows, measured, base, load_w)

        def resp(n):
            model.pumps = np.ones(len(sites), int)
            model.pumps[rows[n]] = pumps[n]
            return posture_seconds(A, rows[n], model.probs(A, rows[n]))

        r0 = resp(("today", "none"))
        c0, b0 = weighted_cvar(r0, counts), np.array([counts[boroughs == b] @ r0[boroughs == b] for b in t_b]) / np.maximum(b_w, 1)
        model.pumps = base.copy()
        d0 = dispatch_response(A, today_rows, model, dmodel) if sc == "lgbm" else None
        for n in layouts:
            if n[0] == "today":
                continue
            r = resp(n)
            d = r - r0
            ds = boot @ d / N
            bd = np.array([counts[boroughs == b] @ r[boroughs == b] for b in t_b]) / np.maximum(b_w, 1) - b0
            rec = {"layout": n[0], "rule": n[1], "scorer": sc, "d_mean_s": round(float(counts @ d / N), 2), "lo": round(float(np.percentile(ds, 2.5)), 2), "hi": round(float(np.percentile(ds, 97.5)), 2),
                   "d_slowest10_s": round(weighted_cvar(r, counts) - c0, 2), "within_6min": round(float(counts @ (r <= 360) / N * 100), 1), "d_within_6min": round(float(counts @ ((r <= 360).astype(float) - (r0 <= 360)) / N * 100), 1),
                   "calls_faster_30s_pct": round(float(counts[d <= -30].sum() / N * 100), 1), "calls_slower_30s_pct": round(float(counts[d >= 30].sum() / N * 100), 1), "calls_slower_120s_pct": round(float(counts[d >= 120].sum() / N * 100), 2),
                   "boroughs_worse_1s": int((bd >= 1).sum()), "worst_borough_s": round(float(bd.max()), 1), "boroughs_better_5s": int((bd <= -5).sum())}
            if d0 is not None:
                model.pumps = np.ones(len(sites), int)
                model.pumps[rows[n]] = pumps[n]
                rec["d_mean_dispatch"] = round(float(counts @ (dispatch_response(A, rows[n], model, dmodel) - d0) / N), 2)
            records.append(rec)
        pd.DataFrame(records).to_csv(OUT / "scores.csv", index=False)
    df = pd.DataFrame(records)
    (OUT / "summary.json").write_text(json.dumps({"doc": __doc__, "n_calls_2025": int(N), "bootstrap": BOOT, "rows": records}, indent=2, default=str) + "\n")
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 200)
    print(df.pivot_table(index=["layout", "rule"], columns="scorer", values="d_mean_s", sort=False)[list(SCORERS)].to_string())
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
