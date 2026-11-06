#!/usr/bin/env python3
"""Option A: move stations to speed up London's slowest boroughs, with no borough slower.

The target rule was fixed before this script was run: docs/targeted_areas_rule.md.

Target: the 5 boroughs with the slowest recorded average attendance in 2024 (boroughs with 1,000+ calls).
Layouts: best-swap search (as placement_methods.py) on the wider site list, chosen on 2024 demand with LightGBM and busy engines, pumps moving with
their station. The final layout must keep, on 2024 demand: every borough's average no worse than today (0 s), the London average no worse, and the
slowest 10 % no worse. Every step of the search keeps these limits.
  target_<k>    objective = average attendance in the 5 target boroughs, up to k moved
  citywide_<k>  objective = London average, same search and limits (shows what targeting costs London and gains the targets)
Also scored: the planner's wide Move 10 and Move 20 (section 5.4), which were not built for the target boroughs.

Scoring as score_wide_layouts.py: 2025 calls, busy engines, LightGBM, road and straight-line Kolesar, land rules B and C by moving failing sites to the
nearest passing candidate. No street refinement on the searched layouts (the planner layouts have it). 2026 is not used.

Writes data/figures/placement/targeted/: targets.csv, layouts/, scores.csv, boroughs.csv, summary.json.

With --rule rule_b or rule_c, the targeted layouts are searched again using only new sites that pass that land rule, written to targeted/<rule>/.

Run:  PYTHONPATH=src .venv/bin/python scripts/targeted_areas.py [--rule rule_c]
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

from firestation.busy_placement import ErlangBusyModel, PumpPlan  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid, haversine_km, posture_seconds, weighted_cvar  # noqa: E402
from firestation.fire_station_location import (build_house_candidates, build_travel_matrix, busy_prob_vector, drop_far_incidents,  # noqa: E402
                                               fit_scorer_kolesar, recorded_station_turnout, snap_to_cells)
from firestation.ml_scorer import cell_boroughs  # noqa: E402
from firestation.station_capacity import default_pump_table  # noqa: E402
from placement_analysis import classify  # noqa: E402
from score_wide_layouts import add_rules  # noqa: E402

OUT = ROOT / "data" / "figures" / "placement" / "targeted"
WIDE = ROOT / "data" / "figures" / "placement" / "recommended_wide"
N_TARGET, MIN_CALLS = 5, 1000
LIMITS = (10, 20)
SCORERS = ("lgbm", "road", "kolesar")
BOOT = 1000
NEAR_KM = 3.0  # relocation candidates: sites this close to the station being closed
SITE_LATLON = (None, None)


def borough_means(r, w, boroughs, names):
    return np.array([w[boroughs == b] @ r[boroughs == b] / max(w[boroughs == b].sum(), 1e-9) for b in names])


def search(A, w, wt, model, start, existing, limit, ok, value_fn, log="", allowed=None):
    """Best swap that keeps every limit. Each step prices swaps with fixed busy chances on the objective weights ``wt`` and on all calls ``w``, and
    also tries opening near the closed station (relocations, which are the swaps most likely to keep every borough). The candidates are then checked
    in full, and the best one that keeps every limit (``ok``) and improves ``value_fn`` is taken."""
    ex = set(int(i) for i in existing)
    idx = [int(i) for i in start]
    n_sites = A.shape[0]
    value = value_fn(idx)
    lat, lon = SITE_LATLON
    for step in range(400):
        p_site = model.probs(A, idx)
        moved = len(ex - set(idx))
        idx_set = set(idx)
        rest = np.array([i for i in range(n_sites) if i not in idx_set and (allowed is None or allowed[i])])
        rest_ex = np.isin(rest, list(ex))
        tried = set()
        for c in idx:
            others = np.array([i for i in idx if i != c])
            sub = A[others]
            two = np.argpartition(sub, 1, axis=0)[:2]
            cols = np.arange(sub.shape[1])
            sw = sub[two[0], cols] > sub[two[1], cols]
            i1, i2 = np.where(sw, two[1], two[0]), np.where(sw, two[0], two[1])
            a, b = sub[i1, cols], sub[i2, cols]
            pa = p_site[others[i1]]
            after = moved + (c in ex) - rest_ex.astype(int)
            o = rest[after <= limit]
            if not len(o):
                continue
            S = A[o]
            pc = p_site[c]
            cost = np.where(S < a, (1 - pc) * S + pc * a, np.where(S < b, (1 - pa) * a + pa * S, (1 - pa) * a + pa * b))
            ct, ca = cost @ wt, cost @ w
            near = haversine_km(lon[o], lat[o], lon[c], lat[c]) <= NEAR_KM
            for j in list(np.argsort(ct)[:3]) + list(np.argsort(ca)[:2]) + list(np.argsort(np.where(near, ct, np.inf))[:3]):
                if np.isfinite(ct[j]) and (j in np.argsort(ct)[:3] or near[j] or j in np.argsort(ca)[:2]):
                    tried.add((c, int(o[j])))
        best = None
        for c, o in tried:
            new = [i for i in idx if i != c] + [o]
            val = value_fn(new)
            if val < value - 1e-3 and (best is None or val < best[0]) and ok(new):
                best = (val, new)
        if best is None:
            break
        value, idx = best
    print(f"  {log}: {len(ex - set(idx))} moved, objective {value:.2f} s after {step + 1} steps, limits kept: {ok(idx)}", flush=True)
    return idx, value


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=("rule_b", "rule_c"), help="Search only new sites passing this land rule (re-optimised, not repaired); writes to targeted/<rule>/")
    args = ap.parse_args()
    global OUT
    if args.rule:
        OUT = OUT / args.rule
    (OUT / "layouts").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 200)
    spec = get_city("london")
    houses = load_fire_stations(spec)
    incidents = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024, 2025]), houses)
    demand = incidents[incidents["cal_year"] == 2024].reset_index(drop=True)
    ev = incidents[incidents["cal_year"] == 2025].reset_index(drop=True)

    # 1. Target boroughs: recorded 2024 only.
    rec = demand.dropna(subset=["borough", "travel_seconds"]).assign(b=lambda d: d["borough"].astype(str).str.upper())
    tb = rec.groupby("b")["travel_seconds"].agg(calls="size", recorded_mean_s="mean")
    tb = tb[tb["calls"] >= MIN_CALLS].sort_values("recorded_mean_s", ascending=False)
    targets = tb.index[:N_TARGET].tolist()
    tb.assign(target=tb.index.isin(targets)).round(1).to_csv(OUT / "targets.csv")
    print("Target boroughs (recorded 2024):\n", tb.head(8).round(1).to_string(), flush=True)

    cells = build_demand_grid(demand, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    w = cells["n_incidents"].to_numpy(float)
    boroughs = np.asarray(cell_boroughs(cells, demand)).astype(str)
    names = sorted(set(boroughs) - {""})
    is_t = np.isin(boroughs, targets)
    wt = w * is_t
    cidx, _ = snap_to_cells(ev, cells)
    n25 = np.bincount(cidx, minlength=len(cells)).astype(float)
    N = n25.sum()
    pumps_today = default_pump_table(houses).set_index("station")["n_pumps"].reindex(houses["facilityname"].astype(str)).fillna(1).astype(int).to_numpy()
    tab, new_turn = recorded_station_turnout([2024])
    norm = {k.upper(): v for k, v in tab.items()}
    kmodel, _, _ = fit_scorer_kolesar(demand, houses)

    # 2. Search on the wide pool with LightGBM.
    pool = build_house_candidates(houses, cells, demand_candidates=300, spread_km=1.5)
    ex_rows = np.flatnonzero(pool["is_existing"].to_numpy())
    pnames = pool["site_name"].astype(str).str.upper().to_numpy()
    turn = np.array([norm.get(n, new_turn) if e else new_turn for n, e in zip(pnames, pool["is_existing"])])
    T, _ = build_travel_matrix(pool, cells, surrogate=TravelTimeSurrogate(), scorer="lgbm", kolesar=kmodel, incidents=demand)
    A = T + turn[:, None]
    base = np.ones(len(pool), int)
    base[ex_rows] = pumps_today
    plan = PumpPlan(base, ex_rows, pool["lat"].to_numpy(float), pool["lon"].to_numpy(float), extra="move")
    model = ErlangBusyModel.calibrate(A, ex_rows, busy_prob_vector(pool, demand), base, w, plan)
    global SITE_LATLON
    SITE_LATLON = (pool["lat"].to_numpy(float), pool["lon"].to_numpy(float))
    today = list(ex_rows)
    r0 = model.response(A, today)
    W = w.sum()
    m0, c0, b0 = w @ r0 / W, weighted_cvar(r0, w), borough_means(r0, w, boroughs, names)
    print(f"Pool {len(pool)} sites. Today (2024, LightGBM): London {m0:.1f} s, targets {wt @ r0 / wt.sum():.1f} s", flush=True)

    def excess(r):
        """Seconds by which the limits are broken: every borough, the London average and the slowest 10 % (0 = all kept)."""
        return float(np.clip(borough_means(r, w, boroughs, names) - b0, 0, None).sum() + max(w @ r / W - m0, 0) + max(weighted_cvar(r, w) - c0, 0))

    def mean_over(wp):
        return lambda idx: float(wp @ model.response(A, idx) / wp.sum())

    def ok(idx):
        return excess(model.response(A, idx)) <= 1e-6

    obj = {"target": (wt, mean_over(wt)), "citywide": (w, mean_over(w))}
    allowed = None
    if args.rule:
        ex_mask = pool["is_existing"].to_numpy(bool)
        tg = add_rules(classify(pool["lat"].to_numpy(float), pool["lon"].to_numpy(float)))
        allowed = ex_mask | tg[args.rule].to_numpy(bool)
        obj = {"target": obj["target"]}
        print(f"Only new sites passing {args.rule}: {int((allowed & ~ex_mask).sum())} of {int((~ex_mask).sum())}", flush=True)
    layouts = {"today": pd.DataFrame({"site_name": houses["facilityname"].astype(str), "lat": houses["lat"], "lon": houses["lon"], "is_existing": True,
                                      "n_pumps": pumps_today})}
    for name, (wp, fn) in obj.items():
        idx = today
        for k in LIMITS:
            t0 = time.time()
            idx, _ = search(A, w, wp, model, idx, ex_rows, k, ok, fn, log=f"{name}_{k}", allowed=allowed)
            L = pool.iloc[idx][["site_name", "lat", "lon", "is_existing"]].assign(n_pumps=np.maximum(model.pumps_for(idx)[idx], 1)).reset_index(drop=True)
            L.to_csv(OUT / "layouts" / f"{name}_{k}.csv", index=False)
            layouts[f"{name}_{k}"] = L
            print(f"    {time.time() - t0:.0f} s", flush=True)
    for k in LIMITS if not args.rule else ():
        layouts[f"planner_{k}"] = pd.read_csv(WIDE / f"replace_{k}_boroughs1s" / "planned_fire_stations.csv")[["site_name", "lat", "lon", "is_existing", "n_pumps"]]

    # 3. Land rules for the targeted layouts.
    key = lambda lat, lon: f"{float(lat):.6f},{float(lon):.6f}"  # noqa: E731
    cand = pool[~pool["is_existing"].to_numpy(bool)].reset_index(drop=True)
    new_pts = pd.concat([L.loc[~L["is_existing"].astype(bool), ["lat", "lon"]] for n, L in layouts.items() if n.startswith("target")]).drop_duplicates()
    both = pd.concat([cand[["lat", "lon"]], new_pts], ignore_index=True)
    tags = add_rules(classify(both["lat"].to_numpy(float), both["lon"].to_numpy(float)))
    cand_t = tags.iloc[:len(cand)].reset_index(drop=True)
    tag_of = {key(r.lat, r.lon): tags.iloc[len(cand) + i] for i, r in enumerate(new_pts.itertuples())}
    for n in [n for n in layouts if n.startswith("target") and not args.rule]:
        for rule in ("rule_b", "rule_c"):
            q = layouts[n].copy()
            q["is_existing"] = q["is_existing"].astype(bool)
            okc = cand_t[rule].to_numpy(bool)
            used = {key(r.lat, r.lon) for r in q[~q.is_existing].itertuples() if bool(tag_of[key(r.lat, r.lon)][rule])}
            fails = 0
            for i, r in q[~q.is_existing].iterrows():
                if bool(tag_of[key(r.lat, r.lon)][rule]):
                    continue
                d = haversine_km(cand["lon"].to_numpy(float), cand["lat"].to_numpy(float), r.lon, r.lat)
                free = okc & np.array([key(a, b) not in used for a, b in zip(cand.lat, cand.lon)])
                j = int(np.where(free, d, np.inf).argmin())
                q.loc[i, ["lat", "lon", "site_name"]] = [cand.lat[j], cand.lon[j], f"Repaired site {j}"]
                used.add(key(cand.lat[j], cand.lon[j]))
                fails += 1
            print(f"  {n} {rule}: {fails} sites moved", flush=True)
            layouts[f"{n}|{rule}"] = q

    # 4. Score everything on 2025 calls under three models (as score_wide_layouts.py).
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
    turnout = np.array([norm.get(r.site_name.upper(), new_turn) if r.is_existing else new_turn for r in sites.itertuples()])
    measured = busy_prob_vector(sites, demand)
    rng = np.random.default_rng(0)
    boot = rng.multinomial(int(N), n25 / N, size=BOOT).astype(float)
    tN = n25[is_t].sum()
    records, brows = [], []
    for sc in SCORERS:
        print(f"Scoring with {sc} ({len(sites)} sites)…", flush=True)
        Ts, _ = build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer=sc, kolesar=kmodel, incidents=demand)
        As = Ts + turnout[:, None]
        bp = np.ones(len(sites), int)
        bp[rows["today"]] = pumps["today"]
        m = ErlangBusyModel.calibrate(As, rows["today"], measured, bp, w)

        def resp(n):
            m.pumps = np.ones(len(sites), int)
            m.pumps[rows[n]] = pumps[n]
            return posture_seconds(As, rows[n], m.probs(As, rows[n]))

        q0 = resp("today")
        bt0 = borough_means(q0, n25, boroughs, names)
        for n in layouts:
            if n == "today":
                continue
            q = resp(n)
            d = q - q0
            bd = borough_means(q, n25, boroughs, names) - bt0
            nt = ~np.isin(names, targets)
            tboot = boot[:, is_t] @ d[is_t] / boot[:, is_t].sum(1)
            lay, rule = (n.split("|") + ["none"])[:2]
            records.append({"layout": lay, "rule": rule, "scorer": sc, "moved": int(sum(~layouts[n]["is_existing"].astype(bool))),
                            "d_target_s": round(float(n25[is_t] @ d[is_t] / tN), 2), "target_lo": round(float(np.percentile(tboot, 2.5)), 2),
                            "target_hi": round(float(np.percentile(tboot, 97.5)), 2), "d_london_s": round(float(n25 @ d / N), 2),
                            "d_rest_s": round(float(n25[~is_t] @ d[~is_t] / n25[~is_t].sum()), 2),
                            "d_slowest10_s": round(weighted_cvar(q, n25) - weighted_cvar(q0, n25), 2),
                            "boroughs_worse_1s": int((bd >= 1).sum()), "worst_borough_s": round(float(bd.max()), 1), "worst_borough": names[int(bd.argmax())],
                            "worst_other_borough_s": round(float(bd[nt].max()), 1),
                            "target_calls_30s_faster_pct": round(float(n25[is_t & (d <= -30)].sum() / tN * 100), 1),
                            "calls_30s_slower_pct": round(float(n25[d >= 30].sum() / N * 100), 1)})
            if rule == "none":
                brows += [{"layout": lay, "scorer": sc, "borough": b, "target": b in targets, "d_s": round(float(x), 2)} for b, x in zip(names, bd)]
        pd.DataFrame(records).to_csv(OUT / "scores.csv", index=False)
    df = pd.DataFrame(records)
    pd.DataFrame(brows).to_csv(OUT / "boroughs.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"doc": __doc__, "targets": targets, "n_calls_2025": int(N), "rows": records}, indent=2, default=str) + "\n")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
