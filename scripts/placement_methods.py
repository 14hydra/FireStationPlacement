#!/usr/bin/env python3
"""Which placement methods could beat Move 10? A screen on one consistent scoring.

Chosen on 2024 calls, scored on 2025 calls, each at its 0.9 km demand cell. 2026 is not used. Search: repeated best swap (close one site, open
one), with a limit on stations moved. It is not proven optimal and has no borough or slowest-10 % floors, so results are upper-end screening numbers.
Layouts worth keeping should be re-solved with the exact planner and its floors.

Methods:
  mean        minimise average attendance, standard candidate sites (today's stations + busiest cells + one per 3 km block)
  over6       minimise the chance of attendance over 6 minutes. Chance from a curve fitted on 2024 recorded calls
  mean_away   minimise average, with away time kept apart from busy time: moving a station does not reduce away time
  mean_wide   minimise average, many more candidate sites (300 busiest cells + one per 1.5 km block)
  engines     keep all buildings, only move second engines between stations
  engines_day as engines, but the second engines may sit differently in each period of the day

Scored with the busy-engine model, the learned dispatch model, and road and straight-line travel times.

Writes data/figures/placement/methods/: curve.csv, engines.csv, calibration.json, summary.json, layouts/.

Run:  PYTHONPATH=src .venv/bin/python scripts/placement_methods.py [--skip-wide]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from busy_engine_check import PERIODS  # noqa: E402
from firestation.busy_placement import ErlangBusyModel, PumpPlan, erlang_b  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.dispatch import dispatch_response, load_dispatch  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid, posture_seconds, weighted_cvar  # noqa: E402
from firestation.fire_station_location import (build_house_candidates, build_travel_matrix, busy_prob_vector, drop_far_incidents,  # noqa: E402
                                               fit_scorer_kolesar, recorded_station_turnout, snap_to_cells)
from firestation.station_capacity import default_pump_table  # noqa: E402
from gap_windows import prepare  # noqa: E402

OUT = ROOT / "data" / "figures" / "placement" / "methods"
CHECKPOINTS = (5, 10, 15, 20, 30, 40, 60, 102)
ENGINE_MOVES = (5, 10, 20, 40)
POOLS = {"standard": {"demand_candidates": 80, "spread_km": 3.0}, "wide": {"demand_candidates": 300, "spread_km": 1.5}}


class SplitBusy(ErlangBusyModel):
    """Chance a site cannot respond = fixed away share + (1 - away) x Erlang busy chance."""

    away: np.ndarray

    def probs(self, A, idx):
        p = super().probs(A, idx)
        i = np.asarray(list(idx), dtype=int)
        out = np.zeros_like(p)
        out[i] = self.away[i] + (1 - self.away[i]) * p[i]
        return np.clip(out, 0, 0.95)


def fit_logistic(x, y1, n, iters=30):
    """P(over 6 min) = logit^-1(a + b (x - 300)/60) from counts per cell."""
    z = (x - 300) / 60
    a, b = 0.0, 1.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + b * z)))
        v = n * p * (1 - p) + 1e-9
        g = np.array([(y1 - n * p).sum(), ((y1 - n * p) * z).sum()])
        H = np.array([[v.sum(), (v * z).sum()], [(v * z).sum(), (v * z * z).sum()]])
        a, b = np.array([a, b]) + np.linalg.solve(H, g)
    return float(a), float(b)


def blend(F, A, idx, p_site):
    """(1-p) f(t1) + p f(t2), t1 and t2 the two fastest open sites by time."""
    idx = np.asarray(idx, dtype=int)
    sub = A[idx]
    two = np.argpartition(sub, 1, axis=0)[:2]
    cols = np.arange(sub.shape[1])
    swap = sub[two[0], cols] > sub[two[1], cols]
    first = np.where(swap, two[1], two[0])
    second = np.where(swap, two[0], two[1])
    p = p_site[idx[first]]
    return (1 - p) * F[idx[first], cols] + p * F[idx[second], cols]


def search(A, F, w, model, start, existing, limit, log=None):
    """Best-swap search for at most ``limit`` moved stations. Each swap is priced with fixed busy chances, then checked in full."""
    ex = set(int(i) for i in existing)
    idx = [int(i) for i in start]
    W = w.sum()
    n_sites = A.shape[0]
    value = float(w @ blend(F, A, idx, model.probs(A, idx))) / W
    for step in range(600):
        p_site = model.probs(A, idx)
        moved = len(ex - set(idx))
        rest = np.array([i for i in range(n_sites) if i not in set(idx)])
        rest_ex = np.isin(rest, list(ex))
        tried = []
        for c in idx:
            others = np.array([i for i in idx if i != c])
            sub = A[others]
            two = np.argpartition(sub, 1, axis=0)[:2]
            cols = np.arange(sub.shape[1])
            sw = sub[two[0], cols] > sub[two[1], cols]
            i1, i2 = np.where(sw, two[1], two[0]), np.where(sw, two[0], two[1])
            a, b = sub[i1, cols], sub[i2, cols]
            fa, fb = F[others[i1], cols], F[others[i2], cols]
            pa = p_site[others[i1]]
            after = moved + (c in ex) - rest_ex.astype(int)
            o = rest[after <= limit]
            if not len(o):
                continue
            S, FS = A[o], F[o]
            pc = p_site[c]
            cost = np.where(S < a, (1 - pc) * FS + pc * fa, np.where(S < b, (1 - pa) * fa + pa * FS, (1 - pa) * fa + pa * fb))
            tot = cost @ w / W
            for j in np.argsort(tot)[:2]:
                tried.append((float(tot[j]), c, int(o[j])))
        tried.sort(key=lambda r: r[0])
        done = True
        for _, c, o in tried[:12]:
            new = [i for i in idx if i != c] + [o]
            val = float(w @ blend(F, A, new, model.probs(A, new))) / W
            if val < value - 1e-4 * max(abs(value), 1e-9) * 0.01:
                idx, value, done = new, val, False
                break
        if done:
            break
    return idx, value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-wide", action="store_true")
    args = ap.parse_args()
    (OUT / "layouts").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 200)
    spec = get_city("london")
    houses = load_fire_stations(spec)
    incidents = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024, 2025]), houses)
    demand = incidents[incidents["cal_year"] == 2024].reset_index(drop=True)
    ev = incidents[incidents["cal_year"] == 2025].reset_index(drop=True)
    cells = build_demand_grid(demand, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    w = cells["n_incidents"].to_numpy(float)
    c24, _ = snap_to_cells(demand, cells)
    c25, _ = snap_to_cells(ev, cells)
    n25 = np.bincount(c25, minlength=len(cells)).astype(float)
    over24 = np.bincount(c24, weights=(demand["travel_seconds"].to_numpy(float) > 360).astype(float), minlength=len(cells))
    n24 = np.bincount(c24, minlength=len(cells)).astype(float)
    over25_real = float((ev["travel_seconds"].to_numpy(float) > 360).mean())
    pumps_today = default_pump_table(houses).set_index("station")["n_pumps"].reindex(houses["facilityname"].astype(str)).fillna(1).astype(int).to_numpy()
    turn_tab, turn_new = recorded_station_turnout([2024])
    norm = {k.upper(): v for k, v in turn_tab.items()}
    kmodel, _, _ = fit_scorer_kolesar(demand, houses)
    dmodel = load_dispatch()

    # Busy (all engines out) vs other reasons per station, from recorded engine periods.
    prep = prepare(2024, {})["calls"]
    busy_share = prep.groupby("ground_name")["cat_no_buffer"].apply(lambda s: float((s == "empty").mean()))
    rng = np.random.default_rng(0)
    boot = rng.multinomial(int(n25.sum()), n25 / n25.sum(), size=300).astype(float)

    curve, engine_rows, today_saved = [], [], {}
    for pool, pool_kw in POOLS.items():
        if pool == "wide" and args.skip_wide:
            continue
        sites = build_house_candidates(houses, cells, **pool_kw)
        ex_rows = np.flatnonzero(sites["is_existing"].to_numpy())
        print(f"Pool {pool}: {len(sites)} sites", flush=True)
        names = sites["site_name"].str.upper().to_numpy()
        turnout = np.array([norm.get(n, turn_new) if e else turn_new for n, e in zip(names, sites["is_existing"])])
        As = {}
        for sc in ("lgbm", "road", "kolesar"):
            T, _ = build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer=sc, kolesar=kmodel, incidents=demand)
            As[sc] = T + turnout[:, None]
            print(f"  matrix {sc} done", flush=True)
        base = np.ones(len(sites), int)
        base[ex_rows] = pumps_today
        lat, lon = sites["lat"].to_numpy(float), sites["lon"].to_numpy(float)
        plan = PumpPlan(base, ex_rows, lat, lon, extra="move")
        measured = busy_prob_vector(sites, demand)
        models = {sc: ErlangBusyModel.calibrate(As[sc], ex_rows, measured, base, w, plan) for sc in As}
        # Split model on the main travel times.
        emp = np.array([busy_share.get(n, np.nan) for n in names])
        emp = np.where(np.isfinite(emp), emp, np.nanmedian(emp))
        A = As["lgbm"]
        sp0 = ErlangBusyModel.calibrate(A, ex_rows, emp, base, w, plan)
        split = SplitBusy(sp0.erlangs_per_call, sp0.pumps, sp0.load_weights, plan)
        split.away = np.clip((measured - emp) / (1 - emp), 0, 0.49)
        split.away[~sites["is_existing"].to_numpy()] = float(np.median(split.away[ex_rows]))

        today = list(ex_rows)
        r0 = {sc: posture_seconds(As[sc], today, models[sc].probs(As[sc], today)) for sc in As}
        a, b = fit_logistic(r0["lgbm"], over24, n24)
        Fo = 1 / (1 + np.exp(-(a + b * (A - 300) / 60)))
        if pool == "standard":
            cal = {"a": a, "b_per_minute": b, "over6_2024_recorded": float(over24.sum() / n24.sum()), "over6_2025_recorded": over25_real,
                   "over6_2024_curve": float(w @ blend(Fo, A, today, models["lgbm"].probs(A, today)) / w.sum())}
            (OUT / "calibration.json").write_text(json.dumps(cal, indent=2) + "\n")
            print("Curve:", cal, flush=True)

        def score(name, limit, idx):
            idx = [int(i) for i in idx]
            row = {"method": name, "moves_limit": limit, "moved": len(set(ex_rows.tolist()) - set(idx)), "sites_in_pool": len(sites)}
            n = n25.sum()
            for sc in As:
                r = posture_seconds(As[sc], idx, models[sc].probs(As[sc], idx))
                d = boot @ (r - r0[sc]) / n
                row[f"d_mean_{sc}"] = round(float(n25 @ (r - r0[sc]) / n), 2)
                row[f"d_lo_{sc}"], row[f"d_hi_{sc}"] = round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)
            r, rb = posture_seconds(A, idx, models["lgbm"].probs(A, idx)), r0["lgbm"]
            row["d_mean_dispatch"] = round(float(n25 @ (dispatch_response(A, idx, models["lgbm"], dmodel) - dispatch_response(A, today, models["lgbm"], dmodel)) / n), 2)
            row["d_mean_split"] = round(float(n25 @ (posture_seconds(A, idx, split.probs(A, idx)) - posture_seconds(A, today, split.probs(A, today))) / n), 2)
            row["d_slowest10"] = round(weighted_cvar(r, n25) - weighted_cvar(rb, n25), 2)
            ov = blend(Fo, A, idx, models["lgbm"].probs(A, idx))
            ov0 = blend(Fo, A, today, models["lgbm"].probs(A, today))
            row["d_over6_points"] = round(float(n25 @ (ov - ov0) / n * 100), 2)
            row["d_over6_lo"], row["d_over6_hi"] = [round(float(x), 2) for x in np.percentile(boot @ (ov - ov0) / n * 100, [2.5, 97.5])]
            row["calls_faster_30s_pct"] = round(float(n25[(r - rb) <= -30].sum() / n * 100), 1)
            row["calls_slower_30s_pct"] = round(float(n25[(r - rb) >= 30].sum() / n * 100), 1)
            row["calls_over6_avoided_per_year"] = round(float(-(n25 @ (ov - ov0))))
            return row

        methods = {"mean": (A, models["lgbm"])} if pool == "wide" else {"mean": (A, models["lgbm"]), "over6": (Fo, models["lgbm"]), "mean_away": (A, split)}
        for mname, (F, model) in methods.items():
            label = "mean_wide" if pool == "wide" else mname
            idx = today
            for k in CHECKPOINTS:
                idx, val = search(A, F, w, model, idx, ex_rows, k)
                row = score(label, k, idx)
                curve.append(row)
                pd.DataFrame(curve).to_csv(OUT / "curve.csv", index=False)
                out = sites.iloc[idx][["site_name", "lat", "lon", "is_existing"]].assign(n_pumps=model.pumps_for(idx)[idx])
                out.to_csv(OUT / "layouts" / f"{label}_{k}.csv", index=False)
                print({x: row[x] for x in ("method", "moved", "d_mean_lgbm", "d_mean_dispatch", "d_mean_split", "d_mean_road", "d_mean_kolesar", "d_over6_points", "d_slowest10",
                                           "calls_faster_30s_pct", "calls_slower_30s_pct", "calls_over6_avoided_per_year")}, flush=True)

        if pool == "standard":
            # Second engines only: per site, value is linear in its busy chance, so greedy search is simple and exact per step.
            sub = A[ex_rows]
            two = np.argpartition(sub, 1, axis=0)[:2]
            cols = np.arange(sub.shape[1])
            sw = sub[two[0], cols] > sub[two[1], cols]
            i1, i2 = np.where(sw, two[1], two[0]), np.where(sw, two[0], two[1])
            t1, t2 = sub[i1, cols], sub[i2, cols]
            ep = models["lgbm"].erlangs_per_call[ex_rows]
            S = len(ex_rows)

            def greedy(weights, rate):
                """Move second engines one at a time. Returns value per move count, in seconds of average attendance summed over `weights`."""
                calls = np.bincount(i1, weights=weights, minlength=S)
                gain = np.bincount(i1, weights=weights * (t2 - t1), minlength=S)
                base_v = (weights * t1).sum()
                pum = pumps_today.copy()
                load = ep * calls * rate
                vals = {0: base_v + (gain * erlang_b(load, pum)).sum()}
                for m in range(1, max(ENGINE_MOVES) + 1):
                    now = gain * erlang_b(load, pum)
                    lose = gain * erlang_b(load, np.maximum(pum - 1, 1)) - now
                    win = gain * erlang_b(load, pum + 1) - now
                    donors, takers = np.flatnonzero(pum >= 2), np.flatnonzero(pum < 3)
                    d, t = donors[lose[donors].argmin()], takers[win[takers].argmin()]
                    if lose[d] + win[t] >= 0 or d == t:
                        vals[m] = vals[m - 1]
                        continue
                    pum[d] -= 1
                    pum[t] += 1
                    vals[m] = base_v + (gain * erlang_b(load, pum)).sum()
                return vals

            W = w.sum()
            all_year = greedy(w, 1.0)
            engine_rows += [{"method": "engines", "moved_engines": m, "d_mean_s": round((all_year[m] - all_year[0]) / W, 2)} for m in ENGINE_MOVES]
            hours = demand["hour_of_call"].to_numpy()
            by_p = {}
            for pname, hrs in PERIODS.items():
                mask = np.isin(hours, list(hrs))
                wp = np.bincount(c24[mask], minlength=len(cells)).astype(float)
                by_p[pname] = greedy(wp, 24 / len(hrs))
            for m in ENGINE_MOVES:
                engine_rows.append({"method": "engines_day", "moved_engines": m, "d_mean_s": round(sum(v[m] - v[0] for v in by_p.values()) / n24.sum(), 2)})
            pd.DataFrame(engine_rows).to_csv(OUT / "engines.csv", index=False)
            print(pd.DataFrame(engine_rows).to_string(index=False), flush=True)

    t = pd.DataFrame(curve)
    t.to_csv(OUT / "curve.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"doc": __doc__, "curve": curve, "engines": engine_rows}, indent=2, default=str) + "\n")
    print(t.to_string(index=False))
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
