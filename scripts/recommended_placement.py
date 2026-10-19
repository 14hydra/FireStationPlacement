#!/usr/bin/env python3
"""Recommended station placement: efficiency first, with floors for the slowest calls.

The project's defensible position: favour busy areas (alpha = 1: each area counts by its
calls), but no one may be left worse off than today:

  - every area that meets LFB's 10-minute standard today keeps it (cap);
  - the slowest 10 % of calls are no slower on average than today (floor);
  - the overall average is no worse than today (floor).

For each scenario (add 3 stations, move up to 10, full redesign) the exact solver finds
the best layout under those rules, with busy engines (Erlang loss, load-dependent), pumps
moving with their station, LightGBM travel times, and each new site refined to a street
junction. Then the alpha check: what does favouring busy areas cost quieter ones?

  efficiency score = 100 * (E_today - E) / (E_today - E_best)   E: call-weighted average
  equity score     = 100 * (Q_today - Q) / (Q_today - Q_best)   Q: area average (each area once)

E_best is the unconstrained alpha = 1 optimum and Q_best the alpha = 0 optimum (both busy-
aware, before street refinement, so refinement can push a score slightly past 100). Also
per-borough averages today vs plan.

Writes data/figures/placement/recommended/<scenario>/: the planner's map and figures,
planned / opened / closed stations, boroughs.csv, summary.json; and summary.json at the top.

Run:  PYTHONPATH=src .venv/bin/python scripts/recommended_placement.py [--scenarios expand_3,replace_10]
      (station-count sweep: --scenarios expand_1,expand_2,expand_5,replace_5 --keep-boroughs 1)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.fire_station_location import run_fire_station_plan  # noqa: E402
from firestation.ml_scorer import cell_boroughs  # noqa: E402

OUT = ROOT / "data" / "figures" / "placement" / "recommended"
SCENARIOS = ("expand_3", "replace_10", "redesign")  # default; any expand_<n> or replace_<n> also works


def scenario_kw(name: str) -> dict:
    """``expand_<n>``: add n stations; ``replace_<n>``: move up to n; ``redesign``: all stations."""
    if name == "redesign":
        return {"mode": "redesign"}
    mode, n = name.rsplit("_", 1)
    return {"expand": {"mode": "expand", "add_stations": int(n)},
            "replace": {"mode": "replace", "replace_r": int(n)}}[mode]
COMMON = dict(city_id="london", scorer="lgbm", demand_years=[2024], eval_years=[2025], max_incident_rows=None,
              use_busy=True, objective="response_time")


def run(houses, incidents, scenario, quiet=True, **kw):
    mkw = scenario_kw(scenario)
    if mkw["mode"] == "redesign":
        mkw["n_houses"] = len(houses)
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx:
        return run_fire_station_plan(houses=houses, incidents=incidents, **COMMON, **mkw, **kw)


def area_mean(resp, w):
    a = w > 0
    return float(resp[a].mean())


def main() -> int:
    from plan_fire_station_locations import _write_figures, _write_map

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--refine-km", type=float, default=1.0)
    ap.add_argument("--keep-boroughs", default="",
                    help="Comma list of per-borough tolerances in seconds (e.g. 0,1,5); empty = citywide floors only")
    ap.add_argument("--busy-rounds", type=int, default=4, help="Max busy-rate update rounds")
    ap.add_argument("--solver-time", type=float, default=600, help="Time limit per solve (s)")
    args = ap.parse_args()
    tols = [float(t) for t in args.keep_boroughs.split(",") if t != ""] or [None]
    spec = get_city("london")
    houses = load_fire_stations(spec)
    incidents = load_incidents(spec, max_rows=None, years=[2024, 2025])
    demand = incidents[incidents["cal_year"] == 2024]
    top = {}
    lim = dict(busy_rounds=args.busy_rounds, solver_time_s=args.solver_time)
    for sc0 in args.scenarios.split(","):
        print(f"\n== {sc0}: reference solves for the scores (alpha 1 and 0, no floors)…", flush=True)
        best_e = run(houses, incidents, sc0, alpha=1.0, **lim)
        best_q = run(houses, incidents, sc0, alpha=0.0, **lim)
        for tol in tols:
            sc = sc0 if tol is None else f"{sc0}_boroughs{tol:g}s"
            t0 = time.time()
            out = OUT / sc
            out.mkdir(parents=True, exist_ok=True)
            print(f"\n== {sc}: recommended (alpha 1, busy engines, floors"
                  f"{'' if tol is None else f', every borough within {tol:g} s'}, street refinement)…", flush=True)
            rec = run(houses, incidents, sc0, quiet=False, alpha=1.0, keep_slowest=True, keep_average=True,
                      keep_boroughs=tol, refine_km=args.refine_km, **lim)
            s = rec.summary
            cells = rec.cells
            w = cells["weight"].to_numpy(float)
            cur, plan = cells["response_s_current"].to_numpy(float), cells["response_s"].to_numpy(float)
            W = w.sum()
            E_today, E = float((w * cur).sum() / W), float((w * plan).sum() / W)
            E_best = best_e.summary["plan"]["expected_response_s"]
            Q_today, Q = area_mean(cur, w), area_mean(plan, w)
            Q_best = best_q.summary["plan"]["area_mean_response_s"]
            cur_m, plan_m = s["baselines"]["current_fire_stations"], s["plan"]

            cells = cells.assign(borough=cell_boroughs(cells, demand))
            b = (cells.assign(wc=w * cur, wp=w * plan).groupby("borough")
                 .agg(calls=("weight", "sum"), today=("wc", "sum"), plan=("wp", "sum")))
            b["today_s"], b["plan_s"] = b["today"] / b["calls"], b["plan"] / b["calls"]
            b["change_s"] = b["plan_s"] - b["today_s"]
            b = b[["calls", "today_s", "plan_s", "change_s"]].sort_values("change_s").round(2)
            b.to_csv(out / "boroughs.csv")

            result = {
                "scenario": sc,
                "rules": "alpha 1; busy engines; slowest 10% and average no worse than today; 10-minute cap; "
                         "pumps move with stations; street refinement"
                         + ("" if tol is None else f"; every borough's average within {tol:g} s of today"),
                "borough_tolerance_s": tol,
                "average_s": {"today": round(E_today, 1), "plan": round(E, 1), "best_possible": E_best},
                "slowest10_s": {"today": cur_m["cvar90_response_s"], "plan": plan_m["cvar90_response_s"]},
                "area_average_s": {"today": round(Q_today, 1), "plan": round(Q, 1), "best_possible": Q_best},
                "efficiency_score": round(100 * (E_today - E) / (E_today - E_best), 1) if E_today != E_best else None,
                "equity_score": round(100 * (Q_today - Q) / (Q_today - Q_best), 1) if Q_today != Q_best else None,
                "within_6min": {"today": cur_m["lfb_first_engine"]["share_within_6min"],
                                "plan": plan_m["lfb_first_engine"]["share_within_6min"]},
                "eval_2025_mean_s": {"today": round(s["eval"]["current"]["mean_s"], 1),
                                     "plan": round(s["eval"]["plan"]["mean_s"], 1),
                                     "recorded": round(s["eval"]["observed_attendance"]["mean_s"], 1)},
                "boroughs_worse_by_1s_or_more": b.index[b["change_s"] >= 1].tolist(),
                "boroughs_better_by_5s_or_more": int((b["change_s"] <= -5).sum()),
                "opened": rec.opened["site_name"].astype(str).tolist() if len(rec.opened) else [],
                "closed": rec.closed["site_name"].astype(str).tolist() if len(rec.closed) else [],
                "pump_moves": s.get("pump_moves"),
                "floors": s.get("floors"),
                "solver": {k: v for k, v in s["solver"].items() if k != "cap"},
                "street_refinement": s.get("street_refinement"),
                "run_minutes": round((time.time() - t0) / 60, 1),
            }
            (out / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
            rec.selected.to_csv(out / "planned_fire_stations.csv", index=False)
            if len(rec.opened):
                rec.opened.to_csv(out / "opened_fire_stations.csv", index=False)
            if len(rec.closed):
                rec.closed.to_csv(out / "closed_fire_stations.csv", index=False)
            try:
                _write_figures(rec, out, 10.0)
                _write_map(rec, out / "fire_station_plan_map.html", 10.0)
            except Exception as e:  # figures are a convenience; keep the numbers
                print("   figure/map failed:", e)
            top[sc] = {k: result[k] for k in ("average_s", "slowest10_s", "area_average_s", "efficiency_score",
                                              "equity_score", "within_6min", "eval_2025_mean_s",
                                              "boroughs_worse_by_1s_or_more", "opened", "closed")}
            print(json.dumps(top[sc], indent=1, default=str), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "summary" + ("" if tols == [None] else "_boroughs") + (
        "" if args.scenarios == ",".join(SCENARIOS) else "_" + args.scenarios.replace(",", "-"))
    (OUT / f"{name}.json").write_text(json.dumps(top, indent=2, default=str) + "\n")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
