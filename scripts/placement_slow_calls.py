#!/usr/bin/env python3
"""Placement for the slow calls: average vs 90th-percentile objectives.

LFB's standard is about slow calls (first engine within 10 minutes on 90 % of calls),
but p-median placement minimises the average. This script solves each mode four ways
(exact solver, proven optimum, nearest engine free) and scores every layout on the same
measures:

  objective          travel times          what it minimises
  response_time      lgbm (median trips)   average attendance
  cvar               lgbm                  average of the slowest 10 % of calls (CVaR)
  response_time      lgbm_q90              average 90th-percentile attendance
  cvar               lgbm_q90              slowest 10 % of 90th-percentile attendance

Measures (2024 demand grid, all calls):
  mean_s        call-weighted average attendance (median model)
  cvar90_s      average of the slowest 10 % of calls (median model)
  q90_mean_s    call-weighted average 90th-percentile attendance (quantile model)
  meets_std     share of calls whose area meets LFB's 90 %-within-10-minutes standard,
                i.e. 90th-percentile attendance <= 600 s (quantile model)
  worst_q90_s   slowest area's 90th-percentile attendance

Writes data/figures/placement_slow_calls/: slow_calls.csv, summary.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/placement_slow_calls.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.facility import weighted_cvar  # noqa: E402
from firestation.fire_station_location import run_fire_station_plan  # noqa: E402

OUT = ROOT / "data" / "figures" / "placement_slow_calls"
MODES = {"expand_3": {"mode": "expand", "add_stations": 3},
         "replace_10": {"mode": "replace", "replace_r": 10},
         "redesign": {"mode": "redesign"}}
VARIANTS = [("response_time", "lgbm"), ("cvar", "lgbm"), ("response_time", "lgbm_q90"), ("cvar", "lgbm_q90")]


def run(houses, incidents, mode_kw, objective, scorer):
    kw = dict(mode_kw)
    if kw["mode"] == "redesign":
        kw["n_houses"] = len(houses)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_fire_station_plan(city_id="london", houses=houses, incidents=incidents, scorer=scorer,
                                     objective=objective, demand_years=[2024], eval_years=[2025],
                                     max_incident_rows=None, **kw)


def measures(idx, A_med, A_q90, w):
    med = A_med[idx].min(axis=0)
    # Each area is served by its nearest station under the median model.
    q90 = A_q90[np.asarray(idx)[A_med[idx].argmin(axis=0)], np.arange(A_med.shape[1])]
    W = w.sum()
    return {"mean_s": float((w * med).sum() / W), "cvar90_s": weighted_cvar(med, w),
            "q90_mean_s": float((w * q90).sum() / W), "meets_std": float((w * (q90 <= 600)).sum() / W),
            "worst_q90_s": float(q90[w > 0].max())}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = get_city("london")
    houses = load_fire_stations(spec)
    incidents = load_incidents(spec, max_rows=None, years=[2024, 2025])
    rows = []
    for name, mkw in MODES.items():
        res = {v: run(houses, incidents, mkw, *v) for v in VARIANTS}
        A_med = res[("response_time", "lgbm")].attendance
        A_q90 = res[("response_time", "lgbm_q90")].attendance
        cells = res[("response_time", "lgbm")].cells
        w = cells["weight"].to_numpy(float)
        today = res[("response_time", "lgbm")].candidates.index[
            res[("response_time", "lgbm")].candidates["is_existing"]].tolist()
        base = measures(today, A_med, A_q90, w)
        rows.append({"scenario": name, "objective": "today", "times": "-", **base})
        for (obj, sc), r in res.items():
            idx = r.candidates.index[r.candidates["candidate_id"].isin(r.selected["candidate_id"])].tolist()
            m = measures(idx, A_med, A_q90, w)
            rows.append({"scenario": name, "objective": obj, "times": sc, **m,
                         "optimal": r.summary["solver"]["optimal"], "gap": r.summary["solver"].get("gap"),
                         "n_opened": r.summary["n_opened"],
                         "opened": "; ".join(r.opened["site_name"].astype(str)) if len(r.opened) else ""})
            print(f"{name:10s} {obj:13s} {sc:8s} mean {m['mean_s']:.1f}  cvar90 {m['cvar90_s']:.1f}  "
                  f"q90 {m['q90_mean_s']:.1f}  meets 10-min/90% {m['meets_std']:.3f}  "
                  f"(today {base['mean_s']:.1f} / {base['cvar90_s']:.1f} / {base['q90_mean_s']:.1f} / "
                  f"{base['meets_std']:.3f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "slow_calls.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"doc": __doc__, "rows": df.round(4).to_dict(orient="records")},
                                                 indent=2) + "\n")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
