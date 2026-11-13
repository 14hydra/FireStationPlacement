#!/usr/bin/env python3
"""Test B of docs/prediction_relocations_and_late_2026.md: the Part 2 main layout on August-December 2026, once.

The same scoring as scripts/final_test_2026.py (its functions are reused unchanged): demand cells, busy-engine rates,
turnout and the Kolesar fit from 2024; the layouts and pass marks of docs/prediction_2026.md; three travel models. Only
the calls being counted change: 1 August 2026 to the last complete month.

When it runs (fixed in the prediction):
  - A month counts once the download holds calls on or after that month's last day.
  - All of August-December complete: the test runs once on August-December.
  - Otherwise, from 2027-02-15: it runs once on the complete months from August if there are at least 3, labelled
    "partial (August to <month>)"; with fewer than 3 it records "not run" and never runs.
  - Before 2027-02-15 with any month missing: it refuses and says what is missing.
  - lfb_travel_training.csv must hold trips in the window's last month (rebuild it with the same build script).

  --check aug-dec-2025   dry run on August-December 2025 (a used year): checks the code runs. Any number of times.
  --check jan-jul-2026   dry run on January-July 2026, already scored by final_test_2026.py: must reproduce its
                         primary scores (target, London, worst borough) within 0.05 s. Any number of times.
  (no flag)              the real test. Refuses if the prediction or this script has uncommitted changes, or if it has
                         already been run (data/figures/placement/final_test_2026_late/result.json exists).

Run:  PYTHONPATH=src .venv/bin/python scripts/final_test_2026_late.py --check jan-jul-2026
      PYTHONPATH=src .venv/bin/python scripts/final_test_2026_late.py --check aug-dec-2025
      PYTHONPATH=src .venv/bin/python scripts/final_test_2026_late.py          # once only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import final_test_2026 as F  # noqa: E402

from firestation.busy_placement import ErlangBusyModel  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid, posture_seconds, weighted_cvar  # noqa: E402
from firestation.fire_station_location import (build_travel_matrix, busy_prob_vector, drop_far_incidents,  # noqa: E402
                                               fit_scorer_kolesar, recorded_station_turnout, snap_to_cells)
from firestation.ml_scorer import cell_boroughs  # noqa: E402
from firestation.station_capacity import default_pump_table  # noqa: E402
from firestation.travel_time_model import attach_weather, load_weather, predict  # noqa: E402

PL = ROOT / "data" / "figures" / "placement"
OUT = PL / "final_test_2026_late"
PREDICTION = ROOT / "docs" / "prediction_relocations_and_late_2026.md"
DEADLINE = date(2027, 2, 15)
MONTHS = [8, 9, 10, 11, 12]
MIN_MONTHS = 3
CHECKS = {"aug-dec-2025": ("2025-08-01", "2025-12-31"), "jan-jul-2026": ("2026-01-01", "2026-07-31")}
REPRODUCE = PL / "final_test_2026" / "result.json"


def dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["date_of_call"], format="mixed")


def complete_months(newest: pd.Timestamp) -> list[int]:
    """August-December 2026 months whose last day is covered by the download, in order, stopping at the first gap."""
    out = []
    for m in MONTHS:
        if newest >= pd.Timestamp(2026, m, 1) + pd.offsets.MonthEnd(0):
            out.append(m)
        else:
            break
    return out


def drive_check(window: tuple[pd.Timestamp, pd.Timestamp]) -> dict:
    """Pass mark 3, as final_test_2026.drive_check, on trips in the window."""
    df = pd.read_csv(ROOT / "data" / "processed" / "lfb_travel_training.csv", low_memory=False)
    d = dates(df)
    te = df[(d >= window[0]) & (d <= window[1]) & (df["pump_order"] == 1) & (df["deployed_from_location"] == "Home Station")
            & (df["busy_flag"] == 0) & df["borough"].astype(str).str.upper().isin(F.TARGETS)].reset_index(drop=True)
    newest = d.max()
    del df
    if newest < window[1].replace(day=1):  # no trips in the window's last month: the file was not rebuilt
        raise SystemExit(f"lfb_travel_training.csv ends {newest.date()}, before {window[1].strftime('%B %Y')}: rebuild it with the same build script")
    te = attach_weather(te, load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    bundle = joblib.load(ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib")
    p = np.concatenate([predict(bundle, te.iloc[i:i + 50000]) for i in range(0, len(te), 50000)])
    y = te["drive_s"].to_numpy(float)
    rng = np.random.default_rng(window[0].year)
    bs = [np.polyfit(p[i], y[i], 1)[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
    return {"trips": int(len(y)), "slope": round(float(np.polyfit(p, y, 1)[0]), 3), "slope_lo": round(float(np.percentile(bs, 2.5)), 3),
            "slope_hi": round(float(np.percentile(bs, 97.5)), 3), "recorded_minus_model_s": round(float((y - p).mean()), 1)}


def score(window: tuple[pd.Timestamp, pd.Timestamp], out: Path, label: str) -> dict:
    """final_test_2026.main's scoring, with the calls counted taken from the window."""
    spec = get_city("london")
    houses = load_fire_stations(spec)
    incidents = drop_far_incidents(load_incidents(spec, max_rows=None, years=sorted({2024, window[0].year})), houses)
    demand = incidents[incidents["cal_year"] == 2024].reset_index(drop=True)
    d = dates(incidents)
    ev = incidents[(d >= window[0]) & (d <= window[1])].reset_index(drop=True)
    print(f"{label}: {len(ev)} calls scored ({window[0].date()} to {window[1].date()})", flush=True)
    cells = build_demand_grid(demand, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    w = cells["n_incidents"].to_numpy(float)
    boroughs = np.asarray(cell_boroughs(cells, demand)).astype(str)
    names = sorted(set(boroughs) - {""})
    is_t = np.isin(boroughs, F.TARGETS)
    cidx, _ = snap_to_cells(ev, cells)
    n = np.bincount(cidx, minlength=len(cells)).astype(float)
    N, tN = n.sum(), n[is_t].sum()
    pumps_today = default_pump_table(houses).set_index("station")["n_pumps"].reindex(houses["facilityname"].astype(str)).fillna(1).astype(int).to_numpy()
    tab, new_turn = recorded_station_turnout([2024])
    norm = {k.upper(): v for k, v in tab.items()}
    kmodel, _, _ = fit_scorer_kolesar(demand, houses)

    layouts = {"today": pd.DataFrame({"site_name": houses["facilityname"].astype(str), "lat": houses["lat"], "lon": houses["lon"],
                                      "is_existing": True, "n_pumps": pumps_today})}
    for k, f in F.LAYOUTS.items():
        layouts[k] = pd.read_csv(f)[["site_name", "lat", "lon", "is_existing", "n_pumps"]]
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
    boot = np.random.default_rng(0).multinomial(int(N), n / N, size=F.BOOT).astype(float)

    records, brows = [], []
    for sc in F.SCORERS:
        print(f"Scoring with {sc} ({int(N)} calls, {len(sites)} sites)", flush=True)
        Ts, _ = build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer=sc, kolesar=kmodel, incidents=demand)
        As = Ts + turnout[:, None]
        bp = np.ones(len(sites), int)
        bp[rows["today"]] = pumps["today"]
        m = ErlangBusyModel.calibrate(As, rows["today"], measured, bp, w)

        def resp(k):
            m.pumps = np.ones(len(sites), int)
            m.pumps[rows[k]] = pumps[k]
            return posture_seconds(As, rows[k], m.probs(As, rows[k]))

        q0 = resp("today")
        b0 = F.borough_means(q0, n, boroughs, names)
        for k in F.LAYOUTS:
            q = resp(k)
            dq = q - q0
            bd = F.borough_means(q, n, boroughs, names) - b0
            tb = boot[:, is_t] @ dq[is_t] / boot[:, is_t].sum(1)
            records.append({"layout": k, "scorer": sc, "moved": int((~layouts[k]["is_existing"].astype(bool)).sum()),
                            "d_target_s": round(float(n[is_t] @ dq[is_t] / tN), 2), "target_lo": round(float(np.percentile(tb, 2.5)), 2),
                            "target_hi": round(float(np.percentile(tb, 97.5)), 2), "d_london_s": round(float(n @ dq / N), 2),
                            "d_slowest10_s": round(weighted_cvar(q, n) - weighted_cvar(q0, n), 2),
                            "boroughs_worse_1s": int((bd >= 1).sum()), "worst_borough_s": round(float(bd.max()), 1),
                            "worst_borough": names[int(bd.argmax())], "calls_30s_slower_pct": round(float(n[dq >= 30].sum() / N * 100), 1)})
            brows += [{"layout": k, "scorer": sc, "borough": b, "target": b in F.TARGETS, "d_s": round(float(x), 2)} for b, x in zip(names, bd)]
    df = pd.DataFrame(records)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scores.csv", index=False)
    pd.DataFrame(brows).to_csv(out / "boroughs.csv", index=False)

    p = df[df["layout"] == F.PRIMARY].set_index("scorer")
    drive = drive_check(window)
    verdict = {"p1_targets_10s_faster_all_models": bool((p["d_target_s"] <= -10).all()),
               "p2_no_borough_1s_worse_any_model": bool((p["boroughs_worse_1s"] == 0).all()),
               "p3_drive_slope_0.9_to_1.1": bool(0.9 <= drive["slope"] <= 1.1)}
    print(df.to_string(index=False))
    return {"window": [str(window[0].date()), str(window[1].date())], "label": label, "calls": int(N), "target_calls": int(tN),
            "primary": F.PRIMARY,
            "primary_scores": p[["d_target_s", "target_lo", "target_hi", "d_london_s", "boroughs_worse_1s", "worst_borough_s"]].to_dict("index"),
            "drive_check": drive, "verdict": verdict, "all_pass": all(verdict.values()), "rows": records}


def newest_call() -> pd.Timestamp:
    spec = get_city("london")
    inc = load_incidents(spec, max_rows=None, years=[2026])
    return dates(inc).max()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", choices=sorted(CHECKS), help="Dry run on an already-used window")
    args = ap.parse_args()

    if args.check:
        a, b = (pd.Timestamp(x) for x in CHECKS[args.check])
        res = score((a, b), OUT / f"check_{args.check}", f"dry run {args.check}")
        res["dry_run"] = True
        if args.check == "jan-jul-2026":
            pub = json.loads(REPRODUCE.read_text())["primary_scores"]
            diffs = {sc: {k: round(res["primary_scores"][sc][k] - v, 3) for k, v in pub[sc].items()} for sc in pub}
            res["reproduces_final_test_2026"] = bool(all(abs(x) <= 0.05 for v in diffs.values() for x in v.values()))
            res["difference_from_final_test_2026"] = diffs
            print("Reproduces final_test_2026:", res["reproduces_final_test_2026"], json.dumps(diffs))
        (OUT / f"check_{args.check}" / "result.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
        return 0 if res.get("reproduces_final_test_2026", True) else 1

    if (OUT / "result.json").exists():
        print("Already run once; see", OUT / "result.json")
        return 1
    if not F.committed_clean(PREDICTION, Path(__file__).resolve()):
        print("Commit docs/prediction_relocations_and_late_2026.md and this script first.")
        return 1
    newest = newest_call()
    months = complete_months(newest)
    today = date.today()
    print(f"Newest call in the download: {newest.date()}; complete months from August: {months}", flush=True)
    if len(months) < len(MONTHS) and today < DEADLINE:
        print(f"Not yet: {len(months)} of 5 months complete and it is before {DEADLINE}. Download newer data, or wait.")
        return 1
    if len(months) < MIN_MONTHS:
        OUT.mkdir(parents=True, exist_ok=True)
        res = {"status": "not run", "reason": f"{len(months)} complete months from August 2026 by {DEADLINE}",
               "newest_call": str(newest.date()), "decided_on": str(today)}
        (OUT / "result.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps(res, indent=2))
        return 0
    end = pd.Timestamp(2026, months[-1], 1) + pd.offsets.MonthEnd(0)
    label = "full (August to December)" if len(months) == len(MONTHS) else f"partial (August to {end.strftime('%B')})"
    res = score((pd.Timestamp("2026-08-01"), end), OUT, label)
    res.update({"status": "run", "dry_run": False, "newest_call": str(newest.date()), "run_on": str(today)})
    (OUT / "result.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
