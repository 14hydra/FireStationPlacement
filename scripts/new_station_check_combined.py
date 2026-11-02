#!/usr/bin/env python3
"""New-location test for the combined model (learned street speeds + LightGBM correction).

Same 20 station groups as scripts/new_station_check.py. For each group the street speeds and
the correction are both fitted without that group's stations, and predict its 2025 trips. The
LightGBM-alone predictions come from that script's saved folds, so the comparison is on the
same trips with the same settings (1 seed).

Writes data/figures/validation/new_stations_combined/summary.json and by_distance.csv.

Run:  PYTHONPATH=src .venv/bin/python scripts/new_station_check_combined.py [--rounds 3]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from firestation import ml_scorer  # noqa: E402
from firestation.street_speeds import StreetSpeedModel  # noqa: E402
from firestation.travel_time_model import DRIVE_CLIP_S, attach_weather, load_weather  # noqa: E402
from train_travel_time_lfb import FINAL_GROUPS, TRAIN_YEARS, TUNED_PARAMS, Split, fit_one, metrics  # noqa: E402

OUT = ROOT / "data" / "figures" / "validation" / "new_stations_combined"
LGBM_FOLDS = ROOT / "data" / "figures" / "validation" / "new_stations" / "folds"
DATA = ROOT / "data" / "processed" / "lfb_travel_training.csv"
FOLDS = 20
KEY = ["incident_number", "deployed_from_station", "y"]


def summarize(parts: list[pd.DataFrame], params: dict, t0: float) -> None:
    d = pd.concat(parts, ignore_index=True)
    lg = pd.concat([pd.read_parquet(f, columns=KEY + ["unseen", "kolesar"]) for f in LGBM_FOLDS.glob(f"fold_{FOLDS}_*.parquet")])
    d = d.drop_duplicates(KEY).merge(lg.drop_duplicates(KEY), on=KEY, how="inner")
    sc = (d["pump_order"] == 1) & (d["deployed_from_location"] == "Home Station") & (d["busy_flag"] == 0)
    preds = {"lightgbm_alone": "unseen", "combined": "combined", "street_speeds_alone": "base", "kolesar": "kolesar"}
    y = d["y"].to_numpy()
    out = {"doc": __doc__, "folds_done": len(parts), "trips": int(len(d)), "params": params,
           "all": {k: metrics(y, d[c].to_numpy()) for k, c in preds.items()},
           "placement": {k: metrics(y[sc], d[c].to_numpy()[sc]) for k, c in preds.items()}}
    st = d.assign(a=(d["unseen"] - d["y"]).abs(), b=(d["combined"] - d["y"]).abs()).groupby("deployed_from_station")[["a", "b"]].mean()
    out["stations_where_combined_beats_lightgbm"] = f"{int((st['b'] < st['a']).sum())} of {len(st)}"
    lab = pd.cut(d["road_km"], [0, 2, 4, 6, 100], labels=["<2", "2-4", "4-6", "6+"])
    rows = []
    for b in lab.cat.categories:
        m = (lab == b).to_numpy()
        rows.append({"road_km": b, "trips": int(m.sum()), **{f"mae_{k}": round(float(np.abs(d[c].to_numpy()[m] - y[m]).mean()), 2) for k, c in preds.items()},
                     **{f"bias_{k}": round(float((d[c].to_numpy()[m] - y[m]).mean()), 2) for k, c in preds.items()}})
    bd = pd.DataFrame(rows)
    bd.to_csv(OUT / "by_distance.csv", index=False)
    out["by_distance"] = rows
    out["minutes_this_run"] = round((time.time() - t0) / 60, 1)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("doc", "by_distance", "params")}, indent=1))
    print(bd.to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    (OUT / "folds").mkdir(parents=True, exist_ok=True)
    m = json.loads((ROOT / "data" / "processed" / "models" / "travel_time_lfb.metrics.json").read_text())
    params = {**TUNED_PARAMS, "n_estimators": int(m["lightgbm_params"]["n_estimators"])}
    t0 = time.time()
    inc = pd.read_csv(DATA, low_memory=False)
    inc = attach_weather(inc, load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    inc = inc[inc.cal_year.isin(TRAIN_YEARS + (2025,))].reset_index(drop=True)
    keep64 = ("incident_number", "drive_s", "start_lat", "start_lon", "dest_lat", "dest_lon")
    floats = [c for c in inc.select_dtypes("float64").columns if c not in keep64]
    inc[floats] = inc[floats].astype("float32")
    clip = lambda p: np.clip(p, *DRIVE_CLIP_S)  # noqa: E731

    # The same station groups as new_station_check.py.
    stations = np.array(sorted(inc["deployed_from_station"].dropna().unique()))
    fold_of = dict(zip(stations, np.random.default_rng(0).permutation(len(stations)) % FOLDS))
    f = inc["deployed_from_station"].map(fold_of)
    tr_year = inc.cal_year.isin(TRAIN_YEARS)
    router = ml_scorer._router()
    parts = []
    for i in range(FOLDS):
        fp = OUT / "folds" / f"fold_{i}.parquet"
        if fp.exists():
            parts.append(pd.read_parquet(fp))
            continue
        t1 = time.time()
        tr, te = inc[tr_year & (f != i)], inc[(inc.cal_year == 2025) & (f == i)]
        sm = StreetSpeedModel().fit(router, tr["start_lat"], tr["start_lon"], tr["dest_lat"], tr["dest_lon"],
                                    tr["drive_s"].to_numpy(float), rounds=args.rounds)
        tr = tr.assign(street_base_s=sm.predict(router, tr["start_lat"], tr["start_lon"], tr["dest_lat"], tr["dest_lon"]))
        te = te.assign(street_base_s=sm.predict(router, te["start_lat"], te["start_lon"], te["dest_lat"], te["dest_lon"]))
        s = Split(tr, te)
        xtr, xte = s.x(FINAL_GROUPS)
        btr, bte = s.tr["street_base_s"].to_numpy(float), s.te["street_base_s"].to_numpy(float)
        xtr["street_base_s"], xte["street_base_s"] = btr, bte
        cor = fit_one(xtr, s.ytr - btr, params)
        o = s.te[["incident_number", "deployed_from_station", "road_km", "pump_order", "busy_flag", "deployed_from_location"]].copy()
        o["y"], o["base"], o["combined"] = s.yte, clip(bte), clip(bte + cor.predict(xte))
        o.to_parquet(fp)
        parts.append(o)
        del sm, tr, te, s, xtr, xte, cor
        gc.collect()
        print(f"fold {i + 1}/{FOLDS}: {len(o):,} trips, MAE {np.abs(o['combined'] - o['y']).mean():.1f} s, {time.time() - t1:.0f} s", flush=True)
        summarize(parts, params, t0) if (i + 1) % 5 == 0 else None
    summarize(parts, params, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
