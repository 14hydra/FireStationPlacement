#!/usr/bin/env python3
"""Do traffic counts and roadworks along the route improve the travel time model?

Same settings and seed as production, 2021-2024 to 2025, with features from
scripts/build_traffic_features.py:

  A   production
  A2  production with another seed: how much the error moves by chance alone
  T   A + traffic
  R   A + roadworks
  TR  A + traffic + roadworks

Scored on all trips, placement's trips (first engine, home station, not busy), trips over 6 km,
rush hour and trips with a road closure or lane restriction on the route. Differences from A
have paired 95 % intervals, resampling days.

Writes data/figures/analysis/model_improvement/traffic_features.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/compare_traffic_features.py
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from firestation.travel_time_model import DRIVE_CLIP_S, attach_weather, load_weather  # noqa: E402
from train_travel_time_lfb import FINAL_GROUPS, TUNED_PARAMS, Split, fit_one  # noqa: E402

OUT = ROOT / "data" / "figures" / "analysis" / "model_improvement"
DATA = ROOT / "data" / "processed" / "lfb_travel_training.csv"
FEATURES = ROOT / "data" / "processed" / "trip_traffic_features.parquet"
TRAFFIC = ["route_flow_k", "route_flow_per_lane", "route_hour_flow_k", "route_counted_share"]
WORKS = ["works_closure", "works_lanes", "works_any"]


def main() -> int:
    m = json.loads((ROOT / "data" / "processed" / "models" / "travel_time_lfb.metrics.json").read_text())
    params = {**TUNED_PARAMS, "n_estimators": int(m["lightgbm_params"]["n_estimators"])}
    inc = pd.read_csv(DATA, low_memory=False)
    inc = inc[inc["cal_year"].between(2021, 2025)].reset_index(drop=True)
    f = pd.read_parquet(FEATURES)
    assert len(f) == len(inc) and (f["incident_number"].astype(str).to_numpy() == inc["incident_number"].astype(str).to_numpy()).all()
    inc[TRAFFIC + WORKS] = f[TRAFFIC + WORKS].to_numpy()
    del f
    inc = attach_weather(inc, load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    s = Split(inc[inc["cal_year"] < 2025], inc[inc["cal_year"] == 2025])
    del inc
    gc.collect()
    xtr, xte = s.x(FINAL_GROUPS)
    base = list(xtr.columns)
    for c in TRAFFIC + WORKS:
        xtr[c], xte[c] = s.tr[c].to_numpy(), s.te[c].to_numpy()
    te = s.te
    y = s.yte
    h = pd.to_numeric(te["hour_of_call"], errors="coerce").fillna(12)
    place = ((te["pump_order"] == 1) & (te["deployed_from_location"] == "Home Station") & (te["busy_flag"] == 0)).to_numpy()
    slices = {"all": np.ones(len(y), bool), "placement": place, "over 6 km": te["road_km"].to_numpy() >= 6,
              "rush hour": (h.between(7, 9) | h.between(16, 19)).to_numpy(),
              "closure or lane works on route": (te["works_closure"] + te["works_lanes"]).to_numpy() > 0}
    days = pd.factorize(te["date_of_call"])[0]
    preds, imp = {}, {}
    for name, cols, seed in [("A production", base, 42), ("A2 other seed", base, 43), ("T traffic", base + TRAFFIC, 42),
                             ("R roadworks", base + WORKS, 42), ("TR both", base + TRAFFIC + WORKS, 42)]:
        model = fit_one(xtr[cols], s.ytr, params, seed=seed)
        preds[name] = np.clip(model.predict(xte[cols]), *DRIVE_CLIP_S)
        g = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=cols)
        imp[name] = (g / g.sum()).reindex([c for c in TRAFFIC + WORKS if c in cols]).round(4).to_dict()
        print(name, {k: round(float(np.abs(preds[name][v] - y[v]).mean()), 2) for k, v in slices.items()}, flush=True)
        del model
        gc.collect()

    rng = np.random.default_rng(0)
    n_days = days.max() + 1
    out = {"n": {k: int(v.sum()) for k, v in slices.items()}, "mae_s": {}, "change_from_A_s": {}}
    ae_a = np.abs(preds["A production"] - y)
    for name, p in preds.items():
        ae = np.abs(p - y)
        out["mae_s"][name] = {k: round(float(ae[v].mean()), 2) for k, v in slices.items()}
        if name.startswith("A production"):
            continue
        out["change_from_A_s"][name] = {}
        for k, v in slices.items():
            d, dd = (ae - ae_a)[v], days[v]
            tot, cnt = np.bincount(dd, d, n_days), np.bincount(dd, minlength=n_days)
            w = rng.multinomial(n_days, np.full(n_days, 1 / n_days), size=500)
            boot = (w * tot).sum(1) / np.maximum((w * cnt).sum(1), 1)
            out["change_from_A_s"][name][k] = [round(float(d.mean()), 2), *np.percentile(boot, [2.5, 97.5]).round(2).tolist()]
    out["importance_gain_share"] = imp
    out["features_summary"] = te[TRAFFIC + WORKS].describe().round(3).to_dict()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traffic_features.json").write_text(json.dumps({"doc": __doc__, "settings": params, **out}, indent=2) + "\n")
    print(json.dumps(out["change_from_A_s"], indent=1))
    print(json.dumps(imp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
