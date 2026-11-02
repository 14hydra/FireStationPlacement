#!/usr/bin/env python3
"""Does training for the trips placement scores make the travel time model better on them?

Placement scores the first engine, from its home station, not busy. Same settings and seed
as production, 2021-2024 to 2025, scored on all trips and on placement's trips:

  A  production: every trip, equal weight
  B  trains only on placement's trips
  C  every trip, placement's trips weighted 3x
  D  every trip except those where the engine was probably not at its station
     (sent from another station, or turnout under 30 s)

Writes data/figures/analysis/model_improvement/placement_training.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/compare_placement_training.py
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


def scenario(df: pd.DataFrame) -> pd.Series:
    return (df["pump_order"] == 1) & (df["deployed_from_location"] == "Home Station") & (df["busy_flag"] == 0)


def main() -> int:
    m = json.loads((ROOT / "data" / "processed" / "models" / "travel_time_lfb.metrics.json").read_text())
    params = {**TUNED_PARAMS, "n_estimators": int(m["lightgbm_params"]["n_estimators"])}
    inc = pd.read_csv(DATA, low_memory=False)
    inc = attach_weather(inc, load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    tr = inc[inc.cal_year.between(2021, 2024)].reset_index(drop=True)
    te = inc[inc.cal_year == 2025].reset_index(drop=True)
    del inc
    gc.collect()
    y = te["drive_s"].to_numpy(float)
    sc = scenario(te).to_numpy()
    tsc = scenario(tr).to_numpy()
    away = (tr["deployed_from_location"] != "Home Station") | (tr["turnout_s"] < 30)
    out = {}
    for name, keep, weight in [("A production", np.ones(len(tr), bool), None), ("B placement trips only", tsc, None),
                               ("C placement trips x3", np.ones(len(tr), bool), np.where(tsc, 3.0, 1.0)),
                               ("D without likely-away engines", ~away.to_numpy(), None)]:
        s = Split(tr[keep], te)
        xtr, xte = s.x(FINAL_GROUPS)
        p = np.clip(fit_one(xtr, s.ytr, params, sample_weight=weight).predict(xte), *DRIVE_CLIP_S)
        e = np.abs(p - y)
        out[name] = {"train_trips": int(keep.sum()), "all_mae_s": round(float(e.mean()), 2),
                     "placement_mae_s": round(float(e[sc].mean()), 2),
                     "placement_medae_s": round(float(np.median(e[sc])), 2)}
        print(name, out[name], flush=True)
        del s, xtr, xte
        gc.collect()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "placement_training.json").write_text(json.dumps({"doc": __doc__, "settings": params, "results": out}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
