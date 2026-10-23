#!/usr/bin/env python3
"""Attach embedding-informed regime bias to the saved travel-time model (no retrain).

WARNING: only safe when ``predict()`` sees the same feature columns the model was
trained on (full street / dest features). Prefer baking calibration into
``scripts/train_travel_time_lfb.py`` (already does ``fit_regime_bias`` on train).
A CSV missing route features yields huge bogus bias and must not be written back.

Run:  PYTHONPATH=src .venv/bin/python scripts/calibrate_travel_time_regimes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.travel_time_model import (  # noqa: E402
    apply_regime_bias,
    fit_regime_bias,
    predict,
    trip_regime,
)

MODEL = ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib"
METRICS = MODEL.with_suffix(".metrics.json")
TRIPS = ROOT / "data" / "processed" / "lfb_travel_training.csv"
OUT = ROOT / "data" / "figures" / "analysis" / "model_improvement"


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def load_years(years):
    cols = ["cal_year", "drive_s", "road_km", "crow_km", "date_of_call", "hour_of_call",
            "busy_flag", "pump_order", "dest_exact", "borough", "incident_group",
            "property_category", "deployed_from_location", "signals_per_km", "junctions_per_km",
            "dest_lat", "dest_lon"]
    # Only request columns that exist
    peek = pd.read_csv(TRIPS, nrows=2)
    usecols = [c for c in cols if c in peek.columns]
    df = pd.read_csv(TRIPS, usecols=usecols)
    return df[df["cal_year"].isin(list(years))].dropna(subset=["drive_s", "road_km"]).copy()


def main() -> int:
    print("loading model + trips…", flush=True)
    bundle = joblib.load(MODEL)
    tr = load_years(range(2021, 2025))
    te = load_years([2025])
    # Cap for speed if huge
    if len(tr) > 250_000:
        tr = tr.sample(250_000, random_state=0)
    print(f"  train {len(tr):,}  test {len(te):,}", flush=True)

    # Predictions WITHOUT existing regime bias
    b0 = {k: v for k, v in bundle.items() if k != "regime_bias"}
    print("  predicting train…", flush=True)
    ptr = predict(b0, tr)
    print("  predicting test (before)…", flush=True)
    pte0 = predict(b0, te)
    ytr = tr["drive_s"].to_numpy(float)
    yte = te["drive_s"].to_numpy(float)

    bias = fit_regime_bias(tr, ytr, ptr, only=("long_sparse",))
    print("  regime bias (train pred−y):", {k: round(v, 2) for k, v in bias.items()}, flush=True)

    pte1 = apply_regime_bias(te, pte0, bias)
    # Also report by slice
    reg = trip_regime(te)

    def slice_mae(p):
        out = {"all": round(mae(yte, p), 2)}
        for name in ("dense_short", "long_sparse", "rush_other", "other"):
            m = reg == name
            if m.any():
                out[name] = round(mae(yte[m], p[m]), 2)
        return out

    before, after = slice_mae(pte0), slice_mae(pte1)
    print("MAE before:", before)
    print("MAE after: ", after)

    bundle["regime_bias"] = bias
    bundle["regime_bias_note"] = (
        "Post-hoc correction from scripts/calibrate_travel_time_regimes.py; "
        "long_sparse only (embedding PC3 / outer long drives)."
    )
    joblib.dump(bundle, MODEL)
    if METRICS.exists():
        meta = json.loads(METRICS.read_text())
    else:
        meta = {}
    meta["regime_bias"] = bias
    meta["regime_calibration_2025"] = {"before": before, "after": after}
    METRICS.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "regime_calibration.json").write_text(
        json.dumps({"bias": bias, "before": before, "after": after}, indent=2) + "\n")
    print("wrote", MODEL)
    print("wrote", OUT / "regime_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
