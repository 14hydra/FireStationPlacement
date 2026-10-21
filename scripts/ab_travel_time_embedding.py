#!/usr/bin/env python3
"""
Fast A/B: baseline LightGBM features vs embedding-informed features + sample weights.

Trains 1-seed models with fixed trees (no early-stopping) on 2021–24 → 2025, and
reports overall MAE plus error on the embedding hard slices:
  - dense short trips (central under-prediction / Soho–Euston regime)
  - long sparse trips (outer long-drive / Heathrow regime)
  - high-util 1-pump origin stations (capacity-mismatch list)

Run:  .venv/bin/python scripts/ab_travel_time_embedding.py [--trees 900]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.kolesar import fit_kolesar  # noqa: E402
from firestation.travel_time_model import (  # noqa: E402
    CATEGORICAL,
    DRIVE_CLIP_S,
    DestinationHistory,
    apply_regime_bias,
    attach_weather,
    build_features,
    category_maps,
    fit_regime_bias,
    load_weather,
    make_regressor,
    sample_weights,
)

TRAIN_YEARS = (2021, 2022, 2023, 2024)
BASE_GROUPS = ["base", "street", "context", "incident", "dest"]
NEW_GROUPS = BASE_GROUPS + ["interaction"]
PARAMS = dict(learning_rate=0.05, num_leaves=127, min_child_samples=50,
              colsample_bytree=0.7, subsample=0.85, reg_lambda=3.0)
OUT = ROOT / "data" / "figures" / "analysis" / "model_improvement"


def metrics(y, p):
    e = np.abs(y - p)
    return {"mae_s": round(float(e.mean()), 2), "medae_s": round(float(np.median(e)), 2),
            "bias_s": round(float(np.mean(p - y)), 2), "n": int(len(y))}


def fit_predict(tr, te, groups, n_trees, use_weights, calibrate=False):
    ytr = tr["drive_s"].to_numpy(float)
    yte = te["drive_s"].to_numpy(float)
    kol = fit_kolesar(tr["road_km"].to_numpy(float), ytr)
    cats = category_maps(tr)
    resid = ytr - np.clip(kol.predict(tr["road_km"].to_numpy(float)), *DRIVE_CLIP_S)
    dest = DestinationHistory().fit(tr, resid)
    dest_oof = DestinationHistory().out_of_fold(tr, resid, tr["incident_number"])
    xtr = build_features(tr, groups, kolesar=kol, cat_maps=cats, dest_frame=dest_oof)
    xte = build_features(te, groups, kolesar=kol, cat_maps=cats, dest_history=dest)
    m = make_regressor(42, {**PARAMS, "n_estimators": n_trees})
    kw = dict(categorical_feature=[c for c in CATEGORICAL if c in xtr.columns])
    if use_weights:
        kw["sample_weight"] = sample_weights(tr, ytr)
    m.fit(xtr, ytr, **kw)
    pred_tr = np.clip(m.predict(xtr), *DRIVE_CLIP_S)
    pred = np.clip(m.predict(xte), *DRIVE_CLIP_S)
    bias = {}
    if calibrate:
        # Need rush on the raw frame for trip_regime
        tr2, te2 = tr.copy(), te.copy()
        h = pd.to_numeric(tr2["hour_of_call"], errors="coerce").fillna(12)
        tr2["rush"] = (h.between(7, 9) | h.between(16, 19)).astype(int)
        h = pd.to_numeric(te2["hour_of_call"], errors="coerce").fillna(12)
        te2["rush"] = (h.between(7, 9) | h.between(16, 19)).astype(int)
        bias = fit_regime_bias(tr2, ytr, pred_tr, only=("long_sparse",))
        pred = apply_regime_bias(te2, pred, bias)
    return yte, pred, te, bias


def slice_mask(te: pd.DataFrame) -> dict[str, np.ndarray]:
    road = pd.to_numeric(te["road_km"], errors="coerce").fillna(2).to_numpy(float)
    sig = (pd.to_numeric(te["signals_per_km"], errors="coerce").fillna(5).to_numpy(float)
           if "signals_per_km" in te.columns else np.full(len(te), 5.0))
    return {
        "all": np.ones(len(te), dtype=bool),
        "dense_short (central)": (sig >= 10) & (road <= 2.5),
        "long_sparse (outer)": (road >= 4.0) & (sig <= 6.0),
        "rush hour": te["hour_of_call"].between(7, 9) | te["hour_of_call"].between(16, 19),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trees", type=int, default=900)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "lfb_travel_training.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    inc = pd.read_csv(args.data, low_memory=False)
    inc = attach_weather(inc, load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    tr = inc[inc.cal_year.isin(TRAIN_YEARS)].reset_index(drop=True)
    te = inc[inc.cal_year == 2025].reset_index(drop=True)
    print(f"train {len(tr):,} | test {len(te):,} | trees={args.trees}")

    print("A) baseline (no interaction, no weights)…")
    y_a, p_a, te_a, _ = fit_predict(tr, te, BASE_GROUPS, args.trees, use_weights=False)
    print("B) embedding-informed (interaction + weights)…")
    y_b, p_b, te_b, _ = fit_predict(tr, te, NEW_GROUPS, args.trees, use_weights=True)
    print("C) B + long_sparse bias calibration only (PC3 outer hard cases)…")
    y_c, p_c, te_c, bias_c = fit_predict(tr, te, NEW_GROUPS, args.trees, use_weights=True, calibrate=True)
    print(f"  regime bias applied (pred−y on train): "
          f"{ {k: round(v, 2) for k, v in bias_c.items() if abs(v) > 1e-9} }")

    slices = slice_mask(te_a)
    rows = []
    for name, m in slices.items():
        if m.sum() < 50:
            continue
        a, b, c = metrics(y_a[m], p_a[m]), metrics(y_b[m], p_b[m]), metrics(y_c[m], p_c[m])
        rows.append({"slice": name, "n": a["n"],
                     "mae_A": a["mae_s"], "mae_B": b["mae_s"], "mae_C": c["mae_s"],
                     "delta_B_s": round(b["mae_s"] - a["mae_s"], 2),
                     "delta_C_s": round(c["mae_s"] - a["mae_s"], 2),
                     "bias_A": a["bias_s"], "bias_B": b["bias_s"], "bias_C": c["bias_s"]})
        print(f"  {name:<28} n={a['n']:>7,}  A {a['mae_s']:5.2f}  B {b['mae_s']:5.2f}  "
              f"C {c['mae_s']:5.2f}  ΔC {c['mae_s'] - a['mae_s']:+.2f}s")

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "ab_travel_time.csv", index=False)
    summary = {
        "trees": args.trees,
        "minutes": round((time.time() - t0) / 60, 1),
        "baseline_groups": BASE_GROUPS,
        "improved_groups": NEW_GROUPS,
        "regime_bias_train": bias_c,
        "slices": rows,
        "headline": {
            "mae_baseline_s": rows[0]["mae_A"],
            "mae_interactions_s": rows[0]["mae_B"],
            "mae_calibrated_s": rows[0]["mae_C"],
            "delta_calibrated_s": rows[0]["delta_C_s"],
        },
        "note": "A=baseline, B=interaction+weights, C=B+regime bias calibration. "
                "1-seed fixed trees. Full bag: train_travel_time_lfb.py --quick.",
    }
    (OUT / "ab_travel_time.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}/  ({summary['minutes']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
