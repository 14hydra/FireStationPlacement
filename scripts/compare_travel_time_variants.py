#!/usr/bin/env python3
"""Fair comparison of the travel time model with and without the embedding-informed changes.

Every variant uses the production settings (TUNED_PARAMS, 2,110 trees, the same seed),
trains on 2021-2024 and is scored on 2025 trips, overall and on the hard cases the
station embedding pointed at:

  A  production features (base, street, context, incident, dest), no sample weights
  B  A + interaction features (rush x signals, rush x junctions, night x signals,
     dense_short, long_sparse)
  C  A + sample weights (long trips and dense short trips up-weighted, implausible
     speeds down-weighted)
  D  B + C (the embedding model)
  E  D + long_sparse bias correction fitted on D's predictions for its own training trips
     (in-sample, as first shipped)
  F  D + long_sparse bias correction fitted honestly: a model trained on 2021-2023 with
     D's settings predicts 2024, and its mean error on 2024 long_sparse trips is the
     correction applied to D's 2025 predictions

Slices: all trips; dense_short (>= 10 signals/km, <= 2.5 km); long_sparse (>= 4 km,
<= 6 signals/km); rush hour; trips over 6 km.

Writes data/figures/analysis/model_improvement/fair_comparison.csv and .json.

Run:  PYTHONPATH=src .venv/bin/python scripts/compare_travel_time_variants.py
"""

from __future__ import annotations

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

from firestation.travel_time_model import (  # noqa: E402
    DRIVE_CLIP_S,
    INTERACTION_FEATURES,
    apply_regime_bias,
    attach_weather,
    fit_regime_bias,
    load_weather,
    sample_weights,
    trip_regime,
)
from train_travel_time_lfb import TUNED_PARAMS, Split, fit_one, metrics  # noqa: E402

OUT = ROOT / "data" / "figures" / "analysis" / "model_improvement"
DATA = ROOT / "data" / "processed" / "lfb_travel_training.csv"
PROD_GROUPS = ["base", "street", "context", "incident", "dest"]
ALL_GROUPS = PROD_GROUPS + ["interaction"]


def with_rush(df: pd.DataFrame) -> pd.DataFrame:
    h = pd.to_numeric(df["hour_of_call"], errors="coerce").fillna(12)
    return df.assign(rush=(h.between(7, 9) | h.between(16, 19)).astype(int))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = json.loads((ROOT / "data" / "processed" / "models" / "travel_time_lfb.metrics.json").read_text())
    params = {**TUNED_PARAMS, "n_estimators": int(m["lightgbm_params"]["n_estimators"])}
    print("settings:", params, flush=True)
    clip = lambda p: np.clip(p, *DRIVE_CLIP_S)  # noqa: E731

    inc = pd.read_csv(DATA, low_memory=False)
    inc = attach_weather(inc, load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    inc = inc[inc.cal_year.between(2021, 2025)].reset_index(drop=True)

    # F's honest correction: D's settings trained on 2021-2023, scored on 2024.
    print("F: honest long_sparse correction (train 2021-2023, measure on 2024)…", flush=True)
    s24 = Split(inc[inc.cal_year.between(2021, 2023)], inc[inc.cal_year == 2024])
    x24tr, x24te = s24.x(ALL_GROUPS)
    m24 = fit_one(x24tr, s24.ytr, params, sample_weight=sample_weights(s24.tr, s24.ytr))
    honest_bias = fit_regime_bias(with_rush(s24.te), s24.yte, clip(m24.predict(x24te)))
    print("  honest bias (2024, pred - y):", {k: round(v, 2) for k, v in honest_bias.items()}, flush=True)
    del s24, x24tr, x24te, m24
    gc.collect()

    s = Split(inc[inc.cal_year.between(2021, 2024)], inc[inc.cal_year == 2025])
    del inc
    gc.collect()
    xtr, xte = s.x(ALL_GROUPS)
    prod_cols = [c for c in xtr.columns if c not in INTERACTION_FEATURES]
    sw = sample_weights(s.tr, s.ytr)
    preds, notes = {}, {}
    for name, cols, w in [("A production", prod_cols, None), ("B + interactions", list(xtr.columns), None),
                          ("C + sample weights", prod_cols, sw), ("D + both (embedding model)", list(xtr.columns), sw)]:
        t1 = time.time()
        model = fit_one(xtr[cols], s.ytr, params, sample_weight=w)
        preds[name] = clip(model.predict(xte[cols]))
        print(f"  {name}: MAE {np.abs(preds[name] - s.yte).mean():.2f} s ({time.time() - t1:.0f} s)", flush=True)
        if name.startswith("D"):
            insample = fit_regime_bias(with_rush(s.tr), s.ytr, clip(model.predict(xtr[cols])))
            notes = {"in_sample_bias": insample, "honest_bias": honest_bias}
            te_r = with_rush(s.te)
            preds["E D + in-sample correction"] = apply_regime_bias(te_r, preds[name], insample)
            preds["F D + honest correction"] = apply_regime_bias(te_r, preds[name], honest_bias)
        del model
        gc.collect()

    reg = trip_regime(with_rush(s.te))
    km = s.te["road_km"].to_numpy(float)
    rush = with_rush(s.te)["rush"].to_numpy() == 1
    slices = {"all": np.ones(len(s.yte), bool), "dense_short (central)": reg == "dense_short",
              "long_sparse (outer)": reg == "long_sparse", "rush hour": rush, "over 6 km": km >= 6}
    rows = []
    for name, p in preds.items():
        for sl, msk in slices.items():
            e = p[msk] - s.yte[msk]
            rows.append({"variant": name, "slice": sl, "n": int(msk.sum()), "mae_s": round(float(np.abs(e).mean()), 2),
                         "medae_s": round(float(np.median(np.abs(e))), 2), "bias_s": round(float(e.mean()), 2)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fair_comparison.csv", index=False)
    wide = df.pivot_table(index="variant", columns="slice", values="mae_s")
    (OUT / "fair_comparison.json").write_text(json.dumps({
        "doc": __doc__, "settings": params, "seed": 42, "n_test_2025": int(len(s.yte)),
        "mae_by_variant_and_slice": wide.round(2).to_dict(orient="index"),
        "overall": {k: metrics(s.yte, p) for k, p in preds.items()}, **notes,
        "minutes": round((time.time() - t0) / 60, 1)}, indent=2, default=str) + "\n")
    pd.set_option("display.width", 200)
    print(wide.round(2).to_string())
    print("notes:", json.dumps(notes, default=lambda v: round(v, 2)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
