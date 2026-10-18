#!/usr/bin/env python3
"""90th-percentile driving-time model (quantile LightGBM) for slow-call placement.

The main model (train_travel_time_lfb.py) uses absolute-error loss, so it predicts the
median trip. LFB's standard is about the slow calls (first engine within 10 minutes on
90 % of calls), so this script trains the same features with quantile loss at 0.9: the
prediction is the time 90 % of such trips beat.

Same split and features as the main model (FINAL_GROUPS, trained 2021–24, tested 2025);
trees by early stopping on 2024 (trained 2021–23). Reports on 2025:
  - coverage: share of trips at or under the prediction (target 0.90), overall and by
    road-distance band;
  - pinball (quantile) loss vs the median model + a constant offset.

Saves data/processed/models/travel_time_lfb_q90.joblib in the main bundle format, so
firestation.travel_time_model.predict works on it, and the planner scorer ``lgbm_q90``.

Run:  .venv/bin/python scripts/train_travel_time_quantile.py [--alpha 0.9]
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
sys.path.insert(0, str(ROOT / "scripts"))

from train_travel_time_lfb import FINAL_GROUPS, TRAIN_YEARS, TUNED_PARAMS, Split  # noqa: E402

from firestation.travel_time_model import CATEGORICAL, DRIVE_CLIP_S, make_regressor  # noqa: E402

BANDS_KM = [0, 1, 2, 3, 5, 8, 100]


def pinball(y, q, alpha):
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def main() -> int:
    import joblib
    import lightgbm as lgb

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "lfb_travel_training.csv")
    ap.add_argument("--test-year", type=int, default=2025)
    args = ap.parse_args()
    tag = f"q{int(round(args.alpha * 100))}"
    out = ROOT / "data" / "processed" / "models" / f"travel_time_lfb_{tag}.joblib"
    t0 = time.time()

    inc = pd.read_csv(args.data, low_memory=False)
    inc = inc[inc.cal_year.isin(TRAIN_YEARS + (args.test_year,))].reset_index(drop=True)
    params = {**TUNED_PARAMS, "objective": "quantile", "alpha": args.alpha}
    cats = lambda x: [c for c in CATEGORICAL if c in x.columns]  # noqa: E731

    print(f"Early stopping on 2024 (quantile {args.alpha})…", flush=True)
    v = Split(inc[inc.cal_year.between(2021, 2023)], inc[inc.cal_year == 2024])
    vtr, vva = v.x(FINAL_GROUPS)
    m = make_regressor(42, {**params, "n_estimators": 6000})
    m.fit(vtr, v.ytr, eval_set=[(vva, v.yte)], eval_metric="quantile", categorical_feature=cats(vtr),
          callbacks=[lgb.early_stopping(150, verbose=False)])
    n_trees = int(m.best_iteration_ * 1.15)
    print(f"  best {m.best_iteration_} trees → {n_trees} for the 2021–24 fit", flush=True)

    s = Split(inc[inc.cal_year.isin(TRAIN_YEARS)], inc[inc.cal_year == args.test_year])
    xtr, xte = s.x(FINAL_GROUPS)
    print(f"Final fit on {len(xtr):,} trips…", flush=True)
    model = make_regressor(42, {**params, "n_estimators": n_trees})
    model.fit(xtr, s.ytr, categorical_feature=cats(xtr))
    q = np.clip(model.predict(xte), *DRIVE_CLIP_S)
    y = s.yte

    # Comparison: the median model's predictions + one constant offset fitted on 2024.
    med = joblib.load(ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib")
    from firestation.travel_time_model import predict_bag

    med_te = predict_bag(med["models"], xte[med["features"]])
    med_va = predict_bag(med["models"], vva[med["features"]])
    offset = float(np.quantile(v.yte - med_va, args.alpha))
    cov = float(np.mean(y <= q))
    bands = pd.cut(s.te["road_km"], BANDS_KM)
    by_band = (pd.DataFrame({"band": bands.astype(str), "cov": y <= q, "q": q, "y": y})
               .groupby("band", sort=False).agg(n=("cov", "size"), coverage=("cov", "mean"),
                                               mean_q90_s=("q", "mean"), mean_actual_s=("y", "mean")))
    metrics = {
        "alpha": args.alpha, "n_trees": n_trees, "coverage_2025": round(cov, 4),
        "pinball_2025": round(pinball(y, q, args.alpha), 3),
        "pinball_2025_median_plus_offset": round(pinball(y, med_te + offset, args.alpha), 3),
        "median_plus_offset_s": round(offset, 1),
        "coverage_2025_median_plus_offset": round(float(np.mean(y <= med_te + offset)), 4),
        "mean_q90_minus_median_s": round(float(np.mean(q - med_te)), 1),
        "by_road_km_band": by_band.round(3).reset_index().to_dict(orient="records"),
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "by_road_km_band"}, indent=2))
    print(by_band.round(3).to_string())

    payload = {**{k: med[k] for k in ("features", "feature_groups", "categorical", "category_maps",
                                       "borough_categories", "dest_history", "kolesar_baseline_params",
                                       "kolesar_baseline_distance", "drive_clip_s", "train_years")},
               "kind": f"lightgbm_v3_{tag}",
               "model": f"gradient-boosted decision trees (LightGBM), quantile loss alpha={args.alpha}",
               "models": [model], "lightgbm_params": {**params, "n_estimators": n_trees}, "seeds": [42],
               "quantile": args.alpha, "test_year": args.test_year, "metrics_test": metrics,
               "target": f"{args.alpha:.0%} quantile of drive_s (LFB TravelTimeSeconds)"}
    # Destination history / Kolesar / categories must be the ones this model was trained with.
    payload.update({"category_maps": s.cats, "borough_categories": s.cats["borough"],
                    "dest_history": {"sizes_m": list(s.dest.sizes_m), "shrink": s.dest.shrink,
                                     "tables": {str(k): t.to_dict() for k, t in s.dest.tables.items()}},
                    "kolesar_baseline_params": s.kol.to_dict(), "features": list(xtr.columns)})
    joblib.dump(payload, out)
    out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    print(f"Saved {out} ({(time.time() - t0) / 60:.0f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
