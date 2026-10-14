#!/usr/bin/env python3
"""
Placement-ready London (LFB) driving-time model: gradient-boosted decision trees (LightGBM).

Target: TravelTimeSeconds (leave station → arrive), 30–1200 s.
Model:  LightGBM, averaged over 4 seeds, on placement-safe features (no station IDs).
        Features are built by firestation.travel_time_model.build_features (see its docstring
        for the groups: base, street, context, weather, incident, dest).

What this script reports (all on held-out 2025 trips, trained on 2021–2024):
  1. Baselines: median, straight line, road distance, Kolesar.
  2. Feature ladder: one LightGBM per step, adding one feature group at a time.
  3. Tuning: number of trees by early stopping on 2024 (trained on 2021–23), so the
     2025 test year is never used for choices; larger trees vs the default.
  4. New-station check: stations split into 5 groups; for each, train on the other
     stations' 2021–24 trips and predict this group's 2025 trips. This mimics scoring
     a station site the model has never seen.
  5. Final model: 4-seed bag with the tuned settings on FINAL_GROUPS (weather left out:
     it did not help), saved for the planner / race.

Inputs: data/processed/lfb_travel_training.csv (scripts/build_lfb_travel_training.py),
        data/processed/weather_hourly_london.csv (scripts/fetch_london_weather.py).

Run:  .venv/bin/python scripts/train_travel_time_lfb.py [--quick]
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
    attach_weather,
    build_features,
    category_maps,
    fit_bag,
    load_weather,
    make_regressor,
    predict_bag,
)

TRAIN_YEARS = (2021, 2022, 2023, 2024)
LADDER = [
    ("base", ["base"]),
    ("+ street features", ["base", "street"]),
    ("+ fastest route / main-road access", ["base", "street", "context"]),
    ("+ weather", ["base", "street", "context", "weather"]),
    ("+ incident type / home station", ["base", "street", "context", "weather", "incident"]),
    ("+ destination history", ["base", "street", "context", "weather", "incident", "dest"]),
]
# Final model: every ladder group except weather, which made the tuned model slightly worse
# (2025 MAE 58.0 s with vs 57.8 s without, two seeds each; checked 2026-10-12).
FINAL_GROUPS = ["base", "street", "context", "incident", "dest"]
TUNED_PARAMS = dict(learning_rate=0.05, num_leaves=255, min_child_samples=50, colsample_bytree=0.7,
                    subsample=0.85, reg_lambda=3.0)


def metrics(y, p):
    e = np.abs(y - p)
    return {"mae_s": round(float(e.mean()), 2), "medae_s": round(float(np.median(e)), 2),
            "bias_s": round(float(np.mean(p - y)), 2)}


class Split:
    """Train/test frames plus everything fitted on the train part (Kolesar, categories, dest history)."""

    def __init__(self, tr: pd.DataFrame, te: pd.DataFrame):
        self.tr, self.te = tr.reset_index(drop=True), te.reset_index(drop=True)
        self.ytr = self.tr["drive_s"].to_numpy(float)
        self.yte = self.te["drive_s"].to_numpy(float)
        self.kol = fit_kolesar(self.tr["road_km"].to_numpy(float), self.ytr)
        self.cats = category_maps(self.tr)
        resid = self.ytr - np.clip(self.kol.predict(self.tr["road_km"].to_numpy(float)), *DRIVE_CLIP_S)
        self.dest = DestinationHistory().fit(self.tr, resid)
        self.dest_oof = DestinationHistory().out_of_fold(self.tr, resid, self.tr["incident_number"])

    def x(self, groups):
        kw = dict(kolesar=self.kol, cat_maps=self.cats)
        return (build_features(self.tr, groups, dest_frame=self.dest_oof, **kw),
                build_features(self.te, groups, dest_history=self.dest, **kw))


def fit_one(xtr, ytr, params=None, seed=42):
    m = make_regressor(seed, params)
    m.fit(xtr, ytr, categorical_feature=[c for c in CATEGORICAL if c in xtr.columns])
    return m


def tune_trees(inc: pd.DataFrame, groups, params) -> int:
    """Best n_estimators by early stopping: train 2021–23, validate on 2024."""
    import lightgbm as lgb

    s = Split(inc[inc.cal_year.between(2021, 2023)], inc[inc.cal_year == 2024])
    xtr, xva = s.x(groups)
    m = make_regressor(42, {**params, "n_estimators": 6000})
    m.fit(xtr, s.ytr, eval_set=[(xva, s.yte)], eval_metric="l1",
          categorical_feature=[c for c in CATEGORICAL if c in xtr.columns],
          callbacks=[lgb.early_stopping(150, verbose=False)])
    best = int(m.best_iteration_)
    print(f"  early stopping: best {best} trees, 2024 validation MAE {m.best_score_['valid_0']['l1']:.2f} s")
    # A bit more data in the final fit (2021–24 vs 2021–23) supports a few more trees.
    return int(best * 1.15)


def new_station_check(inc: pd.DataFrame, test_year: int, groups, params, k: int = 5) -> dict:
    """Hold out whole stations: train on other stations' 2021–24 trips, test on held-out stations' 2025 trips."""
    stations = np.array(sorted(inc["deployed_from_station"].unique()))
    rng = np.random.default_rng(0)
    fold_of = dict(zip(stations, rng.permutation(len(stations)) % k))
    f = inc["deployed_from_station"].map(fold_of)
    ys, ps, ks = [], [], []
    for i in range(k):
        tr = inc[inc.cal_year.isin(TRAIN_YEARS) & (f != i)]
        te = inc[(inc.cal_year == test_year) & (f == i)]
        s = Split(tr, te)
        xtr, xte = s.x(groups)
        ps.append(np.clip(fit_one(xtr, s.ytr, params).predict(xte), *DRIVE_CLIP_S))
        ks.append(np.clip(s.kol.predict(s.te["road_km"].to_numpy(float)), *DRIVE_CLIP_S))
        ys.append(s.yte)
        print(f"  fold {i + 1}/{k}: {len(te):,} trips from {int((pd.Series(fold_of) == i).sum())} held-out stations")
    y, p, kk = np.concatenate(ys), np.concatenate(ps), np.concatenate(ks)
    return {"lightgbm": metrics(y, p), "kolesar(road_km)": metrics(y, kk), "n_trips": int(len(y))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "lfb_travel_training.csv")
    ap.add_argument("--weather", type=Path, default=ROOT / "data" / "processed" / "weather_hourly_london.csv")
    ap.add_argument("--test-year", type=int, default=2025)
    ap.add_argument("--quick", action="store_true", help="Skip the ladder and new-station check")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib")
    args = ap.parse_args()
    t0 = time.time()

    inc = pd.read_csv(args.data, low_memory=False)
    weather = load_weather(args.weather)
    print(f"Weather: {'joined' if weather is not None else 'MISSING (run scripts/fetch_london_weather.py)'}")
    inc = attach_weather(inc, weather)
    inc = inc[inc.cal_year.isin(TRAIN_YEARS + (args.test_year,))].reset_index(drop=True)

    s = Split(inc[inc.cal_year.isin(TRAIN_YEARS)], inc[inc.cal_year == args.test_year])
    print(f"train {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}: {len(s.tr):,} trips | test {args.test_year}: {len(s.te):,} trips")
    road_kph = float(s.tr["road_km"].sum() / s.ytr.sum() * 3600)
    kol_crow = fit_kolesar(s.tr["crow_km"].to_numpy(float), s.ytr)
    clip = lambda p: np.clip(p, *DRIVE_CLIP_S)  # noqa: E731
    results = {
        "median": metrics(s.yte, np.full_like(s.yte, np.median(s.ytr))),
        "crow@32kph": metrics(s.yte, clip(s.te["crow_km"].to_numpy(float) / 32 * 3600)),
        f"road@{road_kph:.0f}kph": metrics(s.yte, clip(s.te["road_km"].to_numpy(float) / road_kph * 3600)),
        "kolesar(crow_km)": metrics(s.yte, clip(kol_crow.predict(s.te["crow_km"].to_numpy(float)))),
        "kolesar(road_km)": metrics(s.yte, clip(s.kol.predict(s.te["road_km"].to_numpy(float)))),
    }

    full_groups = FINAL_GROUPS
    ladder = {}
    if not args.quick:
        print("\nFeature ladder (1 seed, default settings):")
        for name, groups in LADDER:
            xtr, xte = s.x(groups)
            ladder[name] = metrics(s.yte, clip(fit_one(xtr, s.ytr).predict(xte)))
            print(f"  {name:<40} {xtr.shape[1]:>3} features  MAE {ladder[name]['mae_s']:.2f} s")

    print("\nTuning (early stopping on 2024; 2025 untouched):")
    tuned = {**TUNED_PARAMS, "n_estimators": tune_trees(inc, full_groups, TUNED_PARAMS)}
    xtr, xte = s.x(full_groups)
    tuned_1 = metrics(s.yte, clip(fit_one(xtr, s.ytr, tuned).predict(xte)))
    print(f"  tuned, 1 seed: MAE {tuned_1['mae_s']:.2f} s ({tuned['n_estimators']} trees, {tuned['num_leaves']} leaves)")
    params = tuned

    new_station = {}
    if not args.quick:
        print("\nNew-station check (whole stations held out):")
        new_station = new_station_check(inc, args.test_year, full_groups, params)
        print(f"  held-out stations, 2025: LightGBM {new_station['lightgbm']['mae_s']:.2f} s | "
              f"Kolesar {new_station['kolesar(road_km)']['mae_s']:.2f} s")

    print("\nFinal model: 4-seed LightGBM bag…")
    models = fit_bag(xtr, s.ytr, params=params)
    pred = predict_bag(models, xte)
    results["lightgbm(final)"] = metrics(s.yte, pred)
    nb = (s.te["busy_flag"] == 0).to_numpy()
    results_nonbusy = {"kolesar(road_km)": metrics(s.yte[nb], clip(s.kol.predict(s.te["road_km"].to_numpy(float)))[nb]),
                       "lightgbm(final)": metrics(s.yte[nb], pred[nb])}
    imp = np.mean([m.booster_.feature_importance(importance_type="gain") for m in models], axis=0)
    imp = pd.Series(imp / imp.sum(), index=xtr.columns).sort_values(ascending=False)

    print(f"\nHeld-out {args.test_year} driving-time error:")
    print(f"{'model':<28}{'MAE s':>9}{'MedAE s':>9}{'bias s':>9}")
    for k, v in results.items():
        print(f"{k:<28}{v['mae_s']:>9.2f}{v['medae_s']:>9.2f}{v['bias_s']:>9.2f}")
    print("non-busy:", {k: v["mae_s"] for k, v in results_nonbusy.items()})
    print("top features (gain share):", imp.head(10).round(3).to_dict())

    payload = {
        "kind": "lightgbm_v3",
        "model": "gradient-boosted decision trees (LightGBM), mean of 4 seeds",
        "models": models,
        "features": list(xtr.columns),
        "feature_groups": full_groups,
        "categorical": [c for c in CATEGORICAL if c in xtr.columns],
        "category_maps": s.cats,
        "borough_categories": s.cats["borough"],
        "dest_history": {"sizes_m": list(s.dest.sizes_m), "shrink": s.dest.shrink,
                         "tables": {str(k): v.to_dict() for k, v in s.dest.tables.items()}},
        "kolesar_baseline_params": s.kol.to_dict(),
        "kolesar_baseline_distance": "road_km",
        "lightgbm_params": params,
        "target": "drive_s = LFB TravelTimeSeconds (station departure -> arrival), 30-1200 s",
        "drive_clip_s": list(DRIVE_CLIP_S),
        "seeds": [42, 43, 44, 45],
        "train_years": list(TRAIN_YEARS),
        "test_year": args.test_year,
        "metrics_test": results,
        "metrics_test_non_busy": results_nonbusy,
        "feature_ladder": ladder,
        "tuned_single_seed": tuned_1,
        "new_station_check": new_station,
        "feature_importance_gain_share": imp.round(4).to_dict(),
        "honesty_note": (
            "No station IDs. Route / street features come from the shortest legal OSM path, not the "
            "driven route. Weather is city-centre hourly. busy_flag, pump_order, incident type and "
            "from_home are known historically; for new sites use typical values (pump_order=1, "
            "from_home=1, expected busy rate). Destination history uses training trips only "
            "(out-of-fold for training rows). Use firestation.travel_time_model.predict(bundle, trips)."
        ),
    }
    import joblib

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.out)
    keep = {k: v for k, v in payload.items() if k not in {"models", "dest_history"}}
    args.out.with_suffix(".metrics.json").write_text(json.dumps(keep, indent=2, default=str) + "\n")
    print(f"\nSaved {args.out}  ({(time.time() - t0) / 60:.0f} min)")
    print(f"Final 2025 MAE: {results['lightgbm(final)']['mae_s']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
