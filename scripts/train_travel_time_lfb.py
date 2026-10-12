#!/usr/bin/env python3
"""
Placement-ready London (LFB) drive-time trainer (v2).

Target: TravelTimeSeconds (leave station → arrive), 30–1200 s.
Model:  LightGBM bag-of-4 on placement-safe features (no station IDs).

Features
--------
Context: hour, dow, month, rush, night, weekend, busy_flag, borough, pump_order,
         dest_exact, circuity (road/crow), kolesar_prior_s
Street (when present in CSV): ROUTE_FEATURES from emvro.route_features
Weather (Open-Meteo London hourly): temp_c, precip_mm, wind_kph, rh_pct,
         is_rain, is_heavy_rain, is_cold, is_hot

Chronological split: train 2021–2024, test 2025.

Run:  .venv/bin/python scripts/train_travel_time_lfb.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.kolesar import fit_kolesar  # noqa: E402
from emvro.route_features import ROUTE_FEATURES  # noqa: E402

DRIVE_CLIP_S = (30.0, 1200.0)
TRAIN_YEARS = (2021, 2022, 2023, 2024)
SEEDS = (42, 43, 44, 45)

BASE_FEATURES = [
    "road_km",
    "crow_km",
    "circuity",
    "kolesar_prior_s",
    "hour",
    "dow",
    "month",
    "rush",
    "night",
    "weekend",
    "busy_flag",
    "pump_order",
    "dest_exact",
    "borough_code",
]
WEATHER_FEATURES = [
    "temp_c",
    "precip_mm",
    "wind_kph",
    "rh_pct",
    "is_rain",
    "is_heavy_rain",
    "is_cold",
    "is_hot",
]
CATS = ["borough_code"]


def _load_weather(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    w = pd.read_csv(path, parse_dates=["time"])
    w["weather_hour"] = w["time"].dt.floor("h")
    return w.drop_duplicates("weather_hour").set_index("weather_hour")


def attach_weather(df: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    if weather is None:
        for c in WEATHER_FEATURES:
            out[c] = np.nan
        return out
    # Join on London local hour of mobilisation / call
    ts = pd.to_datetime(out["date_of_call"], errors="coerce") + pd.to_timedelta(
        out["hour_of_call"].astype(int), unit="h"
    )
    key = ts.dt.floor("h")
    joined = weather.reindex(key)
    for c in WEATHER_FEATURES:
        out[c] = joined[c].to_numpy() if c in weather.columns else np.nan
    return out


def available_street_features(df: pd.DataFrame) -> list[str]:
    return [c for c in ROUTE_FEATURES if c in df.columns and df[c].notna().mean() > 0.5]


def add_features(
    df: pd.DataFrame,
    borough_cats: list[str],
    *,
    street: list[str],
    kolesar_model,
    use_weather: bool,
) -> pd.DataFrame:
    d = pd.to_datetime(df["date_of_call"], errors="coerce")
    road = pd.to_numeric(df["road_km"], errors="coerce").to_numpy(dtype=float)
    crow = pd.to_numeric(df["crow_km"], errors="coerce").to_numpy(dtype=float)
    out = pd.DataFrame(index=df.index)
    out["road_km"] = road
    out["crow_km"] = crow
    with np.errstate(divide="ignore", invalid="ignore"):
        circ = np.where((crow > 0.05) & np.isfinite(crow) & np.isfinite(road), road / crow, 1.25)
    out["circuity"] = np.clip(circ, 0.8, 4.0)
    out["kolesar_prior_s"] = np.clip(kolesar_model.predict(road), *DRIVE_CLIP_S)
    out["hour"] = df["hour_of_call"].astype(int).to_numpy()
    out["dow"] = d.dt.dayofweek.to_numpy()
    out["month"] = d.dt.month.to_numpy()
    h = out["hour"]
    out["rush"] = (h.between(7, 9) | h.between(16, 19)).astype(int)
    out["night"] = ((h >= 22) | (h <= 5)).astype(int)
    out["weekend"] = (out["dow"] >= 5).astype(int)
    out["busy_flag"] = df["busy_flag"].astype(int).to_numpy()
    if "pump_order" in df.columns:
        out["pump_order"] = pd.to_numeric(df["pump_order"], errors="coerce").fillna(1).clip(1, 5)
    else:
        out["pump_order"] = 1
    if "dest_exact" in df.columns:
        out["dest_exact"] = pd.to_numeric(df["dest_exact"], errors="coerce").fillna(0).astype(int)
    else:
        out["dest_exact"] = 0
    codes = {b: i for i, b in enumerate(borough_cats)}
    out["borough_code"] = df["borough"].astype(str).str.upper().map(codes).astype(float)
    for c in street:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    if use_weather:
        for c in WEATHER_FEATURES:
            out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    feats = BASE_FEATURES + street + (WEATHER_FEATURES if use_weather else [])
    return out[feats]


def make_regressor(seed: int, *, n_features: int):
    import lightgbm as lgb

    # Scale capacity mildly with feature count; L1 objective matches MAE.
    leaves = 63 if n_features >= 20 else 47
    return lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=leaves,
        min_child_samples=35,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        reg_alpha=0.1,
        objective="l1",
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )


def fit_bag(x: pd.DataFrame, y: np.ndarray, cats: list[str]):
    models = []
    for s in SEEDS:
        m = make_regressor(s, n_features=x.shape[1])
        use_cats = [c for c in cats if c in x.columns]
        m.fit(x, y, categorical_feature=use_cats)
        models.append(m)
    return models


def predict_bag(models, x: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict(x) for m in models], axis=0)


def metrics(y, p):
    e = np.abs(y - p)
    return {"mae_s": float(e.mean()), "medae_s": float(np.median(e)), "bias_s": float(np.mean(p - y))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "lfb_travel_training.csv")
    ap.add_argument("--weather", type=Path, default=ROOT / "data" / "processed" / "weather_hourly_london.csv")
    ap.add_argument("--test-year", type=int, default=2025)
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--no-street", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib")
    args = ap.parse_args()

    inc = pd.read_csv(args.data)
    weather = None if args.no_weather else _load_weather(args.weather)
    if weather is not None:
        print(f"Weather rows: {len(weather):,} ({args.weather.name})")
        inc = attach_weather(inc, weather)
    else:
        print("Weather: not used")

    street = [] if args.no_street else available_street_features(inc)
    print(f"Street features available: {len(street)}" + (f" ({', '.join(street[:4])}…)" if street else ""))

    train_years = [y for y in TRAIN_YEARS if (inc.cal_year == y).any()]
    tr = inc[inc.cal_year.isin(train_years)].reset_index(drop=True)
    te = inc[inc.cal_year == args.test_year].reset_index(drop=True)
    print(f"train years {train_years}: {len(tr):,} rows | test {args.test_year}: {len(te):,} rows")

    borough_cats = sorted(tr["borough"].astype(str).str.upper().unique())
    kol = fit_kolesar(tr["road_km"].to_numpy(dtype=float), tr["drive_s"].to_numpy(dtype=float))

    use_weather = weather is not None and not args.no_weather
    xtr = add_features(tr, borough_cats, street=street, kolesar_model=kol, use_weather=use_weather)
    xte = add_features(te, borough_cats, street=street, kolesar_model=kol, use_weather=use_weather)
    ytr, yte = tr["drive_s"].to_numpy(dtype=float), te["drive_s"].to_numpy(dtype=float)

    kol_te = np.clip(kol.predict(xte["road_km"]), *DRIVE_CLIP_S)
    kol_crow = fit_kolesar(xtr["crow_km"].to_numpy(), ytr)
    kol_crow_te = np.clip(kol_crow.predict(xte["crow_km"]), *DRIVE_CLIP_S)
    road_kph = float(xtr["road_km"].sum() / max(ytr.sum(), 1) * 3600)

    # Ablation ladder for honesty
    results = {
        "median": metrics(yte, np.full_like(yte, np.median(ytr))),
        "crow@32kph": metrics(yte, np.clip(xte["crow_km"].to_numpy() / 32.0 * 3600, *DRIVE_CLIP_S)),
        f"road@{road_kph:.0f}kph": metrics(yte, np.clip(xte["road_km"].to_numpy() / road_kph * 3600, *DRIVE_CLIP_S)),
        "kolesar(crow_km)": metrics(yte, kol_crow_te),
        "kolesar(road_km)": metrics(yte, kol_te),
    }

    # Distance+time only (no street/weather) — matches prior best without streets
    base_cols = [c for c in BASE_FEATURES]
    models_base = fit_bag(xtr[base_cols], ytr, CATS)
    pred_base = np.clip(predict_bag(models_base, xte[base_cols]), *DRIVE_CLIP_S)
    results["lightgbm(base)"] = metrics(yte, pred_base)

    models = fit_bag(xtr, ytr, CATS)
    pred = np.clip(predict_bag(models, xte), *DRIVE_CLIP_S)
    tag = "lightgbm(full)"
    results[tag] = metrics(yte, pred)

    nb = (xte["busy_flag"] == 0).to_numpy()
    results_nonbusy = {
        "kolesar": metrics(yte[nb], kol_te[nb]),
        "lightgbm(base)": metrics(yte[nb], pred_base[nb]),
        "lightgbm(full)": metrics(yte[nb], pred[nb]),
        "median": metrics(yte[nb], np.full(nb.sum(), np.median(ytr))),
    }

    print(f"\nHeld-out {args.test_year} driving-time error (TravelTimeSeconds):")
    print(f"{'model':<28}{'MAE s':>9}{'MedAE s':>9}{'bias s':>9}")
    for k, v in results.items():
        print(f"{k:<28}{v['mae_s']:>9.1f}{v['medae_s']:>9.1f}{v['bias_s']:>9.1f}")
    print("non-busy:", {k: round(v["mae_s"], 1) for k, v in results_nonbusy.items()})
    print("Kolesar road_km:", {k: round(v, 3) for k, v in kol.to_dict().items()})
    print(f"Features ({len(xtr.columns)}): {list(xtr.columns)}")

    # Feature importance (mean gain across bag)
    try:
        imp = np.mean([m.booster_.feature_importance(importance_type="gain") for m in models], axis=0)
        order = np.argsort(-imp)[:15]
        top = {xtr.columns[i]: float(imp[i]) for i in order}
    except Exception:  # noqa: BLE001
        top = {}

    payload = {
        "kind": "lightgbm_v2",
        "models": models,
        "features": list(xtr.columns),
        "categorical": CATS,
        "borough_categories": borough_cats,
        "street_features": street,
        "weather_features": WEATHER_FEATURES if use_weather else [],
        "kolesar_baseline_params": kol.to_dict(),
        "kolesar_baseline_distance": "road_km",
        "target": "drive_s = LFB TravelTimeSeconds (station departure -> arrival), 30-1200 s",
        "drive_clip_s": list(DRIVE_CLIP_S),
        "seeds": list(SEEDS),
        "train_years": train_years,
        "test_year": args.test_year,
        "metrics_test": results,
        "metrics_test_non_busy": results_nonbusy,
        "feature_importance_gain_top15": top,
        "excluded_features": ["station name/ID (not placement-safe)"],
        "honesty_note": (
            "Placement-safe: no station IDs. road_km / street features from shortest legal OSM path "
            "(not the driven route). Weather is city-centre Open-Meteo hourly (not trip-local). "
            "busy_flag and pump_order are known historically; for new sites assume typical values "
            "(busy_rate prior / pump_order=1). Kolesar prior is an input feature, not the residual target."
        ),
    }
    import joblib

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.out)
    (args.out.with_suffix(".metrics.json")).write_text(
        json.dumps({k: v for k, v in payload.items() if k not in {"models"}}, indent=2, default=str)
    )
    print(f"\nSaved {args.out}")
    print(f"Best test MAE: {results[tag]['mae_s']:.1f}s (base {results['lightgbm(base)']['mae_s']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
