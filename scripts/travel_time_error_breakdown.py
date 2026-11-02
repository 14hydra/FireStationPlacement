#!/usr/bin/env python3
"""Where the production travel time model's error on 2025 trips comes from.

  slices  error overall and on the trips placement scores (first pump, home station, not busy)
  trimmed the same with each slice's worst 5 % of trips (largest absolute error) removed
  routes  one station to one ~100 m spot: average predicted vs average recorded time per route
  history the same trips predicted by the route's own recorded median from 2021-2024
  compare the classic formula and the never-seen-stations test on all trips and placement's trips

Writes data/figures/validation/travel_time_error/summary.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/travel_time_error_breakdown.py
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

from firestation.kolesar import KolesarModel  # noqa: E402
from firestation.travel_time_model import DRIVE_CLIP_S, attach_weather, load_weather, predict  # noqa: E402

OUT = ROOT / "data" / "figures" / "validation" / "travel_time_error"
DATA = ROOT / "data" / "processed" / "lfb_travel_training.csv"
ROUTE = ["deployed_from_station", "cx", "cy"]


def with_route(df: pd.DataFrame) -> pd.DataFrame:
    h = pd.to_numeric(df["hour_of_call"], errors="coerce").fillna(12)
    band = np.select([h.between(7, 9), h.between(10, 15), h.between(16, 19), h.between(20, 23)], [1, 2, 3, 4], 0)
    return df.assign(cx=(df["dest_lat"] / 0.0009).round(), cy=(df["dest_lon"] / 0.0014).round(), band=band)


def scenario(df: pd.DataFrame) -> pd.Series:
    return (df["pump_order"] == 1) & (df["deployed_from_location"] == "Home Station") & (df["busy_flag"] == 0)


def errors(y: np.ndarray, p: np.ndarray) -> dict:
    ae = np.abs(p - y)
    keep = ae <= np.quantile(ae, 0.95)
    return {"n": int(len(y)), "mae_s": round(float(ae.mean()), 1), "medae_s": round(float(np.median(ae)), 1),
            "bias_s": round(float((p - y).mean()), 1), "mae_without_worst_5pct_s": round(float(ae[keep].mean()), 1),
            "worst_5pct_share_of_error_pct": round(float(ae[~keep].sum() / ae.sum() * 100), 1)}


def route_errors(q: pd.DataFrame, min_trips: int) -> dict:
    g = q.groupby(ROUTE)
    k = g["y"].transform("size")
    r = q[k >= min_trips].groupby(ROUTE).agg(n=("y", "size"), y_mean=("y", "mean"), y_med=("y", "median"),
                                             y_sd=("y", "std"), p_mean=("p", "mean"), p_med=("p", "median"))
    q = q[k >= min_trips]
    # A route's recorded average is itself uncertain: E|mean error| from sampling alone is about 0.8 x its SE.
    sampling = 0.8 * r["y_sd"] / np.sqrt(r["n"])
    return {"min_trips": min_trips, "routes": int(len(r)), "trips": int(r["n"].sum()),
            "trip_mae_s": round(float(np.abs(q["p"] - q["y"]).mean()), 1),
            "route_mean_vs_mean_mae_s": round(float((r["p_mean"] - r["y_mean"]).abs().mean()), 1),
            "route_mean_bias_s": round(float((r["p_mean"] - r["y_mean"]).mean()), 1),
            "route_median_vs_median_mae_s": round(float((r["p_med"] - r["y_med"]).abs().mean()), 1),
            "route_median_bias_s": round(float((r["p_med"] - r["y_med"]).mean()), 1),
            "sampling_noise_in_recorded_mean_s": round(float(sampling.mean()), 1),
            "spread_around_route_median_s": round(float((q["y"] - q.groupby(ROUTE)["y"].transform("median")).abs().mean()), 1)}


def history_baseline(q: pd.DataFrame, past: pd.DataFrame, keys: list[str], min_trips: int = 5) -> dict:
    h = past.groupby(keys)["drive_s"].agg(["size", "median"])
    m = q.join(h[h["size"] >= min_trips]["median"].rename("hist"), on=keys, how="inner")
    return {"trips": int(len(m)), "history_median_mae_s": round(float((m["hist"] - m["y"]).abs().mean()), 1),
            "model_mae_s": round(float((m["p"] - m["y"]).abs().mean()), 1)}


def compare_methods(te: pd.DataFrame, y: np.ndarray, p: np.ndarray, bundle: dict, sc: np.ndarray) -> dict:
    """Model vs classic formula, and the saved never-seen-stations folds, on the same trips."""
    kol = KolesarModel.from_dict(bundle["kolesar_baseline_params"])
    k = np.clip(kol.predict(te["road_km"].to_numpy(float)), *DRIVE_CLIP_S)
    mae = lambda a, b: round(float(np.abs(a - b).mean()), 1)  # noqa: E731
    out = {"all": {"model_mae_s": mae(p, y), "kolesar_mae_s": mae(k, y)},
           "placement": {"model_mae_s": mae(p[sc], y[sc]), "kolesar_mae_s": mae(k[sc], y[sc])}}
    folds = sorted((ROOT / "data/figures/validation/new_stations/folds").glob("fold_20_*.parquet"))
    key = ["incident_number", "deployed_from_station", "y"]
    un = pd.concat([pd.read_parquet(f, columns=key + ["unseen"]) for f in folds]).drop_duplicates(key)
    m = te[key[:2]].assign(y=y, sc=sc).merge(un, on=key, how="left")
    assert m["unseen"].notna().mean() > 0.99, "never-seen folds do not cover the 2025 trips"
    m = m.dropna(subset=["unseen"])
    out["all"]["never_seen_mae_s"] = mae(m["unseen"], m["y"])
    out["placement"]["never_seen_mae_s"] = mae(m["unseen"][m["sc"]], m["y"][m["sc"]])
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, low_memory=False)
    past = with_route(df[df["cal_year"].between(2021, 2024) & (df["dest_exact"] == 1) & scenario(df)])[
        ROUTE + ["band", "drive_s"]]
    te = df[df["cal_year"] == 2025].reset_index(drop=True)
    del df
    te = attach_weather(te, load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    bundle = joblib.load(ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib")
    p = np.concatenate([predict(bundle, te.iloc[i:i + 50000]) for i in range(0, len(te), 50000)])
    y = te["drive_s"].to_numpy(float)

    sc = scenario(te).to_numpy()
    exact = te["dest_exact"].to_numpy() == 1
    slices = {"all 2025 trips": np.ones(len(te), bool), "first pump, home station, not busy (placement)": sc,
              "placement, exact address": sc & exact, "every other trip": ~sc}
    out = {"doc": __doc__, "slices": {k: errors(y[m], p[m]) for k, m in slices.items()}}

    q = with_route(te[sc & exact]).assign(y=y[sc & exact], p=p[sc & exact])
    out["routes"] = [route_errors(q, k) for k in (5, 10, 20)]
    out["history"] = {"route": history_baseline(q, past, ROUTE),
                      "route + time of day": history_baseline(q, past, ROUTE + ["band"])}
    out["compare"] = compare_methods(te, y, p, bundle, sc)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "doc"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
