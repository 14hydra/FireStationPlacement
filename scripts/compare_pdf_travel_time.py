#!/usr/bin/env python3
"""Compare the advisor-style street PDF travel-time model to LightGBM + Kolesar.

Statistical simulation model (``StreetTimePDFModel``):
  - estimate empirical pace PDFs by road class from 2021–2024 trips
  - for each 2025 trip, sample paces and recombine with length shares
  - point guess = sample median; also score distributional fit (interval coverage, CRPS)

Fair setup matches the production travel-time tests: train 2021–2024, test 2025.

Writes ``data/figures/analysis/pdf_travel_time/``:
  summary.json, mae_table.csv, comparison.png

Run:
  PYTHONPATH=src .venv/bin/python scripts/compare_pdf_travel_time.py
  PYTHONPATH=src .venv/bin/python scripts/compare_pdf_travel_time.py --max-rows 120000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.kolesar import fit_kolesar  # noqa: E402
from firestation.street_time_pdf import StreetTimePDFModel  # noqa: E402
import joblib  # noqa: E402

from firestation.travel_time_model import (  # noqa: E402
    DRIVE_CLIP_S,
    attach_weather,
    load_weather,
    predict as lgbm_predict,
)

OUT = ROOT / "data" / "figures" / "analysis" / "pdf_travel_time"
DATA = ROOT / "data" / "processed" / "lfb_travel_training.csv"
MODEL = ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib"

USECOLS = [
    "incident_number",
    "cal_year",
    "date_of_call",
    "hour_of_call",
    "borough",
    "station_ground",
    "deployed_from_station",
    "deployed_from_location",
    "pump_order",
    "busy_flag",
    "start_lat",
    "start_lon",
    "dest_lat",
    "dest_lon",
    "dest_exact",
    "crow_km",
    "road_km",
    "turnout_s",
    "drive_s",
    "share_major",
    "share_primary",
    "share_secondary",
    "share_tertiary",
    "share_residential",
    "share_minor",
    "share_oneway",
    "share_20mph",
    "share_bus_lane",
    "share_bridge_tunnel",
    "mean_lanes",
    "signals_per_km",
    "calming_per_km",
    "crossings_per_km",
    "give_way_per_km",
    "junctions_per_km",
    "turns_per_km",
    "sharp_turns_per_km",
    "roundabouts_per_km",
]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    ae = np.abs(e)
    return {
        "n": int(len(y)),
        "mae_s": float(ae.mean()),
        "medae_s": float(np.median(ae)),
        "bias_s": float(e.mean()),
        "rmse_s": float(np.sqrt(np.mean(e**2))),
        "corr": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 1e-9 else float("nan"),
    }


def crps_empirical(samples: np.ndarray, y: np.ndarray) -> float:
    """Mean CRPS for empirical predictive samples [n, m] vs observations y [n]."""
    # CRPS ≈ mean_j |x_j - y| - 0.5 mean_{j,k} |x_j - x_k|
    n, m = samples.shape
    term1 = np.mean(np.abs(samples - y[:, None]), axis=1)
    # pairwise within each row (sample estimate)
    # use random pairs for speed
    rng = np.random.default_rng(0)
    j = rng.integers(0, m, size=(n, min(80, m)))
    k = rng.integers(0, m, size=(n, min(80, m)))
    term2 = 0.5 * np.mean(np.abs(np.take_along_axis(samples, j, axis=1) - np.take_along_axis(samples, k, axis=1)), axis=1)
    return float(np.mean(term1 - term2))


def interval_coverage(samples: np.ndarray, y: np.ndarray, q_lo: float, q_hi: float) -> dict:
    lo = np.quantile(samples, q_lo, axis=1)
    hi = np.quantile(samples, q_hi, axis=1)
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    return {
        "level": f"{int(100*(q_hi-q_lo))}%",
        "coverage": float(inside.mean()),
        "mean_width_s": float(width.mean()),
        "median_width_s": float(np.median(width)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-rows", type=int, default=0, help="Optional subsample for a faster dry run")
    ap.add_argument("--n-samples", type=int, default=60)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("loading trips…", flush=True)
    df = pd.read_csv(DATA, usecols=lambda c: c in USECOLS, low_memory=False)
    df = df[df["cal_year"].between(2021, 2025)].reset_index(drop=True)
    if args.max_rows and len(df) > args.max_rows:
        # keep year mix
        df = (
            df.groupby("cal_year", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), max(1, args.max_rows // 5)), random_state=42))
            .reset_index(drop=True)
        )
        print(f"subsampled to {len(df):,} rows", flush=True)

    weather_path = ROOT / "data" / "processed" / "weather_hourly_london.csv"
    if weather_path.exists():
        df = attach_weather(df, load_weather(weather_path))

    tr = df[df["cal_year"].between(2021, 2024)].reset_index(drop=True)
    te = df[df["cal_year"] == 2025].reset_index(drop=True)
    y = te["drive_s"].to_numpy(float)
    print(f"train {len(tr):,} · test {len(te):,}", flush=True)

    # --- baselines ---
    kol = fit_kolesar(tr["road_km"].to_numpy(float), tr["drive_s"].to_numpy(float))
    pred_kol = np.clip(kol.predict(te["road_km"].to_numpy(float)), *DRIVE_CLIP_S)

    print("predicting LightGBM…", flush=True)
    if not MODEL.exists():
        raise SystemExit(f"missing production model: {MODEL}")
    bundle = joblib.load(MODEL)
    # Production file may still carry an in-sample regime_bias that predict() refuses;
    # drop it so we score the LightGBM core the same way as a clean retrain.
    if bundle.get("regime_bias") and not bundle.get("regime_bias_out_of_sample"):
        bundle = {**bundle, "regime_bias": None}
    chunks = []
    for i in range(0, len(te), 50_000):
        chunks.append(np.asarray(lgbm_predict(bundle, te.iloc[i : i + 50_000]), dtype=float))
    pred_lgbm = np.clip(np.concatenate(chunks), *DRIVE_CLIP_S)

    # --- PDF models ---
    print("fitting street-class PDF model…", flush=True)
    pdf = StreetTimePDFModel().fit(tr)
    print("  class pace medians (s/km):", {k: round(float(np.median(v)), 1) for k, v in pdf.class_paces.items()}, flush=True)

    print("sampling street-class PDF…", flush=True)
    samp_street = pdf.sample(te, n_samples=args.n_samples, seed=0, method="street_class")
    pred_street = np.median(samp_street, axis=1)

    print("sampling regime PDF…", flush=True)
    samp_regime = pdf.sample(te, n_samples=args.n_samples, seed=1, method="regime")
    pred_regime = np.median(samp_regime, axis=1)

    results = {
        "constant_median_train": metrics(y, np.full_like(y, float(np.median(tr["drive_s"])))),
        "kolesar_road_km": metrics(y, pred_kol),
        "lightgbm_production": metrics(y, pred_lgbm),
        "pdf_street_class_median": metrics(y, pred_street),
        "pdf_regime_median": metrics(y, pred_regime),
    }

    dist = {
        "pdf_street_class": {
            "crps_s": crps_empirical(samp_street, y),
            "interval_50": interval_coverage(samp_street, y, 0.25, 0.75),
            "interval_80": interval_coverage(samp_street, y, 0.10, 0.90),
        },
        "pdf_regime": {
            "crps_s": crps_empirical(samp_regime, y),
            "interval_50": interval_coverage(samp_regime, y, 0.25, 0.75),
            "interval_80": interval_coverage(samp_regime, y, 0.10, 0.90),
        },
    }
    # Point models as degenerate distributions for CRPS-ish comparison: |p-y|
    dist["lightgbm_as_point"] = {"mae_as_crps_proxy_s": results["lightgbm_production"]["mae_s"]}
    dist["kolesar_as_point"] = {"mae_as_crps_proxy_s": results["kolesar_road_km"]["mae_s"]}

    # First-engine / not-busy slice if available
    slices = {}
    if "pump_order" in te.columns and "busy_flag" in te.columns:
        m = (pd.to_numeric(te["pump_order"], errors="coerce") == 1) & (
            pd.to_numeric(te["busy_flag"], errors="coerce").fillna(0) == 0
        )
        if m.sum() > 1000:
            slices["first_engine_not_busy"] = {
                "n": int(m.sum()),
                "lightgbm": metrics(y[m], pred_lgbm[m]),
                "kolesar": metrics(y[m], pred_kol[m]),
                "pdf_street_class": metrics(y[m], pred_street[m]),
                "pdf_regime": metrics(y[m], pred_regime[m]),
            }

    winner_point = min(results, key=lambda k: results[k]["mae_s"])
    summary = {
        "doc": (
            "Advisor-style statistical travel-time model vs production LightGBM and Kolesar. "
            "PDF model samples fire-engine pace (s/km) from empirical distributions by OSM "
            "road class (attributed from whole-trip times), then rebuilds trip time from "
            "length shares. Train 2021–2024, test 2025. Point score = sample median MAE; "
            "also CRPS and interval coverage for the PDF models."
        ),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_samples": args.n_samples,
        "pdf_model": pdf.to_summary(),
        "point_metrics": results,
        "distributional": dist,
        "slices": slices,
        "winner_point_mae": winner_point,
        "verdict": (
            f"Best point MAE: {winner_point} "
            f"({results[winner_point]['mae_s']:.1f} s). "
            "PDF models are built for distribution / simulation use; "
            "LightGBM remains the point predictor for placement if it wins MAE."
        ),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([{"model": k, **v} for k, v in results.items()]).to_csv(OUT / "mae_table.csv", index=False)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    order = [
        ("constant_median_train", "Constant"),
        ("kolesar_road_km", "Kolesar"),
        ("pdf_regime_median", "PDF regime"),
        ("pdf_street_class_median", "PDF street-class"),
        ("lightgbm_production", "LightGBM"),
    ]
    names = [a[1] for a in order]
    maes = [results[a[0]]["mae_s"] for a in order]
    colors = ["#94a3b8", "#d97706", "#0f766e", "#1d5bbf", "#15803d"]
    bars = ax.barh(names[::-1], maes[::-1], color=colors[::-1])
    ax.set_xlabel("MAE on 2025 drive time (s)")
    ax.set_title("Point accuracy (lower is better)")
    for b, v in zip(bars, maes[::-1]):
        ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=9)
    ax.set_xlim(0, max(maes) * 1.18)

    ax = axes[1]
    # coverage vs nominal
    levels = ["50%", "80%"]
    cov_s = [dist["pdf_street_class"]["interval_50"]["coverage"], dist["pdf_street_class"]["interval_80"]["coverage"]]
    cov_r = [dist["pdf_regime"]["interval_50"]["coverage"], dist["pdf_regime"]["interval_80"]["coverage"]]
    nominal = [0.50, 0.80]
    x = np.arange(len(levels))
    w = 0.35
    ax.bar(x - w / 2, cov_s, w, label="PDF street-class", color="#1d5bbf")
    ax.bar(x + w / 2, cov_r, w, label="PDF regime", color="#0f766e")
    ax.plot(x, nominal, "k--", label="ideal coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of 2025 trips inside interval")
    ax.set_title("PDF calibration (distributional check)")
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Travel-time models: statistical PDF simulation vs LightGBM / Kolesar",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "PDF paces attributed from whole-trip times (no per-street GPS). Train 2021–24 · test 2025.",
        ha="center",
        fontsize=8,
        color="#5b6570",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(OUT / "comparison.png", dpi=140)
    plt.close(fig)

    print(json.dumps({k: round(v["mae_s"], 2) for k, v in results.items()}, indent=2))
    print("winner:", winner_point)
    print("CRPS street/regime:", dist["pdf_street_class"]["crps_s"], dist["pdf_regime"]["crps_s"])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
