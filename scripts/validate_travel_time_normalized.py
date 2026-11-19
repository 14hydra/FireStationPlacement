#!/usr/bin/env python3
"""Advisor-style travel-time validation: normalized error + full residual histogram.

Absolute MAE (~60 s) hides that a 60 s miss on a 5-minute trip is a large relative
error. This script reports:

  - absolute MAE / median abs error (as now)
  - normalized error: MAPE, median |%|, sMAPE, and error as % of actual
  - outlier count (trips with |error| > 400 s)
  - full residual histogram (and APE histogram) for board / advisor review

Same production model and 2025 holdout as other travel-time checks.

Writes ``data/figures/validation/travel_time_normalized/``:
  summary.json, metrics.csv, error_distribution.png

Run:
  PYTHONPATH=src .venv/bin/python scripts/validate_travel_time_normalized.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.travel_time_model import DRIVE_CLIP_S, attach_weather, load_weather, predict  # noqa: E402

OUT = ROOT / "data" / "figures" / "validation" / "travel_time_normalized"
DATA = ROOT / "data" / "processed" / "lfb_travel_training.csv"
MODEL = ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib"
OUTLIER_S = 400.0
MIN_Y_FOR_PCT = 30.0  # avoid exploding % on tiny drive times


def placement_mask(df: pd.DataFrame) -> np.ndarray:
    return (
        (df["pump_order"] == 1)
        & (df["deployed_from_location"].astype(str) == "Home Station")
        & (pd.to_numeric(df["busy_flag"], errors="coerce").fillna(0) == 0)
    ).to_numpy(bool)


def load_preds(te: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(MODEL)
    if bundle.get("regime_bias") and not bundle.get("regime_bias_out_of_sample"):
        bundle = {**bundle, "regime_bias": None}
    chunks = []
    for i in range(0, len(te), 50_000):
        chunks.append(np.asarray(predict(bundle, te.iloc[i : i + 50_000]), dtype=float))
    return np.clip(np.concatenate(chunks), *DRIVE_CLIP_S)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    ae = np.abs(err)
    # % of actual (advisor "normalized error"); guard tiny y
    ok = y >= MIN_Y_FOR_PCT
    ape = np.full_like(ae, np.nan)
    ape[ok] = ae[ok] / y[ok] * 100.0
    # sMAPE: 2|p-y|/(|p|+|y|) — symmetric, bounded
    denom = np.abs(p) + np.abs(y)
    smape = np.where(denom > 1e-6, 2.0 * ae / denom * 100.0, np.nan)
    n_out = int((ae > OUTLIER_S).sum())
    return {
        "n": int(len(y)),
        "mae_s": float(ae.mean()),
        "medae_s": float(np.median(ae)),
        "bias_s": float(err.mean()),
        "rmse_s": float(np.sqrt(np.mean(err**2))),
        "mape_pct": float(np.nanmean(ape)),
        "median_ape_pct": float(np.nanmedian(ape)),
        "smape_pct": float(np.nanmean(smape)),
        "median_smape_pct": float(np.nanmedian(smape)),
        "pct_within_10pct": float(np.nanmean(ape <= 10)),
        "pct_within_20pct": float(np.nanmean(ape <= 20)),
        "pct_within_30pct": float(np.nanmean(ape <= 30)),
        "pct_within_60s": float(np.mean(ae <= 60)),
        "n_abs_error_gt_400s": n_out,
        "share_abs_error_gt_400s_pct": float(n_out / len(y) * 100),
        "mae_without_gt_400s": float(ae[ae <= OUTLIER_S].mean()) if (ae <= OUTLIER_S).any() else None,
        "example_60s_on_5min_pct": 60.0 / 300.0 * 100.0,
    }


def percentiles(a: np.ndarray, qs=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> dict:
    return {f"p{q}": float(np.nanpercentile(a, q)) for q in qs}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading…", flush=True)
    df = pd.read_csv(DATA, low_memory=False)
    te = df[df["cal_year"] == 2025].reset_index(drop=True)
    del df
    te = attach_weather(te, load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    print(f"predicting on {len(te):,} 2025 trips…", flush=True)
    p = load_preds(te)
    y = te["drive_s"].to_numpy(float)
    err = p - y
    ae = np.abs(err)
    place = placement_mask(te)

    slices = {
        "all_2025": np.ones(len(te), bool),
        "placement_first_home_not_busy": place,
        "placement_exact_address": place & (te["dest_exact"].to_numpy() == 1),
    }
    by_slice = {name: metrics(y[m], p[m]) for name, m in slices.items()}

    # Residual / APE distributions for the two main slices
    dist = {}
    for name, m in (("all_2025", slices["all_2025"]), ("placement", slices["placement_first_home_not_busy"])):
        e = err[m]
        a = ae[m]
        yy = y[m]
        ape = np.where(yy >= MIN_Y_FOR_PCT, a / yy * 100.0, np.nan)
        dist[name] = {
            "residual_p_minus_y_s": percentiles(e),
            "abs_error_s": percentiles(a),
            "ape_pct": percentiles(ape[np.isfinite(ape)]),
            "n_abs_error_gt_400s": int((a > OUTLIER_S).sum()),
        }

    summary = {
        "doc": (
            "Advisor travel-time validation: absolute error alone is incomplete; "
            "normalized (percent) error and the full residual histogram matter. "
            f"Outliers flagged at |error| > {OUTLIER_S:.0f} s. Production LightGBM, 2025 holdout."
        ),
        "slices": by_slice,
        "distributions": dist,
        "advisor_note": (
            "A 60 s MAE on a 5-minute (300 s) trip is a 20% relative error. "
            "Prefer median APE / MAPE alongside MAE, and show the full error histogram "
            "rather than mean/median alone."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = [{"slice": k, **v} for k, v in by_slice.items()]
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)

    # --- figure: full error distribution ---
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))

    # 1) residual histogram (all)
    ax = axes[0, 0]
    clip = (-400, 400)
    ax.hist(err, bins=80, range=clip, color="#1d5bbf", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(np.median(err), color="#b45309", ls="--", lw=1.2, label=f"median {np.median(err):.0f} s")
    ax.set_title("Signed error (predicted − actual), all 2025")
    ax.set_xlabel("Error (seconds)")
    ax.set_ylabel("Trips")
    ax.legend(fontsize=8)
    n_out_all = int((ae > OUTLIER_S).sum())
    ax.text(
        0.98,
        0.95,
        f"|error|>{OUTLIER_S:.0f}s: {n_out_all:,} trips\n({n_out_all/len(te)*100:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#7f1d1d",
    )

    # 2) abs error histogram (all) — log-ish view of tail
    ax = axes[0, 1]
    ax.hist(ae, bins=80, range=(0, 400), color="#0f766e", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(by_slice["all_2025"]["mae_s"], color="#b45309", ls="--", label=f"MAE {by_slice['all_2025']['mae_s']:.0f} s")
    ax.axvline(by_slice["all_2025"]["medae_s"], color="#7c3aed", ls="--", label=f"MedAE {by_slice['all_2025']['medae_s']:.0f} s")
    ax.set_title("Absolute error distribution (clipped at 400 s)")
    ax.set_xlabel("|Error| (seconds)")
    ax.set_ylabel("Trips")
    ax.legend(fontsize=8)

    # 3) APE histogram (normalized) — placement
    ax = axes[1, 0]
    yy = y[place]
    aa = ae[place]
    ape_p = aa[yy >= MIN_Y_FOR_PCT] / yy[yy >= MIN_Y_FOR_PCT] * 100.0
    ax.hist(ape_p, bins=70, range=(0, 100), color="#b45309", alpha=0.85, edgecolor="white", linewidth=0.3)
    med_ape = float(np.median(ape_p))
    mape = float(np.mean(ape_p))
    ax.axvline(med_ape, color="#1d5bbf", ls="--", label=f"median APE {med_ape:.0f}%")
    ax.axvline(mape, color="#7c3aed", ls="--", label=f"MAPE {mape:.0f}%")
    ax.axvline(20, color="#7f1d1d", ls=":", label="20% (60s on 5 min)")
    ax.set_title("Normalized error on placement trips (|err|/actual)")
    ax.set_xlabel("Absolute percent error (%)")
    ax.set_ylabel("Trips")
    ax.legend(fontsize=8)

    # 4) summary callout panel
    ax = axes[1, 1]
    ax.axis("off")
    a = by_slice["all_2025"]
    pl = by_slice["placement_first_home_not_busy"]
    text = (
        "Travel time model validation (2025)\n\n"
        f"All trips (n={a['n']:,})\n"
        f"  MAE {a['mae_s']:.0f} s · MedAE {a['medae_s']:.0f} s\n"
        f"  MAPE {a['mape_pct']:.0f}% · median APE {a['median_ape_pct']:.0f}%\n"
        f"  |error|>400 s: {a['n_abs_error_gt_400s']:,} "
        f"({a['share_abs_error_gt_400s_pct']:.1f}%)\n\n"
        f"Placement trips (n={pl['n']:,})\n"
        f"  MAE {pl['mae_s']:.0f} s · MedAE {pl['medae_s']:.0f} s\n"
        f"  MAPE {pl['mape_pct']:.0f}% · median APE {pl['median_ape_pct']:.0f}%\n"
        f"  within 20% of actual: {pl['pct_within_20pct']*100:.0f}%\n"
        f"  |error|>400 s: {pl['n_abs_error_gt_400s']:,}\n\n"
        "Advisor: prefer normalized error + full\n"
        "histogram over mean/median alone.\n"
        "Example: 60 s on a 5-min trip = 20%."
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#cbd5e1"))

    fig.suptitle("Travel Time Model Validation — absolute + normalized error", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "error_distribution.png", dpi=140)
    plt.close(fig)

    print(json.dumps({"all": {k: round(v, 2) if isinstance(v, float) else v for k, v in a.items()},
                      "placement": {k: round(v, 2) if isinstance(v, float) else v for k, v in pl.items()}}, indent=2))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
