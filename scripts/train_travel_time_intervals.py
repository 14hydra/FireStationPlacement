#!/usr/bin/env python3
"""p10/p50/p90 travel-time quantiles + conformal prediction intervals.

Builds reliable drive-time *ranges* (not just a point MAE) for evaluation:

  1. Fit quantile LightGBM at 0.10, 0.25, 0.50, 0.75, 0.90 on the same features
     as the production median model (FINAL_GROUPS).
  2. Conformalize the 50% (p25–p75) and 80% (p10–p90) bands on 2024
     (models trained on 2021–23) — asymmetric CQR by default.
  3. Refit quantiles on 2021–24, apply the 2024 conformal adjustments, score 2025.
  4. Report coverage + width overall and on first-engine / home / not-busy trips.

Writes:
  data/processed/models/travel_time_lfb_intervals.joblib
  data/processed/models/travel_time_lfb_intervals.metrics.json
  data/figures/analysis/travel_time_intervals/{summary.json,coverage.png,mae_table.csv}

Run:
  PYTHONPATH=src .venv/bin/python scripts/train_travel_time_intervals.py
  PYTHONPATH=src .venv/bin/python scripts/train_travel_time_intervals.py --quick
  PYTHONPATH=src .venv/bin/python scripts/train_travel_time_intervals.py --n-estimators 900
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
sys.path.insert(0, str(ROOT / "scripts"))

from train_travel_time_lfb import FINAL_GROUPS, TRAIN_YEARS, TUNED_PARAMS, Split  # noqa: E402

from firestation.conformal_quantiles import (  # noqa: E402
    apply_cqr,
    fit_cqr_asymmetric,
    fit_cqr_symmetric,
    finite_clip,
    interval_metrics,
    point_metrics,
)
from firestation.travel_time_model import (  # noqa: E402
    CATEGORICAL,
    DRIVE_CLIP_S,
    make_regressor,
    predict_bag,
)

OUT_DIR = ROOT / "data" / "figures" / "analysis" / "travel_time_intervals"
MODEL_OUT = ROOT / "data" / "processed" / "models" / "travel_time_lfb_intervals.joblib"
ALPHAS = (0.10, 0.25, 0.50, 0.75, 0.90)
INTERVALS = {
    "50": {"lo_alpha": 0.25, "hi_alpha": 0.75, "nominal": 0.50},
    "80": {"lo_alpha": 0.10, "hi_alpha": 0.90, "nominal": 0.80},
}


def placement_mask(df: pd.DataFrame) -> np.ndarray:
    return (
        (pd.to_numeric(df["pump_order"], errors="coerce") == 1)
        & (df["deployed_from_location"].astype(str) == "Home Station")
        & (pd.to_numeric(df["busy_flag"], errors="coerce").fillna(0) == 0)
    ).to_numpy(bool)


def fit_quantile(xtr, ytr, alpha: float, n_estimators: int, seed: int = 42):
    params = {**TUNED_PARAMS, "objective": "quantile", "alpha": alpha, "n_estimators": n_estimators}
    m = make_regressor(seed, params)
    cats = [c for c in CATEGORICAL if c in xtr.columns]
    m.fit(xtr, ytr, categorical_feature=cats)
    return m, params


def predict_q(model, x) -> np.ndarray:
    return finite_clip(np.asarray(model.predict(x), dtype=float))


def tune_trees(inc: pd.DataFrame, n_cap: int) -> int:
    """Early-stop once on median quantile (α=0.5); reuse tree count for all alphas."""
    import lightgbm as lgb

    print("tuning trees on 2024 (quantile 0.5)…", flush=True)
    s = Split(inc[inc.cal_year.between(2021, 2023)], inc[inc.cal_year == 2024])
    xtr, xva = s.x(FINAL_GROUPS)
    params = {**TUNED_PARAMS, "objective": "quantile", "alpha": 0.5, "n_estimators": n_cap}
    m = make_regressor(42, params)
    m.fit(
        xtr,
        s.ytr,
        eval_set=[(xva, s.yte)],
        eval_metric="quantile",
        categorical_feature=[c for c in CATEGORICAL if c in xtr.columns],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    n = int(max(200, m.best_iteration_ * 1.15))
    print(f"  best {m.best_iteration_} → {n} trees", flush=True)
    return n


def fit_all_quantiles(xtr, ytr, n_estimators: int) -> dict[str, object]:
    models = {}
    for a in ALPHAS:
        print(f"  fitting quantile {a:.2f} on {len(xtr):,} trips…", flush=True)
        m, _ = fit_quantile(xtr, ytr, a, n_estimators)
        models[f"{a:.2f}"] = m
    return models


def predict_all(models: dict[str, object], x) -> dict[str, np.ndarray]:
    return {k: predict_q(m, x) for k, m in models.items()}


def build_interval_specs(
    cal_y: np.ndarray,
    cal_preds: dict[str, np.ndarray],
    *,
    asymmetric: bool = True,
) -> dict[str, dict]:
    specs = {}
    for name, meta in INTERVALS.items():
        lo = cal_preds[f"{meta['lo_alpha']:.2f}"]
        hi = cal_preds[f"{meta['hi_alpha']:.2f}"]
        mis = 1.0 - meta["nominal"]
        cqr = (
            fit_cqr_asymmetric(cal_y, lo, hi, mis)
            if asymmetric
            else fit_cqr_symmetric(cal_y, lo, hi, mis)
        )
        raw = interval_metrics(cal_y, lo, hi, meta["nominal"])
        clo, chi = apply_cqr(lo, hi, cqr)
        cal = interval_metrics(cal_y, clo, chi, meta["nominal"])
        specs[name] = {
            "lo_alpha": meta["lo_alpha"],
            "hi_alpha": meta["hi_alpha"],
            "nominal": meta["nominal"],
            "cqr": cqr,
            "calibration_raw": raw,
            "calibration_conformal": cal,
        }
        print(
            f"  {name}% CQR on 2024: raw cov {raw['coverage']:.3f} → "
            f"conformal {cal['coverage']:.3f} "
            f"(median width {cal['median_width_s']:.0f} s)",
            flush=True,
        )
    return specs


def score_year(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    specs: dict[str, dict],
    mask: np.ndarray | None = None,
) -> dict:
    if mask is not None:
        y = y[mask]
        preds = {k: v[mask] for k, v in preds.items()}
    out = {"point_p50": point_metrics(y, preds["0.50"])}
    for name, spec in specs.items():
        lo = preds[f"{spec['lo_alpha']:.2f}"]
        hi = preds[f"{spec['hi_alpha']:.2f}"]
        out[f"interval_{name}_raw"] = interval_metrics(y, lo, hi, spec["nominal"])
        clo, chi = apply_cqr(lo, hi, spec["cqr"])
        out[f"interval_{name}_conformal"] = interval_metrics(y, clo, chi, spec["nominal"])
    return out


def main() -> int:
    import joblib

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "lfb_travel_training.csv")
    ap.add_argument("--test-year", type=int, default=2025)
    ap.add_argument("--n-estimators", type=int, default=None, help="Tree count (default: tune on 2024, or 600 with --quick)")
    ap.add_argument("--quick", action="store_true", help="600 trees, skip retune")
    ap.add_argument("--symmetric", action="store_true", help="Symmetric CQR instead of asymmetric")
    ap.add_argument("--max-rows", type=int, default=0, help="Optional subsample for dry runs")
    args = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    usecols = None  # full CSV; Split/build_features need the usual columns
    print("loading trips…", flush=True)
    inc = pd.read_csv(args.data, low_memory=False)
    inc = inc[inc.cal_year.isin(TRAIN_YEARS + (args.test_year,))].reset_index(drop=True)
    if args.max_rows and len(inc) > args.max_rows:
        # keep year balance roughly
        n_per = max(1, args.max_rows // int(inc.cal_year.nunique()))
        parts = [g.sample(min(len(g), n_per), random_state=0) for _, g in inc.groupby("cal_year")]
        inc = pd.concat(parts, ignore_index=True)
        print(f"  subsampled to {len(inc):,} rows", flush=True)

    if args.n_estimators is not None:
        n_trees = args.n_estimators
    elif args.quick:
        n_trees = 600
    else:
        # Prefer production tree count when present; else early-stop.
        prod = ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib"
        if prod.exists():
            bundle = joblib.load(prod)
            n_trees = int((bundle.get("lightgbm_params") or {}).get("n_estimators") or 900)
            print(f"using production n_estimators={n_trees}", flush=True)
        else:
            n_trees = tune_trees(inc, 4000)

    # --- conformal calibration: train 2021–23, calibrate on 2024 ---
    print("calibration fit (2021–23 → 2024)…", flush=True)
    s_cal = Split(inc[inc.cal_year.between(2021, 2023)], inc[inc.cal_year == 2024])
    xtr_cal, xcal = s_cal.x(FINAL_GROUPS)
    models_cal = fit_all_quantiles(xtr_cal, s_cal.ytr, n_trees)
    preds_cal = predict_all(models_cal, xcal)
    specs = build_interval_specs(s_cal.yte, preds_cal, asymmetric=not args.symmetric)

    # --- final fit 2021–24, score 2025 ---
    print("final fit (2021–24 → 2025)…", flush=True)
    s = Split(inc[inc.cal_year.isin(TRAIN_YEARS)], inc[inc.cal_year == args.test_year])
    xtr, xte = s.x(FINAL_GROUPS)
    models = fit_all_quantiles(xtr, s.ytr, n_trees)
    preds_te = predict_all(models, xte)
    y = s.yte

    # Production median MAE for reference (strip invalid regime_bias if needed)
    prod_mae = None
    prod_path = ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib"
    if prod_path.exists():
        from firestation.travel_time_model import predict as lgbm_predict

        pb = joblib.load(prod_path)
        if pb.get("regime_bias") and not pb.get("regime_bias_out_of_sample"):
            pb = {**pb, "regime_bias": None}
        chunks = []
        for i in range(0, len(s.te), 50_000):
            chunks.append(np.asarray(lgbm_predict(pb, s.te.iloc[i : i + 50_000]), dtype=float))
        pred_prod = np.clip(np.concatenate(chunks), *DRIVE_CLIP_S)
        prod_mae = point_metrics(y, pred_prod)

    overall = score_year(y, preds_te, specs)
    place = placement_mask(s.te)
    placement = score_year(y, preds_te, specs, mask=place) if place.sum() > 1000 else {}

    # Also score raw production point vs p50 on placement
    if prod_mae is not None and place.sum() > 1000:
        placement["point_production"] = point_metrics(y[place], pred_prod[place])
        overall["point_production"] = prod_mae

    summary = {
        "doc": (
            "Quantile LightGBM (p10/p25/p50/p75/p90) with asymmetric conformal calibration "
            "on 2024. Final models trained 2021–2024, scored on 2025. Intervals: 50% = "
            "p25–p75, 80% = p10–p90, each expanded by CQR so coverage matches the target."
        ),
        "n_train": int(len(s.tr)),
        "n_test": int(len(s.te)),
        "n_placement": int(place.sum()),
        "n_estimators": n_trees,
        "alphas": list(ALPHAS),
        "cqr_mode": "symmetric" if args.symmetric else "asymmetric",
        "intervals": {
            k: {
                "lo_alpha": v["lo_alpha"],
                "hi_alpha": v["hi_alpha"],
                "nominal": v["nominal"],
                "cqr": v["cqr"],
                "calibration_2024_raw": v["calibration_raw"],
                "calibration_2024_conformal": v["calibration_conformal"],
            }
            for k, v in specs.items()
        },
        "test_2025": overall,
        "test_2025_placement": placement,
        "verdict": _verdict(overall, placement),
        "elapsed_s": round(time.time() - t0, 1),
    }

    payload = {
        "kind": "lightgbm_v3_intervals_cqr",
        "model": "quantile LightGBM p10/p25/p50/p75/p90 + asymmetric CQR",
        "features": list(xtr.columns),
        "feature_groups": list(FINAL_GROUPS),
        "categorical": [c for c in CATEGORICAL if c in xtr.columns],
        "category_maps": s.cats,
        "borough_categories": s.cats["borough"],
        "dest_history": {
            "sizes_m": list(s.dest.sizes_m),
            "shrink": s.dest.shrink,
            "tables": {str(k): t.to_dict() for k, t in s.dest.tables.items()},
        },
        "kolesar_baseline_params": s.kol.to_dict(),
        "kolesar_baseline_distance": "road_km",
        "drive_clip_s": list(DRIVE_CLIP_S),
        "train_years": list(TRAIN_YEARS),
        "test_year": args.test_year,
        "quantile_models": {k: [m] for k, m in models.items()},
        "quantile_alphas": list(ALPHAS),
        "intervals": {
            k: {"lo_alpha": v["lo_alpha"], "hi_alpha": v["hi_alpha"], "nominal": v["nominal"], "cqr": v["cqr"]}
            for k, v in specs.items()
        },
        "lightgbm_params": {**TUNED_PARAMS, "n_estimators": n_trees, "objective": "quantile"},
        "metrics_test": summary,
        # point API compatibility: p50 as "models"
        "models": [models["0.50"]],
        "quantile": 0.5,
        "target": "drive_s quantiles + conformal intervals",
    }
    joblib.dump(payload, MODEL_OUT)
    MODEL_OUT.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = []
    for label, block in (("all_2025", overall), ("placement_2025", placement)):
        if not block:
            continue
        rows.append({"slice": label, "model": "p50", **block.get("point_p50", {})})
        if "point_production" in block:
            rows.append({"slice": label, "model": "production_median", **block["point_production"]})
        for name in ("50", "80"):
            for kind in ("raw", "conformal"):
                key = f"interval_{name}_{kind}"
                if key in block:
                    rows.append({"slice": label, "model": f"{name}pct_{kind}", **block[key]})
    pd.DataFrame(rows).to_csv(OUT_DIR / "mae_table.csv", index=False)

    _plot(summary, OUT_DIR / "coverage.png")
    print(json.dumps({"verdict": summary["verdict"], "placement": placement}, indent=2, default=str)[:2500])
    print(f"wrote {MODEL_OUT}")
    print(f"wrote {OUT_DIR} ({(time.time() - t0) / 60:.1f} min)")
    return 0


def _verdict(overall: dict, placement: dict) -> str:
    parts = []
    p50 = overall.get("point_p50", {})
    parts.append(f"p50 MAE {p50.get('mae_s', float('nan')):.1f} s on all 2025")
    if placement.get("point_p50"):
        parts.append(f"{placement['point_p50']['mae_s']:.1f} s on placement trips")
    for name in ("50", "80"):
        key = f"interval_{name}_conformal"
        if key in overall:
            m = overall[key]
            parts.append(
                f"{name}% conformal coverage {m['coverage']:.3f} "
                f"(target {m['nominal']:.2f}, median width {m['median_width_s']:.0f} s)"
            )
        if placement and key in placement:
            m = placement[key]
            parts.append(
                f"placement {name}% cov {m['coverage']:.3f}, median width {m['median_width_s']:.0f} s"
            )
    return "; ".join(parts) + "."


def _plot(summary: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    labels, covs, noms, colors = [], [], [], []
    for slice_name, color in (("test_2025", "#1d5bbf"), ("test_2025_placement", "#15803d")):
        block = summary.get(slice_name) or {}
        for name in ("50", "80"):
            key = f"interval_{name}_conformal"
            if key not in block:
                continue
            labels.append(f"{'all' if 'placement' not in slice_name else 'place'} {name}%")
            covs.append(block[key]["coverage"])
            noms.append(block[key]["nominal"])
            colors.append(color)
    x = np.arange(len(labels))
    ax.bar(x, covs, color=colors, width=0.7)
    for i, (c, n) in enumerate(zip(covs, noms)):
        ax.plot([i - 0.35, i + 0.35], [n, n], "k--", lw=1.2)
        ax.text(i, c + 0.02, f"{c:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Coverage")
    ax.set_title("Conformal interval coverage (2025)")

    ax = axes[1]
    widths, wlabels, wcolors = [], [], []
    for slice_name, color in (("test_2025", "#1d5bbf"), ("test_2025_placement", "#15803d")):
        block = summary.get(slice_name) or {}
        for name in ("50", "80"):
            key = f"interval_{name}_conformal"
            if key not in block:
                continue
            widths.append(block[key]["median_width_s"])
            wlabels.append(f"{'all' if 'placement' not in slice_name else 'place'} {name}%")
            wcolors.append(color)
    y = np.arange(len(wlabels))
    ax.barh(y, widths, color=wcolors)
    ax.set_yticks(y)
    ax.set_yticklabels(wlabels)
    ax.set_xlabel("Median interval width (s)")
    ax.set_title("How wide are the reliable ranges?")
    for yi, w in zip(y, widths):
        ax.text(w + 2, yi, f"{w:.0f} s", va="center", fontsize=9)
    fig.suptitle("Travel-time prediction intervals (quantile LightGBM + CQR)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
