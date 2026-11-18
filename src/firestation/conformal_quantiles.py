"""Conformalized quantile regression (CQR) for travel-time intervals.

Train quantile LightGBM models (e.g. p10 / p50 / p90), then expand the raw
quantile band with a calibration set so held-out coverage matches a target
level (Romano, Patterson, Candès — Conformalized Quantile Regression).

Asymmetric CQR (default): separate lower/upper adjustments so under- and
over-coverage are balanced. Symmetric CQR uses a single max-residual radius.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .travel_time_model import DRIVE_CLIP_S, predict_bag


def finite_clip(a: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(a, dtype=float), *DRIVE_CLIP_S)


def conformal_level(n: int, miscoverage: float) -> float:
    """Finite-sample quantile level (1 − α)(1 + 1/n), capped at 1."""
    if n <= 0:
        return 1.0
    return float(min(1.0, np.ceil((n + 1) * (1.0 - miscoverage)) / n))


def fit_cqr_symmetric(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, miscoverage: float) -> dict[str, float]:
    scores = np.maximum(lo - y, y - hi)
    n = int(len(scores))
    radius = float(np.quantile(scores, conformal_level(n, miscoverage), method="higher"))
    return {"mode": "symmetric", "miscoverage": float(miscoverage), "radius_s": radius, "n_cal": n}


def fit_cqr_asymmetric(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, miscoverage: float) -> dict[str, float]:
    n = int(len(y))
    level = conformal_level(n, miscoverage / 2.0)
    adj_lo = float(np.quantile(lo - y, level, method="higher"))
    adj_hi = float(np.quantile(y - hi, level, method="higher"))
    return {
        "mode": "asymmetric",
        "miscoverage": float(miscoverage),
        "adj_lo_s": adj_lo,
        "adj_hi_s": adj_hi,
        "n_cal": n,
    }


def apply_cqr(lo: np.ndarray, hi: np.ndarray, cal: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if cal.get("mode") == "symmetric":
        r = float(cal["radius_s"])
        return finite_clip(lo - r), finite_clip(hi + r)
    return finite_clip(lo - float(cal["adj_lo_s"])), finite_clip(hi + float(cal["adj_hi_s"]))


def interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, nominal: float) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    return {
        "nominal": float(nominal),
        "coverage": float(inside.mean()),
        "mean_width_s": float(width.mean()),
        "median_width_s": float(np.median(width)),
        "mean_lo_s": float(lo.mean()),
        "mean_hi_s": float(hi.mean()),
        "n": int(len(y)),
    }


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
    ae = np.abs(e)
    return {
        "n": int(len(y)),
        "mae_s": float(ae.mean()),
        "medae_s": float(np.median(ae)),
        "bias_s": float(e.mean()),
        "corr": float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else float("nan"),
    }


def predict_quantile_bundle(bundle: dict, trips: pd.DataFrame, alpha: float | str) -> np.ndarray:
    """Predict one stored quantile from an intervals bundle (keys like '0.1' or 0.1)."""
    from .kolesar import KolesarModel
    from .travel_time_model import (
        DestinationHistory,
        attach_weather,
        build_features,
        _infer_feature_groups,
    )

    key = f"{float(alpha):.2f}"
    models = bundle["quantile_models"][key]
    groups = list(bundle.get("feature_groups") or _infer_feature_groups(bundle["features"]))
    if "dest" in groups and "dest_history" not in bundle:
        groups = [g for g in groups if g != "dest"]
    dh = None
    if "dest" in groups:
        dh = DestinationHistory(
            tuple(bundle["dest_history"]["sizes_m"]),
            bundle["dest_history"]["shrink"],
            {int(k): pd.DataFrame(v) for k, v in bundle["dest_history"]["tables"].items()},
        )
    if "weather" in groups and "temp_c" not in trips.columns:
        trips = attach_weather(trips, None)
    cats = bundle.get("category_maps") or {
        "borough": list(bundle.get("borough_categories") or []),
        "incident_group": [],
        "property_category": [],
    }
    x = build_features(
        trips,
        groups,
        kolesar=KolesarModel.from_dict(bundle["kolesar_baseline_params"]),
        cat_maps=cats,
        dest_history=dh,
    )
    for c in bundle["features"]:
        if c not in x.columns:
            x[c] = np.nan
    return finite_clip(predict_bag(models if isinstance(models, list) else [models], x[bundle["features"]]))


def predict_intervals(
    bundle: dict,
    trips: pd.DataFrame,
    *,
    level: str = "80",
    conformal: bool = True,
) -> dict[str, np.ndarray]:
    """Return lo / mid / hi drive-time arrays for a named interval level ('50' or '80')."""
    spec = bundle["intervals"][str(level)]
    lo = predict_quantile_bundle(bundle, trips, spec["lo_alpha"])
    hi = predict_quantile_bundle(bundle, trips, spec["hi_alpha"])
    mid = predict_quantile_bundle(bundle, trips, 0.5)
    if conformal and "cqr" in spec:
        lo, hi = apply_cqr(lo, hi, spec["cqr"])
    # keep ordering
    lo2 = np.minimum(lo, hi)
    hi2 = np.maximum(lo, hi)
    mid = np.clip(mid, lo2, hi2)
    return {"lo": lo2, "mid": mid, "hi": hi2}


def summarize_cqr(cal: dict[str, Any]) -> dict[str, Any]:
    out = {k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v) for k, v in cal.items()}
    return out
