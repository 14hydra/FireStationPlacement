"""Transferable trip-spread PDF (Ricky street-PDF idea, without London street graph).

Ricky's full street PDF simulator needs London OSM edges + street-speed offsets.
Megacities only have point travel times (Kolesar). The part that *does* transfer is
the empirical **trip spread**: recorded_time / point_time ratios by regime
(rush × route-length bin), learned on London 2024 holdout-style trips.

Use for Monte Carlo scoring of megacity layouts — not for the placement solver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "data" / "processed" / "models" / "street_pdf_transfer.joblib"
N_DIST = 6
DRIVE_LO, DRIVE_HI = 30.0, 1200.0


def _rush(hour: np.ndarray) -> np.ndarray:
    h = np.asarray(hour, dtype=float)
    return ((h >= 7) & (h <= 9)) | ((h >= 16) & (h <= 19))


def regime_ids(point_s: np.ndarray, hour: np.ndarray, dist_edges: np.ndarray) -> np.ndarray:
    rush = _rush(hour).astype(int)
    db = np.clip(np.searchsorted(dist_edges, point_s) - 1, 0, N_DIST - 1)
    # home/first-engine flag fixed to 1 for placement-style megacity scoring
    home = 1
    return rush * 2 * N_DIST + home * N_DIST + db


def fit_kolesar_ratio_pdf(
    trips: pd.DataFrame,
    *,
    fit_years: tuple[int, ...] = (2021, 2022, 2023),
    pdf_year: int = 2024,
    seed: int = 0,
) -> dict[str, Any]:
    """Learn Kolesar point model + 2024 residual ratio pools (London LFB trips)."""
    from .kolesar import fit_kolesar

    y = pd.to_numeric(trips["drive_s"], errors="coerce").to_numpy(float)
    road = pd.to_numeric(trips["road_km"], errors="coerce").to_numpy(float)
    year = pd.to_numeric(trips["cal_year"], errors="coerce").to_numpy(int)
    hour = pd.to_numeric(trips.get("hour_of_call", 12), errors="coerce").fillna(12).to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(road) & (road > 0.05) & (y >= DRIVE_LO) & (y <= DRIVE_HI)
    fit_m = ok & np.isin(year, fit_years)
    pdf_m = ok & (year == pdf_year)
    # Prefer first-engine / home / not-busy for placement-like pools when columns exist
    if {"pump_order", "deployed_from_location", "busy_flag"}.issubset(trips.columns):
        place = (
            (pd.to_numeric(trips["pump_order"], errors="coerce") == 1)
            & (trips["deployed_from_location"].astype(str) == "Home Station")
            & (pd.to_numeric(trips["busy_flag"], errors="coerce").fillna(0) == 0)
        ).to_numpy(bool)
        if (pdf_m & place).sum() >= 5000:
            pdf_m = pdf_m & place
        if (fit_m & place).sum() >= 20_000:
            fit_m = fit_m & place

    kol = fit_kolesar(road[fit_m], y[fit_m])
    point = np.clip(kol.predict(road), DRIVE_LO, DRIVE_HI)
    edges = np.quantile(point[pdf_m], np.linspace(0, 1, N_DIST + 1))
    edges[0], edges[-1] = 0.0, max(float(edges[-1]), float(point[pdf_m].max()) + 1.0)
    reg = regime_ids(point, hour, edges)
    pools: dict[int, np.ndarray] = {}
    global_ratio = y[pdf_m] / np.maximum(point[pdf_m], 1.0)
    global_ratio = global_ratio[np.isfinite(global_ratio)]
    for rg in np.unique(reg[pdf_m]):
        m = pdf_m & (reg == rg)
        if m.sum() < 200:
            pools[int(rg)] = global_ratio.astype(np.float32)
        else:
            pools[int(rg)] = (y[m] / np.maximum(point[m], 1.0)).astype(np.float32)

    return {
        "kind": "kolesar_ratio_transfer_pdf",
        "note": (
            "Transferable slice of the street PDF simulator: empirical drive_s / Kolesar "
            "ratios by rush × length bin on London 2024 (placement-like trips when available). "
            "For megacity Monte Carlo scoring only — not a placement scorer."
        ),
        "kolesar": kol.to_dict(),
        "dist_edges": edges.astype(float),
        "n_dist": N_DIST,
        "pools": pools,
        "n_fit": int(fit_m.sum()),
        "n_pdf": int(pdf_m.sum()),
        "seed": seed,
    }


def save_pdf(bundle: dict[str, Any], path: Path = DEFAULT_PATH) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_pdf(path: Path = DEFAULT_PATH) -> dict[str, Any]:
    import joblib

    return joblib.load(path)


def sample_ratios(
    point_s: np.ndarray,
    hour: np.ndarray,
    bundle: dict[str, Any],
    *,
    n_draws: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Return array [n, n_draws] of multiplicative ratios for each point time."""
    rng = np.random.default_rng(seed)
    edges = np.asarray(bundle["dist_edges"], dtype=float)
    pools = {int(k): np.asarray(v, dtype=float) for k, v in bundle["pools"].items()}
    fallback = np.concatenate(list(pools.values())) if pools else np.array([1.0])
    reg = regime_ids(np.asarray(point_s, dtype=float), hour, edges)
    out = np.empty((len(point_s), n_draws), dtype=np.float32)
    for r in np.unique(reg):
        m = reg == r
        pool = pools.get(int(r), fallback)
        out[m] = rng.choice(pool, size=(int(m.sum()), n_draws), replace=True)
    return out


def score_layout_draws(
    response_today_s: np.ndarray,
    response_plan_s: np.ndarray,
    drive_plan_s: np.ndarray,
    weights: np.ndarray,
    hour: np.ndarray,
    bundle: dict[str, Any],
    *,
    n_draws: int = 100,
    seed: int = 0,
    target_s: float = 360.0,
) -> dict[str, Any]:
    """Monte Carlo attendance under London transfer PDF, paired today vs plan.

    Turnout is held fixed: turnout = response_plan - drive_plan; today's drive is
    inferred as response_today - turnout.
    """
    today = np.asarray(response_today_s, dtype=float)
    plan = np.asarray(response_plan_s, dtype=float)
    drive_plan = np.asarray(drive_plan_s, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    turnout = np.clip(plan - drive_plan, 20.0, 180.0)
    drive_today = np.clip(today - turnout, DRIVE_LO, DRIVE_HI)
    drive_plan = np.clip(drive_plan, DRIVE_LO, DRIVE_HI)

    # Sample ratios from today's drive length regime (same U for both = paired)
    ratios = sample_ratios(drive_today, hour, bundle, n_draws=n_draws, seed=seed)
    sim_today = turnout[:, None] + drive_today[:, None] * ratios
    sim_plan = turnout[:, None] + drive_plan[:, None] * ratios

    def wmean(a: np.ndarray) -> np.ndarray:
        return w @ a  # [n_draws]

    avg_today = wmean(sim_today)
    avg_plan = wmean(sim_plan)
    delta = avg_plan - avg_today
    within_today = w @ (sim_today <= target_s)
    within_plan = w @ (sim_plan <= target_s)

    point_delta = float(w @ (plan - today))
    return {
        "n_cells": int(len(today)),
        "n_draws": int(n_draws),
        "point_kolesar": {
            "average_s_today": float(w @ today),
            "average_s_plan": float(w @ plan),
            "delta_s": point_delta,
            "within_6min_today": float(w @ (today <= target_s)),
            "within_6min_plan": float(w @ (plan <= target_s)),
        },
        "pdf_transfer": {
            "average_delta_s_mean": float(delta.mean()),
            "average_delta_s_p10": float(np.percentile(delta, 10)),
            "average_delta_s_p50": float(np.percentile(delta, 50)),
            "average_delta_s_p90": float(np.percentile(delta, 90)),
            "prob_plan_faster": float((delta < 0).mean()),
            "within_6min_today_mean": float(within_today.mean()),
            "within_6min_plan_mean": float(within_plan.mean()),
            "within_6min_lift_mean": float((within_plan - within_today).mean()),
            "within_6min_lift_p10": float(np.percentile(within_plan - within_today, 10)),
            "within_6min_lift_p90": float(np.percentile(within_plan - within_today, 90)),
        },
    }
