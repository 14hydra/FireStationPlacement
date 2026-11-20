#!/usr/bin/env python3
"""Exploratory (eMentor meeting 2): a per-street statistical travel time model that samples trip times from distributions.

Nothing in the pipeline is replaced; this only compares the simulator with the production LightGBM model on 2025 trips.

Model ("street PDF simulator"):
  1. Street means. Every street segment e has a mean time t_e from the learned street speeds (section 1.4: pace from the
     street's road class and features). Each segment then gets its own multiplier (1 + u_e), learned from 2021-2023 trips
     that drove it (ridge regression on the trip residuals, so streets with few trips stay near their class average).
     The ridge strength is chosen on January to June 2024 trips.
  2. Trip spread. LFB records whole-trip times, not times per street (no GPS), so the spread cannot be estimated street by
     street. It is an empirical distribution of recorded / mean-route time, learned on 2024 trips (not used in step 1),
     separately for each regime: rush hour or not x first engine from its home station or not x 6 route-length bins.
  3. A trip's time = start-up + off-graph hop + sum over its route of t_e (1 + u_e), times a ratio drawn from its regime.
     200 draws per trip give a distribution; its median is the point guess.

Scored on 2025 (all trips, and the trips placement uses): point MAE, the distribution against the recorded one
(percentiles, Wasserstein gap, KS, share over 6 minutes), CRPS and 50 % / 80 % interval coverage. Compared with LightGBM
(one prediction per trip) and LightGBM + sampled spread (scripts/travel_time_distribution.py, learned the same way on 2024).

Writes data/figures/analysis/street_pdf_model/ (summary.json, percentiles.csv, test_samples.npz).
Run:  PYTHONPATH=src .venv/bin/python scripts/explore_street_pdf_model.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import lsqr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation import ml_scorer  # noqa: E402
from firestation.street_speeds import StreetSpeedModel, edge_design  # noqa: E402
from firestation.travel_time_model import DRIVE_CLIP_S, attach_weather, load_weather, predict  # noqa: E402

OUT = ROOT / "data" / "figures" / "analysis" / "street_pdf_model"
DATA = ROOT / "data" / "processed" / "lfb_travel_training.csv"
DAMPS = (30.0, 100.0, 300.0, 1000.0)  # ridge strengths tried (in seconds of route time)
N_DRAWS = 200
N_DIST = 6
PCTS = (5, 10, 25, 50, 75, 90, 95, 99)


def routes(router, times: np.ndarray, s_idx: np.ndarray, d_idx: np.ndarray):
    """Quickest route of every trip as (trip, edge) pairs, plus route time."""
    e = router.edge_id.tocoo()
    n = router.csr.shape[0]
    g = csr_matrix((times[e.data - 1], (e.row, e.col)), shape=(n, n))
    nodes = np.arange(n)
    rows, cols, rtime = [], [], np.zeros(len(s_idx))
    for o in np.unique(s_idx):
        dist, pred = dijkstra(g, directed=True, indices=o, return_predecessors=True)
        has = pred >= 0
        eid = np.full(n, -1)
        eid[has] = np.asarray(router.edge_id[pred[has], nodes[has]]).ravel() - 1
        trips = np.flatnonzero(s_idx == o)
        rtime[trips] = dist[d_idx[trips]]
        cur = d_idx[trips].copy()
        live = (cur != o) & (pred[cur] >= 0)
        while live.any():
            t, c = trips[live], cur[live]
            rows.append(t.astype(np.int32))
            cols.append(eid[c].astype(np.int32))
            cur[live] = pred[c]
            live = (cur != o) & (pred[cur] >= 0)
    return np.concatenate(rows), np.concatenate(cols), rtime


def regime(df: pd.DataFrame, base: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h = pd.to_numeric(df["hour_of_call"], errors="coerce").fillna(12).to_numpy()
    rush = ((h >= 7) & (h <= 9)) | ((h >= 16) & (h <= 19))
    home = ((df["pump_order"] == 1) & (df["deployed_from_location"] == "Home Station") & (df["busy_flag"] == 0)).to_numpy()
    db = np.clip(np.searchsorted(edges, base) - 1, 0, N_DIST - 1)
    return rush.astype(int) * 2 * N_DIST + home.astype(int) * N_DIST + db


def sample(point: np.ndarray, reg: np.ndarray, pools: dict, rng) -> np.ndarray:
    out = np.empty((len(point), N_DRAWS), np.float32)
    for r in np.unique(reg):
        m = reg == r
        out[m] = point[m, None] * rng.choice(pools[r], (m.sum(), N_DRAWS))
    return np.clip(out, *DRIVE_CLIP_S)


def crps(draws: np.ndarray, y: np.ndarray, k: int = 50) -> float:
    d = draws[:, :k]
    a = np.abs(d - y[:, None]).mean(1)
    b = np.abs(d[:, :, None] - d[:, None, :]).mean((1, 2))
    return float((a - 0.5 * b).mean())


def dist_stats(sim: np.ndarray, y: np.ndarray) -> dict:
    q = np.linspace(0.005, 0.995, 199)
    xs = np.sort(np.concatenate([sim, y]))
    ks = np.abs(np.searchsorted(np.sort(sim), xs, "right") / len(sim) - np.searchsorted(np.sort(y), xs, "right") / len(y)).max()
    return {"wasserstein_s": round(float(np.abs(np.quantile(sim, q) - np.quantile(y, q)).mean()), 1), "ks": round(float(ks), 3),
            "sd_s": round(float(sim.std()), 1), "over_360s_pct": round(float((sim > 360).mean() * 100), 1)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_csv(DATA, low_memory=False)
    df = df[df["cal_year"].between(2021, 2025)].reset_index(drop=True)
    month = pd.to_datetime(df["date_of_call"], format="mixed").dt.month.to_numpy()
    yr = df["cal_year"].to_numpy()
    y = df["drive_s"].to_numpy(float)
    fit_m, tune_m, pdf_m, te_m = yr <= 2023, (yr == 2024) & (month <= 6), yr == 2024, yr == 2025

    router = ml_scorer._router()
    sm = StreetSpeedModel.load()
    Z = edge_design(router)
    te_ = sm.edge_times(router, Z)
    s_idx, s_off = router.snap(df["start_lat"].to_numpy(), df["start_lon"].to_numpy())
    d_idx, d_off = router.snap(df["dest_lat"].to_numpy(), df["dest_lon"].to_numpy())
    print(f"Routing {len(df):,} trips…", flush=True)
    r, c, rt = routes(router, te_, s_idx, d_idx)
    const = sm.coef[0] + sm.coef[1] * (s_off + d_off)
    base = const + rt
    M = csr_matrix((te_[c].astype(np.float32), (r, c)), shape=(len(df), len(te_)))
    del r, c
    gc.collect()
    n_edges_used = int((np.asarray(M[fit_m].getnnz(axis=0)) > 0).sum())
    print(f"  {time.time() - t0:.0f} s; {M.nnz:,} trip-street pairs; {n_edges_used:,} streets driven in 2021-23", flush=True)

    # 1. per-street multipliers: residual ~ M u, ridge, Huber-style clip of residuals
    Mf = M[fit_m]
    res = np.clip(y[fit_m] - base[fit_m], -180, 180)
    tune = {}
    fits = {}
    for dmp in DAMPS:
        u = lsqr(Mf, res, damp=dmp, iter_lim=150)[0]
        p = np.clip(base[tune_m] + M[tune_m] @ u, *DRIVE_CLIP_S)
        tune[dmp] = float(np.abs(p - y[tune_m]).mean())
        fits[dmp] = u
        print(f"  ridge {dmp:>6}: Jan-Jun 2024 MAE {tune[dmp]:.2f} s", flush=True)
    best = min(tune, key=tune.get)
    u = fits[best]
    mean_t = np.clip(base + M @ u, *DRIVE_CLIP_S)
    base_c = np.clip(base, *DRIVE_CLIP_S)
    del Mf, M, fits
    gc.collect()

    # 2. trip spread by regime, learned on 2024
    edges = np.quantile(mean_t[pdf_m], np.linspace(0, 1, N_DIST + 1))
    reg = regime(df, mean_t, edges)
    pools = {}
    for rg in np.unique(reg):
        m = pdf_m & (reg == rg)
        pools[rg] = (y[m] / mean_t[m]) if m.sum() >= 200 else (y[pdf_m] / mean_t[pdf_m])
    # Save the simulator so other scripts can use it (scripts/simulate_calls_over_6min.py). Turnout spread for the same
    # 2024 trips: first engines from home, deviation from their station's 2024 median turnout.
    fe = pdf_m & ((df["pump_order"] == 1) & (df["deployed_from_location"] == "Home Station")).to_numpy()
    tos = pd.to_numeric(df.loc[fe, "turnout_s"], errors="coerce")
    tdev = (tos - tos.groupby(df.loc[fe, "deployed_from_station"]).transform("median")).dropna().to_numpy(float)
    joblib.dump({"street_offsets": u.astype(np.float32), "dist_edges": edges, "n_dist": N_DIST,
                 "pools": {int(k): v.astype(np.float32) for k, v in pools.items()}, "turnout_dev_s": tdev.astype(np.float32),
                 "ridge": best, "note": "street PDF simulator; see scripts/explore_street_pdf_model.py"},
                ROOT / "data" / "processed" / "models" / "street_pdf_sim.joblib")

    # LightGBM on 2025 and its own sampled spread (same regimes, learned on 2024)
    w = attach_weather(df[te_m | pdf_m].reset_index(drop=True), load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    bundle = joblib.load(ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib")
    lg = np.concatenate([predict(bundle, w.iloc[i:i + 50000]) for i in range(0, len(w), 50000)])
    del w
    lgb = np.full(len(df), np.nan)
    lgb[te_m | pdf_m] = lg
    reg_l = regime(df, lgb, np.quantile(lgb[pdf_m], np.linspace(0, 1, N_DIST + 1)))
    pools_l = {rg: y[pdf_m & (reg_l == rg)] / lgb[pdf_m & (reg_l == rg)] for rg in np.unique(reg_l[pdf_m])}

    rng = np.random.default_rng(0)
    T = np.flatnonzero(te_m)
    yt = y[T]
    sim = sample(mean_t[T], reg[T], pools, rng)
    sim_l = sample(lgb[T], reg_l[T], {k: pools_l.get(k, y[pdf_m] / lgb[pdf_m]) for k in np.unique(reg_l[T])}, rng)
    placement = ((df["pump_order"] == 1) & (df["deployed_from_location"] == "Home Station") & (df["busy_flag"] == 0)).to_numpy()[T]

    summary = {"doc": __doc__, "ridge_tuning_mae_s": tune, "ridge_chosen": best, "streets_with_own_offset": n_edges_used,
               "street_multiplier_spread": {"p5": round(float(np.percentile(u[u != 0] / np.maximum(te_[u != 0], 1e-3), 5)), 3),
                                            "p95": round(float(np.percentile(u[u != 0] / np.maximum(te_[u != 0], 1e-3), 95)), 3)}}
    prow = []
    for name, sl in (("all_2025_trips", np.ones(len(T), bool)), ("placement_trips", placement)):
        ys, S, SL = yt[sl], sim[sl], sim_l[sl]
        med, medl = np.median(S, 1), np.median(SL, 1)
        cov = lambda D, lo, hi: round(float(((ys >= np.percentile(D, lo, 1)) & (ys <= np.percentile(D, hi, 1))).mean()), 3)  # noqa: E731
        one, one_l = S[:, 0], SL[:, 0]  # one simulated time per trip, for the distribution comparison
        summary[name] = {
            "trips": int(sl.sum()), "mean_recorded_s": round(float(ys.mean()), 1),
            "point_mae_s": {"lightgbm": round(float(np.abs(lgb[T][sl] - ys).mean()), 1),
                            "street_speeds_alone": round(float(np.abs(base_c[T][sl] - ys).mean()), 1),
                            "street_speeds_with_own_street_offsets": round(float(np.abs(mean_t[T][sl] - ys).mean()), 1),
                            "street_pdf_median_of_draws": round(float(np.abs(med - ys).mean()), 1),
                            "lightgbm_plus_spread_median_of_draws": round(float(np.abs(medl - ys).mean()), 1)},
            "distribution_vs_recorded": {"recorded": dist_stats(ys, ys), "lightgbm_point": dist_stats(lgb[T][sl], ys),
                                         "street_pdf_sim": dist_stats(one, ys), "lightgbm_plus_spread_sim": dist_stats(one_l, ys)},
            "crps_s": {"street_pdf": round(crps(S, ys), 1), "lightgbm_plus_spread": round(crps(SL, ys), 1)},
            "coverage_50_80": {"street_pdf": [cov(S, 25, 75), cov(S, 10, 90)], "lightgbm_plus_spread": [cov(SL, 25, 75), cov(SL, 10, 90)]}}
        for k, v in (("recorded", ys), ("lightgbm_point", lgb[T][sl]), ("street_pdf_sim", one), ("lightgbm_plus_spread_sim", one_l)):
            prow.append({"trips": name, "series": k, **{f"p{q}": round(float(np.percentile(v, q)), 1) for q in PCTS}})
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    pd.DataFrame(prow).to_csv(OUT / "percentiles.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    np.savez_compressed(OUT / "test_samples.npz", recorded=yt, lightgbm=lgb[T], street_mean=mean_t[T], street_sim=sim[:, 0],
                        lgb_sim=sim_l[:, 0], placement=placement)
    print(json.dumps({k: v for k, v in summary.items() if k != "doc"}, indent=2, default=str))
    print(pd.DataFrame(prow).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
