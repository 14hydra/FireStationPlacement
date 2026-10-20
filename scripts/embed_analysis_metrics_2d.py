#!/usr/bin/env python3
"""
Every per-station analysis metric in the project, condensed to a 2D picture.

Joins all station-level artifacts under data/figures/ into one feature matrix (one row per
current LFB station), standardises it, and embeds it with PCA, UMAP and t-SNE. Layout-level
metrics (named placement layouts from layout_scoring + robustness + recommended summaries)
cannot join to stations, so they get their own small matrix and PCA.

Station features, by source:
  capacity     data_exploration/station_capacity/station_pumps_vs_demand.csv, two_pump_inference.csv
  busyness     data_exploration/station_busyness/station_busyness.csv, station_busyness_by_hour.csv
  trips        data/processed/lfb_travel_training.csv, 2025 trips by deployed_from_station
               (distance, drive time, speed, turnout, street mix of the route)
  model        saved LightGBM travel-time model (trained 2021-24) scored on the same 2025
               trips: held-out error by origin station, plus the Kolesar curve's error
  area maps    travel_time/maps/travel_time_cells.csv, each cell given to its nearest station
  plans        placement/planner_runs/resize (assigned demand, pump change, turnout),
               placement/exact_runs (demand share lost when 5 sites open),
               placement/recommended (how often a station survives layouts that close sites)

Not joinable to current stations (documented in summary.json): validation/closure_2014*
(the 10 stations closed in 2014 no longer exist), validation/dispatch (city / closure-area
level), placement/tradeoff* and slow_calls (layout level, different layout keys).

Writes data/figures/analysis/station_embedding/:
  station_feature_matrix.csv, station_embedding_2d.csv, pca_loadings.csv, top_features.txt,
  layout_feature_matrix.csv, layout_embedding_2d.csv, summary.json, ANALYSIS.md,
  00_summary.png, 01_pca.png, 02_umap.png, 02b_tsne.png, 03_feature_loadings.png,
  04_layout_pca.png, 05_archetypes.png

Run:  .venv/bin/python scripts/embed_analysis_metrics_2d.py [--refresh-trips]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIG = ROOT / "data" / "figures"
PROC = ROOT / "data" / "processed"
OUT = FIG / "analysis" / "station_embedding"
TRIP_CACHE = OUT / "station_trip_metrics_2025.csv"
EVAL_YEAR = 2025
SEED = 42
NEAR_CONSTANT_MODE_SHARE = 0.95
BASEMAP = PROC / "london_basemap.pkl"

# Reference palette (dataviz skill, light mode), as in plot_station_busyness.py.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE, ORANGE, GRAY = "#256abf", "#eb6834", "#9a9893"
GREEN = "#3f8f5b"
CLUSTER_COLORS = ["#256abf", "#eb6834", "#3f8f5b", "#b5892b", "#7a6f9b", "#52514e"]
# Stable names for k-means groups ordered by mean PC1 (quiet → busy). Filled after embed.
CLUSTER_NAME_TEMPLATE = [
    "Outer long-drive",
    "Two-pump suburban",
    "Busy multi-pump",
    "Strained 1-pump urban",
    "Extreme demand cores",
]

SOURCES = {
    "capacity": ["n_pumps", "n_incidents", "busy_rate", "nearest_station_km", "incidents_per_nn_km",
                 "incidents_per_pump", "named_two_pump", "n_multi_dispatch", "multi_dispatch_rate", "n_home_trips"],
    "busyness": ["calls_per_day", "engine_hours_per_day", "median_job_min", "utilisation_pct", "all_out_pct",
                 "covered_by_other_pct", "util_night_pct", "util_peak_pct", "util_peak_to_trough"],
    "trips": ["trips_2025", "drive_median_s", "drive_p90_s", "road_km_median", "speed_kph_median",
              "turnout_median_s", "busy_trip_pct", "first_pump_pct", "outside_ground_pct", "share_major_mean",
              "share_residential_mean", "share_20mph_mean", "signals_per_km_mean", "junctions_per_km_mean"],
    "model": ["lgbm_mae_s", "lgbm_bias_s", "kolesar_mae_s", "lgbm_gain_vs_kolesar_s"],
    "area maps": ["area_first_engine_min", "area_rush_slowdown_pct", "area_traffic_delay_pct", "area_n_cells"],
    "plans": ["plan_assigned_share", "resize_delta_pumps", "plan_turnout_s", "keep_rate_closing_layouts",
              "share_lost_expand5_pct"],
}
SOURCE_OF = {c: s for s, cols in SOURCES.items() for c in cols}

LABELS = {
    "n_pumps": "pumps", "n_incidents": "incidents (2024-25)", "busy_rate": "busy rate",
    "nearest_station_km": "km to nearest station", "incidents_per_nn_km": "incidents per km to neighbour",
    "incidents_per_pump": "incidents per pump", "named_two_pump": "named two-pump station",
    "n_multi_dispatch": "multi-pump dispatches", "multi_dispatch_rate": "multi-pump dispatch rate",
    "n_home_trips": "home-station trips", "calls_per_day": "calls per day",
    "engine_hours_per_day": "engine hours per day", "median_job_min": "median job length",
    "utilisation_pct": "utilisation %", "all_out_pct": "% of year all engines out",
    "covered_by_other_pct": "% own-area calls covered by others", "util_night_pct": "utilisation 00-06",
    "util_peak_pct": "utilisation at peak hour", "util_peak_to_trough": "peak / trough utilisation",
    "trips_2025": "trips 2025", "drive_median_s": "median drive time", "drive_p90_s": "90th pct drive time",
    "road_km_median": "median road km", "speed_kph_median": "median speed", "turnout_median_s": "median turnout",
    "busy_trip_pct": "% trips flagged busy", "first_pump_pct": "% trips as first pump",
    "outside_ground_pct": "% trips outside own ground", "share_major_mean": "route share on major roads",
    "share_residential_mean": "route share residential", "share_20mph_mean": "route share 20 mph",
    "signals_per_km_mean": "signals per km", "junctions_per_km_mean": "junctions per km",
    "lgbm_mae_s": "LightGBM MAE (2025)", "lgbm_bias_s": "LightGBM bias (2025)",
    "kolesar_mae_s": "Kolesar MAE (2025)", "lgbm_gain_vs_kolesar_s": "LightGBM gain over Kolesar",
    "area_first_engine_min": "area first-engine time", "area_rush_slowdown_pct": "area rush slowdown %",
    "area_traffic_delay_pct": "area traffic delay %", "area_n_cells": "map cells nearest",
    "plan_assigned_share": "planned demand share", "resize_delta_pumps": "resize pump change",
    "plan_turnout_s": "plan turnout", "keep_rate_closing_layouts": "kept in closing layouts",
    "share_lost_expand5_pct": "% demand lost to 5 new sites",
}


# --------------------------------------------------------------------------- #
# Station features
# --------------------------------------------------------------------------- #


def capacity_features() -> pd.DataFrame:
    d = FIG / "data_exploration" / "station_capacity"
    cap = pd.read_csv(d / "station_pumps_vs_demand.csv").set_index("station")
    t = cap[["lat", "lon", "n_pumps", "n_incidents", "busy_rate", "nearest_station_km", "incidents_per_nn_km",
             "incidents_per_pump"]].copy()
    t["named_two_pump"] = cap["named_two_pump"].astype(int)
    tp = pd.read_csv(d / "two_pump_inference.csv").set_index("station")
    t["n_multi_dispatch"] = tp["n_multi_dispatch"]
    t["multi_dispatch_rate"] = tp["rate"]
    t["n_home_trips"] = tp["n_home"]
    return t


def busyness_features() -> pd.DataFrame:
    d = FIG / "data_exploration" / "station_busyness"
    b = pd.read_csv(d / "station_busyness.csv").set_index("station")
    t = b[["calls_per_day", "engine_hours_per_day", "median_job_min", "utilisation_pct", "all_out_pct",
           "covered_by_other_pct"]].copy()
    h = pd.read_csv(d / "station_busyness_by_hour.csv").set_index("station")
    h = h[[c for c in h.columns if c.isdigit()]].astype(float)
    t["util_night_pct"] = h[[str(i) for i in range(0, 7)]].mean(axis=1)
    t["util_peak_pct"] = h.max(axis=1)
    t["util_peak_to_trough"] = h.max(axis=1) / h.min(axis=1).clip(lower=0.1)
    return t


def trip_features(refresh: bool) -> pd.DataFrame:
    """Per-origin-station trip stats and held-out travel-time model error, 2025 trips."""
    if TRIP_CACHE.exists() and not refresh:
        return pd.read_csv(TRIP_CACHE, index_col="station")
    import joblib

    from firestation.kolesar import KolesarModel
    from firestation.travel_time_model import DRIVE_CLIP_S, attach_weather, build_features, load_weather, predict_bag

    bundle = joblib.load(PROC / "models" / "travel_time_lfb.joblib")
    route_cols = list(bundle["street_features"])
    cols = ["cal_year", "date_of_call", "hour_of_call", "borough", "station_ground", "deployed_from_station",
            "deployed_from_location", "pump_order", "busy_flag", "dest_exact", "crow_km", "road_km", "turnout_s",
            "drive_s"] + route_cols
    print("loading 2025 trips…", flush=True)
    df = pd.read_csv(PROC / "lfb_travel_training.csv", usecols=lambda c: c in set(cols), low_memory=False)
    df = df[(df["cal_year"] == bundle["test_year"]) & df["drive_s"].between(*DRIVE_CLIP_S)].reset_index(drop=True)

    print(f"scoring {len(df):,} trips with the saved LightGBM bundle (trained {bundle['train_years']})…", flush=True)
    groups = ["base", "street"] + (["weather"] if bundle.get("weather_features") else [])
    trips = attach_weather(df, load_weather(PROC / "weather_hourly_london.csv")) if "weather" in groups else df
    kol = KolesarModel.from_dict(bundle["kolesar_baseline_params"])
    x = build_features(trips, groups, kolesar=kol, cat_maps={"borough": list(bundle["borough_categories"]),
                                                              "incident_group": [], "property_category": []})
    df["pred_lgbm"] = predict_bag(bundle["models"], x[bundle["features"]])
    df["pred_kol"] = np.clip(kol.predict(df["road_km"].to_numpy(float)), *DRIVE_CLIP_S)
    df["err_lgbm"] = df["pred_lgbm"] - df["drive_s"]
    df["speed_kph"] = df["road_km"] / (df["drive_s"] / 3600)

    g = df.groupby("deployed_from_station")
    t = pd.DataFrame({
        "trips_2025": g.size(),
        "drive_median_s": g["drive_s"].median(),
        "drive_p90_s": g["drive_s"].quantile(0.9),
        "road_km_median": g["road_km"].median(),
        "speed_kph_median": g["speed_kph"].median(),
        "turnout_median_s": g["turnout_s"].median(),
        "busy_trip_pct": 100 * g["busy_flag"].mean(),
        "first_pump_pct": 100 * g["pump_order"].apply(lambda s: (s == 1).mean()),
        "outside_ground_pct": 100 * (df["station_ground"] != df["deployed_from_station"]).groupby(
            df["deployed_from_station"]).mean(),
        "share_major_mean": g["share_major"].mean(),
        "share_residential_mean": g["share_residential"].mean(),
        "share_20mph_mean": g["share_20mph"].mean(),
        "signals_per_km_mean": g["signals_per_km"].mean(),
        "junctions_per_km_mean": g["junctions_per_km"].mean(),
        "lgbm_mae_s": g["err_lgbm"].apply(lambda e: e.abs().mean()),
        "lgbm_bias_s": g["err_lgbm"].mean(),
        "kolesar_mae_s": (df["pred_kol"] - df["drive_s"]).abs().groupby(df["deployed_from_station"]).mean(),
    })
    t["lgbm_gain_vs_kolesar_s"] = t["kolesar_mae_s"] - t["lgbm_mae_s"]
    t.index.name = "station"
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(TRIP_CACHE)
    return t


def area_map_features(stations: pd.DataFrame) -> pd.DataFrame:
    """Travel-time map cells averaged over the cells whose nearest station is this one."""
    from firestation.facility import haversine_km

    cells = pd.read_csv(FIG / "travel_time" / "maps" / "travel_time_cells.csv")
    st = stations[["lat", "lon"]]
    d = haversine_km(cells["lon"].to_numpy()[:, None], cells["lat"].to_numpy()[:, None],
                     st["lon"].to_numpy()[None, :], st["lat"].to_numpy()[None, :])
    cells["station"] = st.index.to_numpy()[np.argmin(d, axis=1)]
    p = cells.pivot_table(index="station", columns="map", values="value", aggfunc="mean")
    out = pd.DataFrame(index=st.index)
    out["area_first_engine_min"] = p.get("A_first_engine_min")
    out["area_rush_slowdown_pct"] = p.get("B_rush_slowdown_pct")
    out["area_traffic_delay_pct"] = p.get("C_traffic_delay_pct")
    out["area_n_cells"] = cells[cells["map"] == "A_first_engine_min"].groupby("station").size()
    return out


def plan_features(stations: pd.Index) -> pd.DataFrame:
    P = FIG / "placement"
    out = pd.DataFrame(index=stations)

    def existing(path: Path) -> pd.DataFrame:
        d = pd.read_csv(path)
        return d[d["is_existing"].astype(bool)].set_index("site_name")

    rz = existing(P / "planner_runs" / "resize" / "planned_fire_stations.csv")
    out["plan_assigned_share"] = rz["assigned_share"]
    out["resize_delta_pumps"] = rz["delta_pumps"]
    out["plan_turnout_s"] = rz["turnout_s"]

    e5 = existing(P / "exact_runs" / "expand_5" / "planned_fire_stations.csv")
    out["share_lost_expand5_pct"] = 100 * (1 - e5["assigned_share"] / rz["assigned_share"].replace(0, np.nan))

    kept, n = pd.Series(0.0, index=stations), 0
    for path in sorted((P / "recommended").glob("*/planned_fire_stations.csv")):
        names = set(existing(path).index)
        if len(names) < len(stations):
            kept += stations.isin(names).astype(float)
            n += 1
    out["keep_rate_closing_layouts"] = kept / max(n, 1)
    out.attrs["n_closing_layouts"] = n
    return out


def station_matrix(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    cap = capacity_features()
    t = cap.join(busyness_features()).join(trip_features(refresh)).join(area_map_features(cap))
    plans = plan_features(cap.index)
    t = t.join(plans)
    t.attrs["n_closing_layouts"] = plans.attrs["n_closing_layouts"]
    t.index.name = "station"
    feats = [c for c in SOURCE_OF if c in t.columns]
    return t[["lat", "lon"]], t[feats]


# --------------------------------------------------------------------------- #
# Layout features
# --------------------------------------------------------------------------- #


def layout_matrix() -> pd.DataFrame:
    P = FIG / "placement"
    s = pd.read_csv(P / "layout_scoring" / "scores.csv")
    s["key"] = s["scorer"] + "/" + s["response"]
    wide = s.pivot_table(index="layout", columns="key", values=["mean_s", "slowest10_s", "within_6min"])
    wide.columns = [f"{m}_{k}" for m, k in wide.columns]

    r = pd.read_csv(P / "robustness" / "robustness.csv")
    nb = r[r["group_type"] != "borough"]
    rob = nb.pivot_table(index="layout", columns="group", values=["d_mean_s", "d_slowest10_s"])
    rob.columns = [f"{m}_{re.sub(r'[^a-z0-9]+', '_', g.lower()).strip('_')}" for m, g in rob.columns]
    bor = r[r["group_type"] == "borough"].groupby("layout")["d_mean_s"]
    rob["boroughs_worse_1s"] = bor.apply(lambda x: (x >= 1).sum())
    rob["borough_d_mean_max"] = bor.max()
    rob["borough_d_mean_std"] = bor.std()

    t = wide.join(rob)
    t.loc["today", rob.columns] = 0.0

    def counts(name: str) -> tuple[float, float]:
        m = re.search(r"(expand|replace)_(\d+)", name)
        if name == "today":
            return 0.0, 0.0
        if m:
            k = float(m.group(2))
            return (k, 0.0) if m.group(1) == "expand" else (k, k)
        return np.nan, np.nan

    rec = {}
    for p in (P / "recommended").glob("*/summary.json"):
        j = json.load(open(p))
        rec["lgbm_" + p.parent.name] = j
    t["n_opened"], t["n_closed"] = zip(*[counts(i) for i in t.index])
    for lay, j in rec.items():
        if lay in t.index:
            t.loc[lay, "n_opened"] = len(j.get("opened", []))
            t.loc[lay, "n_closed"] = len(j.get("closed", []))
            t.loc[lay, "efficiency_score"] = j.get("efficiency_score")
            t.loc[lay, "equity_score"] = j.get("equity_score")
    t.index.name = "layout"
    return t


# --------------------------------------------------------------------------- #
# Standardise + embed
# --------------------------------------------------------------------------- #


def standardise(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.fillna(X.median())
    dropped = []
    for c in X.columns:
        mode_share = X[c].round(9).value_counts(normalize=True).iloc[0]
        if X[c].std() < 1e-9 or mode_share >= NEAR_CONSTANT_MODE_SHARE:
            dropped.append(c)
    X = X.drop(columns=dropped)
    corr = X.corr().abs().to_numpy()
    dupes = [c for j, c in enumerate(X.columns) if (corr[:j, j] > 0.999).any()]
    dropped += [f"{c} (duplicate)" for c in dupes]
    X = X.drop(columns=dupes)
    Z = (X - X.mean()) / X.std(ddof=0)
    return Z, dropped


def pca_fit(Z: pd.DataFrame, n: int):
    from sklearn.decomposition import PCA

    p = PCA(n_components=min(n, *Z.shape), random_state=SEED).fit(Z.to_numpy())
    return p, p.transform(Z.to_numpy())


def embed(Z: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import umap
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    pca, pcs = pca_fit(Z, 10)
    E = pd.DataFrame(index=Z.index)
    E["pca1"], E["pca2"] = pcs[:, 0], pcs[:, 1]
    u = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=SEED).fit_transform(Z.to_numpy())
    E["umap1"], E["umap2"] = u[:, 0], u[:, 1]
    ts = TSNE(n_components=2, perplexity=20, init="pca", random_state=SEED).fit_transform(Z.to_numpy())
    E["tsne1"], E["tsne2"] = ts[:, 0], ts[:, 1]

    # Clusters on the PCs that carry 80% of the variance (less noise than all features).
    k_pc = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.8) + 1)
    space = pcs[:, :k_pc]
    sil = {}
    for k in range(3, 7):
        lab = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(space)
        sil[k] = float(silhouette_score(space, lab))
    best_k = max(sil, key=sil.get)
    lab = KMeans(n_clusters=best_k, n_init=20, random_state=SEED).fit_predict(space)
    # Order clusters by mean PC1 so colours are stable and meaningful.
    order = {c: i for i, c in enumerate(pd.Series(pcs[:, 0]).groupby(lab).mean().sort_values().index)}
    E["cluster"] = [order[c] for c in lab]
    meta = {"pca": pca, "k_pcs_for_clusters": k_pc, "silhouette_by_k": sil, "k": best_k}
    return E, meta


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif",
    })


# Compact heatmap tick labels (vertical ticks need short strings).
HEAT_SHORT = {
    "plan_assigned_share": "plan demand",
    "incidents_per_nn_km": "inc / nn-km",
    "n_incidents": "incidents",
    "incidents_per_pump": "inc / pump",
    "share_lost_expand5_pct": "% lost to 5 sites",
    "signals_per_km_mean": "signals / km",
    "n_home_trips": "home trips",
    "util_night_pct": "util 00–06",
    "trips_2025": "trips 2025",
    "calls_per_day": "calls / day",
    "utilisation_pct": "utilisation %",
    "resize_delta_pumps": "resize Δ pumps",
    "engine_hours_per_day": "engine h / day",
    "util_peak_pct": "util peak",
    "n_multi_dispatch": "multi-dispatch",
    "multi_dispatch_rate": "multi-disp. rate",
    "junctions_per_km_mean": "junctions / km",
    "drive_median_s": "median drive",
    "lgbm_mae_s": "LGBM MAE",
}


def _short_label(name: str, max_len: int = 18) -> str:
    s = HEAT_SHORT.get(name) or LABELS.get(name, name)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _pick_extremes(E: pd.DataFrame, x: str, y: str, n: int = 6) -> list[str]:
    """A few stations that define the axes / corners — not a crowd of near-duplicates."""
    xy = E[[x, y]].to_numpy()
    med = np.median(xy, axis=0)
    std = xy.std(axis=0) + 1e-9
    zn = (xy - med) / std
    picks: list[str] = []
    # True corners of the cloud
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        score = sx * zn[:, 0] + sy * zn[:, 1]
        picks.append(E.index[int(np.argmax(score))])
    # Farthest from centre for any remaining slots
    r = np.hypot(zn[:, 0], zn[:, 1])
    for i in np.argsort(-r):
        if len(picks) >= n:
            break
        name = E.index[int(i)]
        if name not in picks:
            picks.append(name)
    # Drop near-duplicates in data space so labels don't pile up
    kept: list[str] = []
    for name in picks:
        p = E.loc[name, [x, y]].to_numpy(dtype=float)
        if any(np.hypot(*(p - E.loc[k, [x, y]].to_numpy(dtype=float)) / std) < 0.55 for k in kept):
            continue
        kept.append(name)
        if len(kept) >= n:
            break
    return kept


def _annotate_sparse(ax, E: pd.DataFrame, x: str, y: str, n: int = 6):
    """Callouts with short leader lines; kept inside the axes so they never hit tick labels."""
    from matplotlib import patheffects

    names = _pick_extremes(E, x, y, n=n)
    if not names:
        return
    xs, ys = E[x].to_numpy(dtype=float), E[y].to_numpy(dtype=float)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    dx, dy = (x1 - x0) or 1.0, (y1 - y0) or 1.0
    # Modest pad — labels stay inside these limits
    pad_x, pad_y = 0.08 * dx, 0.10 * dy
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # Keep text away from spines (room for axis titles / ticks)
    margin_x, margin_y = 0.10 * (xlim[1] - xlim[0]), 0.08 * (ylim[1] - ylim[0])
    cx, cy = float(np.median(xs)), float(np.median(ys))
    halo = [patheffects.withStroke(linewidth=3.4, foreground=SURFACE)]
    placed_xy: list[tuple[float, float]] = []

    def _clamp(tx, ty):
        return (
            float(np.clip(tx, xlim[0] + margin_x, xlim[1] - margin_x)),
            float(np.clip(ty, ylim[0] + margin_y, ylim[1] - margin_y)),
        )

    for name in names:
        px, py = float(E.loc[name, x]), float(E.loc[name, y])
        vx, vy = px - cx, py - cy
        norm = (vx ** 2 + vy ** 2) ** 0.5 or 1.0
        # Prefer offset along the larger radial component; for left-edge points,
        # bias upward/downward so we don't collide with the y-axis title.
        near_left = (px - xlim[0]) < 0.22 * (xlim[1] - xlim[0])
        near_right = (xlim[1] - px) < 0.22 * (xlim[1] - xlim[0])
        if near_left:
            ox, oy = 0.04 * dx, (0.08 * dy if vy >= 0 else -0.08 * dy)
        elif near_right:
            ox, oy = -0.04 * dx, (0.07 * dy if abs(vy) > 1e-9 else 0.06 * dy)
        else:
            ox = 0.055 * dx * (vx / norm)
            oy = 0.055 * dy * (vy / norm)
        tx, ty = _clamp(px + ox, py + oy)
        for _ in range(10):
            hit = False
            for qx, qy in placed_xy:
                if abs(tx - qx) < 0.16 * dx and abs(ty - qy) < 0.11 * dy:
                    ty += 0.07 * dy * (1 if ty >= cy else -1)
                    tx, ty = _clamp(tx, ty)
                    hit = True
                    break
            if not hit:
                break
        placed_xy.append((tx, ty))
        ax.annotate(
            name, (px, py), xytext=(tx, ty),
            textcoords="data", fontsize=8, color=INK, fontweight="medium",
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.65, shrinkA=2, shrinkB=4),
            path_effects=halo, zorder=5,
        )


def _pca_crosshairs(ax, x0=0.0, y0=0.0):
    ax.axhline(y0, color=GRID, lw=1.0, zorder=0)
    ax.axvline(x0, color=GRID, lw=1.0, zorder=0)


def _draw_basemap(ax, loc: pd.DataFrame):
    """Major roads + Thames; builds the cache on first use via plot_station_busyness."""
    from matplotlib.collections import LineCollection

    try:
        from plot_station_busyness import load_basemap
        bm = load_basemap()
        ax.add_collection(LineCollection(bm["roads"], colors="#e4e3df", linewidths=0.55, zorder=1))
        ax.add_collection(LineCollection(bm["river"], colors="#b9d6ee", linewidths=3.2, zorder=1,
                                         capstyle="round"))
    except Exception:
        pass  # points-only map is fine if OSM graph is unavailable
    pad = 0.025
    ax.set_xlim(loc["lon"].min() - pad, loc["lon"].max() + pad)
    ax.set_ylim(loc["lat"].min() - pad, loc["lat"].max() + pad)
    ax.set_aspect(1 / np.cos(np.deg2rad(51.5)))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _scatter_pumps(ax, E, F, x, y):
    for pumps, color in [(1, GRAY), (2, ORANGE)]:
        m = F["n_pumps"] == pumps
        ax.scatter(E.loc[m, x], E.loc[m, y], s=42, c=color, edgecolor="white", linewidth=0.7,
                   label=f"{pumps} pump{'s' if pumps > 1 else ''}  ({int(m.sum())})", zorder=2)


def _scatter_seq(fig, ax, E, F, x, y, col):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    sc = ax.scatter(E[x], E[y], s=42, c=F[col], cmap=cmap, edgecolor="white", linewidth=0.7, zorder=2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label(LABELS.get(col, col), color=INK2, fontsize=8)
    return sc


def plot_pair(E, F, x, y, title, axis_labels, path, *, pca_axes: bool = False):
    style()
    fig = plt.figure(figsize=(14.5, 7.2))
    # Leave a clear band under the figure title so subplot titles never collide
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.97, top=0.80, bottom=0.22, wspace=0.30)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[0, 1])
    _scatter_pumps(a1, E, F, x, y)
    a1.set_title("Coloured by number of pumps", pad=8)
    a1.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _scatter_seq(fig, a2, E, F, x, y, "incidents_per_pump")
    a2.set_title("Coloured by incidents per pump", pad=8)
    short_x = axis_labels[0].split(":")[0].strip() if ":" in axis_labels[0] else axis_labels[0]
    short_y = axis_labels[1].split(":")[0].strip() if ":" in axis_labels[1] else axis_labels[1]
    for a in (a1, a2):
        if pca_axes:
            _pca_crosshairs(a)
        a.set_xlabel(short_x)
        a.set_ylabel(short_y)
        a.yaxis.labelpad = 8
    if pca_axes:
        # Only the two farthest stations — keeps the panel readable
        from matplotlib import patheffects
        r = np.hypot(
            (E[x] - E[x].median()) / (float(E[x].std()) or 1.0),
            (E[y] - E[y].median()) / (float(E[y].std()) or 1.0),
        )
        halo = [patheffects.withStroke(linewidth=3.2, foreground=SURFACE)]
        for name in r.nlargest(2).index:
            px, py = float(E.loc[name, x]), float(E.loc[name, y])
            off = (10, 10) if py >= float(E[y].median()) else (10, -12)
            a1.annotate(name, (px, py), xytext=off, textcoords="offset points",
                        fontsize=8.5, color=INK, fontweight="medium", path_effects=halo,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    fig.text(0.08, 0.955, title, fontsize=14, fontweight="bold", color=INK, ha="left", va="top")
    if pca_axes:
        # One compact loadings line — not stacked under the title
        fig.text(0.08, 0.045,
                 f"{axis_labels[0]}   ·   {axis_labels[1]}",
                 fontsize=7.5, color=INK2, ha="left", va="bottom")
        fig.text(0.08, 0.018,
                 "← quiet / long drives · busy / dense → on PC1"
                 "      ↑ multi-pump capacity · 1-pump strain ↓ on PC2",
                 fontsize=8, color=INK2, ha="left", va="bottom")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _axis_name(load: pd.Series, k: int = 3) -> str:
    top = load.abs().sort_values(ascending=False).index[:k]
    return ", ".join(("+" if load[c] > 0 else "−") + LABELS.get(c, c) for c in top)


def plot_loadings(pca, cols, path):
    style()
    L = pd.DataFrame(pca.components_[:2].T, index=cols, columns=["PC1", "PC2"])
    fig = plt.figure(figsize=(14, 8), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[3, 3, 2])
    for j, pc in enumerate(["PC1", "PC2"]):
        ax = fig.add_subplot(gs[:, j])
        top = L[pc].reindex(L[pc].abs().sort_values().index[-16:])
        ax.barh([LABELS.get(c, c) for c in top.index], top, color=[BLUE if v > 0 else ORANGE for v in top],
                height=0.7)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f"{pc} loadings ({100 * pca.explained_variance_ratio_[j]:.0f}% of variance)")
        ax.set_xlabel("loading (blue = higher score, orange = lower)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
    ax = fig.add_subplot(gs[0, 2])
    ev = 100 * pca.explained_variance_ratio_
    ax.bar(np.arange(1, len(ev) + 1), ev, color=SEQ[3], width=0.7)
    ax.plot(np.arange(1, len(ev) + 1), np.cumsum(ev), color=INK2, marker="o", ms=3, lw=1)
    ax.set_title("Variance explained")
    ax.set_xlabel("principal component")
    ax.set_ylabel("% (bars) / cumulative % (line)")
    ax.set_xticks(np.arange(1, len(ev) + 1))
    ax = fig.add_subplot(gs[1, 2])
    src = pd.DataFrame({"src": [SOURCE_OF[c] for c in cols], "PC1": L["PC1"] ** 2, "PC2": L["PC2"] ** 2})
    src = src.groupby("src")[["PC1", "PC2"]].sum().mul(100).sort_values("PC1")
    yy = np.arange(len(src))
    ax.barh(yy - 0.18, src["PC1"], height=0.36, color=BLUE, label="PC1")
    ax.barh(yy + 0.18, src["PC2"], height=0.36, color=ORANGE, label="PC2")
    ax.set_yticks(yy, src.index, fontsize=8)
    ax.set_xlabel("% of axis (squared loadings)")
    ax.set_title("Share of each data source")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.suptitle("What the station PCA axes mean", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cluster_names(k: int, E: pd.DataFrame | None = None, F: pd.DataFrame | None = None) -> list[str]:
    """Name clusters quiet→busy (PC1 order). Prefer profile-based labels when F is given."""
    if E is None or F is None or "cluster" not in E.columns:
        names = list(CLUSTER_NAME_TEMPLATE[:k])
        while len(names) < k:
            names.append(f"Group {len(names) + 1}")
        return names
    names = []
    for c in range(k):
        m = E["cluster"] == c
        g = F.loc[m]
        n = int(m.sum())
        mean_pc1 = float(E.loc[m, "pca1"].mean())
        mean_pumps = float(g["n_pumps"].mean())
        mean_util = float(g["utilisation_pct"].mean())
        # Tiny far-right clusters are central demand outliers (e.g. Soho / Euston).
        if n <= 3 and mean_pc1 > E["pca1"].quantile(0.85):
            names.append("Central demand outliers")
        elif mean_pc1 < E["pca1"].quantile(0.35):
            names.append("Outer two-pump" if mean_pumps >= 1.6 else "Outer long-drive")
        elif mean_pc1 > E["pca1"].quantile(0.70):
            if mean_util >= F["utilisation_pct"].median() and mean_pumps < 1.5:
                names.append("Strained 1-pump urban")
            else:
                names.append("Busy multi-pump")
        elif mean_pumps >= 1.6:
            names.append("Outer two-pump" if mean_pc1 < E["pca1"].median() else "Busy multi-pump")
        elif mean_util >= F["utilisation_pct"].quantile(0.60):
            names.append("Strained 1-pump urban")
        else:
            names.append("Mid-load single-pump")
    # De-dupe if profiles collide
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n} ({seen[n]})")
        else:
            seen[n] = 1
            out.append(n)
    return out


def plot_summary(E, F, Z, loc, pca, meta, path):
    style()
    k = meta["k"]
    colors = CLUSTER_COLORS[:k]
    names = cluster_names(k, E, F)
    ev = 100 * pca.explained_variance_ratio_
    # Extra room for shared legend + vertical heatmap tick labels
    fig = plt.figure(figsize=(17.5, 13.2))
    gs = fig.add_gridspec(
        3, 3, width_ratios=[1, 1, 1.05], height_ratios=[1.05, 0.12, 1.15],
        left=0.11, right=0.96, top=0.91, bottom=0.05, hspace=0.24, wspace=0.24,
    )

    # --- PCA: colour only — station names live on 01_pca / 05_archetypes ---
    ax = fig.add_subplot(gs[0, 0])
    handles = []
    for c in range(k):
        m = E["cluster"] == c
        h = ax.scatter(E.loc[m, "pca1"], E.loc[m, "pca2"], s=38, c=colors[c], edgecolor="white",
                       linewidth=0.55, label=f"{names[c]}  ({int(m.sum())})", zorder=2)
        handles.append(h)
    _pca_crosshairs(ax)
    ax.set_xlabel(f"PC1 · demand / density ({ev[0]:.0f}%)")
    ax.set_ylabel(f"PC2 · capacity vs strain ({ev[1]:.0f}%)")
    ax.yaxis.labelpad = 8
    ax.set_title(f"PCA · {Z.shape[1]} metrics → 2D")

    # --- UMAP ---
    ax = fig.add_subplot(gs[0, 1])
    for c in range(k):
        m = E["cluster"] == c
        ax.scatter(E.loc[m, "umap1"], E.loc[m, "umap2"], s=38, c=colors[c], edgecolor="white",
                   linewidth=0.55, zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"UMAP · {Z.shape[1]} metrics → 2D")

    # --- Map ---
    ax = fig.add_subplot(gs[0, 2])
    _draw_basemap(ax, loc)
    for c in range(k):
        m = E["cluster"] == c
        ax.scatter(loc.loc[m, "lon"], loc.loc[m, "lat"], s=36, c=colors[c], edgecolor="white",
                   linewidth=0.55, zorder=3)
    ax.set_title("Same groups on the city map")

    # --- Shared legend row ---
    leg_ax = fig.add_subplot(gs[1, :])
    leg_ax.axis("off")
    leg = leg_ax.legend(
        handles, [f"{names[c]}  ({int((E['cluster'] == c).sum())})" for c in range(k)],
        loc="center", ncol=min(k, 5), frameon=False, fontsize=9,
        title="Archetype (ordered quiet → busy on PC1)", title_fontsize=9,
        handletextpad=0.4, columnspacing=1.4,
    )
    leg.get_title().set_color(INK2)

    # --- Heatmap transposed: metrics as rows (horizontal labels never collide) ---
    ax = fig.add_subplot(gs[2, :])
    prof = Z.groupby(E["cluster"]).mean()
    n_feat = 10
    spread = prof.abs().max().sort_values(ascending=False).index[:n_feat]
    mat = prof[spread].T  # metrics × archetypes
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-1.8, vmax=1.8)
    ax.set_yticks(range(len(spread)), [_short_label(c, 20) for c in spread], fontsize=9)
    ax.tick_params(axis="y", pad=6, length=0)
    col_labels = [f"{names[c]}\n(n={int((E['cluster'] == c).sum())})" for c in range(k)]
    ax.set_xticks(range(k), col_labels, fontsize=8.5)
    ax.tick_params(axis="x", pad=4, length=0, labeltop=True, labelbottom=False)
    for t, c in zip(ax.get_xticklabels(), colors):
        t.set_color(c)
        t.set_fontweight("bold")
        t.set_linespacing(1.1)
    for i in range(len(spread)):
        for j in range(k):
            v = float(mat.iloc[i, j])
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(v) >= 0.85 else INK, fontweight="medium")
    ax.grid(False)
    ax.set_title(f"Archetype profiles · mean z-score on the {n_feat} most separating metrics", pad=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015, fraction=0.025)
    cb.outline.set_visible(False)
    cb.set_label("z-score", fontsize=8, color=INK2)
    fig.text(0.12, 0.965, f"London's {len(E)} fire stations · all analysis metrics condensed to 2D",
             fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.12, 0.935,
             "PC1 → busy / dense streets   ·   PC2 ↑ multi-pump capacity / ↓ 1-pump strain",
             fontsize=8.5, color=INK2, ha="left", va="top")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_archetypes(E, F, Z, pca, meta, path):
    """Single-page reading aid: PCA with named archetypes + key contrasts."""
    style()
    k = meta["k"]
    colors = CLUSTER_COLORS[:k]
    names = cluster_names(k, E, F)
    ev = 100 * pca.explained_variance_ratio_
    fig = plt.figure(figsize=(15.5, 9.8))
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1.35, 1], height_ratios=[1.2, 1.05, 0.20],
        left=0.09, right=0.97, top=0.90, bottom=0.06, hspace=0.28, wspace=0.30,
    )

    ax = fig.add_subplot(gs[:2, 0])
    handles = []
    for c in range(k):
        m = E["cluster"] == c
        h = ax.scatter(E.loc[m, "pca1"], E.loc[m, "pca2"], s=52, c=colors[c], edgecolor="white",
                       linewidth=0.7, label=f"{names[c]}  ({int(m.sum())})", zorder=2)
        handles.append(h)
    _pca_crosshairs(ax)
    # Only the two true outliers — names for the rest live in the text panel
    outliers = E.nlargest(2, "pca1").index.tolist()
    from matplotlib import patheffects
    halo = [patheffects.withStroke(linewidth=3.2, foreground=SURFACE)]
    for name in outliers:
        px, py = float(E.loc[name, "pca1"]), float(E.loc[name, "pca2"])
        ax.annotate(name, (px, py), xytext=(8, 8), textcoords="offset points",
                    fontsize=8.5, color=INK, fontweight="medium", path_effects=halo,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xlabel(f"PC1 ({ev[0]:.0f}%)  quiet ←→ busy")
    ax.set_ylabel(f"PC2 ({ev[1]:.0f}%)  strain ←→ capacity")
    ax.yaxis.labelpad = 10
    ax.set_title("Station archetypes in PCA space")

    leg_ax = fig.add_subplot(gs[2, 0])
    leg_ax.axis("off")
    leg_ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="center",
                  ncol=2, columnspacing=1.3, handletextpad=0.4)

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    strained = F[(F["n_pumps"] == 1) & (F["utilisation_pct"] >= F["utilisation_pct"].quantile(0.75))]
    spare = F[(F["n_pumps"] >= 2) & (F["utilisation_pct"] <= F["utilisation_pct"].quantile(0.40))]
    top_s = strained.sort_values("utilisation_pct", ascending=False).index[:6]
    top_p = spare.sort_values("utilisation_pct").index[:6]
    block = (
        f"PCA 2D captures {ev[0] + ev[1]:.0f}% of variance across {Z.shape[1]} features.\n\n"
        f"Upsize · high-util 1-pump ({len(strained)})\n"
        + "  " + ", ".join(top_s) + "\n\n"
        f"Downsize · lower-util 2-pump ({len(spare)})\n"
        + "  " + ", ".join(top_p) + "\n\n"
        "Opposite PCA quadrants: bottom-right vs top-left.\n"
        "Fig 03 loadings — util columns dominate PC1."
    )
    ax.text(0.0, 0.98, "What to notice", transform=ax.transAxes, va="top",
            fontsize=12, fontweight="bold", color=INK)
    ax.text(0.0, 0.86, block, transform=ax.transAxes, va="top", fontsize=9.5,
            color=INK, linespacing=1.55)

    # Horizontal bars — long archetype names never collide
    ax = fig.add_subplot(gs[1, 1])
    g = F.copy()
    g["cluster"] = E["cluster"].to_numpy()
    order = list(range(k))
    means = g.groupby("cluster")[["incidents_per_pump", "utilisation_pct"]].mean().reindex(order)
    y = np.arange(k)
    h = 0.36
    ax.barh(y + h / 2, means["incidents_per_pump"] / means["incidents_per_pump"].max(), h,
            color=BLUE, label="incidents / pump")
    ax.barh(y - h / 2, means["utilisation_pct"] / means["utilisation_pct"].max(), h,
            color=ORANGE, label="utilisation %")
    ax.set_yticks(y, [f"{names[c]}  ({int((E['cluster'] == c).sum())})" for c in order], fontsize=8)
    ax.set_xlabel("scaled to panel max")
    ax.set_title("Load pressure by archetype", pad=18)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.02),
              ncol=2, borderaxespad=0)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()
    fig.text(0.09, 0.96, "Reading the embedding · where capacity and demand pull apart",
             fontsize=14, fontweight="bold", color=INK, ha="left", va="top")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_analysis_md(E, F, Z, L, ev, meta, names, path: Path):
    """Human-readable analysis of the embedding figures (also mirrored under docs/)."""
    k = meta["k"]
    sil = meta["silhouette_by_k"][meta["k"]]
    strained = F[(F["n_pumps"] == 1) & (F["utilisation_pct"] >= F["utilisation_pct"].quantile(0.75))]
    spare = F[(F["n_pumps"] >= 2) & (F["utilisation_pct"] <= F["utilisation_pct"].quantile(0.40))]
    # Per-archetype one-liners from profiles
    blurbs = []
    for c in range(k):
        m = E["cluster"] == c
        g = F.loc[m]
        blurbs.append(
            f"- **{names[c]}** (n={int(m.sum())}): "
            f"mean util {g['utilisation_pct'].mean():.0f}%, "
            f"mean pumps {g['n_pumps'].mean():.1f}, "
            f"median drive {g['drive_median_s'].mean():.0f}s · "
            f"e.g. {', '.join(E.index[m][:5])}"
        )
    lines = [
        "# Station metric embedding — analysis",
        "",
        f"{len(E)} current LFB stations · **{Z.shape[1]} standardised metrics** · PCA / UMAP / t-SNE.",
        "Figures: `data/figures/analysis/station_embedding/`.",
        "",
        "## How to read the figures",
        "",
        "| File | What it shows |",
        "|---|---|",
        "| `00_summary.png` | PCA + UMAP + map of archetypes, plus a z-score profile heatmap |",
        "| `01_pca.png` | Same PCA coloured by pumps and by incidents/pump (actionable axes) |",
        "| `02_umap.png` / `02b_tsne.png` | Local neighbourhood views (axes have no units) |",
        "| `03_feature_loadings.png` | What PC1/PC2 actually load on; share by data source |",
        "| `04_layout_pca.png` | Separate PCA of *named placement layouts* (not stations) |",
        "| `05_archetypes.png` | Named groups + capacity-mismatch callouts |",
        "",
        "## What the 2D view is",
        "",
        "Every joinable per-station number in the project (capacity, busyness, 2025 trips,",
        "LightGBM error, area maps, plan flags) is z-scored and projected.",
        f"**PCA PC1+PC2 explain {100 * (ev[0] + ev[1]):.0f}%** of variance "
        f"({100 * ev[0]:.0f}% + {100 * ev[1]:.0f}%). "
        f"k-means on the leading PCs picks **k={meta['k']}** "
        f"(silhouette {sil:.2f}). UMAP/t-SNE are for local neighbourhoods only.",
        "",
        "## Axis reading (PCA)",
        "",
        f"- **PC1 ({100 * ev[0]:.0f}%) — busyness / street density:** "
        f"{_axis_name(L['PC1'], 5)}. Right = high utilisation and dense junctions/signals "
        "(inner London); left = longer median drives, quieter outskirts.",
        f"- **PC2 ({100 * ev[1]:.0f}%) — capacity vs strain:** "
        f"{_axis_name(L['PC2'], 5)}. Up = two-pump / multi-dispatch houses; "
        "down = single-pump stations with high all-out / neighbour-cover rates.",
        f"- **PC3 ({100 * ev[2]:.0f}%) — model hard cases + volume:** "
        f"{_axis_name(L['PC3'], 4)}. Stations where both LightGBM and Kolesar struggle, "
        "plus sheer incident volume.",
        "",
        "The pump-coloured PCA (`01_pca.png`) almost splits horizontally: almost all "
        "2-pump stations sit above PC2 ≈ 0. The incidents/pump colouring darkens on the "
        "**right** — Soho, Euston, Shoreditch are the load extremes.",
        "",
        "## Archetypes",
        "",
        *blurbs,
        "",
        "On the map (`00_summary.png`), outer long-drive / outer two-pump ring London; "
        "strained 1-pump and busy multi-pump fill the inner boroughs; the central outliers "
        "are the densest cores.",
        "",
        "## Capacity mismatches the embedding surfaces",
        "",
        f"- **{len(strained)} high-utilisation 1-pump stations** "
        f"(top quartile util): {', '.join(strained.sort_values('utilisation_pct', ascending=False).index[:10])}.",
        f"- **{len(spare)} lower-utilisation 2-pump stations** "
        f"(bottom 40% util): {', '.join(spare.sort_values('utilisation_pct').index[:10])}.",
        "",
        "These are the stations the **resize** planner tends to upsize / downsize. "
        "They sit in opposite PCA quadrants: bottom-right (busy · 1-pump strain) vs "
        "top-left (quiet · spare capacity).",
        "",
        "## Layout embedding (separate matrix)",
        "",
        "Named placement layouts (recommended / exact / scored) form a second PCA "
        "(`04_layout_pca.png`). Layout PC1 is essentially **seconds saved vs today** — "
        "redesign / large replace at one end, status quo at the other.",
        "",
        "## Honesty",
        "",
        "- Correlated demand / util columns inflate PC1; always check loadings "
        "(`03_feature_loadings.png`).",
        "- Area-map cells are assigned by crow-fly nearest station, not LFB ground.",
        f"- Clusters are a reading aid (silhouette {sil:.2f}), not operational station types.",
        "- 2014 closed stations and city-level tradeoff tables are not in the station matrix.",
        "",
    ]
    path.write_text("\n".join(lines))


def plot_layouts(LE, LF, lpca, path):
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6), layout="constrained", gridspec_kw={"width_ratios": [3, 2]})
    scorer = [i.split("_")[0] if i != "today" else "today" for i in LE.index]
    cmap = {"today": INK, "lgbm": BLUE, "kolesar": ORANGE, "road": "#3f8f5b"}
    for s in cmap:
        m = np.array(scorer) == s
        a1.scatter(LE.loc[m, "pca1"], LE.loc[m, "pca2"], s=60, c=cmap[s], edgecolor="white", linewidth=0.6,
                   label=s, marker="*" if s == "today" else "o")
    for name, (x, y) in LE[["pca1", "pca2"]].iterrows():
        a1.annotate(name.replace("lgbm_", ""), (x, y), xytext=(5, 3), textcoords="offset points", fontsize=7.5,
                    color=INK2)
    ev = 100 * lpca.explained_variance_ratio_
    a1.set_xlabel(f"PC1 ({ev[0]:.0f}%)")
    a1.set_ylabel(f"PC2 ({ev[1]:.0f}%)")
    a1.legend(frameon=False, fontsize=8, title="layout from scorer", title_fontsize=8)
    a1.set_title(f"{len(LE)} placement layouts, PCA of {LF.shape[1]} city-level metrics")
    L = pd.Series(lpca.components_[0], index=LF.columns)
    top = L.reindex(L.abs().sort_values().index[-14:])
    a2.barh(top.index, top, color=[BLUE if v > 0 else ORANGE for v in top], height=0.7)
    a2.axvline(0, color=INK2, lw=0.8)
    a2.tick_params(axis="y", labelsize=7.5)
    a2.grid(axis="y", visible=False)
    a2.set_xlabel("loading (d_* = change vs today in s; positive = slower)")
    a2.set_title("Layout PC1 loadings")
    fig.suptitle("Placement layouts in 2D", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--refresh-trips", action="store_true", help="re-score 2025 trips with the travel-time model")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    loc, F = station_matrix(args.refresh_trips)
    F.to_csv(OUT / "station_feature_matrix.csv")
    Z, dropped = standardise(F)
    E, meta = embed(Z)
    pca = meta["pca"]
    loc.join(E).to_csv(OUT / "station_embedding_2d.csv")

    L = pd.DataFrame(pca.components_.T, index=Z.columns, columns=[f"PC{i + 1}" for i in range(pca.n_components_)])
    L.insert(0, "source", [SOURCE_OF[c] for c in Z.columns])
    L.to_csv(OUT / "pca_loadings.csv")
    ev = pca.explained_variance_ratio_
    lines = []
    for j in range(3):
        col = L[f"PC{j + 1}"]
        lines.append(f"PC{j + 1} ({100 * ev[j]:.1f}% of variance)")
        for c in col.abs().sort_values(ascending=False).index[:8]:
            lines.append(f"  {col[c]:+.3f}  {c}  ({LABELS.get(c, c)}; {SOURCE_OF[c]})")
        lines.append("")
    for c in range(meta["k"]):
        m = E["cluster"] == c
        prof = Z[m].mean().sort_values()
        lines.append(f"group {c + 1} (n={m.sum()}): high {', '.join(prof.index[-4:][::-1])}; "
                     f"low {', '.join(prof.index[:3])}")
        lines.append(f"  e.g. {', '.join(E.index[m][:8])}")
    (OUT / "top_features.txt").write_text("\n".join(lines) + "\n")

    names = cluster_names(meta["k"], E, F)
    E = E.copy()
    E["archetype"] = E["cluster"].map(dict(enumerate(names)))
    # Re-write embedding CSV with archetype labels for downstream use.
    loc.join(E).to_csv(OUT / "station_embedding_2d.csv")

    plot_summary(E, F, Z, loc, pca, meta, OUT / "00_summary.png")
    plot_pair(E, F, "pca1", "pca2", "Stations in PCA space",
              (f"PC1 ({100 * ev[0]:.0f}%): {_axis_name(L['PC1'])}", f"PC2 ({100 * ev[1]:.0f}%): "
               f"{_axis_name(L['PC2'])}"), OUT / "01_pca.png", pca_axes=True)
    plot_pair(E, F, "umap1", "umap2", "Stations in UMAP space (axes have no units)", ("UMAP 1", "UMAP 2"),
              OUT / "02_umap.png")
    plot_pair(E, F, "tsne1", "tsne2", "Stations in t-SNE space (axes have no units)", ("t-SNE 1", "t-SNE 2"),
              OUT / "02b_tsne.png")
    plot_loadings(pca, list(Z.columns), OUT / "03_feature_loadings.png")

    LF = layout_matrix()
    LF.to_csv(OUT / "layout_feature_matrix.csv")
    LZ, ldropped = standardise(LF)
    lpca, lpcs = pca_fit(LZ, 5)
    LE = pd.DataFrame({"pca1": lpcs[:, 0], "pca2": lpcs[:, 1]}, index=LZ.index)
    LE.to_csv(OUT / "layout_embedding_2d.csv")
    plot_layouts(LE, LZ, lpca, OUT / "04_layout_pca.png")
    plot_archetypes(E, F, Z, pca, meta, OUT / "05_archetypes.png")
    write_analysis_md(E, F, Z, L, ev, meta, names, OUT / "ANALYSIS.md")
    # Mirror under docs/ for the project doc tree
    docs_md = ROOT / "docs" / "station_embedding_analysis.md"
    docs_md.write_text(
        (OUT / "ANALYSIS.md").read_text()
        + "\n## How to regenerate\n\n"
        + "```bash\n.venv/bin/python scripts/embed_analysis_metrics_2d.py\n```\n"
        + "\nFigures and CSVs: `data/figures/analysis/station_embedding/`.\n"
    )

    from scipy.stats import spearmanr

    summary = {
        "method": "Per-station features joined by station name; median impute; drop constant or near-constant "
                  f"columns (one value in >= {NEAR_CONSTANT_MODE_SHARE:.0%} of rows) and exact duplicates (|r| > 0.999); z-score; PCA (sklearn), "
                  "UMAP (umap-learn, n_neighbors=15, min_dist=0.3), t-SNE (perplexity 20); k-means groups on "
                  f"the first {meta['k_pcs_for_clusters']} PCs (80% variance), k chosen by silhouette.",
        "n_stations": int(len(F)),
        "n_features_joined": int(F.shape[1]),
        "n_features_used": int(Z.shape[1]),
        "dropped_columns": dropped,
        "features_by_source": {s: [c for c in cols if c in Z.columns] for s, cols in SOURCES.items()},
        "missing_values_imputed": {c: int(v) for c, v in F.isna().sum().items() if v},
        "pca_variance_explained": [round(float(v), 4) for v in ev],
        "pca_cumulative": [round(float(v), 4) for v in np.cumsum(ev)],
        "pc1_top": _axis_name(L["PC1"], 5), "pc2_top": _axis_name(L["PC2"], 5),
        "umap_vs_pca_spearman": {
            "umap1~pca1": round(float(spearmanr(E["umap1"], E["pca1"])[0]), 3),
            "umap2~pca2": round(float(spearmanr(E["umap2"], E["pca2"])[0]), 3)},
        "clusters": {"k": meta["k"], "names": names,
                     "silhouette_by_k": {str(k): round(v, 3) for k, v in meta["silhouette_by_k"].items()},
                     "sizes": E["cluster"].value_counts().sort_index().tolist()},
        "trip_and_model_metrics": f"{EVAL_YEAR} trips from data/processed/lfb_travel_training.csv grouped by "
                                  "deployed_from_station; LightGBM error is held out (model trained 2021-24); "
                                  "cached in station_trip_metrics_2025.csv",
        "n_closing_layouts_for_keep_rate": int(F.attrs.get("n_closing_layouts", 0)) or None,
        "layout_matrix": {"n_layouts": int(len(LF)), "n_features_used": int(LZ.shape[1]),
                          "dropped_columns": ldropped,
                          "pca_variance_explained": [round(float(v), 4) for v in lpca.explained_variance_ratio_],
                          "pc1_top": _axis_name(pd.Series(lpca.components_[0], index=LZ.columns), 5)},
        "not_joined": {
            "validation/closure_2014*": "the 10 stations closed in 2014 are not current stations",
            "validation/dispatch": "city-wide / closure-area rows, not stations",
            "placement/tradeoff, tradeoff_busy, slow_calls": "layout level with scenario x alpha keys that do not "
                                                             "match the named layouts; left out of the layout PCA",
        },
        "honesty": "Descriptive only. Many inputs measure the same thing (incidents, calls/day, engine hours, "
                   "multi-dispatches all track demand), so PCA axes are dominated by whichever theme has the most "
                   "columns; read loadings, not just positions. Area-map metrics give each map cell to its "
                   "crow-fly nearest station, which is not the LFB station ground. Plan metrics come from "
                   "specific planner runs with their own demand years. UMAP/t-SNE distances are only "
                   "meaningful locally; their axes have no units. Groups are k-means on ~100 points with modest "
                   "silhouette, so treat them as a reading aid, not station types.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_features_used", "dropped_columns", "pca_cumulative",
                                              "pc1_top", "pc2_top", "clusters")}, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
