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

Writes data/figures/analysis_embedding/:
  station_feature_matrix.csv, station_embedding_2d.csv, pca_loadings.csv, top_features.txt,
  layout_feature_matrix.csv, layout_embedding_2d.csv, summary.json,
  00_summary.png, 01_pca.png, 02_umap.png, 02b_tsne.png, 03_feature_loadings.png, 04_layout_pca.png

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
OUT = FIG / "analysis_embedding"
TRIP_CACHE = OUT / "station_trip_metrics_2025.csv"
EVAL_YEAR = 2025
SEED = 42
NEAR_CONSTANT_MODE_SHARE = 0.95
BASEMAP = PROC / "london_basemap.pkl"

# Reference palette (dataviz skill, light mode), as in plot_station_busyness.py.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE, ORANGE, GRAY = "#256abf", "#eb6834", "#9a9893"
CLUSTER_COLORS = ["#256abf", "#eb6834", "#3f8f5b", "#b5892b", "#7a6f9b", "#52514e"]

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


def _label_extremes(ax, E: pd.DataFrame, x: str, y: str, n: int = 10):
    xy = E[[x, y]].to_numpy()
    r = np.hypot(*((xy - np.median(xy, axis=0)) / xy.std(axis=0)).T)
    for i in np.argsort(-r)[:n]:
        ax.annotate(E.index[i], xy[i], xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)


def _scatter_pumps(ax, E, F, x, y):
    for pumps, color in [(1, GRAY), (2, ORANGE)]:
        m = F["n_pumps"] == pumps
        ax.scatter(E.loc[m, x], E.loc[m, y], s=34, c=color, edgecolor="white", linewidth=0.6,
                   label=f"{pumps} pump{'s' if pumps > 1 else ''} ({m.sum()})")
    ax.legend(frameon=False, fontsize=8, loc="best")


def _scatter_seq(fig, ax, E, F, x, y, col):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    sc = ax.scatter(E[x], E[y], s=34, c=F[col], cmap=cmap, edgecolor="white", linewidth=0.6)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.01)
    cb.outline.set_visible(False)
    cb.set_label(LABELS.get(col, col), color=INK2, fontsize=8)


def plot_pair(E, F, x, y, title, axis_labels, path):
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6), layout="constrained")
    _scatter_pumps(a1, E, F, x, y)
    a1.set_title("Coloured by number of pumps")
    _scatter_seq(fig, a2, E, F, x, y, "incidents_per_pump")
    a2.set_title("Coloured by incidents per pump")
    for a in (a1, a2):
        _label_extremes(a, E, x, y)
        a.set_xlabel(axis_labels[0])
        a.set_ylabel(axis_labels[1])
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.savefig(path, dpi=160)
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


def plot_summary(E, F, Z, loc, pca, meta, path):
    style()
    k = meta["k"]
    colors = CLUSTER_COLORS[:k]
    fig = plt.figure(figsize=(16, 10), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15])
    panels = [("pca1", "pca2", "PCA"), ("umap1", "umap2", "UMAP"), ("tsne1", "tsne2", "t-SNE")]
    for j, (x, y, name) in enumerate(panels[:2]):
        ax = fig.add_subplot(gs[0, j])
        for c in range(k):
            m = E["cluster"] == c
            ax.scatter(E.loc[m, x], E.loc[m, y], s=30, c=colors[c], edgecolor="white", linewidth=0.5,
                       label=f"group {c + 1} ({m.sum()})")
        if name == "PCA":
            _label_extremes(ax, E, x, y, n=8)
            ev = 100 * pca.explained_variance_ratio_
            ax.set_xlabel(f"PC1 ({ev[0]:.0f}%)")
            ax.set_ylabel(f"PC2 ({ev[1]:.0f}%)")
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(f"{name} of {Z.shape[1]} station metrics")
    fig.axes[0].legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    if BASEMAP.exists():
        import pickle

        from matplotlib.collections import LineCollection

        bm = pickle.loads(BASEMAP.read_bytes())
        ax.add_collection(LineCollection(bm["roads"], colors="#e4e3df", linewidths=0.5, zorder=1))
        ax.add_collection(LineCollection(bm["river"], colors="#b9d6ee", linewidths=3.5, zorder=1))
    for c in range(k):
        m = E["cluster"] == c
        ax.scatter(loc.loc[m, "lon"], loc.loc[m, "lat"], s=34, c=colors[c], edgecolor="white", linewidth=0.5,
                   zorder=3)
    pad = 0.03
    ax.set_xlim(loc["lon"].min() - pad, loc["lon"].max() + pad)
    ax.set_ylim(loc["lat"].min() - pad, loc["lat"].max() + pad)
    ax.grid(False)
    ax.set_aspect(1 / np.cos(np.deg2rad(51.5)))
    ax.set_title("Same groups on the map")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, :])
    prof = Z.groupby(E["cluster"]).mean()
    spread = prof.abs().max().sort_values(ascending=False).index[:22]
    prof = prof[spread]
    im = ax.imshow(prof.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-1.6, vmax=1.6)
    ax.set_yticks(range(k), [f"group {c + 1} (n={int((E['cluster'] == c).sum())})" for c in range(k)])
    for t, c in zip(ax.get_yticklabels(), colors):
        t.set_color(c)
        t.set_fontweight("bold")
    ax.set_xticks(range(len(spread)), [LABELS.get(c, c) for c in spread], rotation=40, ha="right", fontsize=8)
    for i in range(k):
        for j in range(len(spread)):
            v = prof.iloc[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 1 else INK)
    ax.grid(False)
    ax.set_title("Group profiles: mean z-score of the 22 metrics that separate groups most")
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
    cb.outline.set_visible(False)
    cb.set_label("z-score (blue below, red above city median)", fontsize=8, color=INK2)
    fig.suptitle(f"London's {len(E)} fire stations, every analysis metric condensed to 2D", x=0.01, ha="left",
                 fontsize=15, fontweight="bold", color=INK)
    fig.savefig(path, dpi=150)
    plt.close(fig)


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

    plot_summary(E, F, Z, loc, pca, meta, OUT / "00_summary.png")
    plot_pair(E, F, "pca1", "pca2", "Stations in PCA space",
              (f"PC1 ({100 * ev[0]:.0f}%): {_axis_name(L['PC1'])}", f"PC2 ({100 * ev[1]:.0f}%): "
               f"{_axis_name(L['PC2'])}"), OUT / "01_pca.png")
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
        "clusters": {"k": meta["k"], "silhouette_by_k": {str(k): round(v, 3) for k, v in
                                                          meta["silhouette_by_k"].items()},
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
