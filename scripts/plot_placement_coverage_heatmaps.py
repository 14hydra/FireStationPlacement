#!/usr/bin/env python3
"""Before / after / difference + demand coverage heatmaps for move-10 placement.

Prefer the recommended fair layout (``replace_10_boroughs1s``) by recovering
per-cell times from its Folium HTML, then join real call demand from
``exact_runs/replace_10/demand_cells.csv``. Falls back to that CSV alone if
the HTML is missing.

Demand is the reason some stations move and others stay: high-call cells that
are slow today get new sites; low-call / already-fast neighbourhoods keep
their station or lose a redundant one.

Writes:
  data/figures/placement/coverage_heatmaps/
    01_before_after_diff.png     — 2×2: before, after, Δ, demand
    02_bad_coverage_before.png   — cells over 6 min today (sized by demand)
    03_improvement_only.png      — where the plan is ≥5 s faster
    04_move_rationale.png        — closed/opened with local demand + Δ
    cells.csv                    — cell table with demand weights
    move_rationale.csv           — per-station local demand / response stats
  data/figures/final/07_coverage_heatmaps.png  — board copy of panel 01

Run:  PYTHONPATH=src .venv/bin/python scripts/plot_placement_coverage_heatmaps.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "data" / "figures"
REC = FIG / "placement/recommended/replace_10_boroughs1s"
EXACT = FIG / "placement/exact_runs/replace_10"
OUT = FIG / "placement/coverage_heatmaps"
FINAL = FIG / "final"

INK, MUTED = "#15202b", "#5b6570"
DIV_TIME = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a3a2", "#e34948", "#9e1f1e"]
DIV_DIFF = ["#1d5bbf", "#6ea0e0", "#d6e4f5", "#f0efec", "#f5d0c8", "#e07060", "#b91c1c"]
DIV_DEMAND = ["#f8fafc", "#cbd5e1", "#94a3b8", "#64748b", "#334155", "#0f172a"]

# Radius (degrees ≈ km/111) used for “local” demand around a station
LOCAL_DEG = 0.025  # ~2.8 km


def load_cells_from_html(path: Path) -> pd.DataFrame:
    """Parse Folium circle markers + Δ/plan/was popups written by the planner."""
    html = path.read_text()
    blocks = re.split(r"circleMarker\(", html)[1:]
    rows = []
    for block in blocks:
        m = re.match(r"\s*\[([0-9.\-]+),\s*([0-9.\-]+)\]", block)
        if not m:
            continue
        lat, lon = float(m.group(1)), float(m.group(2))
        p = re.search(
            r"Δ=([+\-\d.]+)s\s*·\s*plan=([0-9.]+)\s*min\s*·\s*was=([0-9.]+)\s*min",
            block,
        )
        if not p:
            continue  # station marker, not a demand cell
        delta_s = float(p.group(1))
        plan_min = float(p.group(2))
        was_min = float(p.group(3))
        rows.append({
            "lat": lat,
            "lon": lon,
            "response_s": plan_min * 60.0,
            "response_s_current": was_min * 60.0,
            "response_s_delta": delta_s,  # plan − current (negative = faster)
            "weight": 1.0,
        })
    if len(rows) < 100:
        raise ValueError(f"Only {len(rows)} demand cells parsed from {path}")
    return pd.DataFrame(rows)


def attach_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Join real call weights from exact_runs demand_cells (nearest lat/lon)."""
    demand_path = EXACT / "demand_cells.csv"
    if not demand_path.exists():
        print("no demand_cells.csv — keeping uniform weight")
        return df
    dem = pd.read_csv(demand_path)
    if not {"lat", "lon", "weight"}.issubset(dem.columns):
        return df

    out = df.copy()
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.c_[dem["lat"].to_numpy(), dem["lon"].to_numpy()])
        dist, ix = tree.query(np.c_[out["lat"].to_numpy(), out["lon"].to_numpy()], k=1)
    except ImportError:
        dlat = out["lat"].to_numpy()[:, None] - dem["lat"].to_numpy()[None, :]
        dlon = out["lon"].to_numpy()[:, None] - dem["lon"].to_numpy()[None, :]
        dist2 = dlat * dlat + dlon * dlon
        ix = np.argmin(dist2, axis=1)
        dist = np.sqrt(dist2[np.arange(len(ix)), ix])

    out["weight"] = dem["weight"].to_numpy()[ix].astype(float)
    if "n_incidents" in dem.columns:
        out["n_incidents"] = dem["n_incidents"].to_numpy()[ix]
    # Guard: if HTML cells are far from the demand grid, warn
    far = float(np.mean(np.asarray(dist) > 0.01))  # > ~1 km
    if far > 0.05:
        print(f"warning: {far:.0%} of cells >1 km from nearest demand cell")
    print(
        f"demand attached: weight sum={out['weight'].sum():.0f}, "
        f"median cell weight={out['weight'].median():.0f}"
    )
    return out


def load_cells() -> tuple[pd.DataFrame, str]:
    html = REC / "fire_station_plan_map.html"
    csv = EXACT / "demand_cells.csv"
    if html.exists():
        try:
            df = load_cells_from_html(html)
            df = attach_demand(df)
            return df, "replace_10_boroughs1s (fair move-10)"
        except ValueError as e:
            print("HTML parse failed:", e)
    if csv.exists():
        df = pd.read_csv(csv)
        return df, "exact_runs/replace_10"
    raise FileNotFoundError("No demand cell source found for coverage heatmaps")


def _demand_sizes(weight: np.ndarray, s_lo: float = 6.0, s_hi: float = 55.0) -> np.ndarray:
    w = np.asarray(weight, float)
    w = np.clip(w, 1.0, None)
    # sqrt scale so hubs dominate without blotting out the map
    q_lo, q_hi = np.nanpercentile(np.sqrt(w), [5, 95])
    t = (np.sqrt(w) - q_lo) / max(q_hi - q_lo, 1e-9)
    t = np.clip(t, 0.0, 1.0)
    return s_lo + t * (s_hi - s_lo)


def _local_mask(df: pd.DataFrame, lat: float, lon: float, deg: float = LOCAL_DEG) -> np.ndarray:
    return (
        (df["lat"].to_numpy(float) - lat) ** 2 + (df["lon"].to_numpy(float) - lon) ** 2
    ) <= deg ** 2


def station_local_stats(df: pd.DataFrame, stations: pd.DataFrame, role: str) -> pd.DataFrame:
    """Per station: local call weight, local before/after mean response, rationale."""
    w = df["weight"].to_numpy(float)
    w_total = float(w.sum())
    before = df["response_s_current"].to_numpy(float)
    after = df["response_s"].to_numpy(float)
    if "response_s_delta" in df.columns:
        delta = df["response_s_delta"].to_numpy(float)
    else:
        delta = after - before

    rows = []
    for _, s in stations.iterrows():
        m = _local_mask(df, float(s["lat"]), float(s["lon"]))
        if not m.any():
            continue
        ww = w[m]
        ww_sum = float(ww.sum())
        share = ww_sum / w_total
        b = float(np.average(before[m], weights=ww))
        a = float(np.average(after[m], weights=ww))
        d = float(np.average(delta[m], weights=ww))
        over6_b = float(ww[before[m] > 360].sum() / ww_sum) if ww_sum else 0.0
        over6_a = float(ww[after[m] > 360].sum() / ww_sum) if ww_sum else 0.0
        name = str(s.get("site_name", s.get("candidate_id", "?")))
        short = name.replace("New site on ", "").replace("New site near ", "")
        if len(short) > 36:
            short = short[:33] + "…"

        if role == "opened":
            # Open where demand is material AND neighbourhood was slow / gains a lot
            if share >= 0.012 and (b >= 330 or d <= -15):
                why = "High local call load + slow today → open here"
            elif d <= -20:
                why = "Large local speed-up for nearby calls"
            elif share >= 0.015:
                why = "Dense call cluster; fills a coverage gap"
            else:
                why = "Improves a weak pocket of the map"
        else:  # closed
            # Close where demand is thin OR neighbourhood stays fast after move
            if share < 0.008 and a <= 360:
                why = "Low local demand; coverage stays OK after move"
            elif a <= b + 5 and a <= 360:
                why = "Nearby demand still covered in ≤6 min"
            elif share < 0.010:
                why = "Thin call weight relative to denser sites"
            else:
                why = "Capacity relocates to higher-demand gaps"

        rows.append({
            "role": role,
            "site_name": name,
            "short_name": short,
            "lat": float(s["lat"]),
            "lon": float(s["lon"]),
            "local_weight": ww_sum,
            "local_share": share,
            "local_before_s": b,
            "local_after_s": a,
            "local_delta_s": d,
            "local_over6_before": over6_b,
            "local_over6_after": over6_a,
            "why": why,
        })
    return pd.DataFrame(rows)


def plot_board(df: pd.DataFrame, label: str) -> Path:
    planned = pd.read_csv(REC / "planned_fire_stations.csv") if (REC / "planned_fire_stations.csv").exists() else None
    closed = pd.read_csv(REC / "closed_fire_stations.csv") if (REC / "closed_fire_stations.csv").exists() else None
    opened = pd.read_csv(REC / "opened_fire_stations.csv") if (REC / "opened_fire_stations.csv").exists() else None

    before = df["response_s_current"].to_numpy(float) / 60.0
    after = df["response_s"].to_numpy(float) / 60.0
    if "response_s_delta" in df.columns:
        diff_s = df["response_s_delta"].to_numpy(float)
    else:
        diff_s = df["response_s"].to_numpy(float) - df["response_s_current"].to_numpy(float)
    diff_min = diff_s / 60.0
    weight = df["weight"].to_numpy(float)
    sizes = _demand_sizes(weight)

    time_cmap = ListedColormap(DIV_TIME)
    time_norm = BoundaryNorm([3, 4, 5, 5.75, 6.25, 7, 8, 11], len(DIV_TIME))
    lim = float(np.nanpercentile(np.abs(diff_min[np.isfinite(diff_min)]), 99))
    lim = float(np.clip(max(lim, 1.0), 1.0, 4.0))
    diff_cmap = ListedColormap(DIV_DIFF)
    diff_norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    dem_cmap = ListedColormap(DIV_DEMAND)
    # log-ish demand colour
    dem_v = np.log1p(weight)
    dem_norm = BoundaryNorm(
        np.nanpercentile(dem_v, [0, 20, 40, 60, 80, 95, 100]), len(DIV_DEMAND)
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 12.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.06, wspace=0.08, hspace=0.18)
    ax_b, ax_a = axes[0]
    ax_d, ax_dem = axes[1]

    def _base(ax, vals, cmap, norm, title, show_closed=False, show_opened=False, show_stations=False):
        order = np.argsort(sizes)  # big demand on top
        ax.scatter(
            df["lon"].to_numpy()[order], df["lat"].to_numpy()[order],
            c=np.asarray(vals)[order], s=sizes[order], cmap=cmap, norm=norm,
            alpha=0.9, edgecolors="none", rasterized=True,
        )
        if show_stations and planned is not None:
            keep = planned if closed is None else planned[
                ~planned["site_name"].isin(set(closed["site_name"]))
            ] if "site_name" in planned.columns and "site_name" in closed.columns else planned
            ax.scatter(keep["lon"], keep["lat"], s=10, c="#1e293b", marker="s", zorder=5, alpha=0.7)
        if show_closed and closed is not None:
            ax.scatter(closed["lon"], closed["lat"], s=42, c="black", marker="x",
                       zorder=6, linewidths=1.6, label="closed")
        if show_opened and opened is not None:
            ax.scatter(opened["lon"], opened["lat"], s=48, c="#16a34a", marker="^",
                       zorder=6, label="opened")
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=6)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=7, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color("#e6eaef")

    _base(ax_b, before, time_cmap, time_norm, "Before — today’s stations", show_stations=True)
    _base(ax_a, after, time_cmap, time_norm, "After — move up to 10 stations",
          show_closed=True, show_opened=True)
    # Difference: emphasize |Δ|≥5s but still size by demand
    sizes_diff = np.where(np.abs(diff_s) >= 5.0, sizes * 1.15, sizes * 0.55)
    order = np.argsort(np.abs(diff_min))
    sc_diff = ax_d.scatter(
        df["lon"].to_numpy()[order], df["lat"].to_numpy()[order],
        c=diff_min[order], s=sizes_diff[order], cmap=diff_cmap, norm=diff_norm,
        alpha=0.9, edgecolors="none", rasterized=True,
    )
    if opened is not None:
        ax_d.scatter(opened["lon"], opened["lat"], s=48, c="#16a34a", marker="^", zorder=6)
    if closed is not None:
        ax_d.scatter(closed["lon"], closed["lat"], s=42, c="black", marker="x", zorder=6, linewidths=1.6)
    ax_d.set_title("Difference (after − before)", fontsize=12, fontweight="bold", color=INK, pad=6)
    ax_d.set_aspect("equal", adjustable="box")
    ax_d.tick_params(labelsize=7, colors=MUTED)

    order_d = np.argsort(weight)
    sc_dem = ax_dem.scatter(
        df["lon"].to_numpy()[order_d], df["lat"].to_numpy()[order_d],
        c=dem_v[order_d], s=sizes[order_d], cmap=dem_cmap, norm=dem_norm,
        alpha=0.92, edgecolors="none", rasterized=True,
    )
    if opened is not None:
        ax_dem.scatter(opened["lon"], opened["lat"], s=48, c="#16a34a", marker="^",
                       zorder=6, label="opened (into demand)")
    if closed is not None:
        ax_dem.scatter(closed["lon"], closed["lat"], s=42, c="#dc2626", marker="x",
                       zorder=6, linewidths=1.6, label="closed (leave demand)")
    ax_dem.set_title("Call demand — why some stations move", fontsize=12, fontweight="bold", color=INK, pad=6)
    ax_dem.set_aspect("equal", adjustable="box")
    ax_dem.tick_params(labelsize=7, colors=MUTED)
    ax_dem.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.92)

    ax_b.set_ylabel("Latitude")
    ax_d.set_ylabel("Latitude")
    ax_d.set_xlabel("Longitude")
    ax_dem.set_xlabel("Longitude")

    # Colorbars
    for ax, cmap, norm, label_cb in (
        (ax_b, time_cmap, time_norm, "minutes"),
        (ax_a, time_cmap, time_norm, "minutes"),
    ):
        cax = ax.inset_axes([0.08, -0.10, 0.84, 0.035])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.set_label(label_cb, fontsize=8, color=MUTED)
        cb.ax.tick_params(labelsize=7)
    cax_d = ax_d.inset_axes([0.08, -0.10, 0.84, 0.035])
    cb_d = fig.colorbar(sc_diff, cax=cax_d, orientation="horizontal")
    cb_d.set_label("minutes (blue = faster · red = slower)", fontsize=8, color=MUTED)
    cb_d.ax.tick_params(labelsize=7)
    cax_dem = ax_dem.inset_axes([0.08, -0.10, 0.84, 0.035])
    cb_dem = fig.colorbar(sc_dem, cax=cax_dem, orientation="horizontal")
    cb_dem.set_label("call weight (darker = more incidents)", fontsize=8, color=MUTED)
    cb_dem.ax.tick_params(labelsize=7)

    w_n = weight / weight.sum()
    mean_delta = float((diff_s * w_n).sum())
    pct_faster = float(w_n[diff_s < -5].sum() * 100)
    over6_before = float(w_n[before > 6].sum() * 100)
    over6_after = float(w_n[after > 6].sum() * 100)

    fig.suptitle(
        "Coverage and call demand — where to move stations, and where not to\n"
        f"{label}  ·  demand-weighted mean Δ {mean_delta:+.1f} s  ·  "
        f"{pct_faster:.0f}% of call-weight ≥5 s faster  ·  "
        f"share over 6 min {over6_before:.0f}% → {over6_after:.0f}%\n"
        "Dot size ∝ local call weight. Open (▲) into high-demand gaps; close (✕) where demand is thin or still covered.",
        fontsize=13, fontweight="bold", color=INK, y=0.995,
    )
    ax_a.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "01_before_after_diff.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)
    return path


def plot_bad_coverage(df: pd.DataFrame) -> Path:
    before = df["response_s_current"].to_numpy(float) / 60.0
    weight = df["weight"].to_numpy(float)
    sizes = _demand_sizes(weight, 5, 50)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    ax.scatter(df["lon"], df["lat"], c="#dbe3ec", s=sizes * 0.5, edgecolors="none", alpha=0.85)
    bad = before > 6
    ax.scatter(
        df.loc[bad, "lon"], df.loc[bad, "lat"], c=before[bad], s=sizes[bad],
        cmap=ListedColormap(["#fca5a5", "#ef4444", "#9f1239"]),
        vmin=6, vmax=11, edgecolors="none",
    )
    ax.set_title("Bad coverage today — over 6 min (sized by call demand)", fontweight="bold", color=INK)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    share = float(weight[bad].sum() / weight.sum() * 100)
    ax.text(
        0.02, 0.02,
        f"{share:.0f}% of call-weight over LFB’s 6-minute average target",
        transform=ax.transAxes, fontsize=10, color=MUTED,
    )
    path = OUT / "02_bad_coverage_before.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)
    return path


def plot_improvement(df: pd.DataFrame) -> Path:
    if "response_s_delta" in df.columns:
        diff_s = df["response_s_delta"].to_numpy(float)
    else:
        diff_s = df["response_s"].to_numpy(float) - df["response_s_current"].to_numpy(float)
    weight = df["weight"].to_numpy(float)
    sizes = _demand_sizes(weight, 6, 48)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    ax.scatter(df["lon"], df["lat"], c="#e8eef4", s=sizes * 0.45, edgecolors="none")
    gain = diff_s <= -5
    ax.scatter(
        df.loc[gain, "lon"], df.loc[gain, "lat"], c=-diff_s[gain] / 60.0, s=sizes[gain],
        cmap="Blues", vmin=0, vmax=2, edgecolors="none",
    )
    opened = REC / "opened_fire_stations.csv"
    if opened.exists():
        o = pd.read_csv(opened)
        ax.scatter(o["lon"], o["lat"], s=55, c="#16a34a", marker="^", zorder=5, label="new station")
        ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Where the plan helps most (≥5 s faster; sized by demand)", fontweight="bold", color=INK)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    path = OUT / "03_improvement_only.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)
    return path


def plot_move_rationale(df: pd.DataFrame) -> Path:
    closed = pd.read_csv(REC / "closed_fire_stations.csv")
    opened = pd.read_csv(REC / "opened_fire_stations.csv")
    rat_c = station_local_stats(df, closed, "closed")
    rat_o = station_local_stats(df, opened, "opened")
    rat = pd.concat([rat_c, rat_o], ignore_index=True)
    rat.to_csv(OUT / "move_rationale.csv", index=False)
    print("wrote", OUT / "move_rationale.csv")

    weight = df["weight"].to_numpy(float)
    sizes = _demand_sizes(weight, 4, 36)
    dem_v = np.log1p(weight)
    dem_cmap = ListedColormap(DIV_DEMAND)
    dem_norm = BoundaryNorm(np.nanpercentile(dem_v, [0, 20, 40, 60, 80, 95, 100]), len(DIV_DEMAND))

    fig = plt.figure(figsize=(15.5, 8.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_t.axis("off")

    order = np.argsort(weight)
    ax.scatter(
        df["lon"].to_numpy()[order], df["lat"].to_numpy()[order],
        c=dem_v[order], s=sizes[order], cmap=dem_cmap, norm=dem_norm,
        alpha=0.88, edgecolors="none", rasterized=True,
    )
    ax.scatter(rat_c["lon"], rat_c["lat"], s=70, c="#b91c1c", marker="x",
               zorder=6, linewidths=2.0, label="close")
    ax.scatter(rat_o["lon"], rat_o["lat"], s=80, c="#15803d", marker="^",
               zorder=6, label="open")
    # Short labels on map for a few high-share sites
    for _, r in rat.nlargest(6, "local_share").iterrows():
        ax.annotate(
            r["short_name"].split("/")[0].strip()[:18],
            (r["lon"], r["lat"]),
            textcoords="offset points", xytext=(6, 6),
            fontsize=7, color=INK, alpha=0.9,
        )
    ax.set_title("Demand map with moves", fontweight="bold", color=INK)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.92)
    ax.tick_params(labelsize=8, colors=MUTED)

    # Table: open vs close with demand + reason
    lines = [
        "Why open vs close (local ~3 km call weight)",
        "",
    ]
    lines.append("OPEN  — move capacity into demand")
    for _, r in rat_o.sort_values("local_share", ascending=False).iterrows():
        lines.append(
            f"  ▲ {r['short_name'][:34]:<34}  "
            f"{r['local_share']*100:4.1f}% calls  "
            f"{r['local_before_s']/60:4.1f}→{r['local_after_s']/60:4.1f} min  "
            f"Δ{r['local_delta_s']:+.0f}s"
        )
        lines.append(f"      {r['why']}")
    lines.append("")
    lines.append("CLOSE — leave thin / still-covered demand")
    for _, r in rat_c.sort_values("local_share", ascending=True).iterrows():
        lines.append(
            f"  ✕ {r['short_name'][:34]:<34}  "
            f"{r['local_share']*100:4.1f}% calls  "
            f"{r['local_before_s']/60:4.1f}→{r['local_after_s']/60:4.1f} min  "
            f"Δ{r['local_delta_s']:+.0f}s"
        )
        lines.append(f"      {r['why']}")

    ax_t.text(
        0.0, 1.0, "\n".join(lines),
        transform=ax_t.transAxes, va="top", ha="left",
        fontsize=8.2, family="monospace", color=INK,
        linespacing=1.35,
    )
    fig.suptitle(
        "Demand decides the moves — open into high-call gaps; close where calls are few or still fast",
        fontsize=13, fontweight="bold", color=INK, y=0.98,
    )
    path = OUT / "04_move_rationale.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)
    return path


def main() -> int:
    df, label = load_cells()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "cells.csv", index=False)
    print(f"loaded {len(df)} cells from {label}")
    board = plot_board(df, label)
    plot_bad_coverage(df)
    plot_improvement(df)
    plot_move_rationale(df)
    FINAL.mkdir(parents=True, exist_ok=True)
    dest = FINAL / "07_coverage_heatmaps.png"
    shutil.copy2(board, dest)
    # Also ship rationale next to the pack for the board / appendix
    shutil.copy2(OUT / "04_move_rationale.png", FINAL / "07b_move_rationale.png")
    print("copied →", dest.relative_to(ROOT))
    print("copied →", (FINAL / "07b_move_rationale.png").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
