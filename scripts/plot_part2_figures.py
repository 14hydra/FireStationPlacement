#!/usr/bin/env python3
"""Part 2 poster figures. Reads saved results only, except the map, which re-scores the main layout per square with LightGBM.

  11_part2_map.png            change in modelled attendance per 0.9 km square, main layout (Rule B targeted Move 10),
                              LightGBM, busy engines; target boroughs outlined; stations closed and opened marked
  12_part2_three_models.png   target-borough change under three models, 2025 and 2026, main layout vs citywide vs planner
  13_part2_no_borough_worse.png  every borough's change in 2026, three models, main layout vs planner Move 10

The map uses no call counts from 2025 or 2026 (only which squares have 2024 calls), so it is not a new use of 2026.
Writes data/figures/final/ and caches the map's per-square numbers in data/figures/part2/.

Run:  PYTHONPATH=src .venv/bin/python scripts/plot_part2_figures.py [--rescore]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PL = ROOT / "data" / "figures" / "placement"
FIN = ROOT / "data" / "figures" / "final"
CACHE = ROOT / "data" / "figures" / "part2"
LAT0 = 51.5  # replaced by the 2024 calls' mean latitude when known (map_cells.csv stores it)
TARGETS = ["HILLINGDON", "HAVERING", "ENFIELD", "BROMLEY", "REDBRIDGE"]
MAIN = PL / "targeted" / "rule_b" / "layouts" / "target_10.csv"

# Reference palette (dataviz skill): categorical slots 1-3, diverging blue <-> red with a grey midpoint.
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#9e1f1f"])
MODELS = [("lgbm", "LightGBM"), ("road", "Road formula"), ("kolesar", "Straight-line formula")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def rescore() -> pd.DataFrame:
    from firestation.busy_placement import ErlangBusyModel
    from firestation.cities import get_city, load_fire_stations, load_incidents
    from firestation.facility import TravelTimeSurrogate, build_demand_grid, posture_seconds
    from firestation.fire_station_location import build_travel_matrix, busy_prob_vector, drop_far_incidents, fit_scorer_kolesar, recorded_station_turnout
    from firestation.ml_scorer import cell_boroughs
    from firestation.station_capacity import default_pump_table

    spec = get_city("london")
    houses = load_fire_stations(spec)
    demand = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024]), houses).reset_index(drop=True)
    cells = build_demand_grid(demand, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    w = cells["n_incidents"].to_numpy(float)
    pumps_today = default_pump_table(houses).set_index("station")["n_pumps"].reindex(houses["facilityname"].astype(str)).fillna(1).astype(int).to_numpy()
    tab, new_turn = recorded_station_turnout([2024])
    norm = {k.upper(): v for k, v in tab.items()}
    kmodel, _, _ = fit_scorer_kolesar(demand, houses)
    today = pd.DataFrame({"site_name": houses["facilityname"].astype(str), "lat": houses["lat"], "lon": houses["lon"], "is_existing": True,
                          "n_pumps": pumps_today})
    main = pd.read_csv(MAIN)
    key = lambda r: str(r.site_name) if bool(r.is_existing) else f"{float(r.lat):.6f},{float(r.lon):.6f}"  # noqa: E731
    sites, index = [], {}
    for L in (today, main):
        for r in L.itertuples():
            if key(r) not in index:
                index[key(r)] = len(sites)
                sites.append({"site_name": str(r.site_name), "lat": float(r.lat), "lon": float(r.lon), "is_existing": bool(r.is_existing)})
    sites = pd.DataFrame(sites)
    rows = [np.array([index[key(r)] for r in L.itertuples()]) for L in (today, main)]
    pumps = [np.maximum(L["n_pumps"].fillna(1).astype(int).to_numpy(), 1) for L in (today, main)]
    turnout = np.array([norm.get(r.site_name.upper(), new_turn) if r.is_existing else new_turn for r in sites.itertuples()])
    T, _ = build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer="lgbm", kolesar=kmodel, incidents=demand)
    A = T + turnout[:, None]
    bp = np.ones(len(sites), int)
    bp[rows[0]] = pumps[0]
    m = ErlangBusyModel.calibrate(A, rows[0], busy_prob_vector(sites, demand), bp, w)
    out = []
    for r, p in zip(rows, pumps):
        m.pumps = np.ones(len(sites), int)
        m.pumps[r] = p
        out.append(posture_seconds(A, r, m.probs(A, r)))
    CACHE.mkdir(parents=True, exist_ok=True)
    c = cells[["_ix", "_iy", "lat", "lon", "n_incidents"]].assign(borough=np.asarray(cell_boroughs(cells, demand)).astype(str),
                                                                   today_s=out[0], main_s=out[1])
    c["lat0"] = float(pd.to_numeric(demand["dest_lat"], errors="coerce").dropna().mean())
    c.to_csv(CACHE / "map_cells.csv", index=False)
    return c


def fig_map(c: pd.DataFrame) -> None:
    global LAT0
    if "lat0" in c:
        LAT0 = float(c["lat0"].iloc[0])
    houses = pd.read_csv(MAIN)
    from firestation.cities import get_city, load_fire_stations
    today = load_fire_stations(get_city("london"))
    kept = set(houses.loc[houses["is_existing"].astype(bool), "site_name"].astype(str))
    closed = today[~today["facilityname"].astype(str).isin(kept)]
    opened = houses[~houses["is_existing"].astype(bool)]
    ix, iy = c["_ix"] - c["_ix"].min(), c["_iy"] - c["_iy"].min()
    grid = np.full((iy.max() + 1, ix.max() + 1), np.nan)
    grid[iy, ix] = np.clip(c["main_s"] - c["today_s"], -60, 60)
    tmask = np.zeros_like(grid)
    tmask[iy, ix] = c["borough"].str.upper().isin(TARGETS).astype(float)
    x0, y0 = c["_ix"].min(), c["_iy"].min()
    dlat = 0.9 / 110.574  # as build_demand_grid: cell_km / km per degree, longitude at the 2024 calls' mean latitude
    dlon = 0.9 / (111.320 * np.cos(np.radians(LAT0)))
    ext = [x0 * dlon, (ix.max() + 1 + x0) * dlon, y0 * dlat, (iy.max() + 1 + y0) * dlat]
    fig, ax = plt.subplots(figsize=(11, 8.6))
    im = ax.imshow(grid, origin="lower", extent=ext, cmap=DIV, norm=TwoSlopeNorm(0, -60, 60), interpolation="nearest")
    pad = np.pad(tmask, 1)  # close the outline where a target borough reaches the edge of the grid
    xs = ext[0] + (np.arange(pad.shape[1]) - 0.5) * dlon
    ys = ext[2] + (np.arange(pad.shape[0]) - 0.5) * dlat
    ax.contour(xs, ys, pad, levels=[0.5], colors=INK, linewidths=1.6)
    ax.set_xlim(ext[0] - dlon, ext[1] + dlon)
    ax.set_ylim(ext[2] - dlat, ext[3] + dlat)
    ax.scatter(closed["lon"], closed["lat"], marker="x", s=70, c=INK, linewidths=2, label="Station closed", zorder=4)
    ax.scatter(opened["lon"], opened["lat"], marker="o", s=70, facecolors=SURF, edgecolors=INK, linewidths=2, label="Station opened", zorder=4)
    ax.plot([], [], color=INK, lw=1.6, label="5 target boroughs")
    ax.set_aspect(1 / np.cos(np.radians(51.5)))
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("Change in modelled attendance (s); blue = faster", color=INK2)
    cb.outline.set_visible(False)
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Moving 10 stations: where calls get faster (blue) or slower (red)", loc="left", color=INK, fontsize=15, pad=12)
    fig.text(0.02, 0.02, "LightGBM, busy engines, 0.9 km squares with 2024 calls; capped at ±60 s. Rule B targeted Move 10.",
             color=MUTED, fontsize=9)
    fig.savefig(FIN / "11_part2_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_models() -> None:
    s25a = pd.read_csv(PL / "targeted" / "scores.csv").query("rule == 'none'")
    s25b = pd.read_csv(PL / "targeted" / "rule_b" / "scores.csv").assign(layout=lambda d: "rule_b_" + d["layout"])
    s25 = pd.concat([s25a, s25b])
    s26 = pd.read_csv(PL / "final_test_2026" / "scores.csv")
    lays = [("rule_b_target_10", "Targeted Move 10 (Rule B sites)", S1), ("citywide_10", "Citywide Move 10", S2), ("planner_10", "Planner Move 10", S3)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, (yr, s) in zip(axes, (("2025 calls", s25), ("2026 calls (fixed prediction)", s26))):
        x = np.arange(3)
        bw = 0.26
        for k, (lay, lab, col) in enumerate(lays):
            v = s[s["layout"] == lay].set_index("scorer").reindex([m for m, _ in MODELS])["d_target_s"].to_numpy()
            bars = ax.bar(x + (k - 1) * bw, -v, bw - 0.03, color=col, label=lab, edgecolor=SURF, linewidth=2)
            for b, val in zip(bars, -v):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.3, f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=INK2)
        ax.axhline(10, color=INK, lw=1.2, ls=(0, (4, 3)))
        ax.text(2.45, 10.25, "pass mark: 10 s", ha="right", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(x, [n for _, n in MODELS])
        ax.set_title(yr, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Seconds faster in the 5 target boroughs")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Targeted moves help the slowest boroughs most, under every travel model", x=0.01, ha="left", color=INK, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIN / "12_part2_three_models.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_boroughs() -> None:
    b = pd.read_csv(PL / "final_test_2026" / "boroughs.csv")
    lays = [("rule_b_target_10", "Targeted Move 10 (Rule B sites)", S1), ("planner_10", "Planner Move 10", S3)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 9), sharey=True)
    order = b[(b["layout"] == "rule_b_target_10") & (b["scorer"] == "lgbm")].sort_values("d_s")["borough"].tolist()
    y = {n: i for i, n in enumerate(order)}
    for ax, (sc, name) in zip(axes, MODELS):
        for k, (lay, lab, col) in enumerate(lays):
            q = b[(b["layout"] == lay) & (b["scorer"] == sc)]
            ax.scatter(q["d_s"], q["borough"].map(y) + (k - 0.5) * 0.3, s=40, color=col, edgecolor=SURF, linewidth=1.5, label=lab, zorder=3)
            bad = q[q["d_s"] >= 1]
            for r in bad.itertuples():
                ax.annotate(f"+{r.d_s:.1f}", (r.d_s, y[r.borough] + (k - 0.5) * 0.3), xytext=(5, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK2)
        ax.axvspan(1, 12, color="#f2a3a2", alpha=0.25, lw=0)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.axvline(1, color=INK, lw=1, ls=(0, (4, 3)))
        ax.set_xlim(-32, 12)
        ax.set_title(name, loc="left", color=INK)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Change in borough average (s)")
    labels = [n.title() + (" ●" if n in TARGETS else "") for n in order]
    axes[0].set_yticks(range(len(order)), labels, fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=2, fontsize=10)
    fig.suptitle("2026: no borough 1 s slower with targeted moves (shaded = fails the 1 s limit; ● = target borough)",
                 x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIN / "13_part2_no_borough_worse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true", help="Recompute the map's per-square numbers (a few minutes)")
    args = ap.parse_args()
    FIN.mkdir(parents=True, exist_ok=True)
    f = CACHE / "map_cells.csv"
    c = rescore() if args.rescore or not f.exists() else pd.read_csv(f)
    fig_map(c)
    fig_models()
    fig_boroughs()
    print("wrote 11_part2_map.png, 12_part2_three_models.png, 13_part2_no_borough_worse.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
