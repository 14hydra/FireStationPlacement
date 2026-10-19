#!/usr/bin/env python3
"""
Station capacity vs incident density figures (London LFB).

Builds per-station pump counts (inferred two-pump houses + named LFB list),
joins 2024–2025 incident demand by station ground, and writes charts + CSV.

Run:
  PYTHONPATH=src .venv/bin/python scripts/analyze_station_capacity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.station_capacity import (  # noqa: E402
    LFB_NAMED_TWO_PUMP,
    default_pump_table,
    infer_two_pump_stations,
    station_incident_load,
)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#f7f5f1",
    "figure.facecolor": "#f7f5f1",
    "axes.edgecolor": "#2c2c2c",
    "axes.grid": True,
    "grid.color": "#d9d4cb",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
INK = "#1f3a5f"
ACCENT = "#c45c26"
OK = "#2f6f4e"
MUTED = "#6b7280"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "figures" / "data_exploration" / "station_capacity")
    ap.add_argument("--demand-years", type=str, default="2024,2025")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    years = [int(x) for x in args.demand_years.replace(" ", "").split(",") if x]
    spec = get_city("london")
    houses = load_fire_stations(spec)
    incidents = load_incidents(spec, max_rows=None, years=years)

    pumps = default_pump_table(houses)
    load = station_incident_load(incidents, houses)
    df = pumps.merge(load.drop(columns=["lat", "lon"], errors="ignore"), on="station", how="left")
    df["n_incidents"] = df["n_incidents"].fillna(0).astype(int)
    df["incidents_per_pump"] = df["n_incidents"] / df["n_pumps"].clip(lower=1)
    df["named_two_pump"] = df["station"].isin(LFB_NAMED_TWO_PUMP)
    df = df.sort_values("n_incidents", ascending=False).reset_index(drop=True)

    inferred = infer_two_pump_stations()
    summary = {
        "n_stations": int(len(df)),
        "n_pumps_total": int(df["n_pumps"].sum()),
        "n_two_pump": int((df["n_pumps"] >= 2).sum()),
        "n_named_two_pump_list": len(LFB_NAMED_TWO_PUMP),
        "demand_years": years,
        "n_incidents": int(df["n_incidents"].sum()),
        "mean_incidents_per_pump": round(float(df["incidents_per_pump"].mean()), 1),
        "corr_incidents_vs_pumps": round(float(df["n_incidents"].corr(df["n_pumps"])), 3),
        "honesty": (
            "Pump counts: top ~40 stations by home multi-dispatch in mobilisation "
            "records (matches LFB's ~40 two-pump stations), always including the "
            "named LFB org-chart list. Demand = incidents by station_ground in the "
            "planner extract for the listed years."
        ),
    }
    df.to_csv(args.out_dir / "station_pumps_vs_demand.csv", index=False)
    inferred.to_csv(args.out_dir / "two_pump_inference.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ---- 1. scatter pumps vs incidents ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    one = df[df.n_pumps == 1]
    two = df[df.n_pumps >= 2]
    ax.scatter(one["n_incidents"], np.full(len(one), 1), s=36, c=MUTED, alpha=0.75, label="1 pump", zorder=2)
    ax.scatter(two["n_incidents"], np.full(len(two), 2), s=48, c=ACCENT, alpha=0.85, label="2 pumps", zorder=3)
    # jitter y slightly already discrete; add tiny x-noise? skip
    ax.set_yticks([1, 2])
    ax.set_xlabel(f"Incidents at station ground ({'/'.join(map(str, years))})")
    ax.set_ylabel("Pumps at station")
    ax.set_title("Firetrucks vs incident load — do busier grounds get a second pump?")
    ax.legend(frameon=False, loc="lower right")
    # annotate extremes
    for _, r in df.nlargest(5, "n_incidents").iterrows():
        ax.annotate(r.station, (r.n_incidents, r.n_pumps), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color=INK)
    fig.tight_layout()
    fig.savefig(args.out_dir / "01_pumps_vs_incidents.png", dpi=160)
    plt.close()

    # ---- 2. incidents per pump ranking ----
    top = df.sort_values("incidents_per_pump", ascending=True).tail(25)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [ACCENT if n >= 2 else INK for n in top["n_pumps"]]
    ax.barh(top["station"].astype(str), top["incidents_per_pump"], color=colors)
    ax.set_xlabel("Incidents per pump")
    ax.set_title("Highest load per firetruck (navy=1 pump · terracotta=2 pumps)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "02_incidents_per_pump.png", dpi=160)
    plt.close()

    # ---- 3. distribution comparison ----
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bins = np.linspace(0, df["n_incidents"].quantile(0.98), 24)
    ax.hist(one["n_incidents"], bins=bins, alpha=0.65, color=MUTED, label=f"1 pump (n={len(one)})")
    ax.hist(two["n_incidents"], bins=bins, alpha=0.65, color=ACCENT, label=f"2 pumps (n={len(two)})")
    ax.axvline(one["n_incidents"].median(), color=MUTED, ls="--", lw=1.4)
    ax.axvline(two["n_incidents"].median(), color=ACCENT, ls="--", lw=1.4)
    ax.set_xlabel("Incidents at station ground")
    ax.set_ylabel("Stations")
    ax.set_title("Incident-load distributions by station size")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out_dir / "03_load_by_size_hist.png", dpi=160)
    plt.close()

    # ---- 4. map-like scatter (lon/lat) sized by incidents, colored by pumps ----
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    sizes = 20 + 180 * (df["n_incidents"] / max(df["n_incidents"].max(), 1))
    colors = [ACCENT if n >= 2 else INK for n in df["n_pumps"]]
    ax.scatter(df["lon"], df["lat"], s=sizes, c=colors, alpha=0.8, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Stations: marker size = incident load · colour = pumps (navy=1, terracotta=2)")
    ax.set_aspect("equal", adjustable="box")
    # legend proxies
    ax.scatter([], [], s=60, c=INK, label="1 pump")
    ax.scatter([], [], s=60, c=ACCENT, label="2 pumps")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out_dir / "04_map_pumps_demand.png", dpi=160)
    plt.close()

    # ---- 5. mismatch: high load / 1 pump vs low load / 2 pumps ----
    med = df["n_incidents"].median()
    under = df[(df.n_pumps == 1) & (df.n_incidents >= df.n_incidents.quantile(0.75))].sort_values("n_incidents", ascending=False)
    over = df[(df.n_pumps >= 2) & (df.n_incidents <= df.n_incidents.quantile(0.40))].sort_values("n_incidents")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=False)
    ax = axes[0]
    show = under.head(12).iloc[::-1]
    ax.barh(show["station"].astype(str), show["n_incidents"], color=INK)
    ax.set_title("Busy grounds with only 1 pump")
    ax.set_xlabel("Incidents")
    ax = axes[1]
    show = over.head(12).iloc[::-1]
    ax.barh(show["station"].astype(str), show["n_incidents"], color=ACCENT)
    ax.set_title("2-pump stations with lighter load")
    ax.set_xlabel("Incidents")
    fig.suptitle("Capacity mismatches — candidates to upsize / downsize", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(args.out_dir / "05_capacity_mismatches.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ---- 6. summary card ----
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axis("off")
    lines = [
        f"Stations {summary['n_stations']} · pumps {summary['n_pumps_total']} "
        f"({summary['n_two_pump']} with a second engine)",
        f"Incidents {summary['n_incidents']:,} ({'/'.join(map(str, years))}) · "
        f"mean load/pump {summary['mean_incidents_per_pump']:.0f}",
        f"Correlation(incidents, pumps) = {summary['corr_incidents_vs_pumps']:+.2f}",
        f"1-pump med load {one.n_incidents.median():.0f} · 2-pump med load {two.n_incidents.median():.0f}",
        f"High-load 1-pump stations: {len(under)} · light-load 2-pump: {len(over)}",
    ]
    ax.text(0.02, 0.92, "LFB station capacity vs demand", fontsize=15, fontweight="bold",
            transform=ax.transAxes, va="top", color=INK)
    ax.text(0.02, 0.72, "\n".join(lines), fontsize=11, transform=ax.transAxes, va="top",
            linespacing=1.5, color="#222")
    fig.tight_layout()
    fig.savefig(args.out_dir / "00_summary.png", dpi=160)
    plt.close()

    print(json.dumps(summary, indent=2))
    print("Wrote", args.out_dir)
    for p in sorted(args.out_dir.glob("*.png")):
        print(" ", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
