#!/usr/bin/env python3
"""Figures for docs/results.md that no other script draws.

  placement/station_sweep.png   saving vs number of stations added or moved (2025 calls,
                                learned dispatch, all three travel time models, 95 % intervals)
  validation/dispatch/rank_shares.png
                                which station sends the first engine: recorded 2025 shares
                                by rank vs the dispatch model and the nearest-station rules

Run:  PYTHONPATH=src .venv/bin/python scripts/plot_results_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "data" / "figures"
SCORERS = {"lgbm": ("LightGBM", "o"), "road": ("Kolesar, road km", "s"), "kolesar": ("Kolesar, straight line", "^")}


def station_sweep() -> None:
    d = pd.read_csv(FIG / "placement" / "layout_scoring" / "scores.csv")
    d = d[(d["response"] == "dispatch") & d["layout"].str.fullmatch(r"lgbm_(expand|replace)_\d+_boroughs1s")]
    d = d.assign(mode=d["layout"].str.extract(r"lgbm_(expand|replace)_")[0],
                 n=d["layout"].str.extract(r"_(\d+)_boroughs")[0].astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (mode, title) in zip(axes, [("expand", "Add stations (none close)"),
                                         ("replace", "Move existing stations")]):
        for sc, (label, mk) in SCORERS.items():
            s = d[(d["mode"] == mode) & (d["scorer"] == sc)].sort_values("n")
            if s.empty:
                continue
            ax.errorbar(s["n"], -s["d_mean_s"], yerr=[s["d_mean_hi"] - s["d_mean_s"], s["d_mean_s"] - s["d_mean_lo"]],
                        marker=mk, capsize=3, label=label)
        ax.set_xlabel("stations " + ("added" if mode == "expand" else "moved (up to)"))
        ax.set_title(title, fontsize=10)
        ax.set_xticks(sorted(d.loc[d["mode"] == mode, "n"].unique()))
        ax.axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("average first-engine time saved (s)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Saving on 2025 calls vs number of stations (pumps fixed at 142; every borough within 1 s)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "placement" / "station_sweep.png", dpi=150)
    plt.close(fig)


def rank_shares() -> None:
    r = pd.read_csv(FIG / "validation" / "dispatch" / "rank_shares.csv")
    r = r[r["rank"].astype(str).str.fullmatch(r"\d+")].astype({"rank": int}).head(5)
    cols = [("recorded", "recorded (LFB 2025)"), ("dispatch_all_day", "dispatch model"),
            ("nearest_or_next_if_busy", "nearest, or next if busy")]
    x = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (c, label) in enumerate(cols):
        ax.bar(x + (i - 1) * 0.27, 100 * r[c], 0.27, label=label)
    ax.set_xticks(x, [f"{k}{'st' if k == 1 else 'nd' if k == 2 else 'rd' if k == 3 else 'th'} fastest"
                      for k in r["rank"]])
    ax.set_ylabel("share of first engines (%)")
    ax.set_title("Which station sends the first engine (2025 calls, model trained 2021-24)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "validation" / "dispatch" / "rank_shares.png", dpi=150)
    plt.close(fig)


def main() -> int:
    station_sweep()
    rank_shares()
    print("Wrote station_sweep.png, rank_shares.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
