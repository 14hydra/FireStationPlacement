#!/usr/bin/env python3
"""Draw the method diagram: data -> travel time -> busy engines -> solver -> validation.

The numbers are read from the saved results, so they stay in step with the figures.

Writes data/figures/final/method_diagram.png.

Run:  .venv/bin/python scripts/build_method_diagram.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "data" / "figures"
OUT = FIG / "final" / "method_diagram.png"

INK, MUTED, GRID = "#15202b", "#5b6570", "#e6eaef"
BLUE, ORANGE, GREEN, GREY = "#1d5bbf", "#d97706", "#15803d", "#94a3b8"

BOX_W, BOX_H, GAP, TOP = 3.3, 4.5, 0.55, 5.4


def steps() -> list[tuple[str, str, list[str]]]:
    pl = json.loads((FIG / "validation/travel_time_error/summary.json").read_text())["compare"]["placement"]
    return [
        ("1  Data", GREY, [
            "LFB incident and\nmobilisation records",
            "OpenStreetMap roads",
            "Hourly weather",
        ]),
        ("2  Travel time", BLUE, [
            "Legal route from each\nstation to each call",
            "Street features along\nthe route",
            f"LightGBM drive time:\n{pl['model_mae_s']:.1f} s error vs\n{pl['kolesar_mae_s']:.1f} s classic formula",
        ]),
        ("3  Busy engines", ORANGE, [
            "Erlang chance each\nstation is busy, from\nits pumps and call load",
            "Dispatch model, to score\nlayouts: which of the 8\nfastest open stations\nsends the first engine",
        ]),
        ("4  Solver", GREEN, [
            "p-median integer\nprogram (HiGHS, exact)",
            "Genetic algorithm if no\nlayout is found in time",
            "Minimises average first-\nengine time on 2024\ncalls, with borough limits",
        ]),
        ("5  Validation", INK, [
            "Layouts vs today on\nheld-out 2025 calls",
            "Three travel time\nmodels",
            "2014 closures",
            "Stations never seen\nin training",
            "Time of day, severity,\nborough, income",
        ]),
    ]


def main() -> int:
    items = steps()
    fig, ax = plt.subplots(figsize=(16, 6.2))
    ax.set_axis_off()
    width = len(items) * BOX_W + (len(items) - 1) * GAP
    ax.set_xlim(-0.1, width + 0.1)
    ax.set_ylim(0.2, TOP + 0.75)
    ax.text(0, TOP + 0.45, "How the project works: from LFB records to tested station layouts",
            fontsize=18, fontweight="bold", color=INK, va="center")

    for i, (title, color, lines) in enumerate(items):
        x = i * (BOX_W + GAP)
        y0 = TOP - BOX_H
        ax.add_patch(FancyBboxPatch((x, y0), BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor="white", edgecolor=color, linewidth=1.8))
        ax.add_patch(FancyBboxPatch((x, TOP - 0.62), BOX_W, 0.62, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor=color, edgecolor=color, linewidth=1.8))
        ax.text(x + BOX_W / 2, TOP - 0.31, title, ha="center", va="center", fontsize=14,
                fontweight="bold", color="white")
        # Evenly spaced lines, sized by how many text rows each has.
        rows = [t.count("\n") + 1 for t in lines]
        row_h = 0.24
        spare = (BOX_H - 0.62 - sum(rows) * row_h) / (len(lines) + 1)
        cy = TOP - 0.62
        for text, n in zip(lines, rows):
            cy -= spare
            ax.text(x + BOX_W / 2, cy - n * row_h / 2, text, ha="center", va="center", fontsize=11.5,
                    color=INK, linespacing=1.25)
            cy -= n * row_h
        if i < len(items) - 1:
            ax.add_patch(FancyArrowPatch((x + BOX_W + 0.06, y0 + BOX_H / 2), (x + BOX_W + GAP - 0.06, y0 + BOX_H / 2),
                                         arrowstyle="-|>", mutation_scale=20, linewidth=2, color=MUTED))

    ax.text(width / 2, 0.42, "Travel time model trained on 2021–24 trips   ·   layouts chosen on 2024 calls   ·   "
            "layouts scored on 2025 calls",
            ha="center", va="center", fontsize=12, color=MUTED)
    ax.plot([0, width], [0.78, 0.78], color=GRID, linewidth=1)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
