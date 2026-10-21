#!/usr/bin/env python3
"""
Assemble the 10 most important project figures into data/figures/final/.

Copies (does not move) the source PNGs, renames them for a clear story order,
and writes INDEX.md with one-line takeaways for ISEF judges.

Also draws 00_roadmap.png — a one-page “what to look at” board.

Run:  .venv/bin/python scripts/build_final_figures.py
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "data" / "figures"
OUT = FIG / "final"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#256abf", "#eb6834"

# Story order for judges. Paths are relative to data/figures/.
FIGURES = [
    {
        "num": "01",
        "name": "travel_time_accuracy",
        "src": "reports/weekly_2026-10-19/1_travel_time_accuracy.png",
        "title": "Our travel-time model beats the classic formula",
        "glance": "LightGBM error 57.6 s vs Kolesar 68.7 s on held-out 2025 trips.",
        "say": "Street-aware trees beat the Kolesar distance→time curve on real 2025 calls.",
    },
    {
        "num": "02",
        "name": "street_features",
        "src": "data_exploration/london_eda/08_street_shap.png",
        "title": "Which street traits slow fire engines",
        "glance": "SHAP ranking of route features the model uses (signals, junctions, road class).",
        "say": "The model is not a black box — these street features drive its predictions.",
    },
    {
        "num": "03",
        "name": "first_engine_map",
        "src": "travel_time/maps/travel_time_maps.png",
        "title": "Predicted first-engine time across London",
        "glance": "City map of how long the nearest engine takes to arrive under our model.",
        "say": "This is what we optimise when we move or add stations.",
    },
    {
        "num": "04",
        "name": "station_busyness",
        "src": "data_exploration/station_busyness/station_busyness.png",
        "title": "Central stations’ engines are out most often",
        "glance": "Utilisation by station — inner London is stretched; outer London is quieter.",
        "say": "Capacity and demand are uneven — placement alone is not the whole story.",
    },
    {
        "num": "05",
        "name": "capacity_mismatches",
        "src": "data_exploration/station_capacity/05_capacity_mismatches.png",
        "title": "Where to add or free up an engine",
        "glance": "Busy 1-pump houses (Euston…) vs quieter 2-pump houses (Addington…).",
        "say": "A fixed pump budget can be reallocated along this mismatch.",
    },
    {
        "num": "06",
        "name": "station_types",
        "src": "analysis/station_embedding/06_isef_story.png",
        "title": "102 stations → five types at a glance",
        "glance": "Map + busy×capacity picture + what it means for the model and resize.",
        "say": "One poster that ties busyness, capacity, and model hard cases together.",
    },
    {
        "num": "07",
        "name": "layouts_beat_today",
        "src": "reports/weekly_2026-10-19/3_three_models.png",
        "title": "Better layouts still win under all three scorers",
        "glance": "Seconds saved vs today’s stations under LightGBM, Kolesar road, Kolesar crow.",
        "say": "Placement gains are not an artifact of one travel-time model.",
    },
    {
        "num": "08",
        "name": "price_of_fairness",
        "src": "reports/weekly_2026-10-19/2_price_of_fairness.png",
        "title": "Efficiency vs equal coverage",
        "glance": "The α tradeoff curve — how much average time you give up for fairness.",
        "say": "We can choose how hard to equalise boroughs; the cost is explicit.",
    },
    {
        "num": "09",
        "name": "closure_2014",
        "src": "reports/weekly_2026-10-19/4_closure_2014.png",
        "title": "2014 ten-station closure test",
        "glance": "Nearest vs learned dispatch vs oracle — which predicts the real 2014 slowdown.",
        "say": "Dispatch learning closes much of the gap between ‘assume nearest’ and the oracle.",
    },
    {
        "num": "10",
        "name": "results_summary",
        "src": "reports/weekly_2026-10-19/6_summary_table.png",
        "title": "Numbers on one page",
        "glance": "Compact results table for the board / abstract.",
        "say": "If a judge only has 30 seconds, show this after figure 01 and 07.",
    },
]


def draw_roadmap(path: Path) -> None:
    """At-a-glance board: 10 boxes with titles + one-line takeaways."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
    })
    fig = plt.figure(figsize=(14, 10.5))
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax.axis("off")
    ax.text(0.0, 0.98, "Final figures — read in order", fontsize=20, fontweight="bold",
            color=INK, transform=ax.transAxes, va="top")
    ax.text(0.0, 0.935,
            "London fire-station placement · street-aware travel time · ISEF board pack",
            fontsize=11, color=INK2, transform=ax.transAxes, va="top")

    # 2 columns × 5 rows
    for i, fig_meta in enumerate(FIGURES):
        col, row = i % 2, i // 2
        x0 = 0.0 if col == 0 else 0.52
        y0 = 0.82 - row * 0.155
        # card background
        ax.add_patch(plt.Rectangle((x0, y0 - 0.12), 0.46, 0.13, transform=ax.transAxes,
                                   facecolor="#f4f3ef", edgecolor=GRID, linewidth=1.0,
                                   clip_on=False))
        ax.text(x0 + 0.015, y0 + 0.005, f"{fig_meta['num']}", fontsize=16, fontweight="bold",
                color=BLUE, transform=ax.transAxes, va="top")
        ax.text(x0 + 0.07, y0 + 0.005, fig_meta["title"], fontsize=11, fontweight="bold",
                color=INK, transform=ax.transAxes, va="top")
        wrapped = textwrap.fill(fig_meta["glance"], width=48)
        ax.text(x0 + 0.07, y0 - 0.045, wrapped, fontsize=9, color=INK2,
                transform=ax.transAxes, va="top", linespacing=1.35)
        ax.text(x0 + 0.015, y0 - 0.105, f"{fig_meta['num']}_{fig_meta['name']}.png",
                fontsize=7.5, color=ORANGE, transform=ax.transAxes, va="bottom",
                fontfamily="monospace")

    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_index(copied: list[dict]) -> None:
    lines = [
        "# Final figures (ISEF board pack)",
        "",
        "Ten figures that tell the whole project. **Open `00_roadmap.png` first**, then 01→10.",
        "",
        "| # | File | At a glance |",
        "|---|---|---|",
    ]
    for f in copied:
        lines.append(f"| {f['num']} | `{f['num']}_{f['name']}.png` | **{f['title']}** — {f['glance']} |")
    lines += [
        "",
        "## What to say (30-second version)",
        "",
    ]
    for f in copied:
        lines.append(f"{f['num']}. {f['say']}")
    lines += [
        "",
        "## Sources",
        "",
        "These are copies of figures elsewhere under `data/figures/`. Regenerate with:",
        "",
        "```bash",
        ".venv/bin/python scripts/build_final_figures.py",
        "```",
        "",
        "Related: `analysis/station_embedding/ISEF_GUIDE.md`, `docs/model_improvements_from_embedding.md`.",
        "",
    ]
    (OUT / "INDEX.md").write_text("\n".join(lines))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Clear previous numbered copies (keep folder)
    for p in OUT.glob("*.png"):
        p.unlink()
    for p in OUT.glob("*.md"):
        if p.name in {"INDEX.md", "README.md"}:
            p.unlink()

    copied = []
    missing = []
    for f in FIGURES:
        src = FIG / f["src"]
        dst = OUT / f"{f['num']}_{f['name']}.png"
        if not src.exists():
            missing.append(str(src.relative_to(ROOT)))
            continue
        shutil.copy2(src, dst)
        copied.append(f)
        print(f"  {dst.name}  ←  {f['src']}")

    draw_roadmap(OUT / "00_roadmap.png")
    write_index(copied)
    (OUT / "README.md").write_text(
        "# Final figures\n\n"
        "See [INDEX.md](INDEX.md). Start with **00_roadmap.png**, then **01…10**.\n"
    )
    if missing:
        print("MISSING:")
        for m in missing:
            print(f"  {m}")
        return 1
    print(f"wrote {OUT.relative_to(ROOT)}/ ({len(copied)} figures + roadmap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
