#!/usr/bin/env python3
"""Project significance infographic for the ISEF final-figures pack.

Leads with Part 2 (slowest boroughs faster, none worse — 2026 PASS), equity,
and the NYC FDNY weak-label external check. Method MAE stays as support.

Writes ``data/figures/final/14_project_significance.png``.
Does not modify figures 01–10.

Run:  PYTHONPATH=src .venv/bin/python scripts/build_project_infographic.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "figures" / "final" / "14_project_significance.png"
NYC = ROOT / "data" / "figures" / "validation" / "nyc_fdny" / "summary.json"

INK = "#15202b"
MUTED = "#5b6570"
RULE = "#d7dee7"
PANEL = "#ffffff"
BG = "#eef2f6"
BLUE = "#1d5bbf"
ORANGE = "#c2410c"
GREEN = "#15803d"
TEAL = "#0f766e"


def _panel(ax, x, y, w, h, *, ec=RULE, fc=PANEL, lw=1.2, rs=0.04):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={rs}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            clip_on=False,
        )
    )


def _bar(ax, x, y, w, h, color):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none", clip_on=False))


def _nyc_tile() -> tuple[str, str]:
    if NYC.exists():
        h = json.loads(NYC.read_text()).get("headline") or {}
        mae = h.get("nyc_refit_mae_s")
        corr = h.get("nyc_refit_corr")
        if mae is not None and corr is not None:
            return (f"r = {corr:.2f}", f"NYC FDNY fires (weak labels)\nMAE {mae:.0f} s · dist+hour+boro")
    return ("External", "NYC FDNY open CAD\nweak-label check")


def main() -> int:
    fig = plt.figure(figsize=(17, 11), dpi=160, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 11)
    ax.set_axis_off()
    ax.set_facecolor(BG)

    ax.imshow(
        [[0.0, 0.12], [0.22, 0.04]],
        extent=[0, 17, 0, 11],
        origin="lower",
        cmap="Blues",
        alpha=0.16,
        aspect="auto",
        interpolation="bicubic",
        zorder=0,
    )

    # Header
    _panel(ax, 0.35, 9.35, 16.3, 1.35, ec=BLUE, fc="#f7faff", lw=1.6, rs=0.06)
    _bar(ax, 0.35, 9.35, 0.18, 1.35, BLUE)
    ax.text(
        0.75,
        10.35,
        "EMV Route Optimization",
        fontsize=28,
        fontweight="bold",
        color=INK,
        va="center",
    )
    ax.text(
        0.75,
        9.72,
        "Part 2: can a few moves speed London’s slowest boroughs — with no borough worse?",
        fontsize=13,
        color=MUTED,
        va="center",
    )
    ax.text(16.3, 10.35, "CSEF / ISEF", fontsize=12, fontweight="bold", color=BLUE, ha="right", va="center")
    ax.text(16.3, 9.72, "Zhao & Akalin", fontsize=11, color=MUTED, ha="right", va="center")

    # Three story panels
    _panel(ax, 0.35, 6.85, 5.3, 2.25, ec=ORANGE, lw=1.5)
    _bar(ax, 0.35, 8.75, 5.3, 0.35, ORANGE)
    ax.text(0.55, 8.92, "The question that can fail", fontsize=13, fontweight="bold", color="white", va="center")
    ax.text(
        0.55,
        8.55,
        "Targets fixed first from 2024 records:\n"
        "Hillingdon, Havering, Enfield,\n"
        "Bromley, Redbridge.\n"
        "\n"
        "Pass marks written before 2026 was used:\n"
        "≥10 s faster in targets · no borough\n"
        "≥1 s worse · under all three models.",
        fontsize=10.5,
        color=INK,
        va="top",
        linespacing=1.32,
    )

    _panel(ax, 5.85, 6.85, 5.5, 2.25, ec=BLUE, lw=1.5)
    _bar(ax, 5.85, 8.75, 5.5, 0.35, BLUE)
    ax.text(6.05, 8.92, "How we check it", fontsize=13, fontweight="bold", color="white", va="center")
    ax.text(
        6.05,
        8.55,
        "Street-aware LightGBM travel times\n"
        "(45.5 s first-engine MAE vs 53.8 s classic)\n"
        "+ busy engines + learned dispatch\n"
        "+ fairness rules on every borough.\n"
        "\n"
        "Score under LightGBM, road Kolesar,\n"
        "and straight-line Kolesar — a layout\n"
        "counts only if it wins on all three.",
        fontsize=10.5,
        color=INK,
        va="top",
        linespacing=1.32,
    )

    _panel(ax, 11.55, 6.85, 5.1, 2.25, ec=GREEN, lw=1.5)
    _bar(ax, 11.55, 8.75, 5.1, 0.35, GREEN)
    ax.text(11.75, 8.92, "Why it matters", fontsize=13, fontweight="bold", color="white", va="center")
    ax.text(
        11.75,
        8.55,
        "People who wait longest get help;\n"
        "no borough’s average is sacrificed.\n"
        "\n"
        "Checked on a pre-registered 2026\n"
        "holdout — and on the only other open\n"
        "megacity CAD we found (NYC FDNY),\n"
        "with the nearest-station limit stated.",
        fontsize=10.5,
        color=INK,
        va="top",
        linespacing=1.32,
    )

    # Metric strip — Part 2 first
    nyc_label, nyc_sub = _nyc_tile()
    metrics = [
        ("−13.7 s", "slowest boroughs, 2026\n(also −16.1 / −13.0 Kolesar)", BLUE),
        ("0 boroughs", "≥1 s worse under any\nof the three models", GREEN),
        ("−9.6 s", "poorest income fifth\n(Move 10, LightGBM)", ORANGE),
        (nyc_label, nyc_sub, TEAL),
    ]
    x0 = 0.35
    for label, sub, color in metrics:
        _panel(ax, x0, 4.95, 3.95, 1.6, ec=color, lw=1.6)
        ax.text(x0 + 0.25, 6.15, label, fontsize=24, fontweight="bold", color=color, va="center")
        ax.text(x0 + 0.25, 5.35, sub, fontsize=11, color=MUTED, va="center")
        x0 += 4.15

    # Evidence pipeline
    _panel(ax, 0.35, 2.55, 16.3, 2.15, ec=RULE, lw=1.2)
    ax.text(0.55, 4.4, "Evidence chain judges can follow", fontsize=13, fontweight="bold", color=INK, va="center")
    steps = [
        ("Fix targets", "slowest 5 boroughs\nfrom 2024 records"),
        ("Search moves", "≤10 stations ·\nRule B land filter"),
        ("Three models", "LightGBM + both\nKolesar formulas"),
        ("2026 PASS", "−13.7 / −16.1 / −13.0 s\nworst borough +0.4 s"),
        ("NYC check", "open FDNY CAD\nr = 0.24 weak labels"),
    ]
    sx = 0.7
    for i, (title, body) in enumerate(steps):
        ec = GREEN if title == "2026 PASS" else (BLUE if i % 2 == 0 else TEAL)
        _panel(ax, sx, 2.8, 2.7, 1.35, ec=ec, lw=1.3, rs=0.05)
        ax.text(sx + 1.35, 3.85, title, fontsize=11, fontweight="bold", color=INK, ha="center", va="center")
        ax.text(sx + 1.35, 3.25, body, fontsize=9.5, color=MUTED, ha="center", va="center")
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(sx + 2.95, 3.45),
                xytext=(sx + 2.75, 3.45),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4),
            )
        sx += 3.15

    # Bottom line
    _panel(ax, 0.35, 0.35, 16.3, 1.95, ec=INK, fc="#15202b", lw=0)
    ax.text(0.7, 1.85, "Bottom line", fontsize=12, fontweight="bold", color="#93c5fd", va="center")
    ax.text(
        0.7,
        1.15,
        "A pre-registered Move-10 layout speeds London’s five slowest boroughs by 13–16 seconds under every\n"
        "travel model, with no borough ≥1 s worse — confirmed on 2026 holdout calls. Poorest neighbourhoods gain\n"
        "most. Outside London, open CAD is rare; NYC FDNY gives a weak-label distance–time check (r ≈ 0.24),\n"
        "limited by the missing responding company. Gains are still modelled; no station has moved yet.",
        fontsize=11.5,
        color="#f8fafc",
        va="center",
        linespacing=1.4,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
