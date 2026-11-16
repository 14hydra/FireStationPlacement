#!/usr/bin/env python3
"""Project significance + novelty infographic for the ISEF final-figures pack.

Leads with what is easy to defend as new:
  1) a promise written before 2026 that then PASSed
  2) the recorded “missing local engine” delay
Part 2 savings and equity stay as proof. NYC is a small external note.

Writes ``data/figures/final/14_project_significance.png``.
Also writes ``data/figures/final/15_what_is_new.png`` (plain-English novelty).

Does not modify figures 01–10.

Run:  PYTHONPATH=src .venv/bin/python scripts/build_project_infographic.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "figures" / "final" / "14_project_significance.png"
OUT_NEW = ROOT / "data" / "figures" / "final" / "15_what_is_new.png"

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


def build_significance() -> None:
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

    _panel(ax, 0.35, 9.35, 16.3, 1.35, ec=BLUE, fc="#f7faff", lw=1.6, rs=0.06)
    _bar(ax, 0.35, 9.35, 0.18, 1.35, BLUE)
    ax.text(0.75, 10.35, "EMV Route Optimization", fontsize=28, fontweight="bold", color=INK, va="center")
    ax.text(
        0.75,
        9.72,
        "What’s new: we made a promise before seeing 2026 — and we found a hidden delay in real calls",
        fontsize=13,
        color=MUTED,
        va="center",
    )
    ax.text(16.3, 10.35, "CSEF / ISEF", fontsize=12, fontweight="bold", color=BLUE, ha="right", va="center")
    ax.text(16.3, 9.72, "Zhao & Akalin · ASI", fontsize=11, color=MUTED, ha="right", va="center")

    # Three novelty panels (plain English)
    _panel(ax, 0.35, 6.85, 5.3, 2.25, ec=ORANGE, lw=1.5)
    _bar(ax, 0.35, 8.75, 5.3, 0.35, ORANGE)
    ax.text(0.55, 8.92, "1 · A promise that could fail", fontsize=13, fontweight="bold", color="white", va="center")
    ax.text(
        0.55,
        8.55,
        "Before using 2026 calls we wrote:\n"
        "move ≤10 stations → help the 5 slowest\n"
        "boroughs by ≥10 seconds, and make no\n"
        "borough ≥1 second worse — under three\n"
        "different travel models.\n"
        "\n"
        "That test PASSed on 2026 data.",
        fontsize=10.5,
        color=INK,
        va="top",
        linespacing=1.28,
    )

    _panel(ax, 5.85, 6.85, 5.5, 2.25, ec=BLUE, lw=1.5)
    _bar(ax, 5.85, 8.75, 5.5, 0.35, BLUE)
    ax.text(6.05, 8.92, "2 · A hidden delay in the records", fontsize=13, fontweight="bold", color="white", va="center")
    ax.text(
        6.05,
        8.55,
        "On about 13% of calls the local station\n"
        "looks free and is closer — but another\n"
        "station arrives first.\n"
        "\n"
        "Those calls take about +148 seconds\n"
        "longer (2024). Moving stations alone\n"
        "cannot fix that fully.",
        fontsize=10.5,
        color=INK,
        va="top",
        linespacing=1.28,
    )

    _panel(ax, 11.55, 6.85, 5.1, 2.25, ec=GREEN, lw=1.5)
    _bar(ax, 11.55, 8.75, 5.1, 0.35, GREEN)
    ax.text(11.75, 8.92, "3 · Streets, not just distance", fontsize=13, fontweight="bold", color="white", va="center")
    ax.text(
        11.75,
        8.55,
        "We learn travel time from the streets\n"
        "fire engines actually drive (signals,\n"
        "turns, 20 mph roads…), then place\n"
        "stations with that model.\n"
        "\n"
        "We also check against London’s real\n"
        "2014 station closures.",
        fontsize=10.5,
        color=INK,
        va="top",
        linespacing=1.28,
    )

    # Metric strip — promise result + gap + equity
    metrics = [
        ("−13.7 s", "slowest boroughs, 2026\n(promise: ≥10 s · PASS)", BLUE),
        ("0 boroughs", "made ≥1 s worse\n(under all 3 models)", GREEN),
        ("+148 s", "extra wait when local\nlooks free but isn’t sent", ORANGE),
        ("−9.6 s", "poorest income fifth\ngains most (Move 10)", TEAL),
    ]
    x0 = 0.35
    for label, sub, color in metrics:
        _panel(ax, x0, 4.95, 3.95, 1.6, ec=color, lw=1.6)
        ax.text(x0 + 0.25, 6.15, label, fontsize=24, fontweight="bold", color=color, va="center")
        ax.text(x0 + 0.25, 5.35, sub, fontsize=11, color=MUTED, va="center")
        x0 += 4.15

    _panel(ax, 0.35, 2.55, 16.3, 2.15, ec=RULE, lw=1.2)
    ax.text(0.55, 4.4, "Simple story for judges", fontsize=13, fontweight="bold", color=INK, va="center")
    steps = [
        ("Learn streets", "travel times from\nreal LFB trips"),
        ("Check reality", "2014 closures +\nbusy / dispatch"),
        ("Make a promise", "targets + pass marks\nbefore 2026"),
        ("Test once", "2026 PASS\n−13.7 / −16.1 / −13.0 s"),
        ("Say the limit", "missing-engine gap\n~13% of calls, +148 s"),
    ]
    sx = 0.7
    for i, (title, body) in enumerate(steps):
        ec = GREEN if "PASS" in body or title == "Test once" else (ORANGE if i == 4 else (BLUE if i % 2 == 0 else TEAL))
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

    _panel(ax, 0.35, 0.35, 16.3, 1.95, ec=INK, fc="#15202b", lw=0)
    ax.text(0.7, 1.85, "Bottom line", fontsize=12, fontweight="bold", color="#93c5fd", va="center")
    ax.text(
        0.7,
        1.15,
        "What’s new is not “placing fire stations.” It’s learning from real streets, checking on a real closure,\n"
        "writing a pass/fail promise before the 2026 data, and showing a recorded delay when the local engine\n"
        "looks free but isn’t sent. Targeted Move-10 helps the slowest boroughs by 13–16 s with no borough worse.\n"
        "Gains are still modelled — no station has moved yet.",
        fontsize=11.5,
        color="#f8fafc",
        va="center",
        linespacing=1.4,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", OUT)


def build_what_is_new() -> None:
    """One-page plain-English novelty card for the board."""
    fig = plt.figure(figsize=(11, 8.5), dpi=160, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.set_axis_off()

    _panel(ax, 0.35, 7.35, 10.3, 0.85, ec=BLUE, fc="#f7faff", lw=1.6)
    ax.text(0.55, 7.9, "What is new about this project?", fontsize=20, fontweight="bold", color=INK, va="center")
    ax.text(0.55, 7.55, "Say this in the interview (plain words)", fontsize=11, color=MUTED, va="center")

    items = [
        (
            ORANGE,
            "We made a bet before looking",
            "We wrote the target boroughs and the pass marks first, then ran 2026 once.\n"
            "It passed: about 13–16 seconds faster in the slowest boroughs, no borough ≥1 s worse.",
        ),
        (
            BLUE,
            "We found a hidden wait in real calls",
            "About 13% of the time the local station looks free and closer, but another station arrives first.\n"
            "Those calls are roughly 2½ minutes slower (+148 s). Moving stations cannot fix all of that.",
        ),
        (
            GREEN,
            "We use streets, then check the past",
            "Travel times come from street traits on real fire trips — not distance alone.\n"
            "We also check against London’s real 2014 station closures and who really gets sent.",
        ),
        (
            TEAL,
            "What is NOT the new part",
            "Placing stations on a map is old. A slightly better error score alone is not the story.\n"
            "The story is: promise → test → limit we found in the records.",
        ),
    ]
    y = 6.35
    for color, title, body in items:
        _panel(ax, 0.35, y - 1.15, 10.3, 1.25, ec=color, lw=1.4)
        _bar(ax, 0.35, y - 1.15, 0.14, 1.25, color)
        ax.text(0.7, y - 0.15, title, fontsize=13, fontweight="bold", color=INK, va="center")
        ax.text(0.7, y - 0.7, body, fontsize=11, color=MUTED, va="center", linespacing=1.35)
        y -= 1.45

    OUT_NEW.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_NEW, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", OUT_NEW)


def main() -> int:
    build_significance()
    build_what_is_new()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
