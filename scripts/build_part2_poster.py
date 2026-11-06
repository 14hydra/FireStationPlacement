#!/usr/bin/env python3
"""Part 2 poster layout: the targeted question first, Part 1's method underneath.

Same convention as build_synopsis_poster.py (which is kept for Part 1): figures in place, empty slots for the title and
captions, which Ricky and Sarp write themselves. Figures come from data/figures/final/ (run plot_part2_figures.py first).

  Row 1  11 map of the main layout (dominant) | 12 three-model chart, 2025 and 2026
  Row 2  13 no borough worse | method diagram | 01 travel time accuracy | 09 2014 closure test

Writes data/figures/part2_poster/part2_poster.png and IMAGE_SOURCES.md.

Run:  .venv/bin/python scripts/build_part2_poster.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_synopsis_poster as bp  # noqa: E402

OUT = bp.ROOT / "data" / "figures" / "part2_poster"
ROW1 = [("11_part2_map.png", "Where calls get faster or slower, main layout (LightGBM)"),
        ("12_part2_three_models.png", "Target-borough gain under three models, 2025 and the 2026 fixed test")]
ROW2 = [("13_part2_no_borough_worse.png", "Every borough's change in 2026: targeted vs planner"),
        ("method_diagram.png", "Method: data, travel time, busy engines, solver, validation"),
        ("01_travel_time_accuracy.png", "Travel time model accuracy (Part 1)"),
        ("09_closure_2014.png", "2014 closures test (Part 1)")]


def main() -> int:
    W, H, m, g = bp.W, bp.H, 90, 36
    bp.canvas = Image.new("RGB", (W, H), bp.BG)
    bp.draw = ImageDraw.Draw(bp.canvas)
    header_h = 260
    bp._slot((m, m, W - m, m + header_h))
    y0 = m + header_h + g

    row1_h = 1900
    left_w = int((W - 2 * m - g) * 0.52)
    fh = int(row1_h * 0.82)
    bp._fit(bp._load(ROW1[0][0]), (m, y0, m + left_w, y0 + fh))
    bp._slot((m, y0 + fh + 8, m + left_w, y0 + row1_h))
    rl = m + left_w + g
    ch = int(row1_h * 0.56)
    bp._fit(bp._load(ROW1[1][0]), (rl, y0, W - m, y0 + ch))
    bp._slot((rl, y0 + ch + 8, W - m, y0 + row1_h))  # big slot: question, hypothesis, pass marks, result table

    y2 = y0 + row1_h + g
    row2_h = H - m - y2
    fourth = (W - 2 * m - 3 * g) // 4
    for i, (name, _) in enumerate(ROW2):
        x = m + i * (fourth + g)
        f2 = int(row2_h * 0.74)
        bp._fit(bp._load(name), (x, y2, x + fourth, y2 + f2))
        bp._slot((x, y2 + f2 + 8, x + fourth, y2 + row2_h))

    OUT.mkdir(parents=True, exist_ok=True)
    bp.canvas.save(OUT / "part2_poster.png", dpi=(bp.DPI, bp.DPI))
    lines = ["# Part 2 poster: image sources", "", "Built by `scripts/build_part2_poster.py`. Grey slots are for text the students write.", "",
             "| Position | Figure | Shows |", "|---|---|---|"]
    for pos, items in (("Row 1", ROW1), ("Row 2", ROW2)):
        lines += [f"| {pos} | `data/figures/final/{n}` | {d} |" for n, d in items]
    lines += ["", "The large slot under the three-model chart is for the question, the pass marks fixed in advance and the 2026 result table",
              "(`docs/results.md` 5.8). Numbers to use are in `docs/notebook_part2.md`."]
    (OUT / "IMAGE_SOURCES.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "part2_poster.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
