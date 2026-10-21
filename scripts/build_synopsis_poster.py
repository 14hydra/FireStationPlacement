#!/usr/bin/env python3
"""Assemble a text-free synopsis poster from the final figure pack.

Layout only: project figures in place, empty slots where title / takeaways /
captions will go later. No synopsis copy is written into the artboard.

Writes:
  data/figures/synopsis_poster/synopsis_poster.html   (print / edit in browser)
  data/figures/synopsis_poster/synopsis_poster.png     (high-res composite)
  data/figures/synopsis_poster/IMAGE_SOURCES.md        (repo paths for every image)

Run:  .venv/bin/python scripts/build_synopsis_poster.py
Requires: pillow (already in the project venv if matplotlib is present).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data" / "figures" / "final"
OUT = ROOT / "data" / "figures" / "synopsis_poster"

# Soft slate atmosphere (avoid purple / cream-terracotta / broadsheet defaults).
BG = (244, 247, 250)
PANEL = (255, 255, 255)
INK_FAINT = (180, 188, 198)
SLOT = (232, 238, 244)
SLOT_EDGE = (168, 178, 190)
RULE = (210, 218, 226)

# Poster: ~36×24 in at 150 dpi (common board aspect).
W, H, DPI = 5400, 3600, 150


def _load(name: str) -> Image.Image:
    path = FINAL / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/build_final_figures.py first")
    return Image.open(path).convert("RGB")


def _fit(img: Image.Image, box: tuple[int, int, int, int], *, cover: bool = False) -> None:
    """Paste ``img`` into ``box`` (l,t,r,b) on the global canvas, letterbox or cover."""
    global canvas
    l, t, r, b = box
    bw, bh = r - l, b - t
    iw, ih = img.size
    scale = max(bw / iw, bh / ih) if cover else min(bw / iw, bh / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    if cover:
        x0 = (nw - bw) // 2
        y0 = (nh - bh) // 2
        resized = resized.crop((x0, y0, x0 + bw, y0 + bh))
        canvas.paste(resized, (l, t))
    else:
        x = l + (bw - nw) // 2
        y = t + (bh - nh) // 2
        # white mat behind letterboxed figures
        draw.rectangle([l, t, r, b], fill=PANEL, outline=RULE, width=2)
        canvas.paste(resized, (x, y))


def _slot(box: tuple[int, int, int, int], *, dashed: bool = True) -> None:
    """Empty text region — light fill + dashed border so writers see the hole."""
    l, t, r, b = box
    draw.rectangle([l, t, r, b], fill=SLOT, outline=SLOT_EDGE, width=2)
    if not dashed:
        return
    # Corner ticks only (no words) so the slot reads as “write here”.
    tick = 28
    for x, y, dx, dy in (
        (l, t, 1, 1), (r, t, -1, 1), (l, b, 1, -1), (r, b, -1, -1),
    ):
        draw.line([(x, y), (x + dx * tick, y)], fill=INK_FAINT, width=3)
        draw.line([(x, y), (x, y + dy * tick)], fill=INK_FAINT, width=3)


def build_png() -> Path:
    global canvas, draw
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    m = 90  # outer margin
    g = 36  # gutter

    # --- Header: brand / title / authors (empty) ---
    header_h = 220
    _slot((m, m, W - m, m + header_h))
    # Split hint: left = title, right = authors (invisible divider)
    mid = int(W * 0.68)
    draw.line([(mid, m + 40), (mid, m + header_h - 40)], fill=RULE, width=2)

    y0 = m + header_h + g

    # --- Row 1: accuracy + street features | dominant map ---
    row1_h = 1180
    left_w = int((W - 2 * m - g) * 0.38)
    right_l = m + left_w + g
    # left column: two stacked figures + text under each
    col_h = (row1_h - g) // 2
    fig_h = int(col_h * 0.72)
    _fit(_load("01_travel_time_accuracy.png"), (m, y0, m + left_w, y0 + fig_h))
    _slot((m, y0 + fig_h + 8, m + left_w, y0 + col_h))
    y1b = y0 + col_h + g
    _fit(_load("02_street_features.png"), (m, y1b, m + left_w, y1b + fig_h))
    _slot((m, y1b + fig_h + 8, m + left_w, y1b + col_h))

    # map dominates right
    map_h = int(row1_h * 0.78)
    _fit(_load("03_first_engine_map.png"), (right_l, y0, W - m, y0 + map_h), cover=False)
    _slot((right_l, y0 + map_h + 8, W - m, y0 + row1_h))

    y2 = y0 + row1_h + g

    # --- Row 2: busyness | capacity | station types ---
    row2_h = 780
    third = (W - 2 * m - 2 * g) // 3
    for i, name in enumerate(("04_station_busyness.png", "05_capacity_mismatches.png",
                              "06_station_types.png")):
        x = m + i * (third + g)
        fh = int(row2_h * 0.70)
        _fit(_load(name), (x, y2, x + third, y2 + fh))
        _slot((x, y2 + fh + 8, x + third, y2 + row2_h))

    y3 = y2 + row2_h + g

    # --- Row 3: layouts | fairness | closure | numbers ---
    row3_h = H - m - y3
    fourth = (W - 2 * m - 3 * g) // 4
    for i, name in enumerate(("07_layouts_beat_today.png", "08_price_of_fairness.png",
                              "09_closure_2014.png", "10_results_summary.png")):
        x = m + i * (fourth + g)
        fh = int(row3_h * 0.68)
        _fit(_load(name), (x, y3, x + fourth, y3 + fh))
        _slot((x, y3 + fh + 8, x + fourth, y3 + row3_h))

    out = OUT / "synopsis_poster.png"
    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(out, dpi=(DPI, DPI))
    return out


def build_html() -> Path:
    """Editable / printable HTML twin — same slots, relative image links."""
    # Paths relative to the HTML file (synopsis_poster/ → ../final/)
    rel = "../final"

    def fig(src: str, alt: str) -> str:
        return f'<div class="fig"><img src="{rel}/{src}" alt="{alt}"></div>'

    def slot(cls: str = "") -> str:
        return f'<div class="slot {cls}"></div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Synopsis poster (images + text slots)</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500&family=Source+Sans+3:wght@400;600&display=swap');
  :root {{
    --bg: #f4f7fa;
    --panel: #ffffff;
    --slot: #e8eef4;
    --edge: #a8b2be;
    --rule: #d2dae2;
    --ink: #1a2330;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; background: #c5ccd4; }}
  .board {{
    width: 17in;
    height: 11in;
    margin: 0.4in auto;
    padding: 0.35in;
    background:
      radial-gradient(120% 80% at 80% 10%, #e8eef5 0%, transparent 55%),
      linear-gradient(165deg, #f7f9fb 0%, var(--bg) 45%, #eef2f6 100%);
    display: grid;
    grid-template-rows: 0.9in 1fr 1fr 1fr;
    gap: 0.18in;
    box-shadow: 0 12px 40px rgba(20, 30, 40, 0.18);
  }}
  .header {{
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 0.18in;
  }}
  .row1 {{
    display: grid;
    grid-template-columns: 0.9fr 1.35fr;
    gap: 0.18in;
    min-height: 0;
  }}
  .stack {{
    display: grid;
    grid-template-rows: 1fr 1fr;
    gap: 0.18in;
    min-height: 0;
  }}
  .pair {{
    display: grid;
    grid-template-rows: 1fr auto;
    gap: 0.08in;
    min-height: 0;
  }}
  .map-block {{
    display: grid;
    grid-template-rows: 1fr auto;
    gap: 0.08in;
    min-height: 0;
  }}
  .row2, .row3 {{
    display: grid;
    gap: 0.18in;
    min-height: 0;
  }}
  .row2 {{ grid-template-columns: repeat(3, 1fr); }}
  .row3 {{ grid-template-columns: repeat(4, 1fr); }}
  .fig {{
    background: var(--panel);
    border: 1px solid var(--rule);
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
  }}
  .fig img {{
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
  }}
  .slot {{
    background: var(--slot);
    border: 1.5px dashed var(--edge);
    border-radius: 2px;
    min-height: 0.55in;
    position: relative;
  }}
  .slot.hero {{ min-height: 100%; }}
  /* corner ticks — no text */
  .slot::before, .slot::after {{
    content: "";
    position: absolute;
    width: 10px; height: 10px;
    border-color: var(--edge);
    border-style: solid;
  }}
  .slot::before {{ top: 4px; left: 4px; border-width: 2px 0 0 2px; }}
  .slot::after {{ bottom: 4px; right: 4px; border-width: 0 2px 2px 0; }}
  @media print {{
    body {{ background: white; }}
    .board {{
      margin: 0; box-shadow: none;
      width: 100%; height: 100vh;
      page-break-inside: avoid;
    }}
  }}
</style>
</head>
<body>
  <div class="board">
    <div class="header">
      {slot("hero")}
      {slot("hero")}
    </div>
    <div class="row1">
      <div class="stack">
        <div class="pair">
          {fig("01_travel_time_accuracy.png", "01 travel time accuracy")}
          {slot()}
        </div>
        <div class="pair">
          {fig("02_street_features.png", "02 street features")}
          {slot()}
        </div>
      </div>
      <div class="map-block">
        {fig("03_first_engine_map.png", "03 first engine map")}
        {slot()}
      </div>
    </div>
    <div class="row2">
      <div class="pair">
        {fig("04_station_busyness.png", "04 station busyness")}
        {slot()}
      </div>
      <div class="pair">
        {fig("05_capacity_mismatches.png", "05 capacity mismatches")}
        {slot()}
      </div>
      <div class="pair">
        {fig("06_station_types.png", "06 station types")}
        {slot()}
      </div>
    </div>
    <div class="row3">
      <div class="pair">
        {fig("07_layouts_beat_today.png", "07 layouts")}
        {slot()}
      </div>
      <div class="pair">
        {fig("08_price_of_fairness.png", "08 price of fairness")}
        {slot()}
      </div>
      <div class="pair">
        {fig("09_closure_2014.png", "09 closure 2014")}
        {slot()}
      </div>
      <div class="pair">
        {fig("10_results_summary.png", "10 results summary")}
        {slot()}
      </div>
    </div>
  </div>
</body>
</html>
"""
    out = OUT / "synopsis_poster.html"
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


SOURCES_MD = """# Synopsis poster — image sources

Poster files (this folder):

| File | What |
|---|---|
| `synopsis_poster.png` | High-res composite (images + empty text slots) |
| `synopsis_poster.html` | Same layout, print/edit in a browser |

All artwork is copied from the **final ISEF figure pack**. Refresh that pack first if figures change:

```bash
.venv/bin/python scripts/build_final_figures.py
.venv/bin/python scripts/build_synopsis_poster.py
```

## Figures used (repo paths)

| Slot on poster | File in `data/figures/final/` | Upstream source (under `data/figures/`) |
|---|---|---|
| Top-left (accuracy) | `01_travel_time_accuracy.png` | `reports/weekly_2026-10-19/1_travel_time_accuracy.png` |
| Below accuracy (streets) | `02_street_features.png` | `data_exploration/london_eda/08_street_shap.png` |
| Hero map | `03_first_engine_map.png` | `travel_time/maps/travel_time_maps.png` |
| Mid row left | `04_station_busyness.png` | `data_exploration/station_busyness/station_busyness.png` |
| Mid row centre | `05_capacity_mismatches.png` | `data_exploration/station_capacity/05_capacity_mismatches.png` |
| Mid row right | `06_station_types.png` | `analysis/station_embedding/06_isef_story.png` |
| Bottom 1 | `07_layouts_beat_today.png` | `reports/weekly_2026-10-19/3_three_models.png` |
| Bottom 2 | `08_price_of_fairness.png` | `reports/weekly_2026-10-19/2_price_of_fairness.png` |
| Bottom 3 | `09_closure_2014.png` | `reports/weekly_2026-10-19/4_closure_2014.png` |
| Bottom 4 | `10_results_summary.png` | `reports/weekly_2026-10-19/6_summary_table.png` |

Story index: [`data/figures/final/INDEX.md`](../final/INDEX.md).

## Text slots (empty on purpose)

Dashed grey boxes are for later copy — title, authors, and a short takeaway under each figure. Nothing is filled in yet.
"""


def main() -> int:
    png = build_png()
    html = build_html()
    (OUT / "IMAGE_SOURCES.md").write_text(SOURCES_MD)
    print("wrote", png.relative_to(ROOT))
    print("wrote", html.relative_to(ROOT))
    print("wrote", (OUT / "IMAGE_SOURCES.md").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
