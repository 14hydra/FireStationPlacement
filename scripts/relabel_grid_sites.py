#!/usr/bin/env python3
"""Rename new sites still labelled by grid cell ("Demand cell 36", "Spread cell 12") after
their nearest street ("New site near Kennington Road"), in saved placement outputs.

Street refinement keeps a new site at its grid position when no nearby junction lowers the
average time, and older runs left those sites with their grid-cell label. New runs name
them directly (``site_refine.refine_sites``); this updates the outputs already written.
Only labels change: positions, times and every other number stay as they are.

Run:  PYTHONPATH=src .venv/bin/python scripts/relabel_grid_sites.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.ml_scorer import _router  # noqa: E402
from firestation.site_refine import _street  # noqa: E402

PLACEMENT = ROOT / "data" / "figures" / "placement"
GRID = re.compile(r"(?:Demand|Spread) cell \d+")


def main() -> int:
    csvs = [p for p in PLACEMENT.glob("recommended/*/*.csv") if "fire_stations" in p.name]
    csvs += list(PLACEMENT.glob("layout_scoring/layouts/*.csv"))
    coords: dict[str, set] = {}
    for p in csvs:
        df = pd.read_csv(p)
        for r in df[df["site_name"].astype(str).str.fullmatch(GRID.pattern)].itertuples():
            coords.setdefault(r.site_name, set()).add((round(float(r.lat), 6), round(float(r.lon), 6)))
    clash = {k: v for k, v in coords.items() if len(v) > 1}
    if clash:
        raise SystemExit(f"same label at different positions, not renaming: {clash}")
    r = _router()
    names = {}
    for label, ((lat, lon),) in coords.items():
        (node,), _ = r.snap(np.array([lat]), np.array([lon]))
        names[label] = f"New site near {_street(r, node)}"
    print(f"{len(names)} grid-cell labels:", names)
    sub = lambda s: GRID.sub(lambda m: names.get(m.group(0), m.group(0)), s)  # noqa: E731
    files = csvs + [p for p in PLACEMENT.glob("recommended/**/*") if p.suffix in (".json", ".html")]
    n = 0
    for p in files:
        s = p.read_text()
        t = sub(s)
        if t != s:
            p.write_text(t)
            n += 1
    print(f"updated {n} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
