#!/usr/bin/env python3
"""How much could moving engines' away time to quieter hours save?

Calculation, not a simulation. Away share per hour comes from gap_windows.py (2024 settings). For each engine class and day type,
total away hours per day stay fixed and only their timing changes. Cost per call = chance of a gap beyond the available-state rate x 148 s
(first_engine_gap.py, 2024), or x 48 over-6-minute points. The saving is an upper limit: it assumes the duties can move freely,
neighbours do not interfere, and the extra time does not change with the hour. 2023 uses the same away pattern on its own calls, as a check.

Writes data/figures/validation/gap_shift/: shift.csv, summary.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/gap_shift.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from gap_windows import prepare  # noqa: E402

OUT = ROOT / "data" / "figures" / "validation" / "gap_shift"
AWAY = ROOT / "data" / "figures" / "validation" / "gap_windows" / "away_by_hour.csv"
EXTRA_S, EXTRA_OVER6 = 148.2, 48.0
EMIT = {1: (0.89, 0.038), 2: (0.587, 0.049)}  # chance of a gap call when away / available (gap_windows.py, 2024)


def flat(u: np.ndarray, hours: range) -> np.ndarray:
    out = np.zeros(24)
    out[list(hours)] = u.sum() / len(hours)
    return out


def quietest(u: np.ndarray, calls: np.ndarray, hours: range, cap: float) -> np.ndarray:
    """Fill the quietest allowed hours first, up to the cap, keeping total away time."""
    out, left = np.zeros(24), u.sum()
    for h in sorted(hours, key=lambda x: calls[x]):
        out[h] = min(cap, left)
        left -= out[h]
    return out if left <= 1e-9 else flat(u, hours)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    away = pd.read_csv(AWAY)
    rows = []
    for year in (2024, 2023):
        d = prepare(year, {})
        calls, pumps = d["calls"], d["pumps"]
        n_all = len(calls)
        e = calls[calls["cat"].isin(["own_first", "real_gap"])].copy()
        e["pumps"] = e["ground"].map(pumps).fillna(1).clip(upper=2).astype(int)
        e["weekend"] = (e["t0"].dt.dayofweek >= 5).astype(int)
        counts = e.groupby(["pumps", "weekend", "HourOfCall"]).size()
        plans = {"today": None, "flat 08:00-18:00": ("flat", range(8, 18)), "flat 08:00-20:00": ("flat", range(8, 20)),
                 "flat 07:00-22:00": ("flat", range(7, 22)), "quietest hours in 08:00-18:00": ("quiet", range(8, 18)),
                 "quietest hours in 08:00-20:00": ("quiet", range(8, 20)), "quietest hours in 07:00-22:00": ("quiet", range(7, 22)),
                 "any hour, quietest first (not realistic)": ("quiet", range(24)), "25 % less away time": ("scale", 0.75), "50 % less away time": ("scale", 0.5)}
        for name, plan in plans.items():
            sec, pts = 0.0, 0.0
            for (k, wk), g in away.groupby(["pumps", "weekend"]):
                u = g.set_index("hour")["away_pct"].reindex(range(24)).to_numpy() / 100
                c = np.array([counts.get((k, wk, h), 0) for h in range(24)], float)
                if plan is None:
                    new = u
                elif plan[0] == "flat":
                    new = flat(u, plan[1])
                elif plan[0] == "quiet":
                    new = quietest(u, c, plan[1], cap=u.max())
                else:
                    new = u * plan[1]
                excess = (c * new).sum() * (EMIT[k][0] - EMIT[k][1])
                sec += excess * EXTRA_S / n_all
                pts += excess * EXTRA_OVER6 / n_all
            rows.append({"year": year, "plan": name, "cost_s": round(sec, 2), "cost_over6_points": round(pts, 2)})
    t = pd.DataFrame(rows)
    base = t[t["plan"] == "today"].set_index("year")
    t["saving_s"] = (t["year"].map(base["cost_s"]) - t["cost_s"]).round(2)
    t["saving_over6_points"] = (t["year"].map(base["cost_over6_points"]) - t["cost_over6_points"]).round(2)
    t.to_csv(OUT / "shift.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"doc": __doc__, "rows": t.to_dict(orient="records")}, indent=2) + "\n")
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
