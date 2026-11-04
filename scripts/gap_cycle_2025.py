#!/usr/bin/env python3
"""One-off fresh test of the 14-day cycle found in gap_windows.py, on 2025.

Prediction, fixed before this was run (docs/results.md, 2b.3): with the 2024 model settings unchanged, 2025 shows at least
+3.0 points at 14 days and 0 or below at 7 days, on repeat_predicts(). Both must hold. 2026 is not used.
Ranges come from resampling stations and are extra to the prediction.

Writes data/figures/validation/gap_windows/cycle_2025.csv and cycle_2025.json.

Run once:  PYTHONPATH=src .venv/bin/python scripts/gap_cycle_2025.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from gap_windows import (FIT_YEAR, Grid, OUT, apply_pars, lag_corr, midday_day, prepare, repeat_cycle, repeat_predicts, split_fit, STEP_MIN)  # noqa: E402,F401


def gap_calls(d: dict) -> pd.DataFrame:
    e = d["calls"][d["calls"]["cat"].isin(["own_first", "real_gap"])].copy()
    e["gap"] = (e["cat"] == "real_gap").astype(int)
    return e


def main() -> int:
    if os.path.exists(OUT / "cycle_2025.json"):
        print("Already run once; see cycle_2025.json")
        return 1
    cache: dict = {}
    d24 = prepare(FIT_YEAR, cache)
    e24 = gap_calls(d24)
    stations = sorted(s for s in e24["ground"].unique() if s in d24["pumps"].index)
    e24 = e24[e24["ground"].isin(stations)]
    pars = split_fit(Grid(FIT_YEAR, stations), e24, d24["pumps"])
    d = prepare(2025, cache)
    e = gap_calls(d)
    print("Calls stamped outside 2025, dropped:", int((e["t0"].dt.year != 2025).sum()))
    e = e[e["t0"].dt.year == 2025]
    known = [s for p in pars.values() for s in p["stations"]]
    stations = sorted(s for s in e["ground"].unique() if s in known)
    e = e[e["ground"].isin(stations)]
    grid = Grid(2025, stations)
    post = apply_pars(grid, e, pars)
    rows = repeat_predicts(2025, grid, post, e)

    # Ranges: same calculation, stations resampled.
    away_day = midday_day(grid, post) * STEP_MIN / 60 >= 3
    m = e[(e["HourOfCall"] >= 10) & (e["HourOfCall"] < 16)].copy()
    m["s"], m["d"] = m["ground"].map(grid.index), (m["t0"] - grid.start).dt.days
    m["stratum"] = m["s"].astype(str) + "|" + (m["t0"].dt.dayofweek >= 5).astype(str) + "|" + m["HourOfCall"].astype(str)
    rng = np.random.default_rng(2025)
    draws = rng.poisson(1.0, (1000, len(stations)))
    for row in rows:
        k = row["days_earlier"]
        v = m[m["d"] >= k]
        flag = away_day[v["s"].to_numpy(), v["d"].to_numpy() - k]
        adj = (v["gap"] - v.groupby("stratum")["gap"].transform("mean")).to_numpy()
        s = v["s"].to_numpy()
        n = len(stations)
        sf, cf = np.bincount(s[flag], adj[flag], n), np.bincount(s[flag], minlength=n)
        sn, cn = np.bincount(s[~flag], adj[~flag], n), np.bincount(s[~flag], minlength=n)
        b = (draws @ sf) / (draws @ cf) - (draws @ sn) / (draws @ cn)
        row["lo"], row["hi"] = round(float(np.percentile(b, 2.5) * 100), 2), round(float(np.percentile(b, 97.5) * 100), 2)
    t = pd.DataFrame(rows)
    pick = t.set_index("days_earlier")["difference_same_station_hour"]
    verdict = bool(pick[14] >= 3.0 and pick[7] <= 0)
    cy = pd.DataFrame(repeat_cycle(2025, grid, post, e, seed=2025))
    raw = cy[(cy["source"] == "raw gap calls") & cy["days_apart"].isin([5, 7, 8, 9, 14, 21, 23, 28, 37])]
    print(t.to_string(index=False))
    print(raw.to_string(index=False))
    print("Prediction holds:", verdict)
    t.to_csv(OUT / "cycle_2025.csv", index=False)
    (OUT / "cycle_2025.json").write_text(json.dumps({"prediction": __doc__, "holds": verdict, "rows": rows, "raw_correlation": raw.to_dict(orient="records")}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
