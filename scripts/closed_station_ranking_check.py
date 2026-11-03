#!/usr/bin/env python3
"""Does the closed-station ranking agree with the 2014 damage once the units match?

closure_2014_test.py step 5 compares two different things and gets a negative rank
correlation: the predicted saving is averaged over ALL London calls (so a station with few
calls in its area scores low), while the recorded damage is per call INSIDE the station's
area. This script compares like with like, with the current scoring model (dispatch, all
day) as well as nearest-or-next-if-busy:

  per call   predicted change in the station's former area vs recorded damage per call
  total      predicted saving across London vs recorded damage x calls in the area,
             the same measure as the saving (seconds per London call)

Also for the 2014 test itself (no 2025 network): predicted vs recorded effect in each area.

Each correlation gets a permutation p-value (all orderings of 10 stations would be 3.6
million, so 100,000 random ones) and a bootstrap 95 % range. The bootstrap redraws each
recorded effect from its own interval and resamples the 10 stations.

Limits: 10 stations; predicted savings use 2025 calls and today's network, recorded damage
is 2014; former areas are the 2013 ones.

Writes data/figures/validation/closure_2014/ranking_like_for_like.csv and .json.

Run:  PYTHONPATH=src .venv/bin/python scripts/closed_station_ranking_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.busy_placement import ErlangBusyModel  # noqa: E402
from firestation.cities import get_city, load_incidents  # noqa: E402
from firestation.dispatch import dispatch_response, load_dispatch  # noqa: E402
from firestation.facility import build_demand_grid, posture_seconds  # noqa: E402
from firestation.fire_station_location import (  # noqa: E402
    busy_prob_vector,
    drop_far_incidents,
    recorded_station_turnout,
    snap_to_cells,
)
from firestation.london_closures_2014 import CLOSED_2014, CLOSED_2014_COORDS  # noqa: E402
from firestation.ml_scorer import lgbm_matrix  # noqa: E402
from firestation.station_capacity import default_pump_table  # noqa: E402

RAW = ROOT / "data" / "raw" / "london"
OUT = ROOT / "data" / "figures" / "validation" / "closure_2014"
TRIPS_2014 = ROOT / "data" / "processed" / "lfb_travel_training_2009_2014.parquet"


def spearman(a, b) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def test_corr(a, b, se_b, *, n_perm=100_000, n_boot=5000, seed=0) -> dict:
    """Rank correlation, two-sided permutation p, bootstrap range including the error in b."""
    a, b, se_b = (np.asarray(x, float) for x in (a, b, se_b))
    rng = np.random.default_rng(seed)
    r = spearman(a, b)
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    perm = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(n_perm)])
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        if len(np.unique(i)) < 3:
            continue
        v = spearman(a[i], (b + rng.normal(0, se_b))[i])
        if np.isfinite(v):
            boots.append(v)
    return {"spearman": round(r, 2), "p_permutation": round(float((np.abs(perm) >= abs(r) - 1e-12).mean()), 3),
            "lo": round(float(np.percentile(boots, 2.5)), 2), "hi": round(float(np.percentile(boots, 97.5)), 2)}


def former_area_owner(cells: pd.DataFrame) -> np.ndarray:
    """For each cell, the station whose engine most often arrived first in 2013."""
    t = pd.read_parquet(TRIPS_2014, columns=["cal_year", "pump_order", "dest_lat", "dest_lon", "deployed_from_station"])
    t = t[(t["cal_year"] == 2013) & (t["pump_order"] == 1)]
    c, _ = snap_to_cells(t, cells)
    owner = pd.DataFrame({"cell": c, "st": t["deployed_from_station"].to_numpy()}).groupby("cell")["st"].agg(
        lambda s: s.value_counts().index[0])
    return owner.reindex(np.arange(len(cells))).to_numpy()


def main() -> int:
    stations = pd.read_csv(RAW / "london_fire_stations.csv")[["name", "lat", "lon"]]
    closed = pd.DataFrame([(n, *CLOSED_2014_COORDS[n]) for n in CLOSED_2014], columns=["name", "lat", "lon"])
    sites = pd.concat([stations, closed], ignore_index=True).rename(columns={"name": "site_name"})
    sites["is_existing"] = True
    is_closed = sites["site_name"].isin(CLOSED_2014).to_numpy()
    today = np.flatnonzero(~is_closed)

    inc = drop_far_incidents(load_incidents(get_city("london"), max_rows=None, years=[2024, 2025]),
                             stations.rename(columns={"name": "facilityname"}))
    dem, ev = inc[inc["cal_year"] == 2024], inc[inc["cal_year"] == 2025]
    cells = build_demand_grid(dem, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    c25, _ = snap_to_cells(ev, cells)
    w = np.bincount(c25, minlength=len(cells)).astype(float)
    T, _ = lgbm_matrix(sites, cells, dem)
    tt, tnew = recorded_station_turnout([2024])
    A = T + np.array([tt.get(n, tnew) for n in sites["site_name"]])[:, None]
    ptab = default_pump_table(stations.rename(columns={"name": "facilityname"})).set_index("station")["n_pumps"]
    pumps = sites["site_name"].map(ptab).fillna(1).astype(int).to_numpy()
    prev = json.loads((OUT / "summary.json").read_text())
    p13 = {r["station"]: r["pumps_2013"] for r in prev["ranking"]}
    bm = ErlangBusyModel.calibrate(A, today, busy_prob_vector(sites, dem), pumps, cells["n_incidents"].to_numpy(float))
    dmodel = load_dispatch()
    owner = former_area_owner(cells)

    def responses(idx, extra=None):
        bm.pumps = pumps.copy()
        if extra is not None:
            bm.pumps[extra] = max(int(p13[sites.at[extra, "site_name"]]), 1)
        return {"busy": posture_seconds(A, idx, bm.probs(A, idx)), "dispatch": dispatch_response(A, idx, bm, dmodel)}

    base = responses(today)
    rec = pd.read_csv(OUT / "closure_test.csv").set_index("area")
    drows = pd.read_csv(ROOT / "data" / "figures" / "validation" / "dispatch" / "closure_areas.csv").set_index("area")
    rows = []
    for j in np.flatnonzero(is_closed):
        name = sites.at[j, "site_name"]
        area = owner == name
        r = responses(np.r_[today, j], j)
        row = {"station": name, "cells_2025_in_area": int((area & (w > 0)).sum()), "calls_2025_in_area": int(w[area].sum()),
               "recorded_per_call_s": float(rec.at[name, "recorded_s"]),
               "recorded_se_s": float((rec.at[name, "hi"] - rec.at[name, "lo"]) / 3.92),
               "calls_2014": int(rec.at[name, "calls_2014"]),
               "pred_2014_per_call_dispatch_s": float(drows.at[name, "dispatch_all_day"])}
        for kind in ("busy", "dispatch"):
            d = base[kind] - r[kind]
            row[f"saving_london_{kind}_s"] = float((w * d).sum() / w.sum())
            row[f"saving_area_{kind}_s"] = float((w[area] * d[area]).sum() / max(w[area].sum(), 1e-9))
        rows.append(row)
    t = pd.DataFrame(rows)
    # Recorded effect as a total, the same kind of measure as the London-wide saving (only ranks are used).
    t["recorded_total_s_calls"] = t["recorded_per_call_s"] * t["calls_2014"]
    t["recorded_total_se"] = t["recorded_se_s"] * t["calls_2014"]
    t.to_csv(OUT / "ranking_like_for_like.csv", index=False)

    out = {
        "doc": __doc__,
        "as_published_unlike_units (London-wide saving vs per-call damage)": {
            k: test_corr(t[f"saving_london_{k}_s"], t["recorded_per_call_s"], t["recorded_se_s"]) for k in ("busy", "dispatch")},
        "total_vs_total (London-wide saving vs damage x calls)": {
            k: test_corr(t[f"saving_london_{k}_s"], t["recorded_total_s_calls"], t["recorded_total_se"]) for k in ("busy", "dispatch")},
        "per_call_vs_per_call (saving in the area, 2025 network, vs damage per call)": {
            k: test_corr(t[f"saving_area_{k}_s"], t["recorded_per_call_s"], t["recorded_se_s"]) for k in ("busy", "dispatch")},
        "2014_test_itself (predicted vs recorded per-call effect, dispatch all day)":
            test_corr(t["pred_2014_per_call_dispatch_s"], t["recorded_per_call_s"], t["recorded_se_s"]),
        "calls_vs_london_saving": round(spearman(t["calls_2025_in_area"], t["saving_london_dispatch_s"]), 2),
    }
    (OUT / "ranking_like_for_like.json").write_text(json.dumps(out, indent=2) + "\n")
    pd.set_option("display.width", 200)
    print(t[["station", "calls_2025_in_area", "saving_london_dispatch_s", "saving_area_dispatch_s", "recorded_per_call_s",
             "calls_2014", "recorded_total_s_calls"]].round(2).sort_values("saving_london_dispatch_s", ascending=False).to_string(index=False))
    print(json.dumps({k: v for k, v in out.items() if k != "doc"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
