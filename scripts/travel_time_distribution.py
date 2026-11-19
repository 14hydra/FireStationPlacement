#!/usr/bin/env python3
"""Does the travel time model reproduce the real distribution of drive times, not just the average? (eMentor meeting 2)

Production LightGBM model, 2025 trips (never used in training). Two versions are compared with the recorded drive times:

  point      one prediction per trip (what placement uses)
  simulated  the eMentor's statistical idea in its simplest form: prediction x a random ratio (recorded / predicted)
             drawn from trips with a similar predicted time. Ratios are learned on January to June 2025 and the
             comparison is on July to December 2025, so the spread is not learned from the trips it is tested on.

Also reports the error relative to the average recorded drive time (normalised error).
Trips: all 2025 trips, and the trips placement scores (first engine, from its home station, local engine not busy).

Writes data/figures/validation/travel_time_distribution/ (summary.json, percentiles.csv, distribution.png).
Run:  PYTHONPATH=src .venv/bin/python scripts/travel_time_distribution.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.travel_time_model import attach_weather, load_weather, predict  # noqa: E402

OUT = ROOT / "data" / "figures" / "validation" / "travel_time_distribution"
N_BINS = 20
PCTS = (5, 10, 25, 50, 75, 90, 95, 99)


def wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    q = np.linspace(0.005, 0.995, 199)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def ks(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.concatenate([a, b]))
    return float(np.abs(np.searchsorted(np.sort(a), x, "right") / len(a) - np.searchsorted(np.sort(b), x, "right") / len(b)).max())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "processed" / "lfb_travel_training.csv", low_memory=False)
    te = df[df["cal_year"] == 2025].reset_index(drop=True)
    del df
    te = attach_weather(te, load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    bundle = joblib.load(ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib")
    p = np.concatenate([predict(bundle, te.iloc[i:i + 50000]) for i in range(0, len(te), 50000)])
    y = te["drive_s"].to_numpy(float)
    month = pd.to_datetime(te["date_of_call"], format="mixed").dt.month.to_numpy()
    placement = ((te["pump_order"] == 1) & (te["deployed_from_location"] == "Home Station") & (te["busy_flag"] == 0)).to_numpy()

    summary, prow = {}, []
    sims = {}
    for name, sl in (("all_2025_trips", np.ones(len(te), bool)), ("placement_trips", placement)):
        cal, tst = sl & (month <= 6), sl & (month >= 7)
        edges = np.quantile(p[cal], np.linspace(0, 1, N_BINS + 1))
        bc, bt = np.clip(np.searchsorted(edges, p[cal]) - 1, 0, N_BINS - 1), np.clip(np.searchsorted(edges, p[tst]) - 1, 0, N_BINS - 1)
        ratio = y[cal] / p[cal]
        rng = np.random.default_rng(0)
        sim = np.empty(tst.sum())
        for b in range(N_BINS):
            pool = ratio[bc == b]
            sim[bt == b] = p[tst][bt == b] * rng.choice(pool, (bt == b).sum())
        yt, pt = y[tst], p[tst]
        sims[name] = (yt, pt, sim)
        err = pt - yt
        yall, pall = y[sl], p[sl]
        summary[name] = {
            "trips_2025": int(sl.sum()),
            "mean_recorded_s": round(float(yall.mean()), 1), "median_recorded_s": round(float(np.median(yall)), 1),
            "mae_s": round(float(np.abs(pall - yall).mean()), 1), "median_abs_error_s": round(float(np.median(np.abs(pall - yall))), 1),
            "mae_over_mean_recorded_pct": round(float(np.abs(pall - yall).mean() / yall.mean() * 100), 1),
            "median_abs_pct_error": round(float(np.median(np.abs(pall - yall) / yall) * 100), 1),
            "trips_within_20pct": round(float((np.abs(pall - yall) / yall <= 0.2).mean() * 100), 1),
            "test_half": {"trips": int(tst.sum()),
                          "point_vs_recorded": {"wasserstein_s": round(wasserstein(pt, yt), 1), "ks": round(ks(pt, yt), 3),
                                                "sd_s": round(float(pt.std()), 1)},
                          "simulated_vs_recorded": {"wasserstein_s": round(wasserstein(sim, yt), 1), "ks": round(ks(sim, yt), 3),
                                                    "sd_s": round(float(sim.std()), 1)},
                          "recorded_sd_s": round(float(yt.std()), 1),
                          "over_360s_pct": {"recorded": round(float((yt > 360).mean() * 100), 1), "point": round(float((pt > 360).mean() * 100), 1),
                                            "simulated": round(float((sim > 360).mean() * 100), 1)},
                          "mean_error_s": round(float(err.mean()), 1)}}
        for k, v in (("recorded", yt), ("point", pt), ("simulated", sim)):
            prow.append({"trips": name, "series": k, **{f"p{q}": round(float(np.percentile(v, q)), 1) for q in PCTS}})
    pd.DataFrame(prow).to_csv(OUT / "percentiles.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"doc": __doc__, **summary}, indent=2) + "\n")
    np.savez_compressed(OUT / "placement_test_half.npz", recorded=sims["placement_trips"][0], point=sims["placement_trips"][1],
                        simulated=sims["placement_trips"][2])
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(prow).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
