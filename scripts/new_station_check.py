#!/usr/bin/env python3
"""New-location test: how accurate is the travel time model for a station it has never seen?

Placement scores sites that have no station yet, so the accuracy that matters is on trips
from stations left out of training. Stations are split into ``--folds`` groups (default
20, about 5 stations each; every station is held out exactly once). For each group, a
LightGBM model (tuned settings, 1 seed) is trained on the other stations' 2021-24 trips
and predicts this group's 2025 trips. A reference model trained on every station (also 1
seed) gives the "seen" error on the same 2025 trips, so the gap is the cost of a new site.

This is leave-one-firehouse-out in effect: each held-out station's trips are predicted by
a model that never saw that station, with 95 % rather than 99 % of the training data. The
Kolesar road-distance formula is refit inside each fold for comparison.

Reports overall, per station (does LightGBM beat Kolesar at a station it has never seen?)
and by trip length (new sites far from today's stations make longer trips).

Writes data/figures/validation/new_stations/: summary.json, per_station.csv, by_distance.csv,
per_station.png.

Run:  PYTHONPATH=src .venv/bin/python scripts/new_station_check.py [--folds 20]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from firestation.travel_time_model import DRIVE_CLIP_S, attach_weather, load_weather  # noqa: E402
from train_travel_time_lfb import FINAL_GROUPS, TRAIN_YEARS, TUNED_PARAMS, Split, fit_one, metrics  # noqa: E402

OUT = ROOT / "data" / "figures" / "validation" / "new_stations"
DATA = ROOT / "data" / "processed" / "lfb_travel_training.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--trees", type=int, default=None, help="n_estimators (default: from the saved model's metrics)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.loads((ROOT / "data" / "processed" / "models" / "travel_time_lfb.metrics.json").read_text())
    params = {**TUNED_PARAMS, "n_estimators": args.trees or int(m["lightgbm_params"]["n_estimators"])}
    print("params:", params, flush=True)

    inc = pd.read_csv(DATA, low_memory=False)
    inc = attach_weather(inc, load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    inc = inc[inc.cal_year.isin(TRAIN_YEARS + (2025,))].reset_index(drop=True)
    clip = lambda p: np.clip(p, *DRIVE_CLIP_S)  # noqa: E731

    t0 = time.time()
    ref = Split(inc[inc.cal_year.isin(TRAIN_YEARS)], inc[inc.cal_year == 2025])
    xtr, xte = ref.x(FINAL_GROUPS)
    seen = pd.Series(clip(fit_one(xtr, ref.ytr, params).predict(xte)), index=ref.te["incident_number"].astype(str)
                     + "|" + ref.te["deployed_from_station"].astype(str))
    seen = seen[~seen.index.duplicated()]
    print(f"reference model (all stations): {time.time() - t0:.0f} s", flush=True)

    stations = np.array(sorted(inc["deployed_from_station"].dropna().unique()))
    fold_of = dict(zip(stations, np.random.default_rng(0).permutation(len(stations)) % args.folds))
    f = inc["deployed_from_station"].map(fold_of)
    parts = []
    for i in range(args.folds):
        t1 = time.time()
        s = Split(inc[inc.cal_year.isin(TRAIN_YEARS) & (f != i)], inc[(inc.cal_year == 2025) & (f == i)])
        xtr, xte = s.x(FINAL_GROUPS)
        te = s.te[["incident_number", "deployed_from_station", "road_km"]].copy()
        te["y"] = s.yte
        te["unseen"] = clip(fit_one(xtr, s.ytr, params).predict(xte))
        te["kolesar"] = clip(s.kol.predict(s.te["road_km"].to_numpy(float)))
        parts.append(te)
        print(f"  fold {i + 1}/{args.folds}: {len(te):,} trips, "
              f"{int((pd.Series(fold_of) == i).sum())} stations, {time.time() - t1:.0f} s", flush=True)
    d = pd.concat(parts, ignore_index=True)
    key = d["incident_number"].astype(str) + "|" + d["deployed_from_station"].astype(str)
    d["seen"] = key.map(seen).to_numpy()
    d = d.dropna(subset=["seen"])
    for c in ("seen", "unseen", "kolesar"):
        d[f"err_{c}"] = (d[c] - d["y"]).abs()

    st = d.groupby("deployed_from_station").agg(
        trips=("y", "size"), mae_seen=("err_seen", "mean"), mae_unseen=("err_unseen", "mean"),
        mae_kolesar=("err_kolesar", "mean"), bias_unseen=("unseen", "mean"), mean_y=("y", "mean")).round(2)
    st["bias_unseen"] = (st["bias_unseen"] - st["mean_y"]).round(2)
    st["new_site_penalty_s"] = (st["mae_unseen"] - st["mae_seen"]).round(2)
    st["lgbm_beats_kolesar"] = st["mae_unseen"] < st["mae_kolesar"]
    st = st.drop(columns="mean_y").sort_values("new_site_penalty_s")
    st.to_csv(OUT / "per_station.csv")

    bins = [0, 1, 2, 3, 4, 6, 100]
    d["road_km_bin"] = pd.cut(d["road_km"], bins, labels=["<1", "1-2", "2-3", "3-4", "4-6", "6+"])
    bd = d.groupby("road_km_bin", observed=True).apply(lambda g: pd.Series({
        "trips": len(g), **{f"mae_{c}": g[f"err_{c}"].mean() for c in ("seen", "unseen", "kolesar")},
        "bias_unseen": (g["unseen"] - g["y"]).mean()})).round(2)
    bd.to_csv(OUT / "by_distance.csv")

    summary = {
        "doc": __doc__, "folds": args.folds, "params": params, "n_trips_2025": int(len(d)),
        "n_stations": int(len(st)),
        "overall": {c: metrics(d["y"].to_numpy(), d[c].to_numpy()) for c in ("seen", "unseen", "kolesar")},
        "stations_where_lgbm_beats_kolesar_unseen": int(st["lgbm_beats_kolesar"].sum()),
        "median_new_site_penalty_s": float(st["new_site_penalty_s"].median()),
        "worst_stations_unseen": st.sort_values("mae_unseen").tail(5)[["trips", "mae_seen", "mae_unseen",
                                                                       "mae_kolesar"]].to_dict("index"),
        "by_distance": bd.to_dict("index"),
        "run_minutes": round((time.time() - t0) / 60, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("overall", "stations_where_lgbm_beats_kolesar_unseen", "n_stations",
                                              "median_new_site_penalty_s")}, indent=1))
    print(bd.to_string())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(st["mae_kolesar"], st["mae_unseen"], s=np.sqrt(st["trips"]) * 1.5, alpha=0.6)
    lim = [min(st[["mae_kolesar", "mae_unseen"]].min()) - 5, max(st[["mae_kolesar", "mae_unseen"]].max()) + 5]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Kolesar road-distance formula: error (s)")
    ax.set_ylabel("LightGBM, station never seen in training: error (s)")
    ax.set_title(f"2025 trips per station (MAE). LightGBM better at "
                 f"{summary['stations_where_lgbm_beats_kolesar_unseen']} of {len(st)} stations", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "per_station.png", dpi=150)
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
