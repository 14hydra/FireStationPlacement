#!/usr/bin/env python3
"""Does the model's error in outer London shrink or grow the Part 2 savings? 2025 recorded data only; 2026 is not used.

The model under-predicts recorded attendance in outer London (results.md 5.2: +28 s beyond 20 km). A layout's saving is a
difference between two modelled times, so a fixed offset cancels. What does not cancel:

  1. Drive time.  If recorded drive time rises less (or more) than one second per modelled second, a modelled saving shows
     up scaled by that slope.  Fitted on 2025 first-engine trips from the home station with the engine not busy (the trips
     placement scores), in the 5 target boroughs and elsewhere.  Also with predictions from the never-seen-stations test,
     which is closer to a new site.  Slope = how recorded time changes with modelled time across trips (least squares,
     ranges from resampling trips).
  2. Who responds.  A moved station only helps when its own engine responds.  If the local station responds less often than
     the model expects, the saving shrinks by about recorded share / modelled share (station-ground table from
     placement_analysis.py q1, 2025).

Adjusted saving = modelled saving x drive slope x (recorded share / modelled share).  This is a rough first-order check,
not a re-score: it ignores that the second station's time also changes.

Writes data/figures/placement/outer_error/summary.json and by_borough.csv.

Run:  PYTHONPATH=src .venv/bin/python scripts/outer_error_check.py
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

OUT = ROOT / "data" / "figures" / "placement" / "outer_error"
TARGETS = pd.read_csv(ROOT / "data" / "figures" / "placement" / "targeted" / "targets.csv").query("target")["b"].tolist()
BOOT = 500


def fit(y: np.ndarray, p: np.ndarray, rng) -> dict:
    """Recorded = a + b x modelled. Returns slope, bias and a resampled 95 % range for the slope."""
    b = np.polyfit(p, y, 1)[0]
    bs = []
    for _ in range(BOOT):
        i = rng.integers(0, len(y), len(y))
        bs.append(np.polyfit(p[i], y[i], 1)[0])
    return {"trips": int(len(y)), "slope": round(float(b), 3), "slope_lo": round(float(np.percentile(bs, 2.5)), 3),
            "slope_hi": round(float(np.percentile(bs, 97.5)), 3), "recorded_minus_model_s": round(float((y - p).mean()), 1),
            "mean_model_s": round(float(p.mean()), 1)}


def binned(y: np.ndarray, p: np.ndarray) -> list[dict]:
    q = pd.qcut(p, 5, labels=False, duplicates="drop")
    g = pd.DataFrame({"q": q, "y": y, "p": p}).groupby("q").agg(model_s=("p", "mean"), recorded_s=("y", "mean"), trips=("y", "size"))
    return g.round(1).reset_index(drop=True).to_dict("records")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2025)
    df = pd.read_csv(ROOT / "data" / "processed" / "lfb_travel_training.csv", low_memory=False)
    te = df[(df["cal_year"] == 2025) & (df["pump_order"] == 1) & (df["deployed_from_location"] == "Home Station")
            & (df["busy_flag"] == 0)].reset_index(drop=True)
    del df
    te = attach_weather(te, load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    bundle = joblib.load(ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib")
    te["p"] = np.concatenate([predict(bundle, te.iloc[i:i + 50000]) for i in range(0, len(te), 50000)])
    te["y"] = te["drive_s"].astype(float)
    te["b"] = te["borough"].astype(str).str.upper()
    te["target"] = te["b"].isin(TARGETS)

    folds = sorted((ROOT / "data/figures/validation/new_stations/folds").glob("fold_20_*.parquet"))
    key = ["incident_number", "deployed_from_station", "y"]
    un = pd.concat([pd.read_parquet(f, columns=key + ["unseen"]) for f in folds]).drop_duplicates(key)
    te = te.merge(un, on=key, how="left")

    out: dict = {"doc": __doc__, "targets": TARGETS, "drive": {}, "drive_binned": {}}
    for name, m in (("target boroughs", te["target"]), ("rest of London", ~te["target"]), ("all", te["target"] | ~te["target"])):
        s = te[m]
        out["drive"][name] = {"production model": fit(s["y"].to_numpy(), s["p"].to_numpy(), rng)}
        u = s.dropna(subset=["unseen"])
        out["drive"][name]["never-seen stations"] = fit(u["y"].to_numpy(), u["unseen"].to_numpy(), rng)
        out["drive_binned"][name] = binned(s["y"].to_numpy(), s["p"].to_numpy())

    rows = []
    for b in TARGETS:
        s = te[te["b"] == b]
        f = fit(s["y"].to_numpy(), s["p"].to_numpy(), rng)
        rows.append({"borough": b, **f})

    # 2. Who responds: station grounds mapped to the borough where most of their calls are.
    g = pd.read_csv(ROOT / "data" / "figures" / "placement" / "analysis" / "q1_by_ground.csv")
    gb = te.assign(g=te["station_ground"].astype(str).str.upper()).groupby("g")["b"].agg(lambda x: x.mode().iat[0])
    g["b"] = g["g"].str.upper().map(gb)
    g["target"] = g["b"].isin(TARGETS)
    resp = {}
    for name, m in (("target boroughs", g["target"]), ("rest of London", ~g["target"])):
        s = g[m]
        w = s["calls"]
        resp[name] = {"grounds": int(len(s)), "calls": int(w.sum()),
                      "recorded_minus_model_attendance_s": round(float(np.average(s["resid"], weights=w)), 1),
                      "model_nearest_pct": round(float(np.average(s["model_nearest"], weights=w)), 1),
                      "recorded_nearest_pct": round(float(np.average(s["recorded_nearest"], weights=w)), 1),
                      "gap_share_pct": round(float(np.average(s["gap_share"], weights=w)), 1),
                      "empty_share_pct": round(float(np.average(s["empty_share"], weights=w)), 1)}
        resp[name]["share_ratio"] = round(resp[name]["recorded_nearest_pct"] / resp[name]["model_nearest_pct"], 3)
    out["responder"] = resp
    for r in rows:
        s = g[g["b"] == r["borough"]]
        r["recorded_minus_model_attendance_s"] = round(float(np.average(s["resid"], weights=s["calls"])), 1)
        r["share_ratio"] = round(float(np.average(s["recorded_nearest"], weights=s["calls"]) / np.average(s["model_nearest"], weights=s["calls"])), 3)
    pd.DataFrame(rows).to_csv(OUT / "by_borough.csv", index=False)
    out["by_borough"] = rows

    # 3. Adjusted savings for the main targeted layouts (target-borough change, s; LightGBM / road / Kolesar).
    sc = pd.read_csv(ROOT / "data" / "figures" / "placement" / "targeted" / "scores.csv")
    rb = pd.read_csv(ROOT / "data" / "figures" / "placement" / "targeted" / "rule_b" / "scores.csv")
    d = out["drive"]["target boroughs"]
    ratio = resp["target boroughs"]["share_ratio"]
    adj = {}
    for label, t, lay in (("Targeted up to 10", sc, "target_10"), ("Targeted up to 20", sc, "target_20"),
                          ("Targeted 10, Rule B re-optimised", rb, "target_10"), ("Targeted 20, Rule B re-optimised", rb, "target_20")):
        q = t[(t["layout"] == lay) & (t["rule"] == "none")].set_index("scorer")["d_target_s"]
        adj[label] = {}
        for sname in ("production model", "never-seen stations"):
            f = d[sname]
            adj[label][sname] = {s: {"modelled": float(q[s]), "adjusted": round(float(q[s]) * f["slope"] * ratio, 2),
                                     "adjusted_low": round(float(q[s]) * f["slope_lo"] * ratio, 2)} for s in q.index}
    out["adjusted_target_savings"] = adj
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "doc"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
