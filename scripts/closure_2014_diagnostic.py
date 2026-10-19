#!/usr/bin/env python3
"""Why can't the models tell which 2014 closure areas suffered most? A decomposition.

For first-engine trips (PumpOrder 1, engine left from a station) in each closed station's
former area (as in closure_2014_test.py) in 2013 and in 2014 after the closures:

  1. Turnout vs drive: how much of the recorded slowdown is slower turnout and how much
     longer driving.
  2. Travel time: the pre-2014 models' predicted drive for the station that actually
     responded, vs the recorded drive. If this tracks the recorded change, the travel-time
     model is not the problem.
  3. Dispatch: was the responding station the model's nearest open station (by predicted
     attendance), the second nearest, or further? And how much slower the recorded drive
     is than the model's nearest-station drive.

Each change is a difference in differences (area change minus the change in the rest of
London). Per-area correlations with the recorded drive change show which part the
nearest-station prediction misses.

Writes data/figures/validation/closure_2014/diagnostic.csv and diagnostic.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/closure_2014_diagnostic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from closure_2014_test import MODEL_PATH, OUT, RAW, TRIPS, load_eval_incidents  # noqa: E402

from firestation.facility import build_demand_grid  # noqa: E402
from firestation.fire_station_location import snap_to_cells  # noqa: E402
from firestation.kolesar import KolesarModel  # noqa: E402
from firestation.london_closures_2014 import CLOSED_2014, CLOSED_2014_COORDS, closure_date  # noqa: E402
from firestation.travel_time_model import DRIVE_CLIP_S, predict  # noqa: E402


def main() -> int:
    import joblib

    from firestation.ml_scorer import lgbm_matrix

    bundle = joblib.load(MODEL_PATH)
    trips = pd.read_parquet(TRIPS)
    trips = trips[(trips["pump_order"] == 1)].copy()
    trips["period"] = np.where(trips["date_of_call"] >= pd.Timestamp(closure_date), "post",
                               np.where(trips["cal_year"] == 2013, "pre", None))
    trips = trips[trips["period"].notna() & (trips["cal_year"] >= 2013)].reset_index(drop=True)
    trips = trips[~((trips["period"] == "post") & (trips["cal_year"] != 2014))].reset_index(drop=True)

    # Same cells and former areas as the closure test.
    inc = load_eval_incidents()
    pre_inc = inc[inc["date"].dt.year == 2013]
    cells = build_demand_grid(pre_inc, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    pre13 = trips[trips["period"] == "pre"]
    c13, _ = snap_to_cells(pre13, cells)
    owner = (pd.DataFrame({"cell": c13, "st": pre13["deployed_from_station"].to_numpy()})
             .groupby("cell")["st"].agg(lambda s: s.value_counts().index[0]))
    cell_owner = owner.reindex(np.arange(len(cells))).to_numpy()
    tc, tkm = snap_to_cells(trips, cells)
    trips["cell"] = tc
    trips = trips[tkm <= 1.0].reset_index(drop=True)
    trips["area"] = pd.Series(cell_owner[trips["cell"].to_numpy()]).where(
        lambda s: s.isin(CLOSED_2014), "rest of London").to_numpy()

    # 2. Predicted drive for the station that actually responded.
    print(f"Predicting {len(trips):,} first-engine trips (2013 and 2014 after closures)…", flush=True)
    trips["pred_lgbm"] = predict(bundle, trips)
    trips["pred_kol_road"] = np.clip(KolesarModel.from_dict(bundle["kolesar_baseline_params"]).predict(
        trips["road_km"].to_numpy(float)), *DRIVE_CLIP_S)
    trips["pred_kol_crow"] = np.clip(KolesarModel.from_dict(bundle["kolesar_crow_params"]).predict(
        trips["crow_km"].to_numpy(float)), *DRIVE_CLIP_S)

    # 3. Dispatch: rank of the responding station among open stations by predicted attendance.
    stations = pd.read_csv(RAW / "london_fire_stations.csv")[["name", "lat", "lon"]]
    closed = pd.DataFrame([(n, *CLOSED_2014_COORDS[n]) for n in CLOSED_2014], columns=["name", "lat", "lon"])
    sites = pd.concat([stations, closed], ignore_index=True).rename(columns={"name": "site_name"})
    sites["is_existing"] = True
    T, _ = lgbm_matrix(sites, cells, pre_inc, model_path=MODEL_PATH)  # cached by the closure test
    turn = trips[trips["period"] == "pre"].groupby("deployed_from_station")["turnout_s"].median()
    A = T + sites["site_name"].map(turn).fillna(turn.median()).to_numpy()[:, None]
    is_closed = sites["site_name"].isin(CLOSED_2014).to_numpy()
    name_to_row = {n: i for i, n in enumerate(sites["site_name"])}
    rank = np.full(len(trips), -1)
    nearest_drive = np.full(len(trips), np.nan)
    for period in ("pre", "post"):
        m = (trips["period"] == period).to_numpy()
        open_rows = np.arange(len(sites)) if period == "pre" else np.flatnonzero(~is_closed)
        order = open_rows[np.argsort(A[open_rows], axis=0)]  # per cell: open stations, fastest first
        cells_m = trips.loc[m, "cell"].to_numpy()
        actual = trips.loc[m, "deployed_from_station"].map(name_to_row).to_numpy()
        pos = np.full(m.sum(), 9)
        for k in range(8):
            pos[(order[k, cells_m] == actual) & (pos == 9)] = k + 1
        rank[m] = pos
        nearest_drive[m] = T[order[0, cells_m], cells_m]
    trips["responder_rank"] = rank
    trips["model_nearest_drive"] = nearest_drive

    def summarise(g):
        return pd.Series({
            "n": len(g), "attendance_s": (g["turnout_s"] + g["drive_s"]).mean(), "turnout_s": g["turnout_s"].mean(),
            "drive_s": g["drive_s"].mean(), "pred_lgbm_actual_station": g["pred_lgbm"].mean(),
            "pred_kol_road_actual_station": g["pred_kol_road"].mean(),
            "pred_kol_crow_actual_station": g["pred_kol_crow"].mean(),
            "model_nearest_drive": g["model_nearest_drive"].mean(),
            "share_nearest": (g["responder_rank"] == 1).mean(), "share_second": (g["responder_rank"] == 2).mean(),
            "share_further": (g["responder_rank"] >= 3).mean(),
        })

    s = trips.groupby(["area", "period"]).apply(summarise).unstack("period")
    cols = ["attendance_s", "turnout_s", "drive_s", "pred_lgbm_actual_station", "pred_kol_road_actual_station",
            "pred_kol_crow_actual_station", "model_nearest_drive"]
    did = pd.DataFrame({c: (s[(c, "post")] - s[(c, "pre")]) - (s.loc["rest of London", (c, "post")]
                                                              - s.loc["rest of London", (c, "pre")]) for c in cols})
    for c in ("share_nearest", "share_second", "share_further"):
        did[f"{c}_pre"], did[f"{c}_post"] = s[(c, "pre")], s[(c, "post")]
    did["n_post"] = s[("n", "post")]
    did = did.drop(index="rest of London")
    areas = did.loc[CLOSED_2014]
    w = trips[(trips["area"] != "rest of London")]
    pooled = summarise(w[w["period"] == "post"]) - summarise(w[w["period"] == "pre"])
    rest = summarise(trips[(trips["area"] == "rest of London") & (trips["period"] == "post")]) \
        - summarise(trips[(trips["area"] == "rest of London") & (trips["period"] == "pre")])
    all_did = (pooled - rest)[cols]
    corr = {c: round(float(np.corrcoef(areas[c], areas["drive_s"])[0, 1]), 3)
            for c in ["pred_lgbm_actual_station", "pred_kol_road_actual_station", "pred_kol_crow_actual_station",
                      "model_nearest_drive"]}
    # Accuracy of each model on the actual responding station's drive, affected areas after closure.
    post_aff = trips[(trips["period"] == "post") & (trips["area"] != "rest of London")]
    acc = {k: {"mae_s": round(float((post_aff[k] - post_aff["drive_s"]).abs().mean()), 1),
               "bias_s": round(float((post_aff[k] - post_aff["drive_s"]).mean()), 1)}
           for k in ("pred_lgbm", "pred_kol_road", "pred_kol_crow", "model_nearest_drive")}
    pre_aff = trips[(trips["period"] == "pre") & (trips["area"] != "rest of London")]
    acc_pre = {k: {"mae_s": round(float((pre_aff[k] - pre_aff["drive_s"]).abs().mean()), 1),
                   "bias_s": round(float((pre_aff[k] - pre_aff["drive_s"]).mean()), 1)}
               for k in ("pred_lgbm", "pred_kol_road", "pred_kol_crow")}

    pd.set_option("display.width", 250)
    show = areas[["attendance_s", "turnout_s", "drive_s", "pred_lgbm_actual_station", "model_nearest_drive",
                  "share_nearest_pre", "share_nearest_post", "share_further_post", "n_post"]].round(2)
    print(show.to_string())
    print("\nAll 10 areas (difference in differences, s):", all_did.round(1).to_dict())
    print("Correlation across areas with the recorded drive change:", corr)
    print("Drive accuracy, actual responding station, affected areas — 2014 after closures:", acc)
    print("  same areas, 2013:", acc_pre)
    areas.to_csv(OUT / "diagnostic.csv")
    (OUT / "diagnostic.json").write_text(json.dumps({
        "doc": __doc__, "all_areas_did_s": all_did.round(2).to_dict(), "correlation_with_recorded_drive_change": corr,
        "drive_accuracy_affected_post": acc, "drive_accuracy_affected_2013": acc_pre,
        "dispatch_share_all_affected": {p: {
            "nearest": round(float((trips[(trips['period'] == p) & (trips['area'] != 'rest of London')]['responder_rank'] == 1).mean()), 3),
            "second": round(float((trips[(trips['period'] == p) & (trips['area'] != 'rest of London')]['responder_rank'] == 2).mean()), 3),
            "further": round(float((trips[(trips['period'] == p) & (trips['area'] != 'rest of London')]['responder_rank'] >= 3).mean()), 3)}
            for p in ("pre", "post")},
        "per_area": areas.round(3).to_dict(orient="index")}, indent=2, default=str) + "\n")
    print("Wrote", OUT / "diagnostic.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
