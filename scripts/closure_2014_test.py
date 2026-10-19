#!/usr/bin/env python3
"""The 2014 closure test: which travel-time model predicts what really happened?

On 9 January 2014 London closed 10 fire stations. Placement asks a model what happens
when the set of stations changes, and this is the one real change we can check against.

Step 3 - pre-closure models (trained only on 2009-2013 engine trips):
  LightGBM (same features and settings as the current model, FINAL_GROUPS, trees by early
  stopping on 2013 with 2009-12 training), Kolesar on road km, Kolesar on straight-line km.
  Accuracy on 2014 trips after the closures.

Step 4 - the test:
  - Each closed station's former area: 0.9 km cells whose first engine in 2013 most often
    came from that station (LFB mobilisation records, independent of any model).
  - Recorded effect: mean recorded first-engine attendance (incident records) in those
    cells, 9 Jan - 31 Dec 2014 minus 2013, minus the same change in all other cells
    (difference in differences: removes London-wide trends). 95 % bootstrap interval.
  - Predicted effect: each model's attendance (drive + each station's 2013 median turnout)
    with the 2013 stations minus the 2014 stations, same cells, same difference in
    differences; nearest engine free, and with busy engines (Erlang loss calibrated on 2013
    busy rates, pump counts per station for 2013 and 2014 from the mobilisation records).
  - Score per model: error and correlation across the 10 areas.

Step 5 - ranking the closed stations: each closed station put back, one at a time, into
today's 102 stations; predicted saving on 2025 calls (current LightGBM model, busy
engines), compared with the recorded damage in its area (Spearman rank correlation).

Inputs: data/processed/lfb_travel_training_2009_2014.parquet (build_lfb_travel_training.py
--era 2009-2014), data/raw/london/lfb_incidents_2009_2017.csv, lfb_mobilisations_2009_2014.csv.
Writes data/figures/validation/closure_2014/ and data/processed/models/travel_time_lfb_pre2014.joblib.

Run:  PYTHONPATH=src .venv/bin/python scripts/closure_2014_test.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_travel_time_lfb import FINAL_GROUPS, TUNED_PARAMS, Split, metrics  # noqa: E402

from firestation.busy_placement import ErlangBusyModel  # noqa: E402
from firestation.facility import build_demand_grid, haversine_km, posture_seconds  # noqa: E402
from firestation.fire_station_location import snap_to_cells  # noqa: E402
from firestation.kolesar import fit_kolesar  # noqa: E402
from firestation.london_closures_2014 import CLOSED_2014, CLOSED_2014_COORDS, closure_date  # noqa: E402
from firestation.travel_time_model import CATEGORICAL, DRIVE_CLIP_S, make_regressor, predict_bag  # noqa: E402

RAW = ROOT / "data" / "raw" / "london"
OUT = ROOT / "data" / "figures" / "validation" / "closure_2014"
MODEL_PATH = ROOT / "data" / "processed" / "models" / "travel_time_lfb_pre2014.joblib"
TRIPS = ROOT / "data" / "processed" / "lfb_travel_training_2009_2014.parquet"
CLIP = lambda p: np.clip(p, *DRIVE_CLIP_S)  # noqa: E731


def _num(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


# --------------------------------------------------------------------------- step 3
def train_pre2014(trips: pd.DataFrame) -> dict:
    import joblib
    import lightgbm as lgb

    cats = lambda x: [c for c in CATEGORICAL if c in x.columns]  # noqa: E731
    pre = trips[trips["date_of_call"] < pd.Timestamp(closure_date)]
    post = trips[trips["date_of_call"] >= pd.Timestamp(closure_date)]
    print(f"Step 3: pre-closure trips {len(pre):,} (2009 - 8 Jan 2014), post-closure 2014 trips {len(post):,}",
          flush=True)
    v = Split(pre[pre.cal_year <= 2012], pre[pre.cal_year == 2013])
    vtr, vva = v.x(FINAL_GROUPS)
    m = make_regressor(42, {**TUNED_PARAMS, "n_estimators": 6000})
    m.fit(vtr, v.ytr, eval_set=[(vva, v.yte)], eval_metric="l1", categorical_feature=cats(vtr),
          callbacks=[lgb.early_stopping(150, verbose=False)])
    n_trees = int(m.best_iteration_ * 1.15)
    print(f"  early stopping on 2013: {m.best_iteration_} trees → {n_trees}", flush=True)
    s = Split(pre, post)
    xtr, xte = s.x(FINAL_GROUPS)
    model = make_regressor(42, {**TUNED_PARAMS, "n_estimators": n_trees})
    model.fit(xtr, s.ytr, categorical_feature=cats(xtr))
    kol_crow = fit_kolesar(s.tr["crow_km"].to_numpy(float), s.ytr)
    acc = {"lightgbm": metrics(s.yte, CLIP(model.predict(xte))),
           "kolesar(road_km)": metrics(s.yte, CLIP(s.kol.predict(s.te["road_km"].to_numpy(float)))),
           "kolesar(crow_km)": metrics(s.yte, CLIP(kol_crow.predict(s.te["crow_km"].to_numpy(float))))}
    print("  2014 (post-closure) trip accuracy:", {k: v["mae_s"] for k, v in acc.items()}, flush=True)
    bundle = {"kind": "lightgbm_v3_pre2014", "models": [model], "features": list(xtr.columns),
              "feature_groups": FINAL_GROUPS, "categorical": cats(xtr), "category_maps": s.cats,
              "borough_categories": s.cats["borough"],
              "dest_history": {"sizes_m": list(s.dest.sizes_m), "shrink": s.dest.shrink,
                               "tables": {str(k): t.to_dict() for k, t in s.dest.tables.items()}},
              "kolesar_baseline_params": s.kol.to_dict(), "kolesar_crow_params": kol_crow.to_dict(),
              "kolesar_baseline_distance": "road_km", "drive_clip_s": list(DRIVE_CLIP_S),
              "lightgbm_params": {**TUNED_PARAMS, "n_estimators": n_trees}, "seeds": [42],
              "train_period": "2009-01-01 to 2014-01-08", "metrics_2014_post_closure": acc}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


# --------------------------------------------------------------------------- data
def load_eval_incidents() -> pd.DataFrame:
    from pyproj import Transformer

    cols = ["IncidentNumber", "DateOfCall", "CalYear", "HourOfCall", "IncGeo_BoroughName", "Easting_m", "Northing_m",
            "Easting_rounded", "Northing_rounded", "FirstPumpArriving_AttendanceTime"]
    i = pd.read_csv(RAW / "lfb_incidents_2009_2017.csv", usecols=cols, dtype={"IncidentNumber": str}, low_memory=False)
    i = i[i["CalYear"].isin([2013, 2014])].copy()
    e = pd.to_numeric(i["Easting_m"], errors="coerce").fillna(i["Easting_rounded"])
    n = pd.to_numeric(i["Northing_m"], errors="coerce").fillna(i["Northing_rounded"])
    lon, lat = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True).transform(e.to_numpy(), n.to_numpy())
    return pd.DataFrame({"incident_number": _num(i["IncidentNumber"]),
                         "date": pd.to_datetime(i["DateOfCall"], format="%d-%b-%y", errors="coerce"),
                         "hour_of_call": i["HourOfCall"], "borough": i["IncGeo_BoroughName"].astype(str).str.upper(),
                         "dest_lat": lat, "dest_lon": lon,
                         "attendance_s": pd.to_numeric(i["FirstPumpArriving_AttendanceTime"], errors="coerce")}
                        ).dropna(subset=["date", "dest_lat", "attendance_s"])


def pumps_by_year() -> pd.DataFrame:
    """Pumps per station and year: distinct home-station resource codes (station code + 1 digit)."""
    m = pd.read_csv(RAW / "lfb_mobilisations_2009_2014.csv", usecols=["CalYear", "Resource_Code",
                    "DeployedFromStation_Name", "DeployedFromLocation"], low_memory=False)
    m = m[(m["DeployedFromLocation"] == "Home Station") & m["Resource_Code"].astype(str).str.match(r"^[A-Z]\d{3}$")]
    c = m.groupby(["DeployedFromStation_Name", "CalYear", "Resource_Code"]).size().reset_index(name="n")
    c = c[c["n"] >= 100]  # an appliance in regular use that year
    return c.groupby(["DeployedFromStation_Name", "CalYear"]).size().unstack(fill_value=0)


# --------------------------------------------------------------------------- travel matrices
def travel_matrices(sites: pd.DataFrame, cells: pd.DataFrame, incidents: pd.DataFrame, bundle: dict) -> dict:
    from firestation.kolesar import KolesarModel
    from firestation.ml_scorer import _router, lgbm_matrix

    T = {}
    print(f"  LightGBM (pre-2014 model) {len(sites)} sites × {len(cells)} cells…", flush=True)
    T["lightgbm"], _ = lgbm_matrix(sites, cells, incidents, model_path=MODEL_PATH)
    n_s, n_c = len(sites), len(cells)
    s_lat = np.repeat(sites["lat"].to_numpy(float), n_c)
    s_lon = np.repeat(sites["lon"].to_numpy(float), n_c)
    c_lat = np.tile(cells["lat"].to_numpy(float), n_s)
    c_lon = np.tile(cells["lon"].to_numpy(float), n_s)
    road = _router().road_km(s_lat, s_lon, c_lat, c_lon).reshape(n_s, n_c)
    crow = haversine_km(s_lon, s_lat, c_lon, c_lat).reshape(n_s, n_c)
    road = np.where(np.isfinite(road), road, crow * 1.4)
    T["kolesar(road_km)"] = CLIP(KolesarModel.from_dict(bundle["kolesar_baseline_params"]).predict(road))
    T["kolesar(crow_km)"] = CLIP(KolesarModel.from_dict(bundle["kolesar_crow_params"]).predict(crow))
    return T


# --------------------------------------------------------------------------- main
def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    trips = pd.read_parquet(TRIPS)
    bundle = train_pre2014(trips)

    print("Step 4: closure test…", flush=True)
    stations = pd.read_csv(RAW / "london_fire_stations.csv")[["name", "lat", "lon"]]
    closed = pd.DataFrame([(n, *CLOSED_2014_COORDS[n]) for n in CLOSED_2014], columns=["name", "lat", "lon"])
    sites = pd.concat([stations, closed], ignore_index=True).rename(columns={"name": "site_name"})
    sites["is_existing"] = True
    is_closed = sites["site_name"].isin(CLOSED_2014).to_numpy()
    pre_idx = np.arange(len(sites))
    post_idx = np.flatnonzero(~is_closed)

    inc = load_eval_incidents()
    pre_inc = inc[inc["date"].dt.year == 2013]
    post_inc = inc[(inc["date"] >= pd.Timestamp(closure_date)) & (inc["date"].dt.year == 2014)]
    cells = build_demand_grid(pre_inc, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    pre_c, pre_km = snap_to_cells(pre_inc, cells)
    post_c, post_km = snap_to_cells(post_inc, cells)
    keep_pre, keep_post = pre_km <= 1.0, post_km <= 1.0
    pre_c, post_c = pre_c[keep_pre], post_c[keep_post]
    pre_att = pre_inc["attendance_s"].to_numpy()[keep_pre]
    post_att = post_inc["attendance_s"].to_numpy()[keep_post]

    # Former areas: the station whose engine most often arrived first in each cell in 2013.
    t13 = trips[(trips["cal_year"] == 2013) & (trips["pump_order"] == 1)]
    first, _ = snap_to_cells(t13, cells)
    owner = (pd.DataFrame({"cell": first, "st": t13["deployed_from_station"].to_numpy()})
             .groupby("cell")["st"].agg(lambda s: s.value_counts().index[0]))
    cell_owner = owner.reindex(np.arange(len(cells))).to_numpy()
    area = {c: np.flatnonzero(cell_owner == c) for c in CLOSED_2014}
    affected = np.isin(cell_owner, CLOSED_2014)

    # Recorded effect (difference in differences), with a bootstrap over incidents.
    rng = np.random.default_rng(0)

    def did(cells_in, pc, pa, qc, qa):
        a_pre, a_post = np.isin(pc, cells_in), np.isin(qc, cells_in)
        o_pre, o_post = ~np.isin(pc, np.flatnonzero(affected)), ~np.isin(qc, np.flatnonzero(affected))
        return (qa[a_post].mean() - pa[a_pre].mean()) - (qa[o_post].mean() - pa[o_pre].mean())

    rec = {}
    for name, cl in list(area.items()) + [("all 10 areas", np.flatnonzero(affected))]:
        est = did(cl, pre_c, pre_att, post_c, post_att)
        boots = []
        for _ in range(300):
            i1, i2 = rng.integers(0, len(pre_c), len(pre_c)), rng.integers(0, len(post_c), len(post_c))
            boots.append(did(cl, pre_c[i1], pre_att[i1], post_c[i2], post_att[i2]))
        rec[name] = {"recorded_s": float(est), "lo": float(np.percentile(boots, 2.5)),
                     "hi": float(np.percentile(boots, 97.5)), "n_cells": int(len(cl)),
                     "calls_2014": int(np.isin(post_c, cl).sum())}
        print(f"  recorded {name:14s} {est:+6.1f} s [{rec[name]['lo']:+.1f}, {rec[name]['hi']:+.1f}]", flush=True)

    # Turnout (2013 medians per station), pumps, busy rates.
    turn = trips[(trips["cal_year"] == 2013) & (trips["deployed_from_location"] == "Home Station")] \
        .groupby("deployed_from_station")["turnout_s"].median()
    turnout = sites["site_name"].map(turn).fillna(turn.median()).to_numpy()
    pumps = pumps_by_year()
    p13 = sites["site_name"].map(pumps.get(2013, pd.Series(dtype=int))).fillna(1).clip(lower=1).astype(int).to_numpy()
    p14 = sites["site_name"].map(pumps.get(2014, pd.Series(dtype=int))).fillna(1).clip(lower=1).astype(int).to_numpy()
    # Busy rate per station area: share of its area's first engines that came from elsewhere
    # (area = nearest station open at the time, as in the training set's busy flag).
    from build_lfb_travel_training import nearest_open_station

    st_table = sites.set_index("site_name")[["lat", "lon"]]
    ground13 = nearest_open_station(st_table, t13["dest_lat"].to_numpy(), t13["dest_lon"].to_numpy(),
                                    t13["date_of_call"])
    busy_rate = pd.Series((t13["deployed_from_station"].to_numpy() != ground13).astype(float)).groupby(ground13).mean()
    measured = sites["site_name"].map(busy_rate).fillna(busy_rate.mean()).to_numpy()
    load_w = np.bincount(pre_c, minlength=len(cells)).astype(float)
    w14 = np.bincount(post_c, minlength=len(cells)).astype(float)

    T = travel_matrices(sites, cells, pre_inc.assign(hour_of_call=pre_inc["hour_of_call"]), bundle)
    pred = {}
    for model_name, Tm in T.items():
        A = Tm + turnout[:, None]
        r_pre_free, r_post_free = A[pre_idx].min(axis=0), A[post_idx].min(axis=0)
        bm = ErlangBusyModel.calibrate(A, pre_idx, measured, p13, load_w)
        bm.pumps = p13.copy()
        r_pre_busy = posture_seconds(A, pre_idx, bm.probs(A, pre_idx))
        bm.pumps = p14.copy()
        r_post_busy = posture_seconds(A, post_idx, bm.probs(A, post_idx))
        for kind, (a, b) in {"free": (r_pre_free, r_post_free), "busy": (r_pre_busy, r_post_busy)}.items():
            d = b - a
            other = ~affected & (w14 > 0)
            base = float((w14[other] * d[other]).sum() / w14[other].sum())
            out = {}
            for name, cl in list(area.items()) + [("all 10 areas", np.flatnonzero(affected))]:
                ww = w14[cl]
                out[name] = float((ww * d[cl]).sum() / max(ww.sum(), 1e-9)) - base
            pred[f"{model_name} / {kind}"] = out

    rows = []
    for name in CLOSED_2014 + ["all 10 areas"]:
        rows.append({"area": name, **{k: rec[name][k] for k in ("recorded_s", "lo", "hi", "n_cells", "calls_2014")},
                     **{f"pred: {k}": v[name] for k, v in pred.items()}})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "closure_test.csv", index=False)
    per = table[table["area"] != "all 10 areas"]
    score = {}
    for k in pred:
        e = per[f"pred: {k}"] - per["recorded_s"]
        score[k] = {"mae_s_across_areas": round(float(e.abs().mean()), 2),
                    "bias_s": round(float(e.mean()), 2),
                    "corr": round(float(np.corrcoef(per[f"pred: {k}"], per["recorded_s"])[0, 1]), 3),
                    "all_areas_pred_s": round(float(table.loc[table["area"] == "all 10 areas", f"pred: {k}"].iloc[0]), 1)}
    print(pd.DataFrame(score).T.to_string(), flush=True)

    # Step 5: rank closed stations by the saving from reopening each in today's network.
    print("Step 5: ranking the closed stations (reopen one at a time, 2025 calls)…", flush=True)
    from firestation.cities import get_city, load_incidents
    from firestation.fire_station_location import drop_far_incidents
    from firestation.ml_scorer import lgbm_matrix

    spec = get_city("london")
    inc2 = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024, 2025]),
                              stations.rename(columns={"name": "facilityname"}))
    dem, ev = inc2[inc2["cal_year"] == 2024], inc2[inc2["cal_year"] == 2025]
    cells25 = build_demand_grid(dem, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    c25, _ = snap_to_cells(ev, cells25)
    w25 = np.bincount(c25, minlength=len(cells25)).astype(float)
    T25, _ = lgbm_matrix(sites, cells25, dem)
    from firestation.fire_station_location import recorded_station_turnout

    tt, tnew = recorded_station_turnout([2024])
    turn25 = np.array([tt.get(n, turn.get(n, tnew)) for n in sites["site_name"]])
    A25 = T25 + turn25[:, None]
    from firestation.fire_station_location import busy_prob_vector
    from firestation.station_capacity import default_pump_table

    today = post_idx
    ptab = default_pump_table(stations.rename(columns={"name": "facilityname"})).set_index("station")["n_pumps"]
    p25 = sites["site_name"].map(ptab).fillna(1).astype(int).to_numpy()
    meas25 = busy_prob_vector(sites.assign(is_existing=True), dem)
    bm25 = ErlangBusyModel.calibrate(A25, today, meas25, p25, cells25["n_incidents"].to_numpy(float))
    bm25.pumps = p25.copy()
    base_r = posture_seconds(A25, today, bm25.probs(A25, today))
    base_mean = float((w25 * base_r).sum() / w25.sum())
    rank = []
    for j in np.flatnonzero(is_closed):
        idx = np.r_[today, j]
        bm25.pumps = p25.copy()
        bm25.pumps[j] = max(int(p13[j]), 1)
        r = posture_seconds(A25, idx, bm25.probs(A25, idx))
        rank.append({"station": sites.at[j, "site_name"], "pumps_2013": int(p13[j]),
                     "saving_mean_s": round(base_mean - float((w25 * r).sum() / w25.sum()), 2),
                     "extra_calls_within_6min": int((w25 * ((r <= 360) & (base_r > 360))).sum()),
                     "extra_calls_within_10min": int((w25 * ((r <= 600) & (base_r > 600))).sum()),
                     "recorded_damage_2014_s": round(rec[sites.at[j, "site_name"]]["recorded_s"], 1)})
    rank = pd.DataFrame(rank).sort_values("saving_mean_s", ascending=False)
    rank["rank_saving"] = np.arange(1, len(rank) + 1)
    rank["rank_recorded_damage"] = rank["recorded_damage_2014_s"].rank(ascending=False).astype(int)
    spearman = float(rank[["saving_mean_s", "recorded_damage_2014_s"]].rank().corr().iloc[0, 1])
    rank.to_csv(OUT / "closed_station_ranking.csv", index=False)
    print(rank.to_string(index=False))
    print(f"  Spearman (predicted saving today vs recorded 2014 damage): {spearman:.2f}")

    (OUT / "summary.json").write_text(json.dumps({
        "doc": __doc__, "pre2014_model_accuracy_on_2014_trips": bundle["metrics_2014_post_closure"],
        "recorded": rec, "predicted": pred, "score": score, "ranking_spearman": spearman,
        "ranking": rank.to_dict(orient="records"),
        "pumps_2013_2014_changed": {n: [int(a), int(b)] for n, a, b in zip(sites["site_name"], p13, p14) if a != b},
        "minutes": round((time.time() - t0) / 60, 1)}, indent=2, default=str) + "\n")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
