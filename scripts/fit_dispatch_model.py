#!/usr/bin/env python3
"""Fit and test the dispatch model (firestation.dispatch): which station sends the first engine.

Part A - today's network: trained on 2021-2024 first-engine trips, tested on 2025.
  Candidates per demand cell: the 8 open stations with the fastest predicted attendance
  (LightGBM travel times + recorded turnout); busy chances from the Erlang model.
  Compared with "nearest station" and "nearest, or second nearest when busy" on:
    share of first engines by rank (predicted vs recorded), accuracy of the most likely
    responder, and expected first-engine drive per cell vs the recorded drive.
  Saves data/processed/models/dispatch_lfb.joblib.

Part B - the 2014 closures: trained on 2009-2013 only (pre-2014 LightGBM travel times,
  2013 pumps and busy rates), then predicts the 2014 layout (closed stations removed,
  2014 pump counts). Expected first-engine drive change per closed station's former area
  (difference in differences, as in closure_2014_diagnostic.py) vs the recorded change;
  the "nearest station" assumption reached a correlation of 0.42 across the 10 areas and
  the actual responding station 0.88.

Writes data/figures/validation/dispatch/summary.json, rank_shares.csv, closure_areas.csv.

Run:  PYTHONPATH=src .venv/bin/python scripts/fit_dispatch_model.py
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

from firestation.busy_placement import ErlangBusyModel  # noqa: E402
from firestation.dispatch import K, SLOTS, DispatchModel, candidates, responder_counts, slot_busy, slot_factors, slot_of, calibrate_slot_factors  # noqa: E402,E501
from firestation.facility import build_demand_grid  # noqa: E402
from firestation.fire_station_location import snap_to_cells  # noqa: E402

OUT = ROOT / "data" / "figures" / "validation" / "dispatch"
MODEL = ROOT / "data" / "processed" / "models" / "dispatch_lfb.joblib"


def calls_to_cells(trips, cells, row_of):
    c, km = snap_to_cells(trips, cells)
    resp = trips["deployed_from_station"].map(row_of).to_numpy()
    ok = (km <= 1.0) & pd.notna(resp)
    return c[ok], resp[ok].astype(int), ok


def fit_variant(A, open_idx, p_all, p_slots, pumps, cells_n, calls, hourly: bool) -> tuple:
    """Fit the all-day (``hourly=False``) or hour-aware model. ``calls`` = (cells, responder rows, slots)."""
    c, r, h = calls
    parts, layouts = [], {}
    for slot in (range(len(SLOTS)) if hourly else [None]):
        pb = p_all if slot is None else p_slots[slot]
        rows, feats = candidates(A, open_idx, pb, pumps, slot=slot)
        m = np.ones(len(c), bool) if slot is None else h == slot
        parts.append((feats, responder_counts(rows, c[m], r[m]), np.bincount(c[m], minlength=cells_n).astype(float)))
        layouts[slot] = (rows, feats)
    return DispatchModel(hourly=hourly).fit(parts), layouts


def predict_variant(model, A, open_idx, p_all, p_slots, pumps, cells_n):
    """{slot or None: (rows [k, cells], P [k, cells])}."""
    out = {}
    for slot in (range(len(SLOTS)) if model.hourly else [None]):
        pb = p_all if slot is None else p_slots[slot]
        rows, feats = candidates(A, open_idx, pb, pumps, slot=slot)
        out[slot] = (rows, model.proba(feats, cells_n))
    return out


def per_call(pred, c, h):
    """Each call's candidate rows [k, calls] and probabilities, from its own slot when hourly."""
    if None in pred:
        rows, P = pred[None]
        return rows[:, c], P[:, c]
    k = next(iter(pred.values()))[0].shape[0]
    rows, P = np.zeros((k, len(c)), int), np.zeros((k, len(c)))
    for slot, (rw, pp) in pred.items():
        m = h == slot
        rows[:, m], P[:, m] = rw[:, c[m]], pp[:, c[m]]
    return rows, P


def evaluate(preds, T, p_all, calls, drive, n_cells):
    """Rank shares, top-1 accuracy and expected drive per cell vs recorded, for each method."""
    c, r, h = calls
    rows0, _ = per_call(next(iter(preds.values())), c, h)
    k = rows0.shape[0]
    rank = np.full(len(c), k + 1)
    for q in range(k):
        rank[(rows0[q] == r) & (rank > k)] = q + 1
    shares = {"rank": list(range(1, k + 1)) + [f">{k}"],
              "recorded": list(np.bincount(rank - 1, minlength=k + 1) / len(rank))}
    top1, exp_call = {}, {}
    for name, pred in preds.items():
        rows, P = per_call(pred, c, h)
        shares[name] = list(P.mean(axis=1)) + [0.0]
        top1[name] = float((rows[P.argmax(axis=0), np.arange(len(c))] == r).mean())
        exp_call[name] = (P * T[rows, c[None, :]]).sum(axis=0)
    p1 = p_all[rows0[0]]
    shares["nearest_or_next_if_busy"] = [float((1 - p1).mean()), float(p1.mean())] + [0.0] * (k - 1)
    top1["nearest"] = float((rank == 1).mean())
    exp_call["nearest_or_next_if_busy"] = (1 - p1) * T[rows0[0], c] + p1 * T[rows0[1], c]
    exp_call["nearest"] = T[rows0[0], c]
    n_c = np.bincount(c, minlength=n_cells).astype(float)
    m = n_c >= 20
    rec = np.bincount(c, weights=drive, minlength=n_cells) / np.maximum(n_c, 1)
    fit = {}
    for name, e in exp_call.items():
        v = np.bincount(c, weights=e, minlength=n_cells) / np.maximum(n_c, 1)
        fit[name] = {"cell_mae_s": round(float(np.average(np.abs(v[m] - rec[m]), weights=n_c[m])), 1),
                     "bias_s": round(float(np.average(v[m] - rec[m], weights=n_c[m])), 1),
                     "corr": round(float(np.corrcoef(v[m], rec[m])[0, 1]), 3)}
    return pd.DataFrame(shares), top1, fit, exp_call, rank


def part_a():
    from firestation.cities import get_city, load_fire_stations, load_incidents
    from firestation.fire_station_location import (build_house_candidates, busy_prob_vector, drop_far_incidents,
                                                   recorded_station_turnout)
    from firestation.ml_scorer import lgbm_matrix
    from firestation.station_capacity import default_pump_table

    print("Part A: today's network (train 2021-2024, test 2025)…", flush=True)
    spec = get_city("london")
    houses = load_fire_stations(spec)
    inc = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024]), houses)
    cells = build_demand_grid(inc, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    cand = build_house_candidates(houses, cells, demand_candidates=80, spread_km=3.0)
    T, _ = lgbm_matrix(cand, cells, inc)  # same cache as the planner
    existing = np.flatnonzero(cand["is_existing"].to_numpy())
    tt, tnew = recorded_station_turnout([2024])
    names = cand["site_name"].astype(str)
    A = T + np.array([tt.get(n, tnew) for n in names])[:, None]
    pumps = names.map(default_pump_table(houses).set_index("station")["n_pumps"]).fillna(1).astype(int).to_numpy()
    bm = ErlangBusyModel.calibrate(A, existing, busy_prob_vector(cand, inc), pumps,
                                   cells["n_incidents"].to_numpy(float))
    bm.pumps = pumps
    p_all = bm.probs(A, existing)

    trips = pd.read_csv(ROOT / "data" / "processed" / "lfb_travel_training.csv",
                        usecols=["cal_year", "hour_of_call", "pump_order", "deployed_from_station", "dest_lat",
                                 "dest_lon", "drive_s"])
    trips = trips[trips["pump_order"] == 1]
    row_of = {n: i for i, n in zip(existing, names.to_numpy()[existing])}
    data = {}
    for split, years in (("train", range(2021, 2025)), ("test", [2025])):
        t = trips[trips["cal_year"].isin(list(years))]
        c, r, ok = calls_to_cells(t, cells, row_of)
        data[split] = ((c, r, slot_of(t["hour_of_call"].to_numpy()[ok])), t["drive_s"].to_numpy()[ok])
    p_rate = slot_busy(bm, A, existing, slot_factors(trips[trips["cal_year"] < 2025]["hour_of_call"]))
    c_tr, r_tr, h_tr = data["train"][0]
    nearest_row = existing[np.argmin(A[existing], axis=0)]
    f_cal = calibrate_slot_factors(bm, A, existing, c_tr, h_tr, nearest_row[c_tr] == r_tr)
    p_cal = slot_busy(bm, A, existing, f_cal)
    print("calibrated load factor per slot:", np.round(f_cal, 2), flush=True)
    models, preds = {}, {}
    for name, hourly, ps in (("dispatch_all_day", False, p_rate), ("dispatch_by_hour_call_rate", True, p_rate),
                             ("dispatch_by_hour_calibrated", True, p_cal)):
        models[name], _ = fit_variant(A, existing, p_all, ps, pumps, len(cells), data["train"][0], hourly)
        preds[name] = predict_variant(models[name], A, existing, p_all, ps, pumps, len(cells))
    shares, top1, fit, _, rank = evaluate(preds, T, p_all, data["test"][0], data["test"][1], len(cells))
    # By time of day: share of first engines from the nearest station, recorded vs predicted (2025).
    c_te, r_te, h_te = data["test"][0]
    by_slot = []
    for h in range(len(SLOTS)):
        m = h_te == h
        row = {"slot": f"{SLOTS[h][0]:02d}-{SLOTS[h][1]:02d}h", "calls": int(m.sum()),
               "recorded_nearest": float((nearest_row[c_te[m]] == r_te[m]).mean())}
        for name, pred in preds.items():
            rows, P = per_call(pred, c_te[m], h_te[m])
            row[name] = float(P[0].mean())
        row["busy_blend_call_rate"] = float(1 - p_rate[h][nearest_row[c_te[m]]].mean())
        by_slot.append(row)
    by_slot = pd.DataFrame(by_slot)
    print(by_slot.round(3).to_string(index=False))
    slot_err = {k: round(float((by_slot[k] - by_slot["recorded_nearest"]).abs().mean()), 4)
                for k in list(preds) + ["busy_blend_call_rate"]}
    print("mean error in nearest-station share across time slots:", slot_err, flush=True)
    print(shares.round(3).to_string(index=False))
    print("top-1 accuracy:", {k: round(v, 3) for k, v in top1.items()})
    print("expected vs recorded drive per cell (2025, cells with 20+ calls):", json.dumps(fit), flush=True)
    import joblib

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": models["dispatch_by_hour_calibrated"], "all_day_model": models["dispatch_all_day"],
                 "slot_load_factors": f_cal, "slots": SLOTS, "k": K,
                 "trained_on": "2021-2024 first engines"}, MODEL)
    by_slot.to_csv(OUT / "by_time_of_day.csv", index=False)
    return {"rank_shares_2025": shares.round(4).to_dict(orient="records"), "top1_accuracy_2025": top1,
            "drive_per_cell_2025": fit, "share_outside_top8": float((rank > K).mean()),
            "by_time_of_day_2025": by_slot.round(4).to_dict(orient="records"),
            "nearest_share_error_across_slots": slot_err, "slot_load_factors": list(np.round(f_cal, 3))}, shares


def part_b():
    from closure_2014_test import MODEL_PATH, RAW, TRIPS, load_eval_incidents, pumps_by_year
    from firestation.ml_scorer import lgbm_matrix
    from firestation.london_closures_2014 import CLOSED_2014, CLOSED_2014_COORDS, closure_date
    from build_lfb_travel_training import nearest_open_station

    print("Part B: the 2014 closures (train 2009-2013, predict 2014)…", flush=True)
    stations = pd.read_csv(RAW / "london_fire_stations.csv")[["name", "lat", "lon"]]
    closed = pd.DataFrame([(n, *CLOSED_2014_COORDS[n]) for n in CLOSED_2014], columns=["name", "lat", "lon"])
    sites = pd.concat([stations, closed], ignore_index=True).rename(columns={"name": "site_name"})
    sites["is_existing"] = True
    is_closed = sites["site_name"].isin(CLOSED_2014).to_numpy()
    pre_idx, post_idx = np.arange(len(sites)), np.flatnonzero(~is_closed)
    inc = load_eval_incidents()
    pre_inc = inc[inc["date"].dt.year == 2013]
    cells = build_demand_grid(pre_inc, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    nc = len(cells)
    T, _ = lgbm_matrix(sites, cells, pre_inc, model_path=MODEL_PATH)  # cached by the closure test
    trips = pd.read_parquet(TRIPS)
    trips = trips[trips["pump_order"] == 1]
    turn = trips[(trips["cal_year"] == 2013) & (trips["deployed_from_location"] == "Home Station")] \
        .groupby("deployed_from_station")["turnout_s"].median()
    A = T + sites["site_name"].map(turn).fillna(turn.median()).to_numpy()[:, None]
    pumps = pumps_by_year()
    p13 = sites["site_name"].map(pumps.get(2013, pd.Series(dtype=int))).fillna(1).clip(lower=1).astype(int).to_numpy()
    p14 = sites["site_name"].map(pumps.get(2014, pd.Series(dtype=int))).fillna(1).clip(lower=1).astype(int).to_numpy()
    t13 = trips[trips["cal_year"] == 2013]
    st_table = sites.set_index("site_name")[["lat", "lon"]]
    g13 = nearest_open_station(st_table, t13["dest_lat"].to_numpy(), t13["dest_lon"].to_numpy(), t13["date_of_call"])
    rate = pd.Series((t13["deployed_from_station"].to_numpy() != g13).astype(float)).groupby(g13).mean()
    measured = sites["site_name"].map(rate).fillna(rate.mean()).to_numpy()
    c13, _ = snap_to_cells(t13, cells)
    bm = ErlangBusyModel.calibrate(A, pre_idx, measured, p13, np.bincount(c13, minlength=nc).astype(float))
    pre_trips = trips[trips["date_of_call"] < pd.Timestamp(closure_date)]
    factors = slot_factors(pre_trips["hour_of_call"])
    row_of = {n: i for i, n in enumerate(sites["site_name"])}
    c, r, ok = calls_to_cells(pre_trips, cells, row_of)
    calls = (c, r, slot_of(pre_trips["hour_of_call"].to_numpy()[ok]))
    bm.pumps = p13.copy()
    near_pre_row = pre_idx[np.argmin(A[pre_idx], axis=0)]
    f_cal = calibrate_slot_factors(bm, A, pre_idx, c, calls[2], near_pre_row[c] == r)
    pb_pre, ps_pre, pc_pre = bm.probs(A, pre_idx), slot_busy(bm, A, pre_idx, factors), slot_busy(bm, A, pre_idx, f_cal)
    bm.pumps = p14.copy()
    pb_post, ps_post, pc_post = (bm.probs(A, post_idx), slot_busy(bm, A, post_idx, factors),
                                 slot_busy(bm, A, post_idx, f_cal))

    # Evaluation trips (2013 and 2014 after the closures) in former areas vs the rest.
    owner = (pd.DataFrame({"cell": c13, "st": t13["deployed_from_station"].to_numpy()})
             .groupby("cell")["st"].agg(lambda s: s.value_counts().index[0]))
    cell_owner = owner.reindex(np.arange(nc)).to_numpy()
    ev = trips[(trips["cal_year"] == 2013) | ((trips["date_of_call"] >= pd.Timestamp(closure_date))
                                               & (trips["cal_year"] == 2014))]
    ec, ekm = snap_to_cells(ev, cells)
    ev = ev[ekm <= 1.0].assign(cell=ec[ekm <= 1.0])
    ev["post"] = (ev["cal_year"] == 2014).to_numpy()
    ev["slot"] = slot_of(ev["hour_of_call"].to_numpy())
    ev["area"] = pd.Series(cell_owner[ev["cell"].to_numpy()]).where(lambda s: s.isin(CLOSED_2014), "rest").to_numpy()
    n = np.arange(nc)

    def expected_per_trip(pred_pre, pred_post):
        out = np.zeros(len(ev))
        for post, pred in ((False, pred_pre), (True, pred_post)):
            m = (ev["post"] == post).to_numpy()
            rows, P = per_call(pred, ev["cell"].to_numpy()[m], ev["slot"].to_numpy()[m])
            out[m] = (P * T[rows, ev["cell"].to_numpy()[m][None, :]]).sum(axis=0)
        return out

    rows_pre = np.argsort(A[pre_idx], axis=0)[:2]
    rows_post = post_idx[np.argsort(A[post_idx], axis=0)[:2]]
    nearest = {False: T[pre_idx[rows_pre[0]], n], True: T[rows_post[0], n]}
    busy = {False: (1 - pb_pre[pre_idx[rows_pre[0]]]) * T[pre_idx[rows_pre[0]], n]
            + pb_pre[pre_idx[rows_pre[0]]] * T[pre_idx[rows_pre[1]], n],
            True: (1 - pb_post[rows_post[0]]) * T[rows_post[0], n] + pb_post[rows_post[0]] * T[rows_post[1], n]}
    cell_e = ev["cell"].to_numpy()
    ev["nearest"] = np.where(ev["post"], nearest[True][cell_e], nearest[False][cell_e])
    ev["nearest_or_next_if_busy"] = np.where(ev["post"], busy[True][cell_e], busy[False][cell_e])
    names = ["nearest", "nearest_or_next_if_busy"]
    for name, hourly, sp, sq in (("dispatch_all_day", False, ps_pre, ps_post),
                                 ("dispatch_by_hour_call_rate", True, ps_pre, ps_post),
                                 ("dispatch_by_hour_calibrated", True, pc_pre, pc_post)):
        model, _ = fit_variant(A, pre_idx, pb_pre, sp, p13, nc, calls, hourly)
        ev[name] = expected_per_trip(predict_variant(model, A, pre_idx, pb_pre, sp, p13, nc),
                                     predict_variant(model, A, post_idx, pb_post, sq, p14, nc))
        names.append(name)
    cols = ["drive_s"] + names
    g = ev.groupby(["area", "post"])[cols].mean().unstack("post")
    did = pd.DataFrame({k: (g[(k, True)] - g[(k, False)]) - (g.loc["rest", (k, True)] - g.loc["rest", (k, False)])
                        for k in cols}).drop(index="rest").loc[CLOSED_2014]
    aff, rest = ev[ev["area"] != "rest"], ev[ev["area"] == "rest"]
    allarea = {k: float((aff[aff.post][k].mean() - aff[~aff.post][k].mean())
                        - (rest[rest.post][k].mean() - rest[~rest.post][k].mean())) for k in cols}
    score = {k: {"corr_across_areas": round(float(np.corrcoef(did[k], did["drive_s"])[0, 1]), 3),
                 "mae_across_areas_s": round(float((did[k] - did["drive_s"]).abs().mean()), 1),
                 "all_10_areas_s": round(allarea[k], 1)} for k in names}
    print(did.round(1).to_string())
    print("recorded drive change, all 10 areas:", round(allarea["drive_s"], 1))
    print(json.dumps(score, indent=1), flush=True)
    return {"closure_score": score, "recorded_all_areas_drive_s": allarea["drive_s"]}, did


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    a, shares = part_a()
    b, did = part_b()
    shares.to_csv(OUT / "rank_shares.csv", index=False)
    did.to_csv(OUT / "closure_areas.csv")
    (OUT / "summary.json").write_text(json.dumps({"doc": __doc__, "today": a, "closure_2014": b},
                                                 indent=2, default=str) + "\n")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
