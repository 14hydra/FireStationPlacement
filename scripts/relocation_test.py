#!/usr/bin/env python3
"""Test A of docs/prediction_relocations_and_late_2026.md: do the models predict what happened when engines really moved?

Events come from data/figures/validation/relocations/events.csv (scripts/relocation_events.py). For each event:

  - Area: 0.9 km cells whose first engine, in the 365 days before the move, most often came from the rebuilt station
    (first-engine trips from the mobilisation records, the 2014 test's rule; see the 2026-11-12 note in the prediction).
    Control: every cell outside all 8 event areas.
  - Windows: "during" = the move dates (from 9 January 2014 for Plaistow and Deptford, which have no "before" window);
    "before" = the 180 days ending 30 days before the move; "after" = the 180 days starting 30 days after it ended.
  - Recorded effect: difference in differences of mean recorded first-engine attendance (during minus comparison window,
    area minus control), averaged over the comparison windows. 95 % range: 1,000 draws, resampling calls within each
    window, separately in the area and in the control.
  - Predicted effect: per cell, the change in modelled first-engine attendance (during layout minus comparison layout),
    weighted by the during window's recorded calls per cell, then the same area-minus-control difference (as in
    scripts/closure_2014_test.py). Within each comparison window: that window's turnout medians and pump counts (from
    the mobilisation records), Erlang busy model calibrated on that window's calls and busy rates. During layout: the
    rebuilt station is removed and its engines work from the station they moved to, keeping their own turnout median
    and busy-model load per call (a separate row at the destination's site, with its travel times).
  - Travel models: the pre-2014 LightGBM model and the 2009-2013 road and straight-line Kolesar fits for events 1-7;
    the current model and Part 2's Kolesar fits for Lee Green (reported only). Each with nearest engine free, with busy
    engines (primary: LightGBM busy), and the dispatch model (dispatch_lfb.joblib, trained on 2021-2024 trips).

Pass marks (primary LightGBM busy, scored events 1-6): direction, overall size inside the recorded 95 % range, mean
absolute error per event 20 s or less.

  --check-2014   the code check: runs the same code on the 2014 closures (2013 vs 9 Jan - 31 Dec 2014, areas from 2013
                 first engines, pumps by calendar year as in closure_2014_test.py). Must reproduce results.md 3.1 within
                 0.5 s (recorded +70.5, LightGBM busy +67.5). Allowed any number of times; reads no event outcome.
  --predictions-only   the models' predicted effect for every event, without reading recorded attendance in any event
                 area (inputs: calls per cell, comparison-window turnout, pumps and busy rates). Commit its output
                 (relocations/predicted.csv) before the real test; the test checks it reproduces those numbers.
  (no flag)      the real test. Runs the check first and stops if it fails. Refuses to run if the prediction or this
                 script has uncommitted changes, or if it has already been run (relocations/result.json exists).

Run:  PYTHONPATH=src .venv/bin/python scripts/relocation_test.py --check-2014
      PYTHONPATH=src .venv/bin/python scripts/relocation_test.py          # once only
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from closure_2014_test import MODEL_PATH as PRE2014_MODEL, travel_matrices  # noqa: E402
from final_test_2026 import committed_clean  # noqa: E402

from firestation.busy_placement import ErlangBusyModel  # noqa: E402
from firestation.facility import build_demand_grid, haversine_km, posture_seconds  # noqa: E402
from firestation.fire_station_location import snap_to_cells  # noqa: E402
from firestation.london_closures_2014 import CLOSED_2014, CLOSED_2014_COORDS, closure_date  # noqa: E402

RAW = ROOT / "data" / "raw" / "london"
OUT = ROOT / "data" / "figures" / "validation" / "relocations"
EVENTS = OUT / "events.csv"
PREDICTED = OUT / "predicted.csv"  # --predictions-only output, committed before the test
PREDICTION = ROOT / "docs" / "prediction_relocations_and_late_2026.md"
CLOSURE = pd.Timestamp(closure_date)
PRIMARY = "lightgbm / busy"
CHECK = {"recorded": 70.5, PRIMARY: 67.5, "tolerance": 0.5}
PASS_MAE_S = 20.0
BETTER_BY_S = 5.0
BOOT = 1000

# Fixed in the prediction: event number, rebuilt station, destination, set.
EVENT_SET = {
    "LEYTONSTONE": (1, "scored"), "SHADWELL": (2, "scored"), "PURLEY": (3, "scored"), "DOCKHEAD": (4, "scored"),
    "PLAISTOW": (5, "scored"), "DEPTFORD": (6, "scored"), "DAGENHAM": (7, "reported"), "LEE GREEN": (8, "reported"),
}
AFTER_ONLY = {"PLAISTOW", "DEPTFORD"}  # started before the 2014 closures
# Note of 2026-11-12 in the prediction: in every window, any other event whose move covers more than half of it is in
# place (its station out, its engines at their destination), so each layout is the network that really ran.
CONCURRENT = True
CURRENT_MODEL = {"LEE GREEN"}  # inside the current model's training years


# --------------------------------------------------------------------------- raw records
def _incident_number(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def load_incidents(start: str, end: str) -> pd.DataFrame:
    """Incident records with a first-engine attendance, start <= date <= end."""
    from pyproj import Transformer

    a, b = pd.Timestamp(start), pd.Timestamp(end)
    cols = ["IncidentNumber", "DateOfCall", "CalYear", "HourOfCall", "IncGeo_BoroughName", "Easting_m", "Northing_m", "Easting_rounded",
            "Northing_rounded", "FirstPumpArriving_AttendanceTime", "FirstPumpArriving_DeployedFromStation"]
    files = {"lfb_incidents_2009_2017.csv": (2009, 2017, "%d-%b-%y"), "lfb_incidents_2018_2023.csv": (2018, 2023, "%Y-%m-%d"),
             "lfb_incidents_2024_onwards.csv": (2024, 2100, "%Y-%m-%d")}
    frames = []
    for f, (y0, y1, fmt) in files.items():
        if y1 < a.year or y0 > b.year:
            continue
        i = pd.read_csv(RAW / f, usecols=cols, dtype={"IncidentNumber": str}, low_memory=False, encoding="utf-8-sig")
        i = i[i["CalYear"].between(a.year, b.year)].copy()
        i["date"] = pd.to_datetime(i["DateOfCall"].astype(str).str[:10], format=fmt, errors="coerce")
        frames.append(i[(i["date"] >= a) & (i["date"] <= b)])
    i = pd.concat(frames, ignore_index=True)
    e = pd.to_numeric(i["Easting_m"], errors="coerce").fillna(i["Easting_rounded"])
    n = pd.to_numeric(i["Northing_m"], errors="coerce").fillna(i["Northing_rounded"])
    lon, lat = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True).transform(e.to_numpy(), n.to_numpy())
    return pd.DataFrame({"incident": _incident_number(i["IncidentNumber"]).to_numpy(), "date": i["date"].to_numpy(), "hour_of_call": i["HourOfCall"].to_numpy(),
                         "borough": i["IncGeo_BoroughName"].astype(str).str.upper().to_numpy(), "dest_lat": lat, "dest_lon": lon,
                         "attendance_s": pd.to_numeric(i["FirstPumpArriving_AttendanceTime"], errors="coerce").to_numpy(),
                         "first_station": i["FirstPumpArriving_DeployedFromStation"].astype(str).str.strip().str.upper().to_numpy()}
                        ).dropna(subset=["date", "dest_lat", "attendance_s"]).reset_index(drop=True)


def load_mobilisations(start: str, end: str) -> pd.DataFrame:
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for f in sorted(glob.glob(str(RAW / "lfb_mobilisations_*.csv"))):
        y0 = int(Path(f).stem.split("_")[2])
        y1 = int(Path(f).stem.split("_")[-1])
        if y1 < a.year or y0 > b.year:
            continue
        d = pd.read_csv(f, usecols=["IncidentNumber", "DateAndTimeMobilised", "Resource_Code", "DeployedFromStation_Name",
                                    "DeployedFromLocation", "TurnoutTimeSeconds", "TravelTimeSeconds", "PumpOrder",
                                    "DelayCode_Description"], dtype={"IncidentNumber": str}, encoding="utf-8-sig", low_memory=False)
        iso = ("2009" in f) or ("2015" in f)  # as in relocation_events.py
        d["date"] = pd.to_datetime(d["DateAndTimeMobilised"].astype(str).str[:10], format="%Y-%m-%d" if iso else "%d/%m/%Y",
                                   errors="coerce")
        frames.append(d[(d["date"] >= a) & (d["date"] <= b)])
    m = pd.concat(frames, ignore_index=True)
    m["incident"] = _incident_number(m["IncidentNumber"])
    m["st"] = m["DeployedFromStation_Name"].astype(str).str.strip().str.upper()
    m["turnout_s"] = pd.to_numeric(m["TurnoutTimeSeconds"], errors="coerce")
    m["travel_s"] = pd.to_numeric(m["TravelTimeSeconds"], errors="coerce")
    return m[["date", "incident", "Resource_Code", "st", "DeployedFromLocation", "turnout_s", "travel_s", "PumpOrder",
              "DelayCode_Description"]]


def engine_trips(mob: pd.DataFrame, inc: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Engine trips with the filters of build_lfb_travel_training.py: left from a station, not on outside duty or a wrong
    address, an LFB station, travel 30-1200 s, joined to the incident location, 0-15 km straight line. Road routing is
    not needed here (it drops few trips)."""
    from build_lfb_travel_training import EXCLUDED_DELAYS, MAX_CROW_KM, TRAVEL_CLIP_S

    m = mob[mob["DeployedFromLocation"].isin(["Home Station", "Other Station"]) & ~mob["DelayCode_Description"].isin(EXCLUDED_DELAYS)
            & mob["st"].isin(sites["key"]) & mob["travel_s"].between(*TRAVEL_CLIP_S)]
    t = m.merge(inc[["incident", "dest_lat", "dest_lon"]].drop_duplicates("incident"), on="incident", how="inner")
    s = sites.drop_duplicates("key").set_index("key")
    km = haversine_km(t["st"].map(s["lon"]).to_numpy(), t["st"].map(s["lat"]).to_numpy(), t["dest_lon"].to_numpy(),
                      t["dest_lat"].to_numpy())
    return t[(km > 0) & (km <= MAX_CROW_KM)].reset_index(drop=True)


def in_window(df: pd.DataFrame, w: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    return df[(df["date"] >= w[0]) & (df["date"] <= w[1])]


def pumps_in(mob: pd.DataFrame, w) -> pd.Series:
    """Pumps per station: home-station engine codes in regular use (100+ a year, pro rata), as in closure_2014_test.py."""
    m = in_window(mob, w)
    m = m[(m["DeployedFromLocation"] == "Home Station") & m["Resource_Code"].astype(str).str.match(r"^[A-Z]\d{3}$")]
    c = m.groupby(["st", "Resource_Code"]).size()
    c = c[c >= 100 * ((w[1] - w[0]).days + 1) / 365]
    return c.groupby(level=0).size()


def turnout_in(trips: pd.DataFrame, w) -> pd.Series:
    """Median turnout per station, home-station trips (as closure_2014_test.py)."""
    t = in_window(trips, w)
    return t[t["DeployedFromLocation"] == "Home Station"].groupby("st")["turnout_s"].median()


def busy_rates(trips: pd.DataFrame, sites: pd.DataFrame, open_idx) -> pd.Series:
    """Share of first-engine trips in each open station's ground (nearest open station, straight line) that came from
    another station: the training set's busy flag, as in closure_2014_test.py."""
    t = trips[trips["PumpOrder"] == 1]
    s = sites.iloc[np.asarray(open_idx)]
    d = haversine_km(t["dest_lon"].to_numpy()[:, None], t["dest_lat"].to_numpy()[:, None],
                     s["lon"].to_numpy()[None, :], s["lat"].to_numpy()[None, :])
    ground = s["key"].to_numpy()[d.argmin(axis=1)]
    return pd.Series((t["st"].to_numpy() != ground).astype(float)).groupby(ground).mean()


def cell_owners(trips: pd.DataFrame, cells: pd.DataFrame) -> np.ndarray:
    """Station that most often sent the first engine to each cell (first-engine trips), as in closure_2014_test.py."""
    t = trips[trips["PumpOrder"] == 1]
    c, _ = snap_to_cells(t, cells)
    o = pd.DataFrame({"cell": c, "st": t["st"].to_numpy()}).groupby("cell")["st"].agg(lambda s: s.value_counts().index[0])
    return o.reindex(np.arange(len(cells))).fillna("").to_numpy()


# --------------------------------------------------------------------------- one comparison
@dataclass
class Half:
    """One "during vs comparison window" difference in differences."""

    cmp: tuple  # comparison window (start, end)
    during: tuple
    open_cmp: np.ndarray  # site rows open in the comparison window
    open_during: np.ndarray
    pumps_cmp_w: tuple | None = None  # window for pump counts (default: cmp)
    pumps_during_w: tuple | None = None  # None: comparison pumps, plus moved engines
    # Engines working from another station in either window: {"row": moved row, "from": station row, "n": engines,
    # "turnout": their turnout median or None (= the station's median in the comparison window)}
    moved: list = field(default_factory=list)


def calls_cells(inc, w, cells):
    x = in_window(inc, w)
    c, km = snap_to_cells(x, cells)
    return c[km <= 1.0], x["attendance_s"].to_numpy()[km <= 1.0], x[km <= 1.0]


def recorded_half(inc, h: Half, cells, area, control, rng, draws):
    """Point estimate and bootstrap draws of the recorded difference in differences."""
    out = []
    groups = {}
    for name, w in (("cmp", h.cmp), ("during", h.during)):
        c, a, _ = calls_cells(inc, w, cells)
        groups[name] = (a[area[c]], a[control[c]])
    est = (groups["during"][0].mean() - groups["cmp"][0].mean()) - (groups["during"][1].mean() - groups["cmp"][1].mean())
    for _ in range(draws):
        m = {k: [v[rng.integers(0, len(v), len(v))].mean() for v in g] for k, g in groups.items()}
        out.append((m["during"][0] - m["cmp"][0]) - (m["during"][1] - m["cmp"][1]))
    return float(est), np.array(out), int(len(groups["during"][0]))


def predicted_half(T: dict, sites, inc, mob, trips, h: Half, cells, area, control, dispatch):
    """Predicted difference in differences per travel model and response model."""
    keys = sites["key"].to_numpy()
    turn = turnout_in(trips, h.cmp)
    turnout = pd.Series(keys).map(turn).fillna(turn.median()).to_numpy(dtype=float).copy()
    p_cmp = pd.Series(keys).map(pumps_in(mob, h.pumps_cmp_w or h.cmp)).fillna(1).clip(lower=1).astype(int).to_numpy()
    if h.pumps_during_w is not None:
        p_dur = pd.Series(keys).map(pumps_in(mob, h.pumps_during_w)).fillna(1).clip(lower=1).astype(int).to_numpy()
    else:
        p_dur = p_cmp.copy()
    c_cmp, _, _ = calls_cells(inc, h.cmp, cells)
    c_dur, _, _ = calls_cells(inc, h.during, cells)
    load_w = np.bincount(c_cmp, minlength=len(cells)).astype(float)
    w_dur = np.bincount(c_dur, minlength=len(cells)).astype(float)
    rate = busy_rates(in_window(trips, h.cmp), sites, h.open_cmp)
    measured = pd.Series(keys).map(rate).fillna(rate.mean()).to_numpy()
    p_cmp, p_dur = p_cmp.copy(), p_dur.copy()
    for m in h.moved:  # the station's crews keep their turnout and engine count wherever they work from
        t = turnout[m["from"]] if m["turnout"] is None else m["turnout"]
        turnout[m["row"]] = turnout[m["from"]] = t
        for p in (p_cmp, p_dur):
            p[m["row"]] = p[m["from"]] = m["n"]
    a_m, c_m = area & (w_dur > 0), control & (w_dur > 0)
    out = {}
    for model_name, Tm in T.items():
        Tm = Tm.copy()
        A = Tm + turnout[:, None]
        bm = ErlangBusyModel.calibrate(A, h.open_cmp, measured, p_cmp, load_w)
        for m in h.moved:  # load per call follows the crews: copy from whichever of the two rows was calibrated
            src, dst = (m["from"], m["row"]) if m["from"] in set(h.open_cmp) else (m["row"], m["from"])
            bm.erlangs_per_call[dst] = bm.erlangs_per_call[src]
        r = {}
        for kind, idx, pumps in (("cmp", h.open_cmp, p_cmp), ("during", h.open_during, p_dur)):
            bm.pumps = pumps.copy()
            r[("free", kind)] = A[idx].min(axis=0)
            r[("busy", kind)] = posture_seconds(A, idx, bm.probs(A, idx))
            if dispatch is not None and model_name == "lightgbm":
                from firestation.dispatch import dispatch_response

                r[("dispatch", kind)] = dispatch_response(A, idx, bm, dispatch)
        for resp in ("free", "busy", "dispatch"):
            if (resp, "cmp") not in r:
                continue
            d = r[(resp, "during")] - r[(resp, "cmp")]
            out[f"{model_name} / {resp}"] = float(w_dur[a_m] @ d[a_m] / w_dur[a_m].sum() - w_dur[c_m] @ d[c_m] / w_dur[c_m].sum())
    return out


# --------------------------------------------------------------------------- sites
def site_table() -> pd.DataFrame:
    """Today's stations plus the 10 closed in 2014 (the 2013 set); keys are upper-case names."""
    st = pd.read_csv(RAW / "london_fire_stations.csv")[["name", "lat", "lon"]]
    closed = pd.DataFrame([(n, *CLOSED_2014_COORDS[n]) for n in CLOSED_2014], columns=["name", "lat", "lon"])
    s = pd.concat([st, closed], ignore_index=True).rename(columns={"name": "site_name"})
    s["is_existing"] = True
    s["key"] = s["site_name"].astype(str).str.strip().str.upper()
    return s


def open_rows(sites: pd.DataFrame, w) -> np.ndarray:
    """2013 set for a window ending before 9 January 2014, else the 2014 set (closed stations out)."""
    if w[1] < CLOSURE:
        return np.arange(len(sites))
    return np.flatnonzero(~sites["site_name"].isin(CLOSED_2014).to_numpy())


# --------------------------------------------------------------------------- the 2014 code check
def check_2014() -> dict:
    print("Code check: the 2014 closures through this script's code…", flush=True)
    sites = site_table()
    w13 = (pd.Timestamp("2013-01-01"), pd.Timestamp("2013-12-31"))
    w14 = (CLOSURE, pd.Timestamp("2014-12-31"))
    inc = load_incidents("2013-01-01", "2014-12-31")
    mob = load_mobilisations("2013-01-01", "2014-12-31")
    pre = in_window(inc, w13)
    cells = build_demand_grid(pre, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    trips = engine_trips(mob, inc, sites)
    owner = cell_owners(in_window(trips, w13), cells)
    area = np.isin(owner, [c.upper() for c in CLOSED_2014])
    T = travel_matrices(sites, cells, pre, joblib.load(PRE2014_MODEL))
    h = Half(cmp=w13, during=w14, open_cmp=open_rows(sites, w13), open_during=open_rows(sites, w14),
             pumps_cmp_w=w13, pumps_during_w=(pd.Timestamp("2014-01-01"), pd.Timestamp("2014-12-31")))
    rec, boots, n = recorded_half(inc, h, cells, area, ~area, np.random.default_rng(0), 300)
    pred = predicted_half(T, sites, inc, mob, trips, h, cells, area, ~area, dispatch=None)
    res = {"recorded_s": round(rec, 2), "recorded_lo": round(float(np.percentile(boots, 2.5)), 2),
           "recorded_hi": round(float(np.percentile(boots, 97.5)), 2), "area_cells": int(area.sum()), "area_calls_2014": n,
           "predicted": {k: round(v, 2) for k, v in pred.items()}, "published": CHECK}
    res["pass"] = bool(abs(rec - CHECK["recorded"]) <= CHECK["tolerance"]
                       and abs(pred[PRIMARY] - CHECK[PRIMARY]) <= CHECK["tolerance"])
    print(json.dumps(res, indent=1), flush=True)
    return res


# --------------------------------------------------------------------------- the test
def event_windows(e) -> dict:
    s, t = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
    w = {"area": (s - pd.Timedelta(days=365), s - pd.Timedelta(days=1)),
         "during": (max(s, CLOSURE) if e["station"] in AFTER_ONLY else s, t),
         "after": (t + pd.Timedelta(days=31), t + pd.Timedelta(days=210))}
    if e["station"] not in AFTER_ONLY:
        w["before"] = (s - pd.Timedelta(days=210), s - pd.Timedelta(days=31))
    return w


def in_place(w, others: dict) -> list:
    """Other events whose move covers more than half of window w (their engines count as moved in w)."""
    days = (w[1] - w[0]).days + 1
    out = []
    for st, (a, b) in others.items():
        overlap = (min(w[1], b) - max(w[0], a)).days + 1
        if overlap > days / 2:
            out.append(st)
    return out


def run_test(recorded: bool = True) -> dict:
    """The test. recorded=False computes only the models' predictions (no recorded attendance in any event area)."""
    ev = pd.read_csv(EVENTS)
    ev = ev[ev["station"].isin(EVENT_SET)].copy()
    ev["event"] = ev["station"].map(lambda s: EVENT_SET[s][0])
    ev["set"] = ev["station"].map(lambda s: EVENT_SET[s][1])
    ev = ev.sort_values("event").reset_index(drop=True)
    assert len(ev) == 8, "events.csv no longer lists the 8 fixed events"
    wins = {r.station: event_windows(r._asdict()) for r in ev.itertuples()}

    sites0 = site_table()
    # One moved-engine row per event, at the destination's site.
    moved_rows = []
    for r in ev.itertuples():
        d = sites0[sites0["key"] == r.moved_to].iloc[0]
        moved_rows.append({"site_name": f"{r.station} engines at {r.moved_to}", "lat": d["lat"], "lon": d["lon"],
                           "is_existing": True, "key": f"MOVED:{r.station}"})
    sites = pd.concat([sites0, pd.DataFrame(moved_rows)], ignore_index=True)
    base_rows = np.arange(len(sites0))
    row_of = {k: i for i, k in enumerate(sites["key"])}

    print("Loading records…", flush=True)
    eras = {"2013-16": ("2012-10-01", "2016-12-31"), "2023": ("2022-01-01", "2024-06-30")}
    inc = pd.concat([load_incidents(*v) for v in eras.values()], ignore_index=True)
    mob = pd.concat([load_mobilisations(*v) for v in eras.values()], ignore_index=True)
    trips = engine_trips(mob, inc, sites0)
    cells = build_demand_grid(inc, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    print(f"  {len(inc):,} calls, {len(cells)} cells", flush=True)
    areas = {}
    for r in ev.itertuples():
        areas[r.station] = cell_owners(in_window(trips, wins[r.station]["area"]), cells) == r.station
    control = ~np.logical_or.reduce(list(areas.values()))

    # Travel matrices on the base sites; moved rows copy their destination's row.
    def with_moved(T):
        out = {}
        for k, M in T.items():
            M2 = np.vstack([M, np.zeros((len(moved_rows), M.shape[1]))])
            for r in ev.itertuples():
                M2[row_of[f"MOVED:{r.station}"]] = M[row_of[r.moved_to]]
            out[k] = M2
        return out

    inc_old = in_window(inc, (pd.Timestamp(eras["2013-16"][0]), pd.Timestamp(eras["2013-16"][1])))
    T_old = with_moved(travel_matrices(sites0, cells, inc_old, joblib.load(PRE2014_MODEL)))
    T_new = with_moved(current_matrices(sites0, cells, in_window(inc, (pd.Timestamp(eras["2023"][0]), pd.Timestamp(eras["2023"][1])))))
    from firestation.dispatch import load_dispatch

    dispatch = load_dispatch()

    # Each event's move, engines and pre-move turnout (used when it is in place during another event's window).
    moves = {r.station: (pd.Timestamp(r.start), pd.Timestamp(r.end)) for r in ev.itertuples()}
    n_eng = {r.station: len(str(r.own_engines).split()) for r in ev.itertuples()}
    pre_turn = {r.station: float(turnout_in(trips, wins[r.station]["area"]).get(r.station, np.nan)) for r in ev.itertuples()}

    rng = np.random.default_rng(0)
    rows, boot = [], {}
    for r in ev.itertuples():
        w = wins[r.station]
        T = T_new if r.station in CURRENT_MODEL else T_old
        halves, recs, preds, concurrent = [], [], [], {}
        for cmp_name in ("before", "after"):
            if cmp_name not in w:
                continue
            others = {o: moves[o] for o in moves if o != r.station}
            moved_cmp = in_place(w[cmp_name], others) if CONCURRENT else []
            moved_dur = (in_place(w["during"], others) if CONCURRENT else []) + [r.station]

            def layout(win, gone):
                rows_ = np.setdiff1d(open_rows(sites0, win), [row_of[g] for g in gone])
                return np.r_[rows_, [row_of[f"MOVED:{g}"] for g in gone]].astype(int)

            mv = [{"row": row_of[f"MOVED:{g}"], "from": row_of[g], "n": n_eng[g],
                   "turnout": None if g == r.station else pre_turn[g]} for g in sorted(set(moved_cmp) | set(moved_dur))]
            h = Half(cmp=w[cmp_name], during=w["during"], open_cmp=layout(w[cmp_name], moved_cmp),
                     open_during=layout(w["during"], moved_dur), moved=mv)
            concurrent[cmp_name] = {"comparison": moved_cmp, "during": [g for g in moved_dur if g != r.station]}
            if recorded:
                est, b, n_area = recorded_half(inc, h, cells, areas[r.station], control, rng, BOOT)
                recs.append((est, b))
            else:
                n_area = int(areas[r.station][calls_cells(inc, w["during"], cells)[0]].sum())
            preds.append(predicted_half(T, sites, inc, mob, trips, h, cells, areas[r.station], control, dispatch))
            halves.append(cmp_name)
        pred = {k: float(np.mean([p[k] for p in preds])) for k in preds[0]}
        if recorded:
            rec = float(np.mean([x[0] for x in recs]))
            b = np.mean([x[1] for x in recs], axis=0)
            boot[r.station] = b
        row = {"event": r.event, "station": r.station, "moved_to": r.moved_to, "set": r.set,
               "km": round(float(haversine_km(sites0.loc[row_of[r.station], "lon"], sites0.loc[row_of[r.station], "lat"],
                                              sites0.loc[row_of[r.moved_to], "lon"], sites0.loc[row_of[r.moved_to], "lat"])), 1),
               "comparison": "+".join(halves),
               "other_moves_in_place": "; ".join(f"{k}: cmp [{' '.join(v['comparison'])}] during [{' '.join(v['during'])}]"
                                                for k, v in concurrent.items() if v["comparison"] or v["during"]), "during_from": str(w["during"][0].date()), "during_to": str(w["during"][1].date()),
               "area_cells": int(areas[r.station].sum()), "area_calls_during": n_area,
               "travel_models": "current (2021-24)" if r.station in CURRENT_MODEL else "pre-2014",
               **({"recorded_s": round(rec, 2), "lo": round(float(np.percentile(b, 2.5)), 2),
                   "hi": round(float(np.percentile(b, 97.5)), 2)} if recorded else {}),
               **{f"pred: {k}": round(v, 2) for k, v in pred.items()}}
        rows.append(row)
        print(f"  {r.event} {r.station:12s} " + (f"recorded {rec:+6.1f} [{row['lo']:+.1f}, {row['hi']:+.1f}]  " if recorded else "")
              + f"LightGBM busy {pred[PRIMARY]:+6.1f}", flush=True)
    table = pd.DataFrame(rows)
    if not recorded:
        sc = table[table["set"] == "scored"]
        wts = sc["area_calls_during"].to_numpy(float)
        return {"table": table, "overall_pred_s": {c[len("pred: "):]: round(float(sc[c].to_numpy() @ wts / wts.sum()), 2)
                                                   for c in table.columns if c.startswith("pred: ")}, "cells": int(len(cells))}

    # Scoring over the scored events.
    sc = table[table["set"] == "scored"]
    wts = sc["area_calls_during"].to_numpy(float)
    overall = np.array([boot[s] for s in sc["station"]]).T @ wts / wts.sum()
    rec_all = float(sc["recorded_s"].to_numpy() @ wts / wts.sum())
    clear = (sc["lo"] > 0) | (sc["hi"] < 0)
    models = {}
    for col in [c for c in table.columns if c.startswith("pred: ")]:
        k = col[len("pred: "):]
        p = sc[col].to_numpy()
        pa = float(p @ wts / wts.sum())
        models[k] = {"direction_ok": bool((np.sign(p[clear.to_numpy()]) == np.sign(sc["recorded_s"].to_numpy()[clear.to_numpy()])).all()),
                     "overall_pred_s": round(pa, 2),
                     "overall_in_range": bool(np.percentile(overall, 2.5) <= pa <= np.percentile(overall, 97.5)),
                     "mae_s": round(float(np.abs(p - sc["recorded_s"].to_numpy()).mean()), 2),
                     "bias_s": round(float((p - sc["recorded_s"].to_numpy()).mean()), 2)}
        models[k]["mae_ok"] = models[k]["mae_s"] <= PASS_MAE_S
    prim = models[PRIMARY]
    best = min(models, key=lambda k: models[k]["mae_s"])
    second = sorted(models[k]["mae_s"] for k in models)[1] if len(models) > 1 else None
    return {"events": table.to_dict(orient="records"), "table": table,
            "overall_recorded_s": round(rec_all, 2), "overall_lo": round(float(np.percentile(overall, 2.5)), 2),
            "overall_hi": round(float(np.percentile(overall, 97.5)), 2), "events_with_clear_recorded_effect": sc.loc[clear, "station"].tolist(),
            "models": models, "primary": PRIMARY,
            "verdict": {"1_direction": prim["direction_ok"], "2_overall_size": prim["overall_in_range"], "3_mae_20s": prim["mae_ok"]},
            "all_pass": bool(prim["direction_ok"] and prim["overall_in_range"] and prim["mae_ok"]),
            "lowest_mae_model": best,
            "lowest_mae_better_by_5s": bool(second is not None and second - models[best]["mae_s"] >= BETTER_BY_S),
            "cells": int(len(cells))}


def current_matrices(sites, cells, inc) -> dict:
    """Lee Green: the current LightGBM model and Part 2's Kolesar fits (2024)."""
    from firestation.cities import get_city, load_fire_stations
    from firestation.cities import load_incidents as load_planner
    from firestation.facility import TravelTimeSurrogate
    from firestation.fire_station_location import build_travel_matrix, drop_far_incidents, fit_scorer_kolesar

    spec = get_city("london")
    houses = load_fire_stations(spec)
    demand = drop_far_incidents(load_planner(spec, max_rows=None, years=[2024]), houses)
    kmodel, _, _ = fit_scorer_kolesar(demand, houses)
    names = {"lgbm": "lightgbm", "road": "kolesar(road_km)", "kolesar": "kolesar(crow_km)"}
    return {names[sc]: build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer=sc, kolesar=kmodel,
                                           incidents=inc)[0] for sc in names}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-2014", action="store_true", help="Run only the code check on the 2014 closures")
    ap.add_argument("--predictions-only", action="store_true",
                    help="Write the models' predicted effects per event (reads no recorded attendance in any event area)")
    args = ap.parse_args()
    t0 = time.time()
    if args.predictions_only:
        res = run_test(recorded=False)
        res.pop("table").to_csv(PREDICTED, index=False)
        print(json.dumps(res, indent=2))
        return 0
    if args.check_2014:
        (OUT / "check_2014").mkdir(parents=True, exist_ok=True)
        res = check_2014()
        (OUT / "check_2014" / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
        return 0 if res["pass"] else 1
    if (OUT / "result.json").exists():
        print("Already run once; see", OUT / "result.json")
        return 1
    if not committed_clean(PREDICTION, Path(__file__).resolve(), EVENTS):
        print("Commit the prediction, events.csv and this script first.")
        return 1
    chk = check_2014()
    if not chk["pass"]:
        print("Code check failed: the script does not reproduce the 2014 results. Not running the test.")
        return 1
    res = run_test()
    res["code_check_2014"] = chk
    if PREDICTED.exists():  # the committed predictions must be what the test scored
        fixed = pd.read_csv(PREDICTED).set_index("station")
        now = res["table"].set_index("station")
        cols = [c for c in fixed.columns if c.startswith("pred: ")]
        res["matches_committed_predictions"] = bool(np.allclose(fixed.loc[now.index, cols], now[cols], atol=0.01))
    res["minutes"] = round((time.time() - t0) / 60, 1)
    res.pop("table").to_csv(OUT / "relocation_test.csv", index=False)
    (OUT / "result.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("events", "code_check_2014")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
