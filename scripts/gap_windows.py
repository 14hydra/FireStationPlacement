#!/usr/bin/env python3
"""When does each station's local engine seem to be unavailable, and is that real?

Model: each station is in one of two hidden states on a 15-minute grid, "available" or "away".
Calls where the local engine arrived first and gap calls (first_engine_gap.py) are the only input.
Away means a gap call is likely. The chance of going away and of coming back depends on the hour and on
weekday/weekend. Fitted with the Baum-Welch method, separately for one-engine and two-engine stations.
Calls within 20 minutes of an engine leaving its last incident are ignored, so return trips cannot create a window.

Windows made from gap calls are bound to contain gap calls, so that is not evidence. Tests use what the model did not see:

  1. Hidden calls: hide 20 % of calls and predict them, against a baseline of station x hour x weekday/weekend.
     Scored only where the neighbouring calls are in a different 500 m square.
  2. Same model on other years: settings fixed on 2024, then 2021-23. 2025-26 stay unused for the final test.
  3. Trips to other areas: mobilisations of the station's engines to calls in other station areas, per free-engine hour,
     comparing likely away (above 80 %) with likely available (below 20 %) in the same station x hour x weekday/weekend.
     Read against the simulated baseline in test 7, not against 1. Well below it means the engine was used less for all calls;
     equal to it means only how the engine was chosen for local calls changed.
     Placebo: the same comparison with windows moved 7 days.
  4. Delay codes and depot: on those trips, how often "On outside duty", "Other Station", and how long attendance takes.
  5. Does the model reproduce the runs in gap_cause/clustering.csv? Simulated from the fitted model.
  6. Repeat cycle: correlation of a station's away time on days k apart, from the model and from raw gap calls, each year.
     Then: does away time k days earlier predict gap calls today? 7 and 21 days act as comparison lags.
  7. Built-in bias in test 3: calls just after a trip are ignored, which removes evidence around trips. So labels are
     simulated independently of the trips and test 3 repeated. Any drop below 1 there comes from the method.
     Also repeated with no return-trip allowance, which should bias the other way.

Limits: this finds periods when the engine or how it was chosen changed. It cannot say why. A window needs calls to be seen,
so short windows and quiet stations are missed. Estimates at night are weak because few windows occur.

Writes data/figures/validation/gap_windows/: parameters.csv, hidden_calls.csv, other_area_trips.csv, codes_on_trips.csv,
runs_check.csv, repeat_cycle.csv, repeat_predicts.csv, away_by_hour.csv, by_station.csv, windows_2024.csv, by_quarter_hour.csv,
trips_method_bias.csv, summary.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/gap_windows.py
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

from busy_engine_check import MOB_FILES, RAW, station_sweep  # noqa: E402
from first_engine_gap import attach_road, group_calls, load_year, sec  # noqa: E402

OUT = ROOT / "data" / "figures" / "validation" / "gap_windows"
FIT_YEAR = 2024
CHECK_YEARS = (2021, 2022, 2023)
STEP_MIN = 15
STEPS_DAY = 24 * 60 // STEP_MIN
BUFFER_S = 1200
HIDE = 0.2
EM_ITERS = 40
N_BOOT = 300
HI, LO = 0.8, 0.2
BINS = [(0, 15), (15, 30), (30, 60), (60, 120), (120, 240), (240, 480), (480, 1440)]
OUTSIDE = "On outside duty when mobilised"
CACHE = os.environ.get("GAP_CACHE")


def prepare(year: int, mob_cache: dict) -> dict:
    if CACHE and (Path(CACHE) / f"{year}.pkl").exists():
        return pd.read_pickle(Path(CACHE) / f"{year}.pkl")
    c, m = load_year(year, mob_cache)
    c = attach_road(c)
    c["cat"], _ = group_calls(c, m, buffer_s=BUFFER_S)
    c["cat_no_buffer"], _ = group_calls(c, m, buffer_s=0.0)
    use = m.groupby("Resource_Code").size()
    regular = use[use >= 100].index
    pumps = pd.Series(regular.str[:3]).value_counts()
    extra = pd.read_csv(RAW / MOB_FILES[year], usecols=["IncidentNumber", "Resource_Code", "CalYear", "DelayCode_Description", "AttendanceTimeSeconds"],
                        dtype={"IncidentNumber": str}, low_memory=False, encoding="utf-8-sig")
    extra = extra[extra["CalYear"] == year].drop(columns="CalYear").drop_duplicates(["IncidentNumber", "Resource_Code"])
    m = m.merge(extra, on=["IncidentNumber", "Resource_Code"], how="left")
    keep = ["IncidentNumber", "t0", "HourOfCall", "ground", "ground_name", "first_name", "own_km", "first_km", "cat", "cat_no_buffer", "east", "north", "att"]
    out = {"calls": c[keep].copy(), "mob": m[m["Resource_Code"].isin(regular)].copy(), "pumps": pumps}
    if CACHE:
        Path(CACHE).mkdir(parents=True, exist_ok=True)
        pd.to_pickle(out, Path(CACHE) / f"{year}.pkl")
    return out


class Grid:
    """Calls and clock labels for one year on the 15-minute grid. Every station shares the same steps."""

    def __init__(self, year: int, stations: list[str]):
        self.year = year
        self.start = pd.Timestamp(f"{year}-01-01")
        self.T = int((pd.Timestamp(f"{year + 1}-01-01") - self.start) / pd.Timedelta(minutes=STEP_MIN))
        self.stations = stations
        self.index = {s: i for i, s in enumerate(stations)}
        times = self.start + pd.to_timedelta(np.arange(self.T) * STEP_MIN, unit="min")
        self.hour = times.hour.to_numpy()
        self.weekend = (times.dayofweek >= 5).astype(int)
        self.cell = self.hour + 24 * self.weekend  # 48 clock groups

    def step(self, t: pd.Series) -> np.ndarray:
        return ((t - self.start).dt.total_seconds().to_numpy() // (STEP_MIN * 60)).astype(int)

    def counts(self, e: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        g = np.zeros((len(self.stations), self.T), dtype=np.int16)
        loc = np.zeros_like(g)
        i, t = e["ground"].map(self.index).to_numpy(), self.step(e["t0"])
        y = e["gap"].to_numpy()
        np.add.at(g, (i[y == 1], t[y == 1]), 1)
        np.add.at(loc, (i[y == 0], t[y == 0]), 1)
        return g, loc


def forward_backward(grid: Grid, g, loc, par, want_xi=False):
    """Scaled forward-backward across all stations at once. Returns P(away) and, if asked, counts for refitting."""
    S, T = g.shape
    p_in, p_out = par["p_in"][grid.cell], par["p_out"][grid.cell]
    e_a, e_u = par["e_a"], par["e_u"]
    la, lu = np.log(e_a)[:, None], np.log(1 - e_a)[:, None]
    logA = g * la + loc * lu
    logU = g * np.log(e_u) + loc * np.log(1 - e_u)
    # Keep only the difference: most steps have no calls and need no likelihood.
    r = np.exp(logU - logA)  # likelihood away / available
    start = float(par["p_in"].mean() / (par["p_in"].mean() + par["p_out"].mean()))
    a = np.full(S, start)  # P(away | calls so far), after this step's calls
    fwd = np.zeros((S, T), dtype=np.float32)
    a = a * r[:, 0] / (a * r[:, 0] + (1 - a))
    fwd[:, 0] = a
    for t in range(1, T):
        pred = a * (1 - p_out[t]) + (1 - a) * p_in[t]
        num = pred * r[:, t]
        a = num / (num + (1 - pred))
        fwd[:, t] = a
    post = np.zeros((S, T), dtype=np.float32)
    post[:, -1] = fwd[:, -1]
    if want_xi:
        n = len(par["p_in"])
        to_u, from_a, to_a, from_u = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    b = post[:, -1].astype(float)
    for t in range(T - 2, -1, -1):
        f = fwd[:, t].astype(float)
        po, pi = p_out[t + 1], p_in[t + 1]
        pred = np.clip(f * (1 - po) + (1 - f) * pi, 1e-12, 1 - 1e-12)
        # P(state t, state t+1 | all) from the smoothed next state.
        uu = f * (1 - po) * b / pred
        au = (1 - f) * pi * b / pred
        ua = f * po * (1 - b) / (1 - pred)
        aa = (1 - f) * (1 - pi) * (1 - b) / (1 - pred)
        b = uu + ua
        post[:, t] = b
        if want_xi:
            k = grid.cell[t + 1]
            to_u[k] += au.sum()
            from_a[k] += au.sum() + aa.sum()
            to_a[k] += ua.sum()
            from_u[k] += ua.sum() + uu.sum()
    if want_xi:
        return post, (to_u, from_a, to_a, from_u)
    return post


def fit(grid: Grid, g, loc, *, iters: int = EM_ITERS) -> dict:
    S = g.shape[0]
    par = {"p_in": np.full(48, 0.01), "p_out": np.full(48, 0.15), "e_a": np.full(S, 0.06), "e_u": 0.9}
    for _ in range(iters):
        post, (to_u, from_a, to_a, from_u) = forward_backward(grid, g, loc, par, want_xi=True)
        par["p_in"] = (to_u + 0.5) / (from_a + 100)
        par["p_out"] = (to_a + 2) / (from_u + 20)
        gg, ll = g.astype(float), loc.astype(float)
        gap_a = ((1 - post) * gg).sum(1)
        n_a = ((1 - post) * (gg + ll)).sum(1)
        overall = gap_a.sum() / max(n_a.sum(), 1)
        par["e_a"] = np.clip((gap_a + 20 * overall) / (n_a + 20), 0.002, 0.5)  # small stations lean on the average
        par["e_u"] = float(np.clip((post * gg).sum() / max((post * (gg + ll)).sum(), 1), 0.5, 0.995))
    return par


def split_fit(grid: Grid, e: pd.DataFrame, pumps: pd.Series) -> tuple[dict, dict]:
    """One set of settings for one-engine and one for two-engine stations."""
    pars = {}
    for k in (1, 2):
        stations = [s for s in grid.stations if min(int(pumps.get(s, 1)), 2) == k]
        sub = Grid(grid.year, stations)
        gk, lk = sub.counts(e[e["ground"].isin(stations)])
        pars[k] = {**fit(sub, gk, lk), "stations": stations}
    return pars


def apply_pars(grid: Grid, e: pd.DataFrame, pars: dict) -> np.ndarray:
    """P(away) for grid.stations, using each class's settings. Stations new to the class use the class average."""
    post = np.zeros((len(grid.stations), grid.T), dtype=np.float32)
    for k, p in pars.items():
        have = {s: i for i, s in enumerate(p["stations"])}
        stations = [s for s in grid.stations if s in have]
        if not stations:
            continue
        sub = Grid(grid.year, stations)
        par = {**p, "e_a": p["e_a"][[have[s] for s in stations]]}
        gk, lk = sub.counts(e[e["ground"].isin(stations)])
        post[[grid.index[s] for s in stations]] = forward_backward(sub, gk, lk, par)
    return post


def auc(y: np.ndarray, x: np.ndarray) -> float:
    r = pd.Series(x).rank().to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def score_hidden(year: int, e: pd.DataFrame, hide: np.ndarray, pred: np.ndarray, base: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Scores hidden calls whose neighbours among visible calls are in a different 500 m square."""
    d = e.assign(hide=hide, pred=pred, base=base, sq=(e["east"] // 500).astype(int).astype(str) + "_" + (e["north"] // 500).astype(int).astype(str))
    d = d.sort_values(["ground", "t0"])
    vis = d["sq"].where(~d["hide"])
    d["prev_sq"] = vis.groupby(d["ground"]).ffill().groupby(d["ground"]).shift(1)
    d["next_sq"] = vis.groupby(d["ground"]).bfill().groupby(d["ground"]).shift(-1)
    s = d[d["hide"] & (d["prev_sq"] != d["sq"]) & (d["next_sq"] != d["sq"])]
    y = s["gap"].to_numpy()
    row = {"year": year, "hidden_calls_scored": len(s), "gap_share_pct": round(float(y.mean() * 100), 1),
           "auc_model": round(auc(y, s["pred"].to_numpy()), 3), "auc_baseline": round(auc(y, s["base"].to_numpy()), 3),
           "logloss_model": round(logloss(y, s["pred"].to_numpy()), 4), "logloss_baseline": round(logloss(y, s["base"].to_numpy()), 4)}
    edges = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    tab = s.assign(bucket=pd.cut(s["pred"], edges, include_lowest=True)).groupby("bucket", observed=True).agg(
        calls=("gap", "size"), predicted_pct=("pred", "mean"), actual_pct=("gap", "mean"), baseline_pct=("base", "mean")).reset_index()
    for col in ("predicted_pct", "actual_pct", "baseline_pct"):
        tab[col] = (tab[col] * 100).round(1)
    tab["bucket"] = tab["bucket"].astype(str)
    return row, tab.assign(year=year)


def hidden_test(year: int, grid: Grid, e: pd.DataFrame, pumps: pd.Series, frozen: dict | None, seed: int) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    hide = rng.random(len(e)) < HIDE
    train = e[~hide]
    pars = frozen if frozen is not None else split_fit(grid, train, pumps)
    post = apply_pars(grid, train, pars)
    p_u = post[e["ground"].map(grid.index).to_numpy(), grid.step(e["t0"])]
    e_a = {s: float(v) for p in pars.values() for s, v in zip(p["stations"], p["e_a"])}
    e_u = {s: float(p["e_u"]) for p in pars.values() for s in p["stations"]}
    pred = p_u * e["ground"].map(e_u).fillna(0.9).to_numpy() + (1 - p_u) * e["ground"].map(e_a).fillna(0.06).to_numpy()
    wk = (e["t0"].dt.dayofweek >= 5).astype(int)
    keys = e["ground"].astype(str) + "|" + e["HourOfCall"].astype(str) + "|" + wk.astype(str)
    train_rate = train.assign(k=keys[~hide]).groupby("k")["gap"].agg(["sum", "size"])
    overall = train["gap"].mean()
    base = ((keys.map(train_rate["sum"]).fillna(0) + 5 * overall) / (keys.map(train_rate["size"]).fillna(0) + 5)).to_numpy()
    return score_hidden(year, e, hide, pred, base)


def busy_grid(grid: Grid, mob: pd.DataFrame, pumps: pd.Series) -> np.ndarray:
    """True where every regular engine of the station is on a recorded incident at the step's midpoint."""
    mid = sec(pd.Series(grid.start + pd.to_timedelta((np.arange(grid.T) + 0.5) * STEP_MIN, unit="min")))
    busy = np.zeros((len(grid.stations), grid.T), dtype=bool)
    b = mob.dropna(subset=["t_left"])
    dur = (b["t_left"] - b["t0"]).dt.total_seconds()
    b = b[(dur > 0) & (dur < 6 * 3600)]
    for s, gb in b.groupby("home"):
        if s not in grid.index:
            continue
        ev, n, _ = station_sweep(sec(gb["t0"]), sec(gb["t_left"]) + BUFFER_S, int(pumps.get(s, 1)))
        j = np.searchsorted(ev, mid, side="right") - 1
        ok = j >= 0
        busy[grid.index[s]][ok] = n[j[ok]] >= pumps.get(s, 1)
    return busy


def rate_ratio(nu, eu, na, ea) -> float:
    """Mantel-Haenszel rate ratio across station x clock groups. Inputs are arrays, one value per group."""
    tot = eu + ea
    ok = tot > 0
    top = (nu * ea / np.where(ok, tot, 1)).sum()
    bot = (na * eu / np.where(ok, tot, 1)).sum()
    return float(top / bot) if bot > 0 else np.nan


def trips_test(year: int, grid: Grid, post: np.ndarray, calls: pd.DataFrame, mob: pd.DataFrame, pumps: pd.Series, seed: int,
               classes: dict | None = None, boot: int = N_BOOT, placebo: bool = True) -> tuple[list, list]:
    ground = calls.drop_duplicates("IncidentNumber").set_index("IncidentNumber")["ground"]
    t = mob[mob["home"].isin(grid.stations) & (mob["t0"].dt.year == year)].copy()
    t["inc_ground"] = t["IncidentNumber"].map(ground)
    t = t[t["inc_ground"].notna() & (t["inc_ground"] != t["home"])]
    t["s"], t["step"] = t["home"].map(grid.index), grid.step(t["t0"])
    busy = busy_grid(grid, mob, pumps)
    S = len(grid.stations)
    trips = np.zeros((S, grid.T), dtype=np.int16)
    np.add.at(trips, (t["s"].to_numpy(), t["step"].to_numpy()), 1)
    rng = np.random.default_rng(seed)

    def counts(p: np.ndarray, lo_hi: tuple[float, float] = (HI, LO)) -> tuple:
        hi, lo = lo_hi
        nu, eu, na, ea = (np.zeros((S, 48)) for _ in range(4))
        free = ~busy
        for i in range(S):
            for mask, n_arr, e_arr in ((p[i] >= hi, nu, eu), (p[i] < lo, na, ea)):
                m = mask & free[i]
                np.add.at(n_arr[i], grid.cell[m], trips[i][m])
                np.add.at(e_arr[i], grid.cell[m], STEP_MIN / 60)
        return nu, eu, na, ea

    results = []
    variants = {"as fitted": post}
    if placebo:
        variants["windows moved 7 days (placebo)"] = np.roll(post, 7 * STEPS_DAY, axis=1)
    classes = classes or {"all": np.ones(S, dtype=bool)}
    for name, p in variants.items():
        NU, EU, NA, EA = counts(p)
        for label, keep in classes.items():
            nu, eu, na, ea = NU[keep], EU[keep], NA[keep], EA[keep]
            rr = rate_ratio(nu.ravel(), eu.ravel(), na.ravel(), ea.ravel())
            draws = []
            for _ in range(boot):
                w = rng.poisson(1.0, len(nu))[:, None]
                draws.append(rate_ratio((nu * w).ravel(), (eu * w).ravel(), (na * w).ravel(), (ea * w).ravel()))
            both = (eu > 0) & (ea > 0)
            results.append({"year": year, "windows": name, "stations": label, "away_free_hours": round(float(eu.sum())), "available_free_hours": round(float(ea.sum())),
                            "trips_away": int(nu.sum()), "trips_available": int(na.sum()),
                            "trips_per_100h_away_matched": round(float(nu[both].sum() / max(eu[both].sum(), 1) * 100), 2),
                            "trips_per_100h_available_matched": round(float(na[both].sum() / max(ea[both].sum(), 1) * 100), 2),
                            "rate_ratio": round(rr, 3), "ratio_lo": round(float(np.nanpercentile(draws, 2.5)), 3) if boot else np.nan,
                            "ratio_hi": round(float(np.nanpercentile(draws, 97.5)), 3) if boot else np.nan})

    # Delay codes on these trips: same station x clock group, away vs available.
    t["p_away"] = post[t["s"].to_numpy(), t["step"].to_numpy()]
    t["busy"] = busy[t["s"].to_numpy(), t["step"].to_numpy()]
    t = t[~t["busy"]]
    t["outside"] = (t["DelayCode_Description"] == OUTSIDE).astype(float)
    t["other_depot"] = (t["DeployedFromLocation"] == "Other Station").astype(float)
    t["att_s"] = pd.to_numeric(t["AttendanceTimeSeconds"], errors="coerce")
    t["cellkey"] = t["s"].astype(str) + "|" + pd.Series(grid.cell[t["step"].to_numpy()], index=t.index).astype(str)
    a, b = t[t["p_away"] >= HI], t[t["p_away"] < LO]
    shared = set(a["cellkey"]) & set(b["cellkey"])
    a, b = a[a["cellkey"].isin(shared)], b[b["cellkey"].isin(shared)]
    weight = a.groupby("cellkey").size()
    rows = []
    for col, label in (("outside", "on outside duty, %"), ("other_depot", "from another station, %"), ("att_s", "attendance, s")):
        scale = 1 if col == "att_s" else 100
        va = a.groupby("cellkey")[col].mean()
        vb = b.groupby("cellkey")[col].mean().reindex(va.index)
        ok = vb.notna() & va.notna()
        rows.append({"year": year, "measure": label, "trips_away": len(a), "trips_available_matched": len(b),
                     "when_away": round(float(np.average(va[ok], weights=weight[va.index][ok])) * scale, 2),
                     "when_available": round(float(np.average(vb[ok], weights=weight[va.index][ok])) * scale, 2)})
    return results, rows


def simulate_labels(grid: Grid, e: pd.DataFrame, pars: dict, seed: int) -> np.ndarray:
    """Gap labels drawn from the fitted model at the real call times, independently of anything else."""
    rng = np.random.default_rng(seed)
    S = len(grid.stations)
    p_in, p_out = np.zeros((S, 48)), np.zeros((S, 48))
    e_a, e_u = np.zeros(S), np.zeros(S)
    for p in pars.values():
        for k, s in enumerate(p["stations"]):
            i = grid.index[s]
            p_in[i], p_out[i], e_a[i], e_u[i] = p["p_in"], p["p_out"], p["e_a"][k], p["e_u"]
    state = rng.random(S) < 0.05
    away = np.zeros((S, grid.T), dtype=bool)
    for t in range(grid.T):
        k = grid.cell[t]
        u = rng.random(S)
        state = np.where(state, u >= p_out[:, k], u < p_in[:, k])
        away[:, t] = state
    i, t = e["ground"].map(grid.index).to_numpy(), grid.step(e["t0"])
    return (rng.random(len(e)) < np.where(away[i, t], e_u[i], e_a[i])).astype(int)


def simulate_runs(grid: Grid, e: pd.DataFrame, pars: dict, seed: int) -> pd.DataFrame:
    """Runs as in gap_cause_checks.py, measured on simulated labels."""
    sim = simulate_labels(grid, e, pars, seed)
    return run_table(e, sim).rename(columns={"after_gap_pct": "sim_after_gap", "after_local_pct": "sim_after_local", "diff_points": "sim_diff"})


def run_table(e: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    d = e.assign(y=y).sort_values(["ground", "t0"])
    t = sec(d["t0"]) / 60
    same = d["ground"].to_numpy()[1:] == d["ground"].to_numpy()[:-1]
    dt = np.r_[np.inf, np.where(same, t[1:] - t[:-1], np.inf)]
    sx, sy = (d["east"] // 500).astype(int).to_numpy(), (d["north"] // 500).astype(int).to_numpy()
    other = np.r_[False, (sx[1:] != sx[:-1]) | (sy[1:] != sy[:-1])]
    v = d["y"].to_numpy()
    prev = np.r_[0, v[:-1]]
    rows = []
    for a, b in BINS:
        m = other & (dt >= a) & (dt < b)
        x, p = v[m], prev[m]
        rows.append({"window_min": f"{a}-{b}", "after_gap_pct": round(float(x[p == 1].mean() * 100), 1),
                     "after_local_pct": round(float(x[p == 0].mean() * 100), 1),
                     "diff_points": round(float((x[p == 1].mean() - x[p == 0].mean()) * 100), 1)})
    return pd.DataFrame(rows)


def windows_from(grid: Grid, post: np.ndarray, e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gaps = e[e["gap"] == 1]
    gs = {s: np.sort(grid.step(x["t0"])) for s, x in gaps.groupby("ground")}
    ls = {s: np.sort(grid.step(x["t0"])) for s, x in e[e["gap"] == 0].groupby("ground")}
    for s, i in grid.index.items():
        on = post[i] >= 0.5
        d = np.diff(np.r_[0, on.astype(int), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            gv = gs.get(s, np.array([], dtype=int))
            lv = ls.get(s, np.array([], dtype=int))
            n_gap = int(np.searchsorted(gv, b) - np.searchsorted(gv, a))
            n_loc = int(np.searchsorted(lv, b) - np.searchsorted(lv, a))
            first = np.searchsorted(gv, a)
            inner = (gv[first + n_gap - 1] - gv[first]) * STEP_MIN / 60 if n_gap else 0.0
            # Longest it could be: from the last local-first call before the first gap to the next one after the last gap.
            before = lv[np.searchsorted(lv, gv[first]) - 1] if n_gap and np.searchsorted(lv, gv[first]) > 0 else np.nan
            j = np.searchsorted(lv, gv[first + n_gap - 1], side="right") if n_gap else len(lv)
            after = lv[j] if j < len(lv) else np.nan
            rows.append({"station": s, "start": grid.start + pd.Timedelta(minutes=int(a) * STEP_MIN),
                         "end": grid.start + pd.Timedelta(minutes=int(b) * STEP_MIN), "hours": (b - a) * STEP_MIN / 60,
                         "first_to_last_gap_h": inner, "local_to_local_h": (after - before) * STEP_MIN / 60,
                         "gap_calls": n_gap, "local_calls": n_loc, "peak_p_away": round(float(post[i, a:b].max()), 3)})
    return pd.DataFrame(rows)


def midday_day(grid: Grid, x: np.ndarray) -> np.ndarray:
    days = grid.T // STEPS_DAY
    return x[:, :days * STEPS_DAY].reshape(len(grid.stations), days, STEPS_DAY)[:, :, 40:64].sum(2).astype(float)


def lag_corr(year: int, r: np.ndarray, source: str, seed: int, max_lag: int = 60) -> list:
    S = r.shape[0]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, S, (N_BOOT, S))
    rows = []
    for k in range(1, max_lag + 1):
        x, y = r[:, :-k], r[:, k:]
        xy, xx, yy = (x * y).sum(1), (x * x).sum(1), (y * y).sum(1)
        boot = xy[draws].sum(1) / np.sqrt(xx[draws].sum(1) * yy[draws].sum(1))
        rows.append({"year": year, "source": source, "days_apart": k, "correlation": round(float(xy.sum() / np.sqrt(xx.sum() * yy.sum())), 3),
                     "lo": round(float(np.percentile(boot, 2.5)), 3), "hi": round(float(np.percentile(boot, 97.5)), 3)})
    return rows


def repeat_cycle(year: int, grid: Grid, post: np.ndarray, e: pd.DataFrame, seed: int) -> list:
    """10:00-16:00 per station per day, minus station and date averages, correlated k days apart."""
    centre = lambda d: d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()  # noqa: E731
    rows = lag_corr(year, centre(midday_day(grid, post)), "model away time", seed)
    # Raw: gap calls beyond what the station's usual rate at that hour would give. No model involved.
    wk = (e["t0"].dt.dayofweek >= 5).astype(int)
    usual = e.groupby(["ground", e["HourOfCall"], wk])["gap"].transform("mean")
    excess = np.zeros((len(grid.stations), grid.T), dtype=np.float32)
    np.add.at(excess, (e["ground"].map(grid.index).to_numpy(), grid.step(e["t0"])), (e["gap"] - usual).to_numpy())
    rows += lag_corr(year, centre(midday_day(grid, excess)), "raw gap calls", seed + 1)
    return rows


def repeat_predicts(year: int, grid: Grid, post: np.ndarray, e: pd.DataFrame) -> list:
    """Midday gap share today by whether the same station looked away for 3+ midday hours k days earlier."""
    away_day = midday_day(grid, post) * STEP_MIN / 60 >= 3
    m = e[(e["HourOfCall"] >= 10) & (e["HourOfCall"] < 16)].copy()
    m["s"] = m["ground"].map(grid.index)
    m["d"] = (m["t0"] - grid.start).dt.days
    m = m[m["d"] < away_day.shape[1]]
    m["stratum"] = m["s"].astype(str) + "|" + (m["t0"].dt.dayofweek >= 5).astype(str) + "|" + m["HourOfCall"].astype(str)
    rows = []
    for k in (7, 14, 21, 28):
        v = m[m["d"] >= k]
        flag = away_day[v["s"].to_numpy(), v["d"].to_numpy() - k]
        adj = v["gap"] - v.groupby("stratum")["gap"].transform("mean")
        rows.append({"year": year, "days_earlier": k, "calls_flagged": int(flag.sum()), "gap_pct_flagged": round(float(v["gap"][flag].mean() * 100), 1),
                     "gap_pct_not_flagged": round(float(v["gap"][~flag].mean() * 100), 1),
                     "difference_same_station_hour": round(float((adj[flag].mean() - adj[~flag].mean()) * 100), 2)})
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    mob_cache: dict = {}
    data = {y: prepare(y, mob_cache) for y in (FIT_YEAR, *CHECK_YEARS)}
    mob_cache.clear()

    hidden, calib, trips, codes, cycle_rows, pred_rows, bias_rows = [], [], [], [], [], [], []
    pars24 = None
    for year in (FIT_YEAR, *CHECK_YEARS):
        d = data[year]
        calls, mob, pumps = d["calls"], d["mob"], d["pumps"]
        e = calls[calls["cat"].isin(["own_first", "real_gap"])].copy()
        e["gap"] = (e["cat"] == "real_gap").astype(int)
        if year == FIT_YEAR:
            stations = sorted(s for s in e["ground"].unique() if s in pumps.index)
            e = e[e["ground"].isin(stations)]
            grid = Grid(year, stations)
            pars24 = split_fit(grid, e, pumps)
            row, tab = hidden_test(year, grid, e, pumps, None, seed=year)
        else:
            known = [s for p in pars24.values() for s in p["stations"]]
            stations = sorted(s for s in e["ground"].unique() if s in known)
            e = e[e["ground"].isin(stations)]
            grid = Grid(year, stations)
            row, tab = hidden_test(year, grid, e, pumps, pars24, seed=year)
        hidden.append(row)
        calib.append(tab)
        print(row, flush=True)
        print(tab.to_string(index=False), flush=True)

        post = apply_pars(grid, e, pars24)
        have = {s: k for k, p in pars24.items() for s in p["stations"]}
        cls_y = np.array([have[s] for s in stations])
        classes = {"all": np.ones(len(stations), dtype=bool), "one engine": cls_y == 1, "two engines": cls_y == 2}
        r, cd = trips_test(year, grid, post, calls, mob, pumps, seed=year, classes=classes)
        trips += r
        codes += cd
        print(pd.DataFrame(r).to_string(index=False), flush=True)
        print(pd.DataFrame(cd).to_string(index=False), flush=True)

        # What the method does on its own: labels with no link to the trips.
        for n_sim in range(10):
            sim_e = e.assign(gap=simulate_labels(grid, e, pars24, seed=1000 * year + n_sim))
            sim_post = apply_pars(grid, sim_e, pars24)
            rs, _ = trips_test(year, grid, sim_post, calls, mob, pumps, seed=n_sim, classes=classes, boot=0, placebo=False)
            bias_rows += [{"year": year, "check": "simulated, no link to trips", "stations": x["stations"], "rate_ratio": x["rate_ratio"], "sim": n_sim} for x in rs]
        # No allowance for return trips: engines on the way back now look away, which pushes the ratio up.
        e0 = calls[calls["cat_no_buffer"].isin(["own_first", "real_gap"]) & calls["ground"].isin(stations)].copy()
        e0["gap"] = (e0["cat_no_buffer"] == "real_gap").astype(int)
        r0, _ = trips_test(year, grid, apply_pars(grid, e0, pars24), calls, mob, pumps, seed=year, classes=classes, boot=0, placebo=False)
        bias_rows += [{"year": year, "check": "no return-trip allowance", "stations": x["stations"], "rate_ratio": x["rate_ratio"], "sim": 0} for x in r0]
        print(pd.DataFrame(bias_rows)[lambda x: x["year"] == year].groupby(["check", "stations"])["rate_ratio"].agg(["mean", "min", "max"]).round(3).to_string(), flush=True)

        cycle_rows += repeat_cycle(year, grid, post, e, seed=year)
        pred_rows += repeat_predicts(year, grid, post, e)
        cy = pd.DataFrame(cycle_rows)[lambda x: x["year"] == year]
        print(cy[cy["days_apart"] <= 30].pivot(index="days_apart", columns="source", values="correlation").T.to_string(), flush=True)
        print(pd.DataFrame(pred_rows)[lambda x: x["year"] == year].to_string(index=False), flush=True)

        if year == FIT_YEAR:
            print("gap calls 'away' setting:", {k: round(p["e_u"], 3) for k, p in pars24.items()},
                  "available, median:", {k: round(float(np.median(p["e_a"])), 3) for k, p in pars24.items()}, flush=True)
            actual = run_table(e, e["gap"].to_numpy())
            sim = simulate_runs(grid, e, pars24, seed=1)
            runs = actual.merge(sim, on="window_min")
            print(runs.to_string(index=False), flush=True)
            runs.to_csv(OUT / "runs_check.csv", index=False)

            pm = [dict(pumps=k, wk=w, hour=h, p_in=round(float(p["p_in"][h + 24 * w]), 5), p_out=round(float(p["p_out"][h + 24 * w]), 4))
                  for k, p in pars24.items() for w in (0, 1) for h in range(24)]
            pm = pd.DataFrame(pm)
            eng = [{"pumps": k, "e_u": round(p["e_u"], 3), "e_a_median": round(float(np.median(p["e_a"])), 3), "stations": len(p["stations"])} for k, p in pars24.items()]
            pm.to_csv(OUT / "parameters.csv", index=False)

            cls = np.array([min(int(pumps.get(s, 1)), 2) for s in stations])
            hrs = []
            for k in (1, 2):
                v = post[cls == k]
                for w in (0, 1):
                    for h in range(24):
                        m = grid.cell == h + 24 * w
                        hrs.append({"pumps": k, "weekend": w, "hour": h, "away_pct": round(float(v[:, m].mean() * 100), 1)})
            hrs = pd.DataFrame(hrs)
            hrs.to_csv(OUT / "away_by_hour.csv", index=False)
            print(hrs.pivot_table(index="hour", columns=["pumps", "weekend"], values="away_pct").to_string(), flush=True)

            w = windows_from(grid, post, e)
            w.to_csv(OUT / "windows_2024.csv", index=False)
            sure = w[w["gap_calls"] >= 2]
            print(f"Windows {len(w)}; with 2+ gap calls {len(sure)}; hours median {sure['hours'].median()}, IQR {sure['hours'].quantile([.25, .75]).tolist()}", flush=True)
            print("Start hour of 2+ windows:", sure["start"].dt.hour.value_counts().sort_index().to_dict(), flush=True)
            print("End hour of 2+ windows:", sure["end"].dt.hour.value_counts().sort_index().to_dict(), flush=True)
            print("Longest:", sure.nlargest(10, "hours").to_string(index=False), flush=True)
            print("Length, hours, windows with 2+ gap calls:\n", sure[["hours", "first_to_last_gap_h", "local_to_local_h"]].describe().round(2).to_string(), flush=True)

            st = pd.DataFrame({"station": stations, "pumps": cls, "away_all_pct": (post.mean(1) * 100).round(1),
                               "away_10_16_pct": (post[:, (grid.hour >= 10) & (grid.hour < 16)].mean(1) * 100).round(1),
                               "gap_share_pct": (e.groupby("ground")["gap"].mean().reindex(stations).to_numpy() * 100).round(1)})
            st24 = st
            q = e.assign(qh=e["t0"].dt.hour * 4 + e["t0"].dt.minute // 15, wk=e["t0"].dt.dayofweek >= 5)
            q = q[(q["t0"].dt.hour >= 7) & (q["t0"].dt.hour < 20)].groupby(["wk", "qh"])["gap"].agg(["size", "mean"]).reset_index()
            q["clock"] = (q["qh"] // 4).astype(str).str.zfill(2) + ":" + (q["qh"] % 4 * 15).astype(str).str.zfill(2)
            q["mean"] = (q["mean"] * 100).round(1)
            q.to_csv(OUT / "by_quarter_hour.csv", index=False)
            print(q.pivot(index="clock", columns="wk", values="mean").to_string(), flush=True)
        else:
            s_y = pd.Series(post[:, (grid.hour >= 10) & (grid.hour < 16)].mean(1) * 100, index=stations)
            st24[f"away_10_16_{year}"] = st24["station"].map(s_y).round(1)

    st24.to_csv(OUT / "by_station.csv", index=False)
    print(st24.describe().round(1).to_string(), flush=True)
    corr = st24.filter(like="away_10_16").corr(method="spearman").round(2)
    print(corr.to_string(), flush=True)
    pd.DataFrame(hidden).to_csv(OUT / "hidden_calls.csv", index=False)
    pd.concat(calib).to_csv(OUT / "hidden_calls_calibration.csv", index=False)
    pd.DataFrame(trips).to_csv(OUT / "other_area_trips.csv", index=False)
    pd.DataFrame(codes).to_csv(OUT / "codes_on_trips.csv", index=False)
    pd.DataFrame(cycle_rows).to_csv(OUT / "repeat_cycle.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUT / "repeat_predicts.csv", index=False)
    bias = pd.DataFrame(bias_rows).groupby(["year", "check", "stations"])["rate_ratio"].agg(["mean", "min", "max"]).round(3).reset_index()
    bias.to_csv(OUT / "trips_method_bias.csv", index=False)
    print(bias.to_string(index=False))
    pc = pd.DataFrame(cycle_rows)
    print(pc[pc["source"] == "raw gap calls"].pivot(index="days_apart", columns="year", values="correlation").to_string())
    print(pc[pc["source"] == "model away time"].pivot(index="days_apart", columns="year", values="correlation").to_string())
    (OUT / "summary.json").write_text(json.dumps({"doc": __doc__, "settings": eng, "hidden_calls": hidden, "other_area_trips": trips, "codes_on_trips": codes, "trips_method_bias": bias.to_dict(orient="records"), "repeat_predicts": pred_rows,
                                                  "station_rank_corr": corr.to_dict()}, indent=2, default=str) + "\n")
    print(pd.DataFrame(hidden).to_string(index=False))
    print(pd.DataFrame(trips).to_string(index=False))
    print(pd.DataFrame(codes).to_string(index=False))
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
