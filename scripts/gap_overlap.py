#!/usr/bin/env python3
"""Are neighbouring stations' engines away at the same time, and what does that cost?

Uses 2021-24 and the 2024 settings of the gap_windows.py model. 2025-26 are not used.

  1. Pairs of calls: a call in one station's area, then within 60 minutes a call in a neighbouring station's area.
     Is the second more likely to be a gap call if the first was? No model involved. Neighbours are the 3 nearest stations;
     the comparison is station pairs at least 10 km apart. Null: 200 shuffles within station x hour x weekday/weekend. Pairs where
     the second station's engine was the one that reached the first call are left out, so one engine covering cannot create a link.
  2. Overlap in the model: how often station and neighbour are both away at 10:00-16:00, against the same pair on days 7 or 21 days apart.
     The model is unsure at most moments, so this understates any real link.
  3. Passing over the next station: on each gap call, the next-nearest station by road (local one left out) either reached it first,
     had every engine out, or looked free and was passed over too. Extra seconds compared with "reached it first", same
     500 m square and period. Limit = share of all calls x extra seconds. This is an upper limit for keeping the next
     station available. Not every passed-over engine was away: road distance is only an approximation of who is quickest.
     Check: does the model, using only that station's own area's calls, say it was more often away when passed over? Simulated labels
     give the baseline, because engines on trips leave less evidence.

Writes data/figures/validation/gap_overlap/: pairs.csv, overlap_model.csv, passed_over.csv, passed_over_model_check.csv, summary.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/gap_overlap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from busy_engine_check import PERIODS, RAW, station_sweep  # noqa: E402
from first_engine_gap import fixed_effects, sec  # noqa: E402
from gap_windows import BUFFER_S, CHECK_YEARS, FIT_YEAR, Grid, apply_pars, prepare, simulate_labels, split_fit  # noqa: E402

OUT = ROOT / "data" / "figures" / "validation" / "gap_overlap"
N_NEAR = 3
FAR_M = 10_000
PAIR_MIN = 60
N_PERM = 200
CANDIDATES = 4
MARGINS = (200, 500)  # metres of road distance treated as a tie, primary first


def station_table(names: dict) -> pd.DataFrame:
    from pyproj import Transformer

    st = pd.read_csv(RAW / "london_fire_stations.csv")
    st["key"] = st["name"].str.upper()
    st = st.set_index("key")
    rows = {c: n for c, n in names.items() if n in st.index}
    t = pd.DataFrame({"code": list(rows), "name": list(rows.values())})
    t["lat"], t["lon"] = st["lat"].reindex(t["name"]).to_numpy(float), st["lon"].reindex(t["name"]).to_numpy(float)
    t["x"], t["y"] = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True).transform(t["lon"].to_numpy(), t["lat"].to_numpy())
    return t


def neighbour_sets(t: pd.DataFrame, seed: int = 0) -> tuple[list, list]:
    xy = t[["x", "y"]].to_numpy()
    dist = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    np.fill_diagonal(dist, np.inf)
    rng = np.random.default_rng(seed)
    near, far = [], []
    codes = t["code"].to_numpy()
    for i in range(len(t)):
        near += [(codes[i], codes[j]) for j in np.argsort(dist[i])[:N_NEAR]]
        pool = np.flatnonzero((dist[i] >= FAR_M) & np.isfinite(dist[i]))
        far += [(codes[i], codes[j]) for j in rng.choice(pool, N_NEAR, replace=False)]
    return near, far


def pairs_of_calls(e: pd.DataFrame, pairs: list, names: dict) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of (first call, next call in the other area within the time limit)."""
    t = sec(e["t0"]) / 60
    by = {s: np.flatnonzero(e["ground"].to_numpy() == s) for s in e["ground"].unique()}
    by = {s: i[np.argsort(t[i])] for s, i in by.items()}
    first_name = e["first_name"].to_numpy()
    a_all, b_all = [], []
    for s, j in pairs:
        if s not in by or j not in by:
            continue
        ia, ib = by[s], by[j]
        pos = np.searchsorted(t[ib], t[ia], side="right")
        ok = pos < len(ib)
        a, b = ia[ok], ib[pos[ok]]
        keep = (t[b] - t[a] <= PAIR_MIN) & (first_name[a] != names.get(j, "?"))
        a_all.append(a[keep])
        b_all.append(b[keep])
    return np.concatenate(a_all), np.concatenate(b_all)


def pair_test(year: int, e: pd.DataFrame, near: list, far: list, names: dict) -> list:
    y = e["gap"].to_numpy()
    wk = (e["t0"].dt.dayofweek >= 5).astype(int)
    stratum = pd.factorize(e["ground"].astype(str) + "|" + e["HourOfCall"].astype(str) + "|" + wk.astype(str))[0]
    hour = e["HourOfCall"].to_numpy()
    rng = np.random.default_rng(year)
    order = np.argsort(stratum, kind="stable")
    perms = []
    for _ in range(N_PERM):
        p = np.empty_like(y)
        p[order] = y[np.lexsort((rng.random(len(y)), stratum))]
        perms.append(p)
    rows = []
    for label, sets in (("neighbours", near), ("10 km or more apart", far)):
        a, b = pairs_of_calls(e, sets, names)
        for scope, m in (("all day", np.ones(len(a), dtype=bool)), ("10:00-16:00", (hour[a] >= 10) & (hour[a] < 16))):
            a2, b2 = a[m], b[m]
            diff = lambda v: v[b2][v[a2] == 1].mean() - v[b2][v[a2] == 0].mean()  # noqa: E731
            null = np.array([diff(p) for p in perms])
            rows.append({"year": year, "stations": label, "hours": scope, "pairs": len(a2),
                         "next_gap_after_gap_pct": round(float(y[b2][y[a2] == 1].mean() * 100), 1),
                         "next_gap_after_local_pct": round(float(y[b2][y[a2] == 0].mean() * 100), 1),
                         "difference": round(float(diff(y) * 100), 2), "shuffled_lo": round(float(np.percentile(null, 2.5) * 100), 2),
                         "shuffled_hi": round(float(np.percentile(null, 97.5) * 100), 2)})
    return rows


def model_overlap(year: int, grid: Grid, post: np.ndarray, near: list, far: list) -> list:
    mid = (grid.hour >= 10) & (grid.hour < 16)
    rows = []
    days = 96
    for label, sets in (("neighbours", near), ("10 km or more apart", far)):
        idx = [(grid.index[a], grid.index[b]) for a, b in sets if a in grid.index and b in grid.index]
        i, j = np.array([x[0] for x in idx]), np.array([x[1] for x in idx])
        same = (post[i][:, mid] * post[j][:, mid]).mean()
        other = np.mean([(post[i][:, mid] * np.roll(post[j], k * days, axis=1)[:, mid]).mean() for k in (7, 21)])
        rows.append({"year": year, "stations": label, "both_away_same_day_pct": round(float(same * 100), 2), "both_away_days_apart_pct": round(float(other * 100), 2),
                     "away_pct": round(float(post[i][:, mid].mean() * 100), 1)})
    # How often is at least one of the 3 nearest away while this station is away?
    near_by = {}
    for a, b in near:
        if a in grid.index and b in grid.index:
            near_by.setdefault(grid.index[a], []).append(grid.index[b])
    p_some, p_all, w = 0.0, 0.0, 0.0
    for i, js in near_by.items():
        none = np.prod(1 - post[js][:, mid], axis=0)
        every = np.prod(post[js][:, mid], axis=0)
        pi = post[i, mid]
        p_some += ((1 - none) * pi).sum()
        p_all += (every * pi).sum()
        w += pi.sum()
    rows[0]["given_away_some_near_also_pct"] = round(float(p_some / w * 100), 1)
    rows[0]["given_away_all_near_also_pct"] = round(float(p_all / w * 100), 1)
    return rows


def next_station(c: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """For gap calls: the nearest station by road, leaving out the local one."""
    from pyproj import Transformer

    from firestation.ml_scorer import _router

    lon, lat = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True).transform(c["east"].to_numpy(), c["north"].to_numpy())
    xy = t[["x", "y"]].to_numpy()
    d = np.hypot(c["east"].to_numpy()[:, None] - xy[:, 0], c["north"].to_numpy()[:, None] - xy[:, 1])
    pos = {code: i for i, code in enumerate(t["code"])}
    own = c["ground"].map(pos).to_numpy()
    d[np.arange(len(c)), own] = np.inf
    nearest = np.argsort(d, axis=1)[:, :CANDIDATES]
    router = _router()
    rows = np.repeat(np.arange(len(c)), CANDIDATES)
    st = nearest.ravel()
    km = np.asarray(router.road_km(t["lat"].to_numpy()[st], t["lon"].to_numpy()[st], lat[rows], lon[rows]), dtype=float).reshape(len(c), CANDIDATES)
    km = np.where(np.isfinite(km), km, np.inf)
    best = km.argmin(1)
    out = c.copy()
    idx = nearest[np.arange(len(c)), best]
    out["next_code"] = t["code"].to_numpy()[idx]
    out["next_name"] = t["name"].to_numpy()[idx]
    out["next_km"] = km[np.arange(len(c)), best]
    return out


def all_out(calls: pd.DataFrame, codes: np.ndarray, mob: pd.DataFrame, pumps: pd.Series) -> np.ndarray:
    """Was every engine of that station on a recorded incident (or on its way back) when the call came?"""
    t = sec(calls["t0"])
    out = np.zeros(len(calls), dtype=bool)
    b = mob.dropna(subset=["t_left"])
    dur = (b["t_left"] - b["t0"]).dt.total_seconds()
    b = b[(dur > 0) & (dur < 6 * 3600)]
    groups = dict(tuple(b.groupby("home")))
    for s in np.unique(codes):
        if s not in groups:
            continue
        n_p = int(pumps.get(s, 1))
        ev, n, _ = station_sweep(sec(groups[s]["t0"]), sec(groups[s]["t_left"]) + BUFFER_S, n_p)
        i = np.flatnonzero(codes == s)
        j = np.searchsorted(ev, t[i], side="left") - 1
        ok = j >= 0
        out[i[ok]] = n[j[ok]] >= n_p
    return out


def passed_over(year: int, d: dict, grid: Grid, post: np.ndarray, pars: dict, e: pd.DataFrame, t: pd.DataFrame) -> tuple[list, list]:
    calls = d["calls"]
    period = {h: p for p, r in PERIODS.items() for h in r}
    g = e[e["gap"] == 1].reset_index(drop=True)
    g = next_station(g, t)
    g = g[np.isfinite(g["next_km"]) & g["first_km"].notna()].reset_index(drop=True)
    g["over6"] = (g["att"] > 360).astype(float)
    g["period"] = g["HourOfCall"].map(period)
    g["next_busy"] = all_out(g, g["next_code"].to_numpy(), d["mob"], d["pumps"])
    rows, checks = [], []
    for margin in MARGINS:
        reached = (g["first_name"] == g["next_name"]) | (g["first_km"] <= g["next_km"] + margin / 1000)
        lab = np.select([reached, g["next_busy"]], ["reached_first", "next_all_out"], "passed_over")
        fe = fixed_effects(g, lab, groups=("next_all_out", "passed_over"), n_boot=300, seed=year).set_index("group")
        for name in ("reached_first", "next_all_out", "passed_over"):
            m = lab == name
            row = {"year": year, "tie_margin_m": margin, "group": name, "gap_calls": int(m.sum()), "share_of_gap_calls_pct": round(float(m.mean() * 100), 1),
                   "share_of_all_calls_pct": round(float(m.sum() / len(calls) * 100), 2), "mean_att_s": round(float(g.loc[m, "att"].mean()), 1),
                   "over6_pct": round(float(g.loc[m, "over6"].mean() * 100), 1), "first_km_mean": round(float(g.loc[m, "first_km"].mean()), 2)}
            if name != "reached_first":
                x = fe.loc[name]
                share = m.sum() / len(calls)
                row.update(extra_s=x["extra_s"], extra_lo=x["extra_lo"], extra_hi=x["extra_hi"], extra_over6_points=x["extra_over6_points"],
                           limit_s=round(float(share * x["extra_s"]), 2), limit_lo=round(float(share * x["extra_lo"]), 2), limit_hi=round(float(share * x["extra_hi"]), 2),
                           limit_over6_points=round(float(share * x["extra_over6_points"]), 2))
            rows.append(row)
        if margin == MARGINS[0]:
            known = g["next_code"].isin(grid.stations).to_numpy()
            s_idx = g["next_code"].map(grid.index)
            step = grid.step(g["t0"])
            wk = (g["t0"].dt.dayofweek >= 5).astype(int)
            key = g["next_code"].astype(str) + "|" + g["HourOfCall"].astype(str) + "|" + wk.astype(str)

            def compare(p: np.ndarray) -> tuple[float, float]:
                v = pd.Series(np.nan, index=g.index, dtype=float)
                v[known] = p[s_idx[known].astype(int).to_numpy(), step[known]]
                use = (lab != "next_all_out") & known
                dfr = pd.DataFrame({"v": v[use], "k": key[use], "passed": lab[use] == "passed_over"})
                both = dfr.groupby("k")["passed"].transform("nunique") == 2
                dfr = dfr[both]
                w = dfr[dfr["passed"]].groupby("k").size()
                a, b = dfr[dfr["passed"]].groupby("k")["v"].mean(), dfr[~dfr["passed"]].groupby("k")["v"].mean()
                return float(np.average(a, weights=w) * 100), float(np.average(b.reindex(a.index), weights=w) * 100)

            obs = compare(post)
            sims = []
            for n in range(10):
                sim_e = e.assign(gap=simulate_labels(grid, e, pars, seed=year * 100 + n))
                sims.append(compare(apply_pars(grid, sim_e, pars)))
            sims = np.array(sims)
            checks.append({"year": year, "away_when_passed_over_pct": round(obs[0], 1), "away_when_reached_first_pct": round(obs[1], 1), "difference": round(obs[0] - obs[1], 1),
                           "simulated_difference_mean": round(float((sims[:, 0] - sims[:, 1]).mean()), 1),
                           "simulated_difference_lo": round(float((sims[:, 0] - sims[:, 1]).min()), 1), "simulated_difference_hi": round(float((sims[:, 0] - sims[:, 1]).max()), 1)})
    return rows, checks


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    cache: dict = {}
    data = {y: prepare(y, cache) for y in (FIT_YEAR, *CHECK_YEARS)}
    d24 = data[FIT_YEAR]
    e24 = d24["calls"][d24["calls"]["cat"].isin(["own_first", "real_gap"])].copy()
    e24["gap"] = (e24["cat"] == "real_gap").astype(int)
    stations = sorted(s for s in e24["ground"].unique() if s in d24["pumps"].index)
    e24 = e24[e24["ground"].isin(stations)]
    pars = split_fit(Grid(FIT_YEAR, stations), e24, d24["pumps"])
    known = [s for p in pars.values() for s in p["stations"]]

    pair_rows, overlap_rows, passed_rows, check_rows = [], [], [], []
    for year in (FIT_YEAR, *CHECK_YEARS):
        d = data[year]
        e = d["calls"][d["calls"]["cat"].isin(["own_first", "real_gap"]) & d["calls"]["ground"].isin(known)].copy()
        e["gap"] = (e["cat"] == "real_gap").astype(int)
        e = e.reset_index(drop=True)
        names = e.drop_duplicates("ground").set_index("ground")["ground_name"].to_dict()
        t = station_table(names)
        near, far = neighbour_sets(t)
        sts = sorted(e["ground"].unique())
        grid = Grid(year, sts)
        post = apply_pars(grid, e, pars)
        pair_rows += pair_test(year, e, near, far, names)
        overlap_rows += model_overlap(year, grid, post, near, far)
        print(pd.DataFrame(pair_rows)[lambda x: x["year"] == year].to_string(index=False), flush=True)
        print(pd.DataFrame(overlap_rows)[lambda x: x["year"] == year].to_string(index=False), flush=True)
        r, chk = passed_over(year, d, grid, post, pars, e, t)
        passed_rows += r
        check_rows += chk
        print(pd.DataFrame(r).to_string(index=False), flush=True)
        print(pd.DataFrame(chk).to_string(index=False), flush=True)

    for name, rows in (("pairs", pair_rows), ("overlap_model", overlap_rows), ("passed_over", passed_rows), ("passed_over_model_check", check_rows)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"doc": __doc__, "pairs": pair_rows, "overlap_model": overlap_rows, "passed_over": passed_rows, "model_check": check_rows}, indent=2, default=str) + "\n")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
