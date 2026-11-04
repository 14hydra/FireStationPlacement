#!/usr/bin/env python3
"""Is the "local engine looks free, another station arrives first" gap real?

Every call is put in one group from LFB's records (no travel-time or dispatch model):

  own_first        the first engine came from the call's own station area (the baseline)
  empty            every regular engine of the local station was out when the call came
  other_own_same   another station arrived first; a local engine was sent at the same time (within 60 s)
  other_own_later  another station arrived first; a local engine was sent later
  other_closer     another station arrived first, no local engine sent, and it is closer by road
  real_gap         another station arrived first, no local engine sent, local station closer by road
  other_unknown    as above, but a road distance could not be found

Only real_gap can be time lost because a nearer engine that looked free did not respond. Even
then the local engine may have been away on duties the records do not show.

Extra time per group: regression with fixed effects for each square and period of the day, so
each group is compared with the baseline in the same place at the same time of day. 95 % ranges
come from resampling whole squares. Most it could save = share of calls x extra time. This is an
upper limit, not what a real change would achieve.

Checks: every year 2021-26 (2025-26 records have times only to the minute; 2026 is part of a year;
2021 was affected by Covid), and on 2024 and 2025 the return-trip allowance, what counts as a
regular engine, the square size and hour-by-hour comparison.

Writes data/figures/validation/first_engine_gap/: by_year.csv, sensitivity.csv, by_period.csv, summary.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/first_engine_gap.py
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

from busy_engine_check import MOB_FILES, PERIODS, RAW, station_sweep, when  # noqa: E402

OUT = ROOT / "data" / "figures" / "validation" / "first_engine_gap"
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
GROUPS = ("empty", "other_own_same", "other_own_later", "other_closer", "real_gap", "other_unknown")
BASE = "own_first"
SAME_S = 60
sec = lambda s: (s - pd.Timestamp("2000-01-01")).dt.total_seconds().to_numpy()  # noqa: E731


def load_year(year: int, mob_cache: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = RAW / MOB_FILES[year]
    if path not in mob_cache:
        mob_cache[path] = pd.read_csv(path, usecols=["IncidentNumber", "CalYear", "HourOfCall", "Resource_Code",
                                                     "DateAndTimeMobilised", "DateAndTimeLeft", "DeployedFromStation_Code",
                                                     "DeployedFromStation_Name", "DeployedFromLocation"],
                                      dtype={"IncidentNumber": str}, low_memory=False, encoding="utf-8-sig")
    m = mob_cache[path]
    m = m[m["CalYear"] == year].copy()
    m["t0"], m["t_left"] = when(m["DateAndTimeMobilised"]), when(m["DateAndTimeLeft"])
    m = m[m["Resource_Code"].astype(str).str.match(r"^[A-Z]\d{3}$")].dropna(subset=["t0"])
    m["home"] = m["Resource_Code"].str[:3]
    inc = pd.read_csv(RAW / ("lfb_incidents_2024_onwards.csv" if year >= 2024 else "lfb_incidents_2018_2023.csv"),
                      usecols=["IncidentNumber", "CalYear", "Easting_rounded", "Northing_rounded", "IncidentGroup",
                               "IncidentStationGround", "FirstPumpArriving_AttendanceTime",
                               "FirstPumpArriving_DeployedFromStation"],
                      dtype={"IncidentNumber": str}, low_memory=False, encoding="utf-8-sig")
    inc = inc[inc["CalYear"] == year].dropna(subset=["FirstPumpArriving_AttendanceTime", "IncidentStationGround",
                                                     "Easting_rounded", "Northing_rounded"])
    first = m.sort_values("t0").drop_duplicates("IncidentNumber")[["IncidentNumber", "t0", "HourOfCall"]]
    c = inc.merge(first, on="IncidentNumber").reset_index(drop=True)
    names = m[m["DeployedFromLocation"] == "Home Station"].groupby("DeployedFromStation_Code")["DeployedFromStation_Name"].agg(
        lambda s: s.value_counts().index[0])
    n2c = {v.upper(): k for k, v in names.items()}
    c["ground_name"] = c["IncidentStationGround"].astype(str).str.upper()
    c["first_name"] = c["FirstPumpArriving_DeployedFromStation"].astype(str).str.upper()
    c["ground"] = c["ground_name"].map(n2c)
    c = c.dropna(subset=["ground"]).reset_index(drop=True)
    c["att"] = c["FirstPumpArriving_AttendanceTime"].astype(float)
    c["over6"] = (c["att"] > 360).astype(float)
    c["period"] = c["HourOfCall"].map({h: p for p, r in PERIODS.items() for h in r})
    c["east"], c["north"] = c["Easting_rounded"].astype(float), c["Northing_rounded"].astype(float)
    # When each call's local engine was first sent to it, if ever.
    own = m.groupby(["IncidentNumber", "home"])["t0"].min().rename("own_t0").reset_index()
    c = c.merge(own, how="left", left_on=["IncidentNumber", "ground"], right_on=["IncidentNumber", "home"]).drop(columns="home")
    return c, m


def attach_road(c: pd.DataFrame) -> pd.DataFrame:
    """Road km from the local station and from the station that arrived first, where they differ."""
    from pyproj import Transformer

    from firestation.ml_scorer import _router

    st = pd.read_csv(RAW / "london_fire_stations.csv")
    st["key"] = st["name"].str.upper()
    st = st.set_index("key")
    lon, lat = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True).transform(c["east"].to_numpy(), c["north"].to_numpy())
    c["own_km"], c["first_km"] = np.nan, np.nan
    todo = np.flatnonzero((c["first_name"] != c["ground_name"]).to_numpy() & c["own_t0"].isna().to_numpy())
    router = _router()
    for col, names in (("own_km", c["ground_name"]), ("first_km", c["first_name"])):
        n = names.to_numpy()[todo]
        known = np.isin(n, st.index.to_numpy())
        i = todo[known]
        km = router.road_km(st["lat"].reindex(n[known]).to_numpy(float), st["lon"].reindex(n[known]).to_numpy(float), lat[i], lon[i])
        c.loc[i, col] = km
    return c


def group_calls(c: pd.DataFrame, m: pd.DataFrame, *, min_use: int = 100, buffer_s: float = 0.0) -> tuple[np.ndarray, dict]:
    use = m.groupby("Resource_Code").size()
    regular = use[use >= min_use].index
    pumps = pd.Series(regular.str[:3]).value_counts()
    b = m[m["Resource_Code"].isin(regular)].dropna(subset=["t_left"])
    dur = (b["t_left"] - b["t0"]).dt.total_seconds()
    b = b[(dur > 0) & (dur < 6 * 3600)]
    t = sec(c["t0"])
    out = np.zeros(len(c))
    g = c["ground"].to_numpy()
    for s, gb in b.groupby("home"):
        if s not in pumps.index:
            continue
        ev, n, _ = station_sweep(sec(gb["t0"]), sec(gb["t_left"]) + buffer_s, int(pumps[s]))
        i = np.flatnonzero(g == s)
        j = np.searchsorted(ev, t[i], side="left") - 1
        ok = j >= 0
        out[i[ok]] = n[j[ok]]
    empty = out >= c["ground"].map(pumps).fillna(1).to_numpy()
    own_first = (c["first_name"] == c["ground_name"]).to_numpy()
    lag = (c["own_t0"] - c["t0"]).dt.total_seconds().to_numpy()
    own_sent = np.isfinite(lag)
    known = c["own_km"].notna().to_numpy() & c["first_km"].notna().to_numpy() & np.isfinite(c["own_km"].to_numpy()) & np.isfinite(c["first_km"].to_numpy())
    closer = c["first_km"].to_numpy() <= c["own_km"].to_numpy()
    cat = np.select(
        [empty, own_first, own_sent & (lag <= SAME_S), own_sent, ~known, closer],
        ["empty", "own_first", "other_own_same", "other_own_later", "other_unknown", "other_closer"], "real_gap")
    return cat, {"regular_engines": int(len(regular))}


def fixed_effects(c: pd.DataFrame, cat: np.ndarray, *, square_m: int = 500, by_hour: bool = False, n_boot: int = 500, seed: int = 0,
                  groups: tuple = GROUPS) -> pd.DataFrame:
    """Extra seconds and extra over-6-minute share per group against the baseline (anything not in groups)."""
    GROUPS = groups  # noqa: N806
    sq = (c["east"] // square_m).astype(int).astype(str) + "_" + (c["north"] // square_m).astype(int).astype(str)
    cells = sq + "|" + (c["HourOfCall"].astype(str) if by_hour else c["period"].astype(str))
    fe = pd.factorize(cells)[0]
    squares = pd.factorize(sq)[0]
    X = np.column_stack([(cat == g).astype(float) for g in GROUPS])

    def demean(a):
        a = a.astype(float)
        return a - (np.bincount(fe, weights=a) / np.bincount(fe))[fe]

    Xd = np.column_stack([demean(X[:, k]) for k in range(X.shape[1])])
    n_sq = squares.max() + 1
    k = len(GROUPS)
    # Fixed effects sit inside squares, so per-square sums let the bootstrap avoid refitting from scratch.
    xx = np.zeros((n_sq, k, k))
    np.add.at(xx, squares, Xd[:, :, None] * Xd[:, None, :])
    rows = []
    rng = np.random.default_rng(seed)
    draws = rng.poisson(1.0, (n_boot, n_sq)).astype(float)
    res = {}
    for col in ("att", "over6"):
        yd = demean(c[col].to_numpy())
        xy = np.zeros((n_sq, k))
        np.add.at(xy, squares, Xd * yd[:, None])
        est = np.linalg.lstsq(xx.sum(0), xy.sum(0), rcond=None)[0]
        boots = np.array([np.linalg.lstsq(np.tensordot(w, xx, 1), w @ xy, rcond=None)[0] for w in draws])
        res[col] = (est, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0), boots)
    share = np.array([(cat == g).mean() for g in GROUPS])
    for i, g in enumerate(GROUPS):
        rows.append({"group": g, "calls": int((cat == g).sum()), "share_pct": round(share[i] * 100, 2),
                     "mean_att_s": round(float(c.loc[cat == g, "att"].mean()), 1),
                     "over6_pct": round(float(c.loc[cat == g, "over6"].mean() * 100), 1),
                     "extra_s": round(float(res["att"][0][i]), 1), "extra_lo": round(float(res["att"][1][i]), 1),
                     "extra_hi": round(float(res["att"][2][i]), 1),
                     "extra_over6_points": round(float(res["over6"][0][i] * 100), 1),
                     "limit_s": round(float(share[i] * res["att"][0][i]), 2),
                     "limit_lo": round(float(share[i] * res["att"][1][i]), 2),
                     "limit_hi": round(float(share[i] * res["att"][2][i]), 2),
                     "limit_over6_points": round(float(share[i] * res["over6"][0][i] * 100), 2)})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    mob_cache: dict = {}
    by_year, sens, period_rows = [], [], []
    for year in YEARS:
        c, m = load_year(year, mob_cache)
        c = attach_road(c)
        cat, info = group_calls(c, m)
        t = fixed_effects(c, cat).assign(year=year, calls_total=len(c), mean_att_all=round(float(c["att"].mean()), 1),
                                         own_first_mean_s=round(float(c.loc[cat == BASE, "att"].mean()), 1),
                                         own_first_share_pct=round(float((cat == BASE).mean() * 100), 2),
                                         over6_all_pct=round(float(c["over6"].mean() * 100), 1), **info)
        by_year.append(t)
        print(f"\n{year}: {len(c):,} calls, average {c['att'].mean():.1f} s, over 6 min {c['over6'].mean():.1%}, "
              f"local engine first {(cat == BASE).mean():.1%}", flush=True)
        print(t.drop(columns=["year", "calls_total", "mean_att_all", "own_first_mean_s", "own_first_share_pct", "over6_all_pct", "regular_engines"]).to_string(index=False), flush=True)
        for p in PERIODS:
            mask = (c["period"] == p).to_numpy()
            period_rows.append({"year": year, "period": p, "calls": int(mask.sum()),
                                **{f"{g}_pct": round(float((cat[mask] == g).mean() * 100), 2) for g in GROUPS}})
        if year in (2024, 2025):
            variants = {"as run": {}, "return trip +10 min": {"buffer_s": 600}, "return trip +20 min": {"buffer_s": 1200},
                        "regular engine: 50+ trips": {"min_use": 50}, "regular engine: 300+ trips": {"min_use": 300},
                        "squares 250 m": {"fe": {"square_m": 250}}, "squares 1 km": {"fe": {"square_m": 1000}},
                        "by hour, not period": {"fe": {"by_hour": True}}}
            for name, kw in variants.items():
                fe_kw = kw.pop("fe", {})
                cat2, _ = group_calls(c, m, **kw)
                r = fixed_effects(c, cat2, n_boot=100, **fe_kw).set_index("group")
                sens.append({"year": year, "variant": name, **{f"{g}_share": r.at[g, "share_pct"] for g in ("empty", "real_gap")},
                             "empty_limit_s": r.at["empty", "limit_s"], "real_gap_extra_s": r.at["real_gap", "extra_s"],
                             "real_gap_limit_s": r.at["real_gap", "limit_s"],
                             "real_gap_limit_over6": r.at["real_gap", "limit_over6_points"]})
            print(pd.DataFrame(sens)[lambda d: d["year"] == year].to_string(index=False), flush=True)
    by = pd.concat(by_year, ignore_index=True)
    by.to_csv(OUT / "by_year.csv", index=False)
    pd.DataFrame(sens).to_csv(OUT / "sensitivity.csv", index=False)
    pd.DataFrame(period_rows).to_csv(OUT / "by_period.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"doc": __doc__, "by_year": by.to_dict(orient="records"), "sensitivity": sens}, indent=2, default=str) + "\n")
    print(pd.DataFrame(period_rows).pivot(index="period", columns="year", values="real_gap_pct").to_string())
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
