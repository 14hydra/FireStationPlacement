#!/usr/bin/env python3
"""What causes the "closer local engine looks free, another station arrives first" gap?

Uses the groups from first_engine_gap.py ("gap" = real_gap; "local" = own_first; other calls left out).

Delay codes: "On outside duty when mobilised" shows an engine was away from its station when sent.
Does it rise and fall with the gap across hours of the week and across stations?

Clustering: if a local engine is really unavailable for a while, its area's calls should go to
other stations in runs. For calls in one station area, in order of time, compare
P(gap | previous call was a gap) with P(gap | previous call was local), by time since the
previous call. Null: shuffle the labels within station x hour of day x weekday/weekend, which keeps each
station's rate at each time of day but removes runs. Pairs must be in different 500 m squares,
so the same awkward street cannot create a run.

A run shows the engine (or its dispatch) changed for a while. It does not say why: training, visits,
repairs, standing by at another station and temporary dispatch changes would all look alike.

Also: gap by weekday/weekend and hour, and by number of engines at the local station.

Writes data/figures/validation/gap_cause/: delay_codes.csv, outside_duty_by_hour.csv, clustering.csv,
clustering_by_period.csv, by_week_hour.csv, by_pumps.csv, summary.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/gap_cause_checks.py
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

from busy_engine_check import MOB_FILES, PERIODS, RAW  # noqa: E402
from first_engine_gap import YEARS, attach_road, group_calls, load_year  # noqa: E402

OUT = ROOT / "data" / "figures" / "validation" / "gap_cause"
OUTSIDE = "On outside duty when mobilised"
BINS = [(0, 15), (15, 30), (30, 60), (60, 120), (120, 240), (240, 480), (480, 1440)]
N_PERM = 500
SQUARE_M = 500


def delay_table(year: int, mob_cache: dict) -> pd.DataFrame:
    path = RAW / MOB_FILES[year]
    if path not in mob_cache:
        mob_cache[path] = pd.read_csv(path, usecols=["CalYear", "HourOfCall", "Resource_Code", "DateAndTimeMobilised",
                                                     "DelayCode_Description", "DeployedFromLocation", "AttendanceTimeSeconds", "PumpOrder"],
                                      low_memory=False, encoding="utf-8-sig")
    m = mob_cache[path]
    m = m[(m["CalYear"] == year) & m["Resource_Code"].astype(str).str.match(r"^[A-Z]\d{3}$")].copy()
    m["home"] = m["Resource_Code"].str[:3]
    t = pd.to_datetime(m["DateAndTimeMobilised"], dayfirst=True, errors="coerce", format="mixed")
    m["weekend"] = t.dt.dayofweek >= 5
    m["delay"] = m["DelayCode_Description"].replace("NULL", np.nan).fillna("(none recorded)")
    m["outside"] = m["delay"] == OUTSIDE
    return m


def pair_stats(y: np.ndarray, prev: np.ndarray, valid: np.ndarray) -> float:
    a, p = y[valid], y[prev[valid]]
    if (p == 1).sum() == 0 or (p == 0).sum() == 0:
        return np.nan
    return float(a[p == 1].mean() - a[p == 0].mean())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    delay_cache: dict = {}
    mob_cache: dict = {}
    clus, clus_p, weekly, by_pumps, delay_rows, outside_rows, corr_rows = [], [], [], [], [], [], []
    for year in YEARS:
        c, m = load_year(year, mob_cache)
        c = attach_road(c)
        cat, _ = group_calls(c, m)
        c["cat"] = cat
        c["weekend"] = c["t0"].dt.dayofweek >= 5
        use = m.groupby("Resource_Code").size()
        pumps = pd.Series(use[use >= 100].index.str[:3]).value_counts()
        e = c[c["cat"].isin(["own_first", "real_gap"])].sort_values(["ground", "t0"]).reset_index(drop=True)
        e["gap"] = (e["cat"] == "real_gap").astype(int)
        e["pumps"] = e["ground"].map(pumps).fillna(1).astype(int)

        # ---- when
        w = e.groupby(["weekend", "HourOfCall"])["gap"].agg(["size", "mean"]).reset_index().assign(year=year)
        weekly.append(w)
        by_pumps.append(e.assign(p=e["pumps"].clip(upper=2)).groupby(["period", "p"])["gap"].agg(["size", "mean"]).reset_index().assign(year=year))

        # ---- delay codes
        d = delay_table(year, delay_cache)
        tab = d["delay"].value_counts(normalize=True).head(10).mul(100).round(2)
        delay_rows += [{"year": year, "delay": k, "pct_of_mobilisations": v} for k, v in tab.items()]
        out = d.groupby(["weekend", "HourOfCall"])["outside"].mean().rename("outside").reset_index()
        hw = e.groupby(["weekend", "HourOfCall"])["gap"].mean().rename("gap").reset_index().merge(out, on=["weekend", "HourOfCall"])
        outside_rows.append(hw.assign(year=year))
        sd = d.groupby("home")["outside"].mean().rename("outside").to_frame().join(e.groupby("ground")["gap"].agg(["mean", "size"]))
        sd = sd[sd["size"] >= 100]
        d["period"] = d["HourOfCall"].map({h: p for p, r in PERIODS.items() for h in r})
        sp = d.groupby(["home", "period"])["outside"].mean().rename("outside").reset_index().merge(
            e.groupby(["ground", "period"])["gap"].agg(["mean", "size"]).reset_index(), left_on=["home", "period"], right_on=["ground", "period"])
        sp = sp[sp["size"] >= 30]
        # Station x period, removing what is common to the period and to the station.
        res = sp.copy()
        for col in ("outside", "mean"):
            res[col] = res[col] - res.groupby("period")[col].transform("mean") - res.groupby("home")[col].transform("mean") + res[col].mean()
        fo = d[d["PumpOrder"].astype(str) == "1"] if "PumpOrder" in d else d
        att = pd.to_numeric(fo["AttendanceTimeSeconds"], errors="coerce")
        corr_rows.append({"year": year, "outside_share_pct": round(float(d["outside"].mean() * 100), 2),
                          "corr_week_hours": round(float(hw["gap"].corr(hw["outside"])), 2),
                          "corr_week_hours_rank": round(float(hw["gap"].corr(hw["outside"], method="spearman")), 2),
                          "corr_stations": round(float(sd["mean"].corr(sd["outside"])), 2),
                          "corr_station_x_period_within": round(float(res["mean"].corr(res["outside"])), 2),
                          "attendance_first_engine_outside_s": round(float(att[fo["outside"]].mean()), 1),
                          "attendance_first_engine_not_outside_s": round(float(att[~fo["outside"]].mean()), 1)})

        # ---- clustering
        t = e["t0"].astype("int64").to_numpy() / 6e10 if str(e["t0"].dtype).endswith("[ns]") else (e["t0"] - pd.Timestamp("2000-01-01")).dt.total_seconds().to_numpy() / 60
        y = e["gap"].to_numpy()
        same = (e["ground"].to_numpy()[1:] == e["ground"].to_numpy()[:-1])
        prev = np.r_[0, np.arange(len(e) - 1)]
        dt = np.r_[np.inf, np.where(same, t[1:] - t[:-1], np.inf)]
        sqx = (e["east"] // SQUARE_M).astype(int).to_numpy()
        sqy = (e["north"] // SQUARE_M).astype(int).to_numpy()
        other_sq = np.r_[False, (sqx[1:] != sqx[:-1]) | (sqy[1:] != sqy[:-1])]
        stratum = pd.factorize(e["ground"].astype(str) + "|" + e["HourOfCall"].astype(str) + "|" + e["weekend"].astype(str))[0]
        rng = np.random.default_rng(year)
        by_stratum = np.argsort(stratum, kind="stable")
        perms = []
        for _ in range(N_PERM):
            yp = np.empty_like(y)
            yp[by_stratum] = y[np.lexsort((rng.random(len(y)), stratum))]
            perms.append(yp)
        masks = {f"{a}-{b} min": other_sq & (dt >= a) & (dt < b) for a, b in BINS}
        for name, mask in masks.items():
            obs = pair_stats(y, prev, mask)
            null = np.array([pair_stats(p, prev, mask) for p in perms])
            a, pv = y[mask], y[prev[mask]]
            clus.append({"year": year, "window": name, "pairs": int(mask.sum()),
                         "p_gap_after_gap": round(float(a[pv == 1].mean() * 100), 1), "p_gap_after_local": round(float(a[pv == 0].mean() * 100), 1),
                         "diff_points": round(obs * 100, 1), "null_mean": round(float(np.nanmean(null) * 100), 1),
                         "null_lo": round(float(np.nanpercentile(null, 2.5) * 100), 1), "null_hi": round(float(np.nanpercentile(null, 97.5) * 100), 1),
                         "p_value": round(float(((np.abs(null - np.nanmean(null)) >= abs(obs - np.nanmean(null))).sum() + 1) / (N_PERM + 1)), 4)})
        short = other_sq & (dt < 60)
        per = e["period"].to_numpy()
        for p_name in PERIODS:
            mask = short & (per == p_name)
            obs = pair_stats(y, prev, mask)
            null = np.array([pair_stats(p, prev, mask) for p in perms])
            a, pv = y[mask], y[prev[mask]]
            clus_p.append({"year": year, "period": p_name, "pairs": int(mask.sum()), "p_gap_after_gap": round(float(a[pv == 1].mean() * 100), 1),
                           "p_gap_after_local": round(float(a[pv == 0].mean() * 100), 1), "diff_points": round(obs * 100, 1),
                           "null_mean": round(float(np.nanmean(null) * 100), 1), "null_hi": round(float(np.nanpercentile(null, 97.5) * 100), 1)})
        print(f"\n{year}", pd.DataFrame(clus)[lambda x: x["year"] == year].drop(columns="year").to_string(index=False), flush=True)
        print(pd.DataFrame(clus_p)[lambda x: x["year"] == year].drop(columns="year").to_string(index=False), flush=True)
        print(corr_rows[-1], flush=True)

    pd.DataFrame(delay_rows).to_csv(OUT / "delay_codes.csv", index=False)
    pd.concat(outside_rows).round(4).to_csv(OUT / "outside_duty_by_hour.csv", index=False)
    pd.DataFrame(clus).to_csv(OUT / "clustering.csv", index=False)
    pd.DataFrame(clus_p).to_csv(OUT / "clustering_by_period.csv", index=False)
    pd.concat(weekly).round(4).to_csv(OUT / "by_week_hour.csv", index=False)
    pd.concat(by_pumps).round(4).to_csv(OUT / "by_pumps.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"doc": __doc__, "delay_vs_gap": corr_rows, "clustering": clus, "clustering_by_period": clus_p}, indent=2, default=str) + "\n")
    print(pd.DataFrame(corr_rows).to_string(index=False))
    print(pd.DataFrame(delay_rows)[lambda x: x["year"] == 2024].to_string(index=False))
    h = pd.concat(outside_rows)
    h = h[h["year"] == 2024].assign(gap=lambda x: (x["gap"] * 100).round(1), outside=lambda x: (x["outside"] * 100).round(2))
    print(h.pivot(index="HourOfCall", columns="weekend", values=["gap", "outside"]).to_string())
    print(pd.concat(by_pumps)[lambda x: x["year"] == 2024].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
