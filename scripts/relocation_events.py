#!/usr/bin/env python3
"""Find London's temporary engine relocations (station rebuilds) from mobilisation records only.

Reads only which station each engine was deployed from and when. It never reads attendance, travel or turnout times, so
the event list is fixed without seeing any outcome. Used by docs/prediction_relocations_and_late_2026.md.

  1. Low periods: a station's share of London's mobilisations (14-day centred window) falls below a quarter of its
     whole-life median for 14+ days. Stations with fewer than 1,000 active days are skipped. Periods less than 30 days
     apart are merged into one episode.
  2. The station's own engines: resource codes with 30+ deployments in the 90 days before the episode, 80 % or more of
     them from that station.
  3. Where they went: the station those engines were deployed from during the episode, and its share.

Writes data/figures/validation/relocations/events.csv.

Run:  .venv/bin/python scripts/relocation_events.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "london"
OUT = ROOT / "data" / "figures" / "validation" / "relocations"


def load() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(RAW / "lfb_mobilisations_*.csv"))):
        d = pd.read_csv(f, usecols=["DateAndTimeMobilised", "Resource_Code", "DeployedFromStation_Name"], encoding="utf-8-sig",
                        low_memory=False)
        iso = ("2009" in f) or ("2015" in f)  # older files use ISO dates, newer ones day/month/year
        d["date"] = pd.to_datetime(d["DateAndTimeMobilised"].str[:10], format="%Y-%m-%d" if iso else "%d/%m/%Y", errors="coerce")
        frames.append(d[["date", "Resource_Code", "DeployedFromStation_Name"]])
    m = pd.concat(frames).dropna(subset=["date"])
    m["st"] = m["DeployedFromStation_Name"].astype(str).str.strip().str.upper()
    return m


def low_periods(m: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    days = pd.date_range(m["date"].min(), m["date"].max())
    total = m.groupby("date").size().reindex(days, fill_value=0).rolling(14, center=True).sum()
    periods = []
    for st, g in m.groupby("st"):
        s = g.groupby("date").size().reindex(days, fill_value=0)
        active = s[s > 0]
        if len(active) < 1000:
            continue
        share = (s.rolling(14, center=True).sum() / total)[active.index.min():active.index.max()]
        low = (share < 0.25 * share.median()).to_numpy()
        i = 0
        while i < len(low):
            if low[i]:
                j = i
                while j < len(low) and low[j]:
                    j += 1
                if j - i >= 14:
                    periods.append((st, share.index[i], share.index[j - 1]))
                i = j
            else:
                i += 1
    episodes = []
    for st in sorted({p[0] for p in periods}):
        cur = None
        for _, a, b in sorted(p for p in periods if p[0] == st):
            if cur and (a - cur[1]).days <= 30:
                cur[1] = max(cur[1], b)
            else:
                if cur:
                    episodes.append((st, cur[0], cur[1]))
                cur = [a, b]
        episodes.append((st, cur[0], cur[1]))
    return episodes


def main() -> int:
    m = load()
    rows = []
    for st, a, b in low_periods(m):
        pre = m[(m["date"] < a) & (m["date"] >= a - pd.Timedelta(days=90))]
        c = pre.groupby("Resource_Code").agg(n=("st", "size"), own=("st", lambda s: (s == st).mean()))
        own = c[(c["n"] >= 30) & (c["own"] >= 0.8)].index.tolist()
        during = m[m["Resource_Code"].isin(own) & (m["date"] >= a) & (m["date"] <= b)]
        top = during["st"].value_counts()
        rows.append({"station": st, "start": a.date(), "end": b.date(), "days": (b - a).days + 1, "own_engines": " ".join(own),
                     "moved_to": top.index[0] if len(top) else "", "share": round(top.iloc[0] / len(during), 2) if len(top) else 0.0,
                     "deployments": len(during)})
    e = pd.DataFrame(rows).sort_values("start")
    OUT.mkdir(parents=True, exist_ok=True)
    e.to_csv(OUT / "events.csv", index=False)
    print(e.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
