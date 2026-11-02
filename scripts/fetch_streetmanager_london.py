#!/usr/bin/env python3
"""Stream DfT Street Manager monthly permit archives and keep one row per London work.

Each month's zip is about 1 GB, so it is read as a stream (curl | tar) and never stored; only
works inside a box around Greater London are kept, with their location, type, traffic
management and start/end dates. Open Government Licence v3.0, no sign-up.

Writes data/raw/roadworks/<year>-<month>.csv.gz. Months already written are skipped.

Run:  .venv/bin/python scripts/fetch_streetmanager_london.py 2021-01 2025-12
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "roadworks"
URL = "https://opendata.manage-roadworks.service.gov.uk/permit/{y}/{m:02d}.zip"
BOX = (500000, 565000, 150000, 205000)  # British National Grid: easting, northing around Greater London
COORD = re.compile(rb'"works_location_coordinates":"[A-Z]+\s*\(+\s*(\d+)(?:\.\d+)?\s+(\d+)')
SPLIT = b'{"event_reference":'
FIELDS = ["highway_authority", "street_name", "work_category", "traffic_management_type", "works_location_type",
          "road_category", "is_traffic_sensitive", "proposed_start_date", "proposed_end_date"]


def fetch_month(y: int, m: int) -> tuple[pd.DataFrame, int]:
    cmd = f"curl -sSL --retry 3 {URL.format(y=y, m=m)} | tar -xOf -"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=1 << 22)
    works: dict[str, dict] = {}
    buf, seen = b"", 0
    while chunk := proc.stdout.read(1 << 22):
        parts = (buf + chunk).split(SPLIT)
        buf = parts.pop()
        for part in parts:
            if not part:
                continue
            seen += 1
            c = COORD.search(part)
            if not c or not (BOX[0] <= int(c[1]) <= BOX[1] and BOX[2] <= int(c[2]) <= BOX[3]):
                continue
            try:
                d = json.loads(SPLIT + part)["object_data"]
            except ValueError:
                continue
            w = works.setdefault(d["work_reference_number"], {"easting": int(c[1]), "northing": int(c[2])})
            w.update({k: d[k] for k in FIELDS if d.get(k) is not None})
            if d.get("actual_start_date_time"):
                w["actual_start"] = d["actual_start_date_time"]
            if d.get("actual_end_date_time"):
                w["actual_end"] = d["actual_end_date_time"]
    proc.wait()
    return pd.DataFrame.from_dict(works, orient="index").rename_axis("work_reference_number").reset_index(), seen


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    first, last = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
    for period in pd.period_range(first, last, freq="M"):
        fp = OUT / f"{period}.csv.gz"
        if fp.exists():
            continue
        t0 = time.time()
        df, seen = fetch_month(period.year, period.month)
        if df.empty:
            print(f"{period}: no London works from {seen:,} events; nothing written", flush=True)
            continue
        df.to_csv(fp, index=False)
        print(f"{period}: {len(df):,} London works from {seen:,} events, {fp.stat().st_size / 1e6:.1f} MB, "
              f"{time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
