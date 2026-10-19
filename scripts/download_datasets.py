#!/usr/bin/env python3
"""Download the London data this project uses into data/raw/london/.

London Datastore (Open Government Licence):
  - LFB incident records 2009-2017, 2018-2023 (xlsx, converted to csv) and 2024 onwards
  - LFB mobilisation records 2009-2014 and 2015-2020 (xlsx, converted), 2021-2024, 2025 onwards
Geofabrik (OpenStreetMap, ODbL):
  - Greater London extract (.osm.pbf), used to build the drive graph

Existing files are skipped unless --force. Then build the derived tables:

  python scripts/build_lfb_planner_incidents.py
  python scripts/build_london_fire_stations.py
  python scripts/build_lfb_travel_training.py

Run:  .venv/bin/python scripts/download_datasets.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "london"

DATASTORE = "https://data.london.gov.uk/download"
FILES = {
    "lfb_incidents_2018_2023.xlsx": f"{DATASTORE}/em8xy/f5066d66-c7a3-415f-9629-026fbda61822/"
    "LFB%20Incident%20data%20from%202018%20-%202023.xlsx",
    "lfb_incidents_2024_onwards.xlsx": f"{DATASTORE}/em8xy/58m/LFB%20Incident%20data%20from%202024%20onwards.xlsx",
    "lfb_mobilisations_2021_2024.csv": f"{DATASTORE}/24r65/3ff29fb5-3935-41b2-89f1-38571059237e/"
    "LFB%20Mobilisation%20data%20from%202021%20-%202024.csv",
    "lfb_mobilisations_2025.csv": f"{DATASTORE}/24r65/7d5b4e2f-3ddb-48b9-8a4d-bf86bdf6a4ef/"
    "LFB%20Mobilisation%20data%20from%202025.csv",
    # Pre-2014 history for the 2014 closure test (scripts/build_lfb_2014_closure_data.py).
    "lfb_incidents_2009_2017.csv": f"{DATASTORE}/em8xy/73728cf4-b70e-48e2-9b97-4e4341a2110d/"
    "LFB%20Incident%20data%20from%202009%20-%202017.csv",
    "lfb_mobilisations_2009_2014.xlsx": f"{DATASTORE}/24r65/fcbd2e97-b5bf-4117-a50f-d596181bc8d3/"
    "LFB%20Mobilisation%20data%20from%20January%202009%20-%202014.xlsx",
    "lfb_mobilisations_2015_2020.xlsx": f"{DATASTORE}/24r65/329ef4d3-7ebb-4198-8bcf-92ad6711e50d/"
    "LFB%20Mobilisation%20data%20from%202015%20-%202020.xlsx",
    "greater-london-latest.osm.pbf": "https://download.geofabrik.de/europe/united-kingdom/england/"
    "greater-london-latest.osm.pbf",
}


def download(name: str, url: str, force: bool) -> Path:
    path = RAW / name
    if path.exists() and not force:
        print(f"skip  {name} (exists)")
        return path
    print(f"get   {name} …", flush=True)
    tmp = path.with_suffix(path.suffix + ".part")
    # The Datastore refuses Python's default user agent on some files (HTTP 403).
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (FireStationPlacement research)"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)
    print(f"      {path.stat().st_size / 1e6:,.0f} MB")
    return path


def xlsx_to_csv(xlsx: Path, force: bool) -> None:
    import pandas as pd

    csv = xlsx.with_suffix(".csv")
    if csv.exists() and not force:
        print(f"skip  {csv.name} (exists)")
        return
    print(f"conv  {xlsx.name} -> {csv.name} (a few minutes) …", flush=True)
    pd.read_excel(xlsx, engine="openpyxl").to_csv(csv, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="Re-download / re-convert existing files")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    for name, url in FILES.items():
        path = download(name, url, args.force)
        if path.suffix == ".xlsx":
            xlsx_to_csv(path, args.force)
    print("Done:", RAW)
    return 0


if __name__ == "__main__":
    sys.exit(main())
