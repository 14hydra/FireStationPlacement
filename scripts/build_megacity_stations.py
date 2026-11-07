#!/usr/bin/env python3
"""Fetch OSM amenity=fire_station points for one or all megacities.

Tries several Overpass mirrors. If all fail (or return too few sites), writes a
documented synthetic station lattice so placement can still run.

Writes ``data/raw/megacities/<id>/fire_stations.csv`` with columns
facilityname, lat, lon (planner-ready).

Run:
  PYTHONPATH=src .venv/bin/python scripts/build_megacity_stations.py
  PYTHONPATH=src .venv/bin/python scripts/build_megacity_stations.py --city tokyo
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.megacities import get_megacity, iter_megacities, list_megacities  # noqa: E402

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def _clean_name(raw, i: int) -> str:
    s = str(raw).strip() if raw is not None and str(raw).strip() and str(raw) != "nan" else ""
    if not s:
        return f"Fire station {i+1}"
    s = re.sub(r"\s+", " ", s)
    return s[:80]


def _overpass_ql(bbox: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    # Overpass bbox is south,west,north,east
    return f"""
    [out:json][timeout:180];
    (
      node["amenity"="fire_station"]({south},{west},{north},{east});
      way["amenity"="fire_station"]({south},{west},{north},{east});
      relation["amenity"="fire_station"]({south},{west},{north},{east});
    );
    out center tags;
    """


def fetch_stations_overpass(
    bbox: tuple[float, float, float, float], *, timeout_s: float = 12.0
) -> pd.DataFrame:
    """Query Overpass mirrors directly (avoids some osmnx SSL path issues)."""
    body = _overpass_ql(bbox).encode("utf-8")
    ctx = ssl.create_default_context()
    last_err: Exception | None = None
    data = None
    for url in OVERPASS_ENDPOINTS:
        try:
            req = Request(url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urlopen(req, timeout=timeout_s, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            print(f"  Overpass OK via {url}")
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  Overpass failed ({url}): {exc}")
    if data is None:
        raise RuntimeError(f"all Overpass endpoints failed: {last_err}")

    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags") or {}
        if el.get("type") == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            c = el.get("center") or {}
            lat, lon = c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            continue
        rows.append(
            {
                "facilityname": tags.get("name"),
                "lat": float(lat),
                "lon": float(lon),
                "osm_element": el.get("type"),
                "osm_id": el.get("id"),
                "source": "osm_overpass",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["facilityname", "lat", "lon", "osm_element", "osm_id", "source"])
    out = pd.DataFrame(rows)
    out["facilityname"] = [_clean_name(n, i) for i, n in enumerate(out["facilityname"])]
    out = out.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["lat", "lon"])
    out["_k"] = out["lat"].round(4).astype(str) + "|" + out["lon"].round(4).astype(str)
    out = out.drop_duplicates(subset=["_k"]).drop(columns=["_k"]).reset_index(drop=True)
    seen: dict[str, int] = {}
    uniq = []
    for n in out["facilityname"]:
        if n not in seen:
            seen[n] = 0
            uniq.append(n)
        else:
            seen[n] += 1
            uniq.append(f"{n} ({seen[n]+1})")
    out["facilityname"] = uniq
    return out


def fetch_stations_osmnx(bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.timeout = 180
    for ep in OVERPASS_ENDPOINTS:
        try:
            ox.settings.overpass_url = ep
            g = ox.features_from_bbox(bbox, tags={"amenity": "fire_station"}).reset_index()
            if g.empty:
                continue
            pt = g.geometry.representative_point()
            names = g["name"] if "name" in g.columns else pd.Series([None] * len(g))
            element = g["element"] if "element" in g.columns else "way"
            oid = g["id"] if "id" in g.columns else pd.Series(range(len(g)))
            out = pd.DataFrame(
                {
                    "facilityname": [_clean_name(n, i) for i, n in enumerate(names)],
                    "lat": pt.y.astype(float),
                    "lon": pt.x.astype(float),
                    "osm_element": element.astype(str),
                    "osm_id": pd.to_numeric(oid, errors="coerce"),
                    "source": "osm_osmnx",
                }
            )
            return out.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  osmnx failed ({ep}): {exc}")
    raise RuntimeError("osmnx could not fetch fire stations")


def synthetic_stations(city, n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Fallback lattice biased toward the urban core when OSM is unavailable."""
    rng = np.random.default_rng(seed + abs(hash(city.id)) % 10_000)
    west, south, east, north = city.bbox
    clon, clat = city.center
    # ~sqrt(n) grid with jitter, denser near center via rejection
    rows = []
    attempts = 0
    while len(rows) < n and attempts < n * 40:
        attempts += 1
        lon = rng.uniform(west, east)
        lat = rng.uniform(south, north)
        # prefer center
        d = ((lon - clon) / max(east - west, 1e-6)) ** 2 + ((lat - clat) / max(north - south, 1e-6)) ** 2
        if rng.random() > np.exp(-3.5 * d):
            continue
        rows.append(
            {
                "facilityname": f"{city.name} Station {len(rows)+1}",
                "lat": float(lat),
                "lon": float(lon),
                "osm_element": None,
                "osm_id": None,
                "source": "synthetic_fallback",
            }
        )
    return pd.DataFrame(rows)


def build_one(
    city_id: str,
    *,
    force: bool = False,
    allow_synthetic: bool = True,
    synthetic_only: bool = False,
    n_synthetic: int = 48,
) -> Path:
    city = get_megacity(city_id)
    out = city.fire_stations_path
    if out.exists() and not force:
        print(f"{city.id}: keep existing {out} ({sum(1 for _ in out.open())-1} rows)")
        return out

    df = pd.DataFrame()
    source = "none"
    if synthetic_only:
        print(f"{city.id}: synthetic stations (n={n_synthetic})", flush=True)
        df = synthetic_stations(city, n=n_synthetic)
        source = "synthetic_fallback"
    else:
        print(f"{city.id}: fetching OSM fire stations in {city.bbox} …", flush=True)
        try:
            df = fetch_stations_overpass(city.bbox, timeout_s=12.0)
            source = "osm_overpass"
        except Exception as exc:  # noqa: BLE001
            print(f"{city.id}: direct Overpass failed ({exc}); trying osmnx…")
            try:
                df = fetch_stations_osmnx(city.bbox)
                source = "osm_osmnx"
            except Exception as exc2:  # noqa: BLE001
                print(f"{city.id}: osmnx also failed ({exc2})")

        if len(df) < 8:
            if not allow_synthetic:
                raise RuntimeError(f"{city.id}: only {len(df)} OSM fire stations")
            print(f"{city.id}: OSM returned {len(df)} stations — using synthetic fallback (n={n_synthetic})")
            df = synthetic_stations(city, n=n_synthetic)
            source = "synthetic_fallback"

    city.raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    (city.raw_dir / "stations_meta.json").write_text(
        json.dumps({"source": source, "n_stations": int(len(df))}, indent=2) + "\n"
    )
    print(f"{city.id}: wrote {len(df)} stations ({source}) → {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", choices=list_megacities(), default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip Overpass and write a synthetic station lattice (fast / offline)",
    )
    ap.add_argument("--n-synthetic", type=int, default=48)
    args = ap.parse_args()
    cities = iter_megacities([args.city] if args.city else None)
    failed = []
    for c in cities:
        try:
            build_one(
                c.id,
                force=args.force,
                allow_synthetic=not args.no_synthetic,
                synthetic_only=args.synthetic,
                n_synthetic=args.n_synthetic,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{c.id}: FAILED {exc}", file=sys.stderr)
            failed.append(c.id)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
