#!/usr/bin/env python3
"""Fetch / cache an urban-extent polygon per megacity (Nominatim).

Writes ``data/raw/megacities/<id>/urban_extent.geojson`` used to clip demand
so heatmaps follow the city/coast instead of a bounding-box ellipse.

Run:
  PYTHONPATH=src .venv/bin/python scripts/build_megacity_urban_extent.py
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.megacities import get_megacity, iter_megacities, list_megacities  # noqa: E402

# Nominatim search hints (prefer metro / city that matches our core bbox)
NOMINATIM_QUERY: dict[str, dict[str, str]] = {
    "tokyo": {"q": "Tokyo Metropolis, Japan"},
    "delhi": {"q": "Delhi, India"},
    "shanghai": {"q": "Shanghai, China"},
    "dhaka": {"q": "Dhaka District, Bangladesh"},
    "cairo": {"q": "Cairo, Egypt"},
    "sao_paulo": {"q": "São Paulo, Brazil"},
    "mexico_city": {"q": "Mexico City, Mexico"},
    "beijing": {"q": "Beijing, China"},
    "mumbai": {"q": "Mumbai City District, Maharashtra, India"},
    "osaka": {"q": "Osaka, Japan"},
}


def fetch_extent(city_id: str) -> dict:
    params = {
        **NOMINATIM_QUERY[city_id],
        "format": "geojson",
        "polygon_geojson": "1",
        "limit": "1",
    }
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "EMVRouteOptimization/1.0 (research; megacity placement)"},
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=45, context=ctx) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    feats = data.get("features") or []
    if not feats:
        raise RuntimeError(f"{city_id}: Nominatim returned no polygon")
    feat = feats[0]
    geom = feat.get("geometry") or {}
    if geom.get("type") not in {"Polygon", "MultiPolygon"}:
        raise RuntimeError(f"{city_id}: geometry type {geom.get('type')} not polygonal")
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "city_id": city_id,
                    "name": (feat.get("properties") or {}).get("display_name"),
                    "source": "nominatim",
                },
                "geometry": geom,
            }
        ],
    }


def build_one(city_id: str, *, force: bool = False) -> Path:
    city = get_megacity(city_id)
    out = city.raw_dir / "urban_extent.geojson"
    if out.exists() and not force:
        print(f"{city.id}: keep {out}")
        return out
    print(f"{city.id}: fetching urban extent…", flush=True)
    fc = fetch_extent(city.id)
    city.raw_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fc) + "\n")
    print(f"{city.id}: wrote {out}")
    time.sleep(1.1)  # Nominatim usage policy
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", choices=list_megacities(), default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    failed = []
    for c in iter_megacities([args.city] if args.city else None):
        try:
            build_one(c.id, force=args.force)
        except Exception as exc:  # noqa: BLE001
            print(f"{c.id}: FAILED {exc}", file=sys.stderr)
            failed.append(c.id)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
