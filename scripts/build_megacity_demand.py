#!/usr/bin/env python3
"""Build population-proxy demand incidents for megacity placement.

If ``data/raw/megacities/<id>/pop.tif`` exists (WorldPop/GHSL GeoTIFF), sample
from it. Otherwise synthesise a polycentric urban density field inside the city
bbox (documented proxy — not real fire calls).

Writes ``incidents_proxy.csv`` with dest_lat, dest_lon, weight, travel_seconds
(nearest-station Kolesar estimate so the planner can fit a local curve).

Run:
  PYTHONPATH=src .venv/bin/python scripts/build_megacity_demand.py
  PYTHONPATH=src .venv/bin/python scripts/build_megacity_demand.py --city delhi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.kolesar import KolesarModel  # noqa: E402
from firestation.megacities import get_megacity, iter_megacities, list_megacities  # noqa: E402


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _synthetic_weights(
    lat: np.ndarray,
    lon: np.ndarray,
    center: tuple[float, float],
    bbox: tuple[float, float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Polycentric urban density: main CBD + 4 satellites + background."""
    clon, clat = center
    west, south, east, north = bbox
    span_lon = max(east - west, 1e-6)
    span_lat = max(north - south, 1e-6)
    # characteristic scale ~ 1/4 of bbox
    sd_lon = 0.22 * span_lon
    sd_lat = 0.22 * span_lat
    cores = [(clon, clat, 1.0)]
    offsets = [(-0.28, -0.18), (0.30, 0.12), (-0.15, 0.28), (0.22, -0.25)]
    for i, (dx, dy) in enumerate(offsets):
        cores.append((clon + dx * span_lon, clat + dy * span_lat, 0.45 - 0.05 * i))
    w = np.zeros(len(lat), dtype=float)
    for cx, cy, amp in cores:
        w += amp * np.exp(
            -0.5 * (((lon - cx) / sd_lon) ** 2 + ((lat - cy) / sd_lat) ** 2)
        )
    # mild edge fade
    edge = (
        np.clip((lon - west) / span_lon, 0, 1)
        * np.clip((east - lon) / span_lon, 0, 1)
        * np.clip((lat - south) / span_lat, 0, 1)
        * np.clip((north - lat) / span_lat, 0, 1)
    )
    w *= 0.25 + 0.75 * edge
    w *= 0.85 + 0.3 * rng.random(len(w))
    return np.maximum(w, 1e-6)


def _from_geotiff(path: Path, bbox, n_cells: int, rng: np.random.Generator) -> pd.DataFrame:
    import rasterio
    from rasterio.windows import from_bounds

    west, south, east, north = bbox
    with rasterio.open(path) as src:
        window = from_bounds(west, south, east, north, transform=src.transform)
        data = src.read(1, window=window, boundless=True, fill_value=0)
        transform = src.window_transform(window)
    data = np.where(np.isfinite(data), data, 0.0).astype(float)
    data = np.maximum(data, 0.0)
    if data.sum() <= 0:
        raise ValueError(f"empty population raster: {path}")
    # Aggregate to ~n_cells by stride
    h, w = data.shape
    stride = max(1, int(np.sqrt((h * w) / max(n_cells, 1))))
    rows, cols, weights, lats, lons = [], [], [], [], []
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            block = data[i : i + stride, j : j + stride]
            wt = float(block.sum())
            if wt <= 0:
                continue
            # pixel center of block
            r = i + min(stride, h - i) / 2.0
            c = j + min(stride, w - j) / 2.0
            lon, lat = rasterio.transform.xy(transform, r, c)
            if not (south <= lat <= north and west <= lon <= east):
                continue
            rows.append(i)
            cols.append(j)
            weights.append(wt)
            lats.append(lat)
            lons.append(lon)
    df = pd.DataFrame({"dest_lat": lats, "dest_lon": lons, "weight": weights})
    if len(df) > n_cells:
        # keep heaviest cells
        df = df.nlargest(n_cells, "weight")
    return df.reset_index(drop=True)


def _load_urban_rings(city) -> list[np.ndarray]:
    """Exterior rings from cached Nominatim urban_extent.geojson as (N,2) lon/lat."""
    path = city.raw_dir / "urban_extent.geojson"
    if not path.exists():
        return []
    fc = json.loads(path.read_text())
    rings: list[np.ndarray] = []

    def _add_polygon(coords):
        if not coords:
            return
        # GeoJSON polygon: first ring is exterior
        exterior = np.asarray(coords[0], dtype=float)
        if exterior.ndim == 2 and len(exterior) >= 3:
            rings.append(exterior[:, :2])

    for feat in fc.get("features") or []:
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "Polygon":
            _add_polygon(coords)
        elif gtype == "MultiPolygon":
            for poly in coords or []:
                _add_polygon(poly)
    return rings


def _points_in_rings(lon: np.ndarray, lat: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """True if point is inside any ring (matplotlib path; fallback ray casting)."""
    if not rings:
        return np.ones(len(lon), dtype=bool)
    try:
        from matplotlib.path import Path as MplPath

        mask = np.zeros(len(lon), dtype=bool)
        pts = np.column_stack([lon, lat])
        for ring in rings:
            mask |= MplPath(ring).contains_points(pts)
        return mask
    except Exception:  # noqa: BLE001
        # Ray casting fallback
        mask = np.zeros(len(lon), dtype=bool)
        for ring in rings:
            x, y = ring[:, 0], ring[:, 1]
            n = len(ring)
            for i in range(len(lon)):
                if mask[i]:
                    continue
                xi, yi = lon[i], lat[i]
                inside = False
                j = n - 1
                for k in range(n):
                    xk, yk = x[k], y[k]
                    xj, yj = x[j], y[j]
                    if ((yk > yi) != (yj > yi)) and (
                        xi < (xj - xk) * (yi - yk) / (yj - yk + 1e-15) + xk
                    ):
                        inside = not inside
                    j = k
                if inside:
                    mask[i] = True
        return mask


def _grid_proxy(city, n_cells: int, rng: np.random.Generator) -> pd.DataFrame:
    """City-shaped demand: dense grid clipped to urban polygon (else soft ellipse)."""
    west, south, east, north = city.bbox
    clon, clat = city.center
    # Fine grid over bbox, then clip hard to the city outline when available
    n_side = int(np.ceil(np.sqrt(n_cells * 3.5)))
    lons = np.linspace(west + 0.005 * (east - west), east - 0.005 * (east - west), n_side)
    lats = np.linspace(south + 0.005 * (north - south), north - 0.005 * (north - south), n_side)
    lon_g, lat_g = np.meshgrid(lons, lats)
    lat = lat_g.ravel()
    lon = lon_g.ravel()
    w = _synthetic_weights(lat, lon, city.center, city.bbox, rng)

    rings = _load_urban_rings(city)
    if rings:
        inside = _points_in_rings(lon, lat, rings)
        w = np.where(inside, w, 0.0)
        print(f"  clipped to urban polygon ({inside.sum()} / {len(inside)} cells inside)")
    else:
        # Fallback ellipse when Nominatim cache is missing
        span_lon = max(east - west, 1e-6)
        span_lat = max(north - south, 1e-6)
        rx, ry = 0.42 * span_lon, 0.38 * span_lat
        ell = ((lon - clon) / rx) ** 2 + ((lat - clat) / ry) ** 2
        w = w * np.clip(1.1 - ell, 0.0, 1.0) ** 1.6
        w[ell > 1.05] = 0.0
        print("  no urban_extent.geojson — using elliptical fallback")

    df = pd.DataFrame({"dest_lat": lat, "dest_lon": lon, "weight": w})
    df = df[df["weight"] > 0]
    if df.empty:
        raise RuntimeError(f"{city.id}: urban mask removed all demand cells")
    cutoff = float(df["weight"].quantile(0.22))
    df = df[df["weight"] >= cutoff]
    if len(df) > n_cells:
        df = df.nlargest(n_cells, "weight")
    return df.reset_index(drop=True)


def _attach_travel_seconds(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Nearest-station crow Kolesar as a stand-in observed attendance."""
    model = KolesarModel.london_default()
    turnout = 70.0
    slat = stations["lat"].to_numpy(float)
    slon = stations["lon"].to_numpy(float)
    dlat = df["dest_lat"].to_numpy(float)
    dlon = df["dest_lon"].to_numpy(float)
    # broadcast min distance
    km = np.full(len(df), np.inf)
    for i in range(0, len(stations), 40):
        chunk_lat = slat[i : i + 40]
        chunk_lon = slon[i : i + 40]
        d = _haversine_km(dlat[:, None], dlon[:, None], chunk_lat[None, :], chunk_lon[None, :])
        km = np.minimum(km, d.min(axis=1))
    drive = model.predict(km)
    # expand each cell into weight-proportional pseudo-incidents (capped)
    rows = []
    for i, r in df.iterrows():
        n = int(max(1, min(40, round(float(r["weight"]) / max(df["weight"].median(), 1e-9) * 8))))
        for _ in range(n):
            rows.append(
                {
                    "dest_lat": r["dest_lat"],
                    "dest_lon": r["dest_lon"],
                    "travel_seconds": float(drive[i] + turnout),
                    "weight": float(r["weight"]) / n,
                }
            )
    out = pd.DataFrame(rows)
    # planner weights by row count in a cell unless severity; keep one row per
    # sample so cell_weights = counts. Drop weight col after expanding.
    return out.drop(columns=["weight"], errors="ignore")


def build_one(city_id: str, *, n_cells: int = 2500, force: bool = False, seed: int = 42) -> Path:
    city = get_megacity(city_id)
    out = city.incidents_path
    if out.exists() and not force:
        print(f"{city.id}: keep existing {out}")
        return out
    if not city.fire_stations_path.exists():
        raise FileNotFoundError(f"{city.id}: build stations first ({city.fire_stations_path})")
    stations = pd.read_csv(city.fire_stations_path)
    rng = np.random.default_rng(seed + abs(hash(city.id)) % 10_000)
    pop_tif = city.raw_dir / "pop.tif"
    if pop_tif.exists():
        print(f"{city.id}: sampling demand from {pop_tif}")
        cells = _from_geotiff(pop_tif, city.bbox, n_cells, rng)
        source = "geotiff"
    else:
        print(f"{city.id}: synthesising polycentric population-proxy demand ({n_cells} cells)")
        cells = _grid_proxy(city, n_cells, rng)
        source = "synthetic_polycentric"
    incidents = _attach_travel_seconds(cells, stations)
    import json

    city.raw_dir.mkdir(parents=True, exist_ok=True)
    incidents.to_csv(out, index=False)
    (city.raw_dir / "demand_meta.json").write_text(
        json.dumps(
            {
                "source": source,
                "n_cells": int(len(cells)),
                "n_incident_rows": int(len(incidents)),
                "note": "Population proxy for placement transfer; not real fire incidents.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{city.id}: wrote {len(incidents)} proxy incidents → {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", choices=list_megacities(), default=None)
    ap.add_argument("--n-cells", type=int, default=2500)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    failed = []
    for c in iter_megacities([args.city] if args.city else None):
        try:
            build_one(c.id, n_cells=args.n_cells, force=args.force, seed=args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"{c.id}: FAILED {exc}", file=sys.stderr)
            failed.append(c.id)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
