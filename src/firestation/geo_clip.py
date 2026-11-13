"""Point-in-polygon helpers for megacity urban extents."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_feature_collection(geojson_path: Path) -> dict | None:
    if not geojson_path.exists():
        return None
    return json.loads(geojson_path.read_text())


def load_rings(geojson_path: Path) -> list[np.ndarray]:
    fc = load_feature_collection(geojson_path)
    if not fc:
        return []
    rings: list[np.ndarray] = []

    def _add_polygon(coords):
        if not coords:
            return
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


def study_extent_path(raw_dir: Path) -> Path:
    """Prefer bbox-clipped study outline; fall back to full Nominatim polygon."""
    study = raw_dir / "urban_extent_study.geojson"
    if study.exists():
        return study
    return raw_dir / "urban_extent.geojson"


def clip_extent_to_bbox(
    geojson_path: Path,
    bbox: tuple[float, float, float, float],
    *,
    out_path: Path | None = None,
    pad_frac: float = 0.02,
) -> Path | None:
    """Intersect Nominatim urban outline with the study bbox (coast/admin shape).

    Full metro polygons (e.g. Tokyo Metropolis) sprawl far past the placement
    bbox, so the map border and heatmap disagree. Writing a study-clipped
    outline makes heat and the blue border share the same city-shaped mask.
    """
    if not geojson_path.exists():
        return None
    try:
        from shapely.geometry import box, mapping, shape
    except ImportError:
        return None

    fc = json.loads(geojson_path.read_text())
    feats = fc.get("features") or []
    if not feats:
        return None
    geom = shape(feats[0]["geometry"])
    west, south, east, north = bbox
    pad_lon = pad_frac * max(east - west, 1e-6)
    pad_lat = pad_frac * max(north - south, 1e-6)
    study = box(west - pad_lon, south - pad_lat, east + pad_lon, north + pad_lat)
    clipped = geom.intersection(study)
    if clipped.is_empty:
        return None
    # Drop tiny island shards left after the cut
    if clipped.geom_type == "GeometryCollection":
        polys = [g for g in clipped.geoms if g.geom_type in {"Polygon", "MultiPolygon"} and not g.is_empty]
        if not polys:
            return None
        from shapely.ops import unary_union

        clipped = unary_union(polys)
    if clipped.geom_type == "MultiPolygon":
        parts = [g for g in clipped.geoms if g.area >= 0.002 * clipped.area]
        if parts:
            from shapely.ops import unary_union

            clipped = unary_union(parts) if len(parts) > 1 else parts[0]
    props = dict((feats[0].get("properties") or {}))
    props["clip"] = "study_bbox"
    out = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": props,
                "geometry": mapping(clipped),
            }
        ],
    }
    dest = out_path or (geojson_path.parent / "urban_extent_study.geojson")
    dest.write_text(json.dumps(out) + "\n")
    return dest


def points_in_rings(lon: np.ndarray, lat: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
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
        mask = np.zeros(len(lon), dtype=bool)
        for ring in rings:
            x, y = ring[:, 0], ring[:, 1]
            n = len(ring)
            for i in range(len(lon)):
                if mask[i]:
                    continue
                xi, yi = float(lon[i]), float(lat[i])
                inside = False
                j = n - 1
                for k in range(n):
                    xk, yk = float(x[k]), float(y[k])
                    xj, yj = float(x[j]), float(y[j])
                    if ((yk > yi) != (yj > yi)) and (
                        xi < (xj - xk) * (yi - yk) / (yj - yk + 1e-15) + xk
                    ):
                        inside = not inside
                    j = k
                if inside:
                    mask[i] = True
        return mask


def filter_frame_inside(
    df: pd.DataFrame,
    geojson_path: Path,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Keep rows whose lat/lon fall inside the urban extent (no-op if missing)."""
    if df is None or len(df) == 0:
        return df
    if lat_col not in df.columns or lon_col not in df.columns:
        return df
    rings = load_rings(geojson_path)
    if not rings:
        return df
    mask = points_in_rings(
        df[lon_col].to_numpy(float),
        df[lat_col].to_numpy(float),
        rings,
    )
    return df.loc[mask].reset_index(drop=True)


def bbox_area_km2(bbox: tuple[float, float, float, float]) -> float:
    """Rough geographic area of a lon/lat bbox in km²."""
    west, south, east, north = bbox
    mid = 0.5 * (south + north)
    return max((east - west) * 111.32 * float(np.cos(np.radians(mid))) * (north - south) * 110.57, 1.0)


# Greater London LFB: ~102 stations over the study bbox (~0.038 / km² of bbox).
LONDON_STATION_DENSITY_PER_KM2 = 102 / 2680.0  # ≈ bbox area of london registry entry


def target_station_count(
    bbox: tuple[float, float, float, float],
    *,
    density: float = LONDON_STATION_DENSITY_PER_KM2,
    lo: int = 70,
    hi: int = 120,
) -> int:
    """Station count for a realistic dense network (London-like density, clipped)."""
    n = int(round(density * bbox_area_km2(bbox)))
    return int(np.clip(n, lo, hi))
