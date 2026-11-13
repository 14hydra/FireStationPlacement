#!/usr/bin/env python3
"""Clip megacity heat / stations / border to the study city outline and rebuild the map.

Does not re-run placement. Writes study-clipped ``urban_extent_study.geojson``,
re-exports viewer GeoJSON inside that outline, densifies demand heat for display,
and rebuilds ``megacities.html``.

Run:
  PYTHONPATH=src .venv/bin/python scripts/refresh_megacity_map_borders.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.geo_clip import (  # noqa: E402
    clip_extent_to_bbox,
    filter_frame_inside,
    study_extent_path,
)
from firestation.megacities import iter_megacities, list_megacities  # noqa: E402

WEB_DATA = ROOT / "web" / "megacity_maps" / "data"


def _geojson_points(df: pd.DataFrame, props: list[str], lat="lat", lon="lon") -> dict:
    feats = []
    for _, r in df.iterrows():
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(r[lon]), float(r[lat])]},
                "properties": {k: (None if pd.isna(r[k]) else r[k]) for k in props if k in r.index},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def _densify_heat(cells: pd.DataFrame, extent_path: Path, n_side: int = 48) -> pd.DataFrame:
    """Fine grid inside the study outline; weight from nearest planner cell."""
    from firestation.geo_clip import load_rings, points_in_rings

    if cells.empty:
        return cells
    rings = load_rings(extent_path)
    lon0, lon1 = float(cells["lon"].min()), float(cells["lon"].max())
    lat0, lat1 = float(cells["lat"].min()), float(cells["lat"].max())
    # Expand slightly toward the outline bbox
    if rings:
        all_pts = np.vstack(rings)
        lon0, lon1 = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
        lat0, lat1 = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
    pad_lon = 0.02 * max(lon1 - lon0, 1e-6)
    pad_lat = 0.02 * max(lat1 - lat0, 1e-6)
    lons = np.linspace(lon0 + pad_lon, lon1 - pad_lon, n_side)
    lats = np.linspace(lat0 + pad_lat, lat1 - pad_lat, n_side)
    gg_lon, gg_lat = np.meshgrid(lons, lats)
    lon = gg_lon.ravel()
    lat = gg_lat.ravel()
    if rings:
        mask = points_in_rings(lon, lat, rings)
        lon, lat = lon[mask], lat[mask]
    if len(lon) == 0:
        return cells

    src_lon = cells["lon"].to_numpy(float)
    src_lat = cells["lat"].to_numpy(float)
    src_w = cells["weight"].to_numpy(float) if "weight" in cells.columns else np.ones(len(cells))
    # Nearest-neighbour weight (chunked)
    w = np.zeros(len(lon), dtype=float)
    before = cells["response_s_current"].to_numpy(float) if "response_s_current" in cells.columns else None
    after = cells["response_s"].to_numpy(float) if "response_s" in cells.columns else None
    delta = cells["response_s_delta"].to_numpy(float) if "response_s_delta" in cells.columns else None
    if delta is None and before is not None and after is not None:
        delta = after - before
    b_out = np.full(len(lon), np.nan) if before is not None else None
    a_out = np.full(len(lon), np.nan) if after is not None else None
    d_out = np.full(len(lon), np.nan) if delta is not None else None

    for i0 in range(0, len(lon), 400):
        i1 = min(i0 + 400, len(lon))
        d2 = (lon[i0:i1, None] - src_lon[None, :]) ** 2 + (lat[i0:i1, None] - src_lat[None, :]) ** 2
        j = d2.argmin(axis=1)
        # Drop points whose nearest planner cell is far (outside coverage)
        near = d2.min(axis=1) <= (0.035**2)
        idx = np.arange(i0, i1)[near]
        jj = j[near]
        w[idx] = src_w[jj]
        if before is not None:
            b_out[idx] = before[jj]
        if after is not None:
            a_out[idx] = after[jj]
        if delta is not None:
            d_out[idx] = delta[jj]

    keep = w > 0
    out = pd.DataFrame({"lat": lat[keep], "lon": lon[keep], "weight": w[keep]})
    if b_out is not None:
        out["response_s_current"] = b_out[keep]
    if a_out is not None:
        out["response_s"] = a_out[keep]
    if d_out is not None:
        out["response_s_delta"] = d_out[keep]
    return out if len(out) else cells


def refresh_city(city_id: str) -> None:
    from firestation.megacities import get_megacity

    city = get_megacity(city_id)
    full = city.raw_dir / "urban_extent.geojson"
    study = clip_extent_to_bbox(full, city.bbox, out_path=city.raw_dir / "urban_extent_study.geojson")
    extent = study_extent_path(city.raw_dir)
    print(f"{city.id}: study extent → {extent.name} ({'new' if study else 'reuse'})")

    dest = WEB_DATA / city.id
    if not dest.exists():
        print(f"{city.id}: no viewer data, skip")
        return

    fig = city.fig_dir
    cells_path = fig / "demand_cells.csv"
    if not cells_path.exists():
        print(f"{city.id}: no demand_cells.csv, skip")
        return
    cells = pd.read_csv(cells_path)
    if "response_s_delta" not in cells.columns and {"response_s", "response_s_current"} <= set(cells.columns):
        cells["response_s_delta"] = cells["response_s"] - cells["response_s_current"]

    cells = filter_frame_inside(cells, extent, lat_col="lat", lon_col="lon")
    heat = _densify_heat(cells, extent, n_side=52)
    heat = filter_frame_inside(heat, extent, lat_col="lat", lon_col="lon")
    props = [c for c in ("weight", "share", "response_s", "response_s_current", "response_s_delta") if c in heat.columns]
    (dest / "demand_cells.geojson").write_text(json.dumps(_geojson_points(heat, props)))
    print(f"{city.id}: demand heat {len(cells)} planner cells → {len(heat)} display points")

    # Re-clip station layers from figure CSVs when present
    planned = fig / "planned_fire_stations.csv"
    closed_p = fig / "closed_fire_stations.csv"
    opened_p = fig / "opened_fire_stations.csv"
    if planned.exists():
        planned_df = pd.read_csv(planned)
        closed = pd.read_csv(closed_p) if closed_p.exists() else pd.DataFrame()
        opened = pd.read_csv(opened_p) if opened_p.exists() else pd.DataFrame()
        closed_names = set(closed["site_name"].astype(str)) if len(closed) and "site_name" in closed.columns else set()
        name_col = "site_name" if "site_name" in planned_df.columns else "facilityname"
        kept = planned_df[planned_df["is_existing"] == True].copy() if "is_existing" in planned_df.columns else planned_df.copy()
        if name_col in kept.columns and closed_names:
            kept = kept[~kept[name_col].astype(str).isin(closed_names)]
        kept = filter_frame_inside(kept, extent)
        closed = filter_frame_inside(closed, extent) if len(closed) else closed
        opened = filter_frame_inside(opened, extent) if len(opened) else opened

        def pts(frame, key=name_col):
            if frame is None or len(frame) == 0:
                return {"type": "FeatureCollection", "features": []}
            f = frame.copy()
            if key in f.columns:
                f = f.rename(columns={key: "name"})
            else:
                f["name"] = f.index.astype(str)
            return _geojson_points(f, [c for c in ("name", "n_pumps") if c in f.columns])

        (dest / "stations_kept.geojson").write_text(json.dumps(pts(kept)))
        (dest / "stations_closed.geojson").write_text(json.dumps(pts(closed)))
        (dest / "stations_opened.geojson").write_text(json.dumps(pts(opened)))
        print(f"{city.id}: stations kept={len(kept)} closed={len(closed)} opened={len(opened)}")

    # Copy study outline into viewer data
    if extent.exists():
        (dest / "urban_extent.geojson").write_text(extent.read_text())


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", choices=list_megacities(), default=None)
    args = ap.parse_args()
    for c in iter_megacities([args.city] if args.city else None):
        try:
            refresh_city(c.id)
        except Exception as exc:  # noqa: BLE001
            print(f"{c.id}: FAILED {exc}", file=sys.stderr)
    # Rebuild HTML with clipped extent + tighter heat
    from build_megacity_map_html import build

    build()
    return 0


if __name__ == "__main__":
    # Allow importing sibling script
    sys.path.insert(0, str(ROOT / "scripts"))
    raise SystemExit(main())
