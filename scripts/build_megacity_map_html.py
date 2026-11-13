#!/usr/bin/env python3
"""Build megacities.html with city-shaped demand heat + study outline border."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "web" / "megacity_maps" / "data"
OUT = ROOT / "web" / "megacity_maps" / "megacities.html"
COPY = ROOT / "data" / "figures" / "megacities" / "megacities_map.html"
DEFAULT_KEY = "AIzaSyBconoJs9_60A1VCt-lj_DzO2ZRE0PShQA"


def _api_key() -> str:
    env = ROOT / ".env"
    if env.exists():
        m = re.search(r"GOOGLE_MAPS_API_KEY\s*=\s*(\S+)", env.read_text())
        if m:
            return m.group(1).strip().strip('"').strip("'")
    return DEFAULT_KEY


def _simplify_extent(fc: dict, *, tol: float = 0.0015) -> dict:
    """Shrink Nominatim outlines so the HTML payload stays manageable."""
    try:
        from shapely.geometry import mapping, shape
    except ImportError:
        return fc
    feats = []
    for f in fc.get("features") or []:
        geom = f.get("geometry")
        if not geom:
            continue
        g = shape(geom)
        if g.is_empty:
            continue
        simple = g.simplify(tol, preserve_topology=True)
        feats.append(
            {
                "type": "Feature",
                "properties": f.get("properties") or {},
                "geometry": mapping(simple),
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def _load_extent(cid: str) -> dict | None:
    for path in (
        DATA / cid / "urban_extent.geojson",
        ROOT / "data" / "raw" / "megacities" / cid / "urban_extent_study.geojson",
        ROOT / "data" / "raw" / "megacities" / cid / "urban_extent.geojson",
    ):
        if path.exists():
            return _simplify_extent(json.loads(path.read_text()))
    return None


def _tighten_heat_js(template: str) -> str:
    """Keep soft-circle glow small so heat does not spill past the city outline."""
    template = re.sub(
        r"radius:\s*280\s*\+\s*420\s*\*\s*t",
        "radius: 160 + 220 * t",
        template,
    )
    template = re.sub(
        r"radius:\s*120\s*\+\s*260\s*\*\s*t",
        "radius: 70 + 140 * t",
        template,
    )
    template = re.sub(
        r"radius:\s*320([,\s]*\n\s*fillColor:\s*deltaColor)",
        r"radius: 200\1",
        template,
    )
    # Caveat line
    template = template.replace(
        "Demand follows the real city/admin polygon (Nominatim), so heat hugs the land — not a square bbox.",
        "Demand + border use the study outline (admin/coast ∩ core bbox), so heat stays inside the city shape.",
    )
    return template


def build() -> Path:
    key = _api_key()
    cities_meta = json.loads((DATA / "cities.json").read_text())
    bundle = []
    for c in cities_meta:
        cid = c["id"]
        d = DATA / cid
        if not d.exists():
            continue
        demand = json.loads((d / "demand_cells.geojson").read_text())
        feats = demand.get("features") or []
        weights = [float((f.get("properties") or {}).get("weight") or 0) for f in feats]
        if weights:
            cut = float(np.quantile(weights, 0.18))
            feats = [f for f, w in zip(feats, weights) if w >= cut]
        extent = _load_extent(cid)
        if extent and extent.get("features"):
            try:
                from shapely.geometry import Point, shape

                mask = shape(extent["features"][0]["geometry"]).buffer(1e-5)
                feats = [
                    f
                    for f in feats
                    if mask.contains(Point(f["geometry"]["coordinates"]))
                ]
            except Exception:  # noqa: BLE001
                pass
        for f in feats:
            p = f.get("properties") or {}
            f["properties"] = {
                k: p[k]
                for k in ("weight", "response_s", "response_s_current", "response_s_delta")
                if k in p
            }
        demand["features"] = feats
        bundle.append(
            {
                "id": cid,
                "name": c.get("name", cid),
                "center": c.get("center"),
                "metrics": c.get("metrics") or {},
                "summary": json.loads((d / "summary.json").read_text())
                if (d / "summary.json").exists()
                else c,
                "kept": json.loads((d / "stations_kept.geojson").read_text()),
                "closed": json.loads((d / "stations_closed.geojson").read_text()),
                "opened": json.loads((d / "stations_opened.geojson").read_text()),
                "demand": demand,
                "extent": extent,
                "rationale": json.loads((d / "move_rationale.json").read_text())
                if (d / "move_rationale.json").exists()
                else [],
                "caveat": c.get("caveat"),
            }
        )
    template = OUT.read_text() if OUT.exists() else ""
    if "function heatColor" not in template:
        raise SystemExit("Run the circle-based HTML generator once; template missing heatColor")
    template = _tighten_heat_js(template)
    template = re.sub(
        r"const API_KEY = .*?;",
        f"const API_KEY = {json.dumps(key)};",
        template,
        count=1,
    )
    # Avoid re.sub on the multi-MB payload (unicode escapes). Splice by markers.
    start = template.find("const CITIES = ")
    end = template.find("\nconst state", start)
    if start < 0 or end < 0:
        raise SystemExit("Could not locate CITIES payload markers in megacities.html")
    payload = "const CITIES = " + json.dumps(bundle, separators=(",", ":")) + ";"
    template = template[:start] + payload + template[end:]
    OUT.write_text(template)
    COPY.parent.mkdir(parents=True, exist_ok=True)
    COPY.write_text(template)
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB,", len(bundle), "cities")
    return OUT


if __name__ == "__main__":
    build()
