#!/usr/bin/env python3
"""Build megacities.html with visible demand heat (soft circles; no Visualization API)."""
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


def build() -> Path:
    # Prefer the already-generated working file if present and fresh enough; else rebuild.
    # Always rebuild from data/ for correctness.
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
            cut = float(np.quantile(weights, 0.28))
            feats = [f for f, w in zip(feats, weights) if w >= cut]
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
                "rationale": json.loads((d / "move_rationale.json").read_text())
                if (d / "move_rationale.json").exists()
                else [],
                "caveat": c.get("caveat"),
            }
        )
    # Keep the last good HTML template from disk if it has soft-circle heatColor
    template = OUT.read_text() if OUT.exists() else ""
    if "function heatColor" not in template:
        raise SystemExit("Run the circle-based HTML generator once; template missing heatColor")
    # Swap payload + key
    template = re.sub(
        r"const API_KEY = .*?;",
        f"const API_KEY = {json.dumps(key)};",
        template,
        count=1,
    )
    template = re.sub(
        r"const CITIES = \[.*?\];\nconst state",
        "const CITIES = " + json.dumps(bundle, separators=(",", ":")) + ";\nconst state",
        template,
        count=1,
        flags=re.S,
    )
    OUT.write_text(template)
    COPY.parent.mkdir(parents=True, exist_ok=True)
    COPY.write_text(template)
    print("wrote", OUT, OUT.stat().st_size)
    return OUT


if __name__ == "__main__":
    build()
