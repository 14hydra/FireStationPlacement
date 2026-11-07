#!/usr/bin/env python3
"""Download a drive graph for a megacity core bbox (optional for Kolesar crow runs).

Writes ``data/raw/megacities/<id>/drive.graphml``. Large bboxes may OOM or time out;
failures are non-fatal for the placement orchestrator (crow Kolesar still runs).

Run:
  PYTHONPATH=src .venv/bin/python scripts/build_megacity_graph.py --city dhaka
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.megacities import get_megacity, iter_megacities, list_megacities  # noqa: E402


def build_one(city_id: str, *, force: bool = False) -> Path:
    import osmnx as ox

    city = get_megacity(city_id)
    out = city.graph_path
    if out.exists() and not force:
        print(f"{city.id}: keep existing {out}")
        return out
    ox.settings.use_cache = True
    ox.settings.timeout = 300
    print(f"{city.id}: downloading drive network {city.bbox} …", flush=True)
    G = ox.graph_from_bbox(city.bbox, network_type="drive", simplify=True)
    city.raw_dir.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, out)
    print(f"{city.id}: wrote graph nodes={G.number_of_nodes()} edges={G.number_of_edges()} → {out}")
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
