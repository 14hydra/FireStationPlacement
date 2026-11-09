# Megacity placement transfer runs

Apply the same **move-up-to-10** fire-station placement pipeline used for London to
the world’s ten largest urban agglomerations (core bboxes), then browse results in
a Google Maps viewer.

**London’s ISEF final figures (`data/figures/final/01`–`10`) are not modified.**

## Important caveats

- Other cities do **not** publish LFB-style mobilisation travel times. These runs use
  **population-proxy demand** (synthetic polycentric density, or a local `pop.tif`
  if you drop one in `data/raw/megacities/<id>/`) and **Kolesar** travel times.
- Results are **modelled transfers**, not validated local attendance savings.
- OSM `amenity=fire_station` completeness varies by country. If Overpass is
  unreachable, the station builder falls back to a **synthetic lattice**
  (`source=synthetic_fallback` in `stations_meta.json`) so the run still completes.

## Cities

Tokyo, Delhi, Shanghai, Dhaka, Cairo, São Paulo, Mexico City, Beijing, Mumbai, Osaka, plus **London** and **New York**  
(registry: `src/firestation/megacities.py`). Stations are clipped to each city outline so markers are not drawn outside the border.

## Rebuild

```bash
# All ten (synthetic stations + proxy demand + replace-10 + viewer GeoJSON)
PYTHONPATH=src .venv/bin/python scripts/run_megacity_placement.py --force

# One city
PYTHONPATH=src .venv/bin/python scripts/run_megacity_placement.py --city dhaka --force

# Try live OSM stations instead of synthetic (needs working Overpass)
PYTHONPATH=src .venv/bin/python scripts/run_megacity_placement.py --city tokyo --try-osm --force

# Optional OSM drive graph (not required for crow Kolesar)
PYTHONPATH=src .venv/bin/python scripts/build_megacity_graph.py --city dhaka
```

Outputs:

| Path | Contents |
|---|---|
| `data/raw/megacities/<id>/` | `fire_stations.csv`, `incidents_proxy.csv` |
| `data/figures/megacities/<id>/` | plan CSVs + `summary.json` |
| `data/figures/megacities/summary.csv` | Cross-city table |
| `web/megacity_maps/data/` | GeoJSON + `cities.json` for the map UI |

## Map viewer (single HTML)

Open either file in a browser (no server / API key; needs network once for map tiles):

- [`web/megacity_maps/megacities.html`](../web/megacity_maps/megacities.html)
- [`data/figures/megacities/megacities_map.html`](../data/figures/megacities/megacities_map.html)

City dropdown, kept / closed / opened stations, demand heat, coverage Δ, and
per-move rationale. Rebuild the HTML after re-running placement by regenerating
from `web/megacity_maps/data/` (same embed used above).

## Realistic station counts

Non-London cities use a **London-like station density** (~0.038 stations per km² of the study bbox, clipped to 70–120), inside the city outline. OSM stations are preferred when Overpass works; otherwise synthetic fill. This stops sparse 48-station synthetics from exaggerating move-10 savings versus London’s 102 real stations.
