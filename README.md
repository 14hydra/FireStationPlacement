# Fire Station Placement — London Fire Brigade (CSEF / ISEF)

**Optimizing London Fire Brigade station placement with a street-characteristic
travel-time model** (gradient-boosted decision trees, LightGBM),
scored on held-out 2025 LFB calls.

Team: **Ricky Zhao** and **Sarp Akalin** (ASI), CSEF/ISEF.

> **Status legend used in these docs:** **EXISTS** = code or data in this repo today ·
> **PLANNED** = in the project plan, not built / not run yet. Please do not read
> PLANNED items as results.

## Research question and hypothesis

**Research question.** Does placing fire stations with *street-aware* driving-time
predictions beat placement by straight-line distance or the classic Kolesar
distance→time curve, when scored on real 2025 LFB calls?

**Hypothesis.** A travel-time model that sees street characteristics (road class,
width, speed limits, one-ways, junction density, …) will
rank candidate station layouts differently from distance-only or Kolesar-only
placement, and the street-aware layout will do better on held-out calls
(mean first-engine attendance and share of calls within 10 min).

**Not our claim:** that we have beaten LFB's real operational planning (see
[Limits](#limits-and-honesty)).

## Standards we score against

London Fire Brigade (London Safety Plan 2022) — **not** NFPA 1710 / the US 4-minute rule:

| Standard | Target |
|---|---|
| First engine | average **≤ 6 min**; **> 90 %** of calls within **10 min** |
| Second engine | average ≤ 8 min; 95th percentile within 12 min |

Encoded in `src/firestation/lfb_standards.py` (**EXISTS**).

## What we score

A candidate placement is scored with

```
attendance  =  predicted driving time (station → incident)  +  that station's recorded turnout
```

LFB mobilisation records give driving time (`TravelTimeSeconds`) and turnout
(`TurnoutTimeSeconds`) separately, so the travel-time model is trained on driving time
alone. The planner adds each station's median recorded turnout, and the median over all stations
(73 s in 2024) for new sites.

## Three models

| # | Model | Role | Status |
|---|---|---|---|
| 1 | **Travel-time LightGBM, no station IDs** — gradient-boosted trees, 4-seed average | Predict driving time to any point from any site | **EXISTS** (`train_travel_time_lfb.py`, `firestation.travel_time_model`): mobilisation `TravelTimeSeconds`, real station origins, road distance, route street features, fastest route, incident type, destination history; held-out 2025 MAE **57.6 s** (62.4 s on stations held out of training) vs Kolesar 68.7 s; OS NGD street attrs **PLANNED** |
| 2 | **Route model** with learned segment speeds | Dijkstra under learned EMV edge speeds; FOI GPS when available | Shortest-distance routing on directed London graph **EXISTS** (`firestation.road_distance`); firetruck weights + race viz **EXIST**; learned speeds scaffold **EXISTS** (`segment_speeds.py`); FOI **PLANNED** |
| 3 | **Placement on a directed street graph** | redesign / replace / expand / resize | All modes **EXIST**; LightGBM, road, Kolesar and crow scorers **EXIST** (`--scorer lgbm\|road\|kolesar\|crow`), with each station's recorded turnout |
Details: [`docs/travel_time_training.md`](docs/travel_time_training.md),
[`docs/fire_station_location.md`](docs/fire_station_location.md).

## Validation plan

1. **2014 ten-station closures** — on 9 Jan 2014 LFB closed Belsize, Bow, Clerkenwell,
   Downham, Kingsland, Knightsbridge, Silvertown, Southwark, Westminster, Woolwich.
   Counterfactual + ranking scripts **EXIST** on today's network with approximate
   closed-station coords; the true pre-2014 train → post-2014 test needs the 2009–2017
   LFB extracts (**PLANNED** ingest).
2. **Leave-one-firehouse-out** — hide one station's calls and predict its area from the rest. **PLANNED**
3. **Multi-scorer** — score layouts under Kolesar / crow / ML; planner supports
   `--scorer kolesar|crow` today (**EXISTS**); full cross-table comparison script **PLANNED**

Temporal split: the travel-time model trains on **2021–2024** and tests on **2025**.
The planner takes demand and evaluation years from `--demand-years` / `--eval-years`.

## Datasets

All raw files live in `data/raw/london/` (not in git). `scripts/download_datasets.py`
fetches them; see [`docs/datasets.md`](docs/datasets.md).

| Role | Source | Status |
|---|---|---|
| Incidents (location, station ground, attendance) | **LFB incident records** 2018–2023 and 2024 onwards (London Datastore) | **EXISTS** |
| Engine trips (deployed-from station, turnout, driving time) | **LFB mobilisation records** 2021–2024 and 2025 onwards (London Datastore) | **EXISTS** |
| Station locations | OpenStreetMap fire stations, name-matched to LFB (`scripts/build_london_fire_stations.py`) | **EXISTS** |
| Street network | OpenStreetMap, Geofabrik Greater London extract → directed drive graph (`firestation.road_distance`) | **EXISTS** |
| Street network (detail) | **Ordnance Survey NGD** (width, speed limit, class) | **PLANNED** |
| 2014 closures | Hard-coded list in `firestation.london_closures_2014` | **EXISTS** |

## Code

| Piece | Path |
|---|---|
| London data download | `scripts/download_datasets.py` |
| Station locations (OSM) | `scripts/build_london_fire_stations.py` |
| Engine-trip training set (real origin, road distance) | `scripts/build_lfb_travel_training.py` |
| Shortest legal road distance (one-way streets respected) | `src/firestation/road_distance.py` |
| LightGBM travel-time trainer | `scripts/train_travel_time_lfb.py` |
| Kolesar piecewise `T(d)` (fit + predict) | `src/firestation/kolesar.py` |
| LFB planner CSV builder (with `busy_flag`) | `scripts/build_lfb_planner_incidents.py` |
| Planner drive matrices from the LightGBM model / road graph | `src/firestation/ml_scorer.py` |
| Firehouse planner (redesign, replace, expand) | `src/firestation/fire_station_location.py`, `src/firestation/facility.py`, `scripts/plan_fire_station_locations.py` |
| LFB 6 / 10-min standards scoring, turnout + drive | `src/firestation/lfb_standards.py` |
| Busy-engine rates (`deployed ≠ ground`) | `src/firestation/busy_engines.py` |
| 2014 closure list + counterfactual + ranking | `src/firestation/london_closures_2014.py`, `scripts/validate_london_2014_closures.py`, `scripts/rank_closed_stations_2014.py` |
| Segment-speed route scaffold | `src/firestation/segment_speeds.py`, `scripts/fit_segment_speeds.py` |
| Incident density map | `scripts/plot_incident_density.py` |
| Street features along each route | `src/firestation/route_features.py` |
| Initial data analysis + street feature impact | `scripts/analyze_london.py` → `data/figures/data_exploration/london_eda/` |
| Station capacity / busyness | `scripts/analyze_station_capacity.py`, `plot_station_busyness.py` |
| All station metrics → 2D embedding | `scripts/embed_analysis_metrics_2d.py` → `data/figures/analysis/station_embedding/` |
| Figure index | [`data/figures/README.md`](data/figures/README.md) · [`scripts/README.md`](scripts/README.md) |

Still **PLANNED**: OS NGD download,
SHAP/ablations, leave-one-firehouse-out CV, pre-2014 LFB ingest, a multi-scorer comparison table.

## Quick start

```bash
cd FireStationPlacement
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
brew install libomp            # macOS: LightGBM needs it

# 1. Download LFB incidents + mobilisations and the London OSM extract (~600 MB)
python scripts/download_datasets.py

# 2. Derived tables
PYTHONPATH=src python scripts/build_lfb_planner_incidents.py
python scripts/build_london_fire_stations.py
python scripts/build_lfb_travel_training.py      # builds the road graph on first run
python scripts/fetch_london_weather.py

# 3. Travel-time model (LightGBM; ~5 h with all checks, --quick for the model only)
python scripts/train_travel_time_lfb.py

# 4. Incident density map -> data/figures/data_exploration/london_incident_density.html
PYTHONPATH=src python scripts/plot_incident_density.py

# 4b. Initial analysis + street feature impact -> data/figures/data_exploration/london_eda/ (~15 min)
python scripts/analyze_london.py

# 4c. All station metrics condensed to 2D -> data/figures/analysis/station_embedding/
python scripts/embed_analysis_metrics_2d.py

# 5. Placement planner
PYTHONPATH=src python scripts/plan_fire_station_locations.py \
  --city london --mode redesign --demand-years 2024 --eval-years 2025
PYTHONPATH=src python scripts/plan_fire_station_locations.py --city london --mode replace --replace 10
PYTHONPATH=src python scripts/plan_fire_station_locations.py --city london --mode expand --add-stations 2

# 6. 2014 closure scaffolds
PYTHONPATH=src python scripts/validate_london_2014_closures.py
PYTHONPATH=src python scripts/rank_closed_stations_2014.py

# 7. 3D race: newly placed station vs existing (firetruck routing)
PYTHONPATH=src python scripts/build_london_drive_graph.py                 # central London OSM graph
PYTHONPATH=src python scripts/build_fire_station_race_3d.py
open data/figures/placement/planner_runs/fire_station_race_3d.html

# Synthetic smoke test (no real data)
PYTHONPATH=src python scripts/plan_fire_station_locations.py --demo
```
Defaults: `--city london`, `--scorer lgbm` (decision-tree travel times; `kolesar` for other cities), `--threshold-min 10`, LFB
`score_first_engine` in the summary, drive + turnout attendance scoring. Modelled cover
rates are optimistic vs observed attendance (nearest engine always free unless `--use-busy`).

## Docs

| Doc | Content |
|---|---|
| [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) | Research plan, prior work, validation, references |
| [`docs/travel_time_training.md`](docs/travel_time_training.md) | LFB driving-time model; trip origin; road distance; baselines |
| [`docs/fire_station_location.md`](docs/fire_station_location.md) | Placement: modes, KPIs, scorers, closures, busy engines |
| [`docs/datasets.md`](docs/datasets.md) | London data sources and how they are joined |
| [`docs/london_eda.md`](docs/london_eda.md) | Initial analysis; which street features affect driving time |
| [`docs/station_embedding_analysis.md`](docs/station_embedding_analysis.md) | Reading the 2D station-metric embedding (PCA / UMAP / archetypes) |
| [`docs/model_improvements_from_embedding.md`](docs/model_improvements_from_embedding.md) | How the embedding changed the travel-time + resize models |
| ISEF final figure pack | [`data/figures/final/`](data/figures/final/) (`00_roadmap.png` → `01`…`10`) |
| [`docs/model_log.md`](docs/model_log.md) | Catalog of model implementations + blank entry |

## Limits and honesty

- Station sites in `london_fire_stations.csv` come from OpenStreetMap building footprints,
  name-matched to LFB stations (8 pinned by hand, see the script).
- Engines are assumed to take the shortest-distance legal route; emergency exemptions
  (contraflow, bus gates) are not modelled.
- `busy_flag` (engine deployed from a station other than the incident's ground station)
  is a proxy for "home engine unavailable".
- Candidate sites are discrete; no land cost, staffing, or planning constraints.
- Street-feature gains over a good distance model may be small. The claim is about
  **placement ranking**, and we will report that honestly whichever way it comes out.
