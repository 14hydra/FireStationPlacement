# Travel-time model: LFB driving time

Model 1 of the CSEF/ISEF study: predict **driving time from a station to an
incident** with **no station IDs**, so the same model can score stations that
do not exist yet. Used by the placement scorer in `docs/fire_station_location.md`.

> **EXISTS** = in the repo today. **PLANNED** = in the plan, not built or run.
>
> **Model:** gradient-boosted decision trees (**LightGBM**), mean of 4 seeds, 44
> placement-safe features (`firestation.travel_time_model`). Trained on 701,000 LFB engine
> trips 2021–2024 (real station origin → incident), tested on 200,624 trips in 2025:
> **MAE 57.6 s** on recorded driving time (`TravelTimeSeconds`), vs 68.7 s for the Kolesar
> curve on road distance. On stations the model never saw in training: **62.4 s**.

## Results (held-out 2025 trips)

| Model | MAE | Median abs. error |
|---|---|---|
| Constant guess (train median) | 111.9 s | 85.0 s |
| Straight line @ 32 km/h | 95.2 s | 72.1 s |
| Road distance @ 35 km/h | 80.8 s | 54.8 s |
| Kolesar (straight line) | 71.4 s | 48.9 s |
| Kolesar (road distance) | 68.7 s | 45.8 s |
| **LightGBM (final)** | **57.6 s** | **35.3 s** |

Non-busy trips only (engine from the incident's own station area): LightGBM 50.0 s vs
Kolesar 58.7 s.

### Feature ladder (one LightGBM per step, 1 seed, default settings)

| Step | Features | MAE |
|---|---|---|
| Base: distance, circuity, Kolesar prior, time, busy engine, pump order, borough | 14 | 62.82 s |
| + street characteristics of the route | 33 | 60.09 s |
| + fastest route at posted limits, main-road access | 37 | 59.62 s |
| + weather | 45 | 59.75 s |
| + incident type, property type, home-station flag | 48 | 59.05 s |
| + destination history | 52 | 58.85 s |
| Without weather, tuned (2,110 trees, 255 leaves; trees chosen on 2024) | 44 | 57.80 s |
| **Final: without weather, tuned, mean of 4 seeds** | 44 | **57.63 s** |

Weather does not help: London weather barely changes engine driving speed. With the tuned
settings the model is slightly better without it (57.8 s vs 58.0 s, same on two seeds), so the
final model leaves weather out (`FINAL_GROUPS` in the trainer).

### New-station check

Placement scores sites with no station yet, so a plain time split (where every test
station was also in training) is optimistic. The trainer also splits the 102 stations into
5 groups; for each group it trains on the other stations' 2021–24 trips and predicts the
held-out stations' 2025 trips (201k trips in total):

| | MAE on held-out stations |
|---|---|
| LightGBM | **62.4 s** |
| Kolesar (road distance) | 68.7 s |

The model loses about 4.8 s on unseen stations but still beats Kolesar by about 6 s, so it
transfers to new sites. This is the accuracy to quote for placement.

### Most important features (share of LightGBM gain)

Road distance (21 %), fastest-route time at posted limits (18 %), Kolesar prior (7 %),
borough (4 %), destination history 300 m (4 %), straight-line distance (3 %), hour (2 %),
destination history 1 km (2 %), incident's distance to a main road (2 %), fastest-route
distance (2 %).
Street-feature impact in detail: `docs/london_eda.md`.

## Target and trip origin

- **Target:** LFB mobilisation records (London Datastore) give `TravelTimeSeconds`
  (leaving the station → arriving), recorded separately from `TurnoutTimeSeconds`.
- **Origin:** each mobilisation names the station the engine left from
  (`DeployedFromStation_Name`); the origin is that station's building from OpenStreetMap
  (`scripts/build_london_fire_stations.py`). Engines already out on the road and trips with a
  wrong address are dropped. Real origins vs station-area centres: 72.8 → 67.9 s.
- **Road distance:** shortest legal route on London's directed OSM drive graph
  (`firestation.road_distance`; 131,549 junctions, 306,131 directed segments, 30,145 one-way).
  One-way streets are only used in their legal direction. 67.9 → 64.2 s.
- Placement is scored as `predicted_drive + station's recorded turnout`.

## Features (`firestation.travel_time_model.build_features`)

| Group | Features | Source |
|---|---|---|
| base | `road_km`, `crow_km`, `circuity` (road/straight), `kolesar_prior_s` (Kolesar curve on road km, an input only), hour, day of week, month, rush, night, weekend, `busy_flag`, `pump_order`, `dest_exact`, borough | training set |
| street | road-class shares, one-way / 20 mph / bus-lane / bridge-tunnel shares, lanes, signals, traffic calming, crossings, give-way, junctions, turns, sharp turns, roundabouts per km — along the shortest legal route | `firestation.route_features` |
| context | `fastest_route_s` / `fastest_route_km` (fastest route at posted speed limits, class default when untagged), `origin_main_road_km` / `dest_main_road_km` (distance to the nearest A road / trunk / motorway) | `RouteFeatureRouter.route_context` |
| weather (tested, not in final model) | temperature, precipitation, wind, humidity, rain / heavy rain / cold / hot flags (central London, hourly) | Open-Meteo archive, `scripts/fetch_london_weather.py` |
| incident | incident group (fire / special service / false alarm), property category, engine left from its home station | LFB incident + mobilisation records |
| dest | `dest_resid_300m` / `dest_resid_1km`: mean of (observed − Kolesar prior) for past trips to the same 300 m / 1 km square, shrunk toward 0 (`sum / (n + 20)`), and trip counts | training trips only; out-of-fold (5 folds by incident) for training rows |

All features are placement-safe: none identifies the station. For a hypothetical site use
`pump_order = 1`, `from_home = 1` and an expected busy rate. Destination history depends only
on the destination, so it transfers to new sites (the new-station check fits it without the
held-out stations' trips).

## Design

| Piece | Choice |
|---|---|
| Learner | LightGBM, L1 objective (matches MAE), mean of 4 seeds (42–45). No parametric residual target; Kolesar is an input feature and a baseline. |
| Tuning | `learning_rate` 0.05, `num_leaves` 255, `min_child_samples` 50, `colsample_bytree` 0.7, `reg_lambda` 3; trees by early stopping on 2024 with 2021–23 training (1,835 → ×1.15 = 2,110 for the 2021–24 fit). 2025 is never used for choices. |
| Split | Train 2021–2024, test 2025 (chronological); plus the new-station check. |
| Prediction | `firestation.travel_time_model.predict(bundle, trips)` builds the same features from a saved bundle (used by the planner / visualisations). |

## Still planned

- Linear regression and small neural-net baselines on the same features.
- Quantile LightGBM (prediction intervals) for 90th-percentile placement.
- Learned per-street speeds for routing (instead of shortest / posted-limit routes).
- OS NGD carriageway width and speed data.
- Score placements with this model in `plan_fire_station_locations.py`.

## Code

| Script / module | Role |
|---|---|
| `scripts/build_london_fire_stations.py` | Real station locations (OpenStreetMap) |
| `scripts/build_lfb_travel_training.py` | One row per engine trip: origin, destination, distances, route street + context features, incident type, `drive_s` |
| `scripts/fetch_london_weather.py` | Hourly central-London weather (Open-Meteo, no key) |
| `firestation/road_distance.py`, `firestation/route_features.py` | Directed drive graph; shortest / fastest routes; street characteristics |
| `firestation/travel_time_model.py` | Feature building, destination history, LightGBM fit / predict |
| `scripts/train_travel_time_lfb.py` | Baselines, feature ladder, tuning, new-station check, final model |

## Run

```bash
python scripts/download_datasets.py            # LFB incidents + mobilisations, London OSM extract
PYTHONPATH=src python scripts/build_lfb_planner_incidents.py
python scripts/build_london_fire_stations.py      # real station locations (OpenStreetMap)
python scripts/build_lfb_travel_training.py    # engine trips + route features (~5 min; builds the graph on first run)
python scripts/fetch_london_weather.py         # hourly weather
python scripts/train_travel_time_lfb.py        # ~5 h with ladder + new-station check
python scripts/train_travel_time_lfb.py --quick   # tuning + final model only
```

Writes `data/processed/models/travel_time_lfb.joblib` (about 240 MB, not in git) and
`travel_time_lfb.metrics.json` (all metrics above, parameters, feature importance).
