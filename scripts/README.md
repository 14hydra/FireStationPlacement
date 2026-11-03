# Scripts

Runnable entry points. Library code lives in `src/firestation/`.

| Script | Does |
|---|---|
| `download_datasets.py` | Fetch LFB + OSM inputs into `data/raw/london/` |
| `build_london_fire_stations.py` | OSM stations name-matched to LFB |
| `build_london_drive_graph.py` | Directed London drive graph |
| `build_lfb_travel_training.py` | Engine-trip training rows (origin → incident) |
| `build_lfb_planner_incidents.py` | Planner demand CSV (with busy flag) |
| `train_travel_time_lfb.py` | LightGBM travel-time model (2021–24 → 2025) |
| `train_travel_time_quantile.py` | Quantile / interval companion |
| `train_street_speeds.py` | Learned street speeds + LightGBM correction; compared with LightGBM alone |
| `analyze_london.py` | London EDA + street-feature panels |
| `analyze_station_capacity.py` | Pumps vs demand, two-pump inference |
| `plot_station_busyness.py` | Utilisation / all-out by station |
| `plot_incident_density.py` | City incident density map |
| `plot_travel_time_maps.py` | First-engine area maps |
| `plan_fire_station_locations.py` | Placement planner (redesign / replace / expand / resize) |
| `recommended_placement.py` | Recommended layout pack |
| `score_layouts.py` | Score named layouts under LightGBM / Kolesar / road |
| `placement_tradeoff.py` | Efficiency–equity α curves |
| `placement_robustness.py` | Layouts by time / severity / borough / income |
| `plot_results_figures.py` | Station-count sweep and dispatch-share figures for `docs/results.md` |
| `relabel_grid_sites.py` | Name grid-cell sites in saved layouts after their nearest street |
| `placement_slow_calls.py` | Optimise the slowest calls |
| `build_fire_station_race_3d.py` | 3D race visual for a planner run |
| `embed_analysis_metrics_2d.py` | Join all station metrics → PCA / UMAP / t-SNE |
| `closure_2014_*.py`, `rank_closed_stations_2014.py`, `validate_london_2014_closures.py` | 2014 closure validation |
| `fit_dispatch_model.py` | Which station sends first |
| `neutral_siting_test.py` | Neutral test: models pick among hidden real stations; recorded trips score the picks |
| `closed_station_ranking_check.py` | Closed-station ranking vs 2014 damage, compared like for like |
| `new_station_check.py` | Hold-out new stations (`--summarize-only` to report saved folds) |
| `new_station_check_combined.py` | Hold-out new stations for street speeds + LightGBM correction |
| `build_method_diagram.py` | Method diagram: data, travel time, busy engines, solver, validation |
| `compare_placement_training.py` | Training for first-engine trips from the home station, compared with production |
| `fetch_streetmanager_london.py` | Stream Street Manager roadworks archives, keep London works |
| `build_traffic_features.py` | DfT traffic counts and roadworks along each trip's route |
| `compare_traffic_features.py` | Do traffic and roadworks improve the travel time model? |
| `travel_time_error_breakdown.py` | Travel time error on first-engine trips, without the worst 5 %, and per route |
| `plot_weekly_report.py` | Slide pack under `data/figures/reports/` |

Typical order after a fresh download: build stations → graph → travel training → train model → capacity / busyness → planner / recommended → `embed_analysis_metrics_2d.py`.
