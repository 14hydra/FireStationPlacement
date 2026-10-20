# Figures and results

Every output the scripts write, grouped by part of the project. The numbers that matter are
collected in [docs/results.md](../../docs/results.md).

| Folder | What | Written by |
|---|---|---|
| `data_exploration/london_eda/` | First look at LFB response data; street feature effects (SHAP, ablation) | `scripts/analyze_london.py` |
| `data_exploration/station_busyness/` | How often each station's own engine is out | `scripts/plot_station_busyness.py`, `build_live_day.py` |
| `data_exploration/station_capacity/` | Pumps per station vs demand | `scripts/analyze_station_capacity.py` |
| `data_exploration/london_incident_density.html` | Incident density map | `scripts/plot_incident_density.py` |
| `travel_time/maps/` | Predicted first-engine time across London | `scripts/plot_travel_time_maps.py` |
| `travel_time/outliers/` | Where the travel time model is most wrong | (outlier analysis) |
| `placement/recommended/` | Recommended layouts (add / move / redesign; borough floors; station-count sweep) | `scripts/recommended_placement.py` |
| `placement/layout_scoring/` | Every layout scored under three travel time models, bootstrap intervals | `scripts/score_layouts.py` |
| `placement/robustness/` | Layouts by time of day, severity, borough and neighbourhood income | `scripts/placement_robustness.py` |
| `placement/tradeoff/`, `tradeoff_busy/` | The α curve (efficiency vs equal coverage), without / with busy engines | `scripts/placement_tradeoff.py` |
| `placement/slow_calls/` | Optimising the slowest 10 % of calls | `scripts/placement_slow_calls.py` |
| `placement/station_sweep.png`, `validation/dispatch/rank_shares.png` | Saving vs number of stations; who sends the first engine | `scripts/plot_results_figures.py` |
| `placement/exact_runs/` | Early exact-solver runs (nearest engine free) | `scripts/plan_fire_station_locations.py` |
| `placement/planner_runs/` | Single planner runs from the command line; 3D race visual | `scripts/plan_fire_station_locations.py`, `build_fire_station_race_3d.py` |
| `validation/closure_2014/` | The 2014 closure test and its diagnostic | `scripts/closure_2014_test.py`, `closure_2014_diagnostic.py` |
| `validation/closure_2014_ranking/` | Earlier ranking of the closed stations | `scripts/validate_london_2014_closures.py`, `rank_closed_stations_2014.py` |
| `validation/dispatch/` | Dispatch model: which station sends the first engine | `scripts/fit_dispatch_model.py` |
| `validation/new_stations/` | Travel time accuracy for stations the model never saw | `scripts/new_station_check.py` |
| `reports/` | Slide-ready figures for progress reports | `scripts/plot_weekly_report.py` |
