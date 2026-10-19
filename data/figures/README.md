# Figures and results

Every output the scripts write, grouped by part of the project.

```
data/figures/
├── analysis/station_embedding/   # all station metrics → 2D (PCA / UMAP / t-SNE)
├── data_exploration/             # EDA, busyness, capacity, incident density
├── travel_time/                  # area maps + model outlier panels
├── placement/                    # recommended layouts, scoring, tradeoffs, planner
├── validation/                   # 2014 closures, dispatch model
└── reports/                      # slide-ready weekly packs
```

| Folder | What | Written by |
|---|---|---|
| `analysis/station_embedding/` | Every joinable station metric condensed to 2D; archetypes + ANALYSIS.md | `scripts/embed_analysis_metrics_2d.py` |
| `data_exploration/london_eda/` | First look at LFB response data; street feature effects (SHAP, ablation) | `scripts/analyze_london.py` |
| `data_exploration/station_busyness/` | How often each station's own engine is out | `scripts/plot_station_busyness.py`, `build_live_day.py` |
| `data_exploration/station_capacity/` | Pumps per station vs demand | `scripts/analyze_station_capacity.py` |
| `data_exploration/london_incident_density.html` | Incident density map | `scripts/plot_incident_density.py` |
| `travel_time/maps/` | Predicted first-engine time across London | `scripts/plot_travel_time_maps.py` |
| `travel_time/outliers/` | Where the travel time model is most wrong | `scripts/train_travel_time_lfb.py` (outlier pass) |
| `placement/recommended/` | Recommended layouts (add / move / redesign; borough floors; station-count sweep) | `scripts/recommended_placement.py` |
| `placement/layout_scoring/` | Every layout scored under three travel time models, bootstrap intervals | `scripts/score_layouts.py` |
| `placement/robustness/` | Layouts by time of day, severity, borough and neighbourhood income | `scripts/placement_robustness.py` |
| `placement/tradeoff/`, `tradeoff_busy/` | The α curve (efficiency vs equal coverage), without / with busy engines | `scripts/placement_tradeoff.py` |
| `placement/slow_calls/` | Optimising the slowest 10 % of calls | `scripts/placement_slow_calls.py` |
| `placement/exact_runs/` | Early exact-solver runs (nearest engine free) | `scripts/plan_fire_station_locations.py` |
| `placement/planner_runs/` | Single planner runs from the command line; 3D race visual | `scripts/plan_fire_station_locations.py`, `build_fire_station_race_3d.py` |
| `validation/closure_2014/` | The 2014 closure test and its diagnostic | `scripts/closure_2014_test.py`, `closure_2014_diagnostic.py` |
| `validation/closure_2014_ranking/` | Earlier ranking of the closed stations | `scripts/validate_london_2014_closures.py`, `rank_closed_stations_2014.py` |
| `validation/dispatch/` | Dispatch model: which station sends the first engine | `scripts/fit_dispatch_model.py` |
| `validation/new_stations/` | Travel time accuracy for stations the model never saw | `scripts/new_station_check.py` |
| `reports/` | Slide-ready figures for progress reports | `scripts/plot_weekly_report.py` |

**Start here for the cross-metric picture:** [`analysis/station_embedding/ANALYSIS.md`](analysis/station_embedding/ANALYSIS.md) (also mirrored at [`docs/station_embedding_analysis.md`](../../docs/station_embedding_analysis.md)).
