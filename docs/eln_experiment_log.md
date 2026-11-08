# Electronic Lab Notebook — Experiment Log (filled)

**Name:** Sarp Akalin  
**Project:** Optimizing Fire Station Placement Based on Street Characteristics Using a Decision Tree Travel Time Model  
**Repo write-ups:** https://github.com/14hydra/EMVRouteOptimization · results: `docs/results.md`

Paste the table below into the **Experiment log** section of the ASI ELN (replace / extend the existing rows). Embed the GitHub or `docs/…` link in the “link” word as required.

| Date | Experiment Title | Link to Write-up |
|---|---|---|
| 9/10/2026 | CAD travel time model (early baseline) | [link](https://github.com/14hydra/EMVRouteOptimization) |
| 9/11/2026 | All three main routing models and improvements to time model | [link](https://github.com/14hydra/EMVRouteOptimization) |
| 9/11/2026 | Route ETA | [link](https://github.com/14hydra/EMVRouteOptimization) |
| 9/13/2026 | Google Maps control | [link](https://github.com/14hydra/EMVRouteOptimization) |
| 9/15/2026 | SF data exploration | [link](https://github.com/14hydra/EMVRouteOptimization) |
| 9/18/2026 | All three routes and race visualization | [link](https://github.com/14hydra/EMVRouteOptimization) |
| 9/27/2026 | Hybrid / improved travel-time prediction (MAE push) | [link](https://github.com/14hydra/EMVRouteOptimization) |
| 9/28/2026 | Multi-city probe (Paris) + travel-time MAE ≈ 64 s | [link](https://github.com/14hydra/EMVRouteOptimization) |
| 9/29/2026 | Further travel-time model improvements | [link](https://github.com/14hydra/EMVRouteOptimization) |
| 9/30/2026 | Firehouse placing model (first placement prototype) | [link](https://github.com/14hydra/EMVRouteOptimization) |
| 10/4/2026 | Switch study city to London Fire Brigade (LFB) data | [link](docs/datasets.md) |
| 10/4/2026 | Project overview + LFB planner incident extract | [link](PROJECT_OVERVIEW.md) |
| 10/5/2026 | LightGBM travel-time model (no station IDs); real station origins + road distance | [link](docs/travel_time_training.md) |
| 10/5/2026 | Firehouse new-vs-old 3D race (London-only router) | [link](https://github.com/14hydra/EMVRouteOptimization) |
| 10/12/2026 | Tuned LightGBM (~60 s MAE all trips) + improved race visuals | [link](docs/results.md) |
| 10/14/2026 | Kolesar baselines vs LightGBM; station busyness figures; outlier checks | [link](docs/results.md) |
| 10/16/2026 | LightGBM as placement scorer; p-median / HiGHS placement solver; street-level candidates | [link](docs/fire_station_location.md) |
| 10/18/2026 | Placement for slow calls (CVaR) + equity floors (average / slowest 10% / borough) | [link](docs/results.md) |
| 10/18/2026 | 2014 ten-station closure natural experiment (pre-2014 train → post-closure test) | [link](docs/results.md) |
| 10/19/2026 | Busy-engine (Erlang) + learned dispatch model (which of 8 nearest responds) | [link](docs/results.md) |
| 10/19/2026 | Dispatch time-of-day model; placement robustness (TOD / severity / station-count sweep) | [link](docs/results.md) |
| 10/21/2026 | Learned street speeds + LightGBM correction (route / segment model) | [link](docs/results.md) |
| 11/1/2026 | Extra travel-time holdout / ablation tests | [link](docs/results.md) |
| 11/2/2026 | Neutral siting test (hidden real stations; LightGBM vs Kolesar vs random) | [link](docs/results.md) |
| 11/4/2026 | First-engine “gap” analysis (local station free but another arrives first) | [link](docs/results.md) |
| 11/4/2026 | Coverage heatmaps (before / after / Δ / demand) for move-10 layouts | [link](docs/results.md) |
| 11/5/2026 | Part 2 targeted placement: slowest 5 boroughs; pre-registered pass marks | [link](docs/targeted_areas_rule.md) |
| 11/5/2026 | Citywide vs targeted move-10/20; Rule B land sites; three-model scoring | [link](docs/results.md) |
| 11/5/2026 | Locked 2026 final test (prediction written first; Jan–Jul 2026 calls) | [link](docs/prediction_2026.md) |
| 11/6/2026 | Megacity transfer runs (top-10 agglomerations; Kolesar + proxy demand) + map viewer | [link](docs/megacities.md) |

## Short notes for write-ups (optional bullets under each log row)

- **LightGBM travel time:** train 2021–24, test 2025; street features on OSM shortest routes; first-engine MAE **45.5 s** vs Kolesar road **~54–69 s** (see `docs/results.md` §1).
- **Dispatch + busy:** nearest-free assumption understates real attendance; dispatch model raises 2014 area correlation to **~0.59**.
- **Placement Move 10:** citywide ~**5–6 s**; targeted Rule B Move 10 passes on 2025/2026 under all three scorers (LightGBM, road Kolesar, crow Kolesar).
- **Neutral test:** street model barely beats distance for *choosing* sites (&lt;1 s); any model beats random by 3–4 s.
- **Megacities:** transfer/proxy only — not LFB-grade validation.

## How to paste into Google Docs

1. Open the ELN Google Doc (not only the PDF export).
2. Select the Experiment log table → insert rows as needed.
3. Copy columns Date / Experiment Title / Link from this file (or from `docs/eln_experiment_log.tsv`).
4. For each Link cell: Insert → Link → paste the URL and display text `link`.
