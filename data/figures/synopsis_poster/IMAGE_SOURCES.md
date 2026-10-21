# Synopsis poster — image sources

Poster files (this folder):

| File | What |
|---|---|
| `synopsis_poster.png` | High-res composite (images + empty text slots) |
| `synopsis_poster.html` | Same layout, print/edit in a browser |

All artwork is copied from the **final ISEF figure pack**. Refresh that pack first if figures change:

```bash
.venv/bin/python scripts/build_final_figures.py
.venv/bin/python scripts/build_synopsis_poster.py
```

## Figures used (repo paths)

| Slot on poster | File in `data/figures/final/` | Upstream source (under `data/figures/`) |
|---|---|---|
| Top-left (accuracy) | `01_travel_time_accuracy.png` | `reports/weekly_2026-10-19/1_travel_time_accuracy.png` |
| Below accuracy (streets) | `02_street_features.png` | `data_exploration/london_eda/08_street_shap.png` |
| Hero map | `03_first_engine_map.png` | `travel_time/maps/travel_time_maps.png` |
| Mid row left | `04_station_busyness.png` | `data_exploration/station_busyness/station_busyness.png` |
| Mid row centre | `05_capacity_mismatches.png` | `data_exploration/station_capacity/05_capacity_mismatches.png` |
| Mid row right | `06_station_types.png` | `analysis/station_embedding/06_isef_story.png` |
| Bottom 1 | `07_layouts_beat_today.png` | `reports/weekly_2026-10-19/3_three_models.png` |
| Bottom 2 | `08_price_of_fairness.png` | `reports/weekly_2026-10-19/2_price_of_fairness.png` |
| Bottom 3 | `09_closure_2014.png` | `reports/weekly_2026-10-19/4_closure_2014.png` |
| Bottom 4 | `10_results_summary.png` | `reports/weekly_2026-10-19/6_summary_table.png` |

Story index: [`data/figures/final/INDEX.md`](../final/INDEX.md).

## Text slots (empty on purpose)

Dashed grey boxes are for later copy — title, authors, and a short takeaway under each figure. Nothing is filled in yet.
