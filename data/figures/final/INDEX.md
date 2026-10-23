# Final figures (ISEF board pack)

Ten figures that tell the whole project. Open **01 → 10** in order.

| # | File | At a glance |
|---|---|---|
| 01 | `01_travel_time_accuracy.png` | Street-aware trees beat the classic formula — **58.9 s** vs Kolesar **68.7 s**. |
| 02 | `02_street_features.png` | Top street traits that slow engines (plain labels, top 8). |
| 03 | `03_first_engine_map.png` | Predicted first-engine time across London (single map). |
| 04 | `04_station_busyness.png` | Busiest stations — engines out most often. |
| 05 | `05_capacity_mismatches.png` | Move pumps: busy 1-engine ↔ quiet 2-engine (same scale). |
| 06 | `06_station_types.png` | Station types + what that means for the model and resize. |
| 07 | `07_layouts_beat_today.png` | Move-10 fair layout wins under every travel-time model. |
| 08 | `08_price_of_fairness.png` | Stricter borough rules → smaller citywide gain. |
| 09 | `09_closure_2014.png` | 2014 closures: nearest vs learned dispatch vs oracle. |
| 10 | `10_results_summary.png` | Plain-English results on one page. |

Regenerate:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_clear_final_figures.py
```

Upstream data still comes from weekly / embedding / busyness / capacity / map scripts.
