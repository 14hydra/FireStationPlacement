# Final figures (ISEF board pack)

Ten figures that tell the whole project. Open **01 → 10** in order.

| # | File | At a glance |
|---|---|---|
| 01 | `01_travel_time_accuracy.png` | Street-aware trees beat the classic distance→time formula. |
| 02 | `02_street_features.png` | Top street traits that slow engines (plain labels, top 8). |
| 03 | `03_first_engine_map.png` | Predicted first-engine time across London (single map). |
| 04 | `04_station_busyness.png` | Busiest stations — engines out most often. |
| 05 | `05_capacity_mismatches.png` | Move pumps: busy 1-engine ↔ quiet 2-engine (same scale). |
| 06 | `06_station_types.png` | Station types + what that means for the model and resize. |
| 07 | `07_coverage_heatmaps.png` | **Before / after / Δ / demand** — coverage change plus call-weight so opens/closes are demand-justified (dot size ∝ demand). |
| 07b | `07b_move_rationale.png` | Per-station local demand + response Δ with plain-language why each site opens or closes. |
| 08 | `08_price_of_fairness.png` | Stricter borough rules → smaller citywide gain. |
| 09 | `09_closure_2014.png` | 2014 closures: nearest vs learned dispatch vs oracle. |
| 10 | `10_results_summary.png` | Plain-English results on one page. |

Also `method_diagram.png`: how the project works, from data to validation (`scripts/build_method_diagram.py`).

| 14 | `14_project_significance.png` | **Significance:** Part 2 2026 PASS (−13.7 s slowest boroughs, 0 boroughs worse), equity (−9.6 s poorest fifth), NYC FDNY weak-label check (r ≈ 0.44 on fires). |

Regenerate:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_clear_final_figures.py
```

Upstream data still comes from weekly / embedding / busyness / capacity / map scripts.
