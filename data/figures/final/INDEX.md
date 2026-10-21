# Final figures (ISEF board pack)

Ten figures that tell the whole project. **Open `00_roadmap.png` first**, then 01→10.

| # | File | At a glance |
|---|---|---|
| 01 | `01_travel_time_accuracy.png` | **Our travel-time model beats the classic formula** — LightGBM error 57.6 s vs Kolesar 68.7 s on held-out 2025 trips. |
| 02 | `02_street_features.png` | **Which street traits slow fire engines** — SHAP ranking of route features the model uses (signals, junctions, road class). |
| 03 | `03_first_engine_map.png` | **Predicted first-engine time across London** — City map of how long the nearest engine takes to arrive under our model. |
| 04 | `04_station_busyness.png` | **Central stations’ engines are out most often** — Utilisation by station — inner London is stretched; outer London is quieter. |
| 05 | `05_capacity_mismatches.png` | **Where to add or free up an engine** — Busy 1-pump houses (Euston…) vs quieter 2-pump houses (Addington…). |
| 06 | `06_station_types.png` | **102 stations → five types at a glance** — Map + busy×capacity picture + what it means for the model and resize. |
| 07 | `07_layouts_beat_today.png` | **Better layouts still win under all three scorers** — Seconds saved vs today’s stations under LightGBM, Kolesar road, Kolesar crow. |
| 08 | `08_price_of_fairness.png` | **Efficiency vs equal coverage** — The α tradeoff curve — how much average time you give up for fairness. |
| 09 | `09_closure_2014.png` | **2014 ten-station closure test** — Nearest vs learned dispatch vs oracle — which predicts the real 2014 slowdown. |
| 10 | `10_results_summary.png` | **Numbers on one page** — Compact results table for the board / abstract. |

## What to say (30-second version)

01. Street-aware trees beat the Kolesar distance→time curve on real 2025 calls.
02. The model is not a black box — these street features drive its predictions.
03. This is what we optimise when we move or add stations.
04. Capacity and demand are uneven — placement alone is not the whole story.
05. A fixed pump budget can be reallocated along this mismatch.
06. One poster that ties busyness, capacity, and model hard cases together.
07. Placement gains are not an artifact of one travel-time model.
08. We can choose how hard to equalise boroughs; the cost is explicit.
09. Dispatch learning closes much of the gap between ‘assume nearest’ and the oracle.
10. If a judge only has 30 seconds, show this after figure 01 and 07.

## Sources

These are copies of figures elsewhere under `data/figures/`. Regenerate with:

```bash
.venv/bin/python scripts/build_final_figures.py
```

Related: `analysis/station_embedding/ISEF_GUIDE.md`, `docs/model_improvements_from_embedding.md`.
