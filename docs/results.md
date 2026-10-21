# Results and comparisons

Every comparison the project makes, in one place, with where each number comes from and
which figures show it. Update this file when a script is rerun. Last updated 2026-10-19.

**How to read the numbers**

- Times are first-engine **attendance** (turnout + drive) in seconds unless a table says
  "drive". Lower is better; a negative change is a saving.
- "2025 calls" = held-out calls the models never saw (trained or placed on 2021–2024 / 2024).
- Placement scores use the **learned dispatch model** (which station really responds) and
  busy engines, unless a table says otherwise. Modelled times run about 18 s below recorded
  (314 s modelled vs 332 s recorded for today's stations), so compare layouts with each
  other, not with recorded times.
- Each travel time model gives a different "today", so changes are always against today's
  stations under the same model.
- Intervals are 95 % paired bootstrap intervals (resampling calls).

---

## 1. Travel time model

### 1.1 Accuracy on 2025 trips (drive time)

Source: `data/processed/models/travel_time_lfb.metrics.json` (`scripts/train_travel_time_lfb.py`).

| Model | MAE (s) | Median error (s) | Bias (s) |
|---|---|---|---|
| Median of training trips | 111.9 | 85.0 | −36.9 |
| Straight line at 32 km/h | 95.2 | 72.1 | −65.8 |
| Road distance at 35 km/h | 80.8 | 54.8 | −3.5 |
| Kolesar formula, straight line | 71.4 | 48.9 | −16.9 |
| Kolesar formula, road distance | 68.7 | 45.8 | −16.1 |
| **LightGBM (street features)** | **57.6** | **35.3** | −17.7 |
| LightGBM, trips where the local engine was free | 50.0 | 30.7 | −17.2 |

### 1.2 Stations the model has never seen (new-location test)

Source: `data/figures/validation/new_stations/` (`scripts/new_station_check.py`; 20 groups
of about 5 stations, each held out once). Earlier 5-group version in the metrics file:
LightGBM 62.4 s vs Kolesar (road) 68.7 s.

All 20 groups (102 stations, 200,624 trips from 2025).

| Model | MAE (s) | Median error (s) | Bias (s) |
|---|---|---|---|
| LightGBM, station seen in training | 58.3 | 35.6 | −18.6 |
| **LightGBM, station never seen** | **62.3** | **41.2** | −11.5 |
| Kolesar, road distance (refit without the station) | 68.8 | 45.9 | −16.2 |

- A new site costs LightGBM about 4 s of accuracy (median per station 3.6 s, range −0.5
  to 22.4 s), and it still beats the Kolesar formula at 98 of 102 stations (not at Harold
  Hill, Wallington, Woodside, Eltham; all outer London, within 6 s).
- By trip length (MAE, seen / never seen / Kolesar): under 1 km 45 / 48 / 51 s; 2–3 km
  54 / 58 / 63 s; 4–6 km 73 / 77 / 88 s; over 6 km 104 / 109 / 126 s. LightGBM keeps its
  lead on the long trips that new sites far from today's stations would make.

Figure: `validation/new_stations/per_station.png`.

### 1.3 Trained before the 2014 closures, tested on 2014 trips after them

Source: `data/figures/validation/closure_2014/summary.json`.

| Model (trained 2009–2013) | MAE (s) | Median error (s) | Bias (s) |
|---|---|---|---|
| LightGBM | **64.8** | **38.3** | −30.4 |
| Kolesar, road distance | 73.4 | 45.8 | −30.4 |
| Kolesar, straight line | 74.8 | 48.3 | −30.1 |

### 1.4 Learned street speeds and the correction model

Source: `data/figures/travel_time/street_speeds/` (`scripts/train_street_speeds.py`).
A pace for every street segment from its characteristics, fitted so summed route times
match 2021–2024 trips, with routes re-chosen under the learned speeds each round (3
rounds). LightGBM then predicts only the correction (recorded drive minus street-speed
time). All LightGBM fits: same settings, one seed.

| Model (2025 trips) | MAE (s) | Median error (s) | Bias (s) |
|---|---|---|---|
| Kolesar, road distance | 68.7 | 45.8 | −16.1 |
| Fastest route at posted speed limits, calibrated to drive time (77.6 s + 0.78 × route time) | 68.6 | 45.2 | −18.5 |
| Street speeds alone | 65.9 | 42.5 | −17.8 |
| Street speeds + correction (time, season, turns, busy engines, call density only) | 62.9 | 40.4 | −18.7 |
| LightGBM alone | 57.8 | 35.5 | −17.7 |
| **Street speeds + correction (all features)** | **57.8** | **35.3** | −17.7 |

By trip length (MAE, s):

| Road km | Trips | Kolesar | Street speeds | LightGBM | Combined |
|---|---|---|---|---|---|
| < 1 | 17,336 | 50.4 | 49.4 | 44.8 | 44.8 |
| 1–2 | 51,525 | 55.1 | 53.8 | 47.7 | 47.6 |
| 2–3 | 54,690 | 63.0 | 60.7 | 53.5 | 53.4 |
| 3–4 | 36,913 | 73.9 | 70.5 | 60.8 | 60.7 |
| 4–6 | 30,900 | 88.2 | 83.1 | 72.5 | 72.3 |
| 6+ | 9,260 | 125.8 | 119.3 | 103.3 | 104.5 |

Learned-speed routes beat the fastest route at posted speed limits (calibrated the same robust way on 2021–2024) by 2.7 s overall and by 9.7 s on trips over 6 km (119.3 against 129.0 s). The combined model is as accurate as LightGBM alone; on the longest trips it is about 1 s
less accurate but slightly less biased (−8.9 vs −10.8 s), so on 2025 trips the street-speed
base does not yet show the hoped-for advantage on long trips.

Learned street effects (95 % bootstrap intervals, resampling trips):

| Road type | Pace (s/km) | Speed |
|---|---|---|
| Motorway / trunk | 50.1 [49.8, 50.4] | 72 km/h |
| A road | 52.4 [52.2, 52.7] | 69 km/h |
| B road | 58.2 [57.7, 58.5] | 62 km/h |
| Local distributor | 63.1 [63.0, 63.5] | 57 km/h |
| Residential | 71.2 [71.1, 71.8] | 51 km/h |
| Minor / unclassified | 88.4 [87.3, 89.5] | 41 km/h |

| Street feature | Effect |
|---|---|
| 20 mph limit | +8.7 s/km [8.6, 8.9] |
| One-way | +2.1 s/km [1.8, 2.3] |
| Bridge / tunnel | +1.2 s/km [0.8, 1.9] |
| Bus lane | −2.4 s/km [−2.6, −1.7] |
| 2+ lanes | −3.2 s/km [−3.4, −3.0] |
| Each traffic calming feature | +1.03 s [1.01, 1.07] |
| Each pedestrian crossing | +0.62 s [0.56, 0.64] |
| Each junction | +0.58 s [0.56, 0.60] |
| Each traffic signal | +0.56 s [0.55, 0.57] |
| Each roundabout | +0.03 s [−0.03, 0.09] |
| Each give-way / stop sign | −0.48 s [−0.59, −0.44] |
| Fixed start-up time per trip | 70 s |

These describe what the model learned, not cause and effect (a 20 mph street may be slow
because it is narrow and lined with parked cars). The negative give-way effect is likely
because give-way signs mark side-street exits onto main roads, where engines have priority.

Figures: `travel_time/street_speeds/coefficients.png`, `speed_map.png`.

Figures: `reports/weekly_2026-10-19/1_travel_time_accuracy.png`,
`data_exploration/london_eda/07_model_comparison.png`, `08_street_shap.png`,
`09_street_ablation.png`, `10_street_effects_per_km.png`, `travel_time/outliers/`.

---

## 2. Which station responds (busy engines and dispatch)

Source: `data/figures/validation/dispatch/` (`scripts/fit_dispatch_model.py`). Trained on
2021–2024 first engines, tested on 2025.

### 2.1 Share of first engines by rank (fastest open station = 1)

| Rank | Recorded 2025 | Dispatch model | Nearest, or next if busy |
|---|---|---|---|
| 1st | 69.6 % | 71.5 % | 72.3 % |
| 2nd | 18.1 % | 17.7 % | 27.7 % |
| 3rd | 6.3 % | 6.0 % | 0 % |
| 4th | 2.9 % | 2.6 % | 0 % |
| 5th | 1.2 % | 1.1 % | 0 % |

### 2.2 Expected drive per demand cell vs recorded (2025, cells with 20+ calls)

| Response model | Cell MAE (s) | Bias (s) | Correlation |
|---|---|---|---|
| Nearest station always | 47.8 | −46.6 | 0.883 |
| Nearest, or next if busy (Erlang) | 25.4 | −19.3 | 0.878 |
| **Dispatch model** | **23.4** | −16.5 | **0.892** |

### 2.3 Time of day: share sent by the nearest station (2025)

| Slot | Recorded | Dispatch, by hour (calibrated) | Dispatch, all day |
|---|---|---|---|
| 00–04 | 77.8 % | 78.9 % | 71.0 % |
| 04–08 | 79.6 % | 80.5 % | 71.5 % |
| 08–12 | 68.2 % | 70.4 % | 71.5 % |
| 12–16 | 60.5 % | 63.5 % | 71.6 % |
| 16–20 | 68.5 % | 71.0 % | 71.6 % |
| 20–24 | 72.7 % | 74.0 % | 71.3 % |

The nearest station is least available in the early afternoon (training, fire-safety
visits), not at the evening call peak.

Busy engines alone: today's modelled average rises from 284 s (all engines free) to 312 s
(Erlang busy model) and 314 s (dispatch model), against 332 s recorded.

Figures: `validation/dispatch/rank_shares.png`, `reports/weekly_2026-10-19/5_dispatch_time_of_day.png`,
`data_exploration/station_busyness/`.

---

## 3. The 2014 closures (main real-world test)

Source: `data/figures/validation/closure_2014/` (`closure_2014_test.py`, `closure_2014_diagnostic.py`)
and `validation/dispatch/summary.json` (part B). Ten stations closed on 9 January 2014.
Effect = change in first-engine attendance in the closed stations' former areas, 2014 vs
2013, minus the change in the rest of London.

### 3.1 London-wide effect across the 10 areas

| | Effect (s) |
|---|---|
| **Recorded** | **+70.5** |
| Kolesar, road distance (nearest engine free) | +71.0 |
| LightGBM (nearest engine free) | +76.8 |
| Kolesar, straight line (nearest engine free) | +78.3 |
| LightGBM (busy engines) | +67.5 |

### 3.2 Which areas were hit hardest: correlation across the 10 areas

| What the model is given | LightGBM | Kolesar road | Kolesar straight |
|---|---|---|---|
| Nearest open station responds | 0.29 (attendance) / 0.42 (drive) | 0.08 | 0.16 |
| The station that actually responded (drive) | **0.88** | 0.84 | 0.75 |

| Response model (LightGBM drive, trained 2009–13) | Correlation | Area MAE (s) |
|---|---|---|
| Nearest station | 0.42 | 17.2 |
| Nearest, or next if busy | 0.31 | 16.3 |
| **Dispatch model** | **0.56** | **13.1** |

Reading: the travel time model gets the drive right when it knows who responded; the
remaining error is dispatch (after the closures the nearest station sent the first engine
on as few as 38 % of calls in some areas).

Figures: `reports/weekly_2026-10-19/4_closure_2014.png`.

---

## 4. Choosing the objective

### 4.1 Efficiency vs equal coverage (the α dial)

Source: `data/figures/placement/tradeoff_busy/tradeoff.csv` (busy engines, 2024 demand, all
calls). E = call-weighted average, Q = area average (each area once). Scores: 0 = today,
100 = best possible for that measure.

| Scenario | α | E (s) | Q (s) | Efficiency score | Equity score | Beats today on both |
|---|---|---|---|---|---|---|
| Today | – | 309.5 | 354.0 | 0 | 0 | – |
| Add 3 | 1 | 305.7 | 353.2 | 100 | 13 | yes |
| Add 3 | 0.5 | 307.4 | 348.6 | 55 | 86 | yes |
| Add 3 | 0 | 308.4 | 347.7 | 29 | 100 | yes |
| Move up to 10 | 1 | 304.6 | 353.0 | 100 | 8 | yes |
| Move up to 10 | 0.5 | 307.9 | 345.1 | 33 | 67 | yes |
| Move up to 10 | 0 | 315.3 | 340.8 | −118 | 100 | no |
| Full redesign | 1 | 304.3 | 358.6 | 100 | −26 | no |
| Full redesign | 0.5 | 307.6 | 344.3 | 37 | 55 | yes |
| Full redesign | 0 | 333.2 | 336.2 | −456 | 100 | no |

Same pattern for serious calls and fires in homes (full table in the CSV). Without busy
engines (`placement/tradeoff/`), every α from 0.25 to 1 beats today on both measures.

### 4.2 Average vs slowest calls

Source: `data/figures/placement/slow_calls/slow_calls.csv` (nearest engine free, 2024 demand).

| Scenario | Objective | Average (s) | Slowest 10 % (s) |
|---|---|---|---|
| Today | – | 281.9 | 406.7 |
| Move up to 10 | minimise average | 276.3 | 405.8 |
| Move up to 10 | minimise slowest 10 % | 284.2 | 394.9 |
| Full redesign | minimise average | 268.6 | 403.0 |
| Full redesign | minimise slowest 10 % | 287.3 | 398.6 |

Choice made: minimise the average (α = 1) with floors (slowest 10 %, the average and every
borough no worse than today), which keeps most of the efficiency gain without anyone losing.

Figures: `reports/weekly_2026-10-19/2_price_of_fairness.png`, `placement/tradeoff_busy/tradeoff.png`.

---

## 5. Recommended layouts

Source: `data/figures/placement/recommended/<scenario>/summary.json`. Rules: α = 1, busy
engines, slowest 10 % and average no worse than today, 10-minute cap, pumps move with
stations (142 in total), street refinement, every borough within the stated tolerance.
"Planning" columns are on 2024 demand with the solver's busy model; the last column is
2025 calls.

| Scenario | Borough tolerance | Average (s) | Slowest 10 % (s) | Area avg (s) | Eff. score | Equity score | Within 6 min | 2025 avg (s) | Solver |
|---|---|---|---|---|---|---|---|---|---|
| Today | – | 309.5 | 423.2 | 354.0 | 0 | 0 | 81.6 % | 311.7 | – |
| Add 1 | 1 s | 308.1 | 423.0 | 353.8 | 103 | 11 | 82.5 % | 310.5 | exact |
| Add 2 | 1 s | 307.5 | 423.0 | 353.7 | 87 | 10 | 82.5 % | 310.0 | exact |
| Add 3 | 1 s | 306.4 | 423.1 | 353.4 | 98 | 13 | 82.5 % | 308.9 | exact |
| Add 5 | 1 s | 304.5 | 421.5 | 351.9 | 106 | 30 | 83.5 % | 307.0 | exact |
| Move up to 5 | 1 s | 306.7 | 420.5 | 351.0 | 79 | 36 | 82.4 % | 308.9 | GA + swaps (fallback) |
| **Move up to 10** | **1 s** | **304.0** | **420.2** | **349.7** | **98** | **33** | **83.8 %** | **306.4** | exact |
| Move up to 10 | 0 s | 305.6 | 420.9 | 351.2 | 69 | 22 | 82.7 % | 308.0 | GA + swaps (fallback) |
| Move up to 10 | 5 s | 303.4 | 420.0 | 349.8 | 109 | 32 | 84.3 % | 306.2 | exact |
| Move up to 10 | none | 302.9 | 420.9 | 351.3 | 117 | 21 | 83.6 % | 305.7 | exact |
| Full redesign | 1 s | 293.3 | 409.5 | 338.0 | 144 | 73 | 86.8 % | 296.1 | GA + swaps (fallback) |
| Full redesign | none | 290.5 | 406.8 | 337.0 | 170 | 76 | 88.4 % | 293.7 | exact |

Scores above 100 come from street refinement after the solve. "GA + swaps" means the exact
solver found no layout within its time limit and the genetic algorithm plus swap search
supplied one, so it is not proven optimal.

Figures: per scenario `01_metrics_comparison.png`, `03_response_surface_map.png`,
`04_relocations.png`, `fire_station_plan_map.html`.

---

## 6. Do the layouts hold up under other travel time models?

Source: `data/figures/placement/layout_scoring/scores.csv` (`scripts/score_layouts.py`).
2025 calls, dispatch model, 1000 bootstrap resamples. Today: LightGBM 314.4 s, Kolesar
road 324.6 s, Kolesar straight 330.5 s.

Change in average (s); a layout "wins" if the interval is below 0 under all three models.

| Layout | LightGBM | Kolesar road | Kolesar straight | Wins under all 3 | Slowest 10 % no worse under all 3 |
|---|---|---|---|---|---|
| Add 1 (1 s) | −1.3 | −0.9 | −0.5 | yes | yes |
| Add 2 (1 s) | −1.4 | −1.0 | −0.7 | yes | no (+0.0) |
| Add 3 (1 s) | −2.3 | −1.8 | −1.3 | yes | no (+0.1) |
| Add 5 (1 s) | −3.8 | −3.2 | −2.4 | yes | yes |
| Move up to 5 (1 s) | −2.3 | −2.4 | −1.5 | yes | yes |
| **Move up to 10 (1 s)** | **−5.5** | **−4.7** | **−2.5** | **yes** | **yes** |
| Full redesign (1 s) | −13.8 | −10.6 | −5.1 | yes | yes |
| Full redesign (no borough rule) | −16.4 | −12.7 | −5.4 | yes | yes |
| Add 3, sites chosen by Kolesar road | −2.0 | −2.2 | −1.6 | yes | no |
| Move 10, sites chosen by Kolesar road | −3.1 | −5.2 | −3.2 | yes | no (+3.2) |
| Move 10, sites chosen by Kolesar straight | −2.9 | −3.7 | −4.2 | yes | yes |

Move up to 10 (1 s), slowest 10 %: LightGBM −3.8 [−4.1, −3.6], Kolesar road −4.9
[−5.3, −4.5], Kolesar straight −3.6 [−4.0, −3.2].

Every LightGBM-chosen layout beats today under all three models. Gains are largest under
LightGBM (the model that chose them) and smallest under the straight-line formula, so the
spread between models is the honest uncertainty on the size of the gain.

Figures: `reports/weekly_2026-10-19/3_three_models.png`, `placement/station_sweep.png`.

---

## 7. Robustness: time of day, severity, borough, income

Source: `data/figures/placement/robustness/` (`scripts/placement_robustness.py`). 2025
calls; LightGBM rows use travel times for each period's hour and the hourly dispatch
model; Kolesar rows have no hour. 500 bootstrap resamples (100 for boroughs).

### 7.1 Time of day (LightGBM): change in average, [95 % interval]

| Period | Calls | Today (s) | Add 3 | Move up to 10 | Full redesign |
|---|---|---|---|---|---|
| Night 00–07 | 21,939 | 288.1 | −1.8 [−2.0, −1.7] | −4.7 [−5.0, −4.4] | −12.2 [−12.7, −11.7] |
| Morning rush 07–10 | 12,320 | 318.5 | −2.6 [−2.9, −2.4] | −4.6 [−5.1, −4.2] | −13.8 [−14.5, −13.1] |
| Day 10–16 | 38,789 | 328.8 | −2.4 [−2.6, −2.3] | −5.6 [−5.8, −5.4] | −16.1 [−16.5, −15.7] |
| Evening rush 16–20 | 30,027 | 326.5 | −2.0 [−2.1, −1.8] | −5.4 [−5.7, −5.1] | −15.1 [−15.6, −14.5] |
| Evening 20–24 | 26,193 | 297.7 | −2.0 [−2.2, −1.9] | −5.0 [−5.3, −4.7] | −13.1 [−13.6, −12.6] |
| All day | 129,268 | 314.1 | −2.2 | −5.2 | −14.3 |

Slowest 10 % by period: move up to 10 and the redesign improve in every period (−3.3 to
−5.6 s and −12.6 to −16.1 s). Add 3 makes the slowest 10 % about 0.2–0.3 s worse at night
and in both rush hours (the floor holds all day, not in every period).

### 7.2 Call severity: change in average (s)

| Calls | Add 3 (LGBM / road / straight) | Move up to 10 | Full redesign |
|---|---|---|---|
| All (129,268) | −2.2 / −1.8 / −1.3 | −5.2 / −4.7 / −2.5 | −14.3 / −10.6 / −5.1 |
| Fires and special services (68,263) | −1.6 / −1.4 / −1.0 | −4.7 / −4.3 / −2.2 | −13.9 / −10.0 / −4.4 |
| Fires in homes (4,878) | −1.6 / −1.4 / −1.1 | −5.5 / −5.2 / −3.4 | −14.9 / −11.2 / −5.9 |

Every layout is faster for every severity group under all three models.

### 7.3 Neighbourhood income: change in average (s)

Each call takes the ONS FYE 2020 equivalised household income after housing costs of its
nearest London MSOA (about 980 neighbourhoods, split into fifths).

| Income fifth | Add 3 (LGBM / road / straight) | Move up to 10 | Full redesign |
|---|---|---|---|
| Q1 (poorest) | −1.5 / −1.4 / −1.2 | **−9.6 / −9.3 / −5.9** | **−19.2 / −16.4 / −9.7** |
| Q2 | −1.7 / −1.5 / −1.4 | −4.1 / −3.8 / −2.9 | −15.6 / −11.9 / −7.8 |
| Q3 | −1.3 / −1.3 / −1.0 | −3.1 / −2.3 / −0.9 | −14.3 / −8.9 / −4.6 |
| Q4 | −2.0 / −1.4 / −0.5 | −2.8 / −2.1 / 0.0 | −9.8 / −6.2 / −0.1 |
| Q5 (richest) | −4.7 / −3.8 / −2.4 | −5.6 / −5.4 / −1.8 | −11.2 / −7.9 / −1.2 |

Moving stations helps the poorest fifth most, under every model. Adding stations helps the
richest fifth most (new sites go to central London, which is both busy and wealthy).

### 7.4 Boroughs

With a 1 s borough tolerance, under LightGBM:

- Add 3: 11 boroughs better, 11 slightly worse; worst Barnet +1.0 s.
- Move up to 10: 28 better; worst City of London +2.6 s (few calls).
- Full redesign: all 33 better.

Under the Kolesar formulas some boroughs get worse. In move up to 10: City of London up to
+8.8 s, Richmond +5.1 s, Camden +3.1 s. In the redesign: Tower Hamlets up to +11.8 s and
Lewisham +10.5 s (straight-line). The borough rule is enforced with the model that chose the
layout, so a borough's protection is only as good as that model's map of it.

Figures: `placement/robustness/time_of_day.png`, `income.png`, `boroughs.png`.

---

## 8. Figures

Have (path under `data/figures/`) and still needed for the board / paper.

| # | Figure | Shows | Status |
|---|---|---|---|
| 1 | `reports/weekly_2026-10-19/1_travel_time_accuracy.png` | LightGBM vs baselines (MAE) | have |
| 2 | `data_exploration/london_eda/08_street_shap.png`, `10_street_effects_per_km.png` | Which street features matter | have |
| 3 | `validation/new_stations/per_station.png` | Accuracy at stations never seen | have |
| 4 | `validation/dispatch/rank_shares.png` | Who sends the first engine: model vs recorded | have |
| 5 | `reports/weekly_2026-10-19/5_dispatch_time_of_day.png` | Nearest-station availability by hour | have |
| 6 | `reports/weekly_2026-10-19/4_closure_2014.png` | 2014 closures: recorded vs predicted | have |
| 7 | `reports/weekly_2026-10-19/2_price_of_fairness.png` | α trade-off | have |
| 8 | `placement/recommended/replace_10_boroughs1s/04_relocations.png` | Map of the recommended moves | have (draft); needs a poster-quality map with street names |
| 9 | `placement/recommended/replace_10_boroughs1s/03_response_surface_map.png` | Response time map before/after | have (draft) |
| 10 | `reports/weekly_2026-10-19/3_three_models.png` | Gain under three models | have; add the sweep layouts |
| 11 | `placement/station_sweep.png` | Saving vs number of stations | have |
| 12 | `placement/robustness/time_of_day.png` | Gain by time of day | have |
| 13 | `placement/robustness/income.png` | Who gains, by income | have |
| 14 | `placement/robustness/boroughs.png` | Change by borough, three models | have |
| 15 | Method diagram | Data → travel time → dispatch/busy → solver → validation | **needed** |
| 16 | `travel_time/street_speeds/speed_map.png`, `coefficients.png` | Learned fire engine speed per street; which features slow engines | have |

---

## 9. Limitations to state

- Modelled attendance is about 18 s below recorded (314 vs 332 s); comparisons between
  layouts are fair, absolute times are optimistic.
- Gains depend on the travel time model: under the straight-line formula they are about
  half the LightGBM gain. Borough-level protection holds under LightGBM, not always under
  the Kolesar formulas.
- Some layouts (move up to 5, move up to 10 with 0 s borough tolerance, the borough-limited
  redesign) come from the genetic algorithm fallback and are not proven optimal.
- The dispatch model cannot see LFB's own moves of engines between stations (after the
  2014 closures, the likely cause of the remaining error).
- Demand is past calls (2024), not predicted future calls; land, cost and staffing are not
  modelled.
