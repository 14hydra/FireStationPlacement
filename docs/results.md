# Results and comparisons

Every comparison the project makes, in one place, with where each number comes from and
which figures show it. Update this file when a script is rerun. Last updated 2026-11-04.

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

### 1.1b Error on first-engine trips from the home station, and per route (production model, 2025)

Source: `data/figures/validation/travel_time_error/summary.json` (`scripts/travel_time_error_breakdown.py`). Placement scores the first engine, from its home station, with the local engine not busy, so 45.5 s is the error that applies to it.

| Trips | n | MAE (s) | Median error (s) | Bias (s) | MAE without each slice's worst 5 % (s) |
|---|---|---:|---:|---:|---:|
| All 2025 trips | 200,624 | 57.6 | 35.3 | −17.7 | 44.4 |
| **First engine, home station, not busy (placement)** | 87,937 | **45.5** | 29.4 | −14.5 | 35.2 |
| The same, exact address only | 32,155 | 50.3 | 31.2 | −17.0 | 38.3 |
| Every other trip | 112,687 | 67.1 | 41.3 | −20.2 | 52.3 |

On the same first-engine trips the classic formula (road distance) has 53.8 s and the never-seen-stations test 50.3 s. Figures 01 and 10 show these numbers.

The worst 5 % are chosen by their error, so that column describes the error's long tail and is not an accuracy claim. They carry 26–28 % of the error in each slice.

Per route (one station to one spot of about 100 m; first-engine trips to exact addresses). The model predicts a typical (median) time, so the median comparison is the fair one; recorded means sit higher because of slow outliers.

| Routes with at least | Routes | Trips | Single-trip MAE (s) | Route median vs recorded median, MAE (s) | Bias (s) | Route mean vs recorded mean, MAE (s) | Bias (s) | Sampling noise in recorded mean (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 trips | 847 | 6,458 | 42.9 | 20.0 | −2.0 | 23.3 | −10.8 | 17.4 |
| 10 trips | 149 | 2,256 | 43.7 | 14.3 | −1.8 | 18.7 | −13.3 | 13.6 |
| 20 trips | 20 | 647 | 42.3 | 10.7 | +5.0 | 13.3 | −7.6 | 9.3 |

Most of the single-trip error is variation between trips on the same route (39–42 s around the route's own median), and the recorded averages are themselves noisy, so the route figures partly reflect that noise and likely overstate the model's error against a route's true average. Only 20 routes have 20+ such trips, and repeat spots are not typical of London. As an out-of-sample check, predicting 2025 trips from the route's own 2021–24 median gives 47.1 s against the model's 44.0 s on the same 13,165 trips (48.9 s against 45.4 s when also matching time of day, 3,844 trips).

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

### 1.3b Embedding-informed changes, tested like for like

Source: `data/figures/analysis/model_improvement/fair_comparison.{csv,json}`
(`scripts/compare_travel_time_variants.py`). Production settings (2,110 trees, same seed),
2021–2024 → 2025, MAE in seconds.

| Variant | All | Dense short | Long sparse | Over 6 km |
|---|---|---|---|---|
| Production | **57.80** | **49.40** | 78.78 | **103.30** |
| + interaction features | 57.85 | 49.43 | 78.83 | 103.55 |
| + sample weights | 57.82 | 49.41 | 78.70 | 103.69 |
| + both | 57.82 | 49.42 | 78.64 | 103.37 |
| + both + long-sparse correction (fitted in sample) | 57.80 | 49.42 | 78.51 | 103.30 |
| + both + long-sparse correction (fitted on unseen 2024 trips) | 57.85 | 49.42 | 78.92 | 103.53 |

No variant improves on production by more than noise, so the production model is unchanged
and these stay as options (`--interactions`, `--sample-weights`, `--regime-bias`).

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

### 1.5 Further attempts to improve the model (2021–24 to 2025, one seed, production settings)

**Stations never seen, combined model.** Sources: `data/figures/validation/new_stations_combined/summary.json` (`scripts/new_station_check_combined.py`). Street speeds and correction are both fitted without the held-out stations; same 20 groups and same 199,194 trips as section 1.2.

| Model, station never seen | All trips, MAE (s) | First-engine trips, MAE (s) |
|---|---:|---:|
| Street speeds + LightGBM correction | 62.05 | 50.16 |
| LightGBM alone | 62.22 | 50.30 |
| Street speeds alone | 66.16 | 52.02 |
| Kolesar, road distance | 68.69 | 53.91 |

The combined model is better at 62 of 102 stations. Over 6 km it matches LightGBM alone (108.64 s against 108.59 s), so the expected long-trip advantage does not appear on real trips, though this test cannot show behaviour beyond the distances in the data.

**Training for first-engine trips.** Source: `data/figures/analysis/model_improvement/placement_training.json` (`scripts/compare_placement_training.py`).

| Training | Trips | All trips, MAE (s) | First-engine trips, MAE (s) |
|---|---:|---:|---:|
| Production: every trip | 701,000 | 57.80 | 45.64 |
| First-engine trips only | 319,060 | 70.74 | 45.76 |
| First-engine trips weighted 3× | 701,000 | 58.13 | 45.61 |
| Without engines probably away from their station | 625,006 | 58.13 | 45.73 |

**Traffic counts and roadworks along the route.** Sources: `data/figures/analysis/model_improvement/traffic_features.json` and `traffic_features_residuals.json` (`scripts/build_traffic_features.py`, `compare_traffic_features.py`). Traffic is DfT road traffic counts: 2021–24 daily flow, hourly shape 07:00–18:59, main roads only. Roadworks are DfT Street Manager permits active on the day of the trip (1.49 million works on the carriageway). Change in MAE from production, with paired 95 % intervals resampling days:

| Added | All trips (s) | First-engine trips (s) | Over 6 km (s) | Rush hour (s) |
|---|---:|---:|---:|---:|
| Nothing, different seed (chance alone) | +0.02 [−0.01, +0.06] | +0.01 [−0.03, +0.06] | −0.03 [−0.38, +0.32] | +0.02 [−0.04, +0.10] |
| Traffic | +0.01 [−0.03, +0.05] | +0.03 [−0.01, +0.07] | −0.16 [−0.51, +0.22] | +0.09 [+0.02, +0.16] |
| Roadworks | +0.03 [−0.01, +0.07] | −0.01 [−0.06, +0.04] | +0.01 [−0.33, +0.39] | +0.02 [−0.05, +0.09] |
| Both | 0.00 [−0.04, +0.05] | +0.01 [−0.04, +0.05] | −0.10 [−0.50, +0.28] | +0.02 [−0.05, +0.10] |

Neither improves the model beyond chance. Roadworks do have a real but small effect: on first-engine trips, the production model's median shortfall rises from 2.4 s with no road closure on the route to 9.1 s with three or more, but 80 % of those trips have none (rank correlation with the error 0.02). Limits of this test: roadworks are by day and located at the nearest junction, routes are the model's shortest-distance routes and not the ones driven, and counts measure traffic volume on main roads, not congestion or speed. It therefore does not rule out traffic-aware travel times, such as Google's, which need a billing account and were not tried. Roadworks could not be used for scoring future sites in any case, because they are not known in advance.

## 2. Which station responds (busy engines and dispatch)

Source: `data/figures/validation/dispatch/` (`scripts/fit_dispatch_model.py`). Trained on
2021–2024 first engines, tested on 2025.

### 2.1 Share of first engines by rank (fastest open station = 1)

| Rank | Recorded 2025 | Dispatch model | Nearest, or next if busy |
|---|---|---|---|
| 1st | 69.6 % | 69.1 % | 72.3 % |
| 2nd | 18.1 % | 20.4 % | 27.7 % |
| 3rd | 6.3 % | 6.5 % | 0 % |
| 4th | 2.9 % | 2.4 % | 0 % |
| 5th | 1.2 % | 0.9 % | 0 % |

### 2.2 Expected drive per demand cell vs recorded (2025, cells with 20+ calls)

| Response model | Cell MAE (s) | Bias (s) | Correlation |
|---|---|---|---|
| Nearest station always | 47.8 | −46.6 | 0.883 |
| Nearest, or next if busy (Erlang) | 25.4 | −19.3 | 0.878 |
| **Dispatch model** | **23.4** | −16.1 | **0.891** |

### 2.3 Time of day: share sent by the nearest station (2025)

| Slot | Recorded | Dispatch, by hour (calibrated) | Dispatch, all day |
|---|---|---|---|
| 00–04 | 77.8 % | 77.3 % | 68.7 % |
| 04–08 | 79.6 % | 79.2 % | 69.1 % |
| 08–12 | 68.2 % | 67.3 % | 69.1 % |
| 12–16 | 60.5 % | 58.0 % | 69.4 % |
| 16–20 | 68.5 % | 67.9 % | 69.3 % |
| 20–24 | 72.7 % | 71.8 % | 69.0 % |

The nearest station is least available in the early afternoon (training, fire-safety
visits), not at the evening call peak.

Busy engines alone: today's modelled average rises from 284 s (all engines free) to 312 s
(Erlang busy model) and 314 s (dispatch model), against 332 s recorded.

Figures: `validation/dispatch/rank_shares.png`, `reports/weekly_2026-10-19/5_dispatch_time_of_day.png`,
`data_exploration/station_busyness/`.

---

## 2b. Why the first engine does not come from the local station (recorded, no model)

Source: `data/figures/validation/first_engine_gap/` (`scripts/first_engine_gap.py`; first look in
`scripts/busy_engine_check.py`). Engine busy periods are replayed from the mobilisation records. Extra time compares
each group with calls where the local engine arrived first, in the same 500 m square and period of the day; 95 % ranges
resample squares. "Limit" = share of calls x extra time: the most a perfect fix could recover, not an expected saving.

2024 (127,153 calls, average 322.7 s, 31.6 % over 6 min; local engine first on 71.9 %):

| Group | Share | Average | Over 6 min | Extra time | Limit |
|---|---|---|---|---|---|
| All local engines out | 5.8 % | 429 s | 66 % | +149 s [145, 154] | 8.6 s |
| Other first, local engine also sent | 4.7 % | 325 s | 33 % | +24 to +55 s | 1.2 s |
| Other first and closer by road | 4.5 % | 366 s | 46 % | +46 s [42, 51] | 2.1 s |
| **Other first, local engine looks free, not sent, local closer** | **13.2 %** | **438 s** | **69 %** | **+148 s [144, 153]** | **19.5 s** |

The last group ("real gap") across years:

| Year | Share of calls | Extra time | Limit | Limit, points of calls over 6 min |
|---|---|---|---|---|
| 2021 | 11.7 % | +142 s | 16.6 s | 5.4 |
| 2022 | 13.9 % | +154 s | 21.4 s | 6.8 |
| 2023 | 13.3 % | +146 s | 19.5 s | 6.3 |
| 2024 | 13.2 % | +148 s | 19.5 s | 6.3 |
| 2025 | 14.1 % | +153 s | 21.6 s | 6.9 |
| 2026 (part) | 14.1 % | +160 s | 22.5 s | 7.0 |

Share of calls in this group by period, 2024: night 5.3 %, morning 7.5 %, midday 22.2 %, evening 12.8 %, late 9.0 %.
On 2024 and 2025 the limit stays 18.0 to 21.8 s with 10 or 20 minutes allowed for the return trip, a different definition of a
regular engine, 250 m or 1 km squares, or comparison by hour (`sensitivity.csv`).

Reading: the gap is stable and larger than the modelled Move 10 saving. The cause is not established. Engines may be away on
duties the records do not show, which the midday peak suggests but does not prove. 2025-26 times are to the minute; distances
are by road, not time; 2021 was affected by Covid.

### 2b.1 Looking for the cause

Source: `data/figures/validation/gap_cause/` (`scripts/gap_cause_checks.py`).

**Runs of gap calls.** Calls in one station area in time order, previous call in a different 500 m square; chance the next
call is a gap after a gap vs after a local-first call. Null: 500 shuffles within station x hour x weekday/weekend (p values below
0.002 cannot be shown with 500 shuffles).

| Time since previous call (2024) | After a gap | After local first | Difference | Shuffled 95 % range |
|---|---|---|---|---|
| 0-15 min | 61.3 % | 37.7 % | 23.5 | 1.5 to 9.1 |
| 15-30 min | 43.1 % | 14.5 % | 28.6 | 2.4 to 9.1 |
| 30-60 min | 46.0 % | 11.8 % | 34.3 | 3.0 to 8.0 |
| 1-2 h | 38.0 % | 11.8 % | 26.2 | 2.9 to 6.7 |
| 2-4 h | 27.2 % | 12.0 % | 15.2 | 2.0 to 4.9 |
| 4-8 h | 16.5 % | 11.6 % | 4.9 | −0.5 to 2.2 |
| 8-24 h | 14.5 % | 14.8 % | −0.3 | −1.7 to 0.6 |

Same pattern in every year 2021-26 (`clustering.csv`). Windows under 30 minutes may include engines returning from a call.

**Time of day (2024).** Gap share jumps from 10.5 % at 09:00 to 31.9 % at 11:00 on weekdays, and 9.6 % to 30.0 % at weekends, then
falls through the afternoon; about 6 % overnight.

**Delay code.** "On outside duty when mobilised" is 0.24 % of mobilisations in 2024. Across hours of the week its rate tracks the
gap (correlation 0.78 to 0.85 per year, peaking at 12:00 to 14:00, about 1 %). Across stations it does not (−0.06 to −0.27; about 0
within station and period). First engines with this code average 498 s, against 323 s for others (2024).

**Engines at the station (2024, gap share).** Midday: one engine 30.9 %, two 21.4 %. Night: 6.2 %, 6.0 %.

Reading: boundaries and data errors do not fit, because runs fade within hours and appear across different squares. The local
engine, or how it is dispatched, changes for several hours (lengths in 2b.2), mostly around midday, in a clock-driven pattern. That fits engines away on
routine duties but does not prove it, and it does not exclude temporary dispatch changes, repairs or crewing.

### 2b.2 Estimating the windows

Source: `data/figures/validation/gap_windows/` (`scripts/gap_windows.py`). Two hidden states per station on a 15-minute grid, fitted
only on gap and local-first calls in 2024, ignoring calls within 20 minutes of an engine's last incident. Settings then fixed for 2021-23.
2025-26 were not used.

**Hidden calls.** 20 % of calls hidden and predicted, neighbouring calls in a different square. Baseline: station x hour x weekday/weekend.

| Year | Model AUC | Baseline AUC | Model log loss | Baseline log loss |
|---|---|---|---|---|
| 2024 (fitted) | 0.720 | 0.659 | 0.380 | 0.409 |
| 2021 | 0.706 | 0.649 | 0.359 | 0.381 |
| 2022 | 0.719 | 0.652 | 0.396 | 0.428 |
| 2023 | 0.726 | 0.653 | 0.382 | 0.413 |

2024: where the model predicts above 80 %, 81.5 % were gap calls and the baseline predicted 29.8 %, but that is 92 of 18,278 calls.
Only 4.8 % of hidden calls were predicted above 40 %, so most individual moments stay uncertain.

**Trips to other station areas, one-engine stations.** Rate ratio, likely away vs likely available, same station x hour x weekday/weekend. The
simulated baseline is what the method gives when labels have no link to the trips (10 simulations): well below 1, because calls just after a
trip are ignored. Compare with it, not with 1.

| Year | Ratio (95 % range) | Simulated baseline | Drop against baseline | No return-trip allowance | Windows moved 7 days |
|---|---|---|---|---|---|
| 2024 (fitted) | 0.166 (0.131-0.208) | 0.531 | 69 % | 0.297 | 1.14 |
| 2021 | 0.263 (0.175-0.363) | 0.559 | 53 % | 0.423 | 1.23 |
| 2022 | 0.235 (0.191-0.282) | 0.546 | 57 % | 0.334 | 1.11 |
| 2023 | 0.186 (0.148-0.225) | 0.522 | 64 % | 0.298 | 1.12 |

Two-engine stations are mixed (2024 0.635 vs baseline 0.847; 2022 0.913 vs 0.856; 2021 1.128 vs 0.860), as expected if a second engine can
still respond. Moved windows sit somewhat above 1, unexplained. Delay codes and depot on trips during windows do not separate clearly: few trips, and the
direction changes by year (`codes_on_trips.csv`).

**Length and timing, 2024, 2,331 of 8,453 windows with two or more gap calls.** First to last gap call: median 2.0 h (middle half 0.75-3.75 h), a lower
limit. Model window: median 5.75 h (4.5-7.25 h), which leans on the model's settings. 72 % start 09:00-10:59; ends spread over 13:00-18:00.
The model gives about 28 % of 10:00-16:00 as away, but station differences are compressed by the shared settings (23-33 %), so these
are not a station ranking.

**Repeat cycle.** Seen in 2024, then checked on 2021-23. Midday gap share by whether the same station looked away for 3+ midday hours k days earlier
(same station and hour, percentage points):

| Days earlier | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|
| 7 | −2.49 | −0.85 | −1.67 | −0.83 |
| **14** | **+5.03** | **+5.30** | **+7.27** | **+4.97** |
| 21 | +1.24 | −2.66 | −2.90 | −1.09 |
| **28** | **+3.60** | **+3.63** | **+4.82** | **+4.51** |

Raw gap calls without the model, correlation for the same station k days apart: 14 days 0.018 to 0.048, 28 days 0.015 to 0.040, 7 days −0.014 to −0.025, 8 days
−0.012 to −0.019. Smaller peaks at 5, 9, 23 and 37 days. Correlations are small because a day holds few calls.

Reading: during windows, one-engine stations' engines also make far fewer trips elsewhere, which counts against causes that only affect local calls. Trips do not stop,
and the test cannot separate duties, crewing, repairs or a dispatch change that covers all calls. A cycle at 14 and 28 days but not 7 or 8 suggests a
rota, but which one is unknown.

### 2b.3 Fresh test of the 14-day cycle on 2025

Prediction written on 2026-11-04, before any 2025 result was calculated: with the 2024 model settings unchanged and the same measure as the table above,
2025 will show **at least +3.0 points at 14 days, and 0 or below at 7 days**. Both must hold. One run only. 2026 stays unused.

**Result: the prediction as written failed.** The 14-day part held; the 7-day part did not. Run once (a first attempt crashed on a date edge before calculating anything; one call stamped in 2026 was dropped).
Source: `data/figures/validation/gap_windows/cycle_2025.csv` (`scripts/gap_cycle_2025.py`).

| Days earlier | Gap share if flagged | If not flagged | Same station and hour, points | 95 % range (stations resampled) | Predicted |
|---|---|---|---|---|---|
| 7 | 29.5 % | 27.1 % | +0.99 | −0.61 to +2.63 | 0 or below: **not met** |
| 14 | 33.8 % | 26.4 % | +5.29 | +3.46 to +7.08 | +3.0 or above: **met** |
| 21 | 26.6 % | 27.8 % | −2.63 | −4.16 to −1.20 | not predicted |
| 28 | 31.0 % | 27.1 % | +1.93 | +0.29 to +3.54 | not predicted |

Raw gap calls, correlation: 14 days 0.041 (0.030 to 0.052), 7 days 0.005 (−0.008 to 0.015), 21 days −0.017, 28 days 0.013 (−0.001 to 0.028).

Reading, after seeing the result so not a confirmed test: the 14-day signal is the same size as in 2021-24 and clearly larger than at 7 days, but the 7-day value was not below zero as predicted,
and the 28-day signal is weaker than in earlier years. A 14-day pattern is supported; the exact prediction is not. Any stricter claim needs a new prediction, tested on 2026.

### 2b.4 Neighbouring stations

Source: `data/figures/validation/gap_overlap/` (`scripts/gap_overlap.py`). 2021-24, 2024 model settings. 2025-26 not used.

**Pairs of calls, no model, 10:00-16:00.** A call, then within 60 minutes a call in a neighbouring station's area (3 nearest stations), leaving out pairs where the neighbour's engine reached the
first call. Is the second a gap call? Comparison: station pairs at least 10 km apart. Null: 200 shuffles within station x hour x weekday/weekend.

| Year | Neighbours: after gap / after local | Difference | Shuffled 95 % range | 10 km+ apart: difference | Shuffled 95 % range |
|---|---|---|---|---|---|
| 2024 | 34.4 % / 26.9 % | 7.5 | −0.7 to 2.7 | 0.5 | −1.6 to 2.0 |
| 2021 | 36.0 % / 23.8 % | 12.2 | −1.0 to 3.1 | 1.5 | −1.2 to 2.5 |
| 2022 | 36.6 % / 26.6 % | 10.0 | −1.3 to 2.4 | 2.8 | −1.2 to 2.4 |
| 2023 | 35.3 % / 27.5 % | 7.9 | −1.0 to 2.8 | 1.6 | −1.2 to 2.4 |

Neighbours' gap calls come together more than chance, so there is no sign that neighbours currently avoid being away together. It does not show why: joint activity, shared
dispatch effects or knock-on effects that were not removed would all fit.

**Overlap in the model, 10:00-16:00, 2024.** Both away 8.05 % of the time; the same pairs 7 or 21 days apart 8.00 %. The model is unsure at most moments, which understates any link, so this does
not contradict the pairs test. While a station is likely away, at least one of its 3 nearest is too 63.6 % of the time, and all three 2.7 %. Other years are similar.

**Passing over the next-nearest station by road (2024).** Extra seconds against gap calls where that station reached the call first, same 500 m square and period. Limit = share of all calls x extra seconds.

| On a gap call, the next-nearest station... | Share of gap calls | Share of all calls | Mean attendance | Extra (95 % range) | Limit | Limit, over-6-min points |
|---|---|---|---|---|---|---|
| reached it first | 63.0 % | 7.70 % | 410 s | baseline | - | - |
| had every engine out | 9.8 % | 1.20 % | 509 s | +106 s (95-118) | 1.27 s | 0.33 |
| looked free, passed over | 27.1 % | 3.31 % | 487 s | +74 s (66-81) | **2.45 s** | **0.61** |

Passed over, limit in other years: 2021 2.17 s, 2022 3.72 s, 2023 2.66 s (0.58 to 0.91 points). With a 500 m tie margin instead of 200 m, 1.82 to 3.40 s.

**Check.** Using only its own area's calls, the model says the next-nearest station was likely away 30.1 % of the time when passed over and 22.1 % when it reached the call first (2024); differences 6.6 to 9.4 points across
years, against about 0 on simulated labels. So being passed over is linked to that station looking away, but most passed-over stations do not look away, and road distance is an imperfect guide to who is quickest.

Reading: keeping the next-nearest station available has an upper limit of about 2 to 4 s across all calls, or 0.5 to 0.9 points. That is below the planned target (10 s or 3 points), so staggering between neighbours
cannot reach it alone. The remainder of the gap's 19.5 s limit comes from the local engine's own absence. Moving that away time to quieter hours is tested in 2b.5.

### 2b.5 Moving away time to quieter hours

Source: `scripts/gap_shift.py`, `data/figures/validation/gap_shift/`. A calculation, not a simulation. Total away hours per day stay fixed for each engine class and day type, and only their timing changes. Each call costs
the chance of a gap above the available-state rate x 148 s, or x 48 over-6-minute points. The savings are upper limits: duties are assumed free to move, with no effect from neighbours.

| Plan | Cost, 2024 | Saving, 2024 | Saving, over-6-min points | Saving, 2023 |
|---|---|---|---|---|
| Today | 11.02 s | - | - | - |
| Spread evenly, 08:00-18:00 | 11.21 s | -0.19 s | -0.06 | -0.22 s |
| Spread evenly, 08:00-20:00 | 11.43 s | -0.41 s | -0.13 | -0.45 s |
| Spread evenly, 07:00-22:00 | 11.10 s | -0.08 s | -0.03 | -0.10 s |
| Quietest hours in 08:00-18:00 | 10.95 s | 0.07 s | 0.02 | 0.07 s |
| Quietest hours in 08:00-20:00 | 10.95 s | 0.07 s | 0.02 | 0.09 s |
| Quietest hours in 07:00-22:00 | 10.12 s | 0.90 s | 0.29 | 0.91 s |
| Quietest hours of the whole day (not realistic) | 5.63 s | 5.39 s | 1.75 | 5.50 s |
| 25 % less away time | 8.27 s | 2.75 s | 0.89 | 2.77 s |
| 50 % less away time | 5.51 s | 5.51 s | 1.78 | 5.54 s |

Reading: within realistic hours, moving away time saves about 0 to 1 s, because daytime demand is too flat for the timing to matter. Only doing less away time, or moving it into the night, saves more. This does not reach the
planned target (10 s or 3 points).

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
| **Dispatch model** (all day) | **0.59** | **13.0** |
| Dispatch model (by hour, calibrated) | 0.64 | 12.4 |

The all-day model is the one that scores layouts, so it is the headline result; with 10 areas, the two dispatch versions cannot be told apart (95 % range of 0.64: about 0.20 to 0.96).

Reading: the travel time model gets the drive right when it knows who responded; the
remaining error is dispatch (after the closures the nearest station sent the first engine
on as few as 38 % of calls in some areas).

Figures: `reports/weekly_2026-10-19/4_closure_2014.png`.

### 3.3 Ranking the closed stations, compared like for like

`scripts/closed_station_ranking_check.py`, `validation/closure_2014/ranking_like_for_like.*`. Predictions use
the dispatch model, today's stations and 2025 calls, except the last row. Damage = recorded change per call in the
area, as in 3.1. Rank correlation across 10 stations; range includes the error in the recorded effects.

| Comparison | Rank correlation | 95 % range | Permutation p |
|---|---|---|---|
| London-wide saving vs damage per call (units do not match) | −0.15 | −0.87 to 0.71 | 0.68 |
| London-wide saving vs damage × calls in the area | +0.22 | −0.58 to 0.87 | 0.54 |
| Saving per call in the area vs damage per call | +0.33 | −0.53 to 0.87 | 0.35 |
| 2014 test: predicted vs recorded effect per call (all-day dispatch) | +0.68 | −0.08 to 0.96 | 0.035 |

The −0.42 first reported compared unlike units with the older nearest-or-next model (−0.33 when rerun with
the 73 s new-site turnout). Silvertown shows the difference: the most damage per call (+126 s) but few calls.
The ranking of which station to reopen today cannot be told apart from chance with 10 stations, so it is
not validated. The last row is optimistic: the dispatch model was improved while watching this test.

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

### 5.1 Screening other placement methods

Source: `scripts/placement_methods.py`, `data/figures/placement/methods/`. Chosen on 2024 calls, scored with the models on 2025 calls. 2026 is not used. Search: repeated best swap, with a limit on stations moved, no borough
or slowest-10 % floors, no street refinement. It is weaker than the exact planner, so compare these rows with each other, not with the table above.

Change from today in average attendance (s), busy-engine model:

| Stations moved (limit) | Average | Over 6 min objective | Average, away time kept apart | Average, many more sites |
|---|---|---|---|---|
| 5 | -2.80 | -3.03 | -2.77 | -3.45 |
| 10 | -4.86 | -4.52 | -4.63 | -5.25 |
| 15 | -6.12 | -5.88 | -5.99 | -6.88 |
| 20 | -7.12 | -6.61 | -6.96 | -7.55 |
| 30 | -8.51 | -8.16 | -8.44 | -9.76 |
| 40 | -9.50 | -9.07 (36 moved) | -8.77 | -11.07 |
| No limit (search stops) | -9.61 (41 moved) | -9.07 (36 moved) | -9.74 (56 moved) | -12.94 (53 moved) |

Candidate sites: 370 for the first three columns, 989 for the last.

The widest pool, under the other scores. The 95 % ranges are narrow (at most 0.3 s and 0.08 points either way) because they only cover which calls occur, not model error:

| Moved | Busy engines | Road | Kolesar | Learned dispatch | Slowest 10 % | Over 6 min, points | Over-6 calls avoided per year | Calls 30+ s faster | Calls 30+ s slower |
|---|---|---|---|---|---|---|---|---|---|
| 10 | -5.25 | -4.52 | -3.17 | -5.03 | +7.59 | -1.49 | 1,924 | 10.3 % | 3.3 % |
| 20 | -7.55 | -6.91 | -5.39 | -6.87 | +9.43 | -1.98 | 2,555 | 17.3 % | 7.8 % |
| 30 | -9.76 | -9.18 | -7.38 | -8.66 | +8.93 | -2.51 | 3,243 | 21.0 % | 9.3 % |
| 40 | -11.07 | -11.07 | -8.81 | -9.72 | +8.00 | -2.86 | 3,695 | 23.9 % | 11.6 % |
| 53 | -12.94 | -13.98 | -11.15 | -11.56 | +8.50 | -3.28 | 4,246 | 28.3 % | 13.8 % |

Moving second engines only, with every building kept (calculated from 2024 calls, not scored on 2025): 10 engines -0.84 s, and no further gain at 40. With a different arrangement in each period of the day: 10 engines -1.07 s, and
no further gain at 40.

Over-6-minute curve (logistic, attendance to chance, per cell): 33.0 % modelled against 31.9 % recorded in 2024. Calls avoided are modelled, not observed.

Reading:

- The number of stations moved matters far more than the method. No method moves Move 10 beyond about 5 s in this search.
- Optimising over 6 minutes directly adds almost nothing: -1.35 points at 10 moved against -1.29 for the average. Keeping away time apart also changes little.
- More candidate sites help most: 0.4 s at 10 moved, 1.3 s at 30, 3.3 s with no limit. The exact planner already refines sites within 1 km, and reaches -6.0 s at 10 moved, so the gain may not simply add to the planner's.
- Moving engines without moving buildings saves about 1 s at most.
- The search stops at 41 moved and -9.6 s on the standard sites, while the exact redesign reaches -18.0 s with 66 moved, or -15.6 s with 46 moved and the 1 s borough floor. So the search is well short of what is possible.
- Without a floor, the slowest 10 % get worse in every screened layout (+4 to +24 s). The exact planner's slowest-10 % floor would reduce the gains here. New sites are grid cells and are not checked for whether a station could be built.

---

### 5.2 Checks that inform placement

Source: `scripts/placement_analysis.py`, `data/figures/placement/analysis/`. Layouts use the screening search of 5.1, not the exact planner, and have no borough or slowest-10 % floors, so compare rows with each other.

**Does the model reproduce recorded calls? (2025)** Recorded 329 s; learned dispatch model 315 s. Removing gap calls gives 311 s; removing gap and empty-station calls gives 302 s. The size of the overall miss is about the size of the gap, but this does not prove the gap causes it.

| Period | Gap share | Recorded (s) | Recorded without gap calls (s) | All-day model (s) |
|---|---|---|---|---|
| Night 00-07 | 5.6 % | 329 | 323 | 313 |
| Morning 07-10 | 7.7 % | 315 | 306 | 314 |
| Midday 10-16 | 21.9 % | 341 | 309 | 315 |
| Evening 16-20 | 13.0 % | 339 | 318 | 316 |
| Late 20-24 | 8.5 % | 306 | 295 | 315 |

The model does not vary by hour here, so part of the period difference is traffic. It matches how often the nearest station responds (69.2 % vs 69.4 % recorded; by station ground, r = 0.92), so it gets the rate of diversion right but not the delay. It gives the recorded station an average chance of 57 %.

The error is uneven and persistent: average recorded minus model is +7 s within 5 km of central London, rising to +28 s beyond 20 km; across squares with 30+ calls it correlates 0.63 between 2024 and 2025. By station ground it ranges from -6.5 s (Chelsea) to +52 s (Woodford), and does not follow gap share (r = -0.04). Areas whose nearest station changes under the screened Move 10 have a smaller error (+4.6 s vs +19.0 s elsewhere). Outer-area closures are the least reliable.

**Does the gap follow the station, area, workload or engines?** 102 stations, 2021-25, 10:00-16:00. Share of variation in station-year gap share explained (weighted):

| Explained by (each with year) | R² |
|---|---|
| Year only | 0.04 |
| Workload | 0.20 |
| Area (distance from centre, local road distance, calls) | 0.31 |
| Number of engines | 0.42 |
| Engines + workload + area | 0.53 |
| Station identity | 0.81 |

Predicting a held-out year from the other years: station identity r = 0.81, engines + workload + area r = 0.67. In 180 1 km squares served by two stations, a station's gap share elsewhere predicts its share in the shared square: slope 0.88 [0.56, 1.15], where 1 means the station and 0 means the place. Neighbouring stations are only weakly alike once other factors are removed (r = 0.08). Within a station, years with more workload have a lower gap share: -1.4 points for 10 % more [-2.2, -0.7]. This could run backwards, since an engine that is away makes fewer trips. One-engine stations: 30.4 %; two-engine: 21.7 %. No station changed engine count, so the engine effect is not causal. No crew data exist, so a station-level trait is not proven to be crew or duties.

Reading: moving a station would probably not remove the gap, and lowering workload has no support as a fix. Whether the pattern would travel with a moved station cannot be tested here.

**Which savings hold up?** Choosing on each year 2021-24 and scoring on 2025 (s):

| Chosen on | Move 10 | Move 20 | Move 40 |
|---|---|---|---|
| 2021 | -5.19 | -7.83 | -9.93 |
| 2022 | -5.05 | -8.16 | -11.32 |
| 2023 | -5.50 | -7.99 | -11.13 |
| 2024 | -5.39 | -7.94 | -11.05 |

Wide site pool. Standard pool: -4.0 to -4.8, -6.6 to -7.1, -9.6 to -9.9. Sites chosen in two different years share on average 6.2 of 10 opened and 8.3 of 10 closed (wide pool). Reverting one move at a time, every move in Move 10 and Move 20 helps in every year 2021-25; contributions correlate 0.88-0.95 between 2024 and 2025. At 40 moves, 2 of 40 failed the 2021-23 check; dropping them changes 2025 by 0.08 s. Savings are spread out: in wide-pool Move 10 the best move gives 1.15 of 5.39 s, the best 5 give 3.11 s (58 %). Only demand changes between years; travel times are fixed, so this is not a test of the travel model.

**Where could stations go?** Local OpenStreetMap extract, 2025 score, searching only among allowed sites (wide pool; change in average, LightGBM):

| Rule for new sites | Sites | Move 10 | Move 20 | Move 40 | No limit |
|---|---|---|---|---|---|
| All | 887 | -5.48 | -7.78 | -10.95 | -13.33 (58 moved) |
| A: not parks, water, schools, hospitals etc.; road within 100 m; main road within 300 m | 624 | -5.12 | -7.61 | -11.01 | -11.75 (48) |
| B: A + commercial, industrial, retail or brownfield plot within 300 m | 481 | -5.08 | -7.17 | -10.59 | -11.60 (55) |
| C: B + site itself not on residential land | 180 | -4.49 | -6.73 | -8.39 (34) | -8.39 (34) |

Grouped together, 201 of 887 new wide-pool sites sit on excluded land. All 10 official Move 10 sites, and the Add 3 and redesign sites, pass the excluded-land and road checks; 8 of 10 Move 10 sites have a plot within 300 m. This checks land type only, not whether any plot is for sale, large enough or allowed by planning rules. Rule labels come from OSM and can be missing or out of date.

**Do engines and buildings need to move together?** One formula, moves chosen on 2024, scored on 2025 (standard pool). All layouts keep 142 engines.

| Option | Change in average (s) |
|---|---|
| Move engines only: 1 / 3 / 5 moves | -0.32 / -0.49 / -0.60 |
| Move engines only: no gain after 6 moves | -0.66 |
| Move 10, engines go with the building | -4.82 |
| Move 10, new sites get one engine, the spare goes to the nearest kept station | -4.76 |
| Move 10 + 9 further engine moves | -5.73 |
| Move 20, engines go with the building | -7.05 |
| Move 20 + 10 further engine moves | -8.44 |

Engines alone are not a substitute for moving stations, and they need not follow the building. As an addition, 9 or 10 engine moves add 0.9 to 1.4 s with no construction. The busy-engine formula does not model away time.

---

## 6. Do the layouts hold up under other travel time models?

Source: `data/figures/placement/layout_scoring/scores.csv` (`scripts/score_layouts.py`).
2025 calls, dispatch model (retrained 2026-10-23), 1000 bootstrap resamples. Today: LightGBM 314.9 s,
Kolesar road 324.5 s, Kolesar straight 330.1 s.

Change in average (s); a layout "wins" if the interval is below 0 under all three models.

| Layout | LightGBM | Kolesar road | Kolesar straight | Wins under all 3 | Slowest 10 % no worse under all 3 |
|---|---|---|---|---|---|
| Add 1 (1 s) | −1.2 | −0.8 | −0.3 | yes | yes |
| Add 2 (1 s) | −1.4 | −1.1 | −0.6 | yes | yes |
| Add 3 (1 s) | −2.1 | −1.6 | −1.0 | yes | no (+0.0) |
| Add 5 (1 s) | −3.7 | −3.0 | −2.0 | yes | yes |
| Move up to 5 (1 s) | −2.4 | −2.4 | −1.4 | yes | yes |
| **Move up to 10 (1 s)** | **−5.6** | **−4.6** | **−2.2** | **yes** | **yes** |
| Full redesign (1 s) | −14.4 | −10.7 | −4.7 | yes | yes |
| Full redesign (no borough rule) | −16.8 | −12.1 | −5.2 | yes | yes |
| Add 3, sites chosen by Kolesar road | −2.1 | −2.0 | −1.3 | yes | yes |
| Move 10, sites chosen by Kolesar road | −3.5 | −4.8 | −2.7 | yes | no (+2.5) |
| Move 10, sites chosen by Kolesar straight | −2.9 | −3.1 | −3.5 | yes | yes |

Move up to 10 (1 s), slowest 10 %: LightGBM −4.2 [−4.5, −4.0], Kolesar road −4.7 [−5.1, −4.3], Kolesar straight −3.6 [−4.0, −3.3].

Every LightGBM-chosen layout beats today under all three models. Gains are largest under
LightGBM (the model that chose them) and smallest under the straight-line formula, so the
spread between models is the honest uncertainty on the size of the gain.

Figures: `reports/weekly_2026-10-19/3_three_models.png`, `placement/station_sweep.png`.

---

### 6.1 Neutral test: hidden stations scored on recorded trips

The table above is scored by the same models that choose the layouts. This test is not
(`scripts/neutral_siting_test.py`, `data/figures/validation/neutral_siting/`). Twenty of today's
102 stations are hidden; all models are refit without their trips (2021–24), so they look like
new sites; 82 stay open; each model picks k of the 20 on 2024 calls with the project's solver and
busy engines. Picks are scored with recorded 2025–26 drive times from those stations (half the
trips choose the fastest open station, the other half give its time, then the reverse). 20 hidden
groups (4 splits of 5); 98 % of calls could be scored; average with the 82 alone is about 362 s.

Average seconds saved compared with the 82 stations alone:

| Sites picked | LightGBM | Kolesar road | Kolesar straight | Random | Hindsight (rough limit) |
|---|---|---|---|---|---|
| 3 of 20 | 7.08 | 6.74 | 6.86 | 4.20 | 7.50 |
| 5 of 20 | 10.79 | 10.20 | 10.59 | 6.96 | 11.22 |
| 10 of 20 | 18.16 | 17.97 | 18.01 | 13.74 | 18.47 |

LightGBM's extra saving, [95 % interval across groups], and groups won / tied / lost:

| Sites picked | vs road | W/T/L | vs straight | W/T/L | Sites shared with road, straight |
|---|---|---|---|---|---|
| 3 of 20 | +0.34 [−0.00, 0.69] | 8/11/1 | +0.22 [−0.10, 0.57] | 7/9/4 | 2.55, 2.45 |
| 5 of 20 | +0.59 [0.26, 0.95] | 10/10/0 | +0.20 [−0.26, 0.67] | 8/5/7 | 4.45, 3.85 |
| 10 of 20 | +0.19 [−0.01, 0.39] | 9/8/3 | +0.15 [−0.18, 0.44] | 13/2/5 | 9.25, 8.60 |

With the learned dispatch model deciding who responds (unrecorded pairs filled from a curve on the
held-out trips), the extra saving is smaller: vs road +0.09, +0.33 [0.06, 0.58], −0.02; vs straight
+0.14, +0.09, −0.17 (only 5 of 20 vs road has an interval above zero).

**Which of two stations arrives first** (2025–26 calls with engines from two home stations; each
station predicted by a model that never saw it; split 1 only):

| Pairs | Number | LightGBM right | Road right | Straight right |
|---|---|---|---|---|
| All | 118,398 | 80.0 % | 80.2 % | 78.8 % |
| Road distances within 25 % | 35,745 | 60.9 % | 61.5 % | 58.3 % |

LightGBM minus road: −0.2 points [−0.37, −0.06] on all pairs, −0.6 [−1.12, −0.20] on close pairs.
LightGBM minus straight: +1.2 [1.04, 1.45] and +2.6 [2.01, 3.27].

Reading: the hypothesis gets weak support. LightGBM's picks were never significantly worse and
sometimes slightly better, but mostly the same sites, and the difference is under 1 s. Choosing with
any model matters far more (3 to 4 s better than random). LightGBM predicts times better
(section 1.2) but does not order two stations better than road distance, and placement depends on order.

Limits: candidates are real stations rather than free grid sites, where models could differ more;
only part of the network changes; groups share stations across splits, so intervals are a little narrow;
six comparisons were made without correction; second engines are not a random sample of trips.

---

## 7. Robustness: time of day, severity, borough, income

Source: `data/figures/placement/robustness/` (`scripts/placement_robustness.py`), rerun
2026-11-01 with the current dispatch model. 2025 calls; LightGBM rows use travel times for each period's hour and the hourly dispatch
model; Kolesar rows have no hour. 500 bootstrap resamples (100 for boroughs).

### 7.1 Time of day (LightGBM): change in average, [95 % interval]

| Period | Calls | Today (s) | Add 3 | Move up to 10 | Full redesign |
|---|---|---|---|---|---|
| Night 00–07 | 21,939 | 287.0 | −1.8 [−1.9, −1.6] | −4.7 [−5.0, −4.4] | −12.0 [−12.6, −11.5] |
| Morning rush 07–10 | 12,320 | 318.2 | −2.2 [−2.4, −2.0] | −4.6 [−5.1, −4.1] | −13.7 [−14.4, −12.9] |
| Day 10–16 | 38,789 | 330.4 | −2.0 [−2.1, −1.9] | −5.5 [−5.7, −5.3] | −15.8 [−16.2, −15.4] |
| Evening rush 16–20 | 30,027 | 325.9 | −1.7 [−1.8, −1.6] | −5.3 [−5.6, −5.0] | −14.4 [−14.9, −13.9] |
| Evening 20–24 | 26,193 | 296.4 | −1.5 [−1.6, −1.4] | −4.9 [−5.2, −4.6] | −12.7 [−13.2, −12.2] |
| All day | 129,268 | 314.0 | −1.8 | −5.1 | −14.0 |

Slowest 10 % by period: move up to 10 and the redesign improve in every period (−3.2 to
−5.9 s and −11.9 to −15.7 s). Add 3 makes the slowest 10 % 0.1–0.3 s worse at night and in
both rush hours (the floor holds all day, not in every period).

### 7.2 Call severity: change in average (s)

| Calls | Add 3 (LGBM / road / straight) | Move up to 10 | Full redesign |
|---|---|---|---|
| All (129,268) | −1.8 / −1.6 / −1.0 | −5.1 / −4.6 / −2.2 | −14.0 / −10.7 / −4.7 |
| Fires and special services (68,263) | −1.3 / −1.2 / −0.8 | −4.6 / −4.3 / −2.0 | −13.5 / −10.2 / −4.2 |
| Fires in homes (4,878) | −1.3 / −1.2 / −0.8 | −5.4 / −5.2 / −3.1 | −14.6 / −11.4 / −5.8 |

These three layouts are faster for every severity group under all three models.

### 7.3 Neighbourhood income: change in average (s)

Each call takes the ONS FYE 2020 equivalised household income after housing costs of its
nearest London MSOA (about 980 neighbourhoods, split into fifths).

| Income fifth | Add 3 (LGBM / road / straight) | Move up to 10 | Full redesign |
|---|---|---|---|
| Q1 (poorest) | −1.2 / −1.1 / −0.7 | **−9.6 / −9.3 / −5.7** | **−19.3 / −17.1 / −9.5** |
| Q2 | −1.6 / −1.6 / −1.3 | −3.9 / −3.6 / −2.6 | −15.2 / −11.9 / −7.8 |
| Q3 | −1.0 / −1.1 / −0.7 | −3.0 / −2.3 / −0.6 | −13.9 / −8.9 / −4.2 |
| Q4 | −1.4 / −1.3 / −0.6 | −2.8 / −1.9 / +0.3 | −9.1 / −6.3 / −0.1 |
| Q5 (richest) | −4.0 / −2.8 / −1.8 | −5.5 / −5.2 / −1.4 | −10.7 / −7.2 / +0.1 |

Moving stations helps the poorest fifth most, under every model. Adding stations helps the
richest fifth most (new sites go to central London, which is both busy and wealthy). Under
the straight-line formula two fifths get no gain: Q4 in move up to 10 and Q5 in the redesign.

### 7.4 Boroughs

With a 1 s borough tolerance, under LightGBM:

- Add 3: 12 boroughs better, 8 slightly worse; worst Barnet +0.9 s.
- Move up to 10: 25 better, 7 worse; worst City of London +2.7 s (few calls).
- Full redesign: 32 better, Sutton unchanged.

Under the Kolesar formulas some boroughs get worse. In move up to 10: City of London up to
+10.3 s, Richmond +5.6 s, Tower Hamlets +3.7 s (straight-line). In the redesign: Tower Hamlets up to +11.0 s and
Lewisham +10.1 s (straight-line). The borough rule is enforced with the model that chose the
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
| 15 | `final/method_diagram.png` | Data → travel time → dispatch/busy → solver → validation | have |
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
