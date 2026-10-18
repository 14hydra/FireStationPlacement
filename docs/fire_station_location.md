# Firehouse location planner

Plan **where fire stations should sit** so the first engine meets LFB attendance
standards. Drive times come from the LightGBM decision-tree model by default; road / Kolesar / crow scorers are baselines (all **EXIST**).

> **EXISTS** = in the repo today. **PLANNED** = in the plan, not built or run.

## Research question

Does placing stations with street-aware (or Kolesar) drive times beat crow-flies
placement on held-out **2025** LFB calls, once attendance = drive + turnout?

## KPIs: LFB standards (not NFPA / US 4-minute)

| KPI | Target | Code |
|---|---|---|
| First engine, mean attendance | **≤ 6 min** | `firestation.lfb_standards.FIRST_ENGINE_AVG_S` |
| First engine, share within 10 min | **> 90 %** | `FIRST_ENGINE_P90_S`, `FIRST_ENGINE_TARGET_SHARE` |
| Second engine (secondary) | mean ≤ 8 min; 95 % within 12 min | `SECOND_ENGINE_*` |

`score_first_engine(response_s, weights)` returns mean, p90, shares within 6 / 10 min,
and pass/fail (**EXISTS**). The planner summary includes `lfb_first_engine` on the
demand grid and, when `--eval-years` is set, held-out eval plus optional
`observed_attendance` from CSV **`travel_seconds`** (**EXISTS**). Coverage bars use
`--threshold-min` (default **10** min, LFB tail target).

## What is scored: turnout + drive

```
attendance(station s → incident i) = predicted_drive(s, i) + turnout(s)
```

- LFB mobilisation records give turnout (`TurnoutTimeSeconds`) separately from drive.
  For London the planner uses each station's **median recorded turnout** on home-station
  trips in the demand years (`recorded_station_turnout`; 2024: 57–91 s, median 73 s).
  New sites get the median over all stations (73 s) (**EXISTS**).
- Without mobilisation data (custom cities), existing houses get a Kolesar-residual
  estimate and new sites `DEFAULT_TURNOUT_S = 60 s`.
- `lfb_standards.attendance_seconds(drive_s, turnout_s)` implements the sum (**EXISTS**).

## Modes

| Mode | Meaning | Status |
|---|---|---|
| `redesign` | Place exactly `k` stations from the pool (`--n-houses`; default = current count) | **EXISTS** |
| `replace` | Close `r` / open `r` (`--replace r`) — e.g. `r = 10` mirrors 2014 closures | **EXISTS** |
| `expand` | Keep all current stations, add `m` new ones (`--add-stations m`) | **EXISTS** |

Candidates (**EXISTS**): existing stations plus top demand-cell
centroids (`--demand-candidates`). **PLANNED**: additional candidates on street-graph
nodes. Objectives: p-median `response_time` or max-cover `coverage`.

## Recommended placement (`scripts/recommended_placement.py`)

The project's position: **favour busy areas, but leave no one worse off than today.**

- alpha = 1 (each area counts by its calls), busy engines on, pumps move with stations,
  each new site refined to a street junction, LightGBM travel times;
- floors (`--keep-slowest`, `--keep-average`, on by default in the CLI): the slowest 10 %
  of calls and the overall average may not be worse than today; both are linear limits,
  so the exact solver still applies (checked against brute force), and the busy-aware
  loop only keeps layouts that meet them under the full busy model;
- 10-minute cap: no area that meets the standard today loses it.

Then the alpha check: efficiency score (share of the best possible average improvement)
and equity score (share of the best possible area-average improvement), plus a
per-borough table, so the cost of favouring busy areas is visible.
Outputs: `data/figures/recommended_placement/<scenario>/` (map, figures, stations,
`boroughs.csv`, `summary.json`).

Results (busy engines; today 309.5 s average, 423.2 s slowest 10 %, 81.6 % within 6 min;
2025 calls 311.7 s modelled, 332.0 s recorded):

| Scenario | Average | Slowest 10 % | Area avg | Within 6 min | 2025 calls | Boroughs worse ≥ 1 s |
|---|---|---|---|---|---|---|
| Add 3 | 306.0 | 422.9 | 353.3 | 82.5 % | 308.5 | Lambeth |
| Move up to 10 | 302.9 | 420.9 | 351.3 | 83.6 % | 305.7 | 6 (Kingston +33 s) |
| Full redesign (66 moves) | 290.5 | 406.8 | 337.0 | 88.4 % | 293.7 | 4 |

Scores above 100 (efficiency 108 / 117 / 170) mean the recommended layout beat the
reference "best possible", which is unrefined (street refinement adds 1–8 s) and, with busy
engines, a heuristic loop rather than a proven optimum; read them as "at or past the
reference". Equity scores: 15 / 21 / 76.

**Price of fairness** (`--keep-boroughs SECONDS`: no borough's average may get worse
than today by more than SECONDS). Citywide floors alone still let boroughs lose (move 10:
Kingston +33 s, Tower Hamlets +21 s). Busy engines on; today 309.5 s / 423.2 s:

| Scenario | Borough rule | Average | Slowest 10 % | Boroughs worse ≥ 1 s | Moves |
|---|---|---|---|---|---|
| Add 3 | none | 306.0 | 422.9 | 1 | 3 |
| Add 3 | 0 s / 1 s / 5 s | 306.6 / 306.4 / 306.0 | ≈ 423 | 0 / 0 / 1 | 3 |
| Move up to 10 | none | 302.9 | 420.9 | 6 | 10 |
| Move up to 10 | 5 s | 303.4 | 420.0 | 2 | 10 |
| Move up to 10 | 1 s | 304.0 | 420.2 | 0 | 10 |
| Move up to 10 | 0 s | 305.6 | 420.9 | 0 | 10 |
| Full redesign | none | 290.5 | 406.8 | 4 | 66 |
| Full redesign | 1 s | 293.3 | 409.5 | 0 | 46 |

Protecting every borough costs about 1 s (move 10, 1 s rule) to 3 s (full redesign).
Borough floors make the exact problem hard: when it finds no layout in time, a fallback
(genetic algorithm seeded with today's layout, then a feasible-swap search that only
accepts swaps meeting every floor) is used and reported (not proven optimal); this was
needed for move 10 at 0 s and the full redesign.


## Optimiser: exact integer program

`redesign`, `replace` and `expand` are solved as integer programs with HiGHS
(`scipy.optimize.milp`, `firestation.facility.solve_exact`), to a **proven optimum**: the
summary's `solver` block gives the status, the solver's lower bound and the gap (0 = proven).

- p-median: open-site variables `y_s` (0/1) and assignment variables `x_sj`; minimise
  Σ w_j t_sj x_sj / Σ w_j with every cell served once, only by open sites, and `k` sites.
- `--cap-min` (default 10, LFB's standard): no area that today's stations reach within this
  many minutes may be left beyond it. Areas already over it today are not constrained (the
  objective still pulls them closer). Today's layout always satisfies the cap, so every
  mode has a solution.
- `replace --replace r` keeps at least `n − r` current stations (so it may move fewer than
  `r` if that is better); `expand` fixes all current stations open.
- Pairs a cell would never use (slower than a station that is always open) are dropped
  before solving, which keeps the answer exact and makes `expand` solve in well under a second.

This replaces the earlier greedy-plus-swaps search, which had no optimality guarantee.

## Optimiser: genetic algorithm (`--solver genetic`)

`firestation.facility.solve_genetic`. Each individual is a set of open sites; tournament
selection, elitism, a crossover that keeps the sites both parents share, and a mutation
that swaps one site (half the time for a nearby one). Stops after 400 generations or 80
without improvement. Its point is the **busy-engine objective**: with `--use-busy` it
minimises (1 − p)·t₁ + p·t₂ (nearest engine busy with probability p, next nearest answers),
which the integer program cannot express linearly.

Check against the exact solver (same problem, nearest engine): add 3 → same optimum;
move 10 → within 0.04 %; full redesign → within 0.01 %. On the busy-engine objective
(move 10), the genetic algorithm beats the exact solver's layout by 0.5 s (304.0 vs 304.5 s).

## Street-level sites (`--refine-km`, default 1)

After solving, each newly opened site (not today's stations) moves to the best road
junction within `--refine-km` (`firestation.site_refine`):

1. Candidates: junctions on A / B / local distributor roads (primary, secondary,
   tertiary in OpenStreetMap; not motorways, trunk roads or residential side streets).
2. Quick score for every candidate: one Dijkstra run at posted speed limits to every
   demand cell, converted to attendance seconds by a straight-line fit on the site's own
   LightGBM times.
3. The best 6 get full LightGBM travel times and are scored with the planner's real
   objective (busy engines when on, the 10-minute cap enforced); the best move is kept
   only if it improves the layout.

Sites are named by the roads at the junction (e.g. "New site on Kilburn High Road").
About 15–30 s per new site. Results: add 3 → 2 sites move 190–310 m, 278.3 → 278.0 s;
move up to 10 with busy engines → all 10 move 30–920 m, 303.9 → 302.9 s
(2025 calls 311.7 → 305.6 s). Gains are small because a 0.9 km cell centre is already
close to its best junction; the point is a real, buildable street location.

## Slow calls: 90th-percentile model and CVaR objective

- **90th-percentile travel times** (`scripts/train_travel_time_quantile.py` →
  `travel_time_lfb_q90.joblib`, scorer `--scorer lgbm_q90`): LightGBM with quantile loss
  (0.9) on the same features. 2025 coverage 86.9 % (target 90 %; 83 % on trips over 8 km,
  as 2025 trips run slower than training years); pinball loss 18.6 vs 19.4 for the median
  model plus a constant offset. On average the 90th percentile is 93 s above the median.
- **`--objective cvar`**: minimises the average of the slowest 10 % of calls (CVaR,
  Rockafellar–Uryasev), linear, so the exact solver still applies (checked against brute
  force on a small problem); also in the genetic algorithm and the busy-aware solver.
  CVaR problems are harder: move-10 and redesign hit the 15-minute limit and report a gap.
- `scripts/placement_slow_calls.py` solves each mode four ways (average or CVaR, median
  or 90th-percentile times) and scores every layout on all measures
  (`data/figures/placement_slow_calls/`). Move up to 10, all calls:

  | Objective | Average | Slowest 10 % | Avg 90th pct |
  |---|---|---|---|
  | today | 281.9 | 406.7 | 353.7 |
  | average, median times | 276.3 | 405.8 | 349.3 |
  | CVaR, median times | 284.2 | 394.9 | 356.7 |
  | average, 90th-pct times | 277.2 | 404.6 | 349.0 |
  | CVaR, 90th-pct times | 282.1 | 396.8 | 354.7 |

  Optimising the average barely helps the slowest calls (−1 s); optimising them cuts the
  slowest 10 % by about 12 s at a cost of 2 s on the average. Every layout keeps 99.5 %+ of
  calls in areas whose modelled 90th percentile is under 10 minutes, so that measure does
  not separate layouts (recorded: about 94 % of calls within 10 minutes).

## Demand weighting and the trade-off dial

- `--severity all|serious|dwelling_fire`: which calls count (all; fires + special services;
  fires in homes). Uses `incident_group` / `property_category` from the planner CSV.
- `--alpha a`: each area counts as (weighted calls)^a. 1 = fastest call-weighted average,
  0 = every area with calls counts the same. `scripts/placement_tradeoff.py` sweeps it.
- Candidates: today's stations, the 80 busiest 0.9 km cells, and the busiest cell in every
  3 km block (`--spread-km`), so the solver can also place stations in quieter areas.
- Home calls: LFB withholds exact coordinates for dwellings, so the planner CSV uses LFB's
  50 m rounded grid reference for them (`dest_exact = 0`). Earlier planner runs had left out
  every dwelling call (about 60 % of calls).

## Scorers (multi-scorer comparison)

| Scorer | Travel time from… | Status |
|---|---|---|
| **crow** | Haversine surrogate at a flat speed with a 1.3 detour factor | **EXISTS** (`--scorer crow`) |
| **Kolesar** | Piecewise `T = a + b√d` / `c + e·d` on distance; refit on London trips (`firestation.kolesar`) | **EXISTS** (`--scorer kolesar`) |
| **road** | Kolesar curve (road-km fit from the travel-time trainer) on the shortest legal route over the directed London drive graph | **EXISTS** (`--scorer road`) |
| **lgbm** (street-aware) | LightGBM travel-time model (`docs/travel_time_training.md`) on the full route features from every site to every demand cell | **EXISTS** (`--scorer lgbm`, `firestation.ml_scorer`) |

Default scorer: **lgbm** for London (the decision-tree model is London-only), **kolesar** for `--demo` and custom cities.

**lgbm scenario.** A site has no station ID, so new and existing sites are scored alike:
first pump, from its home station, not busy, on the most common call type (false alarm in
a dwelling). Each cell gets the most common borough of its calls. Time of day is a weighted
average over hours 03 / 08 / 13 / 18 / 21, weighted by when demand calls happen. The first
run routes 182 sites × ~1,900 cells and takes about 10 min; the matrix is then cached in
`data/processed/planner_cache/` (also for `road`).

**Expand by 2, demand 2024, eval 2025** (modelled first-engine mean, current → plan):

| Scorer | Demand grid | 2025 calls |
|---|---|---|
| lgbm | 284.3 → 279.9 s | 287.1 → 284.5 s |
| road | 293.7 → 289.5 s | 297.7 → 295.1 s |
| kolesar | 301.7 → 297.3 s | 305.2 → 302.5 s |

All three open the same two sites (near Soho and Farringdon). Observed 2025 attendance is
339 s mean, so every scorer is still optimistic: they assume the nearest engine is free.

## Demand and evaluation split

| Use | Years | Status |
|---|---|---|
| Demand (where calls happen) | **2023–24** target | **2024 only** until 2023 is ingested |
| Evaluation (held-out) | **2025** | **EXISTS** via `--demand-years` / `--eval-years` |

CLI defaults: `--demand-years 2024`, `--eval-years 2025`. When `cal_year` is present
in the planner CSV, `run_fire_station_plan` builds the demand grid from demand years and
reports eval KPIs on eval-year incidents (**EXISTS**). Non-LFB cities still use
`--max-incidents` subsampling without year split.

## Busy engines

LFB's first pump is often not from the home station. `firestation.busy_engines` marks
**busy** when `FirstPumpArriving_DeployedFromStation != IncidentStationGround`
(**EXISTS**). `build_lfb_planner_incidents.py` writes `busy_flag` and deploy columns.

- **`--use-busy`** (**EXISTS**, `firestation.busy_placement`): busy engines in both the
  **optimisation** and the scores. Expected first-engine time = (1 − p)·t₁ + p·t₂, where p
  is the chance the nearest station has no free pump.
  - **Erlang loss model**: p = B(a, c) for a station with c pumps and offered load a.
    Each station's a is calibrated so B reproduces its measured 2024 busy rate today
    (mean 28 %); load scales with the calls the station is nearest to, so opening a
    station next door takes calls and busyness off its neighbours. New sites get the
    median erlangs per call.
  - **Exact solver**: every area gets a primary and a backup station; with p fixed this
    is linear (Daskin's expected-coverage idea). It solves, recomputes p from the new
    loads, and re-solves until the layout repeats, keeping the best. Each round is a
    proven optimum for its fixed p; the loop itself is a heuristic.
  - **Genetic algorithm**: optimises the full load-dependent model directly.
  - The summary's `busy_vs_ignoring_busy` compares with the plan that assumes the
    nearest engine is free, scored under the busy model.
- First result (move up to 10, all calls): today 309.5 s with busy engines (vs 281.9 s
  assuming free engines; observed 2024–25 mean about 332 s). Plan ignoring busy engines
  305.8 s; busy-aware exact 304.6 s; busy-aware genetic 304.1 s. 7 of the 10 moved sites
  change when busy engines are modelled.
- **Pumps move with their station** (`busy_placement.PumpPlan`, all modes but resize):
  kept stations keep their pumps; each closed station's pumps go to an opened site
  (closed and opened sites paired to minimise total distance moved); in expand mode a new
  station takes the second pump of the nearest two-pump station (`--expand-pumps move`,
  default; London stays at 142 pumps) or gets new pumps (`--expand-pumps new`). The busy
  model uses each layout's pumps inside the optimisation; the summary lists `pump_moves`.
  Measured 2024 busy rate: 32 % at one-pump stations, 22 % at two-pump stations.
  Move up to 10 (busy-aware): 309.5 → 303.9 s with pumps moving (304.6 s when new sites
  always had one pump). Add 3 with pumps moved from two-pump stations: 306.3 s (305.7 s
  with three new pumps, 145 in total).
- Without `--use-busy`, modelled cover is **optimistic** vs observed attendance
  (always-free nearest engine).

Caveat: busy ≠ "home engine physically unavailable" — also cross-border and
closest-appliance mobilisation.

## Three-model scoring (`scripts/score_layouts.py`)

Every layout scored on held-out 2025 calls (48,115) under LightGBM, Kolesar on road km and
Kolesar on straight-line km, with busy engines (each layout's own pumps) and recorded
turnout. Change in average first-engine time vs today, 95 % paired bootstrap interval:

| Layout (chosen by → rules: busy, floors) | LightGBM | Road Kolesar | Straight-line Kolesar |
|---|---|---|---|
| Add 3, LightGBM | −3.2 [−3.2, −3.1] | −2.6 | −2.0 |
| Add 3, road Kolesar | −2.8 | −2.5 | −2.3 |
| Add 3, straight-line Kolesar | −2.9 | −2.4 | −2.4 |
| Move 10, LightGBM | **−6.0** [−6.2, −5.9] | **−4.7** | −2.4 |
| Move 10, road Kolesar | −3.8 | −5.2 | −3.1 |
| Move 10, straight-line Kolesar | −3.6 | −3.9 | **−4.2** |
| Full redesign, LightGBM | −18.0 | −13.1 | −6.0 |

- Every layout beats today under all three models (all intervals below 0).
- Each model favours the layouts chosen with it. The LightGBM layouts beat the
  distance-chosen ones under LightGBM and road Kolesar, but not under straight-line
  Kolesar, so "street-aware placement is better" holds under 2 of 3 models. The 2014
  closure test is what decides which travel-time model to trust.
- The size of the gain depends on the model: the full redesign saves 18 s under LightGBM
  but 6 s under straight-line Kolesar.
- With borough floors (rescored the same way; average / slowest 10 % change vs today):

  | Layout | LightGBM | Road Kolesar | Straight-line Kolesar |
  |---|---|---|---|
  | Move 10, no borough rule | −6.0 / −2.2 | −4.7 / −1.7 | −2.4 / −1.9 |
  | Move 10, every borough ≤ 1 s | −5.3 / −3.2 | −4.6 / −3.0 | −2.3 / −3.2 |
  | Full redesign, no borough rule | −18.0 / −15.1 | −13.1 / −7.1 | −6.0 / −0.8 |
  | Full redesign, every borough ≤ 1 s | −15.6 / −13.0 | −12.2 / −9.3 | −6.1 / −5.0 |

  Borough floors cost a little on the average under LightGBM but make the slowest-call
  gains larger and consistent across all three models: the fairer layouts are also the
  more robust ones.
- The intervals are narrow because they only resample calls; they do not include
  travel-time model error, which the spread across the three models shows is larger.
- Slowest 10 %: the LightGBM layouts improve it under all three models (move 10: −1.7 to
  −2.2 s); the road-chosen move-10 layout makes it worse under all three (+0.7 to +2.9 s)
  although it met the floor in planning: floors are enforced on 2024 demand with the
  planner's busy model, and 2025 calls can differ.

## Validation

1. **2014 ten-station closures** — list in `firestation.london_closures_2014` (**EXISTS**).
   `scripts/validate_london_2014_closures.py` and `scripts/rank_closed_stations_2014.py`
   run counterfactuals on **today's network** with approximate closed-station coords
   (**EXISTS**). True pre-2014 train → post-2014 test needs older LFB extracts
   (**PLANNED** ingest).
2. **Leave-one-firehouse-out** — **PLANNED**.
3. **Multi-scorer table** — planner runs one scorer per invocation (**EXISTS**);
   batch comparison script **PLANNED**.

## Run (London)

Defaults: `--city london`, `--threshold-min 10`, year split as above.

```bash
# Planner extract from raw LFB CSV
PYTHONPATH=src python scripts/build_lfb_planner_incidents.py

# Redesign / replace / expand (Kolesar scorer)
PYTHONPATH=src python scripts/plan_fire_station_locations.py \
  --city london --mode redesign --demand-years 2024 --eval-years 2025
PYTHONPATH=src python scripts/plan_fire_station_locations.py \
  --city london --mode replace --replace 10
PYTHONPATH=src python scripts/plan_fire_station_locations.py \
  --city london --mode expand --add-stations 2

# Optional busy blend in reported metrics only
PYTHONPATH=src python scripts/plan_fire_station_locations.py \
  --city london --mode expand --add-stations 3 --use-busy

# Synthetic smoke test
PYTHONPATH=src python scripts/plan_fire_station_locations.py --demo

# 2014 closure scaffolds (today's-data approximation)
PYTHONPATH=src python scripts/validate_london_2014_closures.py
PYTHONPATH=src python scripts/rank_closed_stations_2014.py

# Custom city
PYTHONPATH=src python scripts/plan_fire_station_locations.py \
  --city custom --fire-stations path/houses.csv --incidents path/incidents.csv \
  --mode redesign --n-houses 20
```

`london` in `firestation.cities` uses `data/raw/london/london_fire_stations.csv` and
`lfb_incidents_planner.csv`.

## Outputs (`data/figures/fire_station_plan/`)

| File | Content |
|---|---|
| `01_metrics_comparison.png` | E[T] + coverage bars vs current / random |
| `02_response_cdf.png` | Demand-weighted response CDF |
| `03_response_surface_map.png` | Before/after response surfaces |
| `04_relocations.png` | Closed → opened moves |
| `05_site_demand_share.png` | Top sites by assigned demand |
| `fire_station_plan_map.html` | Interactive Folium map |
| `planned_fire_stations.csv` | Selected sites + assigned demand share |
| `closed_fire_stations.csv` / `opened_fire_stations.csv` | Replace-mode moves |
| `demand_cells.csv` | Grid cells with planned / current response |
| `candidates.csv` | Full candidate pool |
| `summary.json` | Metrics, `lfb_first_engine`, `demand_years`, `eval_years`, `eval`, honesty text |

## Honesty limits

- **Modelled vs observed:** predicted attendance uses nearest-engine drive + turnout;
  observed `travel_seconds` reflects real dispatch (busy engines, cross-border pumps).
  Compare via eval block `observed_attendance` — expect modelled cover to look **better**
  unless `--use-busy` (partial correction, not full dispatch simulation).
- Station sites are OpenStreetMap building locations, name-matched to LFB stations.
- Discrete candidates only; no land cost, staffing, borough politics, or planning rules.

## Code

| Piece | Path |
|---|---|
| City registry (London + custom) | `src/firestation/cities.py` |
| Planner (redesign, replace, expand) | `src/firestation/fire_station_location.py` |
| Demand grid, exact p-median / max-cover integer program (HiGHS) | `src/firestation/facility.py` |
| Kolesar model | `src/firestation/kolesar.py` |
| LFB standards + attendance sum | `src/firestation/lfb_standards.py` |
| Busy-engine rates | `src/firestation/busy_engines.py` |
| 2014 closures | `src/firestation/london_closures_2014.py` |
| LFB planner CSV | `scripts/build_lfb_planner_incidents.py` |
| 2014 validation scaffolds | `scripts/validate_london_2014_closures.py`, `scripts/rank_closed_stations_2014.py` |
| CLI | `scripts/plan_fire_station_locations.py` |
