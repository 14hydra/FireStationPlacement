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

- **`--use-busy`** (**EXISTS**): blends nearest and next-nearest attendance in
  **reported** response via `expected_first_arrival`; **optimisation still uses
  nearest engine only** (`posture_response` vs `plan_*` on matrix `A`).
- Without `--use-busy`, modelled cover is **optimistic** vs observed attendance
  (always-free nearest engine).

Caveat: busy ≠ "home engine physically unavailable" — also cross-border and
closest-appliance mobilisation.

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
