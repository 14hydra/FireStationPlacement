# Option A: target rule, fixed before any targeted layout was built

Written 2026-11-05, before running `scripts/targeted_areas.py` or looking at any targeted-layout result.
It should be committed before the results are quoted, so the order is on record.

**Question.** Can moving a few stations make London's slowest areas clearly faster while no other area gets slower?

**Target areas.** The 5 boroughs with the slowest *recorded* average first-engine attendance in 2024
(`travel_seconds`, all 2024 calls, boroughs with at least 1,000 calls). Recorded times are used so the choice does not
depend on any travel model. Nothing from 2025 or 2026 is used to choose them.

**Layouts.** Up to 10 moves, and separately up to 20, on the wider site list (989 sites), chosen on 2024 demand with
LightGBM travel times and busy engines (best-swap search, the same engine as `placement_methods.py`).

- Objective: the call-weighted average attendance in the 5 target boroughs.
- Limits, checked on 2024 demand: no borough's average gets worse at all (0 s), the London average is no worse, and the
  slowest 10 % of calls are no worse.

**Scoring.** 2025 calls, busy engines, under LightGBM, road Kolesar and straight-line Kolesar, plus land rules B and C
(sites that fail move to the nearest passing site, as in `score_wide_layouts.py`). 2026 is not used.

**Success, decided now.**

1. Main: the target boroughs' average improves by at least 10 s under all three models.
2. No other area slower: no borough is 1 s or more worse under any of the three models.
3. For comparison: the same numbers for the untargeted wide Move 10 and Move 20 layouts (section 5.4), which were not
   built for these boroughs.

Whatever the result, it is reported as it comes out, including partial passes.
