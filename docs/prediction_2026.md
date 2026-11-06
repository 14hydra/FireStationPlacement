# Part 2 final test: prediction for 2026, fixed before 2026 is used

Written 2026-11-05. Nothing from 2026 has been used to choose, tune or score anything in Part 2. The only thing read from
the 2026 file so far is how many calls it has in each month (January to July 2026, 80,780 calls in the planner file), to
fix the test window below.

This file and `scripts/final_test_2026.py` are committed together before the test is run. The script refuses to run if
either has uncommitted changes, and refuses to run a second time.

## What is tested

**Question (Part 2).** Can relocating a few fire stations make London's slowest boroughs clearly faster without making any
borough slower?

**Main layout (fixed now).** The targeted Move 10 searched only on Rule B sites (new sites need a commercial, industrial,
retail or brownfield plot within 300 m): `data/figures/placement/targeted/rule_b/layouts/target_10.csv`. It is the
buildable version, and it was chosen after the 2025 results were seen (results.md 5.6), so 2026 is its first clean test.

**Target boroughs (unchanged, from recorded 2024 times):** Hillingdon, Havering, Enfield, Bromley, Redbridge.

**Data.** All 2026 calls dated 1 January to 31 July 2026 in `data/raw/london/lfb_incidents_planner.csv`, after the usual
removal of calls far from any station. Later months are not part of this test, even if they are downloaded later.

**Scoring.** Exactly as `scripts/targeted_areas.py` scored 2025: demand cells, busy-engine rates, turnout and the Kolesar fit
all from 2024; only the calls counted change. Three travel models: LightGBM (production model, trained 2021-24), road
Kolesar and straight-line Kolesar. Ranges from resampling calls (1,000 draws).

`python scripts/final_test_2026.py --check-year 2025` runs the identical code on 2025 and reproduces the published 2025
numbers (results.md 5.6), which checks the script before the one real run.

## Predictions and pass marks

1. **Target boroughs faster.** The main layout speeds up the 5 target boroughs' average by at least 10 s under all three
   travel models.
2. **No borough slower.** No London borough is 1 s or more slower under any of the three models.
3. **The outer-London model error does not shrink the saving.** On 2026 first-engine trips from the home station with the
   engine not busy, in the 5 target boroughs, recorded drive time rises by 0.9 to 1.1 s per modelled second (the slope of
   recorded on modelled time). In 2025 it was 1.00 [0.98, 1.02] (`scripts/outer_error_check.py`, results.md 5.7). A slope in
   this range means the model's outer-London error is a fixed offset that cancels when two layouts are compared.

The test passes only if all three hold. Expected values, from 2025: -12.2 / -14.3 / -10.8 s (LightGBM / road / Kolesar),
worst borough +0.5 s, slope about 1.0. The Kolesar margin is only 0.8 s, so pass mark 1 can fail.

## Also reported, not pass marks

Scored the same way and reported whatever they show, so the main layout is not the only one that could have looked good:
the targeted Move 10 and up to 20 without land rules, the Rule B up to 20 (13 moved), the citywide Move 10 and the planner
Move 10 and Move 20 (section 5.4). Also: London average, slowest 10 % of calls, share of calls 30+ s slower, every
borough's change, and the drive-time offset in seconds.

## How the result will be reported

Whatever comes out, in results.md and the notebook, including a partial pass or a fail. No layout, rule, borough list,
model or date window is changed after the run. If the script has a bug that stops it running, the fix is committed
with a note before it is run again, and the note says what changed.
