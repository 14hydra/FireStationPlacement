# Fixed predictions: real engine relocations, and August-December 2026

Written 2026-11-11, before either test is run. Commit this file before writing or running either test script. Like
`docs/prediction_2026.md`, nothing below is changed after a test is run; any bug fix is committed with a note saying what
changed, before the test is run again.

---

## Test A: do the models predict what happened when engines really moved?

### Why

Every Part 2 result is modelled: no station has moved. Between 2013 and 2016 London rebuilt several stations, and while
each was rebuilt its engine worked from a nearby station. These are real relocations of the kind Part 2 proposes (moves
of about 2 to 8 km). If the models predict how much slower each area became, the Part 2 predictions have real-world
support. The 2014 closures (results.md section 3) are already a check of this kind, but they are one event whose result
we have seen; these are new events.

### What has been looked at so far (disclosure)

- `scripts/relocation_events.py` found the events from mobilisation records only: which station each engine was
  deployed from, and when. It does not read attendance, travel or turnout times. Its output
  (`data/figures/validation/relocations/events.csv`) lists the stations, dates, engine codes, where the engines went and
  how many deployments they made. No attendance time for any event area has been calculated.
- The 2014 closure test (results.md 3) used 2013-2014 attendance in the 10 closed areas and in "the rest of London",
  which includes the Plaistow and Deptford areas. Their attendance was not looked at separately.

### The events (fixed now)

| Event | Station rebuilt | Engine worked from | Distance | Dates (from `events.csv`) | Set |
|---|---|---|---|---|---|
| 1 | Leytonstone | Leyton | 1.8 km | 2014-12-06 to 2016-02-21 | Scored |
| 2 | Shadwell | Poplar | 2.8 km | 2015-01-16 to 2016-04-17 | Scored |
| 3 | Purley | Mitcham | 8.0 km | 2015-02-27 to 2016-03-16 | Scored |
| 4 | Dockhead | Deptford | 3.1 km | 2015-11-21 to 2016-05-10 | Scored |
| 5 | Plaistow | Stratford | 2.9 km | 2013-10-01 to 2015-05-30 | Scored, after-window only (see below) |
| 6 | Deptford | New Cross | 1.7 km | 2013-10-18 to 2015-11-15 | Scored, after-window only (see below) |
| 7 | Dagenham | Barking (51 %), elsewhere | 5.7 km | 2014-12-01 to 2016-01-23 | Reported, not scored: engines split |
| 8 | Lee Green | Lewisham | 1.8 km | 2023-01-12 to 2023-10-27 | Reported, not scored: inside the current model's training years |

Distances are straight-line, between today's station sites. Wennington (2020-12-23 to 2021-01-08) is left out: 17 days, and its engine did not move.

Plaistow and Deptford started before the 2014 closures on 9 January 2014, which changed nearby areas too. For these two,
the period before the move is not used. The "during" window starts on 9 January 2014, and the comparison is with the
period after the move only.

### Method (fixed now)

- **Area of each event:** 0.9 km cells whose first engine, in the 365 days before the move started, most often came from
  the rebuilt station (incident records, `FirstPumpArriving_DeployedFromStation`). Same rule as the 2014 test.
- **Windows:** "during" = the move dates. "Before" = the 180 days ending 30 days before the move. "After" = the 180 days
  starting 30 days after it ended.
- **Recorded effect:** mean recorded first-engine attendance in the area, "during" minus the comparison window, minus the
  same change in the control cells (difference in differences, as in the 2014 test). The control cells are every cell
  outside all 8 event areas. With both comparison windows, the effect is the average of the "during minus before" and
  "during minus after" effects. 95 % range from resampling calls, 1,000 draws.
- **Predicted effect:** each model's first-engine attendance per cell, weighted by that window's recorded calls per
  cell, put through the same difference in differences.
  - Stations: the 2014 station set (2013 set for any window before 9 January 2014). In "during", the rebuilt station is
    removed and its engines are added to the station they worked from.
  - Busy engines: Erlang model as in Part 2, calibrated on the comparison window.
  - Turnout: each station's median in the comparison window. Moved engines keep their own station's median.
  - Travel models: the pre-2014 LightGBM model (`travel_time_lfb_pre2014.joblib`, trained 2009-2013, so it has not seen
    any scored "during" window), road Kolesar and straight-line Kolesar fitted on 2009-2013 trips, as in
    `scripts/closure_2014_test.py`. For Lee Green (reported only): the current model and the 2024 Kolesar fits.
  - Also reported: the dispatch model (results.md 3.2).
- **Code check before the real run:** run on the 2014 closures, the same code must reproduce the published 2014 results
  (results.md 3.1) within 0.5 s: recorded +70.5 s and LightGBM with busy engines +67.5 s. The test script runs this
  check first and stops if it fails.

### Pass marks

Primary model: pre-2014 LightGBM with busy engines (the kind of model Part 2 scores with). Scored events: 1 to 6.

1. **Direction.** In every scored event where the recorded 95 % range does not include 0, the predicted effect has the
   same sign.
2. **Overall size.** The call-weighted average predicted effect over the scored events lies inside the 95 % range of
   the call-weighted average recorded effect.
3. **Error per event.** The mean absolute difference between predicted and recorded effect over the scored events is
   20 s or less. For comparison, the 2014 test's error per area with busy engines was 16.3 s (results.md 3.2); 20 s
   allows for the smaller areas here.

Test A passes only if all three hold. The same three marks are reported for road Kolesar, straight-line Kolesar and the
dispatch model, but they are not pass marks. "Model X predicts real moves better than model Y" is claimed only if its
error per event is at least 5 s lower.

**What to expect.** The models should predict that each area got slower, by less than in the 2014 closures (+70 s),
because the engine stayed nearby. Event 3 (Purley to Mitcham, 8 km) should show the largest effect; the other scored moves
were 1.7 to 3.1 km. Recorded effects in small areas may be noisy, so mark 1 only counts events with a clear recorded effect.

**What each result would mean.** If it passes, the models predicted real relocations within about 20 s, which supports
Part 2's modelled gains. If it fails, Part 2's headline stays a modelled result, and the failed mark says where the
models go wrong: direction, overall size, or per-area error.

---

## Test B: the Part 2 main layout on August-December 2026

### What

The same test as `docs/prediction_2026.md`, with the same main layout (Rule B targeted Move 10), target boroughs,
scoring and pass marks, on calls dated 1 August to 31 December 2026. No August 2026 or later data has been downloaded:
the newest call in `data/raw/london/lfb_incidents_planner.csv` is dated 2026-07-31.

### When it runs

- A month counts once the download holds calls from the following month, or calls on that month's last day.
- The test runs once, when all of August to December 2026 is in the download.
- If that has not happened by 2027-02-15, it runs once on the complete months from August that are available, provided
  there are at least 3. It is then labelled "partial (August to <month>)".
- With fewer than 3 complete months by 2027-02-15, it is not run, and is reported as not run.
- When new data is downloaded, `data/processed/lfb_travel_training.csv` is rebuilt with the same build script and no
  other change, so that pass mark 3 can use the new trips.

### Pass marks (unchanged from the January-July test)

1. Target boroughs at least 10 s faster under all three travel models.
2. No borough 1 s or more slower under any model.
3. Drive-time slope in the target boroughs between 0.9 and 1.1.

**Expected:** about -12 to -14 s (LightGBM), -14 to -16 s (road) and -11 to -13 s (straight-line Kolesar), from 2025
(-12.2 / -14.3 / -10.8) and January-July 2026 (-13.7 / -16.1 / -13.0). The straight-line margin was 0.8 s in 2025, and
August-December covers different seasons from January-July, so mark 1 can still fail.

### Script

`scripts/final_test_2026_late.py`: the same code as `final_test_2026.py`, with the date window and output folder
(`data/figures/placement/final_test_2026_late/`) changed. Before the 2026 data exists, it is dry-run on August-December
2025 to check that it runs, and committed. Like the January-July script, it refuses to run if this file or the script has
uncommitted changes, and refuses to run a second time.

---

## Reporting

Both tests are reported whatever they show, in results.md and the notebook, including partial passes and fails. Neither
test changes the Part 2 headline, which is the January-July 2026 result.

---

## Notes added 2026-11-12, while writing the test scripts (before either test was run)

No recorded attendance for any event area has been calculated. What was run: the 2014 code check, the model predictions
(`--predictions-only`, which reads calls per cell, turnout, pump counts and busy rates, but no attendance in any event
area), and Test B's dry runs on already-used months.

**Test A: changes to the method above, and why**

1. **Area rule.** The area is defined from first-engine trips in the mobilisation records (`PumpOrder` 1, with the
   training-set filters), which is what the 2014 test actually did. The wording above ("incident records,
   `FirstPumpArriving_DeployedFromStation`") did not match the 2014 code: with it, the code check failed (LightGBM busy
   +68.6 s against +67.5 s; 57 area cells against 58).
2. **Code check.** It found one bug (latitude and longitude swapped in the busy-rate distances; LightGBM busy +64.6 s).
   After both fixes the script reproduces every 2014 number in results.md 3.1 to 0.01 s (recorded +70.5; LightGBM +76.8
   free, +67.5 busy; road Kolesar +71.0 / +59.5; straight-line Kolesar +78.3 / +62.7).
3. **The network in each window.** Several moves overlap. In every window (comparison or "during"), any other event whose
   move covers more than half of the window is in place: its station is out and its engines work from their
   destination. Without this, Dockhead's "before" window (April-October 2015) treated Deptford as open while Deptford was
   being rebuilt, and Deptford's ground looked 94 % busy to the busy-engine model.
4. **Details the method above left open, now fixed.**
   - The predicted change per cell is weighted by the "during" window's calls, as in the 2014 code.
   - Recorded 95 % range: calls are resampled within each window, separately in the area and in the control.
   - Pumps per station in a window are home-station engine codes used 100+ times a year (pro rata), as in the 2014 code.
   - Moved engines are a separate row at the destination's site, with their own station's turnout, engine count and
     busy-model load per call. For another event in place, its crews' turnout is their median in the 365 days before
     its move.
   - Busy rates are the training set's busy flag: the share of first engines in each open station's ground (nearest
     open station) that came from another station.
   - The dispatch model is the saved one (`dispatch_lfb.joblib`), trained on 2021-2024 trips. That is after events 1-7,
     but includes Lee Green's year.

**Test A: model predictions, fixed now** (`data/figures/validation/relocations/predicted.csv`; the test checks that it
reproduces these numbers)

| Event | Station | Area cells | Calls during | LightGBM busy | Road Kolesar busy | Straight-line Kolesar busy |
|---|---|---|---|---|---|---|
| 1 | Leytonstone | 12 | 608 | +48.5 | +43.3 | +56.4 |
| 2 | Shadwell | 2 | 398 | +53.3 | +50.1 | +65.2 |
| 3 | Purley | 38 | 362 | +158.4 | +151.2 | +150.7 |
| 4 | Dockhead | 7 | 324 | −32.1 | −22.2 | −16.0 |
| 5 | Plaistow | 9 | 1,030 | +72.6 | +79.9 | +66.0 |
| 6 | Deptford | 6 | 1,341 | +74.9 | +79.2 | +64.5 |
| 7 | Dagenham (reported) | 33 | 1,295 | +92.1 | +89.7 | +98.4 |
| 8 | Lee Green (reported) | 14 | 562 | +68.0 | +63.4 | +56.8 |

Call-weighted average over events 1-6, LightGBM busy: +67.1 s. This replaces the guess under "What to expect" above:
the models predict about as much slowing as in the 2014 closures, not less, and much more for Purley.

**Dockhead (event 4).** The models predict it got *faster*. Its area comes from November 2014 to November 2015, when
Deptford (its destination) was itself being rebuilt for 7 of the 12 months, so Dockhead's engine was covering part of
Deptford's ground. One of its 7 cells is 0.1 km from Deptford station. Moving Dockhead's engines to Deptford then speeds
up those cells. This was noticed from the model predictions, before any recorded attendance was looked at.
**Decision (2026-11-13, before any recorded attendance for any event area was calculated):** keep event 4 scored. The models were given the real network in each window, including Deptford being closed, and still predict Dockhead got faster. That is a genuine, checkable prediction, and an unusual one, so it is a harder test of the models, not a broken one. Dropping it now, because its prediction looks odd, would weaken the test. The pass marks stay on events 1-6 as fixed. As a secondary check only, the results also report the same scores on events 1-3, 5 and 6 (without Dockhead); this does not change pass or fail.

**Shadwell (event 2)** has only 2 area cells. It is kept, as fixed; its recorded range will be wide.

**Test B: dry runs** (`data/figures/placement/final_test_2026_late/check_*`)

- January-July 2026: reproduces `final_test_2026/result.json` exactly (0.00 s on every primary score).
- August-December 2025 (a used year): −13.1 / −15.3 / −11.9 s, drive slope 0.967, all three marks met.
- The training-trips file must have trips in the window's last month: the January-July file ends on 2026-07-30, not
  the 31st, so "trips up to the last day" would have refused a file that is complete.
