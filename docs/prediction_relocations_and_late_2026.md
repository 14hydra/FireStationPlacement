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
