# Notebook entries for Part 2 (draft facts to copy)

Synopsys requires a project notebook, and judges may read it. These are the facts, dates and numbers for the Part 2 entries,
in order. **Write the entries yourselves, in your own words and handwriting or your own notebook file.** Add who did what and
what you thought at each step; this file can't record that. Times are local (Pacific) from git and file times.

---

### 2026-11-04: the 14-day cycle prediction (supporting finding)

- Prediction written before calculating 2025: at least +3.0 points at 14 days **and** 0 or below at 7 days.
- Result: **failed as written.** 14 days +5.3 points (range +3.5 to +7.1), met. 7 days +1.0 (range -0.6 to +2.6), not met.
- Lesson: a pattern can be real (14 days) and the exact prediction still fail. We report it as failed.
- Source: results.md 2b.3.

### 2026-11-05 (day): why we changed the question

- Moving 10 stations saves about 5-7 s London-wide at best (results.md 5.3); removing every limit does not beat about 7 s.
- 46 moves give about 15-18 s under LightGBM and road, about 9 s under the straight-line formula (5.4, 5.5).
- Options compared fairly (docs/research_question_evaluation.md). Chosen: Option A, making the slowest boroughs faster
  without making any borough slower. Reason: it has a testable hypothesis with pass marks set first, and the gain is large
  where it matters.

### 2026-11-05 15:31: target rule and pass marks written (before any targeted layout)

- File: docs/targeted_areas_rule.md (file time 15:31).
- Targets: the 5 boroughs with the slowest recorded 2024 attendance (Hillingdon 384.5 s, Havering 371.4, Enfield 366.5,
  Bromley 366.1, Redbridge 358.3). Recorded times, so no model is involved in choosing them.
- Pass marks: (1) target boroughs at least 10 s faster under all three travel models; (2) no borough 1 s or more slower.
- **Be honest if asked:** the rule was committed to git at 16:29 (commit 75c6b90), after the first targeted runs had started
  (the Rule B run's output is timed 16:14). The order rests on the file time, 15:31, not on git.

### 2026-11-05 afternoon: 2025 results against the pre-set test

| Pre-set test | Mark 1 | Mark 2 |
|---|---|---|
| Up to 10 moves | Missed under straight-line formula (-9.8 s) | Passed (worst +0.6 s) |
| Up to 20 moves (15 used) | Passed (-15.2 / -17.2 / -14.1 s) | Passed |
| Up to 10, Rule B, sites shifted | Missed (-7.1 s) | Missed (+2.2 s) |

- Added **after** seeing results: searching again only on Rule B sites (commercial or industrial land within 300 m).
  10 moves: -12.2 / -14.3 / -10.8 s, no borough worse. Label it as decided after the results.
- The main targeted run was rerun with the final script: identical results.
- The ordinary planner Move 10 already gives these boroughs -10.2 / -9.5 / -5.7 s; targeting adds 1.7-4.1 s and keeps every
  borough protected. Cost to London: 1.8 s of average against the same search aimed at the whole city.
- About 2 % of calls are 30+ s slower (near closed stations), even though no borough average is.

### 2026-11-05 19:45: outer-London model error check (2025 only)

- Worry: the model is about 26 s too fast in the target boroughs. Does that shrink the savings?
- Test: recorded vs modelled drive time on 12,497 first-engine trips there. Slope 1.00 (0.98 to 1.02); offset +17.8 s, the
  same at every trip length. Local station responds 74.9 % recorded vs 76.0 % modelled.
- Conclusion: the error is a fixed offset, which cancels when comparing layouts; savings shrink by 1-5 % at most.
- Source: results.md 5.7.

### 2026-11-05 20:02: 2026 prediction committed (commit c40c34b)

- File: docs/prediction_2026.md, with scripts/final_test_2026.py.
- Main layout: Rule B targeted Move 10. Data: January to July 2026 calls only.
- Pass marks: (1) at least 10 s faster under all three models; (2) no borough 1+ s worse; (3) drive slope 0.9-1.1.
- Expected: about -12.2 / -14.3 / -10.8 s; the straight-line margin was only 0.8 s, so it could fail.
- Dry run of the script on 2025 reproduced all 21 published 2025 rows exactly.
- Disclosure: Part 1's hidden-stations test had already used some 2026 drive times, not for anything in Part 2.

### 2026-11-05 20:39: 2026 test run once

| Pass mark | Result |
|---|---|
| 1. Targets 10+ s faster, all models | **Pass:** -13.7 / -16.1 / -13.0 s |
| 2. No borough 1+ s worse | **Pass:** worst +0.4 s |
| 3. Drive slope 0.9-1.1 | **Pass:** 1.04 |

- Other layouts: citywide and planner Move 10 again made 2-3 boroughs slower under the straight-line formula (up to +7.7 s).
- Gains larger than 2025, but 2026 is only January to July, so don't call it an improvement.
- Still modelled: no station has moved.
- Source: results.md 5.8; data/figures/placement/final_test_2026/result.json.

### What we would do next (for the "further research" question)

- Score the 2026 August-December calls when released, as a second check (not part of the fixed test).
- Follow up with LFB (an email has been sent; awaiting a reply), for example on whether the 14-day pattern matches a training or maintenance schedule.
- Check real sites: is land for sale, and how big are the plots? Our land rules use OpenStreetMap labels only.
- Test a stricter rule (site itself non-residential), which halves the gain in the outer boroughs.
