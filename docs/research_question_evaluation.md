# Which research question to take to Synopsys, CSEF and ISEF (2026-11-05)

Also a handoff: section 5 lists what was done in this session so another Claude Code session can pick it up.
Numbers are from `docs/results.md` (sections 5.4 to 5.6), which stays the source of truth.

## 1. Recommendation: option A

**Question:** Can relocating a few fire stations make London's slowest boroughs clearly faster without making any borough slower?

**Hypothesis (fixed before the test):** moving 10 stations, chosen with the street-learned travel model, speeds up the 5 boroughs with the
slowest recorded 2024 attendance by at least 10 s, with no borough 1 s or more slower, under all three travel models.

**What the test found (2025 calls, not used to choose anything):**

- Target boroughs (Hillingdon, Havering, Enfield, Bromley, Redbridge): -11.9 s (LightGBM), -12.0 s (road), -9.8 s (straight-line Kolesar).
  Every one of the 5 boroughs gains under every model, from about -5 s (Enfield, Hillingdon) to -25 s (Redbridge, road).
- No borough is 1 s or more slower under any model; the worst is +0.6 s.
- Using only sites next to commercial or industrial land (Rule B) and searching again: -12.2 / -14.3 / -10.8 s, still with no borough worse.
  This passes both pass marks, **but this version was added after the first results, so it is not one of the tests fixed in advance.** The fixed
  version of the land rule (failing sites shifted to the nearest allowed site) gets -10.7 / -11.4 / -7.1 s and one borough 2.2 s worse under
  Kolesar, so it misses both marks under Kolesar.
- The cost: London's overall average improves by 2.4 s instead of 4.2 s for the same search aimed at the whole city.
- Honest result, against the tests fixed in advance: up to 20 moves (15 used) passes both marks; up to 10 moves misses the first by 0.2 s under
  Kolesar; no fixed land-rule test passes in full. The re-searched Rule B layout passes both, but only as an analysis added afterwards; 2026 is
  where it can be tested as a fixed prediction. Corrected 2026-11-05 (an earlier version called the Rule B result a pass of the fixed test).
- How much comes from targeting: the planner's ordinary Move 10 already speeds these boroughs up by -10.2 / -9.5 / -5.7 s. Targeting adds 1.7 /
  2.5 / 4.1 s there and keeps every borough protected under all three models, which the planner's layout did not.
- Rerun check: the main case was rerun with the final script and gave identical layouts and scores.

**Why A wins:**

1. **It has a real, testable hypothesis with a pass mark written down first.** That is ISEF's Research Question (10 points) and Synopsys's
   Scientific Thought. "How many stations must move" has no prediction that could fail.
2. **Its main result is large where it matters and survives every check we have.** About 10-12 s in the slowest boroughs under all three
   models, against 2.5-6 s London-wide for the general Move 10. It is also the first layout whose "no borough worse" protection holds under
   the two models that did not choose it. The general planner's protection failed under road and Kolesar (up to +8 s).
3. **The impact is easy to state:** the people who wait longest get help, no borough's average gets worse, and it costs London about 2 s of its average. Single calls can still lose: about 2 % of all calls are 30+ s slower.
   That serves ISEF's Creativity and Impact (20) and the interview's "potential impact".
4. **It is the easiest to defend in an interview (35 % of ISEF).** The method is: pick the slowest boroughs from recorded data, search for moves,
   reject any move that slows a borough, check under three models and land rules, score on a year not used. Ricky and Sarp can explain every step.
5. **It uses the whole project:** the travel model, busy engines, the solver, the three-model rule, the land rules and the held-out year.

**Weak points to state openly:** results are modelled, not observed; the travel model is least accurate in outer London, where the target boroughs are (recorded times there run up to about 28 s above the model, section 5.2); the search is not proven optimal; the strictest land rule (site not on
residential land) halves the gain (-7.1 / -7.5 / -4.5 s, with Havering +7 s under Kolesar); the idea of protecting every area is not new in the
literature; under Kolesar the first pass mark was missed by 0.2 s.

## 2. The options compared

Ratings are out of 5, for this project's current evidence. ISEF weights in brackets.

| Option | Question and design (25) | Execution and evidence (20) | Creativity and impact (20) | Interview / defend (35) | Verdict |
|---|---|---|---|---|---|
| **A. Slowest areas faster, none slower** | 5: fixed target rule and pass marks | 4: passes under three models and Rule B; misses Rule C | 4: clear benefit to the worst-served | 5: simple story, every step explainable | **Best** |
| "How many stations must move before it matters" | 3: descriptive, nothing to fail | 4: well tested in 5.5 | 3: "about 15 s needs 46 moves" | 4 | Good supporting section |
| B. Public records reveal unavailable engines | 3: testable, but one fixed prediction failed | 3: AUC 0.72, cause unknown | 5: most original, real recorded effect | 2: hardest to explain | Strong second part of A, or main question if LFB explains the pattern |
| C. Price of fairness | 4 | 4 | 3 | 4 | Already a result inside A (about 1.8 s against the citywide search with the same limits) |
| D. Poorer areas gain most | 2: found after looking | 3 | 4 | 3 | Mention, don't lead |
| G. Better travel model, better layouts? | 4 | 5 | 2: under 1 s difference | 4 | Methods section |
| F, E, H, I | | | | | Supporting or weaker |

The fair tie-break between A and "how many stations" was the three-model rescore of A's targeted layouts with the target fixed first. That
test is now done and A came out ahead.

**Suggested shape of the board and paper:** A as the question; the travel model and the three-model check as the method; "how many stations"
as the scale-up curve; B as the finding that explains part of the remaining delay (the one large effect seen in recorded calls).

## 3. Judging criteria: what we have

Details in `docs/fair_criteria.md`.

| Fair | Have | Missing |
|---|---|---|
| Synopsys | Official 4 x 10 criteria page and the student judging FAQ (science-fair.org) | Rules page for teams and continuation projects (not read) |
| CSEF (California) | Only an old help article: no numerical rubric, panel agreement, PDF presentation plus interview | Current rules and judge guidance: csef.usc.edu would not load (certificate error) |
| ISEF | Official Grand Award criteria page (societyforscience.org), science and engineering rubrics with points | The ISEF Rules and Guidelines book ("ISEF guide"); the criteria page does not link a PDF |

Please upload the ISEF Rules and Guidelines PDF and anything from CSEF (rules, judge guide) if you have them.

Rules that matter now: Synopsys requires a notebook (grades 9-12) and all work done within the past 12 months, so check the project start date.
ISEF judges only this year's new work on continuing projects (Form 7), and every team member must show contribution and understanding.

## 4. Next steps (no schedule)

1. **Commit now** `docs/targeted_areas_rule.md`, `scripts/targeted_areas.py` and `data/figures/placement/targeted/`, so the rule and results
   are timestamped. The rule was written before the run in this session, but nothing in git shows that yet. Be honest about this if asked.
2. **2026 final test:** before touching 2026, write and commit the prediction (for example: the Rule B targeted Move 10 speeds up the same
   5 boroughs by at least 10 s under all three models with no borough 1+ s worse), then score it once. Include the new 14-day cycle prediction
   in the same commit.
3. **Strengthen the layout:** run the exact planner with the target-borough objective and the 0 s borough floor, add street refinement, and
   check whether it beats the swap search. Report whichever is better, scored the same way.
4. **Sensitivity of the target rule:** report the 3 and 7 slowest boroughs as a check, clearly labelled as after the fact.
5. **Rule C:** say plainly that strict non-residential siting halves the gain; show the Rule B layout as the main one.
6. **Notebook:** record the target rule, its date, the pass marks and the partial pass. Copy the 14-day prediction and its failed result too.
7. **Poster:** rebuild around A: a map of the 5 boroughs before and after, the three-model bar chart, and the "no borough worse" chart.
8. **Admin:** Sarp replaces `data/processed/models/travel_time_lfb.joblib` after pulling; check the 12-month rule; wait for LFB's reply.

## 5. What was done in this session (handoff)

- Read `docs/handoff_2026-11-05.md` (earlier context: results, options, preferences).
- Fetched official criteria and wrote `docs/fair_criteria.md`.
- Wrote the target rule before any run: `docs/targeted_areas_rule.md`.
- New script `scripts/targeted_areas.py` (listed in `scripts/README.md`). Best-swap search that keeps, at every step, every borough no worse
  (0 s), the London average and the slowest 10 % no worse. A first version that required every single swap to keep the limits found no move,
  and a penalty version was replaced before it produced results; the final version also tries relocations within 3 km of the closed station.
  `--rule rule_b|rule_c` searches only sites passing that land rule.
- Outputs: `data/figures/placement/targeted/` (scores.csv, boroughs.csv, targets.csv, layouts/, summary.json) and `targeted/rule_b/`,
  `targeted/rule_c/`.
- `docs/results.md`: new section 5.6; "Last updated" set to 2026-11-05.
- Nothing committed. 2026 not used. No payments.
- Run: `PYTHONPATH=src .venv/bin/python scripts/targeted_areas.py` (about 25 minutes), then with `--rule rule_b` and `--rule rule_c`.
