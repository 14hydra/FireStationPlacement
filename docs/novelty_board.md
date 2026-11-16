# What is new? (board / interview card)

Use plain words. Do **not** lead with “we invented fire-station placement.”

## One-sentence novelty

We learn how London’s streets slow fire engines, check that on a real station closure, write a pass/fail promise *before* seeing 2026, and also show a recorded delay when the local engine looks free but isn’t sent.

## Three easy points to say

1. **Promise first, then test**  
   Targets and pass marks were fixed before 2026. Targeted Move-10 PASSed: about **13–16 s** faster in the five slowest boroughs, **no borough ≥1 s worse**.

2. **Hidden delay in the records**  
   On about **13%** of calls the local station looks free and closer, but another arrives first — about **+148 s** slower. Moving stations alone cannot fix all of that.

3. **Streets + a real closure**  
   Travel times use street traits from real LFB trips (not distance alone), and we check against the **2014** closures with a learned dispatch model.

## What not to claim as the new part

- “Putting stations on a map” (old OR literature)
- “Our MAE is a few seconds better” as the whole story
- NYC open data as full proof (it doesn’t name which firehouse responded)

## Figures

- `data/figures/final/14_project_significance.png` — promise + gap + results  
- `data/figures/final/15_what_is_new.png` — interview one-pager  

Regenerate: `PYTHONPATH=src .venv/bin/python scripts/build_project_infographic.py`
