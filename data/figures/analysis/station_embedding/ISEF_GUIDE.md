# ISEF guide — reading the station picture

**Start with** `06_isef_story.png` (one page). Everything else is backup detail.

## The research question (one sentence)

Can a street-aware travel-time model place London fire stations better than
distance-only rules, when scored on real 2025 calls under LFB’s 6 / 10-minute standard?

## What this figure pack adds

We summarised **45 numbers for each of London’s 102 stations**
(busyness, engines, drive times, model error) into a 2D picture so judges can see:

1. **Busy centres vs quiet outskirts** — inner stations face dense streets and high utilisation.
2. **Capacity mismatches** — some single-engine houses are stretched while some two-engine
   houses have spare capacity.
3. **Where the travel-time model struggles** — long outer drives and dense central shorts.

## Station types (colours)

- **Outer long-drive** (33 stations)
- **Outer two-pump** (24 stations)
- **Busy multi-pump** (14 stations)
- **Strained 1-pump urban** (29 stations)
- **Central demand outliers** (2 stations)

## Capacity mismatch (actionable)

- Busy 1-engine stations (21): Euston, Islington, Shoreditch, Chelsea, Bethnal Green, Kensington, Fulham, Poplar
- Quieter 2-engine stations (23): Addington, Kingston, Orpington, Sidcup, Twickenham, Bromley, Sutton, Stanmore

## How we used this to improve the models

- **Travel time:** rush × street-density features, sample weights on hard trips,
  and calibration for **outer long drives** (highest-error slice).
  Quick A/B: overall MAE ≈ flat; outer long-drive MAE improved ~0.4 s.
  See `docs/model_improvements_from_embedding.md`.
- **Resize planner:** blends utilisation into pump moves so engines leave quiet
  two-pump houses and go to busy one-pump houses.

PCA note for judges who ask: the first two axes capture **50%** of the 45-metric pattern (32% + 18%).
