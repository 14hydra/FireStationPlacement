# Model improvements from the station embedding

Source picture: `data/figures/analysis/station_embedding/06_isef_story.png`
(ISEF one-pager) and `ISEF_GUIDE.md`.

## What the picture told us

| Finding | Model implication |
|---|---|
| PC1 = util + signals/km + junctions/km | Dense central trips need street-density × time-of-day features |
| PC3 = high MAE + long outer drives | Emphasise / calibrate long sparse routes |
| 21 busy 1-engine vs 23 quiet 2-engine | Resize should blend utilisation, not only nearest-catchment load |

## Changes tried

### Travel-time (`src/firestation/travel_time_model.py`, `scripts/train_travel_time_lfb.py`)

Tested, and off by default because they did not improve the model (see Honest results):

1. **Interaction features** (placement-safe): `rush_x_signals_per_km`, `rush_x_junctions_per_km`,
   `night_x_signals_per_km`, `dense_short`, `long_sparse` — group `interaction`
   (`--interactions`).
2. **Sample weights**: up-weight long trips and dense-short trips; soft-downweight impossible
   speeds (`--sample-weights`).
3. **Regime bias calibration**: post-hoc correction for `long_sparse` trips only
   (`--regime-bias`; now fitted out of sample, see below). `predict()` applies it when a
   saved bundle has `regime_bias`.

### Resize / capacity (`src/firestation/station_capacity.py`, `fire_station_location.py`)

- `plan_resize_capacity(..., util_pct=, util_weight=0.35)` blends observed utilisation into
  donor/receiver ranking so busy 1-pump houses are preferred receivers and quiet 2-pump houses
  preferred donors.
- Utilisation loaded from `data/figures/data_exploration/station_busyness/station_busyness.csv`.

## Honest results

**Like-for-like test** (`scripts/compare_travel_time_variants.py`,
`data/figures/analysis/model_improvement/fair_comparison.{csv,json}`, 2026-10-23): every
variant uses the production settings (2,110 trees, same seed), trains on 2021–2024 and is
scored on all 200,624 trips from 2025. MAE in seconds:

| Variant | All | Dense short (central) | Long sparse (outer) | Over 6 km | Rush hour |
|---|---:|---:|---:|---:|---:|
| A production | **57.80** | **49.40** | 78.78 | **103.30** | 58.46 |
| B + interactions | 57.85 | 49.43 | 78.83 | 103.55 | 58.45 |
| C + sample weights | 57.82 | 49.41 | 78.70 | 103.69 | 58.44 |
| D + both | 57.82 | 49.42 | 78.64 | 103.37 | 58.44 |
| E D + correction fitted on its own training trips | 57.80 | 49.42 | 78.51 | 103.30 | 58.40 |
| F D + correction fitted out of sample (2024) | 57.85 | 49.42 | 78.92 | 103.53 | 58.42 |

Every variant is within 0.05 s of production overall and within 0.3 s on the hard slices,
which is within the noise of a single seed. So the production model is unchanged
(`travel_time_lfb.joblib`, 4 seeds, 57.6 s) and the three changes stay as options.

Why the earlier numbers looked better: the "58.9 s vs 60.2 s" comparison used a 1-seed,
900-tree model against a different baseline, not the 4-seed, 2,110-tree production model
(57.6 s). The bias correction was fitted on the model's predictions for its own training
trips, where tree models are much closer to the truth than on new trips: it measured
−5.7 s there against −13.2 s on unseen 2024 trips. And `calibrate_travel_time_regimes.py`
(removed) scored trips without their street features (79.9 s MAE), measured a −49 s bias
from that, and wrote it into the production model file; any copy of
`travel_time_lfb.joblib` with a `regime_bias` entry should be retrained.

**Dispatch** (figure 09): cover-density features + temperature calibration → area
correlation **0.57** (nearest 0.48). Resize `util_weight` default **0.45** so busy
1-pump houses rank higher as pump receivers.

## Figures for judges

| File | Role |
|---|---|
| `06_isef_story.png` | One-page story |
| `ISEF_GUIDE.md` | Plain-language walkthrough |
| `00_summary.png` | Full embedding dashboard |
| `05_archetypes.png` | Capacity mismatch callouts |
