# Model improvements from the station embedding

Source picture: `data/figures/analysis/station_embedding/06_isef_story.png`
(ISEF one-pager) and `ISEF_GUIDE.md`.

## What the picture told us

| Finding | Model implication |
|---|---|
| PC1 = util + signals/km + junctions/km | Dense central trips need street-density × time-of-day features |
| PC3 = high MAE + long outer drives | Emphasise / calibrate long sparse routes |
| 21 busy 1-engine vs 23 quiet 2-engine | Resize should blend utilisation, not only nearest-catchment load |

## Changes shipped

### Travel-time (`src/firestation/travel_time_model.py`, `scripts/train_travel_time_lfb.py`)

1. **Interaction features** (placement-safe): `rush_x_signals_per_km`, `rush_x_junctions_per_km`,
   `night_x_signals_per_km`, `dense_short`, `long_sparse` — group `interaction` in `FINAL_GROUPS`.
2. **Sample weights**: up-weight long trips and dense-short trips; soft-downweight impossible speeds.
3. **Regime bias calibration**: post-hoc correction for `long_sparse` only (outer hard cases).
   Wired into `predict()` when the saved bundle has `regime_bias`.

Fast A/B: `scripts/ab_travel_time_embedding.py` → `data/figures/analysis/model_improvement/`.

### Resize / capacity (`src/firestation/station_capacity.py`, `fire_station_location.py`)

- `plan_resize_capacity(..., util_pct=, util_weight=0.35)` blends observed utilisation into
  donor/receiver ranking so busy 1-pump houses are preferred receivers and quiet 2-pump houses
  preferred donors.
- Utilisation loaded from `data/figures/data_exploration/station_busyness/station_busyness.csv`.

## Honest results (1-seed, 700 trees, 2025 holdout)

From `data/figures/analysis/model_improvement/ab_travel_time.json`:

| Slice | Baseline MAE | Improved + calibrate | Δ |
|---|---:|---:|---:|
| All 2025 trips | 59.04 s | 59.09 s | +0.05 s |
| Dense short (central) | 50.35 s | 50.44 s | +0.09 s |
| **Long sparse (outer)** | **80.52 s** | **80.16 s** | **−0.36 s** |
| Rush hour | 59.80 s | 59.73 s | −0.07 s |

Overall MAE is essentially unchanged in this quick A/B; the gain is concentrated on the
**outer long-drive** slice the embedding flagged (PC3). Full 4-seed bag:
`python scripts/train_travel_time_lfb.py --quick`.

## Figures for judges

| File | Role |
|---|---|
| `06_isef_story.png` | One-page story |
| `ISEF_GUIDE.md` | Plain-language walkthrough |
| `00_summary.png` | Full embedding dashboard |
| `05_archetypes.png` | Capacity mismatch callouts |
