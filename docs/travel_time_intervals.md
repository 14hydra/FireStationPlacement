# Travel-time prediction intervals (quantile + conformal)

Point MAE alone understates uncertainty: the same route varies ~40 s trip-to-trip.
This build adds **calibrated ranges** around drive time for evaluation and training checks.

## Method

1. **Quantile LightGBM** at p10 / p25 / p50 / p75 / p90 on the same placement-safe
   features as the production median model (`FINAL_GROUPS`).
2. **Asymmetric conformalized quantile regression (CQR)** on 2024
   (models trained on 2021–23): expand each band so empirical coverage matches the
   target (50% = p25–p75, 80% = p10–p90).
3. Refit quantiles on **2021–2024**, apply the 2024 CQR adjustments, score **2025**.

Code: `src/firestation/conformal_quantiles.py`,
`scripts/train_travel_time_intervals.py`.

Artifacts:

- `data/processed/models/travel_time_lfb_intervals.joblib`
- `data/figures/analysis/travel_time_intervals/` (`summary.json`, `coverage.png`, `mae_table.csv`)

Predict:

```python
from firestation.conformal_quantiles import predict_intervals
out = predict_intervals(bundle, trips, level="80")  # lo / mid / hi
```

## Held-out 2025 results

| Slice | p50 MAE | 50% cov / width | 80% cov / width |
|---|---:|---|---|
| All trips | 58.7 s | 0.504 / 76 s | 0.806 / 163 s |
| First engine, home, not busy | 46.4 s | 0.515 / 66 s | 0.821 / 143 s |

Raw bands under-cover (placement 0.46 / 0.75); CQR hits the targets. Production median on
the same placement trips: 46.6 s — keep it for point scoring; use intervals for reliable
ranges in evaluation.

## How to read the numbers

- **Coverage** should sit near 0.50 / 0.80 after conformalization. Raw quantiles usually
  under-cover; CQR fixes that by widening the band.
- **Width** is the reliability cost: an 80% placement interval is about **2.4 minutes**
  wide (median).
- **p50 MAE** stays tied with production; intervals do not replace it for placement scoring.

## Run

```bash
PYTHONPATH=src .venv/bin/python scripts/train_travel_time_intervals.py
PYTHONPATH=src .venv/bin/python scripts/train_travel_time_intervals.py --quick   # 600 trees
```
