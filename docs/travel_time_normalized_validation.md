# Travel-time validation: normalized error + full histogram

Advisor note: absolute MAE alone is incomplete. A **60 s** miss on a **5-minute**
trip is a **20%** relative error. Prefer **normalized error** and the **full residual
histogram**, and call out extreme outliers.

## What we report

Script: `scripts/validate_travel_time_normalized.py`  
Artifacts: `data/figures/validation/travel_time_normalized/`

| Metric | Meaning |
|---|---|
| MAE / MedAE | Absolute seconds (as before) |
| MAPE / median APE | \|error\| / actual × 100% |
| sMAPE | Symmetric percent error |
| % within 10/20/30% | Share of trips with small relative error |
| \|error\| > 400 s | Extreme-outlier count |

## 2025 results (production LightGBM)

| Slice | MAE | MedAE | MAPE | Median APE | Within 20% | \|err\| > 400 s |
|---|---:|---:|---:|---:|---:|---:|
| All trips | 58.9 s | 36.1 s | 22.4% | 14.3% | 64% | **1,862** (0.9%) |
| Placement (1st, home, free) | 46.6 s | 30.1 s | 22.7% | 15.5% | 61% | 422 (0.5%) |

Dropping the >400 s tail only lowers all-trip MAE to 54.4 s — the bulk error is not
only those outliers.

Figure: `error_distribution.png` (signed residuals, absolute errors, placement APE
histogram, summary panel).

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_travel_time_normalized.py
```
