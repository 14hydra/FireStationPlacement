# Statistical PDF travel-time model (advisor recommendation)

Advisor framing: do not predict each trip’s travel time; build a **statistical model**
that samples times matching the PDF of real observed travel times (by street / road
class), then compare **distributions**.

## What we built

Because LFB training rows have whole-trip `drive_s` and OSM **route length shares**, not
GPS times per edge:

1. Fit empirical **pace PDFs** (s/km) per road class from 2021–2024 trips (length-share
   attribution of each trip’s pace).
2. Optionally fit coarser **regime** PDFs: distance bin × dominant class × rush hour.
3. For a 2025 trip, sample paces / times and recombine with that trip’s shares.
4. Point prediction = **median of samples**; also score CRPS and interval coverage.

Code: `src/firestation/street_time_pdf.py`. Compare script:
`scripts/compare_pdf_travel_time.py` → `data/figures/analysis/pdf_travel_time/`.

## Held-out 2025 results (n=200,624)

| Model | MAE (s) | MedAE (s) | Dist. score |
|---|---:|---:|---|
| Constant | 111.9 | 85.0 | — |
| PDF street-class | 84.0 | 56.7 | CRPS 63.9 s |
| PDF regime | 72.2 | 47.5 | CRPS 53.7 s |
| Kolesar (road km) | 68.7 | 45.8 | — |
| **LightGBM** | **58.9** | **36.1** | MAE as point proxy |

First-engine / not-busy: LightGBM 47.4 s vs PDF regime 56.0 s vs street-class 64.1 s.

PDF 50% / 80% predictive intervals under-cover slightly (street-class 0.39 / 0.66;
regime 0.48 / 0.77 vs nominal 0.50 / 0.80)—expected with whole-trip attribution noise.

## Verdict

**LightGBM is better for point travel-time prediction** used by placement. Keep it as the
production scorer. The PDF model is the right tool when the goal is a **reusable
simulated travel-time environment** (sample from city-specific PDFs; swap parameters for
another city) rather than lowest MAE.

Rerun:

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_pdf_travel_time.py
```
