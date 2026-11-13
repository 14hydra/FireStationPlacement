# External data to validate travel-time prediction (megacities)

Goal: find open datasets like LFB’s (known origin station, destination, recorded drive/attendance seconds) for cities on the megacity map.

**What LFB gives us that most cities do not:** mobilise→arrive split (or drive seconds), **which station the engine left from**, and a geocoded incident location.

## Verdict (short)

| City | Usable for travel-time model validation? | Why |
|---|---|---|
| **London** | **Yes (already)** | LFB mobilisation + incidents |
| **New York** | **Partial — run** | Open FDNY travel seconds + official alarm-box lat/lon + firehouses; **no responding company** |
| Tokyo | No (aggregate stats only) | White-paper CSVs: counts/avg activity times, not trip-level OD |
| Delhi / Mumbai / Beijing / Shanghai | No public trip-level CAD found | Papers use closed agency data |
| Mexico City | Weak | Emergency registry has lat/lon + “attended” flags; **no travel seconds / station origin** |
| São Paulo / Cairo / Dhaka / Osaka | No suitable open CAD found | — |

## New York (FDNY) — weak-label result

Script: `scripts/validate_nyc_fdny_travel.py`  
Artifacts: `data/figures/validation/nyc_fdny/` (`summary.json`, `mae_table.csv`, `scatter_kolesar.png`)

**Sources**

- Incidents: [Fire Incident Dispatch Data](https://data.cityofnewyork.us/Public-Safety/Fire-Incident-Dispatch-Data/8m42-w767) (`8m42-w767`) — `INCIDENT_TRAVEL_TM_SECONDS_QY`
- Destinations: [In-Service Alarm Box Locations](https://data.cityofnewyork.us/Public-Safety/In-Service-Alarm-Box-Locations/v57i-gtxb) (`v57i-gtxb`) — lat/lon by `borobox`
- Origins: [FDNY Firehouse Listing](https://data.cityofnewyork.us/Public-Safety/FDNY-Firehouse-Listing/hc8x-tcnd) (`hc8x-tcnd`) — **nearest** house (responding company not in open CAD)

**Design:** filter valid travel 30–1200 s with ≥1 engine; join alarm box → dest; **nearest Engine house** → crow km; road km ≈ crow × 1.3. Score on holdout:
- London-default Kolesar (transfer)
- NYC-refit Kolesar
- Distance + hour + borough linear model

**Headline slice:** structural / non-structural **fires** with travel **&lt; 600 s** (cleaner than medical calls under nearest-station labels).

**Headline numbers** (see `summary.json`):

| Protocol | n | MAE (s) | Corr |
|---|---:|---:|---:|
| All calls · NYC Kolesar | ~53k | 84.4 | 0.24 |
| All calls · dist+hour+borough | ~53k | ~73 | ~0.40 |
| Fires &lt;10 min · NYC Kolesar | ~4.1k | 57.7 | 0.34 |
| **Fires &lt;10 min · dist+hour+borough** | **~4.1k** | **55.3** | **0.44** |
| Fires &lt;10 min · constant | ~4.1k | 62.3 | — |

**Reading (honest):**

- Restricting to fires and adding hour/borough lifts correlation from **0.24 → 0.44** and MAE from **84 → 55 s**.
- Still weak labels: the ceiling remains the **missing responding company**.
- London Kolesar still under-predicts absolute NYC times; do **not** claim a LightGBM transfer.
- This does **not** validate megacity placement savings.

## Other cities (quick notes)

- **Tokyo Fire Department open data:** annual white-paper tables. Not per-trip OD.
- **Beijing / Mumbai / Delhi:** academic studies use **non-public** agency extracts.
- **Mexico City RUSE:** locations without travel seconds / station origin.
- **Chicago:** IG says CFD public data are **not adequate** to measure response times.

## Recommendation

1. Keep **London LFB** as the primary validation.  
2. Cite NYC as an **external weak-label** check (open CAD exists; distance correlates; nearest-station ceiling).  
3. Do not treat megacity Kolesar placement savings as validated travel-time accuracy outside London.
