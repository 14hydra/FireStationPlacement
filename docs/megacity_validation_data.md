# External data to validate travel-time prediction (megacities)

Goal: find open datasets like LFB’s (known origin station, destination, recorded drive/attendance seconds) for cities on the megacity map.

**What LFB gives us that most cities do not:** mobilise→arrive split (or drive seconds), **which station the engine left from**, and a geocoded incident location.

## Verdict (short)

| City | Usable for travel-time model validation? | Why |
|---|---|---|
| **London** | **Yes (already)** | LFB mobilisation + incidents |
| **New York** | **Partial — best next candidate** | Open FDNY travel seconds + firehouse list; **no responding company**; dest only ZIP / alarm-box street |
| Tokyo | No (aggregate stats only) | White-paper CSVs: counts/avg activity times, not trip-level OD |
| Delhi / Mumbai / Beijing / Shanghai | No public trip-level CAD found | Papers use closed agency data |
| Mexico City | Weak | Emergency registry has lat/lon + “attended” flags; **no travel seconds / station origin** |
| São Paulo / Cairo / Dhaka / Osaka | No suitable open CAD found | — |

## Best option: New York (FDNY)

**Fire Incident Dispatch Data** (NYC Open Data `8m42-w767`):

- https://data.cityofnewyork.us/Public-Safety/Fire-Incident-Dispatch-Data/8m42-w767  
- Key field: `INCIDENT_TRAVEL_TM_SECONDS_QY` (assignment → first on scene)  
- Also: timestamps, borough, ZIP, alarm box number + street text  
- **Missing vs LFB:** which engine company / firehouse responded; exact lat/lon (HIPAA aggregation)

**FDNY Firehouse Listing** (`hc8x-tcnd`): station names + lat/lon  
https://data.cityofnewyork.us/Public-Safety/FDNY-Firehouse-Listing/hc8x-tcnd

**EMS Incident Dispatch Data** (`76xm-jjuj`): same style of travel seconds for ambulances (not fire engines).

### Realistic NYC validation design (weaker than LFB)

1. Geocode `alarm_box_location` (or ZIP centroid) → destination.  
2. Assume **nearest FDNY firehouse** as origin (or nearest with an engine).  
3. Predict travel with London LightGBM (transfer) and/or Kolesar on OSM road km.  
4. Compare to `INCIDENT_TRAVEL_TM_SECONDS_QY` (filter valid indicators / 30–1200 s).  

This tests **transfer** of the street/time model, not a fair “train on NYC” experiment, and nearest-station error will inflate MAE (same issue we saw on London 2014 closures before the dispatch model).

## Other cities (quick notes)

- **Tokyo Fire Department open data:** annual white-paper tables (出場件数, average activity time). Not per-trip origins/destinations.  
- **Beijing / Mumbai / Delhi:** academic studies report response times from **non-public** agency extracts.  
- **Mexico City RUSE** (datos.cdmx.gob.mx): emergency situations with coordinates; not a fire CAD travel-time table.  
- **Chicago:** IG explicitly says CFD data are **not adequate to measure response times** publicly.

## Recommendation

1. Keep **London LFB** as the primary validation.  
2. If we want a second city: build a **NYC FDNY weak-label** pipeline (nearest firehouse → alarm-box dest → travel seconds) and report MAE with that caveat.  
3. Do not treat megacity Kolesar placement savings as validated travel-time accuracy outside London.
