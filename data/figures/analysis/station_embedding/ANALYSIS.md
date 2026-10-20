# Station metric embedding — analysis

102 current LFB stations · **45 standardised metrics** · PCA / UMAP / t-SNE.
Figures: `data/figures/analysis/station_embedding/`.

## How to read the figures

| File | What it shows |
|---|---|
| `00_summary.png` | PCA + UMAP + map of archetypes, plus a z-score profile heatmap |
| `01_pca.png` | Same PCA coloured by pumps and by incidents/pump (actionable axes) |
| `02_umap.png` / `02b_tsne.png` | Local neighbourhood views (axes have no units) |
| `03_feature_loadings.png` | What PC1/PC2 actually load on; share by data source |
| `04_layout_pca.png` | Separate PCA of *named placement layouts* (not stations) |
| `05_archetypes.png` | Named groups + capacity-mismatch callouts |

## What the 2D view is

Every joinable per-station number in the project (capacity, busyness, 2025 trips,
LightGBM error, area maps, plan flags) is z-scored and projected.
**PCA PC1+PC2 explain 50%** of variance (32% + 18%). k-means on the leading PCs picks **k=5** (silhouette 0.25). UMAP/t-SNE are for local neighbourhoods only.

## Axis reading (PCA)

- **PC1 (32%) — busyness / street density:** +utilisation %, +utilisation 00-06, +utilisation at peak hour, +signals per km, +junctions per km. Right = high utilisation and dense junctions/signals (inner London); left = longer median drives, quieter outskirts.
- **PC2 (18%) — capacity vs strain:** +multi-pump dispatch rate, +pumps, +multi-pump dispatches, −% of year all engines out, −% own-area calls covered by others. Up = two-pump / multi-dispatch houses; down = single-pump stations with high all-out / neighbour-cover rates.
- **PC3 (11%) — model hard cases + volume:** +Kolesar MAE (2025), +LightGBM MAE (2025), +incidents (2024-25), +incidents per pump. Stations where both LightGBM and Kolesar struggle, plus sheer incident volume.

The pump-coloured PCA (`01_pca.png`) almost splits horizontally: almost all 2-pump stations sit above PC2 ≈ 0. The incidents/pump colouring darkens on the **right** — Soho, Euston, Shoreditch are the load extremes.

## Archetypes

- **Outer long-drive** (n=33): mean util 7%, mean pumps 1.0, median drive 300s · e.g. Heathrow, Hayes, Ealing, Beckenham, Bexley
- **Outer two-pump** (n=24): mean util 6%, mean pumps 2.0, median drive 288s · e.g. Ilford, Plumstead, Dagenham, Enfield, Southall
- **Busy multi-pump** (n=14): mean util 9%, mean pumps 2.0, median drive 250s · e.g. Lambeth, Paddington, Tottenham, Kentish Town, Hammersmith
- **Strained 1-pump urban** (n=29): mean util 11%, mean pumps 1.0, median drive 264s · e.g. Dowgate, Shoreditch, Chelsea, Bethnal Green, Stratford
- **Central demand outliers** (n=2): mean util 15%, mean pumps 1.5, median drive 264s · e.g. Soho, Euston

On the map (`00_summary.png`), outer long-drive / outer two-pump ring London; strained 1-pump and busy multi-pump fill the inner boroughs; the central outliers are the densest cores.

## Capacity mismatches the embedding surfaces

- **21 high-utilisation 1-pump stations** (top quartile util): Euston, Islington, Shoreditch, Chelsea, Bethnal Green, Kensington, Fulham, Poplar, Stratford, Holloway.
- **23 lower-utilisation 2-pump stations** (bottom 40% util): Addington, Kingston, Orpington, Sidcup, Twickenham, Bromley, Sutton, Stanmore, Finchley, Edmonton.

These are the stations the **resize** planner tends to upsize / downsize. They sit in opposite PCA quadrants: bottom-right (busy · 1-pump strain) vs top-left (quiet · spare capacity).

## Layout embedding (separate matrix)

Named placement layouts (recommended / exact / scored) form a second PCA (`04_layout_pca.png`). Layout PC1 is essentially **seconds saved vs today** — redesign / large replace at one end, status quo at the other.

## Honesty

- Correlated demand / util columns inflate PC1; always check loadings (`03_feature_loadings.png`).
- Area-map cells are assigned by crow-fly nearest station, not LFB ground.
- Clusters are a reading aid (silhouette 0.25), not operational station types.
- 2014 closed stations and city-level tradeoff tables are not in the station matrix.
