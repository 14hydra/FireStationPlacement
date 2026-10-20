"""Street-level refinement: move each newly opened site to the best nearby road junction.

The solver picks new sites from demand-cell centres (0.9 km grid). A real station sits on
a street, so after solving, each new site is moved, one at a time, to the junction within
``radius_km`` that lowers the layout's objective the most:

1. Candidates: junctions on A / B / local distributor roads (``RouteFeatureRouter.site_nodes``;
   not motorways or residential side streets) within ``radius_km``.
2. Shortlist: every candidate is scored quickly. One Dijkstra run per junction on posted
   speed limits gives the fastest-route time to every demand cell; a straight-line fit of
   the site's LightGBM times on its own fastest-route times converts these to attendance
   seconds, and the layout is scored with each junction swapped in (nearest engine).
3. Final pick: the ``shortlist`` best junctions get full LightGBM travel times
   (``build_travel_matrix``) and are scored with the planner's real objective
   (busy engines included when on). A move is kept only if it beats the current site
   and keeps the 10-minute cap.

Today's stations are real buildings and are not moved.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


def _fastest_rows(router, nodes: np.ndarray, cell_nodes: np.ndarray, chunk: int = 16) -> np.ndarray:
    from scipy.sparse.csgraph import dijkstra

    out = np.empty((len(nodes), len(cell_nodes)))
    for s in range(0, len(nodes), chunk):
        d = dijkstra(router.csr_time, directed=True, indices=nodes[s:s + chunk])
        out[s:s + chunk] = d[:, cell_nodes]
    return out


def _street(router, node: int) -> str:
    """Name(s) of the main roads meeting at a junction, from OpenStreetMap."""
    G, nid = router.G, router.nodes[int(node)]
    names = []
    for _, _, d in list(G.out_edges(nid, data=True)) + list(G.in_edges(nid, data=True)):
        n = d.get("name")
        for x in (n if isinstance(n, list) else [n]):
            if isinstance(x, str) and x and x not in names:
                names.append(x)
    return " / ".join(names[:2]) if names else f"junction {int(nid)}"


def refine_sites(
    A: np.ndarray,
    candidates: pd.DataFrame,
    cells: pd.DataFrame,
    selected: Sequence[int],
    movable: Sequence[int],
    weights: np.ndarray,
    *,
    make_rows: Callable[[pd.DataFrame], np.ndarray],
    objective: Callable[[np.ndarray, list[int], pd.DataFrame], float],
    cap_ok: Callable[[np.ndarray, list[int], pd.DataFrame], bool] | None = None,
    radius_km: float = 1.0,
    shortlist: int = 6,
    passes: int = 1,
) -> dict[str, Any]:
    """Move each site in ``movable`` to its best nearby junction (see module docstring).

    ``make_rows(sites_df)`` returns attendance seconds (sites x cells) for new sites;
    ``objective(A, idx, candidates)`` scores a layout (lower is better) on a matrix whose
    rows are ``candidates``. Returns the extended candidates / matrix, new layout and a
    log of moves.
    """
    from .ml_scorer import _router

    t0 = time.perf_counter()
    r = _router()
    w = np.asarray(weights, dtype=float)
    W = max(float(w.sum()), 1e-9)
    cell_nodes, _ = r.snap(cells["lat"].to_numpy(float), cells["lon"].to_numpy(float))
    site_nodes = r.site_nodes
    sl, so = r.node_lat[site_nodes], r.node_lon[site_nodes]
    kx = 111.32 * np.cos(np.radians(51.5))

    cand = candidates.copy()
    A = np.asarray(A, dtype=float)
    idx = [int(i) for i in selected]
    moving = [int(i) for i in movable]
    log = []
    current_val = objective(A, idx, cand)
    for p in range(passes):
        changed = False
        for k, site in enumerate(list(moving)):
            lat0, lon0 = float(cand.at[site, "lat"]), float(cand.at[site, "lon"])
            near = np.hypot((so - lon0) * kx, (sl - lat0) * 110.574) <= radius_km
            nodes = site_nodes[near]
            if not len(nodes):
                continue
            # 1. quick score: fastest-route time, calibrated to this site's LightGBM row
            f_nodes = _fastest_rows(r, nodes, cell_nodes)
            (own_node,), _ = r.snap(np.array([lat0]), np.array([lon0]))
            f_own = _fastest_rows(r, np.array([own_node]), cell_nodes)[0]
            ok = np.isfinite(f_own) & (f_own < 900)
            b, a = np.polyfit(f_own[ok], A[site, ok], 1) if ok.sum() > 10 else (1.0, 60.0)
            proxy = a + b * np.where(np.isfinite(f_nodes), f_nodes, 1e6)
            others = [i for i in idx if i != site]
            base = A[others].min(axis=0)
            quick = (w * np.minimum(base[None, :], proxy)).sum(axis=1) / W
            top = nodes[np.argsort(quick)[:shortlist]]
            # 2. full LightGBM rows for the shortlist, scored with the real objective
            new = pd.DataFrame({"site_name": [f"New site on {_street(r, n)}" for n in top],
                                "layer": "street_junction", "lat": r.node_lat[top], "lon": r.node_lon[top],
                                "is_existing": False})
            rows = make_rows(new)
            start = len(cand)
            new.insert(0, "candidate_id", np.arange(start, start + len(new)))
            cand = pd.concat([cand, new], ignore_index=True)
            A = np.vstack([A, rows])
            best_val, best_row = current_val, None
            for j in range(len(top)):
                trial = [start + j if i == site else i for i in idx]
                if cap_ok is not None and not cap_ok(A, trial, cand):
                    continue
                v = objective(A, trial, cand)
                if v < best_val - 1e-6:
                    best_val, best_row = v, start + j
            if best_row is not None:
                moved_m = 1000 * float(np.hypot((cand.at[best_row, "lon"] - lon0) * kx,
                                                (cand.at[best_row, "lat"] - lat0) * 110.574))
                log.append({"pass": p + 1, "from": site, "to": best_row, "from_lat": lat0, "from_lon": lon0,
                            "to_lat": float(cand.at[best_row, "lat"]), "to_lon": float(cand.at[best_row, "lon"]),
                            "moved_m": round(moved_m), "gain_s": round(current_val - best_val, 3),
                            "n_junctions_considered": int(len(nodes))})
                idx = [best_row if i == site else i for i in idx]
                moving[k] = best_row
                current_val = best_val
                changed = True
        if not changed:
            break
    # A site no junction could improve on keeps its grid position; name it after its nearest street.
    for site in moving:
        if not str(cand.at[site, "site_name"]).startswith("New site"):
            (node,), _ = r.snap(np.array([float(cand.at[site, "lat"])]), np.array([float(cand.at[site, "lon"])]))
            cand.at[site, "site_name"] = f"New site near {_street(r, node)}"
    return {"candidates": cand, "A": A, "indices": idx, "moved_sites": moving, "value": current_val,
            "log": log, "solve_s": round(time.perf_counter() - t0, 1)}
