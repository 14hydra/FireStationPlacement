"""Discrete facility location: demand grid, candidate sites, p-median / max-cover.

Historical incidents are binned into a demand grid::

    w_c      = incident weight of grid cell c
    t(s, c)  = response seconds from site s to cell c
    S        = the k sites chosen from the candidates

    objective="response_time"   minimize  E[T] = sum_c w_c * min_{s in S} t(s,c) / sum_c w_c
    objective="coverage"        maximize  C(T) = sum_c w_c * 1[min_{s in S} t(s,c) <= T] / sum_c w_c

``response_time`` is the p-median problem, ``coverage`` is maximal covering
location (Church & ReVelle, 1974). ``solve_exact`` writes either as an integer
program and solves it with HiGHS (via ``scipy.optimize.milp``) to a proven optimum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0


def haversine_km(lon1, lat1, lon2, lat2):
    """Vectorized great-circle distance in km (broadcasts like numpy)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return EARTH_R_KM * 2.0 * np.arcsin(np.sqrt(a))


@dataclass
class TravelTimeSurrogate:
    """Crow-flies -> drive seconds: ``fixed + km * detour / speed``.

    ``detour_factor`` is the classic street-network circuity correction (~1.3 in
    dense grids); ``speed_kph`` is an *effective* door-to-scene speed including
    intersections, so it is well below posted limits.
    """

    speed_kph: float = 28.0
    detour_factor: float = 1.30
    fixed_seconds: float = 15.0  # acceleration + first-intersection overhead
    meta: dict[str, Any] = field(default_factory=dict)

    def seconds(self, lon1, lat1, lon2, lat2):
        km = haversine_km(lon1, lat1, lon2, lat2) * self.detour_factor
        return self.fixed_seconds + km / max(self.speed_kph, 1.0) * 3600.0

    def matrix(self, from_df: pd.DataFrame, to_df: pd.DataFrame) -> np.ndarray:
        """Seconds matrix [len(from_df), len(to_df)] using lon/lat columns."""
        flon = from_df["lon"].to_numpy(dtype=float)[:, None]
        flat = from_df["lat"].to_numpy(dtype=float)[:, None]
        tlon = to_df["lon"].to_numpy(dtype=float)[None, :]
        tlat = to_df["lat"].to_numpy(dtype=float)[None, :]
        return np.asarray(self.seconds(flon, flat, tlon, tlat), dtype=float)


def build_demand_grid(
    incidents: pd.DataFrame,
    *,
    lat_col: str = "dest_lat",
    lon_col: str = "dest_lon",
    cell_km: float = 0.75,
    weight_col: str | None = None,
    severity_col: str | None = None,
    min_incidents: int = 1,
) -> pd.DataFrame:
    """Bin historical incidents into an equal-area-ish demand grid.

    Returns one row per non-empty cell with the demand-weighted centroid (the
    point we actually route to) and the cell's share of citywide demand.

    ``severity_col`` (e.g. ``initial_severity_level_code``, 1 = most acute)
    up-weights acute calls: weight = 1 + (max_level - level) / max_level.
    """
    df = incidents[[c for c in {lat_col, lon_col, weight_col, severity_col} if c]].copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col])
    if not len(df):
        raise ValueError("No incidents with usable coordinates for the demand grid")

    if weight_col and weight_col in df.columns:
        w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        w = pd.Series(1.0, index=df.index)
    if severity_col and severity_col in df.columns:
        sev = pd.to_numeric(df[severity_col], errors="coerce")
        top = float(sev.max()) if sev.notna().any() else np.nan
        if top == top and top > 0:
            w = w * (1.0 + (top - sev.fillna(top)) / top)
    df["_w"] = w

    lat0 = float(df[lat_col].mean())
    dlat = cell_km / 110.574
    dlon = cell_km / (111.320 * max(math.cos(math.radians(lat0)), 0.1))
    df["_iy"] = np.floor(df[lat_col] / dlat).astype(int)
    df["_ix"] = np.floor(df[lon_col] / dlon).astype(int)

    # Demand-weighted centroid per cell = Σ(w·coord) / Σw (no groupby.apply).
    df["_wpos"] = df["_w"].clip(lower=1e-9)
    df["_wlat"] = df[lat_col] * df["_wpos"]
    df["_wlon"] = df[lon_col] * df["_wpos"]
    g = df.groupby(["_ix", "_iy"], sort=False)
    agg = g.agg(
        n_incidents=("_w", "size"),
        weight=("_w", "sum"),
        _wsum=("_wpos", "sum"),
        _wlat=("_wlat", "sum"),
        _wlon=("_wlon", "sum"),
    )
    agg["lat"] = agg["_wlat"] / agg["_wsum"]
    agg["lon"] = agg["_wlon"] / agg["_wsum"]
    cells = agg.drop(columns=["_wsum", "_wlat", "_wlon"]).reset_index()
    cells = cells[cells["n_incidents"] >= max(1, int(min_incidents))]
    cells = cells.sort_values("weight", ascending=False).reset_index(drop=True)
    cells.insert(0, "cell_id", np.arange(len(cells)))
    cells["share"] = cells["weight"] / cells["weight"].sum()
    cells["cell_km"] = cell_km
    cells["cell_dlat"] = dlat
    cells["cell_dlon"] = dlon
    return cells


def add_demand_cell_candidates(
    candidates: pd.DataFrame, cells: pd.DataFrame, *, top_n: int = 60, spread_km: float | None = None
) -> pd.DataFrame:
    """Add demand-cell centroids as extra on-road post candidates.

    ``top_n`` busiest cells, plus (``spread_km``) the busiest cell in every
    ``spread_km`` x ``spread_km`` block, so the solver can also place stations in
    quieter outer areas. Without these the optimizer can only pick from existing
    facilities, which biases posts toward legacy station real estate.
    """
    if (top_n <= 0 and not spread_km) or not len(cells):
        return candidates
    top = cells.head(max(int(top_n), 0))
    if spread_km:
        f = float(spread_km) / float(cells["cell_km"].iloc[0])
        block = (np.floor(cells["_ix"] / f).astype(int).astype(str) + "_"
                 + np.floor(cells["_iy"] / f).astype(int).astype(str))
        spread = cells.loc[cells["n_incidents"].groupby(block).idxmax().to_numpy()]
        top = pd.concat([top, spread[~spread["cell_id"].isin(top["cell_id"])]])
    extra = pd.DataFrame(
        {
            "post_name": ["Demand cell " + str(int(c)) for c in top["cell_id"]],
            "layer": "demand_cell",
            "borough": None,
            "lon": top["lon"].to_numpy(dtype=float),
            "lat": top["lat"].to_numpy(dtype=float),
            "is_anchor": False,
        }
    )
    out = pd.concat([candidates.drop(columns=["candidate_id"]), extra], ignore_index=True)
    out.insert(0, "candidate_id", np.arange(len(out)))
    return out


@dataclass
class SelectionResult:
    objective: str
    indices: list[int]
    value: float
    assignment: np.ndarray  # cell -> position in `indices`
    response_s: np.ndarray  # cell -> travel seconds from its nearest post
    status: str = ""
    optimal: bool = False
    bound: float = float("nan")  # best possible objective value (solver's lower bound)
    gap: float = float("nan")  # relative gap between ``value`` and ``bound``
    solve_s: float = float("nan")
    n_vars: int = 0
    cap: dict = field(default_factory=dict)


TAIL_BETA = 0.9  # CVaR level: average over the slowest 10 % of (weighted) calls


def weighted_cvar(values: np.ndarray, weights: np.ndarray, beta: float = TAIL_BETA) -> float:
    """Average of the slowest (1 - beta) share of demand (Rockafellar–Uryasev CVaR)."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(-v)
    v, w = v[order], w[order]
    tail = (1.0 - beta) * w.sum()
    if tail <= 0:
        return float(v.max())
    take = np.minimum(w, np.maximum(tail - (np.cumsum(w) - w), 0.0))
    return float((v * take).sum() / tail)


def objective_value(
    T: np.ndarray,
    indices: Sequence[int],
    weights: np.ndarray,
    *,
    objective: str,
    threshold_s: float,
) -> float:
    """Lower is better for both objectives (coverage is negated demand covered)."""
    best = T[list(indices), :].min(axis=0)
    total = max(weights.sum(), 1e-9)
    if objective == "coverage":
        return -float((weights * (best <= threshold_s)).sum() / total)
    if objective == "cvar":
        return weighted_cvar(best, weights)
    return float((weights * best).sum() / total)


def _cvar_augment(A, lo, hi, cols, vals, cells, w, W, n_cells, beta=TAIL_BETA):
    """Add CVaR variables: eta (the 90th-percentile level) and u_j >= cost_j - eta.

    Objective becomes eta + sum_j w_j u_j / ((1 - beta) W); returns the new matrix,
    bounds, objective tail and the number of added variables.
    """
    from scipy.sparse import coo_matrix, hstack, vstack

    nv = A.shape[1]
    rows = coo_matrix((np.r_[vals, -np.ones(n_cells), -np.ones(n_cells)],
                       (np.r_[cells, np.arange(n_cells), np.arange(n_cells)],
                        np.r_[cols, np.full(n_cells, nv), nv + 1 + np.arange(n_cells)])),
                      shape=(n_cells, nv + 1 + n_cells))
    A2 = vstack([hstack([A, coo_matrix((A.shape[0], 1 + n_cells))]), rows]).tocsr()
    c_tail = np.r_[1.0, w / ((1.0 - beta) * W)]
    return A2, lo + [-np.inf] * n_cells, hi + [0.0] * n_cells, c_tail, 1 + n_cells


def _add_floors(A, lo, hi, cols, vals, cells, fw, n_cells, max_mean=None, max_cvar=None, group_floor=None):
    """Floor limits on the layout's call-weighted average and slowest-10 % average.

    ``vals`` are the assignment costs at columns ``cols`` for cells ``cells``; both limits
    are linear (CVaR via its own eta / u variables). Returns the new matrix, bounds and the
    number of variables added.
    """
    from scipy.sparse import coo_matrix, hstack, vstack

    FW = max(float(np.sum(fw)), 1e-9)
    added = 0
    if max_cvar is not None:
        nv = A.shape[1]
        A, lo, hi, c_tail, n_new = _cvar_augment(A, lo, hi, cols, vals, cells, fw, FW, n_cells)
        row = coo_matrix((c_tail, (np.zeros(n_new, int), nv + np.arange(n_new))), shape=(1, A.shape[1]))
        A = vstack([A, row]).tocsr()
        lo, hi = lo + [-np.inf], hi + [float(max_cvar)]
        added += n_new
    if max_mean is not None:
        row = coo_matrix((fw[cells] * vals / FW, (np.zeros(len(cols), int), cols)), shape=(1, A.shape[1]))
        A = vstack([A, row]).tocsr()
        lo, hi = lo + [-np.inf], hi + [float(max_mean)]
    if group_floor is not None:
        # One row per group (e.g. borough): its call-weighted average <= its limit.
        group, limits = np.asarray(group_floor[0]), np.asarray(group_floor[1], dtype=float)
        g = group[cells]
        use = g >= 0
        gw = np.bincount(group[group >= 0], weights=fw[group >= 0], minlength=len(limits))
        coef = fw[cells][use] * vals[use] / np.maximum(gw[g[use]], 1e-9)
        rows = coo_matrix((coef, (g[use], cols[use])), shape=(len(limits), A.shape[1]))
        A = vstack([A, rows]).tocsr()
        lo, hi = lo + [-np.inf] * len(limits), hi + list(limits)
    return A, lo, hi, added


def group_means(resp: np.ndarray, fw: np.ndarray, group: np.ndarray, n_groups: int) -> np.ndarray:
    """Call-weighted average per group (cells with group -1 ignored)."""
    ok = group >= 0
    num = np.bincount(group[ok], weights=(fw * resp)[ok], minlength=n_groups)
    den = np.bincount(group[ok], weights=fw[ok], minlength=n_groups)
    return num / np.maximum(den, 1e-9)


def floors_met(resp, fw, max_mean=None, max_cvar=None, group_floor=None, tol=1e-3) -> bool:
    if max_mean is not None and float((fw * resp).sum() / max(float(fw.sum()), 1e-9)) > max_mean + tol:
        return False
    if max_cvar is not None and weighted_cvar(resp, fw) > max_cvar + tol:
        return False
    if group_floor is not None:
        limits = np.asarray(group_floor[1], dtype=float)
        if (group_means(resp, fw, np.asarray(group_floor[0]), len(limits)) > limits + tol).any():
            return False
    return True


def _floor_report(resp, fw, max_mean, max_cvar, group_floor=None) -> dict:
    mean = float((fw * resp).sum() / max(float(fw.sum()), 1e-9))
    cvar = weighted_cvar(resp, fw)
    out = {"max_mean": max_mean, "mean": round(mean, 3), "max_cvar": max_cvar, "cvar": round(cvar, 3),
           "met": floors_met(resp, fw, max_mean, max_cvar, group_floor)}
    if group_floor is not None:
        limits = np.asarray(group_floor[1], dtype=float)
        gm = group_means(resp, fw, np.asarray(group_floor[0]), len(limits))
        out["groups_over_limit"] = int((gm > limits + 1e-3).sum())
    return out


def solve_exact(
    T: np.ndarray,
    weights: np.ndarray,
    k: int,
    *,
    objective: str = "response_time",
    threshold_s: float = 600.0,
    cap_s: float | None = None,
    cap_cells: np.ndarray | None = None,
    fixed_open: Sequence[int] = (),
    allowed: Sequence[int] | None = None,
    min_from: tuple[Sequence[int], int] | None = None,
    backup_p: np.ndarray | None = None,
    backup_k: int = 30,
    max_mean: float | None = None,
    max_cvar: float | None = None,
    floor_weights: np.ndarray | None = None,
    group_floor: tuple[np.ndarray, np.ndarray] | None = None,
    time_limit_s: float = 900.0,
    rel_gap: float = 1e-4,
) -> SelectionResult:
    """Choose ``k`` sites (rows of ``T``) by integer programming, solved exactly with HiGHS.

    ``response_time`` (p-median)::

        min  sum_j w_j sum_s t_sj x_sj / sum_j w_j
        s.t. sum_s x_sj = 1           every cell is served
             x_sj <= y_s              only by an open site
             sum_s y_s = k            k sites
             y_s = 1 for s in fixed_open,   y binary, 0 <= x <= 1

    ``coverage`` (maximal covering)::

        max  sum_j w_j z_j / sum_j w_j
        s.t. z_j <= sum_{s: t_sj <= threshold} y_s,   sum_s y_s = k,   0 <= z <= 1

    ``cap_s`` (response_time only): no cell may be served from a site more than ``cap_s``
    away. ``cap_cells`` (bool per cell) says which cells the cap applies to (the planner
    passes cells today's stations reach in time: no area may lose the standard); by
    default, cells with no allowed site inside the cap are exempt. Recorded in ``cap``.
    ``min_from=(subset, n)`` requires at least ``n`` open sites from ``subset``
    (replace mode: keep at least ``n - r`` current stations).

    ``backup_p`` (response_time only; per cell, <= 0.5): busy-engine version. Each cell
    gets a primary ``x1`` and a backup ``x2`` station (different, both open) and costs
    ``(1 - p_j) t_primary + p_j t_backup``; ``x1_sj + x2_sj <= y_s``. Candidate pairs are
    each cell's ``backup_k`` nearest sites (and, with fixed open sites, none slower than
    the second-fastest fixed one); ``cap['backup_outside_k']`` counts cells whose real
    second-nearest open site fell outside that list.

    Floors (response_time / cvar): ``max_mean`` caps the layout's call-weighted average
    and ``max_cvar`` its average over the slowest 10 % of calls, both with
    ``floor_weights`` (default ``weights``) and the same costs as the objective (the busy
    blend with ``backup_p``). The planner passes today's values: "no worse than today".
    ``group_floor=(group_per_cell, limits)`` caps each group's (e.g. borough's) average.

    Pairs (s, j) that can never be used are dropped before solving, which keeps the
    answer exact: with fixed open sites, a cell is never served by a site slower than
    its fastest fixed site. With ``x`` continuous, the optimum still assigns each cell
    wholly to its nearest open site.
    """
    import time

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix, csr_matrix, vstack

    if objective not in {"response_time", "coverage", "cvar"}:
        raise ValueError(f"Unknown objective={objective!r}")
    tail = objective == "cvar"
    if tail:
        objective = "response_time"  # same assignment structure; only the cost changes
    T = np.asarray(T, dtype=float)
    w = np.asarray(weights, dtype=float)
    W = max(float(w.sum()), 1e-9)
    n_cells = T.shape[1]
    n_extra = 0
    fw = w if floor_weights is None else np.asarray(floor_weights, dtype=float)
    floors = max_mean is not None or max_cvar is not None or group_floor is not None
    pool = np.array(sorted(set(range(T.shape[0]) if allowed is None else map(int, allowed)) | set(map(int, fixed_open))))
    pos = {int(s): i for i, s in enumerate(pool)}
    n_y = len(pool)
    k = int(min(max(1, k), n_y))
    fixed = np.array([pos[int(s)] for s in fixed_open], dtype=int)
    Tp = T[pool, :]
    cap_info: dict[str, Any] = {}

    y_lb = np.zeros(n_y)
    y_lb[fixed] = 1.0
    rows = [csr_matrix(np.ones((1, n_y)))]
    lo, hi = [float(k)], [float(k)]
    if min_from is not None:
        subset, n_min = min_from
        row = np.zeros((1, n_y))
        row[0, [pos[int(s)] for s in subset if int(s) in pos]] = 1.0
        rows.append(csr_matrix(row))
        lo.append(float(n_min))
        hi.append(np.inf)

    t0 = time.perf_counter()
    if objective == "response_time":
        keep = np.isfinite(Tp)
        if len(fixed):
            best_fixed = Tp[fixed].min(axis=0)
            keep &= Tp < best_fixed[None, :]
            keep[fixed[Tp[fixed].argmin(axis=0)], np.arange(n_cells)] = True
        if cap_s is not None:
            inside = Tp <= cap_s
            exempt = ~inside.any(axis=0) if cap_cells is None else ~np.asarray(cap_cells, bool) | ~inside.any(axis=0)
            keep &= inside | exempt[None, :]
            cap_info = {"cap_s": float(cap_s), "n_cells_exempt": int(exempt.sum()),
                        "demand_share_exempt": round(float(w[exempt].sum() / W), 5)}
    if objective == "response_time" and backup_p is not None:
        pj = np.clip(np.asarray(backup_p, dtype=float), 0.0, 0.5)
        rank = np.argsort(np.argsort(Tp, axis=0), axis=0)
        pair = rank < int(backup_k)
        if len(fixed) >= 2:
            second_fixed = np.sort(Tp[fixed], axis=0)[1]
            pair &= Tp <= second_fixed[None, :]
            two = fixed[np.argsort(Tp[fixed], axis=0)[:2]]
            pair[two[0], np.arange(n_cells)] = True
            pair[two[1], np.arange(n_cells)] = True
        pair |= keep
        si, cj = np.nonzero(pair)
        prim_ok = keep[si, cj]
        n_x = len(si)
        idx = np.arange(n_x)
        tw = w[cj] * Tp[si, cj] / W
        c = np.r_[np.zeros(n_y), tw * (1 - pj[cj]), tw * pj[cj]]
        nv = n_y + 2 * n_x
        assign1 = coo_matrix((np.ones(n_x), (cj, n_y + idx)), shape=(n_cells, nv))
        assign2 = coo_matrix((np.ones(n_x), (cj, n_y + n_x + idx)), shape=(n_cells, nv))
        link = coo_matrix((np.r_[np.ones(n_x), np.ones(n_x), -np.ones(n_x)],
                           (np.r_[idx, idx, idx], np.r_[n_y + idx, n_y + n_x + idx, si])), shape=(n_x, nv))
        Acon = vstack([csr_matrix((r.toarray()[0], (np.zeros(n_y, int), np.arange(n_y))), shape=(1, nv))
                       for r in rows] + [assign1, assign2, link]).tocsr()
        lo_b = lo + [1.0] * (2 * n_cells) + [-np.inf] * n_x
        hi_b = hi + [1.0] * (2 * n_cells) + [0.0] * n_x
        if tail:
            cost = Tp[si, cj]
            Acon, lo_b, hi_b, c_tail, n_extra = _cvar_augment(
                Acon, lo_b, hi_b, np.r_[n_y + idx, n_y + n_x + idx], np.r_[cost * (1 - pj[cj]), cost * pj[cj]],
                np.r_[cj, cj], w, W, n_cells)
            c = np.r_[np.zeros(nv), c_tail]
        if floors:
            cost = Tp[si, cj]
            Acon, lo_b, hi_b, n_f = _add_floors(
                Acon, lo_b, hi_b, np.r_[n_y + idx, n_y + n_x + idx], np.r_[cost * (1 - pj[cj]), cost * pj[cj]],
                np.r_[cj, cj], fw, n_cells, max_mean, max_cvar, group_floor)
            c = np.r_[c, np.zeros(n_f)]
            n_extra += n_f
        res = milp(
            c,
            constraints=LinearConstraint(Acon, np.array(lo_b), np.array(hi_b)),
            integrality=np.r_[np.ones(n_y), np.zeros(2 * n_x + n_extra)],
            bounds=Bounds(np.r_[y_lb, np.zeros(2 * n_x), np.zeros(n_extra)],
                          np.r_[np.ones(n_y), prim_ok.astype(float), np.ones(n_x), np.full(n_extra, np.inf)]),
            options={"time_limit": float(time_limit_s), "mip_rel_gap": float(rel_gap), "disp": False},
        )
        solve_s = time.perf_counter() - t0
        if res.x is None:
            raise RuntimeError(f"Placement problem has no solution: {res.message}")
        chosen = [int(pool[i]) for i in np.flatnonzero(res.x[:n_y] > 0.5)]
        sub = T[chosen, :]
        srt = np.sort(sub, axis=0)
        open_pos = np.array([pos[i] for i in chosen])
        in_list = pair[open_pos]
        true_second = np.argsort(sub, axis=0)[1] if len(chosen) > 1 else np.zeros(n_cells, int)
        outside = ~in_list[true_second, np.arange(n_cells)]
        blend = (1 - pj) * srt[0] + pj * srt[min(1, len(chosen) - 1)]
        value = weighted_cvar(blend, w) if tail else float((w * blend).sum() / W)
        cap_info = {**cap_info, "backup_k": int(backup_k), "backup_outside_k": int(outside.sum())}
        if floors:
            cap_info["floors"] = _floor_report(blend, fw, max_mean, max_cvar, group_floor)
        return SelectionResult(
            objective="cvar_busy" if tail else "response_time_busy", indices=chosen, value=value,
            assignment=sub.argmin(axis=0), response_s=(1 - pj) * srt[0] + pj * srt[min(1, len(chosen) - 1)],
            status=str(res.message), optimal=bool(res.status == 0),
            bound=float(getattr(res, "mip_dual_bound", np.nan)), gap=float(getattr(res, "mip_gap", np.nan)),
            solve_s=round(solve_s, 2), n_vars=int(nv), cap=cap_info,
        )
    if objective == "response_time":
        si, cj = np.nonzero(keep)
        n_x = len(si)
        c = np.r_[np.zeros(n_y), w[cj] * Tp[si, cj] / W]
        idx = np.arange(n_x)
        assign = coo_matrix((np.ones(n_x), (cj, n_y + idx)), shape=(n_cells, n_y + n_x))
        link = coo_matrix((np.r_[np.ones(n_x), -np.ones(n_x)], (np.r_[idx, idx], np.r_[n_y + idx, si])),
                          shape=(n_x, n_y + n_x))
        A = vstack([csr_matrix((r.toarray()[0], (np.zeros(n_y, int), np.arange(n_y))), shape=(1, n_y + n_x))
                    for r in rows] + [assign, link]).tocsr()
        lo = lo + [1.0] * n_cells + [-np.inf] * n_x
        hi = hi + [1.0] * n_cells + [0.0] * n_x
        n_z = n_x
        if tail:
            A, lo, hi, c_tail, n_extra = _cvar_augment(A, lo, hi, n_y + idx, Tp[si, cj], cj, w, W, n_cells)
            c = np.r_[np.zeros(n_y + n_x), c_tail]
        if floors:
            A, lo, hi, n_f = _add_floors(A, lo, hi, n_y + idx, Tp[si, cj], cj, fw, n_cells, max_mean, max_cvar,
                                         group_floor)
            c = np.r_[c, np.zeros(n_f)]
            n_extra += n_f
    else:
        reach = Tp <= threshold_s
        si, cj = np.nonzero(reach)
        n_z = n_cells
        c = np.r_[np.zeros(n_y), -w / W]
        link = coo_matrix((np.r_[np.ones(n_cells), -np.ones(len(si))],
                           (np.r_[np.arange(n_cells), cj], np.r_[n_y + np.arange(n_cells), si])),
                          shape=(n_cells, n_y + n_cells))
        A = vstack([csr_matrix((r.toarray()[0], (np.zeros(n_y, int), np.arange(n_y))), shape=(1, n_y + n_cells))
                    for r in rows] + [link]).tocsr()
        lo = lo + [-np.inf] * n_cells
        hi = hi + [0.0] * n_cells

    res = milp(
        c,
        constraints=LinearConstraint(A, np.array(lo), np.array(hi)),
        integrality=np.r_[np.ones(n_y), np.zeros(n_z + n_extra)],
        bounds=Bounds(np.r_[y_lb, np.zeros(n_z), np.zeros(n_extra)],
                      np.r_[np.ones(n_y), np.ones(n_z), np.full(n_extra, np.inf)]),
        options={"time_limit": float(time_limit_s), "mip_rel_gap": float(rel_gap), "disp": False},
    )
    solve_s = time.perf_counter() - t0
    if res.x is None:
        raise RuntimeError(f"Placement problem has no solution: {res.message}")
    chosen = [int(pool[i]) for i in np.flatnonzero(res.x[:n_y] > 0.5)]
    sub = T[chosen, :]
    value = objective_value(T, chosen, w, objective="cvar" if tail else objective, threshold_s=threshold_s)
    bound = float(getattr(res, "mip_dual_bound", np.nan))
    if floors:
        cap_info["floors"] = _floor_report(sub.min(axis=0), fw, max_mean, max_cvar, group_floor)
    return SelectionResult(
        objective="cvar" if tail else objective,
        indices=chosen,
        value=float(value),
        assignment=sub.argmin(axis=0),
        response_s=sub.min(axis=0),
        status=str(res.message),
        optimal=bool(res.status == 0),
        bound=bound,
        gap=float(getattr(res, "mip_gap", np.nan)),
        solve_s=round(solve_s, 2),
        n_vars=int(n_y + n_z),
        cap=cap_info,
    )


def posture_seconds(T: np.ndarray, idx: Sequence[int], busy_p: np.ndarray | None = None) -> np.ndarray:
    """Per-cell first-engine seconds for open sites ``idx``.

    With ``busy_p`` (per candidate row), the nearest site's engine is out with probability
    p and the next nearest answers: (1 - p) * t1 + p * t2 (Larson-style backup cover).
    """
    sub = T[np.asarray(list(idx), dtype=int), :]
    if busy_p is None or sub.shape[0] < 2:
        return sub.min(axis=0)
    two = np.argpartition(sub, 1, axis=0)[:2]
    cols = np.arange(sub.shape[1])
    a, b = sub[two[0], cols], sub[two[1], cols]
    first = np.where(a <= b, two[0], two[1])
    t1, t2 = np.minimum(a, b), np.maximum(a, b)
    p = np.asarray(busy_p, dtype=float)[np.asarray(list(idx), dtype=int)[first]]
    return (1.0 - p) * t1 + p * t2


@dataclass
class GeneticResult:
    indices: list[int]
    value: float
    response_s: np.ndarray
    history: list[float]  # best objective per generation
    generations: int
    evaluations: int
    solve_s: float
    params: dict = field(default_factory=dict)


def solve_genetic(
    T: np.ndarray,
    weights: np.ndarray,
    k: int,
    *,
    fixed_open: Sequence[int] = (),
    allowed: Sequence[int] | None = None,
    min_from: tuple[Sequence[int], int] | None = None,
    busy_p: np.ndarray | None = None,
    busy_fn: Any = None,
    cap_s: float | None = None,
    cap_cells: np.ndarray | None = None,
    cap_penalty: float = 5.0,
    population: int = 80,
    generations: int = 400,
    elite: int = 4,
    tournament: int = 3,
    mutation_rate: float = 0.3,
    patience: int = 80,
    seed: int = 42,
    objective: str = "response_time",
    max_mean: float | None = None,
    max_cvar: float | None = None,
    floor_weights: np.ndarray | None = None,
    group_floor: tuple[np.ndarray, np.ndarray] | None = None,
    floor_penalty: float = 20.0,
    seeds: Sequence[Sequence[int]] = (),
) -> GeneticResult:
    """Choose ``k`` sites with a genetic algorithm (minimises demand-weighted first-engine time).

    Each individual is a set of ``k`` open sites (``fixed_open`` always included).

    - fitness: demand-weighted average first-engine seconds, nearest open site, or with
      ``busy_p`` the busy-engine blend (1 - p) t1 + p t2 that the integer program cannot
      express linearly. ``busy_fn(open_sites) -> p per candidate`` makes ``p`` depend on
      the layout (e.g. ``busy_placement.ErlangBusyModel.probs``). ``cap_s`` adds ``cap_penalty`` x the seconds by which any area
      exceeds the cap (areas no site reaches in time are exempt). Floors ``max_mean`` /
      ``max_cvar`` (as in ``solve_exact``) add ``floor_penalty`` x the seconds over.
    - selection: tournaments of ``tournament``; the best ``elite`` survive unchanged.
    - crossover: the child keeps sites both parents share, then fills up from the sites
      only one parent has (random order).
    - mutation: with probability ``mutation_rate`` per child, swap one free site for an
      unused one, half the time a near neighbour of the dropped site.
    - repair: ``min_from=(subset, n)`` keeps at least ``n`` sites from ``subset``.

    ``seeds``: full layouts (e.g. today's stations) placed in the first population.
    With floors or a cap, the best layout that breaks none of them is returned (falling
    back to the best overall only if none was found); ``params['feasible']`` says which.

    Stops after ``generations`` or ``patience`` generations without improvement.
    """
    import time

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    T = np.asarray(T, dtype=float)
    w = np.asarray(weights, dtype=float)
    W = max(float(w.sum()), 1e-9)
    fw = w if floor_weights is None else np.asarray(floor_weights, dtype=float)
    FW = max(float(fw.sum()), 1e-9)
    fixed = sorted({int(i) for i in fixed_open})
    pool = sorted(set(range(T.shape[0]) if allowed is None else map(int, allowed)) - set(fixed))
    pool_arr = np.array(pool)
    m = int(k) - len(fixed)
    if m <= 0 or not pool:
        resp = posture_seconds(T, fixed, busy_fn(fixed) if busy_fn is not None else busy_p)
        v0 = weighted_cvar(resp, w) if objective == "cvar" else float((w * resp).sum() / W)
        return GeneticResult(fixed, v0, resp, [], 0, 1, 0.0)
    m = min(m, len(pool))
    keep_set, n_keep = (set(map(int, min_from[0])) & set(pool), int(min_from[1])) if min_from else (set(), 0)
    keep_arr = np.array(sorted(keep_set))
    # Near neighbours for mutation: 10 closest candidate sites by travel time profile.
    prof = T[pool_arr][:, :: max(1, T.shape[1] // 400)]
    d2 = ((prof[:, None, :] - prof[None, :, :]) ** 2).sum(axis=2)
    near = np.argsort(d2, axis=1)[:, 1:11]
    pos_in_pool = {s: i for i, s in enumerate(pool)}
    base = T[fixed].min(axis=0) if fixed and busy_p is None and busy_fn is None else None

    def busy_of(idx):
        return busy_fn(idx) if busy_fn is not None else busy_p
    exempt = None
    if cap_s is not None:
        exempt = ~(T[fixed + pool] <= cap_s).any(axis=0)
        if cap_cells is not None:
            exempt |= ~np.asarray(cap_cells, bool)
    n_eval = 0

    def fitness(genes: np.ndarray) -> float:
        nonlocal n_eval
        n_eval += 1
        if base is not None:
            resp = np.minimum(base, T[genes].min(axis=0))
        else:
            idx = fixed + genes.tolist()
            resp = posture_seconds(T, idx, busy_of(idx))
        val = weighted_cvar(resp, w) if objective == "cvar" else float((w * resp).sum() / W)
        if max_mean is not None:
            val += floor_penalty * max(0.0, float((fw * resp).sum() / FW) - max_mean)
        if max_cvar is not None:
            val += floor_penalty * max(0.0, weighted_cvar(resp, fw) - max_cvar)
        if group_floor is not None:
            lim = np.asarray(group_floor[1], dtype=float)
            over = group_means(resp, fw, np.asarray(group_floor[0]), len(lim)) - lim
            val += floor_penalty * float(np.maximum(over, 0.0).sum())
        if cap_s is not None:
            near = T[fixed + genes.tolist()].min(axis=0)  # cap on the nearest station, not the busy blend
            over = np.where(exempt, 0.0, np.maximum(near - cap_s, 0.0))
            val += cap_penalty * float((w * over).sum() / W)
        return val

    constrained = cap_s is not None or max_mean is not None or max_cvar is not None or group_floor is not None

    def violation(genes: np.ndarray) -> float:
        idx = fixed + genes.tolist()
        resp = posture_seconds(T, idx, busy_of(idx))
        v = 0.0
        if max_mean is not None:
            v += max(0.0, float((fw * resp).sum() / FW) - max_mean)
        if max_cvar is not None:
            v += max(0.0, weighted_cvar(resp, fw) - max_cvar)
        if group_floor is not None:
            lim = np.asarray(group_floor[1], dtype=float)
            v += float(np.maximum(group_means(resp, fw, np.asarray(group_floor[0]), len(lim)) - lim, 0).sum())
        if cap_s is not None:
            v += float(np.maximum(np.where(exempt, 0.0, T[idx].min(axis=0) - cap_s), 0.0).sum())
        return v

    def repair(genes: np.ndarray) -> np.ndarray:
        genes = np.unique(genes)
        if len(genes) < m:
            free = np.setdiff1d(pool_arr, genes)
            genes = np.r_[genes, rng.choice(free, m - len(genes), replace=False)]
        if n_keep:
            have = np.isin(genes, keep_arr)
            short = n_keep - int(have.sum())
            if short > 0:
                drop = rng.choice(np.flatnonzero(~have), short, replace=False)
                genes[drop] = rng.choice(np.setdiff1d(keep_arr, genes), short, replace=False)
        return np.sort(genes[:m])

    def random_individual() -> np.ndarray:
        if n_keep:
            g = rng.choice(keep_arr, min(n_keep, m), replace=False)
            rest = rng.choice(np.setdiff1d(pool_arr, g), m - len(g), replace=False)
            return repair(np.r_[g, rest])
        return np.sort(rng.choice(pool_arr, m, replace=False))

    pop = [random_individual() for _ in range(population)]
    fixed_set = set(fixed)
    for i, sd in enumerate(seeds):
        if i < population:
            g = np.array(sorted(int(x) for x in sd if int(x) not in fixed_set and int(x) in set(pool)))
            pop[i] = repair(g) if len(g) != m or n_keep else np.sort(g)
    # Replace mode: seed part of the population near today's layout (all kept + random others).
    if n_keep and len(keep_arr) >= m:
        for i in range(population // 4):
            g = keep_arr.copy()
            swap = rng.choice(m, size=min(m, int(rng.integers(1, 4))), replace=False)
            g[swap] = rng.choice(np.setdiff1d(pool_arr, g), len(swap), replace=False)
            pop[i] = repair(g)
    fit = np.array([fitness(g) for g in pop])
    best_i = int(fit.argmin())
    best, best_val = pop[best_i].copy(), float(fit[best_i])
    history, stale, gen = [best_val], 0, 0
    feas, feas_val = None, np.inf

    def update_feasible():
        nonlocal feas, feas_val
        if not constrained:
            return
        for i in np.argsort(fit)[:5]:
            if fit[i] < feas_val - 1e-9 and violation(pop[i]) <= 1e-3:
                feas, feas_val = pop[i].copy(), float(fit[i])
                break

    update_feasible()

    def pick() -> np.ndarray:
        c = rng.choice(population, tournament, replace=False)
        return pop[int(c[fit[c].argmin()])]

    for gen in range(1, generations + 1):
        order = np.argsort(fit)
        children = [pop[i].copy() for i in order[:elite]]
        while len(children) < population:
            a, b = pick(), pick()
            common = np.intersect1d(a, b)
            rest = np.setxor1d(a, b)
            rng.shuffle(rest)
            child = np.r_[common, rest[: m - len(common)]]
            if rng.random() < mutation_rate:
                j = int(rng.integers(len(child)))
                unused = np.setdiff1d(pool_arr, child)
                if rng.random() < 0.5:
                    nb = pool_arr[near[pos_in_pool[int(child[j])]]]
                    nb = np.setdiff1d(nb, child)
                    child[j] = rng.choice(nb) if len(nb) else rng.choice(unused)
                else:
                    child[j] = rng.choice(unused)
            children.append(repair(child))
        pop = children
        fit = np.array([fitness(g) for g in pop])
        i = int(fit.argmin())
        if fit[i] < best_val - 1e-9:
            best, best_val, stale = pop[i].copy(), float(fit[i]), 0
        else:
            stale += 1
        update_feasible()
        history.append(best_val)
        if stale >= patience:
            break

    feasible = None
    if constrained:
        feasible = feas is not None
        if feas is not None:
            best = feas
    chosen = fixed + best.tolist()
    resp = posture_seconds(T, chosen, busy_of(chosen))
    return GeneticResult(
        indices=sorted(int(i) for i in chosen),
        value=weighted_cvar(resp, w) if objective == "cvar" else float((w * resp).sum() / W),
        response_s=resp,
        history=history,
        generations=gen,
        evaluations=n_eval,
        solve_s=round(time.perf_counter() - t0, 2),
        params={"population": population, "max_generations": generations, "elite": elite,
                "tournament": tournament, "mutation_rate": mutation_rate, "patience": patience,
                "seed": seed, "busy_aware": busy_p is not None or busy_fn is not None,
                "busy_load_dependent": busy_fn is not None, "cap_s": cap_s, "feasible": feasible},
    )


def swap_search(
    T: np.ndarray,
    weights: np.ndarray,
    start: Sequence[int],
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    allowed: Sequence[int],
    fixed_open: Sequence[int] = (),
    min_from: tuple[Sequence[int], int] | None = None,
    busy_fn: Any = None,
    objective: str = "response_time",
    cap_s: float | None = None,
    cap_cells: np.ndarray | None = None,
    max_mean: float | None = None,
    max_cvar: float | None = None,
    floor_weights: np.ndarray | None = None,
    group_floor: tuple[np.ndarray, np.ndarray] | None = None,
    neighbours: int = 12,
    max_swaps: int = 200,
) -> dict[str, Any]:
    """Feasible-swap local search (Teitz–Bart style) from a layout that meets every floor.

    Each step tries closing one open site and opening one of its ``neighbours`` nearest
    unused sites, and applies the swap that lowers the objective most while keeping every
    floor and the cap. Every layout it passes through is valid, so it can only improve on
    ``start``. Used when the exact solver finds no layout in time.
    """
    import time

    t0 = time.perf_counter()
    w = np.asarray(weights, dtype=float)
    W = max(float(w.sum()), 1e-9)
    fw = w if floor_weights is None else np.asarray(floor_weights, dtype=float)
    fixed = {int(i) for i in fixed_open}
    pool = np.array(sorted(set(map(int, allowed)) - fixed))
    kx = 111.32 * np.cos(np.radians(51.5))
    keep_set, n_keep = (set(map(int, min_from[0])), int(min_from[1])) if min_from else (set(), 0)

    def evaluate(idx):
        resp = posture_seconds(T, idx, busy_fn(idx) if busy_fn is not None else None)
        # The cap is on the nearest station's time (as in solve_exact), not the busy blend.
        if cap_s is not None and cap_cells is not None and (T[idx].min(axis=0)[cap_cells] > cap_s + 1e-6).any():
            return np.inf
        if not floors_met(resp, fw, max_mean, max_cvar, group_floor):
            return np.inf
        return weighted_cvar(resp, w) if objective == "cvar" else float((w * resp).sum() / W)

    cur = sorted(int(i) for i in start)
    cur_val = evaluate(cur)
    log = []
    for _ in range(max_swaps):
        cur_set = set(cur)
        unused = pool[~np.isin(pool, cur)]
        best = (cur_val, None, None)
        for s in cur:
            if s in fixed:
                continue
            d = np.hypot((lon[unused] - lon[s]) * kx, (lat[unused] - lat[s]) * 110.574)
            for t in unused[np.argsort(d)[:neighbours]]:
                if n_keep and s in keep_set and t not in keep_set and len(cur_set & keep_set) - 1 < n_keep:
                    continue
                trial = sorted(cur_set - {s} | {int(t)})
                v = evaluate(trial)
                if v < best[0] - 1e-6:
                    best = (v, s, int(t))
        if best[1] is None:
            break
        cur = sorted(set(cur) - {best[1]} | {best[2]})
        log.append({"close": best[1], "open": best[2], "value": round(best[0], 3)})
        cur_val = best[0]
    return {"indices": cur, "value": float(cur_val), "swaps": log, "solve_s": round(time.perf_counter() - t0, 1)}


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if not len(values):
        return float("nan")
    order = np.argsort(values)
    v = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float("nan")
    return float(np.interp(q * cw[-1], cw, v))


def posture_metrics(
    response_s: np.ndarray,
    weights: np.ndarray,
    *,
    threshold_s: float,
    label: str = "",
    n_posts: int | None = None,
) -> dict[str, Any]:
    """Demand-weighted response-time / coverage summary for one posture."""
    total = max(float(np.sum(weights)), 1e-9)
    finite = np.isfinite(response_s)
    areas = finite & (np.asarray(weights) > 0)  # areas with demand, each counted once
    return {
        "label": label,
        "n_posts": n_posts,
        "expected_response_s": round(float((weights[finite] * response_s[finite]).sum() / total), 1),
        "median_response_s": round(_weighted_quantile(response_s[finite], weights[finite], 0.5), 1),
        "p90_response_s": round(_weighted_quantile(response_s[finite], weights[finite], 0.9), 1),
        "cvar90_response_s": round(weighted_cvar(response_s[finite], weights[finite]), 1),
        "worst_cell_response_s": round(float(np.max(response_s[areas])), 1) if areas.any() else None,
        "coverage_share_within_threshold": round(
            float((weights[finite] * (response_s[finite] <= threshold_s)).sum() / total), 4
        ),
        "area_mean_response_s": round(float(response_s[areas].mean()), 1) if areas.any() else None,
        "areas_over_threshold_share": round(float((response_s[areas] > threshold_s).mean()), 4)
        if areas.any() else None,
        "threshold_s": float(threshold_s),
        "demand_cells": int(len(response_s)),
        "areas_with_demand": int(areas.sum()),
    }
