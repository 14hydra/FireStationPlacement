"""Optimal fire station facility location (London LFB primary).

Places or relocates fire stations so the first engine reaches historical
demand fast enough to meet the LFB standards. The scored quantity is

    attendance(station s -> cell j) = predicted_drive(s, j) + turnout(s)

and is judged against the LFB first-engine KPIs (mean <= 6 min, > 90 % within
10 min; ``firestation.lfb_standards.score_first_engine``). Travel-only scoring (no
turnout) was WRONG for LFB: public attendance is mobilise -> arrive, so drive
alone understates response and misses the 6/10-minute standards.

Building blocks:

- travel scorers (``scorer=``): ``lgbm`` (LightGBM travel-time model on the full
  route features, ``firestation.ml_scorer``), ``road`` (Kolesar on shortest legal road
  km), ``kolesar`` (crow distance -> Kolesar piecewise model, refit on LFB trips) and
  ``crow`` (flat-speed surrogate)
- turnout: each station's median recorded turnout (LFB mobilisation
  ``TurnoutTimeSeconds``, home-station trips in the demand years); new sites get the
  median over all stations. Without mobilisation data, a prior from attendance
  residuals vs. Kolesar drive, and ``DEFAULT_TURNOUT_S`` for new sites
- discrete facility location: p-median / max-cover as an integer program, solved to a
  proven optimum with HiGHS (``facility.solve_exact``); optional per-cell time cap
- optional busy-engine blend (``use_busy``) and demand/eval year split

Modes
-----
``redesign`` - choose the best ``k`` sites from the candidate set (full redesign).
``replace``  - keep at least ``n - r`` current houses; close up to ``r`` and open as
               many new candidates (constant fleet size).
``expand``   - keep ALL current houses; open the best ``m`` new ones.

redesign / replace / expand are integer programs solved to a proven optimum.
``resize``   - keep ALL current house *locations*; reallocate ``total_pumps``
               across stations (1–3 engines each) under a capacity-aware busy
               model (``firestation.station_capacity``).

No land cost, zoning, staffing or borough-boundary constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .busy_engines import estimate_busy_rates, expected_first_arrival
from .kolesar import KolesarModel, fit_kolesar
from .lfb_standards import DEFAULT_TURNOUT_S, attendance_seconds, score_first_engine
from .facility import (
    TravelTimeSurrogate,
    add_demand_cell_candidates,
    build_demand_grid,
    haversine_km,
    objective_value,
    posture_metrics,
    GeneticResult,
    SelectionResult,
    solve_exact,
    solve_genetic,
    swap_search,
    weighted_cvar,
    floors_met,
    group_means,
)
from .station_capacity import (
    DEFAULT_TOTAL_PUMPS,
    capacity_busy_prob,
    default_pump_table,
    plan_resize_capacity,
    resolve_pump_vector,
)

SCORERS = ("lgbm", "lgbm_q90", "road", "kolesar", "crow")

# How much each call counts as demand (LFB IncidentGroup / PropertyCategory).
SEVERITY_SCHEMES = {
    "all": "every call counts 1",
    "serious": "fires and special services count 1, false alarms 0",
    "dwelling_fire": "only fires in homes count",
}


def severity_weight(incidents: pd.DataFrame, scheme: str = "all") -> np.ndarray:
    """Demand weight per call under a severity scheme (see ``SEVERITY_SCHEMES``)."""
    if scheme not in SEVERITY_SCHEMES:
        raise ValueError(f"severity must be one of {list(SEVERITY_SCHEMES)}, got {scheme!r}")
    if scheme == "all":
        return np.ones(len(incidents))
    if "incident_group" not in incidents.columns:
        raise ValueError("severity weighting needs incident_group; rebuild with build_lfb_planner_incidents.py")
    group = incidents["incident_group"].astype(str).str.upper()
    if scheme == "serious":
        return group.isin(["FIRE", "SPECIAL SERVICE"]).to_numpy(dtype=float)
    prop = incidents["property_category"].astype(str).str.upper()
    return (group.eq("FIRE") & prop.eq("DWELLING")).to_numpy(dtype=float)


def cell_weights(cells: pd.DataFrame, incidents: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    """Sum of per-call weights in each demand-grid cell (same binning as ``build_demand_grid``)."""
    dlat, dlon = float(cells["cell_dlat"].iloc[0]), float(cells["cell_dlon"].iloc[0])
    key = pd.DataFrame({"_ix": np.floor(incidents["dest_lon"].to_numpy(float) / dlon).astype(int),
                        "_iy": np.floor(incidents["dest_lat"].to_numpy(float) / dlat).astype(int), "w": w})
    tot = key.groupby(["_ix", "_iy"])["w"].sum()
    return tot.reindex(pd.MultiIndex.from_frame(cells[["_ix", "_iy"]])).fillna(0.0).to_numpy()


def solver_weights(weights: np.ndarray, alpha: float) -> np.ndarray:
    """Weight each area by demand**alpha: 1 = calls count (efficiency), 0 = areas count equally."""
    w = np.asarray(weights, dtype=float)
    return np.where(w > 0, np.power(np.maximum(w, 1e-12), float(alpha)), 0.0)
TRAVEL_TRAINING = Path(__file__).resolve().parents[2] / "data" / "processed" / "lfb_travel_training.csv"
MODES = ("redesign", "replace", "expand", "resize")


def build_house_candidates(
    houses: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    demand_candidates: int = 80,
    spread_km: float | None = None,
) -> pd.DataFrame:
    """Existing fire stations + top demand centroids (+ one per ``spread_km`` block) as sites."""
    existing = pd.DataFrame(
        {
            "site_name": houses["facilityname"].astype(str),
            "layer": "existing_fire_station",
            "lon": houses["lon"].astype(float),
            "lat": houses["lat"].astype(float),
            "is_existing": True,
            "borough": houses["borough"] if "borough" in houses.columns else None,
        }
    )
    cand = existing.copy()
    cand.insert(0, "candidate_id", np.arange(len(cand)))
    # add_demand_cell_candidates expects post_name/layer/lon/lat
    post_shaped = cand.rename(columns={"site_name": "post_name"})
    post_shaped["is_anchor"] = True
    enriched = add_demand_cell_candidates(post_shaped, cells, top_n=demand_candidates, spread_km=spread_km)
    # Normalize columns for this module
    out = pd.DataFrame(
        {
            "candidate_id": np.arange(len(enriched)),
            "site_name": enriched["post_name"].astype(str),
            "layer": enriched["layer"].astype(str),
            "lon": enriched["lon"].astype(float),
            "lat": enriched["lat"].astype(float),
        }
    )
    out["is_existing"] = out["layer"].eq("existing_fire_station")
    if "borough" in enriched.columns:
        out["borough"] = enriched["borough"]
    return out.reset_index(drop=True)


def build_travel_matrix(
    candidates: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    surrogate: TravelTimeSurrogate | None = None,
    scorer: str = "kolesar",
    kolesar: KolesarModel | None = None,
    incidents: pd.DataFrame | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Candidate × demand **drive** seconds (turnout is added by the caller).

    ``scorer``:
      * ``lgbm``    — LightGBM travel-time model on routes over the London drive graph
                      (``incidents`` sets the hour weights and cell boroughs)
      * ``road``    — shortest legal road km → Kolesar (road-km fit)
      * ``kolesar`` — crow-flies km → :class:`KolesarModel` (London default if none given)
      * ``crow``    — flat-speed surrogate
    """
    if scorer not in SCORERS:
        raise ValueError(f"scorer must be one of {SCORERS}, got {scorer!r}")
    surrogate = surrogate or TravelTimeSurrogate()
    meta: dict[str, Any] = {"scorer": scorer, "surrogate_speed_kph": surrogate.speed_kph}
    if scorer in {"lgbm", "lgbm_q90"}:
        from .ml_scorer import MODEL_PATH, Q90_MODEL_PATH, lgbm_matrix

        T, m = lgbm_matrix(candidates, cells, incidents,
                           model_path=Q90_MODEL_PATH if scorer == "lgbm_q90" else MODEL_PATH)
        return T, {**meta, **m}
    if scorer == "road":
        from .ml_scorer import road_matrix

        T, m = road_matrix(candidates, cells)
        return T, {**meta, **m}
    if scorer == "kolesar":
        model = kolesar or KolesarModel.london_default()
        km = haversine_km(
            candidates["lon"].to_numpy(dtype=float)[:, None],
            candidates["lat"].to_numpy(dtype=float)[:, None],
            cells["lon"].to_numpy(dtype=float)[None, :],
            cells["lat"].to_numpy(dtype=float)[None, :],
        )
        meta["kolesar"] = model.to_dict()
        return np.maximum(model.predict(km), 15.0), meta
    return surrogate.matrix(candidates, cells), meta


def _solver_meta(res) -> dict[str, Any]:
    if isinstance(res, GeneticResult):
        return {"solver": "genetic algorithm", "optimal": None, "objective_value": round(res.value, 4),
                "generations": res.generations, "evaluations": res.evaluations, "solve_s": res.solve_s,
                "params": res.params, "history_every_10": [round(v, 3) for v in res.history[::10]]}
    return {"solver": "HiGHS integer program (scipy.optimize.milp)", "status": res.status,
            "optimal": res.optimal, "objective_value": round(res.value, 4),
            "lower_bound": round(res.bound, 4) if np.isfinite(res.bound) else None,
            "gap": res.gap, "solve_s": res.solve_s, "n_vars": res.n_vars, "cap": res.cap,
            "warm_start": (res.cap or {}).get("warm_start"),
            "bound_note": ("lower bound of the round that produced the layout, busy rates held fixed"
                           if "busy" in str(res.objective) else "lower bound: no layout can score better")}


def _solve(T, weights, k, *, solver, objective, threshold_s, cap_s, busy_model=None, start_idx=None,
           seed=42, **kw):
    """Exact integer program or genetic algorithm; ``busy_model`` makes either busy-aware."""
    if busy_model is not None and objective == "coverage":
        raise ValueError("busy-aware placement optimises response_time or cvar")
    if solver == "genetic":
        if objective == "coverage":
            raise ValueError("the genetic solver optimises response_time or cvar")
        fn = (lambda idx: busy_model.probs(T, idx)) if busy_model is not None else None
        return solve_genetic(T, weights, k, cap_s=cap_s, busy_fn=fn, seed=seed, objective=objective, **kw)
    if solver != "exact":
        raise ValueError(f"solver must be exact|genetic, got {solver!r}")
    if busy_model is not None:
        from .busy_placement import solve_busy_exact

        out = solve_busy_exact(T, weights, k, busy_model, start_idx=start_idx, objective=objective,
                               threshold_s=threshold_s, cap_s=cap_s, **kw)
        if out["round"] == 0 and any(kw.get(f) is not None for f in ("max_mean", "max_cvar", "group_floor")):
            # No round produced a valid improvement: the exact solver found no layout in time,
            # or its layouts broke a floor once busy rates were recomputed. Fall back to the
            # genetic algorithm (seeded with today's layout) and a feasible-swap search, which
            # keep only layouts meeting every floor under the full busy model.
            ga_kw = {k2: v for k2, v in kw.items() if k2 not in {"max_rounds", "time_limit_s", "backup_k"}}
            print("  no valid improving layout from the exact rounds; genetic algorithm + feasible-swap search…",
                  flush=True)
            ga = solve_genetic(T, weights, k, cap_s=cap_s, busy_fn=lambda idx: busy_model.probs(T, idx),
                               seed=seed, objective=objective, seeds=[list(start_idx)], **ga_kw)
            print(f"  genetic: feasible={ga.params.get('feasible')} value={ga.value:.2f} (today {out['value']:.2f})",
                  flush=True)
            sw_kw = {k2: v for k2, v in ga_kw.items() if k2 in {"fixed_open", "min_from", "cap_cells", "max_mean",
                                                                 "max_cvar", "floor_weights", "group_floor"}}
            allowed = kw.get("allowed") or range(T.shape[0])
            lat, lon = busy_model.pump_plan.lat, busy_model.pump_plan.lon
            start = list(ga.indices) if ga.params.get("feasible") else list(start_idx)
            sw = swap_search(T, weights, start, lat=lat, lon=lon, allowed=allowed,
                             busy_fn=lambda idx: busy_model.probs(T, idx), objective=objective, cap_s=cap_s, **sw_kw)
            print(f"  swap search: {len(sw['swaps'])} swaps, value {sw['value']:.2f}", flush=True)
            if sw["value"] < out["value"] - 1e-9:
                res = SelectionResult(
                    objective=f"{objective}_busy", indices=sw["indices"], value=sw["value"],
                    assignment=T[sw["indices"]].argmin(axis=0), response_s=busy_model.response(T, sw["indices"]),
                    status=f"fallback: feasible-swap search from {'genetic algorithm' if start != list(start_idx) else 'today'}"
                           f" ({len(sw['swaps'])} swaps); exact rounds gave no valid improvement",
                    optimal=False, solve_s=sw["solve_s"],
                    bound=(fb := float(np.nanmin([r.get("bound_fixed_p", np.nan) for r in out["rounds"]] + [np.inf]))),
                    gap=(sw["value"] - fb) / sw["value"] if np.isfinite(fb) else float("nan"),
                    cap={"swaps": sw["swaps"], "genetic_feasible": ga.params.get("feasible"),
                         "exact_rounds": out["rounds"]})
                return res
        res = SelectionResult(
            objective=f"{objective}_busy", indices=out["indices"], value=out["value"],
            assignment=T[out["indices"]].argmin(axis=0), response_s=out["response_s"],
            status=f"busy-aware: best of {len(out['rounds'])} rounds (round {out['round']})",
            optimal=all(r.get("optimal_fixed_p", True) for r in out["rounds"]),
            # Lower bound for the round that produced the layout, with that round's busy rates fixed.
            bound=float(out.get("bound_fixed_p", np.nan)),
            gap=max((r["gap"] for r in out["rounds"] if "gap" in r), default=float("nan")),
            solve_s=out["solve_s"], cap={"busy_rounds": out["rounds"]})
        return res
    return solve_exact(T, weights, k, objective=objective, threshold_s=threshold_s, cap_s=cap_s,
                       warm_start=start_idx, **kw)


def plan_redesign(
    T: np.ndarray,
    weights: np.ndarray,
    k: int,
    *,
    objective: str = "response_time",
    threshold_s: float = 600.0,
    cap_s: float | None = None,
    solver: str = "exact",
    busy_model: Any = None,
    start_idx: Sequence[int] | None = None,
    cap_cells: np.ndarray | None = None,
    floors: dict | None = None,
) -> dict[str, Any]:
    """Best ``k`` sites from the whole candidate set."""
    res = _solve(T, weights, k, solver=solver, objective=objective, threshold_s=threshold_s, cap_s=cap_s,
                 busy_model=busy_model, start_idx=start_idx, cap_cells=cap_cells, **(floors or {}))
    return {"indices": res.indices, "value": res.value, "response_s": res.response_s,
            "objective": objective, "solver": _solver_meta(res)}


def plan_replace(
    T: np.ndarray,
    weights: np.ndarray,
    existing_idx: Sequence[int],
    *,
    replace_r: int,
    new_pool: Sequence[int] | None = None,
    objective: str = "response_time",
    threshold_s: float = 600.0,
    cap_s: float | None = None,
    solver: str = "exact",
    busy_model: Any = None,
    start_idx: Sequence[int] | None = None,
    cap_cells: np.ndarray | None = None,
    floors: dict | None = None,
) -> dict[str, Any]:
    """Keep the fleet size; close up to ``r`` current stations and open as many new sites.

    Solved jointly and exactly: the best layout with at least ``n - r`` current stations.
    """
    existing = [int(i) for i in existing_idx]
    ex = set(existing)
    n = len(existing)
    r = int(min(max(0, replace_r), n))
    pool = [i for i in range(T.shape[0]) if i not in ex] if new_pool is None else [
        int(i) for i in new_pool if int(i) not in ex]
    res = _solve(T, weights, n, solver=solver, objective=objective, threshold_s=threshold_s, cap_s=cap_s,
                 busy_model=busy_model, start_idx=start_idx or existing, cap_cells=cap_cells,
                 allowed=existing + pool, min_from=(existing, n - r), **(floors or {}))
    chosen = set(res.indices)
    return {
        "indices": res.indices,
        "closed": [i for i in existing if i not in chosen],
        "opened": [i for i in res.indices if i not in ex],
        "value": res.value,
        "response_s": res.response_s,
        "objective": objective,
        "solver": _solver_meta(res),
    }


def random_replace_baseline(
    T: np.ndarray,
    weights: np.ndarray,
    existing_idx: Sequence[int],
    *,
    replace_r: int,
    new_pool: Sequence[int],
    objective: str = "response_time",
    threshold_s: float = 480.0,
    seed: int = 0,
) -> dict[str, Any]:

    rng = np.random.default_rng(seed)
    existing = [int(i) for i in existing_idx]
    pool = [int(i) for i in new_pool if int(i) not in set(existing)]
    r = min(int(replace_r), len(existing), len(pool))
    if r <= 0:
        selected = list(existing)
        closed, opened = [], []
    else:
        closed = [int(x) for x in rng.choice(existing, size=r, replace=False)]
        opened = [int(x) for x in rng.choice(pool, size=r, replace=False)]
        selected = [i for i in existing if i not in set(closed)] + opened
    best = T[selected, :].min(axis=0)
    return {
        "indices": selected,
        "closed": closed,
        "opened": opened,
        "value": objective_value(T, selected, weights, objective=objective, threshold_s=threshold_s),
        "response_s": best,
        "label": "random_replace",
    }


def plan_expand(
    T: np.ndarray,
    weights: np.ndarray,
    existing_idx: Sequence[int],
    add_m: int,
    *,
    new_pool: Sequence[int] | None = None,
    objective: str = "response_time",
    threshold_s: float = 600.0,
    cap_s: float | None = None,
    solver: str = "exact",
    busy_model: Any = None,
    start_idx: Sequence[int] | None = None,
    cap_cells: np.ndarray | None = None,
    floors: dict | None = None,
) -> dict[str, Any]:
    """Keep ALL current stations; open the best ``add_m`` new sites."""
    existing = [int(i) for i in existing_idx]
    ex = set(existing)
    pool = [i for i in range(T.shape[0]) if i not in ex] if new_pool is None else [
        int(i) for i in new_pool if int(i) not in ex]
    m = int(min(max(0, add_m), len(pool)))
    res = _solve(T, weights, len(existing) + m, solver=solver, objective=objective, threshold_s=threshold_s,
                 cap_s=cap_s, busy_model=busy_model, start_idx=start_idx or existing, cap_cells=cap_cells,
                 fixed_open=existing, allowed=existing + pool, **(floors or {}))
    return {
        "indices": res.indices,
        "closed": [],
        "opened": [i for i in res.indices if i not in ex],
        "value": res.value,
        "response_s": res.response_s,
        "objective": objective,
        "solver": _solver_meta(res),
    }


def _expand_value(T, idx, weights, objective, threshold_s) -> float:

    return objective_value(T, idx, weights, objective=objective, threshold_s=threshold_s)


def random_expand_baseline(
    T: np.ndarray,
    weights: np.ndarray,
    existing_idx: Sequence[int],
    add_m: int,
    *,
    new_pool: Sequence[int],
    objective: str = "response_time",
    threshold_s: float = 480.0,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    existing = [int(i) for i in existing_idx]
    pool = [int(i) for i in new_pool if int(i) not in set(existing)]
    m = min(int(add_m), len(pool))
    opened = [int(x) for x in rng.choice(pool, size=m, replace=False)] if m > 0 else []
    selected = existing + opened
    return {
        "indices": selected,
        "opened": opened,
        "value": _expand_value(T, selected, weights, objective, threshold_s),
        "response_s": T[selected, :].min(axis=0),
        "label": "random_expand",
    }


# --------------------------------------------------------------------------- #
# Turnout / Kolesar fit / busy / year-split helpers
# --------------------------------------------------------------------------- #


def _norm_name(x: Any) -> str:
    return " ".join(str(x).strip().lower().split())


def _station_ground_pairs(
    incidents: pd.DataFrame, houses: pd.DataFrame
) -> pd.DataFrame | None:
    """Rows of (station, crow_km, observed_s) for non-busy calls (house = origin)."""
    need = {"station_ground", "travel_seconds", "dest_lat", "dest_lon"}
    if not need.issubset(incidents.columns):
        return None
    df = incidents.dropna(subset=list(need)).copy()
    if "deployed_from_station" in df.columns:
        same = df["station_ground"].map(_norm_name) == df["deployed_from_station"].map(_norm_name)
        df = df[same]
    df["_key"] = df["station_ground"].map(_norm_name)
    h = houses.assign(_key=houses["facilityname"].map(_norm_name))[["_key", "lat", "lon"]]
    h = h.drop_duplicates("_key")
    df = df.merge(h, on="_key", how="inner")
    if df.empty:
        return None
    df["crow_km"] = haversine_km(df["lon"], df["lat"], df["dest_lon"], df["dest_lat"])
    df["observed_s"] = df["travel_seconds"].astype(float)
    df = df[df["crow_km"].between(0.05, 40.0) & df["observed_s"].between(60, 1800)]
    return df[["_key", "crow_km", "observed_s"]].rename(columns={"_key": "station"}) if len(df) else None


def fit_scorer_kolesar(
    incidents: pd.DataFrame,
    houses: pd.DataFrame,
    *,
    min_pairs: int = 200,
    max_pairs: int = 50_000,
    turnout_s: float = DEFAULT_TURNOUT_S,
    seed: int = 0,
) -> tuple[KolesarModel, dict[str, Any], pd.DataFrame | None]:
    """Fit a **drive-only** Kolesar model from house→dest crow km.

    LFB ``travel_seconds`` is attendance (turnout + drive), so ``turnout_s`` is
    subtracted before fitting; the planner adds turnout back per station.
    Falls back to the London default curve when too few usable pairs exist.
    """
    pairs = _station_ground_pairs(incidents, houses)
    if pairs is None or len(pairs) < min_pairs:
        return (
            KolesarModel.london_default(),
            {"source": "london_default", "n_pairs": 0 if pairs is None else int(len(pairs))},
            pairs,
        )
    fit_df = pairs.sample(max_pairs, random_state=seed) if len(pairs) > max_pairs else pairs
    drive = np.maximum(fit_df["observed_s"].to_numpy() - float(turnout_s), 30.0)
    model = fit_kolesar(fit_df["crow_km"].to_numpy(), drive)
    return model, {"source": "fit_house_to_dest_crow_km", "n_pairs": int(len(fit_df)),
                   "turnout_removed_s": float(turnout_s)}, pairs


def estimate_station_turnout(
    pairs: pd.DataFrame | None,
    model: KolesarModel,
    *,
    default_s: float = DEFAULT_TURNOUT_S,
    min_n: int = 20,
    clip: tuple[float, float] = (30.0, 180.0),
) -> dict[str, float]:
    """Per-station turnout: default + median(observed - default - Kolesar drive)."""
    if pairs is None or pairs.empty:
        return {}
    resid = pairs["observed_s"].to_numpy() - default_s - model.predict(pairs["crow_km"].to_numpy())
    df = pd.DataFrame({"station": pairs["station"].to_numpy(), "resid": resid})
    g = df.groupby("station")["resid"].agg(["median", "size"])
    g = g[g["size"] >= min_n]
    return {k: float(np.clip(default_s + v, *clip)) for k, v in g["median"].items()}


def recorded_station_turnout(
    years: Sequence[int] | None = None,
    *,
    path: Path = TRAVEL_TRAINING,
    min_n: int = 50,
) -> tuple[dict[str, float], float] | None:
    """Median recorded turnout per station from LFB mobilisation records.

    Uses home-station engine trips (``scripts/build_lfb_travel_training.py``) in ``years``;
    returns ``({station: seconds}, median over stations)`` or None if the file is missing.
    """
    if not path.exists():
        return None
    t = pd.read_csv(path, usecols=["cal_year", "deployed_from_station", "deployed_from_location", "turnout_s"])
    t = t[t["deployed_from_location"].eq("Home Station") & t["turnout_s"].between(1, 600)]
    if years:
        t = t[t["cal_year"].isin({int(y) for y in years})]
    g = t.groupby("deployed_from_station")["turnout_s"].agg(["median", "size"])
    g = g[g["size"] >= min_n]
    if g.empty:
        return None
    return {str(k): float(v) for k, v in g["median"].items()}, float(g["median"].median())


def resolve_turnout_vector(
    candidates: pd.DataFrame,
    turnout_s: float | Mapping[str, float] | None,
    estimated: Mapping[str, float] | None = None,
    *,
    default_s: float = DEFAULT_TURNOUT_S,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Turnout per candidate row. Scalar → all rows; dict/estimates → existing by name."""
    n = len(candidates)
    if isinstance(turnout_s, (int, float)) and not isinstance(turnout_s, bool):
        return np.full(n, float(turnout_s)), {"source": "scalar", "default_s": float(turnout_s)}
    table = {_norm_name(k): float(v) for k, v in (estimated or {}).items()}
    source = "estimated_station_prior" if table else "default"
    if isinstance(turnout_s, Mapping):
        table.update({_norm_name(k): float(v) for k, v in turnout_s.items()})
        source = "mapping" + ("+estimated" if estimated else "")
    vec = np.full(n, float(default_s))
    existing = candidates["is_existing"].to_numpy(dtype=bool)
    keys = candidates["site_name"].map(_norm_name).to_numpy()
    hit = 0
    for i in range(n):
        if existing[i] and keys[i] in table:
            vec[i] = table[keys[i]]
            hit += 1
    vec = np.clip(vec, 0.0, 300.0)
    ex = vec[existing] if existing.any() else vec
    return vec, {
        "source": source, "default_s": float(default_s), "n_existing_with_prior": hit,
        "existing_min_s": float(ex.min()), "existing_median_s": float(np.median(ex)),
        "existing_max_s": float(ex.max()),
    }


def busy_prob_vector(
    candidates: pd.DataFrame, incidents: pd.DataFrame, *, min_n: int = 20
) -> np.ndarray | None:
    """Busy probability per candidate (new sites get the existing-house mean)."""
    if not {"station_ground", "deployed_from_station"}.issubset(incidents.columns):
        return None
    rates = estimate_busy_rates(incidents)["by_station"]
    rates = rates[rates["n_incidents"] >= min_n]
    if rates.empty:
        return None
    table = {_norm_name(k): float(v) for k, v in zip(rates["station_ground"], rates["busy_prob"])}
    keys = candidates["site_name"].map(_norm_name).to_numpy()
    existing = candidates["is_existing"].to_numpy(dtype=bool)
    p = np.array([table.get(k, np.nan) if e else np.nan for k, e in zip(keys, existing)])
    mean_p = float(np.nanmean(p)) if np.isfinite(p).any() else float(np.mean(list(table.values())))
    return np.where(np.isfinite(p), p, mean_p)


def posture_response(A: np.ndarray, idx: Sequence[int], busy_p: np.ndarray | None = None) -> np.ndarray:
    """Per-cell first-engine seconds; with ``busy_p`` blend nearest with next-nearest."""
    idx = np.asarray(list(idx), dtype=int)
    sub = A[idx, :]
    if busy_p is None or len(idx) < 2:
        return sub.min(axis=0)
    order = np.argsort(sub, axis=0)[:2]
    cols = np.arange(sub.shape[1])
    t1, t2 = sub[order[0], cols], sub[order[1], cols]
    return expected_first_arrival(t1, busy_p[idx[order[0]]], t2)


def snap_to_cells(points: pd.DataFrame, cells: pd.DataFrame, *, chunk: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Nearest demand cell (crow km) for each incident; returns (cell_idx, km)."""
    lat0 = float(cells["lat"].mean())
    kx = 111.32 * np.cos(np.radians(lat0))
    cx, cy = cells["lon"].to_numpy(float) * kx, cells["lat"].to_numpy(float) * 110.574
    px, py = points["dest_lon"].to_numpy(float) * kx, points["dest_lat"].to_numpy(float) * 110.574
    idx = np.empty(len(points), dtype=int)
    dist = np.empty(len(points))
    for s in range(0, len(points), chunk):
        d2 = (px[s:s + chunk, None] - cx[None, :]) ** 2 + (py[s:s + chunk, None] - cy[None, :]) ** 2
        j = d2.argmin(axis=1)
        idx[s:s + chunk] = j
        dist[s:s + chunk] = np.sqrt(d2[np.arange(len(j)), j])
    return idx, dist


def drop_far_incidents(
    incidents: pd.DataFrame, houses: pd.DataFrame, *, max_km: float = 60.0
) -> pd.DataFrame:
    """Drop geocode outliers farther than ``max_km`` from the median house location."""
    c_lat, c_lon = float(houses["lat"].median()), float(houses["lon"].median())
    km = haversine_km(incidents["dest_lon"], incidents["dest_lat"], c_lon, c_lat)
    return incidents[np.asarray(km) <= max_km]


def _split_years(
    incidents: pd.DataFrame,
    demand_years: Sequence[int] | None,
    eval_years: Sequence[int] | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Demand / eval incident frames by ``cal_year`` (no-op without that column)."""
    if "cal_year" not in incidents.columns:
        return incidents, None
    yr = pd.to_numeric(incidents["cal_year"], errors="coerce")
    demand = incidents
    if demand_years:
        demand = incidents[yr.isin({int(y) for y in demand_years})]
        if demand.empty:
            raise ValueError(f"No incidents for demand_years={list(demand_years)}")
    ev = None
    if eval_years:
        ev = incidents[yr.isin({int(y) for y in eval_years})]
        if ev.empty:
            ev = None
    return demand.reset_index(drop=True), None if ev is None else ev.reset_index(drop=True)


@dataclass
class FireStationPlanResult:
    mode: str
    city_id: str
    candidates: pd.DataFrame
    cells: pd.DataFrame
    selected: pd.DataFrame
    closed: pd.DataFrame
    opened: pd.DataFrame
    metrics: dict[str, Any]
    baselines: dict[str, Any]
    travel_meta: dict[str, Any]
    summary: dict[str, Any]
    attendance: np.ndarray | None = None  # candidate x cell attendance seconds (drive + turnout)


def run_fire_station_plan(
    *,
    city_id: str,
    houses: pd.DataFrame,
    incidents: pd.DataFrame,
    mode: str = "redesign",
    n_houses: int | None = None,
    replace_r: int = 0,
    add_stations: int = 0,
    cell_km: float = 0.9,
    demand_candidates: int = 80,
    spread_km: float | None = 3.0,
    objective: str = "response_time",
    threshold_min: float = 10.0,
    cap_min: float | None = 10.0,
    max_incident_rows: int | None = 200_000,
    seed: int = 42,
    scorer: str = "kolesar",
    turnout_s: float | Mapping[str, float] | None = None,
    demand_years: Sequence[int] | None = None,
    eval_years: Sequence[int] | None = None,
    use_busy: bool = False,
    pumps: Mapping[str, int] | None = None,
    total_pumps: int | None = None,
    min_pumps: int = 1,
    max_pumps: int = 3,
    new_site_pumps: int = 1,
    expand_pumps: str = "move",
    refine_km: float = 0.0,
    keep_slowest: bool = False,
    keep_average: bool = False,
    keep_boroughs: float | None = None,
    busy_rounds: int | None = None,
    solver_time_s: float | None = None,
    severity: str = "all",
    alpha: float = 1.0,
    solver: str = "exact",
) -> FireStationPlanResult:
    """End-to-end facility location for one city.

    Response = drive (``scorer``) + turnout. ``turnout_s``: scalar (all sites), a
    ``{station_name: seconds}`` mapping, or ``None`` (estimate a per-station prior
    from LFB attendance residuals; new sites get ``DEFAULT_TURNOUT_S``).
    ``demand_years`` builds the demand grid; ``eval_years`` (if present in the data)
    re-scores current vs plan on held-out incidents snapped to demand cells.
    ``pumps`` / ``total_pumps`` control per-station engine counts (``resize`` mode
    reallocates a fixed pump budget; other modes attach ``n_pumps`` for reporting
    and optional busy scaling when ``use_busy`` is set).
    ``severity`` weights calls by type (``SEVERITY_SCHEMES``); ``alpha`` sets the trade-off
    the solver optimises: each area counts as (its weighted calls)**alpha, so 1 minimises
    the call-weighted average time and 0 the average over areas (equal coverage).
    Metrics are always reported with the severity weights.
    ``expand_pumps``: where a new station's pump comes from when more sites open than
    close: ``move`` (second pump of the nearest two-pump station; total fixed) or ``new``.
    Pumps always move with their station (``busy_placement.PumpPlan``).
    ``keep_slowest`` / ``keep_average``: floors, the plan's slowest-10 % average / overall
    average (severity-weighted calls, busy engines when on) may not be worse than today's.
    ``keep_boroughs`` (seconds, e.g. 1.0): no borough's call-weighted average may get worse
    than today's by more than this.
    ``refine_km`` > 0 moves each newly opened site to the best road junction within that
    distance after solving (``site_refine``; today's stations stay put).
    ``solver``: ``exact`` (integer program, proven optimum, nearest engine always free) or
    ``genetic`` (genetic algorithm; with ``use_busy`` it optimises the busy-engine blend).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be {'|'.join(MODES)}, got {mode!r}")
    if scorer not in SCORERS:
        raise ValueError(f"scorer must be {'|'.join(SCORERS)}, got {scorer!r}")

    has_years = "cal_year" in incidents.columns
    n_raw = len(incidents)
    incidents = drop_far_incidents(incidents, houses)
    n_far = n_raw - len(incidents)
    incidents, eval_inc = _split_years(incidents, demand_years, eval_years)
    if max_incident_rows is not None:
        if len(incidents) > max_incident_rows:
            incidents = incidents.sample(max_incident_rows, random_state=seed).reset_index(drop=True)
        if eval_inc is not None and len(eval_inc) > max_incident_rows:
            eval_inc = eval_inc.sample(max_incident_rows, random_state=seed).reset_index(drop=True)

    # Grid geometry from all calls (so sites and travel times are shared across schemes);
    # each cell's demand is then its severity-weighted call count.
    cells = build_demand_grid(incidents, lat_col="dest_lat", lon_col="dest_lon", cell_km=cell_km)
    # resize only needs existing sites — skip demand-cell candidates for speed
    demand_cand = 0 if mode == "resize" else demand_candidates
    candidates = build_house_candidates(houses, cells, demand_candidates=demand_cand,
                                        spread_km=None if mode == "resize" else spread_km)
    sev_w = severity_weight(incidents, severity)
    cells["weight"] = cell_weights(cells, incidents, sev_w)
    cells["share"] = cells["weight"] / max(float(cells["weight"].sum()), 1e-9)
    weights = cells["weight"].to_numpy(dtype=float)
    w_opt = solver_weights(weights, alpha)
    threshold_s = float(threshold_min) * 60.0
    cap_s = float(cap_min) * 60.0 if cap_min else None

    existing_idx = candidates.index[candidates["is_existing"]].tolist()
    if not existing_idx:
        existing_idx = [i for i in range(len(houses)) if i < len(candidates)]

    # --- pump / engine capacity per site -----------------------------------
    if pumps is None:
        pump_df = default_pump_table(houses)
        pumps = {str(r.station): int(r.n_pumps) for r in pump_df.itertuples()}
    baseline_pumps = resolve_pump_vector(
        candidates, pumps=pumps, default_pumps=1, new_site_pumps=new_site_pumps
    )
    candidates["n_pumps_baseline"] = baseline_pumps
    print(
        f"Pumps: {int(baseline_pumps[candidates['is_existing'].to_numpy()].sum())} on "
        f"{int(candidates['is_existing'].sum())} existing stations "
        f"({int((baseline_pumps[candidates['is_existing'].to_numpy()] >= 2).sum())} with ≥2)"
    )

    # --- Kolesar drive model (scorer + turnout prior) -----------------------
    kmodel, kmeta, pairs = fit_scorer_kolesar(incidents, houses, seed=seed)
    print(f"Kolesar drive model: {kmeta['source']} (pairs={kmeta['n_pairs']})")

    # --- turnout per candidate ---------------------------------------------
    estimated = None
    turnout_default = DEFAULT_TURNOUT_S
    recorded = recorded_station_turnout(demand_years) if turnout_s is None and city_id == "london" else None
    if recorded is not None:
        estimated, turnout_default = recorded
    elif turnout_s is None and kmeta["source"] != "london_default":
        estimated = estimate_station_turnout(pairs, kmodel)
    turnout_vec, turnout_meta = resolve_turnout_vector(candidates, turnout_s, estimated, default_s=turnout_default)
    if recorded is not None:
        turnout_meta["source"] = "recorded_mobilisation_median"
        turnout_meta["years"] = list(demand_years) if demand_years else "all"
    print("Turnout:", turnout_meta)

    print(f"Building {scorer} travel matrix ({len(candidates)} sites × {len(cells)} cells)…")
    T, travel_meta = build_travel_matrix(
        candidates,
        cells,
        surrogate=TravelTimeSurrogate(),
        scorer=scorer,
        kolesar=kmodel,
        incidents=incidents,
    )
    travel_meta["kolesar_fit"] = kmeta
    travel_meta["turnout"] = turnout_meta
    travel_meta["pumps"] = {
        "baseline_total": int(baseline_pumps[candidates["is_existing"].to_numpy()].sum()),
        "n_two_plus": int((baseline_pumps[candidates["is_existing"].to_numpy()] >= 2).sum()),
        "min_pumps": int(min_pumps),
        "max_pumps": int(max_pumps),
    }
    # Attendance = drive + station turnout; everything below optimises/scores A.
    A = attendance_seconds(T, turnout_vec[:, None])

    busy_base = busy_prob_vector(candidates, incidents)
    busy_p = None
    busy_model = None
    from .busy_placement import PumpPlan

    pump_plan = PumpPlan(baseline_pumps.astype(int), np.asarray(existing_idx, dtype=int),
                         candidates["lat"].to_numpy(float), candidates["lon"].to_numpy(float),
                         extra=expand_pumps, new_site_pumps=new_site_pumps)
    if use_busy and mode != "resize":
        # Load-dependent busy rates (Erlang loss), calibrated to today's measured rates;
        # pumps follow their station in every layout the solver tries.
        from .busy_placement import ErlangBusyModel

        if busy_base is None:
            busy_base = np.full(len(candidates), 0.25)
            print("Busy rates unavailable — using uniform 0.25 prior.")
        busy_model = ErlangBusyModel.calibrate(
            A, existing_idx, busy_base, baseline_pumps, cells["n_incidents"].to_numpy(dtype=float),
            pump_plan=pump_plan)
        busy_p = busy_model.probs(A, existing_idx)
        ex_mask = candidates["is_existing"].to_numpy()
        travel_meta["busy"] = {
            "used": True, "model": "Erlang loss, load-dependent (busy_placement.ErlangBusyModel)",
            "mean_existing_busy_prob": round(float(busy_p[ex_mask].mean()), 4),
            "median_erlangs_per_1000_calls": round(1000 * float(np.median(busy_model.erlangs_per_call[ex_mask])), 3),
            "measured_busy_rate_by_pumps": {
                f"{int(c)}_pump": round(float(np.mean(busy_base[ex_mask & (baseline_pumps == c)])), 4)
                for c in sorted(set(baseline_pumps[ex_mask].astype(int)))},
        }
    # Capacity-aware busy is required for resize
    elif mode == "resize":
        if busy_base is None:
            # Uniform prior when deploy columns are missing
            busy_base = np.full(len(candidates), 0.25)
            print("Busy rates unavailable — using uniform 0.25 prior for capacity model.")
        busy_p = capacity_busy_prob(busy_base, baseline_pumps, baseline_pumps=baseline_pumps)
        travel_meta["busy"] = {
            "used": True,
            "capacity_scaled": True,
            "mean_existing_busy_prob": round(
                float(busy_p[candidates["is_existing"].to_numpy()].mean()), 4
            ),
        }

    # --- current posture ----------------------------------------------------
    cur_sel = existing_idx
    cur_resp = posture_response(A, cur_sel, busy_p) if cur_sel else np.full(len(cells), np.nan)
    cur_metrics = posture_metrics(
        cur_resp, weights, threshold_s=threshold_s, label="current_fire_stations", n_posts=len(cur_sel)
    )
    cur_metrics["lfb_first_engine"] = score_first_engine(cur_resp, weights)
    cur_metrics["total_pumps"] = int(baseline_pumps[np.asarray(cur_sel)].sum()) if cur_sel else 0
    baselines = {"current_fire_stations": cur_metrics}

    # Floors: the plan may not be worse than today on the slowest 10 % / the average.
    floors = {}
    today_mean = float((weights * cur_resp).sum() / max(weights.sum(), 1e-9))
    today_cvar = weighted_cvar(cur_resp, weights)
    if keep_slowest:
        floors["max_cvar"] = today_cvar
    if keep_average:
        floors["max_mean"] = today_mean
    borough_info = None
    if keep_boroughs is not None:
        from .ml_scorer import cell_boroughs

        names = cell_boroughs(cells, incidents)
        uniq = sorted({n for n in names if n})
        code = np.array([uniq.index(n) if n else -1 for n in names])
        today_b = group_means(cur_resp, weights, code, len(uniq))
        floors["group_floor"] = (code, today_b + float(keep_boroughs))
        borough_info = {"n_boroughs": len(uniq), "tolerance_s": float(keep_boroughs)}
    if floors:
        floors["floor_weights"] = weights
    if solver == "exact" and solver_time_s:
        floors["time_limit_s"] = float(solver_time_s)
    if solver == "exact" and busy_rounds and use_busy:
        floors["max_rounds"] = int(busy_rounds)
    # 10-minute cap: no area that today's stations reach in time may lose the standard.
    cap_cells = A[np.asarray(existing_idx)].min(axis=0) <= cap_s if cap_s is not None else None

    closed_df = pd.DataFrame()
    opened_df = pd.DataFrame()
    new_pool = [i for i in range(len(candidates)) if i not in set(existing_idx)]
    plan_pumps = baseline_pumps.copy()

    if mode == "redesign":
        k = int(n_houses) if n_houses is not None else len(existing_idx)
        k = max(1, min(k, len(candidates)))
        print(f"Redesign: placing k={k} fire stations…")
        res = plan_redesign(A, w_opt, k, objective=objective, threshold_s=threshold_s, cap_s=cap_s,
                            solver=solver, busy_model=busy_model, start_idx=existing_idx, cap_cells=cap_cells,
                            floors=floors)
        selected_idx = res["indices"]
        plan_value = res["value"]
        kept = [i for i in selected_idx if i in set(existing_idx)]
        opened_idx = [i for i in selected_idx if i not in set(existing_idx)]
        closed_idx = [i for i in existing_idx if i not in set(selected_idx)]
        closed_df = candidates.loc[closed_idx].copy() if closed_idx else pd.DataFrame()
        opened_df = candidates.loc[opened_idx].copy() if opened_idx else pd.DataFrame()
        extra = {"k": k, "kept_existing": len(kept), "solver": res["solver"]}
    elif mode == "replace":
        r = int(replace_r)
        if r <= 0:
            raise ValueError("--replace must be > 0 for mode=replace")
        print(f"Replace: relocating r={r} of {len(existing_idx)} fire stations…")
        res = plan_replace(
            A, w_opt, existing_idx,
            replace_r=r, new_pool=new_pool, objective=objective, threshold_s=threshold_s, cap_s=cap_s,
            solver=solver, busy_model=busy_model, start_idx=existing_idx, cap_cells=cap_cells, floors=floors,
        )
        selected_idx = res["indices"]
        plan_value = res["value"]
        closed_df = candidates.loc[res["closed"]].copy() if res["closed"] else pd.DataFrame()
        opened_df = candidates.loc[res["opened"]].copy() if res["opened"] else pd.DataFrame()
        rnd = random_replace_baseline(
            A, weights, existing_idx,
            replace_r=r, new_pool=new_pool, objective=objective, threshold_s=threshold_s, seed=seed,
        )
        baselines["random_replace"] = _baseline_metrics(
            A, rnd["indices"], weights, threshold_s,
            busy_model.probs(A, rnd["indices"]) if busy_model else busy_p, "random_replace"
        )
        extra = {"replace_r": r, "n_replaced": len(res["opened"]), "solver": res["solver"]}
    elif mode == "expand":
        m = int(add_stations)
        if m <= 0:
            raise ValueError("--add-stations must be > 0 for mode=expand")
        print(f"Expand: keeping {len(existing_idx)} houses, opening m={m} new…")
        res = plan_expand(
            A, w_opt, existing_idx, m,
            new_pool=new_pool, objective=objective, threshold_s=threshold_s, cap_s=cap_s,
            solver=solver, busy_model=busy_model, start_idx=existing_idx, cap_cells=cap_cells, floors=floors,
        )
        selected_idx = res["indices"]
        plan_value = res["value"]
        opened_df = candidates.loc[res["opened"]].copy() if res["opened"] else pd.DataFrame()
        rnd = random_expand_baseline(
            A, weights, existing_idx, m,
            new_pool=new_pool, objective=objective, threshold_s=threshold_s, seed=seed,
        )
        baselines["random_expand"] = _baseline_metrics(
            A, rnd["indices"], weights, threshold_s,
            busy_model.probs(A, rnd["indices"]) if busy_model else busy_p, "random_expand"
        )
        extra = {"add_stations": m, "solver": res["solver"]}
    else:  # resize
        budget = int(total_pumps) if total_pumps is not None else int(
            baseline_pumps[np.asarray(existing_idx)].sum() or DEFAULT_TOTAL_PUMPS
        )
        print(
            f"Resize: reallocating {budget} pumps across {len(existing_idx)} fixed stations "
            f"(min={min_pumps}, max={max_pumps})…"
        )
        res = plan_resize_capacity(
            A, weights, existing_idx,
            base_busy=busy_base if busy_base is not None else np.full(len(candidates), 0.25),
            baseline_pumps=baseline_pumps,
            total_pumps=budget,
            min_pumps=min_pumps,
            max_pumps=max_pumps,
            threshold_s=threshold_s,
        )
        selected_idx = res["indices"]
        plan_value = res["value"]
        plan_pumps = res["n_pumps_full"]
        busy_p = res["busy_p"]
        extra = {
            "total_pumps": res["total_pumps"],
            "n_moves": res["n_moves"],
            "min_pumps": int(min_pumps),
            "max_pumps": int(max_pumps),
            "n_upgraded": int((res["n_pumps"] > baseline_pumps[np.asarray(existing_idx)]).sum()),
            "n_downgraded": int((res["n_pumps"] < baseline_pumps[np.asarray(existing_idx)]).sum()),
        }

    refine_meta = None
    pre_refine_idx = list(selected_idx)
    new_open = [i for i in selected_idx if i not in set(existing_idx)]
    if refine_km and mode != "resize" and new_open:
        from .busy_placement import PumpPlan as _PumpPlan, extend_busy_model
        from .site_refine import refine_sites

        n0 = len(candidates)
        t_new = float(turnout_meta["default_s"])
        W_opt = max(float(w_opt.sum()), 1e-9)

        def make_rows(df):
            Tn, _ = build_travel_matrix(df, cells, surrogate=TravelTimeSurrogate(), scorer=scorer,
                                        kolesar=kmodel, incidents=incidents)
            return attendance_seconds(Tn, t_new)

        def objective_fn(Ax, idx, cand):
            if busy_model is None:
                resp = Ax[idx].min(axis=0)
            else:
                m = extend_busy_model(busy_model, cand["lat"].to_numpy(float)[n0:], cand["lon"].to_numpy(float)[n0:])
                resp = m.response(Ax, idx)
            return weighted_cvar(resp, w_opt) if objective == "cvar" else float((w_opt * resp).sum() / W_opt)

        def cap_ok(Ax, idx, cand):
            if cap_cells is not None and not bool((Ax[idx].min(axis=0)[cap_cells] <= cap_s).all()):
                return False
            if not floors:
                return True
            if busy_model is None:
                resp = Ax[idx].min(axis=0)
            else:
                m = extend_busy_model(busy_model, cand["lat"].to_numpy(float)[n0:], cand["lon"].to_numpy(float)[n0:])
                resp = m.response(Ax, idx)
            return floors_met(resp, weights, floors.get("max_mean"), floors.get("max_cvar"), floors.get("group_floor"))

        print(f"Refining {len(new_open)} new site(s) to street junctions within {refine_km} km…")
        ref = refine_sites(A, candidates, cells, selected_idx, new_open, w_opt, make_rows=make_rows,
                           objective=objective_fn, cap_ok=cap_ok, radius_km=refine_km)
        n_add = len(ref["candidates"]) - n0
        candidates = ref["candidates"]
        A = ref["A"]
        T = np.vstack([T, A[n0:] - t_new])
        turnout_vec = np.r_[turnout_vec, np.full(n_add, t_new)]
        baseline_pumps = np.r_[baseline_pumps, np.full(n_add, new_site_pumps, dtype=baseline_pumps.dtype)]
        new_lat, new_lon = candidates["lat"].to_numpy(float)[n0:], candidates["lon"].to_numpy(float)[n0:]
        if busy_model is not None:
            busy_model = extend_busy_model(busy_model, new_lat, new_lon)
            pump_plan = busy_model.pump_plan
            busy_p = busy_model.probs(A, existing_idx)
        else:
            pump_plan = _PumpPlan(np.r_[pump_plan.base_pumps, np.full(n_add, new_site_pumps, dtype=int)],
                                  pump_plan.existing, np.r_[pump_plan.lat, new_lat], np.r_[pump_plan.lon, new_lon],
                                  extra=pump_plan.extra, new_site_pumps=new_site_pumps)
        selected_idx = ref["indices"]
        plan_value = ref["value"]
        opened_df = candidates.loc[[i for i in selected_idx if i not in set(existing_idx)]].copy()
        refine_meta = {"radius_km": refine_km, "solve_s": ref["solve_s"], "n_moved": len(ref["log"]),
                       "moves": ref["log"]}
        print(f"  moved {len(ref['log'])} of {len(new_open)} sites in {ref['solve_s']} s")

    pump_moves = []
    if mode != "resize":
        plan_pumps, pump_moves = pump_plan.assign(selected_idx)
        names = candidates["site_name"].astype(str).to_numpy()
        pump_moves = [{**m, "from_name": None if m["from"] is None else names[m["from"]],
                       "to_name": names[m["to"]]} for m in pump_moves]
    plan_busy_p = busy_model.probs(A, selected_idx) if busy_model is not None else busy_p
    busy_compare = None
    if busy_model is not None and mode in {"redesign", "replace", "expand"}:
        # Does modelling busy engines change the answer? Re-solve assuming the nearest
        # engine is always free, then score that layout under the busy model too.
        kw_free = dict(objective=objective, threshold_s=threshold_s, cap_s=cap_s, cap_cells=cap_cells)
        if mode == "redesign":
            free = plan_redesign(A, w_opt, len(selected_idx), **kw_free)
        elif mode == "replace":
            free = plan_replace(A, w_opt, existing_idx, replace_r=int(replace_r), new_pool=new_pool, **kw_free)
        else:
            free = plan_expand(A, w_opt, existing_idx, int(add_stations), new_pool=new_pool, **kw_free)
        free_idx = free["indices"]
        baselines["plan_ignoring_busy"] = _baseline_metrics(
            A, free_idx, weights, threshold_s, busy_model.probs(A, free_idx), "plan_ignoring_busy")
        busy_compare = {
            "sites_differ": int(len(set(free_idx) ^ set(pre_refine_idx)) // 2),  # before street refinement
            "busy_aware_s": round(float((weights * posture_response(A, selected_idx, plan_busy_p)).sum()
                                        / max(weights.sum(), 1e-9)), 2),
            "ignoring_busy_s": baselines["plan_ignoring_busy"]["expected_response_s"],
        }
    plan_resp = posture_response(A, selected_idx, plan_busy_p)
    selected = candidates.loc[selected_idx].copy().reset_index(drop=True)
    selected["plan_rank"] = np.arange(len(selected))
    selected["turnout_s"] = turnout_vec[selected_idx]
    selected["n_pumps"] = plan_pumps[np.asarray(selected_idx)].astype(int)
    selected["n_pumps_baseline"] = baseline_pumps[np.asarray(selected_idx)].astype(int)
    selected["delta_pumps"] = selected["n_pumps"] - selected["n_pumps_baseline"]
    assign = A[selected_idx, :].argmin(axis=0)
    site_w = np.zeros(len(selected_idx))
    for j, a in enumerate(assign):
        site_w[a] += weights[j]
    selected["assigned_demand"] = site_w
    selected["assigned_share"] = site_w / max(float(weights.sum()), 1e-9)

    plan_metrics = posture_metrics(
        plan_resp, weights, threshold_s=threshold_s, label=f"plan_{mode}", n_posts=len(selected_idx)
    )
    plan_metrics["objective_value"] = float(plan_value)  # nearest-engine objective
    plan_metrics["lfb_first_engine"] = score_first_engine(plan_resp, weights)
    plan_metrics["total_pumps"] = int(plan_pumps[np.asarray(selected_idx)].sum())

    # --- held-out evaluation on eval_years incidents ------------------------
    eval_summary = None
    if eval_inc is not None and len(eval_inc):
        ok = eval_inc[["dest_lat", "dest_lon"]].notna().all(axis=1)
        ev = eval_inc[ok].reset_index(drop=True)
        cidx, ckm = snap_to_cells(ev, cells)
        ev_w = severity_weight(ev, severity)
        cur_score = score_first_engine(cur_resp[cidx], ev_w)
        plan_score = score_first_engine(plan_resp[cidx], ev_w)
        cur_metrics["lfb_first_engine_eval"] = cur_score
        plan_metrics["lfb_first_engine_eval"] = plan_score
        eval_summary = {
            "years": [int(y) for y in (eval_years or [])],
            "n_incidents": int(len(ev)),
            "n_incidents_weighted": float(ev_w.sum()),
            "snap_median_km": round(float(np.median(ckm)), 3),
            "snap_p95_km": round(float(np.percentile(ckm, 95)), 3),
            "current": cur_score,
            "plan": plan_score,
        }
        if "travel_seconds" in ev.columns and ev["travel_seconds"].notna().any():
            eval_summary["observed_attendance"] = score_first_engine(ev["travel_seconds"].dropna())

    cells_out = cells.copy()
    cells_out["response_s"] = plan_resp
    cells_out["response_s_current"] = cur_resp
    cells_out["response_s_delta"] = plan_resp - cur_resp
    cells_out["assigned_site"] = [selected_idx[int(a)] for a in assign]
    cells_out["drive_s"] = T[np.asarray(selected_idx)[assign], np.arange(len(cells))]
    cells_out["covered"] = plan_resp <= threshold_s
    cells_out["covered_current"] = cur_resp <= threshold_s

    cur_et = cur_metrics["expected_response_s"]
    plan_et = plan_metrics["expected_response_s"]
    summary = {
        "city_id": city_id,
        "mode": mode,
        "objective": objective,
        "severity": severity,
        "severity_rule": SEVERITY_SCHEMES[severity],
        "alpha": float(alpha),
        "solver_kind": solver,
        "scorer": scorer,
        "demand_years": [int(y) for y in (demand_years or [])] if has_years else [],
        "eval_years": [int(y) for y in (eval_years or [])] if has_years else [],
        "use_busy": bool(busy_p is not None),
        "n_existing_fire_stations": len(existing_idx),
        "n_candidates": len(candidates),
        "n_demand_cells": len(cells),
        "n_incidents": int(len(incidents)),
        "n_incidents_dropped_far": int(n_far),
        "n_selected": len(selected),
        "n_closed": int(len(closed_df)),
        "n_opened": int(len(opened_df)),
        "threshold_min": threshold_min,
        "total_pumps_baseline": int(baseline_pumps[np.asarray(existing_idx)].sum()),
        "total_pumps_plan": int(plan_pumps[np.asarray(selected_idx)].sum()),
        "plan": plan_metrics,
        "baselines": baselines,
        "eval": eval_summary,
        "delta_vs_current_s": round(float(plan_et - cur_et), 1) if cur_et == cur_et else None,
        "travel": travel_meta,
        "busy_vs_ignoring_busy": busy_compare,
        "pump_moves": pump_moves,
        "street_refinement": refine_meta,
        "floors": {"keep_slowest": keep_slowest, "keep_average": keep_average, "keep_boroughs": borough_info,
                   "today_mean_s": round(today_mean, 2), "today_slowest10_s": round(today_cvar, 2)},
        "expand_pumps": expand_pumps,
        **extra,
        "honesty": (
            "Response = predicted drive + station turnout (LFB attendance is mobilise->arrive; "
            "travel-only scoring was wrong). Judged on LFB first-engine KPIs: mean <= 6 min, "
            "> 90% within 10 min. Drive comes from the chosen scorer "
            f"({scorer}). Turnout: {turnout_meta['source']} (existing stations), "
            f"{turnout_meta['default_s']:.0f} s for new sites. Discrete candidates (existing houses + "
            "demand centroids); house sites are OpenStreetMap station buildings. No land, zoning or staffing "
            "constraints. Station size (n_pumps) defaults from LFB two-pump inference "
            "(~40 stations with a second appliance). resize mode reallocates a fixed pump budget "
            "under a capacity-scaled busy model; other modes keep baseline pumps unless overridden. "
            "Busy blend applies when use_busy=True or mode=resize "
            "(location optimisation still uses nearest engine)."
        ),
    }

    return FireStationPlanResult(
        mode=mode,
        city_id=city_id,
        candidates=candidates,
        cells=cells_out,
        selected=selected,
        closed=closed_df.reset_index(drop=True) if len(closed_df) else closed_df,
        opened=opened_df.reset_index(drop=True) if len(opened_df) else opened_df,
        metrics=plan_metrics,
        baselines=baselines,
        travel_meta=travel_meta,
        summary=summary,
        attendance=A,
    )


def _baseline_metrics(A, idx, weights, threshold_s, busy_p, label) -> dict[str, Any]:
    resp = posture_response(A, idx, busy_p)
    out = posture_metrics(resp, weights, threshold_s=threshold_s, label=label, n_posts=len(idx))
    out["lfb_first_engine"] = score_first_engine(resp, weights)
    return out


def demo_inputs(*, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic houses + incidents for --demo."""
    incidents, stations = synthetic_demo_data(seed=seed)
    houses = pd.DataFrame(
        {
            "facilityname": stations["facname"].astype(str),
            "lat": pd.to_numeric(stations["latitude"], errors="coerce"),
            "lon": pd.to_numeric(stations["longitude"], errors="coerce"),
            "borough": stations.get("borough", "DEMO"),
        }
    ).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    houses.insert(0, "house_id", houses.index.astype(int))
    return houses, incidents.reset_index(drop=True)


def synthetic_demo_data(
    *,
    n_incidents: int = 4000,
    center: tuple[float, float] = (-0.12, 51.50),
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fake incidents + stations around central London so --demo runs without data.

    Three demand hotspots plus a diffuse background, enough to show the
    optimizer moving sites off the stations and toward demand.
    """
    rng = np.random.default_rng(seed)
    lon0, lat0 = center
    hotspots = [
        (lon0 + 0.00, lat0 + 0.00, 0.020, 0.45),
        (lon0 - 0.06, lat0 - 0.04, 0.015, 0.30),
        (lon0 + 0.07, lat0 + 0.05, 0.018, 0.25),
    ]
    rows = []
    for hx, hy, sd, share in hotspots:
        n = int(n_incidents * share * 0.8)
        rows.append(pd.DataFrame({"dest_lon": rng.normal(hx, sd, n), "dest_lat": rng.normal(hy, sd * 0.8, n)}))
    n_bg = n_incidents - sum(len(r) for r in rows)
    rows.append(
        pd.DataFrame(
            {
                "dest_lon": rng.uniform(lon0 - 0.12, lon0 + 0.12, n_bg),
                "dest_lat": rng.uniform(lat0 - 0.09, lat0 + 0.09, n_bg),
            }
        )
    )
    incidents = pd.concat(rows, ignore_index=True)
    incidents["travel_seconds"] = rng.uniform(180, 600, len(incidents))
    incidents["borough"] = "DEMO"
    n_st = 10
    stations = pd.DataFrame(
        {
            "facname": [f"DEMO Station {i+1}" for i in range(n_st)],
            "latitude": rng.uniform(lat0 - 0.08, lat0 + 0.08, n_st),
            "longitude": rng.uniform(lon0 - 0.11, lon0 + 0.11, n_st),
            "borough": "DEMO",
        }
    )
    return incidents, stations
