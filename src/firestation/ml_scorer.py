"""Planner drive-time matrices from the road graph and the LightGBM travel-time model.

Two scorers for ``fire_station_location.build_travel_matrix``:

- ``road``:  Kolesar curve on the shortest legal road distance (directed London graph,
             one-way streets respected), with the curve the travel-time trainer fit on
             road km. Same baseline the LightGBM model is compared with (MAE 68.7 s).
- ``lgbm``:  the LightGBM travel-time model (``firestation.travel_time_model``) on the
             full route features from every candidate site to every demand cell.

A site has no station ID in the model, so a new site is scored the same way as an
existing one: first pump, from its home station, not busy. Time of day is averaged over
a few representative hours, weighted by when demand calls happen. The incident is the
most common LFB call type (false alarm in a dwelling), so differences between sites
come from the route, not the call mix.

Matrices take a few minutes (one Dijkstra per site for each route quantity, then one
batch of predictions), so they are cached in ``data/processed/planner_cache/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib"
Q90_MODEL_PATH = ROOT / "data" / "processed" / "models" / "travel_time_lfb_q90.joblib"  # 90th-percentile
# Learned street speeds + LightGBM correction (scripts/train_street_speeds.py).
STREETSPEED_MODEL_PATH = ROOT / "data" / "processed" / "models" / "travel_time_lfb_streetspeed.joblib"
CACHE_DIR = ROOT / "data" / "processed" / "planner_cache"

# (representative hour, hours it stands for); weights come from demand call hours.
TIME_SLOTS = [(3, range(0, 7)), (8, range(7, 10)), (13, range(10, 16)), (18, range(16, 20)), (21, range(20, 24))]
SCENARIO_DATE = "2024-10-16"  # a Wednesday in the last training year
SCENARIO_CALL = {"incident_group": "False Alarm", "property_category": "Dwelling"}

_ROUTER = None
_BUNDLES: dict[Path, dict] = {}


def _router():
    global _ROUTER
    if _ROUTER is None:
        from .road_distance import load_or_build_graph
        from .route_features import RouteFeatureRouter, edge_feature_table

        G = load_or_build_graph()
        _ROUTER = RouteFeatureRouter(G, edge_feature_table(G))
    return _ROUTER


def _bundle(path: Path = MODEL_PATH) -> dict:
    if path not in _BUNDLES:
        import joblib

        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/train_travel_time_lfb.py "
                                    "(or train_travel_time_quantile.py for the 90th-percentile model)")
        _BUNDLES[path] = joblib.load(path)
    return _BUNDLES[path]


def slot_weights(incidents: pd.DataFrame | None) -> list[tuple[int, float]]:
    """(hour, weight) per time slot, weighted by the demand calls' hour of call."""
    if incidents is None or "hour_of_call" not in incidents.columns:
        w = np.array([len(hours) for _, hours in TIME_SLOTS], dtype=float)
    else:
        h = pd.to_numeric(incidents["hour_of_call"], errors="coerce").dropna().astype(int)
        w = np.array([h.isin(list(hours)).sum() for _, hours in TIME_SLOTS], dtype=float)
    w = w / w.sum()
    return [(hour, float(wt)) for (hour, _), wt in zip(TIME_SLOTS, w)]


def cell_boroughs(cells: pd.DataFrame, incidents: pd.DataFrame | None) -> np.ndarray:
    """Most common borough of the demand calls nearest to each cell centroid."""
    if incidents is None or "borough" not in incidents.columns:
        return np.full(len(cells), "", dtype=object)
    from .fire_station_location import snap_to_cells

    inc = incidents.dropna(subset=["dest_lat", "dest_lon", "borough"])
    idx, _ = snap_to_cells(inc, cells)
    mode = pd.Series(inc["borough"].astype(str).str.upper().to_numpy()).groupby(idx).agg(
        lambda s: s.value_counts().index[0])
    return mode.reindex(np.arange(len(cells))).fillna("").to_numpy()


def _cache_key(scorer: str, candidates: pd.DataFrame, cells: pd.DataFrame, extra: Any,
               model_path: Path = MODEL_PATH) -> Path:
    h = hashlib.sha1()
    h.update(scorer.encode())
    for df in (candidates, cells):
        h.update(np.round(df[["lat", "lon"]].to_numpy(dtype=float), 6).tobytes())
    h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    if model_path.exists():
        h.update(str(model_path.stat().st_mtime_ns).encode())
    return CACHE_DIR / f"{scorer}_{len(candidates)}x{len(cells)}_{h.hexdigest()[:12]}.npy"


def _pairs(candidates: pd.DataFrame, cells: pd.DataFrame) -> tuple[np.ndarray, ...]:
    n_c, n_d = len(candidates), len(cells)
    s_lat = np.repeat(candidates["lat"].to_numpy(dtype=float), n_d)
    s_lon = np.repeat(candidates["lon"].to_numpy(dtype=float), n_d)
    d_lat = np.tile(cells["lat"].to_numpy(dtype=float), n_c)
    d_lon = np.tile(cells["lon"].to_numpy(dtype=float), n_c)
    return s_lat, s_lon, d_lat, d_lon


def road_matrix(candidates: pd.DataFrame, cells: pd.DataFrame, *, use_cache: bool = True) -> tuple[np.ndarray, dict]:
    """Kolesar (road-km fit from the travel-time trainer) on shortest legal road distance."""
    from .kolesar import KolesarModel

    params = _bundle()["kolesar_baseline_params"]
    meta = {"scorer": "road", "kolesar_road_km": params}
    path = _cache_key("road", candidates, cells, params)
    if use_cache and path.exists():
        return np.load(path), {**meta, "cache": str(path.relative_to(ROOT))}
    ends = _pairs(candidates, cells)
    km = _router().road_km(*ends).reshape(len(candidates), len(cells))
    from .facility import haversine_km

    crow = haversine_km(ends[1], ends[0], ends[3], ends[2]).reshape(km.shape)
    unreachable = ~np.isfinite(km)
    km = np.where(unreachable, crow * 1.4, km)  # rare: a site snapped off the main component
    T = np.clip(KolesarModel.from_dict(params).predict(km), 30.0, 1200.0)
    meta["n_unreachable_pairs"] = int(unreachable.sum())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, T)
    return T, {**meta, "cache": str(path.relative_to(ROOT))}


def _lgbm_trips(candidates: pd.DataFrame, cells: pd.DataFrame, boroughs: np.ndarray) -> pd.DataFrame:
    """Route every site × cell pair and build the model's input rows (one scenario call each)."""
    from .facility import haversine_km

    r = _router()
    ends = _pairs(candidates, cells)
    print(f"  routing {len(candidates)} sites × {len(cells)} cells on the London drive graph…", flush=True)
    trips = pd.concat([r.route_features(*ends), r.route_context(*ends)], axis=1)
    trips["road_km"] = r.road_km(*ends)
    trips["crow_km"] = haversine_km(ends[1], ends[0], ends[3], ends[2])
    return trips.assign(
        dest_lat=ends[2], dest_lon=ends[3], borough=np.tile(boroughs, len(candidates)),
        date_of_call=SCENARIO_DATE, busy_flag=0, pump_order=1, dest_exact=1,
        deployed_from_location="Home Station", **SCENARIO_CALL,
    )


def speeds_matrix(candidates: pd.DataFrame, cells: pd.DataFrame, *, use_cache: bool = True) -> tuple[np.ndarray, dict]:
    """Drive seconds along quickest routes under learned street speeds (``firestation.street_speeds``)."""
    from .street_speeds import MODEL_PATH as SPEEDS_PATH, StreetSpeedModel, speed_matrix

    model = StreetSpeedModel.load()
    meta = {"scorer": "speeds", "model": str(SPEEDS_PATH.relative_to(ROOT))}
    path = _cache_key("speeds", candidates, cells, list(np.round(model.coef, 8)), SPEEDS_PATH)
    if use_cache and path.exists():
        return np.load(path), {**meta, "cache": str(path.relative_to(ROOT))}
    T = np.clip(speed_matrix(model, _router(), candidates, cells), 30.0, 1200.0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, T)
    return T, {**meta, "cache": str(path.relative_to(ROOT))}


def lgbm_matrix(
    candidates: pd.DataFrame,
    cells: pd.DataFrame,
    incidents: pd.DataFrame | None = None,
    *,
    use_cache: bool = True,
    model_path: Path = MODEL_PATH,
    slots: list[tuple[int, float]] | None = None,
    trips: pd.DataFrame | None = None,
) -> tuple[np.ndarray, dict]:
    """LightGBM predicted drive seconds, candidate × cell, averaged over time slots.

    ``model_path=Q90_MODEL_PATH`` gives 90th-percentile times (the quantile model).
    ``slots``: (hour, weight) pairs to average over instead of the demand-weighted
    ``TIME_SLOTS`` (e.g. ``[(8, 1.0)]`` for the morning rush alone).
    """
    from .travel_time_model import predict

    slots = slot_weights(incidents) if slots is None else slots
    boroughs = cell_boroughs(cells, incidents)
    scenario = {"slots": slots, "date": SCENARIO_DATE, **SCENARIO_CALL,
                "busy_flag": 0, "pump_order": 1, "from_home": 1}
    name = {MODEL_PATH: "lgbm", STREETSPEED_MODEL_PATH: "lgbm_speeds"}.get(model_path,
                                                                         "lgbm_" + model_path.stem.split("_")[-1])
    meta = {"scorer": name, "model": str(model_path.relative_to(ROOT)), "scenario": scenario}
    path = _cache_key(name, candidates, cells, [scenario, list(boroughs)], model_path)
    if use_cache and path.exists():
        return np.load(path), {**meta, "cache": str(path.relative_to(ROOT))}

    bundle = _bundle(model_path)
    trips = _lgbm_trips(candidates, cells, boroughs) if trips is None else trips
    if bundle.get("base") == "street_speeds" and "street_base_s" not in trips.columns:
        trips = trips.assign(street_base_s=speeds_matrix(candidates, cells)[0].ravel())
    T = np.zeros(len(trips))
    step = 100_000  # chunks keep peak memory low on large matrices
    for hour, w in slots:
        print(f"  predicting hour {hour:02d} (weight {w:.2f})…", flush=True)
        for a in range(0, len(trips), step):
            T[a:a + step] += w * predict(bundle, trips.iloc[a:a + step].assign(hour_of_call=hour))
    T = T.reshape(len(candidates), len(cells))
    unreachable = ~np.isfinite(trips["road_km"].to_numpy()).reshape(T.shape)
    meta["n_unreachable_pairs"] = int(unreachable.sum())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, T)
    return T, {**meta, "cache": str(path.relative_to(ROOT))}


def lgbm_matrices_by_hour(candidates: pd.DataFrame, cells: pd.DataFrame, incidents: pd.DataFrame | None = None,
                          hours=tuple(h for h, _ in TIME_SLOTS), **kw) -> dict[int, np.ndarray]:
    """One LightGBM matrix per representative hour (default: the ``TIME_SLOTS`` hours), routing once.

    Also caches their demand-weighted average, which is exactly ``lgbm_matrix``'s all-day
    matrix, so a later ``lgbm_matrix`` call for the same sites and cells is free.
    """
    out, trips = {}, None
    for h in hours:
        key = _cache_key("lgbm", candidates, cells,
                         [{"slots": [(h, 1.0)], "date": SCENARIO_DATE, **SCENARIO_CALL, "busy_flag": 0,
                           "pump_order": 1, "from_home": 1}, list(cell_boroughs(cells, incidents))])
        if trips is None and not key.exists():
            trips = _lgbm_trips(candidates, cells, cell_boroughs(cells, incidents))
        out[h], _ = lgbm_matrix(candidates, cells, incidents, slots=[(h, 1.0)], trips=trips, **kw)
    slots = slot_weights(incidents)
    if not kw.get("model_path") and sorted(hours) == sorted(h for h, _ in slots):
        scenario = {"slots": slots, "date": SCENARIO_DATE, **SCENARIO_CALL,
                    "busy_flag": 0, "pump_order": 1, "from_home": 1}
        path = _cache_key("lgbm", candidates, cells, [scenario, list(cell_boroughs(cells, incidents))])
        if not path.exists():
            np.save(path, sum(w * out[h] for h, w in slots))
    return out
