"""Per-station fire-engine (pump) capacity for London LFB.

LFB has 102 land stations and ~142 pumping appliances: about **40 stations
carry two pumps**, the rest one (London Assembly / LFB fleet notes). Named
two-engine stations appear in LFB's org-chart PDF; the rest of the forty are
recovered from mobilisation records as stations that often send **two home
engines to the same incident**.

Capacity enters placement as ``n_pumps`` on each candidate. More pumps lower
the chance the local station is fully busy, so first-engine attendance can
fall back less often to a neighbour (see ``capacity_busy_prob``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "london"

# LFB "Fire Station with 2 Fire Engines" org chart (stations without extra
# specialist appliances). Incomplete vs the full set of ~40 two-pump houses.
LFB_NAMED_TWO_PUMP = frozenset(
    {
        "Addington", "Brixton", "Bromley", "Enfield", "Feltham", "Hendon",
        "Hillingdon", "Homerton", "Hornsey", "Ilford", "Kentish Town", "Kingston",
        "Lambeth", "Norbury", "North Kensington", "Northolt", "Orpington",
        "Plumstead", "Sidcup", "Southall", "Stoke Newington", "Tooting",
        "Walthamstow", "West Hampstead", "West Norwood",
    }
)

DEFAULT_TOTAL_PUMPS = 142  # 102 stations + ~40 second pumps
MOBILISATION_FILES = ("lfb_mobilisations_2021_2024.csv", "lfb_mobilisations_2025.csv")


def _norm(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def infer_two_pump_stations(
    *,
    mobilisations: pd.DataFrame | None = None,
    min_multi_dispatch: int = 200,
    target_n: int = 40,
) -> pd.DataFrame:
    """Stations that often send ≥2 home engines to one incident → likely 2-pump.

    Returns columns: station, n_multi_dispatch, n_home, rate, n_pumps_inferred.
    """
    if mobilisations is None:
        parts = []
        cols = ["IncidentNumber", "DeployedFromStation_Name", "DeployedFromLocation"]
        for f in MOBILISATION_FILES:
            path = RAW / f
            if path.exists():
                parts.append(
                    pd.read_csv(path, usecols=cols, encoding="utf-8-sig",
                                low_memory=False, dtype={"IncidentNumber": str})
                )
        if not parts:
            # Fall back to the named list only
            return pd.DataFrame(
                {"station": sorted(LFB_NAMED_TWO_PUMP), "n_multi_dispatch": np.nan,
                 "n_home": np.nan, "rate": np.nan, "n_pumps_inferred": 2}
            )
        mobilisations = pd.concat(parts, ignore_index=True)

    home = mobilisations[mobilisations["DeployedFromLocation"] == "Home Station"].copy()
    home["DeployedFromStation_Name"] = home["DeployedFromStation_Name"].astype(str).str.strip()
    cnt = (
        home.groupby(["IncidentNumber", "DeployedFromStation_Name"], as_index=False)
        .size()
        .rename(columns={"size": "n_from_station"})
    )
    multi = cnt[cnt["n_from_station"] >= 2]
    by = multi.groupby("DeployedFromStation_Name").size().rename("n_multi_dispatch")
    total = home.groupby("DeployedFromStation_Name").size().rename("n_home")
    out = pd.concat([by, total], axis=1).fillna(0).reset_index()
    out = out.rename(columns={"DeployedFromStation_Name": "station"})
    out["rate"] = out["n_multi_dispatch"] / out["n_home"].clip(lower=1)
    out = out.sort_values(["n_multi_dispatch", "rate"], ascending=False).reset_index(drop=True)
    # Prefer the empirical top ``target_n`` (matches LFB's ~40 two-pump houses).
    two = set(out.head(int(target_n))["station"].astype(str))
    # Always include the named org-chart stations even if slightly below the cut.
    two |= set(LFB_NAMED_TWO_PUMP)
    # If that exceeds target_n slightly, keep them; honesty note covers it.
    out["n_pumps_inferred"] = out["station"].isin(two).map({True: 2, False: 1}).astype(int)
    # Also keep a threshold-based flag for diagnostics
    out["above_min_multi"] = out["n_multi_dispatch"] >= int(min_multi_dispatch)
    return out


def default_pump_table(houses: pd.DataFrame) -> pd.DataFrame:
    """One row per house with ``n_pumps`` (1 or 2) from LFB inference."""
    names = houses["facilityname"].astype(str) if "facilityname" in houses.columns else houses["name"].astype(str)
    inferred = infer_two_pump_stations()
    lookup = {_norm(s): int(n) for s, n in zip(inferred["station"], inferred["n_pumps_inferred"])}
    # Named list as floor
    for s in LFB_NAMED_TWO_PUMP:
        lookup[_norm(s)] = max(lookup.get(_norm(s), 1), 2)
    out = houses.copy()
    out["station"] = names.astype(str)
    out["n_pumps"] = [lookup.get(_norm(s), 1) for s in out["station"]]
    out["n_pumps"] = out["n_pumps"].astype(int).clip(lower=1)
    return out


def load_pump_overrides(path: Path | None) -> dict[str, int]:
    """Optional CSV with columns station/name + n_pumps (or pumps)."""
    if path is None or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    name_col = next((c for c in ("station", "name", "facilityname", "site_name") if c in df.columns), None)
    pump_col = next((c for c in ("n_pumps", "pumps", "n_engines", "capacity") if c in df.columns), None)
    if name_col is None or pump_col is None:
        raise ValueError(f"{path} needs station + n_pumps columns")
    return {
        _norm(n): max(1, int(p))
        for n, p in zip(df[name_col].astype(str), pd.to_numeric(df[pump_col], errors="coerce").fillna(1))
    }


def resolve_pump_vector(
    candidates: pd.DataFrame,
    *,
    pumps: Mapping[str, int] | None = None,
    default_pumps: int = 1,
    new_site_pumps: int = 1,
) -> np.ndarray:
    """``n_pumps`` per candidate row. Existing sites use ``pumps`` map; new sites get ``new_site_pumps``."""
    pumps = {_norm(k): max(1, int(v)) for k, v in (pumps or {}).items()}
    out = np.full(len(candidates), max(1, int(default_pumps)), dtype=int)
    names = candidates["site_name"].astype(str).to_numpy()
    existing = candidates["is_existing"].to_numpy(dtype=bool) if "is_existing" in candidates.columns else np.ones(len(candidates), bool)
    for i, (name, is_ex) in enumerate(zip(names, existing)):
        if is_ex:
            out[i] = pumps.get(_norm(name), max(1, int(default_pumps)))
        else:
            out[i] = max(1, int(new_site_pumps))
    return out


def capacity_busy_prob(
    base_busy: np.ndarray,
    n_pumps: np.ndarray,
    *,
    baseline_pumps: np.ndarray | None = None,
) -> np.ndarray:
    """Scale empirical busy rates when station size changes.

    With ``n`` independent engines each busy with probability ``p0`` under the
    baseline size ``c0``, the chance *all* engines are busy is roughly
    ``p0 ** (n / c0)``. That shrinks as capacity grows and grows if capacity
    is cut. Clipped to ``[0.02, 0.95]``.
    """
    p0 = np.clip(np.asarray(base_busy, dtype=float), 1e-6, 0.999)
    n = np.asarray(n_pumps, dtype=float).clip(min=1.0)
    c0 = np.asarray(baseline_pumps if baseline_pumps is not None else np.ones_like(n), dtype=float)
    c0 = np.clip(c0, 1.0, None)
    # all-busy ≈ p0^(n/c0)
    p = np.power(p0, n / c0)
    return np.clip(p, 0.02, 0.95)


def station_incident_load(
    incidents: pd.DataFrame,
    houses: pd.DataFrame,
    *,
    ground_col: str = "station_ground",
) -> pd.DataFrame:
    """Incident counts / density proxies per station ground."""
    names = houses["facilityname"].astype(str) if "facilityname" in houses.columns else houses["name"].astype(str)
    base = pd.DataFrame({"station": names.astype(str), "lat": houses["lat"], "lon": houses["lon"]})
    if ground_col not in incidents.columns:
        base["n_incidents"] = 0
        base["busy_rate"] = np.nan
        return base

    g = incidents.copy()
    g[ground_col] = g[ground_col].astype(str).str.strip()
    counts = g.groupby(ground_col).size().rename("n_incidents")
    out = base.copy()
    out["n_incidents"] = out["station"].map(counts).fillna(0).astype(int)

    if {"deployed_from_station", ground_col}.issubset(incidents.columns):
        busy = (
            g[ground_col].astype(str).str.strip()
            != g["deployed_from_station"].astype(str).str.strip()
        )
        br = g.assign(_busy=busy).groupby(ground_col)["_busy"].mean().rename("busy_rate")
        out["busy_rate"] = out["station"].map(br)
    else:
        out["busy_rate"] = np.nan

    # Rough catchment area: mean distance to other stations / 2 (Voronoi proxy)
    lat = out["lat"].to_numpy(float)
    lon = out["lon"].to_numpy(float)
    # haversine to nearest other station
    from .facility import haversine_km

    nn_km = np.full(len(out), np.nan)
    for i in range(len(out)):
        d = haversine_km(lon[i], lat[i], lon, lat)
        d[i] = np.inf
        nn_km[i] = float(np.min(d))
    out["nearest_station_km"] = nn_km
    # density proxy: incidents per km of nearest-neighbour spacing (higher = busier for size)
    out["incidents_per_nn_km"] = out["n_incidents"] / np.clip(out["nearest_station_km"], 0.2, None)
    return out


def plan_resize_capacity(
    A: np.ndarray,
    weights: np.ndarray,
    existing_idx: Sequence[int],
    *,
    base_busy: np.ndarray,
    baseline_pumps: np.ndarray,
    total_pumps: int,
    min_pumps: int = 1,
    max_pumps: int = 3,
    threshold_s: float = 600.0,
    max_moves: int = 80,
) -> dict[str, Any]:
    """Reallocate a fixed pump budget across fixed station sites.

    Starts from ``baseline_pumps`` on ``existing_idx`` (clipped to the budget),
    then greedily moves one pump at a time from a lighter / over-sized station
    to a heavier / under-sized one under a capacity-aware busy blend.
    Locations do not change — only ``n_pumps``.
    """
    from .busy_engines import expected_first_arrival
    from .facility import posture_metrics
    from .lfb_standards import score_first_engine

    idx = np.asarray([int(i) for i in existing_idx], dtype=int)
    n = len(idx)
    if n == 0:
        raise ValueError("resize needs existing stations")
    min_p, max_p = int(min_pumps), int(max_pumps)
    budget = int(total_pumps)
    if budget < n * min_p:
        raise ValueError(f"total_pumps={budget} < n_stations*min_pumps={n * min_p}")
    if budget > n * max_p:
        budget = n * max_p

    pumps = np.array([max(min_p, min(max_p, int(baseline_pumps[i]))) for i in idx], dtype=int)
    while pumps.sum() > budget:
        j = int(np.argmax(pumps))
        if pumps[j] <= min_p:
            break
        pumps[j] -= 1
    while pumps.sum() < budget:
        cands = [k for k in range(n) if pumps[k] < max_p]
        if not cands:
            break
        j = cands[int(np.argmin([pumps[k] for k in cands]))]
        pumps[j] += 1

    base_b = np.asarray(base_busy, dtype=float)
    base_c = np.asarray(baseline_pumps, dtype=float)
    # Restrict travel matrix to existing stations once
    A_ex = np.asarray(A[idx, :], dtype=float)
    w = np.asarray(weights, dtype=float)
    w_sum = max(float(w.sum()), 1e-9)
    # Nearest / second-nearest station index (into idx) per cell — fixed locations
    order = np.argsort(A_ex, axis=0)
    i1, i2 = order[0], order[1] if n >= 2 else order[0]
    t1 = A_ex[i1, np.arange(A_ex.shape[1])]
    t2 = A_ex[i2, np.arange(A_ex.shape[1])] if n >= 2 else t1

    def score(pvec: np.ndarray) -> tuple[float, np.ndarray]:
        # all-busy prob at each existing station under this pump vector
        p_all = np.clip(np.power(np.clip(base_b[idx], 1e-6, 0.999), pvec / np.clip(base_c[idx], 1.0, None)), 0.02, 0.95)
        if n < 2:
            resp = t1.copy()
        else:
            resp = expected_first_arrival(t1, p_all[i1], t2)
        val = float((w * resp).sum() / w_sum)
        return val, resp

    # Demand each station is nearest for — used to prioritise move candidates
    nearest_load = np.bincount(i1, weights=w, minlength=n)

    current, resp = score(pumps)
    moves = 0
    for _ in range(int(max_moves)):
        donors = [i for i in range(n) if pumps[i] > min_p]
        receivers = [j for j in range(n) if pumps[j] < max_p]
        if not donors or not receivers:
            break
        # Rank: prefer donating from low-load oversized, receiving at high-load undersized
        donor_score = nearest_load / pumps.clip(min=1)
        recv_score = nearest_load / pumps.clip(min=1)
        donors = sorted(donors, key=lambda i: donor_score[i])[: min(20, len(donors))]
        receivers = sorted(receivers, key=lambda j: -recv_score[j])[: min(20, len(receivers))]
        best_gain = 0.0
        best_move = None
        for d in donors:
            for r in receivers:
                if d == r:
                    continue
                trial = pumps.copy()
                trial[d] -= 1
                trial[r] += 1
                val, _ = score(trial)
                gain = current - val
                if gain > best_gain + 1e-6:
                    best_gain = gain
                    best_move = (d, r, trial, val)
        if best_move is None:
            break
        _, _, pumps, current = best_move
        moves += 1
        _, resp = score(pumps)

    n_full = np.asarray(baseline_pumps, dtype=float).copy()
    n_full[idx] = pumps
    busy_final = capacity_busy_prob(base_b, n_full, baseline_pumps=base_c)
    metrics = posture_metrics(resp, weights, threshold_s=threshold_s, label="plan_resize", n_posts=n)
    metrics["lfb_first_engine"] = score_first_engine(resp, weights)
    metrics["objective_value"] = float(current)

    return {
        "indices": idx.tolist(),
        "n_pumps": pumps,
        "n_pumps_full": n_full.astype(int),
        "busy_p": busy_final,
        "response_s": resp,
        "value": float(current),
        "n_moves": moves,
        "total_pumps": int(pumps.sum()),
        "metrics": metrics,
    }
