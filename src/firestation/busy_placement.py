"""Busy engines inside placement: an Erlang loss model plus a busy-aware exact solver.

When every pump at a call's nearest station is out, the next nearest station answers:

    expected first-engine time  =  (1 - p) * t1  +  p * t2

``p`` is the chance the nearest station has no free pump. It is not fixed: a station
that takes on more calls is busy more often, and opening a station next door takes
calls (and busyness) off it. We model it with the **Erlang loss formula** (M/G/c/c),
the standard result for a station with ``c`` pumps and offered load ``a`` (erlangs):

    p = B(a, c) = (a^c / c!) / sum_{i=0..c} a^i / i!

Calibration: today's measured busy rate per station (first pump from another station,
``busy_engines``) and its pump count give each station's load ``a0`` by inverting B.
Load scales with the calls a station serves, so each station's erlangs per call is
``a0 / calls0``; new sites get the median over today's stations. Any layout then gives
loads (calls each station is nearest to) and hence busy probabilities.

``solve_busy_exact`` optimises the blend with the integer program: with each area's
``p`` fixed, assigning every area a primary and a backup station is linear (Daskin's
expected-coverage idea). It solves, recomputes ``p`` from the new layout's loads, and
re-solves until the layout stops changing, keeping the best layout under the full model.

Pumps move with their station (``PumpPlan``): in a new layout, each closed station's
pumps go to an opened site (closed and opened sites paired to minimise the total
distance moved), kept stations keep theirs, and in expand mode a new station takes the
second pump of the nearest two-pump station, so London's pump total stays fixed.

Caveats: the measured busy rate also counts calls LFB sends from another station for
other reasons (closer appliance across a ground boundary), so it overstates true
unavailability; calls passed on to the backup station are not added to its load.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .facility import posture_seconds, solve_exact

P_CLIP = (0.01, 0.49)  # p <= 0.5 keeps "primary = nearest" in the linear model


def erlang_b(a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Erlang loss probability B(a, c), vectorised (recursion, stable for small c)."""
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=int)
    b = np.ones(np.broadcast(a, c).shape)
    a_b = np.broadcast_to(a, b.shape)
    c_b = np.broadcast_to(c, b.shape)
    for k in range(1, int(c_b.max(initial=1)) + 1):
        nxt = a_b * b / (k + a_b * b)
        b = np.where(k <= c_b, nxt, b)
    return b


def offered_load(p: np.ndarray, c: np.ndarray, iters: int = 60) -> np.ndarray:
    """Invert B(a, c) = p for a (bisection; B increases with a)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 0.95)
    c = np.asarray(c, dtype=int)
    lo, hi = np.zeros_like(p), np.full_like(p, 50.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        too_busy = erlang_b(mid, c) > p
        hi = np.where(too_busy, mid, hi)
        lo = np.where(too_busy, lo, mid)
    return (lo + hi) / 2


def _km(lat1, lon1, lat2, lon2) -> np.ndarray:
    kx = 111.32 * np.cos(np.radians(51.5))
    return np.hypot((np.asarray(lon1)[:, None] - np.asarray(lon2)[None, :]) * kx,
                    (np.asarray(lat1)[:, None] - np.asarray(lat2)[None, :]) * 110.574)


@dataclass
class PumpPlan:
    """Pumps per candidate site for any layout, moving pumps with their station.

    - kept current stations keep their pumps;
    - closed stations' pumps move to opened sites, pairing closed and opened sites to
      minimise total distance (Hungarian algorithm);
    - opened sites left over (more opened than closed, e.g. expand mode):
      ``extra="move"`` takes the second pump of the nearest two-pump station still
      open (London's total stays fixed); ``extra="new"`` adds ``new_site_pumps``.
    """

    base_pumps: np.ndarray  # per candidate row (current stations: today's pumps)
    existing: np.ndarray  # candidate rows of today's stations
    lat: np.ndarray
    lon: np.ndarray
    extra: str = "move"
    new_site_pumps: int = 1

    def assign(self, idx: Sequence[int]) -> tuple[np.ndarray, list[dict]]:
        from scipy.optimize import linear_sum_assignment

        idx = [int(i) for i in idx]
        ex = set(self.existing.tolist())
        pumps = np.zeros(len(self.base_pumps), dtype=int)
        kept = [i for i in idx if i in ex]
        opened = [i for i in idx if i not in ex]
        closed = [i for i in self.existing.tolist() if i not in set(idx)]
        pumps[kept] = self.base_pumps[kept]
        moves: list[dict] = []
        left = list(opened)
        if opened and closed:
            d = _km(self.lat[closed], self.lon[closed], self.lat[opened], self.lon[opened])
            ci, oi = linear_sum_assignment(d)
            for a, b in zip(ci, oi):
                pumps[opened[b]] = self.base_pumps[closed[a]]
                moves.append({"from": int(closed[a]), "to": int(opened[b]), "pumps": int(self.base_pumps[closed[a]]),
                              "km": round(float(d[a, b]), 2)})
            left = [opened[b] for b in range(len(opened)) if b not in set(oi)]
        for o in left:
            donors = [i for i in kept if pumps[i] >= 2]
            if self.extra == "move" and donors:
                d = _km(self.lat[[o]], self.lon[[o]], self.lat[donors], self.lon[donors])[0]
                give = donors[int(d.argmin())]
                pumps[give] -= 1
                pumps[o] = 1
                moves.append({"from": int(give), "to": int(o), "pumps": 1, "km": round(float(d.min()), 2),
                              "second_pump": True})
            else:
                pumps[o] = int(self.new_site_pumps)
                moves.append({"from": None, "to": int(o), "pumps": int(self.new_site_pumps), "km": None})
        return pumps, moves


@dataclass
class ErlangBusyModel:
    """Busy probability of every candidate site for any layout (see module docstring)."""

    erlangs_per_call: np.ndarray  # per candidate row
    pumps: np.ndarray  # per candidate row
    load_weights: np.ndarray  # calls per cell (all calls: every call keeps a pump busy)
    pump_plan: PumpPlan | None = None  # pumps per layout; None = fixed ``pumps``

    @classmethod
    def calibrate(
        cls,
        A: np.ndarray,
        existing_idx: Sequence[int],
        measured_p: np.ndarray,
        pumps: np.ndarray,
        load_weights: np.ndarray,
        pump_plan: PumpPlan | None = None,
    ) -> "ErlangBusyModel":
        idx = np.asarray(list(existing_idx), dtype=int)
        primary = idx[A[idx].argmin(axis=0)]
        calls0 = np.bincount(primary, weights=load_weights, minlength=A.shape[0])[idx]
        a0 = offered_load(np.asarray(measured_p, float)[idx], np.asarray(pumps)[idx])
        per_call = np.where(calls0 > 0, a0 / np.maximum(calls0, 1e-9), np.nan)
        out = np.full(A.shape[0], float(np.nanmedian(per_call)))
        out[idx] = np.where(np.isfinite(per_call), per_call, out[idx])
        return cls(out, np.asarray(pumps, dtype=int), np.asarray(load_weights, dtype=float), pump_plan)

    def pumps_for(self, idx: Sequence[int]) -> np.ndarray:
        return self.pump_plan.assign(idx)[0] if self.pump_plan is not None else self.pumps

    def probs(self, A: np.ndarray, idx: Sequence[int]) -> np.ndarray:
        """Busy probability per candidate row for open sites ``idx`` (closed rows: 0)."""
        idx = np.asarray(list(idx), dtype=int)
        primary = idx[A[idx].argmin(axis=0)]
        calls = np.bincount(primary, weights=self.load_weights, minlength=A.shape[0])
        p = np.zeros(A.shape[0])
        pumps = self.pumps_for(idx)
        p[idx] = erlang_b(self.erlangs_per_call[idx] * calls[idx], np.maximum(pumps[idx], 1))
        return np.clip(p, 0.0, 0.95)

    def response(self, A: np.ndarray, idx: Sequence[int]) -> np.ndarray:
        return posture_seconds(A, idx, self.probs(A, idx))


def solve_busy_exact(
    A: np.ndarray,
    weights: np.ndarray,
    k: int,
    model: ErlangBusyModel,
    *,
    start_idx: Sequence[int],
    max_rounds: int = 8,
    backup_k: int = 30,
    **solve_kw: Any,
) -> dict[str, Any]:
    """Busy-aware placement: integer program with primary + backup stations, iterated on p.

    ``start_idx`` (today's stations) gives the first round's busy probabilities. Each
    round fixes every area's ``p`` (busy probability of its current nearest station),
    solves exactly, then recomputes loads and ``p`` for the new layout. Returns the best
    layout under the full Erlang model.
    """
    t0 = time.perf_counter()
    w = np.asarray(weights, dtype=float)
    W = max(float(w.sum()), 1e-9)

    def true_value(idx):
        return float((w * model.response(A, idx)).sum() / W)

    idx = list(start_idx)
    best = {"indices": list(idx), "value": true_value(idx), "round": 0}
    seen, rounds = set(), []
    for r in range(1, max_rounds + 1):
        p_site = model.probs(A, idx)
        cur = np.asarray(idx)
        p_cell = np.clip(p_site[cur[A[cur].argmin(axis=0)]], *P_CLIP)
        res = solve_exact(A, w, k, backup_p=p_cell, backup_k=backup_k, **solve_kw)
        val = true_value(res.indices)
        rounds.append({"round": r, "value_full_model": round(val, 3), "value_fixed_p": round(res.value, 3),
                       "optimal_fixed_p": res.optimal, "gap": res.gap, "solve_s": res.solve_s,
                       "backup_outside_k": res.cap.get("backup_outside_k")})
        if val < best["value"] - 1e-9:
            best = {"indices": list(res.indices), "value": val, "round": r}
        key = tuple(sorted(res.indices))
        if key in seen:
            break
        seen.add(key)
        idx = res.indices
    best["rounds"] = rounds
    best["solve_s"] = round(time.perf_counter() - t0, 2)
    best["response_s"] = model.response(A, best["indices"])
    return best


def extend_busy_model(model: ErlangBusyModel, lat: np.ndarray, lon: np.ndarray) -> ErlangBusyModel:
    """The same model with extra candidate rows appended (new sites: median load per call)."""
    n = len(lat)
    pp = model.pump_plan
    new_pp = None
    if pp is not None:
        new_pp = PumpPlan(np.r_[pp.base_pumps, np.full(n, pp.new_site_pumps, dtype=int)], pp.existing,
                          np.r_[pp.lat, lat], np.r_[pp.lon, lon], extra=pp.extra, new_site_pumps=pp.new_site_pumps)
    med = float(np.median(model.erlangs_per_call[pp.existing])) if pp is not None else float(
        np.median(model.erlangs_per_call))
    return ErlangBusyModel(np.r_[model.erlangs_per_call, np.full(n, med)],
                           np.r_[model.pumps, np.full(n, 1, dtype=int)], model.load_weights, new_pp)
