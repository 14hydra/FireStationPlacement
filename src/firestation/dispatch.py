"""Dispatch model: which station sends the first engine?

Placement usually assumes the nearest station responds (or the next nearest when it is
busy). LFB's records show that in 2013 about 30 % of first engines did not come from the
nearest open station, and after the 2014 closures as many as 62 % in some areas; that
assumption, not the travel-time model, is what made closure effects hard to predict per
area (``scripts/closure_2014_diagnostic.py``).

For a call in demand cell j and a layout (open stations), the candidates are the ``K``
open stations with the fastest predicted attendance to j. A LightGBM classifier (decision
trees) scores each candidate from features that exist for any layout, real or planned:

  rank            1 = fastest predicted attendance
  attend_s        predicted attendance (drive + turnout)
  gap_s           seconds slower than the fastest candidate
  gap_ratio       attend_s / fastest
  p_busy          chance the candidate has no free pump (Erlang loss, its pumps and load)
  pumps           its pumps
  p_busy_first    chance the fastest candidate is busy
  p_busy_ahead    chance every faster candidate is busy (product)
  n_within_60s    open stations within 60 s of the fastest (dense cover = more choice)
  pumps_first     pumps at the fastest candidate
  slot            time of day (4-hour slots; hourly models only)

Time of day: engine availability swings between quiet nights and busy evenings, so the
hourly version scores each 4-hour slot with that slot's busy chances (each station's
load scaled by the slot's call rate, ``slot_factors``) and averages the slots, weighted by
when calls happen.

Scores are normalised within each call to probabilities, so the expected first-engine
attendance in cell j is ``sum_k P_k * attend_k``. Trained on calls grouped by cell and
layout (same features), with one weighted row per candidate and outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

K = 8
FEATURES = ["rank", "attend_s", "gap_s", "gap_ratio", "p_busy", "pumps", "p_busy_first", "p_busy_ahead",
            "n_within_60s", "pumps_first"]
SLOTS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]


def slot_of(hour) -> np.ndarray:
    return np.clip(np.asarray(hour, dtype=int) // 4, 0, len(SLOTS) - 1)


def slot_factors(hours) -> np.ndarray:
    """Call rate in each 4-hour slot relative to the all-day average (1 = average)."""
    share = np.bincount(slot_of(hours), minlength=len(SLOTS)) / max(len(hours), 1)
    return share / (1.0 / len(SLOTS))


def calibrate_slot_factors(busy_model, A, open_idx, call_cells, slots, responded_nearest,
                           iters: int = 30) -> np.ndarray:
    """Load factor per slot so the modelled busy chance of each call's nearest station
    matches how often, in that slot, the nearest station did NOT send the first engine.

    LFB's records show the nearest station is least available in the early afternoon
    (training, fire-safety visits), not at the evening call peak, so call rate alone is the
    wrong driver; this ties each slot to what was observed.
    """
    open_idx = np.asarray(list(open_idx), dtype=int)
    nearest = open_idx[np.argmin(A[open_idx], axis=0)]
    base = busy_model.load_weights
    out = np.ones(len(SLOTS))
    try:
        for h in range(len(SLOTS)):
            m = slots == h
            if not m.any():
                continue
            target = 1.0 - float(np.mean(responded_nearest[m]))
            lo, hi = 0.01, 20.0
            for _ in range(iters):
                mid = (lo * hi) ** 0.5
                busy_model.load_weights = base * mid
                p = busy_model.probs(A, open_idx)[nearest[call_cells[m]]].mean()
                lo, hi = (mid, hi) if p < target else (lo, mid)
            out[h] = (lo * hi) ** 0.5
    finally:
        busy_model.load_weights = base
    return out


def slot_busy(busy_model, A, open_idx, factors) -> list[np.ndarray]:
    """Busy chance per candidate row in each slot: the station's load scaled by the slot's call rate."""
    base = busy_model.load_weights
    out = []
    try:
        for f in factors:
            busy_model.load_weights = base * float(f)
            out.append(busy_model.probs(A, open_idx))
    finally:
        busy_model.load_weights = base
    return out


def candidates(A: np.ndarray, open_idx: Sequence[int], p_busy: np.ndarray, pumps: np.ndarray,
               k: int = K, slot: int | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Top-``k`` open stations per cell and their features.

    Returns (rows [k, n_cells]: candidate row per rank, feature frame with one row per
    (cell, rank) in cell-major order).
    """
    open_idx = np.asarray(list(open_idx), dtype=int)
    k = min(k, len(open_idx))
    sub = A[open_idx]
    order = np.argsort(sub, axis=0)[:k]  # [k, cells]
    rows = open_idx[order]
    n = A.shape[1]
    cols = np.arange(n)
    att = A[rows, cols[None, :]]
    pb = np.asarray(p_busy, float)[rows]
    ahead = np.vstack([np.ones((1, n)), np.cumprod(pb, axis=0)[:-1]])
    pm = np.asarray(pumps, float)[rows]
    within = ((att - att[0]) <= 60).sum(axis=0) - 1
    feats = pd.DataFrame({
        "rank": np.repeat(np.arange(1, k + 1)[:, None], n, axis=1).T.ravel(),
        "attend_s": att.T.ravel(),
        "gap_s": (att - att[0]).T.ravel(),
        "gap_ratio": (att / np.maximum(att[0], 1.0)).T.ravel(),
        "p_busy": pb.T.ravel(),
        "pumps": np.asarray(pumps, float)[rows].T.ravel(),
        "p_busy_first": np.repeat(pb[0][None, :], k, axis=0).T.ravel(),
        "p_busy_ahead": ahead.T.ravel(),
        "n_within_60s": np.repeat(within[None, :], k, axis=0).T.ravel(),
        "pumps_first": np.repeat(pm[0][None, :], k, axis=0).T.ravel(),
    })
    if slot is not None:
        feats["slot"] = int(slot)
    return rows, feats


def responder_counts(rows: np.ndarray, call_cells: np.ndarray, responder_rows: np.ndarray) -> np.ndarray:
    """[k, n_cells] count of calls per cell whose first engine came from the rank-k candidate."""
    k, n = rows.shape
    counts = np.zeros((k, n))
    for r in range(k):
        hit = rows[r, call_cells] == responder_rows
        np.add.at(counts[r], call_cells[hit], 1.0)
    return counts


@dataclass
class DispatchModel:
    k: int = K
    params: dict = field(default_factory=lambda: dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                                      min_child_samples=50, subsample=0.9, subsample_freq=1,
                                                      colsample_bytree=0.9, reg_lambda=1.0))
    model: Any = None
    hourly: bool = False

    def fit(self, parts: list[tuple[pd.DataFrame, np.ndarray, np.ndarray]]) -> "DispatchModel":
        """``parts``: (features from ``candidates``, counts [k, cells], calls per cell)."""
        import lightgbm as lgb

        X, y, w = [], [], []
        for feats, counts, n_calls in parts:
            pos = counts.T.ravel()
            neg = np.repeat(n_calls, counts.shape[0]) - pos
            keep = np.repeat(n_calls > 0, counts.shape[0])
            for label, wt in ((1, pos), (0, neg)):
                m = keep & (wt > 0)
                X.append(feats[m])
                y.append(np.full(m.sum(), label))
                w.append(wt[m])
        self.model = lgb.LGBMClassifier(objective="binary", verbose=-1, random_state=0, **self.params)
        cols = FEATURES + (["slot"] if self.hourly else [])
        self.model.fit(pd.concat(X, ignore_index=True)[cols], np.concatenate(y), sample_weight=np.concatenate(w))
        return self

    def proba(self, feats: pd.DataFrame, n_cells: int) -> np.ndarray:
        """[k, n_cells] probability that the rank-k candidate responds (sums to 1 per cell)."""
        cols = FEATURES + (["slot"] if self.hourly else [])
        s = self.model.predict_proba(feats[cols])[:, 1].reshape(n_cells, -1).T
        return s / np.maximum(s.sum(axis=0, keepdims=True), 1e-12)

    def expected(self, A: np.ndarray, open_idx: Sequence[int], p_busy, pumps: np.ndarray,
                 values: np.ndarray | None = None, slot_weights: np.ndarray | None = None) -> np.ndarray:
        """Expected first-engine attendance per cell (or of ``values``, e.g. drive only).

        Hourly model: ``p_busy`` is a list with one array per slot (``slot_busy``) and
        ``slot_weights`` [slots, cells] (or [slots]) the share of each cell's calls per slot.
        """
        V = A if values is None else values
        n = np.arange(A.shape[1])
        if not self.hourly:
            rows, feats = candidates(A, open_idx, p_busy, pumps, self.k)
            return (self.proba(feats, A.shape[1]) * V[rows, n[None, :]]).sum(axis=0)
        sw = np.ones(len(SLOTS)) / len(SLOTS) if slot_weights is None else np.asarray(slot_weights, float)
        sw = sw[:, None] if sw.ndim == 1 else sw
        out = np.zeros(A.shape[1])
        for h, pb in enumerate(p_busy):
            rows, feats = candidates(A, open_idx, pb, pumps, self.k, slot=h)
            out += sw[h] * (self.proba(feats, A.shape[1]) * V[rows, n[None, :]]).sum(axis=0)
        return out / np.maximum(sw.sum(axis=0), 1e-12)


DISPATCH_PATH = __import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "processed" / "models" / \
    "dispatch_lfb.joblib"


def load_dispatch(path=DISPATCH_PATH, by_hour: bool = False) -> DispatchModel:
    """The fitted dispatch model (``scripts/fit_dispatch_model.py``); all-day unless ``by_hour``."""
    import joblib

    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/fit_dispatch_model.py first")
    b = joblib.load(path)
    return b["model"] if by_hour else b["all_day_model"]


def dispatch_response(A: np.ndarray, idx: Sequence[int], busy_model, model: DispatchModel) -> np.ndarray:
    """Expected first-engine attendance per cell for open sites ``idx``.

    Busy chances and pumps come from ``busy_model`` (Erlang loss, pumps moving with their
    station when it has a pump plan); which station responds comes from ``model``.
    """
    idx = list(idx)
    p = busy_model.probs(A, idx)
    pumps = np.maximum(busy_model.pumps_for(idx), 1)
    return model.expected(A, idx, p, pumps)
