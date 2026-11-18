"""Statistical travel-time model: sample from street-conditioned PDFs.

Advisor-style alternative to point predictors (LightGBM / Kolesar):

  1. From historical trips, estimate an empirical distribution of fire-engine
     *pace* (s/km) for each OpenStreetMap road class along the route.
  2. For a new trip, sample a pace from each class PDF and recombine with the
     trip's length shares:  T = road_km * sum_c (share_c * pace_c).
  3. Optionally also keep a coarser *regime* PDF (distance bin × dominant class)
     for trips with thin class coverage.

We only observe whole-trip times, not GPS per street. Pace samples per class
are built by attributing each training trip's drive time across classes in
proportion to length share (standard compositional attribution). That is an
honest stand-in for "travel time PDF per street type" when edge-level times
are unavailable.

Point prediction = median of Monte Carlo samples (or closed-form mixture median
approximation via many samples).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

CLASS_SHARE_COLS = [
    ("major", "share_major"),
    ("primary", "share_primary"),
    ("secondary", "share_secondary"),
    ("tertiary", "share_tertiary"),
    ("residential", "share_residential"),
    ("minor", "share_minor"),
]
PACE_LO, PACE_HI = 15.0, 400.0  # s/km ≈ 240 km/h … 9 km/h
DRIVE_LO, DRIVE_HI = 30.0, 1200.0
MIN_CLASS_SAMPLES = 40
N_DIST_BINS = 8


def _rush(hour: np.ndarray) -> np.ndarray:
    h = np.asarray(hour, dtype=float)
    return ((h >= 7) & (h <= 9)) | ((h >= 16) & (h <= 19))


@dataclass
class StreetTimePDFModel:
    """Empirical pace PDFs by road class (+ regime fallback)."""

    class_paces: dict[str, np.ndarray] = field(default_factory=dict)
    regime_times: dict[tuple[int, str, int], np.ndarray] = field(default_factory=dict)
    dist_edges: np.ndarray | None = None
    global_times: np.ndarray | None = None
    global_paces: np.ndarray | None = None
    n_train: int = 0

    def fit(self, df: pd.DataFrame) -> "StreetTimePDFModel":
        y = pd.to_numeric(df["drive_s"], errors="coerce").to_numpy(float)
        road = pd.to_numeric(df["road_km"], errors="coerce").to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(road) & (road > 0.05) & (y >= DRIVE_LO) & (y <= DRIVE_HI)
        df = df.loc[ok].reset_index(drop=True)
        y = y[ok]
        road = road[ok]
        self.n_train = int(len(df))
        pace = np.clip(y / road, PACE_LO, PACE_HI)

        # --- per road-class pace PDFs (length-share attribution) ---
        shares = {}
        for name, col in CLASS_SHARE_COLS:
            if col in df.columns:
                shares[name] = np.clip(pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(float), 0, 1)
            else:
                shares[name] = np.zeros(len(df))
        share_mat = np.column_stack([shares[n] for n, _ in CLASS_SHARE_COLS])
        share_sum = share_mat.sum(axis=1, keepdims=True)
        share_mat = np.where(share_sum > 1e-6, share_mat / np.maximum(share_sum, 1e-6), 1.0 / share_mat.shape[1])

        class_paces: dict[str, list[float]] = {n: [] for n, _ in CLASS_SHARE_COLS}
        # Weight each trip's pace into every class it uses; keep multiple copies
        # proportional to share so common classes dominate naturally.
        rng = np.random.default_rng(0)
        for i in range(len(df)):
            for j, (name, _) in enumerate(CLASS_SHARE_COLS):
                w = float(share_mat[i, j])
                if w < 0.05:
                    continue
                # replicate ~ proportional to share (cap 5) for empirical mass
                n_rep = max(1, int(round(w * 5)))
                class_paces[name].extend([float(pace[i])] * n_rep)

        self.class_paces = {
            k: np.asarray(v, dtype=float) if len(v) >= MIN_CLASS_SAMPLES else np.asarray([], dtype=float)
            for k, v in class_paces.items()
        }
        # fill empty classes from global pace
        self.global_paces = pace
        for k, v in list(self.class_paces.items()):
            if len(v) < MIN_CLASS_SAMPLES:
                self.class_paces[k] = pace

        # --- regime PDF: distance bin × dominant class × rush ---
        self.dist_edges = np.quantile(road, np.linspace(0, 1, N_DIST_BINS + 1))
        self.dist_edges[0] = 0.0
        self.dist_edges[-1] = max(self.dist_edges[-1], float(road.max()) + 1e-6)
        dist_bin = np.clip(np.digitize(road, self.dist_edges[1:-1], right=True), 0, N_DIST_BINS - 1)
        dom_idx = share_mat.argmax(axis=1)
        dom_name = np.array([CLASS_SHARE_COLS[j][0] for j in dom_idx])
        hour = pd.to_numeric(df.get("hour_of_call", 12), errors="coerce").fillna(12).to_numpy(float)
        rush = _rush(hour).astype(int)

        buckets: dict[tuple[int, str, int], list[float]] = {}
        for i in range(len(df)):
            key = (int(dist_bin[i]), str(dom_name[i]), int(rush[i]))
            buckets.setdefault(key, []).append(float(y[i]))
        self.regime_times = {
            k: np.asarray(v, dtype=float) for k, v in buckets.items() if len(v) >= MIN_CLASS_SAMPLES
        }
        self.global_times = y
        return self

    def _dist_bin(self, road_km: np.ndarray) -> np.ndarray:
        assert self.dist_edges is not None
        return np.clip(np.digitize(road_km, self.dist_edges[1:-1], right=True), 0, N_DIST_BINS - 1)

    def _share_matrix(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for name, col in CLASS_SHARE_COLS:
            if col in df.columns:
                cols.append(np.clip(pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(float), 0, 1))
            else:
                cols.append(np.zeros(len(df)))
        m = np.column_stack(cols)
        s = m.sum(axis=1, keepdims=True)
        return np.where(s > 1e-6, m / np.maximum(s, 1e-6), 1.0 / m.shape[1])

    def sample_class_pace(self, class_name: str, n: int, rng: np.random.Generator) -> np.ndarray:
        pool = self.class_paces.get(class_name)
        if pool is None or len(pool) == 0:
            pool = self.global_paces
        return rng.choice(pool, size=n, replace=True)

    def sample(
        self,
        df: pd.DataFrame,
        *,
        n_samples: int = 40,
        seed: int = 0,
        method: str = "street_class",
    ) -> np.ndarray:
        """Return array [n_trips, n_samples] of simulated drive times (seconds)."""
        rng = np.random.default_rng(seed)
        road = np.maximum(pd.to_numeric(df["road_km"], errors="coerce").fillna(2.0).to_numpy(float), 0.05)
        n = len(df)
        out = np.zeros((n, n_samples), dtype=float)

        if method == "regime":
            shares = self._share_matrix(df)
            dom = shares.argmax(axis=1)
            dist_bin = self._dist_bin(road)
            hour = pd.to_numeric(df.get("hour_of_call", 12), errors="coerce").fillna(12).to_numpy(float)
            rush = _rush(hour).astype(int)
            for i in range(n):
                key = (int(dist_bin[i]), CLASS_SHARE_COLS[int(dom[i])][0], int(rush[i]))
                pool = self.regime_times.get(key)
                if pool is None or len(pool) < 10:
                    # fall back: ignore rush, then global
                    for r in (0, 1):
                        pool = self.regime_times.get((int(dist_bin[i]), CLASS_SHARE_COLS[int(dom[i])][0], r))
                        if pool is not None and len(pool) >= 10:
                            break
                if pool is None or len(pool) < 10:
                    pool = self.global_times
                out[i] = rng.choice(pool, size=n_samples, replace=True)
            return np.clip(out, DRIVE_LO, DRIVE_HI)

        # street_class: T = road_km * sum_c share_c * pace_c
        shares = self._share_matrix(df)
        for i in range(n):
            paces = np.zeros(n_samples, dtype=float)
            for j, (name, _) in enumerate(CLASS_SHARE_COLS):
                w = float(shares[i, j])
                if w <= 0:
                    continue
                paces += w * self.sample_class_pace(name, n_samples, rng)
            out[i] = road[i] * paces
        return np.clip(out, DRIVE_LO, DRIVE_HI)

    def predict(self, df: pd.DataFrame, *, n_samples: int = 40, seed: int = 0, method: str = "street_class") -> np.ndarray:
        """Point prediction = median of PDF samples."""
        samples = self.sample(df, n_samples=n_samples, seed=seed, method=method)
        return np.median(samples, axis=1)

    def predictive_quantiles(
        self,
        df: pd.DataFrame,
        qs: tuple[float, ...] = (0.1, 0.5, 0.9),
        *,
        n_samples: int = 80,
        seed: int = 0,
        method: str = "street_class",
    ) -> dict[float, np.ndarray]:
        samples = self.sample(df, n_samples=n_samples, seed=seed, method=method)
        return {q: np.quantile(samples, q, axis=1) for q in qs}

    def to_summary(self) -> dict[str, Any]:
        return {
            "n_train": self.n_train,
            "class_pace_n": {k: int(len(v)) for k, v in self.class_paces.items()},
            "class_pace_median_s_per_km": {
                k: float(np.median(v)) if len(v) else None for k, v in self.class_paces.items()
            },
            "n_regimes": len(self.regime_times),
            "dist_edges_km": self.dist_edges.tolist() if self.dist_edges is not None else None,
        }
