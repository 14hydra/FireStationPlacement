"""Learned fire engine speed on every street segment.

Each directed street segment e gets a driving time

    t_e = length_e * pace_e + (seconds per point feature on e)
    pace_e = pace[road class] + d_20mph [limit <= 20 mph] + d_bus [bus lane]
             + d_oneway [one-way] + d_multilane [2+ lanes] + d_bridge [bridge / tunnel]

plus a fixed time per junction passed, and each traffic signal, traffic calming
feature (speed hump, chicane, ...), pedestrian crossing, give-way / stop sign and
roundabout on the segment. A trip's time is a fixed start-up time plus the sum of t_e
along its route, plus a pace for the straight-line hop from each end to the street graph.

For fixed routes the trip time is linear in the parameters (the sum along a route of each
segment's length per class, count of signals, ...), so the parameters are fitted by a
robust (Huber) linear regression with every class pace kept positive. Routes depend on
the times, so training alternates (Zhan, Hasan and Ukkusuri, 2013, did the same from taxi
trips): route every trip on the current times (one Dijkstra per starting station, sums
along the shortest-time tree by pointer jumping), refit, and repeat.

The coefficients read directly as seconds per km (paces and adjustments) or seconds per
feature, so they say which street characteristics slow fire engines, with bootstrap
intervals from resampling trips.

Turns are not in the segment times (they belong to a pair of segments, which would need a
turn-expanded graph); the correction model (``scripts/train_street_speeds.py``) adds
them, with time of day, season and busy engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .route_features import CLASSES

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "processed" / "models" / "street_speeds.joblib"
MIN_PACE = 1 / 40.0  # s per metre: no segment faster than 144 km/h

PACE_COLS = [f"pace_{c}" for c in CLASSES]
ADJ_COLS = ["d_20mph", "d_bus_lane", "d_oneway", "d_multilane", "d_bridge_tunnel"]
POINT_COLS = ["per_junction", "per_signal", "per_calming", "per_crossing", "per_give_way", "per_roundabout"]
EDGE_COLS = PACE_COLS + ADJ_COLS + POINT_COLS
TRIP_COLS = ["start_up", "pace_off_graph"] + EDGE_COLS  # start_up = intercept; off-graph metres


def edge_design(router) -> np.ndarray:
    """[n_edges, len(EDGE_COLS)]: each segment's contribution per unit of each parameter."""
    v = router.edge_vals
    L = v["len_total"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        lanes = np.where(v["len_lanes_tagged"] > 0, v["lanes_x_len"] / np.maximum(L, 1e-9), np.nan)
    cols = [v[f"len_{c}"].to_numpy() for c in CLASSES]
    cols += [v["len_20mph"].to_numpy(), v["len_bus_lane"].to_numpy(), v["len_oneway"].to_numpy(),
             L * (np.nan_to_num(lanes) >= 2), v["len_bridge_tunnel"].to_numpy()]
    cols += [v["n_junctions"].to_numpy(), v["n_signals"].to_numpy(), v["n_calming"].to_numpy(),
             v["n_crossing"].to_numpy(), v["n_give_way"].to_numpy(), v["n_roundabout"].to_numpy()]
    return np.column_stack(cols).astype(float)


@dataclass
class StreetSpeedModel:
    """Fitted parameters (``coef`` in TRIP_COLS order; seconds per metre or per feature)."""

    coef: np.ndarray | None = None
    history: list = field(default_factory=list)

    # ------------------------------------------------------------------ segment times
    def edge_times(self, router, Z: np.ndarray | None = None) -> np.ndarray:
        Z = edge_design(router) if Z is None else Z
        L = router.edge_vals["len_total"].to_numpy()
        t = Z @ self.coef[2:]
        return np.maximum(t, L * MIN_PACE) + 1e-3  # strictly positive for Dijkstra

    def edge_speed_kph(self, router) -> np.ndarray:
        """Driving speed on each segment (length over its learned time, point features included)."""
        L = router.edge_vals["len_total"].to_numpy()
        return 3.6 * L / self.edge_times(router)

    # ------------------------------------------------------------------ routing
    @staticmethod
    def route_sums(router, times: np.ndarray, Z: np.ndarray, s_idx, d_idx) -> tuple[np.ndarray, np.ndarray]:
        """Quickest route under ``times`` for each trip (start node, end node).

        Returns (route time in seconds, sum of Z along the route) per trip.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        e = router.edge_id.tocoo()
        n = router.csr.shape[0]
        csr = csr_matrix((times[e.data - 1], (e.row, e.col)), shape=(n, n))
        out_t = np.full(len(s_idx), np.nan)
        out_z = np.zeros((len(s_idx), Z.shape[1]))
        nodes = np.arange(n)
        for o in np.unique(s_idx):
            dist, pred = dijkstra(csr, directed=True, indices=o, return_predecessors=True)
            has = pred >= 0
            eid = np.zeros(n, dtype=int)
            eid[has] = np.asarray(router.edge_id[pred[has], nodes[has]]).ravel()
            vals = np.zeros((n, Z.shape[1]))
            vals[has] = Z[eid[has] - 1]
            S = router._ancestor_sums(pred, vals)
            m = s_idx == o
            out_t[m] = dist[d_idx[m]]
            out_z[m] = S[d_idx[m]]
        return out_t, out_z

    def design(self, router, Z, s_idx, d_idx, off_m, times) -> tuple[np.ndarray, np.ndarray]:
        """Trip design matrix [1, off-graph metres, route sums] and route times under ``times``."""
        t, zs = self.route_sums(router, times, Z, s_idx, d_idx)
        return np.column_stack([np.ones(len(s_idx)), off_m, zs]), t

    # ------------------------------------------------------------------ fitting
    @staticmethod
    def fit_coef(X: np.ndarray, y: np.ndarray, huber_s: float = 60.0, iters: int = 6,
                 weights: np.ndarray | None = None) -> np.ndarray:
        """Huber regression by iteratively reweighted bounded least squares (class paces >= 0)."""
        w = np.ones(len(y)) if weights is None else weights.astype(float)
        coef = None
        for _ in range(iters):
            coef = StreetSpeedModel.weighted_bounded_ls(X, y, w)
            r = np.abs(y - X @ coef)
            hub = np.where(r <= huber_s, 1.0, huber_s / np.maximum(r, 1e-9))
            w = hub if weights is None else hub * weights
        return coef

    @staticmethod
    def weighted_bounded_ls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """min sum w (y - X b)^2 with start-up time, off-graph pace >= 0 and class paces >= MIN_PACE.

        Reduced to a p x p problem through the Cholesky factor of X'WX (same minimiser).
        """
        from scipy.optimize import lsq_linear

        p = X.shape[1]
        lb = np.full(p, -np.inf)
        lb[0] = lb[1] = 0.0
        lb[2:2 + len(PACE_COLS)] = MIN_PACE
        scale = np.maximum(np.sqrt((w[:, None] * X * X).sum(axis=0) / w.sum()), 1e-12)
        Xs = X / scale
        A = Xs.T @ (Xs * w[:, None]) + 1e-9 * np.eye(p)
        b = Xs.T @ (y * w)
        R = np.linalg.cholesky(A).T  # A = R'R
        rhs = np.linalg.solve(R.T, b)
        res = lsq_linear(R, rhs, bounds=(lb * scale, np.full(p, np.inf)), method="bvls")
        return res.x / scale

    def fit(self, router, start_lat, start_lon, dest_lat, dest_lon, y, rounds: int = 5,
            log=print) -> "StreetSpeedModel":
        Z = edge_design(router)
        s_idx, s_off = router.snap(np.asarray(start_lat), np.asarray(start_lon))
        d_idx, d_off = router.snap(np.asarray(dest_lat), np.asarray(dest_lon))
        off = s_off + d_off
        times = router.edge_time.copy()  # round 1: free flow at posted limits
        prev = None
        for k in range(rounds):
            X, _ = self.design(router, Z, s_idx, d_idx, off, times)
            ok = np.isfinite(X).all(axis=1)
            self.coef = self.fit_coef(X[ok], y[ok])
            pred = X @ self.coef
            mae = float(np.nanmean(np.abs(pred - y)))
            new_times = self.edge_times(router, Z)
            changed = float(np.mean(np.abs(new_times - times) / times))
            self.history.append({"round": k + 1, "mae_s": round(mae, 2), "mean_rel_change_edge_time": round(changed, 4)})
            log(f"  round {k + 1}: training MAE {mae:.2f} s; segment times changed {100 * changed:.1f}%")
            times = new_times
            if prev is not None and abs(prev - mae) < 0.05:
                break
            prev = mae
        self._X_last = X  # final routes' design, for bootstrap intervals
        return self

    # ------------------------------------------------------------------ prediction
    def predict(self, router, start_lat, start_lon, dest_lat, dest_lon) -> np.ndarray:
        """Base driving seconds for each trip along its quickest route under learned speeds."""
        Z = edge_design(router)
        s_idx, s_off = router.snap(np.asarray(start_lat), np.asarray(start_lon))
        d_idx, d_off = router.snap(np.asarray(dest_lat), np.asarray(dest_lon))
        t, _ = self.route_sums(router, self.edge_times(router, Z), Z[:, :1], s_idx, d_idx)
        return self.coef[0] + self.coef[1] * (s_off + d_off) + t

    def table(self, ci: np.ndarray | None = None) -> pd.DataFrame:
        """Coefficients in readable units: s/km for paces and adjustments, s each for point features."""
        unit = np.array([1.0, 1000.0] + [1000.0] * (len(PACE_COLS) + len(ADJ_COLS)) + [1.0] * len(POINT_COLS))
        df = pd.DataFrame({"parameter": TRIP_COLS, "value": self.coef * unit,
                           "unit": ["s"] + ["s/km"] * (1 + len(PACE_COLS) + len(ADJ_COLS)) + ["s each"] * len(POINT_COLS)})
        if ci is not None:
            df["lo"], df["hi"] = ci[0] * unit, ci[1] * unit
        df.loc[df["parameter"].isin(PACE_COLS), "speed_kph"] = 3600 / df["value"]
        return df

    def save(self, path: Path = MODEL_PATH) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"coef": self.coef, "columns": TRIP_COLS, "history": self.history}, path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "StreetSpeedModel":
        import joblib

        b = joblib.load(path)
        return cls(coef=np.asarray(b["coef"]), history=b.get("history", []))


def speed_matrix(model: StreetSpeedModel, router, sites: pd.DataFrame, cells: pd.DataFrame) -> np.ndarray:
    """Base driving seconds, site x cell, along quickest routes under learned speeds."""
    s_idx, s_off = router.snap(sites["lat"].to_numpy(float), sites["lon"].to_numpy(float))
    d_idx, d_off = router.snap(cells["lat"].to_numpy(float), cells["lon"].to_numpy(float))
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    times = model.edge_times(router)
    e = router.edge_id.tocoo()
    n = router.csr.shape[0]
    csr = csr_matrix((times[e.data - 1], (e.row, e.col)), shape=(n, n))
    D = dijkstra(csr, directed=True, indices=np.unique(s_idx))
    row = {o: i for i, o in enumerate(np.unique(s_idx))}
    T = np.vstack([D[row[o], d_idx] for o in s_idx])
    return model.coef[0] + model.coef[1] * (s_off[:, None] + d_off[None, :]) + T
