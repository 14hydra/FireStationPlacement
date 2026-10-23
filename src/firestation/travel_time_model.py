"""LightGBM travel-time model for London fire engines: features, training, prediction.

The model is gradient-boosted decision trees (LightGBM), averaged over a few
random seeds, predicting driving time (LFB ``TravelTimeSeconds``, station
departure → arrival). There is no parametric residual target: the Kolesar curve
appears only as one input feature (``kolesar_prior_s``) and as a baseline.

All features are placement-safe: none identifies the station, so the model can
score a site that has no station yet. Feature groups:

- ``BASE_FEATURES``: road / straight-line distance, circuity, Kolesar prior, time
  of day / week / year, busy engine, pump order, exact-location flag, borough
- ``ROUTE_FEATURES``: street characteristics of the shortest legal route
- ``CONTEXT_FEATURES``: fastest route at posted limits, distance of each end to a
  main road
- ``WEATHER_FEATURES``: hourly central-London weather (Open-Meteo)
- ``INCIDENT_FEATURES``: incident type and property category, home-station flag
- ``DEST_FEATURES``: how much slower/faster than the Kolesar curve past trips to the
  same 300 m / 1 km destination square were (shrunk toward 0; out-of-fold on
  training rows). Destination-only, so it transfers to new station sites.

``build_features`` is the single place features are made; the trainer, the
analysis script and any planner/visualisation should call it via ``predict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .route_features import CONTEXT_FEATURES, ROUTE_FEATURES

DRIVE_CLIP_S = (30.0, 1200.0)

BASE_FEATURES = [
    "road_km", "crow_km", "circuity", "kolesar_prior_s", "hour", "dow", "month", "rush", "night",
    "weekend", "busy_flag", "pump_order", "dest_exact", "borough_code",
]
WEATHER_FEATURES = ["temp_c", "precip_mm", "wind_kph", "rh_pct", "is_rain", "is_heavy_rain", "is_cold", "is_hot"]
INCIDENT_FEATURES = ["incident_group_code", "property_code", "from_home"]
DEST_FEATURES = ["dest_resid_300m", "dest_n_300m", "dest_resid_1km", "dest_n_1km"]
# From the station embedding: PC1 = util + signals/junctions; PC3 = long outer hard trips.
# These interactions let the trees specialise dense-rush under-prediction and long sparse routes
# without using station IDs (still placement-safe).
INTERACTION_FEATURES = [
    "rush_x_signals_per_km",
    "rush_x_junctions_per_km",
    "night_x_signals_per_km",
    "dense_short",   # signals / (1 + road_km) — central short trips (Soho / Euston regime)
    "long_sparse",   # road_km / (1 + signals) — outer long-drive hard cases (Heathrow regime)
]
CATEGORICAL = ["borough_code", "incident_group_code", "property_code"]

FEATURE_GROUPS = {
    "base": BASE_FEATURES,
    "street": list(ROUTE_FEATURES),
    "context": list(CONTEXT_FEATURES),
    "weather": WEATHER_FEATURES,
    "incident": INCIDENT_FEATURES,
    "dest": DEST_FEATURES,
    "interaction": INTERACTION_FEATURES,
}

# Default LightGBM settings; scripts/train_travel_time_lfb.py tunes n_estimators by
# early stopping on a validation year.
DEFAULT_PARAMS: dict[str, Any] = dict(
    n_estimators=800, learning_rate=0.03, num_leaves=63, min_child_samples=35, subsample=0.85,
    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.5, reg_alpha=0.1, objective="l1",
)


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #


def load_weather(path: Path) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    w = pd.read_csv(path, parse_dates=["time"])
    w["weather_hour"] = w["time"].dt.floor("h")
    return w.drop_duplicates("weather_hour").set_index("weather_hour")


def attach_weather(df: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    """Add WEATHER_FEATURES by London local hour of call (NaN when no weather table)."""
    out = df.copy()
    if weather is None:
        for c in WEATHER_FEATURES:
            out[c] = np.nan
        return out
    ts = pd.to_datetime(out["date_of_call"], errors="coerce") + pd.to_timedelta(
        out["hour_of_call"].astype(int), unit="h")
    joined = weather.reindex(ts.dt.floor("h"))
    for c in WEATHER_FEATURES:
        out[c] = joined[c].to_numpy() if c in weather.columns else np.nan
    return out


# --------------------------------------------------------------------------- #
# Destination history (target encoding by destination square)
# --------------------------------------------------------------------------- #


@dataclass
class DestinationHistory:
    """Shrunk mean of (observed - Kolesar prior) by destination grid square.

    ``value = sum(residual) / (n + shrink)``: squares with few past trips stay near 0.
    """

    sizes_m: tuple[int, ...] = (300, 1000)
    shrink: float = 20.0
    tables: dict[int, pd.DataFrame] = field(default_factory=dict)

    @staticmethod
    def _cell(df: pd.DataFrame, size_m: int) -> pd.Series:
        lat = pd.to_numeric(df["dest_lat"], errors="coerce")
        lon = pd.to_numeric(df["dest_lon"], errors="coerce")
        dlat = size_m / 111_320.0
        dlon = size_m / (111_320.0 * np.cos(np.radians(51.5)))
        return (np.floor(lat / dlat)).astype("Int64").astype(str) + "_" + (np.floor(lon / dlon)).astype("Int64").astype(str)

    def fit(self, df: pd.DataFrame, residual: np.ndarray) -> "DestinationHistory":
        for s in self.sizes_m:
            g = pd.DataFrame({"cell": self._cell(df, s).to_numpy(), "r": residual}).groupby("cell")["r"]
            self.tables[s] = pd.DataFrame({"sum": g.sum(), "n": g.size()})
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for s in self.sizes_m:
            name = "300m" if s == 300 else "1km" if s == 1000 else f"{s}m"
            t = self.tables[s].reindex(self._cell(df, s).to_numpy())
            n = t["n"].fillna(0).to_numpy()
            out[f"dest_resid_{name}"] = t["sum"].fillna(0).to_numpy() / (n + self.shrink)
            out[f"dest_n_{name}"] = n
        return out

    def out_of_fold(self, df: pd.DataFrame, residual: np.ndarray, groups: pd.Series, k: int = 5) -> pd.DataFrame:
        """Encode each training row using only the other folds (no row sees its own target)."""
        fold = (pd.util.hash_pandas_object(groups.astype(str), index=False).to_numpy() % k).astype(int)
        out = pd.DataFrame(index=df.index, columns=DEST_FEATURES, dtype=float)
        for f in range(k):
            m = fold == f
            enc = DestinationHistory(self.sizes_m, self.shrink).fit(df[~m], residual[~m]).transform(df[m])
            out.loc[m, enc.columns] = enc.to_numpy()
        return out


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def category_maps(train: pd.DataFrame) -> dict[str, list[str]]:
    def cats(col):
        return sorted(train[col].astype(str).str.upper().unique()) if col in train.columns else []

    return {"borough": cats("borough"), "incident_group": cats("incident_group"),
            "property_category": cats("property_category")}


def _codes(series: pd.Series, cats: list[str]) -> np.ndarray:
    m = {c: i for i, c in enumerate(cats)}
    return series.astype(str).str.upper().map(m).astype(float).to_numpy()  # unseen -> NaN


def build_features(
    df: pd.DataFrame,
    groups: list[str],
    *,
    kolesar,
    cat_maps: dict[str, list[str]],
    dest_history: DestinationHistory | None = None,
    dest_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature matrix for the requested FEATURE_GROUPS.

    ``df`` needs the training-set columns (road_km, crow_km, date_of_call, hour_of_call,
    busy_flag, borough, …); missing optional columns become NaN or a typical value.
    For "dest", pass either a fitted ``dest_history`` or a precomputed ``dest_frame``
    (out-of-fold encodings for training rows).
    """
    d = pd.to_datetime(df["date_of_call"], errors="coerce")
    road = pd.to_numeric(df["road_km"], errors="coerce").to_numpy(dtype=float)
    crow = pd.to_numeric(df["crow_km"], errors="coerce").to_numpy(dtype=float)
    out = pd.DataFrame(index=df.index)
    out["road_km"] = road
    out["crow_km"] = crow
    with np.errstate(divide="ignore", invalid="ignore"):
        circ = np.where((crow > 0.05) & np.isfinite(crow) & np.isfinite(road), road / crow, 1.25)
    out["circuity"] = np.clip(circ, 0.8, 4.0)
    out["kolesar_prior_s"] = np.clip(kolesar.predict(road), *DRIVE_CLIP_S)
    h = pd.to_numeric(df["hour_of_call"], errors="coerce").fillna(12).astype(int)
    out["hour"] = h.to_numpy()
    out["dow"] = d.dt.dayofweek.to_numpy()
    out["month"] = d.dt.month.to_numpy()
    out["rush"] = (h.between(7, 9) | h.between(16, 19)).astype(int).to_numpy()
    out["night"] = ((h >= 22) | (h <= 5)).astype(int).to_numpy()
    out["weekend"] = (out["dow"] >= 5).astype(int)

    def col(name, default):
        return pd.to_numeric(df[name], errors="coerce").fillna(default) if name in df.columns else default

    out["busy_flag"] = col("busy_flag", 0)
    out["pump_order"] = np.clip(col("pump_order", 1), 1, 5)
    out["dest_exact"] = col("dest_exact", 0)
    out["borough_code"] = _codes(df["borough"], cat_maps["borough"])
    for c in list(ROUTE_FEATURES) + list(CONTEXT_FEATURES) + WEATHER_FEATURES:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    out["incident_group_code"] = (_codes(df["incident_group"], cat_maps["incident_group"])
                                  if "incident_group" in df.columns else np.nan)
    out["property_code"] = (_codes(df["property_category"], cat_maps["property_category"])
                            if "property_category" in df.columns else np.nan)
    out["from_home"] = ((df["deployed_from_location"] == "Home Station").astype(int)
                        if "deployed_from_location" in df.columns else 1)
    if "dest" in groups:
        enc = dest_frame if dest_frame is not None else dest_history.transform(df)
        for c in DEST_FEATURES:
            out[c] = enc[c].to_numpy(dtype=float)
    if "interaction" in groups:
        def _num(name: str) -> np.ndarray:
            if name in out.columns:
                return pd.to_numeric(out[name], errors="coerce").fillna(0.0).to_numpy(float)
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(float)
            return np.zeros(len(df), dtype=float)

        sig, jun = _num("signals_per_km"), _num("junctions_per_km")
        rush = out["rush"].to_numpy(dtype=float)
        night = out["night"].to_numpy(dtype=float)
        out["rush_x_signals_per_km"] = rush * sig
        out["rush_x_junctions_per_km"] = rush * jun
        out["night_x_signals_per_km"] = night * sig
        out["dense_short"] = sig / (1.0 + np.clip(road, 0.0, None))
        out["long_sparse"] = np.clip(road, 0.0, None) / (1.0 + sig)
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    return out[cols]


def trip_regime(df: pd.DataFrame) -> np.ndarray:
    """Placement-safe trip buckets from the embedding hard-case analysis.

    dense_short — high signals/km, short road (central under-prediction)
    long_sparse — long road, low signals (outer long-drive hard cases)
    rush_other  — rush hour, neither of the above
    other
    """
    road = pd.to_numeric(df["road_km"], errors="coerce").fillna(2.0).to_numpy(float)
    sig = (pd.to_numeric(df["signals_per_km"], errors="coerce").fillna(5.0).to_numpy(float)
           if "signals_per_km" in df.columns else np.full(len(df), 5.0))
    if "rush" in df.columns:
        rush = pd.to_numeric(df["rush"], errors="coerce").fillna(0).to_numpy(float)
    elif "hour_of_call" in df.columns:
        h = pd.to_numeric(df["hour_of_call"], errors="coerce").fillna(12)
        rush = (h.between(7, 9) | h.between(16, 19)).astype(float).to_numpy()
    else:
        rush = np.zeros(len(df))
    regime = np.full(len(df), "other", dtype=object)
    regime[(sig >= 10.0) & (road <= 2.5)] = "dense_short"
    regime[(road >= 4.0) & (sig <= 6.0)] = "long_sparse"
    regime[(regime == "other") & (rush >= 1)] = "rush_other"
    return regime


def fit_regime_bias(
    df: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    *,
    only: tuple[str, ...] = ("long_sparse",),
) -> dict[str, float]:
    """Mean (pred − y) by trip regime on training rows — used as a post-hoc correction.

    Default: correct only ``long_sparse`` (outer long-drive / PC3 hard cases). A/B tests
    showed correcting all regimes (esp. dense_short) can raise overall MAE even when it
    zeroes average bias, because L1 trees already balance over/under errors.
    """
    reg = trip_regime(df)
    resid = np.asarray(pred, float) - np.asarray(y, float)
    out = {name: 0.0 for name in ("dense_short", "long_sparse", "rush_other", "other")}
    for name in only:
        m = reg == name
        if m.any():
            out[name] = float(resid[m].mean())
    return out


def apply_regime_bias(df: pd.DataFrame, pred: np.ndarray, bias: dict[str, float]) -> np.ndarray:
    reg = trip_regime(df)
    adj = pred.astype(float).copy()
    for name, b in bias.items():
        adj[reg == name] -= b  # remove systematic over/under-prediction
    return np.clip(adj, *DRIVE_CLIP_S)


def sample_weights(df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Up-weight embedding hard cases; soft-downweight data-quality outliers.

    - Longer road_km trips (PC3 / outer long-drive) get higher weight.
    - Dense short trips (high signals/km, short road — central under-prediction) get a boost.
    - Impossible speeds / ultra-long drives are soft-downweighted (not dropped), matching the
      outlier analysis in data/figures/travel_time/outliers/.
    """
    road = pd.to_numeric(df["road_km"], errors="coerce").fillna(2.0).to_numpy(float)
    y = np.asarray(y, dtype=float)
    # Slightly stronger long-trip weight (embedding PC3 / outer hard cases in fig 06).
    w = 1.0 + 0.18 * np.clip(road, 0.0, 8.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kph = np.where(y > 0, road / y * 3600.0, np.nan)
    bad = (~np.isfinite(kph)) | (kph < 8.0) | (kph > 70.0) | (y > 900.0)
    w = np.where(bad, 0.35 * w, w)
    if "signals_per_km" in df.columns:
        sig = pd.to_numeric(df["signals_per_km"], errors="coerce").fillna(5.0).to_numpy(float)
        dense_short = (sig >= 10.0) & (road <= 2.5)
        w = np.where(dense_short, w * 1.25, w)
    return w.astype(float)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def make_regressor(seed: int, params: dict[str, Any] | None = None):
    import lightgbm as lgb

    p = {**DEFAULT_PARAMS, **(params or {})}
    return lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **p)


def fit_bag(
    x: pd.DataFrame,
    y: np.ndarray,
    seeds=(42, 43, 44, 45),
    params: dict[str, Any] | None = None,
    sample_weight: np.ndarray | None = None,
):
    cats = [c for c in CATEGORICAL if c in x.columns]
    models = []
    for s in seeds:
        m = make_regressor(s, params)
        kw = dict(categorical_feature=cats)
        if sample_weight is not None:
            kw["sample_weight"] = sample_weight
        m.fit(x, y, **kw)
        models.append(m)
    return models


def predict_bag(models, x: pd.DataFrame) -> np.ndarray:
    return np.clip(np.mean([m.predict(x) for m in models], axis=0), *DRIVE_CLIP_S)


def _infer_feature_groups(feature_names: list[str]) -> list[str]:
    """Recover FEATURE_GROUPS keys for older bundles that only stored ``features``."""
    feats = set(feature_names)
    return [g for g, cols in FEATURE_GROUPS.items() if any(c in feats for c in cols)]


def predict(bundle: dict, trips: pd.DataFrame) -> np.ndarray:
    """Predicted driving seconds for ``trips`` using a saved bundle from the trainer."""
    from .kolesar import KolesarModel

    groups = list(bundle.get("feature_groups") or _infer_feature_groups(bundle["features"]))
    # Drop dest if the bundle has no destination-history tables (legacy models).
    if "dest" in groups and "dest_history" not in bundle:
        groups = [g for g in groups if g != "dest"]
    dh = None
    if "dest" in groups:
        dh = DestinationHistory(tuple(bundle["dest_history"]["sizes_m"]), bundle["dest_history"]["shrink"],
                                {int(k): pd.DataFrame(v) for k, v in bundle["dest_history"]["tables"].items()})
    if "weather" in groups and "temp_c" not in trips.columns:
        trips = attach_weather(trips, None)
    cats = bundle.get("category_maps")
    if cats is None:
        cats = {"borough": list(bundle.get("borough_categories") or []),
                "incident_group": [], "property_category": []}
    x = build_features(trips, groups, kolesar=KolesarModel.from_dict(bundle["kolesar_baseline_params"]),
                       cat_maps=cats, dest_history=dh)
    if bundle.get("base") == "street_speeds":
        # Correction model: learned street-speed time (trips["street_base_s"]) + LightGBM correction.
        base = pd.to_numeric(trips["street_base_s"], errors="coerce").to_numpy(float)
        x["street_base_s"] = base
        corr = np.mean([m.predict(x[bundle["features"]]) for m in bundle["models"]], axis=0)
        pred = np.clip(base + corr, *DRIVE_CLIP_S)
    else:
        # Legacy bundles may list columns build_features already produced under groups.
        missing = [c for c in bundle["features"] if c not in x.columns]
        for c in missing:
            x[c] = np.nan
        pred = predict_bag(bundle["models"], x[bundle["features"]])
    if bundle.get("regime_bias"):
        # Ensure rush exists for trip_regime (predict callers may only pass training columns).
        trips2 = trips
        if "rush" not in trips.columns and "hour_of_call" in trips.columns:
            trips2 = trips.copy()
            h = pd.to_numeric(trips2["hour_of_call"], errors="coerce").fillna(12)
            trips2["rush"] = (h.between(7, 9) | h.between(16, 19)).astype(int)
        pred = apply_regime_bias(trips2, pred, bundle["regime_bias"])
    return pred
