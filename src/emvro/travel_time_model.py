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
CATEGORICAL = ["borough_code", "incident_group_code", "property_code"]

FEATURE_GROUPS = {
    "base": BASE_FEATURES,
    "street": list(ROUTE_FEATURES),
    "context": list(CONTEXT_FEATURES),
    "weather": WEATHER_FEATURES,
    "incident": INCIDENT_FEATURES,
    "dest": DEST_FEATURES,
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
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    return out[cols]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def make_regressor(seed: int, params: dict[str, Any] | None = None):
    import lightgbm as lgb

    p = {**DEFAULT_PARAMS, **(params or {})}
    return lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **p)


def fit_bag(x: pd.DataFrame, y: np.ndarray, seeds=(42, 43, 44, 45), params: dict[str, Any] | None = None):
    cats = [c for c in CATEGORICAL if c in x.columns]
    models = []
    for s in seeds:
        m = make_regressor(s, params)
        m.fit(x, y, categorical_feature=cats)
        models.append(m)
    return models


def predict_bag(models, x: pd.DataFrame) -> np.ndarray:
    return np.clip(np.mean([m.predict(x) for m in models], axis=0), *DRIVE_CLIP_S)


def predict(bundle: dict, trips: pd.DataFrame) -> np.ndarray:
    """Predicted driving seconds for ``trips`` using a saved bundle from the trainer."""
    from .kolesar import KolesarModel

    groups = bundle["feature_groups"]
    dh = None
    if "dest" in groups:
        dh = DestinationHistory(tuple(bundle["dest_history"]["sizes_m"]), bundle["dest_history"]["shrink"],
                                {int(k): pd.DataFrame(v) for k, v in bundle["dest_history"]["tables"].items()})
    if "weather" in groups and "temp_c" not in trips.columns:
        trips = attach_weather(trips, None)
    x = build_features(trips, groups, kolesar=KolesarModel.from_dict(bundle["kolesar_baseline_params"]),
                       cat_maps=bundle["category_maps"], dest_history=dh)
    return predict_bag(bundle["models"], x[bundle["features"]])
