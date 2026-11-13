#!/usr/bin/env python3
"""Weak-label NYC FDNY travel-time check (nearest firehouse + alarm-box dest).

FDNY open data has travel seconds but not which company responded. Destinations
come from the official In-Service Alarm Box Locations map (``v57i-gtxb``).
Origins assume the nearest FDNY firehouse.

Predicts with:
  - London-default Kolesar (crow×1.3) — transfer check
  - NYC-refit Kolesar on holdout — functional-form check
  - Distance + hour + borough linear model — stronger weak-label protocol
  - Headline slice: structural/non-structural fires with travel < 600 s,
    nearest Engine house

Writes ``data/figures/validation/nyc_fdny/``.

Run:
  PYTHONPATH=src .venv/bin/python scripts/validate_nyc_fdny_travel.py
  PYTHONPATH=src .venv/bin/python scripts/validate_nyc_fdny_travel.py --force
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.kolesar import KolesarModel, fit_kolesar  # noqa: E402

RAW = ROOT / "data" / "raw" / "megacities" / "new_york"
OUT = ROOT / "data" / "figures" / "validation" / "nyc_fdny"

FIREHOUSE_URL = "https://data.cityofnewyork.us/resource/hc8x-tcnd.json"
INCIDENT_URL = "https://data.cityofnewyork.us/resource/8m42-w767.json"
ALARMBOX_URL = "https://data.cityofnewyork.us/resource/v57i-gtxb.json"

CTX = ssl.create_default_context()
UA = {"User-Agent": "EMVRouteOptimization/1.0 (research; NYC FDNY validation)"}

BOROUGH_PREFIX = {
    "BRONX": "X",
    "BROOKLYN": "B",
    "MANHATTAN": "M",
    "QUEENS": "Q",
    "RICHMOND / STATEN ISLAND": "R",
    "STATEN ISLAND": "R",
    "RICHMOND": "R",
}


def _get_json(url: str, params: dict | None = None, *, retries: int = 4) -> list | dict:
    full = url if not params else url + "?" + urllib.parse.urlencode(params)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(full, headers=UA)
            with urllib.request.urlopen(req, timeout=120, context=CTX) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed {full}: {last_err}")


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def fetch_firehouses(*, force: bool = False) -> pd.DataFrame:
    path = RAW / "fdny_firehouses.csv"
    if path.exists() and not force:
        print(f"keep {path}")
        return pd.read_csv(path)
    print("fetching FDNY firehouses…", flush=True)
    rows = _get_json(FIREHOUSE_URL, {"$limit": "5000"})
    df = pd.DataFrame(rows)
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    keep = [c for c in ("facilityname", "facilityaddress", "borough", "postcode", "lat", "lon") if c in df.columns]
    out = df[keep].copy()
    RAW.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} houses)")
    return out


def fetch_alarm_boxes(*, force: bool = False) -> pd.DataFrame:
    path = RAW / "fdny_alarm_boxes.csv"
    if path.exists() and not force:
        print(f"keep {path}")
        return pd.read_csv(path)
    print("fetching FDNY alarm box locations…", flush=True)
    rows = _get_json(ALARMBOX_URL, {"$limit": "50000"})
    df = pd.DataFrame(rows)
    df["dest_lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["dest_lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["dest_lat", "dest_lon", "borobox"]).copy()
    df["borobox"] = df["borobox"].astype(str).str.upper().str.strip()
    out = df[["borobox", "location", "borough", "zip", "dest_lat", "dest_lon"]].drop_duplicates("borobox")
    RAW.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} boxes)")
    return out


def fetch_incidents(*, n: int = 80_000, force: bool = False) -> pd.DataFrame:
    path = RAW / "fdny_incidents_sample.csv"
    if path.exists() and not force:
        print(f"keep {path}")
        return pd.read_csv(path)
    print(f"fetching FDNY incidents (up to {n})…", flush=True)
    page = 50_000
    frames = []
    offset = 0
    where = (
        "valid_incident_rspns_time_indc='Y' AND "
        "incident_travel_tm_seconds_qy > 30 AND "
        "incident_travel_tm_seconds_qy < 1200 AND "
        "engines_assigned_quantity > 0"
    )
    while offset < n:
        limit = min(page, n - offset)
        print(f"  offset={offset} limit={limit}", flush=True)
        rows = _get_json(
            INCIDENT_URL,
            {
                "$limit": str(limit),
                "$offset": str(offset),
                "$where": where,
                "$order": "incident_datetime DESC",
                "$select": (
                    "starfire_incident_id,incident_datetime,alarm_box_borough,alarm_box_number,"
                    "alarm_box_location,incident_borough,zipcode,"
                    "incident_travel_tm_seconds_qy,incident_response_seconds_qy,"
                    "engines_assigned_quantity,incident_classification_group"
                ),
            },
        )
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        offset += len(rows)
        if len(rows) < limit:
            break
        time.sleep(0.35)
    if not frames:
        raise RuntimeError("no FDNY incidents returned")
    df = pd.concat(frames, ignore_index=True)
    df["travel_s"] = pd.to_numeric(df["incident_travel_tm_seconds_qy"], errors="coerce")
    df = df.dropna(subset=["travel_s"]).reset_index(drop=True)
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} incidents)")
    return df


def make_borobox(borough: str, number) -> str | None:
    if borough is None or number is None or (isinstance(number, float) and np.isnan(number)):
        return None
    b = str(borough).strip().upper()
    prefix = BOROUGH_PREFIX.get(b)
    if prefix is None:
        return None
    digits = "".join(ch for ch in str(number) if ch.isdigit())
    if not digits:
        return None
    return f"{prefix}{digits.zfill(4)}"


def attach_alarm_boxes(inc: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    out = inc.copy()
    out["borobox"] = [
        make_borobox(b, n) for b, n in zip(out["alarm_box_borough"], out["alarm_box_number"])
    ]
    merged = out.merge(
        boxes[["borobox", "dest_lat", "dest_lon", "location"]],
        on="borobox",
        how="left",
    )
    rate = merged["dest_lat"].notna().mean()
    print(f"alarm-box match rate: {rate:.1%} ({merged['dest_lat'].notna().sum():,} / {len(merged):,})")
    return merged.dropna(subset=["dest_lat", "dest_lon"]).reset_index(drop=True)


def nearest_firehouse(dest_lat: np.ndarray, dest_lon: np.ndarray, houses: pd.DataFrame):
    hlat = houses["lat"].to_numpy(float)
    hlon = houses["lon"].to_numpy(float)
    best_km = np.full(len(dest_lat), np.inf)
    best_i = np.zeros(len(dest_lat), dtype=int)
    for i0 in range(0, len(houses), 40):
        i1 = min(i0 + 40, len(houses))
        d = _haversine_km(dest_lat[:, None], dest_lon[:, None], hlat[None, i0:i1], hlon[None, i0:i1])
        j = d.argmin(axis=1)
        km = d[np.arange(len(dest_lat)), j]
        better = km < best_km
        best_km[better] = km[better]
        best_i[better] = i0 + j[better]
    return best_km, best_i


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    return {
        "n": int(len(y)),
        "mae_s": float(np.mean(np.abs(err))),
        "median_abs_s": float(np.median(np.abs(err))),
        "bias_s": float(np.mean(err)),
        "rmse_s": float(np.sqrt(np.mean(err**2))),
        "corr": float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 and np.std(p) > 1e-9 else float("nan"),
        "y_mean_s": float(np.mean(y)),
        "p_mean_s": float(np.mean(p)),
    }


def _holdout_kolesar(road_km: np.ndarray, y: np.ndarray, train: np.ndarray) -> tuple[dict, KolesarModel]:
    model = fit_kolesar(road_km[train], y[train], robust=True)
    pred = model.predict(road_km[~train])
    return metrics(y[~train], pred), model


def _holdout_linear(
    road_km: np.ndarray,
    y: np.ndarray,
    hour: np.ndarray,
    borough: np.ndarray,
    train: np.ndarray,
) -> dict:
    """travel ~ 1 + sqrt(road) + road + hour + borough dummies (holdout)."""
    from numpy.linalg import lstsq

    boro = pd.get_dummies(pd.Series(borough).fillna("?"), drop_first=True).to_numpy(float)
    x = np.column_stack([np.ones(len(y)), np.sqrt(road_km), road_km, hour.astype(float), boro])
    beta = lstsq(x[train], y[train], rcond=None)[0]
    pred = x[~train] @ beta
    return metrics(y[~train], pred)


def run(*, n_incidents: int = 80_000, force: bool = False, road_factor: float = 1.3, seed: int = 42) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    houses_all = fetch_firehouses(force=force)
    # Engine companies only — ladders/battalions alone are not first-due engines
    houses = houses_all[houses_all["facilityname"].astype(str).str.contains("Engine", case=False, na=False)].copy()
    if len(houses) < 50:
        houses = houses_all
    boxes = fetch_alarm_boxes(force=force)
    inc = fetch_incidents(n=n_incidents, force=force)
    inc = attach_alarm_boxes(inc, boxes)
    inc["hour"] = pd.to_datetime(inc["incident_datetime"], errors="coerce").dt.hour.fillna(12).astype(int)
    inc["is_fire"] = inc["incident_classification_group"].isin(["Structural Fires", "NonStructural Fires"])

    y_all = inc["travel_s"].to_numpy(float)
    crow_km, _ = nearest_firehouse(
        inc["dest_lat"].to_numpy(float),
        inc["dest_lon"].to_numpy(float),
        houses,
    )
    road_km = crow_km * road_factor
    hour = inc["hour"].to_numpy(int)
    is_fire = inc["is_fire"].to_numpy(bool)
    borough = inc["alarm_box_borough"].astype(str).to_numpy()

    base = (crow_km >= 0.05) & (crow_km <= 12.0) & np.isfinite(y_all)
    print(f"matched with distance filter: {base.sum():,} / {len(base):,} (engine houses={len(houses)})")

    rng = np.random.default_rng(seed)
    train_all = rng.random(len(y_all)) < 0.7
    london = KolesarModel.london_default()

    def slice_eval(mask: np.ndarray) -> dict:
        y = y_all[mask]
        road = road_km[mask]
        crow = crow_km[mask]
        tr = train_all[mask]
        hr = hour[mask]
        boro = borough[mask]
        const = np.full_like(y, float(np.median(y[tr])))
        kol, model = _holdout_kolesar(road, y, tr)
        lin = _holdout_linear(road, y, hr, boro, tr)
        return {
            "n": int(mask.sum()),
            "constant_median": metrics(y[~tr], const[~tr]),
            "london_kolesar_road_x1.3": metrics(y, london.predict(road)),
            "nyc_refit_kolesar_holdout": kol,
            "nyc_linear_tod_borough_holdout": lin,
            "nyc_kolesar_params": model.to_dict(),
            "distance_summary": {
                "crow_km_median": float(np.median(crow)),
                "road_km_median": float(np.median(road)),
                "travel_s_median": float(np.median(y)),
            },
        }

    slices = {
        "all_calls": slice_eval(base),
        "travel_under_600s": slice_eval(base & (y_all < 600)),
        "fires_travel_under_600s": slice_eval(base & is_fire & (y_all < 600)),
    }

    # Headline = cleanest weak-label protocol: fires, travel < 10 min, nearest engine,
    # distance + hour + borough (still no responding company).
    head_slice = slices["fires_travel_under_600s"]
    results = {
        "all_calls_constant": slices["all_calls"]["constant_median"],
        "all_calls_nyc_kolesar": slices["all_calls"]["nyc_refit_kolesar_holdout"],
        "all_calls_nyc_linear": slices["all_calls"]["nyc_linear_tod_borough_holdout"],
        "fires_nyc_kolesar": head_slice["nyc_refit_kolesar_holdout"],
        "fires_nyc_linear": head_slice["nyc_linear_tod_borough_holdout"],
        "fires_constant": head_slice["constant_median"],
        # aliases kept for older doc readers
        "constant_median": slices["all_calls"]["constant_median"],
        "nyc_refit_kolesar_road_holdout": slices["all_calls"]["nyc_refit_kolesar_holdout"],
        "london_kolesar_road_x1.3": slices["all_calls"]["london_kolesar_road_x1.3"],
    }

    summary = {
        "doc": (
            "NYC FDNY weak-label travel-time check. Destination = official alarm-box "
            "lat/lon (v57i-gtxb); origin = nearest Engine firehouse (no responding company "
            "in open CAD). Headline uses structural/non-structural fires with travel < 600 s "
            "and a holdout model of distance + hour + borough. Baseline all-calls Kolesar "
            "refit is also reported. Not LFB-grade OD validation; does not validate LightGBM "
            "or placement savings."
        ),
        "n_firehouses_engine": int(len(houses)),
        "n_firehouses_all": int(len(houses_all)),
        "n_alarm_boxes": int(len(boxes)),
        "n_incidents_raw": int(len(pd.read_csv(RAW / "fdny_incidents_sample.csv"))),
        "n_incidents_matched": int(base.sum()),
        "road_factor": road_factor,
        "london_kolesar": london.to_dict(),
        "filters": {
            "valid_incident_rspns_time_indc": "Y",
            "travel_s_raw": "30–1200",
            "engines_assigned_quantity": ">0",
            "crow_km": "0.05–12",
            "headline_slice": "Structural/NonStructural Fires AND travel < 600 s",
            "origin": "nearest facility with 'Engine' in the name",
        },
        "slices": slices,
        "models": results,
        "headline": {
            "slice": "fires_travel_under_600s",
            "nyc_refit_mae_s": head_slice["nyc_linear_tod_borough_holdout"]["mae_s"],
            "nyc_refit_corr": head_slice["nyc_linear_tod_borough_holdout"]["corr"],
            "nyc_kolesar_mae_s": head_slice["nyc_refit_kolesar_holdout"]["mae_s"],
            "nyc_kolesar_corr": head_slice["nyc_refit_kolesar_holdout"]["corr"],
            "constant_mae_s": head_slice["constant_median"]["mae_s"],
            "all_calls_kolesar_mae_s": slices["all_calls"]["nyc_refit_kolesar_holdout"]["mae_s"],
            "all_calls_kolesar_corr": slices["all_calls"]["nyc_refit_kolesar_holdout"]["corr"],
            "london_transfer_mae_s": slices["all_calls"]["london_kolesar_road_x1.3"]["mae_s"],
            "london_transfer_corr": slices["all_calls"]["london_kolesar_road_x1.3"]["corr"],
            "n": head_slice["n"],
        },
        "caveat": (
            "Nearest-station assumption inflates error vs true OD pairs. "
            "Headline restricts to fire calls and adds hour/borough; still weak labels. "
            "Do not treat this as validating London LightGBM or megacity placement savings."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    flat_rows = []
    for slice_name, block in slices.items():
        for model_name in ("constant_median", "london_kolesar_road_x1.3", "nyc_refit_kolesar_holdout", "nyc_linear_tod_borough_holdout"):
            flat_rows.append({"slice": slice_name, "model": model_name, **block[model_name]})
    pd.DataFrame(flat_rows).to_csv(OUT / "mae_table.csv", index=False)

    # Rebuild holdout preds for headline scatter (fires + linear features)
    fire_mask = base & is_fire & (y_all < 600)
    y_f = y_all[fire_mask]
    road_f = road_km[fire_mask]
    tr_f = train_all[fire_mask]
    hr_f = hour[fire_mask]
    boro_f = borough[fire_mask]
    from numpy.linalg import lstsq

    boro_x = pd.get_dummies(pd.Series(boro_f).fillna("?"), drop_first=True).to_numpy(float)
    x_f = np.column_stack([np.ones(len(y_f)), np.sqrt(road_f), road_f, hr_f.astype(float), boro_x])
    beta = lstsq(x_f[tr_f], y_f[tr_f], rcond=None)[0]
    pred_lin = x_f[~tr_f] @ beta
    y_te = y_f[~tr_f]
    kol_m = fit_kolesar(road_f[tr_f], y_f[tr_f], robust=True)
    pred_kol = kol_m.predict(road_f[~tr_f])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    idx = rng.choice(len(y_te), size=min(4000, len(y_te)), replace=False)
    ax = axes[0]
    ax.scatter(y_te[idx], pred_lin[idx], s=10, alpha=0.3, c="#15803d", linewidths=0)
    lim = max(np.percentile(y_te, 99), np.percentile(pred_lin, 99))
    ax.plot([0, lim], [0, lim], color="#15202b", lw=1.2, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    m = head_slice["nyc_linear_tod_borough_holdout"]
    ax.set_title(f"Fires · distance+hour+borough\nMAE {m['mae_s']:.0f} s · corr {m['corr']:.2f}")
    ax.set_xlabel("FDNY recorded travel (s)")
    ax.set_ylabel("Predicted (s)")

    ax = axes[1]
    ax.scatter(y_te[idx], pred_kol[idx], s=10, alpha=0.3, c="#1d5bbf", linewidths=0)
    lim2 = max(np.percentile(y_te, 99), np.percentile(pred_kol, 99), 1)
    ax.plot([0, lim2], [0, lim2], color="#15202b", lw=1.2, ls="--")
    ax.set_xlim(0, lim2)
    ax.set_ylim(0, lim2)
    m2 = head_slice["nyc_refit_kolesar_holdout"]
    ax.set_title(f"Fires · NYC-refit Kolesar\nMAE {m2['mae_s']:.0f} s · corr {m2['corr']:.2f}")
    ax.set_xlabel("FDNY recorded travel (s)")
    ax.set_ylabel("Predicted (s)")
    fig.suptitle(
        "NYC FDNY weak-label check — nearest engine → alarm box (fires, travel < 10 min)",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Weak labels: no responding company. Headline = fires + hour/borough. Still not LFB-grade OD validation.",
        ha="center",
        fontsize=8,
        color="#5b6570",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(OUT / "scatter_kolesar.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    ax.set_axis_off()
    rows = [["Protocol", "MAE (s)", "Corr", "vs constant"]]
    for lab, block, key in [
        ("All calls · Kolesar", slices["all_calls"], "nyc_refit_kolesar_holdout"),
        ("All calls · dist+hour+boro", slices["all_calls"], "nyc_linear_tod_borough_holdout"),
        ("Fires <10 min · Kolesar", head_slice, "nyc_refit_kolesar_holdout"),
        ("Fires <10 min · dist+hour+boro", head_slice, "nyc_linear_tod_borough_holdout"),
    ]:
        m = block[key]
        c = block["constant_median"]["mae_s"]
        rows.append([lab, f"{m['mae_s']:.1f}", f"{m['corr']:.2f}", f"{m['mae_s'] - c:+.1f} s"])
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.15, 1.55)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1d5bbf")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 4:
            cell.set_facecolor("#dcfce7")
    ax.set_title("NYC FDNY weak-label validation — improved protocols", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "mae_table.png", dpi=140)
    plt.close(fig)

    import shutil

    shutil.copy2(OUT / "scatter_kolesar.png", OUT / "scatter_kolesar_road.png")

    print(json.dumps(summary["headline"], indent=2))
    print("wrote", OUT)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-incidents", type=int, default=80_000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--road-factor", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(n_incidents=args.n_incidents, force=args.force, road_factor=args.road_factor, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
