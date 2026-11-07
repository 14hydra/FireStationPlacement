#!/usr/bin/env python3
"""Build megacity inputs (if needed) and run replace-10 placement for each city.

Does **not** touch ``data/figures/final/``. Writes per-city artifacts under
``data/figures/megacities/<id>/``, aggregate ``summary.csv``, and GeoJSON for the
Google Maps viewer under ``web/megacity_maps/data/<id>/``.

Run:
  PYTHONPATH=src .venv/bin/python scripts/run_megacity_placement.py
  PYTHONPATH=src .venv/bin/python scripts/run_megacity_placement.py --city tokyo --skip-graph
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.cities import custom_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.fire_station_location import run_fire_station_plan  # noqa: E402
from firestation.megacities import get_megacity, iter_megacities, list_megacities  # noqa: E402

WEB_DATA = ROOT / "web" / "megacity_maps" / "data"
FIG_ROOT = ROOT / "data" / "figures" / "megacities"
PY = ROOT / ".venv" / "bin" / "python"


def _run_builder(script: str, city_id: str, extra: list[str] | None = None) -> None:
    cmd = [str(PY), str(ROOT / "scripts" / script), "--city", city_id, *(extra or [])]
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT / "src")}
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def _ensure_inputs(
    city_id: str,
    *,
    force: bool,
    skip_graph: bool,
    synthetic_stations: bool = True,
    n_cells: int = 1800,
) -> None:
    city = get_megacity(city_id)
    if force or not city.fire_stations_path.exists():
        extra = []
        if force:
            extra.append("--force")
        if synthetic_stations:
            extra.append("--synthetic")
        _run_builder("build_megacity_stations.py", city_id, extra or None)
    # Urban outline for demand clipping (Nominatim); non-fatal if offline
    extent = city.raw_dir / "urban_extent.geojson"
    if force or not extent.exists():
        try:
            _run_builder(
                "build_megacity_urban_extent.py",
                city_id,
                ["--force"] if force else None,
            )
        except subprocess.CalledProcessError as exc:
            print(f"{city_id}: urban extent fetch failed ({exc}); ellipse fallback")
    if force or not city.incidents_path.exists():
        extra = [f"--n-cells={n_cells}"]
        if force:
            extra.append("--force")
        _run_builder("build_megacity_demand.py", city_id, extra)
    if not skip_graph and (force or not city.graph_path.exists()):
        try:
            _run_builder("build_megacity_graph.py", city_id, ["--force"] if force else None)
        except subprocess.CalledProcessError as exc:
            print(f"{city_id}: graph build failed ({exc}); continuing with crow Kolesar")


def _geojson_points(df: pd.DataFrame, props: list[str], lat="lat", lon="lon") -> dict:
    feats = []
    for _, r in df.iterrows():
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(r[lon]), float(r[lat])]},
                "properties": {k: (None if pd.isna(r[k]) else r[k]) for k in props if k in r.index},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def _pts_geojson(frame: pd.DataFrame, name_key: str = "site_name") -> dict:
    if frame is None or len(frame) == 0:
        return {"type": "FeatureCollection", "features": []}
    f = frame.copy()
    if name_key in f.columns:
        f = f.rename(columns={name_key: "name"})
    elif "facilityname" in f.columns:
        f = f.rename(columns={"facilityname": "name"})
    else:
        f["name"] = f.index.astype(str)
    props = [c for c in ("name", "n_pumps", "assigned_demand", "assigned_share") if c in f.columns]
    return _geojson_points(f, props)


def _export_viewer(city_id: str, out_dir: Path, summary: dict) -> None:
    city = get_megacity(city_id)
    dest = WEB_DATA / city_id
    dest.mkdir(parents=True, exist_ok=True)

    planned = pd.read_csv(out_dir / "planned_fire_stations.csv")
    closed = (
        pd.read_csv(out_dir / "closed_fire_stations.csv")
        if (out_dir / "closed_fire_stations.csv").exists()
        else pd.DataFrame()
    )
    opened = (
        pd.read_csv(out_dir / "opened_fire_stations.csv")
        if (out_dir / "opened_fire_stations.csv").exists()
        else pd.DataFrame()
    )
    cells = pd.read_csv(out_dir / "demand_cells.csv")

    closed_names = (
        set(closed["site_name"].astype(str))
        if len(closed) and "site_name" in closed.columns
        else set()
    )
    name_col = "site_name" if "site_name" in planned.columns else "facilityname"
    kept = planned[planned["is_existing"] == True].copy() if "is_existing" in planned.columns else planned.copy()
    if name_col in kept.columns and closed_names:
        kept = kept[~kept[name_col].astype(str).isin(closed_names)]

    before_c = "response_s_current" if "response_s_current" in cells.columns else None
    after_c = "response_s" if "response_s" in cells.columns else None
    delta_c = "response_s_delta" if "response_s_delta" in cells.columns else None
    if after_c and before_c and delta_c is None:
        cells["response_s_delta"] = cells[after_c] - cells[before_c]
        delta_c = "response_s_delta"

    (dest / "stations_kept.geojson").write_text(json.dumps(_pts_geojson(kept, name_col)))
    (dest / "stations_closed.geojson").write_text(json.dumps(_pts_geojson(closed)))
    (dest / "stations_opened.geojson").write_text(json.dumps(_pts_geojson(opened)))

    cdf = cells.copy()
    cell_props = [c for c in ("weight", "share", before_c, after_c, delta_c) if c and c in cdf.columns]
    (dest / "demand_cells.geojson").write_text(json.dumps(_geojson_points(cdf, cell_props)))

    # Local move rationale (~3 km)
    rationale = []
    LOCAL_DEG = 0.025
    w = cdf["weight"].to_numpy(float) if "weight" in cdf.columns else np.ones(len(cdf))
    w_sum = float(w.sum()) or 1.0
    before = cdf[before_c].to_numpy(float) if before_c else None
    after = cdf[after_c].to_numpy(float) if after_c else None
    delta = cdf[delta_c].to_numpy(float) if delta_c else None

    def local_stats(row, role):
        m = ((cdf["lat"] - float(row["lat"])) ** 2 + (cdf["lon"] - float(row["lon"])) ** 2) <= LOCAL_DEG ** 2
        if not m.any():
            return None
        ww = w[m.to_numpy()]
        share = float(ww.sum() / w_sum)
        b = float(np.average(before[m.to_numpy()], weights=ww)) if before is not None else None
        a = float(np.average(after[m.to_numpy()], weights=ww)) if after is not None else None
        d = float(np.average(delta[m.to_numpy()], weights=ww)) if delta is not None else None
        name = str(row.get("site_name", row.get("name", "?")))
        if role == "opened":
            why = "Open into local demand / coverage gap"
            if share >= 0.015:
                why = "Dense local demand cluster"
            if d is not None and d <= -20:
                why = "Large local modelled speed-up"
        else:
            why = "Close where demand is thin or still covered"
            if share < 0.008:
                why = "Low local demand share"
            elif a is not None and a <= 360:
                why = "Nearby demand still within ~6 min (modelled)"
        return {
            "role": role,
            "name": name,
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "local_share": share,
            "local_before_s": b,
            "local_after_s": a,
            "local_delta_s": d,
            "why": why,
        }

    for _, r in closed.iterrows() if len(closed) else []:
        s = local_stats(r, "closed")
        if s:
            rationale.append(s)
    for _, r in opened.iterrows() if len(opened) else []:
        s = local_stats(r, "opened")
        if s:
            rationale.append(s)
    (dest / "move_rationale.json").write_text(json.dumps(rationale, indent=2) + "\n")

    viewer_summary = {
        "id": city.id,
        "name": city.name,
        "center": {"lat": city.center[1], "lon": city.center[0]},
        "bbox": {"west": city.bbox[0], "south": city.bbox[1], "east": city.bbox[2], "north": city.bbox[3]},
        "pop_millions": city.pop_millions,
        "notes": city.notes,
        "metrics": summary,
        "caveat": "Population-proxy demand + Kolesar travel times; not validated on local fire records.",
    }
    (dest / "summary.json").write_text(json.dumps(viewer_summary, indent=2) + "\n")
    # copy summary next to figures too
    shutil.copy2(dest / "summary.json", out_dir / "viewer_summary.json")


def run_one(
    city_id: str,
    *,
    force: bool = False,
    skip_graph: bool = True,
    replace: int = 10,
    solver: str = "genetic",
    solver_time: float | None = 45.0,
    cell_km: float = 1.5,
    demand_candidates: int = 40,
    synthetic_stations: bool = True,
    n_cells: int = 1800,
) -> dict:
    city = get_megacity(city_id)
    _ensure_inputs(
        city_id,
        force=force,
        skip_graph=skip_graph,
        synthetic_stations=synthetic_stations,
        n_cells=n_cells,
    )

    houses = load_fire_stations(
        custom_city(
            city_id=city.id,
            name=city.name,
            fire_stations=city.fire_stations_path,
            incidents=city.incidents_path,
            graph=city.graph_path if city.graph_path.exists() else None,
            bbox=city.bbox,
            center=city.center,
        )
    )
    incidents = load_incidents(
        custom_city(
            city_id=city.id,
            name=city.name,
            fire_stations=city.fire_stations_path,
            incidents=city.incidents_path,
        ),
        max_rows=80_000,
    )
    # Need travel_seconds for Kolesar fit
    if "travel_seconds" not in incidents.columns:
        raise RuntimeError(f"{city_id}: incidents missing travel_seconds")

    out_dir = city.fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(
        f"\n=== {city.name}: {len(houses)} stations, {len(incidents):,} proxy incidents → replace {replace} ===",
        flush=True,
    )
    result = run_fire_station_plan(
        city_id=city.id,
        houses=houses,
        incidents=incidents,
        mode="replace",
        replace_r=replace,
        cell_km=cell_km,
        demand_candidates=demand_candidates,
        spread_km=4.0,
        objective="response_time",
        threshold_min=10.0,
        cap_min=12.0,
        scorer="kolesar",
        keep_slowest=True,
        keep_average=True,
        keep_boroughs=None,
        use_busy=False,
        dispatch=False,
        refine_km=0.0,
        solver=solver,
        solver_time_s=solver_time,
        max_incident_rows=80_000,
        seed=42,
        alpha=1.0,
    )
    elapsed = time.time() - t0

    result.selected.to_csv(out_dir / "planned_fire_stations.csv", index=False)
    result.cells.to_csv(out_dir / "demand_cells.csv", index=False)
    result.candidates.to_csv(out_dir / "candidates.csv", index=False)
    if len(result.closed):
        result.closed.to_csv(out_dir / "closed_fire_stations.csv", index=False)
    if len(result.opened):
        result.opened.to_csv(out_dir / "opened_fire_stations.csv", index=False)

    summary = dict(result.summary)
    summary["city_id"] = city.id
    summary["city_name"] = city.name
    summary["n_stations"] = int(len(houses))
    summary["n_incidents"] = int(len(incidents))
    summary["elapsed_s"] = round(elapsed, 1)
    summary["scorer"] = "kolesar"
    summary["demand"] = "population_proxy"
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Metrics for aggregate table
    cells = result.cells
    row = {
        "city_id": city.id,
        "city_name": city.name,
        "n_stations": len(houses),
        "n_opened": int(len(result.opened)),
        "n_closed": int(len(result.closed)),
        "average_s_today": summary.get("average_s", {}).get("today") if isinstance(summary.get("average_s"), dict) else summary.get("average_s"),
        "average_s_plan": None,
        "delta_s": None,
        "within_6min_today": None,
        "within_6min_plan": None,
        "within_10min_today": None,
        "within_10min_plan": None,
        "pct_callweight_5s_faster": None,
        "elapsed_s": elapsed,
        "status": "ok",
    }
    # summary structure varies — pull from cells when possible
    w = cells["weight"].to_numpy(float) if "weight" in cells.columns else np.ones(len(cells))
    wn = w / w.sum()
    if "response_s_current" in cells.columns and "response_s" in cells.columns:
        before = cells["response_s_current"].to_numpy(float)
        after = cells["response_s"].to_numpy(float)
        delta = after - before
        row["average_s_today"] = float(wn @ before)
        row["average_s_plan"] = float(wn @ after)
        row["delta_s"] = float(wn @ delta)
        row["within_6min_today"] = float(wn[before <= 360].sum())
        row["within_6min_plan"] = float(wn[after <= 360].sum())
        row["within_10min_today"] = float(wn[before <= 600].sum())
        row["within_10min_plan"] = float(wn[after <= 600].sum())
        row["pct_callweight_5s_faster"] = float(wn[delta <= -5].sum() * 100)
    elif isinstance(summary.get("average_s"), dict):
        row["average_s_today"] = summary["average_s"].get("current") or summary["average_s"].get("today")
        row["average_s_plan"] = summary["average_s"].get("plan") or summary["average_s"].get("selected")
        if row["average_s_today"] is not None and row["average_s_plan"] is not None:
            row["delta_s"] = float(row["average_s_plan"]) - float(row["average_s_today"])

    _export_viewer(city_id, out_dir, {**summary, **{k: row[k] for k in row if k not in summary}})
    print(json.dumps(row, indent=2))
    return row


def write_cities_index() -> None:
    cities = []
    for c in iter_megacities():
        sp = WEB_DATA / c.id / "summary.json"
        if sp.exists():
            cities.append(json.loads(sp.read_text()))
        else:
            cities.append(
                {
                    "id": c.id,
                    "name": c.name,
                    "center": {"lat": c.center[1], "lon": c.center[0]},
                    "pop_millions": c.pop_millions,
                    "metrics": None,
                    "status": "pending",
                }
            )
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    (WEB_DATA / "cities.json").write_text(json.dumps(cities, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", choices=list_megacities(), default=None)
    ap.add_argument("--force", action="store_true", help="Rebuild station/demand inputs")
    ap.add_argument("--skip-graph", action="store_true", default=True)
    ap.add_argument("--with-graph", action="store_true", help="Also try OSM drive graph download")
    ap.add_argument("--try-osm", action="store_true", help="Try Overpass for stations (slow; may fail SSL)")
    ap.add_argument("--replace", type=int, default=10)
    ap.add_argument("--solver", choices=["exact", "genetic"], default="genetic")
    ap.add_argument("--solver-time", type=float, default=45.0)
    ap.add_argument("--n-cells", type=int, default=1800)
    args = ap.parse_args()
    skip_graph = not args.with_graph

    rows = []
    failed = []
    for c in iter_megacities([args.city] if args.city else None):
        try:
            rows.append(
                run_one(
                    c.id,
                    force=args.force,
                    skip_graph=skip_graph,
                    replace=args.replace,
                    solver=args.solver,
                    solver_time=args.solver_time,
                    synthetic_stations=not args.try_osm,
                    n_cells=args.n_cells,
                )
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"{c.id}: FAILED {exc}", file=sys.stderr)
            failed.append(c.id)
            rows.append(
                {
                    "city_id": c.id,
                    "city_name": c.name,
                    "status": f"failed: {exc}",
                }
            )

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(FIG_ROOT / "summary.csv", index=False)
    write_cities_index()
    print("Wrote", FIG_ROOT / "summary.csv")
    print("Viewer index", WEB_DATA / "cities.json")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
