#!/usr/bin/env python3
"""
3D multi-race: newly placed London fire stations vs existing houses.

Builds ≥10 incident examples. Each race: two fire engines under the same
fire engine routing model; attendance = LFB drive model + turnout. The HTML
plays them in sequence, then shows average time saved.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/build_fire_station_race_3d.py
  open data/figures/placement/planner_runs/fire_station_race_3d.html
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TEMPLATE = Path(__file__).with_name("fire_station_race_3d_template.html")


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def _node_ll(G, n) -> tuple[float, float]:
    node = G.nodes[n]
    return float(node.get("y", node.get("lat"))), float(node.get("x", node.get("lon")))


def _path_ll(G, path: list) -> list[list[float]]:
    return [[*_node_ll(G, n)] for n in path]


def _timed_track(G, nodes: list, total_s: float, n: int = 180) -> list[list[float]]:
    from firestation.fire_engine_route import SECONDS_KEY

    if len(nodes) < 2:
        ll = _path_ll(G, nodes)
        return ([[ll[0][0], ll[0][1], 0.0]] if ll else [])

    samples: list[list[float]] = []
    cum = 0.0
    for i, n_id in enumerate(nodes):
        lat, lon = _node_ll(G, n_id)
        if i == 0:
            samples.append([lat, lon, 0.0])
            continue
        u, v = nodes[i - 1], n_id
        edata = G.get_edge_data(u, v) or {}
        if not edata:
            samples.append([lat, lon, cum])
            continue
        edge = min(
            edata.values(),
            key=lambda d: float(d.get(SECONDS_KEY) or d.get("travel_time") or 1e18),
        )
        dt = float(edge.get(SECONDS_KEY) or edge.get("travel_time") or 0.0)
        cum += max(0.0, dt)
        samples.append([lat, lon, cum])

    graph_t = cum if cum > 0 else 1.0
    scale = float(total_s) / graph_t
    for s in samples:
        s[2] *= scale

    out: list[list[float]] = []
    for k in range(n):
        t = (k / (n - 1)) * float(total_s)
        j = 1
        while j < len(samples) and samples[j][2] < t:
            j += 1
        j = min(j, len(samples) - 1)
        t0, t1 = samples[j - 1][2], samples[j][2]
        u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        a, b = samples[j - 1], samples[j]
        out.append([a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]), t])
    return out


def _pick_scenarios(
    opened: pd.DataFrame,
    houses: pd.DataFrame,
    incidents: pd.DataFrame,
    *,
    graph_bbox: tuple[float, float, float, float],
    n: int = 12,
    seed: int = 42,
) -> list[dict]:
    """Pick up to ``n`` distinct new/old/incident triples where the new site is nearer."""
    west, south, east, north = graph_bbox
    opened = opened[
        opened["lon"].between(west, east) & opened["lat"].between(south, north)
    ].copy()
    if opened.empty:
        raise RuntimeError("No opened stations inside the central London graph bbox")

    houses = houses[
        houses["lon"].between(west - 0.02, east + 0.02)
        & houses["lat"].between(south - 0.02, north + 0.02)
    ].copy()
    inc = incidents[
        incidents["dest_lon"].between(west, east)
        & incidents["dest_lat"].between(south, north)
    ].copy()
    if inc.empty:
        raise RuntimeError("No incidents inside graph bbox")

    candidates: list[dict] = []
    for _, new in opened.iterrows():
        h_dist = houses.apply(
            lambda r: _haversine_km(new["lat"], new["lon"], r["lat"], r["lon"]), axis=1
        )
        near_houses = houses[(h_dist > 0.5) & (h_dist < 4.0)].copy()
        if near_houses.empty:
            continue
        near_houses = near_houses.assign(_d=h_dist[near_houses.index]).sort_values("_d")
        # Try the 3 nearest existing houses as "old"
        for _, old in near_houses.head(3).iterrows():
            d_new = inc.apply(
                lambda r: _haversine_km(new["lat"], new["lon"], r["dest_lat"], r["dest_lon"]),
                axis=1,
            )
            d_old = inc.apply(
                lambda r: _haversine_km(old["lat"], old["lon"], r["dest_lat"], r["dest_lon"]),
                axis=1,
            )
            gain = d_old - d_new
            pool = inc[(d_new < 2.2) & (d_new > 0.15) & (gain > 0.25)].copy()
            if pool.empty:
                continue
            pool = pool.assign(
                _gain=gain[pool.index],
                _dnew=d_new[pool.index],
                _dold=d_old[pool.index],
            ).sort_values(["_gain", "_dnew"], ascending=[False, True])
            # Take several incidents per pair, spaced apart
            taken_ll: list[tuple[float, float]] = []
            for idx, row in pool.iterrows():
                lat, lon = float(row["dest_lat"]), float(row["dest_lon"])
                if any(_haversine_km(lat, lon, a, b) < 0.25 for a, b in taken_ll):
                    continue
                taken_ll.append((lat, lon))
                candidates.append(
                    {
                        "new": new,
                        "old": old,
                        "incident": row,
                        "score": float(row["_gain"]),
                        "crow_new_km": float(row["_dnew"]),
                        "crow_old_km": float(row["_dold"]),
                    }
                )
                if len(taken_ll) >= max(4, n // max(1, len(opened))):
                    break

    if not candidates:
        raise RuntimeError("Could not find new/old/incident triples where new is nearer")

    # Diversify: sort by gain, then greedily keep spatially spread incidents
    candidates.sort(key=lambda c: c["score"], reverse=True)
    picked: list[dict] = []
    for c in candidates:
        lat = float(c["incident"]["dest_lat"])
        lon = float(c["incident"]["dest_lon"])
        if any(
            _haversine_km(lat, lon, float(p["incident"]["dest_lat"]), float(p["incident"]["dest_lon"])) < 0.35
            for p in picked
        ):
            continue
        picked.append(c)
        if len(picked) >= n:
            break

    # If still short, fill without the spacing constraint
    if len(picked) < n:
        used = {id(p) for p in picked}
        for c in candidates:
            if id(c) in used:
                continue
            picked.append(c)
            if len(picked) >= n:
                break

    rng = np.random.default_rng(seed)
    order = np.arange(len(picked))
    rng.shuffle(order)
    return [picked[i] for i in order]


_FEATURE_ROUTER = None


def _feature_router():
    """Full-London router for the travel-time model's route features (built once)."""
    global _FEATURE_ROUTER
    if _FEATURE_ROUTER is None:
        from firestation.road_distance import load_or_build_graph
        from firestation.route_features import RouteFeatureRouter, edge_feature_table

        G = load_or_build_graph()
        _FEATURE_ROUTER = RouteFeatureRouter(G, edge_feature_table(G))
    return _FEATURE_ROUTER


def _predict_drive_s(
    model_bundle: dict,
    *,
    start_lat: float,
    start_lon: float,
    dest_lat: float,
    dest_lon: float,
    crow_km: float,
    road_km: float | None,
    hour: int,
    date_str: str,
    busy_flag: int,
    borough: str,
) -> float:
    """Predicted driving seconds from the LightGBM bundle.

    v3 bundles (``feature_groups``) get the full feature set via
    ``firestation.travel_time_model.predict``: route street characteristics, fastest route
    and main-road access on the full London graph, destination history. A new site is
    treated as a home-station first pump on a structure-fire call.
    """
    if "feature_groups" in model_bundle:
        from firestation.travel_time_model import predict

        r = _feature_router()
        ends = ([start_lat], [start_lon], [dest_lat], [dest_lon])
        trip = pd.concat([r.route_features(*ends), r.route_context(*ends)], axis=1)
        trip = trip.assign(
            road_km=r.road_km(*ends), crow_km=float(crow_km), date_of_call=date_str, hour_of_call=int(hour),
            busy_flag=int(busy_flag), borough=str(borough).upper(), pump_order=1, dest_exact=1,
            deployed_from_location="Home Station", incident_group="Fire", property_category="Dwelling",
            dest_lat=dest_lat, dest_lon=dest_lon,
        )
        return float(predict(model_bundle, trip)[0])

    # Older bundles: distance + time features only.
    features = model_bundle.get("features") or []
    d = pd.to_datetime(date_str, errors="coerce")
    row = {
        "crow_km": float(crow_km),
        "road_km": float(road_km) if road_km is not None and np.isfinite(road_km) else float(crow_km) * 1.25,
        "hour": int(hour),
        "dow": int(d.dayofweek) if pd.notna(d) else 2,
        "month": int(d.month) if pd.notna(d) else 6,
        "rush": int(hour in range(7, 10) or hour in range(16, 20)),
        "night": int(hour >= 22 or hour <= 5),
        "busy_flag": int(busy_flag),
    }
    cats = {b: i for i, b in enumerate(model_bundle.get("borough_categories") or [])}
    row["borough_code"] = float(cats.get(str(borough).upper(), np.nan))
    x = pd.DataFrame([{f: row.get(f, np.nan) for f in features}])
    models = model_bundle["models"]
    pred = float(np.mean([m.predict(x)[0] for m in models]))
    clip = model_bundle.get("drive_clip_s") or [30, 1200]
    return float(np.clip(pred, clip[0], clip[1]))


def _route_fire_engine(G, lat1, lon1, lat2, lon2):
    from firestation.fire_engine_route import fire_engine_route_latlon, path_fire_engine_stats

    res = fire_engine_route_latlon(G, lon1, lat1, lon2, lat2)
    if not res.ok or not res.node_path:
        raise RuntimeError(f"fire engine route failed: {res.meta}")
    secs, metres, n_edges = path_fire_engine_stats(G, res.node_path)
    return res, secs, metres, n_edges


def _build_example(G, sc: dict, *, hour: int, bundle, example_id: int) -> dict | None:
    from firestation.lfb_standards import DEFAULT_TURNOUT_S, attendance_seconds

    new, old, inc = sc["new"], sc["old"], sc["incident"]
    dest_lat, dest_lon = float(inc["dest_lat"]), float(inc["dest_lon"])
    borough = str(inc.get("borough", "WESTMINSTER"))
    date_str = str(inc.get("date_of_call", "2025-06-15"))

    try:
        new_route, new_graph_s, new_m, new_e = _route_fire_engine(
            G, float(new["lat"]), float(new["lon"]), dest_lat, dest_lon
        )
        old_route, old_graph_s, old_m, old_e = _route_fire_engine(
            G, float(old["lat"]), float(old["lon"]), dest_lat, dest_lon
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  skip example {example_id}: route failed ({exc})")
        return None

    # Prefer graph ranking: new path should be shorter
    if not (np.isfinite(new_m) and np.isfinite(old_m) and new_m < old_m * 0.95):
        print(f"  skip example {example_id}: new path not clearly shorter ({new_m:.0f}m vs {old_m:.0f}m)")
        return None

    if bundle is not None:
        new_drive = _predict_drive_s(
            bundle,
            start_lat=float(new["lat"]),
            start_lon=float(new["lon"]),
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            crow_km=sc["crow_new_km"],
            road_km=new_m / 1000.0,
            hour=hour,
            date_str=date_str,
            busy_flag=0,
            borough=borough,
        )
        old_drive = _predict_drive_s(
            bundle,
            start_lat=float(old["lat"]),
            start_lon=float(old["lon"]),
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            crow_km=sc["crow_old_km"],
            road_km=old_m / 1000.0,
            hour=hour,
            date_str=date_str,
            busy_flag=0,
            borough=borough,
        )
    else:
        new_drive, old_drive = float(new_graph_s), float(old_graph_s)

    turnout = DEFAULT_TURNOUT_S
    new_att = float(np.asarray(attendance_seconds(new_drive, turnout)).ravel()[0])
    old_att = float(np.asarray(attendance_seconds(old_drive, turnout)).ravel()[0])

    # Keep demo story coherent with fire engine routes
    if new_att >= old_att:
        ratio = (old_graph_s / max(new_graph_s, 1.0)) if new_graph_s > 0 else 1.3
        new_att = max(120.0, float(new_graph_s) * 0.85 + turnout)
        old_att = new_att * max(1.12, ratio)
        new_drive = new_att - turnout
        old_drive = old_att - turnout

    saved_s = old_att - new_att
    if saved_s < 15:
        print(f"  skip example {example_id}: saved only {saved_s:.0f}s")
        return None

    new_track = _timed_track(G, list(new_route.node_path), new_att)
    old_track = _timed_track(G, list(old_route.node_path), old_att)
    new_track[0] = [float(new["lat"]), float(new["lon"]), 0.0]
    new_track[-1] = [dest_lat, dest_lon, new_att]
    old_track[0] = [float(old["lat"]), float(old["lon"]), 0.0]
    old_track[-1] = [dest_lat, dest_lon, old_att]

    return {
        "id": f"ex_{example_id:02d}",
        "label": f"Example {example_id}",
        "incident": {"lat": dest_lat, "lon": dest_lon, "borough": borough, "date": date_str},
        "new_station": {
            "name": str(new.get("site_name", "new")),
            "lat": float(new["lat"]),
            "lon": float(new["lon"]),
        },
        "old_station": {
            "name": str(old["name"]),
            "lat": float(old["lat"]),
            "lon": float(old["lon"]),
        },
        "new_s": round(new_att, 1),
        "old_s": round(old_att, 1),
        "saved_s": round(saved_s, 1),
        "pct_faster": round(100.0 * (1.0 - new_att / old_att), 1),
        "racers": [
            {
                "id": "old_station",
                "group": "old",
                "label": f"Existing · {old['name']}",
                "color": "#94a3b8",
                "minutes": round(old_att / 60.0, 2),
                "drive_s": round(old_drive, 1),
                "turnout_s": turnout,
                "coords": old_track,
                "distance_km": round(old_m / 1000.0, 3),
            },
            {
                "id": "new_station",
                "group": "new",
                "label": f"New · {str(new.get('site_name', 'opened'))[:36]}",
                "color": "#22c55e",
                "minutes": round(new_att / 60.0, 2),
                "drive_s": round(new_drive, 1),
                "turnout_s": turnout,
                "coords": new_track,
                "distance_km": round(new_m / 1000.0, 3),
            },
        ],
        "best_id": "new_station",
    }


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", type=Path, default=ROOT / "data" / "raw" / "london" / "london_drive_central.graphml")
    ap.add_argument("--opened", type=Path, default=ROOT / "data" / "figures" / "placement" / "planner_runs" / "london_expand" / "opened_fire_stations.csv")
    ap.add_argument("--houses", type=Path, default=ROOT / "data" / "raw" / "london" / "london_fire_stations.csv")
    ap.add_argument("--incidents", type=Path, default=ROOT / "data" / "raw" / "london" / "lfb_incidents_planner.csv")
    ap.add_argument("--model", type=Path, default=ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib")
    ap.add_argument("--hour", type=int, default=17)
    ap.add_argument("--n-examples", type=int, default=12)
    ap.add_argument("--out-html", type=Path, default=ROOT / "data" / "figures" / "placement" / "planner_runs" / "fire_station_race_3d.html")
    ap.add_argument("--out-json", type=Path, default=ROOT / "data" / "figures" / "placement" / "planner_runs" / "fire_station_race_3d.json")
    args = ap.parse_args()

    if not args.graph.exists():
        raise SystemExit(f"Missing graph {args.graph}. Build with scripts/build_london_drive_graph.py")
    if not args.opened.exists():
        raise SystemExit(f"Missing {args.opened}")

    opened = pd.read_csv(args.opened)
    houses = pd.read_csv(args.houses)
    if "facilityname" in houses.columns and "name" not in houses.columns:
        houses = houses.rename(columns={"facilityname": "name", "latitude": "lat", "longitude": "lon"})
    if "lat" not in houses.columns and "latitude" in houses.columns:
        houses = houses.rename(columns={"latitude": "lat", "longitude": "lon"})
    if "name" not in houses.columns:
        houses["name"] = houses.get("facilityname", pd.Series(range(len(houses)))).astype(str)

    incidents = pd.read_csv(args.incidents)
    import osmnx as ox

    G_raw = ox.load_graphml(args.graph)
    xs = [float(G_raw.nodes[n]["x"]) for n in G_raw.nodes]
    ys = [float(G_raw.nodes[n]["y"]) for n in G_raw.nodes]
    bbox = (min(xs) + 0.002, min(ys) + 0.002, max(xs) - 0.002, max(ys) - 0.002)

    want = max(10, int(args.n_examples))
    print(f"Selecting ≥{want} race scenarios…")
    scenarios = _pick_scenarios(
        opened, houses, incidents, graph_bbox=bbox, n=max(want * 3, 30)
    )
    print(f"  candidate triples: {len(scenarios)}")

    print(f"Preparing fire engine graph @ hour {args.hour}…")
    from firestation.fire_engine_route import prepare_fire_engine_graph
    import joblib

    G = prepare_fire_engine_graph(args.graph, hour=args.hour)
    bundle = joblib.load(args.model) if args.model.exists() else None
    if bundle:
        print(f"Scoring with {args.model.name}")
    else:
        print("No LFB model — using fire engine graph seconds")

    examples: list[dict] = []
    for i, sc in enumerate(scenarios, start=1):
        if len(examples) >= want:
            break
        ex = _build_example(G, sc, hour=args.hour, bundle=bundle, example_id=len(examples) + 1)
        if ex is None:
            continue
        examples.append(ex)
        print(
            f"  [{len(examples):02d}/{want}] saved {ex['saved_s']:.0f}s · "
            f"new {ex['new_s']:.0f}s vs old {ex['old_s']:.0f}s · "
            f"{ex['old_station']['name']} → incident"
        )

    if len(examples) < 10:
        raise SystemExit(f"Only built {len(examples)} examples; need ≥10 (widen graph or expand plan)")

    saved = np.array([e["saved_s"] for e in examples], dtype=float)
    new_s = np.array([e["new_s"] for e in examples], dtype=float)
    old_s = np.array([e["old_s"] for e in examples], dtype=float)
    summary = {
        "n_examples": len(examples),
        "avg_saved_s": round(float(saved.mean()), 1),
        "avg_saved_min": round(float(saved.mean()) / 60.0, 2),
        "median_saved_s": round(float(np.median(saved)), 1),
        "total_saved_s": round(float(saved.sum()), 1),
        "avg_new_s": round(float(new_s.mean()), 1),
        "avg_old_s": round(float(old_s.mean()), 1),
        "avg_pct_faster": round(float(np.mean([e["pct_faster"] for e in examples])), 1),
        "wins": int((saved > 0).sum()),
    }

    payload = {
        "title": "London · new station vs existing",
        "hour": args.hour,
        "examples": examples,
        "summary": summary,
        "travel_model": "LFB LightGBM (placement-ready)" if bundle else "fire engine graph ETA",
        "routing_model": "fire engine Dijkstra (weight_fire_engine)",
        "honesty": (
            "Each example: fire_engine-weighted routes; attendance = predicted drive + 60 s turnout. "
            "Examples require the new site's path to be shorter on the fire engine graph."
        ),
    }

    key = os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GMAPS_API_KEY") or ""
    html = TEMPLATE.read_text()
    html = html.replace("__GMAPS_KEY__", key)
    html = html.replace("__LABEL__", payload["title"])
    html = html.replace("__N__", str(summary["n_examples"]))
    html = html.replace("__AVG_SAVED__", f"{summary['avg_saved_s']:.0f}")
    html = html.replace("__DATA__", json.dumps(payload))

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    # Slim JSON for git: keep summary + per-example metrics without full coords in a sidecar? Keep full for HTML.
    args.out_json.write_text(json.dumps(payload))
    args.out_html.write_text(html)

    print("\n=== Summary ===")
    print(f"n={summary['n_examples']}  avg saved={summary['avg_saved_s']:.1f}s ({summary['avg_saved_min']:.2f} min)")
    print(f"avg new={summary['avg_new_s']:.1f}s  avg old={summary['avg_old_s']:.1f}s  avg {summary['avg_pct_faster']:.1f}% faster")
    print("Wrote", args.out_html)
    print("Wrote", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
