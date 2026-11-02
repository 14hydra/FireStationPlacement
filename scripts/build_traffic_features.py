#!/usr/bin/env python3
"""Traffic and roadworks along each trip's route, for testing on the travel time model.

Traffic (DfT road traffic counts, Open Government Licence): each main-road segment takes the
flow of the nearest London count point within half that point's link length. Flows use 2021-2024 only, so nothing comes
from the 2025 test year. Hourly shape comes from DfT's raw counts (07:00-18:59 only).
  route_flow_k         length-weighted daily vehicles (thousands) on the counted part of the route
  route_flow_per_lane  the same, per lane
  route_hour_flow_k    vehicles per hour on the counted part at the trip's hour (blank at night)
  route_counted_share  share of the route that has a count

Roadworks (DfT Street Manager, Open Government Licence): each work sits at its nearest junction
and is active from its actual start to actual end, by day; works with no recorded dates use the
proposed ones. Counted along the route on the day of the trip:
  works_closure        road closures
  works_lanes          signals, lane closures, stop/go, give and take, contraflow, priority or convoy working
  works_any            every work touching the carriageway (the two above + carriageway incursions)

Trips are routed on the same shortest-distance tree as the existing street features.

Writes data/processed/trip_traffic_features.parquet, rows matching the 2021-2025 trips of
lfb_travel_training.csv (key: incident_number, deployed_from_station, pump_order).

Run:  PYTHONPATH=src .venv/bin/python scripts/build_traffic_features.py [--months 2025-06 2025-06]
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation import ml_scorer  # noqa: E402

DATA = ROOT / "data" / "processed" / "lfb_travel_training.csv"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "trip_traffic_features.parquet"
LANES = ["Multi-way signals", "Two-way signals", "Lane closure", "Stop/go boards", "Give and take",
         "Priority working", "Contra-flow", "Convoy workings"]
HOURS = list(range(7, 19))


def edge_flows(router) -> np.ndarray:
    """[n_edges, 2 + len(HOURS)]: length x flow columns, plus counted length, for each segment."""
    tf = Transformer.from_crs(4326, 27700, always_xy=True)
    aadf = pd.read_csv(RAW / "traffic_dft" / "london_aadf.csv.gz", usecols=["count_point_id", "year", "easting", "northing", "link_length_km", "all_motor_vehicles"])
    aadf = aadf[aadf["year"].between(2021, 2024)].groupby("count_point_id").agg(
        e=("easting", "last"), n=("northing", "last"), km=("link_length_km", "last"), flow=("all_motor_vehicles", "mean"))
    raw = pd.read_csv(RAW / "traffic_dft" / "london_raw_counts.csv.gz", usecols=["count_point_id", "direction_of_travel", "count_date", "hour", "all_motor_vehicles"])
    hourly = raw.groupby(["count_point_id", "count_date", "hour"])["all_motor_vehicles"].sum().groupby(["count_point_id", "hour"]).mean().unstack("hour")
    # Scale each point's hourly shape to its 2021-2024 flow, so old surveys only supply the shape.
    share = hourly.reindex(columns=HOURS).div(hourly.reindex(columns=HOURS).sum(axis=1), axis=0)
    frac12 = 0.75  # DfT: about three quarters of daily traffic falls in the 12 surveyed hours
    hour_flow = share.reindex(aadf.index).mul(aadf["flow"] * frac12, axis=0)
    hour_flow = hour_flow.apply(lambda c: c.fillna(aadf["flow"] * frac12 / 12))  # no survey: flat shape

    e = router.edge_id.tocoo()
    order = np.argsort(e.data)
    u, v = e.row[order], e.col[order]
    x1, y1 = tf.transform(router.node_lon[u], router.node_lat[u])
    x2, y2 = tf.transform(router.node_lon[v], router.node_lat[v])
    mid = np.c_[(np.asarray(x1) + np.asarray(x2)) / 2, (np.asarray(y1) + np.asarray(y2)) / 2]
    dist, nn = cKDTree(aadf[["e", "n"]].to_numpy()).query(mid, distance_upper_bound=2000)
    # A count stands for its whole junction-to-junction link, so reach half the link's length.
    reach = np.r_[np.maximum(150, aadf["km"].fillna(0).to_numpy() * 500 + 50), 0]
    dist = np.where(dist <= reach[nn], dist, np.inf)
    vals = router.edge_vals
    main = (vals[["len_major", "len_primary", "len_secondary"]].sum(axis=1) > 0).to_numpy()
    ok = np.isfinite(dist) & main
    L = vals["len_total"].to_numpy()
    lanes = (vals["lanes_x_len"] / vals["len_lanes_tagged"].replace(0, np.nan)).fillna(2).clip(lower=1).to_numpy()
    Z = np.zeros((len(L), 3 + len(HOURS)))
    idx = nn[ok]
    Z[ok, 0] = L[ok]
    Z[ok, 1] = L[ok] * aadf["flow"].to_numpy()[idx]
    Z[ok, 2] = L[ok] * aadf["flow"].to_numpy()[idx] / lanes[ok]
    Z[ok, 3:] = L[ok, None] * hour_flow.to_numpy()[idx]
    print(f"counted segments: {ok.sum():,} of {len(L):,}; count points {len(aadf):,}", flush=True)
    return Z


def load_works(router, months: list[str]) -> pd.DataFrame:
    tf = Transformer.from_crs(27700, 4326, always_xy=True)
    files = [f for f in sorted((RAW / "roadworks").glob("*.csv.gz")) if not months or months[0] <= f.name[:7] <= months[1]]
    cols = ["work_reference_number", "easting", "northing", "traffic_management_type", "proposed_start_date", "proposed_end_date", "actual_start", "actual_end"]
    w = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols) for f in files], ignore_index=True)
    # A work can appear in several months: first non-blank per column gives start, last gives end.
    w = w.groupby("work_reference_number", sort=False).agg(
        easting=("easting", "last"), northing=("northing", "last"), tm=("traffic_management_type", "last"),
        ps=("proposed_start_date", "first"), pe=("proposed_end_date", "last"), a_s=("actual_start", "first"), a_e=("actual_end", "last"))
    d = lambda c: pd.to_datetime(c, errors="coerce", utc=True).dt.tz_localize(None).dt.floor("D")  # noqa: E731
    w["start"] = d(w["a_s"]).fillna(d(w["ps"]))
    w["end"] = d(w["a_e"]).fillna(d(w["pe"]))
    w = w.dropna(subset=["start", "end"])
    w = w[(w["end"] >= w["start"]) & ((w["end"] - w["start"]).dt.days <= 400)]
    w["kind"] = np.select([w["tm"].eq("Road closure"), w["tm"].isin(LANES), w["tm"].eq("Some carriageway incursion")], [0, 1, 2], -1)
    w = w[w["kind"] >= 0]
    lon, lat = tf.transform(w["easting"].to_numpy(), w["northing"].to_numpy())
    w["node"], off = router.snap(np.asarray(lat), np.asarray(lon))
    w = w[np.asarray(off) <= 100]
    print(f"works on the carriageway near the network: {len(w):,}", w["kind"].value_counts().to_dict(), flush=True)
    return w[["node", "kind", "start", "end"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", nargs=2, help="Only these months of trips and roadworks (pilot), e.g. 2025-06 2025-06")
    args = ap.parse_args()
    t0 = time.time()
    use = ["incident_number", "cal_year", "date_of_call", "hour_of_call", "deployed_from_station", "pump_order", "start_lat", "start_lon", "dest_lat", "dest_lon"]
    trips = pd.read_csv(DATA, usecols=use, low_memory=False)
    trips = trips[trips["cal_year"].between(2021, 2025)].reset_index(drop=True)
    trips["day"] = pd.to_datetime(trips["date_of_call"], errors="coerce", format="mixed", dayfirst=True).dt.floor("D")
    if args.months:
        p = trips["day"].dt.to_period("M").astype(str)
        trips = trips[p.between(*args.months)].reset_index(drop=True)
    print(f"{len(trips):,} trips, {trips['day'].min()} to {trips['day'].max()}", flush=True)

    router = ml_scorer._router()
    Z = edge_flows(router)
    works = load_works(router, args.months or [])
    s_idx, _ = router.snap(trips["start_lat"].to_numpy(), trips["start_lon"].to_numpy())
    d_idx, _ = router.snap(trips["dest_lat"].to_numpy(), trips["dest_lon"].to_numpy())
    n = router.csr.shape[0]
    nodes = np.arange(n)
    days = trips["day"].to_numpy()
    hours = pd.to_numeric(trips["hour_of_call"], errors="coerce").fillna(12).astype(int).to_numpy()
    L = router.edge_vals["len_total"].to_numpy()
    w_node, w_kind = works["node"].to_numpy(), works["kind"].to_numpy()
    w_s, w_e = works["start"].to_numpy(), works["end"].to_numpy()
    res = np.full((len(trips), 8), np.nan, dtype=np.float32)  # counted, flow, flow/lane, hour flow, total, 3 works
    # Junctions on each distinct route, stored once; works are then counted day by day.
    pair, pair_id = np.unique(s_idx.astype(np.int64) * n + d_idx, return_inverse=True)
    starts, lens, chunks, pos = np.zeros(len(pair), dtype=np.int64), np.zeros(len(pair), dtype=np.int32), [], 0
    for k, o in enumerate(np.unique(s_idx)):
        _, pred = dijkstra(router.csr, directed=True, indices=o, return_predecessors=True)
        has = pred >= 0
        eid = np.zeros(n, dtype=int)
        eid[has] = np.asarray(router.edge_id[pred[has], nodes[has]]).ravel()
        vals = np.zeros((n, Z.shape[1] + 1))
        vals[has, :-1] = Z[eid[has] - 1]
        vals[has, -1] = L[eid[has] - 1]
        S = router._ancestor_sums(pred, vals)
        m = np.flatnonzero(s_idx == o)
        res[m, 0:3] = S[d_idx[m], 0:3]
        res[m, 4] = S[d_idx[m], -1]
        in_hours = (hours[m] >= 7) & (hours[m] <= 18)
        res[m[in_hours], 3] = S[d_idx[m[in_hours]], 3 + hours[m[in_hours]] - 7]
        for pid in np.flatnonzero(pair // n == o):
            path, x = [], int(pair[pid] % n)
            if x != o and pred[x] < 0:
                continue  # unreachable
            while x != o:
                path.append(x)
                x = int(pred[x])
            path.append(o)
            starts[pid], lens[pid] = pos, len(path)
            chunks.append(np.array(path, dtype=np.int32))
            pos += len(path)
        del S, vals
        if (k + 1) % 10 == 0:
            print(f"  station {k + 1}: {time.time() - t0:.0f} s", flush=True)
            gc.collect()
    paths = np.concatenate(chunks)
    del chunks
    print(f"routes: {len(pair):,}, junctions stored {len(paths):,}, {time.time() - t0:.0f} s", flush=True)
    res[:, 5:] = 0
    for day in pd.unique(days):
        if pd.isna(day):
            continue
        mm = np.flatnonzero((days == day) & (lens[pair_id] > 0))
        if not len(mm):
            continue
        active = (w_s <= day) & (w_e >= day)
        pid = pair_id[mm]
        ls = lens[pid]
        idx = np.repeat(starts[pid] - np.r_[0, np.cumsum(ls)[:-1]], ls) + np.arange(ls.sum())
        offs = np.r_[0, np.cumsum(ls)[:-1]]
        on_path = paths[idx]
        for kind in range(3):
            count = np.bincount(w_node[active & (w_kind == kind)], minlength=n)
            res[mm, 5 + kind] = np.add.reduceat(count[on_path], offs)
    counted = res[:, 0]
    out = trips[["incident_number", "deployed_from_station", "pump_order"]].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["route_flow_k"] = res[:, 1] / counted / 1000
        out["route_flow_per_lane"] = res[:, 2] / counted / 1000
        out["route_hour_flow_k"] = res[:, 3] / counted / 1000
        out["route_counted_share"] = counted / res[:, 4]
    out["works_closure"] = res[:, 5]
    out["works_lanes"] = res[:, 6]
    out["works_any"] = res[:, 5] + res[:, 6] + res[:, 7]
    dest = OUT if not args.months else OUT.with_name(f"trip_traffic_features_{args.months[0]}.parquet")
    out.to_parquet(dest)
    print(out.describe().T.round(3).to_string())
    print(f"wrote {dest.name}; {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
