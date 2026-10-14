#!/usr/bin/env python3
"""
How busy is each London fire station? Measured from LFB mobilisation records.

For every engine trip in the year: the engine is committed from mobilisation until it
leaves the incident scene, plus the drive back (LFB leaves DateAndTimeReturned empty,
so the return drive is taken as equal to the recorded outbound TravelTimeSeconds).
Each engine's busy periods are merged (so back-to-back jobs are not double counted).

Per station, using its own engines (the pump codes it sends from home), wherever they
were sent from:
  utilisation_pct      share of the station's engine time spent committed to calls
  all_out_pct          share of the year with every one of its engines out
  covered_by_other_pct share of first-engine calls in its own station area answered
                       by an engine from another station (the effect of being busy)
  calls_per_day, engine_hours_per_day, median_job_min

Writes data/figures/station_busyness/:
  station_busyness.png   one-page poster: map of London (size = calls/day, colour =
                         utilisation), busiest / quietest stations, busy-by-hour heatmap
  station_busyness_ranked.png  all stations ranked (utilisation, every engine out,
                         coverage by neighbours)
  station_busyness_map.html  interactive map, circle size = calls/day, colour = utilisation
  station_busyness.csv

Run:  .venv/bin/python scripts/plot_station_busyness.py [--year 2025]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "london"
OUT = ROOT / "data" / "figures" / "station_busyness"
MOB_FILES = {2021: "lfb_mobilisations_2021_2024.csv", 2022: "lfb_mobilisations_2021_2024.csv",
             2023: "lfb_mobilisations_2021_2024.csv", 2024: "lfb_mobilisations_2021_2024.csv",
             2025: "lfb_mobilisations_2025.csv", 2026: "lfb_mobilisations_2025.csv"}
INC_FILES = ["lfb_incidents_2018_2023.csv", "lfb_incidents_2024_onwards.csv"]

# Reference palette (dataviz skill): sequential blue ramp; text inks; surface.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE = "#eb6834"
# Sequential orange ramp for "how busy" (one hue, light -> dark).
HOT = ["#fde4d6", "#fbc4a6", "#f7a072", "#eb6834", "#c94f1f", "#9c3a14", "#6b270c"]
BASEMAP = ROOT / "data" / "processed" / "london_basemap.pkl"


def _time(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    return t.fillna(pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce"))


def load_trips(year: int) -> pd.DataFrame:
    cols = ["IncidentNumber", "CalYear", "Resource_Code", "DateAndTimeMobilised", "DateAndTimeLeft",
            "TravelTimeSeconds", "DeployedFromStation_Name", "DeployedFromLocation", "PumpOrder"]
    m = pd.read_csv(RAW / MOB_FILES[year], usecols=cols, encoding="utf-8-sig", low_memory=False,
                    dtype={"IncidentNumber": str})
    m = m[m["CalYear"] == year].copy()
    m["start"] = _time(m["DateAndTimeMobilised"])
    left = _time(m["DateAndTimeLeft"])
    back = pd.to_timedelta(pd.to_numeric(m["TravelTimeSeconds"], errors="coerce").fillna(0), unit="s")
    m["end"] = left + back
    m = m.dropna(subset=["start", "end"])
    dur = (m["end"] - m["start"]).dt.total_seconds()
    # Drop impossible records (negative or > 24 h committed).
    return m[(dur > 0) & (dur < 24 * 3600)].assign(job_s=dur)


def _merge(intervals: np.ndarray) -> np.ndarray:
    """Union of [start, end) intervals (seconds), sorted."""
    iv = intervals[np.argsort(intervals[:, 0])]
    out = [iv[0].copy()]
    for a, b in iv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append(np.array([a, b]))
    return np.array(out)


def _seconds(t: pd.Series) -> np.ndarray:
    return (t - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()


def all_out_seconds(merged: list[np.ndarray]) -> float:
    """Seconds in which every engine's merged busy periods overlap (sweep line)."""
    ev = np.concatenate([np.r_[m[:, 0], m[:, 1]] for m in merged])
    step = np.concatenate([np.r_[np.ones(len(m)), -np.ones(len(m))] for m in merged])
    order = np.lexsort((step, ev))  # at equal times, process ends (-1) first
    ev, level = ev[order], np.cumsum(step[order])
    return float(np.diff(ev)[level[:-1] >= len(merged)].sum())


def station_table(year: int, stations: pd.DataFrame) -> pd.DataFrame:
    trips = load_trips(year)
    home = trips[(trips["DeployedFromLocation"] == "Home Station") & trips["DeployedFromStation_Name"].isin(stations.index)]
    fleet = home.groupby("DeployedFromStation_Name")["Resource_Code"].unique()
    y0 = pd.Timestamp(f"{year}-01-01")
    year_s = (min(trips["end"].max(), pd.Timestamp(f"{year + 1}-01-01")) - y0).total_seconds()
    days = year_s / 86400

    rows, hourly = [], {}
    for st, codes in fleet.items():
        g = trips[trips["Resource_Code"].isin(codes)]
        merged = [_merge(np.c_[_seconds(r["start"]), _seconds(r["end"])]) for _, r in g.groupby("Resource_Code")]
        busy_s = sum(float((m[:, 1] - m[:, 0]).sum()) for m in merged)
        hourly[st] = _hourly_share(merged, len(codes), days)
        rows.append({
            "station": st, "engines": len(codes),
            "calls_per_day": len(g) / days,
            "engine_hours_per_day": busy_s / 3600 / days,
            "median_job_min": g["job_s"].median() / 60,
            "utilisation_pct": 100 * busy_s / (len(codes) * year_s),
            "all_out_pct": 100 * all_out_seconds(merged) / year_s,
        })
    t = pd.DataFrame(rows).set_index("station")

    inc = pd.concat([pd.read_csv(RAW / f, usecols=["IncidentNumber", "CalYear", "IncidentStationGround",
                                                   "FirstPumpArriving_DeployedFromStation"], low_memory=False)
                     for f in INC_FILES])
    inc = inc[(inc["CalYear"] == year)].dropna(subset=["IncidentStationGround", "FirstPumpArriving_DeployedFromStation"])
    other = (inc["IncidentStationGround"] != inc["FirstPumpArriving_DeployedFromStation"])
    cov = other.groupby(inc["IncidentStationGround"]).mean().mul(100).rename("covered_by_other_pct")
    t = t.join(cov).join(stations).sort_values("utilisation_pct", ascending=False)
    t.attrs["hourly"] = pd.DataFrame(hourly).T.reindex(t.index)
    return t


def _hourly_share(merged: list[np.ndarray], n_engines: int, days: float) -> np.ndarray:
    """% of engine time out on calls in each hour of the day (0-23), from busy periods in seconds."""
    minutes = np.zeros(24)
    for m in merged:
        a = np.floor(m[:, 0] / 60).astype(np.int64)
        b = np.ceil(m[:, 1] / 60).astype(np.int64)
        lens = b - a
        idx = np.repeat(a, lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
        minutes += np.bincount((idx // 60) % 24, minlength=24)
    return 100 * minutes / (n_engines * days * 60)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "font.family": "sans-serif",
    })


def seq_color(v, vmax):
    i = int(np.clip(v / vmax, 0, 0.999) * len(SEQ))
    return SEQ[i]


def plot_ranked(t: pd.DataFrame, year: int, path: Path):
    style()
    t = t.sort_values("utilisation_pct")
    n = len(t)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 0.19 * n + 2.2), sharey=True, layout="constrained",
                                     gridspec_kw={"width_ratios": [3, 2, 2]})
    y = np.arange(n)
    vmax = t["utilisation_pct"].max()
    a1.barh(y, t["utilisation_pct"], color=[seq_color(v, vmax) for v in t["utilisation_pct"]], height=0.72)
    med = t["utilisation_pct"].median()
    a1.axvline(med, color=INK2, lw=0.9, ls="--")
    a1.text(med, n + 0.4, f"median {med:.0f}%", fontsize=8, color=INK2, ha="center")
    for i, v in enumerate(t["utilisation_pct"]):
        a1.text(v + 0.2, i, f"{v:.0f}%", va="center", fontsize=6.5, color=INK2)
    a1.set_yticks(y, [f"{st} ({e})" for st, e in zip(t.index, t["engines"])], fontsize=7)
    a1.set_xlim(0, vmax * 1.12)
    a1.set_xlabel("% of engine time out on calls (utilisation)")
    a1.set_title("How busy is each station's fleet")
    a1.set_ylim(-0.8, n + 1)
    a1.grid(axis="y", visible=False)

    a2.barh(y, t["all_out_pct"], color=ORANGE, height=0.72)
    a2.set_xlabel("% of the year with every engine out")
    a2.set_title("Every engine out")
    a2.grid(axis="y", visible=False)

    a3.barh(y, t["covered_by_other_pct"], color="#9a9893", height=0.72)
    a3.set_xlabel("% of own-area calls where another\nstation's engine arrived first")
    a3.set_title("Area covered by neighbours")
    a3.grid(axis="y", visible=False)

    fig.suptitle(f"London fire stations ranked by how busy their engines are, {year}", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.supxlabel("(n) = engines at the station (pump codes it sent from home). An engine counts as busy from "
                  "mobilisation until it leaves the scene, plus the drive back (assumed equal to the drive out).\n"
                  "Source: LFB mobilisation and incident records (London Datastore). Single-engine stations have every "
                  "engine out whenever their one engine is out.", x=0.01, ha="left", fontsize=7.5, color=INK2)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def load_basemap() -> dict:
    """Major roads (A roads, trunk, motorway) and the Thames as line arrays, cached."""
    import pickle

    if BASEMAP.exists():
        return pickle.loads(BASEMAP.read_bytes())
    import osmium

    sys_path = str(ROOT / "src")
    import sys

    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from firestation.road_distance import PBF_PATH, load_or_build_graph

    G = load_or_build_graph()
    roads = []
    for u, v, d in G.edges(data=True):
        hw = d.get("highway")
        hw = hw[0] if isinstance(hw, list) else hw
        if hw in ("motorway", "trunk", "primary"):
            if "geometry" in d:
                roads.append(np.asarray(d["geometry"].coords))
            else:
                roads.append(np.array([[G.nodes[u]["x"], G.nodes[u]["y"]], [G.nodes[v]["x"], G.nodes[v]["y"]]]))
    river = []
    for o in osmium.FileProcessor(str(PBF_PATH), osmium.osm.NODE | osmium.osm.WAY).with_locations():
        if o.is_way() and o.tags.get("name") == "River Thames" and o.tags.get("waterway") == "river":
            river.append(np.array([[n.lon, n.lat] for n in o.nodes if n.location.valid()]))
    bm = {"roads": roads, "river": river}
    BASEMAP.parent.mkdir(parents=True, exist_ok=True)
    BASEMAP.write_bytes(pickle.dumps(bm))
    return bm


def hot_color(v, vmax):
    return HOT[int(np.clip(v / vmax, 0, 0.999) * len(HOT))]


def plot_poster(t: pd.DataFrame, year: int, path: Path):
    """One-page figure: map + busiest/quietest + hour-of-day heatmap + headline numbers."""
    from matplotlib import patheffects
    from matplotlib.collections import LineCollection
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

    style()
    t = t.dropna(subset=["lat", "lon"])
    vmax = float(np.ceil(t["utilisation_pct"].max() / 2.5) * 2.5)  # nice bins of 2.5 %
    cmap = ListedColormap(HOT)
    norm = BoundaryNorm(np.linspace(0, vmax, len(HOT) + 1), len(HOT))
    halo = [patheffects.withStroke(linewidth=3, foreground=SURFACE)]

    fig = plt.figure(figsize=(16, 12.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1], height_ratios=[0.11, 1, 0.36],
                          left=0.03, right=0.98, top=0.95, bottom=0.06, hspace=0.22, wspace=0.12)

    # ---- headline -------------------------------------------------------
    fig.text(0.03, 0.975, f"Central London's fire engines are the busiest in the city ({year})",
             fontsize=19, fontweight="bold", color=INK, va="center")
    top15 = t.head(15)
    kpis = [
        (f"{t['utilisation_pct'].median():.0f}%", "of the time, the typical station's\nengines are out on calls"),
        (f"{t['utilisation_pct'].iloc[0]:.0f}%", f"{t.index[0]}, the busiest: its engine is out\n"
                                                  f"{t['engine_hours_per_day'].iloc[0]:.1f} hours a day"),
        (f"{top15['covered_by_other_pct'].mean():.0f}%", "of calls in the 15 busiest stations' areas\n"
                                                         "are reached first by a neighbour's engine"),
        (f"{t['utilisation_pct'].iloc[0] / t['utilisation_pct'].iloc[-1]:.0f}×", f"busier: {t.index[0]} vs "
                                                                                  f"{t.index[-1]}, the quietest"),
    ]
    hdr = fig.add_subplot(gs[0, :])
    hdr.axis("off")
    for i, (big, small) in enumerate(kpis):
        x = 0.005 + i * 0.25
        hdr.text(x, 0.78, big, fontsize=26, fontweight="bold", color=HOT[5], transform=hdr.transAxes, va="center")
        hdr.text(x, 0.22, small, fontsize=10, color=INK2, transform=hdr.transAxes, va="center")

    # ---- map --------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    bm = load_basemap()
    ax.add_collection(LineCollection(bm["roads"], colors="#d9d8d3", linewidths=0.6, zorder=1))
    ax.add_collection(LineCollection(bm["river"], colors="#b9d6ee", linewidths=4.5, zorder=1, capstyle="round"))
    order = t.sort_values("calls_per_day", ascending=False)
    size = 18 + 26 * order["calls_per_day"]
    two = order["engines"] >= 2
    ax.scatter(order.loc[two, "lon"], order.loc[two, "lat"], s=size[two] * 1.6, facecolor="none",
               edgecolor=INK2, linewidth=0.7, zorder=2)
    ax.scatter(order["lon"], order["lat"], s=size, c=order["utilisation_pct"], cmap=cmap, norm=norm,
               edgecolor="white", linewidth=1.0, zorder=3)
    # Busiest stations: rank numbers (matching the bar chart), names would collide in the centre.
    for rank, st in enumerate(t.index[:10], start=1):
        r = t.loc[st]
        ax.annotate(str(rank), (r.lon, r.lat), xytext=(5, 4), textcoords="offset points", fontsize=9,
                    fontweight="bold", color=INK, path_effects=halo, zorder=4)
    for st in t.index[-3:]:
        r = t.loc[st]
        ax.annotate(f"{st} {r.utilisation_pct:.0f}%", (r.lon, r.lat), xytext=(7, 4), textcoords="offset points",
                    fontsize=8.5, color=INK2, path_effects=halo, zorder=4)
    ax.set_xlim(t["lon"].min() - 0.05, t["lon"].max() + 0.05)
    ax.set_ylim(t["lat"].min() - 0.03, t["lat"].max() + 0.03)
    ax.set_aspect(1 / np.cos(np.radians(51.5)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Where the busy stations are", fontsize=13)
    # legends: colour bar + circle sizes + engine ring
    cax = ax.inset_axes([0.02, 0.05, 0.32, 0.025])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks(np.arange(0, vmax + 0.1, 5))
    cb.set_ticklabels([f"{v:.0f}%" for v in np.arange(0, vmax + 0.1, 5)])
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    cax.set_title("% of engine time out on calls", fontsize=9, color=INK2, loc="left")
    for k, c in enumerate((3, 8, 15)):
        ax.scatter([], [], s=18 + 26 * c, color=HOT[3], edgecolor="white", label=f"{c} calls/day")
    ax.scatter([], [], s=150, facecolor="none", edgecolor=INK2, label="ring = 2 engines")
    ax.legend(loc="lower right", fontsize=8.5, labelspacing=1.3, borderpad=0.8, frameon=False)

    # ---- busiest / quietest bars ------------------------------------------
    right = gs[1, 1].subgridspec(2, 1, height_ratios=[15, 8], hspace=0.28)
    for k, (sub, title) in enumerate([(t.head(15), "15 busiest stations"),
                                      (t.tail(8), "8 quietest stations")]):
        a = fig.add_subplot(right[k])
        sub = sub.iloc[::-1]
        y = np.arange(len(sub))
        a.barh(y, sub["utilisation_pct"], color=[hot_color(v, vmax) for v in sub["utilisation_pct"]], height=0.7)
        ranks = [t.index.get_loc(s) + 1 for s in sub.index]
        a.set_yticks(y, [(f"{r}. " if k == 0 and r <= 10 else "") + f"{s}  {'●●' if e >= 2 else '●'}"
                         for r, s, e in zip(ranks, sub.index, sub["engines"])], fontsize=9)
        for i, (v, h) in enumerate(zip(sub["utilisation_pct"], sub["engine_hours_per_day"])):
            a.text(v + 0.2, i, f"{v:.0f}%  ({h:.1f} engine-h/day)", va="center", fontsize=8, color=INK2)
        a.set_xlim(0, vmax * 1.45)
        a.axvline(t["utilisation_pct"].median(), color=INK2, lw=0.8, ls="--")
        a.grid(axis="y", visible=False)
        a.set_title(title, fontsize=12)
        if k == 1:
            a.set_xlabel("% of engine time out on calls   (● = engines; dashed line = median station)", fontsize=9)
        a.tick_params(axis="x", labelsize=8)

    # ---- hour-of-day heatmap ---------------------------------------------
    hm = fig.add_subplot(gs[2, :])
    pos = hm.get_position()
    hm.set_position([0.13, pos.y0, 0.80, pos.height])  # leave room for station names on the left
    hourly = t.attrs["hourly"]
    rows = list(t.index[:12])
    data = pd.concat([hourly.loc[rows], hourly.median().rename("All stations (median)").to_frame().T])
    im = hm.imshow(data.to_numpy(), aspect="auto", cmap=LinearSegmentedColormap.from_list("hot", HOT), vmin=0,
                   interpolation="nearest")
    hm.set_yticks(range(len(data)), data.index, fontsize=8.5)
    hm.set_xticks(range(24), [f"{h:02d}" for h in range(24)], fontsize=8)
    hm.axhline(len(data) - 1.5, color=SURFACE, lw=3)
    hm.grid(False)
    hm.set_xlabel("hour of day")
    peak = int(data.iloc[:-1].mean().idxmax())
    low = int(data.iloc[:-1].mean().idxmin())
    hm.set_title(f"When are the 12 busiest stations busy? Peak around {peak:02d}:00, quietest around {low:02d}:00  "
                 f"(% of engine time out on calls, by hour of day)", fontsize=12)
    cb2 = fig.colorbar(im, cax=fig.add_axes([0.94, pos.y0, 0.008, pos.height]))
    cb2.ax.tick_params(labelsize=8)
    cb2.outline.set_visible(False)

    fig.text(0.03, 0.004,
             "Source: LFB mobilisation and incident records (London Datastore), station sites from OpenStreetMap. "
             "An engine is busy from mobilisation until it leaves the scene, plus the drive back (assumed equal to "
             "the drive out). Engines = pump codes each station sent from home.", fontsize=8, color=INK2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_map(t: pd.DataFrame, year: int, path: Path):
    import folium
    from branca.element import Element

    t = t.dropna(subset=["lat", "lon"])
    m = folium.Map(location=[51.49, -0.12], zoom_start=10, tiles=None, control_scale=True)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Light gray", max_native_zoom=16, max_zoom=18).add_to(m)
    vmax = t["utilisation_pct"].max()
    for st, r in t.sort_values("calls_per_day", ascending=False).iterrows():
        folium.CircleMarker(
            [r.lat, r.lon], radius=float(4 + 2.2 * np.sqrt(r.calls_per_day)),
            color="#ffffff", weight=1.5, fill=True, fill_color=seq_color(r.utilisation_pct, vmax), fill_opacity=0.92,
            tooltip=f"{st}: {r.utilisation_pct:.0f}% utilised",
            popup=folium.Popup(
                f"<b>{st}</b> ({int(r.engines)} engine{'s' if r.engines > 1 else ''})<br>"
                f"Utilisation: <b>{r.utilisation_pct:.1f}%</b><br>"
                f"Every engine out: {r.all_out_pct:.1f}% of the year<br>"
                f"Own-area calls answered by another station: {r.covered_by_other_pct:.0f}%<br>"
                f"Calls: {r.calls_per_day:.1f}/day · engine-hours {r.engine_hours_per_day:.1f}/day<br>"
                f"Median job {r.median_job_min:.0f} min", max_width=300),
        ).add_to(m)
    stops = np.linspace(0, vmax, len(SEQ) + 1)
    swatches = "".join(f'<span style="display:inline-block;width:26px;height:10px;background:{c}"></span>' for c in SEQ)
    legend = f"""
    <div style="position:fixed;top:12px;left:56px;z-index:9999;background:rgba(255,255,255,.96);padding:12px 14px;
      border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,.2);font:12px -apple-system,Segoe UI,sans-serif;color:#222;
      max-width:300px">
      <div style="font-weight:700;font-size:14px;margin-bottom:4px">How busy is each fire station? ({year})</div>
      <div style="margin-bottom:6px">Colour = % of engine time out on calls</div>
      <div>{swatches}</div>
      <div style="display:flex;justify-content:space-between;font-size:10.5px;color:#555">
        <span>{stops[0]:.0f}%</span><span>{stops[-1]:.0f}%</span></div>
      <div style="margin-top:6px">Circle size = calls per day. Click a station for details.</div>
      <div style="margin-top:6px;font-size:10.5px;color:#666">Source: LFB mobilisation records. Busy = mobilised
        until leaving the scene + drive back.</div>
    </div>"""
    m.get_root().html.add_child(Element(legend))
    m.save(str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2025)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(RAW / "london_fire_stations.csv").set_index("name")[["lat", "lon"]]
    t = station_table(args.year, stations)
    t.round(3).to_csv(OUT / "station_busyness.csv")
    plot_ranked(t, args.year, OUT / "station_busyness_ranked.png")
    plot_poster(t, args.year, OUT / "station_busyness.png")
    t.attrs["hourly"].round(2).to_csv(OUT / "station_busyness_by_hour.csv")
    plot_map(t, args.year, OUT / "station_busyness_map.html")
    s = {
        "year": args.year, "stations": len(t),
        "median_utilisation_pct": round(float(t["utilisation_pct"].median()), 1),
        "busiest": t["utilisation_pct"].head(5).round(1).to_dict(),
        "quietest": t["utilisation_pct"].tail(5).round(1).to_dict(),
        "corr_utilisation_vs_covered_by_other": round(float(t["utilisation_pct"].corr(t["covered_by_other_pct"])), 2),
    }
    (OUT / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
