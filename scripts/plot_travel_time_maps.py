#!/usr/bin/env python3
"""
Where in London are fire engines slow, and where is that traffic? Three maps.

  A  First-engine time: mean turnout + driving time of the first engine sent, per
     500 m square, against LFB's 6-minute average standard.
  B  Rush-hour slowdown (congestion): how much slower engines drive into each 1 km
     square on weekday rush hours (07-10, 16-19) than at night (22-06), after
     allowing for trip distance. Street layout is the same at both times, so the
     difference is mostly traffic. Coloured relative to London's typical slowdown.
  C  Held up by traffic: share of engine trips into each 1 km square that LFB coded
     as delayed by "Traffic, roadworks, etc" or "Traffic calming measures".
  B and C pool each square with its 8 neighbours (a 3 km window) to cut noise.

Data: LFB engine trips 2021-2025 (data/processed/lfb_travel_training.csv) and their
delay codes (LFB mobilisation records). Squares are by incident location (where engines
drove to), not by street: without GPS we do not know which streets they used.
Squares with too few trips are left blank.

Writes data/figures/travel_time/maps/: travel_time_maps.png, travel_time_maps.html,
travel_time_cells.csv, summary.json

Run:  .venv/bin/python scripts/plot_travel_time_maps.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
RAW = ROOT / "data" / "raw" / "london"
OUT = ROOT / "data" / "figures" / "travel_time" / "maps"
YEARS = (2021, 2025)
MIN_A, MIN_B, MIN_C = 25, 60, 150  # minimum trips per square (B, C: pooled 3 x 3 window) to shade it

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
# Diverging blue <-> red with a gray midpoint (dataviz reference palette).
DIV = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a3a2", "#e34948", "#9e1f1e"]
HOT = ["#fde4d6", "#fbc4a6", "#f7a072", "#eb6834", "#c94f1f", "#9c3a14", "#6b270c"]
TRAFFIC_CODES = {"Traffic, roadworks, etc", "Traffic calming measures"}


def cells(lat, lon, size_m):
    dlat = size_m / 111_320.0
    dlon = dlat / np.cos(np.radians(51.5))
    return np.floor(np.asarray(lat) / dlat).astype(int), np.floor(np.asarray(lon) / dlon).astype(int), dlat, dlon


def load_trips() -> pd.DataFrame:
    from firestation.kolesar import fit_kolesar

    d = pd.read_csv(ROOT / "data" / "processed" / "lfb_travel_training.csv", low_memory=False,
                    usecols=["incident_number", "cal_year", "date_of_call", "hour_of_call", "pump_order", "borough",
                             "dest_lat", "dest_lon", "road_km", "drive_s", "turnout_s"])
    d = d[d["cal_year"].between(*YEARS)].copy()
    kol = fit_kolesar(d["road_km"].to_numpy(float), d["drive_s"].to_numpy(float))
    d["expected_s"] = np.clip(kol.predict(d["road_km"].to_numpy(float)), 30, 1200)
    d["slow_ratio"] = d["drive_s"] / d["expected_s"]  # >1: slower than distance alone predicts
    h = d["hour_of_call"]
    wkday = pd.to_datetime(d["date_of_call"]).dt.dayofweek < 5
    d["period"] = np.where(wkday & (h.between(7, 9) | h.between(16, 18)), "rush",
                           np.where((h >= 22) | (h <= 5), "night", "other"))
    # Delay codes from the mobilisation records, matched on incident + pump order.
    parts = [pd.read_csv(RAW / f, encoding="utf-8-sig", low_memory=False, dtype={"IncidentNumber": str},
                         usecols=["IncidentNumber", "PumpOrder", "DelayCode_Description"])
             for f in ["lfb_mobilisations_2021_2024.csv", "lfb_mobilisations_2025.csv"]]
    m = pd.concat(parts).drop_duplicates(["IncidentNumber", "PumpOrder"])
    m["traffic"] = m["DelayCode_Description"].isin(TRAFFIC_CODES)
    d = d.merge(m[["IncidentNumber", "PumpOrder", "traffic"]], how="left",
                left_on=["incident_number", "pump_order"], right_on=["IncidentNumber", "PumpOrder"])
    d["traffic"] = d["traffic"].fillna(False).astype(bool)
    return d


def grid(df: pd.DataFrame, size_m: int, value_fn, min_n: int) -> pd.DataFrame:
    iy, ix, dlat, dlon = cells(df["dest_lat"], df["dest_lon"], size_m)
    g = df.assign(iy=iy, ix=ix).groupby(["iy", "ix"])
    out = value_fn(g)
    out = out[out["n"] >= min_n].reset_index()
    out["lat"] = (out["iy"] + 0.5) * dlat
    out["lon"] = (out["ix"] + 0.5) * dlon
    out["dlat"], out["dlon"] = dlat, dlon
    out["borough"] = g["borough"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "").reindex(
        pd.MultiIndex.from_frame(out[["iy", "ix"]])).to_numpy()
    return out


def compute(d: pd.DataFrame):
    first = d[d["pump_order"] == 1].assign(attend_s=lambda x: x["turnout_s"].clip(0, 600) + x["drive_s"])
    a = grid(first, 500, lambda g: g.agg(n=("attend_s", "size"), value=("attend_s", "mean")), MIN_A)
    a["value"] /= 60  # minutes

    iy, ix, dlat, dlon = cells(d["dest_lat"], d["dest_lon"], 1000)
    d = d.assign(iy=iy, ix=ix)
    groups = {k: g for k, g in d.groupby(["iy", "ix"])}
    rows = []
    for (y, x), own in groups.items():
        pool = pd.concat([groups[(y + dy, x + dx)] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                          if (y + dy, x + dx) in groups])
        rush, night = pool.loc[pool["period"] == "rush", "slow_ratio"], pool.loc[pool["period"] == "night", "slow_ratio"]
        rows.append({"iy": y, "ix": x, "n_rush": len(rush), "n_night": len(night), "n_pool": len(pool),
                     "slowdown": 100 * (rush.median() / night.median() - 1) if len(rush) and len(night) else np.nan,
                     "traffic": 100 * pool["traffic"].mean(),
                     "borough": own["borough"].mode().iat[0] if len(own) else ""})
    pooled = pd.DataFrame(rows)
    pooled["lat"], pooled["lon"] = (pooled["iy"] + 0.5) * dlat, (pooled["ix"] + 0.5) * dlon
    pooled["dlat"], pooled["dlon"] = dlat, dlon
    b = pooled.assign(n=pooled[["n_rush", "n_night"]].min(axis=1), value=pooled["slowdown"])
    b = b[b["n"] >= MIN_B].reset_index(drop=True)
    c = pooled.assign(n=pooled["n_pool"], value=pooled["traffic"])
    c = c[c["n"] >= MIN_C].reset_index(drop=True)
    return a, b, c


# --------------------------------------------------------------------------- #
# Static figure
# --------------------------------------------------------------------------- #


def draw_cells(ax, df, cmap, norm):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    rects = [Rectangle((r.lon - r.dlon / 2, r.lat - r.dlat / 2), r.dlon, r.dlat) for r in df.itertuples()]
    pc = PatchCollection(rects, cmap=cmap, norm=norm, linewidth=0, alpha=0.9, zorder=2)
    pc.set_array(df["value"].to_numpy())
    ax.add_collection(pc)
    return pc


def base(ax, bm, extent):
    from matplotlib.collections import LineCollection

    ax.add_collection(LineCollection(bm["roads"], colors="#d4d3ce", linewidths=0.45, zorder=1))
    ax.add_collection(LineCollection(bm["river"], colors="#b9d6ee", linewidths=3.5, zorder=3, capstyle="round"))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(np.radians(51.5)))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def label_top(ax, df, k, fmt, halo):
    top = df.sort_values("value", ascending=False)
    used, spots = [], []
    for r in top.itertuples():
        if len(used) >= k:
            break
        # One label per borough, and none within ~11 km of another, so labels never collide.
        if r.borough in used or any(np.hypot((r.lon - x) * 0.62, r.lat - y) < 0.1 for x, y in spots):
            continue
        used.append(r.borough)
        spots.append((r.lon, r.lat))
        cx, cy = np.mean(ax.get_xlim()), np.mean(ax.get_ylim())
        off = (-60 if r.lon > cx else 14, -22 if r.lat > cy else 14)
        ax.annotate(f"{str(r.borough).title()}\n{fmt(r.value)}", (r.lon, r.lat), xytext=off,
                    textcoords="offset points", fontsize=7.5, color=INK, path_effects=halo, zorder=5,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))


def b_bounds(b: pd.DataFrame) -> list[float]:
    """Diverging bins centred on London's median rush-hour slowdown."""
    m = round(float(b["value"].median()))
    return [m - 15, m - 8, m - 4, m - 1.5, m + 1.5, m + 4, m + 8, m + 15]


def plot_png(a, b, c, stations, path: Path):
    from matplotlib import patheffects
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from plot_station_busyness import load_basemap

    bm = load_basemap()
    halo = [patheffects.withStroke(linewidth=2.5, foreground=SURFACE)]
    extent = (a["lon"].min() - 0.02, a["lon"].max() + 0.02, a["lat"].min() - 0.015, a["lat"].max() + 0.015)
    plt.rcParams.update({"figure.facecolor": SURFACE, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 3, figsize=(21, 8.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.12, wspace=0.03)

    panels = [
        (a, ListedColormap(DIV), BoundaryNorm([3, 4, 5, 5.75, 6.25, 7, 8, 10], len(DIV)), "min",
         "A. How long the first engine takes",
         "Mean turnout + drive of the first engine sent, per 500 m (2021-25).\nBlue = faster than LFB's 6-minute average, red = slower.",
         lambda v: f"{v:.1f} min"),
        (b, ListedColormap(DIV), BoundaryNorm(b_bounds(b), len(DIV)), "%",
         "B. Rush-hour slowdown (congestion)",
         f"% slower on weekday rush hours than at night, allowing for distance (mostly traffic).\n"
         f"London typical: +{b['value'].median():.0f}%. Red = more congested than typical, blue = less.",
         lambda v: f"+{v:.0f}% at rush hour"),
        (c, ListedColormap(HOT), BoundaryNorm([0, 2, 4, 6, 8, 10, 13, 20], len(HOT)), "%",
         "C. Engines held up by traffic",
         f"Share of engine trips that LFB coded as delayed by traffic, roadworks\nor traffic calming (London: {c['value'].median():.0f}%). Coding may vary between crews.",
         lambda v: f"{v:.0f}%"),
    ]
    for ax, (df, cmap, norm, unit, title, sub, fmt) in zip(axes, panels):
        base(ax, bm, extent)
        pc = draw_cells(ax, df, cmap, norm)
        if df is a:
            ax.scatter(stations["lon"], stations["lat"], s=7, color=INK, zorder=4, label="fire station")
            ax.legend(loc="lower right", fontsize=8, frameon=False)
        label_top(ax, df, 2 if df is b else 3, fmt, halo)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK, pad=36)
        ax.text(0, 1.005, sub, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
        cax = ax.inset_axes([0.05, -0.07, 0.6, 0.025])
        cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8)
        cb.set_ticks(norm.boundaries)
        cb.set_ticklabels([f"{v:+g}%" if df is b else f"{v:g}{'' if unit == 'min' else '%'}"
                           for v in norm.boundaries])
        cax.set_title("minutes" if unit == "min" else "", fontsize=8, color=INK2, loc="left")

    fig.text(0.01, 0.955, "Where London's fire engines are slow, and where that is traffic", fontsize=20,
             fontweight="bold", color=INK)
    fig.text(0.01, 0.015,
             "Source: LFB mobilisation and incident records 2021-2025 (London Datastore); road map from OpenStreetMap. "
             "Squares are by incident location (where engines drove to), not individual streets. Blank squares have "
             f"too few trips (A < {MIN_A}; B < {MIN_B} rush and night trips, C < {MIN_C} trips in the pooled 3 km window).",
             fontsize=8.5, color=INK2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Interactive map
# --------------------------------------------------------------------------- #


def plot_html(a, b, c, stations, path: Path):
    import folium
    from branca.element import Element

    m = folium.Map(location=[51.495, -0.12], zoom_start=10, tiles=None, control_scale=True)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Light gray", max_native_zoom=16, max_zoom=18,
        control=False).add_to(m)  # always on; the three maps below are the switchable base layers

    def color(v, bounds, ramp):
        i = int(np.clip(np.searchsorted(bounds, v) - 1, 0, len(ramp) - 1))
        return ramp[i]

    layers = [
        ("A. First-engine time", a, [3, 4, 5, 5.75, 6.25, 7, 8, 10], DIV,
         lambda r: f"<b>{r.value:.1f} min</b> mean first-engine time<br>{int(r.n)} calls · {str(r.borough).title()}", True),
        ("B. Rush-hour slowdown", b, b_bounds(b), DIV,
         lambda r: f"<b>{r.value:+.0f}%</b> slower at rush hour than at night (3 km window)<br>"
                   f"{int(r.n_rush)} rush / {int(r.n_night)} night trips · {str(r.borough).title()}", False),
        ("C. Held up by traffic", c, [0, 2, 4, 6, 8, 10, 13, 20], HOT,
         lambda r: f"<b>{r.value:.1f}%</b> of trips held up by traffic or calming<br>"
                   f"{int(r.n)} trips in the 3 km window · {str(r.borough).title()}", False),
    ]
    for name, df, bounds, ramp, tip, show in layers:
        fg = folium.FeatureGroup(name=name, show=show, overlay=False)
        for r in df.itertuples():
            folium.Rectangle(
                bounds=[[r.lat - r.dlat / 2, r.lon - r.dlon / 2], [r.lat + r.dlat / 2, r.lon + r.dlon / 2]],
                color=None, weight=0, fill=True, fill_color=color(r.value, bounds, ramp), fill_opacity=0.72,
                tooltip=tip(r)).add_to(fg)
        fg.add_to(m)
    st = folium.FeatureGroup(name="Fire stations", show=True)
    for name, r in stations.iterrows():
        folium.CircleMarker([r.lat, r.lon], radius=3, color="#111", weight=1, fill=True, fill_color="#111",
                            fill_opacity=0.9, tooltip=name).add_to(st)
    st.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(Element("""
    <div style="position:fixed;top:12px;left:56px;z-index:9999;max-width:330px;background:rgba(255,255,255,.96);
      padding:12px 14px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,.2);
      font:12px/1.45 -apple-system,Segoe UI,sans-serif;color:#222">
      <div style="font-weight:700;font-size:14px;margin-bottom:4px">Where London's fire engines are slow</div>
      Switch maps at top right; hover a square for numbers.<br>
      <b>A</b> first-engine time (blue faster / red slower than LFB's 6-min average) ·
      <b>B</b> how much slower at weekday rush hour than at night (traffic; red = more than London's typical) ·
      <b>C</b> share of trips LFB coded as held up by traffic or calming.
      <div style="margin-top:6px;font-size:10.5px;color:#666">LFB records 2021-2025. Squares are by incident location,
        not streets driven.</div>
    </div>"""))
    m.save(str(path))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(RAW / "london_fire_stations.csv").set_index("name")[["lat", "lon"]]
    print("Loading trips and delay codes…")
    d = load_trips()
    print(f"{len(d):,} trips; computing squares…")
    a, b, c = compute(d)
    pd.concat([a.assign(map="A_first_engine_min"), b.assign(map="B_rush_slowdown_pct"),
               c.assign(map="C_traffic_delay_pct")]).to_csv(OUT / "travel_time_cells.csv", index=False)
    plot_png(a, b, c, stations, OUT / "travel_time_maps.png")
    plot_html(a, b, c, stations, OUT / "travel_time_maps.html")

    def top(df, k=5):
        t = df.sort_values("value", ascending=False).drop_duplicates("borough").head(k)
        return {str(r.borough).title(): round(float(r.value), 1) for r in t.itertuples()}

    w = a["n"]
    s = {
        "years": list(YEARS), "n_trips": int(len(d)),
        "A_share_of_first_engine_calls_in_squares_over_6_min": round(float(w[a["value"] > 6].sum() / w.sum()), 3),
        "A_slowest_squares_min": top(a),
        "B_median_rush_slowdown_pct": round(float(b["value"].median()), 1),
        "B_most_congested_pct": top(b),
        "C_overall_traffic_delay_pct": round(float(100 * d["traffic"].mean()), 2),
        "C_highest_pct": top(c),
        "corr_B_vs_C": round(float(b.merge(c, on=["iy", "ix"])[["value_x", "value_y"]].corr().iloc[0, 1]), 2),
    }
    (OUT / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
