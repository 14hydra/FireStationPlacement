#!/usr/bin/env python3
"""Robustness of the placement results: time of day, call severity and borough.

Every layout from scripts/score_layouts.py (today's stations, the recommended layouts and
the distance-based ones) is scored on held-out 2025 calls, split three ways, each layout
minus today with a 95 % paired bootstrap interval (resampling calls within the group):

  time of day   LightGBM travel times predicted for each period's representative hour
                (night 00-07 at 03h, morning rush 07-10 at 08h, day 10-16 at 13h,
                evening rush 16-20 at 18h, evening 20-24 at 21h) and the hourly dispatch
                model, whose busy chances use each 4-hour slot's calibrated load factor.
                Each call is scored with the times for its own hour of call.
  severity      all calls; fires and special services ("serious"); fires in homes.
  borough       each of the 33 boroughs (where the call happened).
  income        quintile of neighbourhood income: each call takes the ONS FYE 2020 net
                household income after housing costs (equivalised) of the nearest London
                MSOA population-weighted centroid; quintiles split London's ~980 MSOAs
                into fifths (Q1 poorest). Data: data/raw/income/ (ONS, Open Government Licence).

Severity, borough and income are reported under all three travel-time models (LightGBM with the
hour-specific times above; the two Kolesar formulas have no hour, so they use the all-day
dispatch model). A result holds up if it holds under all three.

Writes data/figures/placement/robustness/: robustness.csv (every layout x group), summary.json,
time_of_day.png, boroughs.png, income.png.

Run:  PYTHONPATH=src .venv/bin/python scripts/placement_robustness.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from firestation.busy_placement import ErlangBusyModel  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.dispatch import DISPATCH_PATH, candidates, dispatch_response, load_dispatch, slot_busy, slot_of  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid  # noqa: E402
from firestation.fire_station_location import (  # noqa: E402
    build_travel_matrix,
    busy_prob_vector,
    drop_far_incidents,
    fit_scorer_kolesar,
    recorded_station_turnout,
    severity_weight,
    snap_to_cells,
)
from firestation.ml_scorer import TIME_SLOTS, lgbm_matrices_by_hour  # noqa: E402
from firestation.station_capacity import default_pump_table  # noqa: E402
from score_layouts import load_layouts, site_key  # noqa: E402

OUT = ROOT / "data" / "figures" / "placement" / "robustness"
INCOME = ROOT / "data" / "raw" / "income"
QUINTILES = ["Q1 (poorest)", "Q2", "Q3", "Q4", "Q5 (richest)"]
PERIODS = ["night 00-07", "morning rush 07-10", "day 10-16", "evening rush 16-20", "evening 20-24"]
MAIN = {"lgbm_expand_3_boroughs1s": "add 3", "lgbm_replace_10_boroughs1s": "move up to 10",
        "lgbm_redesign_boroughs1s": "full redesign"}


def period_of(hour: np.ndarray) -> np.ndarray:
    out = np.zeros(len(hour), int)
    for i, (_, hours) in enumerate(TIME_SLOTS):
        out[np.isin(hour, list(hours))] = i
    return out


def income_quintile(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(quintile label, MSOA income) of each point's nearest London MSOA centroid."""
    from scipy.spatial import cKDTree

    inc = pd.read_excel(INCOME / "ons_msoa_income_fye2020.xlsx", "Net income after housing costs", header=4)
    inc = inc[inc["Region name"] == "London"].iloc[:, [0, 6]]
    inc.columns = ["msoa11cd", "income"]
    cen = pd.read_csv(INCOME / "ons_msoa2011_population_weighted_centroids.csv").merge(inc, on="msoa11cd")
    q = pd.qcut(cen["income"], 5, labels=QUINTILES).astype(str).to_numpy()
    k = np.cos(np.radians(51.5))
    _, i = cKDTree(np.c_[cen["lat"], cen["lon"] * k]).query(np.c_[lat, lon * k])
    return q[i], cen["income"].to_numpy()[i]


def slowest10(x: np.ndarray) -> float:
    k = max(int(np.ceil(0.1 * len(x))), 1)
    return float(np.partition(x, len(x) - k)[len(x) - k:].mean())


def compare(r0: np.ndarray, r: np.ndarray, boot: int, rng) -> dict:
    """Layout minus today on the same calls, with paired bootstrap intervals."""
    n = len(r0)
    d = r - r0
    out = {"n_calls": n, "today_mean_s": r0.mean(), "mean_s": r.mean(), "d_mean_s": d.mean(),
           "today_slowest10_s": slowest10(r0), "slowest10_s": slowest10(r)}
    out["d_slowest10_s"] = out["slowest10_s"] - out["today_slowest10_s"]
    dm, ds = [], []
    for _ in range(0, boot, 50):
        idx = rng.integers(0, n, size=(min(50, boot), n))
        dm.append(d[idx].mean(axis=1))
        ds.extend(slowest10(r[i]) - slowest10(r0[i]) for i in idx)
    dm = np.concatenate(dm)
    out["d_mean_lo"], out["d_mean_hi"] = np.percentile(dm, [2.5, 97.5])
    out["d_slowest10_lo"], out["d_slowest10_hi"] = np.percentile(ds, [2.5, 97.5])
    return {k: (round(float(v), 2) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def hourly_response(A, idx, busy_model, dmodel, factors, dslot) -> np.ndarray:
    """Expected first-engine attendance per cell in one 4-hour dispatch slot."""
    pb = slot_busy(busy_model, A, idx, [factors[dslot]])[0]
    pumps = np.maximum(busy_model.pumps_for(list(idx)), 1)
    rows, feats = candidates(A, idx, pb, pumps, dmodel.k, slot=dslot)
    n = np.arange(A.shape[1])
    return (dmodel.proba(feats, A.shape[1]) * A[rows, n[None, :]]).sum(axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    spec = get_city("london")
    houses = load_fire_stations(spec)
    incidents = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024, 2025]), houses)
    demand = incidents[incidents["cal_year"] == 2024].reset_index(drop=True)
    ev = incidents[incidents["cal_year"] == 2025].reset_index(drop=True)

    pumps_today = default_pump_table(houses).set_index("station")["n_pumps"].reindex(
        houses["facilityname"].astype(str)).fillna(1).astype(int).to_numpy()
    layouts = load_layouts(houses, pumps_today)
    sites, index = [], {}
    for L in layouts.values():
        for r in L.itertuples():
            k = site_key(r)
            if k not in index:
                index[k] = len(sites)
                sites.append({"site_name": str(r.site_name), "lat": float(r.lat), "lon": float(r.lon),
                              "is_existing": bool(r.is_existing)})
    sites = pd.DataFrame(sites)
    rows = {name: np.array([index[site_key(r)] for r in L.itertuples()]) for name, L in layouts.items()}
    pumps = {name: (L["n_pumps"].fillna(1).astype(int).to_numpy() if "n_pumps" in L else np.ones(len(L), int))
             for name, L in layouts.items()}
    print(f"{len(layouts)} layouts, {len(sites)} sites", flush=True)

    cells = build_demand_grid(demand, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    cidx, _ = snap_to_cells(ev, cells)
    load_w = cells["n_incidents"].to_numpy(float)
    hour = pd.to_numeric(ev["hour_of_call"], errors="coerce").fillna(12).astype(int).to_numpy()
    period, dslot = period_of(hour), slot_of(hour)
    groups = {("time", "all day"): np.ones(len(ev), bool)}
    for i, name in enumerate(PERIODS):
        groups[("time", name)] = period == i
    groups[("time", "rush hours (07-10, 16-20)")] = np.isin(period, [1, 3])
    sev = {s: severity_weight(ev, s) > 0 for s in ("all", "serious", "dwelling_fire")}
    for s, m in sev.items():
        groups[("severity", s)] = m
    borough = ev["borough"].astype(str).str.upper().to_numpy()
    for b in sorted(set(borough) - {"NAN", ""}):
        groups[("borough", b)] = borough == b

    quint, _ = income_quintile(ev["dest_lat"].to_numpy(float), ev["dest_lon"].to_numpy(float))
    for q in QUINTILES:
        groups[("income", q)] = quint == q

    turn_tab, turn_new = recorded_station_turnout([2024])
    norm = {k.upper(): v for k, v in turn_tab.items()}
    turnout = np.array([norm.get(r.site_name.upper(), turn_new) if r.is_existing else turn_new
                        for r in sites.itertuples()])
    measured = busy_prob_vector(sites, demand)
    kmodel, _, _ = fit_scorer_kolesar(demand, houses)
    bundle = __import__("joblib").load(DISPATCH_PATH)
    d_hour, d_all, factors = bundle["model"], bundle["all_day_model"], bundle["slot_load_factors"]
    today_rows = rows["today"]
    base_pumps = np.ones(len(sites), int)
    base_pumps[today_rows] = pumps["today"]

    per_call = {}  # (scorer, layout) -> response per 2025 call
    for scorer in ("lgbm", "road", "kolesar"):
        print(f"Travel times: {scorer}…", flush=True)
        T, _ = build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer=scorer,
                                   kolesar=kmodel, incidents=demand)
        A = T + turnout[:, None]
        model = ErlangBusyModel.calibrate(A, today_rows, measured, base_pumps, load_w)
        if scorer == "lgbm":
            Th = lgbm_matrices_by_hour(sites, cells, demand)
            Ah = {h: Th[h] + turnout[:, None] for h in Th}
            combos = sorted(set(zip(period, dslot)))
        for name, r in rows.items():
            model.pumps = np.ones(len(sites), int)
            model.pumps[r] = np.maximum(pumps[name], 1)
            if scorer != "lgbm":
                per_call[scorer, name] = dispatch_response(A, r, model, d_all)[cidx]
                continue
            resp = np.zeros(len(ev))
            for p, s in combos:
                m = (period == p) & (dslot == s)
                resp[m] = hourly_response(Ah[TIME_SLOTS[p][0]], r, model, d_hour, factors, s)[cidx[m]]
            per_call[scorer, name] = resp
        print(f"  today's average: {per_call[scorer, 'today'].mean():.1f} s "
              f"(recorded {pd.to_numeric(ev['travel_seconds'], errors='coerce').mean():.0f} s drive only)", flush=True)

    recs = []
    for (scorer, name), r in per_call.items():
        if name == "today":
            continue
        r0 = per_call[scorer, "today"]
        for (kind, g), m in groups.items():
            if scorer != "lgbm" and kind == "time" and g != "all day":
                continue
            if m.sum() < 200:
                continue
            boot = args.boot if kind != "borough" else max(args.boot // 5, 100)
            recs.append({"layout": name, "scorer": scorer, "group_type": kind, "group": g,
                         **compare(r0[m], r[m], boot, rng)})
        print(f"  scored {scorer} {name}", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "robustness.csv", index=False)

    # Summary: for each layout, does the gain hold in every period / severity, under all three models?
    summ = {}
    for name in df["layout"].unique():
        x = df[df["layout"] == name]
        lg = x[x["scorer"] == "lgbm"]
        b = x[x["group_type"] == "borough"].pivot_table(index="group", columns="scorer", values="d_mean_s")
        summ[name] = {
            "time_of_day_lgbm": lg[lg["group_type"] == "time"].set_index("group")[
                ["n_calls", "today_mean_s", "d_mean_s", "d_mean_lo", "d_mean_hi", "d_slowest10_s"]].to_dict("index"),
            "severity": {s: x[(x["group_type"] == "severity") & (x["group"] == s)].set_index("scorer")[
                ["d_mean_s", "d_mean_lo", "d_mean_hi", "d_slowest10_s"]].to_dict("index") for s in sev},
            "faster_in_every_period_lgbm": bool((lg[lg["group_type"] == "time"]["d_mean_hi"] < 0).all()),
            "faster_for_every_severity_all_models": bool(
                (x[x["group_type"] == "severity"]["d_mean_hi"] < 0).all()),
            "income_quintiles": {q: x[(x["group_type"] == "income") & (x["group"] == q)].set_index("scorer")[
                ["n_calls", "today_mean_s", "d_mean_s", "d_mean_lo", "d_mean_hi", "d_slowest10_s"]].to_dict("index")
                for q in QUINTILES},
            "boroughs_worse_by_1s_any_model": sorted(b.index[(b >= 1).any(axis=1)].tolist()),
            "boroughs_significantly_worse_lgbm": sorted(
                x[(x["group_type"] == "borough") & (x["scorer"] == "lgbm") & (x["d_mean_lo"] > 0)]["group"].tolist()),
        }
    (OUT / "summary.json").write_text(json.dumps(
        {"doc": __doc__, "n_calls_2025": int(len(ev)), "bootstrap": args.boot, "layouts": summ},
        indent=2, default=str) + "\n")
    plots(df)
    pd.set_option("display.width", 220)
    main_rows = df[df["layout"].isin(MAIN) & (df["group_type"] != "borough")]
    print(main_rows.pivot_table(index=["group_type", "group"], columns=["layout", "scorer"],
                                values="d_mean_s").round(1).to_string())
    print("Wrote", OUT)
    return 0


def plots(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = [k for k in MAIN if k in set(df["layout"])]
    if not main:
        return
    # Time of day: change in average and slowest 10 % per period, LightGBM.
    t = df[(df["scorer"] == "lgbm") & (df["group_type"] == "time") & df["group"].isin(PERIODS)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    x = np.arange(len(PERIODS))
    wdt = 0.8 / len(main)
    for ax, (col, lo, hi, title) in zip(axes, [("d_mean_s", "d_mean_lo", "d_mean_hi", "average"),
                                               ("d_slowest10_s", "d_slowest10_lo", "d_slowest10_hi", "slowest 10 %")]):
        for i, name in enumerate(main):
            s = t[t["layout"] == name].set_index("group").reindex(PERIODS)
            ax.bar(x + (i - (len(main) - 1) / 2) * wdt, s[col], wdt, label=MAIN[name],
                   yerr=[s[col] - s[lo], s[hi] - s[col]], capsize=2)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [p.replace(" ", "\n", 1) for p in PERIODS], fontsize=8)
        ax.set_ylabel("change vs today (s)")
        ax.set_title(f"First-engine time, {title} (LightGBM, 2025 calls)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "time_of_day.png", dpi=150)
    plt.close(fig)

    # Income: change in average by neighbourhood income quintile, all three models.
    q = df[(df["group_type"] == "income") & df["layout"].isin(main)]
    fig, axes = plt.subplots(1, len(main), figsize=(4.2 * len(main), 4), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(QUINTILES))
    for ax, name in zip(axes, main):
        for j, sc in enumerate(("lgbm", "road", "kolesar")):
            s = q[(q["layout"] == name) & (q["scorer"] == sc)].set_index("group").reindex(QUINTILES)
            ax.bar(x + (j - 1) * 0.27, s["d_mean_s"], 0.27, label=sc,
                   yerr=[s["d_mean_s"] - s["d_mean_lo"], s["d_mean_hi"] - s["d_mean_s"]], capsize=2)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [t.replace(" ", "\n") for t in QUINTILES], fontsize=8)
        ax.set_title(MAIN[name], fontsize=10)
    axes[0].set_ylabel("change in average vs today (s)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Who gains: change by neighbourhood income (2025 calls)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "income.png", dpi=150)
    plt.close(fig)

    # Boroughs: change in average per borough for each main layout, all three models.
    b = df[(df["group_type"] == "borough") & df["layout"].isin(main)]
    fig, axes = plt.subplots(1, len(main), figsize=(4.2 * len(main), 8), sharey=True)
    axes = np.atleast_1d(axes)
    order = b[(b["layout"] == main[-1]) & (b["scorer"] == "lgbm")].sort_values("d_mean_s")["group"].tolist()
    y = np.arange(len(order))
    for ax, name in zip(axes, main):
        for j, (sc, mk) in enumerate([("lgbm", "o"), ("road", "s"), ("kolesar", "^")]):
            s = b[(b["layout"] == name) & (b["scorer"] == sc)].set_index("group").reindex(order)
            ax.scatter(s["d_mean_s"], y, marker=mk, s=16, label=sc)
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(1, color="r", lw=0.6, ls="--")
        ax.set_title(MAIN[name], fontsize=10)
        ax.set_xlabel("change in average vs today (s)")
    axes[0].set_yticks(y, [o.title() for o in order], fontsize=7)
    axes[0].legend(fontsize=8)
    fig.suptitle("Change by borough (red line: 1 s worse)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "boroughs.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
