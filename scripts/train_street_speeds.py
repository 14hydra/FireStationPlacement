#!/usr/bin/env python3
"""Learned street speeds + a LightGBM correction, compared with LightGBM alone.

1. Street speeds (firestation.street_speeds): a pace for every street segment from its
   characteristics, fitted so the summed route times match 2021-2024 driving times, with
   routes re-chosen under the learned speeds each round. 95 % intervals on every
   coefficient by bootstrap (resampling trips, routes fixed).
2. Correction: LightGBM predicts recorded drive minus the street-speed time, so
   prediction = street-speed base + correction. Two versions:
     full        every feature the current model uses, plus the base time
     restricted  only what street speeds cannot capture: time of day, day of week,
                 season, turns along the route, busy engines and call density near the
                 destination, plus the base time
3. Compared on 2025 trips (MAE, median error, bias), overall and by trip length:
   Kolesar (road km), street speeds alone, LightGBM alone, and the two combined models.
   All LightGBM fits use the same settings and one seed, so they compare like for like.

Writes data/figures/travel_time/street_speeds/ (metrics.json, coefficients.csv,
by_distance.csv, coefficients.png, speed_map.png) and data/processed/models/
street_speeds.joblib, travel_time_lfb_streetspeed.joblib (the full combined model,
usable with firestation.travel_time_model.predict when trips carry street_base_s).

Run:  PYTHONPATH=src .venv/bin/python scripts/train_street_speeds.py [--seeds 4]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from firestation.street_speeds import MODEL_PATH, StreetSpeedModel, edge_design  # noqa: E402
from firestation.travel_time_model import (  # noqa: E402
    CATEGORICAL,
    DRIVE_CLIP_S,
    attach_weather,
    fit_bag,
    load_weather,
    predict_bag,
)
from train_travel_time_lfb import FINAL_GROUPS, TRAIN_YEARS, TUNED_PARAMS, Split, fit_one, metrics  # noqa: E402

OUT = ROOT / "data" / "figures" / "travel_time" / "street_speeds"
DATA = ROOT / "data" / "processed" / "lfb_travel_training.csv"
COMBINED_PATH = ROOT / "data" / "processed" / "models" / "travel_time_lfb_streetspeed.joblib"
RESTRICTED = ["street_base_s", "hour", "dow", "month", "rush", "night", "weekend", "busy_flag", "pump_order",
              "from_home", "turns_per_km", "sharp_turns_per_km", "dest_n_300m", "dest_n_1km"]
LABELS = {"pace_major": "Motorway / trunk", "pace_primary": "A road", "pace_secondary": "B road",
          "pace_tertiary": "Local distributor", "pace_residential": "Residential", "pace_minor": "Minor / unclassified",
          "d_20mph": "20 mph limit", "d_bus_lane": "Bus lane", "d_oneway": "One-way", "d_multilane": "2+ lanes",
          "d_bridge_tunnel": "Bridge / tunnel", "per_junction": "Each junction", "per_signal": "Each traffic signal",
          "per_calming": "Each traffic calming feature", "per_crossing": "Each pedestrian crossing",
          "per_give_way": "Each give-way / stop sign", "per_roundabout": "Each roundabout"}


def by_distance(y, preds: dict, km) -> pd.DataFrame:
    bins = [0, 1, 2, 3, 4, 6, 100]
    lab = pd.cut(km, bins, labels=["<1", "1-2", "2-3", "3-4", "4-6", "6+"])
    rows = []
    for b in lab.categories:
        m = np.asarray(lab == b)
        rows.append({"road_km": b, "trips": int(m.sum()),
                     **{f"mae_{k}": round(float(np.abs(p[m] - y[m]).mean()), 2) for k, p in preds.items()},
                     **{f"bias_{k}": round(float((p[m] - y[m]).mean()), 2) for k, p in preds.items()}})
    q = np.quantile(km, 0.95)
    m = km >= q
    rows.append({"road_km": f"longest 5% (>= {q:.1f} km)", "trips": int(m.sum()),
                 **{f"mae_{k}": round(float(np.abs(p[m] - y[m]).mean()), 2) for k, p in preds.items()},
                 **{f"bias_{k}": round(float((p[m] - y[m]).mean()), 2) for k, p in preds.items()}})
    return pd.DataFrame(rows)


def plots(table: pd.DataFrame, router, model: StreetSpeedModel) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    t = table[table["parameter"].isin(LABELS)].copy()
    t["label"] = t["parameter"].map(LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [6, 5, 6]})
    for ax, (kind, title, unit) in zip(axes, [("pace_", "Pace by road type", "seconds per km (lower = faster)"),
                                              ("d_", "Adjustment to the pace", "extra seconds per km"),
                                              ("per_", "Fixed time per feature passed", "seconds each")]):
        s = t[t["parameter"].str.startswith(kind)]
        y = np.arange(len(s))
        ax.barh(y, s["value"], xerr=[s["value"] - s["lo"], s["hi"] - s["value"]], capsize=3,
                color=np.where(s["value"] >= 0, "#c0504d", "#4f81bd") if kind != "pace_" else "#7f7f7f")
        ax.set_yticks(y, s["label"], fontsize=9)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(unit)
        ax.set_title(title, fontsize=10)
        if kind == "pace_":
            for yi, (v, kph) in enumerate(zip(s["value"], s["speed_kph"])):
                ax.text(v + 2, yi, f"{kph:.0f} km/h", va="center", fontsize=8)
    fig.suptitle("Learned fire engine street speeds (LFB trips 2021-2024; 95% bootstrap intervals)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "coefficients.png", dpi=150)
    plt.close(fig)

    # Map: speed on every segment (length over learned time, point features included).
    v = router.edge_vals
    e = router.edge_id.tocoo()
    order = np.argsort(e.data)
    u_idx, v_idx = e.row[order], e.col[order]
    segs = np.stack([np.c_[router.node_lon[u_idx], router.node_lat[u_idx]],
                     np.c_[router.node_lon[v_idx], router.node_lat[v_idx]]], axis=1)
    kph = model.edge_speed_kph(router)
    L = v["len_total"].to_numpy()
    keep = L > 15
    fig, ax = plt.subplots(figsize=(11, 9))
    lc = LineCollection(segs[keep], array=np.clip(kph[keep], 15, 70), cmap="RdYlGn", linewidths=0.35)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect(1 / np.cos(np.radians(51.5)))
    ax.set_axis_off()
    cb = fig.colorbar(lc, ax=ax, shrink=0.6)
    cb.set_label("learned fire engine speed (km/h)")
    ax.set_title("How fast fire engines move on each London street (learned from LFB trips)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "speed_map.png", dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=1, help="Seeds for the saved combined model (comparison uses 1)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = json.loads((ROOT / "data" / "processed" / "models" / "travel_time_lfb.metrics.json").read_text())
    params = {**TUNED_PARAMS, "n_estimators": int(m["lightgbm_params"]["n_estimators"])}

    inc = pd.read_csv(DATA, low_memory=False)
    inc = attach_weather(inc, load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    inc = inc[inc.cal_year.isin(TRAIN_YEARS + (2025,))].reset_index(drop=True)
    tr_mask = inc.cal_year.isin(TRAIN_YEARS).to_numpy()
    y_all = inc["drive_s"].to_numpy(float)

    # ---- 1. street speeds
    from firestation import ml_scorer

    router = ml_scorer._router()
    print(f"Street speeds: fitting on {tr_mask.sum():,} trips (2021-2024)…", flush=True)
    sm = StreetSpeedModel().fit(router, inc.loc[tr_mask, "start_lat"], inc.loc[tr_mask, "start_lon"],
                                inc.loc[tr_mask, "dest_lat"], inc.loc[tr_mask, "dest_lon"], y_all[tr_mask],
                                rounds=args.rounds)
    # Bootstrap intervals with the final routes held fixed.
    Z = edge_design(router)
    s_idx, s_off = router.snap(inc.loc[tr_mask, "start_lat"].to_numpy(), inc.loc[tr_mask, "start_lon"].to_numpy())
    d_idx, d_off = router.snap(inc.loc[tr_mask, "dest_lat"].to_numpy(), inc.loc[tr_mask, "dest_lon"].to_numpy())
    X, _ = sm.design(router, Z, s_idx, d_idx, s_off + d_off, sm.edge_times(router, Z))
    ytr = y_all[tr_mask]
    r = np.abs(ytr - X @ sm.coef)
    hub = np.where(r <= 60.0, 1.0, 60.0 / np.maximum(r, 1e-9))
    rng = np.random.default_rng(0)
    boots = np.array([StreetSpeedModel.weighted_bounded_ls(X, ytr, hub * rng.poisson(1.0, len(ytr)))
                      for _ in range(args.boot)])
    table = sm.table(np.percentile(boots, [2.5, 97.5], axis=0))
    table.round(3).to_csv(OUT / "coefficients.csv", index=False)
    print(table.round(2).to_string(), flush=True)
    sm.save()
    del X, boots
    print("Street-speed base time for every trip…", flush=True)
    inc["street_base_s"] = sm.predict(router, inc["start_lat"], inc["start_lon"], inc["dest_lat"], inc["dest_lon"])
    plots(table, router, sm)
    ml_scorer._ROUTER = None
    del router, Z
    gc.collect()

    # ---- 2. LightGBM alone and corrections
    clip = lambda p: np.clip(p, *DRIVE_CLIP_S)  # noqa: E731
    s = Split(inc[tr_mask], inc[~tr_mask])
    del inc
    gc.collect()
    xtr, xte = s.x(FINAL_GROUPS)
    base_tr, base_te = s.tr["street_base_s"].to_numpy(float), s.te["street_base_s"].to_numpy(float)
    preds = {"kolesar": clip(s.kol.predict(s.te["road_km"].to_numpy(float))), "street_speeds": clip(base_te)}
    print("LightGBM alone…", flush=True)
    preds["lightgbm"] = clip(fit_one(xtr, s.ytr, params).predict(xte))
    xtr["street_base_s"], xte["street_base_s"] = base_tr, base_te
    print("Street speeds + LightGBM correction (restricted features)…", flush=True)
    mr = fit_one(xtr[RESTRICTED], s.ytr - base_tr, params)
    preds["combined_restricted"] = clip(base_te + mr.predict(xte[RESTRICTED]))
    print("Street speeds + LightGBM correction (all features)…", flush=True)
    mf = fit_one(xtr, s.ytr - base_tr, params)
    preds["combined"] = clip(base_te + mf.predict(xte))

    results = {k: metrics(s.yte, p) for k, p in preds.items()}
    bd = by_distance(s.yte, preds, s.te["road_km"].to_numpy(float))
    bd.to_csv(OUT / "by_distance.csv", index=False)
    imp = pd.Series(mf.booster_.feature_importance(importance_type="gain"), index=xtr.columns)
    summary = {"doc": __doc__, "n_train": int(len(s.ytr)), "n_test_2025": int(len(s.yte)),
               "street_speed_rounds": sm.history, "metrics_2025": results,
               "by_distance": bd.to_dict(orient="records"),
               "correction_importance_gain_share": (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict(),
               "restricted_features": RESTRICTED, "lightgbm_params": params,
               "minutes": round((time.time() - t0) / 60, 1)}
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print("\n2025 driving-time error:")
    for k, v in results.items():
        print(f"  {k:<22} MAE {v['mae_s']:6.2f}  MedAE {v['medae_s']:6.2f}  bias {v['bias_s']:7.2f}")
    print(bd.to_string(index=False))

    # ---- 3. save the full combined model in the planner's bundle format
    models = [mf] if args.seeds == 1 else fit_bag(xtr, s.ytr - base_tr, seeds=tuple(range(42, 42 + args.seeds)),
                                                  params=params)
    import joblib

    joblib.dump({
        "kind": "lightgbm_v3_streetspeed",
        "model": f"learned street speeds + LightGBM correction ({len(models)} seed(s))",
        "base": "street_speeds", "street_speeds_path": str(MODEL_PATH.relative_to(ROOT)),
        "models": models, "features": list(xtr.columns), "feature_groups": FINAL_GROUPS,
        "categorical": [c for c in CATEGORICAL if c in xtr.columns], "category_maps": s.cats,
        "borough_categories": s.cats["borough"],
        "dest_history": {"sizes_m": list(s.dest.sizes_m), "shrink": s.dest.shrink,
                         "tables": {str(k): v.to_dict() for k, v in s.dest.tables.items()}},
        "kolesar_baseline_params": s.kol.to_dict(), "kolesar_baseline_distance": "road_km",
        "lightgbm_params": params, "train_years": list(TRAIN_YEARS), "test_year": 2025,
        "metrics_test": results,
    }, COMBINED_PATH)
    print(f"Saved {MODEL_PATH.name}, {COMBINED_PATH.name}; {(time.time() - t0) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
