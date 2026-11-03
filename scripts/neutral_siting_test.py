#!/usr/bin/env python3
"""A neutral test of the hypothesis: do street-aware travel times choose better sites?

Scoring a layout with one of the models it could have been chosen by favours that model,
and a new site has no recorded trips. So we make the sites real but hide them:

  1. Hide 20 of today's 102 stations (every station hidden once per split of 5 groups).
  2. Retrain LightGBM and refit both Kolesar curves without the hidden stations' trips
     (2021-24), so each hidden station looks like a brand new site to every model.
  3. Keep the other 82 open. Each model picks the best k of the 20 (k = 3, 5, 10) on 2024
     calls, with the project's solver and busy engines.
  4. Score each pick with recorded drive times from those stations (2025-26 trips from the
     home station, used by no model), which needs no model at all.

Recorded scorer: mean recorded drive per station and 0.9 km cell, plus the station's median
recorded turnout, from pairs with at least --min-trips trips. Trips are split in two: one half
decides which open station is fastest, the other gives its time, and then the reverse, so
noise cannot make a layout look better by always picking lucky low averages. Busy chances
are Erlang loss on the same recorded times. The first three open stations with recorded
trips can respond. A cell with no recorded open station is left out of every layout in that
comparison. Also reported with the learned dispatch model (unrecorded pairs filled from a
curve fitted on the same held-out trips).

Baselines: base 82 only; random picks; hindsight (chosen with the recorded times themselves:
a rough upper limit, slightly optimistic).

Test 2, pair order: on 2025-26 calls where engines came from two home stations, does each
model say which arrived faster? A station's trips are predicted by a model that never saw it.
Features that differ by engine order (pump order, busy flag) are set to the same values, as
in placement. "Close pairs" are those with road distances within 25 %.

Intervals: bootstrap over the hidden groups (pair order: over calls). Groups from different
splits share stations, so they are not fully independent and the group intervals are a little narrow.
Pair order uses the first split only, where each station is predicted once.

Writes data/figures/validation/neutral_siting/: picks.csv, scores.csv, pairs.csv, summary.json,
hidden_fold cache in folds/.

Run:  PYTHONPATH=src .venv/bin/python scripts/neutral_siting_test.py [--splits 4] [--summarize-only]
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_travel_time_lfb import FINAL_GROUPS, TRAIN_YEARS, TUNED_PARAMS, Split, fit_one  # noqa: E402

from firestation.busy_placement import ErlangBusyModel  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.dispatch import dispatch_response, load_dispatch  # noqa: E402
from firestation.facility import build_demand_grid, haversine_km  # noqa: E402
from firestation.fire_station_location import (  # noqa: E402
    busy_prob_vector,
    drop_far_incidents,
    plan_expand,
    recorded_station_turnout,
    snap_to_cells,
)
from firestation.kolesar import fit_kolesar  # noqa: E402
from firestation.ml_scorer import _lgbm_trips, cell_boroughs, slot_weights  # noqa: E402
from firestation.station_capacity import default_pump_table  # noqa: E402
from firestation.travel_time_model import CATEGORICAL, DRIVE_CLIP_S, attach_weather, load_weather, predict  # noqa: E402

OUT = ROOT / "data" / "figures" / "validation" / "neutral_siting"
DATA = ROOT / "data" / "processed" / "lfb_travel_training.csv"
METRICS = ROOT / "data" / "processed" / "models" / "travel_time_lfb.metrics.json"
KS = (3, 5, 10)
CHOOSERS = ("lgbm", "road", "straight")
EVAL_YEARS = (2025, 2026)
N_RANDOM = 100
LEVELS = 3
clip = lambda p: np.clip(p, *DRIVE_CLIP_S)  # noqa: E731


def hidden_groups(names: np.ndarray, splits: int, groups: int = 5) -> list[list[str]]:
    out = []
    for s in range(splits):
        perm = np.random.default_rng(100 + s).permutation(len(names))
        out += [sorted(names[perm[perm_idx::groups]]) for perm_idx in range(groups)]
    return out


# --------------------------------------------------------------------------- recorded scorer
class RecordedScorer:
    """Recorded attendance per station × cell, with cross-fitted order and time."""

    def __init__(self, trips: pd.DataFrame, names: np.ndarray, n_cells: int, min_trips: int):
        idx = {s: i for i, s in enumerate(names)}
        t = trips[trips["deployed_from_station"].isin(idx)]
        s = t["deployed_from_station"].map(idx).to_numpy()
        c = t["cell"].to_numpy()
        y = np.clip(t["drive_s"].to_numpy(float), *DRIVE_CLIP_S)
        half = pd.util.hash_array(t["incident_number"].astype(str).to_numpy(object)) % 2  # same call, same half
        turnout = t.groupby("deployed_from_station")["turnout_s"].median().reindex(names).to_numpy()
        self.turnout = turnout
        self.R = []
        for h in (0, 1):
            m = half == h
            n = np.bincount(s[m] * n_cells + c[m], minlength=len(names) * n_cells).astype(float)
            tot = np.bincount(s[m] * n_cells + c[m], weights=y[m], minlength=len(names) * n_cells)
            r = np.where(n >= min_trips, tot / np.maximum(n, 1), np.nan).reshape(len(names), n_cells)
            self.R.append(r + turnout[:, None])
        n = np.bincount(s * n_cells + c, minlength=len(names) * n_cells).astype(float)
        tot = np.bincount(s * n_cells + c, weights=y, minlength=len(names) * n_cells)
        self.full = (np.where(n >= min_trips, tot / np.maximum(n, 1), np.nan).reshape(len(names), n_cells)
                     + turnout[:, None])

    @staticmethod
    def _one(order: np.ndarray, value: np.ndarray, idx: np.ndarray, bm: ErlangBusyModel) -> np.ndarray:
        a = np.where(np.isfinite(order), order, 1e6)
        # Busy chances: load goes to the fastest recorded open station.
        p = bm.probs(a, idx)[idx]
        sub_o, sub_v = a[idx], value[idx]
        rank = np.argsort(sub_o, axis=0)[:LEVELS]
        cols = np.arange(a.shape[1])
        carry, total, weight = np.ones(a.shape[1]), np.zeros(a.shape[1]), np.zeros(a.shape[1])
        for lv in range(LEVELS):
            r = rank[lv]
            ok = (sub_o[r, cols] < 1e6) & np.isfinite(sub_v[r, cols])
            take = carry * (1 - p[r]) * ok if lv < LEVELS - 1 else carry * ok
            total += take * np.where(ok, sub_v[r, cols], 0.0)
            weight += take
            carry = carry * p[r]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(weight > 0, total / weight, np.nan)

    def response(self, idx, bm_a: ErlangBusyModel, bm_b: ErlangBusyModel) -> np.ndarray:
        idx = np.asarray(idx, int)
        a = self._one(self.R[0], self.R[1], idx, bm_a)
        b = self._one(self.R[1], self.R[0], idx, bm_b)
        return np.nanmean(np.stack([a, b]), axis=0)  # nan only if neither half can score the cell


# --------------------------------------------------------------------------- one hidden group
def train_group(inc: pd.DataFrame, hidden: list[str], params: dict, scenario: pd.DataFrame, slots, pair_trips: pd.DataFrame):
    """Fit all three choosers without the hidden stations; drive-time matrices and pair predictions."""
    keep = ~inc["deployed_from_station"].isin(hidden)
    tr = inc[inc["cal_year"].isin(TRAIN_YEARS) & keep]
    s = Split(tr, tr.head(10))
    xtr, _ = s.x(FINAL_GROUPS)
    model = fit_one(xtr, s.ytr, params)
    kol_crow = fit_kolesar(s.tr["crow_km"].to_numpy(float), s.ytr)
    bundle = {"kind": "lightgbm_hidden_group", "models": [model], "features": list(xtr.columns),
              "feature_groups": FINAL_GROUPS, "categorical": [c for c in CATEGORICAL if c in xtr.columns],
              "category_maps": s.cats, "borough_categories": s.cats["borough"],
              "dest_history": {"sizes_m": list(s.dest.sizes_m), "shrink": s.dest.shrink,
                               "tables": {str(k): t.to_dict() for k, t in s.dest.tables.items()}},
              "kolesar_baseline_params": s.kol.to_dict(), "kolesar_baseline_distance": "road_km",
              "drive_clip_s": list(DRIVE_CLIP_S), "seeds": [42]}
    T = np.zeros(len(scenario))
    for hour, w in slots:
        for a in range(0, len(scenario), 100_000):
            T[a:a + 100_000] += w * predict(bundle, scenario.iloc[a:a + 100_000].assign(hour_of_call=hour))
    road_km = scenario["road_km"].to_numpy(float)
    road_km = np.where(np.isfinite(road_km), road_km, scenario["crow_km"].to_numpy(float) * 1.4)  # as in road_matrix
    road = clip(s.kol.predict(road_km))
    crow = np.maximum(kol_crow.predict(scenario["crow_km"].to_numpy(float)), 15.0)
    p = pair_trips[pair_trips["deployed_from_station"].isin(hidden)].copy()
    pred = []
    for a in range(0, len(p), 100_000):
        pred.append(predict(bundle, p.iloc[a:a + 100_000]))
    pairs = p[["incident_number", "deployed_from_station", "drive_s", "road_km", "crow_km"]].copy()
    pairs["lgbm"] = np.concatenate(pred) if pred else []
    pairs["road"] = clip(s.kol.predict(p["road_km"].to_numpy(float)))
    pairs["straight"] = np.maximum(kol_crow.predict(p["crow_km"].to_numpy(float)), 15.0)
    return T, road, crow, pairs


# --------------------------------------------------------------------------- summary
def boot_groups(df: pd.DataFrame, value: str, *, n=5000, seed=0):
    """Mean and 95 % range across hidden groups (resampled)."""
    v = df.groupby("group")[value].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return round(float(v.mean()), 2), round(float(np.percentile(b, 2.5)), 2), round(float(np.percentile(b, 97.5)), 2)


def pair_table(pairs: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    """Pairs of engines from different home stations at the same call."""
    d = pairs.sort_values(["incident_number", "deployed_from_station"]).drop_duplicates(
        ["incident_number", "deployed_from_station"])
    a = d.merge(d, on="incident_number", suffixes=("_a", "_b"))
    a = a[a["deployed_from_station_a"] < a["deployed_from_station_b"]]
    a = a[a["drive_s_a"] != a["drive_s_b"]]
    ratio = a[["road_km_a", "road_km_b"]].max(axis=1) / a[["road_km_a", "road_km_b"]].min(axis=1).clip(lower=0.01)
    calls = pd.factorize(a["incident_number"])[0]
    rng = np.random.default_rng(0)
    rows = []
    for name, m in {"all pairs": np.ones(len(a), bool), "close pairs (road km within 25 %)": ratio.to_numpy() <= 1.25}.items():
        sub = a[m]
        c = calls[m]
        out = {"pairs": name, "n_pairs": int(m.sum()), "n_calls": int(len(np.unique(c)))}
        faster_a = sub["drive_s_a"].to_numpy() < sub["drive_s_b"].to_numpy()
        gap = np.abs(sub["drive_s_a"].to_numpy() - sub["drive_s_b"].to_numpy())
        right, ng = {}, c.max() + 1
        for chooser in CHOOSERS:
            pa, pb = sub[f"{chooser}_a"].to_numpy(), sub[f"{chooser}_b"].to_numpy()
            r = ((pa < pb) == faster_a).astype(float)
            r[pa == pb] = 0.5
            right[chooser] = r
            out[f"{chooser}_right_pct"] = round(float(r.mean() * 100), 2)
            out[f"{chooser}_lost_s"] = round(float(((1 - r) * gap).mean()), 2)
        counts = np.bincount(c, minlength=ng)
        wts = rng.poisson(1.0, (n_boot, ng))  # resample calls
        for other in ("road", "straight"):
            diff = np.bincount(c, weights=right["lgbm"] - right[other], minlength=ng)
            loss = np.bincount(c, weights=(right["lgbm"] - right[other]) * gap, minlength=ng)
            den = wts @ counts
            bd = (wts @ diff) / den * 100
            bl = (wts @ loss) / den
            out[f"lgbm_minus_{other}_pct_lo"] = round(float(np.percentile(bd, 2.5)), 2)
            out[f"lgbm_minus_{other}_pct_hi"] = round(float(np.percentile(bd, 97.5)), 2)
            out[f"lgbm_saves_vs_{other}_s"] = round(float(((right["lgbm"] - right[other]) * gap).mean()), 2)
            out[f"lgbm_saves_vs_{other}_lo"] = round(float(np.percentile(bl, 2.5)), 2)
            out[f"lgbm_saves_vs_{other}_hi"] = round(float(np.percentile(bl, 97.5)), 2)
        rows.append(out)
    return pd.DataFrame(rows)


def summarize() -> int:
    scores = pd.read_csv(OUT / "scores.csv")
    picks = pd.read_csv(OUT / "picks.csv")
    summary = {"doc": __doc__, "n_groups": int(scores["group"].nunique()), "by_k": {}}
    rows = []
    for k in KS:
        sk = scores[scores["k"] == k]
        wide = sk.pivot(index="group", columns="layout", values="gain_s").reset_index()
        wided = sk.pivot(index="group", columns="layout", values="gain_dispatch_s").reset_index()
        pk = picks[picks["k"] == k]
        overlap = []
        for g, gg in pk.groupby("group"):
            sets = {c: set(str(gg.loc[gg["layout"] == c, "stations"].iloc[0]).split("|")) for c in CHOOSERS}
            overlap.append({"group": g, "road": len(sets["lgbm"] & sets["road"]), "straight": len(sets["lgbm"] & sets["straight"])})
        ov = pd.DataFrame(overlap)
        one = {"k": k, "same_sites_as_lgbm_road": round(float(ov["road"].mean()), 2),
               "same_sites_as_lgbm_straight": round(float(ov["straight"].mean()), 2)}
        for L in ("lgbm", "road", "straight", "random", "hindsight"):
            if L in wide:
                one[f"{L}_gain"], one[f"{L}_lo"], one[f"{L}_hi"] = boot_groups(wide.assign(v=wide[L]), "v")
            if L in wided and L != "random":
                one[f"{L}_gain_dispatch"] = round(float(wided[L].mean()), 2)
        for other in ("road", "straight", "random"):
            d = wide.assign(v=wide["lgbm"] - wide[other])
            one[f"lgbm_minus_{other}"], one[f"lgbm_minus_{other}_lo"], one[f"lgbm_minus_{other}_hi"] = boot_groups(d, "v")
            one[f"lgbm_wins_vs_{other}"] = int((d["v"] > 1e-9).sum())
            one[f"ties_vs_{other}"] = int((d["v"].abs() <= 1e-9).sum())
            one[f"lgbm_losses_vs_{other}"] = int((d["v"] < -1e-9).sum())
            if other != "random":
                dd = wided.assign(v=wided["lgbm"] - wided[other])
                one[f"lgbm_minus_{other}_dispatch"], one[f"lgbm_minus_{other}_dispatch_lo"], one[f"lgbm_minus_{other}_dispatch_hi"] = boot_groups(dd, "v")
        rows.append(one)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary_by_k.csv", index=False)
    summary["by_k"] = rows
    summary["cells_scored_share"] = float(scores["share_calls_scored"].mean())
    parts = sorted((OUT / "folds").glob("pairs_*.parquet"))
    if parts:
        # Pairs must come from one complete split: there every station was predicted unseen once.
        first_split = [p for p in parts if int(p.stem.split("_")[1]) < 5]
        ptab = pair_table(pd.concat([pd.read_parquet(p) for p in first_split], ignore_index=True))
        ptab.to_csv(OUT / "pairs.csv", index=False)
        summary["pairs"] = ptab.to_dict(orient="records")
        print(ptab.T.to_string())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    pd.set_option("display.width", 220)
    print(table.T.to_string())
    return 0


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=int, default=4)
    ap.add_argument("--min-trips", type=int, default=3)
    ap.add_argument("--max-groups", type=int, default=None, help="Stop after this many groups (trial run)")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    (OUT / "folds").mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        return summarize()

    t0 = time.time()
    spec = get_city("london")
    houses = load_fire_stations(spec).reset_index(drop=True)
    names = houses["facilityname"].astype(str).to_numpy()
    inc24 = drop_far_incidents(load_incidents(spec, max_rows=None, years=[2024, 2025]), houses)
    dem, ev = inc24[inc24["cal_year"] == 2024], inc24[inc24["cal_year"] == 2025]
    cells = build_demand_grid(dem, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    sites = pd.DataFrame({"site_name": names, "lat": houses["lat"], "lon": houses["lon"], "is_existing": True})
    c25, _ = snap_to_cells(ev, cells)
    calls = np.bincount(c25, minlength=len(cells)).astype(float)
    demand_w = cells["n_incidents"].to_numpy(float)

    inc = pd.read_csv(DATA, low_memory=False)
    inc = attach_weather(inc, load_weather(ROOT / "data" / "processed" / "weather_hourly_london.csv"))
    inc = inc[inc["cal_year"].isin(TRAIN_YEARS + EVAL_YEARS)].reset_index(drop=True)
    floats = [c for c in inc.select_dtypes("float64").columns if c != "drive_s"]
    inc[floats] = inc[floats].astype("float32")

    held = inc[inc["cal_year"].isin(EVAL_YEARS) & (inc["deployed_from_location"] == "Home Station")].dropna(
        subset=["dest_lat", "dest_lon", "drive_s"]).reset_index(drop=True)
    cell_of, km = snap_to_cells(held, cells)
    scorer_trips = held.assign(cell=cell_of)[km <= 1.0]
    rec = RecordedScorer(scorer_trips, names, len(cells), args.min_trips)
    print(f"Recorded trips {len(scorer_trips):,}; pairs with >= {args.min_trips} trips: {np.isfinite(rec.full).sum():,}", flush=True)

    # Neutral fill for the dispatch model only: recorded drive by crow-flies km, from the held-out trips.
    bins = np.arange(0, 41, 0.5)
    by_km = scorer_trips.groupby(pd.cut(scorer_trips["crow_km"], bins), observed=False)["drive_s"].mean()
    mids = (bins[:-1] + bins[1:]) / 2
    ok = by_km.notna().to_numpy()
    crow = haversine_km(sites["lon"].to_numpy(float)[:, None], sites["lat"].to_numpy(float)[:, None],
                        cells["lon"].to_numpy(float)[None, :], cells["lat"].to_numpy(float)[None, :])
    fill = np.interp(crow, mids[ok], np.maximum.accumulate(by_km.to_numpy()[ok])) + rec.turnout[:, None]
    A_rec_filled = np.where(np.isfinite(rec.full), rec.full, fill)

    # Features that differ between engines at one call are set to the placement scenario.
    pair_trips = held.assign(pump_order=1, busy_flag=0)

    print("Routing all stations to all cells…", flush=True)
    boroughs = cell_boroughs(cells, dem)
    scenario = _lgbm_trips(sites, cells, boroughs)
    slots = slot_weights(dem)
    m = json.loads(METRICS.read_text())
    params = {**TUNED_PARAMS, "n_estimators": int(m["lightgbm_params"]["n_estimators"])}
    pumps = default_pump_table(houses).set_index("station")["n_pumps"].reindex(names).fillna(1).astype(int).to_numpy()
    measured = busy_prob_vector(sites, dem)
    turn_tab, turn_new = recorded_station_turnout([2024])
    norm = {k.upper(): v for k, v in turn_tab.items()}
    today = np.arange(len(names))
    dmodel = load_dispatch()

    def busy_for(A):
        bm = ErlangBusyModel.calibrate(A, today, measured, pumps, demand_w)
        bm.pumps = pumps.copy()
        return bm

    bm_rec = [busy_for(np.where(np.isfinite(R), R, 1e6)) for R in rec.R]
    bm_full = busy_for(A_rec_filled)
    groups = hidden_groups(names, args.splits)
    if args.max_groups:
        groups = groups[:args.max_groups]
    print(f"{len(groups)} hidden groups of about {len(groups[0])} stations", flush=True)

    all_scores, all_picks = [], []
    rng = np.random.default_rng(1)
    for gi, hidden in enumerate(groups):
        fp, fq = OUT / "folds" / f"group_{gi:02d}.json", OUT / "folds" / f"pairs_{gi:02d}.parquet"
        if fp.exists():
            saved = json.loads(fp.read_text())
            all_scores += saved["scores"]
            all_picks += saved["picks"]
            continue
        t1 = time.time()
        Tl, Tr, Ts, pairs = train_group(inc, hidden, params, scenario, slots, pair_trips)
        pairs.to_parquet(fq)
        H = np.array([np.flatnonzero(names == h)[0] for h in hidden])
        B = np.setdiff1d(today, H)
        # Choosers get today's recorded turnout for stations they know; hidden ones are new sites.
        turn = np.array([turn_new if i in set(H) else norm.get(names[i].upper(), turn_new) for i in range(len(names))])
        As = {c: T.reshape(len(names), len(cells)) + turn[:, None] for c, T in zip(CHOOSERS, (Tl, Tr, Ts))}
        As["hindsight"] = A_rec_filled
        picks: dict[tuple[str, int], np.ndarray] = {}
        for name, A in As.items():
            bm = busy_for(A)
            for k in KS:
                with contextlib.redirect_stdout(io.StringIO()):
                    r = plan_expand(A, demand_w, B, k, new_pool=H, objective="response_time", solver="exact",
                                    busy_model=bm, start_idx=list(B), floors={"max_rounds": 4, "time_limit_s": 120})
                picks[(name, k)] = np.array(r["opened"], int)
                all_picks.append({"group": gi, "layout": name, "k": k,
                                  "stations": "|".join(sorted(names[r["opened"]]))})
        # Scoring: only cells every layout in this group can score, weighted by 2025 calls.
        base_r = rec.response(B, *bm_rec)
        rnd = {k: [rng.choice(H, k, replace=False) for _ in range(N_RANDOM)] for k in KS}
        resp = {key: rec.response(np.r_[B, v], *bm_rec) for key, v in picks.items()}
        mask = np.isfinite(base_r) & (calls > 0)
        for r in resp.values():
            mask &= np.isfinite(r)
        share = float(calls[mask].sum() / calls.sum())
        def avg(r):
            return float((calls[mask] * r[mask]).sum() / calls[mask].sum())
        base_mean = avg(base_r)
        base_d = dispatch_response(A_rec_filled, B, bm_full, dmodel)
        def avg_d(r):
            return float((calls * r).sum() / calls.sum())
        rows = []
        for (name, k), v in picks.items():
            rows.append({"group": gi, "layout": name, "k": k, "gain_s": base_mean - avg(resp[(name, k)]),
                         "gain_dispatch_s": avg_d(base_d) - avg_d(dispatch_response(A_rec_filled, np.r_[B, v], bm_full, dmodel)),
                         "base_mean_s": base_mean, "share_calls_scored": share})
        for k in KS:
            g = [base_mean - avg(rec.response(np.r_[B, v], *bm_rec)) for v in rnd[k]]
            rows.append({"group": gi, "layout": "random", "k": k, "gain_s": float(np.mean(g)), "gain_dispatch_s": np.nan,
                         "base_mean_s": base_mean, "share_calls_scored": share})
        fp.write_text(json.dumps({"hidden": hidden, "scores": rows, "picks": [p for p in all_picks if p["group"] == gi]}))
        all_scores += rows
        del Tl, Tr, Ts, pairs, As
        gc.collect()
        print(f"group {gi + 1}/{len(groups)}: {time.time() - t1:.0f} s; k=5 gains " +
              ", ".join(f"{r['layout']} {r['gain_s']:.2f}" for r in rows if r["k"] == 5), flush=True)
    pd.DataFrame(all_scores).to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(all_picks).to_csv(OUT / "picks.csv", index=False)
    print(f"{(time.time() - t0) / 60:.0f} min", flush=True)
    return summarize()


if __name__ == "__main__":
    sys.exit(main())
