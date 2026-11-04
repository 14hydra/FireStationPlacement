#!/usr/bin/env python3
"""Five checks that inform placement. Uses the screening search from placement_methods.py, not the exact planner.

  q1  Does the model reproduce recorded calls? Modelled vs recorded attendance and responder by area and hour, 2025.
  q2  Does the gap follow the station, the area, the workload or the engines? Station-year panel, 10:00-16:00, 2021-25.
      Includes calls in 1 km squares shared by two station grounds, which compares stations in the same place.
  q3  Which savings hold up? Choose on each year 2021-24, score on all years. Per-move contribution, filtered on 2021-23 and tested on 2025.
  q4  Where could stations go? Local OSM extract: land use and road access per candidate site, then search again on allowed sites.
  q5  Do engines and buildings need to move together? Same formula for all layouts; moves chosen on 2024, scored on 2025.

Writes data/figures/placement/analysis/. Run: GAP_CACHE=<dir> PYTHONPATH=src .venv/bin/python scripts/placement_analysis.py q1 q2 q3 q4 q5 [--pool wide]
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from firestation.busy_placement import ErlangBusyModel, PumpPlan, erlang_b  # noqa: E402
from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.dispatch import candidates, dispatch_response, load_dispatch  # noqa: E402
from firestation.facility import TravelTimeSurrogate, build_demand_grid, posture_seconds, weighted_cvar  # noqa: E402
from firestation.fire_station_location import (build_house_candidates, build_travel_matrix, busy_prob_vector, drop_far_incidents,  # noqa: E402
                                               fit_scorer_kolesar, recorded_station_turnout, snap_to_cells)
from firestation.station_capacity import default_pump_table  # noqa: E402
from gap_windows import prepare  # noqa: E402
from placement_methods import POOLS, search  # noqa: E402

OUT = ROOT / "data" / "figures" / "placement" / "analysis"
CACHE = Path(os.environ.get("MATRIX_CACHE", ROOT / "data" / "processed" / "planner_cache"))
YEARS = (2021, 2022, 2023, 2024, 2025)
PBF = ROOT / "data" / "raw" / "london" / "greater-london-latest.osm.pbf"
MID = range(10, 16)


def setup(pool="standard"):
    spec = get_city("london")
    houses = load_fire_stations(spec)
    inc = drop_far_incidents(load_incidents(spec, max_rows=None, years=list(YEARS)), houses)
    print("calls by year", inc["cal_year"].value_counts().sort_index().to_dict(), flush=True)
    d24 = inc[inc["cal_year"] == 2024].reset_index(drop=True)
    cells = build_demand_grid(d24, lat_col="dest_lat", lon_col="dest_lon", cell_km=0.9)
    sites = build_house_candidates(houses, cells, **POOLS[pool])
    ex = np.flatnonzero(sites["is_existing"].to_numpy())
    tt, tn = recorded_station_turnout([2024])
    norm = {k.upper(): v for k, v in tt.items()}
    names = sites["site_name"].str.upper().to_numpy()
    turnout = np.array([norm.get(n, tn) if e else tn for n, e in zip(names, sites["is_existing"])])
    km, _, _ = fit_scorer_kolesar(d24, houses)
    As = {}
    for sc in ("lgbm", "road", "kolesar"):
        f = CACHE / f"analysis_{pool}_{sc}.npy"
        if f.exists():
            T = np.load(f)
        else:
            T, _ = build_travel_matrix(sites, cells, surrogate=TravelTimeSurrogate(), scorer=sc, kolesar=km, incidents=d24)
            np.save(f, T)
        As[sc] = T + turnout[:, None]
        print("matrix", sc, flush=True)
    pumps = default_pump_table(houses).set_index("station")["n_pumps"].reindex(houses["facilityname"].astype(str)).fillna(1).astype(int).to_numpy()
    base = np.ones(len(sites), int); base[ex] = pumps
    plan = PumpPlan(base, ex, sites["lat"].to_numpy(float), sites["lon"].to_numpy(float), extra="move")
    per = {}
    from pyproj import Transformer
    bng = Transformer.from_crs(27700, 4326, always_xy=True)
    for y in YEARS:
        d = inc[inc["cal_year"] == y].reset_index(drop=True)
        if d.empty:  # the planner file starts in 2024; earlier years come from the recorded first-engine table
            q = prepare(y, {})["calls"]
            q = q[(q["east"] > 0) & (q["north"] > 0)]
            lo, la = bng.transform(q["east"].to_numpy(float), q["north"].to_numpy(float))
            d = pd.DataFrame(dict(dest_lat=la, dest_lon=lo, station_ground=q["ground_name"].to_numpy(), deployed_from_station=q["first_name"].to_numpy(), travel_seconds=q["att"].to_numpy(float), hour_of_call=q["HourOfCall"].to_numpy()))
            d = drop_far_incidents(d, houses).reset_index(drop=True)
        c, _ = snap_to_cells(d, cells)
        per[y] = dict(d=d, c=c, n=np.bincount(c, minlength=len(cells)).astype(float), measured=busy_prob_vector(sites, d))
    return dict(houses=houses, cells=cells, sites=sites, ex=ex, As=As, plan=plan, base=base, per=per, pool=pool)


def busy(S, y, sc="lgbm"):
    p = S["per"][y]
    return ErlangBusyModel.calibrate(S["As"][sc], S["ex"], p["measured"], S["base"], p["n"], S["plan"])


def mean_at(S, idx, model, y, sc="lgbm"):
    n = S["per"][y]["n"]
    return float(n @ posture_seconds(S["As"][sc], idx, model.probs(S["As"][sc], idx)) / n.sum())


def q1(S):
    A, ex = S["As"]["lgbm"], list(S["ex"])
    dm = load_dispatch()
    rows, resid = [], {}
    for y in (2024, 2025):
        p = S["per"][y]; m = busy(S, 2024)
        bm = posture_seconds(A, ex, m.probs(A, ex))
        dr = dispatch_response(A, ex, m, dm)
        t = p["d"]["travel_seconds"].to_numpy(float)
        ok = np.isfinite(t) & (t > 0)
        rec = np.bincount(p["c"][ok], weights=t[ok], minlength=len(bm)); nn = np.bincount(p["c"][ok], minlength=len(bm)).astype(float)
        with np.errstate(all="ignore"):
            r = rec / nn - dr
        resid[y] = (r, nn)
        rows.append(dict(year=y, recorded=rec.sum() / nn.sum(), busy_model=nn @ bm / nn.sum(), dispatch_model=nn @ dr / nn.sum()))
    summ = dict(overall=rows)
    r, nn = resid[2025]; r0, n0 = resid[2024]
    use = nn >= 30
    mean_r = np.average(r[use], weights=nn[use])
    summ["cells_30plus"] = int(use.sum()); summ["share_calls_in_them"] = float(nn[use].sum() / nn.sum())
    summ["cell_weighted_mae"] = float(np.average(np.abs(r[use]), weights=nn[use]))
    summ["cell_weighted_sd_after_mean"] = float(np.sqrt(np.average((r[use] - mean_r) ** 2, weights=nn[use])))
    both = use & (n0 >= 30)
    summ["cell_residual_corr_2024_2025"] = float(np.corrcoef(r[both], r0[both])[0, 1])
    # sampling noise alone per cell, to compare with the spread
    d = S["per"][2025]["d"]; c = S["per"][2025]["c"]; t = d["travel_seconds"].to_numpy(float)
    sd = pd.Series(t).groupby(c).std().reindex(range(len(nn))).to_numpy()
    summ["cell_noise_sd"] = float(np.sqrt(np.nansum((sd[use] ** 2 / nn[use]) * nn[use]) / nn[use].sum()))
    dev = np.abs(r[use] - mean_r) * nn[use]; o = np.sort(dev)[::-1]
    summ["top10pct_cells_share_abs_error"] = float(o[: max(1, len(o) // 10)].sum() / o.sum())
    cc = S["cells"]; cen = np.hypot((cc["lat"].to_numpy() - 51.5074) * 111, (cc["lon"].to_numpy() + 0.1278) * 69)
    bands = pd.cut(cen, [0, 5, 10, 15, 20, 60], labels=["0-5", "5-10", "10-15", "15-20", "20+"])
    band = pd.DataFrame(dict(b=bands, r=r, n=nn))[nn >= 1]
    band = band.groupby("b", observed=True).apply(lambda g: pd.Series(dict(calls=g.n.sum(), resid=np.average(g.r, weights=g.n)))).reset_index()
    band.to_csv(OUT / "q1_by_distance_from_centre.csv", index=False)
    # responder
    m = busy(S, 2024); pb = m.probs(A, ex); pumps = np.maximum(m.pumps_for(ex), 1)
    rws, feats = candidates(A, ex, pb, pumps, dm.k)
    P = dm.proba(feats, A.shape[1])
    near = rws[0]
    names = S["sites"]["site_name"].str.upper().to_numpy()
    dep = d["deployed_from_station"].astype(str).str.upper().to_numpy()
    lookup = {n: i for i, n in enumerate(names)}
    di = np.array([lookup.get(x, -1) for x in dep]); known = di >= 0
    rec_near = (di[known] == near[c[known]])
    hit = (rws[:, c[known]] == di[known]).T
    Pcall = P[:, c[known]].T
    summ["responder_recorded_nearest"] = float(rec_near.mean())
    summ["responder_model_nearest"] = float(P[0, c[known]].mean())
    summ["responder_avg_prob_on_recorded_station"] = float((Pcall * hit).sum(axis=1).mean())
    summ["responder_recorded_in_model_list"] = float(hit.any(axis=1).mean())
    gr = pd.DataFrame(dict(g=d["station_ground"].astype(str).str.upper().to_numpy()[known], c=c[known], rec=rec_near, mod=P[0, c[known]]))
    # by hour
    hr = pd.to_numeric(d["hour_of_call"], errors="coerce").to_numpy()
    dr = dispatch_response(A, ex, m, dm)
    pr = pd.Series(np.where(hr < 7, "Night 00-07", np.where(hr < 10, "Morning 07-10", np.where(hr < 16, "Midday 10-16", np.where(hr < 20, "Evening 16-20", "Late 20-24")))))
    tb = pd.DataFrame(dict(p=pr, t=t, m=dr[c])).dropna()
    tb = tb[tb.t > 0]
    prep = prepare(2025, {})["calls"].copy()
    prep["p"] = pd.cut(prep["HourOfCall"], [-1, 6, 9, 15, 19, 23], labels=["Night 00-07", "Morning 07-10", "Midday 10-16", "Evening 16-20", "Late 20-24"]).astype(str)
    h = tb.groupby("p").agg(calls=("t", "size"), recorded=("t", "mean"), all_day_model=("m", "mean"))
    h["gap_share"] = prep.groupby("p")["cat"].apply(lambda s: (s == "real_gap").mean() * 100)
    h["recorded_without_gap"] = prep[prep.cat != "real_gap"].groupby("p")["att"].mean()
    h["recorded_prep"] = prep.groupby("p")["att"].mean()
    h["recorded_without_gap_or_empty"] = prep[~prep.cat.isin(["real_gap", "empty"])].groupby("p")["att"].mean()
    h.reset_index().round(2).to_csv(OUT / "q1_by_period.csv", index=False)
    summ["recorded_all_prep"] = float(prep["att"].mean()); summ["recorded_without_gap"] = float(prep[prep.cat != "real_gap"]["att"].mean())
    summ["recorded_without_gap_or_empty"] = float(prep[~prep.cat.isin(["real_gap", "empty"])]["att"].mean())
    # by station ground: error vs gap
    g = pd.DataFrame(dict(g=d["station_ground"].astype(str).str.upper(), t=t, m=dr[c])).dropna(); g = g[g.t > 0]
    gs = g.groupby("g").agg(calls=("t", "size"), rec=("t", "mean"), mod=("m", "mean"))
    gs["resid"] = gs.rec - gs["mod"]
    prep["G"] = prep["ground_name"].astype(str).str.upper()
    gs["gap_share"] = prep.groupby("G")["cat"].apply(lambda s: (s == "real_gap").mean() * 100)
    gs["empty_share"] = prep.groupby("G")["cat"].apply(lambda s: (s == "empty").mean() * 100)
    gs["model_nearest"] = gr.groupby("g")["mod"].mean() * 100; gs["recorded_nearest"] = gr.groupby("g")["rec"].mean() * 100
    gs = gs.dropna(); gs.round(2).to_csv(OUT / "q1_by_ground.csv")
    summ["ground_resid_sd"] = float(gs.resid.std()); summ["ground_resid_range"] = [float(gs.resid.min()), float(gs.resid.max())]
    summ["ground_corr_resid_gap"] = float(gs.resid.corr(gs.gap_share)); summ["ground_corr_resid_empty"] = float(gs.resid.corr(gs.empty_share))
    summ["ground_corr_nearest_model_vs_recorded"] = float(gs.model_nearest.corr(gs.recorded_nearest))
    X = np.column_stack([np.ones(len(gs)), gs.gap_share, gs.empty_share]); b = np.linalg.lstsq(X, gs.resid, rcond=None)[0]
    summ["ground_resid_on_gap_empty_R2"] = float(1 - ((gs.resid - X @ b) ** 2).sum() / ((gs.resid - gs.resid.mean()) ** 2).sum()); summ["ground_slope_gap_s_per_point"] = float(b[1])
    # chosen sites
    lay = pd.read_csv(ROOT / "data/figures/placement/methods/layouts/mean_10.csv")
    idx = np.flatnonzero(S["sites"]["site_name"].isin(lay["site_name"]) )
    if len(idx) == len(ex):
        p1 = np.array(ex)[A[ex].argmin(axis=0)]; p2 = idx[A[idx].argmin(axis=0)]
        ch = (p1 != p2) & (nn > 0); un = (p1 == p2) & (nn > 0)
        summ["move10_changed_cells"] = dict(calls_share=float(nn[ch].sum() / nn.sum()), resid=float(np.average(r[ch], weights=nn[ch])), resid_other=float(np.average(r[un], weights=nn[un])),
                                            mae=float(np.average(np.abs(r[ch & use]), weights=nn[ch & use])), mae_other=float(np.average(np.abs(r[un & use]), weights=nn[un & use])))
        hit2 = np.isin(g.index, []) if False else None
    (OUT / "q1_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float)); print(h.round(1)); print(band.round(1)); print(gs.sort_values("resid").round(1).iloc[list(range(5)) + list(range(-5, 0))])


def r2(y, X, w):
    X = np.column_stack(X); s = np.sqrt(w)
    b = np.linalg.lstsq(X * s[:, None], y * s, rcond=None)[0]
    e = y - X @ b; ym = np.average(y, weights=w)
    return 1 - np.sum(w * e * e) / np.sum(w * (y - ym) ** 2), b


def q2():
    rows, sq = [], []
    for y in YEARS:
        P = prepare(y, {}); c, m, pumps = P["calls"], P["mob"], P["pumps"]
        e = c[c["cat"].isin(["own_first", "real_gap"]) & c["HourOfCall"].isin(MID)].copy(); e["gap"] = (e.cat == "real_gap").astype(float)
        mobs = m.groupby(m["Resource_Code"].str[:3]).size()
        for s, g in e.groupby("ground"):
            allc = c[c.ground == s]
            rows.append(dict(year=y, st=s, sname=g.ground_name.iloc[0], n=len(g), gap=g.gap.mean() * 100, pumps=min(pumps.get(s, 1), 2), wl=mobs.get(s, np.nan) / max(pumps.get(s, 1), 1),
                             own_km=allc.own_km.mean(), east=allc.east.mean(), north=allc.north.mean(), calls=len(allc), empty=(allc.cat == "empty").mean() * 100))
        e["sq"] = (e.east // 1000).astype(int).astype(str) + "_" + (e.north // 1000).astype(int).astype(str); e["year"] = y
        sq.append(e[["sq", "ground", "gap", "year"]])
    t = pd.DataFrame(rows).dropna(subset=["wl", "own_km"]); t = t[t.n >= 30].reset_index(drop=True)
    t["d"] = np.hypot(t.east - 530000, t.north - 180000) / 1000; t["lwl"] = np.log(t.wl); t["lcalls"] = np.log(t.calls)
    y, w = t.gap.to_numpy(), t.n.to_numpy(float)
    yd = pd.get_dummies(t.year, drop_first=True).to_numpy(float); sd = pd.get_dummies(t.st, drop_first=True).to_numpy(float); one = np.ones(len(t))
    two = (t.pumps == 2).to_numpy(float)
    z = lambda v: ((v - v.mean()) / v.std()).to_numpy()
    sets = {"year": [one, yd], "year+engines": [one, yd, two], "year+workload": [one, yd, z(t.lwl)], "year+area": [one, yd, z(t.d), z(t.own_km), z(t.lcalls)],
            "year+engines+workload+area": [one, yd, two, z(t.lwl), z(t.d), z(t.own_km), z(t.lcalls)]}
    sets["year+station"] = [one, yd, sd]; sets["year+station+workload"] = [one, yd, sd, z(t.lwl)]
    tab = [dict(model=k, r2=r2(y, v, w)[0]) for k, v in sets.items()]
    # Out of year: predict each year's station gap from the other years.
    oos = {}
    for k in ("year+engines+workload+area", "year+station"):
        pred = np.zeros(len(t))
        for yy in YEARS:
            tr = (t.year != yy).to_numpy()
            yr = [] if k != "x" else []
            X = np.column_stack([one, two, z(t.lwl), z(t.d), z(t.own_km), z(t.lcalls)]) if "engines" in k else np.column_stack([one, sd])
            b = np.linalg.lstsq(X[tr] * np.sqrt(w[tr])[:, None], y[tr] * np.sqrt(w[tr]), rcond=None)[0]
            pred[~tr] = X[~tr] @ b
        dm = y - t.groupby("year").gap.transform("mean").to_numpy(); pm = pred - pd.Series(pred).groupby(t.year.values).transform("mean").to_numpy()
        oos[k] = float(np.corrcoef(dm, pm)[0, 1])
    # Within station: does the gap move when workload moves?
    def fe(tt):
        v = tt[["gap", "lwl", "calls"]].copy(); v["lcalls"] = np.log(v.calls)
        for col in ("gap", "lwl", "lcalls"):
            v[col] = v[col] - tt.groupby("st")[col if col != "lcalls" else "lwl"].transform("mean") if col != "lcalls" else v[col] - v[col].groupby(tt.st).transform("mean")
            v[col] = v[col] - v[col].groupby(tt.year).transform("mean")
        return float((tt.n * v.gap * v.lwl).sum() / (tt.n * v.lwl ** 2).sum())
    rng = np.random.default_rng(0); sts = t.st.unique(); gr = {s: g for s, g in t.groupby("st")}; bs = []
    for _ in range(500):
        b = pd.concat([gr[s].assign(st=i) for i, s in enumerate(rng.choice(sts, len(sts)))]); bs.append(fe(b))
    # Between stations
    m = t.groupby("st").apply(lambda g: pd.Series(dict(gap=np.average(g.gap, weights=g.n), lwl=g.lwl.mean(), own_km=g.own_km.mean(), d=g.d.mean(), east=g.east.mean(), north=g.north.mean(), pumps=g.pumps.max(), n=g.n.sum(), sname=g.sname.iloc[0])))
    m[["gap", "lwl", "own_km", "d", "east", "north", "pumps", "n"]] = m[["gap", "lwl", "own_km", "d", "east", "north", "pumps", "n"]].astype(float)
    # Year-to-year persistence, station gap
    wide = t.pivot(index="st", columns="year", values="gap"); pc = wide.corr().to_numpy()
    # neighbours
    xy = m[["east", "north"]].to_numpy(); D = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1)); np.fill_diagonal(D, 1e12)
    nb = np.argsort(D, axis=1)[:, :3]
    rr = r2(m.gap.to_numpy(), [np.ones(len(m)), (m.pumps == 2).astype(float), z(m.lwl), z(m.d), z(m.own_km)], m.n.to_numpy())[1]
    res = m.gap.to_numpy() - np.column_stack([np.ones(len(m)), (m.pumps == 2).astype(float), z(m.lwl), z(m.d), z(m.own_km)]) @ rr
    moran_raw = np.corrcoef(m.gap, m.gap.to_numpy()[nb].mean(axis=1))[0, 1]; moran_res = np.corrcoef(res, res[nb].mean(axis=1))[0, 1]
    # Same place, two stations (only 2021-24; 2025 kept out of the rate)
    a = pd.concat(sq); a = a[a.year <= 2024]
    cnt = a.groupby(["sq", "ground"]).gap.agg(["size", "mean"]).reset_index(); cnt = cnt[cnt["size"] >= 15]
    pairs = []
    for s, g in cnt.groupby("sq"):
        if len(g) < 2:
            continue
        g = g.sort_values("size", ascending=False).head(2); pairs.append((s, g.ground.iloc[0], g.ground.iloc[1], g["mean"].iloc[0] * 100, g["mean"].iloc[1] * 100, g["size"].iloc[0], g["size"].iloc[1]))
    pr = pd.DataFrame(pairs, columns=["sq", "a", "b", "ga", "gb", "na", "nb"])
    shared = set(pr.sq)
    out = a[~a.sq.isin(shared)].groupby("ground").gap.mean() * 100
    pr["oa"], pr["ob"] = pr.a.map(out), pr.b.map(out); pr = pr.dropna()
    pr["same"], pr["else"] = pr.ga - pr.gb, pr.oa - pr.ob
    wp = 1 / (1 / pr.na + 1 / pr.nb)
    fit = lambda q, wq: float((wq * q["same"] * q["else"]).sum() / (wq * q["else"] ** 2).sum())
    bs2 = []
    for _ in range(1000):
        i = rng.integers(0, len(pr), len(pr)); bs2.append(fit(pr.iloc[i], wp.iloc[i]))
    # pump changes
    pc2 = t.pivot(index="st", columns="year", values="pumps")
    ch = pc2[pc2.nunique(axis=1) > 1]; pumprows = []
    for s, r in ch.iterrows():
        g = t[t.st == s].set_index("year")
        pumprows.append(dict(st=s, sname=g.sname.iloc[0], pumps=r.to_dict(), gap=g.gap.round(1).to_dict()))
    # model away table check
    bs_ = pd.read_csv(ROOT / "data/figures/validation/gap_windows/by_station.csv").set_index("station")
    m["away"] = bs_["away_10_16_pct"]; k = m.dropna(subset=["away"])
    summ = dict(models=tab, out_of_year_corr=oos, within_slope_points_per_log_workload=fe(t), within_slope_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                workload_sd_within=float((t.lwl - t.groupby("st").lwl.transform("mean")).std()), year_pair_corr_mean=float(pc[np.triu_indices(len(pc), 1)].mean()),
                gap_station_sd=float(m.gap.std()), between=dict(corr_workload=float(m.gap.corr(m.lwl)), corr_d_centre=float(m.gap.corr(m.d)), corr_own_km=float(m.gap.corr(m.own_km)), one_engine=float(np.average(m.gap[m.pumps == 1], weights=m.n[m.pumps == 1])), two_engine=float(np.average(m.gap[m.pumps == 2], weights=m.n[m.pumps == 2]))),
                neighbour_corr_raw=float(moran_raw), neighbour_corr_after_controls=float(moran_res), same_place=dict(squares=len(pr), slope=fit(pr, wp), ci=[float(np.percentile(bs2, 2.5)), float(np.percentile(bs2, 97.5))], corr=float(pr["same"].corr(pr["else"]))),
                engine_changes=pumprows, stations=len(m))
    t.drop(columns=["east", "north"]).round(3).to_csv(OUT / "q2_station_year.csv", index=False); pr.round(2).to_csv(OUT / "q2_same_place.csv", index=False)
    (OUT / "q2_summary.json").write_text(json.dumps(summ, indent=1, default=str)); print(json.dumps(summ, indent=1, default=str))


def q3(S):
    A, ex, plan = S["As"]["lgbm"], S["ex"], S["plan"]; today = list(ex); tag = S["pool"]
    models = {y: busy(S, y) for y in YEARS}
    today_v = {y: mean_at(S, today, models[y], y) for y in YEARS}
    rows, sets = [], {}
    for cy in (2021, 2022, 2023, 2024):
        idx = today
        for k in (10, 20, 40):
            idx, _ = search(A, A, S["per"][cy]["n"], models[cy], idx, ex, k)
            sets[(cy, k)] = list(idx)
            for ey in YEARS:
                rows.append(dict(chosen_on=cy, limit=k, scored_on=ey, moved=len(set(ex.tolist()) - set(idx)), saved=mean_at(S, idx, models[ey], ey) - today_v[ey]))
            print(rows[-1], flush=True)
    r = pd.DataFrame(rows); r.round(3).to_csv(OUT / f"q3_{tag}_year_grid.csv", index=False)
    ov = []
    for k in (10, 20, 40):
        for a in (2021, 2022, 2023, 2024):
            for b in (2021, 2022, 2023, 2024):
                if a < b:
                    oa, ob = set(sets[(a, k)]) - set(ex.tolist()), set(sets[(b, k)]) - set(ex.tolist()); ca, cb = set(ex.tolist()) - set(sets[(a, k)]), set(ex.tolist()) - set(sets[(b, k)])
                    ov.append(dict(limit=k, a=a, b=b, opened_same=len(oa & ob), closed_same=len(ca & cb), of=len(oa)))
    pd.DataFrame(ov).to_csv(OUT / f"q3_{tag}_overlap.csv", index=False)
    summ = dict(overlap=pd.DataFrame(ov).groupby("limit")[["opened_same", "closed_same", "of"]].mean().round(2).to_dict("index"))
    names = S["sites"]["site_name"].to_numpy()
    for k in (10, 20, 40):
        idx = sets[(2024, k)]; mv = plan.assign(idx)[1]; c = []
        full = {y: mean_at(S, idx, models[y], y) for y in YEARS}
        for x in mv:
            back = [i for i in idx if i != x["to"]] + [x["from"]]
            d = dict(frm=names[x["from"]], to=names[x["to"]], km=x["km"], pumps=x["pumps"], i=x["from"], j=x["to"])
            for y in YEARS:
                d[f"s{y}"] = mean_at(S, back, models[y], y) - full[y]
            c.append(d)
        c = pd.DataFrame(c)
        c["worst_2021_23"] = c[["s2021", "s2022", "s2023"]].min(axis=1); c["mean_2021_24"] = c[["s2021", "s2022", "s2023", "s2024"]].mean(axis=1)
        c.sort_values("s2024", ascending=False).round(3).to_csv(OUT / f"q3_{tag}_moves_{k}.csv", index=False)
        def apply(sub):
            lay = list(today)
            for _, q in sub.iterrows():
                lay = [i for i in lay if i != q["i"]] + [int(q["j"])]
            return lay
        res = {}
        keep = c[(c.s2021 > 0) & (c.s2022 > 0) & (c.s2023 > 0)]
        res["kept_positive_2021_23"] = dict(n=len(keep), of=len(c), s2025_full=today_v[2025] - full[2025], s2025_kept=today_v[2025] - mean_at(S, apply(keep), models[2025], 2025), lost=[f"{a}->{b}" for a, b in zip(c[~c.index.isin(keep.index)].frm, c[~c.index.isin(keep.index)].to)])
        cum = []
        o = c.sort_values("mean_2021_24", ascending=False)
        for n in range(1, len(o) + 1):
            cum.append(dict(top_moves=n, s2025=today_v[2025] - mean_at(S, apply(o.head(n)), models[2025], 2025)))
        pd.DataFrame(cum).round(3).to_csv(OUT / f"q3_{tag}_top_moves_{k}.csv", index=False)
        res["cum"] = cum; res["corr_contribution_2024_2025"] = float(c.s2024.corr(c.s2025)); res["min_contrib_2025"] = float(c.s2025.min()); res["negative_2025"] = int((c.s2025 < 0).sum())
        summ[f"k{k}"] = res
    p = r.assign(same=r.chosen_on == r.scored_on)
    summ["self_vs_other"] = p[p.scored_on <= 2024].groupby(["limit", "same"]).saved.mean().round(2).unstack().to_dict("index")
    summ["on_2025_by_choice_year"] = r[r.scored_on == 2025].pivot(index="chosen_on", columns="limit", values="saved").round(2).to_dict("index")
    (OUT / f"q3_{tag}_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


HARD = {"landuse": ["cemetery", "recreation_ground", "forest", "meadow", "village_green", "grass", "allotments", "farmland", "farmyard", "military", "reservoir", "basin", "landfill", "railway", "quarry", "orchard"],
        "leisure": ["park", "nature_reserve", "golf_course", "pitch", "garden", "common", "stadium", "playground", "sports_centre", "track"],
        "natural": ["water", "wood", "wetland", "scrub", "heath", "grassland"], "amenity": ["grave_yard", "school", "hospital", "university", "college"], "aeroway": ["aerodrome"]}
PLOT = ["commercial", "industrial", "retail", "brownfield", "construction", "garages"]


def land_layers():
    import geopandas as gpd, pyogrio
    f = OUT / "osm_land.parquet"
    if f.exists():
        return gpd.read_parquet(f)
    g = pyogrio.read_dataframe(PBF, layer="multipolygons", columns=["landuse", "leisure", "natural", "amenity", "aeroway"], where="landuse IS NOT NULL OR leisure IS NOT NULL OR natural IS NOT NULL OR amenity IS NOT NULL OR aeroway IS NOT NULL")
    g = g.to_crs(27700); g["cls"] = None
    for col, vals in HARD.items():
        g.loc[g[col].isin(vals), "cls"] = "excluded"
    g.loc[g["landuse"].isin(PLOT) & g["cls"].isna(), "cls"] = "plot"
    g.loc[g["landuse"].eq("residential") & g["cls"].isna(), "cls"] = "residential"
    g = g[g.cls.notna()].copy(); g["geometry"] = g.buffer(0); g["detail"] = g[["landuse", "leisure", "natural", "amenity", "aeroway"]].bfill(axis=1).iloc[:, 0]
    g = g[["cls", "detail", "geometry"]]; g.to_parquet(f); return g


def classify(lat, lon, names=None):
    import geopandas as gpd, pyogrio, shapely
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(27700)
    land = land_layers()
    t = pd.DataFrame(dict(excluded=False, detail=""), index=range(len(pts)))
    ex = land[land.cls == "excluded"]; tree = ex.sindex
    a, b = tree.query(pts.values, predicate="intersects"); t.loc[a, "excluded"] = True; t.loc[a, "detail"] = ex.detail.iloc[b].values
    res = land[land.cls == "residential"]; a, _ = res.sindex.query(pts.values, predicate="intersects"); t["residential"] = False; t.loc[a, "residential"] = True
    pl = land[(land.cls == "plot") & (land.area >= 2000)]
    t["plot_300m"] = False; a, _ = pl.sindex.query(pts.values, predicate="dwithin", distance=300); t.loc[a, "plot_300m"] = True
    ro = pyogrio.read_dataframe(PBF, layer="lines", columns=["highway"], where="highway IN ('trunk','trunk_link','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','unclassified','residential','living_street','service')").to_crs(27700)
    main = ro[ro.highway.isin(["trunk", "primary", "secondary", "tertiary"])]
    drive = ro[ro.highway != "service"]
    t["road_100m"] = False; a, _ = drive.sindex.query(pts.values, predicate="dwithin", distance=100); t.loc[a, "road_100m"] = True
    t["main_road_300m"] = False; a, _ = main.sindex.query(pts.values, predicate="dwithin", distance=300); t.loc[a, "main_road_300m"] = True
    return t


def q4(S):
    sites, A0, ex0 = S["sites"], S["As"]["lgbm"], S["ex"]; tag = S["pool"]
    t = classify(sites["lat"].to_numpy(float), sites["lon"].to_numpy(float))
    e = sites["is_existing"].to_numpy(bool)
    t["rule_a"] = ~t.excluded & t.road_100m & t.main_road_300m
    t["rule_b"] = t.rule_a & t.plot_300m
    t["rule_c"] = t.rule_a & t.plot_300m & ~t.residential
    out = pd.concat([sites[["site_name", "lat", "lon", "is_existing"]].reset_index(drop=True), t], axis=1); out.to_csv(OUT / f"q4_{tag}_sites.csv", index=False)
    new = t[~e]
    summ = dict(sites=len(t), new=int((~e).sum()), existing_flagged_excluded=int(t[e].excluded.sum()), new_excluded=int(new.excluded.sum()), new_no_road_100=int((~new.road_100m).sum()), new_no_main_300=int((~new.main_road_300m).sum()), rule_a=int(new.rule_a.sum()), rule_b=int(new.rule_b.sum()), rule_c=int(new.rule_c.sum()), detail=new[new.excluded].detail.value_counts().head(8).to_dict())
    # Recommended and screened layouts
    chk = []
    for label, p in [("official Move 10 (1 s)", ROOT / "data/figures/placement/recommended/replace_10_boroughs1s/opened_fire_stations.csv"), ("official Add 3 (1 s)", ROOT / "data/figures/placement/recommended/expand_3_boroughs1s/opened_fire_stations.csv"), ("official redesign (1 s)", ROOT / "data/figures/placement/recommended/redesign_boroughs1s/opened_fire_stations.csv")]:
        o = pd.read_csv(p); c = classify(o.lat.to_numpy(float), o.lon.to_numpy(float))
        o = pd.concat([o[["site_name", "lat", "lon"]], c], axis=1); o["label"] = label; chk.append(o)
    chk = pd.concat(chk); chk.to_csv(OUT / "q4_official_sites.csv", index=False)
    summ["official"] = chk.groupby("label").agg(sites=("excluded", "size"), excluded=("excluded", "sum"), no_road=("road_100m", lambda x: int((~x).sum())), no_main=("main_road_300m", lambda x: int((~x).sum())), plot_300=("plot_300m", "sum"), residential=("residential", "sum")).to_dict("index")
    # Search on allowed sites
    rows = []; layouts = {}
    n24, n25 = S["per"][2024]["n"], S["per"][2025]["n"]
    dm = load_dispatch()
    for rule in ("none", "rule_a", "rule_b", "rule_c"):
        keep = np.flatnonzero(e | (t[rule].to_numpy() if rule != "none" else True))
        As = {k: v[keep] for k, v in S["As"].items()}
        ex = np.flatnonzero(e[keep])
        plan = PumpPlan(S["base"][keep], ex, sites["lat"].to_numpy(float)[keep], sites["lon"].to_numpy(float)[keep], extra="move")
        ms = {k: ErlangBusyModel.calibrate(v, ex, S["per"][2024]["measured"][keep], S["base"][keep], n24, plan) for k, v in As.items()}
        idx = list(ex)
        base = {k: n25 @ posture_seconds(As[k], ex, ms[k].probs(As[k], ex)) / n25.sum() for k in As}
        for lim in (5, 10, 20, 30, 40, 102):
            idx, _ = search(As["lgbm"], As["lgbm"], n24, ms["lgbm"], idx, ex, lim)
            d = dict(rule=rule, limit=lim, pool_new=int(len(keep) - len(ex)), moved=len(set(ex.tolist()) - set(idx)))
            for k in As:
                d[k] = n25 @ posture_seconds(As[k], idx, ms[k].probs(As[k], idx)) / n25.sum() - base[k]
            r = posture_seconds(As["lgbm"], idx, ms["lgbm"].probs(As["lgbm"], idx)); r0 = posture_seconds(As["lgbm"], ex, ms["lgbm"].probs(As["lgbm"], ex))
            d["slowest10"] = weighted_cvar(r, n25) - weighted_cvar(r0, n25)
            d["dispatch"] = n25 @ (dispatch_response(As["lgbm"], idx, ms["lgbm"], dm) - dispatch_response(As["lgbm"], ex, ms["lgbm"], dm)) / n25.sum()
            rows.append(d); print(d, flush=True)
            pd.DataFrame(rows).round(3).to_csv(OUT / f"q4_{tag}_search.csv", index=False)
            if lim == 10:
                sites.iloc[keep[idx]][["site_name", "lat", "lon", "is_existing"]].to_csv(OUT / f"q4_{tag}_{rule}_move10.csv", index=False)
    (OUT / f"q4_{tag}_summary.json").write_text(json.dumps(summ, indent=1, default=float)); print(json.dumps(summ, indent=1, default=float))


def q5(S):
    A, ex, plan = S["As"]["lgbm"], S["ex"], S["plan"]; lay = ROOT / "data/figures/placement/methods/layouts"
    m = busy(S, 2024); ep = m.erlangs_per_call; n24, n25 = S["per"][2024]["n"], S["per"][2025]["n"]
    names = S["sites"]["site_name"].to_numpy()
    def geo(idx):
        idx = np.asarray(idx); sub = A[idx]; two = np.argpartition(sub, 1, axis=0)[:2]; cols = np.arange(sub.shape[1])
        sw = sub[two[0], cols] > sub[two[1], cols]
        i1, i2 = np.where(sw, two[1], two[0]), np.where(sw, two[0], two[1])
        return idx, i1, sub[i1, cols], sub[i2, cols]
    def val(idx, pum, w):
        idx, i1, t1, t2 = geo(idx); load = ep[idx] * np.bincount(i1, weights=n24, minlength=len(idx))
        p = erlang_b(load, np.maximum(pum, 1)); return float(w @ ((1 - p[i1]) * t1 + p[i1] * t2) / w.sum())
    def greedy(idx, pum, moves):
        idx, i1, t1, t2 = geo(idx); S_ = len(idx); pum = pum.copy(); load = ep[idx] * np.bincount(i1, weights=n24, minlength=S_); gain = np.bincount(i1, weights=n24 * (t2 - t1), minlength=S_); log = []
        for _ in range(moves):
            now = gain * erlang_b(load, pum); lose = gain * erlang_b(load, np.maximum(pum - 1, 1)) - now; win = gain * erlang_b(load, pum + 1) - now
            dn, tk = np.flatnonzero(pum >= 2), np.flatnonzero(pum < 3)
            d, u = dn[lose[dn].argmin()], tk[win[tk].argmin()]
            if lose[d] + win[u] >= 0 or d == u:
                break
            pum[d] -= 1; pum[u] += 1; log.append((names[idx[d]], names[idx[u]]))
        return pum, log
    rows = []; today = np.array(ex); p0 = plan.assign(today)[0][today]; v0 = val(today, p0, n25)
    def add(label, idx, pum, stations, engines):
        rows.append(dict(option=label, stations_moved=stations, extra_engine_moves=engines, engines_total=int(pum.sum()), saved_2025=val(idx, pum, n25) - v0, saved_2024=val(idx, pum, n24) - val(today, p0, n24)))
    add("today", today, p0, 0, 0)
    logs = {}
    for e in (1, 2, 3, 5, 10, 20):
        pu, log = greedy(today, p0, e); add(f"engines only, {e}", today, pu, 0, len(log)); logs[e] = log
    for k in (5, 10, 20):
        f = pd.read_csv(lay / f"mean_{k}.csv"); idx = np.flatnonzero(S["sites"]["site_name"].isin(f["site_name"]))
        if len(idx) != len(ex):
            print("skip", k, len(idx)); continue
        pm = plan.assign(idx)[0][idx]
        add(f"Move {k}, engines go with building", idx, pm, k, 0)
        # Building moves, second engine stays near: new site gets one engine, the spare goes to the nearest kept one-engine site... as a test of pairing
        moves = plan.assign(idx)[1]; pl = pm.copy(); pos = {int(i): a for a, i in enumerate(idx)}; lat, lon = S["sites"]["lat"].to_numpy(float), S["sites"]["lon"].to_numpy(float)
        for mv in moves:
            if mv["pumps"] >= 2:
                j = pos[mv["to"]]; pl[j] = 1
                kept = np.array([a for a, i in enumerate(idx) if i in set(ex.tolist()) and pl[a] < 2])
                dd = np.hypot((lat[idx[kept]] - lat[mv["from"]]) * 111, (lon[idx[kept]] - lon[mv["from"]]) * 69); pl[kept[dd.argmin()]] += 1
        add(f"Move {k}, new site 1 engine, spare to nearest kept station", idx, pl, k, 0)
        for e in (5, 10):
            pu, log = greedy(idx, pm, e); add(f"Move {k} + {e} engine moves", idx, pu, k, len(log))
    r = pd.DataFrame(rows).round(3); r.to_csv(OUT / "q5_engines.csv", index=False); print(r.to_string())
    pd.DataFrame(logs.get(10, []), columns=["from", "to"]).to_csv(OUT / "q5_top_engine_moves.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("steps", nargs="+"); ap.add_argument("--pool", default="standard"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); pd.set_option("display.width", 220, "display.max_columns", 30)
    S = setup(a.pool) if set(a.steps) - {"q2"} else None
    for s in a.steps:
        print("=====", s, flush=True); {"q1": lambda: q1(S), "q2": q2, "q3": lambda: q3(S), "q4": lambda: q4(S), "q5": lambda: q5(S)}[s]()
