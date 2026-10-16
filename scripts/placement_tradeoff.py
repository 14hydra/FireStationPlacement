#!/usr/bin/env python3
"""Efficiency vs equal-coverage trade-off curve for station placement.

For each placement mode and severity scheme, the exact solver is run at several values of
alpha, where each area counts as (its weighted calls)**alpha:

  alpha = 1   fastest call-weighted average (p-median: busy areas count most)
  alpha = 0   fastest area average (every area with calls counts the same)

Each solution is scored on both measures with the severity weights:

  efficiency  E = call-weighted average first-engine time (s)
  equity      Q = average over areas, each area once (s)

Alpha = 1 is the proven best E and alpha = 0 the proven best Q for that mode, so the
curve runs between two optima. The scale puts every layout between today's stations and
those optima:

  efficiency score = 100 * (E_today - E) / (E_today - E_best)
  equity score     = 100 * (Q_today - Q) / (Q_today - Q_best)

0 = no better than today, 100 = the best possible for that measure in that mode, negative
= worse than today. A layout scoring above 0 on both beats today's stations on both.

Writes data/figures/placement_tradeoff/: tradeoff.csv, tradeoff.png, summary.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/placement_tradeoff.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firestation.cities import get_city, load_fire_stations, load_incidents  # noqa: E402
from firestation.fire_station_location import SEVERITY_SCHEMES, run_fire_station_plan  # noqa: E402

OUT = ROOT / "data" / "figures" / "placement_tradeoff"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
MODES = {
    "expand_3": {"mode": "expand", "add_stations": 3},
    "replace_10": {"mode": "replace", "replace_r": 10},
    "redesign": {"mode": "redesign"},
}
SEVERITY_LABEL = {"all": "All calls", "serious": "Fires + special services", "dwelling_fire": "Fires in homes"}
MODE_LABEL = {"expand_3": "Add 3 stations", "replace_10": "Move up to 10", "redesign": "Full redesign (102)"}
MODE_COLOR = {"expand_3": "#2a9d8f", "replace_10": "#e76f51", "redesign": "#264653"}


def run_one(houses, incidents, scenario: str, severity: str, alpha: float, scorer: str) -> dict:
    kw = dict(MODES[scenario])
    if kw["mode"] == "redesign":
        kw["n_houses"] = len(houses)
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_fire_station_plan(
            city_id="london", houses=houses, incidents=incidents, scorer=scorer,
            demand_years=[2024], eval_years=[2025], max_incident_rows=None,
            severity=severity, alpha=alpha, **kw,
        )
    s = res.summary
    cur, plan = s["baselines"]["current_fire_stations"], s["plan"]
    ev = s.get("eval") or {}
    return {
        "scenario": scenario, "severity": severity, "alpha": alpha,
        "E_today": cur["expected_response_s"], "Q_today": cur["area_mean_response_s"],
        "E": plan["expected_response_s"], "Q": plan["area_mean_response_s"],
        "worst_area_today": cur["worst_cell_response_s"], "worst_area": plan["worst_cell_response_s"],
        "areas_over_10min_today": cur["areas_over_threshold_share"],
        "areas_over_10min": plan["areas_over_threshold_share"],
        "within_6min_today": cur["lfb_first_engine"]["share_within_6min"],
        "within_6min": plan["lfb_first_engine"]["share_within_6min"],
        "eval2025_today_s": (ev.get("current") or {}).get("mean_s"),
        "eval2025_plan_s": (ev.get("plan") or {}).get("mean_s"),
        "n_opened": s["n_opened"], "n_closed": s["n_closed"],
        "opened": "; ".join(f"{r.lat:.4f},{r.lon:.4f}" for r in res.opened.itertuples()) if len(res.opened) else "",
        "closed": "; ".join(res.closed["site_name"].astype(str)) if len(res.closed) else "",
        "optimal": s["solver"]["optimal"], "gap": s["solver"]["gap"],
    }


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in df.groupby(["scenario", "severity"], sort=False):
        g = g.copy()
        e_best = g.loc[g["alpha"] == 1.0, "E"].iloc[0]
        q_best = g.loc[g["alpha"] == 0.0, "Q"].iloc[0]
        g["E_best"], g["Q_best"] = e_best, q_best
        g["efficiency_score"] = 100 * (g["E_today"] - g["E"]) / (g["E_today"] - e_best)
        g["equity_score"] = 100 * (g["Q_today"] - g["Q"]) / (g["Q_today"] - q_best)
        g["beats_today_on_both"] = (g["E"] < g["E_today"]) & (g["Q"] < g["Q_today"])
        out.append(g)
    return pd.concat(out, ignore_index=True)


def plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    sev = list(dict.fromkeys(df["severity"]))
    fig, axes = plt.subplots(1, len(sev), figsize=(5.2 * len(sev), 4.8))
    axes = np.atleast_1d(axes)
    for ax, sv in zip(axes, sev):
        d = df[df["severity"] == sv]
        today = d.iloc[0]
        for sc, g in d.groupby("scenario", sort=False):
            g = g.sort_values("alpha")
            ax.plot(g["E"], g["Q"], "-o", color=MODE_COLOR[sc], label=MODE_LABEL[sc], ms=5, lw=1.8)
            for r in g.itertuples():
                if r.alpha in (0.0, 1.0):
                    ax.annotate(f"α={r.alpha:g}", (r.E, r.Q), textcoords="offset points", xytext=(5, 4),
                                fontsize=8, color=MODE_COLOR[sc])
        ax.plot(today["E_today"], today["Q_today"], "*", color="#111", ms=14, label="Today's stations")
        ax.set_title(SEVERITY_LABEL[sv], fontsize=11)
        ax.set_xlabel("Call-weighted average time (s)  ← faster for most calls")
        ax.set_ylabel("Area average time (s)  ← fairer across areas")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Placement trade-off: efficiency vs equal coverage (exact solver, LightGBM travel times)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=",".join(MODES))
    ap.add_argument("--severities", default=",".join(SEVERITY_SCHEMES))
    ap.add_argument("--scorer", default="lgbm")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    spec = get_city("london")
    houses = load_fire_stations(spec)
    incidents = load_incidents(spec, max_rows=None, years=[2024, 2025])
    rows = []
    for sv in args.severities.split(","):
        for sc in args.scenarios.split(","):
            for a in ALPHAS:
                r = run_one(houses, incidents, sc, sv, a, args.scorer)
                rows.append(r)
                print(f"{sv:13s} {sc:10s} α={a:<4g} E={r['E']:.1f} (today {r['E_today']:.1f})  "
                      f"Q={r['Q']:.1f} (today {r['Q_today']:.1f})  optimal={r['optimal']}", flush=True)
    df = add_scores(pd.DataFrame(rows))
    df.to_csv(OUT / "tradeoff.csv", index=False)
    plot(df, OUT / "tradeoff.png")
    (OUT / "summary.json").write_text(json.dumps({
        "alphas": list(ALPHAS), "severity_schemes": SEVERITY_SCHEMES,
        "measures": {"E": "call-weighted average first-engine time (s), demand grid 2024",
                     "Q": "average over areas with calls, each area once (s)"},
        "scale": "score = 100*(today - layout)/(today - best); 0 = today, 100 = best possible in that mode",
        "rows": df.round(3).to_dict(orient="records"),
    }, indent=2) + "\n")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
