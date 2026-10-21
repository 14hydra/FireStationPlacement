#!/usr/bin/env python3
"""Progress figures for the week of 11-19 Oct 2026 (slide-ready PNGs).

Built only from saved results, so every number matches the docs:

  1_travel_time_accuracy.png   travel-time model error vs baselines (2025, held-out stations, 2014)
  2_price_of_fairness.png      recommended layouts: average and slowest-10 % gains, borough rules
  3_three_models.png           recommended layouts scored under all three travel-time models
  4_closure_2014.png           2014 closures: recorded vs predicted slowdown per area
  5_dispatch_time_of_day.png   nearest station sends the first engine, by time of day
  6_summary_table.png          what was built this week

Writes data/figures/reports/weekly_2026-10-19/.

Run:  PYTHONPATH=src .venv/bin/python scripts/plot_weekly_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "data" / "figures"
OUT = FIG / "reports/weekly_2026-10-19"

INK, MUTED, GRID = "#1f2933", "#6b7785", "#e3e7ec"
BLUE, ORANGE, GREEN, GREY, RED = "#2563eb", "#ea8a1b", "#16a34a", "#9aa5b1", "#dc2626"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 13,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def fig_accuracy():
    m = json.loads((ROOT / "data/processed/models/travel_time_lfb.metrics.json").read_text())
    mt = m["metrics_test"]
    pre = json.loads((FIG / "validation/closure_2014/summary.json").read_text())["pre2014_model_accuracy_on_2014_trips"]
    rows = [("Constant guess", mt["median"]["mae_s"], GREY),
            ("Straight line, 32 km/h", mt["crow@32kph"]["mae_s"], GREY),
            ("Road distance, fixed speed", next(v["mae_s"] for k, v in mt.items() if k.startswith("road@")), GREY),
            ("Kolesar formula (straight line)", mt["kolesar(crow_km)"]["mae_s"], ORANGE),
            ("Kolesar formula (road distance)", mt["kolesar(road_km)"]["mae_s"], ORANGE),
            ("LightGBM, stations never seen", m["new_station_check"]["lightgbm"]["mae_s"], BLUE),
            ("LightGBM decision trees", mt["lightgbm(final)"]["mae_s"], BLUE)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    y = np.arange(len(rows))
    a.barh(y, [r[1] for r in rows], color=[r[2] for r in rows])
    a.set_yticks(y, [r[0] for r in rows])
    for i, r in enumerate(rows):
        a.text(r[1] + 1, i, f"{r[1]:.1f} s", va="center", color=INK, fontsize=10)
    a.set_xlabel("Average error predicting driving time (seconds, lower is better)")
    a.set_title("Travel-time model: held-out 2025 trips")
    a.grid(axis="x", color=GRID)
    labels = ["LightGBM", "Kolesar\n(road)", "Kolesar\n(straight)"]
    vals = [pre["lightgbm"]["mae_s"], pre["kolesar(road_km)"]["mae_s"], pre["kolesar(crow_km)"]["mae_s"]]
    b.bar(labels, vals, color=[BLUE, ORANGE, ORANGE])
    for i, v in enumerate(vals):
        b.text(i, v + 1, f"{v:.1f} s", ha="center", color=INK, fontsize=10)
    b.set_ylim(0, max(vals) * 1.2)
    b.set_ylabel("Average error (s)")
    b.set_title("Trained before 2014, tested after\nthe 2014 closures (119k trips)")
    b.grid(axis="y", color=GRID)
    save(fig, "1_travel_time_accuracy.png")


def fig_fairness():
    rec = FIG / "placement/recommended"
    order = [("expand_3", "Add 3", "no rule"), ("expand_3_boroughs1s", "Add 3", "boroughs ≤ 1 s"),
             ("replace_10", "Move 10", "no rule"), ("replace_10_boroughs5s", "Move 10", "boroughs ≤ 5 s"),
             ("replace_10_boroughs1s", "Move 10", "boroughs ≤ 1 s"),
             ("replace_10_boroughs0s", "Move 10", "boroughs ≤ 0 s"),
             ("redesign", "Redesign", "no rule"), ("redesign_boroughs1s", "Redesign", "boroughs ≤ 1 s")]
    rows = []
    for d, sc, rule in order:
        s = json.loads((rec / d / "summary.json").read_text())
        rows.append({"label": f"{sc}\n{rule}", "avg": s["average_s"]["today"] - s["average_s"]["plan"],
                     "slow": s["slowest10_s"]["today"] - s["slowest10_s"]["plan"],
                     "worse": len(s["boroughs_worse_by_1s_or_more"]), "moves": len(s["opened"])})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["avg"], 0.4, color=BLUE, label="Average first-engine time")
    ax.bar(x + 0.2, df["slow"], 0.4, color=GREEN, label="Slowest 10 % of calls")
    for i, r in df.iterrows():
        col = RED if r["worse"] else GREEN
        txt = (f"{r['worse']} borough{'s' if r['worse'] > 1 else ''}\nworse" if r["worse"] else "no borough\nworse")
        ax.text(i, max(r["avg"], r["slow"]) + 0.6, txt,
                ha="center", fontsize=9, color=col)
    ax.set_xticks(x, df["label"], fontsize=10)
    ax.set_ylabel("Seconds faster than today")
    ax.set_xlabel("Scenario and borough rule (no borough's average may get worse than today by more than this)",
                  color=MUTED, fontsize=9)
    ax.set_ylim(0, df[["avg", "slow"]].max().max() * 1.25)
    ax.set_title("Recommended placements and the price of fairness (busy engines, LightGBM travel times)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=GRID)
    save(fig, "2_price_of_fairness.png")


def fig_three_models():
    d = pd.read_csv(FIG / "placement/layout_scoring/scores.csv")
    d = d[d["response"] == "dispatch"]
    layouts = [("lgbm_expand_3", "Add 3"), ("lgbm_replace_10", "Move 10"),
               ("lgbm_replace_10_boroughs1s", "Move 10\n(no borough\nworse)"),
               ("kolesar_replace_10", "Move 10\n(sites chosen\nby Kolesar)"),
               ("lgbm_redesign_boroughs1s", "Redesign\n(no borough\nworse)")]
    scorers = [("lgbm", "LightGBM", BLUE), ("road", "Kolesar (road)", ORANGE), ("kolesar", "Kolesar (straight)", GREY)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=False)
    for ax, (col, lo, hi, title) in zip(axes, [("d_mean_s", "d_mean_lo", "d_mean_hi", "Average first-engine time"),
                                               ("d_slowest10_s", "d_slowest10_lo", "d_slowest10_hi", "Slowest 10 % of calls")]):
        x = np.arange(len(layouts))
        for j, (sc, name, colr) in enumerate(scorers):
            vals = [-float(d[(d.layout == l) & (d.scorer == sc)][col].iloc[0]) for l, _ in layouts]
            err_lo = [-float(d[(d.layout == l) & (d.scorer == sc)][hi].iloc[0]) for l, _ in layouts]
            err_hi = [-float(d[(d.layout == l) & (d.scorer == sc)][lo].iloc[0]) for l, _ in layouts]
            ax.bar(x + (j - 1) * 0.27, vals, 0.27, color=colr, label=name,
                   yerr=[np.array(vals) - np.array(err_lo), np.array(err_hi) - np.array(vals)], capsize=2,
                   error_kw={"elinewidth": 0.8, "ecolor": INK})
        ax.set_xticks(x, [n for _, n in layouts], fontsize=9)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(title)
        ax.grid(axis="y", color=GRID)
    axes[0].set_ylabel("Seconds faster than today (2025 calls)")
    axes[0].legend(frameon=False, title="Scored with", fontsize=9)
    fig.suptitle("Every layout beats today's stations under all three travel-time models "
                 "(learned dispatch, 95 % bootstrap intervals)", fontsize=12, color=INK, y=1.02)
    save(fig, "3_three_models.png")


def fig_closure():
    diag = pd.read_csv(FIG / "validation/closure_2014/diagnostic.csv", index_col=0)
    disp = pd.read_csv(FIG / "validation/dispatch/closure_areas.csv", index_col=0)
    summary = json.loads((FIG / "validation/dispatch/summary.json").read_text())
    rec = diag["drive_s"]
    # Middle panel: best learned dispatch variant by area correlation (usually all-day).
    disp_cols = [c for c in ("dispatch_by_hour_calibrated", "dispatch_ensemble", "dispatch_all_day",
                             "dispatch_by_hour_call_rate") if c in disp.columns]
    scores = summary.get("closure_2014", {}).get("closure_score", {})
    best_name = max(disp_cols, key=lambda c: scores.get(c, {}).get("corr_across_areas", -1))
    best_pred = disp[best_name]
    best_label = {
        "dispatch_all_day": "Learned dispatch (all day)",
        "dispatch_by_hour_calibrated": "Learned dispatch (by time of day)",
        "dispatch_by_hour_call_rate": "Learned dispatch (by hour, call rate)",
        "dispatch_ensemble": "Learned dispatch (ensemble)",
    }.get(best_name, "Learned dispatch model")
    nearest_pred = disp["nearest"] if "nearest" in disp.columns else diag["model_nearest_drive"]
    panels = [(nearest_pred, "Assume nearest station responds", GREY),
              (best_pred, best_label, ORANGE),
              (diag["pred_lgbm_actual_station"], "Given the station that really responded", BLUE)]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), sharex=True, sharey=True)
    lim = (30, 130)
    for ax, (pred, title, colr) in zip(axes, panels):
        ax.plot(lim, lim, color=GRID, lw=1.6, zorder=0)
        err = (pred - rec).abs()
        # Soft residual colouring: closer to the diagonal = darker
        sizes = 48 + 28 * (err / max(float(err.max()), 1.0))
        ax.scatter(rec, pred, s=sizes, color=colr, alpha=0.92, zorder=2,
                   edgecolors="white", linewidths=0.6)
        shown = []
        for name in err.sort_values(ascending=False).index:
            px, py = float(rec[name]), float(pred[name])
            if any(abs(px - a) < 9 and abs(py - b) < 6 for a, b in shown):
                continue
            shown.append((px, py))
            ax.annotate(name, (px, py), fontsize=8.5, color=MUTED, xytext=(5, 3),
                        textcoords="offset points")
        r = float(np.corrcoef(rec, pred)[0, 1])
        mae = float(err.mean())
        ax.set_title(f"{title}\ncorr {r:.2f} · MAE {mae:.0f} s", fontsize=11)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Recorded extra driving time (s)")
        ax.grid(color=GRID, alpha=0.85)
    axes[0].set_ylabel("Predicted extra driving time (s)")
    fig.suptitle("2014 closures: extra driving time in each closed station's former area "
                 "(travel times + dispatch trained only on 2009–2013)", fontsize=12, color=INK, y=1.05)
    save(fig, "4_closure_2014.png")


def fig_time_of_day():
    t = pd.read_csv(FIG / "validation/dispatch/by_time_of_day.csv")
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(t))
    ax.plot(x, 100 * t["recorded_nearest"], "-o", color=INK, lw=2.4, label="Recorded (2025)")
    ax.plot(x, 100 * t["dispatch_by_hour_calibrated"], "--o", color=BLUE, label="Dispatch model, by time of day")
    ax.plot(x, 100 * t["dispatch_all_day"], ":o", color=GREY, label="Dispatch model, all day")
    ax.set_xticks(x, t["slot"])
    ax.set_ylabel("% of first engines from the nearest station")
    ax.set_ylim(55, 85)
    ax.annotate("lowest in the early afternoon\n(training, fire-safety visits),\nnot at the evening call peak",
                (3, 100 * t["recorded_nearest"][3]), xytext=(3.4, 58), fontsize=9, color=MUTED,
                arrowprops={"arrowstyle": "->", "color": MUTED})
    ax.set_title("When is the nearest fire station available?")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(color=GRID)
    save(fig, "5_dispatch_time_of_day.png")


def fig_table():
    rows = [
        ["Travel-time model", "LightGBM, 57.6 s error (Kolesar 68.7 s); 62.4 s on stations it never saw"],
        ["Placement solver", "Exact integer program (HiGHS) with warm starts and lower bounds; genetic algorithm"],
        ["Busy engines", "Erlang loss model: busy chance from each station's pumps and call load"],
        ["Pumps", "Move with their station; London keeps 142"],
        ["Street-level sites", "New stations snap to the best junction on a main road"],
        ["Fairness", "Slowest 10 %, average and every borough no worse than today"],
        ["Slow calls", "90th-percentile model and slowest-10 % objective"],
        ["Validation", "Three travel-time models + bootstrap intervals; 2014 closure test"],
        ["Dispatch model", "Learned which station responds; by time of day"],
        ["Headline", "Move up to 10, no borough worse: −2.5 to −5.5 s average, −3.6 to −4.9 s slowest 10 %"],
    ]
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=["Built this week", "What it does / result"], loc="center",
                   cellLoc="left", colWidths=[0.2, 0.8])
    tab.auto_set_font_size(False)
    tab.set_fontsize(10.5)
    tab.scale(1, 1.6)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif r == len(rows):
            cell.set_facecolor("#e8f0fe")
            cell.get_text().set_weight("bold")
    ax.set_title("Week of 11–19 October: fire station placement (London Fire Brigade)", pad=12)
    save(fig, "6_summary_table.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_accuracy, fig_fairness, fig_three_models, fig_closure, fig_time_of_day, fig_table):
        f()


if __name__ == "__main__":
    main()
