#!/usr/bin/env python3
"""Score megacity Move-10 layouts with Ricky's transferable trip-spread PDF.

Placement stays on Kolesar (point optimizer). This script adds a Monte Carlo check:
London 2024 drive/Kolesar residual ratios (rush × length bin) are applied to each
city's cell drive times so we can report a *distribution* of average savings and
P(within 6 minutes) — the advisor-style PDF use case for transfer cities.

Does not modify data/figures/final/.

Run:
  PYTHONPATH=src .venv/bin/python scripts/megacity_pdf_evaluate.py
  PYTHONPATH=src .venv/bin/python scripts/megacity_pdf_evaluate.py --city tokyo --refit
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

from firestation.megacities import get_megacity, iter_megacities, list_megacities  # noqa: E402
from firestation.transfer_pdf import (  # noqa: E402
    DEFAULT_PATH,
    fit_kolesar_ratio_pdf,
    load_pdf,
    save_pdf,
    score_layout_draws,
)

OUT = ROOT / "data" / "figures" / "megacities" / "pdf_eval"
LFB = ROOT / "data" / "processed" / "lfb_travel_training.csv"


def ensure_pdf(*, refit: bool) -> dict:
    if DEFAULT_PATH.exists() and not refit:
        return load_pdf()
    print("fitting London transfer PDF (Kolesar ratios on 2024)…", flush=True)
    trips = pd.read_csv(LFB, low_memory=False, usecols=lambda c: c in {
        "cal_year", "drive_s", "road_km", "hour_of_call", "pump_order",
        "deployed_from_location", "busy_flag",
    })
    bundle = fit_kolesar_ratio_pdf(trips)
    save_pdf(bundle)
    print(f"  saved {DEFAULT_PATH} (n_pdf={bundle['n_pdf']:,})", flush=True)
    return bundle


def eval_city(city_id: str, bundle: dict, *, n_draws: int, seed: int) -> dict | None:
    city = get_megacity(city_id)
    cells_path = city.fig_dir / "demand_cells.csv"
    if not cells_path.exists():
        print(f"{city_id}: no demand_cells.csv — skip", flush=True)
        return None
    cells = pd.read_csv(cells_path)
    need = {"response_s", "response_s_current", "drive_s", "weight"}
    if not need.issubset(cells.columns):
        print(f"{city_id}: cells missing {need - set(cells.columns)} — skip", flush=True)
        return None
    # No local hour of call on proxy demand — use midday (non-rush) + rush mix via fixed hour=12
    # and also report a rush-hour stress case (hour=8).
    n = len(cells)
    w = cells["weight"].to_numpy(float)
    today = cells["response_s_current"].to_numpy(float)
    plan = cells["response_s"].to_numpy(float)
    drive = cells["drive_s"].to_numpy(float)

    mid = score_layout_draws(
        today, plan, drive, w, np.full(n, 12.0), bundle, n_draws=n_draws, seed=seed
    )
    rush = score_layout_draws(
        today, plan, drive, w, np.full(n, 8.0), bundle, n_draws=n_draws, seed=seed + 1
    )
    out = {
        "city_id": city_id,
        "city_name": city.name,
        "midday": mid,
        "rush_hour": rush,
        "caveat": (
            "London transfer PDF ratios on Kolesar megacity point times; proxy demand; "
            "not validated on local fire attendance."
        ),
    }
    city.fig_dir.mkdir(parents=True, exist_ok=True)
    (city.fig_dir / "pdf_eval.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"{city_id}: point Δ {mid['point_kolesar']['delta_s']:+.1f} s | "
        f"PDF midday Δ {mid['pdf_transfer']['average_delta_s_p50']:+.1f} s "
        f"[{mid['pdf_transfer']['average_delta_s_p10']:+.1f}, "
        f"{mid['pdf_transfer']['average_delta_s_p90']:+.1f}] | "
        f"P(plan faster)={mid['pdf_transfer']['prob_plan_faster']:.2f}",
        flush=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", choices=list_megacities(), default=None)
    ap.add_argument("--refit", action="store_true", help="Refit London transfer PDF")
    ap.add_argument("--n-draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    bundle = ensure_pdf(refit=args.refit)
    rows = []
    for c in iter_megacities([args.city] if args.city else None):
        r = eval_city(c.id, bundle, n_draws=args.n_draws, seed=args.seed)
        if r is None:
            continue
        mid = r["midday"]["pdf_transfer"]
        pk = r["midday"]["point_kolesar"]
        rows.append(
            {
                "city_id": c.id,
                "city_name": c.name,
                "point_delta_s": pk["delta_s"],
                "pdf_delta_p50_s": mid["average_delta_s_p50"],
                "pdf_delta_p10_s": mid["average_delta_s_p10"],
                "pdf_delta_p90_s": mid["average_delta_s_p90"],
                "prob_plan_faster": mid["prob_plan_faster"],
                "within_6min_lift_mean": mid["within_6min_lift_mean"],
                "point_within_6min_lift": pk["within_6min_plan"] - pk["within_6min_today"],
            }
        )
    summary = {
        "doc": __doc__,
        "pdf_model": str(DEFAULT_PATH),
        "n_fit": bundle.get("n_fit"),
        "n_pdf": bundle.get("n_pdf"),
        "cities": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print("wrote", OUT / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
