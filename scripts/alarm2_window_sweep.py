"""Alarm 2: is the surviving Regime 4.9 excess a window-rule artifact?

The inherited specification is ``T = max(N, 250)``.  On S&P that is T = N = 357,
so q = N/T = 1.000 exactly -- the singular boundary.  The rule descends from
arXiv:1203.6228 §6, which fixes T = N for its own measurement, and it is the
*minimum* window satisfying the rank floor T >= N rather than a good one.  The
same paper's §6 reports an optimal S&P window of 700 days and this project's own
Regime 4.3 measured argmin D_emp at 999 days, so every real-data regime has run
at roughly a third of the window length the project itself identified.

Changing T at fixed N moves five things simultaneously: q, the deletion fraction
h/T, the origin count, the independent-block count, and the economic span of the
window.  A single alternative T therefore cannot separate "the excess was a
conditioning artifact" from "a longer window is a different, more persistent
target".  Two arms are required.

    Arm A   h = 42 fixed, T swept.  The predeclared horizon; q and h/T both move.
            This is the operational robustness check.
    Arm B   h/T held near 0.112, T swept (h scaled, kept a multiple of step).
            Isolates q with the deletion geometry fixed, at the price of no
            longer forecasting the predeclared 42 days.

Read: falls in A but holds in B -> h/T, the result stands.  Falls in both -> q,
genuine artifact.  Holds in both -> N, and the headline is robust.

Every cell reuses ``scripts.regime4_9_deletion_attribution`` unchanged, so the
methodology is identical to the published run by construction.  Only the
volatility-matched null is run: the calendar null is not the binding control.
Null permutations are drawn from ``SeedSequence([seed, 492, replicate])`` against
the same unpermuted panel in every cell, so the cells share common random
numbers and the *differences* between them carry less noise than their levels.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime4_9_deletion_attribution import (
    deletion_attribution_series, flag_histories, summarise)
from scripts.regime4_9_deletion_attribution import main as regime4_9_main
from scripts.regime4_4_tangent import empirical_upper_p
from src.data import to_panel


PRIMARY = "flag_nested"
# (arm, T, horizon).  T=357/h=42 is the published cell and is shared by both
# arms; it is listed once and labelled "A+B" in the collected table.
CELLS = (
    ("A+B", 357, 42),
    ("A", 500, 42),
    ("A", 750, 42),
    ("A", 1008, 42),
    ("B", 500, 56),
    ("B", 750, 84),
    ("B", 1008, 112),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="sp500_full")
    parser.add_argument("--indir", type=Path, default=Path("data/cache"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("results/alarm2_window_sweep"))
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--shuffles", type=int, default=99)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--mode", choices=("observed", "full", "collect"),
                        default="observed",
                        help="observed = no nulls (fast); full = run the "
                             "volatility null in every cell; collect = read "
                             "existing outputs and build the curves")
    parser.add_argument("--cells", default="all",
                        help="comma-separated T:h pairs, or 'all'")
    return parser.parse_args(argv)


def selected_cells(spec):
    if spec == "all":
        return CELLS
    wanted = set()
    for token in spec.split(","):
        T, horizon = token.split(":")
        wanted.add((int(T), int(horizon)))
    chosen = tuple(cell for cell in CELLS if (cell[1], cell[2]) in wanted)
    if len(chosen) != len(wanted):
        raise SystemExit(f"unknown cells in {spec!r}; known: "
                         + ", ".join(f"{T}:{h}" for _, T, h in CELLS))
    return chosen


def geometry(panel, T, horizon, step):
    """Design quantities that do not require running anything."""
    N, days = panel.shape
    windows = (days - T) // step + 1
    origins = windows - 2 * (horizon // step)
    block = int(np.ceil((T + 2 * horizon) / step))
    return {
        "N": N, "T": T, "horizon": horizon, "step": step,
        "q": N / T, "deletion_fraction": horizon / T,
        "origins": origins,
        "bootstrap_block_origins": block,
        "independent_blocks": origins / block,
    }


def observed_cell(panel, T, horizon, step):
    """Observed primary statistic for one cell, no null."""
    starts, full, retained = flag_histories(panel, T, step, horizon)
    series = deletion_attribution_series(starts, full, retained, horizon, step)
    row = summarise(series).set_index("component").loc[PRIMARY]
    return {
        "observed_addition_cosine": float(row["mean_addition_cosine"]),
        "observed_full_cosine": float(row["mean_full_cosine"]),
        "deletion_attributed_fraction":
            float(row["mean_deletion_attributed_fraction"]),
        "projection_residual_cosine": float(row["mean_full_residual_cosine"]),
        "positive_addition_fraction": float(row["positive_addition_fraction"]),
        "n_origins_run": int(row["n_origins"]),
    }


def run_full_cell(args, T, horizon):
    """Delegate one cell to the unmodified Regime 4.9 runner."""
    argv = [
        "--label", args.label,
        "--indir", str(args.indir),
        "--outdir", str(args.outdir),
        "--T", str(T),
        "--horizon", str(horizon),
        "--step", str(args.step),
        "--shuffles", str(args.shuffles),
        "--null", "volatility",
        "--seed", str(args.seed),
    ]
    return regime4_9_main(argv)


def collect(args):
    """Build the observed / null-mean / null-sd curves from written outputs."""
    returns = pd.read_parquet(args.indir / f"{args.label}_returns.parquet")
    panel = to_panel(returns)
    rows = []
    for arm, T, horizon in selected_cells(args.cells):
        stem = f"{args.label}_T{T}_h{horizon}_step{args.step}"
        row = {"arm": arm, **geometry(panel, T, horizon, args.step)}
        summary_path = args.outdir / f"{stem}_summary.csv"
        null_path = args.outdir / f"{stem}_volatility_null.csv"
        if summary_path.exists():
            frame = pd.read_csv(summary_path)
            primary = frame.loc[frame["component"] == PRIMARY].iloc[0]
            row.update({
                "observed_addition_cosine": primary["mean_addition_cosine"],
                "observed_full_cosine": primary["mean_full_cosine"],
                "deletion_attributed_fraction":
                    primary["mean_deletion_attributed_fraction"],
                "projection_residual_cosine":
                    primary["mean_full_residual_cosine"],
                "positive_addition_fraction":
                    primary["positive_addition_fraction"],
                "n_origins_run": primary["n_origins"],
            })
        else:
            row.update(observed_cell(panel, T, horizon, args.step))
        if null_path.exists():
            null = pd.read_csv(null_path).drop_duplicates(
                subset=["replicate", "component"], keep="first")
            values = null.loc[null["component"] == PRIMARY,
                              "mean_addition_cosine"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) > 1:
                observed = row["observed_addition_cosine"]
                row.update({
                    "n_null": len(values),
                    "null_mean": float(np.mean(values)),
                    "null_sd": float(np.std(values, ddof=1)),
                    "null_q95": float(np.quantile(values, .95)),
                    "excess": float(observed - np.mean(values)),
                    "z": float((observed - np.mean(values))
                               / np.std(values, ddof=1)),
                    "mde_excess": float(np.quantile(values, .95)
                                        - np.mean(values)),
                    "p": empirical_upper_p(observed, values),
                })
        rows.append(row)
    table = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / f"{args.label}_sweep_collected.csv", index=False)
    return table


def _show(table):
    design = ["arm", "T", "horizon", "q", "deletion_fraction", "origins",
              "independent_blocks"]
    print("\nDesign")
    print(table[design].to_string(
        index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nObserved (no null required)")
    print(table[["arm", "T", "horizon", "observed_addition_cosine",
                 "observed_full_cosine", "deletion_attributed_fraction",
                 "projection_residual_cosine"]].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"))
    if "null_mean" in table:
        columns = ["arm", "T", "horizon", "q", "observed_addition_cosine",
                   "null_mean", "null_sd", "excess", "z", "mde_excess", "p",
                   "n_null"]
        available = table.dropna(subset=["null_mean"])
        if len(available):
            print("\nVolatility-matched null")
            print(available[columns].to_string(
                index=False, float_format=lambda value: f"{value:.4f}"))


def main(argv=None):
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    cells = selected_cells(args.cells)

    if args.mode == "full":
        for index, (arm, T, horizon) in enumerate(cells, start=1):
            print(f"\n=== cell {index}/{len(cells)}: arm {arm}, "
                  f"T={T}, h={horizon} ===", flush=True)
            run_full_cell(args, T, horizon)
        _show(collect(args))
        return 0

    if args.mode == "collect":
        _show(collect(args))
        return 0

    returns = pd.read_parquet(args.indir / f"{args.label}_returns.parquet")
    panel = to_panel(returns)
    rows = []
    for arm, T, horizon in cells:
        print(f"  observed: arm {arm}, T={T}, h={horizon}", flush=True)
        rows.append({"arm": arm, **geometry(panel, T, horizon, args.step),
                     **observed_cell(panel, T, horizon, args.step)})
    table = pd.DataFrame(rows)
    table.to_csv(args.outdir / f"{args.label}_sweep_observed.csv", index=False)
    _show(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
