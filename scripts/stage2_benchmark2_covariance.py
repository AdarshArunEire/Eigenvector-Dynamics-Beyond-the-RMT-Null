"""Stage 2 Benchmark Family 2: established full-covariance estimators."""
import argparse
import sys
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.covariance_benchmarks import (
    choose_ewma_half_life, estimate_ewma, estimate_ledoit_wolf,
    estimate_nonlinear_shrinkage, estimate_oas, estimate_sample,
    evaluate_covariance_estimator, rolling_covariance_examples,
    split_covariance_examples, summarise_covariance_benchmarks)
from src.data import to_panel
from src.forecast import ChronologicalSplits


DEFAULT_LABELS = ("sp500_full", "nikkei_full", "dax_full", "cac40_full")
# Frozen after the first validation-only smoke test selected the old upper
# boundary (252).  The final log-spaced grid includes fast crisis adaptation
# through an effectively uniform weighting of the finite T-day input window.
EWMA_HALF_LIFE_GRID = (5, 10, 21, 42, 63, 126, 252, 504, 1008,
                       2016, 4032, 8064, 16128, float("inf"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=DEFAULT_LABELS + ("all",),
                        default="all")
    parser.add_argument("--indir", type=Path, default=Path("data/cache"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("results/stage2/benchmark2_covariance"))
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=42)
    parser.add_argument("--train-end", default="2013-12-31")
    parser.add_argument("--validation-start", default="2015-07-01")
    parser.add_argument("--validation-end", default="2018-06-30")
    parser.add_argument("--test-start", default="2020-01-01")
    return parser.parse_args(argv)


def run_panel(label, args, specification):
    returns = pd.read_parquet(args.indir / f"{label}_returns.parquet")
    panel = to_panel(returns)
    N, days = panel.shape
    T = args.T or max(N, 250)
    examples = rolling_covariance_examples(
        panel, returns.index, T, args.horizon, args.step)
    splits = split_covariance_examples(examples, specification)

    selected_half_life, tuning = choose_ewma_half_life(
        examples, splits, EWMA_HALF_LIFE_GRID)
    tuned_label = ("uniform" if selected_half_life == float("inf")
                   else f"half-life {selected_half_life:g}")
    estimators = (
        ("2.1 Sample covariance", estimate_sample),
        (f"2.2 Tuned EWMA ({tuned_label})",
         partial(estimate_ewma, half_life=selected_half_life)),
        ("2.3 Ledoit-Wolf", estimate_ledoit_wolf),
        ("2.4 Ledoit-Wolf QIS / RIE",
         estimate_nonlinear_shrinkage),
        ("2.6 RiskMetrics EWMA (lambda 0.94)",
         partial(estimate_ewma, decay=.94)),
        ("2.7 OAS", estimate_oas),
    )
    series = pd.concat([
        evaluate_covariance_estimator(
            examples, splits, estimator, name, include_splits={"test"})
        for name, estimator in estimators
    ], ignore_index=True)
    summary = summarise_covariance_benchmarks(series)

    for frame in (series, summary, tuning):
        frame.insert(0, "label", label)
        frame.insert(1, "N", N)
        frame.insert(2, "T", T)
        frame["horizon"] = args.horizon
        frame["step"] = args.step
    tuning["selected"] = tuning["half_life"] == selected_half_life

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_T{T}_h{args.horizon}_step{args.step}"
    series.to_csv(args.outdir / f"{stem}_series.csv", index=False)
    summary.to_csv(args.outdir / f"{stem}_summary.csv", index=False)
    tuning.to_csv(args.outdir / f"{stem}_ewma_tuning.csv", index=False)

    primary = summary.loc[(summary["split"] == "test") &
                          summary["metric"].isin((
                              "relative_frobenius",
                              "gaussian_nll_per_asset",
                              "gmv_long_short_annualised_volatility",
                              "gmv_long_only_annualised_volatility"))]
    print(f"\n{label}: N={N}, days={days}, T={T}")
    print("  What is tested: full covariance forecasts against the next 42 returns.")
    print("  Setup: identical past information and dates; EWMA tuned on validation only.")
    selected_text = ("uniform weighting (no decay)" if
                     selected_half_life == float("inf") else
                     f"{selected_half_life:g} trading days")
    print(f"  Selected EWMA half-life: {selected_text}")
    print(primary.pivot(index="estimator", columns="metric", values="mean")
          .to_string(float_format=lambda value: f"{value:.6g}"))
    print("  Verdict: BASELINES RECORDED - learned forecasts have not entered yet.")
    return series, summary, tuning


def main(argv=None):
    args = parse_args(argv)
    if args.T is not None and args.label == "all":
        raise SystemExit("--T override requires one --label")
    specification = ChronologicalSplits(
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start)
    labels = DEFAULT_LABELS if args.label == "all" else (args.label,)
    output = [run_panel(label, args, specification) for label in labels]
    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.concat([value[1] for value in output], ignore_index=True).to_csv(
        args.outdir / "all_panels_summary.csv", index=False)
    pd.concat([value[2] for value in output], ignore_index=True).to_csv(
        args.outdir / "all_panels_ewma_tuning.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
