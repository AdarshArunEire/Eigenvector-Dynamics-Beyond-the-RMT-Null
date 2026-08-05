"""Stage 2 extended covariance benchmarks: CVC, HCAL and BAHC."""
import argparse
import sys
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.covariance_benchmarks import (
    estimate_bahc, estimate_cvc, estimate_hcal,
    evaluate_covariance_estimator, rolling_covariance_examples,
    split_covariance_examples, summarise_covariance_benchmarks)
from src.data import to_panel
from src.forecast import ChronologicalSplits


DEFAULT_LABELS = ("sp500_full", "nikkei_full", "dax_full", "cac40_full")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=DEFAULT_LABELS + ("all",),
                        default="all")
    parser.add_argument("--estimator", choices=("cvc", "hcal", "bahc", "all"),
                        default="all")
    parser.add_argument("--indir", type=Path, default=Path("data/cache"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("results/stage2/benchmark2_structural"))
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=42)
    parser.add_argument("--bahc-bootstraps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--all-splits", action="store_true",
                        help="also compute purged/train/validation rows; test is sufficient for comparison")
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
    available = {
        "cvc": ("2.9 Cross-validated eigenvalue shrinkage",
                partial(estimate_cvc, folds=10, seed=args.seed)),
        "hcal": ("2.10 HCAL", estimate_hcal),
        "bahc": (f"2.5 BAHC ({args.bahc_bootstraps} bootstraps)",
                 partial(estimate_bahc, bootstraps=args.bahc_bootstraps,
                         seed=args.seed)),
    }
    selected = available.values() if args.estimator == "all" else (
        available[args.estimator],)
    include_splits = None if args.all_splits else {"test"}
    series = pd.concat([
        evaluate_covariance_estimator(
            examples, splits, estimator, name, include_splits=include_splits)
        for name, estimator in selected
    ], ignore_index=True)
    summary = summarise_covariance_benchmarks(series)
    for frame in (series, summary):
        frame.insert(0, "label", label)
        frame.insert(1, "N", N)
        frame.insert(2, "T", T)
        frame["horizon"] = args.horizon
        frame["step"] = args.step
        frame["bahc_bootstraps"] = args.bahc_bootstraps
        frame["seed"] = args.seed

    args.outdir.mkdir(parents=True, exist_ok=True)
    estimator_stem = args.estimator
    stem = f"{label}_{estimator_stem}_T{T}_h{args.horizon}_step{args.step}"
    series.to_csv(args.outdir / f"{stem}_series.csv", index=False)
    summary.to_csv(args.outdir / f"{stem}_summary.csv", index=False)
    primary = summary.loc[(summary["split"] == "test") &
                          summary["metric"].isin((
                              "relative_frobenius",
                              "gaussian_nll_per_asset",
                              "gmv_long_short_annualised_volatility",
                              "gmv_long_only_annualised_volatility"))]
    print(f"\n{label}: N={N}, days={days}, T={T}")
    print("  What is tested: eigenvalue-CV and hierarchical covariance filters.")
    print("  Setup: untouched test origins; same T-day input and next-42-day target.")
    print(primary.pivot(index="estimator", columns="metric", values="mean")
          .to_string(float_format=lambda value: f"{value:.6g}"))
    print("  Verdict: EXTENDED BASELINES RECORDED.")
    return summary


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
    summaries = [run_panel(label, args, specification) for label in labels]
    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.concat(summaries, ignore_index=True).to_csv(
        args.outdir / f"all_panels_{args.estimator}_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
