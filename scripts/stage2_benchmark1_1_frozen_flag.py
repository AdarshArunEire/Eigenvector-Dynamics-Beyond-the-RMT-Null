"""Benchmark 1.1 — Frozen Flag: predict zero eigenspace rotation."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime4_8_robustness import rolling_flag_frames
from src.data import to_panel
from src.forecast import (ChronologicalSplits, assign_chronological_splits,
                          frozen_flag_series, summarise_benchmark)


DEFAULT_LABELS = ("sp500_full", "nikkei_full", "dax_full", "cac40_full")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", choices=DEFAULT_LABELS + ("all",), default="all")
    p.add_argument("--indir", type=Path, default=Path("data/cache"))
    p.add_argument("--outdir", type=Path,
                   default=Path("results/stage2/benchmark1_1_frozen_flag"))
    p.add_argument("--T", type=int, default=None, help="one-panel override")
    p.add_argument("--step", type=int, default=14)
    p.add_argument("--horizon", type=int, default=42)
    p.add_argument("--train-end", default="2013-12-31")
    p.add_argument("--validation-start", default="2015-07-01")
    p.add_argument("--validation-end", default="2018-06-30")
    p.add_argument("--test-start", default="2020-01-01")
    return p.parse_args(argv)


def run_panel(label, args, specification):
    returns = pd.read_parquet(args.indir / f"{label}_returns.parquet")
    panel = to_panel(returns)
    N, days = panel.shape
    T = args.T or max(N, 250)
    starts, frames, _ = rolling_flag_frames(panel, T, args.step)
    series = frozen_flag_series(starts, frames, returns.index, T,
                                args.horizon, args.step)
    series = assign_chronological_splits(series, specification)
    summary = summarise_benchmark(series)
    for position, (column, value) in enumerate((
            ("label", label), ("N", N), ("days", days), ("T", T),
            ("step", args.step), ("horizon", args.horizon))):
        summary.insert(position, column, value)
    for name, value in vars(specification).items():
        summary[name] = value
    series.insert(0, "label", label)
    series.insert(1, "N", N)
    series.insert(2, "T", T)

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_T{T}_h{args.horizon}_step{args.step}"
    series.to_csv(args.outdir / f"{stem}_series.csv", index=False)
    summary.to_csv(args.outdir / f"{stem}_summary.csv", index=False)

    test = summary.loc[(summary["split"] == "test") &
                       summary["component"].isin(
                           ["market_1", "top_3", "top_6", "flag_nested"])]
    print(f"\n{label}: N={N}, days={days}, T={T}")
    print("  What is tested: loss from pretending the Flag does not rotate.")
    print("  Setup: purged chronological train/validation/test; h=42 days.")
    print("  Test component      mean      median       IQR       examples/effective")
    for row in test.itertuples():
        print(f"  {row.component:<15} {row.mean_loss:.5f}   {row.median_loss:.5f}   "
              f"[{row.q25_loss:.5f}, {row.q75_loss:.5f}]   "
              f"{row.n_examples}/{row.n_nonoverlap_targets}")
    print("  Verdict: BASELINE RECORDED — later models must reduce untouched test loss.")
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
    combined = pd.concat([run_panel(label, args, specification)
                          for label in labels], ignore_index=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.outdir / "all_panels_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
