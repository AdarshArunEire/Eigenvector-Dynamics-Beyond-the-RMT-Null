"""Collect and validate the frozen Stage 2 covariance benchmark leaderboard."""
import argparse
from pathlib import Path

import pandas as pd


PRIMARY_METRICS = (
    "relative_frobenius",
    "gaussian_nll_per_asset",
    "gmv_long_short_annualised_volatility",
    "gmv_long_only_annualised_volatility",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-dir", type=Path,
                        default=Path("results/stage2/benchmark2_covariance"))
    parser.add_argument("--structural-dir", type=Path,
                        default=Path("results/stage2/benchmark2_structural"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("results/stage2/benchmark2_leaderboard"))
    return parser.parse_args(argv)


def _series_files(standard_dir, structural_dir):
    standard = list(standard_dir.glob("*_series.csv"))
    structural = [path for path in structural_dir.glob("*_series.csv")
                  if any(f"_{name}_" in path.name
                         for name in ("cvc", "hcal", "bahc"))]
    return standard + structural


def collect(args):
    files = _series_files(args.standard_dir, args.structural_dir)
    if not files:
        raise FileNotFoundError("no benchmark series files found")
    frames = [pd.read_csv(path) for path in files]
    required = {"label", "estimator", "example", "split", "target_date",
                *PRIMARY_METRICS}
    for path, frame in zip(files, frames):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
    series = pd.concat(frames, ignore_index=True)
    series = series.loc[series["split"] == "test"].copy()
    series = series.drop_duplicates(["label", "estimator", "example"])

    expected = series.groupby("label")["example"].apply(
        lambda values: frozenset(values))
    for (label, estimator), group in series.groupby(["label", "estimator"]):
        actual = frozenset(group["example"])
        if actual != expected[label]:
            raise ValueError(
                f"date mismatch for {label}/{estimator}: "
                f"{len(actual)} rows versus expected {len(expected[label])}")
        if group[list(PRIMARY_METRICS)].isna().any().any():
            raise ValueError(f"missing primary score for {label}/{estimator}")

    long = series.melt(
        id_vars=["label", "estimator", "example", "target_date"],
        value_vars=list(PRIMARY_METRICS), var_name="metric", value_name="loss")
    summary = long.groupby(["label", "estimator", "metric"],
                           as_index=False).agg(
        n_examples=("loss", "size"),
        mean=("loss", "mean"),
        median=("loss", "median"),
        q25=("loss", lambda x: x.quantile(.25)),
        q75=("loss", lambda x: x.quantile(.75)),
    )
    summary["rank"] = summary.groupby(["label", "metric"])["mean"].rank(
        method="min", ascending=True).astype(int)
    leaderboard = summary.pivot_table(
        index=["label", "estimator"], columns="metric", values="mean").reset_index()
    winners = summary.loc[summary["rank"] == 1].sort_values(
        ["label", "metric", "estimator"])

    args.outdir.mkdir(parents=True, exist_ok=True)
    series.to_csv(args.outdir / "all_benchmarks_test_series.csv", index=False)
    summary.to_csv(args.outdir / "all_benchmarks_test_summary.csv", index=False)
    leaderboard.to_csv(args.outdir / "all_benchmarks_test_leaderboard.csv",
                       index=False)
    winners.to_csv(args.outdir / "all_benchmarks_test_winners.csv", index=False)
    return series, summary, winners


def main(argv=None):
    args = parse_args(argv)
    series, _, winners = collect(args)
    counts = series.groupby(["label", "estimator"]).size()
    print("What is tested: benchmark-output integrity and the combined leaderboard.")
    print("Setup: exact equality of untouched test example IDs; lower is better.")
    print(f"Validated {counts.size} panel-estimator cells; row counts "
          f"{sorted(counts.unique())}.")
    print(winners[["label", "metric", "estimator", "mean"]].to_string(
        index=False, float_format=lambda value: f"{value:.6g}"))
    print("Verdict: YAY - every estimator has the same complete test rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
