"""Stage 2 Oracle Line: information ceilings for Flag reconstruction."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.covariance_benchmarks import (
    choose_ewma_half_life, covariance_scores, realised_covariance,
    rolling_covariance_examples, split_covariance_examples,
    summarise_covariance_benchmarks)
from src.data import to_panel
from src.forecast import ChronologicalSplits, frozen_flag_losses
from src.oracle_line import build_oracle_forecasts
from src.overlap import spectral


DEFAULT_LABELS = ("sp500_full", "nikkei_full", "dax_full", "cac40_full")
PRIMARY_METRICS = (
    "relative_frobenius",
    "gaussian_nll_per_asset",
    "gmv_long_short_annualised_volatility",
    "gmv_long_only_annualised_volatility",
)
EWMA_HALF_LIFE_GRID = (5, 10, 21, 42, 63, 126, 252, 504, 1008,
                       2016, 4032, 8064, 16128, float("inf"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=DEFAULT_LABELS + ("all",),
                        default="all")
    parser.add_argument("--indir", type=Path, default=Path("data/cache"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("results/stage2/oracle_line"))
    parser.add_argument("--benchmark-series", type=Path, default=Path(
        "results/stage2/benchmark2_leaderboard/all_benchmarks_test_series.csv"))
    parser.add_argument("--ewma-tuning", type=Path, default=Path(
        "results/stage2/benchmark2_covariance/all_panels_ewma_tuning.csv"))
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=42)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--train-end", default="2013-12-31")
    parser.add_argument("--validation-start", default="2015-07-01")
    parser.add_argument("--validation-end", default="2018-06-30")
    parser.add_argument("--test-start", default="2020-01-01")
    return parser.parse_args(argv)


def selected_half_life(label, path, examples, splits):
    """Reuse the frozen Family 2 validation choice, or reproduce it."""
    if path.exists():
        tuning = pd.read_csv(path)
        rows = tuning.loc[(tuning["label"] == label) &
                          tuning["selected"].astype(str).str.lower().eq("true")]
        if len(rows) == 1:
            return float(rows.iloc[0]["half_life"]), "frozen Family 2 output"
    chosen, _ = choose_ewma_half_life(
        examples, splits, EWMA_HALF_LIFE_GRID)
    return chosen, "recomputed from validation"


def block_bootstrap_interval(reference, candidate, block_length,
                             repetitions, rng):
    """Circular calendar-block interval for paired mean improvement."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("paired score arrays must be one-dimensional and equal")
    difference = reference - candidate
    n = len(difference)
    block_length = min(max(1, int(block_length)), n)
    blocks = int(np.ceil(n / block_length))
    draws = np.empty(int(repetitions), dtype=float)
    offsets = np.arange(block_length)
    for draw in range(len(draws)):
        starts = rng.integers(0, n, size=blocks)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        draws[draw] = float(np.mean(difference[indices]))
    return tuple(float(value) for value in np.quantile(draws, [.025, .975]))


def compare_pair(label, candidate_name, reference_name, metric,
                 candidate, reference, T, horizon, step, repetitions, rng,
                 reference_type):
    """One paired comparison with an overlap-aware uncertainty interval."""
    candidate = candidate.sort_values("example")
    reference = reference.sort_values("example")
    merged = candidate[["example", metric]].merge(
        reference[["example", metric]], on="example", suffixes=("_candidate",
                                                                 "_reference"),
        validate="one_to_one")
    candidate_values = merged[f"{metric}_candidate"].to_numpy(float)
    reference_values = merged[f"{metric}_reference"].to_numpy(float)
    block_length = int(np.ceil((T + horizon) / step))
    lower, upper = block_bootstrap_interval(
        reference_values, candidate_values, block_length, repetitions, rng)
    candidate_mean = float(np.mean(candidate_values))
    reference_mean = float(np.mean(reference_values))
    improvement = reference_mean - candidate_mean
    return {
        "label": label,
        "candidate": candidate_name,
        "reference_type": reference_type,
        "reference": reference_name,
        "metric": metric,
        "n_examples": len(merged),
        "block_length_origins": min(block_length, len(merged)),
        "candidate_mean": candidate_mean,
        "reference_mean": reference_mean,
        "absolute_improvement": improvement,
        "percent_improvement": 100.0 * improvement /
            max(abs(reference_mean), np.finfo(float).tiny),
        "win_fraction": float(np.mean(candidate_values < reference_values)),
        "improvement_ci_low": lower,
        "improvement_ci_high": upper,
    }


def run_panel(label, args, specification, benchmark_series):
    returns = pd.read_parquet(args.indir / f"{label}_returns.parquet")
    panel = to_panel(returns)
    N, days = panel.shape
    T = args.T or max(N, 250)
    examples = rolling_covariance_examples(
        panel, returns.index, T, args.horizon, args.step)
    splits = split_covariance_examples(examples, specification)
    split_by_example = splits.set_index("example")["split"].to_dict()
    half_life, half_life_source = selected_half_life(
        label, args.ewma_tuning, examples, splits)

    rows = []
    test_examples = [row for row in examples
                     if split_by_example[row.example] == "test"]
    for position, example in enumerate(test_examples, start=1):
        future_start = example.current_start + args.horizon
        future_rolling = panel[:, future_start:future_start + T]
        if future_rolling.shape[1] != T:
            raise RuntimeError("future rolling oracle window is incomplete")
        forecasts = build_oracle_forecasts(
            example.estimation_returns, future_rolling,
            example.realised_returns, half_life)
        target = realised_covariance(example.realised_returns)
        future_flag = forecasts.future_state.flag
        for estimator, forecast in forecasts.covariance.items():
            scores = covariance_scores(forecast, target, compute_long_only=True)
            input_loss = frozen_flag_losses(
                future_flag, forecasts.input_flag[estimator])["flag_nested"]
            _, reconstructed_vectors = spectral(
                forecasts.correlation[estimator])
            reconstruction_loss = frozen_flag_losses(
                future_flag, reconstructed_vectors[:, :6])["flag_nested"]
            rows.append({
                "label": label,
                "N": N,
                "T": T,
                "horizon": args.horizon,
                "step": args.step,
                "example": example.example,
                "estimator": estimator,
                "split": "test",
                "current_date": example.current_date,
                "target_window_start": example.target_window_start,
                "target_date": example.target_date,
                "ewma_half_life": half_life,
                "input_flag_loss": input_loss,
                "post_reconstruction_flag_loss": reconstruction_loss,
                **scores,
            })
        if position % 20 == 0 or position == len(test_examples):
            print(f"  {label}: {position}/{len(test_examples)} origins", flush=True)

    series = pd.DataFrame(rows)
    summary = summarise_covariance_benchmarks(series.drop(columns=[
        "label", "N", "T", "horizon", "step", "ewma_half_life"]))
    for column, value in reversed((
            ("label", label), ("N", N), ("T", T),
            ("horizon", args.horizon), ("step", args.step),
            ("ewma_half_life", half_life))):
        summary.insert(0, column, value)

    rng = np.random.default_rng(args.seed + sum(map(ord, label)))
    comparisons = []
    control_name = "Control - frozen Flag/QIS/EWMA"
    control = series.loc[series["estimator"] == control_name]
    oracle_names = [name for name in series["estimator"].unique()
                    if name.startswith("Oracle")]
    for oracle_name in oracle_names:
        candidate = series.loc[series["estimator"] == oracle_name]
        for metric in PRIMARY_METRICS:
            comparisons.append(compare_pair(
                label, oracle_name, control_name, metric, candidate, control,
                T, args.horizon, args.step, args.bootstrap_repetitions, rng,
                "common reconstruction control"))

    panel_benchmarks = benchmark_series.loc[benchmark_series["label"] == label]
    if not panel_benchmarks.empty:
        for metric in PRIMARY_METRICS:
            means = panel_benchmarks.groupby("estimator")[metric].mean()
            best_name = str(means.idxmin())
            best = panel_benchmarks.loc[
                panel_benchmarks["estimator"] == best_name]
            for candidate_name in [control_name, *oracle_names]:
                candidate = series.loc[series["estimator"] == candidate_name]
                comparisons.append(compare_pair(
                    label, candidate_name, best_name, metric, candidate, best,
                    T, args.horizon, args.step, args.bootstrap_repetitions, rng,
                    "best Family 2 estimator"))
    comparisons = pd.DataFrame(comparisons)

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_T{T}_h{args.horizon}_step{args.step}"
    series.to_csv(args.outdir / f"{stem}_series.csv", index=False)
    summary.to_csv(args.outdir / f"{stem}_summary.csv", index=False)
    comparisons.to_csv(args.outdir / f"{stem}_comparisons.csv", index=False)
    print(f"  EWMA half-life: {half_life:g} ({half_life_source})")
    return series, summary, comparisons


def main(argv=None):
    args = parse_args(argv)
    if args.T is not None and args.label == "all":
        raise SystemExit("--T override requires one --label")
    specification = ChronologicalSplits(
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start)
    benchmark_series = (pd.read_csv(args.benchmark_series)
                        if args.benchmark_series.exists() else pd.DataFrame())
    labels = DEFAULT_LABELS if args.label == "all" else (args.label,)
    outputs = [run_panel(label, args, specification, benchmark_series)
               for label in labels]
    args.outdir.mkdir(parents=True, exist_ok=True)
    all_series = pd.concat([value[0] for value in outputs], ignore_index=True)
    all_summary = pd.concat([value[1] for value in outputs], ignore_index=True)
    all_comparisons = pd.concat([value[2] for value in outputs],
                                ignore_index=True)
    all_series.to_csv(args.outdir / "all_panels_series.csv", index=False)
    all_summary.to_csv(args.outdir / "all_panels_summary.csv", index=False)
    all_comparisons.to_csv(args.outdir / "all_panels_comparisons.csv", index=False)

    primary = all_summary.loc[
        (all_summary["split"] == "test") &
        all_summary["metric"].isin(PRIMARY_METRICS)]
    print("\nWhat is tested: how much future information is required before the")
    print("rolling Flag route becomes a competitive full covariance forecast.")
    print(primary.pivot_table(index=["label", "estimator"], columns="metric",
                              values="mean").to_string(
        float_format=lambda value: f"{value:.6g}"))
    print("Verdict: ORACLE CEILINGS RECORDED - interpret against the common")
    print("control and best Family 2 estimator; no oracle is deployable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
