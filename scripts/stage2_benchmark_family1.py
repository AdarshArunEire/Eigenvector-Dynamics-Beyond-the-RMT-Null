"""Run Stage 2 geometric Benchmark Family 1 on the frozen test origins."""
import argparse
import sys
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime4_8_robustness import rolling_flag_frames
from src.covariance_benchmarks import estimate_bahc, estimate_hcal
from src.data import to_panel
from src.family1_benchmarks import (
    COMPONENTS, combine_panel_skills, compare_with_frozen,
    constant_velocity_flag, erse_flag, ewma_flag, factor_cm_iewma_flag,
    filtered_flag, long_flag_loss_rows, retained_window_flag,
    stationary_roll_forward_flag)
from src.forecast import (ChronologicalSplits, assign_chronological_splits,
                          frozen_flag_series)


DEFAULT_LABELS = ("sp500_full", "nikkei_full", "dax_full", "cac40_full")
ESTIMATORS = {
    "constant_velocity": ("1.2 Constant Velocity", "benchmark1_2_constant_velocity"),
    "erse": ("1.3 ERSE Direction", "benchmark1_3_erse_direction"),
    "hcal": ("1.4 HCAL Flag", "benchmark1_4_hcal_flag"),
    "bahc": ("1.5 BAHC Flag", "benchmark1_5_bahc_flag"),
    "retained_window": ("1.6 Retained-Window Flag", "benchmark1_6_retained_window"),
    "roll_forward": ("1.7 Stationary Roll-Forward", "benchmark1_7_roll_forward"),
    "riskmetrics_ewma": ("1.8 RiskMetrics EWMA Flag", "benchmark1_8_riskmetrics_ewma"),
    "tuned_ewma": ("1.9 Validation-Geometric EWMA Flag", "benchmark1_9_tuned_ewma"),
    "cm_iewma": ("1.10 Factor CM-IEWMA Flag", "benchmark1_10_cm_iewma"),
}

EWMA_HALF_LIVES = (5., 10., 21., 42., 63., 126., 252., 504., 1008.,
                   2016., 4032., float("inf"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=DEFAULT_LABELS + ("all",), default="all")
    parser.add_argument("--estimator", choices=tuple(ESTIMATORS), required=True)
    parser.add_argument("--indir", type=Path, default=Path("data/cache"))
    parser.add_argument("--outdir", type=Path, default=Path("results/stage2"))
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=42)
    parser.add_argument("--delta", type=float, default=.25)
    parser.add_argument("--bahc-bootstraps", type=int, default=100)
    parser.add_argument("--cm-factor-rank", type=int, default=20)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--train-end", default="2013-12-31")
    parser.add_argument("--validation-start", default="2015-07-01")
    parser.add_argument("--validation-end", default="2018-06-30")
    parser.add_argument("--test-start", default="2020-01-01")
    return parser.parse_args(argv)


def _prediction(estimator, past, current, current_returns, args,
                selected_half_life=None):
    if estimator == "constant_velocity":
        return constant_velocity_flag(past, current)
    if estimator == "erse":
        return erse_flag(current_returns, args.delta)
    if estimator == "hcal":
        return filtered_flag(current_returns, estimate_hcal)
    if estimator == "bahc":
        method = partial(estimate_bahc, bootstraps=args.bahc_bootstraps,
                         seed=args.seed)
        return filtered_flag(current_returns, method)
    if estimator == "retained_window":
        return retained_window_flag(current_returns, args.horizon)
    if estimator == "roll_forward":
        return stationary_roll_forward_flag(current_returns, args.horizon)
    if estimator == "riskmetrics_ewma":
        return ewma_flag(current_returns, decay=.94)
    if estimator == "tuned_ewma":
        if selected_half_life is None:
            raise ValueError("tuned EWMA requires a validation-selected half-life")
        return ewma_flag(current_returns, half_life=selected_half_life)
    if estimator == "cm_iewma":
        return factor_cm_iewma_flag(
            current_returns, factor_rank=args.cm_factor_rank)
    raise ValueError(f"unknown estimator {estimator}")


def _select_ewma_half_life(frozen, frames, starts, panel, T, offset, args):
    """Select one half-life from validation complete-Flag loss only."""
    validation_ids = sorted(set(frozen.loc[
        frozen["split"] == "validation", "example"]))
    rows = []
    for half_life in EWMA_HALF_LIVES:
        losses = []
        for example in validation_ids:
            current_index = int(example) + offset
            start = int(starts[current_index])
            prediction = ewma_flag(panel[:, start:start + T],
                                   half_life=half_life)
            losses.append(long_flag_loss_rows(
                example, frames[current_index + offset], prediction)[-1]["loss"])
        rows.append({"half_life": half_life,
                     "validation_mean_complete_flag_loss": float(sum(losses) / len(losses)),
                     "n_validation_examples": len(losses)})
    tuning = pd.DataFrame(rows).sort_values(
        ["validation_mean_complete_flag_loss", "half_life"],
        kind="stable").reset_index(drop=True)
    selected = float(tuning.iloc[0]["half_life"])
    tuning["selected"] = tuning["half_life"] == selected
    return selected, tuning


def run_panel(label, args, specification):
    returns = pd.read_parquet(args.indir / f"{label}_returns.parquet")
    panel = to_panel(returns)
    N, days = panel.shape
    T = args.T or max(N, 250)
    starts, frames, _ = rolling_flag_frames(panel, T, args.step)
    frozen = assign_chronological_splits(
        frozen_flag_series(starts, frames, returns.index, T,
                           args.horizon, args.step), specification)
    frozen.insert(0, "label", label)
    offset = args.horizon // args.step
    selected_half_life, tuning = None, pd.DataFrame()
    if args.estimator == "tuned_ewma":
        selected_half_life, tuning = _select_ewma_half_life(
            frozen, frames, starts, panel, T, offset, args)
        tuning.insert(0, "label", label)
        print(f"  {label}: validation selected EWMA half-life "
              f"{selected_half_life:g}", flush=True)
    test_ids = set(frozen.loc[frozen["split"] == "test", "example"])
    model_name, _ = ESTIMATORS[args.estimator]
    rows = []
    ordered_ids = sorted(test_ids)
    for position, example in enumerate(ordered_ids, start=1):
        current_index = int(example) + offset
        past = frames[current_index - offset]
        current = frames[current_index]
        target = frames[current_index + offset]
        start = int(starts[current_index])
        prediction = _prediction(
            args.estimator, past, current, panel[:, start:start + T], args,
            selected_half_life=selected_half_life)
        metadata = frozen.loc[
            (frozen["example"] == example) &
            (frozen["component"] == "flag_nested")].iloc[0]
        rows.extend(long_flag_loss_rows(
            example, target, prediction, label=label, N=N, T=T,
            estimator=model_name, split="test",
            current_date=metadata["current_date"],
            target_window_start=metadata["target_window_start"],
            target_date=metadata["target_date"]))
        if (args.estimator in {"hcal", "bahc", "cm_iewma"} and
                (position % 20 == 0 or position == len(ordered_ids))):
            print(f"  {label}: {position}/{len(ordered_ids)} origins", flush=True)
    model = pd.DataFrame(rows)
    comparisons = compare_with_frozen(
        model, frozen.loc[frozen["split"] == "test"], T=T,
        horizon=args.horizon, step=args.step,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed + sum(map(ord, label)))
    comparisons.insert(2, "estimator", model_name)
    return model, comparisons, tuning


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
    outputs = [run_panel(label, args, specification) for label in labels]
    series = pd.concat([output[0] for output in outputs], ignore_index=True)
    comparisons = pd.concat([output[1] for output in outputs], ignore_index=True)
    combined = combine_panel_skills(comparisons)
    model_name, directory = ESTIMATORS[args.estimator]
    outdir = args.outdir / directory
    outdir.mkdir(parents=True, exist_ok=True)
    series.to_csv(outdir / "all_panels_series.csv", index=False)
    comparisons.to_csv(outdir / "all_panels_comparisons.csv", index=False)
    combined.to_csv(outdir / "combined_skill.csv", index=False)
    tuning_frames = [output[2] for output in outputs if not output[2].empty]
    if tuning_frames:
        pd.concat(tuning_frames, ignore_index=True).to_csv(
            outdir / "ewma_tuning.csv", index=False)

    table = combined.set_index("component").loc[list(COMPONENTS)]
    print(f"\nWhat is tested: {model_name} against Frozen Flag on identical test origins.")
    print("Setup: all six Flag losses; equal-market skill; paired calendar-block intervals.")
    print(table[["combined_skill_percent", "worst_panel_skill_percent",
                 "panels_improved", "n_panels"]].to_string(
        float_format=lambda value: f"{value:.4g}"))
    complete = table.loc["flag_nested"]
    verdict = "PASS" if (complete["panels_improved"] == complete["n_panels"]
                         and complete["combined_skill_percent"] > 0) else "NAY"
    print(f"Verdict: {verdict} — complete-Flag combined skill "
          f"{complete['combined_skill_percent']:.2f}%, worst panel "
          f"{complete['worst_panel_skill_percent']:.2f}%, "
          f"{int(complete['panels_improved'])}/{int(complete['n_panels'])} improved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
