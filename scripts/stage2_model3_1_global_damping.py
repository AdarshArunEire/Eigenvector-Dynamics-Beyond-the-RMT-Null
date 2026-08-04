"""Fit and evaluate Model 3.1: one scalar on the complete previous Flag motion."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime4_8_robustness import rolling_flag_frames
from scripts.stage2_benchmark_family1 import DEFAULT_LABELS
from src.data import to_panel
from src.family1_benchmarks import (
    COMPONENTS, DEFAULT_DAMPING_ALPHAS, combine_panel_skills,
    compare_with_frozen, damped_velocity_flag, long_flag_loss_rows,
    retained_window_flag, select_global_damping_alpha)
from src.forecast import (ChronologicalSplits, assign_chronological_splits,
                          frozen_flag_series)


MODEL_NAME = "3.1 Global Damping"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=DEFAULT_LABELS + ("all",),
                        default="all")
    parser.add_argument("--indir", type=Path, default=Path("data/cache"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("results/stage2/model3_1_global_damping"))
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=42)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--train-end", default="2013-12-31")
    parser.add_argument("--validation-start", default="2015-07-01")
    parser.add_argument("--validation-end", default="2018-06-30")
    parser.add_argument("--test-start", default="2020-01-01")
    return parser.parse_args(argv)


def _rename_reference(comparison, old, new):
    """Name the reference loss explicitly while preserving skill columns."""
    return comparison.rename(columns={old: new})


def run_panel(label, args, specification):
    returns = pd.read_parquet(args.indir / f"{label}_returns.parquet")
    panel = to_panel(returns)
    N, _ = panel.shape
    T = args.T or max(N, 250)
    starts, frames, _ = rolling_flag_frames(panel, T, args.step)
    frozen = assign_chronological_splits(
        frozen_flag_series(starts, frames, returns.index, T,
                           args.horizon, args.step), specification)
    frozen.insert(0, "label", label)
    offset = args.horizon // args.step

    validation_ids = sorted(set(frozen.loc[
        frozen["split"] == "validation", "example"]))
    alpha, tuning = select_global_damping_alpha(
        frames, validation_ids, offset, DEFAULT_DAMPING_ALPHAS)
    tuning.insert(0, "label", label)
    print(f"  {label}: selected alpha={alpha:.3f} from "
          f"{len(validation_ids)} validation origins", flush=True)

    model_rows, retained_rows = [], []
    test_ids = sorted(set(frozen.loc[frozen["split"] == "test", "example"]))
    for example in test_ids:
        current_index = int(example) + offset
        start = int(starts[current_index])
        current_returns = panel[:, start:start + T]
        target = frames[current_index + offset]
        prediction = damped_velocity_flag(
            frames[current_index - offset], frames[current_index], alpha)
        retained = retained_window_flag(current_returns, args.horizon)
        metadata = frozen.loc[
            (frozen["example"] == example)
            & (frozen["component"] == "flag_nested")].iloc[0]
        shared = dict(
            label=label, N=N, T=T, split="test", alpha=alpha,
            current_date=metadata["current_date"],
            target_window_start=metadata["target_window_start"],
            target_date=metadata["target_date"])
        model_rows.extend(long_flag_loss_rows(
            example, target, prediction, estimator=MODEL_NAME, **shared))
        retained_rows.extend(long_flag_loss_rows(
            example, target, retained, estimator="1.6 Retained-Window Flag",
            **shared))

    model = pd.DataFrame(model_rows)
    retained = pd.DataFrame(retained_rows)
    frozen_test = frozen.loc[frozen["split"] == "test"]
    comparison_args = dict(
        T=T, horizon=args.horizon, step=args.step,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed + sum(map(ord, label)))
    versus_frozen = compare_with_frozen(
        model, frozen_test, **comparison_args)
    versus_frozen.insert(2, "estimator", MODEL_NAME)
    versus_frozen.insert(3, "reference", "1.1 Frozen Flag")

    versus_retained = compare_with_frozen(
        model, retained, **comparison_args)
    versus_retained = _rename_reference(
        versus_retained, "frozen_mean_loss", "retained_mean_loss")
    versus_retained.insert(2, "estimator", MODEL_NAME)
    versus_retained.insert(3, "reference", "1.6 Retained-Window Flag")
    return model, retained, versus_frozen, versus_retained, tuning


def _print_complete_result(name, combined, comparisons):
    table = combined.set_index("component").loc[list(COMPONENTS)]
    complete = table.loc["flag_nested"]
    panel_rows = comparisons.loc[comparisons["component"] == "flag_nested"]
    clear_positive = int((panel_rows["skill_ci_low"] > 0).sum())
    clear_negative = int((panel_rows["skill_ci_high"] < 0).sum())
    panels = len(panel_rows)
    if (complete["combined_skill_percent"] > 0
            and complete["panels_improved"] == complete["n_panels"]):
        verdict = ("PASS" if clear_positive == panels else
                   f"YAY ON MEANS; {clear_positive}/{panels} CLEAR")
    elif clear_negative == panels:
        verdict = "NAY"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\n{name}")
    print(table[["combined_skill_percent", "worst_panel_skill_percent",
                 "panels_improved", "n_panels"]].to_string(
        float_format=lambda value: f"{value:.4g}"))
    print(f"Verdict: {verdict} - complete-Flag skill "
          f"{complete['combined_skill_percent']:.2f}%, worst panel "
          f"{complete['worst_panel_skill_percent']:.2f}%.")


def main(argv=None):
    args = parse_args(argv)
    if args.T is not None and args.label == "all":
        raise SystemExit("--T override requires one --label")
    if args.horizon < args.step or args.horizon % args.step:
        raise SystemExit("--horizon must be a positive multiple of --step")
    specification = ChronologicalSplits(
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start)
    labels = DEFAULT_LABELS if args.label == "all" else (args.label,)
    outputs = [run_panel(label, args, specification) for label in labels]

    model = pd.concat([output[0] for output in outputs], ignore_index=True)
    retained = pd.concat([output[1] for output in outputs], ignore_index=True)
    versus_frozen = pd.concat(
        [output[2] for output in outputs], ignore_index=True)
    versus_retained = pd.concat(
        [output[3] for output in outputs], ignore_index=True)
    tuning = pd.concat([output[4] for output in outputs], ignore_index=True)
    combined_frozen = combine_panel_skills(versus_frozen)
    combined_retained = combine_panel_skills(versus_retained)

    args.outdir.mkdir(parents=True, exist_ok=True)
    model.to_csv(args.outdir / "all_panels_series.csv", index=False)
    retained.to_csv(args.outdir / "retained_reference_series.csv", index=False)
    tuning.to_csv(args.outdir / "alpha_tuning.csv", index=False)
    versus_frozen.to_csv(
        args.outdir / "versus_frozen_comparisons.csv", index=False)
    combined_frozen.to_csv(
        args.outdir / "versus_frozen_combined_skill.csv", index=False)
    versus_retained.to_csv(
        args.outdir / "versus_retained_comparisons.csv", index=False)
    combined_retained.to_csv(
        args.outdir / "versus_retained_combined_skill.csv", index=False)

    print("\nWhat is tested: whether one validation-fitted scalar can turn the "
          "previous complete Flag rotation into a test-period forecast.")
    print("Setup: alpha in [0,1] on a frozen 0.025 grid; complete-Flag "
          "validation loss only; untouched test origins.")
    _print_complete_result(
        "Against Frozen Flag", combined_frozen, versus_frozen)
    _print_complete_result(
        "Against Retained-Window Flag", combined_retained, versus_retained)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
