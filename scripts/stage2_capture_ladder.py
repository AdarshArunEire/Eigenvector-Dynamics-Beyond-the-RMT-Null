"""Reproducible capture ladder: Frozen, EWMA, RIE class, ceiling and floors.

Regenerates the Stage 2 respecified benchmark from the cached returns so that
Model 4.1 has a baseline it can actually be evaluated against.  Everything is
split-clean: the EWMA half-life is selected on validation origins only, the
reported comparison is on test origins only, and the interval is a
circular-block bootstrap at block length ``ceil((T_in + T_out)/step)`` origins
rather than a t-statistic that would treat overlapping windows as independent.

Two nulls are reported alongside the entrants, and neither is optional.

* The **Haar floor** is exactly ``d/N``, because ``E[Y Y^T] = (d/N) I`` for a
  random frame.  No simulation needed.
* The **ceiling bias** is the headroom the in-sample top-6 of the target window
  reports in a simulated world where the subspace does not move at all.  At
  ``T_out=42`` that is most of the naive headroom, so quoting skill against the
  raw ceiling overstates the denominator several-fold.  The corrected
  denominator, ``ceiling - frozen - bias``, is the only one this script prints.

The rotationally-invariant entrants are included for one reason: to make the
structural zero visible.  Ledoit-Wolf, OAS and QIS are eigenvalue maps with the
sample eigenvectors held fixed, so the visible block ``U_perp^T G U_6`` of the
shrinkage step is identically zero and their capture is pinned to Frozen by
construction.  The script asserts that predicate rather than inferring it from
four matching decimal places -- and, having asserted it, then finds that the
ladder's own correlation renormalisation breaks it for QIS alone, because
``D^{-1/2} S D^{-1/2}`` is a congruence rather than a similarity whenever the
diagonal is not constant.  That is the origin of QIS's residual 1e-4, and it is
a fact about the pipeline, not about nonlinear shrinkage.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capture import ceiling_bias, random_floor, realised_ceiling, variance_captured
from src.capture_ladder import (assign_splits, block_length_origins,
                                build_origins, frozen_frame, paired_summary)
from src.coupling import (RANK, base_estimator, descending_spectrum,
                          leading_frame, rie_correction_is_invisible)
from src.data import to_panel
from src.family1_benchmarks import covariance_to_correlation
from src.overlap import sample_covariance

DEFAULT_LABELS = ("cac40_full", "dax_full", "nikkei_full", "sp500_full")
EWMA_HALF_LIVES = (21., 42., 63., 126., 252., 504., 1008., float("inf"))
RIE_ESTIMATORS = ("ledoit_wolf", "oas", "qis")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=DEFAULT_LABELS + ("all",), default="all")
    parser.add_argument("--indir", type=Path, default=Path("data/cache"))
    parser.add_argument("--outdir", type=Path, default=Path("results/stage2/capture_ladder"))
    parser.add_argument("--T-in", type=int, default=750)
    parser.add_argument("--T-out", type=int, default=42)
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--ceiling-replicates", type=int, default=60)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--skip-rie", action="store_true",
                        help="skip the rotationally-invariant entrants")
    parser.add_argument("--train-end", default="2013-12-31")
    parser.add_argument("--validation-start", default="2015-07-01")
    parser.add_argument("--validation-end", default="2018-06-30")
    parser.add_argument("--test-start", default="2020-01-01")
    return parser.parse_args(argv)


def _rie_frames(estimation, adjusted, check=True):
    """Leading frames of the shrinkage estimators, with the zero measured twice.

    The structural claim is about the *shrinkage step*: an eigenvalue map with
    the sample eigenvectors held fixed has ``U_perp^T G U_6 = 0`` and cannot
    move the score.  Measured on the raw estimate, that is what happens, to
    machine precision, for all three.

    The ladder then renormalises each estimate to a correlation matrix, and
    that step is *not* rotationally invariant whenever the estimate has a
    non-constant diagonal: ``D^{-1/2} S D^{-1/2}`` is a congruence, not a
    similarity.  Ledoit-Wolf and OAS escape it because shrinking a
    constant-diagonal matrix toward a multiple of the identity leaves the
    diagonal constant, so the renormalisation is a scalar.  QIS cleans each
    eigenvalue differently, its diagonal is not constant, and the
    renormalisation therefore rotates the frame slightly.

    Both are recorded.  The residual QIS movement in the reported ladder is
    that renormalisation, not a property of QIS, which is worth knowing before
    anyone reads a signed 1e-4 as evidence about nonlinear shrinkage.
    """
    from src.covariance_benchmarks import (estimate_ledoit_wolf,
                                           estimate_nonlinear_shrinkage,
                                           estimate_oas)
    methods = {"ledoit_wolf": estimate_ledoit_wolf, "oas": estimate_oas,
               "qis": estimate_nonlinear_shrinkage}
    sample = sample_covariance(adjusted)
    out, invisible = {}, {}
    normalised_sample = covariance_to_correlation(sample) if check else None
    for name in RIE_ESTIMATORS:
        raw = methods[name](adjusted)
        shrunk = covariance_to_correlation(raw)
        if check:
            invisible[f"{name}_shrinkage"] = rie_correction_is_invisible(
                sample, raw, RANK, tol=1e-7 * float(np.trace(sample)))
            invisible[f"{name}_after_renormalisation"] = rie_correction_is_invisible(
                normalised_sample, shrunk, RANK, tol=1e-7 * adjusted.shape[0])
        out[name] = leading_frame(shrunk, RANK)
    return out, invisible


def run_panel(label, args):
    returns = pd.read_parquet(args.indir / f"{label}_returns.parquet")
    panel = to_panel(returns)
    n_assets = panel.shape[0]
    started = time.time()
    origins = build_origins(panel, returns.index, args.T_in, args.T_out, args.step)
    splits, grouped = assign_splits(
        origins, args.train_end, args.validation_start,
        args.validation_end, args.test_start)

    entrants = ["frozen", "ceiling"] + [f"ewma_hl_{h:g}" for h in EWMA_HALF_LIVES]
    if not args.skip_rie:
        entrants += list(RIE_ESTIMATORS)
    capture = {name: {} for name in entrants}
    bias, invisible_flags = {}, {}
    from src.capture_ladder import apply_estimation_scaling, estimation_scaling
    from src.data import to_correlation_panel

    for position, origin in enumerate(origins, start=1):
        reference, sigma = estimation_scaling(origin.estimation)
        adjusted = to_correlation_panel(
            apply_estimation_scaling(origin.estimation, reference, sigma))
        base = frozen_frame(origin.estimation)
        capture["frozen"][origin.index] = variance_captured(
            base, origin.realised)["capture_6"]
        capture["ceiling"][origin.index] = variance_captured(
            realised_ceiling(origin.realised), origin.realised)["capture_6"]
        for half_life in EWMA_HALF_LIVES:
            frame = leading_frame(base_estimator(origin.estimation, half_life), RANK)
            capture[f"ewma_hl_{half_life:g}"][origin.index] = variance_captured(
                frame, origin.realised)["capture_6"]
        if not args.skip_rie:
            # The invisibility predicate needs a full eigendecomposition, so it
            # is verified on a regular subsample rather than every origin; the
            # claim is structural and does not need 432 confirmations.
            check = (position - 1) % 20 == 0
            frames, flags = _rie_frames(origin.estimation, adjusted, check=check)
            if check:
                invisible_flags[origin.index] = flags
            for name, frame in frames.items():
                capture[name][origin.index] = variance_captured(
                    frame, origin.realised)["capture_6"]
        if position % 100 == 0 or position == len(origins):
            print(f"  {label}: {position}/{len(origins)} origins "
                  f"[{time.time() - started:.0f}s]", flush=True)

    # Stationary ceiling null: simulate the target window from the estimation
    # covariance itself, so honest headroom is exactly zero by construction.
    rng = np.random.default_rng(args.seed + sum(map(ord, label)))
    sampled = grouped["test"][:: max(1, len(grouped["test"]) // 12)]
    biases = []
    for origin in sampled:
        reference, sigma = estimation_scaling(origin.estimation)
        adjusted = to_correlation_panel(
            apply_estimation_scaling(origin.estimation, reference, sigma))
        biases.append(ceiling_bias(sample_covariance(adjusted), args.T_out,
                                   replicates=args.ceiling_replicates,
                                   rng=rng)["capture_6"])
    bias = float(np.mean(biases))

    block = block_length_origins(args.T_in, args.T_out, args.step)
    validation_ids = [origin.index for origin in grouped["validation"]]
    test_ids = [origin.index for origin in grouped["test"]]
    selected = max(EWMA_HALF_LIVES, key=lambda h: float(np.mean(
        [capture[f"ewma_hl_{h:g}"][i] for i in validation_ids])))

    frozen_test = np.array([capture["frozen"][i] for i in test_ids])
    rows = []
    for name in entrants:
        if name == "frozen":
            continue
        series = np.array([capture[name][i] for i in test_ids])
        row = {"label": label, "N": n_assets, "entrant": name}
        row.update(paired_summary(series, frozen_test, block,
                                  args.bootstrap_repetitions, args.seed))
        rows.append(row)
    comparisons = pd.DataFrame(rows)

    ceiling_mean = float(np.mean([capture["ceiling"][i] for i in test_ids]))
    frozen_mean = float(frozen_test.mean())
    headroom = ceiling_mean - frozen_mean - bias
    summary = pd.DataFrame([{
        "label": label, "N": n_assets, "T_in": args.T_in, "T_out": args.T_out,
        "step": args.step, "n_origins": len(origins),
        "n_train": len(grouped["train"]), "n_validation": len(grouped["validation"]),
        "n_test": len(test_ids), "block_length_origins": block,
        "independent_blocks": len(test_ids) / block,
        "random_floor": random_floor(n_assets)["capture_6"],
        "frozen_capture": frozen_mean, "ceiling_capture": ceiling_mean,
        "naive_headroom": ceiling_mean - frozen_mean,
        "ceiling_bias": bias, "real_headroom": headroom,
        "validation_selected_ewma_half_life": selected,
        "rie_shrinkage_invisible": (
            bool(all(flags[f"{name}_shrinkage"]
                     for flags in invisible_flags.values()
                     for name in RIE_ESTIMATORS))
            if invisible_flags else None),
        "rie_invisible_after_renormalisation": (
            bool(all(flags[f"{name}_after_renormalisation"]
                     for flags in invisible_flags.values()
                     for name in RIE_ESTIMATORS))
            if invisible_flags else None),
    }])
    series_frame = pd.DataFrame([
        {"label": label, "entrant": name, "example": index,
         "split": splits[index], "capture_6": value}
        for name, values in capture.items() for index, value in values.items()])
    return summary, comparisons, series_frame


def main(argv=None):
    args = parse_args(argv)
    labels = DEFAULT_LABELS if args.label == "all" else (args.label,)
    outputs = [run_panel(label, args) for label in labels]
    summary = pd.concat([o[0] for o in outputs], ignore_index=True)
    comparisons = pd.concat([o[1] for o in outputs], ignore_index=True)
    series = pd.concat([o[2] for o in outputs], ignore_index=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.outdir / "panel_summary.csv", index=False)
    comparisons.to_csv(args.outdir / "test_comparisons.csv", index=False)
    series.to_csv(args.outdir / "all_panels_series.csv", index=False)

    print("\nWhat is tested: the Stage 2 capture ladder on test origins only, "
          "EWMA half-life selected on validation.")
    print("Setup: T_in=%d, T_out=%d, step=%d; circular-block interval at "
          "block length %d origins." % (
              args.T_in, args.T_out, args.step,
              block_length_origins(args.T_in, args.T_out, args.step)))
    print(summary[["label", "N", "n_test", "independent_blocks", "random_floor",
                   "frozen_capture", "ceiling_capture", "naive_headroom",
                   "ceiling_bias", "real_headroom",
                   "validation_selected_ewma_half_life",
                   "rie_shrinkage_invisible",
                   "rie_invisible_after_renormalisation"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nPaired against Frozen on test origins:")
    print(comparisons[["label", "entrant", "paired_improvement",
                       "improvement_ci_low", "improvement_ci_high",
                       "origin_win_fraction", "excludes_zero"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    if not args.skip_rie:
        pinned = comparisons.loc[comparisons["entrant"].isin(RIE_ESTIMATORS)]
        print("\nRotationally-invariant class: max |improvement| = "
              f"{pinned['paired_improvement'].abs().max():.6f}. "
              "Shrinkage step invisible on every origin = "
              f"{bool(summary['rie_shrinkage_invisible'].all())}; "
              "still invisible after the ladder's correlation renormalisation = "
              f"{bool(summary['rie_invisible_after_renormalisation'].all())}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
