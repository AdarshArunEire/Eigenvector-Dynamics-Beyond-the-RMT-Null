"""Model 4.1 -- a five-parameter correction confined to the visible block.

    M_t = C_t(theta) + eps * sum_m beta_m (U_perp A^m_t U_6^T + U_6 A^m_t^T U_perp^T)

``C_t(theta)`` is an EWMA correlation of the estimation window, so the family
contains Frozen (``eps=0, theta=inf``) and every EWMA (``eps=0``) exactly.  Each
``A^m_t`` is a cheap observable projected into ``U_perp^T . U_6`` -- the only
block of a symmetric correction the capture score can see -- and only the
scalars are learned.  Not one parameter is spent on something invisible.

Fitting order, fixed in advance:

1.  ``beta`` on **train** origins, four ways, because *what you optimise* is
    itself a modelling choice and this design lets it be tested rather than
    assumed.  All four spend the same parameters and differ only in the loss:

    ===========  ==========================================================
    ``gradient`` **risk loss.**  Maximise realised capture to first order:
                 ``max sum_t <sum_m beta_m A^m/Lambda, Sigma^oi>``.  Only the
                 alignment with realised variance counts, and the ``1/Lambda``
                 weight makes cheap-to-move directions worth more.
    ``ridge``    **geometry loss, metric coordinates.**  Least squares
                 predicting ``Sigma^oi`` from the gap-weighted features.
                 Penalises every entry of the target, including those the
                 score cannot see.
    ``geometric``**pure geometry loss.**  The same least squares with
                 ``1/Lambda`` dropped, so all directions weigh equally: what a
                 model trained to forecast subspace motion, with no reference
                 to the risk metric, would fit.
    ``direct``   **risk loss, exactly.**  Search the sphere for the beta
                 maximising true realised train capture of the exactly
                 corrected frame.  Checks the first-order surrogate.
    ===========  ==========================================================

    The risk and geometry losses coincide only if realised variance were
    isotropic across the complement.  It is not: complement variance
    concentrates near the top of the bulk, which is also where the eigengaps
    are smallest, so the two objectives pull on the same directions for
    opposite reasons.  The reported ``fit_agreement_*`` columns are the cosines
    between each fitted beta and the risk-optimal beta, which is the direct
    measurement of whether the distinction matters on this data.
2.  ``eps``, ``theta`` and the choice among the four ``beta`` on **validation**
    origins, by realised capture.  Note that validation scores the *true*
    capture of the exactly-corrected frame, never the first-order surrogate:
    the perturbation theory chose what to fit, it does not do the scoring.
3.  **Test** origins once, paired against the same-``theta`` EWMA, with a
    circular-block interval.

**Predeclared stopping rule.**  Model 4.1 must beat the validation-selected
EWMA on *validation* origins with a circular-block interval at 57-origin blocks
that excludes zero.  If it does not, it stops.  No re-tuning, no feature
hunting, no third half-life.  ``STOPPING_RULE`` below is evaluated by the
script and printed as PASS or STOP, and the test-set numbers are computed
either way so that a failure is documented rather than buried -- but a STOP
verdict means the test number is a postmortem, not a result.

**Predeclared expectation.**  Real headroom is 0.047-0.055.  EWMA takes
0.0015-0.0032 of it, i.e. 3-6%.  A metric-aware five-parameter model plausibly
reaches 1.5-2x EWMA, so 6-12% of honest headroom, ~0.005 capture, ~0.5% of
residual volatility.  That is economically nil, and it is written down here
before anything is fitted.  The value of the exercise is not the 0.5%: it is
that a model built from the exact first-order condition, spending every
parameter on the only visible block, still cannot close the gap.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capture import realised_ceiling, variance_captured
from src.capture_ladder import (apply_estimation_scaling, assign_splits,
                                block_length_origins, build_origins,
                                estimation_scaling, frozen_frame,
                                paired_summary)
from src.coupling import (ALL_FEATURES, DEFAULT_FEATURES, RANK, base_estimator,
                          causal_standardise, combine, corrected_frame,
                          descending_spectrum, direct_capture_beta,
                          eigengap_matrix, feature_collinearity,
                          feature_fast_slow, feature_hierarchy,
                          feature_momentum, feature_stress, geometric_beta,
                          gradient_beta, realised_visible_target, ridge_beta)
from src.data import to_panel

DEFAULT_LABELS = ("cac40_full", "dax_full", "nikkei_full", "sp500_full")
BASE_HALF_LIVES = (126., 252., 504., float("inf"))
EPSILONS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
#: The four fits differ only in *what they optimise*, never in how many
#: parameters they spend.  ``gradient`` maximises realised risk capture to first
#: order; ``ridge`` predicts the realised subspace motion in gap-weighted
#: coordinates; ``geometric`` predicts the same motion with the eigengap ignored
#: entirely, i.e. pure geometry with no reference to the risk metric;
#: ``direct`` maximises the true realised capture of the exactly-corrected frame
#: by search, and exists to check that the first-order surrogate is not lying.
FITS = ("gradient", "ridge", "geometric", "direct")

#: Evaluated verbatim below.  Written before any result was seen.
STOPPING_RULE = (
    "Model 4.1 beats the validation-selected EWMA on validation origins with a "
    "circular-block interval at 57-origin blocks excluding zero.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=DEFAULT_LABELS + ("all",), default="all")
    parser.add_argument("--indir", type=Path, default=Path("data/cache"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("results/stage2/model4_1_visible_coupling"))
    parser.add_argument("--T-in", type=int, default=750)
    parser.add_argument("--T-out", type=int, default=42)
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--features", nargs="+", choices=ALL_FEATURES,
                        default=list(DEFAULT_FEATURES))
    parser.add_argument("--fast-half-life", type=float, default=42.0)
    parser.add_argument("--slow-half-life", type=float, default=504.0)
    parser.add_argument("--gap-floor", type=float, default=1e-3)
    parser.add_argument("--half-lives", nargs="+", type=float, default=None,
                        help="override the base-kernel grid (theta)")
    parser.add_argument("--epsilons", nargs="+", type=float, default=None,
                        help="override the amplitude grid (epsilon)")
    parser.add_argument("--ridge-penalty", type=float, default=1e-3)
    parser.add_argument("--direct-directions", type=int, default=96,
                        help="sphere grid size for the exact-capture fit")
    parser.add_argument("--direct-origin-cap", type=int, default=140,
                        help="train origins subsampled for the exact-capture fit")
    parser.add_argument("--direct-epsilon", type=float, default=0.02,
                        help="reference amplitude at which the exact fit searches")
    parser.add_argument("--skip-direct", action="store_true",
                        help="omit the exact-capture fit (it is the slow one)")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--train-end", default="2013-12-31")
    parser.add_argument("--validation-start", default="2015-07-01")
    parser.add_argument("--validation-end", default="2018-06-30")
    parser.add_argument("--test-start", default="2020-01-01")
    return parser.parse_args(argv)


def build_design(origins, half_life, features, args):
    """Per-origin spectra, gap matrices, feature stack and realised label.

    Every quantity is a function of the estimation window alone except the
    label, which is the realised target and is used only in fitting.  The
    stress covariate is standardised causally over origins, so origin ``t``
    sees only origins strictly before it.
    """
    spectra, gaps, labels, blocks, gap_bound = [], [], [], [], []
    log_variance = []
    for origin in origins:
        covariance = base_estimator(origin.estimation, half_life)
        values, vectors = descending_spectrum(covariance)
        gap, bound = eigengap_matrix(values, RANK, floor=args.gap_floor)
        spectra.append((values, vectors))
        gaps.append(gap)
        gap_bound.append(bound)
        labels.append(realised_visible_target(origin.realised, vectors, RANK))
        reference, sigma = estimation_scaling(origin.estimation)
        recent = apply_estimation_scaling(
            origin.estimation[:, -args.T_out:], reference, sigma)
        log_variance.append(np.log(max(float(np.mean(recent ** 2)), 1e-300)))
        blocks.append(None)
    stress = causal_standardise(np.asarray(log_variance))

    designs = []
    for position, origin in enumerate(origins):
        values, vectors = spectra[position]
        computed = {}
        if "fast_slow" in features or "stress" in features:
            computed["fast_slow"] = feature_fast_slow(
                origin.estimation, vectors, RANK,
                fast=args.fast_half_life, slow=args.slow_half_life)
        if "momentum" in features:
            previous = origins[max(position - 3, 0)]
            computed["momentum"] = (
                feature_momentum(frozen_frame(previous.estimation), vectors, RANK)
                if position >= 3 else np.zeros_like(gaps[position]))
        if "stress" in features:
            computed["stress"] = feature_stress(
                computed["fast_slow"], stress[position])
        if "hierarchy" in features:
            computed["hierarchy"] = feature_hierarchy(
                origin.estimation, vectors, RANK)
        designs.append(np.stack([computed[name] for name in features]))
    return {"spectra": spectra, "gaps": gaps, "labels": labels,
            "designs": designs, "gap_floor_bound": float(np.mean(gap_bound))}


def capture_series(origins, design, beta, epsilon, indices):
    """Realised capture of the exactly-corrected frame at every origin."""
    out = []
    for index in indices:
        values, vectors = design["spectra"][index]
        block = combine(design["designs"][index], beta)
        frame = corrected_frame(values, vectors, block, epsilon, RANK)
        out.append(variance_captured(frame, origins[index].realised)["capture_6"])
    return np.asarray(out)


def run_panel(label, args):
    returns = pd.read_parquet(args.indir / f"{label}_returns.parquet")
    panel = to_panel(returns)
    n_assets = panel.shape[0]
    started = time.time()
    origins = build_origins(panel, returns.index, args.T_in, args.T_out, args.step)
    splits, grouped = assign_splits(
        origins, args.train_end, args.validation_start,
        args.validation_end, args.test_start)
    train = [o.index for o in grouped["train"]]
    validation = [o.index for o in grouped["validation"]]
    test = [o.index for o in grouped["test"]]
    block = block_length_origins(args.T_in, args.T_out, args.step)
    features = tuple(args.features)

    frozen = np.array([variance_captured(frozen_frame(o.estimation),
                                         o.realised)["capture_6"]
                       for o in origins])
    ceiling = np.array([variance_captured(realised_ceiling(o.realised),
                                          o.realised)["capture_6"]
                        for o in origins])

    half_lives = tuple(args.half_lives) if args.half_lives else BASE_HALF_LIVES
    epsilons = tuple(args.epsilons) if args.epsilons else EPSILONS
    designs, betas, tuning = {}, {}, []
    for half_life in half_lives:
        design = build_design(origins, half_life, features, args)
        designs[half_life] = design
        train_designs = [design["designs"][i] for i in train]
        train_labels = [design["labels"][i] for i in train]
        train_gaps = [design["gaps"][i] for i in train]
        fitted = {
            "gradient": gradient_beta(train_designs, train_labels, train_gaps),
            "ridge": ridge_beta(train_designs, train_labels, train_gaps,
                                penalty=args.ridge_penalty),
            "geometric": geometric_beta(train_designs, train_labels, train_gaps,
                                        penalty=args.ridge_penalty),
        }
        if args.skip_direct:
            fitted["direct"] = fitted["gradient"]
        else:
            # Subsampled so the exact search stays affordable at N=357; the
            # subsample is a stride over train origins, never a random draw
            # that could be re-rolled until it flattered the fit.
            stride = max(1, len(train) // max(1, args.direct_origin_cap))
            sampled = train[::stride]
            fitted["direct"] = direct_capture_beta(
                lambda candidate: capture_series(
                    origins, design, candidate, args.direct_epsilon,
                    sampled).mean(),
                len(features), args.direct_directions, args.seed)[0]
        betas[half_life] = fitted
        collinearity = feature_collinearity(
            [design["designs"][i] for i in train],
            [design["gaps"][i] for i in train])
        for fit in FITS:
            for epsilon in epsilons:
                series = capture_series(origins, design, fitted[fit],
                                        epsilon, validation)
                tuning.append({
                    "label": label, "half_life": half_life, "fit": fit,
                    "epsilon": epsilon,
                    "validation_mean_capture": float(series.mean()),
                    "validation_minus_frozen": float(
                        series.mean() - frozen[validation].mean()),
                    "gap_floor_bound_fraction": design["gap_floor_bound"],
                    "max_offdiagonal_collinearity": float(np.max(
                        np.abs(collinearity - np.eye(len(features))))),
                    **{f"beta_{name}": float(value)
                       for name, value in zip(features, fitted[fit])},
                })
        print(f"  {label}: half-life {half_life:g} done [{time.time() - started:.0f}s]",
              flush=True)

    tuning = pd.DataFrame(tuning)
    # The EWMA baseline is the eps=0 row: same family, same theta grid.
    baseline = tuning.loc[tuning["epsilon"] == 0.0]
    best_baseline = baseline.loc[baseline["validation_mean_capture"].idxmax()]
    best = tuning.loc[tuning["validation_mean_capture"].idxmax()]
    tuning["selected"] = tuning.index == best.name

    theta = float(best["half_life"])
    fit = str(best["fit"])
    epsilon = float(best["epsilon"])
    beta = betas[theta][fit]
    baseline_theta = float(best_baseline["half_life"])

    model_validation = capture_series(origins, designs[theta], beta, epsilon, validation)
    ewma_validation = capture_series(
        origins, designs[baseline_theta], betas[baseline_theta][fit], 0.0, validation)
    gate = paired_summary(model_validation, ewma_validation, block,
                          args.bootstrap_repetitions, args.seed)
    passed = bool(gate["paired_improvement"] > 0 and gate["improvement_ci_low"] > 0)

    model_test = capture_series(origins, designs[theta], beta, epsilon, test)
    ewma_test = capture_series(
        origins, designs[baseline_theta], betas[baseline_theta][fit], 0.0, test)
    versus_ewma = paired_summary(model_test, ewma_test, block,
                                 args.bootstrap_repetitions, args.seed)
    versus_frozen = paired_summary(model_test, frozen[test], block,
                                   args.bootstrap_repetitions, args.seed)

    naive_headroom = float(ceiling[test].mean() - frozen[test].mean())
    summary = pd.DataFrame([{
        "label": label, "N": n_assets, "features": "+".join(features),
        "n_parameters": len(features) + 2,
        "n_train": len(train), "n_validation": len(validation), "n_test": len(test),
        "block_length_origins": block,
        "independent_test_blocks": len(test) / block,
        "selected_half_life": theta, "selected_fit": fit,
        **{f"fit_agreement_{name}": float(np.dot(betas[theta][name],
                                                 betas[theta]["gradient"]))
           for name in FITS if name != "gradient"},
        "selected_epsilon": epsilon,
        "baseline_half_life": baseline_theta,
        "gap_floor_bound_fraction": float(best["gap_floor_bound_fraction"]),
        "max_offdiagonal_collinearity": float(best["max_offdiagonal_collinearity"]),
        "validation_gate_improvement": gate["paired_improvement"],
        "validation_gate_ci_low": gate["improvement_ci_low"],
        "validation_gate_ci_high": gate["improvement_ci_high"],
        "stopping_rule_passed": passed,
        "test_frozen_capture": float(frozen[test].mean()),
        "test_ceiling_capture": float(ceiling[test].mean()),
        "test_naive_headroom": naive_headroom,
        "test_model_capture": versus_ewma["model_mean_capture"],
        "test_versus_ewma": versus_ewma["paired_improvement"],
        "test_versus_ewma_ci_low": versus_ewma["improvement_ci_low"],
        "test_versus_ewma_ci_high": versus_ewma["improvement_ci_high"],
        "test_versus_frozen": versus_frozen["paired_improvement"],
        "test_versus_frozen_ci_low": versus_frozen["improvement_ci_low"],
        "test_versus_frozen_ci_high": versus_frozen["improvement_ci_high"],
        "test_versus_frozen_win_fraction": versus_frozen["origin_win_fraction"],
        **{f"beta_{name}": float(value) for name, value in zip(features, beta)},
    }])
    series = pd.DataFrame({
        "label": label,
        "example": test,
        "split": "test",
        "frozen": frozen[test],
        "ewma": ewma_test,
        "model4_1": model_test,
        "ceiling": ceiling[test],
    })
    return summary, tuning, series


def main(argv=None):
    args = parse_args(argv)
    labels = DEFAULT_LABELS if args.label == "all" else (args.label,)
    outputs = [run_panel(label, args) for label in labels]
    summary = pd.concat([o[0] for o in outputs], ignore_index=True)
    tuning = pd.concat([o[1] for o in outputs], ignore_index=True)
    series = pd.concat([o[2] for o in outputs], ignore_index=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.outdir / "panel_summary.csv", index=False)
    tuning.to_csv(args.outdir / "validation_tuning.csv", index=False)
    series.to_csv(args.outdir / "test_series.csv", index=False)

    print("\nWhat is tested: Model 4.1, a correction confined to the visible "
          f"block U_perp^T . U_6, {summary['n_parameters'].iloc[0]} parameters.")
    print(f"Predeclared stopping rule: {STOPPING_RULE}")
    print("Setup: beta fitted closed-form on train; epsilon, theta and the fit "
          "chosen on validation by realised capture; test scored once.\n")
    print(summary[["label", "N", "selected_half_life", "selected_fit",
                   "selected_epsilon", "gap_floor_bound_fraction",
                   "max_offdiagonal_collinearity",
                   "validation_gate_improvement", "validation_gate_ci_low",
                   "stopping_rule_passed"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nTest origins, paired:")
    print(summary[["label", "test_frozen_capture", "test_model_capture",
                   "test_versus_ewma", "test_versus_ewma_ci_low",
                   "test_versus_ewma_ci_high", "test_versus_frozen",
                   "test_versus_frozen_ci_low", "test_versus_frozen_ci_high"]
                  ].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    beta_columns = [c for c in summary.columns if c.startswith("beta_")]
    print("\nFitted loadings (unit norm; amplitude lives in epsilon):")
    print(summary[["label"] + beta_columns].to_string(
        index=False, float_format=lambda v: f"{v:+.3f}"))
    agreement = [c for c in summary.columns if c.startswith("fit_agreement_")]
    print("\nRisk loss vs geometry loss -- cosine of each fit's beta with the "
          "risk-optimal beta, at the selected half-life.")
    print("A value near 1 means optimising the geometry and optimising the "
          "realised risk pick the same direction on this data.")
    print(summary[["label"] + agreement].to_string(
        index=False, float_format=lambda v: f"{v:+.3f}"))
    print("\nValidation capture by fit, at each fit's own best epsilon:")
    best_by_fit = (tuning.sort_values("validation_mean_capture")
                   .groupby(["label", "fit"], as_index=False).last()
                   .pivot(index="label", columns="fit",
                          values="validation_minus_frozen"))
    print(best_by_fit.to_string(float_format=lambda v: f"{v:+.5f}"))

    passed = summary["stopping_rule_passed"]
    verdict = "PASS" if bool(passed.all()) else "STOP"
    print(f"\nVerdict: {verdict} -- validation gate cleared on "
          f"{int(passed.sum())}/{len(passed)} panels. "
          + ("Proceed to the test reading above."
             if verdict == "PASS" else
             "Predeclared rule: the model stops here. The test numbers above "
             "are a postmortem, not a result, and no re-tuning follows."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
