"""Run the Stage 2 capture ladder and Model 4.1 end to end on all four panels.

    python run_model4_1.py                 # everything, ~15-40 min
    python run_model4_1.py --quick         # skip the exact-capture search
    python run_model4_1.py --labels cac40_full dax_full

Three stages, in order, each gated on the previous one:

1.  **Tests.**  ``tests/test_coupling.py`` and ``tests/test_capture.py``.  These
    verify the algebra Model 4.1 rests on -- that a diagonal correction in the
    sample eigenbasis cannot move the score, that ``score_gradient`` really is
    the derivative of realised capture, and that the two beta estimators behave
    as advertised under planted collinearity.  If they fail, nothing below is
    worth reading and the run stops.

2.  **Capture ladder.**  Frozen, eight EWMA kernels, the three
    rotationally-invariant estimators, the in-sample ceiling, the Haar floor and
    the stationary ceiling-bias null.  Split-clean: half-life chosen on
    validation, comparison on test, circular-block intervals at 57-origin
    blocks.  This regenerates the baseline Model 4.1 is measured against, which
    previously existed only as an uncommitted run.

3.  **Model 4.1.**  The five-parameter visible-block correction, fitted four
    ways -- risk loss, gap-weighted geometry loss, pure geometry loss, and an
    exact capture search -- with the predeclared stopping rule evaluated on
    validation before the test numbers are read.

Everything lands in ``results/stage2/``.  Logs are written per stage so a long
run can be inspected while it is still going.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LADDER_DIR = ROOT / "results" / "stage2" / "capture_ladder"
MODEL_DIR = ROOT / "results" / "stage2" / "model4_1_visible_coupling"
DEFAULT_LABELS = ("cac40_full", "dax_full", "nikkei_full", "sp500_full")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--quick", action="store_true",
                        help="skip the exact-capture sphere search (the slow fit)")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-ladder", action="store_true")
    parser.add_argument("--direct-directions", type=int, default=64)
    parser.add_argument("--direct-origin-cap", type=int, default=100)
    return parser.parse_args(argv)


def _run(name, command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===\n$ {' '.join(command)}", flush=True)
    started = time.time()
    with log_path.open("w") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   bufsize=1)
        for line in process.stdout:
            handle.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        code = process.wait()
    print(f"--- {name}: exit {code} in {time.time() - started:.0f}s "
          f"(log: {log_path.relative_to(ROOT)})", flush=True)
    return code


def main(argv=None):
    args = parse_args(argv)
    labels = list(args.labels)

    if not args.skip_tests:
        code = _run("tests", [sys.executable, "-m", "pytest", "-q",
                              "tests/test_coupling.py", "tests/test_capture.py"],
                    ROOT / "results" / "stage2" / "logs" / "tests.log")
        if code != 0:
            print("\nThe coupling algebra does not hold. Stopping: every number "
                  "below it would be measuring the wrong thing.")
            return code

    if not args.skip_ladder:
        for label in labels:
            code = _run(f"ladder {label}",
                        [sys.executable, "scripts/stage2_capture_ladder.py",
                         "--label", label,
                         "--outdir", str(LADDER_DIR / label)],
                        LADDER_DIR / "logs" / f"{label}.log")
            if code != 0:
                return code

    model_command = [sys.executable,
                     "scripts/stage2_model4_1_visible_coupling.py",
                     "--direct-directions", str(args.direct_directions),
                     "--direct-origin-cap", str(args.direct_origin_cap)]
    if args.quick:
        model_command.append("--skip-direct")
    for label in labels:
        code = _run(f"model 4.1 {label}",
                    model_command + ["--label", label,
                                     "--outdir", str(MODEL_DIR / label)],
                    MODEL_DIR / "logs" / f"{label}.log")
        if code != 0:
            return code

    _combine(labels)
    return 0


def _collect(directory, labels, filename):
    frames = []
    for label in labels:
        path = directory / label / filename
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _combine(labels):
    ladder = _collect(LADDER_DIR, labels, "panel_summary.csv")
    comparisons = _collect(LADDER_DIR, labels, "test_comparisons.csv")
    model = _collect(MODEL_DIR, labels, "panel_summary.csv")
    if ladder.empty or model.empty:
        print("\nNothing to combine.")
        return

    LADDER_DIR.mkdir(parents=True, exist_ok=True)
    ladder.to_csv(LADDER_DIR / "all_panels_summary.csv", index=False)
    comparisons.to_csv(LADDER_DIR / "all_panels_test_comparisons.csv", index=False)
    model.to_csv(MODEL_DIR / "all_panels_summary.csv", index=False)

    merged = ladder.merge(model, on="label", suffixes=("_ladder", "_model"))
    best_ewma = (comparisons.loc[comparisons["entrant"].str.startswith("ewma")]
                 .groupby("label")["paired_improvement"].max())
    merged["ewma_share_of_real_headroom"] = (
        best_ewma.reindex(merged["label"]).to_numpy() / merged["real_headroom"])
    merged["model_share_of_real_headroom"] = (
        merged["test_versus_frozen"] / merged["real_headroom"])
    merged["residual_volatility_cut_percent"] = 100.0 * (
        1.0 - ((1.0 - merged["test_model_capture"]) /
               (1.0 - merged["test_frozen_capture"])) ** 0.5)

    print("\n" + "=" * 78)
    print("COMBINED READING -- honest headroom and what each entrant takes of it")
    print("=" * 78)
    print(merged[["label", "N_ladder", "random_floor", "frozen_capture",
                  "ceiling_capture", "ceiling_bias", "real_headroom",
                  "ewma_share_of_real_headroom",
                  "model_share_of_real_headroom",
                  "residual_volatility_cut_percent"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nModel 4.1: predeclared stopping rule and what optimising each loss chose")
    print(merged[["label", "selected_fit", "selected_epsilon",
                  "selected_half_life", "validation_gate_improvement",
                  "validation_gate_ci_low", "stopping_rule_passed",
                  "fit_agreement_geometric", "fit_agreement_ridge",
                  "fit_agreement_direct"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    passed = merged["stopping_rule_passed"].astype(bool)
    print(f"\nStopping rule cleared on {int(passed.sum())}/{len(passed)} panels.")
    print("Predeclared expectation was 6-12% of honest headroom, ~0.005 capture, "
          "~0.5% of residual volatility.")
    print("Measured share of honest headroom: "
          + ", ".join(f"{row.label} {row.model_share_of_real_headroom:.1%}"
                      for row in merged.itertuples()))


if __name__ == "__main__":
    raise SystemExit(main())
