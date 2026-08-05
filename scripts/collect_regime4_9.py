"""Collect the predeclared complete-Flag verdict for Regime 4.9."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime4_4_tangent import empirical_upper_p
from scripts.regime4_7_flag import holm_adjust
from scripts.regime4_9_deletion_attribution import DEFAULT_LABELS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indir", type=Path,
                        default=Path("results/regime4_9"))
    return parser.parse_args(argv)


def _one(path_pattern, directory):
    matches = sorted(directory.glob(path_pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {path_pattern}, found {len(matches)}")
    return matches[0]


def main(argv=None):
    args = parse_args(argv)
    rows, null_values = [], {"calendar": {}, "volatility": {}}
    for label in DEFAULT_LABELS:
        summary = pd.read_csv(_one(f"{label}_*_summary.csv", args.indir))
        primary = summary.loc[summary["component"] == "flag_nested"].iloc[0]
        row = {
            "label": label,
            "mean_full_cosine": primary["mean_full_cosine"],
            "mean_deletion_attributed_fraction":
                primary["mean_deletion_attributed_fraction"],
            "mean_full_residual_cosine": primary["mean_full_residual_cosine"],
            "mean_addition_cosine": primary["mean_addition_cosine"],
            "positive_addition_fraction": primary["positive_addition_fraction"],
        }
        for kind in ("calendar", "volatility"):
            null = pd.read_csv(_one(f"{label}_*_{kind}_null.csv", args.indir))
            values = null.loc[
                null["component"] == "flag_nested",
                "mean_addition_cosine"].to_numpy(dtype=float)
            if len(values) != 99:
                raise ValueError(f"{label} {kind} has {len(values)} nulls, need 99")
            null_values[kind][label] = values
            row[f"{kind}_null_mean"] = float(np.mean(values))
            row[f"{kind}_null_q95"] = float(np.quantile(values, .95))
            row[f"{kind}_p"] = empirical_upper_p(
                row["mean_addition_cosine"], values)
        rows.append(row)

    panels = pd.DataFrame(rows)
    for kind in ("calendar", "volatility"):
        panels[f"{kind}_p_holm"] = holm_adjust(
            panels[f"{kind}_p"].to_numpy(dtype=float))
    panels["passes_both_raw"] = ((panels["calendar_p"] <= .05)
                                  & (panels["volatility_p"] <= .05))
    panels["passes_both_holm"] = ((panels["calendar_p_holm"] <= .05)
                                   & (panels["volatility_p_holm"] <= .05))
    panels.to_csv(args.indir / "all_panels_primary.csv", index=False)

    observed = float(panels["mean_addition_cosine"].mean())
    combined = {"mean_addition_cosine": observed}
    for kind in ("calendar", "volatility"):
        matrix = np.vstack([null_values[kind][label]
                            for label in DEFAULT_LABELS])
        values = matrix.mean(axis=0)
        combined[f"{kind}_null_mean"] = float(np.mean(values))
        combined[f"{kind}_null_q95"] = float(np.quantile(values, .95))
        combined[f"{kind}_p"] = empirical_upper_p(observed, values)
    combined["passes_both"] = (combined["calendar_p"] <= .05
                               and combined["volatility_p"] <= .05)
    combined_frame = pd.DataFrame([combined])
    combined_frame.to_csv(args.indir / "combined_primary.csv", index=False)

    print("Panel verdicts")
    print(panels.to_string(index=False,
          float_format=lambda value: f"{value:.4f}"))
    print("\nEqual-market verdict")
    print(combined_frame.to_string(index=False,
          float_format=lambda value: f"{value:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
