"""Add the missing p-value for ``mean_full_residual_cosine``.

Regime 4.9 reports ``mean_full_residual_cosine`` in every summary table but
never computed a p-value for it, which made it look like an orphan statistic
with no null.  It is not: the null draws are already the last column of every
``*_calendar_null.csv`` and ``*_volatility_null.csv``.  Only the comparison was
missing.  This script computes it from the existing files -- no re-run of the
permutation pipeline is required.

    python -m scripts.alarm1_residual_cosine_pvalue
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATISTIC = "mean_full_residual_cosine"


def upper_tail_p(observed, draws):
    """Permutation p with the observed value included, so p is never zero."""
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return np.nan
    return float((1 + np.sum(draws >= observed)) / (1 + len(draws)))


def holm(pvalues):
    """Holm step-down adjustment across the four panels."""
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def collect(indir):
    rows = []
    for summary_path in sorted(Path(indir).glob("*_summary.csv")):
        summary = pd.read_csv(summary_path)
        stem = str(summary_path)[:-len("_summary.csv")]
        nulls = {}
        for name in ("calendar", "volatility"):
            path = Path(f"{stem}_{name}_null.csv")
            if path.exists():
                nulls[name] = pd.read_csv(path)
        for _, row in summary.iterrows():
            record = {"label": row["label"], "T": row["T"],
                      "component": row["component"], "observed": row[STATISTIC]}
            for name, frame in nulls.items():
                draws = frame.loc[frame["component"] == row["component"], STATISTIC]
                record[f"{name}_null_mean"] = float(np.mean(draws))
                record[f"{name}_p"] = upper_tail_p(row[STATISTIC], draws)
            rows.append(record)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indir", type=Path, default=Path("results/regime4_9"))
    parser.add_argument("--outfile", type=Path,
                        default=Path("results/regime4_9/residual_cosine_pvalues.csv"))
    args = parser.parse_args(argv)

    table = collect(args.indir)
    if table.empty:
        raise SystemExit(f"no Regime 4.9 summaries found in {args.indir}")
    for name in ("calendar", "volatility"):
        column = f"{name}_p"
        if column in table:
            table[f"{name}_p_holm"] = np.nan
            for component, group in table.groupby("component"):
                table.loc[group.index, f"{name}_p_holm"] = holm(group[column].to_numpy())
    table.to_csv(args.outfile, index=False)
    shown = [c for c in ("label", "component", "observed", "volatility_null_mean",
                         "volatility_p", "volatility_p_holm") if c in table]
    print(table[shown].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {args.outfile}")
    return table


if __name__ == "__main__":
    main()
