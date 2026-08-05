"""Alarm 1 evidence: does the calendar block null preserve volatility clustering?

The deletion-direction alignment measured in Regime 4.9 is driven by volatility
clustering at the horizon scale.  A null that destroys that clustering is not a
control for it.  This script measures the lag-1 autocorrelation of log realised
variance at several non-overlapping aggregation scales, under the observed panel
and under both nulls, and shows that the 21-day calendar permutation annihilates
clustering beyond ~42 days while the 42-day volatility-matched permutation
retains roughly 83% of it.

The table this produces is the main evidence for Alarm 1 and previously existed
only as prose in ALARM1_DRAFT.md.  Run:

    python -m scripts.alarm1_rv_autocorrelation --label sp500_full
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import to_panel
from scripts.regime4_9_deletion_attribution import volatility_matched_block_indices

DEFAULT_SCALES = (1, 14, 42, 63)


def log_rv_autocorrelation(panel, scales=DEFAULT_SCALES):
    """Lag-1 autocorrelation of log realised variance by aggregation scale.

    Realised variance is the cross-sectional mean squared return, which
    estimates the day's overall variance level without needing a model of it.
    Aggregation is non-overlapping so the statistic is not itself a
    rolling-window artefact -- the exact trap Alarm 1 is about.
    """
    daily = (np.asarray(panel, dtype=float) ** 2).mean(axis=0)
    out = {}
    for scale in scales:
        blocks = len(daily) // scale
        if blocks < 3:
            out[scale] = np.nan
            continue
        aggregated = daily[:blocks * scale].reshape(blocks, scale).sum(axis=1)
        series = np.log(aggregated[aggregated > 0])
        out[scale] = float(np.corrcoef(series[:-1], series[1:])[0, 1])
    return out


def calendar_block_indices(days, block_size, rng):
    """Plain block permutation: the Regime 4.7 calendar null."""
    blocks = [np.arange(start, min(start + block_size, days))
              for start in range(0, days, block_size)]
    lengths = np.asarray([len(block) for block in blocks])
    order = np.arange(len(blocks))
    for length in np.unique(lengths):
        eligible = np.flatnonzero(lengths == length)
        if len(eligible) > 1:
            order[eligible] = rng.permutation(eligible)
    return np.concatenate([blocks[index] for index in order])


def run(panel, replicates, calendar_block, volatility_block, bins, seed,
        scales=DEFAULT_SCALES):
    observed = log_rv_autocorrelation(panel, scales)
    calendar, volatility = [], []
    for replicate in range(int(replicates)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0, replicate]))
        calendar.append(log_rv_autocorrelation(
            panel[:, calendar_block_indices(panel.shape[1], calendar_block, rng)],
            scales))
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1, replicate]))
        volatility.append(log_rv_autocorrelation(
            panel[:, volatility_matched_block_indices(
                panel, volatility_block, bins, rng)], scales))
    return pd.DataFrame([{
        "scale_days": scale,
        "observed": observed[scale],
        f"calendar_{calendar_block}_null": np.mean([r[scale] for r in calendar]),
        f"volatility_{volatility_block}_null": np.mean([r[scale] for r in volatility]),
        "n_replicates": int(replicates),
    } for scale in scales])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="sp500_full")
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--outdir", type=Path, default=Path("results/alarm1"))
    parser.add_argument("--replicates", type=int, default=25)
    parser.add_argument("--calendar-block-size", type=int, default=21)
    parser.add_argument("--volatility-block-size", type=int, default=42)
    parser.add_argument("--volatility-bins", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args(argv)

    panel = to_panel(pd.read_parquet(args.cache / f"{args.label}_returns.parquet"))
    table = run(panel, args.replicates, args.calendar_block_size,
                args.volatility_block_size, args.volatility_bins, args.seed)
    table.insert(0, "label", args.label)
    args.outdir.mkdir(parents=True, exist_ok=True)
    path = args.outdir / f"{args.label}_rv_autocorrelation.csv"
    table.to_csv(path, index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print(f"\nwrote {path}")
    return table


if __name__ == "__main__":
    main()
