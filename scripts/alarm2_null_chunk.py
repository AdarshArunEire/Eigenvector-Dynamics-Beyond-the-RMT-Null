"""Resume only the volatility-matched null for one Alarm 2 sweep cell.

``regime4_9_deletion_attribution.main`` recomputes the observed series on every
invocation before it reaches the null loop, which wastes an entire window pass
per resume.  This entry point calls ``run_null`` directly.  It writes the same
per-replicate checkpoint rows to the same path, so a cell can be filled by any
mixture of this runner and the full Regime 4.9 runner.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.regime4_9_deletion_attribution import run_null
from src.data import to_panel


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--shuffles", type=int, default=99)
    parser.add_argument("--indir", type=Path, default=Path("data/cache"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("results/alarm2_window_sweep"))
    parser.add_argument("--calendar-block-size", type=int, default=21)
    parser.add_argument("--volatility-block-size", type=int, default=42)
    parser.add_argument("--volatility-bins", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    panel = to_panel(pd.read_parquet(a.indir / f"{a.label}_returns.parquet"))
    a.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.label}_T{a.T}_h{a.horizon}_step{a.step}"
    args = SimpleNamespace(
        outdir=a.outdir, shuffles=a.shuffles, seed=a.seed, step=a.step,
        horizon=a.horizon, calendar_block_size=a.calendar_block_size,
        volatility_block_size=a.volatility_block_size,
        volatility_bins=a.volatility_bins)
    done = run_null(panel, a.T, args, "volatility", stem)
    print(f"{stem}: {done['replicate'].nunique()} replicates on disk",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
