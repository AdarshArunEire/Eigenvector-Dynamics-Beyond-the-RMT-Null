"""Regime 4.1: the eigenvalue variogram on real returns. Fig. 7 of arXiv:1203.6228.

    python scripts/regime4_1_variogram.py --label nikkei

The first empirical step of the paper, and the one I skipped on the way to
eigenvectors. It asks only about eigenvalues, so it needs no P and no Q, and it
is the cheapest possible test of whether the panel and the preprocessing are
sound before any subspace machinery touches them.

    <(lam_i^s - lam_i^t)^2>_{|t-s|=tau}   against   4 lam_i^2 / T

Three things come out:

1. **A correctness check on the windowing, for free.** The paper states that the
   empirical curve "starts from 0 for tau = 0 and increases to reach the
   stationary noise level at time tau = T ... simply due to the overlapping
   between the sliding periods". That is a known artifact with a known shape. If
   my sliding-window code reproduces the rise-to-plateau at exactly tau = T, the
   pairing is right. If it does not, there is a bug that would have contaminated
   every later regime silently.

2. **The standardisation decision, tested outside the synthetic world.** Regime
   1.5 showed on simulated data that standardising at window=1 removes the
   fat-tail inflation of the null. Eq (4.8) carries the same (nu-2)/(nu-4) factor
   as Eq (4.7), so the same remedy should apply here. If the standardised
   variogram sits closer to 4 lam^2 / T than the raw one, that conclusion
   survives contact with real returns and I can stop fitting nu for the rest of
   stage 1.

3. **Whether the eigenvalues move at all**, which is the paper's actual claim:
   the empirical variogram is "much larger than the theoretical prediction",
   establishing genuine time evolution before any eigenvector question is asked.

Window length is T = N throughout, which is the paper's rule, not a choice.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import to_panel, standardise, to_correlation_panel
from src.overlap import spectral, sample_covariance


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="nikkei")
    p.add_argument("--indir", type=Path, default=Path("data/cache"))
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--T", type=int, default=None,
                   help="window length. Defaults to N, which is the paper's rule.")
    p.add_argument("--modes", type=int, default=3, help="how many eigenvalues to track")
    p.add_argument("--step", type=int, default=None, help="days between window starts")
    return p.parse_args(argv)


def window_eigenvalues(panel, T, step, k, standardised):
    """Top-k correlation eigenvalues of every window. One eigh per window."""
    out = {}
    for s in range(0, panel.shape[1] - T + 1, step):
        w = panel[:, s:s + T]
        if standardised:
            w = standardise(w, window=1)
        lam, _ = spectral(sample_covariance(to_correlation_panel(w)))
        out[s] = lam[:k]
    return out


def variogram(evals, step, k):
    """<(lam_i^s - lam_i^t)^2> binned by lag, including lags below T.

    Short lags are kept deliberately: overlapping windows share data, which
    forces their eigenvalues to agree and drives the variogram to zero at
    tau = 0. That rise is the calibration check, not a nuisance.
    """
    starts = sorted(evals)
    rows = {}
    for i, s in enumerate(starts):
        for t in starts[i + 1:]:
            d = (evals[s] - evals[t]) ** 2
            rows.setdefault(t - s, []).append(d)
    return pd.DataFrame({lag: np.mean(v, axis=0) for lag, v in sorted(rows.items())},
                        index=[f"mode{j + 1}" for j in range(k)]).T


def main(argv=None):
    a = parse_args(argv)
    rets = pd.read_parquet(a.indir / f"{a.label}_returns.parquet")
    panel = to_panel(rets)
    N, total = panel.shape
    T = a.T or N
    step = a.step or max(5, T // 25)
    k = a.modes

    lam_all, _ = spectral(sample_covariance(to_correlation_panel(panel)))
    theory = 4.0 * lam_all[:k] ** 2 / T

    print(f"{a.label}: N={N}, {total} days, T=N={T}, step={step}")
    print(f"  whole-period eigenvalues {np.round(lam_all[:k], 2)}  "
          f"(lam1/N = {lam_all[0] / N:.3f}; paper's Nikkei is 0.358)")
    print(f"  Eq (4.8) null 4*lam_i^2/T = {np.round(theory, 1)}\n")

    out = {}
    for tag, std in (("raw", False), ("standardised", True)):
        v = variogram(window_eigenvalues(panel, T, step, k, std), step, k)
        out[tag] = v

    v = out["raw"]
    lags = [l for l in v.index if l <= 3 * T]
    print("  the overlap artifact: variogram must climb from ~0 and plateau at tau = T")
    print(f"  {'tau':>7} {'tau/T':>6} " + " ".join(f"{f'mode{j+1}':>11}" for j in range(k))
          + "   ratio to null (mode1)")
    for l in [x for x in lags if x <= 2 * T][::max(1, len(lags) // 10)]:
        r = v.loc[l]
        print(f"  {l:>7} {l / T:>6.2f} " + " ".join(f"{r.iloc[j]:>11.2f}" for j in range(k))
              + f"   {r.iloc[0] / theory[0]:>8.1f}x")

    print(f"\n  {'':>14}" + " ".join(f"{f'mode{j+1}':>10}" for j in range(k)))
    for tag in ("raw", "standardised"):
        vv = out[tag]
        beyond = vv.loc[[l for l in vv.index if l >= T]]
        ratio = beyond.iloc[0] / theory
        print(f"  {tag:>12}: " + " ".join(f"{ratio.iloc[j]:>9.1f}x" for j in range(k))
              + f"   at tau = {beyond.index[0]}")
    print(f"\n  A ratio far above 1 is the paper's result -- the eigenvalues of C"
          f"\n  genuinely move, and measurement noise cannot account for it."
          f"\n  Standardising should shrink the ratio if part of the excess was"
          f"\n  fat tails rather than evolution (regime 1.5, Eq 4.7 = Eq 4.8 factor).")

    a.outdir.mkdir(parents=True, exist_ok=True)
    merged = pd.concat({t: out[t] for t in out}, axis=1)
    merged["T"], merged["N"] = T, N
    for j in range(k):
        merged[f"null_mode{j + 1}"] = theory[j]
    path = a.outdir / f"{a.label}_variogram_T{T}.csv"
    merged.to_csv(path)
    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
