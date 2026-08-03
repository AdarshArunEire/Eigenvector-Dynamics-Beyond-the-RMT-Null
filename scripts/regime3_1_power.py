"""Regime 3.1, redone: the detection threshold once returns are fat-tailed.

    python scripts/regime3_1_power.py

The original power curve was measured on Gaussian returns and gave
theta_min ~ 3.2/sqrt(T). Eq (4.7) of arXiv:1203.6228 says the noise floor is
larger by (nu-2)/(nu-4) on Student returns, and since D_inject ~ theta^2/2P
sits against that floor, the threshold should scale as sqrt((nu-2)/(nu-4)).
This checks it, and checks that standardising at window=1 undoes it.

Uses the additivity established in the original 3.1 (D_emp = D_th + D_inject to
within 2%) to avoid scanning theta: with 5% false positives and 80% power the
required injection is exactly the gap between the 95th and 20th percentiles of
the null distribution, so only the null needs simulating.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import standardise
from src.overlap import spectral, sample_covariance, subspace_distance
from src.synth import (spd_from_spectrum, factor_spectrum, gaussian_returns,
                       student_returns, student_factor)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=40)
    p.add_argument("--P", type=int, default=3)
    p.add_argument("--Q", type=int, default=6)
    p.add_argument("--trials", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def null_sample(C, T, P, Q, nu, trials, rng, std=False):
    out = np.empty(trials)
    for k in range(trials):
        pair = []
        for _ in range(2):
            r = (gaussian_returns(C, T, rng) if nu is None
                 else student_returns(C, T, nu, rng))
            if std:
                r = standardise(r, window=1)
            pair.append(spectral(sample_covariance(r))[1])
        out[k] = subspace_distance(pair[0][:, :P], pair[1][:, :Q])
    return out


def theta_min(null, P):
    """Smallest detectable rotation: 5% false positives, 80% power."""
    d_inject = np.quantile(null, 0.95) - np.quantile(null, 0.20)
    return float(np.degrees(np.arccos(np.exp(-P * d_inject))))


def main(argv=None):
    a = parse_args(argv)
    rng = np.random.default_rng(a.seed)
    lam = factor_spectrum(a.N, [12.0, 7.0, 4.0])
    C, _ = spd_from_spectrum(lam, rng)
    Ts = [250, 500, 1000]
    nus = [None, 12.0, 8.0, 6.0]

    print(f"N={a.N}, P={a.P}, Q={a.Q}, {a.trials} null trials per cell")
    print("theta_min in degrees: smallest single-pair rotation detectable at "
          "5% FPR / 80% power\n")
    print(f"  {'T':>6}" + "".join(f"{('nu=inf' if n is None else f'nu={n:.0f}'):>10}" for n in nus)
          + f"{'nu=6 std':>11}")
    tab = {}
    for T in Ts:
        row = []
        for nu in nus:
            th = theta_min(null_sample(C, T, a.P, a.Q, nu, a.trials, rng), a.P)
            row.append(th)
            tab[(T, nu)] = th
        th_std = theta_min(null_sample(C, T, a.P, a.Q, 6.0, a.trials, rng, std=True), a.P)
        tab[(T, "std")] = th_std
        print(f"  {T:>6}" + "".join(f"{x:>10.2f}" for x in row) + f"{th_std:>11.2f}")

    print(f"\n  ratio to the Gaussian column, against the predicted "
          f"sqrt((nu-2)/(nu-4)):")
    print(f"  {'T':>6}" + "".join(f"{f'nu={n:.0f}':>10}" for n in nus[1:]))
    for T in Ts:
        print(f"  {T:>6}" + "".join(f"{tab[(T, n)] / tab[(T, None)]:>10.3f}" for n in nus[1:]))
    print(f"  {'pred':>6}" + "".join(f"{np.sqrt(student_factor(n)):>10.3f}" for n in nus[1:]))

    print(f"\n  standardised nu=6 against Gaussian:")
    for T in Ts:
        print(f"    T={T:<6} {tab[(T, 'std')] / tab[(T, None)]:.3f}   "
              f"(unstandardised: {tab[(T, 6.0)] / tab[(T, None)]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
