"""Regime 4.2, cross-panel view: the windowing check on one common axis.

    python scripts/regime4_2_compare.py

Reads the CSVs `regime4_2_subspace.py` writes and puts all four panels on the
same `tau/T` grid, which is the only axis on which they are comparable -- the
windows differ by more than an order of magnitude in absolute days (T = 26 to
357) but the overlap artifact is a function of `tau/T` alone.

What the table has to show, if the pairing code is right:

    tau/T -> 0     D_emp -> 0        windows share almost all their returns
    tau/T ~ 0.5    D_emp ~ D_th      the curve crosses its own null
    tau/T > 1      flat in D_th      no shared returns; only real motion left

The paper states the shape outright -- the curve "starts from 0 for tau = 0 and
increases to reach the stationary noise level at time tau = T ... simply due to
the overlapping between the sliding periods". Four independent panels
reproducing it is a free correctness check that costs nothing and would have
caught a pairing bug before it contaminated everything downstream.

Values are linearly interpolated onto the common grid. DAX and CAC 40 have
tau/T resolution of only ~0.19 per step because their windows are so short, so
their rows are the interpolated ones; S&P 500 and Nikkei are sampled at ~0.04.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PANELS = [("sp500", "S&P 500"), ("nikkei", "Nikkei"),
          ("dax", "DAX"), ("cac40", "CAC 40")]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", type=Path, default=Path("results"))
    p.add_argument("--P", type=int, default=3)
    p.add_argument("--Q", type=int, default=6)
    p.add_argument("--tag", default="raw", choices=["raw", "standardised"])
    p.add_argument("--grid", type=float, nargs="+",
                   default=[0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    return p.parse_args(argv)


def load(indir, label, P, Q):
    """Find the T=N run for a panel and return (T, N, frame indexed by tau/T)."""
    hits = sorted(indir.glob(f"{label}_subspace_T*_P{P}Q{Q}.csv"))
    if not hits:
        raise SystemExit(f"no CSV for {label} -- run regime4_2_subspace.py --label {label}")
    # T = N is the paper's rule; if several T were run, take the one matching N.
    for path in hits:
        d = pd.read_csv(path, header=[0, 1], index_col=0)
        T = int(d[("T", "Unnamed: 12_level_1")].iloc[0]) if ("T", "Unnamed: 12_level_1") in d \
            else int(d.filter(like="T").iloc[0, 0])
        N = int(d.filter(like="N").iloc[0, 0])
        if T == N:
            d.index = d.index / T
            return T, N, d
    raise SystemExit(f"{label}: no T=N run among {[p.name for p in hits]}")


def main(argv=None):
    a = parse_args(argv)
    cols, meta = {}, {}
    for label, pretty in PANELS:
        T, N, d = load(a.indir, label, a.P, a.Q)
        meta[pretty] = (T, N)
        x = d.index.to_numpy(dtype=float)
        ratio = (d[(a.tag, "d_emp")] / d[(a.tag, "d_th")]).to_numpy(dtype=float)
        cols[pretty] = np.interp(a.grid, x, ratio, left=np.nan, right=np.nan)

    table = pd.DataFrame(cols, index=pd.Index(a.grid, name="tau/T"))

    print(f"Regime 4.2 -- D_emp / D_th against tau/T   [{a.tag}, P={a.P}, Q={a.Q}, T=N]\n")
    head = " | ".join(f"{p} (T={meta[p][0]})" for p, _ in
                      [(k, None) for k in table.columns])
    print(f"| tau/T | {head} |")
    print("|" + "---|" * (len(table.columns) + 1))
    for x, row in table.iterrows():
        cells = " | ".join("--" if not np.isfinite(v) else f"{v:.2f}x" for v in row)
        print(f"| {x:.2f} | {cells} |")

    print("\ncrossing point (tau/T where D_emp first reaches D_th), by interpolation:")
    for label, pretty in PANELS:
        T, N, d = load(a.indir, label, a.P, a.Q)
        x = d.index.to_numpy(dtype=float)
        r = (d[(a.tag, "d_emp")] / d[(a.tag, "d_th")]).to_numpy(dtype=float)
        m = np.isfinite(r)
        cross = np.interp(1.0, r[m], x[m]) if r[m].min() < 1.0 < r[m].max() else np.nan
        # flatness: fractional change in D_emp over tau/T in [1.2, 2.0], against
        # the change over [0.2, 1.0]. Small ratio = the climb has stopped.
        e = d[(a.tag, "d_emp")].to_numpy(dtype=float)
        climb = np.interp(1.0, x, e) - np.interp(0.2, x, e)
        after = np.interp(2.0, x, e) - np.interp(1.2, x, e)
        print(f"  {pretty:>9} (T={T:>3}, N={N:>3}): crosses at tau/T = {cross:.2f}"
              f"   post-T slope / pre-T slope = {after / climb:.2f}")
    print("\n  A crossing near 0.5 and a post/pre ratio well below 1 is the artifact"
          "\n  the paper describes. Residual post-T climb is genuine motion, not overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
