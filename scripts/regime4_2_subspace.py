"""Regime 4.2: the subspace variogram on real returns. Fig. 8 of arXiv:1203.6228.

    python scripts/regime4_2_subspace.py --label sp500

Regime 4.1 asked whether the *eigenvalues* of C move. This asks whether the
*eigenvectors* do, which is the question the whole project exists to answer and
the first time `subspace_distance` is pointed at real data rather than a world I
built.

    D(P,Q;s,t) = -(1/2P) ln det(G^T G),    G = V_t^T U_s

with U_s the top-P eigenvectors of window s and V_t the top-Q of window t,
against the Eq (6.1) null

    D_th(s,t) = D_null(lam^s) + D_null(lam^t)

and the difference between them, which is the quantity the project reports:

    Excess(tau) = <D_emp>_{|t-s|=tau} - <D_th>_{|t-s|=tau}

Four things come out, and three of them are not in the paper.

1. **Fig. 8, replicated.** The paper plots D against tau for four indices at
   T = N. This is that plot. It is the only published number for my own
   statistic that I can check myself against, and until now the instrument had
   never touched real returns at all.

2. **Evolution against mis-specification.** Bun, Bouchaud & Potters
   (arXiv:1603.04364) measure the same kind of departure and read it as
   structure in the true C rather than as C moving. The two readings make
   different predictions and this script tests them: a mis-specified but
   *static* C produces an excess that is FLAT in tau, because the mismatch is
   the same however far apart the two windows are. Genuine evolution produces
   an excess that GROWS with tau. So the slope, not the level, is the
   discriminator, and the level alone -- which is all either paper reports --
   cannot settle it.

3. **Is the rotation persistent enough to forecast?** Linear growth in tau is
   the cheap precondition for a diffusion-like model of the subspace being
   worth fitting at all. Sub-linear or flat means the leading subspace wanders
   without going anywhere, and no forecast of it will work. This script reports
   the fit; it does not report a p-value, for a reason given under `fit_slope`.

4. **Degrees.** Regime 3.1 established that D_emp = D_th + D_inject to 0.1%
   once the injected rotation clears the noise floor, with
   D_inject = -ln(cos theta)/P. Additivity is what makes the subtraction mean
   something, and it also lets the excess be reported as an angle rather than
   as a bare number, which is the only form in which it can be held against
   3.1's detection floor.

Window length is T = N throughout, which is the paper's rule, not a choice.
Lags below T are kept for the same reason as in 4.1: overlapping windows share
returns, so the curve must climb from zero and flatten near tau = T, and
reproducing that shape is a free correctness check on the pairing. Nothing is
*quoted* below tau = T, because the null assumes non-overlapping windows.
Introducing the intersection parameter t of arXiv:2509.25076 would make the
short lags quotable too, and is the obvious next thing to do to this file.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import to_panel, standardise, to_correlation_panel
from src.null_rmt import d_null_sample_vs_true
from src.overlap import spectral, sample_covariance, subspace_distance

# Regime 3.1 measured theta_min * sqrt(T) = 250, 249, 247 degrees across
# T = 250, 500, 1000 on Gaussian returns, and 270, 259, 259 for nu = 6 after
# standardising at window=1. Flat enough in T to extrapolate, but measured at
# N = 40, P = 3, Q = 6 ONLY. The floor depends on N through D_th, so these are
# indicative for a panel of a different size and must not be quoted as the
# panel's own floor. Re-run 3.1 at the panel's N and P, Q to do that.
FLOOR_SQRT_T_GAUSSIAN = 249.0
FLOOR_SQRT_T_STANDARDISED = 262.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="sp500")
    p.add_argument("--indir", type=Path, default=Path("data/cache"))
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--T", type=int, default=None,
                   help="window length. Defaults to N, which is the paper's rule.")
    p.add_argument("--P", type=int, default=3, help="inner block, top-P of window s")
    p.add_argument("--Q", type=int, default=6, help="outer block, top-Q of window t")
    p.add_argument("--step", type=int, default=None, help="days between window starts")
    p.add_argument("--max-lag", type=int, default=None,
                   help="largest lag to bin. Defaults to max(4T, 250) days.")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def window_spectra(panel, T, step, P, Q, standardised):
    """One eigendecomposition per window. Returns (starts, vecs, nulls, gaps).

    `vecs[i]` is the N x Q slice of leading eigenvectors -- top-P is its first P
    columns, so a single slice serves both roles and only one eigh is paid for.

    `nulls[i]` is Eq (7) evaluated on that window's own *estimated* spectrum.
    Eq (6.1) is additive across the two windows, so precomputing one number per
    window turns an O(pairs) sum over P x (N-Q) eigenvalue pairs into an O(1)
    addition per pair. Regime 2.2 measured the cost of feeding the null
    estimated rather than true eigenvalues: a systematic 1-5% on the ratio, and
    slightly *less* scatter than the true spectrum gives, because D_emp and
    D_th(est) come from the same returns and move together.

    `gaps[i]` is (lam_P, lam_{Q+1}) -- the two ends of the gap the null divides
    by. Regime 2.2 found a factor of 17 in the substitution bias driven by how
    far the sampled bulk edge climbs toward lam_P, so this is the exposure
    diagnostic, and unlike the bias itself it is visible without knowing any
    true spectrum.
    """
    starts, vecs, nulls, gaps = [], [], [], []
    for s in range(0, panel.shape[1] - T + 1, step):
        w = panel[:, s:s + T]
        if standardised:
            w = standardise(w, window=1)
        lam, vec = spectral(sample_covariance(to_correlation_panel(w)))
        starts.append(s)
        vecs.append(vec[:, :Q])
        nulls.append(d_null_sample_vs_true(lam, P, Q, T))
        gaps.append((lam[P - 1], lam[Q]))
    return np.array(starts), np.array(vecs), np.array(nulls), np.array(gaps)


def pair_distances(starts, vecs, nulls, P, max_lag):
    """D_emp and D_th for every window pair within max_lag. One batched SVD.

    Building every G first and calling svd once on the stack is not decoration:
    a python loop over 10^5 pairs each doing its own 6x3 decomposition is the
    slowest part of this script by an order of magnitude, and the stacked call
    is exactly the same arithmetic. `_check_batched` verifies that claim against
    src.overlap.subspace_distance on a random subset rather than assuming it.
    """
    i_idx, j_idx, lags = [], [], []
    for i in range(len(starts)):
        for j in range(i + 1, len(starts)):
            lag = starts[j] - starts[i]
            if lag > max_lag:
                break
            i_idx.append(i)
            j_idx.append(j)
            lags.append(lag)
    i_idx, j_idx, lags = map(np.asarray, (i_idx, j_idx, lags))
    if not len(lags):
        raise ValueError("no window pairs within max_lag -- panel too short for this T")

    # U from the earlier window (top P), V from the later (top Q): G = V^T U.
    U = vecs[i_idx][:, :, :P]                       # (n_pairs, N, P)
    V = vecs[j_idx]                                 # (n_pairs, N, Q)
    G = np.einsum("nkq,nkp->nqp", V, U)             # (n_pairs, Q, P)
    sv = np.linalg.svd(G, compute_uv=False)
    d_emp = -np.mean(np.log(np.clip(sv, 1e-300, 1.0)), axis=1)
    d_th = nulls[i_idx] + nulls[j_idx]
    return lags, d_emp, d_th, i_idx, j_idx


def _check_batched(vecs, i_idx, j_idx, d_emp, P, n=25, tol=1e-10):
    """Cross-check the batched path against the module function it replaces."""
    if not len(i_idx):
        return
    rng = np.random.default_rng(0)
    for k in rng.choice(len(i_idx), size=min(n, len(i_idx)), replace=False):
        ref = subspace_distance(vecs[i_idx[k]][:, :P], vecs[j_idx[k]])
        if abs(ref - d_emp[k]) > tol:
            raise AssertionError(
                f"batched D_emp disagrees with src.overlap.subspace_distance at "
                f"pair {k}: {d_emp[k]!r} vs {ref!r}")


def excess_to_angle(excess, P):
    """Invert D_inject = -ln(cos theta)/P for theta, in degrees.

    Legitimate only because regime 3.1 showed the two sources of distance add:
    D_emp = D_th + D_inject to within 0.1% once the injection clears the floor.
    Below the floor the subtraction can go negative, which is noise and not a
    rotation, so it returns NaN there rather than a number that would look like
    a measurement.
    """
    excess = np.asarray(excess, dtype=float)
    cos = np.exp(-P * excess)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).astype(float) * np.where(
        excess > 0, 1.0, np.nan)


def fit_slope(lags, values):
    """Least squares fit of value ~ a + b*lag. Returns (a, b, r).

    Deliberately no p-value. Window pairs at different lags share windows, and
    windows at a step below T share returns, so the residuals are heavily
    dependent and every closed-form standard error is wrong by an unknown and
    probably large factor. An honest interval needs a block bootstrap over
    non-overlapping segments, which is a separate piece of work. The slope is
    reported as a description of the curve; the claim it supports is
    qualitative -- flat, or growing.
    """
    lags = np.asarray(lags, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(lags[ok], values[ok], 1)
    r = float(np.corrcoef(lags[ok], values[ok])[0, 1])
    return float(a), float(b), r


def binned(lags, d_emp, d_th, P):
    """Average D_emp, D_th and excess per lag. One row per distinct lag."""
    rows = []
    for lag in np.unique(lags):
        m = lags == lag
        e, t = float(d_emp[m].mean()), float(d_th[m].mean())
        rows.append({"lag": int(lag), "n_pairs": int(m.sum()),
                     "d_emp": e, "d_th": t, "excess": e - t,
                     "ratio": e / t if t > 0 else np.nan,
                     "angle_deg": float(excess_to_angle(e - t, P))})
    return pd.DataFrame(rows).set_index("lag")


def run(panel, T, step, P, Q, max_lag, standardised):
    starts, vecs, nulls, gaps = window_spectra(panel, T, step, P, Q, standardised)
    lags, d_emp, d_th, i_idx, j_idx = pair_distances(starts, vecs, nulls, P, max_lag)
    _check_batched(vecs, i_idx, j_idx, d_emp, P)
    return binned(lags, d_emp, d_th, P), gaps, len(starts)


def plot(out, label, T, P, Q, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, tag in zip(axes, ("raw", "standardised")):
        v = out[tag]
        ax.plot(v.index / T, v["d_emp"], lw=1.6, label=r"$D_{emp}$")
        ax.plot(v.index / T, v["d_th"], lw=1.4, ls="--", label=r"$D_{th}$, Eq (6.1)")
        ax.axvline(1.0, color="0.6", lw=0.9, ls=":")
        ax.set_xlabel(r"$\tau / T$")
        ax.set_title(tag)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$D(P,Q;s,t)$")
    axes[0].legend(frameon=False)
    fig.suptitle(f"{label}: subspace variogram, T={T}, P={P}, Q={Q} "
                 f"(Fig. 8 of arXiv:1203.6228)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    a = parse_args(argv)
    rets = pd.read_parquet(a.indir / f"{a.label}_returns.parquet")
    panel = to_panel(rets)
    N, total = panel.shape
    T = a.T or N
    step = a.step or max(5, T // 25)
    max_lag = a.max_lag or max(4 * T, 250)
    P, Q = a.P, a.Q

    if not (0 < P <= Q < N):
        raise SystemExit(f"need 0 < P <= Q < N; got P={P}, Q={Q}, N={N}")

    print(f"{a.label}: N={N}, {total} days, T={T}, step={step}, P={P}, Q={Q}")
    print(f"  independent (non-overlapping) windows available: {total / T:.1f}")

    out = {}
    for tag, std in (("raw", False), ("standardised", True)):
        out[tag], gaps, n_win = run(panel, T, step, P, Q, max_lag, std)
        if tag == "raw":
            print(f"  windows: {n_win}, pairs binned to lag <= {max_lag}")
            lo, hi = gaps[:, 0].mean(), gaps[:, 1].mean()
            print(f"  bulk-edge exposure (regime 2.2): mean lam_{P} = {lo:.2f}, "
                  f"mean lam_{Q + 1} = {hi:.2f}, ratio = {lo / hi:.2f}")
            if lo / hi < 2.0:
                print("    ^ narrow. 2.2 found the substitution bias tracks this gap"
                      " -- treat D_th here as soft.")

    # The overlap artifact, exactly as in 4.1: the curve must climb from ~0 and
    # flatten near tau = T. If it does not, the pairing is wrong and everything
    # below it is meaningless.
    v = out["raw"]
    short = [l for l in v.index if l <= 2 * T]
    print(f"\n  windowing check -- D_emp/D_th must rise from ~0 and flatten at tau = T")
    print(f"  {'tau':>7} {'tau/T':>6} {'D_emp':>10} {'D_th':>10} {'ratio':>8}")
    for l in short[::max(1, len(short) // 8)]:
        r = v.loc[l]
        print(f"  {l:>7} {l / T:>6.2f} {r['d_emp']:>10.5f} {r['d_th']:>10.5f} "
              f"{r['ratio']:>7.2f}x")

    print(f"\n  {'':>14} {'tau':>7} {'D_emp':>10} {'D_th':>10} {'excess':>10} "
          f"{'ratio':>8} {'angle':>8}")
    for tag in ("raw", "standardised"):
        vv = out[tag]
        beyond = vv.loc[[l for l in vv.index if l >= T]]
        if not len(beyond):
            continue
        r = beyond.iloc[0]
        ang = f"{r['angle_deg']:>7.1f}" + chr(176) if np.isfinite(r["angle_deg"]) else "      --"
        print(f"  {tag:>12}: {beyond.index[0]:>7} {r['d_emp']:>10.5f} "
              f"{r['d_th']:>10.5f} {r['excess']:>10.5f} {r['ratio']:>7.2f}x {ang:>8}")

    floor_g = FLOOR_SQRT_T_GAUSSIAN / np.sqrt(T)
    floor_s = FLOOR_SQRT_T_STANDARDISED / np.sqrt(T)
    print(f"\n  indicative detection floor at T={T}: {floor_g:.1f}{chr(176)} Gaussian, "
          f"{floor_s:.1f}{chr(176)} standardised")
    print(f"    (regime 3.1's 1/sqrt(T) scaling, calibrated at N=40 P=3 Q=6 -- "
          f"re-run 3.1 at N={N} before quoting)")

    # Gate 3 / Gate 1: flat excess is a static mis-specified C (the BBP reading);
    # growing excess is genuine evolution, and linear growth is the precondition
    # for a diffusion model of the subspace being worth fitting.
    #
    # The discriminating number is `flat frac` -- the share of the fitted excess
    # at the longest lag that the INTERCEPT accounts for. A static mis-specified
    # C puts the whole excess in the intercept and none in the slope, so flat
    # frac -> 1. Genuine evolution starts the two windows agreeing and pushes
    # them apart, so the intercept is near zero and flat frac -> 0. Reporting
    # the level alone, which is what both papers do, cannot separate these.
    print(f"\n  excess vs tau, fitted over tau >= T   "
          f"(flat frac -> 1 static mis-specification, -> 0 evolution)")
    for tag in ("raw", "standardised"):
        vv = out[tag]
        far = vv.loc[[l for l in vv.index if l >= T]]
        if len(far) < 3:
            print(f"  {tag:>12}: too few lags past T ({len(far)}) -- widen --max-lag")
            continue
        a0, b, r = fit_slope(far.index.to_numpy(), far["excess"].to_numpy())
        tau_max = int(far.index[-1])
        pred = a0 + b * tau_max
        frac = a0 / pred if pred > 0 else np.nan
        verdict = ("evolution" if frac < 0.34 else
                   "mixed" if frac < 0.67 else "static offset")
        print(f"  {tag:>12}: intercept {a0:>+9.5f}   slope {b * 100:>+8.5f} /100d   "
              f"r = {r:>+.3f}   flat frac {frac:>+.2f}  -> {verdict}")

    a.outdir.mkdir(parents=True, exist_ok=True)
    merged = pd.concat({t: out[t] for t in out}, axis=1)
    merged["T"], merged["N"], merged["P"], merged["Q"] = T, N, P, Q
    path = a.outdir / f"{a.label}_subspace_T{T}_P{P}Q{Q}.csv"
    merged.to_csv(path)
    print(f"\n  -> {path}")

    if not a.no_plot:
        png = a.outdir / f"{a.label}_subspace_T{T}_P{P}Q{Q}.png"
        plot(out, a.label, T, P, Q, png)
        print(f"  -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
