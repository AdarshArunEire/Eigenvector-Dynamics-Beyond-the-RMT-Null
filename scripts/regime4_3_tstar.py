"""Regime 4.3: the ratio against window length. Fig. 9 of arXiv:1203.6228.

    python scripts/regime4_3_tstar.py --label nikkei

Regime 4.2 fixed $T = N$ and swept the gap $\\tau$. This sweeps $T$ itself, with
$\\tau$ tied to it -- every pair is two **back-to-back** windows, $\\tau = T$ --
and plots $D_{emp}/D_{th}$ against $T$. Different question: not "how fast does
the subspace drift apart" but "at what window length is the rotation clearest".

A maximum is plausible because two effects fight:

- **short T** -- each window is a noisy estimate, $D_{th} \\propto 1/T$ is large,
  and genuine rotation is diluted into the noise floor;
- **long T** -- the window averages over the rotation. A whole-window estimate
  recovers a time-average of C taken across a period in which C moved, which
  smears out the very thing being measured.

The paper reads its maximum at $T^* \\approx 500$ days and calls it a mean
reversion time. Two reasons to hold that loosely, both already in this project:

1. `stage1/paper_comparison.md` §4b argues a mean-reverting process cannot
   produce a maximum at all. For any stationary process, $D(\\tau)$ approaches
   the independent-draw value **from below** as $\\tau \\to \\infty$, so mean
   reversion gives a knee and a plateau. A true interior maximum requires the
   configurations at $T^*$ to be *more* different than two independent draws,
   i.e. anti-correlation -- an oscillation, which is a stronger claim than the
   paper argues for. So this script reports peak-versus-plateau explicitly
   rather than reporting an argmax and calling it $T^*$.
2. Regime 2.2 closes with: *"Still a threat to `T*`, because it is T-dependent
   and does not average away -- and `T*` is found by scanning T."* Feeding the
   null estimated rather than true eigenvalues carries a bias whose sign is set
   by how far the sampled bulk edge climbs toward $\\lambda_P$. This is the one
   regime where that open item bites directly, so `gap` (mean
   $\\hat\\lambda_P / \\hat\\lambda_{Q+1}$) is tracked at **every** $T$, not once.

Two hard limits on the sweep, both reported rather than silently handled:

- **$T \\geq N$.** Below that, $q = N/T > 1$ and the sample covariance is
  rank-deficient: at $N=357$, $T=200$ it has rank 200 and 157 exactly-zero
  eigenvalues, which the Eq (6.1) sum over $j > Q$ runs straight across. The
  S&P panel therefore has almost no room to sweep downward; the Nikkei has the
  most, which is presumably why the paper's Fig. 9 uses it.
- **Independent pairs.** `n_indep = total / (2T)` -- the number of genuinely
  non-overlapping back-to-back pairs. At the long-T end of the curve, exactly
  where the claimed peak sits, this falls to single digits. It is printed beside
  every point so a peak read off two observations is visible as such.
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="nikkei")
    p.add_argument("--indir", type=Path, default=Path("data/cache"))
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--P", type=int, default=3)
    p.add_argument("--Q", type=int, default=6)
    p.add_argument("--n-T", type=int, default=14, help="how many window lengths to try")
    p.add_argument("--T-min", type=int, default=None,
                   help="smallest window. Defaults to N (q = 1), the rank floor.")
    p.add_argument("--T-max", type=int, default=None,
                   help="largest window. Defaults to total/2, so one pair still fits.")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def back_to_back(panel, T, P, Q, standardised, stride=None):
    """Every pair of adjacent, non-overlapping windows of length T.

    Window A is [s, s+T), window B is [s+T, s+2T), so the lag is exactly T and
    the two share no returns at all. That is the configuration Eq (6.1) is
    derived for, and it is what ties tau to T for this sweep.

    Returns (mean D_emp, mean D_th, n_pairs, n_indep, mean gap).
    """
    N, total = panel.shape
    stride = stride or max(5, T // 10)
    d_emp, d_th, gaps = [], [], []
    for s in range(0, total - 2 * T + 1, stride):
        blocks = []
        for a in (s, s + T):
            w = panel[:, a:a + T]
            if standardised:
                w = standardise(w, window=1)
            lam, vec = spectral(sample_covariance(to_correlation_panel(w)))
            blocks.append((lam, vec))
        (lam_s, vec_s), (lam_t, vec_t) = blocks
        d_emp.append(subspace_distance(vec_s[:, :P], vec_t[:, :Q]))
        d_th.append(d_null_sample_vs_true(lam_s, P, Q, T)
                    + d_null_sample_vs_true(lam_t, P, Q, T))
        gaps.append(0.5 * (lam_s[P - 1] / lam_s[Q] + lam_t[P - 1] / lam_t[Q]))
    if not d_emp:
        return None
    return (float(np.mean(d_emp)), float(np.mean(d_th)), len(d_emp),
            total / (2.0 * T), float(np.mean(gaps)))


def sweep(panel, Ts, P, Q, standardised):
    rows = []
    for T in Ts:
        got = back_to_back(panel, int(T), P, Q, standardised)
        if got is None:
            continue
        e, t, n, n_ind, gap = got
        rows.append({"T": int(T), "q": panel.shape[0] / T, "d_emp": e, "d_th": t,
                     "ratio": e / t, "excess": e - t, "n_pairs": n,
                     "n_indep": n_ind, "gap": gap,
                     "angle_deg": float(np.degrees(np.arccos(
                         np.clip(np.exp(-P * (e - t)), -1.0, 1.0)))) if e > t else np.nan})
    return pd.DataFrame(rows).set_index("T")


def describe(v):
    """Peak, or knee? Returns a one-line verdict string.

    A maximum only counts as one if the curve comes back down by more than the
    scatter you would expect from how few independent pairs sit at the far end.
    Otherwise it is a plateau with a noisy tail, which is what mean reversion
    actually predicts -- see the module docstring.
    """
    r = v["ratio"]
    peak_T, peak = int(r.idxmax()), float(r.max())
    last_T, last = int(r.index[-1]), float(r.iloc[-1])
    drop = (peak - last) / peak
    if peak_T == last_T:
        return (f"monotone to the end of the sweep -- no interior maximum "
                f"(max {peak:.2f}x at T={peak_T}, the largest T tried)")

    # Two guards, because the far end of this sweep is where a maximum is
    # cheapest to manufacture and most tempting to believe. A turnover has to
    # be built from more than the last couple of points, and the peak itself
    # has to sit somewhere with enough independent pairs to mean anything.
    tail = r.loc[r.index > peak_T]
    n_indep_at_peak = float(v.loc[peak_T, "n_indep"])
    if len(tail) < 3:
        return (f"apparent turnover at T={peak_T} rests on only {len(tail)} point(s) "
                f"beyond it -- NOT a maximum, the sweep just ended")
    if n_indep_at_peak < 4:
        return (f"apparent peak {peak:.2f}x at T={peak_T} sits on "
                f"{n_indep_at_peak:.1f} independent pairs -- below the floor "
                f"where a turnover can be distinguished from scatter")

    noise = float(tail.std()) / peak
    kind = "interior MAXIMUM" if drop > 0.15 and drop > 2 * noise else "knee / plateau"
    return (f"{kind}: peak {peak:.2f}x at T={peak_T} ({n_indep_at_peak:.1f} indep "
            f"pairs), falls {drop:.0%} to {last:.2f}x at T={last_T}; "
            f"tail scatter {noise:.0%} of peak")


def plot(out, label, N, P, Q, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    # Laid out as their Fig. 9: D_emp and D_th on the same axis against T, so the
    # minimum in D_emp is visible directly. The ratio is in the CSV but is a poor
    # thing to plot -- D_th falls as 1/T while D_emp saturates, so the ratio climbs
    # for reasons that have nothing to do with the market.
    v = out["raw"]
    ax.plot(v.index, v["d_emp"], "-", color="crimson", marker="o", ms=3.5,
            label=r"$D_{emp}(\tau=T)$, raw")
    ax.plot(v.index, v["d_th"], ":", color="green", label=r"$D_{th}(\tau=T)$, Gaussian")
    vs = out["standardised"]
    ax.plot(vs.index, vs["d_emp"], "--", color="crimson", alpha=0.55, marker="o",
            ms=3, label=r"$D_{emp}$, standardised")
    ax.plot(vs.index, vs["d_th"], ":", color="royalblue", alpha=0.8,
            label=r"$D_{th}$, standardised")
    tstar = int(v["d_emp"].idxmin())
    ax.axvline(tstar, color="0.55", lw=0.9, ls="-.")
    ax.annotate(f"$T^*$ = {tstar}", (tstar, v["d_emp"].min()), fontsize=9,
                xytext=(6, 14), textcoords="offset points", color="0.35")
    ax.axvline(N, color="0.8", lw=0.9)
    ax.set_ylabel(r"$D(\tau = T)$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title(f"{label}: $D$ vs window length, P={P}, Q={Q} "
                 f"(Fig. 9 of arXiv:1203.6228)")

    ax2.semilogy(out["raw"].index, out["raw"]["n_indep"], color="0.4", marker="o", ms=3)
    ax2.axhline(4, color="crimson", lw=0.9, ls=":")
    ax2.set_ylabel("indep.\npairs")
    ax2.set_xlabel("window length $T$ (trading days)")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    a = parse_args(argv)
    rets = pd.read_parquet(a.indir / f"{a.label}_returns.parquet")
    panel = to_panel(rets)
    N, total = panel.shape
    T_min = a.T_min or N
    T_max = a.T_max or total // 2
    if T_min >= T_max:
        raise SystemExit(f"{a.label}: no room to sweep -- N={N}, {total} days")
    Ts = np.unique(np.linspace(T_min, T_max, a.n_T).astype(int))

    print(f"{a.label}: N={N}, {total} days, P={a.P}, Q={a.Q}")
    print(f"  sweeping T = {Ts[0]} .. {Ts[-1]} ({len(Ts)} points), "
          f"tau = T (back-to-back windows)")
    print(f"  rank floor T >= N = {N} enforced (q <= 1)\n")

    out = {}
    for tag, std in (("raw", False), ("standardised", True)):
        out[tag] = sweep(panel, Ts, a.P, a.Q, std)

    v = out["raw"]
    print(f"  {'T':>6} {'q':>6} {'D_emp':>9} {'D_th':>9} {'ratio':>8} {'angle':>7} "
          f"{'gap':>6} {'pairs':>6} {'indep':>7}")
    for T, r in v.iterrows():
        ang = f"{r['angle_deg']:>6.1f}" if np.isfinite(r["angle_deg"]) else "     --"
        flag = "  <-- thin" if r["n_indep"] < 4 else ""
        print(f"  {T:>6} {r['q']:>6.2f} {r['d_emp']:>9.5f} {r['d_th']:>9.5f} "
              f"{r['ratio']:>7.2f}x {ang:>7} {r['gap']:>6.2f} {int(r['n_pairs']):>6} "
              f"{r['n_indep']:>7.1f}{flag}")

    print()
    for tag in ("raw", "standardised"):
        print(f"  {tag:>12}: {describe(out[tag])}")

    g = v["gap"]
    print(f"\n  regime 2.2 exposure: bulk gap lam_P/lam_Q+1 runs "
          f"{g.min():.2f} (T={int(g.idxmin())}) to {g.max():.2f} (T={int(g.idxmax())}). "
          f"\n    The null's bias tracks this, so it varies ALONG the sweep axis -- "
          f"a {g.max() / g.min():.1f}x swing in the\n    quantity 2.2 found drives a "
          f"factor of 17. Any T* read off this curve inherits that.")

    a.outdir.mkdir(parents=True, exist_ok=True)
    merged = pd.concat({t: out[t] for t in out}, axis=1)
    merged["N"] = N
    path = a.outdir / f"{a.label}_tstar_P{a.P}Q{a.Q}.csv"
    merged.to_csv(path)
    print(f"\n  -> {path}")
    if not a.no_plot:
        png = a.outdir / f"{a.label}_tstar_P{a.P}Q{a.Q}.png"
        plot(out, a.label, N, a.P, a.Q, png)
        print(f"  -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
