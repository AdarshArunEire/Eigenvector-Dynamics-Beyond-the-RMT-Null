"""Regime 4: D_emp against D_th on real returns, as a function of lag.

    python scripts/regime4.py --label us --T 204 --P 5 --Q 10

Slides a window of length T along the series, takes the eigenbasis of each,
then for every pair separated by lag tau measures the subspace distance and
compares it to Eq (10) evaluated on that pair's own estimated spectra.

Three curves come out, and the gap between them is the point:

  raw            what the instrument reports
  standardised   the same after dividing each day by its cross-sectional
                 volatility, which removes the within-window drift inflation
                 that regime 2.3 measured at 1 + CV^2
  D_RMT          the ceiling: two subspaces with no relationship left

If the peak survives standardisation it is a claim about correlation
structure. If it does not, it was volatility.

Windows overlap, so pairs at a given lag are not independent and the scatter
reported here understates the true uncertainty. Stated rather than corrected.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import to_panel, standardise, cv_squared, to_correlation_panel
from src.null_rmt import d_null_two_samples, d_random_subspaces
from src.overlap import spectral, sample_covariance, subspace_distance


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="us", help="prefix used by fetch_data.py")
    p.add_argument("--indir", type=Path, default=Path("data/cache"))
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--T", type=int, default=204, help="window length in days")
    p.add_argument("--P", type=int, default=5, help="inner block, tracked")
    p.add_argument("--Q", type=int, default=10, help="outer block, searched")
    p.add_argument("--step", type=int, default=20, help="days between window starts")
    p.add_argument("--matrix", choices=["correlation", "covariance"],
                   default="correlation",
                   help="which matrix to eigen-decompose, PER WINDOW. The paper "
                        "works on the correlation matrix ('the top P eigenvectors "
                        "of the true correlation matrix C'), and its Fig. 1 "
                        "variogram axes (0-600 for mode 1) only make sense for "
                        "one: on raw daily covariance 4*lam1^2/T is O(1e-6). "
                        "Running on covariance lets the cross-sectional spread "
                        "of individual volatilities into the top eigenvectors, "
                        "which roughly doubles D_emp and triples D_emp/D_th.")
    p.add_argument("--tol", type=int, default=None, help="lag bin half-width (default step)")
    p.add_argument("--max-names", type=int, default=None,
                   help="subsample to this many names. The lever for q = N/T: "
                        "shrinking N lowers q without costing lag coverage, "
                        "which raising T does.")
    p.add_argument("--seed", type=int, default=0, help="seed for name subsampling")
    p.add_argument("--min-lag", type=int, default=None,
                   help="shortest lag to include. Defaults to T, so windows "
                        "never share data. Going below T lets overlapping "
                        "windows in, which biases D_emp DOWN because shared "
                        "days force the two estimates to agree. The clean way "
                        "to reach short lags is a smaller T.")
    return p.parse_args(argv)


def window_cv2(panel, T, step):
    """CV^2 measured *within* each window, which is what regime 2.3 depends on.

    Series-wide CV^2 is a different and much larger number -- eleven years of
    volatility varies far more than any one window does -- so quoting it
    overstates the inflation for short T and hides that the effect grows with
    window length.
    """
    from src.data import cv_squared
    vals = [cv_squared(panel[:, s:s + T])
            for s in range(0, panel.shape[1] - T + 1, step)]
    return float(np.mean(vals)), float(np.max(vals))


def window_bases(panel, T, step, P, Q, matrix="correlation"):
    """Eigen-decompose once per window; pairing afterwards is cheap.

    `matrix` decides whether each window is scaled to unit per-name variance
    first. This is the single largest lever on the absolute level of D_emp, and
    it is a choice about what question is being asked:

      covariance  -- the top eigenvectors are tilted towards whichever names
                     happened to be most volatile in that window. Volatility
                     rank turnover then reads as eigenvector rotation even when
                     the correlation structure is frozen.
      correlation -- volatility is divided out per name per window, so only the
                     co-movement pattern is left. This is the paper's object.

    Note this is a *different axis* from `standardise`, which divides each day
    by the cross-sectional volatility of that day. Both are usually wanted.
    """
    N, total = panel.shape
    starts = list(range(0, total - T + 1, step))
    out = []
    for s in starts:
        w = panel[:, s:s + T]
        if matrix == "correlation":
            w = to_correlation_panel(w)
        evals, evecs = spectral(sample_covariance(w))
        out.append((s, evals, evecs[:, :max(P, Q)]))
    return out


def curve(bases, T, P, Q, tol, min_lag=None, total=None):
    """Average D_emp and D_th over window pairs, binned by lag.

    Also splits each bin by *when* the pair sits in the sample. The lag axis is
    structurally confounded with period: at long lag the only available pairs
    are start-of-sample against end-of-sample, so a rising curve can mean
    "eigenvectors rotate with elapsed time" or merely "those two eras differ".
    Comparing early-centred against late-centred pairs at the *same* lag
    separates them.
    """
    min_lag = T if min_lag is None else min_lag
    rows = {}
    for i, (s_i, lam_i, v_i) in enumerate(bases):
        for j in range(i + 1, len(bases)):
            s_j, lam_j, v_j = bases[j]
            lag = s_j - s_i
            if lag < min_lag:
                continue
            key = int(round(lag / tol) * tol)
            d_emp = subspace_distance(v_i[:, :P], v_j[:, :Q])
            d_th = d_null_two_samples(lam_i, lam_j, P, Q, T)
            rows.setdefault(key, []).append((d_emp, d_th, 0.5 * (s_i + s_j)))
    recs = []
    mid_cut = 0.5 * (total - T) if total else None
    for lag in sorted(rows):
        a = np.array(rows[lag])
        rec = {"tau": lag, "n_pairs": len(a),
               "D_emp": a[:, 0].mean(), "D_th": a[:, 1].mean(),
               "ratio": a[:, 0].mean() / a[:, 1].mean(),
               "ratio_sd": (a[:, 0] / a[:, 1]).std(ddof=1) if len(a) > 1 else np.nan}
        if mid_cut is not None:
            early, late = a[a[:, 2] < mid_cut], a[a[:, 2] >= mid_cut]
            rec["D_emp_early"] = early[:, 0].mean() if len(early) else np.nan
            rec["D_emp_late"] = late[:, 0].mean() if len(late) else np.nan
            rec["n_early"], rec["n_late"] = len(early), len(late)
        recs.append(rec)
    return pd.DataFrame(recs)


def main(argv=None):
    a = parse_args(argv)
    tol = a.tol or a.step
    rets = pd.read_parquet(a.indir / f"{a.label}_returns.parquet")
    panel = to_panel(rets)
    if a.max_names and a.max_names < panel.shape[0]:
        pick = np.random.default_rng(a.seed).choice(panel.shape[0], a.max_names, False)
        panel = panel[np.sort(pick)]
    N, total = panel.shape
    if a.Q >= N:
        raise SystemExit(f"Q={a.Q} must be < N={N}")
    q = N / a.T
    print(f"{a.label}: N={N}, {total} days, T={a.T}, P={a.P}, Q={a.Q}, step={a.step}, "
          f"matrix={a.matrix}")
    print(f"q = N/T = {q:.3f}" + ("   <- outside the range regime 2.2 calibrated (<=0.16)"
                                  if q > 0.2 else "   (inside calibrated range)"))
    cv_mean, cv_max = window_cv2(panel, a.T, a.step)
    print(f"CV^2 within a {a.T}-day window: mean {cv_mean:.3f}, worst {cv_max:.3f}"
          f"  ->  regime 2.3 inflation ~ {1 + cv_mean:.2f}x typical, {1 + cv_max:.2f}x worst")
    print(f"CV^2 over the whole {total}-day series = {cv_squared(panel):.3f} "
          f"(NOT the relevant number; shown only to contrast)\n")

    if a.min_lag is not None and a.min_lag < a.T:
        print(f"  ! min-lag {a.min_lag} < T={a.T}: overlapping windows share days,"
              f"\n    which forces their estimates to agree and biases D_emp DOWN"
              f"\n    at short lags. A smaller T is the clean route.\n")

    a.outdir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for tag, p in (("raw", panel), ("standardised", standardise(panel))):
        bases = window_bases(p, a.T, a.step, a.P, a.Q, a.matrix)
        df = curve(bases, a.T, a.P, a.Q, tol, a.min_lag, total)
        frames[tag] = df
        print(f"  {tag}: {len(bases)} windows, {int(df['n_pairs'].sum())} pairs, "
              f"{len(df)} lag bins")

    d_rmt = d_random_subspaces(a.P, a.Q, N, "normalised")
    print(f"\n  D_RMT (ceiling, no relationship left) = {d_rmt:.4f}\n")
    print(f"  {'tau':>6} {'n':>5} {'D_emp':>9} {'D_th':>9} {'ratio raw':>10} "
          f"{'ratio std':>10} {'% of D_RMT':>11}")
    r, s = frames["raw"], frames["standardised"].set_index("tau")
    for _, row in r.iterrows():
        st = s.loc[row["tau"], "ratio"] if row["tau"] in s.index else np.nan
        print(f"  {int(row['tau']):>6} {int(row['n_pairs']):>5} {row['D_emp']:>9.4f} "
              f"{row['D_th']:>9.4f} {row['ratio']:>10.3f} {st:>10.3f} "
              f"{100 * row['D_emp'] / d_rmt:>10.1f}%")

    out = a.outdir / f"{a.label}_T{a.T}_P{a.P}Q{a.Q}_{a.matrix[:4]}.csv"
    merged = r.merge(frames["standardised"], on="tau", suffixes=("_raw", "_std"))
    merged["d_rmt"] = d_rmt
    merged.to_csv(out, index=False)
    print(f"\n  -> {out}")

    peak_raw = r.loc[r["ratio"].idxmax()]
    sr = frames["standardised"]
    peak_std = sr.loc[sr["ratio"].idxmax()]
    print(f"\n  peak ratio raw          {peak_raw['ratio']:.3f} at tau = {int(peak_raw['tau'])}")
    print(f"  peak ratio standardised {peak_std['ratio']:.3f} at tau = {int(peak_std['tau'])}")
    if peak_std["ratio"] < 1.1:
        print("  -> the excess does not survive standardisation. That points at"
              "\n     volatility drift rather than eigenvector rotation.")

    hi = r["D_emp"].max() / d_rmt
    if hi > 0.9:
        print(f"  ! D_emp reaches {100 * hi:.0f}% of D_RMT. Above ~90% the measure is"
              f"\n    against its ceiling and cannot tell heavily-rotated from unrelated."
              f"\n    Long-lag numbers in this configuration are compressed.")

    # Is the lag trend actually a lag trend?
    both = r.dropna(subset=["D_emp_early", "D_emp_late"])
    both = both[(both["n_early"] >= 5) & (both["n_late"] >= 5)]
    print("\n  same lag, different era -- is tau doing the work, or the period?\n")
    if both.empty:
        print("    too few pairs in both halves at any lag to compare.")
    else:
        print(f"    {'tau':>6} {'n_early':>8} {'D_emp early':>12} "
              f"{'n_late':>7} {'D_emp late':>11} {'late/early':>11}")
        for _, row in both.iterrows():
            print(f"    {int(row['tau']):>6} {int(row['n_early']):>8} "
                  f"{row['D_emp_early']:>12.4f} {int(row['n_late']):>7} "
                  f"{row['D_emp_late']:>11.4f} "
                  f"{row['D_emp_late'] / row['D_emp_early']:>11.2f}")
        ratio_le = both["D_emp_late"] / both["D_emp_early"]
        span = r["D_emp"].max() / r["D_emp"].min()
        # NOT the mean: this quantity oscillates about 1, so averaging cancels a
        # large effect into a small number. Size is |deviation from 1|.
        typ, worst = (ratio_le - 1).abs().mean(), (ratio_le - 1).abs().max()
        arg = (ratio_le - 1).abs().idxmax()
        print(f"\n    era effect at matched lag   typical {typ:+.0%}, worst {worst:+.0%} "
              f"(at tau = {int(both.loc[arg, 'tau'])}, "
              f"{max(ratio_le[arg], 1 / ratio_le[arg]):.1f}x apart)")
        print(f"    range of late/early         {ratio_le.min():.2f} to {ratio_le.max():.2f}")
        print(f"    lag effect across the range {span:.2f}x")
        if worst > 0.4:
            peak_tau = int(r.loc[r['D_emp'].idxmax(), 'tau'])
            near = both.iloc[(both["tau"] - peak_tau).abs().argsort()[:1]]
            print(f"\n    -> the two eras disagree by {worst:.0%} at the same lag, so the tau"
                  f"\n       axis is not measuring elapsed time alone. At the D_emp peak"
                  f"\n       (tau = {peak_tau}) the halves read "
                  f"{near['D_emp_early'].iloc[0]:.4f} early vs "
                  f"{near['D_emp_late'].iloc[0]:.4f} late -- the peak is largely one era,"
                  f"\n       not an accumulation. Check which windows those pairs are.")
        else:
            print("\n    -> the era effect stays small at every lag, so tau is carrying"
                  "\n       the trend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
