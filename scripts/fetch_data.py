"""Download, clean and diagnose a return panel. Run this before any analysis.

    python scripts/fetch_data.py --universe us --start 2000-01-01 --end 2010-12-31
    python scripts/fetch_data.py --tickers nikkei225.txt --label nikkei --start 2000-01-01 --end 2010-12-31
    python scripts/fetch_data.py --universe cac40 --label cac40_full --end 2026-08-03 \
        --exclude ATO.PA --exclude VIV.PA --exclude AC.PA

Writes into data/cache/ (already gitignored):

    <label>_prices.parquet    raw adjusted closes, the download cache
    <label>_returns.parquet   cleaned log returns, dates x tickers
    <label>_drops.csv         every name considered, and why it was kept or not

Re-running reads the price cache instead of the wire, so the run that produces
a published number is reproducible. Delete the parquet to force a refetch.

The printed report is the point of this script, not a side effect. It is the
last chance to see the panel's defects as numbers before they become part of a
result -- and it is worth reading *before* plotting D_emp/D_th against tau,
because once that curve is on screen every later choice is made by someone who
knows what they are hoping for.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (US_LARGE_CAP_CANDIDATES, fetch_prices, log_returns,
                      filter_by_coverage, to_panel, panel_report,
                      to_correlation_panel, worst_outliers,
                      find_price_breaks, drop_price_breaks,
                      unreliable_names, DEFECT_KINDS)
from src.null_rmt import q_from_mp_edge
from src.overlap import spectral, sample_covariance


# The four pools of arXiv:1203.6228 section 6, plus the original hand list.
#
# `target` is the N the paper reports, for reference only -- it is not enforced
# and I will not hit it. Their SP500 pool cannot be index membership at all:
# roughly 200 constituents changed between 2000 and 2010, so no 500 names have
# both membership and full history. Whatever their pool was, it is not stated.
#
# Filter presets differ by market because the defaults were calibrated on US
# large caps. Tokyo names carry a median 9.2% exactly-zero returns against 1.1%
# for the US panel, so a 10% cap there rejects normal Japanese trading.
UNIVERSES = {
    "us":     dict(file=None,               target=None, cov=0.98, stale=10, zero=0.10),
    "sp500":  dict(file="data/sp500.txt",   target=500,  cov=0.98, stale=10, zero=0.10),
    "nikkei": dict(file="data/nikkei225.txt", target=204, cov=0.98, stale=20, zero=0.20),
    "dax":    dict(file="data/dax.txt",     target=30,   cov=0.98, stale=10, zero=0.10),
    "cac40":  dict(file="data/cac40.txt",   target=39,   cov=0.98, stale=10, zero=0.10),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--universe", choices=sorted(UNIVERSES),
                     help="one of the four pools the paper uses, or 'us' for the "
                          "built-in hand list. Each carries filter defaults suited "
                          "to that market; any --min-coverage/--max-* flag overrides them.")
    src.add_argument("--tickers", type=Path,
                     help="file with one ticker per line. Tokyo codes look "
                          "like 7203.T, Frankfurt SAP.DE, Paris AI.PA")
    p.add_argument("--label", default=None, help="output prefix (default: universe name)")
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default="2010-12-31")
    p.add_argument("--outdir", type=Path, default=Path("data/cache"))
    p.add_argument("--min-coverage", type=float, default=None)
    p.add_argument("--max-stale-run", type=int, default=None)
    p.add_argument("--max-zero-fraction", type=float, default=None)
    p.add_argument("--break-threshold", type=float, default=0.4,
                   help="log-return size above which a move is treated as a "
                        "corporate-action break rather than a market move. "
                        "0.4 is about 50%%; no liquid large cap moves that far "
                        "in a day without an action behind it.")
    p.add_argument("--max-defects", type=int, default=4,
                   help="a name with more flagged moves than this has a broken "
                        "vendor series rather than a few bad days, and is dropped "
                        "whole. LIN.DE threw 290 on the DAX panel.")
    p.add_argument("--batch-size", type=int, default=40,
                   help="tickers per download request. Yahoo rate-limits large "
                        "requests by returning empty data rather than an error, "
                        "so a 500-name pull must be broken up. Lower this if "
                        "batches keep coming back empty.")
    p.add_argument("--pause", type=float, default=2.0,
                   help="seconds between batches")
    p.add_argument("--refetch", action="store_true", help="ignore the price cache")
    p.add_argument("--exclude", action="append", default=[], metavar="TICKER",
                   help="manually exclude a known-bad corporate-action series; repeatable. "
                        "The exclusion is recorded in the drops report.")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    preset = dict(file=None, target=None, cov=0.98, stale=10, zero=0.10)
    if a.tickers:
        tickers = [t.strip() for t in a.tickers.read_text().splitlines() if t.strip()]
        label = a.label or a.tickers.stem
    else:
        preset = UNIVERSES[a.universe]
        label = a.label or a.universe
        if preset["file"] is None:
            tickers = list(US_LARGE_CAP_CANDIDATES)
        else:
            path = Path(preset["file"])
            if not path.exists():
                raise SystemExit(f"{path} not found -- expected the ticker list for "
                                 f"--universe {a.universe}")
            tickers = [t.strip() for t in path.read_text().splitlines() if t.strip()]
    requested_exclusions = list(dict.fromkeys(a.exclude))
    unknown_exclusions = [t for t in requested_exclusions if t not in tickers]
    if unknown_exclusions:
        raise SystemExit(f"--exclude ticker(s) not in the requested universe: "
                         f"{', '.join(unknown_exclusions)}")
    tickers = [t for t in tickers if t not in requested_exclusions]

    # explicit flags win; otherwise take the market's preset
    a.min_coverage = preset["cov"] if a.min_coverage is None else a.min_coverage
    a.max_stale_run = preset["stale"] if a.max_stale_run is None else a.max_stale_run
    a.max_zero_fraction = preset["zero"] if a.max_zero_fraction is None else a.max_zero_fraction
    a.outdir.mkdir(parents=True, exist_ok=True)
    px_path = a.outdir / f"{label}_prices.parquet"
    if a.refetch and px_path.exists():
        px_path.unlink()

    print(f"[1/4] {len(tickers)} candidates, {a.start} to {a.end}"
          + (f"   (paper reports N = {preset['target']})" if preset["target"] else "")
          + f"\n      filters: coverage>={a.min_coverage}, stale<={a.max_stale_run}, "
            f"zero<={a.max_zero_fraction}")
    if requested_exclusions:
        print(f"      manual corporate-action exclusions: "
              f"{', '.join(requested_exclusions)}")
    px = fetch_prices(tickers, a.start, a.end, cache=px_path,
                      batch_size=a.batch_size, pause=a.pause)
    px = px.dropna(axis=1, how="all")
    print(f"      {px.shape[1]} returned data, {len(px)} dates -> {px_path}")
    if px.shape[1] < 5:
        raise SystemExit(
            f"only {px.shape[1]} of {len(tickers)} tickers returned data. That is "
            f"a failed download, not a thin universe.\n"
            f"Yahoo answers a rate-limited request with empty data rather than an "
            f"error, so wait a few minutes and re-run -- whatever did arrive has "
            f"been cached and will not be refetched.\n"
            f"If it persists, lower --batch-size (currently {a.batch_size}) or "
            f"raise --pause (currently {a.pause}s).")

    # A non-positive adjusted close is not a market fact, it is a broken
    # adjustment factor: the vendor has back-applied a cumulative dividend or
    # consolidation adjustment that oversubtracts and drives the old history
    # through zero. `log_returns` refuses to take logs of it, correctly. But
    # one bad name should not take down a 200-name pull, so the name is dropped
    # here, loudly, and the reason is carried into the survivorship report
    # rather than being fixed up silently.
    nonpos = (px <= 0).sum()
    bad = nonpos[nonpos > 0]
    if len(bad):
        print(f"      ! {len(bad)} name(s) have non-positive adjusted prices "
              f"-- broken vendor adjustment, dropped:")
        for t, n in bad.items():
            span = px.index[(px[t] <= 0).fillna(False)]
            print(f"          {t}: {n} days, {span.min().date()} to {span.max().date()}")
        px = px.drop(columns=list(bad.index))
        print(f"        {px.shape[1]} names remain")

    print("[2/4] log returns")
    rets = log_returns(px.ffill(limit=3))

    breaks = find_price_breaks(px, threshold=a.break_threshold)
    junk = unreliable_names(breaks, max_defects=a.max_defects)
    if junk:
        print(f"      ! {len(junk)} name(s) flagged too often to be patched -- the "
              f"series is broken, not the days. Dropped whole:")
        for t_, (nd, nm) in junk.items():
            print(f"          {t_}: {nd} defect(s), {nm} large move(s)")
        px = px.drop(columns=[c for c in junk if c in px.columns])
        rets = rets.drop(columns=[c for c in junk if c in rets.columns])
        breaks = breaks[~breaks["ticker"].isin(junk)]
        print(f"        {px.shape[1]} names remain")

    if len(breaks):
        counts = breaks["kind"].value_counts()
        defects = breaks[breaks["kind"].isin(DEFECT_KINDS)]
        moves = breaks[breaks["kind"] == "large_move"]
        print(f"      {len(breaks)} move(s) past {a.break_threshold} in log terms: "
              + ", ".join(f"{n} {k}" for k, n in counts.items()))
        if len(defects):
            print(f"      ! {len(defects)} are data defects; the return on each is dropped:")
            for _, br in defects.iterrows():
                note = (f"exact {br['split_ratio']} ratio -- corporate action"
                        if br["kind"] == "split" else
                        "single bad close, corrected next day")
                print(f"          {br['ticker']} {br['date'].date()} "
                      f"{br['log_return']:+.3f}  [{br['kind']}] {note}")
            rets = drop_price_breaks(rets, breaks)
        if len(moves):
            worst = moves.reindex(moves["log_return"].abs().sort_values(
                ascending=False).index).head(5)
            print(f"      {len(moves)} are large but are NOT split ratios, so they are "
                  f"market moves and are KEPT. Largest:")
            for _, br in worst.iterrows():
                print(f"          {br['ticker']} {br['date'].date()} "
                      f"{br['log_return']:+.3f}  ({100 * (np.exp(br['log_return']) - 1):+.0f}%)")
            print(f"        These are the days the correlation structure actually "
                  f"moves.\n        Deleting them would remove the signal being measured.")
        breaks.to_csv(a.outdir / f"{label}_breaks.csv", index=False)
        print(f"        -> {a.outdir / f'{label}_breaks.csv'}")

    print("[3/4] filtering")
    kept, report = filter_by_coverage(rets, a.min_coverage, a.max_stale_run,
                                      a.max_zero_fraction)
    for ticker in requested_exclusions:
        report.loc[ticker] = {
            "coverage": np.nan,
            "zero_fraction": np.nan,
            "longest_stale_run": np.nan,
            "dropped_because": "manual exclusion: unresolved corporate action",
        }
    dropped = report[report["dropped_because"].notna()]
    print(f"      kept {kept.shape[1]}, dropped {len(dropped)}")
    for reason in ("coverage", "stale run", "zero returns"):
        n = dropped["dropped_because"].str.contains(reason).sum()
        if n:
            print(f"        {n:>3} on {reason}")
    print(f"      survivorship report -> {a.outdir / f'{label}_drops.csv'}")
    report.to_csv(a.outdir / f"{label}_drops.csv")
    kept.to_parquet(a.outdir / f"{label}_returns.parquet")

    print("[4/4] diagnostics\n")
    panel = to_panel(kept)
    rep = panel_report(panel)
    N, T = rep["N"], rep["T"]
    print(f"  N = {N} names, T = {T} days, q = N/T = {rep['q']:.4f}")
    print(f"  median annualised vol   {rep['annualised_vol_median']:.1%}")
    print(f"  zero-return fraction    {rep['zero_return_fraction']:.3%}")
    print(f"  median kurtosis         {rep['kurtosis_median']:.2f}   (3.0 = Gaussian)")
    print(f"  max kurtosis            {rep['kurtosis_max']:.2f}")
    print(f"  CV^2 of variance path   {rep['cv_squared']:.4f}")

    # MP describes a matrix of common-variance variables, so the edge is only
    # meaningful on the correlation matrix. On a raw covariance the volatility
    # spread across names widens the bulk and the criterion collapses.
    evals, _ = spectral(sample_covariance(to_correlation_panel(panel)))
    Q, edge, s2 = q_from_mp_edge(evals, T)
    print(f"  MP edge (correlation)   {edge:.4f} (sigma^2 = {s2:.4f})  ->  Q = {Q}")
    print(f"  top 8 eigenvalues       {np.round(evals[:8], 3)}  (trace/N = {evals.mean():.3f})")

    outliers = worst_outliers(panel, list(kept.columns), k=5)
    print(f"  largest moves           " + ", ".join(
        f"{o['name']} {o['sd']:.0f}sd ({o['return']:+.0%})" for o in outliers))

    print("\n  read before analysing:")
    if rep["excess_kurtosis_median"] > 1.0:
        print(f"    - kurtosis {rep['kurtosis_median']:.1f} vs 3.0 Gaussian. Eq (7) comes from"
              f"\n      the Wishart 4th moment, so D_th is derived under normality and"
              f"\n      will run low here -- inflating apparent excess. Untested.")
    if rep["cv_squared"] > 0.05:
        print(f"    - CV^2 = {rep['cv_squared']:.3f} implies regime 2.3 inflates D by about"
              f"\n      {1 + rep['cv_squared']:.2f}x with no rotation present. Report the"
              f"\n      standardised curve alongside the raw one.")
    if rep["q"] > 0.3:
        print(f"    - q = {rep['q']:.2f} is outside the range regime 2.2 was calibrated over"
              f"\n      (<= 0.16). The substitution bias there reached -11.5% by q = 0.89"
              f"\n      and its sign depends on bulk density. Measure it in situ.")
    if rep["zero_return_fraction"] > 0.01:
        print(f"    - {rep['zero_return_fraction']:.1%} of returns are exactly zero. Stale prices"
              f"\n      pull the covariance down and manufacture structure.")
    if outliers and outliers[0]["sd"] > 20:
        bad = [o for o in outliers if o["sd"] > 20]
        print(f"    - {len(bad)} move(s) beyond 20 sd, largest {outliers[0]['name']} at"
              f" {outliers[0]['sd']:.0f} sd ({outliers[0]['return']:+.0%})."
              f"\n      That is not a fat tail. Check for an unadjusted split or a bad"
              f"\n      print before trusting any covariance this name appears in.")
    if Q < 2:
        print(f"    - MP finds Q = {Q}: no usable factor block above the noise.")
    elif Q > N // 3:
        print(f"    - MP returns Q = {Q} of {N}, so the bulk does not look like noise."
              f"\n      Do not read this as signal. The MP edge assumes a common scale"
              f"\n      across entries; fat tails make the bulk a mixture of MP laws, so"
              f"\n      the edge is misplaced. Withdrawn -- see BUILDNOTES regime 3.2.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
