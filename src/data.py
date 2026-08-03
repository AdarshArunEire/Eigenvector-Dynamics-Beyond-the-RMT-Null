"""Real return panels: fetch, clean, diagnose.

Everything downstream of this module has been calibrated against synthetic
worlds where the truth was known. Nothing downstream of *this* line has. So the
job here is not just to produce a matrix -- it is to produce a matrix whose
defects are measured and stated rather than silently absorbed.

Three defects matter, in descending order of danger:

1. Stale prices. A name that did not trade reports an unchanged close, which
   becomes a return of exactly zero. Zero returns pull the covariance down and
   manufacture structure that is not there. Dropped silently by a naive
   `dropna()`, because they are not missing -- they are present and wrong.
2. Fat tails. Eq (7) is derived from the Wishart fourth moment, which *is* the
   Gaussian assumption. Real equity kurtosis runs 5-10. `panel_report` measures
   it so the departure is a number rather than a worry.
3. Survivorship. Any list of "names with full history over the period" is
   conditioned on survival. `filter_by_coverage` reports what it dropped and
   why, so the surviving universe is inspectable.

Downloading needs network and so cannot run in CI. Everything else here is
pure and tested.
"""
import numpy as np
import pandas as pd

# Candidate US large caps plausibly trading 2000-2010, for building the
# pipeline against clean data before pointing it at Tokyo. Listed from memory
# and NOT verified: treat it as a starting set, not a universe. Names without
# full history are dropped by filter_by_coverage, which reports what it lost --
# so an imperfect list is safe, just smaller than it looks.
US_LARGE_CAP_CANDIDATES = [
    "AAPL", "MSFT", "JNJ", "XOM", "PG", "KO", "PEP", "WMT", "HD", "MCD",
    "IBM", "INTC", "CSCO", "ORCL", "TXN", "QCOM", "AMGN", "GILD", "MRK", "PFE",
    "ABT", "BMY", "LLY", "MDT", "UNH", "CVX", "COP", "SLB", "HAL", "OXY",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC", "BK",
    "GE", "MMM", "CAT", "DE", "BA", "HON", "UTX", "LMT", "NOC", "RTN",
    "DIS", "CMCSA", "T", "VZ", "LOW", "TGT", "COST", "CVS", "WBA", "KR",
    "NKE", "SBUX", "YUM", "CL", "KMB", "GIS", "K", "SYY", "ADM", "MO",
    "DD", "DOW", "PPG", "SHW", "NEM", "FCX", "NUE", "APD", "ECL", "EMR",
    "ADBE", "ADP", "AMAT", "MU", "AMD", "HPQ", "DELL", "EBAY", "AMZN", "NVDA",
    "SO", "DUK", "NEE", "AEP", "EXC", "D", "SRE", "PEG", "ED", "XEL",
    "SPG", "PLD", "AVB", "EQR", "VNO", "BXP", "O", "PSA", "HST", "KIM",
    "TRV", "ALL", "PGR", "AIG", "MET", "PRU", "AFL", "CB", "L", "HIG",
    "UPS", "FDX", "NSC", "UNP", "CSX", "LUV", "DAL", "R", "EXPD", "CHRW",
    "MAR", "HLT", "CCL", "RCL", "MGM", "WYNN", "DRI", "CMG", "DPZ", "JACK",
    "BBY", "GPS", "ROST", "TJX", "M", "JWN", "KSS", "DDS", "AEO", "ANF",
    "GLW", "MSI", "JNPR", "NTAP", "STX", "WDC", "LRCX", "KLAC", "ADI", "MCHP",
    "SYK", "BSX", "BDX", "BAX", "ZBH", "EW", "VAR", "HOLX", "RMD", "XRAY",
    "APA", "DVN", "EOG", "MRO", "NBL", "PXD", "SWN", "CHK", "BHI", "NOV",
    "PCAR", "CMI", "ITW", "PH", "ROK", "DOV", "SWK", "FAST", "GWW", "MAS",
    "STT", "NTRS", "SCHW", "TROW", "BEN", "IVZ", "CME", "ICE", "MMC", "AON",
    "VMC", "MLM", "IP", "PKG", "SEE", "BLL", "CCK", "AVY", "WY", "LYB",
]


def _close_frame(raw, chunk):
    """Pull the Close block out of a yfinance response, whatever shape it took.

    yfinance returns MultiIndex (field, ticker) columns for several tickers and
    flat columns for one, and with auto_adjust=True there is no 'Adj Close'.
    Both shapes have to be handled or a one-survivor chunk silently vanishes.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        return raw["Close"]
    if "Close" not in raw.columns:
        return pd.DataFrame()
    out = raw[["Close"]]
    return out.rename(columns={"Close": chunk[0]}) if len(chunk) == 1 else out


def fetch_prices(tickers, start, end, cache=None, auto_adjust=True,
                 batch_size=40, pause=2.0, retries=3):
    """Adjusted daily closes as a (dates x tickers) frame.

    Needs network, so this is the one function here that cannot be tested in
    CI. Pass `cache` (a .parquet path) to download once and reuse: the run that
    produces a published number should read the cache, not the wire, or the
    result is not reproducible.

    auto_adjust=True is not optional in practice. On raw closes a 2-for-1 split
    reads as a -50% single-day return, which would dominate every covariance it
    touched.

    Downloads in batches with a pause between them, because Yahoo rate-limits
    and a large single request comes back with the right columns and ZERO ROWS
    rather than an error. That failure mode is the dangerous one: it is silent,
    it poisons the cache if written, and it surfaces hundreds of lines later as
    an unrelated exception. So:

      - only missing tickers are fetched, and successful batches are merged into
        the cache as they arrive, making a throttled run resumable rather than
        wasted;
      - a batch that comes back empty is retried with a longer pause;
      - an empty overall result raises here instead of being cached.
    """
    import time

    tickers = list(tickers)
    have = pd.DataFrame()
    if cache is not None:
        try:
            have = pd.read_parquet(cache)
        except (FileNotFoundError, OSError):
            have = pd.DataFrame()
    if len(have.columns) and not len(have):
        have = pd.DataFrame()               # a poisoned zero-row cache: discard

    missing = [t for t in tickers if t not in have.columns]
    if not missing:
        return have.loc[str(start):str(end), tickers]

    import yfinance as yf                      # imported late: optional dep
    got = []
    for i in range(0, len(missing), batch_size):
        chunk = missing[i:i + batch_size]
        for attempt in range(retries):
            raw = yf.download(chunk, start=start, end=end, auto_adjust=auto_adjust,
                              progress=False, threads=False)
            px = _close_frame(raw, chunk)
            if len(px):
                got.append(px)
                break
            wait = pause * (attempt + 2)
            print(f"      batch {i // batch_size + 1}: no rows for "
                  f"{len(chunk)} tickers, retrying in {wait:.0f}s "
                  f"(attempt {attempt + 2}/{retries})")
            time.sleep(wait)
        else:
            print(f"      batch {i // batch_size + 1}: gave up on {len(chunk)} "
                  f"tickers after {retries} attempts -- rate limited")
        time.sleep(pause)
        if cache is not None and got:        # checkpoint: survive an abort
            pd.concat([have] + got, axis=1).sort_index().to_parquet(cache)

    px = pd.concat([have] + got, axis=1).sort_index() if got else have
    if not len(px) or not len(px.columns):
        raise RuntimeError(
            "the download returned no rows at all. Yahoo answers a rate-limited "
            "request with empty data rather than an error, so this is almost "
            "always throttling: wait a few minutes and re-run, or lower "
            "--batch-size. Nothing has been cached.")
    if cache is not None:
        px.to_parquet(cache)
    keep = [t for t in tickers if t in px.columns]
    return px.loc[str(start):str(end), keep]


def log_returns(prices):
    """Log returns from a (dates x tickers) price frame. Loses the first row."""
    prices = pd.DataFrame(prices).astype(float)
    if (prices <= 0).to_numpy().any():
        raise ValueError("non-positive prices present; cannot take logs")
    return np.log(prices).diff().iloc[1:]


# Ratios companies actually split or consolidate on. Deliberately a whitelist
# rather than "any small rational": allowing every p/q with q<=12 lets real
# crashes through by coincidence -- Quanta Services' 2002 collapse lands within
# 0.02% of 22/7, and a Japanese merger splice within 0.04% of 16/3. Neither is
# a ratio any board has ever voted for.
SPLIT_RATIOS = {
    "21:20": 1.05, "11:10": 1.1, "6:5": 1.2, "5:4": 1.25, "4:3": 4 / 3,
    "7:5": 1.4, "3:2": 1.5, "5:3": 5 / 3, "9:5": 1.8, "7:4": 1.75,
    "2:1": 2.0, "5:2": 2.5, "3:1": 3.0, "7:2": 3.5, "4:1": 4.0,
    "5:1": 5.0, "6:1": 6.0, "7:1": 7.0, "8:1": 8.0, "10:1": 10.0,
    "12:1": 12.0, "15:1": 15.0, "20:1": 20.0, "25:1": 25.0, "50:1": 50.0,
    "100:1": 100.0, "1000:1": 1000.0,
}


def _split_ratio(log_return, rel_tol=2e-4):
    """If this log return is exactly ln of a real split ratio, name it.

    This is the whole discriminator, and it works because a split or a
    consolidation multiplies the price by an EXACT ratio, so the log return
    lands on ln of that ratio to the precision the vendor stores. A market move
    lands on an arbitrary number.

    Worked examples from the panels this was built against:

        4188.T  -0.693147  ->  exp = 2.00000  ->  2:1        artifact
        9201.T  -0.916291  ->  exp = 2.50000  ->  5:2        artifact
        AAPL    -0.731     ->  exp = 2.0772   ->  3.9% off 2:1
        AIG     -0.936     ->  exp = 2.5498   ->  2.0% off 5:2
        PWR     -1.145     ->  exp = 3.1424   ->  4.7% off 3:1

    The last three are Apple's 2000 profit warning, the AIG collapse and the
    Quanta Services collapse. They are among the most informative days in the
    sample and the previous version of this function deleted them.
    """
    r = float(np.exp(abs(log_return)))
    for name, v in SPLIT_RATIOS.items():
        if abs(r - v) < rel_tol * v:
            return name
    return None


def find_price_breaks(prices, threshold=0.4, window=10, step_tol=0.15):
    """Locate corporate-action damage in an adjusted price series.

    Returns one row per suspect date with a `kind`. Only two kinds are defects:

    `split`  the log return is exactly ln of a small rational. A split,
             consolidation or exchange-ratio splice. The return on that day is
             not a return and must be dropped.
    `spike`  one close is wrong and is corrected the next day; the median level
             either side agrees. A bad print. Also dropped.

    Everything else that clears `threshold` is reported as `large_move` and
    LEFT ALONE. A crash and a splice produce the same signature -- big, no
    reversion -- and an earlier version of this function treated them alike,
    which on a 2000-2010 US panel deleted AIG through the Lehman weekend, Apple's
    2000 profit warning, Citigroup in 2009 and 58 other names' worst days. Those
    are precisely the days the correlation structure moves, so removing them
    guts the signal being measured.

    `series_start` / `series_end` mark a break at the first or last observation,
    where there is no level on one side to compare against. Not defects.

    The threshold is a *reporting* floor now, not a deletion rule, so it can
    stay loose without doing damage.
    """
    prices = pd.DataFrame(prices).astype(float)
    lr = np.log(prices.where(prices > 0)).diff()
    rows = []
    for col in prices.columns:
        s, r = prices[col], lr[col]
        for date in r.index[(r.abs() > threshold).fillna(False)]:
            i = prices.index.get_loc(date)
            lo = s.iloc[max(0, i - window - 1):max(0, i - 1)].dropna()
            hi = s.iloc[i + 1:i + 1 + window].dropna()
            before = lo.median() if len(lo) else np.nan
            after = hi.median() if len(hi) else np.nan
            frac = _split_ratio(r[date])
            if not np.isfinite(before):
                kind, ratio = "series_start", np.nan
            elif not np.isfinite(after) or after == 0:
                kind, ratio = "series_end", np.nan
            else:
                ratio = float(before / after)
                if abs(np.log(ratio)) < step_tol:
                    kind = "spike"
                elif frac is not None:
                    kind = "split"
                else:
                    kind = "large_move"
            rows.append({"ticker": col, "date": date, "log_return": float(r[date]),
                         "level_ratio": ratio, "kind": kind,
                         "split_ratio": str(frac) if frac is not None else ""})
    return pd.DataFrame(rows, columns=["ticker", "date", "log_return",
                                       "level_ratio", "kind", "split_ratio"])


DEFECT_KINDS = ("split", "spike")


def drop_price_breaks(returns, breaks):
    """NaN out the return on each `split` or `spike` date. Returns a new frame.

    `large_move` rows are deliberately untouched -- see `find_price_breaks`.

    Deliberately does not interpolate. A break date has no recoverable return,
    and inventing one would put a fabricated number into a covariance window.
    NaN propagates into `filter_by_coverage`, where a name with too many of them
    is dropped on coverage like any other gappy series.
    """
    out = pd.DataFrame(returns).copy()
    b = pd.DataFrame(breaks)
    if not len(b):
        return out
    for _, row in b[b["kind"].isin(DEFECT_KINDS)].iterrows():
        if row["ticker"] in out.columns and row["date"] in out.index:
            out.loc[row["date"], row["ticker"]] = np.nan
    return out


def unreliable_names(breaks, max_defects=4, max_moves=20):
    """Names whose SERIES is broken, not whose days are.

    Two separate counts, because they mean different things:

    `max_defects`  splits and spikes. A company splits a handful of times a
                   decade; a name showing dozens has a feed that mis-applies
                   adjustments, and patching individual days there is pointless.
    `max_moves`    large moves. These are legitimate one at a time -- AIG has
                   five in 2008 and every one is real -- so the bar has to be
                   far higher, and it is only there to catch a series that is
                   simply noise. LIN.DE throws 247 of them, swinging by factors
                   of five and ten repeatedly through 2007-08.

    Counting the two together is what dropped AIG on the first pass, which is
    the single worst name to lose from a 2000-2010 US panel. Returns
    {ticker: (n_defects, n_moves)} for names over either bar.
    """
    b = pd.DataFrame(breaks)
    if not len(b):
        return {}
    d = b[b["kind"].isin(DEFECT_KINDS)].groupby("ticker").size()
    m = b[b["kind"] == "large_move"].groupby("ticker").size()
    names = set(d[d > max_defects].index) | set(m[m > max_moves].index)
    return {t_: (int(d.get(t_, 0)), int(m.get(t_, 0)))
            for t_ in sorted(names, key=lambda x: -(d.get(x, 0) + m.get(x, 0)))}


def longest_stale_run(series):
    """Longest run of consecutive exactly-zero returns.

    The stale-price detector. One or two zeros is a quiet day; a run of thirty
    is a name that is not really trading, and its correlations with everything
    else are fictitious.
    """
    z = (np.asarray(series, dtype=float) == 0.0)
    if not z.any():
        return 0
    best = run = 0
    for flag in z:
        run = run + 1 if flag else 0
        best = max(best, run)
    return int(best)


def filter_by_coverage(returns, min_coverage=0.98, max_stale_run=10,
                       max_zero_fraction=0.10):
    """Drop names that are absent, stale, or barely trading. Returns (kept, dropped).

    `dropped` is a frame of reasons, not a count. The universe that survives
    this filter is the universe the results describe, and the difference
    between it and the one you asked for is survivorship bias made explicit.
    """
    returns = pd.DataFrame(returns)
    if returns.empty or not len(returns.columns):
        raise ValueError(
            "no returns to filter -- the price frame was empty. Check the "
            "[1/4] line: if it says '0 returned data', the download failed "
            "rather than the filter.")
    n = len(returns)
    rows = []
    for col in returns.columns:
        s = returns[col]
        cov = float(s.notna().mean())
        filled = s.fillna(0.0)
        zfrac = float((filled == 0.0).mean())
        stale = longest_stale_run(filled)
        reason = None
        if cov < min_coverage:
            reason = f"coverage {cov:.1%} < {min_coverage:.0%}"
        elif stale > max_stale_run:
            reason = f"stale run {stale} > {max_stale_run}"
        elif zfrac > max_zero_fraction:
            reason = f"zero returns {zfrac:.1%} > {max_zero_fraction:.0%}"
        rows.append({"ticker": col, "coverage": cov, "zero_fraction": zfrac,
                     "longest_stale_run": stale, "dropped_because": reason})
    report = pd.DataFrame(rows).set_index("ticker")
    keep = report.index[report["dropped_because"].isna()]
    out = returns[keep].dropna(axis=0, how="any")
    if out.empty:
        raise ValueError(f"no names survived filtering (started with {len(returns.columns)}, "
                         f"{n} dates)")
    return out, report


def to_panel(returns):
    """(dates x tickers) frame -> N x T array, matching sample_covariance()."""
    return np.asarray(returns, dtype=float).T


def variance_path(panel, window=21):
    """Rolling cross-sectional variance level, one value per day.

    The observable behind regime 2.3. Cross-sectional because a single day's
    average squared return across N names estimates that day's overall variance
    level without needing any model of it.
    """
    panel = np.asarray(panel, dtype=float)
    if panel.ndim != 2:
        raise ValueError(f"panel must be 2-D (N x T), got {panel.shape}")
    if window < 1:
        raise ValueError("window must be >= 1")
    v = (panel ** 2).mean(axis=0)
    if window == 1:
        return v
    pad = window // 2
    padded = np.r_[v[:pad][::-1], v, v[-pad:][::-1]] if pad else v
    return np.convolve(padded, np.ones(window) / window, mode="valid")[:v.size]


def cv_squared(panel, window=21):
    """<c^2>/<c>^2 - 1 for the within-window variance path.

    Regime 2.3 says D is inflated by 1 + this. Estimated from returns, so it
    inherits an upward bias of roughly 2/(N*window) from the noise in the path
    itself -- see stage1/README.md, 2.3.
    """
    s = variance_path(panel, window)
    return float((s ** 2).mean() / s.mean() ** 2 - 1.0)


def standardise(panel, window=21):
    """Divide each day by a rolling estimate of that day's volatility.

    The regime 2.3 remedy: flattens the variance path so CV^2 -> 0 and the
    inflation with it. Measured synthetically to take a 183% inflation down to
    1.3%, and it needs no knowledge of the true path -- only the returns.
    """
    s = variance_path(panel, window)
    # A day on which every name returned exactly zero -- an exchange holiday that
    # leaked into the date index, which happens on the Tokyo and Paris panels --
    # gives s = 0 and a 0/0. Those days carry no information either way, so they
    # are left at unit scale rather than propagating NaN through the whole panel.
    scale = s / s[s > 0].mean() if (s > 0).any() else np.ones_like(s)
    scale = np.where(scale > 0, scale, 1.0)
    return np.asarray(panel, dtype=float) / np.sqrt(scale)[None, :]


def to_correlation_panel(panel):
    """Scale each name to unit variance. Row-wise, not column-wise.

    Required before the Marchenko-Pastur edge means anything. MP describes the
    bulk of a matrix built from variables of a *common* variance; real names
    differ in volatility by factors of several, and that heterogeneity spreads
    the bulk far wider than sampling noise alone would. Fed a raw covariance
    the edge criterion collapses sigma^2 and reports most of the spectrum as
    signal.

    Not to be confused with `standardise`, which divides each *day* by that
    day's cross-sectional volatility and addresses regime 2.3. This one divides
    each *name* by its own volatility over the window. Different axis,
    different purpose, and a panel usually wants both.
    """
    panel = np.asarray(panel, dtype=float)
    sd = panel.std(axis=1, ddof=1, keepdims=True)
    if np.any(sd == 0):
        raise ValueError(f"{int((sd == 0).sum())} name(s) have zero variance")
    return panel / sd


def worst_outliers(panel, tickers=None, k=5):
    """The k largest absolute returns, in units of that name's own sd.

    Kurtosis says a panel has heavy tails; this says which rows and dates are
    responsible. A single move beyond about 20 sd is not a fat tail, it is an
    unadjusted corporate action or a bad print.
    """
    panel = np.asarray(panel, dtype=float)
    z = np.abs(panel - panel.mean(axis=1, keepdims=True))
    z = z / panel.std(axis=1, ddof=1, keepdims=True)
    flat = np.argsort(z, axis=None)[::-1][:k]
    rows, cols = np.unravel_index(flat, z.shape)
    return [{"name": tickers[i] if tickers is not None else int(i),
             "column": int(j), "sd": float(z[i, j]), "return": float(panel[i, j])}
            for i, j in zip(rows, cols)]


def panel_report(panel):
    """Everything about this panel that could invalidate a downstream number."""
    panel = np.asarray(panel, dtype=float)
    N, T = panel.shape
    z = panel - panel.mean(axis=1, keepdims=True)
    sd = z.std(axis=1, ddof=1)
    kurt = (z ** 4).mean(axis=1) / sd ** 4          # 3.0 under a normal
    return {
        "N": N, "T": T, "q": N / T,
        "kurtosis_median": float(np.median(kurt)),
        "kurtosis_max": float(kurt.max()),
        "excess_kurtosis_median": float(np.median(kurt) - 3.0),
        "zero_return_fraction": float((panel == 0.0).mean()),
        "cv_squared": cv_squared(panel),
        "annualised_vol_median": float(np.median(sd) * np.sqrt(252)),
    }
