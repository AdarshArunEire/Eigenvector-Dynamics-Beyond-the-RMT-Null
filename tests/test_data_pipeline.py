"""Stage 1: the real-data pipeline, tested without touching the network.

Every transformation between a price frame and an N x T panel is pure, so all
of it is testable on constructed series where the answer is known. Only
`fetch_prices` needs the wire, and it is deliberately the only function here
with no logic in it worth testing.

The defects these tests pin down are the ones that would otherwise be absorbed
in silence: a split misread as a crash, a stale name misread as a stable one,
a volatility regime misread as rotation.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import (log_returns, longest_stale_run, filter_by_coverage,
                      to_panel, variance_path, cv_squared, standardise,
                      panel_report)


def _prices(n=300, seed=0, drift=0.0002, vol=0.01, cols=("A", "B", "C")):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2004-01-01", periods=n)
    steps = rng.normal(drift, vol, size=(n, len(cols)))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=list(cols))


# --------------------------------------------------------------------------
# Prices to returns.
# --------------------------------------------------------------------------

def test_log_returns_are_exact_on_a_known_path():
    px = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    r = log_returns(px)
    assert r["A"].tolist() == pytest.approx([np.log(1.1), np.log(0.9)])
    assert len(r) == len(px) - 1


def test_log_returns_reject_non_positive_prices():
    with pytest.raises(ValueError, match="non-positive"):
        log_returns(pd.DataFrame({"A": [100.0, 0.0, 99.0]}))


def test_an_unadjusted_split_would_be_a_50_percent_return():
    """Why auto_adjust is not optional.

    A 2-for-1 split on raw closes produces a single -69% log return, tens of
    standard deviations from anything real, and it would dominate every
    covariance entry that name appears in.
    """
    px = _prices(60, seed=1)
    px.iloc[30:, 0] /= 2.0                      # unadjusted split in A
    r = log_returns(px)
    assert r["A"].min() < -0.5
    assert abs(r["A"].min()) > 20 * r["B"].std()


# --------------------------------------------------------------------------
# Stale prices: present, and wrong.
# --------------------------------------------------------------------------

def test_longest_stale_run_counts_consecutive_zeros():
    assert longest_stale_run([0.01, -0.02, 0.03]) == 0
    assert longest_stale_run([0.0, 0.0, 0.01, 0.0]) == 2
    assert longest_stale_run([0.0] * 7) == 7
    assert longest_stale_run([]) == 0


def test_filter_drops_a_stale_name_and_says_why():
    px = _prices(300, seed=2, cols=("GOOD", "STALE", "OK"))
    r = log_returns(px)
    r.iloc[100:140, r.columns.get_loc("STALE")] = 0.0     # 40 days without a trade
    kept, report = filter_by_coverage(r, max_stale_run=10)
    assert "STALE" not in kept.columns
    assert {"GOOD", "OK"} <= set(kept.columns)
    assert "stale run" in report.loc["STALE", "dropped_because"]
    assert report.loc["GOOD", "dropped_because"] is None


def test_filter_drops_a_name_with_too_little_history():
    px = _prices(300, seed=3, cols=("FULL", "SHORT"))
    r = log_returns(px)
    r.iloc[:100, r.columns.get_loc("SHORT")] = np.nan
    kept, report = filter_by_coverage(r, min_coverage=0.98)
    assert "SHORT" not in kept.columns
    assert "coverage" in report.loc["SHORT", "dropped_because"]


def test_the_report_makes_survivorship_inspectable():
    """The gap between the universe asked for and the one that survives IS the
    bias, so it has to be a readable object rather than a count."""
    px = _prices(300, seed=4, cols=("A", "B", "C"))
    r = log_returns(px)
    r.iloc[:200, 1] = np.nan
    _, report = filter_by_coverage(r)
    assert set(report.columns) >= {"coverage", "zero_fraction",
                                   "longest_stale_run", "dropped_because"}
    assert report["dropped_because"].notna().sum() == 1


def test_filter_raises_rather_than_returning_an_empty_universe():
    px = _prices(50, seed=5, cols=("A", "B"))
    r = log_returns(px)
    r.iloc[:, :] = 0.0
    with pytest.raises(ValueError, match="no names survived"):
        filter_by_coverage(r)


# --------------------------------------------------------------------------
# Orientation, and the regime 2.3 observables.
# --------------------------------------------------------------------------

def test_to_panel_matches_the_sample_covariance_convention():
    """sample_covariance expects N x T, frames are dates x tickers."""
    r = log_returns(_prices(120, seed=6))
    panel = to_panel(r)
    assert panel.shape == (r.shape[1], r.shape[0])
    assert panel[0] == pytest.approx(r.iloc[:, 0].to_numpy())


def test_cv_squared_is_near_zero_for_a_flat_volatility_path():
    rng = np.random.default_rng(7)
    assert cv_squared(rng.normal(0, 0.01, size=(60, 2000))) < 0.05


def test_cv_squared_rises_with_a_volatility_regime_change():
    """The observable behind 2.3: it has to see a crisis to be worth anything."""
    rng = np.random.default_rng(8)
    calm = rng.normal(0, 0.01, size=(60, 1600))
    crisis = rng.normal(0, 0.03, size=(60, 400))          # 9x the variance
    assert cv_squared(np.hstack([calm, crisis])) > 0.5


def test_standardise_flattens_the_path_without_damaging_a_flat_one():
    rng = np.random.default_rng(9)
    flat = rng.normal(0, 0.01, size=(60, 2000))
    spiky = np.hstack([rng.normal(0, 0.01, size=(60, 1600)),
                       rng.normal(0, 0.03, size=(60, 400))])
    assert cv_squared(standardise(spiky)) < 0.1 * cv_squared(spiky)
    assert cv_squared(standardise(flat)) < 0.05


def test_variance_path_rejects_bad_shapes():
    with pytest.raises(ValueError, match="2-D"):
        variance_path(np.zeros(10))
    with pytest.raises(ValueError, match="window"):
        variance_path(np.zeros((4, 10)), window=0)


# --------------------------------------------------------------------------
# The report: defects as numbers.
# --------------------------------------------------------------------------

def test_panel_report_recovers_gaussian_kurtosis():
    rng = np.random.default_rng(10)
    rep = panel_report(rng.normal(0, 0.01, size=(80, 4000)))
    assert rep["kurtosis_median"] == pytest.approx(3.0, abs=0.25)
    assert abs(rep["excess_kurtosis_median"]) < 0.25
    assert rep["q"] == pytest.approx(80 / 4000)


def test_panel_report_flags_fat_tails():
    """The assumption Eq (7) rests on, measured rather than hoped for.

    Student-t with 4 degrees of freedom has infinite theoretical kurtosis; the
    sample value is large and unstable. Real equities land between this and
    the Gaussian case, which is exactly why it needs reporting per panel.
    """
    rng = np.random.default_rng(11)
    rep = panel_report(rng.standard_t(4, size=(80, 4000)))
    assert rep["kurtosis_median"] > 5.0
    assert rep["excess_kurtosis_median"] > 2.0
