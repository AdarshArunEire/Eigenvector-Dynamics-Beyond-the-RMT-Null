"""The detector must delete corporate actions and keep crashes.

The earlier version classified on "big move that does not revert", which a
crash and a splice both satisfy. On the SP500 panel that deleted AIG through the
Lehman weekend, Apple's 2000 profit warning and 88 other genuine days. The
discriminator now is that a split multiplies the price by an EXACT ratio, so its
log return is ln of that ratio to vendor precision, while a market move is not.

These tests pin both directions: the artifacts must be caught, and the real
moves must survive.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import (find_price_breaks, drop_price_breaks, unreliable_names,
                      log_returns, _split_ratio, SPLIT_RATIOS)


def _clean(n=60, start=100.0, seed=0):
    rng = np.random.default_rng(seed)
    px = start * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    return pd.Series(px, index=pd.bdate_range("2005-01-03", periods=n))


def _inject_split(s, at, ratio):
    """Scale everything before `at` by `ratio`, with no genuine move on the day.

    Real split artifacts land on ln(ratio) to machine precision because the
    vendor's mis-adjustment is the ONLY thing happening -- there is no market
    move mixed in. Scaling a random walk would leave that day's own return in
    the log return and the exactness would be lost, which is not what the data
    looks like.
    """
    s = s.copy()
    s.iloc[at] = s.iloc[at - 1]
    s.iloc[:at] *= ratio
    return s


def test_clean_series_has_no_breaks():
    df = pd.DataFrame({"A": _clean(), "B": _clean(seed=1)})
    assert find_price_breaks(df).empty


def test_exact_split_is_caught():
    for ratio in (2.0, 1.5, 2.5, 3.0, 10.0):
        s = _inject_split(_clean(), 30, ratio)
        br = find_price_breaks(s.to_frame("A"))
        assert list(br["kind"]) == ["split"], f"ratio {ratio} missed"
        assert br["split_ratio"].iloc[0] in SPLIT_RATIOS


def test_one_day_bad_print_is_a_spike():
    s = _clean()
    s.iloc[30] /= 2.0
    br = find_price_breaks(s.to_frame("A"))
    assert set(br["kind"]) == {"spike"}
    assert len(br) == 2                          # down then back up


@pytest.mark.parametrize("lr,name", [
    (-0.731, "AAPL 2000 profit warning"),
    (-0.936, "AIG, Lehman weekend"),
    (-0.943, "Williams Cos 2002"),
    (-1.145, "Quanta Services 2002"),
    (-0.892, "State Street 2009"),
])
def test_real_crashes_are_not_split_ratios(lr, name):
    assert _split_ratio(lr) is None, f"{name} would be deleted as a corporate action"


def test_non_ratio_step_is_kept_as_a_market_move():
    """A permanent level change that is not an exact ratio is a crash, not a
    splice, and must survive into the returns."""
    s = _clean()
    s.iloc[30:] *= 0.38                          # a 62% one-day collapse
    df = s.to_frame("A")
    br = find_price_breaks(df)
    assert list(br["kind"]) == ["large_move"]
    out = drop_price_breaks(log_returns(df), br)
    assert not out["A"].isna().any(), "a market move was deleted"


def test_drop_removes_only_defects():
    s = _inject_split(_clean(), 30, 2.0)         # split  -> dropped
    s.iloc[45:] *= 0.4                           # crash  -> kept
    df = s.to_frame("A")
    br = find_price_breaks(df)
    assert set(br["kind"]) == {"split", "large_move"}
    out = drop_price_breaks(log_returns(df), br)
    assert out["A"].isna().sum() == 1


def test_drop_does_not_interpolate():
    s = _inject_split(_clean(), 30, 2.0)
    df = s.to_frame("A")
    out = drop_price_breaks(log_returns(df), find_price_breaks(df))
    assert out["A"].isna().any()
    assert not out["A"].ffill().equals(out["A"])


def test_unreliable_names_flags_a_broken_series():
    """LIN.DE threw 290 flags on the DAX panel. Patching days is meaningless
    there -- the feed has come apart and the name has to go."""
    rng = np.random.default_rng(2)
    s = _clean(n=300)
    for i in sorted(rng.choice(np.arange(10, 290), 20, replace=False)):
        s = _inject_split(s, int(i), 3.0)        # repeated exact-ratio jumps
    df = pd.DataFrame({"BROKEN": s, "FINE": _clean(n=300, seed=5)})
    junk = unreliable_names(find_price_breaks(df))
    assert "BROKEN" in junk and "FINE" not in junk
    assert junk["BROKEN"][0] > 4


def test_threshold_only_gates_reporting():
    s = _clean()
    s.iloc[30] *= np.exp(0.5)
    assert len(find_price_breaks(s.to_frame("A"), threshold=0.6)) == 0
    assert len(find_price_breaks(s.to_frame("A"), threshold=0.3)) > 0
