"""Unit tests for the respecified realised-variance capture score."""
import numpy as np
import pytest

from src.capture import (ceiling_bias, equal_weight_market, random_floor,
                         realised_ceiling, skill, variance_captured)


def _haar(n, d, rng):
    return np.linalg.qr(rng.standard_normal((n, d)))[0]


def test_exact_ceiling_attains_the_maximum_against_random_frames():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((23, 42))
    best = variance_captured(realised_ceiling(returns), returns)["capture_6"]
    for _ in range(200):
        other = variance_captured(_haar(23, 6, rng), returns)["capture_6"]
        assert other <= best + 1e-12


def test_haar_frame_captures_d_over_n_in_expectation():
    rng = np.random.default_rng(1)
    for n in (23, 357):
        returns = rng.standard_normal((n, 42))
        mean = np.mean([variance_captured(_haar(n, 6, rng), returns)["capture_6"]
                        for _ in range(400)])
        assert mean == pytest.approx(random_floor(n)["capture_6"], abs=0.01)


def test_frame_orthogonal_to_the_return_span_captures_nothing():
    rng = np.random.default_rng(2)
    returns = rng.standard_normal((357, 42))
    complement = np.linalg.svd(returns, full_matrices=True)[0][:, 42:48]
    assert variance_captured(complement, returns)["capture_6"] < 1e-20


def test_nested_levels_are_monotone():
    rng = np.random.default_rng(3)
    returns = rng.standard_normal((40, 42))
    scores = variance_captured(_haar(40, 6, rng), returns)
    assert scores["capture_1"] <= scores["capture_3"] <= scores["capture_6"]


def test_capture_is_invariant_to_rotation_within_the_top_block():
    rng = np.random.default_rng(4)
    returns = rng.standard_normal((30, 42))
    frame = _haar(30, 6, rng)
    rotation = np.linalg.qr(rng.standard_normal((6, 6)))[0]
    left = variance_captured(frame, returns)["capture_6"]
    right = variance_captured(frame @ rotation, returns)["capture_6"]
    assert left == pytest.approx(right, abs=1e-12)


def test_neutralisation_removes_the_common_market_direction():
    rng = np.random.default_rng(5)
    n = 50
    market = equal_weight_market(n)
    returns = market @ rng.standard_normal((1, 42)) * 12 + rng.standard_normal((n, 42))
    # A frame whose first column IS the market carries most of the raw variance
    # and must score ~0 at level 1 once that common direction is projected out.
    frame = np.linalg.qr(np.hstack([market, rng.standard_normal((n, 5))]))[0]
    raw = variance_captured(frame, returns)
    neutral = variance_captured(frame, returns, neutralise=market)
    assert raw["capture_1"] > 0.5
    assert neutral["capture_1"] < 1e-20


def test_ceiling_bias_is_positive_when_the_subspace_does_not_move():
    rng = np.random.default_rng(6)
    base = rng.standard_normal((30, 300))
    covariance = np.cov(base)
    bias = ceiling_bias(covariance, horizon=42, replicates=50, rng=rng)
    # The truth is stationary by construction, so any headroom is overfitting.
    assert bias["capture_6"] > 0.05


def test_skill_denominator_shrinks_once_bias_is_removed():
    naive = skill(0.641, 0.638, 0.797)
    honest = skill(0.641, 0.638, 0.797, bias=0.112)
    assert honest > 3 * naive
    assert np.isnan(skill(0.641, 0.638, 0.700, bias=0.112))


def test_look_ahead_standardisation_is_the_callers_responsibility():
    rng = np.random.default_rng(7)
    returns = rng.standard_normal((10, 42)) * np.arange(1, 11)[:, None]
    frame = _haar(10, 6, rng)
    own = variance_captured(frame, returns / returns.std(axis=1, ddof=1)[:, None])
    raw = variance_captured(frame, returns)
    assert own["capture_6"] != pytest.approx(raw["capture_6"], abs=1e-6)
