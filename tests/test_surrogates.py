"""Structural tests for the Regime 4.8 robustness machinery."""
import numpy as np
import pytest

from src.surrogates import (iaaft_surrogate, independent_iaaft_panel,
                            remove_equal_weight_factor)


def _lag1(x):
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def test_iaaft_preserves_exact_marginal_and_approximately_preserves_lag_one():
    rng = np.random.default_rng(8)
    noise = rng.standard_t(5, 1000)
    x = np.empty_like(noise)
    x[0] = noise[0]
    for t in range(1, len(x)):
        x[t] = 0.75 * x[t - 1] + noise[t]
    surrogate = iaaft_surrogate(x, np.random.default_rng(9), max_iter=300)
    assert np.sort(surrogate) == pytest.approx(np.sort(x), abs=0)
    assert _lag1(surrogate) == pytest.approx(_lag1(x), abs=0.04)


def test_independent_iaaft_removes_shared_timing_and_is_reproducible():
    rng = np.random.default_rng(10)
    common = rng.standard_normal(1200)
    panel = np.asarray([common + 0.15 * rng.standard_normal(1200)
                        for _ in range(6)])
    first = independent_iaaft_panel(panel, np.random.default_rng(11), max_iter=100)
    second = independent_iaaft_panel(panel, np.random.default_rng(11), max_iter=100)
    assert first == pytest.approx(second, abs=0)
    before = np.mean(np.abs(np.corrcoef(panel)[np.triu_indices(6, 1)]))
    after = np.mean(np.abs(np.corrcoef(first)[np.triu_indices(6, 1)]))
    assert before > 0.9
    assert after < 0.12


def test_equal_weight_factor_removal_annihilates_market_exposure():
    rng = np.random.default_rng(12)
    factor = rng.standard_normal(500)
    betas = np.linspace(0.5, 1.8, 12)
    panel = betas[:, None] * factor + 0.2 * rng.standard_normal((12, 500))
    residual = remove_equal_weight_factor(panel)
    market = panel.mean(axis=0)
    exposures = residual @ (market - market.mean())
    assert exposures == pytest.approx(np.zeros(12), abs=1e-10)
    assert residual.mean(axis=1) == pytest.approx(np.zeros(12), abs=1e-14)


def test_factor_removal_does_not_manufacture_dependence_from_independent_noise():
    rng = np.random.default_rng(13)
    panel = rng.standard_normal((80, 3000))
    residual = remove_equal_weight_factor(panel)
    offdiag = np.corrcoef(residual)[np.triu_indices(80, 1)]
    # Removing an internally estimated equal-weight factor imposes a small
    # negative dependence of order 1/N, but no large common residual factor.
    assert abs(np.mean(offdiag)) < 0.02
    assert np.quantile(np.abs(offdiag), .95) < 0.06
