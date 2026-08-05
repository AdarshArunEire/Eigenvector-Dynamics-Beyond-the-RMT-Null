"""Structural and leakage tests for Stage 2 covariance benchmarks."""
import numpy as np
import pandas as pd
import pytest

from src.covariance_benchmarks import (
    choose_ewma_half_life, condition_covariance, covariance_scores,
    estimate_bahc, estimate_cvc, estimate_ewma, estimate_hcal,
    estimate_ledoit_wolf, estimate_nonlinear_shrinkage, estimate_oas,
    estimate_sample, long_only_gmv_weights, rolling_covariance_examples,
    split_covariance_examples, unconstrained_gmv_weights)
from src.forecast import ChronologicalSplits


def test_estimators_are_symmetric_and_well_conditioned_where_promised():
    # BAHC is an optional third-party filter pinned in requirements.txt; skip
    # rather than fail so a fresh clone without it still reports a green suite
    # for everything this repository actually implements.
    pytest.importorskip("bahc")
    rng = np.random.default_rng(7)
    returns = rng.normal(size=(12, 80))
    for estimator in (estimate_sample, estimate_ledoit_wolf, estimate_oas,
                      estimate_nonlinear_shrinkage, estimate_cvc,
                      estimate_hcal):
        covariance = estimator(returns)
        assert covariance.shape == (12, 12)
        assert np.allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(estimate_ledoit_wolf(returns))[0] > 0
    assert np.linalg.eigvalsh(estimate_oas(returns))[0] > 0


def test_rotationally_invariant_cleaners_keep_full_sample_eigenvectors():
    rng = np.random.default_rng(19)
    returns = rng.normal(size=(12, 80)) * np.arange(1, 13)[:, None]
    _, empirical_vectors = np.linalg.eigh(estimate_sample(returns))
    for estimator in (estimate_nonlinear_shrinkage, estimate_cvc):
        cleaned = estimator(returns)
        representation = empirical_vectors.T @ cleaned @ empirical_vectors
        assert np.linalg.norm(representation - np.diag(np.diag(representation))) < 1e-8


def test_hcal_and_bahc_are_reproducible_covariance_filters():
    pytest.importorskip("bahc")
    rng = np.random.default_rng(23)
    returns = rng.normal(size=(9, 60))
    hcal = estimate_hcal(returns)
    first = estimate_bahc(returns, bootstraps=4, seed=4)
    second = estimate_bahc(returns, bootstraps=4, seed=4)
    assert np.allclose(first, second)
    assert hcal.shape == first.shape == (9, 9)
    empirical_variances = returns.var(axis=1)
    assert np.diag(hcal) == pytest.approx(empirical_variances)
    assert np.diag(first) == pytest.approx(empirical_variances)


def test_ewma_weights_recent_observations_more_and_accepts_equivalent_decay():
    returns = np.zeros((2, 20))
    returns[0, -3:] = [1., -1., 1.]
    returns[1, :3] = [1., -1., 1.]
    recent = estimate_ewma(returns, half_life=3)
    decay = np.exp(np.log(.5) / 3)
    direct = estimate_ewma(returns, decay=decay)
    assert np.allclose(recent, direct)
    assert recent[0, 0] > recent[1, 1]


def test_rolling_examples_use_only_returns_after_forecast_origin_as_target():
    panel = np.arange(2 * 80, dtype=float).reshape(2, 80)
    dates = pd.date_range("2000-01-01", periods=80)
    examples = rolling_covariance_examples(panel, dates, T=20,
                                           horizon=4, step=2)
    row = examples[0]
    assert row.estimation_returns[0, -1] == panel[0, row.current_end]
    assert row.realised_returns[0, 0] == panel[0, row.current_end + 1]
    assert row.realised_returns.shape[1] == 4
    assert row.target_window_start > row.current_date


def test_changing_future_returns_cannot_change_a_past_covariance_forecast():
    rng = np.random.default_rng(11)
    panel = rng.normal(size=(5, 100))
    dates = pd.date_range("2000-01-01", periods=100)
    original = rolling_covariance_examples(panel, dates, 30, 6, 2)[0]
    changed_panel = panel.copy()
    changed_panel[:, original.target_start:] *= 100
    changed = rolling_covariance_examples(changed_panel, dates, 30, 6, 2)[0]
    assert np.array_equal(original.estimation_returns,
                          changed.estimation_returns)
    assert not np.array_equal(original.realised_returns,
                              changed.realised_returns)
    assert np.array_equal(estimate_sample(original.estimation_returns),
                          estimate_sample(changed.estimation_returns))


def test_oracle_covariance_has_zero_frobenius_error():
    rng = np.random.default_rng(13)
    returns = rng.normal(size=(8, 50))
    covariance = estimate_sample(returns)
    scores = covariance_scores(covariance, covariance,
                               compute_long_only=False)
    assert scores["relative_frobenius"] == pytest.approx(0.0, abs=1e-15)
    assert scores["flag_nested_loss"] == pytest.approx(0.0, abs=1e-12)


def test_gmv_identity_weights_are_equal_with_and_without_shorting():
    covariance = np.eye(7)
    expected = np.full(7, 1 / 7)
    assert unconstrained_gmv_weights(covariance) == pytest.approx(expected)
    assert long_only_gmv_weights(covariance) == pytest.approx(expected)


def test_long_only_nnls_solution_satisfies_quadratic_kkt_conditions():
    rng = np.random.default_rng(29)
    loading = rng.normal(size=(10, 10))
    covariance = loading @ loading.T + .2 * np.eye(10)
    weights = long_only_gmv_weights(covariance)
    gradient = covariance @ weights
    positive = weights > 1e-9
    multiplier = float(np.mean(gradient[positive]))
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= 0
    assert gradient[positive] == pytest.approx(
        np.full(positive.sum(), multiplier), abs=1e-8)
    assert np.all(gradient[~positive] >= multiplier - 1e-8)


def test_conditioning_reports_exactly_the_floored_singular_directions():
    covariance = np.diag([3., 1., 0.])
    conditioned, diagnostics = condition_covariance(covariance,
                                                     relative_floor=1e-6)
    assert np.linalg.eigvalsh(conditioned)[0] == pytest.approx(3e-6)
    assert diagnostics["n_eigenvalues_floored"] == 1
    assert np.isinf(diagnostics["raw_condition_number"])


def test_ewma_selection_reads_validation_rows_but_not_test_assignment():
    rng = np.random.default_rng(17)
    panel = rng.normal(size=(8, 260))
    dates = pd.date_range("2000-01-01", periods=260)
    examples = rolling_covariance_examples(panel, dates, 30, 6, 2)
    specification = ChronologicalSplits(
        train_end="2000-03-01", validation_start="2000-03-20",
        validation_end="2000-05-20", test_start="2000-07-01")
    splits = split_covariance_examples(examples, specification)
    selected, table = choose_ewma_half_life(examples, splits, [2, 5, 10])
    changed = splits.copy()
    changed.loc[changed["split"] == "test", "split"] = "purged"
    selected_again, table_again = choose_ewma_half_life(
        examples, changed, [2, 5, 10])
    assert selected_again == selected
    assert table_again.equals(table)
