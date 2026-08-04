"""Cross-check the lean Regime 4.8 path against Regime 4.7 geometry."""
import numpy as np
import pandas as pd
import pytest

from scripts.regime4_7_flag import flag_attribution_series
from scripts.regime4_8_robustness import (flag_persistence_series,
                                         leading_seven,
                                         rolling_flag_frames,
                                         summarise_persistence)
from src.synth import rotate_basis


def test_partial_eigensolver_matches_full_symmetric_decomposition():
    rng = np.random.default_rng(479)
    x = rng.standard_normal((30, 80))
    covariance = x @ x.T
    values, vectors = leading_seven(covariance)
    all_values, all_vectors = np.linalg.eigh(covariance)
    assert values == pytest.approx(all_values[-7:][::-1], abs=1e-11)
    got = vectors @ vectors.T
    want = all_vectors[:, -7:] @ all_vectors[:, -7:].T
    assert got == pytest.approx(want, abs=1e-11)


def test_persistence_exactly_matches_regime4_7_before_erse_attribution():
    full = np.eye(10)
    frames = np.asarray([
        rotate_basis(rotate_basis(full, 0, 7, a), 3, 8, a / 2)[:, :6]
        for a in (0.0, .05, .11, .18, .26)
    ])
    starts = np.arange(len(frames))
    lean = flag_persistence_series(starts, frames, horizon=1, step=1)
    diagnostics = pd.DataFrame({
        "positive_correlation_fraction": np.ones(len(frames)),
        "all_correlations_positive": np.ones(len(frames), dtype=bool),
        "erse_rotations": np.zeros(len(frames)),
    })
    established = flag_attribution_series(
        starts, frames, frames, diagnostics, horizon=1, step=1)
    for component in lean["component"].unique():
        got = lean.loc[lean["component"] == component, "cosine"].to_numpy()
        want = established.loc[
            established["component"] == component, "cosine"].to_numpy()
        assert np.allclose(got, want, atol=1e-12, equal_nan=True)


def test_moving_market_beta_rotation_is_sharply_reduced_by_residualisation():
    rng = np.random.default_rng(480)
    days, assets = 1400, 18
    factor = rng.standard_normal(days)
    beta_start = np.linspace(.3, 1.8, assets)
    beta_end = beta_start[::-1]
    weights = np.linspace(0, 1, days)
    betas = ((1 - weights)[None, :] * beta_start[:, None]
             + weights[None, :] * beta_end[:, None])
    panel = betas * factor + .12 * rng.standard_normal((assets, days))
    raw_starts, raw, _ = rolling_flag_frames(
        panel, T=250, step=50, remove_market=False)
    residual_starts, residual, _ = rolling_flag_frames(
        panel, T=250, step=50, remove_market=True)
    raw_summary = summarise_persistence(flag_persistence_series(
        raw_starts, raw, horizon=100, step=50)).set_index("component")
    residual_summary = summarise_persistence(flag_persistence_series(
        residual_starts, residual, horizon=100, step=50)).set_index("component")
    assert raw_summary.loc["market_1", "mean_cosine"] > 0.4
    assert residual_summary.loc["market_1", "mean_cosine"] < 0
    assert residual_summary.loc["flag_nested", "mean_cosine"] < (
        0.4 * raw_summary.loc["flag_nested", "mean_cosine"])
