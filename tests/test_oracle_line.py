"""Algebraic and information-boundary tests for the Stage 2 Oracle Line."""
import numpy as np
import pytest

from src.forecast import frozen_flag_losses
from src.oracle_line import (
    align_flag_frame, assemble_covariance, build_oracle_forecasts,
    covariance_to_correlation, ordered_minimum_plane_transport)


def random_frame(rng, n, width):
    frame, _ = np.linalg.qr(rng.normal(size=(n, width)))
    return frame


def test_block_alignment_depends_on_flag_subspaces_not_supplied_bases():
    rng = np.random.default_rng(3)
    current = random_frame(rng, 14, 6)
    target = random_frame(rng, 14, 6)
    transformed = target.copy()
    for start, stop in ((0, 1), (1, 3), (3, 6)):
        rotation = random_frame(rng, stop - start, stop - start)
        transformed[:, start:stop] = target[:, start:stop] @ rotation
    first = align_flag_frame(current, target)
    second = align_flag_frame(current, transformed)
    assert first == pytest.approx(second, abs=1e-10)


def test_ordered_transport_is_orthogonal_and_reaches_every_flag_block():
    rng = np.random.default_rng(5)
    current = random_frame(rng, 18, 6)
    target = align_flag_frame(current, random_frame(rng, 18, 6))
    transport = ordered_minimum_plane_transport(current, target)
    assert transport.T @ transport == pytest.approx(np.eye(18), abs=1e-9)
    assert transport @ current == pytest.approx(target, abs=1e-9)


def test_correlation_and_covariance_assembly_preserve_required_diagonal():
    rng = np.random.default_rng(7)
    loading = rng.normal(size=(8, 8))
    covariance = loading @ loading.T + np.eye(8)
    correlation = covariance_to_correlation(covariance)
    volatilities = np.linspace(.1, .8, 8)
    assembled = assemble_covariance(correlation, volatilities)
    assert np.diag(correlation) == pytest.approx(np.ones(8))
    assert np.diag(assembled) == pytest.approx(volatilities ** 2)
    assert np.linalg.eigvalsh(assembled)[0] > 0


def test_oracle_information_ladder_and_scale_boundary():
    rng = np.random.default_rng(11)
    current = rng.normal(size=(10, 80))
    future = current + .15 * rng.normal(size=current.shape)
    realised = rng.normal(size=(10, 12))
    forecasts = build_oracle_forecasts(current, future, realised, 20)

    names = tuple(forecasts.covariance)
    assert names == (
        "Control - frozen Flag/QIS/EWMA",
        "Oracle 1 - future Flag",
        "Oracle 2 - future Flag and spectrum",
        "Oracle 3 - future rolling correlation",
        "Oracle 4 - future correlation and scale",
    )
    future_flag = forecasts.future_state.flag
    for name in names[1:]:
        assert frozen_flag_losses(
            future_flag, forecasts.input_flag[name])["flag_nested"] \
            == pytest.approx(0.0, abs=1e-10)

    changed_realised = 4.0 * realised
    changed = build_oracle_forecasts(current, future, changed_realised, 20)
    for name in names[:-1]:
        assert changed.covariance[name] == pytest.approx(
            forecasts.covariance[name], abs=1e-10)
    assert changed.covariance[names[-1]] == pytest.approx(
        16.0 * forecasts.covariance[names[-1]], rel=1e-9)


def test_control_cannot_read_future_rolling_window():
    rng = np.random.default_rng(13)
    current = rng.normal(size=(9, 70))
    first_future = rng.normal(size=(9, 70))
    second_future = rng.normal(size=(9, 70))
    realised = rng.normal(size=(9, 10))
    first = build_oracle_forecasts(current, first_future, realised, 15)
    second = build_oracle_forecasts(current, second_future, realised, 15)
    control = "Control - frozen Flag/QIS/EWMA"
    assert first.covariance[control] == pytest.approx(
        second.covariance[control], abs=1e-10)
    assert not np.allclose(
        first.covariance["Oracle 3 - future rolling correlation"],
        second.covariance["Oracle 3 - future rolling correlation"])
