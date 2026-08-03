"""Structural tests for ERSE and the Regime 4.6 attribution statistics."""
import numpy as np
import pytest

from scripts.regime4_6_erse import (residualise_tangent,
                                    top_cross_covariance_share)
from src.erse import (deviation_degrees, erse, paired_rotation,
                      rotate_eigenvectors)
from src.synth import rotate_basis


def _basis(n, rng):
    return np.linalg.qr(rng.standard_normal((n, n)))[0]


def test_deviation_degrees_sum_to_ambient_dimension():
    Q = _basis(12, np.random.default_rng(40))
    assert deviation_degrees(Q).sum() == pytest.approx(12.0, abs=1e-10)


def test_per_preserves_pair_and_sets_both_above_floor():
    n = 8
    uniform = np.ones(n) / np.sqrt(n)
    rng = np.random.default_rng(41)
    raw = rng.standard_normal(n)
    low = raw - uniform * np.dot(uniform, raw)
    low /= np.linalg.norm(low)
    new_low, new_high, theta = paired_rotation(low, uniform, delta=0.25)
    pair = np.column_stack([new_low, new_high])
    assert pair.T @ pair == pytest.approx(np.eye(2), abs=1e-10)
    assert np.all(deviation_degrees(pair) >= 0.25 - 1e-10)
    assert abs(theta) < np.pi / 2


def test_erse_rotation_clears_floor_without_losing_orthogonality():
    Q = _basis(20, np.random.default_rng(42))
    corrected, rotations = rotate_eigenvectors(Q, delta=0.25)
    assert corrected.T @ corrected == pytest.approx(np.eye(20), abs=1e-10)
    assert deviation_degrees(corrected).min() >= 0.25 - 1e-10
    assert len(rotations) <= 19


def test_erse_rotation_is_invariant_to_eigenvector_signs():
    rng = np.random.default_rng(420)
    Q = _basis(14, rng)
    signs = rng.choice([-1.0, 1.0], size=14)
    first, _ = rotate_eigenvectors(Q, delta=0.25)
    second, _ = rotate_eigenvectors(Q * signs[None, :], delta=0.25)
    assert first[:, :3] @ first[:, :3].T == pytest.approx(
        second[:, :3] @ second[:, :3].T, abs=1e-10)


def test_pairwise_rayleigh_values_are_mutual_linear_shrinkage():
    n, theta = 6, 0.30
    lam = np.array([10.0, 2.0, 1.5, 1.2, 0.8, 0.5])
    Q = np.eye(n)
    rotated = rotate_basis(Q, 0, 1, theta)
    C = Q @ np.diag(lam) @ Q.T
    values = np.sum(rotated * (C @ rotated), axis=0)
    c2, s2 = np.cos(theta) ** 2, np.sin(theta) ** 2
    assert values[0] == pytest.approx(c2 * lam[0] + s2 * lam[1])
    assert values[1] == pytest.approx(s2 * lam[0] + c2 * lam[1])
    assert values[:2].sum() == pytest.approx(lam[:2].sum())


def test_erse_estimator_preserves_trace_and_reports_assumption():
    n = 10
    # Equicorrelation is strictly positive and has a Perron market direction.
    R = np.full((n, n), 0.4)
    np.fill_diagonal(R, 1.0)
    out = erse(R, delta=0.25)
    assert out["all_correlations_positive"]
    assert out["positive_correlation_fraction"] == 1.0
    assert np.trace(out["estimate"]) == pytest.approx(np.trace(R), abs=1e-10)
    assert out["corrected_vectors"].T @ out["corrected_vectors"] == pytest.approx(
        np.eye(n), abs=1e-10)


def test_residualisation_reports_attributed_energy_without_erasing_remainder():
    E = np.zeros((5, 2))
    F = np.zeros((5, 2))
    E[2, 0], F[3, 1] = 2.0, 3.0
    H = E + F
    residual, attributed, left = residualise_tangent(H, E)
    assert residual == pytest.approx(F)
    assert attributed == pytest.approx(4.0 / 13.0)
    assert left == pytest.approx(9.0 / 13.0)


def test_top_cross_share_is_zero_for_eigenvalue_only_change():
    U = np.eye(5)[:, :2]
    C0 = np.diag([5.0, 3.0, 2.0, 1.0, 0.5])
    C1 = np.diag([4.5, 3.2, 2.1, 1.1, 0.6])
    cross, transition, share = top_cross_covariance_share(U, C0, C1)
    assert transition > 0
    assert cross == pytest.approx(0.0, abs=1e-12)
    assert share == pytest.approx(0.0, abs=1e-12)


def test_top_cross_share_detects_genuine_rotation_across_boundary():
    Q = np.eye(5)
    lam = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
    C0 = Q @ np.diag(lam) @ Q.T
    rotated = rotate_basis(Q, 0, 3, 0.20)
    C1 = rotated @ np.diag(lam) @ rotated.T
    cross, transition, share = top_cross_covariance_share(Q[:, :2], C0, C1)
    assert cross > 0
    assert transition > 0
    assert 0 < share <= 1
