"""Structural tests for the geometry used by Regime 4.4."""
import numpy as np
import pytest

from src.grassmann import (containment_loss, grassmann_exp, grassmann_log,
                           tangent_cosine)
from scripts.regime4_4_tangent import block_permutation_indices, tangent_series
from scripts.regime4_5_coherence import (coherence_statistics, desynchronise,
                                         sequential_procrustes)


def _basis(n, p, rng):
    return np.linalg.qr(rng.standard_normal((n, p)), mode="reduced")[0]


def test_log_is_horizontal_and_exp_recovers_target_subspace():
    rng = np.random.default_rng(10)
    U, V = _basis(30, 4, rng), _basis(30, 4, rng)
    H = grassmann_log(U, V)
    recovered = grassmann_exp(U, H)
    assert U.T @ H == pytest.approx(np.zeros((4, 4)), abs=1e-10)
    assert containment_loss(V, recovered) == pytest.approx(0.0, abs=1e-10)


def test_log_ignores_basis_rotation_inside_same_subspace():
    rng = np.random.default_rng(11)
    U = _basis(25, 3, rng)
    R = _basis(3, 3, rng)
    assert np.linalg.norm(grassmann_log(U, U @ R)) < 1e-10


def test_constant_velocity_extrapolates_a_geodesic():
    rng = np.random.default_rng(12)
    U0 = _basis(35, 3, rng)
    ambient = rng.standard_normal(U0.shape)
    ambient -= U0 @ (U0.T @ ambient)
    H0 = ambient / np.linalg.norm(ambient, ord="fro") * 0.25
    U1 = grassmann_exp(U0, H0)
    U2 = grassmann_exp(U0, 2 * H0)

    incoming_at_u1 = -grassmann_log(U1, U0)
    predicted = grassmann_exp(U1, incoming_at_u1)
    assert containment_loss(U2, predicted) == pytest.approx(0.0, abs=1e-10)


def test_tangent_cosine_and_zero_guard():
    A = np.arange(12, dtype=float).reshape(4, 3)
    assert tangent_cosine(A, A) == pytest.approx(1.0)
    assert tangent_cosine(A, -A) == pytest.approx(-1.0)
    assert np.isnan(tangent_cosine(A, np.zeros_like(A)))


def test_containment_loss_is_basis_invariant_and_supports_q_larger_than_p():
    rng = np.random.default_rng(13)
    V = _basis(30, 7, rng)
    U = V[:, :3]
    R3, R7 = _basis(3, 3, rng), _basis(7, 7, rng)
    assert containment_loss(U @ R3, V @ R7) == pytest.approx(0.0, abs=1e-10)
    assert containment_loss(U, V, normalise=True) == pytest.approx(0.0, abs=1e-10)


def test_regime44_recovers_perfect_direction_and_constant_velocity():
    rng = np.random.default_rng(14)
    U0 = _basis(30, 3, rng)
    H = rng.standard_normal(U0.shape)
    H -= U0 @ (U0.T @ H)
    H *= 0.15 / np.linalg.norm(H, ord="fro")
    bases = np.asarray([grassmann_exp(U0, k * H) for k in range(3)])
    row = tangent_series(np.arange(3), bases, horizon=1, step=1).iloc[0]
    assert row["cosine"] == pytest.approx(1.0, abs=1e-10)
    assert row["constant_velocity_loss"] == pytest.approx(0.0, abs=1e-10)
    assert row["constant_velocity_skill"] == pytest.approx(1.0, abs=1e-10)


def test_block_permutation_keeps_every_day_once():
    indices = block_permutation_indices(23, 5, np.random.default_rng(15))
    assert sorted(indices.tolist()) == list(range(23))
    # Every full block remains consecutive and internally ordered.
    positions = {int(v): i for i, v in enumerate(indices)}
    for start in range(0, 20, 5):
        assert [positions[j + 1] - positions[j] for j in range(start, start + 4)] == [1] * 4


def test_sequential_procrustes_removes_internal_basis_rotations():
    rng = np.random.default_rng(16)
    U = _basis(25, 3, rng)
    path = np.asarray([U @ _basis(3, 3, rng) for _ in range(8)])
    aligned = sequential_procrustes(path)
    for current in aligned[1:]:
        assert np.linalg.norm(current - aligned[0], ord="fro") < 1e-10


def test_coherence_spike_falls_when_asset_histories_are_desynchronised():
    rng = np.random.default_rng(17)
    n_time, n, p = 120, 30, 2
    U = np.eye(n)[:, :p]
    bases = np.repeat(U[None, :, :], n_time, axis=0)
    common_time = rng.standard_normal((n_time, p))
    loadings = rng.standard_normal(n)
    loadings[:p] = 0.0  # horizontal at U
    H = common_time[:, None, :] * loadings[None, :, None]
    H += 0.08 * rng.standard_normal(H.shape)
    H[:, :p, :] = 0.0
    observed = coherence_statistics(H)
    shuffled = coherence_statistics(desynchronise(H, bases, rng))
    assert observed["leading_share"] > 0.8
    assert observed["leading_share"] > 2 * shuffled["leading_share"]
