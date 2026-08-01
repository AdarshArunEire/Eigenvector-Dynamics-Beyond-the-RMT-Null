"""Structural properties of the overlap instrument. No statistics here."""
import numpy as np
import pytest

from src.overlap import (spectral, sample_covariance, overlap_matrix,
                         principal_cosines, subspace_distance)


def _basis(N, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    return Q


def test_containment_gives_zero_distance():
    rng = np.random.default_rng(0)
    Q = _basis(30, rng)
    U, V = Q[:, 2:5], Q[:, 0:8]          # span(U) sits inside span(V)
    assert subspace_distance(U, V) == pytest.approx(0.0, abs=1e-10)
    assert principal_cosines(U, V) == pytest.approx(np.ones(3), abs=1e-10)


def test_orthogonal_subspaces_are_maximally_distant():
    rng = np.random.default_rng(1)
    Q = _basis(30, rng)
    U, V = Q[:, 0:3], Q[:, 10:16]
    # Cosines land at machine epsilon rather than exact zero, so D is large but
    # finite: -log(1e-16) is about 37, not infinity.
    assert np.all(principal_cosines(U, V) < 1e-10)
    assert subspace_distance(U, V) > 20.0


def test_cosines_are_bounded_and_shaped():
    rng = np.random.default_rng(2)
    Qa, Qb = _basis(40, rng), _basis(40, rng)
    s = principal_cosines(Qa[:, :4], Qb[:, :9])
    assert s.shape == (4,)
    assert np.all(s >= 0) and np.all(s <= 1)
    assert np.all(np.diff(s) <= 1e-12)   # descending


def test_distance_equals_log_det_form():
    rng = np.random.default_rng(3)
    Qa, Qb = _basis(25, rng), _basis(25, rng)
    U, V = Qa[:, :3], Qb[:, :7]
    G = overlap_matrix(U, V)
    assert G.shape == (7, 3)
    ref = -np.log(np.linalg.det(G.T @ G)) / (2 * U.shape[1])
    assert subspace_distance(U, V) == pytest.approx(ref, rel=1e-9)


def test_eigh_ordering_is_descending():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((20, 20))
    lam, Q = spectral(A @ A.T)
    assert np.all(np.diff(lam) <= 1e-9)
    assert Q.T @ Q == pytest.approx(np.eye(20), abs=1e-10)


def test_sample_covariance_orientation():
    rng = np.random.default_rng(5)
    R = rng.standard_normal((12, 500))
    assert sample_covariance(R).shape == (12, 12)


def test_p_greater_than_q_is_rejected():
    rng = np.random.default_rng(6)
    Q = _basis(20, rng)
    with pytest.raises(ValueError):
        overlap_matrix(Q[:, :8], Q[:, :3])
