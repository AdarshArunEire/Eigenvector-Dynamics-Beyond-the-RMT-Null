"""Subspace overlap between two eigenbases.

Implements the instrument from Allez & Bouchaud, arXiv:1108.4258, in matrix
notation. The singular values of G = V.T @ U are the cosines of the principal
angles between span(U) and span(V).
"""
import numpy as np

_EPS = 1e-300


def spectral(S):
    """Eigen-decomposition of a real symmetric matrix, eigenvalues descending.

    Uses eigh, not eig: eig does not assume symmetry, returns complex dtype and
    gives no orthogonality guarantee.
    """
    lam, Q = np.linalg.eigh(S)
    order = np.argsort(lam)[::-1]
    return lam[order], Q[:, order]


def sample_covariance(R, demean=False):
    """Sample covariance from an N x T return panel.

    Note the orientation: R is N x T, so this is R @ R.T / T, an N x N matrix.
    (The paper prints R.T @ R, which would be T x T.)
    """
    if R.ndim != 2:
        raise ValueError(f"R must be 2-D (N x T), got shape {R.shape}")
    if demean:
        R = R - R.mean(axis=1, keepdims=True)
    return (R @ R.T) / R.shape[1]


def overlap_matrix(U, V):
    """G = V.T @ U, shape (Q, P). U is N x P (inner), V is N x Q (outer)."""
    if U.shape[0] != V.shape[0]:
        raise ValueError(f"ambient dims differ: {U.shape[0]} vs {V.shape[0]}")
    if U.shape[1] > V.shape[1]:
        raise ValueError(f"need P <= Q, got P={U.shape[1]} Q={V.shape[1]}")
    return V.T @ U


def principal_cosines(U, V):
    """Cosines of the principal angles, descending. Length P."""
    s = np.linalg.svd(overlap_matrix(U, V), compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def principal_angles(U, V):
    """Principal angles in radians, ascending."""
    return np.arccos(principal_cosines(U, V))


def subspace_distance(U, V):
    """D = -(1/2P) ln det(G.T G) = -(1/P) sum_k ln sigma_k.

    Zero iff span(U) is contained in span(V). Not a metric: asymmetric in its
    arguments, no triangle inequality.
    """
    s = principal_cosines(U, V)
    return float(-np.mean(np.log(np.maximum(s, _EPS))))


def top_block(S, k):
    """Top-k eigenvectors of a symmetric matrix, as an N x k orthonormal slice."""
    _, Q = spectral(S)
    return Q[:, :k]
