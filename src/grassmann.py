"""Canonical Grassmann geometry for subspace dynamics.

A point on Gr(N, P) is represented by any N x P orthonormal basis ``U``.
All public quantities here are invariant to replacing that basis by ``U @ R``
for an orthogonal P x P matrix R.
"""
import numpy as np


def _orthonormal_basis(U, name="U", tol=1e-8):
    U = np.asarray(U, dtype=float)
    if U.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {U.shape}")
    if U.shape[0] < U.shape[1]:
        raise ValueError(f"{name} needs N >= P, got shape {U.shape}")
    err = np.linalg.norm(U.T @ U - np.eye(U.shape[1]), ord="fro")
    if err > tol:
        raise ValueError(f"{name} is not orthonormal (error {err:.3g})")
    return U


def grassmann_log(base, target):
    """Canonical logarithm from ``base`` to ``target`` on Gr(N, P).

    Returns an N x P horizontal tangent matrix H at ``base``. Its singular
    values are the principal angles, and ``grassmann_exp(base, H)`` spans the
    target subspace. The construction uses principal vectors rather than an
    inverse of ``base.T @ target``, so it remains stable near 90-degree angles.
    """
    U = _orthonormal_basis(base, "base")
    V = _orthonormal_basis(target, "target")
    if U.shape != V.shape:
        raise ValueError(f"base and target shapes differ: {U.shape} vs {V.shape}")

    left, cosines, right_t = np.linalg.svd(U.T @ V, full_matrices=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    right = right_t.T
    angles = np.arccos(cosines)
    sines = np.sin(angles)

    # V right and U left are corresponding principal-vector bases. Multiplying
    # their orthogonal residual by theta/sin(theta) gives the log without ever
    # forming tan(theta) or an ill-conditioned overlap inverse.
    residual = V @ right - (U @ left) * cosines[None, :]
    scale = np.ones_like(angles)
    active = np.abs(angles) > 1e-10
    scale[active] = angles[active] / sines[active]
    H = (residual * scale[None, :]) @ left.T

    # Remove roundoff in the vertical component. Mathematically this is zero.
    return H - U @ (U.T @ H)


def grassmann_exp(base, tangent):
    """Canonical exponential of a horizontal tangent on Gr(N, P)."""
    U = _orthonormal_basis(base, "base")
    H = np.asarray(tangent, dtype=float)
    if H.shape != U.shape:
        raise ValueError(f"tangent shape {H.shape} does not match base {U.shape}")
    vertical = np.linalg.norm(U.T @ H, ord="fro")
    if vertical > 1e-7:
        raise ValueError(f"tangent is not horizontal (error {vertical:.3g})")

    direction, angles, right_t = np.linalg.svd(H, full_matrices=False)
    right = right_t.T
    out = ((U @ right) * np.cos(angles)[None, :]
           + direction * np.sin(angles)[None, :]) @ right_t
    # The formula is orthonormal in exact arithmetic. QR only repairs roundoff;
    # its sign convention changes the representative, never the subspace.
    return np.linalg.qr(out, mode="reduced")[0]


def tangent_cosine(first, second, eps=1e-14):
    """Frobenius cosine of two tangent matrices at the same base point."""
    A, B = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"tangent shapes differ: {A.shape} vs {B.shape}")
    denom = np.linalg.norm(A, ord="fro") * np.linalg.norm(B, ord="fro")
    if denom <= eps:
        return np.nan
    return float(np.sum(A * B) / denom)


def containment_loss(target, prediction, normalise=False):
    """Projector loss for a target P-space inside a predicted Q-space.

    Returns P - ||target.T @ prediction||_F^2 = sum sin(theta_i)^2.
    Set ``normalise`` for a [0, 1] mean loss across the P target directions.
    """
    U = _orthonormal_basis(target, "target")
    V = _orthonormal_basis(prediction, "prediction")
    if U.shape[0] != V.shape[0] or U.shape[1] > V.shape[1]:
        raise ValueError(f"need equal N and P <= Q, got {U.shape} and {V.shape}")
    loss = float(U.shape[1] - np.linalg.norm(U.T @ V, ord="fro") ** 2)
    loss = max(0.0, loss)  # roundoff can produce -1e-15 at exact containment
    return loss / U.shape[1] if normalise else loss
