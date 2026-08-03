"""Eigenvector Rotation Shrinkage Estimator (ERSE).

Implements Algorithms 1--2 of Liu & Liu, arXiv:2507.01545.  ERSE is a
cross-sectional covariance estimator, not a time-evolution rule: it rotates
pairs of eigenvectors of one sample correlation matrix until every vector has
at least ``delta`` squared projection onto the uniform vector, then recomputes
Rayleigh-quotient eigenvalues in the rotated basis.
"""
import numpy as np

from src.overlap import spectral


def deviation_degrees(vectors):
    """Squared projections of orthonormal columns onto the all-ones vector."""
    Q = np.asarray(vectors, dtype=float)
    if Q.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got shape {Q.shape}")
    return np.sum(Q, axis=0) ** 2


def _wrapped(angle):
    """Return an angle in [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def paired_rotation(q_low, q_high, delta, tol=1e-12):
    """Minimal PER rotation putting the low-deviation vector at ``delta``.

    The inputs must satisfy the feasibility conditions stated in Algorithm 1:
    ``T(q_low) < delta < T(q_high)`` and their total deviation must be at least
    ``2*delta`` so both output vectors can clear the floor.
    """
    low = np.asarray(q_low, dtype=float)
    high = np.asarray(q_high, dtype=float)
    if low.ndim != 1 or high.ndim != 1 or low.shape != high.shape:
        raise ValueError(f"paired vectors need the same 1-D shape, got {low.shape}, {high.shape}")
    if not (0.0 <= delta <= 1.0):
        raise ValueError(f"delta must lie in [0, 1], got {delta}")
    if abs(np.dot(low, low) - 1.0) > 1e-8 or abs(np.dot(high, high) - 1.0) > 1e-8:
        raise ValueError("paired vectors must have unit norm")
    if abs(np.dot(low, high)) > 1e-8:
        raise ValueError("paired vectors must be orthogonal")

    s_low, s_high = float(low.sum()), float(high.sum())
    t_low, t_high = s_low ** 2, s_high ** 2
    if not (t_low < delta + tol and t_high > delta - tol):
        raise ValueError(f"need T(low) < delta < T(high), got {t_low}, {delta}, {t_high}")
    total = t_low + t_high
    if total < 2.0 * delta - tol:
        raise ValueError(
            f"pair cannot put both deviations above delta: total {total} < {2 * delta}")

    # s_low*cos(theta) + s_high*sin(theta) has amplitude sqrt(total).
    # Generate all solutions whose square is delta and choose the smallest
    # absolute rotation, matching Algorithm 1 without its unstable tangent
    # quotient near s_high**2 == delta.
    radius = np.sqrt(total)
    phase = np.arctan2(s_high, s_low)
    target = np.sqrt(delta) / radius if radius > 0 else 0.0
    target = float(np.clip(target, 0.0, 1.0))
    candidates = []
    for signed_target in (target, -target):
        offset = np.arccos(signed_target)
        candidates.extend((_wrapped(phase + offset), _wrapped(phase - offset)))

    feasible = []
    for theta in candidates:
        c, s = np.cos(theta), np.sin(theta)
        new_low_sum = c * s_low + s * s_high
        new_high_sum = -s * s_low + c * s_high
        if (new_low_sum ** 2 >= delta - 1e-9
                and new_high_sum ** 2 >= delta - 1e-9):
            feasible.append(theta)
    if not feasible:
        raise RuntimeError("no feasible PER angle found despite feasible deviation totals")
    theta = min(feasible, key=abs)
    c, s = np.cos(theta), np.sin(theta)
    return c * low + s * high, -s * low + c * high, float(theta)


def rotate_eigenvectors(vectors, delta=0.25, tol=1e-10):
    """Apply ERSE's iterative PER step to a complete orthonormal eigenbasis."""
    Q = np.asarray(vectors, dtype=float).copy()
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"ERSE needs a complete square eigenbasis, got {Q.shape}")
    n = Q.shape[0]
    err = np.linalg.norm(Q.T @ Q - np.eye(n), ord="fro")
    if err > 1e-8:
        raise ValueError(f"vectors are not orthonormal (error {err:.3g})")
    if not (0.0 <= delta <= 1.0):
        raise ValueError(f"delta must lie in [0, 1], got {delta}")

    rotations = []
    while True:
        degree = deviation_degrees(Q)
        low = int(np.argmin(degree))
        if degree[low] >= delta - tol:
            break
        high = int(np.argmax(degree))
        q_low, q_high, theta = paired_rotation(Q[:, low], Q[:, high], delta)
        Q[:, low], Q[:, high] = q_low, q_high
        rotations.append({"low": low, "high": high, "theta": theta})
        if len(rotations) > n - 1:
            raise RuntimeError("ERSE exceeded its n-1 rotation bound")

    # Pairwise rotations preserve this already; QR would alter the deliberately
    # imposed deviations, so validate rather than silently repair.
    err = np.linalg.norm(Q.T @ Q - np.eye(n), ord="fro")
    if err > 1e-8:
        raise RuntimeError(f"ERSE lost orthonormality (error {err:.3g})")
    return Q, rotations


def erse(correlation, delta=0.25):
    """Run ERSE on one sample correlation matrix.

    Returns the estimator and all spectral pieces needed by Regime 4.6.  The
    corrected columns are sorted by their recomputed Rayleigh eigenvalues so
    ``corrected_vectors[:, :P]`` is the estimator's leading P-space.
    """
    R = np.asarray(correlation, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"correlation must be square, got {R.shape}")
    if not np.allclose(R, R.T, atol=1e-10):
        raise ValueError("correlation matrix must be symmetric")

    sample_values, sample_vectors = spectral(R)
    corrected_vectors, rotations = rotate_eigenvectors(sample_vectors, delta)
    corrected_values = np.sum(corrected_vectors * (R @ corrected_vectors), axis=0)
    order = np.argsort(corrected_values)[::-1]
    corrected_values = corrected_values[order]
    corrected_vectors = corrected_vectors[:, order]
    estimate = (corrected_vectors * corrected_values[None, :]) @ corrected_vectors.T

    off_diagonal = R[~np.eye(R.shape[0], dtype=bool)]
    return {
        "estimate": (estimate + estimate.T) / 2.0,
        "sample_values": sample_values,
        "sample_vectors": sample_vectors,
        "corrected_values": corrected_values,
        "corrected_vectors": corrected_vectors,
        "rotations": rotations,
        "deviation_before": deviation_degrees(sample_vectors),
        "deviation_after": deviation_degrees(corrected_vectors),
        "positive_correlation_fraction": float(np.mean(off_diagonal > 0)),
        "all_correlations_positive": bool(np.all(off_diagonal > 0)),
        "minimum_correlation": float(np.min(off_diagonal)),
    }
