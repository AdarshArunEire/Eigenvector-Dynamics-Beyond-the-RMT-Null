"""Partial-flag geometry built from nested Grassmann components.

The project uses the partial flag Flag(N; 1, 3, 6), represented by an
orthonormal N x 6 frame whose nested column spans have dimensions 1, 3 and 6.
The canonical statistic embeds that flag into the product of its nested
Grassmann projectors.  Tangents are therefore tuples of Grassmann tangents at
the same current flag, with each level weighted by inverse dimension so a
six-dimensional component does not dominate merely because it has more
columns.

This is the nested-projector geometry used for Regime 4.7.  It is invariant to
orthogonal changes of basis inside the flag blocks [1], [2:3] and [4:6].
"""
from collections import OrderedDict

import numpy as np

from src.grassmann import grassmann_log


DEFAULT_DIMS = (1, 3, 6)


def validate_flag_frame(frame, dims=DEFAULT_DIMS, name="frame", tol=1e-8):
    """Validate and return an orthonormal frame for a partial flag."""
    U = np.asarray(frame, dtype=float)
    dims = tuple(int(d) for d in dims)
    if U.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {U.shape}")
    if not dims or any(d <= 0 for d in dims) or any(
            left >= right for left, right in zip(dims, dims[1:])):
        raise ValueError(f"dims must be strictly increasing positive integers, got {dims}")
    if dims[-1] > U.shape[1] or U.shape[0] < dims[-1]:
        raise ValueError(f"{name} shape {U.shape} cannot represent dims={dims}")
    U = U[:, :dims[-1]]
    error = np.linalg.norm(U.T @ U - np.eye(dims[-1]), ord="fro")
    if error > tol:
        raise ValueError(f"{name} is not orthonormal (error {error:.3g})")
    return U


def flag_component_bases(frame, dims=DEFAULT_DIMS):
    """Named cumulative and disjoint-block subspaces of a partial flag."""
    U = validate_flag_frame(frame, dims)
    dims = tuple(dims)
    out = OrderedDict()
    out["market_1"] = U[:, :dims[0]]
    for left, right in zip(dims, dims[1:]):
        out[f"block_{left + 1}_{right}"] = U[:, left:right]
    # The already-validated P=3 space and the new outer boundary are retained
    # as cumulative diagnostics.
    for dimension in dims[1:]:
        out[f"top_{dimension}"] = U[:, :dimension]
    return out


def flag_log(base, target, dims=DEFAULT_DIMS):
    """Nested-projector logarithm represented as a tuple of Grassmann logs."""
    U = validate_flag_frame(base, dims, "base")
    V = validate_flag_frame(target, dims, "target")
    if U.shape != V.shape:
        raise ValueError(f"base and target shapes differ: {U.shape} vs {V.shape}")
    return tuple(grassmann_log(U[:, :d], V[:, :d]) for d in dims)


def component_logs(base, target, dims=DEFAULT_DIMS):
    """Grassmann logs for every named cumulative/block flag component."""
    bases = flag_component_bases(base, dims)
    targets = flag_component_bases(target, dims)
    return OrderedDict((name, grassmann_log(bases[name], targets[name]))
                       for name in bases)


def tuple_inner(first, second, dims=DEFAULT_DIMS):
    """Inverse-dimension-weighted inner product of nested tangent tuples."""
    A, B = tuple(first), tuple(second)
    dims = tuple(dims)
    if len(A) != len(dims) or len(B) != len(dims):
        raise ValueError("tangent tuple length must match dims")
    total = 0.0
    for left, right, dimension in zip(A, B, dims):
        left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        if left.shape != right.shape:
            raise ValueError(f"tangent shapes differ: {left.shape} vs {right.shape}")
        total += float(np.sum(left * right)) / dimension
    return total


def tuple_norm(tangent, dims=DEFAULT_DIMS):
    return float(np.sqrt(max(0.0, tuple_inner(tangent, tangent, dims))))


def tuple_cosine(first, second, dims=DEFAULT_DIMS, eps=1e-14):
    denom = tuple_norm(first, dims) * tuple_norm(second, dims)
    if denom <= eps:
        return np.nan
    return float(tuple_inner(first, second, dims) / denom)


def residualise_tuple(tangent, direction, dims=DEFAULT_DIMS, eps=1e-14):
    """Project a complete nested flag tangent off one flag direction."""
    H, E = tuple(tangent), tuple(direction)
    h2, e2 = tuple_inner(H, H, dims), tuple_inner(E, E, dims)
    if e2 <= eps:
        return tuple(np.array(part, copy=True) for part in H), 0.0, 1.0
    coefficient = tuple_inner(H, E, dims) / e2
    residual = tuple(part - coefficient * axis for part, axis in zip(H, E))
    residual_fraction = tuple_inner(residual, residual, dims) / h2 if h2 > eps else np.nan
    attributed = max(0.0, min(1.0, 1.0 - residual_fraction))
    return residual, attributed, residual_fraction


def stack_nested_tangents(tangents, dims=DEFAULT_DIMS):
    """Concatenate nested tangents with weights matching ``tuple_inner``."""
    parts = tuple(tangents)
    if len(parts) != len(tuple(dims)):
        raise ValueError("tangent tuple length must match dims")
    return np.concatenate([np.asarray(part, dtype=float) / np.sqrt(d)
                           for part, d in zip(parts, dims)], axis=-1)
