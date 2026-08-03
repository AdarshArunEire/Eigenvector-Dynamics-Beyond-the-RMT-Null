"""Structural tests for the partial-flag statistic used in Regime 4.7."""
import numpy as np
import pytest

from src.flag import (component_logs, flag_component_bases, flag_log,
                      residualise_tuple, stack_nested_tangents, tuple_cosine,
                      tuple_inner)
from src.synth import rotate_basis


def _basis(n, rng):
    return np.linalg.qr(rng.standard_normal((n, n)))[0]


def test_flag_components_have_expected_dimensions_and_nesting():
    U = _basis(12, np.random.default_rng(50))[:, :6]
    parts = flag_component_bases(U)
    assert {name: value.shape[1] for name, value in parts.items()} == {
        "market_1": 1, "block_2_3": 2, "block_4_6": 3,
        "top_3": 3, "top_6": 6,
    }
    assert np.linalg.norm(parts["market_1"].T @ parts["block_2_3"]) < 1e-12
    assert np.linalg.norm(parts["top_3"].T @ parts["block_4_6"]) < 1e-12


def test_nested_flag_log_is_invariant_to_within_block_gauges():
    rng = np.random.default_rng(51)
    U = _basis(14, rng)[:, :6]
    V = rotate_basis(_basis(14, rng), 0, 8, 0.2)[:, :6]
    R2 = _basis(2, rng)
    R3 = _basis(3, rng)
    gauge = np.zeros((6, 6))
    gauge[0, 0] = -1.0
    gauge[1:3, 1:3] = R2
    gauge[3:6, 3:6] = R3
    first = flag_log(U, V)
    second = flag_log(U @ gauge, V @ gauge)
    assert tuple_inner(first, first) == pytest.approx(
        tuple_inner(second, second), abs=1e-10)


def test_rotation_inside_a_flag_block_is_invisible():
    U = np.eye(10)[:, :6]
    full = np.eye(10)
    V = rotate_basis(full, 1, 2, 0.4)[:, :6]
    logs = component_logs(U, V)
    assert all(np.linalg.norm(value) < 1e-10 for value in logs.values())
    assert all(np.linalg.norm(value) < 1e-10 for value in flag_log(U, V))


def test_cross_boundary_rotation_hits_expected_flag_levels():
    U = np.eye(10)[:, :6]
    V = rotate_basis(np.eye(10), 0, 4, 0.2)[:, :6]
    logs = component_logs(U, V)
    assert np.linalg.norm(logs["market_1"]) > 0
    assert np.linalg.norm(logs["block_4_6"]) > 0
    assert np.linalg.norm(logs["top_3"]) > 0
    # Both rotated modes remain inside the top-six span.
    assert np.linalg.norm(logs["top_6"]) < 1e-10


def test_flag_residualisation_has_exact_energy_accounting():
    E = (np.array([[1.0], [0.0]]), np.zeros((2, 3)), np.zeros((2, 6)))
    F = (np.array([[0.0], [2.0]]), np.zeros((2, 3)), np.zeros((2, 6)))
    H = tuple(left + right for left, right in zip(E, F))
    residual, attributed, left = residualise_tuple(H, E)
    assert tuple_inner(residual, residual) == pytest.approx(4.0)
    assert attributed == pytest.approx(0.2)
    assert left == pytest.approx(0.8)
    assert tuple_cosine(residual, F) == pytest.approx(1.0)


def test_stacked_nested_tangent_implements_the_flag_metric():
    rng = np.random.default_rng(52)
    tangent = tuple(rng.standard_normal((8, d)) for d in (1, 3, 6))
    stacked = stack_nested_tangents(tangent)
    assert np.sum(stacked * stacked) == pytest.approx(
        tuple_inner(tangent, tangent))
