import numpy as np
import pytest

from scripts.regime4_9_deletion_attribution import (
    deletion_attribution_series, flag_histories, transport_objects,
    volatility_matched_block_indices)


def _frame(angle=0.0):
    frame = np.eye(8)[:, :6]
    frame[:, 0] = 0.0
    frame[0, 0] = np.cos(angle)
    frame[6, 0] = np.sin(angle)
    return frame


def test_ordered_transport_preserves_tangent_norm_and_horizontality():
    source = _frame(0.0)
    target = _frame(0.3)
    tangent = np.zeros((8, 1))
    tangent[7, 0] = 0.4
    aligned, [transported] = transport_objects(source, target, [tangent])
    assert np.linalg.norm(transported) == pytest.approx(np.linalg.norm(tangent))
    assert aligned[:, :1].T @ transported == pytest.approx(
        np.zeros((1, 1)), abs=1e-12)


def test_residual_series_recovers_repeated_incoming_direction():
    starts = np.arange(3)
    base = _frame(0.0)
    moved = _frame(0.2)
    full = np.asarray([base, moved, moved])
    retained = np.asarray([base, base, base])
    series = deletion_attribution_series(
        starts, full, retained, horizon=1, step=1)
    nested = series.loc[series["component"] == "flag_nested"].iloc[0]
    assert nested["addition_cosine"] == pytest.approx(1.0, abs=1e-10)


def test_residual_statistic_is_invariant_to_flag_block_bases():
    starts = np.arange(3)
    base = _frame(0.0)
    moved = _frame(0.2)
    full = np.asarray([base, moved, moved])
    retained = np.asarray([base, base, base])
    original = deletion_attribution_series(
        starts, full, retained, horizon=1, step=1)

    def rotate_blocks(frame, angle):
        output = frame.copy()
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        output[:, 1:3] = output[:, 1:3] @ rotation
        block = np.eye(3)
        block[:2, :2] = rotation.T
        output[:, 3:6] = output[:, 3:6] @ block
        return output

    rotated_full = np.asarray([
        rotate_blocks(frame, angle)
        for frame, angle in zip(full, (0.1, -0.3, 0.4))
    ])
    rotated_retained = np.asarray([
        rotate_blocks(frame, angle)
        for frame, angle in zip(retained, (-0.2, 0.25, -0.1))
    ])
    transformed = deletion_attribution_series(
        starts, rotated_full, rotated_retained, horizon=1, step=1)
    assert np.allclose(
        transformed["addition_cosine"].to_numpy(dtype=float),
        original["addition_cosine"].to_numpy(dtype=float),
        atol=1e-10, equal_nan=True)


def test_retained_history_ignores_observations_known_to_expire():
    rng = np.random.default_rng(14)
    panel = rng.normal(size=(9, 70))
    changed = panel.copy()
    changed[:, :10] = rng.normal(scale=100, size=(9, 10))
    _, _, first = flag_histories(panel, T=40, step=10, horizon=10)
    _, _, second = flag_histories(changed, T=40, step=10, horizon=10)
    first_projector = first[0] @ first[0].T
    second_projector = second[0] @ second[0].T
    assert second_projector == pytest.approx(first_projector, abs=1e-10)


def test_volatility_matched_indices_are_an_exact_block_permutation():
    panel = np.arange(2 * 23, dtype=float).reshape(2, 23)
    indices = volatility_matched_block_indices(
        panel, block_size=5, bins=2, rng=np.random.default_rng(15))
    assert np.sort(indices).tolist() == list(range(23))
    # The unique final three-column block cannot be exchanged with a full block.
    assert indices[-3:].tolist() == [20, 21, 22]


def test_volatility_matching_keeps_donor_blocks_in_the_same_stratum():
    panel = np.array([[1., 1., 2., 2., 10., 10., 20., 20.]])
    indices = volatility_matched_block_indices(
        panel, block_size=2, bins=2, rng=np.random.default_rng(16))
    assert set(indices[:4]).issubset(set(range(4)))
    assert set(indices[4:]).issubset(set(range(4, 8)))
