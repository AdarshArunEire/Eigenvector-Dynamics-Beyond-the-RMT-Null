"""Tests for the first-order coupling algebra.

These check the two claims the whole of Model 4.1 rests on: that the metric
sees a correction only through ``U_perp^T G U_6``, and that ``score_gradient``
is the correct derivative of the realised capture at that point.  If either
fails the model is fitting something the score cannot reward.
"""
import numpy as np
import pytest

from src.capture import variance_captured
from src.coupling import (RANK, causal_standardise, combine, corrected_frame,
                          descending_spectrum, eigengap_matrix,
                          feature_collinearity, gradient_beta,
                          realised_visible_target, ridge_beta,
                          rie_correction_is_invisible, score_gradient,
                          visible_block)


def _spd(n, rng, spread=6.0):
    """A well-separated SPD matrix: a genuine 6/7 gap, no accidental ties."""
    basis = np.linalg.qr(rng.standard_normal((n, n)))[0]
    values = np.exp(np.linspace(np.log(spread), 0.0, n))
    return (basis * values) @ basis.T


def _symmetric(n, rng):
    raw = rng.standard_normal((n, n))
    return (raw + raw.T) / 2


# --------------------------------------------------------------------------
# the invisible blocks
# --------------------------------------------------------------------------

def test_diagonal_correction_in_the_eigenbasis_cannot_move_the_frame():
    """Eigenvalue-only edits leave the leading eigenspace exactly fixed."""
    rng = np.random.default_rng(11)
    covariance = _spd(30, rng)
    values, vectors = descending_spectrum(covariance)
    # An arbitrary, large, order-preserving eigenvalue map.
    shrunk = (vectors * (0.3 * values + 0.7 * values.mean())) @ vectors.T
    assert rie_correction_is_invisible(covariance, shrunk)
    base = descending_spectrum(covariance)[1][:, :RANK]
    moved = descending_spectrum(shrunk)[1][:, :RANK]
    assert np.linalg.norm(np.abs(base.T @ moved) - np.eye(RANK)) < 1e-8


def test_ledoit_wolf_style_linear_shrinkage_scores_exactly_zero():
    """The whole rotationally-invariant class is pinned at zero, not merely small."""
    rng = np.random.default_rng(12)
    covariance = _spd(24, rng)
    identity_target = np.trace(covariance) / covariance.shape[0] * np.eye(covariance.shape[0])
    realised = rng.standard_normal((24, 42))
    base = descending_spectrum(covariance)[1][:, :RANK]
    for intensity in (0.05, 0.3, 0.8):
        shrunk = (1 - intensity) * covariance + intensity * identity_target
        assert rie_correction_is_invisible(covariance, shrunk)
        moved = descending_spectrum(shrunk)[1][:, :RANK]
        assert (abs(variance_captured(moved, realised)["capture_6"]
                    - variance_captured(base, realised)["capture_6"]) < 1e-12)


def test_within_block_corrections_are_invisible_to_the_score():
    """Edits confined to the top-6 or complement blocks do not move the span.

    The individual columns *do* rotate -- an edit inside the top-6 block
    reshuffles the six eigenvalues and so remixes their eigenvectors -- but the
    projector, which is the only thing the score contracts against, is fixed to
    machine precision.  Comparing bases rather than projectors would fail here
    for a reason the metric cannot see.
    """
    rng = np.random.default_rng(13)
    covariance = _spd(28, rng)
    values, vectors = descending_spectrum(covariance)
    inner = _symmetric(RANK, rng)
    outer = _symmetric(28 - RANK, rng)
    correction = (vectors[:, :RANK] @ inner @ vectors[:, :RANK].T
                  + vectors[:, RANK:] @ outer @ vectors[:, RANK:].T)
    realised = rng.standard_normal((28, 42))
    base = vectors[:, :RANK]
    moved = descending_spectrum(covariance + 1e-3 * correction)[1][:, :RANK]
    assert np.linalg.norm(base @ base.T - moved @ moved.T) < 1e-9
    assert (abs(variance_captured(moved, realised)["capture_6"]
                - variance_captured(base, realised)["capture_6"]) < 1e-12)
    # ...and the columns really did move, so the projector test has content.
    assert np.linalg.norm(np.abs(base.T @ moved) - np.eye(RANK)) > 1e-6


# --------------------------------------------------------------------------
# the gradient is the actual derivative
# --------------------------------------------------------------------------

def test_score_gradient_matches_a_finite_difference_of_realised_capture():
    """First-order prediction converges at rate eps, i.e. the error is O(eps^2)."""
    rng = np.random.default_rng(14)
    n = 26
    covariance = _spd(n, rng)
    realised = rng.standard_normal((n, 42))
    values, vectors = descending_spectrum(covariance)
    gaps, _ = eigengap_matrix(values)
    block = rng.standard_normal((n - RANK, RANK))
    block /= np.linalg.norm(block)
    target = realised_visible_target(realised, vectors)
    predicted = score_gradient(block, gaps, target)

    errors = []
    for epsilon in (1e-3, 5e-4, 2.5e-4):
        frame = corrected_frame(values, vectors, block, epsilon)
        realised_change = (variance_captured(frame, realised)["capture_6"]
                           - variance_captured(vectors[:, :RANK], realised)["capture_6"])
        errors.append(abs(realised_change - epsilon * predicted) / epsilon)
    # Relative error must fall roughly in proportion to eps.
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.05 * max(abs(predicted), 1e-12) + 1e-9


def test_score_gradient_is_invariant_to_column_sign_flips():
    """Eigenvector signs are arbitrary; the fitted objective must not see them."""
    rng = np.random.default_rng(15)
    n = 22
    covariance = _spd(n, rng)
    realised = rng.standard_normal((n, 40))
    values, vectors = descending_spectrum(covariance)
    gaps, _ = eigengap_matrix(values)
    signs = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    flipped = vectors * signs[None, :]

    arbitrary = _symmetric(n, rng)
    left = score_gradient(visible_block(arbitrary, vectors), gaps,
                          realised_visible_target(realised, vectors))
    right = score_gradient(visible_block(arbitrary, flipped), gaps,
                           realised_visible_target(realised, flipped))
    assert abs(left - right) < 1e-10 * max(1.0, abs(left))


def test_zero_epsilon_returns_the_base_frame_unchanged():
    rng = np.random.default_rng(16)
    covariance = _spd(20, rng)
    values, vectors = descending_spectrum(covariance)
    block = rng.standard_normal((20 - RANK, RANK))
    assert np.array_equal(corrected_frame(values, vectors, block, 0.0),
                          vectors[:, :RANK])


def test_corrected_frame_is_orthonormal_and_spans_a_moved_subspace():
    rng = np.random.default_rng(17)
    n = 25
    covariance = _spd(n, rng)
    values, vectors = descending_spectrum(covariance)
    block = rng.standard_normal((n - RANK, RANK))
    block /= np.linalg.norm(block)
    frame = corrected_frame(values, vectors, block, 0.05)
    assert np.linalg.norm(frame.T @ frame - np.eye(RANK)) < 1e-9
    assert np.linalg.norm(np.abs(frame.T @ vectors[:, :RANK]) - np.eye(RANK)) > 1e-6


# --------------------------------------------------------------------------
# labels, gaps, causality
# --------------------------------------------------------------------------

def test_realised_target_equals_the_dense_trace_normalised_construction():
    rng = np.random.default_rng(18)
    n = 18
    realised = rng.standard_normal((n, 30))
    vectors = np.linalg.qr(rng.standard_normal((n, n)))[0]
    dense = realised @ realised.T / float(np.linalg.norm(realised, "fro") ** 2)
    assert np.allclose(realised_visible_target(realised, vectors),
                       visible_block(dense, vectors), atol=1e-12)


def test_eigengap_floor_binds_and_is_reported():
    values = np.array([3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.5 - 1e-9, 1.0, 0.5])
    gaps, bound = eigengap_matrix(values, floor=1e-6)
    assert gaps.min() == pytest.approx(1e-6)
    assert 0.0 < bound < 1.0
    unbound, none = eigengap_matrix(np.arange(12, 0, -1, dtype=float), floor=1e-6)
    assert none == 0.0 and unbound.min() > 1e-6


def test_eigengap_matrix_rejects_ascending_values():
    with pytest.raises(ValueError, match="descending"):
        eigengap_matrix(np.arange(12, dtype=float))


def test_causal_standardise_never_reads_the_future():
    rng = np.random.default_rng(19)
    values = rng.standard_normal(80)
    baseline = causal_standardise(values, minimum=20)
    perturbed = values.copy()
    perturbed[60:] += 10.0
    assert np.allclose(baseline[:60], causal_standardise(perturbed, minimum=20)[:60])
    assert np.all(baseline[:20] == 0.0)


# --------------------------------------------------------------------------
# the two beta estimators
# --------------------------------------------------------------------------

def _planted(rng, n_origins=40, n=20, n_features=3, truth=(1.0, 0.0, 0.0),
             noise=0.1, duplicate=False):
    designs, targets, gaps = [], [], []
    for _ in range(n_origins):
        design = rng.standard_normal((n_features, n - RANK, RANK))
        if duplicate:
            design[1] = design[0] + 0.02 * rng.standard_normal(design[0].shape)
        gap = np.full((n - RANK, RANK), 1.0)
        signal = combine(design, np.asarray(truth, dtype=float))
        designs.append(design)
        gaps.append(gap)
        targets.append(signal + noise * rng.standard_normal(signal.shape))
    return designs, targets, gaps


def test_gradient_beta_recovers_a_planted_single_feature():
    rng = np.random.default_rng(20)
    designs, targets, gaps = _planted(rng, truth=(1.0, 0.0, 0.0))
    beta = gradient_beta(designs, targets, gaps)
    assert abs(np.linalg.norm(beta) - 1.0) < 1e-12
    assert beta[0] > 0.95


def test_ridge_beta_recovers_a_planted_mixture():
    rng = np.random.default_rng(21)
    truth = np.array([0.6, -0.8, 0.0])
    designs, targets, gaps = _planted(rng, truth=truth, noise=0.05)
    beta = ridge_beta(designs, targets, gaps)
    assert np.linalg.norm(beta - truth / np.linalg.norm(truth)) < 0.1


def test_ridge_discounts_a_duplicated_feature_where_the_gradient_does_not():
    """The exact failure mode that motivates fitting both estimators.

    Feature 1 is a near-copy of feature 0 and the truth loads only feature 0.
    The gradient fit, being linear in beta, has no Gram matrix and loads the
    copy just as heavily; ridge splits or suppresses it.
    """
    rng = np.random.default_rng(22)
    designs, targets, gaps = _planted(rng, truth=(1.0, 0.0, 0.0),
                                      noise=0.05, duplicate=True)
    gradient = gradient_beta(designs, targets, gaps)
    ridge = ridge_beta(designs, targets, gaps)
    assert gradient[1] > 0.6 * gradient[0]          # copy loaded almost fully
    assert abs(ridge[1]) < 0.9 * abs(ridge[0])      # copy discounted
    collinearity = feature_collinearity(designs, gaps)
    assert collinearity[0, 1] > 0.95
    assert abs(collinearity[0, 2]) < 0.3


def test_fit_inputs_are_validated():
    rng = np.random.default_rng(23)
    designs, targets, gaps = _planted(rng, n_origins=4)
    with pytest.raises(ValueError, match="no training origins"):
        gradient_beta([], [], [])
    with pytest.raises(ValueError, match="different lengths"):
        ridge_beta(designs, targets[:2], gaps)
    with pytest.raises(ValueError, match="conformable"):
        gradient_beta(designs, [target[:-1] for target in targets], gaps)


def test_combine_rejects_a_mismatched_beta():
    rng = np.random.default_rng(24)
    design = rng.standard_normal((3, 14, RANK))
    with pytest.raises(ValueError, match="design has 3 features"):
        combine(design, np.ones(2))
