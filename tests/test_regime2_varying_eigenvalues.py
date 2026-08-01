"""Stage 1, regime 2: fixed eigenvectors, time-varying eigenvalues.

The basis is nailed down for all time and only the magnitudes move. There is no
rotation in this world, so any excess the instrument reports is manufactured.

This is not a repeat of regime 1 at a different setting. Regime 1 handed the
same spectrum to both windows, so Eq (10) collapsed to 2 x Eq (7) and its
two-spectrum structure was never executed once. More importantly, regime 1
could only rule out the instrument inventing rotation out of nothing. It could
not rule out the instrument mistaking a change in magnitude for a change in
direction, because nothing was changing. That confound is what this file
exists to close: eigenvalues demonstrably do move in real data, and if that
motion leaked into D then every downstream result would be an artefact of
volatility rather than a statement about correlation structure.

Correctness, not science. Nothing here is a finding.
"""
import numpy as np
import pytest

from src.overlap import spectral, sample_covariance, subspace_distance
from src.null_rmt import (d_null_sample_vs_true, d_null_two_samples,
                          eigenvalue_variogram_null)
from src.synth import (spd_from_spectrum, factor_spectrum, perturb_top,
                       returns_fixed_basis, ramp_path)

N, P, Q, T = 40, 3, 6, 2000
N_TOP, SIGMA = 3, 0.06     # 0/100000 crossings on this spectrum; see perturb_top
TRIALS = 400
TOL = 0.05                 # block-to-block spread of the ratio is ~1% at n=400


@pytest.fixture(scope="module")
def world():
    """Regime 1's spectrum and a basis that never changes again."""
    lam = factor_spectrum(N, [25.0, 10.0, 6.0, 4.0, 3.0, 2.2])
    _, B = spd_from_spectrum(lam, np.random.default_rng(20260801))
    return lam, B


# --------------------------------------------------------------------------
# Structural properties of Eq (10). Regime 1 exercised none of these, because
# it only ever called the formula with lam_s and lam_t identical.
# --------------------------------------------------------------------------

def test_eq10_reduces_to_twice_eq7_when_spectra_match(world):
    """The paper's 'multiplied by a factor 2' remark, as an executable claim."""
    lam, _ = world
    assert d_null_two_samples(lam, lam, P, Q, T) == pytest.approx(
        2.0 * d_null_sample_vs_true(lam, P, Q, T), rel=1e-12)


def test_eq10_is_symmetric_under_window_swap(world):
    """s and t label two windows; nothing distinguishes them."""
    lam, _ = world
    rng = np.random.default_rng(3)
    a = perturb_top(lam, N_TOP, SIGMA, rng)
    b = perturb_top(lam, N_TOP, SIGMA, rng)
    assert not np.allclose(a, b), "spectra must actually differ for this to bite"
    assert d_null_two_samples(a, b, P, Q, T) == pytest.approx(
        d_null_two_samples(b, a, P, Q, T), rel=1e-12)


def test_null_is_invariant_under_a_common_rescaling(world):
    """D depends on the shape of the spectrum, never on its overall level.

    Consequence worth stating plainly: a market-wide change in volatility that
    scales every eigenvalue equally moves the null not at all. Only changes in
    the relative arrangement of the eigenvalues register.
    """
    lam, _ = world
    base = d_null_sample_vs_true(lam, P, Q, T)
    for c in (0.1, 2.0, 50.0):
        assert d_null_sample_vs_true(lam * c, P, Q, T) == pytest.approx(base, rel=1e-10)


# --------------------------------------------------------------------------
# The generators themselves.
# --------------------------------------------------------------------------

def test_perturb_top_leaves_the_bulk_and_the_mean_alone(world):
    lam, _ = world
    rng = np.random.default_rng(4)
    assert perturb_top(lam, N_TOP, 0.0, rng) == pytest.approx(lam)
    draws = np.array([perturb_top(lam, N_TOP, SIGMA, rng) for _ in range(4000)])
    assert np.all(draws[:, N_TOP:] == lam[N_TOP:]), "bulk must not move"
    assert draws[:, :N_TOP].mean(axis=0) == pytest.approx(lam[:N_TOP], rel=0.01)


def test_perturb_top_refuses_to_reorder_the_spectrum(world):
    """A crossing relabels eigenvectors, which is indistinguishable from a
    rotation. Silently resorting would smuggle exactly the signal this regime
    is supposed to exclude, so the guard must raise instead."""
    lam, _ = world
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="reordered"):
        for _ in range(500):
            perturb_top(lam, 6, 0.6, rng)


def test_returns_fixed_basis_needs_T_stated_explicitly(world):
    """A 1-D spectrum must not be read as a one-column panel."""
    lam, B = world
    rng = np.random.default_rng(6)
    assert returns_fixed_basis(B, lam, 500, rng).shape == (N, 500)
    with pytest.raises(ValueError, match="steps"):
        returns_fixed_basis(B, ramp_path(lam, lam, 500), 400, rng)


def test_fixed_basis_draw_agrees_with_the_static_null(world):
    """The new generator must reproduce regime 1 when nothing moves."""
    lam, B = world
    rng = np.random.default_rng(99)
    d = []
    for _ in range(TRIALS):
        _, Qs = spectral(sample_covariance(returns_fixed_basis(B, lam, T, rng)))
        _, Qt = spectral(sample_covariance(returns_fixed_basis(B, lam, T, rng)))
        d.append(subspace_distance(Qs[:, :P], Qt[:, :Q]))
    d_th = d_null_two_samples(lam, lam, P, Q, T)
    assert float(np.mean(d)) == pytest.approx(d_th, rel=TOL)


# --------------------------------------------------------------------------
# 2.1 -- the regime proper.
# --------------------------------------------------------------------------

def test_moving_eigenvalues_are_absorbed_by_the_null(world):
    """Independent spectra per window, one basis. D_num must match D_th.

    If this fails high, the instrument is reading magnitude changes as rotation
    and no later result can be trusted.
    """
    lam, B = world
    rng = np.random.default_rng(4242)
    d, th = np.empty(TRIALS), np.empty(TRIALS)
    for k in range(TRIALS):
        ls = perturb_top(lam, N_TOP, SIGMA, rng)
        lt = perturb_top(lam, N_TOP, SIGMA, rng)
        _, Qs = spectral(sample_covariance(returns_fixed_basis(B, ls, T, rng)))
        _, Qt = spectral(sample_covariance(returns_fixed_basis(B, lt, T, rng)))
        d[k] = subspace_distance(Qs[:, :P], Qt[:, :Q])
        th[k] = d_null_two_samples(ls, lt, P, Q, T)
    assert d.mean() == pytest.approx(th.mean(), rel=TOL), \
        f"D_num={d.mean():.6g} D_th={th.mean():.6g} ratio={d.mean() / th.mean():.4f}"


def test_a_static_null_understates_a_moving_spectrum(world):
    """Quantifies what the shortcut costs.

    Using one whole-period spectrum for both windows, rather than each window's
    own, biases the null low and so reports excess where there is none. Small
    here, but it is a false positive with a known sign.
    """
    lam, _ = world
    rng = np.random.default_rng(5)
    static = d_null_two_samples(lam, lam, P, Q, T)
    moving = np.mean([d_null_two_samples(perturb_top(lam, N_TOP, SIGMA, rng),
                                         perturb_top(lam, N_TOP, SIGMA, rng), P, Q, T)
                      for _ in range(5000)])
    assert moving > static
    assert moving / static == pytest.approx(1.006, abs=0.004)


# --------------------------------------------------------------------------
# 2.2 -- feeding the null estimated eigenvalues instead of true ones.
# --------------------------------------------------------------------------

def _substitution_bias(lam, B, T_win, trials, seed):
    rng = np.random.default_rng(seed)
    r = np.empty(trials)
    for k in range(trials):
        ls = perturb_top(lam, N_TOP, SIGMA, rng)
        lt = perturb_top(lam, N_TOP, SIGMA, rng)
        es, _ = spectral(sample_covariance(returns_fixed_basis(B, ls, T_win, rng)))
        et, _ = spectral(sample_covariance(returns_fixed_basis(B, lt, T_win, rng)))
        r[k] = (d_null_two_samples(es, et, P, Q, T_win)
                / d_null_two_samples(ls, lt, P, Q, T_win))
    return r.mean() - 1.0


def test_estimated_spectrum_biases_the_null_low_and_shrinks_with_T(world):
    """On real data the true spectrum is unavailable and sample eigenvalues get
    substituted. This is the last world in which both are known, so it is the
    only place the cost of that substitution can be measured rather than
    assumed.

    At THESE settings the bias is negative, so it shrinks the null and inflates
    apparent excess. Do not generalise that: holding q fixed and raising N to 80
    flips the sign positive, because a denser bulk spreads upward under sampling
    and closes the gaps instead of opening them. The assertion below is a
    regression guard on this configuration, not a statement about the estimator.
    See stage1/README.md, "2.2".
    """
    lam, B = world
    short = _substitution_bias(lam, B, 250, 400, 701)
    long = _substitution_bias(lam, B, 1000, 400, 702)
    assert short < 0, f"expected a downward bias, got {short:+.4f}"
    assert abs(short) < 0.03, f"bias at T=250 unexpectedly large: {short:+.4f}"
    assert abs(long) < abs(short), f"bias did not shrink with T: {short:+.4f} -> {long:+.4f}"


# --------------------------------------------------------------------------
# 2.3 -- the assumption Eq (10) actually rests on.
# --------------------------------------------------------------------------

def _drift_ratio(lam, B, h, trials, seed):
    rng = np.random.default_rng(seed)
    lo, hi = lam * (1 - h), lam * (1 + h)
    d = np.empty(trials)
    for k in range(trials):
        _, Qs = spectral(sample_covariance(
            returns_fixed_basis(B, ramp_path(lo, hi, T), T, rng)))
        _, Qt = spectral(sample_covariance(
            returns_fixed_basis(B, ramp_path(lo, hi, T), T, rng)))
        d[k] = subspace_distance(Qs[:, :P], Qt[:, :Q])
    return d.mean() / d_null_two_samples(lam, lam, P, Q, T)


def test_common_drift_within_a_window_inflates_D(world):
    """Eq (10) assumes each window has one well-defined spectrum. Let the level
    ramp from (1-h) to (1+h) *during* the window and that assumption fails.

    The whole-window estimate recovers the time-average, but the sampling noise
    is set by the time-average of the square, so D comes out inflated by
    <c^2>/<c>^2 = 1 + h^2/3 for a linear ramp. Zero rotation is present
    throughout: this is the shape a false positive takes.

    Note the contrast with test_null_is_invariant_under_a_common_rescaling. A
    common rescaling that is constant across the window is invisible to D. The
    same rescaling spread out over the window is not.
    """
    lam, B = world
    assert _drift_ratio(lam, B, 0.0, TRIALS, 800) == pytest.approx(1.0, rel=TOL)
    for h in (0.6, 0.8):
        assert _drift_ratio(lam, B, h, TRIALS, 800 + int(10 * h)) == pytest.approx(
            1.0 + h * h / 3.0, rel=0.06), f"h={h}"


def test_variogram_still_holds_when_the_spectrum_moves(world):
    """Eq (9) is the flat line the empirical variogram is compared against. It
    is a static-C statement, so under a genuinely moving spectrum the measured
    variogram must sit *above* it. That gap is the signal Fig. 1 of the paper
    reports, reproduced here with a known answer.
    """
    lam, B = world
    rng = np.random.default_rng(31)
    gaps = []
    for _ in range(TRIALS):
        ls = perturb_top(lam, N_TOP, SIGMA, rng)
        lt = perturb_top(lam, N_TOP, SIGMA, rng)
        es, _ = spectral(sample_covariance(returns_fixed_basis(B, ls, T, rng)))
        et, _ = spectral(sample_covariance(returns_fixed_basis(B, lt, T, rng)))
        gaps.append((es[:N_TOP] - et[:N_TOP]) ** 2)
    emp = np.mean(gaps, axis=0)
    static = eigenvalue_variogram_null(lam[:N_TOP], T)
    expected = static + 2.0 * (SIGMA * lam[:N_TOP]) ** 2
    assert np.all(emp > static), f"emp={emp} static={static}"
    # The inequality above is the claim with teeth: the injected jitter puts the
    # variogram ~4.6x above the static line. The quantitative check below runs
    # looser than 1.2's own 20%, because it stacks a jitter model on top of the
    # same eigenvalue-repulsion effects that already cost 1.2 ~10% per mode.
    assert emp == pytest.approx(expected, rel=0.30), f"emp={emp} expected={expected}"
