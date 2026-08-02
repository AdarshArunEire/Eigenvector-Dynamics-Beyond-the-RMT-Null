"""Stage 1, regime 3: injected drift.

Regimes 1 and 2 asked whether the instrument reports rotation when there is
none. This one asks the mirror question: when a known rotation is present, does
it report the right amount?

The claim under test is not really "can it detect rotation". It is that
`excess = D_emp - D_th` is a legitimate subtraction — that noise-rotation and
real-rotation add. There is a reason to expect it, since for small angles
D = -(1/P) sum_k ln cos(theta_k) ~ (1/2P) sum_k theta_k^2, and squared angles
from independent sources add. But "approximately, for small angles" is doing
work in that sentence, and regimes 1 and 2 could not test it: both had zero
signal, so additivity never had to hold.

Injection is a single Givens rotation, so the truth is known in closed form
rather than simulated: D_inject = -ln(cos theta) / P.

Correctness, not science. Nothing here is a finding.
"""
import numpy as np
import pytest

from src.overlap import spectral, sample_covariance, subspace_distance
from src.null_rmt import d_null_two_samples
from src.synth import (spd_from_spectrum, factor_spectrum, rotate_basis,
                       d_injected, returns_fixed_basis)

N, P, Q, T = 40, 3, 6, 2000
TRIALS = 300
J_PAST_Q = 10              # a bulk mode, comfortably past the Q boundary


@pytest.fixture(scope="module")
def world():
    lam = factor_spectrum(N, [25.0, 10.0, 6.0, 4.0, 3.0, 2.2])
    _, B = spd_from_spectrum(lam, np.random.default_rng(20260801))
    return lam, B


# --------------------------------------------------------------------------
# Does the injection inject anything? Three cases, no noise, exact answers.
#
# Skipping these is how you end up sweeping a parameter for an hour while
# measuring nothing -- the same failure the first attempt at 2.3 walked into.
# --------------------------------------------------------------------------

def test_rotate_basis_stays_orthonormal(world):
    _, B = world
    R = rotate_basis(B, 0, J_PAST_Q, 0.3)
    assert R.T @ R == pytest.approx(np.eye(N), abs=1e-12)


def test_rotation_inside_the_top_block_is_invisible(world):
    """Mixing two directions that are both already in the span is relabelling."""
    _, B = world
    for theta in (0.1, 0.3, 1.0):
        d = subspace_distance(rotate_basis(B, 0, 1, theta)[:, :P], B[:, :Q])
        assert d == pytest.approx(0.0, abs=1e-12), f"theta={theta}"


def test_rotation_into_the_p_to_q_buffer_is_invisible(world):
    """This is precisely what Q > P is for: leakage into modes P..Q-1 lands
    inside the outer block and costs nothing."""
    _, B = world
    for theta in (0.1, 0.3, 1.0):
        d = subspace_distance(rotate_basis(B, 1, 4, theta)[:, :P], B[:, :Q])
        assert d == pytest.approx(0.0, abs=1e-12), f"theta={theta}"


def test_rotation_past_q_matches_the_closed_form(world):
    """Only here does anything register, and the answer is known exactly."""
    _, B = world
    for theta in (0.05, 0.3, 0.8):
        d = subspace_distance(rotate_basis(B, 0, J_PAST_Q, theta)[:, :P], B[:, :Q])
        assert d == pytest.approx(d_injected(theta, P), rel=1e-10), f"theta={theta}"


def test_the_q_boundary_is_where_it_is_claimed_to_be(world):
    """Rotating into mode Q-1 is invisible; into mode Q it is not. Pins the
    boundary rather than assuming it."""
    _, B = world
    inside = subspace_distance(rotate_basis(B, 0, Q - 1, 0.3)[:, :P], B[:, :Q])
    outside = subspace_distance(rotate_basis(B, 0, Q, 0.3)[:, :P], B[:, :Q])
    assert inside == pytest.approx(0.0, abs=1e-12)
    assert outside == pytest.approx(d_injected(0.3, P), rel=1e-10)


# --------------------------------------------------------------------------
# 3.1 -- the regime proper.
# --------------------------------------------------------------------------

def _measure(lam, B, theta, seed):
    """Window s on the original basis, window t on the rotated one. Both noisy."""
    Bt = rotate_basis(B, 0, J_PAST_Q, theta)
    rng = np.random.default_rng(seed)
    d = np.empty(TRIALS)
    for k in range(TRIALS):
        _, Qs = spectral(sample_covariance(returns_fixed_basis(B, lam, T, rng)))
        _, Qt = spectral(sample_covariance(returns_fixed_basis(Bt, lam, T, rng)))
        d[k] = subspace_distance(Qs[:, :P], Qt[:, :Q])
    return d


def test_injected_drift_adds_to_the_null(world):
    """The headline: D_emp = D_th + D_inject.

    If this failed, every excess reported downstream would be measuring
    something other than the rotation that produced it.
    """
    lam, B = world
    d_th = d_null_two_samples(lam, lam, P, Q, T)
    for theta in (0.10, 0.30, 0.80):
        d = _measure(lam, B, theta, 606)
        predicted = d_th + d_injected(theta, P)
        assert d.mean() == pytest.approx(predicted, rel=0.05), (
            f"theta={theta} D_emp={d.mean():.6g} predicted={predicted:.6g}")


def test_recovery_sharpens_as_the_injection_clears_the_noise_floor(world):
    """`(D_emp - D_th) / D_inject` should approach 1 from above.

    It sits high for small injections because the instrument carries a small
    constant positive offset -- the same O(1/T^2) residual regimes 1 and 2 both
    showed -- and a fixed offset matters proportionally more the smaller the
    thing you are dividing it by. Worth knowing before reading any detection
    threshold off a curve: near the floor, recovered magnitude runs about 10%
    high.
    """
    lam, B = world
    d_th = d_null_two_samples(lam, lam, P, Q, T)
    rec = {}
    for theta in (0.05, 0.20):
        d = _measure(lam, B, theta, 606)
        rec[theta] = (d.mean() - d_th) / d_injected(theta, P)
    assert rec[0.05] > 1.0, rec
    assert rec[0.20] == pytest.approx(1.0, rel=0.05), rec
    assert rec[0.20] < rec[0.05], f"recovery should tighten with theta: {rec}"
