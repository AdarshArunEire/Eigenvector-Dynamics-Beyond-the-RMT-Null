"""Stage 1, regime 3.2: choosing P and Q.

Two separate questions, with two very different answers.

Q is solved. The Marchenko-Pastur upper edge is a closed-form threshold that
needs only the observed spectrum and T, so Q stops being a free parameter.

P is not solved, and the interesting result is that it cannot be. Every P is
structurally blind to rotation below its own boundary, and every increase in P
costs threshold superlinearly. There is no dominating choice, only a trade
whose right side depends on where the rotation actually lives -- which is an
empirical question about markets, not something a synthetic sweep decides.

Note the world here differs from regime 3.1: a flat bulk, because the MP edge
is only meaningful when the bulk is genuinely noise. 3.1's `linspace` bulk is
real spread structure and MP correctly refuses to call it noise (test below).
Injection here is whole-block, not single-plane, so D_inject is P-independent
and schemes with different P can be compared on the same physical event.

Correctness, not science. Nothing here is a finding.
"""
import numpy as np
import pytest

from src.data import to_correlation_panel
from src.overlap import spectral, sample_covariance, subspace_distance
from src.null_rmt import mp_upper_edge, q_from_mp_edge
from src.synth import (spd_from_spectrum, factor_spectrum, rotate_basis,
                       returns_fixed_basis)

N, T = 40, 2000
FACTORS = [25.0, 10.0, 6.0, 4.0, 3.0, 2.2]
J0 = 34                     # rotation targets, deep in the bulk and past any Q


@pytest.fixture(scope="module")
def flat_world():
    """Six factors over a flat bulk, so the sample bulk is genuinely MP."""
    lam = np.concatenate([FACTORS, np.ones(N - len(FACTORS))])
    _, B = spd_from_spectrum(lam, np.random.default_rng(20260801))
    return lam, B


# --------------------------------------------------------------------------
# Q: the Marchenko-Pastur edge.
# --------------------------------------------------------------------------

def test_mp_edge_closed_form():
    """lam_+ = sigma2 (1 + sqrt(N/T))^2, and it tightens to sigma2 as T grows."""
    assert mp_upper_edge(1.0, 40, 2000) == pytest.approx((1 + np.sqrt(0.02)) ** 2)
    assert mp_upper_edge(2.5, 40, 2000) == pytest.approx(2.5 * (1 + np.sqrt(0.02)) ** 2)
    assert mp_upper_edge(1.0, 40, 40) == pytest.approx(4.0)     # q=1 -> edge 4 sigma2
    # tightens onto sigma2 as data accumulates, from above
    edges = [mp_upper_edge(1.0, 40, T_) for T_ in (10 ** 4, 10 ** 6, 10 ** 8, 10 ** 10)]
    assert np.all(np.diff(edges) < 0) and np.all(np.array(edges) > 1.0)
    assert edges[-1] == pytest.approx(1.0, abs=1e-3)


def test_mp_edge_recovers_the_noise_level_and_factor_count(flat_world):
    """On the true spectrum: sigma2 back to 1.0, and Q equal to the factor count."""
    lam, _ = flat_world
    Q, edge, sigma2 = q_from_mp_edge(lam, T)
    assert sigma2 == pytest.approx(1.0, rel=1e-6)
    assert edge == pytest.approx(1.3028, abs=1e-3)
    assert Q == len(FACTORS)


def test_mp_edge_picks_the_same_q_from_noisy_samples(flat_world):
    """The point of the rule: it works on estimated spectra, not just true ones."""
    lam, B = flat_world
    rng = np.random.default_rng(4242)
    picks = np.array([q_from_mp_edge(
        spectral(sample_covariance(returns_fixed_basis(B, lam, T, rng)))[0], T)[0]
        for _ in range(200)])
    assert (picks == len(FACTORS)).mean() > 0.95, np.bincount(picks)


def test_mp_edge_refuses_a_bulk_that_is_not_noise():
    """The assumption that must be checked rather than assumed.

    Regime 3.1's spectrum has a bulk spread from 1.3 to 0.4 -- genuine
    structure, not sampling noise. MP is right to report most of it as signal,
    and the large Q is the rule telling you it does not apply, not a bug.
    """
    lam = factor_spectrum(N, FACTORS)          # linspace bulk, 1.3 -> 0.4
    Q, _, _ = q_from_mp_edge(lam, T)
    assert Q > 3 * len(FACTORS), f"expected MP to reject this bulk, got Q={Q}"


def test_mp_edge_over_counts_when_residual_variances_spread():
    """The limit that matters on real data, and that the flat-bulk test hides.

    A flat bulk is the homogeneous case, which is precisely where MP is exact.
    Give the names heterogeneous factor loadings -- so their residual variances
    differ, as real equities do -- and the bulk widens beyond the edge, and the
    criterion reports a large part of the spectrum as signal.

    Three true factors throughout. Q climbs with the spread, so the edge is a
    starting point on real data, not an answer.
    """
    rng = np.random.default_rng(1)
    n, t, factors = 111, 2765, 3
    f = rng.normal(0, 1, (factors, t))
    found = []
    for beta_sd in (0.05, 0.25, 1.00):
        load = np.c_[rng.normal(1.0, beta_sd, n),
                     rng.normal(0.0, beta_sd, n),
                     rng.normal(0.0, beta_sd, n)]
        panel = load @ f + rng.normal(0, 1, (n, t))
        evals, _ = spectral(sample_covariance(to_correlation_panel(panel)))
        found.append(q_from_mp_edge(evals, t)[0])
    assert found[0] == factors, found          # homogeneous: exact
    assert found[-1] > 10 * factors, found     # heterogeneous: badly over-counts
    assert found == sorted(found), found       # monotone in the spread


def test_correlation_scaling_is_required_before_the_edge():
    """Volatility spread across names breaks the edge on a raw covariance.

    MP describes variables of a common variance. Real names differ in
    volatility by factors of several, which widens the bulk on its own, so the
    edge must be taken on the correlation matrix.
    """
    rng = np.random.default_rng(0)
    n, t = 111, 2765
    vols = np.exp(rng.normal(np.log(0.02), 0.5, n))
    panel = (rng.normal(0, 1, (n, 3)) @ rng.normal(0, 1, (3, t)) * 0.5
             + rng.normal(0, 1, (n, t))) * vols[:, None]
    raw = q_from_mp_edge(spectral(sample_covariance(panel))[0], t)[0]
    corr = q_from_mp_edge(
        spectral(sample_covariance(to_correlation_panel(panel)))[0], t)[0]
    assert corr < raw / 2, (corr, raw)


def test_to_correlation_panel_gives_unit_variance_rows():
    rng = np.random.default_rng(3)
    panel = rng.normal(0, 1, (20, 500)) * np.exp(rng.normal(0, 1, (20, 1)))
    out = to_correlation_panel(panel)
    assert out.std(axis=1, ddof=1) == pytest.approx(np.ones(20))
    with pytest.raises(ValueError, match="zero variance"):
        to_correlation_panel(np.zeros((3, 10)))


def test_q_from_mp_edge_rejects_impossible_input():
    with pytest.raises(ValueError):
        q_from_mp_edge(np.array([1.0, -0.5]), T)
    with pytest.raises(ValueError):
        q_from_mp_edge(np.array([]), T)


# --------------------------------------------------------------------------
# P: the blindness structure. Exact and noiseless -- no trials needed.
# --------------------------------------------------------------------------

def _block_rotate(B, P, theta, j0=J0):
    """Tilt each of the top P directions by theta into its own deep bulk mode."""
    R = B.copy()
    for k in range(P):
        R = rotate_basis(R, k, j0 + k, theta)
    return R


def test_whole_block_injection_is_P_independent(flat_world):
    """The reason this injection is used at all.

    A single-plane rotation gives D_inject = -ln(cos theta)/P, which shrinks
    with P and so rigs any comparison in favour of small blocks. Tilting the
    whole block makes all P principal angles equal theta, giving
    D_inject = -ln(cos theta) for every P -- the same physical event costing
    the same D whatever block size was chosen.
    """
    _, B = flat_world
    theta = 0.30
    expected = -np.log(np.cos(theta))
    for P in (1, 2, 3, 5):
        d = subspace_distance(_block_rotate(B, P, theta)[:, :P], B[:, :6])
        assert d == pytest.approx(expected, rel=1e-10), f"P={P}"


def test_small_P_is_structurally_blind_below_its_boundary(flat_world):
    """Rotation in mode m is invisible to any scheme with P <= m.

    Not "hard to see" -- exactly zero, with no noise involved. This is what
    makes "P=1 has the sharpest threshold" a worthless argument on its own.
    """
    _, B = flat_world
    Q, theta = 6, 0.30
    for m in (2, 4):
        for P in range(1, m + 1):                      # P <= m: blind
            d = subspace_distance(rotate_basis(B, m, J0, theta)[:, :P], B[:, :Q])
            assert d == pytest.approx(0.0, abs=1e-12), f"m={m} P={P}"
        for P in (m + 1, m + 2):                       # P > m: registers
            d = subspace_distance(rotate_basis(B, m, J0, theta)[:, :P], B[:, :Q])
            assert d == pytest.approx(-np.log(np.cos(theta)) / P, rel=1e-10)


def test_noise_contribution_grows_sharply_down_the_block(flat_world):
    """Why the threshold worsens superlinearly in P.

    Each mode added to the block sits closer to the bulk, so its gap shrinks
    and its share of the null grows. Across the six factors it grows by more
    than an order of magnitude, which a 1/P average cannot damp.
    """
    lam, _ = flat_world
    tail = lam[6:]
    contrib = np.array([float(np.sum(lam[i] * tail / (lam[i] - tail) ** 2))
                        for i in range(6)])
    assert np.all(np.diff(contrib) > 0), contrib
    assert contrib[-1] / contrib[0] > 20, contrib[-1] / contrib[0]


def test_the_null_grows_with_P_at_fixed_Q(flat_world):
    """The consequence, measured rather than derived."""
    lam, B = flat_world
    Q, trials = 6, 200
    rng = np.random.default_rng(11)
    bases = []
    for _ in range(trials):
        _, a = spectral(sample_covariance(returns_fixed_basis(B, lam, T, rng)))
        _, b = spectral(sample_covariance(returns_fixed_basis(B, lam, T, rng)))
        bases.append((a, b))
    means = [np.mean([subspace_distance(a[:, :P], b[:, :Q]) for a, b in bases])
             for P in (1, 2, 3, 5)]
    assert np.all(np.diff(means) > 0), means
