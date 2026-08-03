"""The accidental-overlap benchmark, and the normalisation caveat pinned in code."""
import pytest
from src.null_rmt import d_random_subspaces, _random_subspace_mass


def test_paper_convention_reproduces_quoted_value():
    # arXiv:1108.4258 (short letter) Fig. 2 caption: D_RMT ~ 0.83 for
    # P=5, Q=10, N=204. Quoted there, not in arXiv:1203.6228.
    assert d_random_subspaces(5, 10, 204, "paper") == pytest.approx(0.8275, abs=2e-3)


def test_normalised_density_integrates_to_one():
    for P, Q, N in [(5, 10, 204), (11, 21, 204), (3, 30, 500)]:
        assert _random_subspace_mass(P, Q, N, "normalised") == pytest.approx(1.0, abs=2e-3)


def test_paper_density_carries_mass_p_over_q():
    # This is the discrepancy: the printed density is not a probability density.
    for P, Q, N in [(5, 10, 204), (11, 21, 204)]:
        assert _random_subspace_mass(P, Q, N, "paper") == pytest.approx(P / Q, abs=2e-3)


def test_conventions_differ_by_exactly_p_over_q():
    P, Q, N = 5, 10, 204
    ratio = d_random_subspaces(P, Q, N, "paper") / d_random_subspaces(P, Q, N, "normalised")
    assert ratio == pytest.approx(P / Q, rel=2e-3)
