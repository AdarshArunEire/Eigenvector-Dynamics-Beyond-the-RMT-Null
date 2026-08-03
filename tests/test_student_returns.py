"""The Student generator must be the stated distribution, not merely fat-tailed.

Two properties do the work downstream, so both are pinned: the covariance must
be exactly C (not C*nu/(nu-2)), otherwise the fat-tail effect gets tangled with
a spectrum change; and the scale must be shared across the cross-section on a
given day, which is what makes `standardise` able to remove it.
"""
import numpy as np
import pytest

from src.synth import (spd_from_spectrum, factor_spectrum, student_returns,
                       student_factor)


def _C(N=20, seed=0):
    rng = np.random.default_rng(seed)
    return spd_from_spectrum(factor_spectrum(N, [8.0, 5.0]), rng)[0], rng


def test_covariance_is_C_not_inflated():
    C, rng = _C()
    r = student_returns(C, 400_000, 8.0, rng)
    emp = (r @ r.T) / r.shape[1]
    assert np.abs(emp - C).max() / np.abs(C).max() < 0.05


def test_marginal_kurtosis_matches_nu():
    C, rng = _C(N=6)
    for nu in (8.0, 12.0):
        r = student_returns(C, 300_000, nu, rng)
        z = r - r.mean(axis=1, keepdims=True)
        k = np.median((z ** 4).mean(axis=1) / z.var(axis=1, ddof=1) ** 2)
        assert k == pytest.approx(3 * student_factor(nu), rel=0.20)


def test_scale_is_common_across_names_on_a_given_day():
    """The defining feature. Cross-sectional mean square must vary far more
    day to day than it would for independent-per-name heavy tails."""
    C, rng = _C(N=200)
    s_student = (student_returns(C, 3000, 6.0, rng) ** 2).mean(axis=0)
    cv2_student = s_student.var() / s_student.mean() ** 2
    L = np.linalg.cholesky(C)
    s_gauss = ((L @ rng.standard_normal((200, 3000))) ** 2).mean(axis=0)
    cv2_gauss = s_gauss.var() / s_gauss.mean() ** 2
    assert cv2_student > 10 * cv2_gauss


def test_student_factor_matches_scale_mixture_moments():
    """(nu-2)/(nu-4) is E[s^2]/E[s]^2 for s = nu/chi2_nu -- the same quantity
    as the 1 + CV^2 of regime 2.3."""
    rng = np.random.default_rng(1)
    for nu in (6.0, 10.0, 20.0):
        s = nu / rng.chisquare(nu, size=2_000_000)
        assert (s ** 2).mean() / s.mean() ** 2 == pytest.approx(
            student_factor(nu), rel=0.05)


def test_gaussian_limit():
    C, rng = _C()
    assert student_factor(1e6) == pytest.approx(1.0, abs=1e-4)
    r = student_returns(C, 200_000, 1e6, rng)
    z = r - r.mean(axis=1, keepdims=True)
    k = np.median((z ** 4).mean(axis=1) / z.var(axis=1, ddof=1) ** 2)
    assert k == pytest.approx(3.0, rel=0.10)


def test_refuses_infinite_fourth_moment():
    C, rng = _C()
    for bad in (4.0, 3.0, 1.0):
        with pytest.raises(ValueError, match="nu"):
            student_returns(C, 10, bad, rng)
        with pytest.raises(ValueError, match="nu"):
            student_factor(bad)
