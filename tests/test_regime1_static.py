"""Stage 1, regime 1: static C.

The null is true by construction, so measured excess must be ~0. This is the
guard against metric-liar #4 (estimated-spectrum contamination): here the true
spectrum is known exactly, so any excess the pipeline reports is manufactured.

If these fail, the instrument is wrong. There is no finding available in this
file.
"""
import numpy as np
import pytest

from src.overlap import spectral, sample_covariance, subspace_distance
from src.null_rmt import (d_null_sample_vs_true, d_null_two_samples,
                          eigenvalue_variogram_null)
from src.synth import spd_from_spectrum, factor_spectrum, gaussian_returns

N, P, Q, T = 40, 3, 6, 2000
TRIALS = 400
TOL = 0.08                 # observed spread is ~1-3% at these settings


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(20260801)
    lam = factor_spectrum(N, [25.0, 10.0, 6.0, 4.0, 3.0, 2.2])
    C, Qtrue = spd_from_spectrum(lam, rng)
    return lam, C, Qtrue, rng


def test_true_vs_sample_matches_eq7(world):
    """Top-P eigenvectors of the true C against top-Q of one sample covariance."""
    lam, C, Qtrue, rng = world
    U = Qtrue[:, :P]
    d = [subspace_distance(U, spectral(sample_covariance(gaussian_returns(C, T, rng)))[1][:, :Q])
         for _ in range(TRIALS)]
    d_emp, d_th = float(np.mean(d)), d_null_sample_vs_true(lam, P, Q, T)
    assert d_emp == pytest.approx(d_th, rel=TOL), f"emp={d_emp:.6g} th={d_th:.6g}"


def test_two_independent_samples_match_eq10(world):
    """Two non-overlapping windows. This is the real-data geometry."""
    lam, C, Qtrue, rng = world
    d = []
    for _ in range(TRIALS):
        _, Qs = spectral(sample_covariance(gaussian_returns(C, T, rng)))
        _, Qt = spectral(sample_covariance(gaussian_returns(C, T, rng)))
        d.append(subspace_distance(Qs[:, :P], Qt[:, :Q]))
    d_emp, d_th = float(np.mean(d)), d_null_two_samples(lam, lam, P, Q, T)
    assert d_emp == pytest.approx(d_th, rel=TOL), f"emp={d_emp:.6g} th={d_th:.6g}"


def test_measured_excess_is_zero(world):
    """The regime-1 statement in the plan's own terms: excess ~ 0."""
    lam, C, Qtrue, rng = world
    d = []
    for _ in range(TRIALS):
        _, Qs = spectral(sample_covariance(gaussian_returns(C, T, rng)))
        _, Qt = spectral(sample_covariance(gaussian_returns(C, T, rng)))
        d.append(subspace_distance(Qs[:, :P], Qt[:, :Q]))
    d_th = d_null_two_samples(lam, lam, P, Q, T)
    excess = np.array(d) - d_th
    se = excess.std(ddof=1) / np.sqrt(TRIALS)
    assert abs(excess.mean()) < max(3 * se, TOL * d_th), (
        f"excess={excess.mean():.3g} +/- {se:.3g}, d_th={d_th:.3g}")


def test_eigenvalue_variogram_matches_eq9(world):
    """Eq (9): <(lam_i^s - lam_i^t)^2> = 4 lam_i^2 / T under a static C."""
    lam, C, Qtrue, rng = world
    gaps = []
    for _ in range(TRIALS):
        ls, _ = spectral(sample_covariance(gaussian_returns(C, T, rng)))
        lt, _ = spectral(sample_covariance(gaussian_returns(C, T, rng)))
        gaps.append((ls[:P] - lt[:P]) ** 2)
    emp = np.mean(gaps, axis=0)
    th = eigenvalue_variogram_null(lam[:P], T)
    assert emp == pytest.approx(th, rel=0.20), f"emp={emp} th={th}"


def test_pipeline_does_not_manufacture_signal_at_larger_T(world):
    """Excess must shrink like 1/T, not sit at a floor."""
    lam, C, Qtrue, rng = world
    out = {}
    for t in (1000, 4000):
        d = []
        for _ in range(200):
            _, Qs = spectral(sample_covariance(gaussian_returns(C, t, rng)))
            _, Qt = spectral(sample_covariance(gaussian_returns(C, t, rng)))
            d.append(subspace_distance(Qs[:, :P], Qt[:, :Q]))
        out[t] = float(np.mean(d))
    assert out[1000] / out[4000] == pytest.approx(4.0, rel=0.15), out
