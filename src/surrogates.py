"""Return-level surrogates and factor removal for Regime 4.8.

IAAFT preserves each asset's observed marginal distribution exactly and its
linear spectrum approximately, while independent phases remove the original
cross-sectional timing.  Rolling factor removal is deliberately performed
inside each estimation window so no future return enters a beta estimate.
"""
import numpy as np


def iaaft_surrogate(series, rng, max_iter=200, atol=1e-8):
    """Iterative amplitude-adjusted Fourier surrogate of one finite series.

    The returned values are a permutation of ``series``.  Iteration stops when
    the relative Fourier-amplitude error ceases to improve materially.
    """
    x = np.asarray(series, dtype=float)
    if x.ndim != 1 or x.size < 4:
        raise ValueError("series must be one-dimensional with at least four values")
    if not np.isfinite(x).all():
        raise ValueError("series contains non-finite values")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if np.ptp(x) == 0:
        return x.copy()

    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))
    scale = np.linalg.norm(target_amp[1:])
    y = rng.permutation(x)
    previous = np.inf
    stagnant = 0
    for _ in range(max_iter):
        spectrum = np.fft.rfft(y)
        phases = np.divide(spectrum, np.abs(spectrum),
                           out=np.ones_like(spectrum),
                           where=np.abs(spectrum) > 0)
        spectral_match = np.fft.irfft(target_amp * phases, n=x.size)
        order = np.argsort(spectral_match, kind="mergesort")
        candidate = np.empty_like(y)
        candidate[order] = sorted_x
        amplitude_error = np.linalg.norm(
            np.abs(np.fft.rfft(candidate))[1:] - target_amp[1:])
        error = amplitude_error / scale if scale > 0 else 0.0
        if previous - error <= atol * max(1.0, previous):
            stagnant += 1
        else:
            stagnant = 0
        y = candidate
        if error <= atol or stagnant >= 8:
            break
        previous = error
    return y


def independent_iaaft_panel(panel, rng, max_iter=200, atol=1e-8):
    """Apply IAAFT independently to every asset row of a return panel."""
    x = np.asarray(panel, dtype=float)
    if x.ndim != 2:
        raise ValueError("panel must have shape (assets, time)")
    return np.asarray([
        iaaft_surrogate(row, rng, max_iter=max_iter, atol=atol) for row in x
    ])


def remove_equal_weight_factor(window):
    """OLS-residualise every asset on the contemporaneous equal-weight return.

    An intercept is included by centring both regressand and factor.  The
    operation is intended for one rolling estimation window, not a full panel.
    """
    x = np.asarray(window, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("window must have shape (assets, at least two days)")
    if not np.isfinite(x).all():
        raise ValueError("window contains non-finite values")
    centred = x - x.mean(axis=1, keepdims=True)
    market = x.mean(axis=0)
    market = market - market.mean()
    denominator = float(market @ market)
    if denominator <= np.finfo(float).eps:
        return centred
    betas = (centred @ market) / denominator
    return centred - betas[:, None] * market[None, :]
