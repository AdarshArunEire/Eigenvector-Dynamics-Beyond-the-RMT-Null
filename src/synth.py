"""Synthetic generators. The correctness oracle: ground truth is exact and free."""
import numpy as np


def spd_from_spectrum(lam, rng):
    """Build C = Q diag(lam) Q.T with a Haar-random orthonormal basis.

    Returns (C, Q) with Q's columns ordered to match lam. A random basis rather
    than the identity so that basis-alignment bugs cannot pass silently.
    """
    lam = np.asarray(lam, dtype=float)
    if np.any(lam <= 0):
        raise ValueError("spectrum must be strictly positive for an SPD matrix")
    if np.any(np.diff(lam) > 1e-12):
        raise ValueError("lam must be in descending order")
    N = lam.size
    Z = rng.standard_normal((N, N))
    Q, Rq = np.linalg.qr(Z)
    Q *= np.sign(np.diag(Rq))          # fix QR sign convention -> Haar
    return (Q * lam) @ Q.T, Q


def factor_spectrum(N, top, bulk_hi=1.3, bulk_lo=0.4):
    """A descending spectrum: a few separated factors over a compressed bulk."""
    top = np.asarray(top, dtype=float)
    if np.any(np.diff(top) > 0):
        raise ValueError("top eigenvalues must be descending")
    if top[-1] <= bulk_hi:
        raise ValueError("top block must sit above the bulk")
    return np.concatenate([top, np.linspace(bulk_hi, bulk_lo, N - top.size)])


def spd_from_basis(Q, lam):
    """C = Q diag(lam) Q.T for a *supplied* basis.

    Regime 2 needs the basis to be the invariant across windows, so it has to be
    passed in rather than drawn fresh. Q's columns must already match lam's
    order.
    """
    lam = np.asarray(lam, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if lam.ndim != 1 or Q.shape != (lam.size, lam.size):
        raise ValueError(f"shape mismatch: Q{Q.shape} against lam{lam.shape}")
    if np.any(lam <= 0):
        raise ValueError("spectrum must be strictly positive for an SPD matrix")
    if np.any(np.diff(lam) > 1e-12):
        raise ValueError("lam must be in descending order")
    return (Q * lam) @ Q.T


def perturb_top(lam, n_top, sigma, rng):
    """Multiplicative lognormal jitter on the top n_top eigenvalues only.

    The basis is untouched, so this moves magnitudes and never directions.
    sigma=0 returns lam unchanged and recovers the static world exactly.

    The jitter has unit mean, so the expected spectrum is preserved and only its
    shape moves. Note that a *common* rescaling lam -> c*lam leaves the null
    invariant (see test_null_is_scale_invariant), so shape movement is the only
    kind that this regime can actually probe.

    Crossings are rejected rather than resorted. A crossing permutes the
    eigenvector labels, and a permuted label is indistinguishable from a rotated
    eigenvector -- the one thing this world must not contain. Resorting would
    hide that; raising makes the usable jitter range explicit.
    """
    lam = np.asarray(lam, dtype=float)
    if not 0 < n_top < lam.size:
        raise ValueError(f"need 0 < n_top < N, got n_top={n_top} N={lam.size}")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        return lam.copy()
    out = lam.copy()
    z = rng.standard_normal(n_top)
    out[:n_top] = lam[:n_top] * np.exp(sigma * z - 0.5 * sigma ** 2)
    if np.any(np.diff(out) > -1e-12):
        raise ValueError(
            f"sigma={sigma} reordered or degenerated the spectrum: {out[:n_top + 1]}. "
            "Lower sigma or widen the gaps in the perturbed block.")
    return out


def gaussian_returns(C, T, rng):
    """N x T draws from N(0, C). True mean is zero, so do not demean."""
    L = np.linalg.cholesky(C)
    return L @ rng.standard_normal((C.shape[0], T))


def returns_fixed_basis(Q, lam_path, T, rng):
    """N x T draws where column t is N(0, Q diag(lam_path[:, t]) Q.T).

    Because the basis never moves, the draw factorises: sample iid normals in
    the eigenbasis and scale component i at time t by sqrt(lam_path[i, t]).
    No per-step Cholesky, so a spectrum that moves *within* the window costs the
    same as one that does not.

    lam_path is (N,) for a spectrum held constant across the window, or (N, T)
    for one that moves during it. The distinction matters: Eq (10) assumes each
    window has a single well-defined spectrum, so the (N, T) case is exactly
    where that assumption is put under strain.

    T is required even in the (N,) case. Inferring it from the array shape
    instead would turn a 1-D spectrum into a one-column panel without
    complaining.
    """
    Q = np.asarray(Q, dtype=float)
    lam_path = np.asarray(lam_path, dtype=float)
    if lam_path.ndim == 1:
        lam_path = np.broadcast_to(lam_path[:, None], (lam_path.size, T))
    elif lam_path.shape[1] != T:
        raise ValueError(f"lam_path has {lam_path.shape[1]} steps, T={T}")
    if lam_path.shape[0] != Q.shape[0]:
        raise ValueError(f"lam_path has {lam_path.shape[0]} modes, Q has {Q.shape[0]}")
    if np.any(lam_path <= 0):
        raise ValueError("spectrum must be strictly positive at every step")
    z = rng.standard_normal((Q.shape[0], T))
    return Q @ (np.sqrt(lam_path) * z)


def ramp_path(lam_start, lam_end, T):
    """(N, T) linear interpolation from lam_start to lam_end across the window.

    The crudest possible within-window drift. Its time-average is the midpoint
    spectrum, which is what a whole-window estimate would recover, so the
    midpoint is the fair spectrum to hand the null when asking whether drift
    inside a window breaks it.
    """
    lam_start = np.asarray(lam_start, dtype=float)
    lam_end = np.asarray(lam_end, dtype=float)
    if lam_start.shape != lam_end.shape:
        raise ValueError("endpoint spectra must have the same shape")
    w = np.linspace(0.0, 1.0, T)[None, :]
    return lam_start[:, None] * (1 - w) + lam_end[:, None] * w
