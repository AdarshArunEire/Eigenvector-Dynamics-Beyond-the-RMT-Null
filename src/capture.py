"""Realised-variance capture: the Stage 2 respecified score.

The forecaster emits an ``N x 6`` orthonormal frame.  It is scored against the
*realised returns* of a disjoint future window, never against another estimated
Flag.  Writing ``P = Y Y^T`` for the rank-d orthogonal projector,

    capture^(d) = || Y[:, :d]^T R_out ||_F^2 / || R_out ||_F^2
                = <P_d, S_out> / tr(S_out),      S_out = R_out R_out^T

so the score is **linear in the projector**.  Two consequences are used
throughout this module and are worth stating once:

1. The optimal frame is the top-d eigenspace of E[S_out / tr S_out | F_t].
   Nothing about the eigenvalues of that expectation matters, only its leading
   eigenvectors.  Any estimator that rescales eigenvalues while holding
   eigenvectors fixed -- which is every rotationally-invariant estimator,
   Ledoit-Wolf and QIS included -- scores exactly zero by construction.
2. E[capture] for a Haar-random d-frame is exactly d/N, unconditionally on
   R_out, because E[Y Y^T] = (d/N) I.  That gives a free, exact random floor.

The in-sample top-d of R_out is the exact attainable maximum, but with
T_out ~ 42 observations it overfits the realisation heavily.  ``ceiling_bias``
measures that overfitting directly by simulating a world in which the subspace
does not move, and must be subtracted before any skill fraction is quoted.
"""
import numpy as np

from src.flag import DEFAULT_DIMS, validate_flag_frame


def _checked(frame, realised, dims):
    Y = validate_flag_frame(frame, dims, "frame")
    R = np.asarray(realised, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"realised must be 2-D (N x T_out), got {R.shape}")
    if R.shape[0] != Y.shape[0]:
        raise ValueError(f"frame has N={Y.shape[0]}, realised has N={R.shape[0]}")
    total = float(np.linalg.norm(R, "fro") ** 2)
    if total <= 0:
        raise ValueError("realised returns carry no variance")
    return Y, R, total


def variance_captured(frame, realised, dims=DEFAULT_DIMS, neutralise=None):
    """Fraction of realised variance spanned by each nested level of a Flag.

    Parameters
    ----------
    frame : (N, 6) orthonormal prediction, available at the forecast origin.
    realised : (N, T_out) realised returns of the *disjoint* target window,
        already standardised using estimation-window volatilities only.  Never
        standardise these with their own volatilities; that is a look-ahead
        leak and it is the easiest mistake to make in this design.
    neutralise : optional (N, k) orthonormal directions projected out of
        ``realised`` before scoring.  Must be identical across every entrant,
        otherwise the denominators differ and the comparison is meaningless.
        Use the common equal-weight market ``1/sqrt(N)``, never an entrant's
        own leading eigenvector.

    Returns
    -------
    dict with keys ``capture_1``, ``capture_3``, ``capture_6``.
    """
    Y, R, _ = _checked(frame, realised, dims)
    if neutralise is not None:
        B = validate_flag_frame(neutralise, (neutralise.shape[1],), "neutralise")
        R = R - B @ (B.T @ R)
    total = float(np.linalg.norm(R, "fro") ** 2)
    if total <= 0:
        raise ValueError("realised returns carry no variance after neutralisation")
    return {f"capture_{d}": float(np.linalg.norm(Y[:, :d].T @ R, "fro") ** 2 / total)
            for d in dims}


def realised_ceiling(realised, dims=DEFAULT_DIMS):
    """Exact attainable maximum: leading eigenvectors of R_out R_out^T.

    This maximises captured variance by construction, so it is the ceiling and
    not an approximation.  No rank condition on T_out is required because the
    object is never claimed to estimate a population quantity -- it is simply
    the best in-sample d-dimensional subspace for that window.  See
    ``ceiling_bias`` for how much of the resulting headroom is overfitting.
    """
    R = np.asarray(realised, dtype=float)
    return np.linalg.svd(R, full_matrices=False)[0][:, :max(dims)]


def random_floor(n_assets, dims=DEFAULT_DIMS):
    """Expected capture of a Haar-random frame: exactly d/N."""
    return {f"capture_{d}": d / float(n_assets) for d in dims}


def ceiling_bias(covariance, horizon, dims=DEFAULT_DIMS, replicates=200,
                 rng=None):
    """Headroom the in-sample ceiling reports when the subspace does NOT move.

    Simulates ``horizon`` returns from ``covariance`` itself, so the frozen
    frame is exactly correct by construction and the honest headroom is zero.
    Whatever ``ceiling - frozen`` comes out of that simulation is pure
    overfitting of the finite target window, and must be subtracted from the
    measured headroom before quoting a skill fraction.

    Measured at T_out=42 this is +0.106 to +0.119 across all four panels while
    N varies 16x, i.e. roughly 70% of the naive headroom.
    """
    C = np.asarray(covariance, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"covariance must be square, got {C.shape}")
    if horizon < 2:
        raise ValueError("horizon must be at least 2")
    rng = np.random.default_rng() if rng is None else rng
    values, vectors = np.linalg.eigh(C)
    frozen = vectors[:, ::-1][:, :max(dims)]
    root = vectors * np.sqrt(np.clip(values, 0.0, None))
    out = {f"capture_{d}": [] for d in dims}
    for _ in range(int(replicates)):
        simulated = root @ rng.standard_normal((C.shape[0], int(horizon)))
        top = realised_ceiling(simulated, dims)
        gap = variance_captured(top, simulated, dims)
        base = variance_captured(frozen, simulated, dims)
        for d in dims:
            out[f"capture_{d}"].append(gap[f"capture_{d}"] - base[f"capture_{d}"])
    return {key: float(np.mean(value)) for key, value in out.items()}


def skill(model_capture, frozen_capture, ceiling_capture, bias=0.0):
    """Fraction of *honest* achievable headroom closed.

    ``bias`` is the ``ceiling_bias`` for the same origin and level.  Passing
    zero reproduces the naive skill, which overstates the denominator by
    roughly 3.4x at T_out=42 and should never be quoted alone.
    """
    headroom = float(ceiling_capture) - float(frozen_capture) - float(bias)
    if headroom <= 0:
        return np.nan
    return float((model_capture - frozen_capture) / headroom)


def equal_weight_market(n_assets):
    """The common, origin-known neutralisation direction ``1/sqrt(N)``."""
    return np.ones((int(n_assets), 1)) / np.sqrt(float(n_assets))
