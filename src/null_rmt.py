"""Analytic nulls: what subspace distance does pure sampling noise produce?

Everything here is a formula, not an experiment. These are rulers and
label-generators, not baselines to beat.
"""
import numpy as np

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


def d_null_sample_vs_true(lam, P, Q, T):
    """Eq (7). Expected D between the top-P eigenvectors of the true C and the
    top-Q eigenvectors of one sample covariance built from T observations.

    lam must be the TRUE spectrum, descending. Feeding an estimated spectrum
    here is the project's declared metric-liar #4.
    """
    lam = np.asarray(lam, dtype=float)
    if np.any(np.diff(lam) > 1e-12):
        raise ValueError("lam must be in descending order")
    if not (0 < P <= Q < lam.size):
        raise ValueError(f"need 0 < P <= Q < N, got P={P} Q={Q} N={lam.size}")
    li = lam[:P][:, None]
    lj = lam[Q:][None, :]
    gap = li - lj
    if np.any(np.abs(gap) < 1e-12):
        raise ValueError("exact eigenvalue degeneracy across the P/Q boundary")
    return float(np.sum(li * lj / gap ** 2) / (2.0 * T * P))


def d_null_two_samples(lam_s, lam_t, P, Q, T):
    """Eq (10). Two non-overlapping windows, each with its own spectrum.

    When lam_s == lam_t this reduces to 2 x Eq (7), which is the paper's
    'multiplied by a factor 2' remark.
    """
    return (d_null_sample_vs_true(lam_s, P, Q, T)
            + d_null_sample_vs_true(lam_t, P, Q, T))


def eigenvalue_variogram_null(lam, T):
    """Eq (9). <(lam_i^s - lam_i^t)^2> = 4 lam_i^2 / T for a static C.

    A flat, parameter-free line. Empirical variograms rising above it are the
    cheapest evidence that the true spectrum moves.
    """
    return 4.0 * np.asarray(lam, dtype=float) ** 2 / T


def d_random_subspaces(P, Q, N, convention="normalised", n=200001):
    """Accidental-overlap benchmark for two uniformly random subspaces.

    The squared principal cosines follow a Jacobi/MANOVA law supported on
    [gamma_-, gamma_+].

    convention='normalised' divides by alpha*pi, which makes the density
    integrate to 1 and so makes the result a genuine mean of -ln sigma over the
    P singular values. This is what subspace_distance() computes.

    convention='paper' divides by beta*pi, reproducing the (unnumbered) D_RMT
    display on p.2 of arXiv:1108.4258 and the 0.83 quoted in its Fig. 2 caption
    for (P,Q,N) = (5,10,204). That paper only quotes the formula; it originates
    in its ref [6], Bouchaud, Laloux, Miceli & Potters, EPJB 55 (2007) 201.
    Note this is NOT Eq (9) of arXiv:1108.4258 -- Eq (9) is the eigenvalue
    variogram implemented above. That density carries mass
    alpha/beta = P/Q, so the value is P/Q times the normalised one. Provided for
    cross-checking against the paper only -- do not compare it against your own
    directly-computed D.
    """
    if convention not in ("normalised", "paper"):
        raise ValueError(f"unknown convention {convention!r}")
    a, b = P / N, Q / N
    r = 2.0 * np.sqrt(a * b * (1 - a) * (1 - b))
    g_lo, g_hi = a + b - 2 * a * b - r, a + b - 2 * a * b + r
    lo, hi = np.sqrt(max(g_lo, 0.0)), np.sqrt(g_hi)
    if hi >= 1.0:
        raise ValueError("support reaches sigma = 1; P+Q too large relative to N")

    t = np.linspace(0.0, np.pi / 2, n)
    s = lo + (hi - lo) * np.sin(t) ** 2
    ds = (hi - lo) * 2 * np.sin(t) * np.cos(t)
    num = np.sqrt(np.clip(s * s - g_lo, 0, None) * np.clip(g_hi - s * s, 0, None))
    scale = a if convention == "normalised" else b
    density = num / (scale * np.pi * s * (1 - s * s))
    return float(_trapz(-np.log(s) * density * ds, t))


def _random_subspace_mass(P, Q, N, convention="normalised", n=200001):
    """Total mass of the benchmark density. Should be 1.0 for 'normalised'."""
    a, b = P / N, Q / N
    r = 2.0 * np.sqrt(a * b * (1 - a) * (1 - b))
    g_lo, g_hi = a + b - 2 * a * b - r, a + b - 2 * a * b + r
    lo, hi = np.sqrt(max(g_lo, 0.0)), np.sqrt(g_hi)
    t = np.linspace(0.0, np.pi / 2, n)
    s = lo + (hi - lo) * np.sin(t) ** 2
    ds = (hi - lo) * 2 * np.sin(t) * np.cos(t)
    num = np.sqrt(np.clip(s * s - g_lo, 0, None) * np.clip(g_hi - s * s, 0, None))
    scale = a if convention == "normalised" else b
    return float(_trapz(num / (scale * np.pi * s * (1 - s * s)) * ds, t))
