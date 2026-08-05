"""Metric-aware first-order coupling for the realised-variance capture score.

The capture score is linear in the rank-d projector,

    capture = <P, S_out> / tr S_out = <P, S~_out>,   P = Y Y^T,

so the Bayes-optimal frame is the top-d eigenspace of M_t = E[S~_out | F_t].
Every entrant on the Stage 2 ladder estimates that expectation by a weighted
realised second moment, C_t = sum_k w_k r_{t-k} r_{t-k}^T.  This module answers
the next question: given C_t, which symmetric corrections can possibly change
the score, and how do you fit them.

**The visible block.**  Write C_t = sum_i lambda_i u_i u_i^T with U_6 the top
six eigenvectors and U_perp the remaining N-6.  Perturb by C_t + eps G.  The
Riesz spectral projector for the group {lambda_1..lambda_6} has first variation

    dP = eps sum_{i<=6} sum_{j>6} (u_j^T G u_i)/(lambda_i - lambda_j)
                                   (u_j u_i^T + u_i u_j^T) + O(eps^2),

with the within-group terms cancelling identically because G is symmetric and
the energy denominators antisymmetrise.  Therefore dP -- and so the entire
change in score -- depends on G only through the off-diagonal block

    G^oi := U_perp^T G U_6   in R^{(N-6) x 6}.

The within-top-6 block, the within-complement block, and the whole diagonal are
invisible to the metric.  Two consequences are load-bearing for this project.

1.  Every rotationally-invariant estimator -- Ledoit-Wolf, OAS, QIS -- is by
    definition an eigenvalue map with the sample eigenvectors held fixed, i.e.
    G is diagonal in the sample eigenbasis and G^oi = 0.  Its score change is
    zero exactly, not approximately, and not merely to first order: a diagonal
    perturbation in the eigenbasis does not rotate eigenvectors at all as long
    as the ordering is preserved.  ``rie_correction_is_invisible`` checks this
    numerically.  The measured agreement of QIS / LW / OAS to four decimals is
    therefore structural, and the residual is preprocessing noise.

2.  The score change is an inner product against a *constructible label*,

        d(capture) = 2 eps < G^oi / Lambda, Sigma^oi_out >,
        Lambda_{ji} = lambda_i - lambda_j,  Sigma^oi_out = U_perp^T S~_out U_6.

    Sigma^oi_out can be computed at every historical origin from the realised
    window, so fitting a correction is supervised learning on a matrix of known
    shape rather than regression on a manifold-valued object.

**The eigengap warning.**  The 1/(lambda_i - lambda_j) weight diverges as the
6/7 gap closes.  The measured gaps are 0.063 (DAX), 0.061 (CAC), 0.633 (S&P),
so the small panels have the most leverage and the least identified labels at
the same time.  ``eigengap_matrix`` therefore takes an explicit floor and
reports how often it binds; a run in which the floor binds on most origins is
reporting the floor, not the data.

**Parameter budget.**  Learning G^oi freely costs (N-6)*6 = 2106 numbers on
S&P against ~420 origins in ~7 independent blocks.  This module instead builds
a small dictionary of cheap observables A^(m) in the visible block and learns
only the scalars beta_m, m <= 4, plus one amplitude eps and the base kernel
theta.  The family contains Frozen (eps=0, theta -> inf), every EWMA (eps=0),
and constant-velocity-style momentum (beta = e_2).
"""
import numpy as np


from src.data import standardise, to_correlation_panel
from src.grassmann import grassmann_log
from src.overlap import sample_covariance

RANK = 6

#: Default dictionary.  ``hierarchy`` is off by default so the standard run
#: spends exactly five parameters (three betas, eps, theta) as predeclared.
DEFAULT_FEATURES = ("fast_slow", "momentum", "stress")
ALL_FEATURES = ("fast_slow", "momentum", "stress", "hierarchy")


# --------------------------------------------------------------------------
# spectral primitives
# --------------------------------------------------------------------------

def descending_spectrum(matrix):
    """Full symmetric eigendecomposition, eigenvalues descending."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got {matrix.shape}")
    symmetric = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    return values[::-1].copy(), vectors[:, ::-1].copy()


def leading_frame(matrix, rank=RANK):
    """Top-``rank`` eigenvectors, descending.

    Uses the full LAPACK divide-and-conquer driver and slices, rather than a
    subset solver.  That is deliberate and measured: at N=357 the ``evr``
    subset path costs 45 ms against 11 ms for the full ``numpy.linalg.eigh``,
    because the subset machinery's fixed overhead dominates long before the
    O(N^3) saving appears.  The ladder makes this call several thousand times
    per panel, so the naive-looking choice is four times faster.
    """
    matrix = np.asarray(matrix, dtype=float)
    symmetric = (matrix + matrix.T) / 2
    return np.linalg.eigh(symmetric)[1][:, ::-1][:, :int(rank)]


def eigengap_matrix(values, rank=RANK, floor=1e-6):
    """``Lambda[j, i] = lambda_i - lambda_{rank+j}`` for the visible block.

    Parameters
    ----------
    values : descending eigenvalues of the base estimator.
    floor : lower clip on the gap.  The perturbative weight is 1/Lambda, so an
        unclipped near-degenerate pair would dominate every fit for reasons
        that have nothing to do with predictability.  Returned alongside the
        matrix is the fraction of entries the floor actually bound, which is a
        diagnostic that must be reported, not silently swallowed.
    """
    values = np.asarray(values, dtype=float)
    rank = int(rank)
    if values.ndim != 1 or values.size <= rank:
        raise ValueError(f"need more than rank={rank} eigenvalues, got {values.shape}")
    if np.any(np.diff(values) > 1e-10):
        raise ValueError("values must be sorted descending")
    if floor <= 0:
        raise ValueError("floor must be positive")
    gaps = values[:rank][None, :] - values[rank:][:, None]
    bound = float(np.mean(gaps < floor))
    return np.maximum(gaps, floor), bound


def visible_block(matrix, vectors, rank=RANK):
    """``U_perp^T X U_6`` -- the only part of a correction the metric sees."""
    matrix = np.asarray(matrix, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    return vectors[:, rank:].T @ matrix @ vectors[:, :rank]


def realised_visible_target(realised, vectors, rank=RANK):
    """``Sigma^oi_out = U_perp^T (R R^T / ||R||_F^2) U_6``.

    Formed from the thin products, never from the N x N second moment: the
    label for a 357-name panel is 351 x 6, but S_out itself is 357 x 357 and
    rank 42.  Trace-normalised, so volatility level cancels exactly -- which is
    why the oracle line found marginal volatility to dominate the *covariance*
    problem while contributing nothing here.
    """
    realised = np.asarray(realised, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if realised.ndim != 2:
        raise ValueError(f"realised must be N x T_out, got {realised.shape}")
    if realised.shape[0] != vectors.shape[0]:
        raise ValueError("realised and vectors disagree on N")
    total = float(np.linalg.norm(realised, "fro") ** 2)
    if total <= 0:
        raise ValueError("realised returns carry no variance")
    return (vectors[:, rank:].T @ realised) @ (vectors[:, :rank].T @ realised).T / total


def score_gradient(block, gaps, target):
    """``d(capture)/d(eps)`` for one unit correction: ``2 <A/Lambda, Sigma^oi>``."""
    block = np.asarray(block, dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    target = np.asarray(target, dtype=float)
    if not (block.shape == gaps.shape == target.shape):
        raise ValueError(
            f"shapes differ: {block.shape}, {gaps.shape}, {target.shape}")
    return 2.0 * float(np.sum(block / gaps * target))


def corrected_frame(values, vectors, block, epsilon, rank=RANK):
    """Top-``rank`` eigenvectors of ``C + eps (U_perp A U_6^T + U_6 A^T U_perp^T)``.

    Computed in the base eigenbasis, where the corrected matrix is diagonal
    plus an arrow border of ``rank`` columns.  This is exact -- not the
    first-order approximation -- so the model is a genuine estimator and the
    perturbation theory is used only to choose *what* to fit, never to score.
    """
    values = np.asarray(values, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    block = np.asarray(block, dtype=float)
    rank = int(rank)
    n = values.size
    if block.shape != (n - rank, rank):
        raise ValueError(
            f"block must be {(n - rank, rank)}, got {block.shape}")
    epsilon = float(epsilon)
    if epsilon == 0.0:
        return vectors[:, :rank].copy()
    reduced = np.diag(values)
    reduced[rank:, :rank] += epsilon * block
    reduced[:rank, rank:] += epsilon * block.T
    return vectors @ leading_frame(reduced, rank)


def rie_correction_is_invisible(covariance, shrunk, rank=RANK, tol=1e-8):
    """True when ``shrunk`` differs from ``covariance`` by eigenvalues only.

    The structural claim behind the Ledoit-Wolf / OAS / QIS collapse, stated as
    a testable predicate rather than an assertion: if the visible block of the
    difference vanishes, the leading eigenspace -- and therefore the capture
    score -- cannot move.
    """
    values, vectors = descending_spectrum(covariance)
    difference = np.asarray(shrunk, dtype=float) - np.asarray(covariance, dtype=float)
    return float(np.linalg.norm(visible_block(difference, vectors, rank))) <= tol


# --------------------------------------------------------------------------
# features -- cheap observables projected into the visible block
# --------------------------------------------------------------------------

def _correlation_of(window):
    """The exact standardised correlation panel Stage 1 Flags are built from."""
    return to_correlation_panel(standardise(np.asarray(window, dtype=float),
                                            window=1))


def _ewma_correlation(adjusted, half_life):
    from src.covariance_benchmarks import estimate_ewma
    from src.family1_benchmarks import covariance_to_correlation
    return covariance_to_correlation(estimate_ewma(adjusted, half_life=half_life))


def _unit(block, eps=1e-300):
    norm = float(np.linalg.norm(block))
    return block / norm if norm > eps else np.zeros_like(block)


def base_estimator(window, half_life):
    """Base estimator ``C_t(theta)``: causal EWMA correlation of the window.

    ``half_life=inf`` is the exact uniform-weight limit, i.e. Frozen's own
    sample correlation, so the family nests the whole ladder at ``eps=0``.
    """
    return _ewma_correlation(_correlation_of(window), half_life)


def feature_fast_slow(window, vectors, rank=RANK, fast=42.0, slow=504.0):
    """Fast-minus-slow EWMA difference, projected onto the visible block.

    This is what EWMA is implicitly exploiting.  Making it an explicit
    correction lets its amplitude be scaled independently of the base kernel,
    which the single-parameter EWMA cannot do.  The 24/24 ladder result says
    this direction carries signal.
    """
    adjusted = _correlation_of(window)
    difference = (_ewma_correlation(adjusted, fast)
                  - _ewma_correlation(adjusted, slow))
    return _unit(visible_block(difference, vectors, rank))


def feature_momentum(previous_frame, vectors, rank=RANK):
    """Realised Grassmann log from the previous frame, in visible coordinates.

    The tangent is taken *at the base estimator's own leading frame*, not at
    the rolling sample Flag, so it lives in the tangent space the correction
    actually acts on.  A Grassmann tangent is horizontal by construction --
    ``U_6^T H = 0`` -- so ``U_perp^T H`` loses nothing: the momentum feature is
    natively visible, and this recovers Model 3.1 as ``beta = e_2``.
    """
    current = np.asarray(vectors, dtype=float)[:, :rank]
    tangent = grassmann_log(current, np.asarray(previous_frame, dtype=float)[:, :rank])
    # Log points backwards along the realised motion; negate so the feature is
    # the direction the subspace has been travelling.
    return _unit(-np.asarray(vectors, dtype=float)[:, rank:].T @ tangent)


def feature_stress(fast_slow_block, stress):
    """Fast/slow direction scaled by standardised log realised variance.

    Lets the frame rotate faster in stress without spending a second direction.
    ``stress`` must be standardised causally -- see ``causal_standardise``.
    """
    return float(stress) * np.asarray(fast_slow_block, dtype=float)


def feature_hierarchy(window, vectors, rank=RANK, **kwargs):
    """Hierarchical-clustering filter minus the sample correlation.

    Structure rather than memory: the only feature in the dictionary that does
    not read the recent past.  Off by default because it costs a clustering per
    origin and adds a fourth scalar.
    """
    from src.covariance_benchmarks import estimate_hcal
    from src.family1_benchmarks import covariance_to_correlation
    adjusted = _correlation_of(window)
    difference = (covariance_to_correlation(estimate_hcal(adjusted, **kwargs))
                  - covariance_to_correlation(sample_covariance(adjusted)))
    return _unit(visible_block(difference, vectors, rank))


def causal_standardise(values, minimum=20):
    """Expanding-window z-scores: entry ``t`` uses only entries ``< t``.

    A full-sample z-score of realised variance would leak the future into
    every origin, including the ones used to fit beta.  Entries before
    ``minimum`` observations are zero, which switches the stress feature off
    rather than letting it fire on two data points.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got {values.shape}")
    out = np.zeros_like(values)
    for index in range(int(minimum), values.size):
        history = values[:index]
        spread = float(np.std(history, ddof=1))
        if spread > 0:
            out[index] = (values[index] - float(np.mean(history))) / spread
    return out


# --------------------------------------------------------------------------
# fitting -- five parameters, both estimators, no autodiff
# --------------------------------------------------------------------------

def gradient_beta(designs, targets, gaps):
    """**Risk loss.**  ``beta_m ∝ sum_t <A^m/Lambda, Sigma^oi>``.

    This is the exact first-order maximiser of realised *capture* on the unit
    sphere.  Because capture is linear in the projector, its first variation is
    linear in beta, so there is no Gram matrix, no local minimum, and each
    feature is loaded in proportion to how often it has historically pointed at
    where the realised variance actually turned out to be.

    Two properties distinguish it from the geometric fits below and they are
    the whole reason all three are computed.  (i) Only the *inner product* with
    the label matters; a prediction that is badly wrong in a direction carrying
    no realised variance costs exactly nothing, because the score cannot see
    it.  (ii) The ``1/Lambda`` weight is not a preconditioner, it is the
    physics: a unit of correction moves the projector by ``1/(lambda_i -
    lambda_j)``, so directions across a small eigengap are cheap to move and
    are therefore worth more per unit of correction.

    The cost of the linearity is that redundancy is not discounted -- two
    collinear features are both loaded near-fully and their shared direction is
    counted twice.  ``ridge_beta`` discounts it.
    """
    designs, targets, gaps = _checked_fit_inputs(designs, targets, gaps)
    scores = np.array([
        sum(score_gradient(design[m], gap, target)
            for design, target, gap in zip(designs, targets, gaps))
        for m in range(designs[0].shape[0])])
    return _normalise_beta(scores)


def ridge_beta(designs, targets, gaps, penalty=1e-3):
    """**Geometry loss, in metric coordinates.**  Gap-weighted least squares.

    Minimises ``sum_t || Sigma^oi_t - sum_m beta_m A^m_t/Lambda_t ||_F^2`` plus
    ``penalty * ||beta||^2``: it asks each feature to *predict* the realised
    visible block, not merely to correlate with it.  Every entry of the target
    is penalised, including entries the capture score would happily ignore, so
    this is subspace-motion prediction rather than risk maximisation -- but it
    is still expressed in the gap-weighted coordinates the metric uses.

    Unlike the gradient fit it inverts the feature Gram, so collinear features
    share one loading instead of duplicating it.  Same parameter count; the
    penalty is fixed, never tuned on test.
    """
    return _least_squares_beta(designs, targets, gaps, penalty, weight=True)


def geometric_beta(designs, targets, gaps, penalty=1e-3):
    """**Pure geometry loss.**  Least squares with the eigengap ignored.

    Identical to ``ridge_beta`` except that ``1/Lambda`` is dropped, so every
    direction of the visible block is weighted equally: this is what a model
    trained to predict *where the subspace goes* would fit, with no reference
    to what the risk metric can see or to how expensive each direction is to
    move.

    It is included precisely so the comparison can be made rather than
    asserted.  If the three fits select near-identical beta the distinction is
    academic and the eigengap weighting is doing nothing; if they diverge, the
    gap between "forecast the geometry" and "forecast the risk" is a measured
    quantity on this data rather than a talking point.  Note the two losses
    coincide only when realised variance is isotropic across the complement,
    which it is not -- complement variance concentrates near the top of the
    bulk, which is also where the eigengaps are smallest.
    """
    return _least_squares_beta(designs, targets, gaps, penalty, weight=False)


def _least_squares_beta(designs, targets, gaps, penalty, weight):
    designs, targets, gaps = _checked_fit_inputs(designs, targets, gaps)
    n_features = designs[0].shape[0]
    gram = np.zeros((n_features, n_features))
    moment = np.zeros(n_features)
    for design, target, gap in zip(designs, targets, gaps):
        weighted = design / gap[None, :, :] if weight else design
        flat = weighted.reshape(n_features, -1)
        gram += flat @ flat.T
        moment += flat @ target.reshape(-1)
    scale = float(np.trace(gram)) / max(n_features, 1)
    solution = np.linalg.solve(
        gram + float(penalty) * max(scale, np.finfo(float).tiny) * np.eye(n_features),
        moment)
    return _normalise_beta(solution)


def direct_capture_beta(evaluate, n_features, n_directions=96, seed=20260805):
    """**Risk loss, exactly.**  Search the sphere for the best realised capture.

    The three closed-form fits above all go through the *first-order* surrogate
    or a squared-error proxy for it.  This one calls ``evaluate(beta)``, which
    is expected to return the true realised capture of the exactly-corrected
    frame, and maximises it over a quasi-uniform grid of directions on the unit
    sphere.  Its only job is to check that the surrogate is not misleading: if
    the closed-form risk fit lands near the direct optimum, the perturbation
    theory is doing its job, and if it does not, the linearisation is the thing
    to distrust rather than the model.

    Antipodal directions are both retained, since the sign of beta is a real
    degree of freedom -- the correction can push the frame either way.
    """
    rng = np.random.default_rng(int(seed))
    directions = rng.standard_normal((int(n_directions), int(n_features)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    scores = np.array([float(evaluate(direction)) for direction in directions])
    if not np.any(np.isfinite(scores)):
        raise ValueError("every candidate direction scored non-finite")
    return directions[int(np.nanargmax(scores))], float(np.nanmax(scores))


def feature_collinearity(designs, gaps):
    """Correlation matrix of the gap-weighted features, pooled over origins.

    Reported so the gradient/ridge divergence is diagnosable rather than
    mysterious.  Off-diagonal entries near one are the case in which the two
    estimators must be expected to disagree.
    """
    designs, gaps = list(designs), list(gaps)
    n_features = designs[0].shape[0]
    gram = np.zeros((n_features, n_features))
    for design, gap in zip(designs, gaps):
        flat = (design / gap[None, :, :]).reshape(n_features, -1)
        gram += flat @ flat.T
    scale = np.sqrt(np.maximum(np.diag(gram), np.finfo(float).tiny))
    return gram / np.outer(scale, scale)


def combine(design, beta):
    """``sum_m beta_m A^(m)`` -- the fitted correction in the visible block."""
    design = np.asarray(design, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if design.shape[0] != beta.size:
        raise ValueError(
            f"design has {design.shape[0]} features, beta has {beta.size}")
    return np.tensordot(beta, design, axes=(0, 0))


def _normalise_beta(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("beta is degenerate; every feature scored exactly zero")
    return vector / norm


def _checked_fit_inputs(designs, targets, gaps):
    designs = [np.asarray(value, dtype=float) for value in designs]
    targets = [np.asarray(value, dtype=float) for value in targets]
    gaps = [np.asarray(value, dtype=float) for value in gaps]
    if not designs:
        raise ValueError("no training origins supplied")
    if not (len(designs) == len(targets) == len(gaps)):
        raise ValueError("designs, targets and gaps have different lengths")
    for design, target, gap in zip(designs, targets, gaps):
        if design.ndim != 3:
            raise ValueError(f"design must be (M, N-6, 6), got {design.shape}")
        if design.shape[1:] != target.shape or target.shape != gap.shape:
            raise ValueError("design, target and gap blocks are not conformable")
    if len({design.shape[0] for design in designs}) != 1:
        raise ValueError("origins disagree on the number of features")
    return designs, targets, gaps
