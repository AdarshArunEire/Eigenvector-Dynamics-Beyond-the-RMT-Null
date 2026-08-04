"""Oracle ceilings connecting partial-Flag forecasts to full covariance.

The Oracle Line is deliberately infeasible.  It adds future information one
piece at a time while keeping a common reconstruction scaffold, so a failure
can be attributed to Flag geometry, the spectrum, the unmodelled complement,
or marginal volatility rather than to a change of estimator elsewhere.
"""
from dataclasses import dataclass

import numpy as np

from src.covariance_benchmarks import (
    estimate_ewma, estimate_nonlinear_shrinkage, realised_covariance)
from src.data import standardise, to_correlation_panel
from src.flag import DEFAULT_DIMS, validate_flag_frame
from src.overlap import sample_covariance, spectral


@dataclass(frozen=True)
class CorrelationState:
    """Empirical Flag plus QIS-cleaned eigenvalues for one rolling window."""

    empirical: np.ndarray
    vectors: np.ndarray
    cleaned_values: np.ndarray
    cleaned_correlation: np.ndarray

    @property
    def flag(self):
        return self.vectors[:, :DEFAULT_DIMS[-1]]


@dataclass(frozen=True)
class OracleForecasts:
    """Forecast matrices and the intermediate objects needed for auditing."""

    covariance: dict
    correlation: dict
    input_flag: dict
    current_state: CorrelationState
    future_state: CorrelationState


def covariance_to_correlation(covariance):
    """Return the unit-diagonal correlation associated with an SPD matrix."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    covariance = (covariance + covariance.T) / 2
    diagonal = np.diag(covariance)
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise ValueError("covariance must have finite positive diagonal")
    inverse_scale = 1.0 / np.sqrt(diagonal)
    correlation = (inverse_scale[:, None] * covariance) * inverse_scale[None, :]
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)
    return correlation


def correlation_state(returns):
    """Reproduce Stage 1 correlation geometry and attach a QIS spectrum.

    QIS is used only as a stable eigenvalue scaffold.  Its guarantee is not
    inherited after rotation; every reconstructed matrix is judged directly
    out of sample.
    """
    returns = np.asarray(returns, dtype=float)
    adjusted = to_correlation_panel(standardise(returns, window=1))
    empirical = sample_covariance(adjusted)
    _, vectors = spectral(empirical)
    qis = estimate_nonlinear_shrinkage(adjusted)
    representation = vectors.T @ qis @ vectors
    off_diagonal = representation - np.diag(np.diag(representation))
    tolerance = 1e-7 * max(np.linalg.norm(qis, "fro"), 1.0)
    if np.linalg.norm(off_diagonal, "fro") > tolerance:
        raise RuntimeError("QIS did not retain the empirical eigenvectors")
    cleaned_values = np.diag(representation).copy()
    floor = max(float(np.max(cleaned_values)) * 1e-12,
                np.finfo(float).tiny)
    cleaned_values = np.maximum(cleaned_values, floor)
    cleaned = (vectors * cleaned_values) @ vectors.T
    return CorrelationState(
        empirical=empirical,
        vectors=vectors,
        cleaned_values=cleaned_values,
        cleaned_correlation=covariance_to_correlation(cleaned),
    )


def align_flag_frame(current, target, dims=DEFAULT_DIMS):
    """Choose the target representative using only its Flag subspaces.

    Each disjoint Flag block is Procrustes-aligned to the corresponding current
    block.  Rotating or reflecting a supplied target basis inside a block thus
    cannot change the result.
    """
    current = validate_flag_frame(current, dims, "current")
    target = validate_flag_frame(target, dims, "target")
    if current.shape != target.shape:
        raise ValueError("current and target Flag frames have different shapes")
    aligned = np.empty_like(target)
    start = 0
    for stop in dims:
        current_block = current[:, start:stop]
        target_block = target[:, start:stop]
        left, _, right_t = np.linalg.svd(target_block.T @ current_block)
        aligned[:, start:stop] = target_block @ (left @ right_t)
        start = stop
    return aligned


def ordered_minimum_plane_transport(current, target, atol=1e-10):
    """Construct a deterministic orthogonal lift mapping one frame to another.

    Successive minimum-plane rotations respect the ordered Flag blocks.  The
    target must already be block-aligned with :func:`align_flag_frame`.
    """
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.ndim != 2 or current.shape != target.shape:
        raise ValueError("current and target must be equal-sized frames")
    n, width = current.shape
    identity = np.eye(n)
    transport = identity.copy()
    fixed = []
    for column in range(width):
        source = transport @ current[:, column]
        destination = target[:, column]
        source /= np.linalg.norm(source)
        destination /= np.linalg.norm(destination)
        cosine = float(np.clip(source @ destination, -1.0, 1.0))
        sine = float(np.sqrt(max(0.0, 1.0 - cosine ** 2)))
        if sine <= atol and cosine > 0:
            fixed.append(destination)
            continue
        if sine <= atol:
            # A deterministic pi rotation handles the exceptional antipode.
            candidates = identity.copy()
            if fixed:
                fixed_frame = np.column_stack(fixed)
                candidates -= fixed_frame @ (fixed_frame.T @ candidates)
            candidates -= source[:, None] * (source @ candidates)[None, :]
            norms = np.linalg.norm(candidates, axis=0)
            index = int(np.argmax(norms))
            if norms[index] <= atol:
                raise ValueError("cannot construct antipodal transport")
            companion = candidates[:, index] / norms[index]
            plane = np.outer(source, source) + np.outer(companion, companion)
            rotation = identity - 2.0 * plane
        else:
            companion = (destination - cosine * source) / sine
            rotation = (
                identity
                + (cosine - 1.0) * (
                    np.outer(source, source) + np.outer(companion, companion))
                + sine * (
                    np.outer(companion, source) - np.outer(source, companion))
            )
        transport = rotation @ transport
        fixed.append(destination)
    if not np.allclose(transport.T @ transport, identity, atol=2e-8):
        raise RuntimeError("Flag transport is not orthogonal")
    if not np.allclose(transport @ current, target, atol=2e-8):
        raise RuntimeError("Flag transport did not reach the aligned target")
    return transport


def reconstruct_from_flag(current_state, target_flag, eigenvalues):
    """Install a target Flag in a full isospectral correlation scaffold."""
    current_flag = current_state.flag
    target_flag = validate_flag_frame(target_flag, DEFAULT_DIMS, "target")
    aligned = align_flag_frame(current_flag, target_flag)
    transport = ordered_minimum_plane_transport(current_flag, aligned)
    rotated_vectors = transport @ current_state.vectors
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.shape != (rotated_vectors.shape[1],):
        raise ValueError("one eigenvalue is required per full-basis direction")
    if np.any(eigenvalues <= 0) or not np.isfinite(eigenvalues).all():
        raise ValueError("reconstruction eigenvalues must be finite and positive")
    covariance = (rotated_vectors * eigenvalues) @ rotated_vectors.T
    return covariance_to_correlation(covariance), aligned


def assemble_covariance(correlation, volatilities):
    """Combine a correlation forecast with positive marginal volatilities."""
    correlation = np.asarray(correlation, dtype=float)
    volatilities = np.asarray(volatilities, dtype=float)
    if correlation.shape != (len(volatilities), len(volatilities)):
        raise ValueError("correlation and volatility dimensions differ")
    if np.any(volatilities <= 0) or not np.isfinite(volatilities).all():
        raise ValueError("volatilities must be finite and positive")
    covariance = (volatilities[:, None] * correlation) * volatilities[None, :]
    return (covariance + covariance.T) / 2


def build_oracle_forecasts(current_returns, future_rolling_returns,
                           realised_returns, ewma_half_life):
    """Build the common control and four progressively informed oracles.

    Oracle 1 knows the exact future rolling Flag.  Oracle 2 additionally knows
    the future-window QIS-cleaned spectrum.  Oracle 3 receives the complete
    future-window QIS-cleaned correlation.  Oracle 4 additionally receives the
    realised next-horizon marginal volatilities.
    """
    current_returns = np.asarray(current_returns, dtype=float)
    future_rolling_returns = np.asarray(future_rolling_returns, dtype=float)
    realised_returns = np.asarray(realised_returns, dtype=float)
    if current_returns.shape != future_rolling_returns.shape:
        raise ValueError("current and future rolling windows must have equal shape")
    if current_returns.shape[0] != realised_returns.shape[0]:
        raise ValueError("rolling and realised panels have different assets")

    current = correlation_state(current_returns)
    future = correlation_state(future_rolling_returns)
    o1_correlation, aligned = reconstruct_from_flag(
        current, future.flag, current.cleaned_values)
    o2_correlation, _ = reconstruct_from_flag(
        current, future.flag, future.cleaned_values)

    past_volatility_covariance = estimate_ewma(
        current_returns, half_life=float(ewma_half_life))
    past_volatilities = np.sqrt(np.maximum(
        np.diag(past_volatility_covariance), np.finfo(float).tiny))
    realised = realised_covariance(realised_returns)
    future_volatilities = np.sqrt(np.maximum(
        np.diag(realised), np.finfo(float).tiny))

    correlations = {
        "Control - frozen Flag/QIS/EWMA": current.cleaned_correlation,
        "Oracle 1 - future Flag": o1_correlation,
        "Oracle 2 - future Flag and spectrum": o2_correlation,
        "Oracle 3 - future rolling correlation": future.cleaned_correlation,
        "Oracle 4 - future correlation and scale": future.cleaned_correlation,
    }
    input_flags = {
        "Control - frozen Flag/QIS/EWMA": current.flag,
        "Oracle 1 - future Flag": aligned,
        "Oracle 2 - future Flag and spectrum": aligned,
        "Oracle 3 - future rolling correlation": future.flag,
        "Oracle 4 - future correlation and scale": future.flag,
    }
    covariance = {}
    for name, correlation in correlations.items():
        volatilities = (future_volatilities if name.startswith("Oracle 4")
                        else past_volatilities)
        covariance[name] = assemble_covariance(correlation, volatilities)
    return OracleForecasts(
        covariance=covariance,
        correlation=correlations,
        input_flag=input_flags,
        current_state=current,
        future_state=future,
    )
