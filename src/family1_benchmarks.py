"""Shared estimators and scoring for Stage 2 Flag benchmarks."""
from collections import OrderedDict

import numpy as np
import pandas as pd

from src.data import standardise, to_correlation_panel
from src.erse import erse
from src.flag import DEFAULT_DIMS
from src.forecast import frozen_flag_losses
from src.oracle_line import align_flag_frame, ordered_minimum_plane_transport
from src.overlap import sample_covariance, spectral


COMPONENTS = (
    "market_1", "block_2_3", "block_4_6", "top_3", "top_6",
    "flag_nested",
)
DEFAULT_DAMPING_ALPHAS = tuple(np.linspace(0.0, 1.0, 41))


def correlation_returns(returns):
    """Return the exact standardised panel used to construct Stage 1 Flags."""
    returns = np.asarray(returns, dtype=float)
    return to_correlation_panel(standardise(returns, window=1))


def constant_velocity_flag(past, current):
    """Repeat the minimum-plane motion from ``past`` to ``current`` once.

    The block alignment removes arbitrary bases inside [1], [2:3] and [4:6].
    Applying the same orthogonal transport a second time produces one valid
    nested Flag rather than three independently extrapolated Grassmann spaces.
    """
    aligned_current = align_flag_frame(past, current)
    transport = ordered_minimum_plane_transport(past, aligned_current)
    return transport @ aligned_current


def _apply_plane_rotation(frame, source, companion, angle):
    """Apply one real plane rotation without constructing an N by N matrix."""
    frame = np.asarray(frame, dtype=float)
    source = np.asarray(source, dtype=float)
    companion = np.asarray(companion, dtype=float)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    source_coordinates = source @ frame
    companion_coordinates = companion @ frame
    return (frame
            + (cosine - 1.0) * (
                np.outer(source, source_coordinates)
                + np.outer(companion, companion_coordinates))
            + sine * (
                np.outer(companion, source_coordinates)
                - np.outer(source, companion_coordinates)))


def _ordered_rotation_steps(past, current, atol=1e-10):
    """Minimum-plane rotation steps carrying ``past`` to aligned ``current``.

    The steps are the low-rank form of ``ordered_minimum_plane_transport``.
    Keeping them explicitly lets a scalar fraction of the complete ordered
    Flag motion be applied without a dense matrix logarithm or independently
    extrapolating the three nested Grassmann components.
    """
    past = np.asarray(past, dtype=float)
    current = np.asarray(current, dtype=float)
    if past.ndim != 2 or past.shape != current.shape:
        raise ValueError("past and current must be equal-sized frames")
    transported = np.array(past, copy=True)
    steps = []
    for column in range(past.shape[1]):
        source = transported[:, column]
        source = source / np.linalg.norm(source)
        destination = current[:, column]
        destination = destination / np.linalg.norm(destination)
        cosine = float(np.clip(source @ destination, -1.0, 1.0))
        angle = float(np.arccos(cosine))
        sine = float(np.sin(angle))
        if sine <= atol and cosine > 0:
            continue
        if sine <= atol:
            raise ValueError("fractional Flag transport is undefined at an antipode")
        companion = (destination - cosine * source) / sine
        steps.append((source, companion, angle))
        transported = _apply_plane_rotation(
            transported, source, companion, angle)
    if not np.allclose(
            transported.T @ transported, np.eye(past.shape[1]), atol=2e-8):
        raise RuntimeError("fractional Flag transport lost orthonormality")
    if not np.allclose(transported, current, atol=2e-8):
        raise RuntimeError("ordered rotation steps did not reach current Flag")
    return steps


def damped_velocity_flag(past, current, alpha):
    """Continue the previous complete Flag rotation by one scalar fraction.

    ``alpha=0`` is Frozen Flag and ``alpha=1`` is Constant Velocity.  A single
    ordered orthogonal motion is damped, preserving nesting throughout; the
    market, core and buffer spaces are not forecast independently.
    """
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    aligned_current = align_flag_frame(past, current)
    prediction = np.array(aligned_current, copy=True)
    for source, companion, angle in _ordered_rotation_steps(
            past, aligned_current):
        prediction = _apply_plane_rotation(
            prediction, source, companion, alpha * angle)
    prediction = np.linalg.qr(prediction, mode="reduced")[0]
    return prediction


def select_global_damping_alpha(frames, validation_examples, offset,
                                alphas=DEFAULT_DAMPING_ALPHAS):
    """Select one full-Flag damping coefficient on validation targets only."""
    frames = np.asarray(frames, dtype=float)
    examples = tuple(sorted({int(value) for value in validation_examples}))
    offset = int(offset)
    alphas = tuple(float(value) for value in alphas)
    if offset <= 0:
        raise ValueError("offset must be positive")
    if not examples:
        raise ValueError("validation_examples cannot be empty")
    if not alphas or any(not np.isfinite(value) or not 0 <= value <= 1
                         for value in alphas):
        raise ValueError("alphas must be finite values in [0, 1]")
    rows = []
    for alpha in alphas:
        losses = []
        for example in examples:
            current_index = example + offset
            if (current_index - offset < 0 or
                    current_index + offset >= len(frames)):
                raise ValueError("validation example is outside the Flag history")
            prediction = damped_velocity_flag(
                frames[current_index - offset], frames[current_index], alpha)
            losses.append(frozen_flag_losses(
                frames[current_index + offset], prediction)["flag_nested"])
        rows.append({
            "alpha": alpha,
            "validation_mean_complete_flag_loss": float(np.mean(losses)),
            "n_validation_examples": len(losses),
        })
    tuning = pd.DataFrame(rows).sort_values(
        ["validation_mean_complete_flag_loss", "alpha"],
        kind="stable").reset_index(drop=True)
    selected = float(tuning.iloc[0]["alpha"])
    tuning["selected"] = tuning["alpha"] == selected
    return selected, tuning


def erse_flag(returns, delta=.25):
    """Leading partial Flag of ERSE fitted to one current rolling window."""
    adjusted = correlation_returns(returns)
    correlation = sample_covariance(adjusted)
    return erse(correlation, delta)["corrected_vectors"][:, :DEFAULT_DIMS[-1]]


def filtered_flag(returns, estimator):
    """Leading partial Flag of a covariance filter on the Stage 1 panel."""
    covariance = estimator(correlation_returns(returns))
    _, vectors = spectral(covariance)
    return vectors[:, :DEFAULT_DIMS[-1]]


def covariance_to_correlation(covariance):
    """Normalise a positive covariance estimate without changing information."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    diagonal = np.diag(covariance)
    floor = max(float(np.max(np.abs(diagonal))) * 1e-12,
                np.finfo(float).tiny)
    scale = np.sqrt(np.maximum(diagonal, floor))
    correlation = covariance / np.outer(scale, scale)
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)
    return correlation


def covariance_flag(covariance):
    """Leading partial Flag of a covariance forecast's correlation matrix."""
    _, vectors = spectral(covariance_to_correlation(covariance))
    return vectors[:, :DEFAULT_DIMS[-1]]


def retained_window_flag(returns, horizon):
    """Flag of observations known to remain in the future rolling window."""
    returns = np.asarray(returns, dtype=float)
    if not 0 < horizon < returns.shape[1] - 1:
        raise ValueError("horizon must leave at least two retained observations")
    retained = correlation_returns(returns[:, int(horizon):])
    _, vectors = spectral(sample_covariance(retained))
    return vectors[:, :DEFAULT_DIMS[-1]]


def stationary_roll_forward_flag(returns, horizon):
    """Known retained scatter plus a stationary fill for unseen returns.

    The oldest ``horizon`` observations are known to leave.  Their replacements
    are unknown, so their expected scatter is filled with the current causal
    correlation estimate.  Only the resulting Flag is returned and scored.
    """
    returns = np.asarray(returns, dtype=float)
    horizon = int(horizon)
    T = returns.shape[1]
    if not 0 < horizon < T - 1:
        raise ValueError("horizon must leave at least two retained observations")
    adjusted = correlation_returns(returns)
    centred = adjusted - adjusted.mean(axis=1, keepdims=True)
    current = sample_covariance(adjusted)
    retained = centred[:, horizon:]
    forecast = (retained @ retained.T + horizon * current) / T
    return covariance_flag(forecast)


def ewma_flag(returns, *, decay=None, half_life=None):
    """Flag of a causal EWMA correlation forecast."""
    from src.covariance_benchmarks import estimate_ewma

    adjusted = standardise(np.asarray(returns, dtype=float), window=1)
    covariance = estimate_ewma(adjusted, decay=decay, half_life=half_life)
    return covariance_flag(covariance)


def factor_cm_iewma_flag(returns, *, factor_rank=20,
                         pairs=None, combination_window=10):
    """Large-universe factor CM-IEWMA forecast, scored only through its Flag.

    This follows the authors' large-universe construction: causal rolling PCA,
    CM-IEWMA on the factor returns, and an EWMA diagonal for unexplained asset
    returns.  The official ``cvxcovariance`` implementation performs the
    iterated EWMAs and convex precision-factor combination.
    """
    try:
        from cvx.covariance.combination import from_sigmas
        from cvx.covariance.ewma import iterated_ewma
    except ImportError as exc:  # pragma: no cover - dependency failure message
        raise ImportError("factor CM-IEWMA requires cvxcovariance") from exc
    from src.covariance_benchmarks import estimate_ewma

    returns = np.asarray(returns, dtype=float)
    adjusted = correlation_returns(returns)
    N, T = adjusted.shape
    rank = min(int(factor_rank), N - 1, T - 2)
    if rank < DEFAULT_DIMS[-1]:
        raise ValueError("factor CM-IEWMA needs rank at least six")
    if pairs is None:
        # Exact three-expert grid used in the paper's large-universe factor
        # experiments, specialised to the selected factor rank.
        pairs = ((max(1, int(np.ceil(rank / 2))), rank),
                 (rank, 3 * rank), (3 * rank, 6 * rank))

    _, exposure = spectral(sample_covariance(adjusted))
    exposure = exposure[:, :rank]
    factors_array = (exposure.T @ adjusted).T
    index = pd.RangeIndex(T)
    factors = pd.DataFrame(factors_array, index=index,
                           columns=[f"factor_{i}" for i in range(rank)])
    min_volatility = rank
    min_correlation = 2 * rank
    experts = {}
    for volatility_half_life, correlation_half_life in pairs:
        sequence = iterated_ewma(
            factors, vola_halflife=volatility_half_life,
            cov_halflife=correlation_half_life,
            min_periods_vola=min_volatility,
            min_periods_cov=min_correlation, mean=False)
        experts[f"{volatility_half_life}-{correlation_half_life}"] = {
            result.time: result.covariance for result in sequence}

    fastest = f"{pairs[0][0]}-{pairs[0][1]}"
    for time, covariance in experts[fastest].items():
        values = covariance.to_numpy(float)
        experts[fastest][time] = pd.DataFrame(
            values + .05 * np.diag(np.diag(values)),
            index=covariance.index, columns=covariance.columns)

    available = sorted(set.intersection(
        *(set(sequence) for sequence in experts.values())))
    if len(available) < combination_window:
        raise ValueError("not enough CM-IEWMA expert history")
    results = list(from_sigmas(experts, factors, means=None).solve(
        window=int(combination_window), times=[available[-1]],
        solver="CLARABEL"))
    if not results or results[-1] is None:
        raise RuntimeError("CM-IEWMA convex combination did not converge")
    factor_covariance = results[-1].covariance.to_numpy(float)

    fitted = exposure @ factors_array.T
    residual = adjusted - fitted
    residual_covariance = estimate_ewma(residual, half_life=21.0)
    diagonal = np.maximum(np.diag(residual_covariance),
                          np.finfo(float).tiny)
    covariance = (exposure @ factor_covariance @ exposure.T +
                  np.diag(diagonal))
    return covariance_flag(covariance)


def long_flag_loss_rows(example, target, prediction, **metadata):
    """One long-format row for every disjoint/cumulative Flag loss."""
    losses = frozen_flag_losses(target, prediction)
    return [dict(metadata, example=int(example), component=component,
                 loss=float(losses[component]))
            for component in COMPONENTS]


def circular_block_interval(values, block_length, repetitions, rng):
    """Circular-block interval for a dependent paired mean."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a non-empty vector")
    block_length = min(max(1, int(block_length)), len(values))
    blocks = int(np.ceil(len(values) / block_length))
    starts = rng.integers(0, len(values), size=(int(repetitions), blocks))
    offsets = np.arange(block_length)
    indices = ((starts[:, :, None] + offsets) % len(values)).reshape(
        int(repetitions), -1)[:, :len(values)]
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [.025, .975]))


def compare_with_frozen(model, frozen, *, T, horizon, step,
                        repetitions=2000, seed=20260803):
    """Complete evidence table for one model paired with Frozen Flag."""
    keys = ["label", "example", "component"]
    left = model.sort_values(keys)
    right = frozen.sort_values(keys)
    merged = left.merge(right[keys + ["loss"]], on=keys,
                        suffixes=("_model", "_frozen"),
                        validate="one_to_one")
    block_length = int(np.ceil((T + 2 * horizon) / step))
    rng = np.random.default_rng(seed)
    rows = []
    for (label, component), group in merged.groupby(
            ["label", "component"], sort=False):
        model_loss = group["loss_model"].to_numpy(float)
        frozen_loss = group["loss_frozen"].to_numpy(float)
        difference = frozen_loss - model_loss
        frozen_mean = float(np.mean(frozen_loss))
        model_mean = float(np.mean(model_loss))
        lower, upper = circular_block_interval(
            difference, block_length, repetitions, rng)
        scale = max(frozen_mean, np.finfo(float).tiny)
        rows.append({
            "label": label,
            "component": component,
            "n_examples": len(group),
            "block_length_origins": min(block_length, len(group)),
            "model_mean_loss": model_mean,
            "model_median_loss": float(np.median(model_loss)),
            "model_q25_loss": float(np.quantile(model_loss, .25)),
            "model_q75_loss": float(np.quantile(model_loss, .75)),
            "frozen_mean_loss": frozen_mean,
            "skill_percent": 100.0 * (frozen_mean - model_mean) / scale,
            "origin_win_fraction": float(np.mean(model_loss < frozen_loss)),
            "paired_improvement": float(np.mean(difference)),
            "improvement_ci_low": lower,
            "improvement_ci_high": upper,
            "skill_ci_low": 100.0 * lower / scale,
            "skill_ci_high": 100.0 * upper / scale,
        })
    return pd.DataFrame(rows)


def combine_panel_skills(comparisons):
    """Equal-market macro skill, worst panel and positive-panel count."""
    rows = []
    for component, group in comparisons.groupby("component", sort=False):
        skills = group["skill_percent"].to_numpy(float)
        rows.append({
            "component": component,
            "combined_skill_percent": float(np.mean(skills)),
            "median_panel_skill_percent": float(np.median(skills)),
            "worst_panel_skill_percent": float(np.min(skills)),
            "best_panel_skill_percent": float(np.max(skills)),
            "panels_improved": int(np.sum(skills > 0)),
            "n_panels": int(len(skills)),
        })
    return pd.DataFrame(rows)
