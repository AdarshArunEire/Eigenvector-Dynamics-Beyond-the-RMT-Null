"""Leakage-safe covariance estimators and scores for Stage 2 benchmarks.

The geometric target and the economic target deliberately share a forecast
origin but not a return window.  Geometry asks for the rolling Flag whose end
date is 42 trading days ahead.  Covariance quality is judged using only the 42
returns strictly after the forecast origin.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.covariance import LedoitWolf, OAS
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from src.forecast import (ChronologicalSplits, assign_chronological_splits,
                          frozen_flag_losses)
from src.overlap import sample_covariance, spectral


@dataclass(frozen=True)
class CovarianceExample:
    """One forecast origin with past-only input and a future realised target."""

    example: int
    current_start: int
    current_end: int
    target_start: int
    target_end: int
    current_date: pd.Timestamp
    target_window_start: pd.Timestamp
    target_date: pd.Timestamp
    estimation_returns: np.ndarray
    realised_returns: np.ndarray


def rolling_covariance_examples(panel, dates, T, horizon=42, step=14):
    """Build examples on the exact origins used by the rolling Flag benchmark.

    The first and last ``horizon`` days are withheld so every row also has the
    past and future rolling Flag required by geometric models.  The realised
    covariance target itself contains exactly the next ``horizon`` returns.
    """
    panel = np.asarray(panel, dtype=float)
    dates = pd.DatetimeIndex(dates)
    if panel.ndim != 2 or panel.shape[1] != len(dates):
        raise ValueError("panel must be N x days and agree with dates")
    if T < 2 or horizon < step or horizon % step:
        raise ValueError("need T >= 2 and horizon a positive multiple of step")
    starts = np.arange(0, panel.shape[1] - T + 1, step, dtype=int)
    offset = horizon // step
    if len(starts) <= 2 * offset:
        raise ValueError("not enough dates for one shared geometry/covariance example")

    examples = []
    for index in range(offset, len(starts) - offset):
        current_start = int(starts[index])
        current_end = current_start + T - 1
        target_start = current_end + 1
        target_end = current_end + horizon
        if target_end >= panel.shape[1]:
            break
        examples.append(CovarianceExample(
            example=len(examples),
            current_start=current_start,
            current_end=current_end,
            target_start=target_start,
            target_end=target_end,
            current_date=dates[current_end],
            target_window_start=dates[target_start],
            target_date=dates[target_end],
            estimation_returns=panel[:, current_start:current_end + 1],
            realised_returns=panel[:, target_start:target_end + 1],
        ))
    if not examples:
        raise ValueError("no complete next-horizon return window is available")
    return examples


def split_covariance_examples(examples,
                              specification=ChronologicalSplits()):
    """Assign the shared examples to frozen chronological splits."""
    metadata = pd.DataFrame([{
        "example": row.example,
        "current_date": row.current_date,
        "target_window_start": row.target_window_start,
        "target_date": row.target_date,
        "component": "covariance",
        "loss": np.nan,
    } for row in examples])
    return assign_chronological_splits(metadata, specification).drop(
        columns=["component", "loss"])


def estimate_sample(returns):
    """Ordinary rolling sample covariance."""
    return sample_covariance(np.asarray(returns, dtype=float))


def estimate_ewma(returns, *, decay=None, half_life=None):
    """Exponentially weighted maximum-likelihood covariance.

    Exactly one of ``decay`` and ``half_life`` is supplied.  Weights sum to one
    and the weighted mean is removed.  No degrees-of-freedom correction is
    applied, matching this project's ``1/T`` sample covariance convention and
    the maximum-likelihood scaling of the other estimators.  Infinite half-life
    is the exact uniform-weight limit.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("returns must be N x T with T >= 2")
    if (decay is None) == (half_life is None):
        raise ValueError("supply exactly one of decay or half_life")
    if half_life is not None:
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        decay = (1.0 if np.isinf(half_life)
                 else float(np.exp(np.log(.5) / half_life)))
    if not 0 < decay <= 1:
        raise ValueError("decay must lie in (0, 1]")
    ages = np.arange(x.shape[1] - 1, -1, -1, dtype=float)
    weights = decay ** ages
    weights /= weights.sum()
    centred = x - x @ weights[:, None]
    return (centred * weights) @ centred.T


def estimate_ledoit_wolf(returns):
    """Ledoit-Wolf linear shrinkage, fitted on observations x assets."""
    x = np.asarray(returns, dtype=float)
    return LedoitWolf(assume_centered=False).fit(x.T).covariance_


def estimate_oas(returns):
    """Oracle Approximating Shrinkage under a Gaussian model."""
    x = np.asarray(returns, dtype=float)
    return OAS(assume_centered=False).fit(x.T).covariance_


def estimate_nonlinear_shrinkage(returns):
    """Ledoit-Wolf quadratic-inverse nonlinear shrinkage (QIS/RIE).

    This is a NumPy implementation of the authors' published QIS formula.  QIS
    retains the empirical eigenvectors, nonlinearly cleans every eigenvalue and
    remains invertible when demeaning makes ``p >= n``.  Formula and reference
    implementation: Ledoit & Wolf (2022), ``pald22/covShrinkage/QIS.py``.
    """
    y = np.asarray(returns, dtype=float).T  # observations x variables
    if y.ndim != 2 or y.shape[0] < 3:
        raise ValueError("QIS requires a 2-D observations x variables matrix")
    y = y - y.mean(axis=0, keepdims=True)
    observations, variables = y.shape
    effective = observations - 1
    concentration = variables / effective
    sample = (y.T @ y) / effective
    eigenvalues, eigenvectors = np.linalg.eigh((sample + sample.T) / 2)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    first_nonnull = max(0, variables - effective)
    nonnull = eigenvalues[first_nonnull:]
    # Numerical rank can be one below the algebraic rank at p ~= n.  The QIS
    # formula treats these directions as the null block, so move the boundary
    # rather than divide by a rounding-scale eigenvalue.
    rank_floor = max(float(eigenvalues[-1]) * np.finfo(float).eps * variables,
                     np.finfo(float).tiny)
    while len(nonnull) and nonnull[0] <= rank_floor:
        first_nonnull += 1
        nonnull = eigenvalues[first_nonnull:]
    if not len(nonnull):
        raise ValueError("QIS received a zero-rank return matrix")
    inverse = 1.0 / nonnull
    m = len(inverse)
    effective_concentration = variables / m
    bandwidth = (min(effective_concentration ** 2,
                     effective_concentration ** -2) ** .35 /
                 variables ** .35)
    level = np.repeat(inverse[:, None], m, axis=1)
    difference = level - level.T
    denominator = difference ** 2 + level ** 2 * bandwidth ** 2
    theta = np.mean(level * difference / denominator, axis=0)
    hilbert = np.mean(level ** 2 * bandwidth / denominator, axis=0)
    amplitude = theta ** 2 + hilbert ** 2

    null_count = variables - m
    if null_count == 0:
        c = concentration
        cleaned = 1.0 / (
            (1 - c) ** 2 * inverse
            + 2 * c * (1 - c) * inverse * theta
            + c ** 2 * inverse * amplitude)
    else:
        c = variables / m
        null_value = 1.0 / ((c - 1) * np.mean(inverse))
        cleaned = np.concatenate((np.full(null_count, null_value),
                                  1.0 / (inverse * amplitude)))
    cleaned *= eigenvalues.sum() / cleaned.sum()
    return (eigenvectors * cleaned) @ eigenvectors.T


def estimate_cvc(returns, *, folds=10, seed=20260803):
    """Isotonic K-fold cross-validated nonlinear eigenvalue shrinkage.

    Each held-out return is projected onto eigenvectors estimated without that
    return.  Rank-wise projected variances are averaged, monotonised by
    decreasing isotonic regression, and installed in the full-sample empirical
    eigenbasis.  This is the iso-10f-CVC construction of Bartz (2016).
    """
    x = np.asarray(returns, dtype=float).T  # observations x assets
    if x.ndim != 2 or x.shape[0] < folds or folds < 2:
        raise ValueError("CVC requires observations >= folds >= 2")
    corrected = np.zeros(x.shape[1], dtype=float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, held_out in splitter.split(x):
        mean = x[train].mean(axis=0)
        training = (x[train] - mean).T
        _, directions = spectral(sample_covariance(training, demean=False))
        projections = (x[held_out] - mean) @ directions
        corrected += np.sum(projections ** 2, axis=0)
    corrected /= x.shape[0]
    corrected = IsotonicRegression(increasing=False).fit_transform(
        np.arange(len(corrected)), corrected)
    positive_floor = max(float(np.mean(corrected)) * 1e-12,
                         np.finfo(float).tiny)
    corrected = np.maximum(corrected, positive_floor)
    _, full_directions = spectral(sample_covariance(x.T))
    return (full_directions * corrected) @ full_directions.T


def estimate_hcal(returns):
    """Average-linkage hierarchical covariance filter without bootstrapping."""
    from bahc import BAHC

    x = np.asarray(returns, dtype=float)
    return np.asarray(BAHC(x, K=1, Nboot=0, method="near",
                           filter_type="covariance").filter_matrix(),
                      dtype=float)


def estimate_bahc(returns, *, bootstraps=100, seed=20260803):
    """Published bootstrap-averaged hierarchical covariance filter."""
    from bahc import BAHC

    if bootstraps < 1:
        raise ValueError("BAHC requires at least one bootstrap")
    x = np.asarray(returns, dtype=float)
    return np.asarray(BAHC(x, K=1, Nboot=int(bootstraps), method="near",
                           filter_type="covariance", seed=seed).filter_matrix(),
                      dtype=float)


def realised_covariance(returns):
    """Sample covariance of the strictly future evaluation returns."""
    return sample_covariance(np.asarray(returns, dtype=float))


def condition_covariance(covariance, relative_floor=1e-8):
    """Symmetrise and apply one transparent eigenvalue floor for scoring."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance contains non-finite values")
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    largest = float(max(values[-1], np.finfo(float).tiny))
    floor = largest * relative_floor
    clipped = np.maximum(values, floor)
    conditioned = (vectors * clipped) @ vectors.T
    raw_condition = (largest / values[0]) if values[0] > 0 else np.inf
    return conditioned, {
        "raw_min_eigenvalue": float(values[0]),
        "raw_max_eigenvalue": largest,
        "raw_condition_number": float(raw_condition),
        "score_eigenvalue_floor": float(floor),
        "n_eigenvalues_floored": int(np.sum(values < floor)),
        "score_condition_number": float(clipped[-1] / clipped[0]),
    }


def unconstrained_gmv_weights(covariance):
    """Fully invested global-minimum-variance weights with shorting allowed."""
    covariance = np.asarray(covariance, dtype=float)
    ones = np.ones(covariance.shape[0])
    raw = np.linalg.solve(covariance, ones)
    denominator = float(ones @ raw)
    if abs(denominator) < np.finfo(float).eps:
        raise np.linalg.LinAlgError("GMV normalisation is numerically zero")
    return raw / denominator


def long_only_gmv_weights(covariance, maxiter=None):
    """Exact long-only GMV weights via a non-negative least-squares problem.

    If ``C=A.T@A``, the non-negative minimiser of
    ``0.5*x.T@C@x - 1.T@x`` is obtained from ``nnls(A, solve(A.T, 1))``.
    Normalising that solution to sum to one satisfies the KKT conditions of the
    original fully-invested long-only minimum-variance problem.
    """
    covariance = np.asarray(covariance, dtype=float)
    n = covariance.shape[0]
    upper = np.linalg.cholesky(covariance).T
    target = np.linalg.solve(upper.T, np.ones(n))
    iterations = 3 * n if maxiter is None else int(maxiter)
    solution, _ = nnls(upper, target, maxiter=iterations)
    if not np.isfinite(solution).all() or solution.sum() <= 0:
        raise RuntimeError("long-only GMV NNLS returned no feasible portfolio")
    return solution / solution.sum()


def covariance_scores(forecast, realised, *, compute_long_only=True):
    """All predeclared covariance, portfolio and eigenspace diagnostics."""
    forecast = np.asarray(forecast, dtype=float)
    realised = np.asarray(realised, dtype=float)
    if forecast.shape != realised.shape:
        raise ValueError("forecast and realised covariance shapes differ")
    conditioned, diagnostics = condition_covariance(forecast)
    relative_frobenius = (np.linalg.norm(forecast - realised, "fro") /
                          max(np.linalg.norm(realised, "fro"),
                              np.finfo(float).tiny))
    sign, logdet = np.linalg.slogdet(conditioned)
    if sign <= 0:
        raise np.linalg.LinAlgError("conditioned forecast is not positive definite")
    gaussian_nll = (
        logdet + np.trace(np.linalg.solve(conditioned, realised))
    ) / forecast.shape[0]

    long_short = unconstrained_gmv_weights(conditioned)
    long_short_variance = float(long_short @ realised @ long_short)
    if compute_long_only:
        long_only = long_only_gmv_weights(conditioned)
        long_only_variance = float(long_only @ realised @ long_only)
    else:
        long_only_variance = np.nan

    _, forecast_vectors = spectral(forecast)
    _, realised_vectors = spectral(realised)
    flag = frozen_flag_losses(realised_vectors[:, :6], forecast_vectors[:, :6])
    return {
        "relative_frobenius": float(relative_frobenius),
        "gaussian_nll_per_asset": float(gaussian_nll),
        "gmv_long_short_variance": long_short_variance,
        "gmv_long_short_annualised_volatility": float(
            np.sqrt(max(long_short_variance, 0.0) * 252)),
        "gmv_long_only_variance": long_only_variance,
        "gmv_long_only_annualised_volatility": float(
            np.sqrt(max(long_only_variance, 0.0) * 252))
            if np.isfinite(long_only_variance) else np.nan,
        "flag_market_1_loss": flag["market_1"],
        "flag_top_3_loss": flag["top_3"],
        "flag_top_6_loss": flag["top_6"],
        "flag_nested_loss": flag["flag_nested"],
        **diagnostics,
    }


def evaluate_covariance_estimator(examples, split_frame, estimator,
                                  estimator_name, *, long_only_split="test",
                                  include_splits=None):
    """Fit a past-only estimator independently at every shared origin."""
    split_by_example = split_frame.set_index("example")["split"].to_dict()
    rows = []
    for example in examples:
        split = split_by_example[example.example]
        if include_splits is not None and split not in include_splits:
            continue
        forecast = estimator(example.estimation_returns)
        target = realised_covariance(example.realised_returns)
        scores = covariance_scores(
            forecast, target, compute_long_only=(split == long_only_split))
        rows.append({
            "example": example.example,
            "estimator": estimator_name,
            "split": split,
            "current_date": example.current_date,
            "target_window_start": example.target_window_start,
            "target_date": example.target_date,
            **scores,
        })
    return pd.DataFrame(rows)


def choose_ewma_half_life(examples, split_frame, candidates):
    """Select EWMA decay once using mean validation Gaussian log loss only."""
    validation_ids = set(split_frame.loc[
        split_frame["split"] == "validation", "example"])
    if not validation_ids:
        raise ValueError("validation split is empty")
    rows = []
    for half_life in candidates:
        losses = []
        for example in examples:
            if example.example not in validation_ids:
                continue
            forecast = estimate_ewma(example.estimation_returns,
                                     half_life=float(half_life))
            target = realised_covariance(example.realised_returns)
            losses.append(covariance_scores(
                forecast, target, compute_long_only=False)[
                    "gaussian_nll_per_asset"])
        rows.append({"half_life": float(half_life),
                     "validation_mean_gaussian_nll_per_asset":
                         float(np.mean(losses)),
                     "n_validation_examples": len(losses)})
    table = pd.DataFrame(rows).sort_values(
        ["validation_mean_gaussian_nll_per_asset", "half_life"],
        ignore_index=True)
    return float(table.iloc[0]["half_life"]), table


def summarise_covariance_benchmarks(series):
    """Mean, median and IQR for every estimator, split and numeric score."""
    identity = {"example", "estimator", "split", "current_date",
                "target_window_start", "target_date"}
    metrics = [column for column in series.columns
               if column not in identity and
               np.issubdtype(series[column].dtype, np.number)]
    rows = []
    for (estimator, split), group in series.groupby(["estimator", "split"],
                                                     sort=False):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            rows.append({
                "estimator": estimator,
                "split": split,
                "metric": metric,
                "n_examples": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                # ``nearest`` avoids invalid inf-inf interpolation for the
                # deliberately reported raw condition number of singular
                # estimators while preserving the empirical quartile.
                "q25": float(np.quantile(values, .25, method="nearest")),
                "q75": float(np.quantile(values, .75, method="nearest")),
            })
    return pd.DataFrame(rows)
