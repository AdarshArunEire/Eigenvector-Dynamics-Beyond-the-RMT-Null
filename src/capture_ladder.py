"""Origin construction and split-clean scoring for the capture ladder.

The Stage 2 respecified design has three independent parameters that must never
again be conflated: ``T_in`` chosen for conditioning, ``T_out`` chosen for
economic relevance and which *is* the horizon, and ``step``.  Estimation and
target windows are disjoint by construction, so deletion contamination is
impossible rather than corrected, and there is no ``h/T`` anywhere.

Everything an entrant may see lives in ``Origin.estimation``; everything it is
scored against lives in ``Origin.realised``.  The realised block is standardised
using **estimation-window** quantities only.  This is the single easiest place
in the whole design to leak the future, so the scaling is written out
explicitly here rather than delegated to ``data.standardise``, whose global
mean divisor would silently read the target window if handed a joined slice.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import to_correlation_panel
from src.overlap import sample_covariance

#: Block length in origins for the circular-block interval: one block spans a
#: full estimation window plus its target, so two origins in different blocks
#: share no return observation.
def block_length_origins(T_in, T_out, step):
    return int(np.ceil((int(T_in) + int(T_out)) / int(step)))


@dataclass(frozen=True)
class Origin:
    """One forecast origin: what is known, what it is scored against."""

    index: int
    estimation_start: int
    target_start: int
    estimation: np.ndarray      # N x T_in, raw returns
    realised: np.ndarray        # N x T_out, standardised by estimation window
    origin_date: pd.Timestamp
    target_date: pd.Timestamp
    target_window_start: pd.Timestamp


def _day_levels(block):
    """Cross-sectional variance level, one value per day. No smoothing."""
    return (np.asarray(block, dtype=float) ** 2).mean(axis=0)


def estimation_scaling(estimation):
    """Per-day reference level and per-name volatilities from the past only.

    Returns ``(reference, sigma)``.  ``reference`` is the mean cross-sectional
    variance level over the estimation window; ``sigma`` are the per-name
    standard deviations of the day-flattened estimation window.  Both are
    functions of the estimation window alone, so applying them to the target
    window cannot leak.
    """
    estimation = np.asarray(estimation, dtype=float)
    levels = _day_levels(estimation)
    positive = levels[levels > 0]
    reference = float(positive.mean()) if positive.size else 1.0
    scale = np.where(levels > 0, levels / reference, 1.0)
    flattened = estimation / np.sqrt(scale)[None, :]
    sigma = flattened.std(axis=1, ddof=1)
    if np.any(sigma <= 0):
        raise ValueError(f"{int((sigma <= 0).sum())} name(s) have zero estimation variance")
    return reference, sigma


def apply_estimation_scaling(block, reference, sigma, flatten_days=False):
    """Standardise a return block with estimation-window quantities only.

    ``flatten_days`` divides each day by its own cross-sectional volatility
    relative to the estimation-window reference.  Day ``t``'s divisor uses only
    day ``t``, so it is causal within the target window, and it makes every day
    of the quarter count equally.

    It defaults to **False**, which is what the reported ladder used: the
    realised block keeps its own volatility path and high-volatility days
    therefore weigh more, which is the honest reading of "how much of next
    quarter's realised risk does my six-factor model span".  The two
    conventions are not close -- day-flattening moves CAC Frozen from 0.641 to
    0.564 -- so the flag exists to make the choice visible, not adjustable
    after the fact.  Calibration against the recorded run: with the default,
    CAC gives Frozen 0.6411 / ceiling 0.7971 against a recorded 0.638 / 0.797
    and DAX 0.5985 / 0.7661 against 0.591 / 0.764.
    """
    block = np.asarray(block, dtype=float)
    if flatten_days:
        levels = _day_levels(block)
        scale = np.where(levels > 0, levels / float(reference), 1.0)
        block = block / np.sqrt(scale)[None, :]
    return block / np.asarray(sigma, dtype=float)[:, None]


def build_origins(panel, dates, T_in, T_out, step, flatten_days=False):
    """Disjoint estimation/target origins over the whole history."""
    panel = np.asarray(panel, dtype=float)
    dates = pd.DatetimeIndex(dates)
    T_in, T_out, step = int(T_in), int(T_out), int(step)
    if panel.ndim != 2:
        raise ValueError(f"panel must be N x days, got {panel.shape}")
    if len(dates) != panel.shape[1]:
        raise ValueError("dates and panel disagree on the number of days")
    if min(T_in, T_out, step) < 1:
        raise ValueError("T_in, T_out and step must all be positive")
    origins = []
    last = panel.shape[1] - T_in - T_out
    for index, start in enumerate(range(0, last + 1, step)):
        estimation = panel[:, start:start + T_in]
        target_start = start + T_in
        reference, sigma = estimation_scaling(estimation)
        realised = apply_estimation_scaling(
            panel[:, target_start:target_start + T_out], reference, sigma,
            flatten_days=flatten_days)
        origins.append(Origin(
            index=index, estimation_start=start, target_start=target_start,
            estimation=estimation, realised=realised,
            origin_date=dates[target_start - 1],
            target_date=dates[target_start + T_out - 1],
            target_window_start=dates[target_start]))
    if not origins:
        raise ValueError(
            f"panel has {panel.shape[1]} days, too short for T_in={T_in} + T_out={T_out}")
    return origins


def frozen_frame(estimation, rank=6):
    """Top-``rank`` eigenvectors of the estimation window's sample correlation.

    The exact Stage 1 construction: day-flatten at ``window=1``, scale each
    name to unit variance, then take the leading eigenvectors of the sample
    correlation.  This is the ladder's Frozen entrant and the base point every
    correction is measured against.
    """
    reference, sigma = estimation_scaling(estimation)
    adjusted = to_correlation_panel(
        apply_estimation_scaling(estimation, reference, sigma))
    values, vectors = np.linalg.eigh(sample_covariance(adjusted))
    return vectors[:, ::-1][:, :int(rank)]


def assign_splits(origins, train_end, validation_start, validation_end,
                  test_start):
    """Split by *target* date, with the target-window purge actually checked.

    An origin belongs to a split only if the window it is scored against ends
    inside that split's dates, and the first validation target window must open
    strictly after the last training target closes.  Splitting on the origin
    date instead would let a training target overlap a validation estimation
    window, which is the standard way this design leaks.
    """
    bounds = [pd.Timestamp(value) for value in
              (train_end, validation_start, validation_end, test_start)]
    if not (bounds[0] < bounds[1] <= bounds[2] < bounds[3]):
        raise ValueError("chronological split dates are not ordered")
    train_end, validation_start, validation_end, test_start = bounds
    splits = {}
    for origin in origins:
        target = origin.target_date
        if target <= train_end:
            splits[origin.index] = "train"
        elif validation_start <= target <= validation_end:
            splits[origin.index] = "validation"
        elif target >= test_start:
            splits[origin.index] = "test"
        else:
            splits[origin.index] = "purged"
    grouped = {name: [origin for origin in origins if splits[origin.index] == name]
               for name in ("train", "validation", "test")}
    empty = [name for name, group in grouped.items() if not group]
    if empty:
        raise ValueError(f"empty chronological split(s): {empty}")
    if (min(o.target_window_start for o in grouped["validation"])
            <= max(o.target_date for o in grouped["train"])):
        raise ValueError("train and validation target windows overlap")
    if (min(o.target_window_start for o in grouped["test"])
            <= max(o.target_date for o in grouped["validation"])):
        raise ValueError("validation and test target windows overlap")
    return splits, grouped


def circular_block_interval(values, block, repetitions=2000, seed=20260805,
                            quantiles=(.025, .975)):
    """Circular-block bootstrap interval for a dependent paired mean."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size:
        raise ValueError("values must be a non-empty vector")
    rng = np.random.default_rng(int(seed))
    block = min(max(1, int(block)), values.size)
    blocks = int(np.ceil(values.size / block))
    starts = rng.integers(0, values.size, size=(int(repetitions), blocks))
    offsets = np.arange(block)
    indices = ((starts[:, :, None] + offsets) % values.size).reshape(
        int(repetitions), -1)[:, :values.size]
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, quantiles))


def paired_summary(model, frozen, block, repetitions=2000, seed=20260805):
    """Paired mean difference with a circular-block interval and win rate."""
    model = np.asarray(model, dtype=float)
    frozen = np.asarray(frozen, dtype=float)
    if model.shape != frozen.shape:
        raise ValueError("model and frozen capture series differ in length")
    difference = model - frozen
    low, high = circular_block_interval(difference, block, repetitions, seed)
    return {
        "n_origins": int(model.size),
        "block_length_origins": int(min(block, model.size)),
        "model_mean_capture": float(model.mean()),
        "reference_mean_capture": float(frozen.mean()),
        "paired_improvement": float(difference.mean()),
        "improvement_ci_low": low,
        "improvement_ci_high": high,
        "origin_win_fraction": float(np.mean(difference > 0)),
        "excludes_zero": bool(low > 0 or high < 0),
    }
