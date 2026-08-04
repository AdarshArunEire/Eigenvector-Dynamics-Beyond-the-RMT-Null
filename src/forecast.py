"""Chronological evaluation primitives for Stage 2 Flag forecasts."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.flag import DEFAULT_DIMS, flag_component_bases, validate_flag_frame
from src.grassmann import containment_loss


@dataclass(frozen=True)
class ChronologicalSplits:
    """Predeclared target-date ranges with purged gaps between them."""

    train_end: str = "2013-12-31"
    validation_start: str = "2015-07-01"
    validation_end: str = "2018-06-30"
    test_start: str = "2020-01-01"

    def timestamps(self):
        values = tuple(pd.Timestamp(value) for value in (
            self.train_end, self.validation_start,
            self.validation_end, self.test_start))
        if not (values[0] < values[1] <= values[2] < values[3]):
            raise ValueError("chronological split dates are not ordered")
        return values


def frozen_flag_losses(target, current, dims=DEFAULT_DIMS):
    """Normalised projector losses when predicting that the Flag holds still.

    Individual disjoint and cumulative components are reported.  The complete
    nested loss is the mean normalised loss over cumulative levels 1, 3 and 6,
    matching the inverse-dimension weighting of the Stage 1 Flag geometry.
    """
    target = validate_flag_frame(target, dims, "target")
    current = validate_flag_frame(current, dims, "current")
    if target.shape != current.shape:
        raise ValueError("target and current flag frames have different shapes")
    targets = flag_component_bases(target, dims)
    predictions = flag_component_bases(current, dims)
    losses = {
        name: containment_loss(targets[name], predictions[name], normalise=True)
        for name in targets
    }
    cumulative = [containment_loss(target[:, :d], current[:, :d], normalise=True)
                  for d in dims]
    losses["flag_nested"] = float(np.mean(cumulative))
    return losses


def frozen_flag_series(starts, frames, dates, T, horizon=42, step=14):
    """Long-format examples for the zero-motion forecast.

    Examples retain a past state even though Benchmark 1.1 does not use it, so
    every later benchmark and model is scored on the identical rows.
    """
    starts = np.asarray(starts, dtype=int)
    frames = np.asarray(frames, dtype=float)
    dates = pd.DatetimeIndex(dates)
    if horizon < step or horizon % step:
        raise ValueError("horizon must be a positive multiple of step")
    if len(starts) != len(frames):
        raise ValueError("starts and frames have different lengths")
    offset = horizon // step
    if len(frames) <= 2 * offset:
        raise ValueError("not enough Flag states for one forecast example")

    rows = []
    for current_index in range(offset, len(frames) - offset):
        past_index, target_index = current_index - offset, current_index + offset
        losses = frozen_flag_losses(frames[target_index], frames[current_index])
        shared = {
            "example": int(current_index - offset),
            "past_date": dates[starts[past_index] + T - 1],
            "current_date": dates[starts[current_index] + T - 1],
            "target_date": dates[starts[target_index] + T - 1],
            "target_window_start": dates[starts[target_index]],
            "past_start": int(starts[past_index]),
            "current_start": int(starts[current_index]),
            "target_start": int(starts[target_index]),
        }
        rows.extend(dict(shared, component=component, loss=loss)
                    for component, loss in losses.items())
    return pd.DataFrame(rows)


def assign_chronological_splits(series, specification=ChronologicalSplits()):
    """Assign train/validation/test by target date and verify target-window purge."""
    out = series.copy()
    for column in ("target_date", "target_window_start"):
        if column not in out:
            raise ValueError(f"missing required column {column}")
        out[column] = pd.to_datetime(out[column])
    train_end, validation_start, validation_end, test_start = specification.timestamps()
    target = out["target_date"]
    out["split"] = "purged"
    out.loc[target <= train_end, "split"] = "train"
    out.loc[(target >= validation_start) & (target <= validation_end),
            "split"] = "validation"
    out.loc[target >= test_start, "split"] = "test"

    examples = out.drop_duplicates("example")
    groups = {name: examples.loc[examples["split"] == name]
              for name in ("train", "validation", "test")}
    if any(group.empty for group in groups.values()):
        counts = {name: len(group) for name, group in groups.items()}
        raise ValueError(f"empty chronological split: {counts}")
    if groups["validation"]["target_window_start"].min() <= groups["train"]["target_date"].max():
        raise ValueError("train and validation target covariance windows overlap")
    if groups["test"]["target_window_start"].min() <= groups["validation"]["target_date"].max():
        raise ValueError("validation and test target covariance windows overlap")
    return out


def _nonoverlap_count(examples):
    """Greedy count of target covariance windows sharing no return observation."""
    ordered = examples.sort_values("target_date")
    last_end = None
    count = 0
    for row in ordered.itertuples():
        if last_end is None or row.target_window_start > last_end:
            count += 1
            last_end = row.target_date
    return count


def summarise_benchmark(series):
    """Mean, median, IQR and effective non-overlap count for every loss."""
    rows = []
    for split in ("train", "validation", "test"):
        split_frame = series.loc[series["split"] == split]
        examples = split_frame.drop_duplicates("example")
        nonoverlap = _nonoverlap_count(examples)
        for component, group in split_frame.groupby("component", sort=False):
            loss = group["loss"].to_numpy(dtype=float)
            rows.append({
                "split": split,
                "component": component,
                "n_examples": int(len(loss)),
                "n_nonoverlap_targets": int(nonoverlap),
                "mean_loss": float(np.mean(loss)),
                "median_loss": float(np.median(loss)),
                "q25_loss": float(np.quantile(loss, .25)),
                "q75_loss": float(np.quantile(loss, .75)),
                "std_loss": float(np.std(loss, ddof=1)) if len(loss) > 1 else np.nan,
                "max_loss": float(np.max(loss)),
            })
    return pd.DataFrame(rows)
