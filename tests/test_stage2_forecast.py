"""Tests for the Stage 2 chronological Flag benchmark."""
import numpy as np
import pandas as pd
import pytest

from src.forecast import (ChronologicalSplits, assign_chronological_splits,
                          frozen_flag_losses, frozen_flag_series,
                          summarise_benchmark)
from src.synth import rotate_basis


def test_frozen_loss_is_zero_for_unchanged_flag_and_basis_invariant():
    frame = np.eye(10)[:, :6]
    transformed = frame.copy()
    transformed[:, 0] *= -1
    transformed[:, 1:3] = transformed[:, 1:3] @ np.linalg.qr(
        np.array([[1., 2.], [2., 0.]]))[0]
    transformed[:, 3:6] = transformed[:, 3:6] @ np.linalg.qr(
        np.array([[1., 2., 0.], [2., 0., 1.], [0., 1., 1.]]))[0]
    losses = frozen_flag_losses(frame, transformed)
    assert all(value == pytest.approx(0, abs=1e-14)
               for value in losses.values())


def test_frozen_loss_detects_known_market_rotation_and_flag_is_component_mean():
    current = np.eye(10)[:, :6]
    target = rotate_basis(np.eye(10), 0, 8, .3)[:, :6]
    losses = frozen_flag_losses(target, current)
    assert losses["market_1"] == pytest.approx(np.sin(.3) ** 2, abs=1e-12)
    assert losses["flag_nested"] == pytest.approx(
        np.mean([losses["market_1"], losses["top_3"], losses["top_6"]]),
        abs=1e-14)


def _toy_series():
    full = np.eye(10)
    frames = np.asarray([rotate_basis(full, 0, 8, angle)[:, :6]
                         for angle in np.linspace(0, .5, 60)])
    starts = np.arange(60) * 2
    dates = pd.date_range("2000-01-01", periods=160, freq="D")
    return frozen_flag_series(starts, frames, dates, T=20, horizon=4, step=2)


def test_series_uses_identical_past_current_future_rows_for_every_component():
    series = _toy_series()
    counts = series.groupby("example")["component"].nunique()
    assert counts.nunique() == 1
    assert counts.iloc[0] == 6
    assert (series["past_date"] < series["current_date"]).all()
    assert (series["current_date"] < series["target_date"]).all()


def test_chronological_assignment_keeps_gaps_purged_and_checks_window_overlap():
    rows = []
    dates = [("2002-01-01", "2001-01-01"),
             ("2004-01-01", "2003-01-01"),
             ("2006-01-01", "2005-01-01"),
             ("2008-01-01", "2007-01-01"),
             ("2010-01-01", "2009-01-01")]
    for example, (target, start) in enumerate(dates):
        rows.append({"example": example, "target_date": target,
                     "target_window_start": start, "component": "flag_nested",
                     "loss": .1})
    spec = ChronologicalSplits(train_end="2002-12-31",
                               validation_start="2005-01-01",
                               validation_end="2006-12-31",
                               test_start="2009-01-01")
    split = assign_chronological_splits(pd.DataFrame(rows), spec)
    assert split["split"].tolist() == ["train", "purged", "validation",
                                        "purged", "test"]


def test_summary_reports_mean_median_iqr_and_nonoverlap_count():
    series = _toy_series()
    # Use compact artificial date boundaries appropriate to the toy history.
    spec = ChronologicalSplits(train_end="2000-01-31",
                               validation_start="2000-02-25",
                               validation_end="2000-03-05",
                               test_start="2000-04-01")
    split = assign_chronological_splits(series, spec)
    summary = summarise_benchmark(split)
    assert {"mean_loss", "median_loss", "q25_loss", "q75_loss",
            "n_nonoverlap_targets"} <= set(summary.columns)
    assert set(summary["split"]) == {"train", "validation", "test"}
