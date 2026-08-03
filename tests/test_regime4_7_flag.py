"""Cross-check Regime 4.7 against the established Grassmann pipeline."""
import numpy as np
import pandas as pd
import pytest

from scripts.regime4_4_tangent import tangent_series
from scripts.regime4_7_flag import (coherence_inputs,
                                    flag_attribution_series, holm_adjust)
from src.synth import rotate_basis


def _path():
    frames = []
    full = np.eye(10)
    for angle in (0.0, 0.05, 0.11, 0.18, 0.26):
        moved = rotate_basis(full, 0, 7, angle)
        moved = rotate_basis(moved, 3, 8, angle / 2)
        frames.append(moved[:, :6])
    return np.asarray(frames)


def test_top_three_persistence_exactly_matches_regime_4_4():
    frames = _path()
    starts = np.arange(len(frames))
    diagnostics = pd.DataFrame({
        "positive_correlation_fraction": np.ones(len(frames)),
        "all_correlations_positive": np.ones(len(frames), dtype=bool),
        "erse_rotations": np.zeros(len(frames)),
    })
    flag = flag_attribution_series(
        starts, frames, frames.copy(), diagnostics, horizon=1, step=1)
    old = tangent_series(starts, frames[:, :, :3], horizon=1, step=1)
    new = flag.loc[flag["component"] == "top_3", "cosine"].to_numpy()
    assert new == pytest.approx(old["cosine"].to_numpy(), abs=1e-12)


def test_top_three_coherence_input_matches_established_tangent_series():
    frames = _path()
    inputs = coherence_inputs(frames, horizon=1, step=1)
    assert inputs["top_3"]["stacked"] == pytest.approx(
        inputs["top_3"]["tangents"][0], abs=1e-12)


def test_holm_adjustment_is_monotone_in_ranked_pvalues():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
