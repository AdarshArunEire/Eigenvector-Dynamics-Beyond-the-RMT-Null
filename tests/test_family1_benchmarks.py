import numpy as np
import pandas as pd
import pytest

from src.family1_benchmarks import (
    combine_panel_skills, compare_with_frozen, constant_velocity_flag,
    covariance_to_correlation, damped_velocity_flag, ewma_flag,
    factor_cm_iewma_flag,
    retained_window_flag,
    select_global_damping_alpha, stationary_roll_forward_flag)
from src.forecast import frozen_flag_losses
from src.synth import rotate_basis


def test_constant_velocity_repeats_a_known_flag_rotation():
    full = np.eye(12)
    past = full[:, :6]
    current = rotate_basis(full, 0, 8, .2)[:, :6]
    expected = rotate_basis(full, 0, 8, .4)[:, :6]
    predicted = constant_velocity_flag(past, current)
    losses = frozen_flag_losses(expected, predicted)
    assert all(value == pytest.approx(0, abs=1e-12)
               for value in losses.values())


def test_constant_velocity_is_invariant_to_flag_block_representatives():
    rng = np.random.default_rng(3)
    full = np.eye(12)
    past = full[:, :6]
    current = rotate_basis(full, 0, 8, .2)[:, :6]
    transformed = current.copy()
    transformed[:, :1] *= -1
    transformed[:, 1:3] @= np.linalg.qr(rng.normal(size=(2, 2)))[0]
    transformed[:, 3:6] @= np.linalg.qr(rng.normal(size=(3, 3)))[0]
    first = constant_velocity_flag(past, current)
    second = constant_velocity_flag(past, transformed)
    losses = frozen_flag_losses(first, second)
    assert all(value == pytest.approx(0, abs=1e-12)
               for value in losses.values())


def test_damped_velocity_has_frozen_and_constant_velocity_endpoints():
    full = np.eye(12)
    past = full[:, :6]
    current = rotate_basis(full, 0, 8, .2)[:, :6]
    frozen = damped_velocity_flag(past, current, 0.)
    full_step = damped_velocity_flag(past, current, 1.)
    assert all(value == pytest.approx(0, abs=1e-12)
               for value in frozen_flag_losses(current, frozen).values())
    expected = constant_velocity_flag(past, current)
    assert all(value == pytest.approx(0, abs=1e-12)
               for value in frozen_flag_losses(expected, full_step).values())


def test_damped_velocity_scales_a_known_complete_rotation():
    full = np.eye(12)
    past = full[:, :6]
    current = rotate_basis(full, 0, 8, .2)[:, :6]
    expected = rotate_basis(full, 0, 8, .3)[:, :6]
    prediction = damped_velocity_flag(past, current, .5)
    assert all(value == pytest.approx(0, abs=1e-12)
               for value in frozen_flag_losses(expected, prediction).values())


def test_damped_velocity_is_invariant_to_flag_block_representatives():
    rng = np.random.default_rng(31)
    full = np.eye(12)
    past = full[:, :6]
    current = rotate_basis(full, 0, 8, .2)[:, :6]
    transformed = current.copy()
    transformed[:, :1] *= -1
    transformed[:, 1:3] @= np.linalg.qr(rng.normal(size=(2, 2)))[0]
    transformed[:, 3:6] @= np.linalg.qr(rng.normal(size=(3, 3)))[0]
    first = damped_velocity_flag(past, current, .25)
    second = damped_velocity_flag(past, transformed, .25)
    assert all(value == pytest.approx(0, abs=1e-12)
               for value in frozen_flag_losses(first, second).values())


def test_damped_velocity_rejects_extrapolation_outside_damping_range():
    frame = np.eye(12)[:, :6]
    with pytest.raises(ValueError, match="alpha"):
        damped_velocity_flag(frame, frame, -0.1)
    with pytest.raises(ValueError, match="alpha"):
        damped_velocity_flag(frame, frame, 1.1)


def test_global_damping_selection_recovers_constant_rotation():
    full = np.eye(12)
    frames = np.asarray([
        rotate_basis(full, 0, 8, angle)[:, :6]
        for angle in np.linspace(0., .8, 9)])
    selected, tuning = select_global_damping_alpha(
        frames, validation_examples=[1, 2, 3, 4, 5], offset=1,
        alphas=(0., .25, .5, .75, 1.))
    assert selected == pytest.approx(1.)
    assert tuning.loc[tuning["selected"], "alpha"].tolist() == [1.]


def test_global_damping_selection_uses_only_supplied_validation_examples():
    full = np.eye(12)
    frames = np.asarray([
        rotate_basis(full, 0, 8, angle)[:, :6]
        for angle in (0., .1, .2, .3, .4, .5, .6, .7, .8)])
    changed = frames.copy()
    changed[7] = rotate_basis(full, 0, 8, -1.)[:, :6]
    args = dict(validation_examples=[1, 2, 3], offset=1,
                alphas=(0., .5, 1.))
    first, _ = select_global_damping_alpha(frames, **args)
    second, _ = select_global_damping_alpha(changed, **args)
    assert second == first


def test_skill_uses_ratio_of_mean_losses_and_combines_markets_equally():
    rows_model, rows_frozen = [], []
    for label, frozen_losses, model_losses in (
            ("large", [1., 3.], [.5, 1.5]),
            ("small", [10., 10.], [15., 15.])):
        for example, (frozen, model) in enumerate(zip(frozen_losses,
                                                       model_losses)):
            rows_frozen.append({"label": label, "example": example,
                                "component": "flag_nested", "loss": frozen})
            rows_model.append({"label": label, "example": example,
                               "component": "flag_nested", "loss": model})
    comparison = compare_with_frozen(
        pd.DataFrame(rows_model), pd.DataFrame(rows_frozen),
        T=2, horizon=1, step=1, repetitions=100, seed=1)
    skills = comparison.set_index("label")["skill_percent"]
    assert skills["large"] == pytest.approx(50.)
    assert skills["small"] == pytest.approx(-50.)
    combined = combine_panel_skills(comparison).iloc[0]
    assert combined["combined_skill_percent"] == pytest.approx(0.)
    assert combined["worst_panel_skill_percent"] == pytest.approx(-50.)
    assert combined["panels_improved"] == 1


def test_new_causal_forecasters_return_valid_nested_frames():
    rng = np.random.default_rng(11)
    returns = rng.normal(size=(12, 80))
    for prediction in (
            retained_window_flag(returns, 10),
            stationary_roll_forward_flag(returns, 10),
            ewma_flag(returns, decay=.94),
            ewma_flag(returns, half_life=21)):
        assert prediction.shape == (12, 6)
        assert prediction.T @ prediction == pytest.approx(np.eye(6), abs=1e-10)


def test_covariance_to_correlation_is_scale_invariant():
    covariance = np.array([[4., 1.], [1., 9.]])
    scale = np.diag([3., .5])
    first = covariance_to_correlation(covariance)
    second = covariance_to_correlation(scale @ covariance @ scale)
    assert second == pytest.approx(first)


def test_retained_window_ignores_observations_known_to_expire():
    rng = np.random.default_rng(12)
    returns = rng.normal(size=(12, 80))
    changed = returns.copy()
    changed[:, :10] = rng.normal(scale=100, size=(12, 10))
    first = retained_window_flag(returns, 10)
    second = retained_window_flag(changed, 10)
    losses = frozen_flag_losses(first, second)
    assert all(value == pytest.approx(0, abs=1e-12)
               for value in losses.values())


def test_factor_cm_iewma_returns_a_valid_flag():
    # cvxcovariance supplies the authors' official CM-IEWMA implementation and
    # is optional; skip cleanly when it is absent.
    pytest.importorskip("cvx.covariance")
    returns = np.random.default_rng(13).normal(size=(12, 90))
    prediction = factor_cm_iewma_flag(returns, factor_rank=6)
    assert prediction.shape == (12, 6)
    assert prediction.T @ prediction == pytest.approx(np.eye(6), abs=1e-10)
