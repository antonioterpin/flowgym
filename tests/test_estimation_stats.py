import jax.numpy as jnp

import flowgym.common.evaluation as eval_mod
from flowgym.common.evaluation import (
    angle_error,
    relative_error,
    absolute_error,
    loss_supervised,
    loss_unsupervised,
    compute_stats,
)


def test_angle_error_zero_and_pi_over_two():
    # Identical vectors -> zero error
    f = jnp.array([[[1.0, 0.0]]])
    gt = jnp.array([[[1.0, 0.0]]])
    err = angle_error(f, gt)
    assert err.shape == (1, 1)
    assert jnp.allclose(err, 0.0)

    # Orthogonal vectors -> pi/2 error
    f = jnp.array([[[1.0, 0.0]]])
    gt = jnp.array([[[0.0, 1.0]]])
    err = angle_error(f, gt)
    assert jnp.allclose(err, 0.5, rtol=1e-6)


def test_relative_error_zero_and_nonzero():
    # Non-zero case
    f = jnp.array([[[3.0, 4.0]]])  # norm = 5
    gt = jnp.array([[[0.0, 8.0]]])  # norm = 8
    err = relative_error(f, gt).squeeze(-1)
    expected = 3.0 / 8.0
    assert err.shape == (1, 1)
    assert jnp.allclose(err, expected, rtol=1e-6)

    # Identical flows -> zero error
    f = jnp.array([[[2.0, 0.0]]])
    gt = jnp.array([[[2.0, 0.0]]])
    err = relative_error(f, gt).squeeze(-1)
    assert jnp.allclose(err, 0.0)


def test_absolute_error():
    f = jnp.array([[[1.0, 2.0]]])
    gt = jnp.array([[[4.0, 6.0]]])
    # Difference vector = [-3, -4] -> norm = 5
    err = absolute_error(f, gt)
    assert err.shape == (1, 1)
    assert jnp.allclose(err, 5.0, rtol=1e-6)


def test_loss_supervised_basic():
    flow = jnp.arange(18, dtype=jnp.float32).reshape((1, 3, 3, 2)) * 0.1
    gt = flow + 0.5
    expected = jnp.mean(jnp.sum(jnp.square(flow - gt), axis=-1))
    loss = loss_supervised(flow, gt)
    assert loss.shape == ()
    assert jnp.allclose(loss, expected, rtol=1e-6)


def test_loss_unsupervised_zero_flow_identity_warp(monkeypatch):
    # Monkey-patch apply_flow_to_image_forward to return img1 unchanged
    monkeypatch.setattr(
        eval_mod, "apply_flow_to_image_forward", lambda img, flow, dt: img
    )
    img1 = jnp.ones((1, 2, 2, 1))
    img2 = jnp.ones((1, 2, 2, 1)) * 2
    flow = jnp.zeros((1, 2, 2, 2))
    loss = loss_unsupervised(img1, img2, flow)
    # Squared error = (2 - 1)^2 = 1 for each pixel, mean over spatial dims -> 1
    assert loss.shape == (1, 1)
    assert jnp.allclose(loss, 1.0, rtol=1e-6)


def test_compute_stats_known_values():
    errors = jnp.array([1.0, 2.0, 3.0, 4.0])
    threshold = 2.5
    stats = compute_stats(errors, threshold=threshold)
    # Expected values
    expected_mean = 2.5
    expected_std = jnp.sqrt(1.25)
    expected_fraction = 0.5
    expected_lq = 1.75
    expected_up = 3.25
    expected_median = 2.5
    iqr = expected_up - expected_lq
    expected_lw = expected_lq - 1.5 * iqr
    expected_uw = expected_up + 1.5 * iqr

    assert jnp.allclose(stats["mean"], expected_mean, rtol=1e-6)
    assert jnp.allclose(stats["std"], expected_std, rtol=1e-6)
    assert jnp.allclose(stats["fraction_below_threshold"], expected_fraction, rtol=1e-6)
    assert jnp.allclose(stats["lower_quartile"], expected_lq, rtol=1e-6)
    assert jnp.allclose(stats["upper_quartile"], expected_up, rtol=1e-6)
    assert jnp.allclose(stats["median"], expected_median, rtol=1e-6)
    assert jnp.allclose(stats["lower_whisker"], expected_lw, rtol=1e-6)
    assert jnp.allclose(stats["upper_whisker"], expected_uw, rtol=1e-6)
