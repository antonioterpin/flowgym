"""Unit tests for the LimaPivEstimator wrapper, loss, and training step."""

import jax
import jax.numpy as jnp
import pytest

from flowgym.flow.lima.lima_piv import LimaPivEstimator
from flowgym.flow.lima.process import bilinear_warp
from flowgym.types import SupervisedExperience


def _small_estimator(**kwargs):
    """A shallow, fast LIMA estimator for unit tests."""
    cfg = dict(
        encoder_channels=(16, 32, 64),  # 3 levels -> stride multiple 8
        refine_levels=3,
        search_range=2,
        optimizer_config={"name": "adam", "learning_rate": 1e-3},
    )
    cfg.update(kwargs)
    return LimaPivEstimator(**cfg)


# --- construction / validation -------------------------------------------


def test_construction_defaults():
    """The estimator builds with paper defaults (LIMA-6, replicate, SR=2)."""
    est = LimaPivEstimator()
    assert est.refine_levels == 6
    assert est.search_range == 2
    assert est.padding_mode == "replicate"
    assert len(est.level_weights) == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"search_range": 0},
        {"search_range": -1},
        {"padding_mode": "bogus"},
        {"refine_levels": 0},
        {"refine_levels": 99},  # > number of encoder levels
        {"lambda_u": -1.0},
        {"level_weights": (0.1, 0.2)},  # wrong length vs refine_levels
    ],
)
def test_construction_rejects_bad_params(kwargs):
    """Invalid hyperparameters raise at construction time."""
    with pytest.raises((ValueError, TypeError)):
        LimaPivEstimator(**kwargs)


def test_level_weights_default_matches_paper_for_lima6():
    """Default per-level loss weights match Table 2 of LIMA-1 for LIMA-6."""
    est = LimaPivEstimator(refine_levels=6)
    assert est.level_weights == (0.0025, 0.005, 0.01, 0.02, 0.08, 0.32)


# --- trainable state / forward -------------------------------------------


def test_create_trainable_state_has_params():
    """create_trainable_state initializes model params and an optimizer."""
    est = _small_estimator()
    dummy = jnp.zeros((2, 32, 32))
    ts = est.create_trainable_state(dummy, jax.random.PRNGKey(0))
    assert ts.params is not None
    assert jax.tree_util.tree_leaves(ts.params)  # non-empty


def test_estimate_returns_full_resolution_flow():
    """_estimate returns a full-resolution (B, H, W, 2) finite flow."""
    est = _small_estimator()
    B, H, W = 2, 40, 48  # not a multiple of 8 -> exercises pad/crop
    dummy = jnp.zeros((B, H, W))
    ts = est.create_trainable_state(dummy, jax.random.PRNGKey(0))
    prev = jax.random.normal(jax.random.PRNGKey(1), (B, H, W)) * 40 + 120
    curr = jax.random.normal(jax.random.PRNGKey(2), (B, H, W)) * 40 + 120
    state = {"images": prev[:, None, ...]}  # state["images"][:, -1] == prev
    flow, _extras, _metrics = est._estimate(curr, state, ts, {})
    assert flow.shape == (B, H, W, 2)
    assert jnp.all(jnp.isfinite(flow))


# --- loss / Jacobian ------------------------------------------------------


def test_jacobian_l1_zero_for_constant_field():
    """A spatially constant displacement field has zero Jacobian penalty."""
    est = _small_estimator()
    flow = jnp.ones((1, 8, 8, 2)) * 3.7
    assert float(est._jacobian_l1(flow)) == pytest.approx(0.0, abs=1e-6)


def test_jacobian_l1_linear_ramp():
    """A linear ramp has a constant gradient equal to the slope."""
    est = _small_estimator()
    # u = 2*x, v = 0 -> |du/dx| = 2 everywhere, all other derivatives 0.
    x = jnp.arange(8.0)
    u = jnp.broadcast_to(2.0 * x[None, None, :], (1, 8, 8))
    flow = jnp.stack([u, jnp.zeros_like(u)], axis=-1)
    assert float(est._jacobian_l1(flow)) == pytest.approx(2.0, abs=1e-5)


def test_jacobian_handles_singleton_levels():
    """The Jacobian penalty is finite on degenerate 1x1 levels."""
    est = _small_estimator()
    flow = jnp.ones((1, 1, 1, 2))
    assert jnp.isfinite(est._jacobian_l1(flow))


# --- training step --------------------------------------------------------


def test_train_step_reduces_loss_on_fixed_batch():
    """A few optimizer steps reduce the loss on a fixed synthetic batch."""
    est = _small_estimator(
        optimizer_config={"name": "adam", "learning_rate": 2e-3}
    )
    B, H, W = 2, 32, 32
    ts = est.create_trainable_state(jnp.zeros((B, H, W)), jax.random.PRNGKey(0))
    train_step = est.create_train_step()

    key = jax.random.PRNGKey(7)
    k1, k2, k3 = jax.random.split(key, 3)
    img1 = jax.random.uniform(k1, (B, H, W)) * 255
    img2 = jax.random.uniform(k2, (B, H, W)) * 255
    # Smooth (per-sample constant) target: both the data and smoothness terms
    # are simultaneously minimizable, so overfitting must reduce the loss.
    const = jax.random.normal(k3, (B, 1, 1, 2)) * 1.5
    gt = jnp.broadcast_to(const, (B, H, W, 2))
    exp = SupervisedExperience(
        state={"images": img1[:, None, ...]},
        obs=(img1, img2),
        ground_truth=gt,
    )

    loss0, ts, _ = train_step(ts, exp)
    for _ in range(25):
        loss, ts, _ = train_step(ts, exp)
    assert jnp.isfinite(loss0) and jnp.isfinite(loss)
    assert float(loss) < float(loss0)


def test_overfits_consistent_pair_with_varying_flow():
    """End-to-end: the refinement loop recovers a spatially-varying flow.

    Builds a self-consistent image pair (img2[p] = img1[p - flow_gt]) from a
    horizontal ramp displacement, overfits a small LIMA, and asserts the
    finest-level prediction error drops well below the zero-flow baseline. A
    spatially-varying target cannot be fit by memorizing a constant, so this
    exercises the warp + correlation matching across pyramid levels (it fails
    if the symmetric warp is anti-corrective).
    """
    est = _small_estimator(
        optimizer_config={"name": "adam", "learning_rate": 3e-3}
    )
    B, H, W = 1, 32, 32
    ts = est.create_trainable_state(jnp.zeros((B, H, W)), jax.random.PRNGKey(0))
    yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    img1 = (jnp.sin(xx * 0.4) + jnp.cos(yy * 0.3) + 2.0)[None] * 60.0
    ramp = jnp.linspace(-1.0, 1.0, W)
    flow_gt = jnp.stack(
        [
            jnp.broadcast_to(ramp[None, None, :], (B, H, W)),
            jnp.zeros((B, H, W)),
        ],
        axis=-1,
    )
    img2 = bilinear_warp(img1[..., None], -flow_gt)[..., 0]
    exp = SupervisedExperience(
        state={"images": img1[:, None, ...]},
        obs=(img1, img2),
        ground_truth=flow_gt,
    )
    step = jax.jit(est.create_train_step())
    loss0, ts, _ = step(ts, exp)
    for _ in range(200):
        loss, ts, _ = step(ts, exp)
    pred, _e, _m = est._estimate(img2, {"images": img1[:, None, ...]}, ts, {})
    baseline = float(jnp.mean(jnp.abs(flow_gt)))  # error of a zero prediction
    mae = float(jnp.mean(jnp.abs(pred - flow_gt)))
    assert float(loss) < float(loss0)
    assert mae < 0.5 * baseline


def test_loss_matches_closed_form_multilevel_weighting():
    """The composed loss equals sum_l w_l (lambda_u*data_l + lambda_J*jac_l)."""
    est = _small_estimator()  # refine_levels=3 -> weights (0.08, 0.16, 0.32)
    B = 1
    ts = est.create_trainable_state(
        jnp.zeros((B, 32, 32)), jax.random.PRNGKey(0)
    )
    sizes = [(4, 4), (8, 8), (16, 16)]  # coarse -> fine native level sizes
    consts = [1.0, 2.0, 3.0]
    gt_val = 0.5

    class _StubModel:
        def apply(self, _variables, _images, _flow_init=None):
            return [
                jnp.full((B, h, w, 2), c)
                for (h, w), c in zip(sizes, consts, strict=True)
            ]

    est.model = _StubModel()  # constant fields -> Jacobian penalty is exactly 0
    train_step = est.create_train_step()
    gt = jnp.full((B, 32, 32, 2), gt_val)
    exp = SupervisedExperience(
        state={"images": jnp.zeros((B, 1, 32, 32))},
        obs=(jnp.zeros((B, 32, 32)), jnp.zeros((B, 32, 32))),
        ground_truth=gt,
    )
    loss, _ts, metrics = train_step(ts, exp)
    # data_l = sum over 2 channels of |c - gt| = 2|c - 0.5|
    data = [2.0 * abs(c - gt_val) for c in consts]
    weights = (0.08, 0.16, 0.32)
    expected = est.lambda_u * sum(
        w * d for w, d in zip(weights, data, strict=True)
    )
    assert float(loss) == pytest.approx(expected, rel=1e-5)
    assert float(metrics["jacobian_loss"]) == pytest.approx(0.0, abs=1e-6)
    assert float(metrics["data_loss"]) == pytest.approx(sum(data), rel=1e-5)


def test_temporal_propagation_path_runs_and_uses_prior():
    """use_temporal_propagation feeds the prior estimate into the model."""
    est = _small_estimator(use_temporal_propagation=True)
    B, H, W = 1, 40, 48  # non-divisible -> also exercises flow_init padding
    ts = est.create_trainable_state(jnp.zeros((B, H, W)), jax.random.PRNGKey(0))
    img = jax.random.normal(jax.random.PRNGKey(1), (B, H, W)) * 30 + 120
    prev = jax.random.normal(jax.random.PRNGKey(2), (B, H, W)) * 30 + 120
    zero_state = {
        "images": prev[:, None, ...],
        "estimates": jnp.zeros((B, 1, H, W, 2)),
    }
    warm_state = {
        "images": prev[:, None, ...],
        "estimates": jnp.full((B, 1, H, W, 2), 2.5),
    }
    flow_zero, _e, _m = est._estimate(img, zero_state, ts, {})
    flow_warm, _e, _m = est._estimate(img, warm_state, ts, {})
    assert flow_zero.shape == (B, H, W, 2)
    assert jnp.all(jnp.isfinite(flow_warm))
    assert not jnp.allclose(flow_zero, flow_warm)


def test_registered_in_all_estimators_and_from_config():
    """LIMA is registered and buildable via make_estimator's from_config."""
    import flowgym

    assert flowgym.ALL_ESTIMATORS["lima_piv"] is LimaPivEstimator
    # from_config filters injected keys (e.g. estimate_shape, jit) and accepts
    # YAML-style list hyperparameters and an optimizer_config.
    est = LimaPivEstimator.from_config(
        {
            "encoder_channels": [16, 32, 64],
            "refine_levels": 3,
            "search_range": 2,
            "padding_mode": "replicate",
            "optimizer_config": {"name": "adam", "learning_rate": 1e-4},
            "estimate_shape": (32, 32, 2),  # injected by make_estimator
            "jit": True,  # injected by make_estimator
            "history_size": 2,  # injected by make_estimator
        }
    )
    assert isinstance(est, LimaPivEstimator)
    assert est.refine_levels == 3
    ts = est.create_trainable_state(
        jnp.zeros((1, 32, 32)), jax.random.PRNGKey(0)
    )
    flow, _e, _m = est._estimate(
        jnp.zeros((1, 32, 32)),
        {"images": jnp.zeros((1, 1, 32, 32))},
        ts,
        {},
    )
    assert flow.shape == (1, 32, 32, 2)


def test_train_step_rejects_non_divisible_images():
    """Training images must be a multiple of 2**levels (clear error)."""
    est = _small_estimator()  # 3 levels -> multiple of 8
    ts = est.create_trainable_state(
        jnp.zeros((1, 32, 32)), jax.random.PRNGKey(0)
    )
    train_step = est.create_train_step()
    B, H, W = 1, 30, 32  # 30 not divisible by 8
    exp = SupervisedExperience(
        state={"images": jnp.zeros((B, 1, H, W))},
        obs=(jnp.zeros((B, H, W)), jnp.zeros((B, H, W))),
        ground_truth=jnp.zeros((B, H, W, 2)),
    )
    with pytest.raises(ValueError):
        train_step(ts, exp)
