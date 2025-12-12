import pytest
import jax
import jax.numpy as jnp

from goggles.history.spec import HistorySpec
from goggles.history import create_history
from goggles.history.types import History
from flowgym.common.base import Estimator


class DummyEstimator(Estimator):
    """Minimal concrete estimator for testing __call__ behavior."""

    def _estimate(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state,
        extras: dict,
    ):
        """Use RNG keys if present, otherwise deterministic estimate from images.

        Returns:
            estimates: (B, 2)
            extras: passed through (unused when there are no extra fields)
            metrics: dict with a simple scalar metric.
        """
        # If RNG subkeys are provided for this step, use them to generate estimates.
        if "keys" in state:
            # state["keys"] is expected to be shape (B, 2) here
            estimates = jax.vmap(jax.random.normal, in_axes=[0, None])(
                state["keys"], (2,)
            )  # (B, 2)
        else:
            # Fallback deterministic estimate: use mean over spatial dims
            mean_val = images.mean(axis=(1, 2), keepdims=False)  # (B,)
            estimates = jnp.stack([mean_val, mean_val], axis=-1)  # (B, 2)

        # For these tests we don't use extras, but we must return something.
        if extras is None:
            extras = {}
        metrics = {"mean_image": images.mean()}
        return estimates, extras, metrics


def _make_state(
    B: int,
    H: int,
    W: int,
    img_T: int,
    est_T: int,
    rng=None,
) -> History:
    """Helper to create a device-resident history with given lengths."""
    cfg = {
        "images": {
            "length": img_T,
            "shape": (H, W),
            "dtype": jnp.float32,
            "init": "zeros",
        },
        "estimates": {
            "length": est_T,
            "shape": (2,),
            "dtype": jnp.float32,
            "init": "zeros",
        },
    }
    spec = HistorySpec.from_config(cfg)
    state = create_history(spec, B, rng)
    return state


@pytest.fixture
def estimator():
    return DummyEstimator()


def test_call_rolls_and_appends_without_keys(estimator):
    """__call__ should roll histories and append new images/estimates when no keys."""
    B, H, W = 3, 8, 8
    IMG_T, EST_T = 3, 2

    # Initial history
    state = _make_state(B, H, W, IMG_T, EST_T, rng=None)

    # Seed prior content to observe rolling
    images_before = jnp.arange(B * IMG_T * H * W, dtype=jnp.float32).reshape(
        B, IMG_T, H, W
    )
    estimates_before = jnp.arange(B * EST_T * 2, dtype=jnp.float32).reshape(B, EST_T, 2)
    state["images"] = images_before
    state["estimates"] = estimates_before

    # New incoming images
    images = jnp.ones((B, H, W), dtype=jnp.float32) * 7.0

    new_state, metrics = estimator(images, state, trainable_state=None)

    # Shapes preserved
    assert new_state["images"].shape == (B, IMG_T, H, W)
    assert new_state["estimates"].shape == (B, EST_T, 2)

    # Last frame equals the input images
    assert jnp.allclose(new_state["images"][:, -1], images)

    # Rolling occurred: previous t=1 becomes new t=0
    assert jnp.allclose(new_state["images"][:, 0], images_before[:, 1])
    assert jnp.allclose(new_state["estimates"][:, 0], estimates_before[:, 1])

    # Metrics returned
    assert "mean_image" in metrics
    assert isinstance(metrics["mean_image"], jnp.ndarray) or isinstance(
        metrics["mean_image"], (float, int)
    )


def test_call_splits_keys_and_updates_estimates(estimator):
    """__call__ should split per-batch keys and use subkeys for stochastic estimates."""
    B, H, W = 4, 6, 5
    IMG_T, EST_T = 2, 2

    state = _make_state(B, H, W, IMG_T, EST_T, rng=None)

    master = jax.random.PRNGKey(0)
    keys = jax.random.split(master, B)  # (B, 2)
    state["keys"] = keys[:, None, :]  # (B, 1, 2) — matches history convention

    # Expected split: per-example split → (B, 2, 2)
    pair = jax.vmap(jax.random.split, in_axes=[0, None])(keys, 2)
    new_keys = pair[:, 0, :]  # (B, 2)
    subkeys = pair[:, 1, :]  # (B, 2)

    images = jnp.zeros((B, H, W), dtype=jnp.float32)
    new_state, metrics = estimator(images, state, trainable_state=None)

    # Expected estimates drawn from subkeys: (B, 2)
    expected_estimates = jax.vmap(jax.random.normal, in_axes=[0, None])(subkeys, (2,))

    # Latest estimates frame equals expected
    assert jnp.allclose(new_state["estimates"][:, -1, :], expected_estimates)

    # Keys stay as a history field with same temporal length (here T=1 → remains 1)
    assert "keys" in new_state
    assert new_state["keys"].shape == (B, 1, 2)

    # The last (and only) keys frame should match the new_keys we computed.
    assert jnp.all(new_state["keys"][:, -1, :] == new_keys)

    # Metrics still present
    assert "mean_image" in metrics


def test_preprocessing_pipeline_applied(estimator):
    """Preprocessing steps must be applied before _estimate and history update."""
    B, H, W = 2, 4, 4
    IMG_T, EST_T = 3, 3

    # Start with zeros history
    state = _make_state(B, H, W, IMG_T, EST_T, rng=None)

    # Patch a simple preprocessing step that adds 1.5 to images
    def add_const(images: jnp.ndarray) -> jnp.ndarray:
        return images + 1.5

    # Inject directly (bypass validate_params plumbing)
    estimator.preprocessing_steps = [add_const]

    images = jnp.zeros((B, H, W), dtype=jnp.float32)
    new_state, metrics = estimator(images, state, trainable_state=None)

    # Last images frame should reflect preprocessing
    assert jnp.allclose(
        new_state["images"][:, -1],
        jnp.full((B, H, W), 1.5, dtype=jnp.float32),
    )

    # _estimate should have seen preprocessed images as well,
    # indirectly verified by the mean_image metric.
    assert "mean_image" in metrics
    assert pytest.approx(float(metrics["mean_image"])) == 1.5


def test_batch_must_match_images_and_estimates(estimator):
    """If _estimate returns a different batch size, __call__ should fail."""
    B, H, W = 3, 5, 5
    IMG_T, EST_T = 2, 2
    state = _make_state(B, H, W, IMG_T, EST_T, rng=None)

    # Make estimator produce a mismatched batch on purpose by overriding _estimate
    def bad_estimate(images, state, trainable_state, extras):
        # Return (B-1, 2) to trigger a shape error inside update_history
        return jnp.zeros((images.shape[0] - 1, 2), dtype=jnp.float32), extras, {}

    estimator._estimate = bad_estimate  # type: ignore[assignment]

    images = jnp.zeros((B, H, W), dtype=jnp.float32)
    with pytest.raises(Exception):
        # Could raise ValueError from update_history, or a downstream shape error
        estimator(images, state, trainable_state=None)
