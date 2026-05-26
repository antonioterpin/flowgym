"""Tests for the synthpix NaN-shape compat shim."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flowgym._synthpix_compat import (
    _has_nan_shape,
    _scrub_nan_placeholders,
    install_restore_state_shim,
)


def test_has_nan_shape_detects_nan_dim():
    """``_has_nan_shape`` flags ShapeDtypeStructs with NaN in their shape."""
    bad = jax.ShapeDtypeStruct((np.nan,), jnp.uint8)
    good = jax.ShapeDtypeStruct((4,), jnp.uint8)
    assert _has_nan_shape(bad), "NaN-dim ShapeDtypeStruct must be detected"
    assert not _has_nan_shape(good), (
        "Integer-dim ShapeDtypeStruct must not be flagged"
    )
    assert not _has_nan_shape(None), "None passes through unflagged"
    assert not _has_nan_shape(jnp.zeros((2, 3))), (
        "Real arrays must not be flagged"
    )


def test_scrub_replaces_nan_placeholders_with_none():
    """``_scrub_nan_placeholders`` rewrites NaN-shape leaves to None."""
    state = {
        "files_scheduler": jax.ShapeDtypeStruct((np.nan,), jnp.uint8),
        "current_flows": jax.ShapeDtypeStruct((1, 32, 32, 2), jnp.float32),
        "batches_generated": 0,
        "other": jnp.array([1.0, 2.0]),
    }
    out = _scrub_nan_placeholders(state)
    assert out is state, "Should mutate in place and return same dict"
    assert state["files_scheduler"] is None, (
        "NaN-shape placeholder must be replaced with None"
    )
    assert isinstance(state["current_flows"], jax.ShapeDtypeStruct), (
        "Integer-shape placeholders must be left untouched"
    )
    assert state["batches_generated"] == 0, "Scalars must be untouched"
    np.testing.assert_array_equal(state["other"], np.array([1.0, 2.0]))


def test_scrub_is_noop_on_clean_state():
    """A state dict without any NaN placeholders is returned unchanged."""
    state = {
        "step": 5,
        "rng": jnp.array([0, 1], dtype=jnp.uint32),
        "current_flows": jax.ShapeDtypeStruct((1, 32, 32, 2), jnp.float32),
    }
    before = dict(state)
    _scrub_nan_placeholders(state)
    assert state == before, "Clean state must be unchanged after scrubbing"


def test_install_shim_is_idempotent():
    """Installing the shim twice patches at most once."""
    first = install_restore_state_shim()
    second = install_restore_state_shim()
    # First call may have already been performed by ``flowgym.make`` at
    # import time; we only assert that the second call is a no-op.
    assert second is False, (
        "Idempotent install: a repeat call must report False"
    )
    # If the first call here actually patched, it returned True; if a
    # previous import already patched, both are False — both shapes are
    # valid, but we did not accidentally re-patch.
    assert first in (True, False)


def test_shim_marks_patched_property():
    """After installation, the patched ``restore_state`` carries the marker.

    Verifying the marker (rather than driving the property through a
    full sampler) keeps the test independent of synthpix's other
    field substitutions, which require a fully-constructed sampler.
    """
    install_restore_state_shim()
    from synthpix.sampler.synthetic import (
        SyntheticImageSampler,
    )

    prop = SyntheticImageSampler.__dict__["restore_state"]
    assert isinstance(prop, property), "restore_state must remain a property"
    assert getattr(prop.fget, "_flowgym_nan_shape_shim", False), (
        "Patched fget must carry the idempotency marker"
    )
