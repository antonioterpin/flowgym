"""Unit tests for flowgym.training.target_transforms.

Covers build_target_transform_from_config for all named transforms,
pipelines, nested kwargs, and error paths.
"""

import jax.numpy as jnp
import pytest

from flowgym.training.target_transforms import (
    build_target_transform_from_config,
)


def test_build_identity():
    """None and 'identity' configs both produce a pass-through transform."""
    transform = build_target_transform_from_config(None)
    x = jnp.array([1.0, 2.0, 3.0])
    assert jnp.allclose(transform(x), x)

    transform = build_target_transform_from_config({"name": "identity"})
    assert jnp.allclose(transform(x), x)

    # identity should swallow extra kwargs from config
    transform = build_target_transform_from_config(
        {"name": "identity", "eps": 1e-8}
    )
    assert jnp.allclose(transform(x), x)


def test_build_log1p():
    """log1p transform applies jnp.log1p(x + eps) element-wise."""
    config = {"name": "log1p", "eps": 1e-8}
    transform = build_target_transform_from_config(config)
    x = jnp.array([0.0, 1.0, 10.0])
    expected = jnp.log1p(x + 1e-8)
    assert jnp.allclose(transform(x), expected)


def test_build_sqrt():
    """sqrt transform applies jnp.sqrt(x + eps) element-wise."""
    config = {"name": "sqrt", "eps": 1e-8}
    transform = build_target_transform_from_config(config)
    x = jnp.array([0.0, 1.0, 4.0])
    expected = jnp.sqrt(x + 1e-8)
    assert jnp.allclose(transform(x), expected)


def test_build_clip():
    """clip transform clamps values to [min, max], supporting partial bounds."""
    config = {"name": "clip", "min": 1.0, "max": 5.0}
    transform = build_target_transform_from_config(config)
    x = jnp.array([0.0, 2.0, 10.0])
    expected = jnp.array([1.0, 2.0, 5.0])
    assert jnp.allclose(transform(x), expected)

    # Partial clip
    config = {"name": "clip", "min": 1.0}
    transform = build_target_transform_from_config(config)
    assert jnp.allclose(transform(x), jnp.array([1.0, 2.0, 10.0]))


def test_build_scale():
    """scale transform multiplies every element by the given factor."""
    config = {"name": "scale", "factor": 2.5}
    transform = build_target_transform_from_config(config)
    x = jnp.array([1.0, 2.0, 3.0])
    expected = jnp.array([2.5, 5.0, 7.5])
    assert jnp.allclose(transform(x), expected)


def test_build_pipeline():
    """pipeline config composes transforms left-to-right in declared order."""
    config = {
        "pipeline": [
            {"name": "clip", "min": 0.0, "max": 10.0},
            {"name": "log1p"},
            {"name": "scale", "factor": 0.5},
        ]
    }
    transform = build_target_transform_from_config(config)
    x = jnp.array([-1.0, 5.0, 20.0])
    # Step 1: clip -> [0.0, 5.0, 10.0]
    # Step 2: log1p -> [log1p(0), log1p(5), log1p(10)]
    # Step 3: scale -> 0.5 * [...]
    expected = 0.5 * jnp.log1p(jnp.array([0.0, 5.0, 10.0]))
    assert jnp.allclose(transform(x), expected)


def test_build_with_nested_kwargs():
    """Nested 'kwargs' dict merges with inline keys when building."""
    config = {"name": "scale", "kwargs": {"factor": 3.0}}
    transform = build_target_transform_from_config(config)
    x = jnp.array([1.0, 2.0])
    assert jnp.allclose(transform(x), jnp.array([3.0, 6.0]))

    # Merged inline and nested
    config = {"name": "log1p", "eps": 1e-8, "kwargs": {}}
    transform = build_target_transform_from_config(config)
    assert jnp.allclose(transform(jnp.array([0.0])), jnp.log1p(1e-8))


def test_unknown_transform():
    """Unrecognized transform name raises a descriptive ValueError."""
    with pytest.raises(ValueError, match="Unknown transform name"):
        build_target_transform_from_config({"name": "nonexistent"})


def test_malformed_pipeline():
    """Non-dict pipeline step raises TypeError naming the offending step."""
    with pytest.raises(
        TypeError, match=r"Pipeline step .* must be a dictionary"
    ):
        build_target_transform_from_config({"pipeline": ["not_a_dict"]})
