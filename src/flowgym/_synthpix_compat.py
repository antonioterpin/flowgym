"""Compatibility shim for synthpix sampler restore.

synthpix's :class:`SyntheticImageSampler.restore_state` substitutes a
``jax.ShapeDtypeStruct((np.nan,), jnp.uint8)`` template placeholder for
the ``files_scheduler`` field whenever the underlying scheduler state
was ``None`` at save time. The intent is "unknown dim — let orbax read
the real shape from disk metadata".

orbax 0.11+ rejects NaN-valued shape entries during deserialization with
``TypeError: 'float' object cannot be interpreted as an integer`` at
``orbax/checkpoint/_src/arrays/numpy_utils.py:53: slice(*x.indices(n))``.
The intended "use the saved shape" semantics is already what orbax does
for ``None`` template leaves, so we patch ``restore_state`` to replace
NaN-shape placeholders with ``None``.

The shim is idempotent and a no-op when synthpix is not installed.
Tracked upstream at synthpix `src/synthpix/sampler/synthetic.py` (the
``np.nan`` sentinel is still on `main`); remove this shim once a
synthpix release drops the NaN placeholder.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_INSTALLED_ATTR = "_flowgym_nan_shape_shim"


def _has_nan_shape(value: Any) -> bool:
    """Detect a ShapeDtypeStruct whose shape contains a NaN dimension.

    Args:
        value: Any object; only ``jax.ShapeDtypeStruct`` instances are
            inspected, every other type returns ``False``.

    Returns:
        ``True`` iff ``value`` is a ``jax.ShapeDtypeStruct`` and at
        least one entry of its ``shape`` tuple is a NaN float.
    """
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if not isinstance(value, jax.ShapeDtypeStruct):
        return False
    return any(isinstance(d, float) and np.isnan(d) for d in value.shape)


def _scrub_nan_placeholders(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Replace NaN-shape ShapeDtypeStruct leaves with ``None`` in-place.

    Args:
        state_dict: Mutable mapping returned by synthpix's
            ``restore_state``. Entries with NaN-shape placeholders are
            rewritten to ``None`` so orbax falls back to the on-disk
            shape during restore.

    Returns:
        The same ``state_dict`` (mutated), for ergonomics.
    """
    for key, value in list(state_dict.items()):
        if _has_nan_shape(value):
            state_dict[key] = None
    return state_dict


def install_restore_state_shim() -> bool:
    """Patch ``SyntheticImageSampler.restore_state`` to scrub NaN shapes.

    No-ops when synthpix is not importable and is idempotent: a second
    call on an already-patched class returns ``False``.

    Returns:
        ``True`` when a fresh patch was applied, ``False`` when synthpix
        is missing or already patched.
    """
    try:
        from synthpix.sampler.synthetic import (  # noqa: PLC0415
            SyntheticImageSampler,
        )
    except ImportError:
        return False

    cls = SyntheticImageSampler
    original_prop = cls.__dict__.get("restore_state")
    if not isinstance(original_prop, property) or original_prop.fget is None:
        # synthpix layout changed; surface but do not raise so flowgym
        # imports do not break on unexpected sampler versions.
        logger.debug(
            "synthpix.SyntheticImageSampler.restore_state is not a "
            "readable property; skipping NaN-shape compat shim."
        )
        return False
    original_fget = original_prop.fget
    if getattr(original_fget, _INSTALLED_ATTR, False):
        return False

    def restore_state(self: Any) -> dict[str, Any]:
        return _scrub_nan_placeholders(original_fget(self))

    setattr(restore_state, _INSTALLED_ATTR, True)
    restore_state.__doc__ = original_fget.__doc__
    # Replacing the class-level property at runtime: synthpix's
    # ``restore_state`` upstream has no setter, so basedpyright flags
    # the attribute assignment. We're intentionally swapping the
    # property descriptor on the class object itself — a class-attribute
    # write, not an instance-attribute write — which is the standard
    # monkey-patch idiom. Targeted ignore documents the reason.
    cls.restore_state = property(  # pyright: ignore[reportAttributeAccessIssue]
        restore_state
    )
    return True
