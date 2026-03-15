"""DISFlowFieldEstimator class."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from goggles.history.types import History

if TYPE_CHECKING:
    from synthpix import SynthpixBatch

from flowgym.common.base.trainable_state import EstimatorTrainableState
from flowgym.flow.base import FlowFieldEstimator
from flowgym.flow.dis import process


class PresetType(Enum):
    """DIS preset types.

    Attributes:
        ULTRAFAST: Ultrafast preset (value 0).
        FAST: Fast preset (value 1).
        MEDIUM: Medium preset (value 2).
        HIGH_QUALITY: High quality preset (value 3).
    """

    ULTRAFAST = 0
    FAST = 1
    MEDIUM = 2
    HIGH_QUALITY = 3


class DISJAXFlowFieldEstimator(FlowFieldEstimator):
    """Dense Inverse Search (DIS) flow estimator with two-frame history."""

    def __init__(
        self,
        preset: PresetType | int = 1,
        start_level: int = 0,
        levels: int = 4,
        level_steps: int = 1,
        patch_size: int = 9,
        patch_stride: int = 4,
        grad_desc_iters: int = 4,
        var_refine_iters: int = 0,
        use_mean_normalization: bool = True,
        use_spatial_propagation: bool = True,
        use_temporal_propagation: bool = False,
        output_full_res: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the DIS estimator.

        Args:
            preset: DIS preset (0=ultrafast,1=fast,2=medium,3=high_quality).
            start_level: Starting level for the pyramid.
            levels: Number of levels for the pyramid.
            level_steps: Number of steps between levels.
            patch_size: Size of matching patches (in pixels).
            patch_stride: Stride between patches (in pixels).
            grad_desc_iters: Number of gradient descent iterations per patch.
            var_refine_iters: Number of variational refinement iterations.
            use_mean_normalization: Enable mean normalization of patches.
            use_spatial_propagation: Enable spatial propagation of flow.
            use_temporal_propagation: Enable temporal propagation of flow.
            output_full_res: Output full resolution flow field.
            **kwargs: Additional keyword arguments for the base class.

        Raises:
            ValueError: If parameter validation fails.
            TypeError: If parameter type is incorrect.
        """
        # Validate and convert preset
        if isinstance(preset, int):
            try:
                preset_enum = PresetType(preset)
            except ValueError:
                preset_vals = ", ".join(
                    f"{e.value}={e.name.lower()}" for e in PresetType
                )
                raise ValueError(
                    f"preset={preset}, but it must be 0, 1, 2, or 3 "
                    f"({preset_vals})"
                ) from None
        elif isinstance(preset, PresetType):
            preset_enum = preset
        else:
            raise TypeError(
                f"preset={preset}, but it must be an int or PresetType."
            )

        self.preset = preset_enum

        if self.preset == PresetType.ULTRAFAST:
            # Ultrafast preset
            self.patch_size = 8
            self.patch_stride = 6
            self.grad_desc_iters = 16
            self.var_refine_iters = 0
            self.start_level = 3
            self.levels = 2
        elif self.preset == PresetType.FAST:
            # Fast preset
            self.patch_size = 8
            self.patch_stride = 5
            self.grad_desc_iters = 12
            self.var_refine_iters = 0  # not implemented yet
            self.start_level = 3
            self.levels = 2

        elif self.preset == PresetType.MEDIUM:
            # Medium preset
            self.patch_size = 12
            self.patch_stride = 4
            self.grad_desc_iters = 16
            self.var_refine_iters = 0  # not implemented yet
            self.start_level = 1
            self.levels = 4

        elif self.preset == PresetType.HIGH_QUALITY:
            # High quality preset
            self.patch_size = 12
            self.patch_stride = 4
            self.grad_desc_iters = 256
            self.var_refine_iters = 0  # not implemented yet
            self.start_level = 0
            self.levels = 5
        else:
            preset_vals = ", ".join(
                f"{e.value}={e.name.lower()}" for e in PresetType
            )
            raise ValueError(
                f"preset={preset}, but it must be 0, 1, 2, or 3 ({preset_vals})"
            )

        for name, val in [
            ("patch_size", patch_size),
            ("patch_stride", patch_stride),
        ]:
            if not isinstance(val, int) or val <= 0:
                raise ValueError(
                    f"{name}={val}, but it must be a positive integer."
                )
        if patch_size % 2 == 0:
            raise ValueError(
                f"patch_size={patch_size}, but it must be an odd integer."
            )
        if patch_stride > patch_size:
            raise ValueError(
                f"patch_stride={patch_stride}, but it must be less "
                "than or equal to patch_size."
            )
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        for name, val in [
            ("grad_desc_iters", grad_desc_iters),
            ("var_refine_iters", var_refine_iters),
        ]:
            if not isinstance(val, int) or val < 0:
                raise ValueError(
                    f"{name}={val}, but it must be a non-negative integer."
                )
        self.grad_desc_iters = grad_desc_iters
        self.var_refine_iters = var_refine_iters

        if not isinstance(start_level, int) or start_level < 0:
            raise ValueError(
                f"start_level={start_level}, but it must be a "
                "non-negative integer."
            )
        self.start_level = start_level

        if not isinstance(levels, int) or levels <= 0:
            raise TypeError(
                f"levels={levels}, but it must be a positive integer."
            )
        self.levels = levels

        if not isinstance(level_steps, int) or level_steps <= 0:
            raise TypeError(
                f"level_steps={level_steps}, but it must be a positive integer."
            )
        self.level_steps = level_steps

        if not isinstance(output_full_res, bool):
            raise TypeError(
                f"output_full_res={output_full_res}, but it must be a boolean."
            )
        self.output_full_res = output_full_res

        self.use_mean_normalization = bool(use_mean_normalization)
        self.use_spatial_propagation = bool(use_spatial_propagation)
        self.use_temporal_propagation = bool(use_temporal_propagation)

        super().__init__(**kwargs)

    def _estimate(
        self,
        image: jnp.ndarray,
        state: History,
        _: None,
        extras: dict,
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the flow field between the two most recent frames.

        Args:
            image: Current batch of frames, shape (B, H, W).
            state: Contains history_images of shape (B, 1, H, W).
            extras: Additional data from history fields, may contain
                cache_payload.

        Returns:
            - Flow field of shape (B, H, W, 2) as float32.
            - placeholder for additional output.
            - placeholder for metrics.
        """
        cache_payload = extras.get("cache_payload")
        if (
            cache_payload is not None
            and getattr(cache_payload, "has_precomputed_errors", False)
            and getattr(cache_payload, "epe", None) is not None
        ):
            B, H, W = image.shape
            if self.use_temporal_propagation:
                flows = state["estimates"][:, -1, ...]
            else:
                flows = jnp.zeros((B, H, W, 2), dtype=jnp.float32)

            metrics: dict[str, jnp.ndarray] = {
                "errors": jnp.asarray(cache_payload.epe)
            }
            relative_epe = getattr(cache_payload, "relative_epe", None)
            if relative_epe is not None:
                metrics["relative_errors"] = jnp.asarray(relative_epe)
            return flows, {}, metrics

        # Convert to
        prev = state["images"][:, 0, ...]
        curr = image
        if self.use_temporal_propagation:
            flow = state["estimates"][:, -1, ...]
        else:
            # Initialize flow to zeros of same shape as prev with 2 channels
            flow = jnp.zeros(
                (*prev.shape[:3], 2),
                dtype=jnp.float32,
            )

        # Process the images to estimate the flow
        flows = process.estimate_dis_flow(
            prev,
            curr,
            start_level=self.start_level,
            levels=self.levels,
            level_steps=self.level_steps,
            grad_desc_iters=self.grad_desc_iters,
            patch_stride=self.patch_stride,
            patch_size=self.patch_size,
            output_full_res=self.output_full_res,
            starting_flow=flow,
            var_refine_iters=self.var_refine_iters,
        )
        return flows, {}, {}

    def get_config(self) -> dict[str, Any]:
        """Get the configuration of the estimator.

        Returns:
            Dictionary containing the configuration parameters.
        """
        return {
            "preset": self.preset.name,
            "start_level": self.start_level,
            "levels": self.levels,
            "level_steps": self.level_steps,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "grad_desc_iters": self.grad_desc_iters,
            "var_refine_iters": self.var_refine_iters,
            "use_mean_normalization": self.use_mean_normalization,
            "use_spatial_propagation": self.use_spatial_propagation,
            "use_temporal_propagation": self.use_temporal_propagation,
            "output_full_res": self.output_full_res,
        }

    def get_cache_id_suffix(
        self,
        trainable_state: EstimatorTrainableState,
    ) -> str:
        """Get a suffix for the cache ID based on the model configuration.

        Args:
            trainable_state: Unused for DIS as it is parameter-free (mostly).

        Returns:
            A string hash of the configuration.
        """
        config = self.get_config()
        # Sort keys to ensure deterministic order
        config_str = json.dumps(config, sort_keys=True)
        h = hashlib.md5(config_str.encode("utf-8")).hexdigest()
        return f"_c{h[:8]}"

    def enrich(
        self,
        batch: SynthpixBatch,
        miss_idxs: jnp.ndarray,
        **kwargs: Any,
    ) -> dict[str, jnp.ndarray] | None:
        """Compute the cache payload for missing keys (EPE).

        Args:
            batch: The batch of data.
            miss_idxs: Indices of samples in the batch that are missing
                from cache.
            **kwargs: Unused.

        Returns:
            Dictionary with "epe" values for the missing indices.
        """
        if batch.flow_fields is None:
            return None

        # Slice batch for missing indices (eager, host-side)
        images1 = batch.images1[miss_idxs]
        images2 = batch.images2[miss_idxs]
        gt_flow = batch.flow_fields[miss_idxs]

        key = jax.random.PRNGKey(0)
        t_state = self.create_trainable_state(images1, key)

        # JIT-compiled computation
        epe_mean, rel_mean = self._compute_cache_payload(
            images1, images2, gt_flow, t_state, key
        )

        return {"epe": epe_mean, "relative_epe": rel_mean}

    @partial(jax.jit, static_argnums=0)
    def _compute_cache_payload(
        self,
        images1: jnp.ndarray,
        images2: jnp.ndarray,
        gt_flow: jnp.ndarray,
        t_state: EstimatorTrainableState,
        rng: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Pure JAX computation of EPE metrics (JIT-compiled).

        Args:
            images1: First frame images (B, H, W).
            images2: Second frame images (B, H, W).
            gt_flow: Ground truth flow fields (B, H, W, 2).
            t_state: Trainable state.
            rng: Random key for state creation.

        Returns:
            Tuple of (epe_mean, relative_epe_mean), both shape (B,).
        """
        B, H, W = images1.shape
        init_flow = jnp.zeros((B, H, W, 2), dtype=jnp.float32)

        state = self.create_state(
            images1, init_flow, image_history_size=2, rng=rng
        )
        state, _ = self(images2, state, t_state)

        est_flow = state["estimates"][:, -1]

        diff = est_flow - gt_flow
        epe = jnp.linalg.norm(diff, axis=-1)
        epe_mean = jnp.mean(epe, axis=(1, 2))

        gt_norm = jnp.linalg.norm(gt_flow, axis=-1)
        rel = (epe**2) / (jnp.maximum(gt_norm, 0.01) ** 2)
        rel_mean = jnp.mean(rel, axis=(1, 2))

        return epe_mean, rel_mean
