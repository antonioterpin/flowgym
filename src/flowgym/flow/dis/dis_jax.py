"""DISFlowFieldEstimator class."""

from typing import Any
import jax.numpy as jnp
from enum import Enum

from goggles.history.types import History
from flowgym.flow.base import FlowFieldEstimator
from flowgym.flow.dis import process


class PresetType(Enum):
    """DIS preset types."""

    ULTRAFAST = 0
    FAST = 1
    MEDIUM = 2
    HIGH_QUALITY = 3


class DISJAXFlowFieldEstimator(FlowFieldEstimator):
    """Dense Inverse Search (DIS) flow field estimator using two-frame history."""

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
            kwargs: Additional keyword arguments for the base class.
        """
        # Validate and convert preset
        if isinstance(preset, int):
            try:
                preset_enum = PresetType(preset)
            except ValueError:
                raise ValueError(
                    f"preset={preset}, but it must be 0, 1, 2, or 3 "
                    f"({', '.join(f'{e.value}={e.name.lower()}' for e in PresetType)})"
                )
        elif isinstance(preset, PresetType):
            preset_enum = preset
        else:
            raise TypeError(f"preset={preset}, but it must be an int or PresetType.")

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
            raise ValueError(
                f"preset={preset}, but it must be 0, 1, 2, or 3 "
                f"({', '.join(f'{e.value}={e.name.lower()}' for e in PresetType)})"
            )

        for name, val in [("patch_size", patch_size), ("patch_stride", patch_stride)]:
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name}={val}, but it must be a positive integer.")
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size={patch_size}, but it must be an odd integer.")
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
                f"start_level={start_level}, but it must be a non-negative integer."
            )
        self.start_level = start_level

        if not isinstance(levels, int) or levels <= 0:
            raise TypeError(f"levels={levels}, but it must be a positive integer.")
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
        __: None,
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the flow field between the two most recent frames.

        Args:
            image: Current batch of frames, shape (B, H, W).
            state: Contains history_images of shape (B, 1, H, W).
            _: Unused parameter.
            __: Unused parameter.

        Returns:
            - Flow field of shape (B, H, W, 2) as float32.
            - placeholder for additional output.
            - placeholder for metrics.
        """
        # Convert to
        prev = state["images"][:, 0, ...]
        curr = image
        if self.use_temporal_propagation:
            flow = state["estimates"][:, -1, ...]
        else:
            # Initialize flow to zeros of the same shape as prev but with 2 channels
            flow = jnp.zeros(
                (prev.shape[0], prev.shape[1], prev.shape[2], 2), dtype=jnp.float32
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
