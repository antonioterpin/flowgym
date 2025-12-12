"""DISFlowFieldEstimator class."""

from typing import Any
import cv2
import numpy as np
import jax.numpy as jnp

from flowgym.flow.base import FlowFieldEstimator
from goggles.history.types import History


class DISFlowFieldEstimator(FlowFieldEstimator):
    """Dense Inverse Search (DIS) flow field estimator using two-frame history."""

    def __init__(
        self,
        preset: int | None = None,
        patch_size: int | None = None,
        patch_stride: int | None = None,
        grad_desc_iters: int | None = None,
        var_refine_iters: int | None = None,
        var_ref_alpha: float | None = None,
        var_ref_delta: float | None = None,
        var_ref_gamma: float | None = None,
        use_mean_normalization: bool | None = None,
        use_spatial_propagation: bool | None = None,
        finest_scale: int | None = None,
        **kwargs: Any,
    ):
        """Initialize the DIS estimator.

        Args:
            preset: DIS preset (0=ultrafast,1=fast,2=medium,3=high_quality).
            patch_size: Size of matching patches (in pixels).
            patch_stride: Stride between patches (in pixels).
            grad_desc_iters: Number of gradient descent iterations per patch.
            var_refine_iters: Number of variational refinement iterations.
            var_ref_alpha: Alpha parameter for variational refinement.
            var_ref_delta: Delta parameter for variational refinement.
            var_ref_gamma: Gamma parameter for variational refinement.
            use_mean_normalization: Enable mean normalization of patches.
            use_spatial_propagation: Enable spatial propagation of flow.
            finest_scale: Finest scale for multi-scale processing.
            kwargs: Additional keyword arguments for the base class.
        """
        # Validate DIS specific parameters
        if preset is not None and preset not in (0, 1, 2, 3):
            raise ValueError(f"preset {preset} must be 0,1,2, or 3.")
        self.dis = cv2.DISOpticalFlow_create(preset)  # type: ignore

        if patch_size is not None and (
            not isinstance(patch_size, int) or patch_size <= 0
        ):
            raise ValueError(f"patch_size {patch_size} must be a positive integer.")
        if patch_size is not None:
            self.dis.setPatchSize(patch_size)

        if patch_stride is not None and (
            not isinstance(patch_stride, int) or patch_stride <= 0
        ):
            raise ValueError(f"patch_stride {patch_stride} must be a positive integer.")
        if patch_stride is not None:
            self.dis.setPatchStride(patch_stride)

        if grad_desc_iters is not None and (
            not isinstance(grad_desc_iters, int) or grad_desc_iters < 0
        ):
            raise ValueError(
                f"grad_desc_iters {grad_desc_iters} must be a non-negative integer."
            )
        if grad_desc_iters is not None:
            self.dis.setGradientDescentIterations(grad_desc_iters)

        if var_refine_iters is not None and (
            not isinstance(var_refine_iters, int) or var_refine_iters < 0
        ):
            raise ValueError(
                f"var_refine_iters {var_refine_iters} must be a non-negative integer."
            )
        if var_refine_iters is not None:
            self.dis.setVariationalRefinementIterations(var_refine_iters)

        if use_mean_normalization is not None and not isinstance(
            use_mean_normalization, bool
        ):
            raise ValueError(
                f"use_mean_normalization {use_mean_normalization} must be a boolean."
            )
        if use_mean_normalization is not None:
            self.dis.setUseMeanNormalization(bool(use_mean_normalization))

        if use_spatial_propagation is not None and not isinstance(
            use_spatial_propagation, bool
        ):
            raise ValueError(
                f"use_spatial_propagation {use_spatial_propagation} must be a boolean."
            )
        if use_spatial_propagation is not None:
            self.dis.setUseSpatialPropagation(bool(use_spatial_propagation))

        if finest_scale is not None and (
            not isinstance(finest_scale, int) or finest_scale < 0
        ):
            raise ValueError(
                f"finest_scale {finest_scale} must be a non-negative integer."
            )
        if finest_scale is not None:
            self.dis.setFinestScale(finest_scale)

        if var_ref_alpha is not None and (
            not isinstance(var_ref_alpha, float) or var_ref_alpha < 0
        ):
            raise ValueError(
                f"var_ref_alpha {var_ref_alpha} must be a non-negative float."
            )
        if var_ref_alpha is not None:
            self.dis.setVariationalRefinementAlpha(var_ref_alpha)

        if var_ref_delta is not None and (
            not isinstance(var_ref_delta, float) or var_ref_delta < 0
        ):
            raise ValueError(
                f"var_ref_delta {var_ref_delta} must be a non-negative float."
            )
        if var_ref_delta is not None:
            self.dis.setVariationalRefinementDelta(var_ref_delta)

        if var_ref_gamma is not None and (
            not isinstance(var_ref_gamma, float) or var_ref_gamma < 0
        ):
            raise ValueError(
                f"var_ref_gamma {var_ref_gamma} must be a non-negative float."
            )
        if var_ref_gamma is not None:
            self.dis.setVariationalRefinementGamma(var_ref_gamma)

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
            Flow field of shape (B, H, W, 2) as float32.
            placeholder for additional output.
            placeholder for metrics.
        """
        # Convert to numpy for OpenCV
        prev = np.asarray(state["images"][:, 0, ...], dtype=np.float32)
        curr = np.asarray(image, dtype=np.float32)

        # OpenCV DIS requires 8-bit single-channel images
        prev_uint8 = np.clip(prev * 255.0, 0, 255).astype(np.uint8)
        curr_uint8 = np.clip(curr * 255.0, 0, 255).astype(np.uint8)

        batch_size, H, W = curr.shape
        flows = np.zeros((batch_size, H, W, 2), dtype=np.float32)

        # Loop per batch element
        for i in range(batch_size):
            flows[i] = self.dis.calc(prev_uint8[i], curr_uint8[i], None)

        return jnp.asarray(flows), {}, {}
