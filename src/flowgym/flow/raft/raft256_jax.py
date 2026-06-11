"""RaftJax256Estimator class.

The RAFT256-PIV estimator reuses the full :class:`RaftJaxEstimator` pipeline
(patchify/fold with the spline window, training step and EPE caching) and only
swaps in the 1/8-resolution :class:`RaftEstimatorModel256` architecture. The
defaults follow the published RAFT256-PIV evaluation configuration: 256x256
interrogation windows with a 64 px shift and 16 refinement iterations.
"""

from __future__ import annotations

from typing import Any

from flax import linen as nn

from flowgym.flow.raft.raft_jax import RaftJaxEstimator
from flowgym.nn.raft256_model import RaftEstimatorModel256


class RaftJax256Estimator(RaftJaxEstimator):
    """RAFT256-PIV estimator: 1/8-resolution convex-upsampling RAFT variant."""

    def __init__(
        self,
        patch_size: int = 256,
        patch_stride: int = 64,
        iters: int = 16,
        **kwargs: Any,
    ):
        """Initialize the RAFT256 estimator.

        Args:
            patch_size: Size of the interrogation windows to process.
            patch_stride: Shift between consecutive windows.
            iters: Number of refinement iterations for flow refinement.
            **kwargs: Additional keyword arguments forwarded to
                :class:`RaftJaxEstimator`.
        """
        super().__init__(
            patch_size=patch_size,
            patch_stride=patch_stride,
            iters=iters,
            **kwargs,
        )

    def _build_model(self) -> nn.Module:
        """Build the 1/8-resolution RAFT256 Flax model.

        Returns:
            The :class:`RaftEstimatorModel256` module.
        """
        return RaftEstimatorModel256(
            hidden_dim=self.hidden_dim,
            context_dim=self.context_dim,
            corr_levels=self.corr_levels,
            corr_radius=self.corr_radius,
            iters=self.iters,
            norm_fn=self.norm_fn,
            dropout=self.dropout,
            train=self.train,
        )
