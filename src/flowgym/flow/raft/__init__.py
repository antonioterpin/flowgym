"""Module for RAFT estimators."""

from flowgym.utils import MissingDependency, optional_import

from .raft256_jax import RaftJax256Estimator
from .raft_jax import RaftJaxEstimator

raft_torch = optional_import("flowgym.flow.raft.raft_piv_pytorch")
if raft_torch is not None:
    RaftTorchEstimator = raft_torch.RaftTorchEstimator
    RaftTorch256Estimator = raft_torch.RaftTorch256Estimator
else:
    RaftTorchEstimator = MissingDependency(
        "raft_piv_pytorch", ["other_methods"]
    )
    RaftTorch256Estimator = MissingDependency(
        "raft_piv_pytorch", ["other_methods"]
    )

__all__ = [
    "RaftJax256Estimator",
    "RaftJaxEstimator",
    "RaftTorch256Estimator",
    "RaftTorchEstimator",
]
