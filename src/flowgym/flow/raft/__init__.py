"""Module for RAFT estimators."""

from flowgym.utils import optional_import, MissingDependency

from .raft_jax import RaftJaxEstimator

raft_torch = optional_import("flowgym.flow.raft.raft_piv_pytorch")
if raft_torch is not None:
    RaftTorchEstimator = raft_torch.RaftTorchEstimator
else:
    RaftTorchEstimator = MissingDependency("raft_piv_pytorch", ["other_methods"])

__all__ = ["RaftJaxEstimator", "RaftTorchEstimator"]
