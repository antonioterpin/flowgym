"""Consensus specific types."""

from jax import Array
from typing import Any, TypeAlias
from collections.abc import Callable

ObjectiveFn: TypeAlias = Callable[..., Any]

FlowSolver: TypeAlias = Callable[
    [Array, Array, Array, ObjectiveFn, float, int | None],
    Array,
]

ConsensusSolver: TypeAlias = Callable[
    [Array, Array, Array, ObjectiveFn, float, int],
    Array,
]

FlowSolverFactory: TypeAlias = Callable[[float], FlowSolver]
ConsensusSolverFactory: TypeAlias = Callable[[float], ConsensusSolver]

AdmmCarry: TypeAlias = tuple[
    Array,  # flows
    Array,  # consensus_flow
    Array,  # consensus_dual
    Array,  # active (scalar bool array)
    Array,  # stopping_time (scalar int array)
    Array,  # primal_residuals
    Array,  # dual_residuals
    Array,  # epris
    Array,  # eduals
]
