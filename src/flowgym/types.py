"""Type aliases used in the flow estimator."""

from typing import TypeAlias
import jax.numpy as jnp

PRNGKey: TypeAlias = jnp.ndarray
ExperimentParams: TypeAlias = dict[str, jnp.ndarray | float | int | bool | str]
