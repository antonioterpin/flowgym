"""Replay buffer implementation."""

from collections import deque
from collections.abc import Iterator
from typing import Any, Generic, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import prefetch_to_device

from flowgym.types import (
    PRNGKey,
)

ExperienceType = TypeVar("ExperienceType")


class ReplayBuffer(Generic[ExperienceType]):
    """A generic replay buffer for storing and sampling experiences.

    Uses `Generic[ExperienceType]` for type safety across different experience
    structures (e.g., `Experience` or `SupervisedExperience`). Type checkers
    can verify that pushed and sampled items match the expected schema.

    The implementation uses `jax.tree_util.tree_map` to handle operations
    agnostically of the specific field structure, ensuring stored items are
    offloaded to the CPU and correctly stacked when sampled.
    """

    def __init__(
        self,
        capacity: int,
        key: PRNGKey | None = None,
    ) -> None:
        """Initializes the ReplayBuffer.

        Args:
            capacity: Maximum number of experiences to store.
            key: Random number generator key.
        """
        if key is None:
            key = jax.device_put(jax.random.PRNGKey(0), jax.devices("cpu")[0])
        self.buffer = deque(maxlen=capacity)

        self.key = cast(jax.Array, key)
        self.cpu_device = jax.devices("cpu")[0]

    def push(self, experience: ExperienceType) -> None:
        """Adds a new experience to the buffer.

        Args:
            experience: The experience to add.
        """

        def to_cpu(x: Any) -> Any:
            """Move JAX array to CPU device.

            Args:
                x: Object to move (typically a JAX array).

            Returns:
                Object moved to CPU device, or unchanged if not an array.
            """
            if isinstance(x, jnp.ndarray):
                return jax.device_put(x, self.cpu_device)
            return x

        # Apply the to_cpu function to all fields in the Experience dataclass
        cpu_experience = jax.tree_util.tree_map(to_cpu, experience)
        self.buffer.append(cpu_experience)

    def sample(self, batch_size: int) -> ExperienceType:
        """Randomly samples a batch of experiences.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            Stacked experience object.

        Raises:
            ValueError: If batch_size exceeds buffer size.
        """
        if batch_size > len(self.buffer):
            buf_size = len(self.buffer)
            raise ValueError(
                f"Requested batch size {batch_size} exceeds buffer size "
                f"{buf_size}"
            )

        self.key, subkey = jax.random.split(self.key)
        with jax.default_device(self.cpu_device):
            indices = jax.random.choice(
                subkey, a=len(self.buffer), shape=(batch_size,), replace=False
            )

        batch = [self.buffer[i.item()] for i in indices]

        return self._process_batch(batch)

    def sample_at(
        self, indices: int | list[int] | np.ndarray
    ) -> ExperienceType:
        """Deterministically sample experiences by given indices.

        Args:
            indices: Indices of experiences to sample.

        Returns:
            Stacked experience object.

        Raises:
            IndexError: If any index is out of range.
        """
        if isinstance(indices, int):
            indices = [indices]

        if np.max(indices) >= len(self.buffer) or np.min(indices) < 0:
            buf_len = len(self.buffer)
            raise IndexError(
                f"Indices {indices} out of range for buffer of length {buf_len}"
            )

        batch = [self.buffer[i] for i in indices]
        return self._process_batch(batch)

    def sample_iter(
        self,
        batch_size: int,
        num_batches: int,
        prefetch: int = 0,
        device: Any = None,
    ) -> Iterator[ExperienceType]:
        """Yield batches of sampled experiences, optionally with GPU prefetch.

        When prefetch > 0, uses Flax's prefetch_to_device for CPU/GPU overlap.

        Args:
            batch_size: Number of experiences per batch.
            num_batches: Number of batches to yield.
            prefetch: Prefetch buffer size. If > 0, enables GPU prefetch.
            device: Target device for prefetch. Defaults to first available.

        Yields:
            ExperienceType: Batches of stacked experiences.

        Raises:
            ValueError: If batch_size exceeds buffer size.
        """
        if batch_size > len(self.buffer):
            buf_size = len(self.buffer)
            raise ValueError(
                f"Requested batch size {batch_size} exceeds buffer size "
                f"{buf_size}"
            )

        def _base_iterator() -> Iterator[ExperienceType]:
            """Sample batches from the buffer.

            Yields:
                ExperienceType: Stacked experience batches.
            """
            key = self.key
            for _ in range(num_batches):
                key, subkey = jax.random.split(key)
                with jax.default_device(self.cpu_device):
                    indices = jax.random.choice(
                        subkey,
                        a=len(self.buffer),
                        shape=(batch_size,),
                        replace=False,
                    )
                batch = [self.buffer[i.item()] for i in indices]
                yield self._process_batch(batch)
            self.key = key

        if prefetch > 0:
            target_device = device if device is not None else jax.devices()[0]

            def _prefetch_iterator() -> Iterator[Any]:
                """Wrap iterator with leading dim for prefetch compatibility.

                Yields:
                    Any: Batches with expanded leading dimension.
                """
                for batch in _base_iterator():
                    yield jax.tree_util.tree_map(
                        lambda x: jnp.expand_dims(x, axis=0), batch
                    )

            prefetched = prefetch_to_device(
                _prefetch_iterator(), size=prefetch, devices=[target_device]
            )
            for batch in prefetched:
                # Squeeze the leading dimension after prefetch
                yield jax.tree_util.tree_map(
                    lambda x: jnp.squeeze(x, axis=0), batch
                )
        else:
            yield from _base_iterator()

    def clear(self) -> None:
        """Clears the entire buffer."""
        self.buffer.clear()

    def _process_batch(self, batch: list[ExperienceType]) -> ExperienceType:
        """Helper function to unpack and stack a batch of experiences.

        Args:
            batch: List of experience objects.

        Returns:
            stacked experience object.
        """
        stacked_experience = jax.tree_util.tree_map(
            lambda *args: jnp.stack(args, axis=0), *batch
        )
        return stacked_experience

    def __len__(self) -> int:
        """Returns the current size of the buffer.

        Returns:
            int: Current size of the buffer.
        """
        return len(self.buffer)
