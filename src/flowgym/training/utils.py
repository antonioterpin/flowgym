"""Utilities for training."""

from collections import deque
from dataclasses import dataclass, fields
import jax.numpy as jnp
import jax
import numpy as np

from flowgym.types import PRNGKey

@jax.tree_util.register_pytree_node_class
@dataclass
class Experience:
    """Dataclass to hold a batch of experiences.
    
    NOTE: When adding experiences to the buffer, each attribute should have shape
    corresponding to a single experience. When sampling a batch, each attribute will
    have an additional leading batch dimension, as shown in the example below.
    
    Attributes:
        next_flow: The flow field after taking the action.
        action: The action taken.
        reward: The reward received.
        images: The current images.
        old_images: The previous images.
        old_flow: The flow field before taking the action.
        
    Example:
    >>>     exp = Experience(
            next_flow=jnp.array([[0.1, 0.2], [0.3, 0.4]]),  # Shape (H, W, 2)
            action=jnp.array([1]),                            # Shape (1,)
            reward=jnp.array([0.5]),                          # Shape (1,)
            images=jnp.array([[0.5, 0.6], [0.7, 0.8]]),        # Shape (H, W, 2)
            old_images=jnp.array([[0.9, 1.0], [1.1, 1.2]]),    # Shape (H, W, 2)
            old_flow=jnp.array([[0.0, 0.1], [0.2, 0.3]]),        # Shape (H, W, 2)
            )
    >>>     buffer.push(exp)
    >>>     batch = buffer.sample(batch_size=32)
    >>>     print(batch.next_flow.shape)  
            (32, H, W, 2)
    """

    next_flow: jnp.ndarray            # (H, W, 2)
    action: jnp.ndarray               # (1, )
    reward: jnp.ndarray               # (1, )
    images: jnp.ndarray               # (H, W, 2)
    old_images: jnp.ndarray           # (H, W, 2)
    old_flow: jnp.ndarray             # (H, W, 2)

    def tree_flatten(self) -> tuple[list[jnp.ndarray], None]:
        """Flattens the Experience dataclass into its fields for JAX PyTree compatibility.
        
        Returns:
            A tuple containing a list of the dataclass fields and None (no auxiliary data).
        """
        children = [getattr(self, f.name) for f in fields(self)]
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children: list[jnp.ndarray]) -> "Experience":
        """Reconstructs the Experience dataclass from its flattened fields.
        
        Args:
            aux: Auxiliary data (not used here).
            children: List of fields to reconstruct the dataclass.
            
        Returns:
            An instance of Experience reconstructed from the fields.
        """
        return cls(*children)
    

class ReplayBuffer:
    """A simple replay buffer for storing and sampling experiences."""

    def __init__(
        self,
        capacity: int,
        key: PRNGKey = jax.device_put(jax.random.PRNGKey(0), jax.devices("cpu")[0])
        ) -> None:
        """Initializes the ReplayBuffer.

        Args:
            capacity: Maximum number of experiences to store.
            key: Random number generator key.
        """
        self.buffer = deque(maxlen=capacity)
        self.key = key
        self.cpu_device = jax.devices("cpu")[0]

    def push(self, experience: Experience) -> None:
        """Adds a new experience to the buffer.
        
        Args:
            experience: The experience to add.
        """
        def to_cpu(x):
            """Moves a JAX array to CPU device."""
            if isinstance(x, jnp.ndarray):
                # We move the array to the specified CPU device
                return jax.device_put(x, self.cpu_device)
            return x

        # Apply the to_cpu function to all fields in the Experience dataclass
        cpu_experience = jax.tree_util.tree_map(to_cpu, experience)
        self.buffer.append(cpu_experience)

    def sample(self, batch_size: int) -> Experience:
        """Randomly samples a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample.
            
        Returns:
            stacked Experience object.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Requested batch size {batch_size} exceeds buffer size {len(self.buffer)}"
            )

        self.key, subkey = jax.random.split(self.key)
        with jax.default_device(self.cpu_device):
                    indices = jax.random.choice(
                        subkey,
                        a=len(self.buffer),
                        shape=(batch_size,),
                        replace=False
                    )
                    
        batch = [self.buffer[i.item()] for i in indices]
        
        return self._process_batch(batch)

    def sample_at(self, indices: int | list[int] | np.ndarray) -> Experience:
        """Deterministically samples experiences by given indices.

        Args:
            indices: Indices of experiences to sample.

        Returns:
            stacked Experience object.
        """
        if isinstance(indices, int):
            indices = [indices]

        if max(indices) >= len(self.buffer) or min(indices) < 0:
            raise IndexError(
                f"Indices {indices} out of range for buffer of length {len(self.buffer)}"
            )

        batch = [self.buffer[i] for i in indices]
        return self._process_batch(batch)

    def clear(self):
        """Clears the entire buffer."""
        self.buffer.clear()

    def _process_batch(self, batch: list[Experience]) -> Experience:
        """Helper function to unpack and stack a batch of experiences.
        
        Args:
            batch: List of Experience objects.
        
        Returns:
            stacked Experience object.
        """
        stacked_experience = jax.tree_util.tree_map(
            lambda *args: jnp.stack(args, axis=0), *batch
        )
        return stacked_experience

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)