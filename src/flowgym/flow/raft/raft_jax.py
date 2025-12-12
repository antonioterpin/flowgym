"""RaftJaxEstimator class."""

from typing import Any, Literal
import jax.numpy as jnp
import jax
import optax

from flowgym.flow.base import FlowFieldEstimator
from flowgym.common.base import NNEstimatorTrainableState
from goggles.history.types import History
from flowgym.nn.raft_model import RaftEstimatorModel
from flowgym.flow.raft.process import unfold_patches, window_2D, fold_patches
from flowgym.types import PRNGKey


NORMKINDS: tuple[str, ...] = ("batch", "group", "instance", "none")


class RaftJaxEstimator(FlowFieldEstimator):
    """RAFT32 flow field estimator."""

    def __init__(
        self,
        patch_size: int = 32,
        patch_stride: int = 8,
        hidden_dim: int = 128,
        context_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        iters: int = 12,
        patches_groups: int = 1,
        norm_fn: Literal["batch", "group", "instance", "none"] = "instance",
        dropout: float = 0.0,
        train: bool = True,
        gamma: float = 0.8,
        use_temporal_propagation: bool = False,
        **kwargs: Any,
    ):
        """Initialize the FlowFormer estimator.

        Args:
            patch_size: Size of the patches to process.
            patch_stride: Stride between patches.
            hidden_dim: Dimension of the hidden state in the update block.
            context_dim: Dimension of the context features.
            corr_levels: Number of levels in the correlation pyramid.
            corr_radius: Radius for correlation lookup.
            iters: Number of iterations for flow refinement.
            patches_groups: Number of groups to divide patches into.
            norm_fn: Normalization function to use ('batch', 'group', 'instance', 'none').
            dropout: Dropout rate.
            train: Whether the model is in training mode.
            gamma: Discount factor for training loss.
            use_temporal_propagation: Whether to use temporal propagation of flow.
            kwargs: Additional keyword arguments for the base class.
        """
        if not isinstance(patch_size, int):
            raise TypeError(f"patch_size must be an int, got {patch_size}.")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")
        self.patch_size = patch_size

        if not isinstance(patch_stride, int):
            raise TypeError(f"patch_stride must be an int, got {patch_stride}.")
        if patch_stride <= 0:
            raise ValueError(f"patch_stride must be positive, got {patch_stride}.")
        self.patch_stride = patch_stride

        if not isinstance(hidden_dim, int):
            raise TypeError(f"hidden_dim must be an int, got {hidden_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        self.hidden_dim = hidden_dim

        if not isinstance(context_dim, int):
            raise TypeError(f"context_dim must be an int, got {context_dim}.")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}.")
        self.context_dim = context_dim

        if not isinstance(corr_levels, int):
            raise TypeError(f"corr_levels must be an int, got {corr_levels}.")
        if corr_levels <= 0:
            raise ValueError(f"corr_levels must be positive, got {corr_levels}.")
        self.corr_levels = corr_levels

        if not isinstance(corr_radius, int):
            raise TypeError(f"corr_radius must be an int, got {corr_radius}.")
        if corr_radius <= 0:
            raise ValueError(f"corr_radius must be positive, got {corr_radius}.")
        self.corr_radius = corr_radius

        if not isinstance(iters, int):
            raise TypeError(f"iters must be an int, got {iters}.")
        if iters <= 0:
            raise ValueError(f"iters must be positive, got {iters}.")
        self.iters = iters

        if not isinstance(patches_groups, int):
            raise TypeError(f"patches_groups must be an int, got {patches_groups}.")
        if patches_groups <= 0:
            raise ValueError(f"patches_groups must be positive, got {patches_groups}.")
        self.patches_groups = patches_groups

        if norm_fn not in NORMKINDS:
            raise ValueError(f"norm_fn must be one of {NORMKINDS}, got {norm_fn}.")
        self.norm_fn = norm_fn

        if not isinstance(dropout, (float, int)):
            raise TypeError(f"dropout must be a number, got {dropout}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")
        self.dropout = float(dropout)

        if not isinstance(gamma, (float, int)):
            raise TypeError(f"gamma must be a number, got {gamma}.")
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0.0, 1.0], got {gamma}.")
        self.gamma = float(gamma)

        if not isinstance(train, bool):
            raise TypeError(f"train must be a bool, got {train}.")
        self.train = train

        if not isinstance(use_temporal_propagation, bool):
            raise TypeError(
                "use_temporal_propagation must be a bool,"
                f" got {use_temporal_propagation}."
            )
        self.use_temporal_propagation = use_temporal_propagation

        self.model = RaftEstimatorModel(
            hidden_dim=self.hidden_dim,
            context_dim=self.context_dim,
            corr_levels=self.corr_levels,
            corr_radius=self.corr_radius,
            iters=self.iters,
            norm_fn=self.norm_fn,
            dropout=self.dropout,
            train=self.train,
        )

        super().__init__(**kwargs)

    def create_trainable_state(
        self,
        dummy_input: jnp.ndarray,
        key: PRNGKey,
    ) -> NNEstimatorTrainableState:
        """Create the initial trainable state of the flow field estimator.

        This method initializes the model parameters and the optimizer state.

        Args:
            dummy_input (jnp.ndarray): Batched dummy input to initialize the state.
            key (jax.random.PRNGKey): JAX random key for parameter initialization.

        Returns:
            NNEstimatorTrainableState:
                The initial trainable state of the estimator.
        """
        # Validate the dummy input shape
        if dummy_input.ndim != 3:
            raise ValueError(
                f"Dummy input must have 3 dimensions (B, H, W), got {dummy_input.ndim}."
            )

        # Input is two images and the initial flow
        dummy_input = jnp.tile(
            dummy_input[..., None], (1, 1, 1, 2)  # prev, curr
        )  # Shape: (B, H, W, 2)

        # Discard extra images
        dummy_input = dummy_input[:1]  # Shape: (1, H, W, 2)

        dummy_input_patches = unfold_patches(
            dummy_input, self.patch_size, self.patch_stride
        ).reshape(
            -1, self.patch_size, self.patch_size, 2
        )  # Shape: (num_patches*B, patch_size, patch_size, 2)

        # Note: to create trainable state in flax, batch size is specific to the training,
        # the network is evaluated in parallel on a single image pair (+ optional flow).
        params = self.model.init(key, dummy_input_patches, dummy_input_patches)[
            "params"
        ]
        tx = optax.adam(learning_rate=1e-4)

        return NNEstimatorTrainableState.create(params, tx)

    def create_train_step(self):
        """Create the training step function for the flow field estimator."""

        def train_step(
            state: History,
            trainable_state: NNEstimatorTrainableState,
            _: NNEstimatorTrainableState,
            __: jnp.ndarray,
            obs: tuple[jnp.ndarray, jnp.ndarray],
            ___: jnp.ndarray,
            ____: jnp.ndarray,
            gt_flow: jnp.ndarray,
            max_norm: float = 1.0,
        ):
            """Train step function for the flow field estimator.

            Args:
                state (History): Current state of the estimator.
                trainable_state (NNEstimatorTrainableState):
                    Current trainable state of the model.
                _ (NNEstimatorTrainableState): Unused parameter.
                __ (jnp.ndarray): Unused parameter.
                obs (Tuple[jnp.ndarray, jnp.ndarray]):
                    Tuple containing previous and current frames.
                ___ (jnp.ndarray): Unused parameter.
                ____ (jnp.ndarray): Unused parameter.
                gt_flow (jnp.ndarray): Ground truth flow for the current frame pair.
                max_norm (float): Maximum norm for gradient clipping.


            Returns:
                loss (jnp.ndarray): The computed loss.
                trainable_state (NNEstimatorTrainableState):
                    The updated state of the model.
                grads (optax.Updates): The computed gradients.
            """
            # Unpack the observation
            prev, curr = obs
            B, H, W = prev.shape

            # Stack the images along the last dimension
            images = jnp.stack([prev, curr], axis=-1)  # (B, H, W, 2)

            # Check if the state has a history of estimates
            flows = (
                state["estimates"][:, -1, ...]
                if self.use_temporal_propagation
                else jnp.zeros((B, H, W, 2), dtype=jnp.float32)
            )

            # Compute exponentially decayed weights
            weights = jnp.array(
                [self.gamma ** (self.iters - 1 - i) for i in range(self.iters)]
            )

            # compute how many valid strides fit
            Kx = int(
                (H / self.patch_stride) - (self.patch_size / self.patch_stride - 1)
            )
            Ky = int(
                (W / self.patch_stride) - (self.patch_size / self.patch_stride - 1)
            )

            # Randomly select patches
            kx, ky = jax.random.split(state["keys"][0][0], 2)
            ix = jax.random.randint(kx, (B,), 0, Kx)
            iy = jax.random.randint(ky, (B,), 0, Ky)
            sx = ix * self.patch_stride  # start x (row)
            sy = iy * self.patch_stride  # start y (col)

            def slice_one(img, sx_i, sy_i):
                return jax.lax.dynamic_slice(
                    img, (sx_i, sy_i, 0), (self.patch_size, self.patch_size, 2)
                )

            # Slice patches
            patches_images = jax.vmap(slice_one)(images, sx, sy)
            patches_flows = jax.vmap(slice_one)(flows, sx, sy)
            patches_gt = jax.vmap(slice_one)(gt_flow, sx, sy)

            def loss_fn(params):
                """Compute the loss for the current parameters."""
                tmp_ts = trainable_state.replace(params=params)

                # Process the image patches to estimate the flow
                flow_predictions = self.model.apply(
                    {"params": tmp_ts.params}, patches_images, patches_flows
                )  # (I, B, 32, 32, 2), I = num iters

                # Compute per-iteration L1 loss (mean over spatial dimensions)
                per_iter_losses = jnp.mean(
                    jnp.abs(flow_predictions - patches_gt), axis=(1, 2, 3, 4)
                )  # (I,)

                # Weighted sum
                loss = jnp.sum(weights * per_iter_losses)

                return loss

            # Compute gradients and update parameters
            loss, grads = jax.value_and_grad(loss_fn)(trainable_state.params)
            grad_norm = jnp.sqrt(sum(jnp.vdot(g, g) for g in jax.tree.leaves(grads)))

            clip_coef = jnp.minimum(1.0, max_norm / (grad_norm + 1e-6))
            grads = jax.tree.map(lambda g: g * clip_coef, grads)

            trainable_state = trainable_state.apply_gradients(grads)

            return loss, trainable_state, grads

        return train_step

    def _estimate(
        self,
        image: jnp.ndarray,
        state: History,
        trainable_state: NNEstimatorTrainableState,
        _: None,
    ) -> jnp.ndarray:
        """Compute the flow field between the two most recent frames.

        Args:
            image (jnp.ndarray): Current batch of frames, shape (B, H, W).
            state (History): Contains history_images of shape (B, 1, H, W).
            trainable_state (NNEstimatorTrainableState): The current state of the model.
            _ (None): Unused parameter.

        Returns:
            jnp.ndarray: Flow field of shape (B, H, W, 2) as float32.
            None: placeholder for additional output.
        """
        B, H, W = image.shape

        # Get the most recent images from the state
        prev = state["images"][:, -1, ...]
        curr = image

        # Create the 2d window
        window_spline_2d = window_2D(self.patch_size, power=2)

        # Stack the images along the last dimension
        images = jnp.stack([prev, curr], axis=-1)  # (B, H, W, 2)

        # Check if the state has a history of estimates
        flows = (
            state["estimates"][:, -1, ...]
            if self.use_temporal_propagation
            else jnp.zeros((B, H, W, 2), dtype=jnp.float32)
        )

        # Unfold the images and flows into patches
        patches_images = unfold_patches(images, self.patch_size, self.patch_stride)
        patches_flows = unfold_patches(flows, self.patch_size, self.patch_stride)
        _, Ny, Nx, _, _, _ = patches_images.shape

        # Reshape to (num_patches*B, patch_size, patch_size, channels)
        patches_images = patches_images.reshape(-1, self.patch_size, self.patch_size, 2)
        patches_flows = patches_flows.reshape(-1, self.patch_size, self.patch_size, 2)
        total_patches = patches_images.shape[0]

        # Pad patches to make them evenly divisible into groups
        group_size = total_patches // self.patches_groups + (
            total_patches % self.patches_groups > 0
        )
        padded_len = group_size * self.patches_groups
        pad_needed = padded_len - total_patches

        patches_images = jnp.pad(
            patches_images,
            pad_width=((0, pad_needed), (0, 0), (0, 0), (0, 0)),
            mode="constant",
        )
        patches_flows = jnp.pad(
            patches_flows,
            pad_width=((0, pad_needed), (0, 0), (0, 0), (0, 0)),
            mode="constant",
        )

        # Split evenly into groups
        patches_images_groups = jnp.split(patches_images, self.patches_groups, axis=0)
        patches_flows_groups = jnp.split(patches_flows, self.patches_groups, axis=0)

        def process_group(carry, group_data):
            img_group, flow_group = group_data
            preds = self.model.apply(
                {"params": trainable_state.params}, img_group, flow_group
            )
            return carry, preds[-1, ...]

        # Sequential scan over groups
        _, flows_groups = jax.lax.scan(
            process_group,
            None,
            (jnp.stack(patches_images_groups), jnp.stack(patches_flows_groups)),
        )

        # Concatenate groups along patch dimension
        flows_patches = flows_groups.reshape(-1, self.patch_size, self.patch_size, 2)
        flows_patches = flows_patches[:total_patches]

        # Reshape back to (B, Ny, Nx, patch_size, patch_size, 2)
        flows_patches = flows_patches.reshape(
            B, Ny, Nx, self.patch_size, self.patch_size, 2
        )

        # Apply the window to the patches
        flows_patches = flows_patches * window_spline_2d[None, None, None, ..., None]

        # Vmap over the batch dimension
        fold_patches_batched = jax.vmap(
            fold_patches,
            in_axes=(0, None, None, None),
        )

        # Fold the patches back to the full image
        flows = fold_patches_batched(flows_patches, H, W, self.patch_stride)

        # Calculate the overlap weights
        patches_weights = (
            jnp.ones_like(flows_patches) * window_spline_2d[None, None, None, ..., None]
        )
        weights = fold_patches_batched(patches_weights, H, W, self.patch_stride)

        # Normalize the flow by the weights to account for overlapping patches
        flows = flows / weights

        return flows, {}, {}
