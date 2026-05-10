"""RaftJaxEstimator class."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from goggles.history.types import History

if TYPE_CHECKING:
    from synthpix import SynthpixBatch

from flowgym.common.base import NNEstimatorTrainableState
from flowgym.flow.base import FlowFieldEstimator
from flowgym.flow.raft.process import fold_patches, unfold_patches, window_2d
from flowgym.nn.raft_model import RaftEstimatorModel
from flowgym.types import PRNGKey, SupervisedExperience, SupervisedTrainStep

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
            norm_fn: Normalization function to use ('batch', 'group',
                'instance', 'none').
            dropout: Dropout rate.
            train: Whether the model is in training mode.
            gamma: Discount factor for training loss.
            use_temporal_propagation: Whether to use temporal propagation
                of flow.
            **kwargs: Additional keyword arguments for the base class.

        Raises:
            TypeError: If parameter types are incorrect.
            ValueError: If parameter values are invalid.
        """
        if not isinstance(patch_size, int):
            raise TypeError(f"patch_size must be an int, got {patch_size}.")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")
        self.patch_size = patch_size

        if not isinstance(patch_stride, int):
            raise TypeError(f"patch_stride must be an int, got {patch_stride}.")
        if patch_stride <= 0:
            raise ValueError(
                f"patch_stride must be positive, got {patch_stride}."
            )
        self.patch_stride = patch_stride

        if not isinstance(hidden_dim, int):
            raise TypeError(f"hidden_dim must be an int, got {hidden_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        self.hidden_dim = hidden_dim

        if not isinstance(context_dim, int):
            raise TypeError(f"context_dim must be an int, got {context_dim}.")
        if context_dim <= 0:
            raise ValueError(
                f"context_dim must be positive, got {context_dim}."
            )
        self.context_dim = context_dim

        if not isinstance(corr_levels, int):
            raise TypeError(f"corr_levels must be an int, got {corr_levels}.")
        if corr_levels <= 0:
            raise ValueError(
                f"corr_levels must be positive, got {corr_levels}."
            )
        self.corr_levels = corr_levels

        if not isinstance(corr_radius, int):
            raise TypeError(f"corr_radius must be an int, got {corr_radius}.")
        if corr_radius <= 0:
            raise ValueError(
                f"corr_radius must be positive, got {corr_radius}."
            )
        self.corr_radius = corr_radius

        if not isinstance(iters, int):
            raise TypeError(f"iters must be an int, got {iters}.")
        if iters <= 0:
            raise ValueError(f"iters must be positive, got {iters}.")
        self.iters = iters

        if not isinstance(patches_groups, int):
            raise TypeError(
                f"patches_groups must be an int, got {patches_groups}."
            )
        if patches_groups <= 0:
            raise ValueError(
                f"patches_groups must be positive, got {patches_groups}."
            )
        self.patches_groups = patches_groups

        if norm_fn not in NORMKINDS:
            raise ValueError(
                f"norm_fn must be one of {NORMKINDS}, got {norm_fn}."
            )
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
            dummy_input: dummy input to initialize the state, shape (B, H, W)
            key: JAX random key for parameter initialization.

        Returns:
            EstimatorTrainableState:
                The initial trainable state of the estimator.

        Raises:
            ValueError: If dummy input shape is invalid.
        """
        # Validate the dummy input shape
        if dummy_input.ndim != 3:
            raise ValueError(
                f"Dummy input must have 3 dimensions (B, H, W), "
                f"got {dummy_input.ndim}."
            )

        # Input is two images and the initial flow
        dummy_input = jnp.tile(
            dummy_input[..., None],
            (1, 1, 1, 2),  # prev, curr
        )  # Shape: (B, H, W, 2)

        # Discard extra images
        dummy_input = dummy_input[:1]  # Shape: (1, H, W, 2)

        dummy_input_patches = unfold_patches(
            dummy_input, self.patch_size, self.patch_stride
        ).reshape(
            -1, self.patch_size, self.patch_size, 2
        )  # Shape: (num_patches*B, patch_size, patch_size, 2)

        # Discard extra patches
        dummy_input_patches = dummy_input_patches[
            :1
        ]  # Shape: (1, patch_size, patch_size, 2)

        # Note: to create trainable state in flax, batch size is specific
        # to the training, the network is evaluated in parallel on a single
        # image pair (+ optional flow).
        params = self.model.init(key, dummy_input_patches, dummy_input_patches)[
            "params"
        ]

        if self.optimizer_config is None:
            raise ValueError("Optimizer configuration is required.")

        return NNEstimatorTrainableState.from_config(
            apply_fn=self.model.apply,
            params=params,
            optimizer_config=self.optimizer_config,
        )

    def create_train_step(self) -> SupervisedTrainStep:
        """Create the training step function for the flow field estimator.

        Returns:
            SupervisedTrainStep: The training step function.
        """

        def train_step(
            trainable_state: NNEstimatorTrainableState,
            experience: SupervisedExperience,
        ) -> tuple[jnp.ndarray, NNEstimatorTrainableState, dict]:
            """Train step function for the flow field estimator.

            Args:
                trainable_state: Current trainable state of the model.
                experience: Supervised experience containing state, obs,
                    ground_truth.

            Returns:
                The computed loss.
                The updated state of the model.
                A metrics dictionary
            """
            # Prepare metrics
            metrics = {}

            # Unpack the experience
            state = experience.state
            images1, images2 = experience.obs
            ground_truth = experience.ground_truth

            # Unpack the observation
            B, H, W = images1.shape

            # Stack the images along the last dimension
            images = jnp.stack([images1, images2], axis=-1)  # (B, H, W, 2)

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
                (H / self.patch_stride)
                - (self.patch_size / self.patch_stride - 1)
            )
            Ky = int(
                (W / self.patch_stride)
                - (self.patch_size / self.patch_stride - 1)
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
            patches_gt = jax.vmap(slice_one)(ground_truth, sx, sy)

            def loss_fn(params: Any) -> tuple[jnp.ndarray, dict]:
                """Compute the loss for the current parameters.

                Args:
                    params: The parameters of the model.

                Returns:
                    The computed loss and metrics.
                """
                # Process the image patches to estimate the flow
                flow_predictions = self.model.apply(
                    {"params": params}, patches_images, patches_flows
                )  # (I, B, 32, 32, 2), I = num iters

                per_iter_losses = jnp.mean(
                    jnp.abs(cast(jnp.ndarray, flow_predictions) - patches_gt),
                    axis=(1, 2, 3, 4),
                )  # (I,)

                # Weighted sum
                loss = jnp.sum(weights * per_iter_losses)

                return loss, {}

            # Compute gradients and update parameters
            (loss, _aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                trainable_state.params
            )

            grad_norm = optax.global_norm(grads)
            metrics["grad_norm"] = grad_norm

            def clip_grads_by_global_norm(grads, clip_norm, grad_norm):
                scale = jnp.minimum(1.0, clip_norm / (grad_norm + 1e-8))
                return jax.tree_util.tree_map(lambda g: g * scale, grads), scale

            clipped_grads, clip_scale = clip_grads_by_global_norm(
                grads, 1.0, grad_norm
            )
            metrics["clip_scale_implied"] = clip_scale  # 1.0 => no clipping
            metrics["clipping_active_implied"] = clip_scale < 1.0

            # Update application
            updates, new_opt_state = trainable_state.tx.update(
                clipped_grads, trainable_state.opt_state, trainable_state.params
            )
            new_params = optax.apply_updates(trainable_state.params, updates)

            # Update metrics
            update_norm = optax.global_norm(updates)
            param_norm = optax.global_norm(trainable_state.params)
            rel_update = update_norm / (param_norm + 1e-8)
            metrics["updates_norm"] = update_norm
            metrics["params_norm"] = param_norm
            metrics["rel_update"] = rel_update
            metrics["loss"] = loss

            def tree_dot(a, b):
                return jax.tree_util.tree_reduce(
                    lambda x, y: x + y,
                    jax.tree_util.tree_map(lambda x, y: jnp.sum(x * y), a, b),
                    initializer=0.0,
                )

            dot = tree_dot(grads, updates)
            cos = dot / ((grad_norm * update_norm) + 1e-8)
            metrics["cosine_similarity"] = cos

            trainable_state = trainable_state.replace(
                params=new_params,
                opt_state=new_opt_state,
                step=trainable_state.step + 1,
            )

            return loss, trainable_state, metrics

        return train_step

    def _estimate(
        self,
        image: jnp.ndarray,
        state: History,
        trainable_state: NNEstimatorTrainableState,
        extras: dict,
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the flow field between the two most recent frames.

        Args:
            image: Current batch of frames, shape (B, H, W).
            state: Contains history_images of shape (B, 1, H, W).
            trainable_state: The current state of the model.
            extras: Additional data from history fields, may contain
                cache_payload.

        Returns:
            Flow field of shape (B, H, W, 2) as float32.
            placeholder for additional output.
            placeholder for additional output.
        """
        # Check if we have cached metrics and can skip computation
        cache_payload = extras.get("cache_payload")
        if (
            cache_payload is not None
            and getattr(cache_payload, "has_precomputed_errors", False)
            and getattr(cache_payload, "epe", None) is not None
        ):
            B, H, W = image.shape
            if self.use_temporal_propagation:
                flows = state["estimates"][:, -1, ...]
            else:
                flows = jnp.zeros((B, H, W, 2), dtype=jnp.float32)

            metrics: dict[str, jnp.ndarray] = {
                "errors": jnp.asarray(cache_payload.epe)
            }
            relative_epe = getattr(cache_payload, "relative_epe", None)
            if relative_epe is not None:
                metrics["relative_errors"] = jnp.asarray(relative_epe)
            return flows, {}, metrics

        B, H, W = image.shape

        # Get the most recent images from the state
        prev = state["images"][:, -1, ...]
        curr = image

        # Create the 2d window
        window_spline_2d = window_2d(self.patch_size, power=2)

        # Stack the images along the last dimension
        images = jnp.stack([prev, curr], axis=-1)  # (B, H, W, 2)

        # Check if the state has a history of estimates
        flows = (
            state["estimates"][:, -1, ...]
            if self.use_temporal_propagation
            else jnp.zeros((B, H, W, 2), dtype=jnp.float32)
        )

        # Unfold the images and flows into patches
        patches_images = unfold_patches(
            images, self.patch_size, self.patch_stride
        )
        patches_flows = unfold_patches(
            flows, self.patch_size, self.patch_stride
        )
        _, Ny, Nx, _, _, _ = patches_images.shape

        # Reshape to (num_patches*B, patch_size, patch_size, channels)
        patches_images = patches_images.reshape(
            -1, self.patch_size, self.patch_size, 2
        )
        patches_flows = patches_flows.reshape(
            -1, self.patch_size, self.patch_size, 2
        )
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
        patches_images_groups = jnp.split(
            patches_images, self.patches_groups, axis=0
        )
        patches_flows_groups = jnp.split(
            patches_flows, self.patches_groups, axis=0
        )

        def process_group(carry, group_data):
            img_group, flow_group = group_data
            preds = self.model.apply(
                {"params": trainable_state.params}, img_group, flow_group
            )
            return carry, cast(jnp.ndarray, preds)[-1, ...]

        # Sequential scan over groups
        _, flows_groups = jax.lax.scan(
            process_group,
            None,
            (jnp.stack(patches_images_groups), jnp.stack(patches_flows_groups)),
        )

        # Concatenate groups along patch dimension
        flows_patches = flows_groups.reshape(
            -1, self.patch_size, self.patch_size, 2
        )
        flows_patches = flows_patches[:total_patches]

        # Reshape back to (B, Ny, Nx, patch_size, patch_size, 2)
        flows_patches = flows_patches.reshape(
            B, Ny, Nx, self.patch_size, self.patch_size, 2
        )

        # Apply the window to the patches
        flows_patches = (
            flows_patches * window_spline_2d[None, None, None, ..., None]
        )

        # Vmap over the batch dimension
        fold_patches_batched = jax.vmap(
            fold_patches,
            in_axes=(0, None, None, None),
        )

        # Fold the patches back to the full image
        flows = fold_patches_batched(flows_patches, H, W, self.patch_stride)

        # Calculate the overlap weights
        patches_weights = (
            jnp.ones_like(flows_patches)
            * window_spline_2d[None, None, None, ..., None]
        )
        weights = fold_patches_batched(patches_weights, H, W, self.patch_stride)

        # Normalize the flow by the weights to account for overlapping patches
        flows = flows / weights

        return flows, {}, {}

    def enrich(
        self,
        batch: SynthpixBatch,
        miss_idxs: jnp.ndarray,
        **kwargs: Any,
    ) -> dict[str, jnp.ndarray] | None:
        """Compute the cache payload for missing keys (EPE).

        Args:
            batch: The batch of data (SynthpixBatch).
            miss_idxs: Indices of samples in the batch that are missing
                from cache.
            **kwargs: Additional arguments, must include 'trainable_state'.

        Returns:
            Dictionary with "epe" values for the missing indices.
        """
        trainable_state = kwargs.get("trainable_state")
        if trainable_state is None:
            # Cannot compute without weights
            return None

        if batch.flow_fields is None:
            return None

        # Slice batch for missing indices (eager, host-side)
        images1 = batch.images1[miss_idxs]
        images2 = batch.images2[miss_idxs]
        gt_flow = batch.flow_fields[miss_idxs]

        # Check if empty (should not happen if check np.all(hit) before)
        if images1.shape[0] == 0:
            return {}

        key = jax.random.PRNGKey(0)

        # JIT-compiled computation
        epe_mean, rel_mean = self._compute_cache_payload(
            images1, images2, gt_flow, trainable_state, key
        )

        return {"epe": epe_mean, "relative_epe": rel_mean}

    @partial(jax.jit, static_argnums=0)
    def _compute_cache_payload(
        self,
        images1: jnp.ndarray,
        images2: jnp.ndarray,
        gt_flow: jnp.ndarray,
        t_state: NNEstimatorTrainableState,
        rng: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Pure JAX computation of EPE metrics (JIT-compiled).

        Args:
            images1: First frame images (B, H, W).
            images2: Second frame images (B, H, W).
            gt_flow: Ground truth flow fields (B, H, W, 2).
            t_state: Trainable state.
            rng: Random key for state creation.

        Returns:
            Tuple of (epe_mean, relative_epe_mean), both shape (B,).
        """
        B, H, W = images1.shape
        init_flow = jnp.zeros((B, H, W, 2), dtype=jnp.float32)

        # Create state
        state = self.create_state(
            images1, init_flow, image_history_size=2, rng=rng
        )

        # Inference
        state, _ = self(images2, state, t_state)
        est_flow = state["estimates"][:, -1]

        # Metrics
        diff = est_flow - gt_flow
        epe = jnp.linalg.norm(diff, axis=-1)
        epe_mean = jnp.mean(epe, axis=(1, 2))

        gt_norm = jnp.linalg.norm(gt_flow, axis=-1)
        rel_err = (epe**2) / (jnp.maximum(gt_norm, 0.01) ** 2)
        rel_epe_mean = jnp.mean(rel_err, axis=(1, 2))

        return epe_mean, rel_epe_mean

    def get_cache_id_suffix(
        self,
        trainable_state: NNEstimatorTrainableState,
    ) -> str:
        """Get a suffix for the cache ID based on model config and params hash.

        Args:
            trainable_state: The current trainable state of the model, used
                to hash the weights.

        Returns:
            A string suffix to append to the cache ID, encoding the model
            configuration and weights.
        """
        # 1. Config hash
        config = {
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "hidden_dim": self.hidden_dim,
            "context_dim": self.context_dim,
            "corr_levels": self.corr_levels,
            "corr_radius": self.corr_radius,
            "iters": self.iters,
            "patches_groups": self.patches_groups,
            "norm_fn": self.norm_fn,
            "use_temporal_propagation": self.use_temporal_propagation,
            # Training related params (dropout, gamma, train) affect
            # training but usually we want to distinguish based on inference
            # behavior. However, if 'train' is True vs False, behavior
            # changes (e.g. dropout).
            "train": self.train,
        }
        config_str = json.dumps(config, sort_keys=True)
        h_config = hashlib.md5(config_str.encode("utf-8")).hexdigest()

        suffix = f"_c{h_config[:8]}"

        # 2. Weights hash
        if trainable_state is not None:
            # Compute hash of params
            leaves = jax.tree_util.tree_leaves(trainable_state.params)
            if leaves:
                # Use mean of each leaf to create a small signature
                # Round to 6 decimals to be robust to GPU reduction jitter
                means = [jnp.mean(x) for x in leaves]
                means_arr = jnp.array(means)

                # Pull to host
                means_np = np.array(means_arr)
                means_np = np.round(means_np, decimals=6)
                h_weights = hashlib.md5(means_np.tobytes()).hexdigest()
                suffix += f"_w{h_weights[:8]}"

        return suffix
