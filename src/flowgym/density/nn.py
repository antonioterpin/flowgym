"""CNN density estimator."""

from typing import Any
import jax
import jax.numpy as jnp
import optax

from flowgym.common.base import (
    Estimator,
    NNEstimatorTrainableState,
)
from flowgym.nn.cnn import CNNDensityModel
from flowgym.common.evaluation import loss_supervised_density
from goggles.history.types import History

NormKind = ("batch", "group", "instance", "none")


class NNDensityEstimator(Estimator):
    """NN density estimator."""

    def __init__(
        self,
        features: list,
        use_residual: bool,
        norm_fn: str = "none",
        **kwargs: Any,
    ):
        """Initialize the estimator with the bandwidth.

        Args:
            background_suppression: Background suppression level.
            intensity_capping: n_std for intensity capping.
            intensity_clipping: Intensity clipping value.
            features: List of features for the CNN.
            use_residual: Whether to use residual connections.
            norm_fn (str): Normalization function to use. Must be one of NormKind.
            kwargs: Additional keyword arguments.
        """
        if not features:
            raise ValueError(f"features must be a non-empty list, got: {features}.")
        if not isinstance(use_residual, bool):
            raise TypeError(f"use_residual must be a boolean, got {use_residual}.")
        if not isinstance(norm_fn, str) or norm_fn not in NormKind:
            raise ValueError(f"norm_fn must be one of {NormKind}, got {norm_fn}.")

        self.model = CNNDensityModel(
            features_list=features,
            use_residual=use_residual,
            norm_fn=norm_fn,
        )

        super().__init__(**kwargs)

    def create_trainable_state(
        self, dummy_input: jnp.ndarray
    ) -> NNEstimatorTrainableState:
        """Create the initial trainable state of the density estimator.

        Args:
            dummy_input: Batched dummy input to initialize the state.

        Returns:
            The initial trainable state of the estimator.
        """
        params = self.model.init(jax.random.PRNGKey(0), dummy_input[0][None, ...])[
            "params"
        ]
        tx = optax.adam(learning_rate=1e-3)
        return NNEstimatorTrainableState.create(params, tx)

    def create_train_step(self):
        """Create a training step function for the estimator.

        Returns:
            Training step function.
        """

        def train_step(state, trainable_state, inputs, targets):
            params = trainable_state.params

            def loss_fn(params):
                tmp_ts = trainable_state.replace(params=params)
                new_state, metrics = self(inputs, state, tmp_ts)
                preds = new_state["estimates"][:, -1]
                loss = loss_supervised_density(preds, targets)
                return loss, new_state

            (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            trainable_state = trainable_state.apply_gradients(grads)
            return loss, state, trainable_state

        return train_step

    def _estimate(
        self,
        images: jnp.ndarray,
        _: History,
        trainable_state: NNEstimatorTrainableState,
        __: dict,
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the density from the image.

        Args:
            images: Input images (B, H, W).
            state: Current state of the estimator.
            trainable_state (NNEstimatorTrainableState):
                Trainable state of the estimator.
            __: Unused parameter.

        Returns:
            - Computed density.
            - None placeholder for additional outputs.
            - None: placeholder for additional outputs.
        """

        def _apply_model(x):  # TODO: verify that this model works
            out = self.model.apply({"params": trainable_state.params}, x)
            # model.apply may return (output, mutated) or just output; handle both cases
            if isinstance(out, tuple):
                return out[0]
            return out

        preds = jax.vmap(_apply_model, in_axes=0, out_axes=0)(images)
        return (
            jnp.expand_dims(preds, axis=1),
            {},
            {},
        )
