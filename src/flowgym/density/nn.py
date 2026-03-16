"""CNN density estimator."""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from goggles.history.types import History

from flowgym.common.base import (
    Estimator,
    NNEstimatorTrainableState,
)
from flowgym.common.evaluation import loss_supervised_density
from flowgym.nn.cnn import CNNDensityModel
from flowgym.types import PRNGKey, SupervisedExperience, SupervisedTrainStep

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
            features: List of features for the CNN.
            use_residual: Whether to use residual connections.
            norm_fn: Normalization function to use. Must be one of NormKind.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If features is empty or norm_fn is not in NormKind.
        """
        if not features:
            raise ValueError(
                f"features must be a non-empty list, got: {features}."
            )
        if not isinstance(norm_fn, str) or norm_fn not in NormKind:
            raise ValueError(
                f"norm_fn must be one of {NormKind}, got {norm_fn}."
            )

        self.model = CNNDensityModel(
            features_list=features,
            use_residual=use_residual,
            norm_fn=norm_fn,
        )

        super().__init__(**kwargs)

    def create_trainable_state(
        self, dummy_input: jnp.ndarray, key: PRNGKey
    ) -> NNEstimatorTrainableState:
        """Create the initial trainable state of the density estimator.

        Args:
            dummy_input: Batched dummy input to initialize the state.
            key: Random key for initialization.

        Returns:
            The initial trainable state of the estimator.
        """
        params = self.model.init(key, dummy_input[0][None, ...])["params"]
        tx = optax.adam(learning_rate=1e-3)
        return NNEstimatorTrainableState.create(
            apply_fn=self.model.apply, params=params, tx=tx
        )

    def create_train_step(self) -> SupervisedTrainStep:
        """Create a training step function for the estimator.

        Returns:
            SupervisedTrainStep: Training step function.
        """

        def train_step(
            trainable_state: NNEstimatorTrainableState,
            experience: SupervisedExperience,
        ) -> tuple[jnp.ndarray, NNEstimatorTrainableState, dict]:
            state = experience.state
            inputs, _ = experience.obs
            targets = experience.ground_truth
            params = trainable_state.params

            def loss_fn(params):
                tmp_ts = trainable_state.replace(params=params)
                new_state, _metrics = self(inputs, state, tmp_ts)
                preds = new_state["estimates"][:, -1]
                loss = loss_supervised_density(preds, targets)
                return loss, new_state

            (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params
            )
            trainable_state = trainable_state.apply_gradients(grads=grads)
            return loss, trainable_state, state

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
            trainable_state: Trainable state of the estimator.

        Returns:
            Tuple of (computed density, empty dict, empty dict).
        """

        def _apply_model(x):  # TODO: verify that this model works
            out = self.model.apply({"params": trainable_state.params}, x)
            # model.apply may return (output, mutated) or just output; handle
            # both cases
            if isinstance(out, tuple):
                return out[0]
            return out

        preds = jax.vmap(_apply_model, in_axes=0, out_axes=0)(images)
        return (
            jnp.expand_dims(preds, axis=1),
            {},
            {},
        )
