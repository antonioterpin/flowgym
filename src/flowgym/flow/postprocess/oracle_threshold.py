"""Learned oracle outlier rejection estimator."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from goggles.history.types import History

from flowgym.common.base import Estimator, NNEstimatorTrainableState
from flowgym.flow.postprocess.data_validation import learned_oracle_threshold
from flowgym.types import PRNGKey, SupervisedExperience, SupervisedTrainStep


class _OracleMaskCNN(nn.Module):
    """Small CNN producing per-pixel mask logits from a flow field."""

    features: tuple[int, ...] = (16, 32)

    @nn.compact
    def __call__(self, flow_field: jnp.ndarray) -> jnp.ndarray:
        x = flow_field
        for feature in self.features:
            x = nn.Conv(feature, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
        logits = nn.Conv(1, kernel_size=(1, 1), padding="SAME")(x)
        return jnp.squeeze(logits, axis=-1)


class LearnedOracleThresholdEstimator(Estimator):
    """Estimator that learns to emulate oracle outlier rejection."""

    def __init__(
        self,
        estimator_list: list[Any] | None = None,
        threshold: float = 0.5,
        oracle_epe_threshold: float = 1.0,
        features: list[int] | tuple[int, ...] = (16, 32),
        **kwargs: Any,
    ) -> None:
        """Initialize the learned oracle threshold estimator.

        Args:
            estimator_list: Optional list of wrapped estimators.
            threshold: Probability threshold to keep a vector as inlier.
            oracle_epe_threshold: EPE threshold used to build oracle labels.
            features: Feature sizes for the CNN head.
            **kwargs: Additional keyword arguments for the base estimator.
        """
        if not isinstance(threshold, (int, float)) or not (
            0.0 < threshold < 1.0
        ):
            raise ValueError(
                f"`threshold` must be in (0, 1), got {threshold}."
            )
        if not isinstance(oracle_epe_threshold, (int, float)) or (
            oracle_epe_threshold <= 0.0
        ):
            raise ValueError(
                "`oracle_epe_threshold` must be positive, got "
                f"{oracle_epe_threshold}."
            )
        if not features:
            raise ValueError("`features` must be non-empty.")
        if not all(isinstance(f, int) and f > 0 for f in features):
            raise ValueError(
                "`features` must contain only positive integers, got "
                f"{features}."
            )

        self.estimator_list = (
            estimator_list if estimator_list is not None else []
        )
        self.threshold = float(threshold)
        self.oracle_epe_threshold = float(oracle_epe_threshold)
        self.features = tuple(features)
        self.model = _OracleMaskCNN(features=self.features)

        super().__init__(**kwargs)

    def _coerce_logits(self, output: Any) -> jnp.ndarray:
        """Convert model output to logits with shape (B, H, W)."""
        logits = output[0] if isinstance(output, tuple) else output
        logits = jnp.asarray(logits)
        if logits.ndim == 4 and logits.shape[-1] == 1:
            logits = jnp.squeeze(logits, axis=-1)
        if logits.ndim != 3:
            raise ValueError(
                "Oracle mask model must return logits of shape (B, H, W) or "
                f"(B, H, W, 1), got {logits.shape}."
            )
        return logits

    def _extract_flow_field(
        self,
        images: jnp.ndarray,
        state: History,
        extras: dict[str, Any],
    ) -> jnp.ndarray:
        """Extract a flow field from cache payload or estimator history."""
        cache_payload = extras.get("cache_payload")
        flow_field: jnp.ndarray | None = None
        if cache_payload is not None:
            payload_flow = getattr(cache_payload, "estimates", None)
            if payload_flow is not None:
                flow_field = jnp.asarray(payload_flow)

        if flow_field is None and extras.get("flow_field") is not None:
            flow_field = jnp.asarray(extras["flow_field"])

        if flow_field is None:
            flow_field = jnp.asarray(state["estimates"][:, -1, ...])

        if flow_field.ndim != 4 or flow_field.shape[-1] != 2:
            raise ValueError(
                "Expected flow field shape (B, H, W, 2), got "
                f"{flow_field.shape}."
            )

        _ = images  # Explicitly unused: flow is obtained from extras/state.
        return flow_field.astype(jnp.float32)

    def create_trainable_state(
        self,
        dummy_input: jnp.ndarray,
        key: PRNGKey,
    ) -> NNEstimatorTrainableState:
        """Create the trainable state for the oracle mask model.

        Args:
            dummy_input: Dummy input with shape (B, H, W) or (B, H, W, 2).
            key: PRNG key for model initialization.

        Returns:
            Initialized neural-network trainable state.
        """
        if dummy_input.ndim == 3:
            dummy_flow = jnp.zeros((*dummy_input.shape, 2), dtype=jnp.float32)
        elif dummy_input.ndim == 4 and dummy_input.shape[-1] == 2:
            dummy_flow = dummy_input.astype(jnp.float32)
        else:
            raise ValueError(
                "dummy_input must have shape (B, H, W) or (B, H, W, 2), "
                f"got {dummy_input.shape}."
            )

        params = self.model.init(key, dummy_flow[:1])["params"]
        optimizer_config = self.optimizer_config or {
            "name": "adam",
            "learning_rate": 1e-3,
        }
        return NNEstimatorTrainableState.from_config(
            apply_fn=self.model.apply,
            params=params,
            optimizer_config=optimizer_config,
        )

    def _estimate(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state: NNEstimatorTrainableState,
        extras: dict[str, Any],
    ) -> tuple[jnp.ndarray, dict[str, Any], dict[str, jnp.ndarray]]:
        """Estimate a flow field and emit the learned inlier mask metric."""
        flow_field = self._extract_flow_field(images, state, extras)
        flow_field, mask, _ = learned_oracle_threshold(
            flow_field=flow_field,
            trainable_state=trainable_state,
            valid=extras.get("valid"),
            state=state,
            threshold_value=self.threshold,
        )
        if mask is None:
            raise ValueError("learned_oracle_threshold returned `mask=None`.")
        return flow_field, {}, {"mask": mask}

    def create_train_step(self) -> SupervisedTrainStep:
        """Create a supervised train step for oracle-mask learning."""

        def train_step(
            trainable_state: NNEstimatorTrainableState,
            experience: SupervisedExperience,
        ) -> tuple[jnp.ndarray, NNEstimatorTrainableState, dict]:
            flow_field = self._extract_flow_field(
                images=experience.obs[0],
                state=experience.state,
                extras={"cache_payload": experience.cache_payload},
            )
            flow_gt = jnp.asarray(experience.ground_truth).astype(jnp.float32)
            if flow_gt.shape != flow_field.shape:
                raise ValueError(
                    "Ground-truth flow shape mismatch: expected "
                    f"{flow_field.shape}, got {flow_gt.shape}."
                )

            labels = (
                jnp.linalg.norm(flow_field - flow_gt, axis=-1)
                <= self.oracle_epe_threshold
            )
            labels_f = labels.astype(jnp.float32)

            if trainable_state.apply_fn is None:
                raise ValueError("trainable_state.apply_fn cannot be None.")

            def loss_fn(params):
                output = trainable_state.apply_fn(
                    {"params": params}, flow_field
                )
                logits = self._coerce_logits(output)
                loss = jnp.mean(
                    optax.sigmoid_binary_cross_entropy(logits, labels_f)
                )
                probs = jax.nn.sigmoid(logits)
                preds = probs > self.threshold
                accuracy = jnp.mean((preds == labels).astype(jnp.float32))
                pred_inlier_frac = jnp.mean(preds.astype(jnp.float32))
                oracle_inlier_frac = jnp.mean(labels_f)
                aux = {
                    "mask_accuracy": accuracy,
                    "pred_inlier_fraction": pred_inlier_frac,
                    "oracle_inlier_fraction": oracle_inlier_frac,
                }
                return loss, aux

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                trainable_state.params
            )
            grad_norm = optax.global_norm(grads)
            new_state = trainable_state.apply_gradients(grads=grads)

            metrics = {
                "loss": loss,
                "grad_norm": grad_norm,
                **aux,
            }
            return loss, new_state, metrics

        return train_step
