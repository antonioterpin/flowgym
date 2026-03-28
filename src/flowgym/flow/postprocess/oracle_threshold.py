"""Learned oracle outlier rejection estimator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from goggles.history.types import History
from jax import lax

from flowgym.common.base import (
    Estimator,
    EstimatorTrainableState,
    NNEstimatorTrainableState,
)
from flowgym.flow.postprocess.data_validation import learned_oracle_threshold
from flowgym.flow.postprocess.oracle_model import OracleMaskCNN
from flowgym.types import (
    CachePayload,
    PRNGKey,
    SupervisedExperience,
    SupervisedTrainStep,
)
from flowgym.utils import load_configuration


class LearnedOracleThresholdEstimator(Estimator):
    """Estimator that learns to emulate oracle outlier rejection."""

    def __init__(
        self,
        estimator_list: str | dict[str, Any] | list[Any] | None = None,
        threshold: float = 0.5,
        oracle_epe_threshold: float = 1.0,
        features: list[int] | tuple[int, ...] = (16, 32),
        include_image_pair: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the learned oracle threshold estimator.

        Args:
            estimator_list: Optional list/path/dict describing wrapped
                sub-estimators. If provided, training uses all sub-estimator
                flow predictions (K runs for K sub-estimators).
            threshold: Probability threshold to keep a vector as inlier.
            oracle_epe_threshold: EPE threshold used to build oracle labels.
            features: Feature sizes for the CNN head.
            include_image_pair: Whether to append the current image pair
                (previous/current grayscale frames) as additional channels.
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
        if not isinstance(include_image_pair, bool):
            raise ValueError(
                "`include_image_pair` must be a boolean, got "
                f"{type(include_image_pair)}."
            )

        self.estimator_list = self._normalize_estimator_list(estimator_list)
        self.threshold = float(threshold)
        self.oracle_epe_threshold = float(oracle_epe_threshold)
        self.features = tuple(features)
        self.include_image_pair = include_image_pair
        self.model = OracleMaskCNN(features=self.features)
        estimator_fns, inner_estimators_support_jit = self._create_estimators(
            self.estimator_list
        )
        self.estimator_fns = tuple(estimator_fns)
        self.num_sub_estimators = len(self.estimator_fns)
        self._inner_estimators_support_jit = inner_estimators_support_jit

        super().__init__(**kwargs)

    def _normalize_estimator_list(
        self,
        estimator_list: str | dict[str, Any] | list[Any] | None,
    ) -> list[dict[str, Any]]:
        """Normalize estimator-list configuration into a list of configs."""
        if estimator_list is None:
            return []
        raw: Any = estimator_list
        if isinstance(raw, str):
            loaded = load_configuration(raw)
            if loaded is None:
                raise ValueError(
                    f"Could not load estimator_list config from: {raw}"
                )
            raw = loaded
        if isinstance(raw, dict):
            raw = raw.get("estimators", [])
        if not isinstance(raw, list):
            raise ValueError(
                "estimator_list must be a list, dict with `estimators`, "
                f"or YAML path, got {type(estimator_list)}."
            )
        normalized: list[dict[str, Any]] = []
        for cfg in raw:
            if not isinstance(cfg, dict):
                raise ValueError(
                    "Each estimator config must be a dictionary, got "
                    f"{type(cfg)}."
                )
            if "estimator" not in cfg:
                raise ValueError(
                    "Each estimator config must include `estimator` key."
                )
            normalized.append(cfg)
        return normalized

    def _create_estimators(
        self,
        estimator_list: list[dict[str, Any]],
    ) -> tuple[list[Callable], bool]:
        """Create sub-estimator callables and check JIT compatibility."""
        if not estimator_list:
            return [], True
        # Local import to avoid circular dependency at module import time.
        from flowgym.make import make_estimator  # noqa: PLC0415

        estimators: list[Callable] = []
        supports_jit = True
        for cfg in estimator_list:
            (
                trainable_state,
                _,
                compute_estimate_fn,
                estimator_model,
            ) = make_estimator(
                model_config=cfg,
                load_from=cfg.get("load_from"),
            )
            supports_jit = supports_jit and estimator_model.supports_jit()

            def estimator_fn(
                input: tuple[jnp.ndarray, History],
                estimator=compute_estimate_fn,
                ts=trainable_state,
            ) -> tuple[History, dict]:
                image, state = input
                new_state, metrics = estimator(
                    image, state, cast(EstimatorTrainableState, ts)
                )
                return new_state, metrics

            estimators.append(estimator_fn)
        return estimators, supports_jit

    def _compute_sub_estimator_flows(
        self,
        images: jnp.ndarray,
        state: History,
    ) -> jnp.ndarray:
        """Run all configured sub-estimators and stack their flow outputs."""
        if self.num_sub_estimators == 0:
            raise ValueError("No sub-estimators configured.")

        input_state = dict(state)
        if "keys" in state:
            keys = state["keys"]
            input_state["keys"] = (
                keys if keys.ndim == 3 else keys[:, jnp.newaxis, :]
            )

        if self._inner_estimators_support_jit:

            def single_estimator(idx: int):
                new_state, _ = lax.switch(
                    idx, self.estimator_fns, (images, input_state)
                )
                return new_state

            states = lax.map(
                single_estimator, jnp.arange(self.num_sub_estimators)
            )
        else:
            state_list = []
            for estimator_fn in self.estimator_fns:
                new_state, _ = estimator_fn((images, input_state))
                state_list.append(new_state)
            states = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0), *state_list
            )

        flows = states["estimates"][:, :, -1, ...]
        return jnp.transpose(flows, (1, 0, 2, 3, 4)).astype(jnp.float32)

    def _flatten_flow_for_training(
        self,
        flow_field: jnp.ndarray,
        flow_gt: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Align flow/GT shapes, flatten K, and return estimator indices."""
        if flow_field.ndim == 5 and flow_gt.ndim == 4:
            flow_gt = jnp.broadcast_to(flow_gt[:, None, ...], flow_field.shape)
        if flow_field.shape != flow_gt.shape:
            raise ValueError(
                "Ground-truth flow shape mismatch: expected "
                f"{flow_field.shape}, got {flow_gt.shape}."
            )
        if flow_field.ndim == 5:
            b, k, h, w, c = flow_field.shape
            estimator_indices = jnp.broadcast_to(
                jnp.arange(k, dtype=jnp.int32)[None, :], (b, k)
            ).reshape(b * k)
            flow_field = flow_field.reshape(b * k, h, w, c)
            flow_gt = flow_gt.reshape(b * k, h, w, c)
        else:
            estimator_indices = jnp.zeros(
                (flow_field.shape[0],), dtype=jnp.int32
            )
        return (
            flow_field.astype(jnp.float32),
            flow_gt.astype(jnp.float32),
            estimator_indices,
        )

    def _build_model_input(
        self,
        flow_field: jnp.ndarray,
        estimator_indices: jnp.ndarray | None = None,
        previous_image: jnp.ndarray | None = None,
        current_image: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Build model input by appending normalized estimator-index channel."""
        if flow_field.ndim != 4 or flow_field.shape[-1] != 2:
            raise ValueError(
                "Expected flow field shape (B, H, W, 2), got "
                f"{flow_field.shape}."
            )
        b, h, w, _ = flow_field.shape
        if estimator_indices is None:
            estimator_indices_f = jnp.zeros((b,), dtype=jnp.float32)
        else:
            estimator_indices_arr = jnp.asarray(estimator_indices)
            if (
                estimator_indices_arr.ndim != 1
                or estimator_indices_arr.shape[0] != b
            ):
                raise ValueError(
                    "estimator_indices must have shape (B,), got "
                    f"{estimator_indices_arr.shape}."
                )
            estimator_indices_f = estimator_indices_arr.astype(jnp.float32)
        norm_denom = jnp.maximum(jnp.max(estimator_indices_f), 1.0)
        estimator_index_channel = jnp.broadcast_to(
            (estimator_indices_f / norm_denom)[:, None, None, None],
            (b, h, w, 1),
        )
        channels = [flow_field.astype(jnp.float32), estimator_index_channel]
        if self.include_image_pair:
            if previous_image is None or current_image is None:
                raise ValueError(
                    "`previous_image` and `current_image` are required when "
                    "`include_image_pair=True`."
                )
            prev = jnp.asarray(previous_image)
            curr = jnp.asarray(current_image)
            if prev.ndim == 4 and prev.shape[-1] == 1:
                prev = jnp.squeeze(prev, axis=-1)
            if curr.ndim == 4 and curr.shape[-1] == 1:
                curr = jnp.squeeze(curr, axis=-1)
            if prev.shape != (b, h, w):
                raise ValueError(
                    "previous_image must have shape "
                    f"(B, H, W)=({b}, {h}, {w}), got {prev.shape}."
                )
            if curr.shape != (b, h, w):
                raise ValueError(
                    "current_image must have shape "
                    f"(B, H, W)=({b}, {h}, {w}), got {curr.shape}."
                )
            channels.extend(
                [
                    prev.astype(jnp.float32)[..., None],
                    curr.astype(jnp.float32)[..., None],
                ]
            )
        return jnp.concatenate(channels, axis=-1)

    def _expand_image_pair_for_flow(
        self,
        flow_field: jnp.ndarray,
        previous_image: jnp.ndarray,
        current_image: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Align image-pair shape to match flattened flow tensor."""
        prev = jnp.asarray(previous_image)
        curr = jnp.asarray(current_image)
        if prev.ndim == 4 and prev.shape[-1] == 1:
            prev = jnp.squeeze(prev, axis=-1)
        if curr.ndim == 4 and curr.shape[-1] == 1:
            curr = jnp.squeeze(curr, axis=-1)
        if flow_field.ndim == 5:
            b, k, h, w, _ = flow_field.shape
            if prev.shape != (b, h, w) or curr.shape != (b, h, w):
                raise ValueError(
                    "Image pair shape mismatch for K-flow input: "
                    f"prev={prev.shape}, curr={curr.shape}, "
                    f"expected ({b}, {h}, {w})."
                )
            prev = jnp.broadcast_to(prev[:, None, ...], (b, k, h, w)).reshape(
                b * k, h, w
            )
            curr = jnp.broadcast_to(curr[:, None, ...], (b, k, h, w)).reshape(
                b * k, h, w
            )
            return prev.astype(jnp.float32), curr.astype(jnp.float32)
        if flow_field.ndim == 4:
            b, h, w, _ = flow_field.shape
            if prev.shape != (b, h, w) or curr.shape != (b, h, w):
                raise ValueError(
                    "Image pair shape mismatch for single-flow input: "
                    f"prev={prev.shape}, curr={curr.shape}, "
                    f"expected ({b}, {h}, {w})."
                )
            return prev.astype(jnp.float32), curr.astype(jnp.float32)
        raise ValueError(
            "flow_field must have shape (B, H, W, 2) or (B, K, H, W, 2), "
            f"got {flow_field.shape}."
        )

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

        if flow_field.ndim not in (4, 5) or flow_field.shape[-1] != 2:
            raise ValueError(
                "Expected flow field shape (B, H, W, 2) or "
                f"(B, K, H, W, 2), got "
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
        if self.include_image_pair:
            dummy_images = jnp.zeros(dummy_input.shape, dtype=jnp.float32)
            dummy_flow = self._build_model_input(
                dummy_flow,
                previous_image=dummy_images,
                current_image=dummy_images,
            )
        else:
            dummy_flow = self._build_model_input(dummy_flow)

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
        metrics: dict[str, jnp.ndarray] = {}
        cache_payload = extras.get("cache_payload")
        cache_payload_estimates = (
            getattr(cache_payload, "estimates", None)
            if cache_payload is not None
            else None
        )
        if (
            self.num_sub_estimators > 0
            and flow_field.ndim == 4
            and cache_payload_estimates is None
            and extras.get("flow_field") is None
        ):
            flows_k = self._compute_sub_estimator_flows(
                images=images,
                state=state,
            )
            _, mask_k, _ = learned_oracle_threshold(
                flow_field=flows_k,
                trainable_state=trainable_state,
                valid=extras.get("valid"),
                state=state,
                threshold_value=self.threshold,
                include_image_pair=self.include_image_pair,
                previous_image=state["images"][:, -1, ...],
                current_image=images,
                estimator_indices=jnp.broadcast_to(
                    jnp.arange(flows_k.shape[1], dtype=jnp.int32)[None, :],
                    (flows_k.shape[0], flows_k.shape[1]),
                ),
            )
            if mask_k is None:
                raise ValueError(
                    "learned_oracle_threshold returned `mask=None`."
                )
            metrics["mask"] = mask_k
            metrics["mask_flow_fields"] = flows_k
            # Keep state estimate shape unchanged for history update.
            return flows_k[:, 0, ...], {}, metrics

        flow_field, mask, _ = learned_oracle_threshold(
            flow_field=flow_field,
            trainable_state=trainable_state,
            valid=extras.get("valid"),
            state=state,
            threshold_value=self.threshold,
            include_image_pair=self.include_image_pair,
            previous_image=state["images"][:, -1, ...],
            current_image=images,
        )
        if mask is None:
            raise ValueError("learned_oracle_threshold returned `mask=None`.")
        return flow_field, {}, {"mask": mask}

    def supports_jit(self) -> bool:
        """Jittable iff all configured sub-estimators are jittable."""
        return self._inner_estimators_support_jit

    def supports_train_step_jit(self) -> bool:
        """Train step stays jittable via host-side precomputation."""
        return True

    def prepare_experience_for_training(
        self,
        experience: SupervisedExperience,
        trainable_state: NNEstimatorTrainableState,
    ) -> SupervisedExperience:
        """Precompute non-jittable sub-estimator flows before train step."""
        del trainable_state
        if (
            self.num_sub_estimators <= 0
            or self._inner_estimators_support_jit
        ):
            return experience

        existing_payload = experience.cache_payload
        existing_estimates = (
            existing_payload.estimates if existing_payload is not None else None
        )
        if existing_estimates is not None:
            return experience

        flows_k = self._compute_sub_estimator_flows(
            images=experience.obs[1],
            state=experience.state,
        )
        payload_base = existing_payload or CachePayload()
        payload = CachePayload(
            has_precomputed_errors=payload_base.has_precomputed_errors,
            epe=payload_base.epe,
            relative_epe=payload_base.relative_epe,
            epe_all=payload_base.epe_all,
            epe_mask=payload_base.epe_mask,
            estimates=flows_k,
            extras=payload_base.extras,
        )
        return SupervisedExperience(
            state=experience.state,
            obs=experience.obs,
            ground_truth=experience.ground_truth,
            cache_payload=payload,
        )

    def create_train_step(self) -> SupervisedTrainStep:
        """Create a supervised train step for oracle-mask learning."""

        def train_step(
            trainable_state: NNEstimatorTrainableState,
            experience: SupervisedExperience,
        ) -> tuple[jnp.ndarray, NNEstimatorTrainableState, dict]:
            cache_estimates = None
            if experience.cache_payload is not None:
                cache_estimates = experience.cache_payload.estimates

            if self.num_sub_estimators > 0 and cache_estimates is None:
                if not self._inner_estimators_support_jit:
                    raise ValueError(
                        "Missing cached flow estimates for non-jittable "
                        "sub-estimators. Call "
                        "`prepare_experience_for_training` before "
                        "the train step."
                    )
                flow_field = self._compute_sub_estimator_flows(
                    images=experience.obs[1],
                    state=experience.state,
                )
            else:
                flow_field = self._extract_flow_field(
                    images=experience.obs[0],
                    state=experience.state,
                    extras={"cache_payload": experience.cache_payload},
                )
            flow_field_raw = flow_field
            flow_gt = jnp.asarray(experience.ground_truth).astype(jnp.float32)
            flow_field, flow_gt, estimator_indices = (
                self._flatten_flow_for_training(
                    flow_field=flow_field, flow_gt=flow_gt
                )
            )
            prev_images_for_model: jnp.ndarray | None = None
            curr_images_for_model: jnp.ndarray | None = None
            if self.include_image_pair:
                prev_images_for_model, curr_images_for_model = (
                    self._expand_image_pair_for_flow(
                        flow_field=flow_field_raw,
                        previous_image=experience.obs[0],
                        current_image=experience.obs[1],
                    )
                )
            model_input = self._build_model_input(
                flow_field=flow_field,
                estimator_indices=estimator_indices,
                previous_image=prev_images_for_model,
                current_image=curr_images_for_model,
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
                    {"params": params}, model_input
                )
                logits = self._coerce_logits(output)
                eps = jnp.asarray(1e-8, dtype=jnp.float32)
                bce = optax.sigmoid_binary_cross_entropy(logits, labels_f)
                pos_frac = jnp.mean(labels_f)
                neg_frac = 1.0 - pos_frac
                pos_weight = 0.5 / jnp.maximum(pos_frac, eps)
                neg_weight = 0.5 / jnp.maximum(neg_frac, eps)
                weights = labels_f * pos_weight + (1.0 - labels_f) * neg_weight
                loss = jnp.mean(bce * weights)
                probs = jax.nn.sigmoid(logits)
                preds = probs > self.threshold
                preds_f = preds.astype(jnp.float32)
                labels_bool = labels.astype(jnp.bool_)

                tp = jnp.sum(preds & labels_bool, axis=(1, 2)).astype(
                    jnp.float32
                )
                tn = jnp.sum((~preds) & (~labels_bool), axis=(1, 2)).astype(
                    jnp.float32
                )
                fp = jnp.sum(preds & (~labels_bool), axis=(1, 2)).astype(
                    jnp.float32
                )
                fn = jnp.sum((~preds) & labels_bool, axis=(1, 2)).astype(
                    jnp.float32
                )

                accuracy = jnp.mean((tp + tn) / (tp + tn + fp + fn + eps))
                precision = jnp.mean(tp / (tp + fp + eps))
                recall = jnp.mean(tp / (tp + fn + eps))
                specificity = jnp.mean(tn / (tn + fp + eps))
                balanced_accuracy = 0.5 * (recall + specificity)
                f1 = jnp.mean(
                    (2.0 * tp) / (2.0 * tp + fp + fn + eps)
                )
                iou = jnp.mean(tp / (tp + fp + fn + eps))
                pred_inlier_frac = jnp.mean(preds_f)
                oracle_inlier_frac = jnp.mean(labels_f)
                aux = {
                    "mask_accuracy": accuracy,
                    "mask_precision": precision,
                    "mask_recall": recall,
                    "mask_specificity": specificity,
                    "mask_balanced_accuracy": balanced_accuracy,
                    "mask_f1": f1,
                    "mask_iou": iou,
                    "pred_inlier_fraction": pred_inlier_frac,
                    "oracle_inlier_fraction": oracle_inlier_frac,
                    "class_weight_pos": pos_weight,
                    "class_weight_neg": neg_weight,
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
