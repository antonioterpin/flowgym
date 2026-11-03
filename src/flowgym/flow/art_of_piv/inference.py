"""Inference logic: estimation and sub-estimator dispatch for ART of PIV."""

from logging import getLogger
from typing import Any

import jax
import jax.numpy as jnp

from flowgym.common.base import (
    EstimatorTrainableState,
    NNEstimatorTrainableState,
)
from flowgym.flow.art_of_piv.metrics import build_inference_metrics
from flowgym.flow.art_of_piv.types import (
    EstimateFn,
    RunRouterFn,
    RunSubEstimatorsFn,
)
from flowgym.types import CachePayload, History

logger = getLogger(__name__)


def _coerce_cache_payload(
    payload: CachePayload | dict[str, Any] | None,
) -> CachePayload | None:
    """Normalize cache payload inputs to ``CachePayload`` for internals.

    Accepts dicts for backward compatibility with tests/callers that still
    pass loose payload dictionaries.

    Args:
        payload: Cache payload candidate from estimator extras.

    Returns:
        Normalized cache payload or ``None`` when payload cannot be coerced.
    """
    if payload is None:
        return None
    if isinstance(payload, CachePayload):
        return payload
    if isinstance(payload, dict):
        return CachePayload.from_enrich_result(payload)
    return None


def estimate(
    images: jnp.ndarray,
    state: History,
    trainable_state: NNEstimatorTrainableState,
    extras: dict[str, Any],
    *,
    use_temporal_propagation: bool,
    use_last_estimate: bool,
    run_router_fn: RunRouterFn,
    run_subestimators_fn: RunSubEstimatorsFn,
    cache_holds_epe: bool = True,
) -> tuple[jnp.ndarray, dict[str, Any], dict[str, Any]]:
    """Compute the flow field between the two most recent frames.

    Args:
        images: Current batch of frames, shape (B, H, W).
        state: Contains "images" (B, T, H, W), "estimates" (B, T, H, W, 2),
            "keys" (B, 2).
        trainable_state: The current state of the model.
        extras: Additional information for the estimation.
        use_temporal_propagation: Whether to use previous flow estimate.
        use_last_estimate: Whether to pass flow to the router.
        run_router_fn: Bound router function that returns (choices, qs,
            exploration_rate, estimate_keys).
        run_subestimators_fn: Bound sub-estimator dispatch function.
        cache_holds_epe: Whether a cached ``epe_all`` is the sub-estimators'
            EPE. Only then can it stand in for running them; other router
            labels (e.g. ``corr16_peak``) are not errors to report.

    Returns:
        Tuple of (flow, out_extras, metrics).
    """
    # Get the most recent images from the state
    prev = state["images"][:, -1, ...]
    curr = images

    # Check if the state has a history of estimates
    if use_temporal_propagation:
        # Use the last estimate from the history
        flow = state["estimates"][:, -1, ...]
    else:
        # Initialize flow to zeros of the same shape as prev but with 2 channels
        flow = jnp.zeros(
            (prev.shape[0], prev.shape[1], prev.shape[2], 2),
            dtype=jnp.float32,
        )

    # Run consolidated router.
    #
    # Read ``is_training`` from ``cache_payload.is_training`` (a static pytree
    # aux field) rather than from ``extras["train"]``. Aux participates in the
    # JIT cache key, so flipping the flag forces a recompile -- acceptable
    # because train and eval each compile once per phase, and in exchange the
    # flag is available as a plain Python bool inside traced code (no need to
    # thread it through as a JAX array). ``eval.py`` flips this flag to
    # ``False`` during validation so the router behaves greedily regardless of
    # the configured exploration schedule (``router.run_router`` zeros out
    # ``exploration_val`` via ``jnp.where(train, exploration_val, 0.0)``).
    cache_payload = _coerce_cache_payload(extras.get("cache_payload"))
    is_training = (
        cache_payload.is_training if cache_payload is not None else True
    )
    choices, qs, exploration_rate, _estimate_keys = run_router_fn(
        prev,
        curr,
        state,
        trainable_state,
        flow=flow if use_last_estimate else None,
        train=is_training,
    )

    # Short-circuit if we have cached EPEs from CachePayload.
    # This optimization is valid because sub-estimators are frozen during
    # both training and evaluation - only the router weights change.
    if (
        cache_holds_epe
        and cache_payload is not None
        and cache_payload.epe_all is not None
    ):
        # Handle potential partial cache misses.
        epe_all = jnp.asarray(cache_payload.epe_all)
        epe_mask_raw = cache_payload.epe_mask
        if epe_mask_raw is not None:
            epe_mask = jnp.asarray(epe_mask_raw, dtype=bool)
        else:
            epe_mask = jnp.ones_like(epe_all, dtype=bool)

        # Short-circuit: use cached EPEs without running sub-estimators.
        B = qs.shape[0]
        choices_idx = jnp.reshape(choices, (B, 1))
        chosen_epe = jnp.take_along_axis(epe_all, choices_idx, axis=-1).reshape(
            B
        )
        chosen_valid = jnp.take_along_axis(
            epe_mask, choices_idx, axis=-1
        ).reshape(B)
        chosen_epe = jnp.where(chosen_valid, chosen_epe, jnp.nan)
        chosen_q = jnp.take_along_axis(qs, choices_idx, axis=-1).reshape(B)

        metrics = build_inference_metrics(
            qs=qs,
            choices=choices,
            exploration_rate=exploration_rate,
            steps=jnp.asarray(trainable_state.extras["steps"][:, 0]),
            estimated_epe=chosen_q,
            cache_payload=cache_payload,
            errors=chosen_epe,  # This feeds into eval.py metrics.
        )
        out_extras = {"choices": choices_idx.reshape(B)}

        # Return dummy flow (zeros)
        # Flow shape: (B, H, W, 2)
        dummy_flow = jnp.zeros_like(prev)[..., None].repeat(2, axis=-1)

        return dummy_flow, out_extras, metrics

    # If we reach here, we actually need to run the estimators
    return run_subestimators_fn(
        state,
        trainable_state,
        curr,
        choices,
        extras,
        qs,
        exploration_rate,
    )


def run_subestimators(
    state: History,
    trainable_state: NNEstimatorTrainableState,
    curr: jnp.ndarray,
    choices: jnp.ndarray,
    extras: dict[str, Any],
    qs: jnp.ndarray,
    exploration_rate: jnp.ndarray,
    *,
    estimator_fns: tuple[EstimateFn, ...],
    estimator_states: tuple[EstimatorTrainableState, ...],
) -> tuple[jnp.ndarray, dict[str, Any], dict[str, Any]]:
    """Run selected sub-estimators and compute metrics.

    Args:
        state: History dict.
        trainable_state: Current trainable state.
        curr: Current frame.
        choices: Selected estimator indices, shape (B,).
        extras: Additional info.
        qs: Q-values, shape (B, num_estimators).
        exploration_rate: Current exploration values.
        estimator_fns: Tuple of sub-estimator callables.
        estimator_states: Tuple of sub-estimator trainable states.

    Returns:
        Tuple of (flows, out_extras, metrics).

    Raises:
        ValueError: If `estimator_fns` and `estimator_states` lengths differ.
    """
    if len(estimator_fns) != len(estimator_states):
        raise ValueError(
            "estimator_fns and estimator_states must have the same length, "
            f"got {len(estimator_fns)} and {len(estimator_states)}."
        )

    def select(
        input_tuple: tuple[int, tuple[jnp.ndarray, History]],
    ) -> History:
        idx, (curr_img, s) = input_tuple

        def branch_fn(i: int):
            new_s = jax.tree.map(lambda x: x[jnp.newaxis, ...], s)
            new_curr = curr_img[jnp.newaxis, ...]
            result_state, _ = estimator_fns[i](
                new_curr, new_s, estimator_states[i], None
            )
            return jax.tree.map(lambda x: x[0], result_state)

        return jax.lax.switch(
            idx,
            [lambda _, i=i: branch_fn(i) for i in range(len(estimator_fns))],
            None,
        )

    states = jax.lax.map(select, (choices, (curr, state)))
    flows = states["estimates"][:, -1]

    # Robust indexing for estimated_epe
    B = qs.shape[0]
    choices_idx = jnp.reshape(choices, (B, 1))
    chosen_q = jnp.take_along_axis(qs, choices_idx, axis=-1).reshape(B)

    out_extras = {"choices": choices_idx.reshape(B)}

    metrics = build_inference_metrics(
        qs=qs,
        choices=choices,
        exploration_rate=exploration_rate,
        steps=jnp.asarray(trainable_state.extras["steps"][:, 0]),
        estimated_epe=chosen_q,
        cache_payload=_coerce_cache_payload(extras.get("cache_payload")),
    )

    return flows, out_extras, metrics
