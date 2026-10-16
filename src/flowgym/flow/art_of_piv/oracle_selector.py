"""ART of PIV oracle selector estimator."""

from typing import Any, NoReturn

import jax.numpy as jnp
from goggles.history.types import History
from jax import lax

from flowgym.common.base import EstimatorTrainableState
from flowgym.flow.art_of_piv.estimator import ArtOfPIVEstimator
from flowgym.flow.art_of_piv.inference import _coerce_cache_payload
from flowgym.types import PRNGKey


class ArtOfPIVOracleEstimator(ArtOfPIVEstimator):
    """Picks, per frame, the library member with the lowest EPE.

    With the library's cached errors in the batch it takes the lowest valid
    cached EPE; otherwise it runs every member and compares its flow with the
    ground truth. It has no router network and is not trained.
    """

    def __init__(self, estimators_list_path: str, **kwargs: Any) -> None:
        """Initialize ArtOfPIVOracleEstimator.

        Args:
            estimators_list_path: Path to the library config file.
            **kwargs: Other ``ArtOfPIVEstimator`` arguments; router and
                training ones are ignored.

        Raises:
            ValueError: If the library has no estimators.
        """
        super().__init__(
            router_config={},
            estimators_list_path=estimators_list_path,
            oracle=True,
            **kwargs,
        )
        if self.num_estimators == 0:
            raise ValueError(f"No estimators in {estimators_list_path}.")

    @classmethod
    def get_init_param_names(cls) -> set[str]:
        """Parent ``__init__`` parameters, without the fixed ones.

        The oracle has no router and always gets the ground truth, so a
        config's ``router_config`` and ``oracle`` are dropped.

        Returns:
            The parameter names ``from_config`` keeps.
        """
        return super().get_init_param_names() - {"router_config", "oracle"}

    def _init_router(self, router_config: dict, **kwargs: Any) -> None:
        """No router network.

        Args:
            router_config: Unused.
            **kwargs: Unused.
        """

    def create_trainable_state(
        self, dummy_input: jnp.ndarray, key: PRNGKey
    ) -> EstimatorTrainableState:
        """No trainable parameters.

        Args:
            dummy_input: Unused.
            key: Unused.

        Returns:
            An empty trainable state.
        """
        return EstimatorTrainableState()

    def create_train_step(self) -> NoReturn:
        """The oracle is not trained.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("The ART of PIV oracle is not trained.")

    def _estimate(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state: EstimatorTrainableState,
        extras: dict,
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Pick the member with the lowest EPE on each frame.

        Args:
            images: Current batch of frames, shape (B, H, W).
            state: Contains "images" (B, T, H, W), "estimates"
                (B, T, H, W, 2) with the ground truth last, and "keys".
            trainable_state: Unused.
            extras: May hold the library's cached errors (``cache_payload``).

        Returns:
            Tuple of (flow, out_extras, metrics).
        """
        cache_payload = _coerce_cache_payload(extras.get("cache_payload"))
        if cache_payload is not None and cache_payload.epe_all is not None:
            epe_all = jnp.asarray(cache_payload.epe_all)
            if cache_payload.epe_mask is not None:
                valid = jnp.asarray(cache_payload.epe_mask, dtype=bool)
                epe_all = jnp.where(valid, epe_all, jnp.inf)
            choices = jnp.argmin(epe_all, axis=-1)
            errors = jnp.min(epe_all, axis=-1)
            # Frames without any cached error
            errors = jnp.where(jnp.isinf(errors), jnp.nan, errors)
            flow = jnp.zeros((*images.shape, 2), dtype=jnp.float32)
            metrics = {"choices": choices, "errors": errors}
            return flow, {"choices": choices}, metrics

        # Run every member, one after the other, on the whole batch
        member_state = {**state, "keys": state["keys"][:, jnp.newaxis, :]}

        def run_member(idx: jnp.ndarray) -> jnp.ndarray:
            branches = [
                lambda fn=fn, ts=ts: fn(images, member_state, ts)[0][
                    "estimates"
                ][:, -1]
                for fn, ts in zip(
                    self.estimator_fns, self.sub_model_states, strict=True
                )
            ]
            return lax.switch(idx, branches)

        flows = lax.map(run_member, jnp.arange(self.num_estimators))
        flows = jnp.moveaxis(flows, 0, 1)  # (B, num_estimators, H, W, 2)
        gt = state["estimates"][:, -1]
        epes = jnp.mean(
            jnp.linalg.norm(flows - gt[:, jnp.newaxis], axis=-1), axis=(2, 3)
        )
        choices = jnp.argmin(epes, axis=-1)
        flow = flows[jnp.arange(flows.shape[0]), choices]
        return (
            flow,
            {"choices": choices},
            {"choices": choices, "q_values": epes},
        )
