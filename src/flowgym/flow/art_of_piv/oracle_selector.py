"""ART of PIV oracle selector estimator."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from goggles import Metrics
from goggles.history.spec import HistoryFieldSpec, HistorySpec
from goggles.history.types import History
from jax import lax

from flowgym.common.base import (
    EstimatorTrainableState,
    NNEstimatorTrainableState,
)
from flowgym.flow.art_of_piv import validation as art_of_piv_validation
from flowgym.flow.base import FlowFieldEstimator
from flowgym.types import PRNGKey

NORM_KINDS: tuple[str, ...] = ("batch", "group", "instance", "none")


class ArtOfPIVOracleEstimator(FlowFieldEstimator):
    """Oracle-selector estimator for the ART of PIV estimator set."""

    def __init__(
        self,
        estimators_list_path: str,
        use_temporal_propagation: bool = False,
        output_full_res: bool = True,
        use_last_estimate: bool = False,
        image_shape: tuple | None = None,
        rng: PRNGKey | int | None = None,
        **kwargs,
    ) -> None:
        """Initialize ArtOfPIVOracleEstimator.

        Args:
            estimators_list_path: Path to the estimators list config file.
            use_temporal_propagation: Whether to use temporal propagation
                of estimates.
            output_full_res: Whether to output full resolution flow fields.
            use_last_estimate: Whether to use the last estimate from
                history.
            image_shape: Shape of the input images (B, H, W), forwarded to
                each sub-estimator so its trainable state can be initialized
                when no checkpoint is provided.
            rng: Random number generator key or seed.
            **kwargs: Additional keyword arguments for the base
                FlowFieldEstimator.

        Raises:
            ValueError: If the estimator list is invalid or empty.
        """
        (
            self.output_full_res,
            self.use_temporal_propagation,
            _,
        ) = art_of_piv_validation.validate_boolean_flags(
            output_full_res=output_full_res,
            use_temporal_propagation=use_temporal_propagation,
            use_last_estimate=use_last_estimate,
        )

        # Load and validate constructor inputs via shared ART of PIV validators.
        estimators_list_config = (
            art_of_piv_validation.validate_and_load_estimators_config(
                estimators_list_path
            )
        )
        rng = art_of_piv_validation.validate_rng(rng)

        # The estimators should take as input the tuple (image, state)
        self.estimator_fns = tuple(
            self._create_estimators(
                estimators_list_config["estimators"],
                rng=rng,
                image_shape=image_shape,
            )
        )

        # Check if the number of estimators is valid
        if len(self.estimator_fns) == 0:
            raise ValueError(
                "No estimators found. Please check the configuration file."
            )
        self.num_estimators = len(self.estimator_fns)

        super().__init__(**kwargs)

    def _create_estimators(
        self,
        configs: list[dict],
        rng: PRNGKey | None = None,
        image_shape: tuple | None = None,
    ) -> list[Callable]:
        """Create the list of estimators based on the configurations.

        Each estimator should specify a name and the required parameters.
        If the whole estimator wants to be jitted, each inner estimator
        should be jittable as well.

        Args:
            configs: List of configurations for each estimator.
            rng: Random number generator key.
            image_shape: Shape of the input images (B, H, W), forwarded to
                ``make_estimator`` so sub-estimator trainable state can be
                initialized when no checkpoint is provided.

        Returns:
            List of estimator callables.
        """
        # Import here to avoid circular imports
        from flowgym.make import make_estimator  # noqa: PLC0415

        estimators: list[Callable] = []
        for cfg in configs:
            if rng is not None:
                rng, subkey = jax.random.split(rng)
            else:
                subkey = None
            (trainable_state, _, compute_estimate_fn, _) = make_estimator(
                estimator_config=cfg,
                image_shape=image_shape,
                load_from=cfg.get("load_from"),
                rng=subkey,
            )
            assert trainable_state is not None

            def estimator_fn(
                input, estimator=compute_estimate_fn, ts=trainable_state
            ):
                image, state = input
                state, metrics = estimator(image, state, ts)
                return state, metrics

            estimators.append(estimator_fn)
        return estimators

    def create_train_step(self) -> Callable:
        """No-op train step to mimic trainable estimators.

        Returns:
            Callable: A no-op train step function.
        """

        def train_step(
            *args, **kwargs
        ) -> tuple[float, EstimatorTrainableState, None]:
            return 0.0, self.create_trainable_state(None, None), None  # pyright: ignore[reportArgumentType]

        return train_step

    def _estimate(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state: NNEstimatorTrainableState,
        extras: dict,
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the flow field between the two most recent frames.

        Note: there is no need to handle key splitting here, as the key
        is splitted in the upper __call__. The keys are stored in the state
        and are of shape (B, 2) already.

        Args:
            images: Current batch of frames, shape (B, H, W).
            state: Contains:
                - "images" of shape (B, T, H, W)
                - "estimates" of shape (B, T, H, W, 2)
                - "keys" of shape (B, 2)
            trainable_state: The current state of the model.
            extras: Additional information for the estimation.

        Returns:
            jnp.ndarray: Flow field of shape (B, H, W, 2) as float32.

        Raises:
            NotImplementedError: If oracle mode is not set.
        """
        # Get the current image
        curr = images

        if self.is_oracle():
            gt_flows = state["estimates"][:, -1, ...]  # (B, H, W, 2)

        # Prepare the input state for estimators
        input_state = state
        input_state["keys"] = state["keys"][:, jnp.newaxis, :]

        # Compute the flow fields using all available algorithms
        def single_estimator(idx: int) -> tuple:
            """Compute the flow field batch for a single estimator.

            Args:
                idx: Index of the estimator.

            Returns:
                The computed flow field for the estimator.
            """
            # Select the estimator based on the index
            new_state, metrics = lax.switch(
                idx, self.estimator_fns, (curr, input_state)
            )

            return new_state, metrics

        states, _ = lax.map(single_estimator, jnp.arange(self.num_estimators))

        # Extract the flow fields from the states
        flows = states["estimates"][
            :, :, -1, ...
        ]  # Shape (num_estimators, B, H, W, 2)
        flows = jnp.transpose(
            flows, (1, 0, 2, 3, 4)
        )  # Shape (B, num_estimators, H, W, 2)

        # Select the best estimator based on the oracle (ground truth)
        if self.is_oracle():
            epes = jnp.linalg.norm(
                flows - gt_flows[:, jnp.newaxis, ...], axis=-1
            )  # (B, num_estimators, H, W)
            mean_epes = jnp.mean(epes, axis=(2, 3))  # (B, num_estimators)
            choices = jnp.argmin(mean_epes, axis=-1)  # (B,)
        else:
            raise NotImplementedError("Only oracle selector is implemented.")

        # Gather the selected flows
        batch_indices = jnp.arange(flows.shape[0])
        selected_flows = flows[batch_indices, choices, ...]  # (B, H, W, 2)
        qs = mean_epes  # For logging purposes

        # Pack the extras and metrics
        out_extras = {
            "choices": choices,
        }

        metrics = {
            "q_values": qs,
            "choices": choices,
        }

        return selected_flows, out_extras, metrics

    def _create_extras(
        self,
    ) -> HistorySpec:
        """Create the extras specification for the estimator.

        Returns:
            HistorySpec: The created extras specification.
        """
        extras_spec = HistorySpec(
            fields={
                "choices": HistoryFieldSpec(
                    length=1,
                    shape=(),
                    dtype=jnp.int32,
                    init="zeros",
                ),
            }
        )
        return extras_spec

    def process_metrics(
        self,
        metrics: dict,
        *,
        flow_field: jnp.ndarray | None = None,
        flow_field_gt: jnp.ndarray | None = None,
    ) -> Metrics:
        """Process and format the metrics for logging.

        Args:
            metrics: Raw metrics dictionary.
            flow_field: Final flow field estimate (unused by this override,
                kept for signature compatibility).
            flow_field_gt: Ground-truth flow field (unused by this override).

        Returns:
            Metrics: Formatted metrics for logging.
        """
        # eval_flow passes these to every estimator; this override does not
        # need them, and del marks them as intentionally unused for linters.
        del flow_field, flow_field_gt
        processed_metrics: dict = {}

        if "q_values" in metrics:
            q_values = metrics["q_values"]
            mean_q = jnp.mean(q_values)
            max_q = jnp.max(q_values)
            min_q = jnp.min(q_values)
            processed_metrics["q_values/mean"] = np.asarray(mean_q)
            processed_metrics["q_values/max"] = np.asarray(max_q)
            processed_metrics["q_values/min"] = np.asarray(min_q)

        if "choices" in metrics:
            choices = metrics["choices"]  # shape (batch_size,)
            batch_size = int(choices.shape[0])

            # Lazy initialization of running statistics
            if not hasattr(self, "_choices_running_counts"):
                # Total counts for each estimator over all seen samples.
                self._choices_running_counts = jnp.zeros(
                    (self.num_estimators,), dtype=jnp.int32
                )
                # Total number of samples seen so far.
                self._choices_total_count = 0

            # Sliding window configuration and state.
            if not hasattr(self, "_choices_window_buffer"):
                # List of (counts_per_estimator, batch_size) for last W batches.
                self._choices_window_buffer: list[tuple[jnp.ndarray, int]] = []
                self._choices_window_sum = jnp.zeros(
                    (self.num_estimators,), dtype=jnp.int32
                )
                self._choices_window_total = 0
                self._choices_window_size = getattr(
                    self, "choices_window_size", 100
                )

            # Per-batch statistics
            # Counts of each estimator in the current batch.
            batch_counts = []
            for i in range(self.num_estimators):
                count_i = jnp.sum(choices == i)
                rate_i = count_i / batch_size  # instantaneous batch rate
                processed_metrics[f"choices/rate_estimator_{i}"] = np.asarray(
                    rate_i
                )
                batch_counts.append(count_i)

            batch_counts = jnp.stack(batch_counts).astype(jnp.int32)

            # --- Running mean over all samples so far ---
            self._choices_running_counts = (
                self._choices_running_counts + batch_counts
            )
            self._choices_total_count += batch_size

            running_rates = self._choices_running_counts / float(
                self._choices_total_count
            )
            for i in range(self.num_estimators):
                processed_metrics[f"choices/running_rate_estimator_{i}"] = (
                    np.asarray(running_rates[i])
                )

            # Sliding-window mean over last W batches
            # Update buffer and aggregate sums.
            self._choices_window_buffer.append((batch_counts, batch_size))
            self._choices_window_sum = self._choices_window_sum + batch_counts
            self._choices_window_total += batch_size

            # Remove oldest batch if buffer is longer than window size.
            if len(self._choices_window_buffer) > self._choices_window_size:
                old_counts, old_batch_size = self._choices_window_buffer.pop(0)
                self._choices_window_sum = self._choices_window_sum - old_counts
                self._choices_window_total -= old_batch_size

            if self._choices_window_total > 0:
                window_rates = self._choices_window_sum / float(
                    self._choices_window_total
                )
                for i in range(self.num_estimators):
                    processed_metrics[f"choices/window_rate_estimator_{i}"] = (
                        np.asarray(window_rates[i])
                    )

        if "exploration_rate" in metrics:
            exploration_rate = metrics["exploration_rate"][0]
            processed_metrics["exploration/rate"] = np.asarray(exploration_rate)

        return processed_metrics
