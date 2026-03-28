"""ConsensusFlowEstimator class."""

import csv
import os
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from goggles import Metrics, get_logger
from goggles.history.types import History
from jax import lax

from flowgym.common.base.trainable_state import EstimatorTrainableState
from flowgym.flow.base import FlowFieldEstimator
from flowgym.flow.consensus.consensus_algorithms import (
    CONSENSUS_REGISTRY,
    validate_experimental_params,
)
from flowgym.flow.consensus.objectives import make_weights
from flowgym.make import make_estimator
from flowgym.types import ExperimentParams, PRNGKey
from flowgym.utils import load_configuration

logger = get_logger(__name__)


class ConsensusFlowEstimator(FlowFieldEstimator):
    """ADMM-based flow field estimator."""

    def __init__(
        self,
        consensus_algorithm: str,
        estimators_list_path: str,
        consensus_config: dict | None = None,
        use_temporal_propagation: bool = False,
        experiment_params: dict | None = None,
        rng: PRNGKey | int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Distributed DIS estimator.

        Args:
            consensus_algorithm: Consensus algorithm to use.
            estimators_list_path: Path to the list of estimators.
            consensus_config: Configuration for the consensus algorithm.
            use_temporal_propagation: Whether to use temporal propagation.
            experiment_params: Experimental parameters.
            rng: Random number generator key or seed.
            **kwargs: Additional keyword arguments for the base class.

        Raises:
            TypeError: If use_temporal_propagation is not a boolean or
                estimators_list_path is not a string.
            ValueError: If estimators_list_path doesn't end with .yaml,
                no estimators are found, or invalid consensus algorithm.
        """
        if not isinstance(use_temporal_propagation, bool):
            raise TypeError(
                f"use_temporal_propagation must be a boolean, "
                f"got {use_temporal_propagation}."
            )
        self.use_temporal_propagation = bool(use_temporal_propagation)

        # Load the DIS algorithms from the specified path
        if not isinstance(estimators_list_path, str):
            msg = "estimators_list_path must be str, "
            msg += f"got {type(estimators_list_path)}."
            raise TypeError(msg)
        if not estimators_list_path.endswith(".yaml"):
            msg = "estimators_list_path must be .yaml, "
            msg += f"got {estimators_list_path}."
            raise ValueError(msg)
        estimators_list_config = load_configuration(estimators_list_path)

        if isinstance(rng, int):
            rng = jax.random.PRNGKey(rng)

        # The estimators should take as input the tuple (image, state)
        self.estimator_fns = tuple(
            self._create_estimators(
                estimators_list_config["estimators"], rng=rng
            )
        )

        # Check if the number of estimators is valid
        if len(self.estimator_fns) == 0:
            raise ValueError(
                "No estimators found. Please check the configuration file."
            )
        self.num_estimators = len(self.estimator_fns)

        if consensus_algorithm not in CONSENSUS_REGISTRY:
            raise ValueError(
                f"Invalid consensus algorithm: {consensus_algorithm}. "
                f"Available options are: {list(CONSENSUS_REGISTRY.keys())}."
            )
        self.consensus_fn = CONSENSUS_REGISTRY[consensus_algorithm]
        self.consensus_config = consensus_config or {}

        # Validate experimental parameters
        if experiment_params is not None:
            validated_params = validate_experimental_params(experiment_params)
            self.experiment_params: ExperimentParams = validated_params
        else:
            self.experiment_params = {}

        for key in self.experiment_params:
            logger.info(
                f"Experimental parameter: {key}={self.experiment_params[key]}"
            )

        super().__init__(**kwargs)

    def _create_estimators(
        self, configs: list[dict], rng: PRNGKey | None = None
    ) -> list[Callable]:
        """Create the list of estimators based on the configurations.

        Each estimator should specify a name and the required parameters.
        If the whole estimator wants to be jitted, each inner estimator
        should be jittable as well.

        Args:
            configs: List of configurations for each estimator
            rng: Random number generator key.

        Returns:
            List of estimator callables.
        """
        estimators: list[Callable] = []
        for cfg in configs:
            if rng is not None:
                rng, subkey = jax.random.split(rng)
            else:
                subkey = None
            (trainable_state, _, compute_estimate_fn, _) = make_estimator(
                model_config=cfg,
                load_from=cfg.get("load_from"),
                rng=subkey,
            )
            trainable_state = cast(EstimatorTrainableState, trainable_state)

            def estimator_fn(
                input, estimator=compute_estimate_fn, ts=trainable_state
            ) -> tuple[History, dict]:
                image, state = input
                state, metrics = estimator(image, state, ts)
                return state, metrics

            estimators.append(estimator_fn)
        return estimators

    def _estimate(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state: EstimatorTrainableState,
        extras: dict,
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the flow field between the two most recent frames.

        Args:
            images: Current batch of frames, shape (B, H, W).
            state: Contains keys "images" and "estimates",
                each with shape (B, T, H, W).
            trainable_state: Unused argument.
            extras: Additional information.

        Returns:
            Tuple containing flow field of shape (B, H, W, 2),
            placeholder dict for additional output, and metrics dict
            from the estimation process.

        Raises:
            ValueError: If oracle masking is requested without oracle=True.
        """
        # Get the most recent images from the state
        prev = state["images"][:, -1, ...]
        curr = images
        metrics = {}

        # Check if the state has a history of estimates
        if not self.use_temporal_propagation:
            # Use the last estimate from the history
            state["estimates"] = state["estimates"].at[:, -1, ...].set(
                jnp.zeros_like(state["estimates"][:, -1, ...])
            )

        # Prepare the input state for estimators
        input_state = state

        # Compute the flow fields using all available algorithms
        def single_estimator(idx: int):
            """Compute the flow field batch for a single estimator.

            Args:
                idx: Index of the estimator.

            Returns:
                state, metrics: The computed flow field for the estimator.
            """
            # Select the estimator based on the index
            new_state, metrics = lax.switch(
                idx, self.estimator_fns, (curr, input_state)
            )

            return new_state, metrics

        states, metrices = lax.map(
            single_estimator, jnp.arange(self.num_estimators)
        )

        # Extract the flow fields from the states
        flows = states["estimates"][
            :, :, -1, ...
        ]  # Shape (num_estimators, B, H, W, 2)
        flows = jnp.transpose(
            flows, (1, 0, 2, 3, 4)
        )  # Shape (B, num_estimators, H, W, 2)

        # Experiment 1: Oracle-based masking
        epe_limit = self.experiment_params.get("epe_limit", None)
        if epe_limit is not None:
            if not self.is_oracle():
                raise ValueError("Oracle masking requires oracle=True.")
            flow_gt = state["estimates"][:, -1, ...]
            # Compute the EPE for each estimator
            epe = jnp.linalg.norm(
                flows - flow_gt[:, jnp.newaxis, ...], axis=-1
            )  # Shape (B, num_estimators, H, W)
            mask = jnp.where(
                epe > epe_limit, 0.0, 1.0
            )  # Shape (B, num_estimators, H, W)
            metrics["oracle_mask"] = mask
        else:
            mask = None

        weights = make_weights(
            flows, prev, curr, self.consensus_config, mask=mask
        )

        # Apply the consensus function to combine the flow estimates
        def map_fn(args):
            flows_i, weights_i = args
            return self.consensus_fn(flows_i, weights_i, self.consensus_config)

        new_flow, consensus_metrics = jax.lax.map(map_fn, (flows, weights))

        for idx, met in enumerate(metrices):
            for key, value in met.items():
                metrics[f"estimator_{idx}_{key}"] = value
        for key, value in consensus_metrics.items():
            metrics[f"consensus_{key}"] = value

        # Check if any of the images is all zeros when in experimental mode
        if len(self.experiment_params) != 0:
            valid = jnp.logical_not(jnp.all(curr == 0, axis=(1, 2)))
            metrics["valid_images"] = valid
            epe = (
                jnp.linalg.norm(
                    new_flow - state["estimates"][:, -1, ...], axis=-1
                )
                * valid[:, jnp.newaxis, jnp.newaxis]
            )
            mean_epes = jnp.mean(epe, axis=(1, 2))
            metrics["epe"] = mean_epes

        return new_flow, {}, metrics

    def process_metrics(self, metrics: dict) -> Metrics:
        """Process and format metrics dictionary.

        Args:
            metrics: Raw metrics dictionary.

        Note:
            Assumption: only the last batch has invalid images.

        Returns:
            Processed metrics object.

        Raises:
            ValueError: If batch size cannot be inferred from metrics.
        """
        # Try to extract the batch size B from any array in metrics
        B = None
        for v in metrics.values():
            if isinstance(v, (np.ndarray, jnp.ndarray)) and v.ndim > 0:
                B = v.shape[0]
                break
        if B is None:
            raise ValueError("Could not infer batch size from metrics.")

        self.total_valid_images = (
            getattr(self, "total_valid_images", 0)
            + np.sum(metrics["valid_images"])
            if "valid_images" in metrics
            else B
        )

        # Convert the raw metrics dictionary into a structured Metrics object
        processed_metrics = Metrics()
        for key, value in metrics.items():
            if key != "valid_images":
                filtered_value = (
                    value[metrics["valid_images"]]
                    if "valid_images" in metrics
                    else value
                )
            else:
                filtered_value = value
            if key == "oracle_mask":
                # Calculate the coverage of the oracle mask
                coverage_map = jnp.any(
                    filtered_value > 0, axis=1
                )  # Shape (jnp.sum(valid), H, W)
                coverage = jnp.mean(
                    coverage_map, axis=(1, 2)
                )  # Shape (jnp.sum(valid),)

                # Update running mean coverage
                running_mean_coverage = getattr(
                    self, "running_mean_coverage", 0.0
                )
                running_mean_coverage = (
                    running_mean_coverage
                    * (self.total_valid_images - coverage.shape[0])
                    + jnp.sum(coverage)
                ) / (self.total_valid_images)
                self.running_mean_coverage = running_mean_coverage

                # Update running max coverage
                running_max_coverage = getattr(
                    self, "running_max_coverage", 0.0
                )
                running_max_coverage = max(
                    running_max_coverage, jnp.max(coverage)
                )
                self.running_max_coverage = running_max_coverage

                # Update running min coverage
                running_min_coverage = getattr(
                    self, "running_min_coverage", 0.0
                )
                running_min_coverage = min(
                    running_min_coverage, jnp.min(coverage)
                )
                self.running_min_coverage = running_min_coverage

                # Store the coverage in the processed metrics
                processed_metrics["oracle_mask_coverage"] = np.array(coverage)

            if key == "epe":
                # Update running mean EPE
                running_mean_epe = getattr(self, "running_mean_epe", 0.0)
                running_mean_epe = (
                    running_mean_epe
                    * (self.total_valid_images - filtered_value.shape[0])
                    + jnp.sum(filtered_value)
                ) / (self.total_valid_images)
                self.running_mean_epe = running_mean_epe

                # Update running max EPE
                running_max_epe = getattr(self, "running_max_epe", 0.0)
                running_max_epe = max(running_max_epe, jnp.max(filtered_value))
                self.running_max_epe = running_max_epe

                # Update running min EPE
                running_min_epe = getattr(self, "running_min_epe", 0.0)
                running_min_epe = min(running_min_epe, jnp.min(filtered_value))
                self.running_min_epe = running_min_epe
        return processed_metrics

    def finalize_metrics(self) -> Metrics:
        """Finalize metrics at the end of evaluation.

        Returns:
            Finalized metrics object.

        Raises:
            TypeError: If log_path is not a string or None.
            ValueError: If log_path doesn't end with .csv.
        """
        finalized_metrics = Metrics()
        if hasattr(self, "running_mean_coverage"):
            finalized_metrics["mean_oracle_mask_coverage"] = np.array(
                self.running_mean_coverage
            )
        if hasattr(self, "running_mean_epe"):
            finalized_metrics["mean_epe"] = np.array(self.running_mean_epe)

        log_path = self.experiment_params.get("log_path", None)
        if not isinstance(log_path, (type(None), str)):
            raise TypeError(
                f"log_path must be a string or None, got {log_path}."
            )
        if log_path is not None:
            if not log_path.endswith(".csv"):
                raise ValueError(
                    f"log_path must be a .csv file, got {log_path}."
                )
            # Define the row data (use getattr to avoid AttributeError)
            row_data = {
                "num_estimators": getattr(self, "num_estimators", None),
                "epe_limit": self.experiment_params.get("epe_limit", None),
                "mean_epe": getattr(self, "running_mean_epe", None),
                "max_epe": getattr(self, "running_max_epe", None),
                "min_epe": getattr(self, "running_min_epe", None),
                "mean_coverage": getattr(self, "running_mean_coverage", None),
                "max_coverage": getattr(self, "running_max_coverage", None),
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            # Check if the CSV already exists
            file_exists = os.path.exists(log_path)

            # Append or create CSV
            with open(log_path, mode="a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=row_data.keys())

                # Write header if creating new file
                if not file_exists:
                    writer.writeheader()

                # Convert JAX or NumPy scalars to plain Python floats
                clean_row = {
                    k: (float(v) if v is not None and hasattr(v, "item") else v)
                    for k, v in row_data.items()
                }

                writer.writerow(clean_row)

        return finalized_metrics
