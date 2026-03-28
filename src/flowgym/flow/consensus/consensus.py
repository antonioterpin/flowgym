"""ConsensusFlowEstimator class."""

import csv
import os
from collections.abc import Callable
from pathlib import Path
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
from flowgym.types import ExperimentParams
from flowgym.utils import append_metrics_to_csv, load_configuration

logger = get_logger(__name__, with_metrics=True)


class ConsensusFlowEstimator(FlowFieldEstimator):
    """Alternating Direction Method of Multipliers flow field estimator."""

    def __init__(
        self,
        consensus_algorithm: str,
        estimators_list_path: str,
        consensus_config: dict | None = None,
        use_temporal_propagation: bool = False,
        experiment_params: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the consensus estimator.

        Args:
            consensus_algorithm: Consensus algorithm to use.
            estimators_list_path: Path to the YAML file listing estimators.
            consensus_config: Optional configuration for the
                consensus algorithm.
            use_temporal_propagation: Whether to use temporal propagation.
            experiment_params: Optional experimental parameters.
            **kwargs: Additional keyword arguments forwarded to the base class.

        Raises:
            ValueError: If required configuration files or values are missing.
        """
        self.counter_logger = 0
        self.use_temporal_propagation = bool(use_temporal_propagation)

        # Load the DIS algorithms from the specified path
        if not estimators_list_path.endswith(".yaml"):
            raise ValueError(
                "estimators_list_path must be a .yaml file, "
                f"got {estimators_list_path}."
            )
        estimators_list_config = load_configuration(estimators_list_path)

        # The estimators should take as input the tuple (image, state)
        estimator_fns, inner_estimators_support_jit = self._create_estimators(
            estimators_list_config["estimators"]
        )
        self.estimator_fns = tuple(estimator_fns)
        self._inner_estimators_support_jit = inner_estimators_support_jit
        if not self._inner_estimators_support_jit:
            logger.warning(
                "ConsensusFlowEstimator contains non-jittable inner "
                "estimators; using Python-loop execution for sub-estimators."
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
                "Using experimental parameter: "
                f"{key} = {self.experiment_params[key]}"
            )

        super().__init__(**kwargs)

    def _create_estimators(
        self, configs: list[dict]
    ) -> tuple[list[Callable], bool]:
        """Create the list of estimators based on the configurations.

        Each estimator should specify a name and the required parameters.
        If the whole estimator wants to be jitted, each inner estimator
        should be jittable as well.

        Args:
            configs: List of configurations for each estimator

        Returns:
            List of estimator callables and whether all estimators support JIT.
        """
        # Import here to avoid circular dependency
        from flowgym.make import make_estimator  # noqa: PLC0415

        estimators: list[Callable] = []
        supports_jit = True
        for cfg in configs:
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
                # Some estimators may be non-trainable and return ``ts=None``.
                # Cast to ``EstimatorTrainableState`` purely to satisfy static
                # typing; the runtime call accepts ``None`` as needed.
                state, metrics = estimator(
                    image, state, cast(EstimatorTrainableState, ts)
                )
                return state, metrics

            estimators.append(estimator_fn)
        return estimators, supports_jit

    def supports_jit(self) -> bool:
        """Consensus is jittable only if all inner estimators are jittable."""
        return self._inner_estimators_support_jit

    def _build_weight_mask_from_estimator_metrics(
        self,
        metrics_per_estimator: list[dict[str, jnp.ndarray]],
        flows_shape: tuple[int, ...],
    ) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
        """Build consensus keep/reject masks from estimator outlier metrics.

        The expected estimator metric is ``postprocess_combined_rejected_mask``
        with shape ``(B, H, W)`` and boolean semantics where True means
        rejected/outlier.

        Args:
            metrics_per_estimator: Metrics emitted by each sub-estimator.
            flows_shape: Shape of the batched flow tensor ``(B, N, H, W, 2)``.

        Returns:
            Tuple ``(keep_mask, rejected_mask)`` with shape ``(B, N, H, W)``
            each, or ``(None, None)`` when no rejection mask is available.
        """
        if len(flows_shape) != 5:
            raise ValueError(
                f"flows_shape must be (B, N, H, W, 2), got {flows_shape}."
            )
        B, N, H, W, _ = flows_shape
        if N != self.num_estimators:
            raise ValueError(
                f"Expected {self.num_estimators} estimators, got {N}."
            )

        rejected_masks: list[jnp.ndarray] = []
        has_any_rejection_mask = False
        for idx in range(self.num_estimators):
            estimator_metrics = metrics_per_estimator[idx]
            rejected = estimator_metrics.get(
                "postprocess_combined_rejected_mask"
            )
            if rejected is None:
                rejected_masks.append(jnp.zeros((B, H, W), dtype=jnp.bool_))
                continue

            rejected = rejected.astype(jnp.bool_)
            if rejected.shape != (B, H, W):
                raise ValueError(
                    "postprocess_combined_rejected_mask must have shape "
                    f"(B, H, W)=({B}, {H}, {W}), got {rejected.shape} "
                    f"for estimator {idx}."
                )
            has_any_rejection_mask = True
            rejected_masks.append(rejected)

        if not has_any_rejection_mask:
            return None, None

        rejected_mask = jnp.stack(rejected_masks, axis=1)  # (B, N, H, W)
        keep_mask = jnp.logical_not(rejected_mask)
        return keep_mask, rejected_mask

    def _estimate(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state: EstimatorTrainableState,
        extras: dict,
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the flow field between the two most recent frames.

        Args:
            images: Current batch of frames.
            state: Estimator state containing history
                (images, estimates, keys, ...).
            trainable_state: Trainable model state (unused for this estimator).
            extras: Additional information passed to the estimator.

        Returns:
            A tuple of (flow, placeholder_dict, metrics) where ``flow`` is the
            estimated flow field for the batch, ``placeholder_dict`` is a
            placeholder for additional outputs, and ``metrics`` contains
            evaluation values collected during estimation.

        Raises:
            ValueError: If experimental configuration requires oracle mode but
                the estimator was not created as an oracle.
        """
        # Get the most recent images from the state
        prev = state["images"][:, -1, ...]
        curr = images
        metrics = {}
        experiment_params = self.experiment_params.copy()
        consensus_config = self.consensus_config.copy()

        # Check if the state has a history of estimates
        if not self.use_temporal_propagation:
            # Use the last estimate from the history
            state["estimates"][:, -1, ...] = jnp.zeros_like(
                state["estimates"][:, -1, ...]
            )

        # Prepare the input state for estimators
        input_state = dict(state)
        if "keys" in state:
            # Inner estimators expect keys with shape (B, 1, 2).
            keys = state["keys"]
            input_state["keys"] = (
                keys if keys.ndim == 3 else keys[:, jnp.newaxis, :]
            )

        metrics_per_estimator: list[dict[str, jnp.ndarray]]
        if self._inner_estimators_support_jit:
            # Compute the flow fields using all available algorithms.
            def single_estimator(idx: int):
                """Compute the flow field batch for a single estimator.

                Args:
                    idx: Index of the estimator.

                Returns:
                    Tuple of the estimator state and metrics for that estimator.
                """
                # Select the estimator based on the index
                new_state, est_metrics = lax.switch(
                    idx, self.estimator_fns, (curr, input_state)
                )
                return new_state, est_metrics

            states, metrices = lax.map(
                single_estimator, jnp.arange(self.num_estimators)
            )
            metrics_per_estimator = [
                {k: v[idx] for k, v in metrices.items()}
                for idx in range(self.num_estimators)
            ]
        else:
            # Non-jittable estimators (e.g., OpenCV baselines) must run in
            # eager Python to avoid tracing NumPy conversions.
            state_list = []
            metrics_per_estimator = []
            for estimator_fn in self.estimator_fns:
                new_state, est_metrics = estimator_fn((curr, input_state))
                state_list.append(new_state)
                metrics_per_estimator.append(est_metrics)
            states = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0), *state_list
            )

        # Extract the flow fields from the states
        flows = states["estimates"][
            :, :, -1, ...
        ]  # Shape (num_estimators, B, H, W, 2)
        flows = jnp.transpose(
            flows, (1, 0, 2, 3, 4)
        )  # Shape (B, num_estimators, H, W, 2)

        # Experimental: Oracle-based masking
        epe_limit = experiment_params.pop("epe_limit", None)
        if epe_limit is not None:
            if not self.is_oracle():
                raise ValueError(
                    "Oracle-based masking requires the estimator "
                    "to be an oracle. Set oracle=True when initializing "
                    "the ConsensusFlowEstimator."
                )
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

        # Non-oracle rejection mask coming from sub-estimator postprocessing.
        subestimator_keep_mask, subestimator_rejected_mask = (
            self._build_weight_mask_from_estimator_metrics(
                metrics_per_estimator, flows.shape
            )
        )
        if subestimator_keep_mask is not None:
            rejected_frac = jnp.mean(
                subestimator_rejected_mask.astype(jnp.float32), axis=(2, 3)
            )  # (B, N)
            for idx in range(self.num_estimators):
                metrics[f"estimator_{idx}_postprocess_rejected_percentage"] = (
                    rejected_frac[:, idx] * 100.0
                )

        # Combine all masks into one keep-mask used by weight computation.
        if mask is not None:
            oracle_keep_mask = mask > 0
            if subestimator_keep_mask is not None:
                mask = jnp.logical_and(oracle_keep_mask, subestimator_keep_mask)
            else:
                mask = oracle_keep_mask
        else:
            mask = subestimator_keep_mask

        weights = make_weights(
            flows, prev, curr, consensus_config, mask=mask
        )  # Shape (B, num_estimators, H, W)

        if len(self.experiment_params) != 0:
            # Only forward experiment parameters explicitly supported by
            # consensus algorithms to avoid passing unrelated metadata.
            forwarded_experiment_keys = {
                "log_metrics",
                "log_path",
                "baseline_performance",
                "oracle_select_weights",
            }
            for key, value in experiment_params.items():
                if key in forwarded_experiment_keys:
                    consensus_config["exp_" + key] = value

        # Experimental: Oracle-based selection of weights
        oracle_select_weights = experiment_params.pop(
            "oracle_select_weights", False
        )
        if oracle_select_weights:
            if not self.is_oracle():
                raise ValueError(
                    "Oracle-based weight selection requires the estimator "
                    "to be an oracle. Set oracle=True when initializing "
                    "the ConsensusFlowEstimator."
                )
            flow_gt = state["estimates"][:, -1, ...]
            # Compute the EPE for each estimator
            epe = jnp.linalg.norm(
                flows - flow_gt[:, jnp.newaxis, ...], axis=-1
            )  # Shape (B, num_estimators, H, W)
            # Get the indices of the best estimators based on EPE
            best_indices = jnp.argmin(epe, axis=1)  # Shape (B, H, W)

            # one-hot encode the best indices to create weights
            mask = jax.nn.one_hot(
                best_indices, num_classes=self.num_estimators
            )  # Shape (B, H, W, num_estimators)
            mask = jnp.transpose(
                mask, (0, 3, 1, 2)
            )  # Shape (B, num_estimators, H, W)
            weights *= mask

        # Apply the consensus function to combine the flow estimates
        def map_fn(args):
            flows_i, weights_i = args
            return self.consensus_fn(flows_i, weights_i, consensus_config)

        new_flow, consensus_metrics = jax.lax.map(map_fn, (flows, weights))

        for idx, met in enumerate(metrics_per_estimator):
            for key, value in met.items():
                if key == "postprocess_combined_rejected_mask":
                    # Internal helper for consensus weighting; avoid logging the
                    # full (B, H, W) mask in global metrics.
                    continue
                metrics[f"estimator_{idx}_{key}"] = value
        for key, value in consensus_metrics.items():
            metrics[f"consensus_{key}"] = value

        # Check if any of the images is all zeros when in experimental mode
        if self.is_oracle():
            valid = jnp.logical_not(jnp.all(curr == 0, axis=(1, 2)))
            metrics["valid_images"] = valid

            # Compute EPE metrics only for valid images
            gt = state["estimates"][:, -1, ...]
            diff = new_flow - gt
            epe = (
                jnp.linalg.norm(diff, axis=-1)
                * valid[:, jnp.newaxis, jnp.newaxis]
            )
            mean_epes = jnp.mean(epe, axis=(1, 2))
            metrics["epe"] = mean_epes

            # Compute relative error only for valid images
            num = jnp.linalg.norm(diff, axis=-1) ** 2
            den = jnp.maximum(jnp.linalg.norm(gt, axis=-1), 0.01) ** 2
            relative_error = num / den * valid[:, jnp.newaxis, jnp.newaxis]
            mean_relative_errors = jnp.mean(relative_error, axis=(1, 2))
            metrics["relative_error"] = mean_relative_errors

        return new_flow, {}, metrics

    def process_metrics(self, metrics: dict) -> Metrics:
        """Process and format metrics collected during evaluation.

        Converts raw numpy/jax arrays into numpy arrays and updates running
        statistics used by this estimator.

        Args:
            metrics: Raw metrics dictionary produced by ``_estimate``.

        Returns:
            A ``Metrics`` object with processed numeric arrays.

        Raises:
            TypeError: If ``log_metrics`` in experiment parameters is not a
                mapping as expected.
            ValueError: If the batch size cannot be inferred from the metrics.
        """
        # Try to extract the batch size B from any array in metrics
        B = None
        for v in metrics.values():
            if isinstance(v, (np.ndarray, jnp.ndarray)) and v.ndim > 0:
                B = v.shape[0]
                break
        if B is None and metrics != {}:
            raise ValueError("Could not infer batch size from metrics.")

        current_valid_count = (
            np.sum(metrics["valid_images"])
            if "valid_images" in metrics
            else (B if B is not None else 0)
        )
        self.total_valid_images = (
            getattr(self, "total_valid_images", 0) + current_valid_count
        )

        # Convert the raw metrics dictionary into a structured Metrics object
        processed_metrics = Metrics()
        for key, value in metrics.items():
            if key != "valid_images":
                v = (
                    value[metrics["valid_images"]]
                    if "valid_images" in metrics
                    else value
                )
                if isinstance(v, (np.ndarray, jnp.ndarray)):
                    processed_metrics[key] = np.array(v)
            if key == "oracle_mask":
                # Calculate the coverage of the oracle mask
                coverage_map = jnp.any(
                    value > 0, axis=1
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
                    * (self.total_valid_images - value.shape[0])
                    + jnp.sum(value)
                ) / (self.total_valid_images)
                self.running_mean_epe = running_mean_epe

                # Update running max EPE
                running_max_epe = getattr(self, "running_max_epe", 0.0)
                running_max_epe = max(running_max_epe, jnp.max(value))
                self.running_max_epe = running_max_epe

                # Update running min EPE
                running_min_epe = getattr(self, "running_min_epe", jnp.inf)
                running_min_epe = min(running_min_epe, jnp.min(value))
                self.running_min_epe = running_min_epe

            if key == "relative_error":
                # No running stats for relative error, just store the mean
                processed_metrics["mean_relative_error"] = np.array(
                    jnp.mean(value)
                )
                processed_metrics[key] = np.array(value)

                # Update running mean relative error
                running_mean_relative_error = getattr(
                    self, "running_mean_relative_error", 0.0
                )
                running_mean_relative_error = (
                    running_mean_relative_error
                    * (self.total_valid_images - value.shape[0])
                    + jnp.sum(value)
                ) / (self.total_valid_images)
                self.running_mean_relative_error = running_mean_relative_error

                # Update running max relative error
                running_max_relative_error = getattr(
                    self, "running_max_relative_error", 0.0
                )
                running_max_relative_error = max(
                    running_max_relative_error, jnp.max(value)
                )
                self.running_max_relative_error = running_max_relative_error

                # Update running min relative error
                running_min_relative_error = getattr(
                    self, "running_min_relative_error", jnp.inf
                )
                running_min_relative_error = min(
                    running_min_relative_error, jnp.min(value)
                )
                self.running_min_relative_error = running_min_relative_error
            if key.endswith("_rejected_percentage"):
                if isinstance(v, (np.ndarray, jnp.ndarray)):
                    rejected_values = np.asarray(v).reshape(-1)
                    finite = np.isfinite(rejected_values)
                    if np.any(finite):
                        rejected_sum = getattr(
                            self, "running_rejected_percentage_sum", {}
                        )
                        rejected_count = getattr(
                            self, "running_rejected_percentage_count", {}
                        )
                        rejected_sum[key] = rejected_sum.get(
                            key, 0.0
                        ) + float(np.sum(rejected_values[finite]))
                        rejected_count[key] = rejected_count.get(
                            key, 0
                        ) + int(np.sum(finite))
                        self.running_rejected_percentage_sum = rejected_sum
                        self.running_rejected_percentage_count = (
                            rejected_count
                        )
            if key in [
                "consensus_final_primal_residuals",
                "consensus_final_dual_residuals",
                "consensus_final_eps_pri",
                "consensus_final_eps_dual",
                "consensus_final_stopping_time",
            ]:
                processed_metrics[key] = np.array(value)

        log_metrics = self.experiment_params.get("log_metrics", {})
        csv_metrics = {}
        if not isinstance(log_metrics, dict):
            raise TypeError(f"log_metrics must be a dict, got {log_metrics}.")
        for metric_name, log_metric in log_metrics.items():
            if (
                log_metric
                and metric_name in metrics
                and metric_name != "consensus_stopping_time"
            ):
                csv_metrics[metric_name] = metrics[metric_name]

        if len(csv_metrics) > 0:
            i = getattr(self, "current_batch_index", 0)
            append_metrics_to_csv(
                csv_metrics, filename="admm_residuals_new_new.csv", batch_idx=i
            )

        if "consensus_stopping_time" in metrics:
            processed_metrics["mean_consensus_stopping_time"] = np.array(
                jnp.mean(metrics["consensus_stopping_time"])
            )
            stopping_time = np.asarray(
                metrics["consensus_stopping_time"]
            )  # (B,)

            if stopping_time.ndim != 1:
                raise ValueError(
                    "Expected consensus_stopping_time to be 1D (B,), "
                    f"got {stopping_time.shape}"
                )

            # Starting index for images in this batch
            idx: int = getattr(self, "current_batch_index", 0)
            if B is None:
                raise ValueError("Batch size must be specified.")
            start_img = idx * B

            # Global image indices: [start_img, ..., start_img + B - 1]
            img_idx = np.arange(start_img, start_img + B)

            cols = [
                img_idx,
                stopping_time,
            ]
            header = [
                "img_idx",
                "stopping_time",
            ]

            data = np.column_stack(cols)

            file_exists = Path("admm_stats_non_iterated.csv").exists()
            with open("admm_stats_non_iterated.csv", "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                writer.writerows(data)

        return processed_metrics

    def _get_estimator_rejected_percentages(self) -> dict[str, float]:
        """Get mean rejected percentages aggregated per estimator."""
        rejected_sum = getattr(self, "running_rejected_percentage_sum", {})
        rejected_count = getattr(
            self, "running_rejected_percentage_count", {}
        )
        if not rejected_sum or not rejected_count:
            return {}

        estimator_values: dict[int, list[float]] = {}
        for key, total in rejected_sum.items():
            count = rejected_count.get(key, 0)
            if count <= 0:
                continue

            parts = key.split("_", 2)
            if len(parts) < 3 or parts[0] != "estimator":
                continue

            try:
                estimator_idx = int(parts[1])
            except ValueError:
                continue

            estimator_values.setdefault(estimator_idx, []).append(total / count)

        return {
            f"estimator_{idx}_mean_rejected_percentage": float(np.mean(values))
            for idx, values in sorted(estimator_values.items())
            if len(values) > 0
        }

    def finalize_metrics(self) -> Metrics:
        """Finalize and optionally persist aggregated metrics.

        This method produces a final ``Metrics`` object and, when configured
        via ``experiment_params['log_path']``, appends a CSV row describing
        the final aggregated statistics.

        Returns:
            Metrics: Final aggregated metrics for the estimator.

        Raises:
            TypeError: If configuration values (for example, ``log_path`` or
                ``baseline_performance``) have incorrect types.
            ValueError: If ``log_path`` does not point to a CSV file.
        """
        finalized_metrics = Metrics()
        eval_summary = getattr(self, "_eval_summary_metrics", {})
        if not isinstance(eval_summary, dict):
            eval_summary = {}
        if hasattr(self, "running_mean_coverage"):
            finalized_metrics["mean_oracle_mask_coverage"] = np.array(
                self.running_mean_coverage
            )
        if hasattr(self, "running_mean_epe"):
            finalized_metrics["mean_epe"] = np.array(self.running_mean_epe)
        elif "mean_epe" in eval_summary:
            finalized_metrics["mean_epe"] = np.array(eval_summary["mean_epe"])
        estimator_rejected = self._get_estimator_rejected_percentages()
        for metric_name, metric_value in estimator_rejected.items():
            finalized_metrics[metric_name] = np.array(metric_value)

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
            regularizer_weights = self.consensus_config.get(
                "regularizer_weights", {}
            )
            if any(v > 0 for v in regularizer_weights.values()):
                regularization = "_".join(
                    f"{k}-{v}" for k, v in regularizer_weights.items() if v > 0
                )
            else:
                regularization = "none"
            row_data = {
                "comparison_profile": self.experiment_params.get(
                    "comparison_profile", None
                ),
                "comparison_tau": self.experiment_params.get(
                    "comparison_tau", None
                ),
                "comparison_checkpoint": self.experiment_params.get(
                    "comparison_checkpoint", None
                ),
                "num_estimators": getattr(self, "num_estimators", None),
                "transformation": getattr(self, "consensus_config", {}).get(
                    "transformation", None
                ),
                "regularization": regularization,
                "loss": getattr(self, "consensus_config", {}).get(
                    "flows_objective_type", None
                ),
                "weights_type": getattr(self, "consensus_config", {}).get(
                    "weights_type", None
                ),
                "epe_limit": self.experiment_params.get("epe_limit", None),
                "mean_epe": getattr(
                    self,
                    "running_mean_epe",
                    eval_summary.get("mean_epe", None),
                ),
                "max_epe": getattr(
                    self,
                    "running_max_epe",
                    eval_summary.get("max_epe", None),
                ),
                "min_epe": getattr(
                    self,
                    "running_min_epe",
                    eval_summary.get("min_epe", None),
                ),
                "mean_coverage": getattr(self, "running_mean_coverage", None),
                "max_coverage": getattr(self, "running_max_coverage", None),
                "min_coverage": getattr(self, "running_min_coverage", None),
                "mean_relative_error": getattr(
                    self,
                    "running_mean_relative_error",
                    eval_summary.get("mean_relative_error", None),
                ),
                "max_relative_error": getattr(
                    self,
                    "running_max_relative_error",
                    eval_summary.get("max_relative_error", None),
                ),
                "min_relative_error": getattr(
                    self,
                    "running_min_relative_error",
                    eval_summary.get("min_relative_error", None),
                ),
            }
            row_data.update(estimator_rejected)

            if "baseline_performance" in self.experiment_params:
                baseline = self.experiment_params["baseline_performance"]
                if not isinstance(baseline, dict):
                    raise TypeError(
                        f"baseline_performance must be a dict, got {baseline}."
                    )
                baseline_mean = baseline.get("mean_epe", None)
                row_mean = row_data.get("mean_epe", None)
                if isinstance(baseline_mean, float) and isinstance(
                    row_mean, float
                ):
                    row_data["relative_mean_epe"] = (
                        row_mean - baseline_mean
                    ) / baseline_mean

                baseline_max = baseline.get("max_epe", None)
                row_max = row_data.get("max_epe", None)
                if isinstance(baseline_max, float) and isinstance(
                    row_max, float
                ):
                    row_data["relative_max_epe"] = (
                        row_max - baseline_max
                    ) / baseline_max

                baseline_min = baseline.get("min_epe", None)
                row_min = row_data.get("min_epe", None)
                if isinstance(baseline_min, float) and isinstance(
                    row_min, float
                ):
                    row_data["relative_min_epe"] = (
                        row_min - baseline_min
                    ) / baseline_min

                baseline_min_rel = baseline.get("min_relative_epe", None)
                row_min_rel = row_data.get("min_relative_error", None)
                if isinstance(baseline_min_rel, float) and isinstance(
                    row_min_rel, float
                ):
                    row_data["relative_min_relative_epe"] = (
                        row_min_rel - baseline_min_rel
                    ) / baseline_min_rel

                baseline_mean_rel = baseline.get("mean_relative_epe", None)
                row_mean_rel = row_data.get("mean_relative_error", None)
                if isinstance(baseline_mean_rel, float) and isinstance(
                    row_mean_rel, float
                ):
                    row_data["relative_mean_relative_epe"] = (
                        row_mean_rel - baseline_mean_rel
                    ) / baseline_mean_rel

                baseline_max_rel = baseline.get("max_relative_epe", None)
                row_max_rel = row_data.get("max_relative_error", None)
                if isinstance(baseline_max_rel, float) and isinstance(
                    row_max_rel, float
                ):
                    row_data["relative_max_relative_epe"] = (
                        row_max_rel - baseline_max_rel
                    ) / baseline_max_rel

            # Ensure directory exists (handle top-level files safely)
            dir_name = os.path.dirname(log_path)
            assert dir_name, f"Invalid log_path with no directory: {log_path}"
            os.makedirs(dir_name, exist_ok=True)

            # Check if the CSV already exists
            file_exists = os.path.exists(log_path)

            # Append or create CSV
            with open(log_path, mode="a", newline="") as csvfile:
                # Ensure stable ordering for fieldnames by converting to a list
                fieldnames = list(row_data.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

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
