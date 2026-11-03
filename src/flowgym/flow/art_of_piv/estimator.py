"""ART of PIV Estimator class."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import goggles as gg
import jax
import jax.numpy as jnp
import numpy as np
from goggles import Metrics
from goggles.history.spec import HistoryFieldSpec, HistorySpec
from goggles.history.types import History

from flowgym.common.base import (
    EstimatorTrainableState,
    NNEstimatorTrainableState,
)
from flowgym.common.preprocess import build_preprocessing_steps
from flowgym.flow.art_of_piv import caching as art_of_piv_caching
from flowgym.flow.art_of_piv import inference as art_of_piv_inference
from flowgym.flow.art_of_piv import router as art_of_piv_router
from flowgym.flow.art_of_piv import training_steps as art_of_piv_training
from flowgym.flow.art_of_piv import validation as art_of_piv_validation
from flowgym.flow.art_of_piv.metrics import (
    ChoiceTracker,
    process_art_of_piv_metrics,
)
from flowgym.flow.art_of_piv.types import EstimateFn, RouterOutputs
from flowgym.flow.base import FlowFieldEstimator
from flowgym.nn.blocks import PostProcess
from flowgym.nn.cnn import build_model_from_config
from flowgym.training import (
    build_exploration_from_config,
    build_loss_from_config,
    build_schedule_from_config,
    build_target_transform_from_config,
)
from flowgym.types import (
    PRNGKey,
    RLTrainStep,
    SupervisedExperience,
    SupervisedTrainStep,
    TrainStep,
)

if TYPE_CHECKING:
    from synthpix.types import SynthpixBatch

    from flowgym.training.caching import CacheManager

logger = gg.get_logger(__name__)


class ArtOfPIVEstimator(FlowFieldEstimator):
    """ART of PIV flow field estimator.

    Attributes:
        train_type: Training paradigm ("supervised" or "rl").
        training_mode: Supervised mode ("bandits", "regression",
            or "classification").
        num_estimators: Number of configured sub-estimators.
        estimator_fns: Callable tuple used to execute sub-estimators.
        sub_models: Instantiated sub-estimator model objects.
        sub_model_states: Trainable state tuple aligned with ``sub_models``.
        target_metric: Sub-estimator cache field the router minimises.
        router_preprocessing_steps: Steps applied to the frames the router
            reads, and not to those of the sub-estimators.
        require_cached_labels: Whether a frame missing from a sub-estimator's
            cache is an error instead of being computed.
    """

    train_type: str
    training_mode: str
    num_estimators: int
    estimator_fns: tuple[EstimateFn, ...]
    sub_models: tuple[FlowFieldEstimator, ...]
    sub_model_states: tuple[EstimatorTrainableState, ...]
    target_metric: str = "epe"
    router_preprocessing_steps: Sequence[
        Callable[[jnp.ndarray], jnp.ndarray]
    ] = ()
    require_cached_labels: bool = False

    def __init__(
        self,
        # Required parameters
        router_config: dict,
        estimators_list_path: str,
        # Feature flags & behavior
        use_temporal_propagation: bool = False,
        output_full_res: bool = True,
        use_last_estimate: bool = False,
        # Training configuration
        train_type: Literal["supervised", "rl"] = "supervised",
        training_mode: Literal["bandits", "regression", "classification"] = (
            "bandits"
        ),
        loss_config: dict | None = None,
        gamma: float = 0,
        target_transform_config: dict | None = None,
        target_metric: Literal["epe", "relative_epe", "corr16_peak"] = "epe",
        router_preprocess: Sequence[dict] | None = None,
        require_cached_labels: bool = False,
        # Exploration strategy
        exploration_config: dict | None = None,
        # Postprocessing
        postprocess: PostProcess | Sequence[str] = (),
        # Initialization & state
        image_shape: tuple[int, int, int] | None = None,
        rng: PRNGKey | int | None = None,
        initialize_estimators: bool = True,
        # Base class arguments
        **kwargs: Any,
    ) -> None:
        """Initialize the ART of PIV estimator.

        Args:
            router_config: Router architecture configuration dict. Two
                formats are supported and auto-detected:

                **Nested format (recommended)** — shared CNN encoder plus
                per-arm MLP heads (``CNNRouterModel``):

                .. code-block:: yaml

                    encoder:
                      features_list: [16, 32, 64]   # required
                      output_dim: 64                # optional, default 64
                      use_residual: false           # optional, default False
                      norm_fn: none                 # optional, default "none"
                    head:
                      hidden_dims: [32, 16]         # required
                      activation: relu              # optional, default "relu"
                      single_arm: false             # optional, default False

                **Flat format (legacy)** — single CNN trunk + MLP head
                (``CNNQEstimatorModel``):

                - ``features`` (required): List of feature maps
                - ``kernel_sizes`` (required): List of kernel sizes
                - ``strides`` (required): List of strides
                - ``norm_fn`` (optional): Normalization function, one of
                  ``"none"``, ``"batch"``, ``"instance"``, ``"group"``
                  (default: ``"none"``)
                - ``mlp_hidden_dim`` (optional): MLP hidden dimension
                  (default: 512)
            estimators_list_path: Path to the YAML file listing
                sub-estimators.
            use_temporal_propagation: Whether to propagate estimates across
                temporal steps.
            output_full_res: Whether to output full resolution flow fields.
            use_last_estimate: Whether to use the previous estimate as
                additional network input.
            train_type: Training paradigm - "supervised" (default) for
                multi-armed bandit learning or "rl" for full reinforcement
                learning.
            training_mode: Supervised training mode - "bandits" (Q-learning
                with rewards), "regression" (predict Q-values directly), or
                "classification" (predict best action).
            loss_config: Loss function configuration.
                Defaults to {"name": "l2"}.
            gamma: Discount factor for future rewards in RL mode.
                Must be in [0, 1).
            target_transform_config: Configuration for transforming
                target values before loss computation.
            target_metric: Sub-estimator cache field the router learns to
                minimise. ``corr16_peak`` needs no ground truth, so it can
                label real images; it requires supervised training from
                cached labels.
            router_preprocess: Preprocessing steps applied to the frames the
                router reads, in addition to the estimator-level ``preprocess``.
                The sub-estimators keep receiving the full frames, so the
                router can run on a crop or a downsampled copy. It applies
                wherever the router reads frames, including training: leave it
                unset when training on frames that are already preprocessed.
                Not supported with ``use_last_estimate``, whose flow input
                has the full-frame size.
            require_cached_labels: Raise when a frame is missing from a
                sub-estimator's cache instead of running the sub-estimator on
                it. For labels written into the caches (e.g. by
                ``scripts/build_label_dataset.py``), whose frames carry a
                placeholder flow that a computed error would be scored on.
            exploration_config: Exploration strategy configuration
                (e.g., epsilon-greedy with schedule).
            postprocess: Sequence of post-processing functions to apply
                to Q-values (e.g., softmax normalization).
            image_shape: Shape of input images (B, H, W) for initialization.
            rng: Random number generator key or integer seed.
            initialize_estimators: Whether to construct sub-estimators eagerly
                during initialization.
            **kwargs: Additional keyword arguments passed to base
                FlowFieldEstimator.

        Raises:
            ValueError: If ``target_metric`` is unknown, or is not EPE with
                RL training.
        """
        # Set default loss config
        if loss_config is None:
            loss_config = {"name": "l2"}

        # VALIDATION PHASE

        # Validate boolean control flags
        (
            self.output_full_res,
            self.use_temporal_propagation,
            self.use_last_estimate,
        ) = art_of_piv_validation.validate_boolean_flags(
            output_full_res, use_temporal_propagation, use_last_estimate
        )

        # Validate train type
        self.train_type = art_of_piv_validation.validate_train_type(train_type)
        if target_metric not in ("epe", "relative_epe", "corr16_peak"):
            raise ValueError(f"Unknown target_metric: {target_metric!r}")
        if target_metric != "epe" and self.train_type != "supervised":
            raise ValueError(
                f"target_metric={target_metric!r} needs supervised training."
            )
        self.target_metric = target_metric
        self.require_cached_labels = require_cached_labels
        if router_preprocess and self.use_last_estimate:
            raise ValueError(
                "router_preprocess is not supported with use_last_estimate."
            )
        self.router_preprocessing_steps = build_preprocessing_steps(
            router_preprocess
        )
        # clahe runs on the host (NumPy) and cannot be traced.
        self._router_preprocess_jittable = not any(
            step["name"] == "clahe" for step in router_preprocess or []
        )

        # Validate and load estimators configuration
        estimators_list_config = (
            art_of_piv_validation.validate_and_load_estimators_config(
                estimators_list_path
            )
        )

        # Validate and convert RNG
        self._rng = art_of_piv_validation.validate_rng(rng)

        # Extract estimator configs and set basic attributes
        self.estimator_configs = estimators_list_config["estimators"]
        self._cache_id_base: str | None = estimators_list_config.get(
            "cache_id_base"
        )
        self.image_shape = image_shape

        # Initialize estimator and caching containers.
        self.num_estimators = len(self.estimator_configs)
        self._sub_cache_managers = {}
        self._unified_epe_index = {}
        self._cache_bootstrapped = False
        self._cache_root_dir: str | None = None
        self._allow_create_missing_sub_caches = False

        self._init_router(
            router_config,
            training_mode=training_mode,
            loss_config=loss_config,
            gamma=gamma,
            target_transform_config=target_transform_config,
            exploration_config=exploration_config,
            postprocess=postprocess,
        )

        if initialize_estimators:
            # Eagerly initialize sub-estimators for clearer object contracts.
            (
                self.estimator_fns,
                self.sub_models,
                self.sub_model_states,
            ) = self._create_estimators(
                self.estimator_configs,
                image_shape=self.image_shape,
                rng=self._rng,
            )
        else:
            self.estimator_fns = ()
            self.sub_models = ()
            self.sub_model_states = ()

        # Call parent constructor
        super().__init__(**kwargs)

        self._choice_tracker = ChoiceTracker(
            num_estimators=self.num_estimators,
            window_size=getattr(self, "choices_window_size", 100),
        )

    def _init_router(
        self,
        router_config: dict,
        *,
        training_mode: Literal["bandits", "regression", "classification"],
        loss_config: dict,
        gamma: float,
        target_transform_config: dict | None,
        exploration_config: dict | None,
        postprocess: PostProcess | Sequence[str],
    ) -> None:
        """Validate and build the router network, its loss and exploration.

        Args:
            router_config: Router architecture configuration.
            training_mode: Supervised training mode.
            loss_config: Loss configuration.
            gamma: Discount factor.
            target_transform_config: Target transform configuration.
            exploration_config: Exploration strategy configuration.
            postprocess: Postprocessing applied to the router output.
        """
        router_config = art_of_piv_validation.normalize_router_config(
            router_config
        )

        # Validate training mode and loss compatibility
        self.training_mode = (
            art_of_piv_validation.validate_training_mode_and_loss(
                training_mode, loss_config
            )
        )

        # Log warnings for training mode / exploration compatibility
        art_of_piv_validation.validate_training_mode_exploration_compatibility(
            training_mode, exploration_config
        )

        # Validate and build postprocessing pipeline
        postprocessing = art_of_piv_validation.validate_and_build_postprocess(
            postprocess
        )

        # Validate and set exploration config with defaults
        exploration_config = (
            art_of_piv_validation.validate_and_set_exploration_defaults(
                exploration_config, training_mode
            )
        )

        # Validate gamma
        self.gamma = art_of_piv_validation.validate_gamma(gamma)

        # Build exploration and schedule
        self.explore = build_exploration_from_config(exploration_config)
        self.schedule = build_schedule_from_config(
            exploration_config.get("schedule", {})
        )

        self.model = build_model_from_config(
            {
                **router_config,
                "num_q_values": self.num_estimators,
                "postprocess": postprocessing,
            }
        )

        # Build loss and target transform
        self.loss_fn = build_loss_from_config(loss_config)
        self.exploration_config = exploration_config
        self.target_transform_config = target_transform_config
        self.target_transform = build_target_transform_from_config(
            target_transform_config
        )

    def _create_estimators(
        self,
        configs: list[dict[str, Any]],
        image_shape: tuple[int, int, int] | None = None,
        rng: PRNGKey | None = None,
    ) -> tuple[
        tuple[EstimateFn, ...],
        tuple[FlowFieldEstimator, ...],
        tuple[EstimatorTrainableState, ...],
    ]:
        """Create the list of estimators based on the configurations.

        Each estimator should specify a name and the required parameters.
        If the whole estimator wants to be jitted, each inner estimator
        should be jittable as well.

        Args:
            configs: List of configurations for each estimator
            image_shape: Shape of the input images (B, H, W).
            rng: Random number generator key.

        Returns:
            estimator_fns:
                Tuple of callables that compute estimates for each
                sub-estimator.
            sub_models:
                Tuple of the instantiated sub-models.
            sub_model_states:
                Tuple of the initial trainable states for each sub-model.

        Raises:
            TypeError: If a constructed sub-estimator model has an unexpected
                type.
        """
        # Import here to avoid circular imports
        from flowgym.make import make_estimator  # noqa: PLC0415

        estimators: list[EstimateFn] = []
        models: list[FlowFieldEstimator] = []
        states: list[EstimatorTrainableState] = []
        for cfg in configs:
            if rng is not None:
                rng, subkey = jax.random.split(rng)
            else:
                subkey = None
            (trainable_state, _, compute_estimate_fn, model) = make_estimator(
                estimator_config=cfg,
                load_from=str(cfg.get("load_from"))
                if cfg.get("load_from") is not None
                else None,
                image_shape=image_shape,
                rng=subkey,
            )
            assert trainable_state is not None

            estimators.append(compute_estimate_fn)
            if not isinstance(model, FlowFieldEstimator):
                raise TypeError(
                    "Sub-estimator model must be a FlowFieldEstimator, got "
                    f"{type(model)}."
                )
            models.append(model)
            states.append(trainable_state)
        return tuple(estimators), tuple(models), tuple(states)

    def supports_jit(self) -> bool:
        """Whether the estimator can be JIT-compiled.

        Returns:
            False when a router preprocessing step runs on the host.
        """
        return self._router_preprocess_jittable

    def cache_spec(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        """One cached EPE (and its validity mask) per library member.

        Returns:
            ``epe_all`` and ``epe_mask``, each of shape ``(num_estimators,)``.
        """
        return {
            "epe_all": ("float32", (self.num_estimators,)),
            "epe_mask": ("bool", (self.num_estimators,)),
        }

    def _cache_id_suffix(
        self,
        trainable_state: NNEstimatorTrainableState,
    ) -> str:
        """Get a suffix for the cache ID based on the sub-models.

        Delegates to each sub-estimator's get_cache_id_suffix method to
        ensure proper encapsulation - each estimator knows which of its
        own parameters affect output vs which are optimization-only.

        Args:
            trainable_state: The current trainable state.

        Returns:
            A string hash of the sub-models' cache IDs.
        """
        # Aggregate cache ID suffixes from all sub-models
        suffixes = []
        for sub_model, sub_state in zip(
            self.sub_models, self.sub_model_states, strict=True
        ):
            # Each estimator knows how to compute its own cache ID
            suffix = sub_model.get_cache_id_suffix(sub_state)
            suffixes.append(suffix)

        # Combine all suffixes into a single hash; the target metric joins it
        # only when it is not EPE, so existing router caches keep their id.
        if self.target_metric != "epe":
            suffixes.append(self.target_metric)
        combined = "".join(suffixes)
        h = hashlib.md5(combined.encode("utf-8")).hexdigest()
        return f"_art_of_piv{h[:8]}"

    def preload_caches(
        self,
        root_dir: str,
        warm_start: Literal["all", "index", "none"] = "all",
    ) -> None:
        """Load all sub-model caches into memory for fast aggregation.

        Delegates to :func:`caching.preload_caches` and
        :func:`caching.build_unified_index`.

        Args:
            root_dir: Root directory containing sub-model caches.
            warm_start: Warm start mode for sub-model caches
                ('all' recommended).
        """
        art_of_piv_caching.preload_caches(
            root_dir=root_dir,
            warm_start=warm_start,
            sub_models=self.sub_models,
            sub_model_states=self.sub_model_states,
            estimator_configs=self.estimator_configs,
            sub_cache_managers=self._sub_cache_managers,
            cache_id_base=self._cache_id_base,
        )

        # Mark cache bootstrap context for cold-start lazy creation/write-back.
        self._cache_bootstrapped = True
        self._cache_root_dir = root_dir
        self._allow_create_missing_sub_caches = True

        # Build unified index for O(1) lookups.
        # Normalize None -> {} so enrich can run in cold-start mode.
        self._build_unified_index()

    def _build_unified_index(self) -> None:
        """Build a unified hash-based index for O(1) lookups.

        Delegates to :func:`caching.build_unified_index`.
        """
        self._unified_epe_index = (
            art_of_piv_caching.build_unified_index(
                sub_models=self.sub_models,
                sub_model_states=self.sub_model_states,
                sub_cache_managers=self._sub_cache_managers,
                cache_id_base=self._cache_id_base,
                target_metric=self.target_metric,
            )
            or {}
        )

    def enrich(
        self,
        batch: SynthpixBatch,
        miss_idxs: np.ndarray,
        trainable_state: EstimatorTrainableState | None = None,
        cache_manager: CacheManager | None = None,
    ) -> dict[str, Any] | None:
        """Compute the cache payload for missing keys.

        Uses the unified index for O(1) lookups. If called through
        `enrich_batch(...)`, cache context is provided to bootstrap
        ART of PIV on cold-start automatically.

        Args:
            batch: The batch of data from the sampler.
            miss_idxs: Indices of samples in the batch that are missing
                from cache.
            trainable_state: Unused placeholder for API compatibility with the
                estimator enrich interface.
            cache_manager: Optional cache manager from enrich_batch used to
                bootstrap sub-cache preload and lazy creation context.

        Returns:
            Dictionary with "epe_all" and "epe_mask" for the missing indices.

        Raises:
            ValueError: If ``require_cached_labels`` and a frame is missing
                from a sub-estimator's cache.
        """
        if not self._cache_bootstrapped and cache_manager is not None:
            self.preload_caches(
                root_dir=str(cache_manager.root_dir),
                warm_start=cache_manager.warm_start,
            )

        # Get EPEs for entire batch from unified index
        payload = art_of_piv_caching.enrich_from_sub_caches(
            batch=batch,
            compute_misses=not self.require_cached_labels,
            sub_models=self.sub_models,
            sub_model_states=self.sub_model_states,
            sub_cache_managers=self._sub_cache_managers,
            unified_epe_index=self._unified_epe_index,
            root_dir=self._cache_root_dir,
            allow_create_missing=self._allow_create_missing_sub_caches,
            cache_id_base=self._cache_id_base,
            target_metric=self.target_metric,
        )
        # Only the misses to fill: padding entries of a partial batch have no
        # labels and are not among them
        uncached = miss_idxs[~payload["epe_mask"][miss_idxs].all(axis=1)]
        if self.require_cached_labels and uncached.size:
            files = getattr(batch, "files", None)
            missing = [files[i] if files else i for i in uncached]
            raise ValueError(
                f"Frames not in the label caches: {missing}. Check that the "
                "cache root_dir and the library match the built dataset."
            )
        # Filter to miss_idxs only (enrich_batch expects data for misses)
        return {k: v[miss_idxs] for k, v in payload.items()}

    def create_trainable_state(
        self,
        dummy_input: jnp.ndarray,
        key: PRNGKey,
    ) -> NNEstimatorTrainableState:
        """Create the initial trainable state of the flow field estimator.

        This method initializes the model parameters and the optimizer state.
        Two possible input shapes for the network are supported:
            1. If `use_last_estimate` is True, the input shape is (B, H, W, 4)
               where the last two channels are the flow field estimated at the
               previous timestep (flow_x, flow_y).
            2. If `use_last_estimate` is False, the input shape is (B, H, W, 2)
               containing only the two images (prev, curr).

        Args:
            dummy_input: Batched dummy input to initialize the state.
            key: Random key for initialization.

        Returns:
            NNEstimatorTrainableState:
                The initial trainable state of the estimator.

        Raises:
            ValueError: If dummy_input shape is invalid.
        """
        # Validate the dummy input shape
        if dummy_input.ndim != 3:
            raise ValueError(
                f"Dummy input must have 3 dimensions (B, H, W), "
                f"got {dummy_input.ndim}."
            )

        dummy_input = self._preprocess_for_router(dummy_input)

        if self.use_last_estimate:
            # Input is two images and a flow field
            dummy_input = jnp.tile(
                dummy_input[..., None],
                (1, 1, 1, 4),  # prev, curr, flow_x, flow_y
            )  # Shape: (B, H, W, 4)
        else:
            # Input is only the two images
            dummy_input = jnp.tile(
                dummy_input[..., None],
                (1, 1, 1, 2),  # prev, curr
            )  # Shape: (B, H, W, 2)

        # Note: to create trainable state in flax, batch size is specific to
        # the training, the network is evaluated in parallel on a single
        # image pair (+ optional flow).
        params = self.model.init(key, dummy_input)["params"]

        extras = {
            # MAB/RL steps, different from optax steps
            "steps": jnp.zeros((dummy_input.shape[0], 1), dtype=jnp.int32)
        }

        return NNEstimatorTrainableState.from_config(
            apply_fn=self.model.apply,
            params=params,
            optimizer_config=self.optimizer_config,
            extras=extras,
        )

    def prepare_experience_for_replay(
        self,
        experience: SupervisedExperience,
        trainable_state: NNEstimatorTrainableState,
    ) -> SupervisedExperience:
        """Enrich experience with computed estimates before storing in
        replay buffer.

        For ART of PIV, the initial state in the experience usually contains
        zeros as estimates. By running the inference here (on the training
        device), we can store much better states in the replay buffer.

        Args:
            experience: The experience object with state, obs, ground_truth.
            trainable_state: The current trainable state.

        Returns:
            The prepared experience with enriched state.

        Raises:
            ValueError: If ``target_metric`` is not EPE: replayed experiences
                drop their cache payload, the only source of such labels.
        """
        if self.target_metric != "epe":
            raise ValueError(
                f"target_metric={self.target_metric!r} does not support "
                "replay (cached labels are not replayed); set "
                "replay_buffer_capacity to 0."
            )
        return art_of_piv_training.prepare_experience_for_replay(
            experience=experience,
            trainable_state=trainable_state,
            estimate_fn=self,
        )

    def create_train_step(self) -> TrainStep:
        """Create a training step function for the estimator.

        Routes to either supervised (MAB) or reinforcement learning training
        based on the `train_type` attribute set during initialization.

        Returns:
            Training step function conforming to TrainStep.
        """
        return art_of_piv_training.create_train_step(
            train_type=cast(Literal["supervised", "rl"], self.train_type),
            create_supervised_fn=self.create_supervised_train_step,
            create_rl_fn=self.create_rl_train_step,
        )

    def create_supervised_train_step(self) -> SupervisedTrainStep:
        """Create a supervised training step for multi-armed bandit mode.

        In this mode, the reward is computed internally as the negative EPE
        between the chosen estimator's output and the ground truth flow.

        Returns:
            SupervisedTrainStep: The supervised training step function.
        """
        return art_of_piv_training.create_supervised_train_step(
            q_estimate_fn=self.q_estimate,
            estimate_fn=self,  # self is callable (__call__)
            run_router_fn=self._run_router,
            loss_fn=self.loss_fn,
            target_transform=self.target_transform,
            schedule=self.schedule,
            training_mode=cast(
                Literal["bandits", "regression", "classification"],
                self.training_mode,
            ),
            use_temporal_propagation=self.use_temporal_propagation,
            num_estimators=self.num_estimators,
            estimator_configs=self.estimator_configs,
            estimator_fns=self.estimator_fns,
            estimator_states=self.sub_model_states,
            target_metric=self.target_metric,
        )

    def create_rl_train_step(self) -> RLTrainStep:
        """Create a reinforcement learning training step.

        In this mode, reward and temporal information come from the environment.

        Returns:
            RLTrainStep: The RL training step function.
        """
        return art_of_piv_training.create_rl_train_step(
            q_estimate_fn=self.q_estimate,
            loss_fn=self.loss_fn,
            use_temporal_propagation=self.use_temporal_propagation,
            gamma=self.gamma,
        )

    def _run_router(
        self,
        prev: jnp.ndarray,
        curr: jnp.ndarray,
        state: History,
        trainable_state: NNEstimatorTrainableState,
        flow: jnp.ndarray | None = None,
        train: bool = True,
    ) -> RouterOutputs:
        """Run router to get actions from Q-values and exploration.

        Args:
            prev: Previous frame of shape (B, H, W).
            curr: Current frame of shape (B, H, W).
            state: Current state of the model (must contain "keys").
            trainable_state: Current trainable state of the model (must
                contain extras["steps"]).
            flow: Optional last flow estimate.
            train: Whether the model is training.

        Returns:
            actions: Chosen actions of shape (B,).
            qs: Q-values of shape (B, num_estimators).
            exploration_val: Exploration values of shape (B,).
            estimate_keys:
                Keys corresponding to the chosen estimates, shape (B, 2).
        """

        # Create a closure that captures trainable_state for q_estimate
        def q_estimate_with_state(prev, curr, flow=None):
            return self.q_estimate(prev, curr, trainable_state, flow=flow)

        return art_of_piv_router.run_router(
            prev,
            curr,
            keys=state["keys"],
            steps=jnp.asarray(trainable_state.extras["steps"]),
            q_estimate_fn=q_estimate_with_state,
            schedule=self.schedule,
            explore=self.explore,
            flow=flow,
            train=train,
        )

    def _preprocess_for_router(self, frames: jnp.ndarray) -> jnp.ndarray:
        """Apply the router-only preprocessing steps.

        Args:
            frames: Batch of frames of shape (B, H, W).

        Returns:
            The frames as the router reads them.
        """
        for step in self.router_preprocessing_steps:
            frames = step(frames)
        return frames

    def q_estimate(
        self,
        prev: jnp.ndarray,
        curr: jnp.ndarray,
        trainable_state: NNEstimatorTrainableState,
        flow: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Estimate the q-value for the current state.

        NOTE: This is a separate method from the router's q_estimate to allow
        it to be called with the trainable_state for proper parameter access.
        The router's q_estimate is a closure that captures the trainable_state
        and calls this method.

        Args:
            prev: Previous frame of shape (B, H, W).
            curr: Current frame of shape (B, H, W).
            trainable_state: The current state of the model.
            flow: Flow field of shape (B, H, W, 2).

        Returns:
            jnp.ndarray: Q-values of shape (B, num_estimators).
        """
        prev = self._preprocess_for_router(prev)
        curr = self._preprocess_for_router(curr)
        return art_of_piv_router.q_estimate(
            prev,
            curr,
            trainable_state,
            model_apply_fn=self.model.apply,
            use_last_estimate=self.use_last_estimate,
            flow=flow,
        )

    def _estimate(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state: NNEstimatorTrainableState,
        extras: dict,
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the flow field between the two most recent frames.

        Delegates to :func:`~flowgym.flow.art_of_piv.inference.estimate`.

        Args:
            images: Current batch of frames, shape (B, H, W).
            state: Contains "images" (B, T, H, W), "estimates"
                (B, T, H, W, 2), "keys" (B, 2).
            trainable_state: The current state of the model.
            extras: Additional information for the estimation.

        Returns:
            Tuple of (flow, out_extras, metrics).
        """
        return art_of_piv_inference.estimate(
            images,
            state,
            trainable_state,
            extras,
            use_temporal_propagation=self.use_temporal_propagation,
            use_last_estimate=self.use_last_estimate,
            run_router_fn=self._run_router,
            run_subestimators_fn=partial(
                art_of_piv_inference.run_subestimators,
                estimator_fns=self.estimator_fns,
                estimator_states=self.sub_model_states,
            ),
            cache_holds_epe=self.target_metric == "epe",
        )

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

        Delegates to :func:`process_art_of_piv_metrics` from the metrics module,
        lazily initializing the :class:`ChoiceTracker` on first call.

        Args:
            metrics: Raw metrics dictionary.
            flow_field: Final flow field estimate (unused by this override,
                kept for signature compatibility).
            flow_field_gt: Ground-truth flow field (unused by this override).

        Returns:
            Formatted metrics for logging.
        """
        # eval_flow passes these to every estimator; this override does not
        # need them, and del marks them as intentionally unused for linters.
        del flow_field, flow_field_gt
        return process_art_of_piv_metrics(metrics, self._choice_tracker)

    def _close_sub_cache_managers(self) -> None:
        """Flush and close all known sub-cache managers.

        This ensures lazy-created sub-caches persist buffered writes when
        evaluation/training finalization runs.
        """
        if not self._sub_cache_managers:
            return

        seen_ids: set[int] = set()
        for cache_id, cm in self._sub_cache_managers.items():
            cm_id = id(cm)
            if cm_id in seen_ids:
                continue
            seen_ids.add(cm_id)
            try:
                cm.close()
            except Exception as e:
                logger.warning(
                    f"Failed to close sub-cache manager {cache_id}: {e}"
                )

    def finalize_metrics(self) -> Metrics:
        """Finalize metrics and persist any buffered sub-cache writes.

        Returns:
            Finalized metrics.
        """
        self._close_sub_cache_managers()
        return super().finalize_metrics()
