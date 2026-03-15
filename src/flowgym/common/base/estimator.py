"""Base class for estimators."""

from __future__ import annotations

import abc
import inspect
from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synthpix.types import SynthpixBatch

    from flowgym.common.base.trainable_state import NNEstimatorTrainableState
    from flowgym.types import SupervisedExperience

from pprint import pformat

import jax
import jax.numpy as jnp
import numpy as np
from goggles import Metrics, get_logger
from goggles.history import create_history, update_history
from goggles.history.spec import HistorySpec
from goggles.history.types import History

from flowgym.common.base.trainable_state import EstimatorTrainableState
from flowgym.common.preprocess import apply_preprocessing, validate_params
from flowgym.types import CachePayload, PRNGKey, TrainStep
from flowgym.utils import load_configuration

logger = get_logger(__name__)


class Estimator(abc.ABC):
    """Base class for estimators."""

    def __init__(
        self,
        preprocessing_steps: list[dict[str, Any]] | None = None,
        optimizer_config: dict[str, Any] | None = None,
        oracle: bool = False,
    ) -> None:
        """Initialize the estimator.

        Args:
            preprocessing_steps:
                List of preprocessing steps to apply to the input image.
                Each step should be a dictionary with a `"name"` key and any
                other parameters. Defaults to None.
            optimizer_config:
                Configuration for the optimizer. Defaults to None.
            oracle:
                Whether the estimator has access to oracle information.

        Raises:
            ValueError: If preprocessing steps are invalid, `optimizer_config`
                is not a dictionary, or `oracle` is not a boolean.
        """
        if preprocessing_steps is None:
            preprocessing_steps = []

        # Validate preprocessing steps
        self.preprocessing_steps = []
        for step in preprocessing_steps:
            if not isinstance(step, dict):
                raise ValueError(
                    f"Preprocessing step {step} must be a dictionary."
                )
            if "name" not in step:
                raise ValueError(
                    f"Preprocessing step {step} must have a 'name' key."
                )
            validate_params(
                step["name"], **{k: v for k, v in step.items() if k != "name"}
            )
            self.preprocessing_steps.append(
                partial(apply_preprocessing, **step)
            )

        if optimizer_config is not None and not isinstance(
            optimizer_config, dict
        ):
            optimizer_type = type(optimizer_config)
            raise ValueError(
                "`optimizer_config` must be a dictionary, got "
                f"{optimizer_type}."
            )
        self.optimizer_config = optimizer_config
        if not isinstance(oracle, bool):
            raise ValueError(f"`oracle` must be a boolean, got {type(oracle)}.")
        self._oracle = oracle

        param_log = {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }
        logger.info(
            f"{self.__class__.__name__} initialized with parameters:\n"
            f"{pformat(param_log)}"
        )

    def create_state(
        self,
        images: jnp.ndarray,
        estimates: jnp.ndarray,
        *,
        image_history_size: int,
        estimate_history_size: int | None = None,
        extras: dict[str, dict[str, Any]] | None = None,
        rng: int | PRNGKey | None = None,
    ) -> History:
        """Create a dict-based history for an estimator.

        This creates a device-resident history. The resulting dict has one key
        per tracked quantity, each with a leading batch dimension `(B, ...)`.
        The `"images"` and `"estimates"` keys are always present.

        Args:
            images: Batched input images of shape `(B, H, W)`.
            estimates: Batched initial estimates of shape
                `(B, *estimate_shape)`.
            image_history_size: Number of timesteps to keep in image history.
            estimate_history_size: Number of timesteps to keep in the estimate
                history. Defaults to `image_history_size`.
            extras: Additional history fields in the
                `HistorySpec` config format. Each entry must define at least:
                `{"length": int, "shape": tuple, "dtype": jnp.dtype,
                "init": str}`.
            rng: Seed or PRNGKey. If provided, `B` additional RNG
                fields are added to the history under the key `"keys"`.
                Defaults to `jax.random.PRNGKey(0)`.

        Returns:
            A JAX-compatible dict with keys:
                - `"images"`: `(B, image_history_size, H, W)`
                - `"estimates"`: `(B, estimate_history_size, *estimate_shape)`
                - any extra fields from `extras`
                - any RNG fields added by `create_history`

        Example:
            >>> import jax, jax.numpy as jnp
            >>> from goggles.history import create_history
            >>> from goggles.history.spec import HistorySpec
            >>>
            >>> images = jnp.ones((3, 32, 32))
            >>> estimates = jnp.zeros((3, 2))
            >>> extras = {
            ...     "reward": {
            ...         "length": 4,
            ...         "shape": (),
            ...         "dtype": jnp.float32,
            ...         "init": "zeros",
            ...     },
            ... }
            >>> rng = jax.random.key(0)
            >>> history = create_state(
            ...     images,
            ...     estimates,
            ...     image_history_size=4,
            ...     estimate_history_size=4,
            ...     extras=extras,
            ...     rng=rng,
            ... )
            >>> print(list(history.keys()))
            ['images', 'estimates', 'keys', 'reward0']

        Raises:
            ValueError: If `images`/`estimates` shapes are invalid or batch
                sizes mismatch.
            TypeError: If `rng` is neither an int seed nor a PRNG key.
        """
        if images.ndim != 3:
            raise ValueError(
                f"`images` must have shape (B, H, W), got {images.shape}."
            )
        if estimates.ndim < 2:
            raise ValueError(
                f"`estimates` must have shape (B, ...), got {estimates.shape}."
            )
        B, _, _ = images.shape
        if estimates.shape[0] != B:
            raise ValueError(
                "Batch size mismatch: "
                f"images ({B}) vs estimates ({estimates.shape[0]})."
            )

        # Apply preprocessing to images
        for step in self.preprocessing_steps:
            images = step(images)

        B, H, W = images.shape

        # Set estimate history size if not provided
        if estimate_history_size is None:
            estimate_history_size = image_history_size

        base_cfg = {
            "images": {
                "length": image_history_size,
                "shape": (H, W),
                "dtype": images.dtype,
                "init": "zeros",
            },
            "estimates": {
                "length": estimate_history_size,
                "shape": estimates.shape[1:],
                "dtype": estimates.dtype,
                "init": "zeros",
            },
        }
        if extras:
            base_cfg.update(extras)

        rng_create = None
        per_batch_keys = None
        if rng is not None:
            if isinstance(rng, int):
                rng = jax.random.PRNGKey(rng)
            elif not (isinstance(rng, PRNGKey) and rng.shape == (2,)):
                raise TypeError(
                    f"`rng` must be an int seed or PRNGKey, got {type(rng)} "
                    f"with shape {getattr(rng, 'shape', None)}."
                )

            # Split master key into one for create_history and one for
            # per-batch keys.
            rng_create, rng_batch = jax.random.split(rng)
            per_batch_keys = jax.random.split(rng_batch, B)  # (B, 2)

        # Create HistorySpec from config
        # Input validation is handled in HistorySpec.from_config
        spec = HistorySpec.from_config(base_cfg)

        history: History = create_history(spec, B, rng_create)

        history["images"] = jnp.tile(
            images[:, None, ...], (1, image_history_size, 1, 1)
        )
        history["estimates"] = jnp.tile(
            estimates[:, None, ...],
            (1, estimate_history_size, *([1] * (estimates.ndim - 1))),
        )

        if per_batch_keys is not None:
            history["keys"] = per_batch_keys[:, None, :]  # (B, 1, 2)

        extra_history_spec = self._create_extras()
        extra_history = create_history(extra_history_spec, B, rng=None)
        for k, v in extra_history.items():
            history[k] = v

        return history

    def create_trainable_state(
        self,
        dummy_input: jnp.ndarray,
        key: PRNGKey,
    ) -> EstimatorTrainableState:
        """Create the trainable state of the estimator.

        Args:
            dummy_input: batched dummy input, shape (B, H, W).
            key: Random key for JAX operations.

        Returns:
            The initial trainable state of the estimator.
        """
        return EstimatorTrainableState()

    def _create_extras(self) -> HistorySpec:
        """Create extra fields for the history.

        Notice that these fields will be initialized in `create_state`,
        and persisted in `__call__`. Moreover, these fields will be
        batched, i.e., they will have a leading batch dimension `(B, ...)`.
        So no need to add a batch dimension here.

        Returns:
            Extra fields for the history.
        """
        return HistorySpec.from_config({})

    def create_train_step(self) -> TrainStep:
        """Create a training step function for the estimator.

        Subclasses should override this to return a function conforming to
        either SupervisedTrainStep or RLTrainStep protocols from flowgym.types.

        Returns:
            Training step function (SupervisedTrainStep or RLTrainStep).

        Raises:
            NotImplementedError: If the estimator does not support training.
        """
        raise NotImplementedError("This estimator does not support training.")

    def __call__(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state: EstimatorTrainableState,
        cache_payload: CachePayload | dict[str, Any] | None = None,
    ) -> tuple[History, dict[str, jnp.ndarray]]:
        """Compute next estimates and roll history forward.

        Steps:
            1) Preprocess input `images`.
            2) Handle per-batch randomness: if `state["keys"]` exists, split
               each key into `(new_key, subkey)`. The `subkey` is exposed to
               `_estimate` for deterministic randomness in this step, while
               `new_key` is persisted for the next step.
            3) Prepare `extras` from history (excluding base fields).
            4) Build `_estimate` extras by injecting `cache_payload` under the
               `cache_payload` key and flattening `cache_payload.extras`.
            5) Call
               `_estimate(images, state_for_step, trainable_state, extras)`.
               The estimator decides how to use cached data.
            6) History Update: roll `images`, `estimates`, and `keys` forward
               using `update_history`.
            7) Persistence: re-attach only the original history extras returned
               by `_estimate` to the state. Transient payload-only keys are not
               persisted.

        Args:
            images: Batched input images of shape `(B, H, W)`.
            state: Dict-based history.
            trainable_state: Trainable state.
            cache_payload: Optional cached data for the batch. Legacy dict
                payloads are converted to `CachePayload`. The estimator is
                responsible for using this data in `_estimate` if provided.
                Typically this payload comes from
                `training.caching.enrich_batch()` in the training/eval loop.
                Note that while used for the current step, this payload is NOT
                persisted in the `state` history to avoid memory bloat.

        Returns:
            A tuple with the updated history and a dictionary of metrics.

        Raises:
            ValueError: If `images` does not have shape `(B, H, W)`.
        """
        if images.ndim != 3:
            raise ValueError(
                f"`images` must have shape (B, H, W), got {images.shape}."
            )

        # Preprocess images
        for step in self.preprocessing_steps:
            images = step(images)

        # Handle per-batch RNG keys if present
        have_keys = "keys" in state
        if have_keys:
            # Robustly slice keys to (B, 2) regardless of temporal dimension
            B = images.shape[0]
            keys_all = jnp.asarray(state["keys"])
            keys_bt = keys_all.reshape(B, -1)[:, -2:]  # (B, 2)

            # Split each per-example key into (new_key, subkey).
            # Result shape: (B, 2, 2).
            pair = jax.vmap(lambda k: jax.random.split(k, 2))(keys_bt)
            new_keys = pair[:, 0]  # (B, 2) → persisted to history
            subkeys = pair[:, 1]  # (B, 2) → used for this step

            # Temporary state view exposing subkeys as (B, 1, 2) for _estimate
            state_for_step = {
                k: v for k, v in state.items() if k in ["images", "estimates"]
            }
            state_for_step["keys"] = subkeys[:, jnp.newaxis, :]  # (B, 1, 2)
        else:
            state_for_step = state

        # Prepare extras from history (excluding base fields)
        have_extras = any(
            k not in ("images", "estimates", "keys") for k in state.keys()
        )
        extras = (
            {
                k: state[k]
                for k in state.keys()
                if k not in ("images", "estimates", "keys")
            }
            if have_extras
            else {}
        )
        original_extra_keys = set(extras.keys())

        # Backward-compat: wrap legacy dict-based cache_payload.
        if isinstance(cache_payload, dict):
            cache_payload = CachePayload.from_enrich_result(cache_payload)

        # Pass cache_payload to _estimate via estimate_extras.
        # Individual estimators decide how to use cached data.
        estimate_extras: dict[str, Any] = dict(extras)
        estimate_extras["cache_payload"] = cache_payload
        if cache_payload is not None:
            estimate_extras.update(cache_payload.extras)

        # Perform estimation (estimator can short-circuit internally if desired)
        estimates, extras, metrics = self._estimate(
            images, state_for_step, trainable_state, estimate_extras
        )

        # Prepare state for rolling history (keep only base keys)
        state_for_next = {
            k: v
            for k, v in state.items()
            if k in ["images", "estimates", "keys"]
        }

        # Roll history forward with new frames
        new_data = {
            "images": images[:, jnp.newaxis, ...],  # (B, 1, H, W)
            "estimates": estimates[:, jnp.newaxis, ...],  # (B, 1, *est_shape)
        }

        if have_keys:
            new_data["keys"] = new_keys[:, jnp.newaxis, :]  # (B, 1, 2)

        # Update core history
        state = update_history(state_for_next, new_data, reset_mask=None)

        # Persist updated extras (filtering out any transient payload keys)
        for k in original_extra_keys:
            v = extras[k]
            shape_str = f" with shape {v.shape}" if hasattr(v, "shape") else ""
            logger.debug(f"Persisting extra key: {k}{shape_str}")
            state[k] = v

        return state, metrics

    @abc.abstractmethod
    def _estimate(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state: EstimatorTrainableState,
        extras: dict[str, Any],
    ) -> tuple[jnp.ndarray, dict[str, Any], dict[str, jnp.ndarray]]:
        """Return the next estimates given the current images and history.

        Args:
            images: Batched input images of shape `(B, H, W)` after
                preprocessing.
            state: Dict-based history (may include prior images/estimates/keys).
            trainable_state: Trainable state (model params, optimizer state,
                etc.).
            extras: Additional data from history fields not including
                images/estimates/keys.

        Returns:
            Batched estimates of shape `(B, *estimate_shape)`.
            Additional data from history fields not including
                images/estimates/keys.
            Optional metrics from the estimation step.

        Raises:
            NotImplementedError: If the subclass does not implement this
                method.
        """
        raise NotImplementedError

    def process_metrics(self, metrics: dict[str, jnp.ndarray]) -> Metrics:
        """Process metrics after estimation.

        Args:
            metrics: The raw metrics from the estimation step.

        Returns:
            Processed metrics.
        """
        # Convert JAX arrays to numpy arrays
        return Metrics(**{k: np.asarray(v) for k, v in metrics.items()})

    def finalize_metrics(self) -> Metrics:
        """Finalize metrics after evaluation.

        Returns:
            Finalized metrics.
        """
        return Metrics()

    def prepare_experience_for_replay(
        self,
        experience: SupervisedExperience,
        trainable_state: NNEstimatorTrainableState,
    ) -> SupervisedExperience:
        """Prepare an experience for storage in the replay buffer.

        This hook is called after training step, before storing the experience
        in the replay buffer. Subclasses can override this to enrich the
        experience with computed data (e.g., estimates, actions) to avoid
        recomputation during replay.

        Args:
            experience: The experience to prepare.
            trainable_state: Current trainable state of the model.

        Returns:
            The prepared experience (may be the same or enriched).
        """
        return experience

    def enrich(
        self,
        batch: SynthpixBatch,
        miss_idxs: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Compute the cache payload for missing keys.

        This is the single hook for estimators to define what data should be
        cached. The infrastructure (CacheManager) handles lookup and storage.

        Subclasses should override this if they produce expensive derived data
        that should be reusable across runs (e.g., EPE, flow estimates).

        Rules for Meta-Estimators:
            Meta-estimators can override this to aggregate cached data from
            their sub-model caches, returning a combined payload
            (e.g., `"epe_all"`, `"epe_mask"`).

        Args:
            batch: The batch of data as returned by the `SynthpixBatch` sampler.
            miss_idxs: Indices of samples in the batch that are missing from
                cache.
            **kwargs: Additional context, potentially including
                `"trainable_state"` for weight-dependent computations.

        Returns:
            A dictionary containing the computed payload for missing samples, or
            None if this estimator does not support caching.
            The dictionary keys must match the expected cache schema
            (e.g., `"epe"`). Values should be numpy-compatible arrays or lists
            aligned with `miss_idxs`.
        """
        return None

    def preload_caches(
        self,
        root_dir: str,
        warm_start: str = "all",
    ) -> None:
        """Preload caches into memory for fast enrichment.

        This method provides a unified interface for cache preloading. The
        default implementation is a no-op.

        Meta-estimators can override this to load their sub-model caches
        into memory at startup for fast aggregation during training.

        Args:
            root_dir: Root directory containing cache files.
            warm_start: Cache loading strategy ('none', 'index', 'all').
        """
        # Intentionally empty - subclasses override to preload their caches
        return None

    def get_cache_id_suffix(
        self,
        trainable_state: EstimatorTrainableState | None,
    ) -> str:
        """Get a suffix for the cache ID based on the model state.

        This allows the cache to be invalidated or namespaced based on the
        specific weights or configuration of the model. The returned string
        is appended to the base class name to form a unique `cache_id`.

        Standard practice is to hash the `trainable_state.params` and the
        estimator's internal configuration.

        Args:
            trainable_state: The current trainable state, or None if the model
                is not trainable or parameters are not yet initialized.

        Returns:
            A string suffix (e.g., "_v1_hash123") or empty string.
        """
        return ""

    def is_oracle(self) -> bool:
        """Check if the estimator is an oracle.

        Returns:
            True if the estimator has access to oracle information, else False.
        """
        return self._oracle

    @classmethod
    def get_init_param_names(cls) -> set[str]:
        """Get `__init__` parameters from the class and its parents.

        Returns:
            A set of parameter names.
        """
        params = set()
        for c in inspect.getmro(cls):
            if "__init__" in c.__dict__:
                sig = inspect.signature(c.__init__)
                for name, _p in sig.parameters.items():
                    if name not in ("self", "args", "kwargs"):
                        params.add(name)
        return params

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Estimator:
        """Construct an estimator from a config dictionary.

        Only keys matching the class and its parents' `__init__` parameters are
        used.

        Args:
            config: Configuration dictionary with parameters for the estimator.

        Returns:
            An instance of the estimator initialized with the provided config.

        Raises:
            ValueError: If preprocess/postprocess configuration is not valid.
        """
        # Get valid keys from the class's __init__ parameters
        valid_keys = cls.get_init_param_names()

        # Extract pre-processing config directory
        if "preprocess" in config:
            preprocess_config = config.pop("preprocess")
            if isinstance(preprocess_config, str):
                # Attempt to read configuration file
                preprocess_config = load_configuration(preprocess_config) or {}
                if preprocess_config != {}:
                    preprocess_config = preprocess_config["preprocessing_steps"]
                else:
                    preprocess_config = []
            if not isinstance(preprocess_config, list):
                raise ValueError(
                    f"Preprocess config {preprocess_config} must be "
                    "a list or a valid YAML file."
                )

            # Add pre-processing parameters to valid keys
            config["preprocessing_steps"] = preprocess_config

        # Extract post-processing config directory
        if "postprocess" in config:
            postprocess_config = config.pop("postprocess")
            if isinstance(postprocess_config, str):
                # Attempt to read configuration file
                postprocess_config = (
                    load_configuration(postprocess_config) or {}
                )
                if postprocess_config != {}:
                    postprocess_config = postprocess_config[
                        "postprocessing_steps"
                    ]
                else:
                    postprocess_config = []
            if not isinstance(postprocess_config, list):
                raise ValueError(
                    f"Postprocess config {postprocess_config} must be "
                    "a list or a valid YAML file."
                )

            # Add post-processing parameters to valid keys
            config["postprocessing_steps"] = postprocess_config

        # Filter the config to only include valid keys
        filtered = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered)
