"""Base class for estimators."""

import abc
from functools import partial
from typing import Any
from collections.abc import Callable

import jax
import jax.numpy as jnp
import inspect
import numpy as np
from rich import print as rich_print
from rich.pretty import Pretty

from flowgym.common.base.trainable_states import EstimatorTrainableState
from flowgym.common.preprocess import apply_preprocessing, validate_params
from flowgym.utils import load_configuration
from flowgym.types import PRNGKey

from goggles.history import create_history, update_history
from goggles.history.spec import HistorySpec
from goggles.history.types import History
from goggles import get_logger, Metrics

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
                Each step should be a dictionary with a 'name' key and other parameters.
                Defaults to None, which means no preprocessing steps are applied.
            optimizer_config:
                Configuration for the optimizer. Defaults to None.
            oracle:
                Whether the estimator has access to oracle information. Defaults to False.
        """
        if preprocessing_steps is None:
            preprocessing_steps = []

        # Validate preprocessing steps
        self.preprocessing_steps = []
        for step in preprocessing_steps:
            if not isinstance(step, dict):
                raise ValueError(f"Preprocessing step {step} must be a dictionary.")
            if "name" not in step:
                raise ValueError(f"Preprocessing step {step} must have a 'name' key.")
            validate_params(
                step["name"], **{k: v for k, v in step.items() if k != "name"}
            )
            self.preprocessing_steps.append(partial(apply_preprocessing, **step))

        if optimizer_config is not None and not isinstance(optimizer_config, dict):
            raise ValueError(
                f"`optimizer_config` must be a dictionary, got {type(optimizer_config)}."
            )
        self.optimizer_config = optimizer_config
        if not isinstance(oracle, bool):
            raise ValueError(f"`oracle` must be a boolean, got {type(oracle)}.")
        self.oracle = oracle

        param_log = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        rich_print(
            f"[bold yellow]{self.__class__.__name__}" " initialized with parameters:"
        )
        rich_print(Pretty(param_log))

    def create_state(
        self,
        images: jnp.ndarray,
        estimates: jnp.ndarray,
        *,
        image_history_size: int,
        estimate_history_size: int | None = None,
        extras: dict[str, dict[str, Any]] | None = None,
        rng: int | PRNGKey | None = jax.random.PRNGKey(0),
    ) -> History:
        """Create a dict-based history for an estimator.

        This create a device-resident history. The resulting dict has one key
        per tracked quantity, each with a leading batch dimension `(B, ...)`.
        In particular, the `"images"` and `"estimates"` keys are always present.

        Args:
            images: Batched input images of shape `(B, H, W)`.
            estimates: Batched initial estimates of shape `(B, *estimate_shape)`.
            image_history_size: Number of timesteps to keep in the image history.
            estimate_history_size: Number of timesteps to keep in the estimate
                history. Defaults to `image_history_size`.
            extras: Additional history fields in the
                `HistorySpec` config format. Each entry must define at least:
                `{"length": int, "shape": tuple, "dtype": jnp.dtype, "init": str}`.
            rng: Seed or PRNGKey. If provided, B additional RNG
                fields will be added to the history under the key "keys".
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
            ...     "reward": {"length": 4, "shape": (), "dtype": jnp.float32, "init": "zeros"},
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
        """
        if images.ndim != 3:
            raise ValueError(f"`images` must have shape (B, H, W), got {images.shape}.")
        if estimates.ndim < 2:
            raise ValueError(
                f"`estimates` must have shape (B, ...), got {estimates.shape}."
            )
        B, _, _ = images.shape
        if estimates.shape[0] != B:
            raise ValueError(
                f"Batch size mismatch: images ({B}) vs estimates ({estimates.shape[0]})."
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

            # Split master key into one for create_history and one for per-batch keys
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
        key,
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

    def create_train_step(self) -> Callable:
        """Create a training step function for the estimator.

        Returns:
            Training step function.
        """
        raise NotImplementedError("This estimator does not support training.")

    def __call__(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state,
    ) -> tuple[History, dict[str, jnp.ndarray]]:
        """Compute next estimates and roll history forward.

        Steps:
            1) Preprocess `images`.
            2) If `state["keys"]` exists, split per-batch keys → (new_keys, subkeys).
                Create a *temporary* view of `state` exposing `subkeys` under "keys"
                so `_estimate` can use randomness deterministically for this step.
            3) Call `_estimate(images, state_for_step, trainable_state)` → estimates.
            4) Roll history forward.
            5) If `state["keys"]` exists, split and persist `new_keys` into `state`.

        Returns:
            updated history (device-resident PyTree).
        """
        if images.ndim != 3:
            raise ValueError(f"`images` must have shape (B, H, W), got {images.shape}.")

        # Preprocess images
        for step in self.preprocessing_steps:
            images = step(images)

        # Handle per-batch RNG keys if present
        have_keys = "keys" in state
        if have_keys:
            # state["keys"] has shape (B, 1, 2); take the last (length-1) slice → (B, 2)
            keys_bt = state["keys"][:, -1, :]  # (B, 2)

            # Split each per-example key into (new_key, subkey); result (B, 2, 2)
            pair = jax.vmap(lambda k: jax.random.split(k, 2))(keys_bt)
            new_keys = pair[:, 0, :]  # (B, 2) → persisted to history
            subkeys = pair[:, 1, :]  # (B, 2) → used for this step

            # Temporary state view exposing subkeys as (B, 2)
            state_for_step = {
                k: v for k, v in state.items() if k in ["images", "estimates"]
            }
            state_for_step["keys"] = subkeys
        else:
            state_for_step = state

        have_extras = any(
            k not in ("images", "estimates", "keys") for k in state.keys()
        )
        if have_extras:
            extras = {
                k: state_for_step[k]
                for k in state_for_step.keys()
                if k not in ("images", "estimates", "keys")
            }
        else:
            extras = {}

        # Estimate
        estimates, extras, metrics = self._estimate(
            images, state_for_step, trainable_state, extras
        )

        # Prepare state for next step (remove extras)
        state_for_next = {
            k: v for k, v in state.items() if k in ["images", "estimates", "keys"]
        }

        # Roll history forward
        new_data = {
            "images": images[:, jnp.newaxis, ...],  # (B, 1, H, W)
            "estimates": estimates[:, jnp.newaxis, ...],  # (B, 1, *estimate_shape)
        }

        # Persist new keys if present
        if have_keys:
            new_data["keys"] = new_keys[:, jnp.newaxis, :]  # (B, 1, 2)

        # Update history
        state = update_history(state_for_next, new_data, reset_mask=None)

        for k, v in extras.items():
            logger.debug(f"Persisting extra key: {k} with shape {v.shape}")
            state[k] = v  # The shape should be decided by the estimator

        return state, metrics

    @abc.abstractmethod
    def _estimate(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state: EstimatorTrainableState,
        extras: dict,
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Return the next estimates given the current images and history.

        Args:
            images: Batched input images of shape (B, H, W) after preprocessing.
            state: Dict-based history (e.g., may include prior images/estimates/keys).
            trainable_state: Trainable state (model params, optimizer state, etc.).
            extras: Additional data from history fields not including images/estimates/keys.

        Returns:
            Batched estimates of shape (B, *estimate_shape).
            Additional data from history fields not including images/estimates/keys.
            Optional metrics from the estimation step.
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

    def is_oracle(self) -> bool:
        """Check if the estimator is an oracle.

        Returns:
            True if the estimator has access to oracle information, False otherwise.
        """
        return self.oracle

    @classmethod
    def get_init_param_names(cls):
        """Get the parameters for the __init__ method of the class and its parents.

        Returns:
            A set of parameter names.
        """
        params = set()
        for c in inspect.getmro(cls):
            if "__init__" in c.__dict__:
                sig = inspect.signature(c.__init__)
                for name, p in sig.parameters.items():
                    if name not in ("self", "args", "kwargs"):
                        params.add(name)
        return params

    @classmethod
    def from_config(cls, config: dict):
        """Construct an estimator from a config dictionary.

        Only keys matching the class and its parents' __init__ parameters will be used.

        Args:
            config: Configuration dictionary with parameters for the estimator.

        Returns:
            An instance of the estimator initialized with the provided config.
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
                postprocess_config = load_configuration(postprocess_config) or {}
                if postprocess_config != {}:
                    postprocess_config = postprocess_config["postprocessing_steps"]
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
