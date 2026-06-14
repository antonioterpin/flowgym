"""Router and postprocess validation helpers for Art of PIV."""

from collections.abc import Sequence
from typing import cast

from flowgym.nn.blocks import PostProcess, build_postprocess_pipeline
from flowgym.nn.registry import get_activation

from .shared import NORM_KINDS, _format_allowed


def normalize_router_config(router_config: dict) -> dict:
    """Validate, normalize, and tag a router architecture config.

    Two YAML schemas are accepted and auto-detected:

    - **Flat (legacy)** — single CNN trunk + MLP head. Required keys:
      ``features``, ``kernel_sizes``, ``strides``. Optional: ``norm_fn``
      (default ``"none"``), ``mlp_hidden_dim`` (default 512). Produces
      a ``CNNQEstimatorModel`` via the ``cnn_q_estimator`` builder.
    - **Nested** — shared CNN ``encoder`` + per-arm MLP ``head``.
      ``encoder.features_list`` and ``head.hidden_dims`` are required;
      everything else has sensible defaults. Produces a
      ``CNNRouterModel`` via the ``cnn_router`` builder.

    The returned dict carries a ``"type"`` key identifying the
    ``flowgym.nn.cnn.MODEL_REGISTRY`` entry, so callers can feed it
    straight into ``build_model_from_config`` without further dispatch.

    Args:
        router_config: Router configuration mapping in either schema.

    Returns:
        A normalized, builder-ready dict tagged with ``"type"``.

    Raises:
        TypeError: If the config (or any sub-config) is of the wrong
            container type.
        ValueError: If required fields are missing or invalid.
    """
    if not isinstance(router_config, dict):
        raise TypeError(
            f"router_config must be a dict, got {type(router_config)}."
        )

    has_encoder = "encoder" in router_config
    has_head = "head" in router_config
    if has_encoder and has_head:
        return _validate_nested_router_config(router_config)
    if has_encoder and not has_head:
        raise ValueError(
            "router_config has 'encoder' but is missing 'head' — the "
            "nested schema requires both keys."
        )
    if has_head and not has_encoder:
        raise ValueError(
            "router_config has 'head' but is missing 'encoder' — the "
            "nested schema requires both keys."
        )
    return _validate_flat_router_config(router_config)


def _validate_flat_router_config(router_config: dict) -> dict:
    """Validate the flat router schema and return a ``cnn_q_estimator`` spec.

    Args:
        router_config: Router config in the flat schema.

    Returns:
        A normalized dict tagged with ``"type": "cnn_q_estimator"`` and the
        keys that ``build_cnn_q_estimator_model_from_config`` expects.

    Raises:
        TypeError: If ``kernel_sizes`` or ``strides`` are not positive ints.
        ValueError: If required keys are missing or shapes disagree.
    """
    features = router_config.get("features")
    kernel_sizes = router_config.get("kernel_sizes")
    strides = router_config.get("strides")

    if features is None:
        raise ValueError("router_config must contain 'features' key.")
    if kernel_sizes is None:
        raise ValueError("router_config must contain 'kernel_sizes' key.")
    if strides is None:
        raise ValueError("router_config must contain 'strides' key.")

    norm_fn = router_config.get("norm_fn", "none")
    mlp_hidden_dim = router_config.get("mlp_hidden_dim", 512)

    if not isinstance(features, (list, tuple)) or len(features) == 0:
        raise ValueError(
            f"features must be a non-empty list or tuple, got {features}."
        )

    if not isinstance(kernel_sizes, (list, tuple)) or len(kernel_sizes) != len(
        features
    ):
        raise ValueError(
            f"kernel_sizes must be a list of the same length as features, "
            f"got {kernel_sizes}."
        )

    if not isinstance(strides, (list, tuple)) or len(strides) != len(features):
        raise ValueError(
            f"strides must be a list of the same length as features, "
            f"got {strides}."
        )

    if len(kernel_sizes) != len(strides):
        raise ValueError(
            f"kernel_sizes and strides must have the same length, "
            f"got {len(kernel_sizes)} and {len(strides)}."
        )

    if not all(isinstance(k, int) and k > 0 for k in kernel_sizes):
        raise TypeError(
            f"All kernel sizes must be positive integers, got {kernel_sizes}."
        )

    if not all(isinstance(s, int) and s > 0 for s in strides):
        raise TypeError(
            f"All strides must be positive integers, got {strides}."
        )

    if not isinstance(mlp_hidden_dim, int) or mlp_hidden_dim <= 0:
        raise ValueError(
            f"mlp_hidden_dim must be a positive integer, got {mlp_hidden_dim}."
        )

    if not isinstance(norm_fn, str) or norm_fn not in NORM_KINDS:
        raise ValueError(
            f"norm_fn must be one of ({_format_allowed(NORM_KINDS)}); "
            f"got {norm_fn!r}."
        )

    return {
        "type": "cnn_q_estimator",
        "features_list": tuple(features),
        "kernel_sizes": tuple(kernel_sizes),
        "strides": tuple(strides),
        "norm_fn": norm_fn,
        "mlp_hidden_dim": mlp_hidden_dim,
    }


def _validate_nested_router_config(router_config: dict) -> dict:
    """Validate the nested router schema and return a ``cnn_router`` spec.

    Args:
        router_config: Router config in the nested encoder/head schema.

    Returns:
        A normalized dict tagged with ``"type": "cnn_router"`` and the keys
        that ``build_cnn_router_model_from_config`` expects.

    Raises:
        TypeError: If ``encoder`` or ``head`` are not dicts.
        ValueError: If any required sub-field is missing or invalid.
    """
    encoder_config = router_config["encoder"]
    head_config = router_config["head"]

    if not isinstance(encoder_config, dict):
        raise TypeError(
            "encoder config must be a dict, "
            f"got {type(encoder_config).__name__}"
        )

    if not isinstance(head_config, dict):
        raise TypeError(
            f"head config must be a dict, got {type(head_config).__name__}"
        )

    # Encoder validation.
    if "features_list" not in encoder_config:
        raise ValueError("encoder config must contain 'features_list' key.")

    features_list = encoder_config["features_list"]
    if not isinstance(features_list, (list, tuple)):
        raise ValueError(
            "encoder features_list must be a list or tuple, "
            f"got {features_list}."
        )
    if len(features_list) == 0:
        raise ValueError("encoder features_list must be non-empty.")
    if not all(isinstance(f, int) and f > 0 for f in features_list):
        raise ValueError(
            "encoder features_list must contain only positive integers, "
            f"got {features_list}."
        )

    output_dim = encoder_config.get("output_dim", 64)
    if not isinstance(output_dim, int) or output_dim <= 0:
        raise ValueError(
            f"encoder output_dim must be a positive integer, got {output_dim}."
        )

    use_residual = encoder_config.get("use_residual", False)
    if not isinstance(use_residual, bool):
        raise ValueError(
            f"encoder use_residual must be a boolean, got {use_residual}."
        )

    norm_fn = encoder_config.get("norm_fn", "none")
    if not isinstance(norm_fn, str) or norm_fn not in NORM_KINDS:
        raise ValueError(
            f"encoder norm_fn must be one of ({_format_allowed(NORM_KINDS)}); "
            f"got {norm_fn!r}."
        )

    # Head validation.
    if "hidden_dims" not in head_config:
        raise ValueError("head config must contain 'hidden_dims' key.")

    hidden_dims = head_config["hidden_dims"]
    if not isinstance(hidden_dims, (list, tuple)):
        raise ValueError(
            f"head hidden_dims must be a list or tuple, got {hidden_dims}."
        )
    if len(hidden_dims) == 0:
        raise ValueError("head hidden_dims must be non-empty.")
    if not all(isinstance(d, int) and d > 0 for d in hidden_dims):
        raise ValueError(
            "head hidden_dims must contain only positive integers, "
            f"got {hidden_dims}."
        )

    activation = head_config.get("activation", "relu")
    if not isinstance(activation, str):
        raise ValueError(f"head activation must be a string, got {activation}.")
    try:
        get_activation(activation)
    except ValueError as e:
        raise ValueError(f"head activation: {e}") from None

    single_arm = head_config.get("single_arm", False)
    if not isinstance(single_arm, bool):
        raise ValueError(
            "head single_arm must be a boolean, "
            f"got {type(single_arm).__name__}."
        )

    return {
        "type": "cnn_router",
        "encoder": {
            "features_list": tuple(features_list),
            "output_dim": output_dim,
            "use_residual": use_residual,
            "norm_fn": norm_fn,
        },
        "head": {
            "hidden_dims": tuple(hidden_dims),
            "activation": activation,
            "single_arm": single_arm,
        },
    }


def validate_and_build_postprocess(
    postprocess: PostProcess | Sequence[str],
) -> PostProcess:
    """Validate the postprocess spec and build the final pipeline.

    Args:
        postprocess: Sequence of callables or postprocess names.

    Returns:
        A normalized postprocess pipeline.

    Raises:
        TypeError: If ``postprocess`` is not a homogeneous sequence of
            callables or strings.
    """
    if isinstance(postprocess, str):
        raise TypeError(
            f"postprocess must be a sequence of callables, got {postprocess}."
        )

    if not isinstance(postprocess, Sequence):
        raise TypeError(
            f"postprocess must be a sequence of callables, got {postprocess}."
        )

    if not all(callable(fn) or isinstance(fn, str) for fn in postprocess):
        raise TypeError(
            "All elements in postprocess must be callables or strings "
            f"got {postprocess!r}."
        )

    if all(isinstance(fn, str) for fn in postprocess):
        postprocess = cast(Sequence[str], postprocess)
        return build_postprocess_pipeline(postprocess)

    if all(callable(fn) for fn in postprocess):
        postprocess = cast(PostProcess, tuple(postprocess))
        return postprocess

    raise TypeError(
        "postprocess must be a sequence of either all callables or all strings."
    )
