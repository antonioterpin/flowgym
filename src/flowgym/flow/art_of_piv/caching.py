"""Caching utilities for Art of PIV meta-estimator.

Handles aggregation of EPEs from multiple sub-model caches using
unified indexing for O(1) lookups.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import goggles as gg
import numpy as np

if TYPE_CHECKING:
    from synthpix.types import SynthpixBatch

    from flowgym.common.base import EstimatorTrainableState
    from flowgym.flow.base import FlowFieldEstimator


logger = gg.get_logger(__name__)


class SubCacheManager(Protocol):
    """Minimal cache-manager contract used by Art of PIV caching helpers.

    Attributes:
        all_data: In-memory cache table keyed by payload field names.
        spec: Payload schema mapping each field to (dtype, per-sample shape).
    """

    all_data: dict[str, np.ndarray] | None
    spec: dict[str, tuple[np.dtype, tuple[int, ...]]]

    def write(
        self, keys: np.ndarray, payload: dict[str, np.ndarray]
    ) -> None: ...

    def close(self) -> None: ...


def _infer_cache_spec_from_payload(
    payload: Mapping[str, Any],
    *,
    miss_count: int,
    cache_id: str,
) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    """Infer a cache spec from a sub-model payload.

    Args:
        payload: Sub-model payload returned by enrich().
        miss_count: Number of miss samples for this payload.
        cache_id: Cache identifier used for error messages.

    Returns:
        Cache spec mapping payload keys to (dtype, per-sample shape).

    Raises:
        ValueError: If payload fields are not aligned with miss_count or are
            not numeric array-like values.
    """
    spec: dict[str, tuple[np.dtype, tuple[int, ...]]] = {}
    for name, value in payload.items():
        array = np.asarray(value)

        if array.ndim == 0:
            raise ValueError(
                f"{cache_id}.{name} must include a batch dimension; "
                f"got shape {array.shape}"
            )
        if array.shape[0] != miss_count:
            raise ValueError(
                f"{cache_id}.{name} first dimension ({array.shape[0]}) does "
                f"not match miss count ({miss_count})"
            )
        if array.dtype == np.dtype("O"):
            raise ValueError(
                f"{cache_id}.{name} must be numeric/boolean, got object dtype"
            )

        spec[name] = (array.dtype, tuple(array.shape[1:]))

    if not spec:
        raise ValueError(f"{cache_id} payload did not contain any fields")

    return spec


def _get_or_create_sub_cache_manager(
    *,
    root_dir: str | None,
    cache_id: str,
    sub_cache_managers: dict[str, SubCacheManager],
    payload: Mapping[str, Any],
    miss_count: int,
) -> SubCacheManager | None:
    """Get an existing sub-cache manager or create one from payload spec.

    Args:
        root_dir: Root cache directory for this run, if available.
        cache_id: Target sub-cache ID.
        sub_cache_managers: Existing cache-manager registry for sub-models.
        payload: Sub-model miss payload used to infer cache spec.
        miss_count: Number of miss samples represented by payload.

    Returns:
        Existing/newly created cache manager, or None when creation is not
        possible (e.g., root_dir is unavailable).
    """
    cm = sub_cache_managers.get(cache_id)
    if cm is not None:
        return cm

    if root_dir is None:
        return None

    # Keep this local import so importing Art of PIV does not eagerly import
    # training.caching (which imports pyarrow at module import time).
    # pyarrow is a caching/runtime-extra dependency, not required for base use.
    from flowgym.training.caching import CacheManager  # noqa: PLC0415

    spec = _infer_cache_spec_from_payload(
        payload,
        miss_count=miss_count,
        cache_id=cache_id,
    )
    cm = CacheManager(
        root_dir=root_dir,
        cache_id=cache_id,
        spec=spec,
        warm_start="none",
    )
    sub_cache_managers[cache_id] = cm
    logger.info(f"Created sub-cache manager on demand: {cache_id}")
    return cm


def preload_caches(
    root_dir: str,
    warm_start: Literal["all", "index", "none"],
    sub_models: Sequence[FlowFieldEstimator],
    sub_model_states: Sequence[EstimatorTrainableState],
    estimator_configs: Sequence[Mapping[str, Any]],
    sub_cache_managers: dict[str, SubCacheManager],
    cache_id_base: str | None = None,
) -> None:
    """Load all sub-model caches into memory for fast aggregation.

    This should be called once at training/eval startup. After this, all lookups
    from `enrich_from_sub_caches` are O(1) RAM lookups.

    Args:
        root_dir: Root directory containing sub-model caches.
        warm_start: Warm start mode for sub-model caches ('all' recommended).
        sub_models: Tuple of sub-model estimator instances.
        sub_model_states: Tuple of sub-model trainable states.
        estimator_configs: List of sub-model configurations.
        sub_cache_managers: Dictionary to populate with cache managers.
        cache_id_base: Optional fixed string to use as the cache-id base
            instead of the sub-model class name. When set, cache dirs are
            expected to be named ``<cache_id_base><suffix>`` (where suffix
            comes from ``model.get_cache_id_suffix``). Defaults to None,
            which preserves the original ``<ClassName><suffix>`` behaviour.
    """
    root_path = Path(root_dir)

    logger.info(
        f"Preloading {len(sub_models)} sub-model caches from {root_dir}..."
    )

    for i, (model, t_state) in enumerate(
        zip(sub_models, sub_model_states, strict=True)
    ):
        cfg = estimator_configs[i]
        model_name = cfg.get("name", cfg.get("estimator", f"SubModel_{i}"))

        suffix = model.get_cache_id_suffix(t_state)
        base = (
            cache_id_base
            if cache_id_base is not None
            else model.__class__.__name__
        )
        cache_id = f"{base}{suffix}"

        cache_dir = root_path / cache_id
        if not cache_dir.exists():
            logger.warning(f"Cache not found for {model_name}: {cache_dir}")
            continue

        meta_path = cache_dir / "meta.json"
        if not meta_path.exists():
            logger.warning(f"No meta.json found for {model_name}: {cache_dir}")
            continue

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            spec = {
                k: (np.dtype(v[0]), tuple(v[1]))
                for k, v in meta["spec"].items()
            }

            # Keep this local import so Art of PIV modules stay importable
            # without eagerly importing training.caching/pyarrow.
            from flowgym.training.caching import CacheManager  # noqa: PLC0415

            cm = CacheManager(
                root_dir=str(root_path),
                cache_id=cache_id,
                spec=spec,
                warm_start=warm_start,
            )
            sub_cache_managers[cache_id] = cm
            logger.info(
                f"Preloaded cache for {model_name}: {cache_id} "
                f"({len(cm.all_data['key']) if cm.all_data else 0} items)"
            )
        except Exception as e:
            logger.warning(f"Failed to preload cache for {model_name}: {e}")


def build_unified_index(
    sub_models: Sequence[FlowFieldEstimator],
    sub_model_states: Sequence[EstimatorTrainableState],
    sub_cache_managers: Mapping[str, SubCacheManager],
    cache_id_base: str | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Build a unified hash-based index for O(1) lookups.

    Aggregates EPEs from all sub-model caches into a single dictionary
    mapping key -> (epe_0, epe_1, ..., epe_n). Only keys present in ALL
    **loaded** sub-model caches are included.

    If some sub-model caches are not loaded (e.g., on first run when caches
    don't exist yet), the unified index will still be built for the available
    caches. This allows partial speedup rather than falling back to no speedup.

    Args:
        sub_models: Tuple of sub-model estimator instances.
        sub_model_states: Tuple of sub-model trainable states.
        sub_cache_managers: Dictionary of cache managers for sub-models.
        cache_id_base: Optional fixed string used as the cache-id base when
            resolving each sub-model's manager from ``sub_cache_managers``.
            Must match the value passed to ``preload_caches``. Defaults to
            None, which uses the sub-model class name.

    Returns:
        Dictionary mapping key -> (epe_array, mask).
    """
    if not sub_cache_managers:
        logger.warning(
            "No sub-cache managers loaded, cannot build unified index"
        )
        return {}

    num_models = len(sub_models)
    cache_data: list[tuple[np.ndarray, np.ndarray, int]] = []
    model_indices: list[int] = []

    # Collect keys and EPEs from each sub-cache that has data loaded
    for i, (model, t_state) in enumerate(
        zip(sub_models, sub_model_states, strict=True)
    ):
        suffix = model.get_cache_id_suffix(t_state)
        base = (
            cache_id_base
            if cache_id_base is not None
            else model.__class__.__name__
        )
        cache_id = f"{base}{suffix}"
        cm = sub_cache_managers.get(cache_id)

        if cm is None or cm.all_data is None:
            logger.debug(
                f"Skipping {cache_id} in unified index (cache not "
                f"loaded or empty)"
            )
            continue

        cache_data.append((cm.all_data["key"], cm.all_data["epe"], i))
        model_indices.append(i)

    if not cache_data:
        logger.warning(
            "No sub-caches with loaded data, cannot build unified index"
        )
        return {}

    # Find common keys across all loaded sub-caches
    common_keys = set(cache_data[0][0].tolist())
    for keys, _, _ in cache_data[1:]:
        common_keys &= set(keys.tolist())

    if not common_keys:
        logger.warning("No common keys across loaded sub-caches")
        return {}

    # Build key-to-index lookup for each cache
    key_to_idx_list = [
        {int(k): idx for idx, k in enumerate(keys)} for keys, _, _ in cache_data
    ]

    # Build unified index
    # Store as tuple: (epe_array, mask) where mask indicates which
    # models have data
    unified: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in common_keys:
        epe_array = np.zeros(num_models, dtype=np.float32)
        mask = np.zeros(num_models, dtype=bool)
        for j, (_, epe_values, model_idx) in enumerate(cache_data):
            idx = key_to_idx_list[j][k]
            epe_array[model_idx] = epe_values[idx]
            mask[model_idx] = True
        unified[k] = (epe_array, mask)

    logger.info(
        f"Built unified EPE index with {len(unified)} entries "
        f"for {len(cache_data)}/{num_models} sub-models"
    )
    return unified


def enrich_from_sub_caches(
    batch: SynthpixBatch,
    compute_misses: bool,
    sub_models: Sequence[FlowFieldEstimator],
    sub_model_states: Sequence[EstimatorTrainableState],
    sub_cache_managers: dict[str, SubCacheManager],
    unified_epe_index: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    root_dir: str | None = None,
    allow_create_missing: bool = False,
    cache_id_base: str | None = None,
) -> dict[str, np.ndarray]:
    """Aggregate EPEs from pre-loaded sub-model caches using unified index.

    This function requires a unified index for O(1) lookups. Use
    `preload_caches()` to build the unified index before calling this.

    Args:
        batch: The SynthpixBatch to enrich.
        compute_misses: If True, compute cache misses by running sub-model
            inference. Set to False during evaluation to avoid expensive
            computation (evaluation will run the model anyway).
        sub_models: Tuple of sub-model estimator instances.
        sub_model_states: Tuple of sub-model trainable states.
        sub_cache_managers: Dictionary of cache managers for sub-models.
        unified_epe_index: Unified index for O(1) lookups (required).
        root_dir: Root directory used when lazily creating missing sub-caches.
        allow_create_missing: Whether to create missing sub-cache managers from
            the first computed payload.
        cache_id_base: Optional fixed base for sub-cache ids; when set, the
            write-back path uses ``<cache_id_base><suffix>`` to match
            ``preload_caches`` (defaults to the sub-model class name).

    Returns:
        Dictionary with "epe_all" (B, num_estimators) and
        "epe_mask" (B, num_estimators).

    Raises:
        ValueError: If a sub-model returns an EPE payload that does not align
            with the miss positions being filled.
    """
    # Keep this local for the same reason as CacheManager above:
    # training.caching imports pyarrow at module import time.
    from flowgym.training.caching import compute_batch_keys  # noqa: PLC0415

    keys = compute_batch_keys(batch)
    batch_size = len(keys)
    num_models = len(sub_models)

    epe_all = np.zeros((batch_size, num_models), dtype=np.float32)
    epe_mask = np.zeros((batch_size, num_models), dtype=bool)

    # Unified index lookup (O(1) per key)
    for i, k in enumerate(keys):
        k_int = int(k)
        if k_int in unified_epe_index:
            epe_values, mask = unified_epe_index[k_int]
            epe_all[i] = epe_values
            epe_mask[i] = mask

    # Compute misses on demand if requested and ground truth available
    if compute_misses and np.any(~epe_mask):
        flow_fields = getattr(batch, "flow_fields", None)
        if flow_fields is None:
            logger.debug("Skipping miss computation: no ground truth in batch")
            return {"epe_all": epe_all, "epe_mask": epe_mask}

        missing_root_dir_warned = False
        for i, (model, t_state) in enumerate(
            zip(sub_models, sub_model_states, strict=True)
        ):
            miss_pos = np.where(~epe_mask[:, i])[0]
            if miss_pos.size == 0:
                continue

            # Compute via sub-model inference
            sub_payload = model.enrich(batch, miss_pos, trainable_state=t_state)
            if sub_payload is None or "epe" not in sub_payload:
                continue

            epe_vals = np.asarray(sub_payload["epe"])
            if epe_vals.ndim == 0 or epe_vals.shape[0] != miss_pos.size:
                raise ValueError(
                    "Sub-model payload 'epe' must align with misses: "
                    f"expected first dimension {miss_pos.size}, got "
                    f"{epe_vals.shape}"
                )
            epe_all[miss_pos, i] = epe_vals
            epe_mask[miss_pos, i] = True

            # Update the unified in-memory index immediately so later batches
            # in the same run can hit without requiring reload/rebuild.
            for rel_idx, pos in enumerate(miss_pos):
                key_int = int(keys[pos])
                existing = unified_epe_index.get(key_int)
                if existing is None:
                    epe_row = np.zeros(num_models, dtype=np.float32)
                    mask_row = np.zeros(num_models, dtype=bool)
                else:
                    epe_row = np.asarray(existing[0], dtype=np.float32).copy()
                    mask_row = np.asarray(existing[1], dtype=bool).copy()
                epe_row[i] = epe_vals[rel_idx]
                mask_row[i] = True
                unified_epe_index[key_int] = (epe_row, mask_row)

            # Best-effort write-back to cache
            suffix = model.get_cache_id_suffix(t_state)
            base = (
                cache_id_base
                if cache_id_base is not None
                else model.__class__.__name__
            )
            cache_id = f"{base}{suffix}"
            cm = sub_cache_managers.get(cache_id)

            if cm is None and allow_create_missing:
                cm = _get_or_create_sub_cache_manager(
                    root_dir=root_dir,
                    cache_id=cache_id,
                    sub_cache_managers=sub_cache_managers,
                    payload=sub_payload,
                    miss_count=miss_pos.size,
                )
                if cm is None and not missing_root_dir_warned:
                    logger.warning(
                        "Skipping persistent sub-cache creation because "
                        "root_dir is unavailable."
                    )
                    missing_root_dir_warned = True

            if cm is not None:
                try:
                    if all(k in sub_payload for k in cm.spec):
                        cm.write(
                            keys[miss_pos],
                            {k: np.asarray(sub_payload[k]) for k in cm.spec},
                        )
                except Exception:
                    logger.debug(
                        f"Cache write-back failed for {cache_id} "
                        "(non-critical, continuing)"
                    )

    return {"epe_all": epe_all, "epe_mask": epe_mask}
