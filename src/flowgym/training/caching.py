"""Caching layer for derived batch data (e.g., model estimates, metrics)."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import goggles as gg
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from flowgym.common.base.trainable_state import EstimatorTrainableState
from flowgym.types import CachePayload

if TYPE_CHECKING:
    from synthpix.types import SynthpixBatch

    from flowgym.common.base import Estimator

# shape_per_sample excludes the batch dimension.
# Example scalar: shape_per_sample=()
# Example vector length M: shape_per_sample=(M,)
CacheSpec: TypeAlias = dict[str, tuple[np.dtype, tuple[int, ...]]]

logger = gg.get_logger(__name__)


def compute_batch_keys(batch: SynthpixBatch) -> np.ndarray:
    """Compute cache keys for a batch.

    Handles both synthetic batches (use PRNG keys) and file-based batches
    (compute MD5 hashes of filenames).

    Args:
        batch: The SynthpixBatch to compute keys for.

    Returns:
        Array of uint64 keys, shape (B,).
    """
    files = getattr(batch, "files", None)
    params = getattr(batch, "params", None)

    if files is not None and params is None:
        # File-based: hash filenames
        hashes = [
            int(hashlib.md5(os.path.basename(f).encode()).hexdigest()[:16], 16)
            for f in files
        ]
        return np.array(hashes, dtype=np.uint64)
    else:
        # Synthetic: use PRNG keys
        keys_np = np.asarray(batch.keys)
        if keys_np.ndim > 1:
            # Reconstruct uint64 from (B, 2) uint32
            return (keys_np[..., 0].astype(np.uint64) << 32) | keys_np[
                ..., 1
            ].astype(np.uint64)
        return keys_np.astype(np.uint64)


class CacheManager:
    """Manager for storing and retrieving derived data for SynthPix batches.

    The CacheManager implements a "read-through" cache. It uses Parquet files
    for storage and supports multiple warm-start strategies to balance memory
    usage and lookup speed.

    It can also act as an aggregate view of multiple physical cache directories
    if initialized with a glob pattern for `cache_id`.
    """

    def __init__(
        self,
        root_dir: str,
        cache_id: str,
        spec: CacheSpec,
        warm_start: Literal["none", "index", "all"] = "index",
    ) -> None:
        """Initialize the cache manager.

        Args:
            root_dir: The root directory for all caches.
            cache_id: Unique identifier for this cache.
            spec: Specification of the data to be cached.
            warm_start: How to warm start the cache.
                "none": No warm start.
                "index": Load keys only (fast).
                "all": Load all data into memory (slow init, fast lookup).
        """
        self.root_dir = Path(root_dir)
        self.cache_id = cache_id

        # Determine source directories and write target
        if "*" in cache_id:
            self.source_dirs = list(self.root_dir.glob(cache_id))
            # Glob-based caches are read-only aggregate views
            self.cache_dir = self.root_dir / cache_id.replace("*", "aggregate")
            self.read_only = True
            num_dirs = len(self.source_dirs)
            logger.info(
                f"Initialized aggregate CacheManager from {num_dirs} "
                f"directories matching '{cache_id}'"
            )
        else:
            self.cache_dir = self.root_dir / cache_id
            self.source_dirs = [self.cache_dir]
            self.read_only = False

        self.spec = spec
        self.warm_start = warm_start

        # In-memory index of keys -> parquet file path/row location (optional)
        self.index: dict[int, tuple[str, int]] = {}
        # In-memory full cache (optional)
        self.all_data: dict[str, np.ndarray] | None = None

        # Buffer for pending writes
        self.pending_buffer: dict[str, list] = {
            "key": [],
            **{name: [] for name in self.spec},
        }

        # Initialize cache directory
        if not self.read_only:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._write_meta()

        if warm_start != "none":
            self._warm_start()

    def _write_meta(self) -> None:
        """Write metadata for the cache."""
        meta = {
            "cache_id": self.cache_id,
            "version": "1.0",
            "spec": {
                name: [str(dtype), list(shape)]
                for name, (dtype, shape) in self.spec.items()
            },
        }
        with open(self.cache_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=4)

    def _warm_start(self) -> None:
        """Warm start the cache."""
        files = []
        for d in self.source_dirs:
            files.extend(list(d.glob("data/*.parquet")))

        if not files:
            return

        num_files = len(files)
        logger.info(
            f"Warm-starting cache ({self.warm_start}) from {num_files} files..."
        )

        if self.warm_start == "all":
            # Load all data into memory
            all_raw: dict[str, list[Any]] = {
                "key": [],
                **{name: [] for name in self.spec},
            }
            for f in files:
                table = pq.read_table(f)
                all_raw["key"].extend(table["key"].to_numpy().tolist())
                for name in self.spec:
                    all_raw[name].extend(table[name].to_numpy().tolist())

            # Convert to numpy arrays for fast lookup
            self.all_data = {}
            self.all_data["key"] = np.array(all_raw["key"], dtype=np.uint64)
            for name in self.spec:
                self.all_data[name] = np.array(
                    all_raw[name], dtype=self.spec[name][0]
                )
            logger.info(
                f"Loaded {len(self.all_data['key'])} items into memory "
                f"({self.all_data['key'].nbytes / 1e6:.2f} MB)."
            )

        elif self.warm_start == "index":
            for f in files:
                table = pq.read_table(f, columns=["key"])
                keys = table["key"].to_numpy()
                file_path = str(f)
                for i, k in enumerate(keys):
                    self.index[int(k)] = (file_path, i)
            logger.info(f"Loaded {len(self.index)} keys into index.")

    def _lookup_pending_buffer(
        self,
        keys_int: list[int],
        hit_mask: np.ndarray,
        payload_out: dict[str, np.ndarray],
    ) -> None:
        """Check pending buffer for cache hits.

        Updates hit_mask and payload_out in-place for keys found in the
        pending buffer.

        Args:
            keys_int: List of integer keys to lookup.
            hit_mask: Boolean array marking which keys were found.
            payload_out: Output payload dictionary to fill.
        """
        for i, k in enumerate(keys_int):
            if k in self.pending_buffer["key"]:
                idx = self.pending_buffer["key"].index(k)
                for name in self.spec:
                    payload_out[name][i] = self.pending_buffer[name][idx]
                hit_mask[i] = True

    def _lookup_memory_cache(
        self,
        keys_u64: np.ndarray,
        keys_int: list[int],
        key_to_input_idxs: dict[int, list[int]],
        hit_mask: np.ndarray,
        payload_out: dict[str, np.ndarray],
    ) -> None:
        """Check in-memory cache for cache hits.

        Updates hit_mask and payload_out in-place for keys found in the
        all_data cache if present.

        Args:
            keys_u64: uint64 array of keys to lookup.
            keys_int: List of integer keys.
            key_to_input_idxs: Mapping from keys to input indices.
            hit_mask: Boolean array marking which keys were found.
            payload_out: Output payload dictionary to fill.
        """
        if self.all_data is None:
            return
        found_idx = np.where(np.isin(self.all_data["key"], keys_u64))[0]
        if len(found_idx) == 0:
            return
        found_key_to_raw_idx: dict[int, int] = {}
        for raw_idx in found_idx.tolist():
            k = int(self.all_data["key"][raw_idx])
            if k not in found_key_to_raw_idx:
                found_key_to_raw_idx[k] = raw_idx
        for k, input_idxs in key_to_input_idxs.items():
            raw_idx = found_key_to_raw_idx.get(k)
            if raw_idx is None:
                continue
            for j in input_idxs:
                if hit_mask[j]:
                    continue
                for name in self.spec:
                    payload_out[name][j] = self.all_data[name][raw_idx]
                hit_mask[j] = True

    def _lookup_index(
        self,
        keys_int: list[int],
        hit_mask: np.ndarray,
        payload_out: dict[str, np.ndarray],
    ) -> None:
        """Check index-based cache for cache hits.

        Reads parquet files referenced by the index for cached keys.
        Updates hit_mask and payload_out in-place.

        Args:
            keys_int: List of integer keys to lookup.
            hit_mask: Boolean array marking which keys were found.
            payload_out: Output payload dictionary to fill.
        """
        if not self.index or self.all_data is not None:
            return
        by_file: dict[str, list[tuple[int, int]]] = {}
        for i, k in enumerate(keys_int):
            if hit_mask[i]:
                continue
            loc = self.index.get(k)
            if loc is not None:
                file_path, row_idx = loc
                by_file.setdefault(file_path, []).append((i, row_idx))
        for file_path, pairs in by_file.items():
            try:
                table = pq.read_table(file_path, columns=list(self.spec))
            except Exception:
                continue
            input_idxs = [i for i, _ in pairs]
            row_idxs = [r for _, r in pairs]
            if not row_idxs:
                continue
            sub = table.take(pa.array(row_idxs, type=pa.int64()))
            for name in self.spec:
                values = sub[name].to_pylist()
                for i, v in zip(input_idxs, values, strict=True):
                    if not hit_mask[i]:
                        payload_out[name][i] = v
                        hit_mask[i] = True

    def _process_parquet_file(
        self,
        table: pa.Table,
        payload_out: dict[str, np.ndarray],
        hit_mask: np.ndarray,
        key_to_input_idxs: dict[int, list[int]],
    ) -> set[int]:
        """Process a single parquet file and update cache results.

        Updates hit_mask and payload_out in-place for found keys.

        Args:
            table: Parquet table to process.
            payload_out: Output payload dictionary to fill.
            hit_mask: Boolean array marking which keys were found.
            key_to_input_idxs: Mapping from keys to input indices.

        Returns:
            Set of keys that were found in this file.
        """
        found_keys_int = [int(k) for k in table["key"].to_numpy().tolist()]
        row_by_key: dict[int, int] = {}
        for row_idx, k in enumerate(found_keys_int):
            if k not in row_by_key:
                row_by_key[k] = row_idx
        if not row_by_key:
            return set()
        found_data = {name: table[name].to_pylist() for name in self.spec}
        found_keys = set()
        for k, row_idx in row_by_key.items():
            input_idxs = key_to_input_idxs.get(k, [])
            for i in input_idxs:
                if not hit_mask[i]:
                    for name in self.spec:
                        payload_out[name][i] = found_data[name][row_idx]
                    hit_mask[i] = True
            if any(hit_mask[i] for i in input_idxs):
                found_keys.add(k)
        return found_keys

    def _lookup_disk(
        self,
        keys_int: list[int],
        hit_mask: np.ndarray,
        payload_out: dict[str, np.ndarray],
        key_to_input_idxs: dict[int, list[int]],
    ) -> None:
        """Check disk cache files for cache hits.

        Scans parquet files from disk for cached keys. Updates hit_mask and
        payload_out in-place for found keys.

        Args:
            keys_int: List of integer keys to lookup.
            hit_mask: Boolean array marking which keys were found.
            payload_out: Output payload dictionary to fill.
            key_to_input_idxs: Mapping from keys to input indices.
        """
        try:
            source_files = []
            for d in self.source_dirs:
                data_dir = d / "data"
                if data_dir.exists():
                    source_files.extend(data_dir.glob("*.parquet"))
            if not source_files:
                return
            source_files = sorted(
                source_files,
                key=lambda p: (p.stat().st_mtime_ns, str(p)),
                reverse=True,
            )
            remaining_keys = {
                k for i, k in enumerate(keys_int) if not hit_mask[i]
            }
            for f in source_files:
                if not remaining_keys:
                    break
                table = pq.read_table(
                    f,
                    columns=["key", *self.spec.keys()],
                    filters=[("key", "in", list(remaining_keys))],
                )
                if table.num_rows == 0:
                    continue
                found_keys = self._process_parquet_file(
                    table, payload_out, hit_mask, key_to_input_idxs
                )
                remaining_keys -= found_keys
                if np.all(hit_mask):
                    break
        except Exception as e:
            logger.debug(
                f"Lookup info: cache directory {self.cache_dir / 'data'} "
                f"not available or empty: {e}"
            )

    def lookup(
        self, keys: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Lookup multiple keys in the cache.

        Lookups proceed in three stages:
        1. **Pending Buffer**: Check recently written items not yet flushed.
        2. **Memory Cache**: If `warm_start="all"`, check the in-memory array.
        3. **Disk**: Query the Parquet files using Arrow's native filtering.

        Args:
            keys: Array of keys to lookup (B,) uint64 or (B, 2) uint32.

        Returns:
            A tuple (payload, hit_mask). `payload` mirrors the `CacheSpec` and
            `hit_mask` is a boolean array indicating which keys were found.
        """
        # Normalize keys to uint64
        keys = np.asarray(keys)
        if keys.ndim > 1:
            keys_u64 = (keys[..., 0].astype(np.uint64) << 32) | keys[
                ..., 1
            ].astype(np.uint64)
        else:
            keys_u64 = keys.astype(np.uint64)
        keys_val = keys_u64.tolist()
        keys_int: list[int] = (
            keys_val if isinstance(keys_val, list) else [keys_val]
        )
        key_to_input_idxs: dict[int, list[int]] = {}
        for j, k in enumerate(keys_int):
            key_to_input_idxs.setdefault(int(k), []).append(j)
        hit_mask = np.zeros(len(keys_u64), dtype=bool)
        payload_out = {
            name: np.zeros(
                (len(keys_u64), *self.spec[name][1]), dtype=self.spec[name][0]
            )
            for name in self.spec
        }
        self._lookup_pending_buffer(keys_int, hit_mask, payload_out)
        if np.all(hit_mask):
            return payload_out, hit_mask
        self._lookup_memory_cache(
            keys_u64, keys_int, key_to_input_idxs, hit_mask, payload_out
        )
        if np.all(hit_mask):
            return payload_out, hit_mask
        self._lookup_index(keys_int, hit_mask, payload_out)
        if np.all(hit_mask):
            return payload_out, hit_mask
        self._lookup_disk(keys_int, hit_mask, payload_out, key_to_input_idxs)
        return payload_out, hit_mask

    def write(self, keys: np.ndarray, payload: dict[str, np.ndarray]) -> None:
        """Buffering write to the cache.

        Args:
            keys: Array of keys.
            payload: Dictionary of data arrays aligned with keys.
        """
        keys = np.asarray(keys)
        if keys.ndim > 1:
            # Reconstruct uint64 from (B, 2) uint32
            keys_u64 = (keys[..., 0].astype(np.uint64) << 32) | keys[
                ..., 1
            ].astype(np.uint64)
            keys = keys_u64

        keys_list = [int(k) for k in keys.tolist()]
        self.pending_buffer["key"].extend(keys_list)
        for name in self.spec:
            # We convert to list for appending
            data = payload[name]
            if isinstance(data, (np.ndarray, jnp.ndarray)):
                data = data.tolist()
            self.pending_buffer[name].extend(data)

        # Amortized flush check (e.g. every 1000 items)
        if len(self.pending_buffer["key"]) >= 1000:
            self.flush()

    def flush(self) -> None:
        """Flush the pending buffer to disk as a new Parquet file."""
        if not self.pending_buffer["key"] or self.read_only:
            return

        filename = (
            self.cache_dir / "data" / f"part-{uuid.uuid4().hex[:8]}.parquet"
        )
        filename.parent.mkdir(parents=True, exist_ok=True)

        arrays = {"key": pa.array(self.pending_buffer["key"], type=pa.uint64())}
        for name, _ in self.spec.items():
            arrays[name] = pa.array(self.pending_buffer[name])

        table = pa.table(arrays)
        pq.write_table(table, filename)

        num_items = len(self.pending_buffer["key"])
        logger.info(f"Flushed {num_items} items to {filename.name}")

        # Clear buffer
        self.pending_buffer = {
            "key": [],
            **{name: [] for name in self.spec},
        }

    def __enter__(self) -> CacheManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.flush()

    def close(self) -> None:
        """Close the cache manager and flush pending writes."""
        self.flush()


def enrich_batch(
    batch: SynthpixBatch,
    model: Estimator,
    *,
    cache_manager: CacheManager | None = None,
    trainable_state: EstimatorTrainableState | None = None,
) -> CachePayload:
    """Enrich a batch with cached data.

    This function orchestrates the read-through cache flow:
    1. Lookup cached entries via CacheManager
    2. Call model.enrich() for missing entries
    3. Write newly computed entries to cache
    4. Return complete payload

    The estimator decides whether to compute misses or return None.

    Args:
        batch: The batch to enrich.
        model: The estimator to use for enrichment.
        cache_manager: Optional CacheManager for storage.
        trainable_state: Current trainable state (passed to model.enrich).

    Returns:
        A CachePayload with enriched data.

    Raises:
        Exception: If model.enrich() or cache_manager.write() raise.
    """
    if cache_manager is None:
        return CachePayload()

    # Compute cache keys for the batch
    keys_np_for_lookup = compute_batch_keys(batch)

    # Check for batch.mask to filter out padding entries
    mask = getattr(batch, "mask", None)
    if mask is not None:
        mask_np = np.asarray(mask)
        valid_indices = np.where(mask_np)[0]
        keys_for_lookup = keys_np_for_lookup[valid_indices]
        batch_size = len(keys_np_for_lookup)
    else:
        valid_indices = np.arange(len(keys_np_for_lookup))
        keys_for_lookup = keys_np_for_lookup
        batch_size = len(keys_np_for_lookup)

    # 1. Lookup (only for valid entries)
    payload_valid, hit = cache_manager.lookup(keys_for_lookup)

    # 2. Handle misses (only for valid entries)
    if not np.all(hit):
        miss_idxs_relative = np.where(~hit)[0]
        miss_idxs_absolute = valid_indices[miss_idxs_relative]

        # Call model.enrich for missing entries
        try:
            payload_miss = model.enrich(
                batch, miss_idxs_absolute, trainable_state=trainable_state
            )
        except Exception as e:
            logger.error(f"Error in model.enrich: {e}")
            logger.error(f"miss_idxs: {miss_idxs_absolute}")
            raise e

        if payload_miss is not None:
            # Merge into valid payload
            for name in cache_manager.spec:
                if name in payload_miss:
                    payload_valid[name][miss_idxs_relative] = payload_miss[name]

            # Write back to cache
            try:
                cache_manager.write(
                    keys_for_lookup[miss_idxs_relative], payload_miss
                )
            except Exception as e:
                logger.error(f"Error in CacheManager.write: {e}")
                logger.error(
                    f"keys_for_lookup shape: {np.shape(keys_for_lookup)}"
                )
                logger.error(f"miss_idxs_relative: {miss_idxs_relative}")
                raise e

    # 3. Expand payload to full batch size (including padding)
    if mask is not None:
        payload = {}
        for name, (dtype, shape) in cache_manager.spec.items():
            full_shape = (batch_size, *shape)
            full_array = np.zeros(full_shape, dtype=dtype)
            full_array[valid_indices] = payload_valid[name]
            payload[name] = full_array
    else:
        payload = payload_valid

    return CachePayload.from_enrich_result(payload)


def build_cache_id(
    model: Estimator,
    trainable_state: Any,
    caching_config: dict,
) -> str:
    """Build a cache ID from model, state, and config.

    Args:
        model: The estimator to build cache ID for.
        trainable_state: Current trainable state (for suffix generation).
        caching_config: Caching configuration dict (contains base cache_id).

    Returns:
        The constructed cache ID string.
    """
    suffix = model.get_cache_id_suffix(trainable_state)
    if "cache_id" in caching_config:
        base = caching_config["cache_id"]
        return f"{base}{suffix}" if suffix else base
    return f"{model.__class__.__name__}{suffix}"
