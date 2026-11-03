# Cache README

This document defines the operational assumptions and responsibilities of the
caching system used during training and evaluation.

## Scope

Caching is coordinated by:

- `CacheManager`: storage/index/lookup/write implementation.
- `enrich_batch(...)`: read-through orchestration.
- `Estimator.enrich(...)`: model hook that computes payload for cache misses.
- `CachePayload`: normalized payload passed into estimator execution.

## Ownership split

### Estimator responsibilities

- Define what is cached by implementing `Estimator.enrich(batch, miss_idxs, **kwargs)`.
- Return payload arrays aligned with `miss_idxs` order.
- Optionally use `trainable_state` from `kwargs` when cache values depend on model weights.
- Return `None` when no miss payload is produced.

### CacheManager responsibilities

- Persist and retrieve cache rows from Parquet.
- Perform key lookup using in-memory buffer, optional warm-start memory/index, then disk.
- Buffer writes and flush to disk.
- Validate data shape compatibility via configured `spec`.

### `enrich_batch(...)` responsibilities

- Compute batch keys.
- Restrict lookup/write to valid (unmasked) samples.
- Lookup hits via `CacheManager`.
- For misses, call `model.enrich(...)` and merge returned payload.
- Write miss payload back through `CacheManager`.
- Expand payload back to full batch shape when masks are present.
- Return normalized `CachePayload`.

## Key assumptions

### Cache keys

- Synthetic batches: keys come from `batch.keys`.
- File-based batches (`files` present and `params` absent): keys are derived from
  `md5(basename(file))[:16]` and stored as `uint64`.
- Key format is normalized to `uint64` internally.

### Payload schema

- Cache storage schema is declared by `CacheManager.spec`:
  `dict[name, (dtype, shape_per_sample)]`.
- Stored arrays must be aligned with queried keys.
- `CachePayload.from_enrich_result(...)` maps known keys:
  - `epe` or `errors` -> `CachePayload.epe`
  - `relative_epe`, `epe_all`, `epe_mask`, `estimates`
  - other keys -> `CachePayload.extras`

### Miss behavior

- Misses are computed only via `Estimator.enrich(...)`.
- If miss payload is `None`, miss entries remain default-initialized (zeros by spec dtype/shape).
- Only keys returned in miss payload are merged and written.
- If duplicate rows for the same key exist across parquet parts, disk lookup
  applies a deterministic policy: prefer the newest part by file mtime.

### Mask handling

- Lookups and miss computation operate only on `mask=True` rows.
- Returned payload is expanded to original batch size with zeros for masked rows.

## Runtime flow

1. Build keys for current batch.
2. Lookup cache rows for valid rows.
3. Compute misses via `Estimator.enrich(...)`.
4. Merge and write miss payload.
5. Convert merged dict to `CachePayload`.
6. Pass `CachePayload` into estimator call path.

## Write semantics

- Writes are buffered in memory (`pending_buffer`).
- Buffer flush occurs:
  - automatically when threshold is reached, or
  - on `flush()`, `close()`, or context exit (`__exit__`).
- Aggregate cache mode (`cache_id` contains `*`) is read-only.

## Configuration assumptions

- Cache identity should include model-specific suffix when cache values depend on
  model config/weights (`Estimator.get_cache_id_suffix(...)`).
- Changing model behavior without changing cache id may cause stale cache usage.
