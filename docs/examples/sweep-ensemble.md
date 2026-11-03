# Sweep, time, and select an ensemble

`examples/14_cache_sweep_ensemble.py` demonstrates the three
estimator-agnostic sweep tools working together, end to end, on a small
fully synthetic dataset (no downloads; runs on CPU):

1. **`scripts/collect_cache.py`** — fill a per-image **error** cache for many
   model configs over one dataset. The error is device-independent, so the
   cache is reproducible and reusable across machines.
2. **`scripts/collect_timing.py`** — write a per-config **`timing.json`**
   (device-dependent inference time, measured `timeit`-style at batch size 1
   with the input pre-loaded on the device) into the *same* cache dirs.
3. **`scripts/select_ensemble.py`** — pick a size-`K` subset minimizing the
   per-image best error, optionally subject to a latency bound, and export
   the chosen configs as a collect-ready `estimators_list` YAML.

## How to run

```bash
uv sync --group dev
uv run python examples/14_cache_sweep_ensemble.py
```

## What the script does

- builds a tiny synthetic dataset (synthpix generates particle images from a
  handful of flow fields) with a `caching` block declaring the error `spec`;
- writes three `dis_jax` model configs differing only in `patch_size`;
- runs `collect_cache.py` over them, then `collect_timing.py`, then
  `select_ensemble.py --K 2 --export-models`.

## Why the three are inter-compatible

Each candidate config maps to a single cache directory
`<cache-root>/<cache_id>/`, where `cache_id` is the base id (from the dataset
config or `--cache-id-base`) plus the estimator's `get_cache_id_suffix` (a
hash of its config). `collect_cache.py` writes the error parquet under
`data/` there; `collect_timing.py` writes `timing.json` there; and
`select_ensemble.py` reads **both** from that directory. So as long as the two
collectors share the same `--cache-root` and base id, error and timing land
together automatically:

```text
=== cache layout (error + timing co-located per config) ===
  demo-sweep_c5a09d90c: error=True timing=True
  demo-sweep_cc712e59d: error=True timing=True
  demo-sweep_cd0430e37: error=True timing=True
```

`timing.json` is optional: without a finite `--time-limit`, selection ranks
purely on the device-independent error and ignores timing entirely.

## Closing the loop

`select_ensemble.py --export-models chosen.yaml` writes the picked subset in
the `estimators_list` format, which feeds straight back into a full-set run:

```bash
scripts/collect_cache.py --estimators-list chosen.yaml --dataset full.yaml \
    --cache-root full_caches/
```
