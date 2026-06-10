# Caching examples

These pages document two example scripts that exercise Flow Gym's
cache-backed evaluation workflow:

- `examples/10_caching.py` for a `dis_jax` estimator
- `examples/11_caching.py` for a `raft_jax` estimator

Both scripts run the same evaluation twice:

1. once with a cold cache
2. once with a warm cache

The point is to show what changes when cached evaluation artifacts are
reused instead of recomputed.

## Scripts

- `examples/10_caching.py`
- `examples/11_caching.py`

## How to run them

These examples assume a local repository checkout with development
dependencies available:

```bash
uv sync --group dev
```

Run the DIS example with:

```bash
uv run python examples/10_caching.py
```

Run the RAFT example with:

```bash
uv run python examples/11_caching.py
```

## What the scripts do

Both scripts:

- create temporary `.mat` files locally
- write temporary dataset and estimator YAML files
- invoke `src.main` in evaluation mode
- measure the duration of two runs

In the first run, the dataset config uses:

```yaml
caching:
  warm_start: index
```

In the second run, the script rewrites the dataset config to use:

```yaml
caching:
  warm_start: all
```

That change is the key action in both examples. The first run builds cache
artifacts. The second run tries to load and reuse them.

## The DIS example

`examples/10_caching.py` uses a small `dis_jax` estimator configuration and a
cache id named `dis_example_cache`.

Use this script when you want the simplest caching walkthrough and the
smallest amount of estimator-specific detail.

## The RAFT example

`examples/11_caching.py` uses `raft_jax` with a reduced configuration so
the example is lighter on memory and runtime than a larger RAFT setup.

Use this script when you want to see the same caching pattern with a
learning-based estimator.

The RAFT script also checks the output for a cache-id update message, which
is useful when you want to verify how caching interacts with estimator-specific
configuration.

## What to look for in the output

When the scripts run successfully, you should see:

- output from the first evaluation run
- output from the second evaluation run
- the duration of each run
- a final comparison that reports whether the warm-cache run was faster

For the RAFT example, you may also see a message confirming that the cache
id was updated with an estimator-specific suffix.

These scripts are meant to be practical references for the real example
files, so the most important thing to compare is the difference between the
first and second run.

## Collecting caches from the command line

The example scripts above generate their own data and configs. To fill a
cache for your own estimator and dataset, you do not need a bespoke script:
the standard evaluation entry point writes the cache for any estimator that
implements `enrich()` (DIS, RAFT, openpiv, art_of_piv, …).

Put the cache `spec` in the dataset config (this is the only
estimator-specific part — it must match what the estimator's `enrich()`
returns):

```yaml
caching:
  spec:
    epe: [float32, []]
    relative_epe: [float32, []]
  warm_start: index
```

Then point `--cache-root` (and optionally `--cache-id`) at the output. These
flags supply the cache location, so the same dataset config can fill many
caches without edits:

```bash
uv run python src/main.py --mode eval \
    --estimator estimator.yaml --dataset dataset.yaml \
    --cache-root caches/
```

The cache lands in `caches/<cache_id>/`, where `<cache_id>` is the base id
plus the estimator's own config/weights suffix (from `get_cache_id_suffix`).

## Sweeping many configs

`scripts/collect_cache.py` runs the command above once per estimator config,
into a shared `--cache-root`. Each config gets its own `<cache_id>` subdir,
runs as an isolated subprocess (so JIT/GPU memory is released between
configs), and the sweep continues past a failing config:

```bash
uv run python scripts/collect_cache.py \
    --models estimators/*.yaml \
    --dataset dataset.yaml \
    --cache-root caches/
```

Nothing in either step is specific to an algorithm — the dataset config's
`spec` is the only knob that depends on the estimator.

## Selecting an ensemble from a cache

`scripts/select_ensemble.py` consumes a directory of such caches and picks a
size-`K` subset minimizing the aggregate per-image best error (greedy, plus
an exact MILP for the mean aggregator with `--exact`):

```bash
uv run python scripts/select_ensemble.py \
    --cache-root caches/ --K 3 --metric epe
```

`--metric` selects the parquet error column (default `epe`). Timing is
optional: pass `--time-limit` only if each cache dir carries a `timing.json`
(timing is device-dependent and collected separately from the
device-independent error).

## Related docs

- [Flow evaluation](flow-eval.md)
- [Evaluation and caching API](../api/evaluation.md)
- [Configuration and data flow](../user-guide/configuration-and-data.md)
