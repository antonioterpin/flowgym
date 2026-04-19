# Caching examples

These pages document two example scripts that exercise FlowGym's
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

## Related docs

- [Flow evaluation](flow-eval.md)
- [Evaluation and caching API](../api/evaluation.md)
- [Configuration and data flow](../user-guide/configuration-and-data.md)
