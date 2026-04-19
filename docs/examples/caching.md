# Caching examples

FlowGym ships two example scripts that demonstrate the repository's
cache-backed evaluation flow. They are lightweight examples of the same
orchestration path used by `src/main.py`, but they make the most sense after
you have already seen the basic evaluation workflow.

## What these examples show

- `examples/10_caching.py`:
  creates temporary `.mat` files, evaluates a `dis_jax` model, then reruns
  the same configuration with cache warm-start enabled.
- `examples/11_caching.py`:
  repeats the pattern with `raft_jax`, using a smaller configuration to
  reduce memory pressure.

Both examples:

- generate temporary input data
- write temporary dataset and model YAML files
- invoke `src/main.py --mode eval`
- compare a cold-cache run with a warm-cache run

## When to use them

- Use this page after [Flow evaluation](flow-eval.md), not before it.
- Use the DIS example when you want the simplest cache walkthrough.
- Use the RAFT example when you need to understand how cache keys interact
  with model-specific configuration.
- Use these examples as integration references, not as minimal package
  quickstarts.

## Prerequisites

These examples assume a local repository checkout with development
dependencies available:

```bash
uv sync --group dev
```

For docs work, you do not need to run them. They are here to document the
intended workflow and to provide a stable place to point readers who want a
real script.

## Related docs

- [First estimate](first-estimate.md)
- [Flow evaluation](flow-eval.md)
- [Getting started](../getting-started/index.md) for the minimal public API path
- [Evaluation and caching API](../api/evaluation.md) for module reference
- [Architecture guide](../guides/architecture.md) for where caching lives in
  the codebase
