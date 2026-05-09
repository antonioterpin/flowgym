# Example workflows

Small, runnable examples of the main FlowGym workflows. They cover
complete paths through the code, from a single in-memory estimate to
configuration-driven evaluation, training, and caching runs.

Each page documents one script under `examples/` and is listed in the
order most readers will want to follow.

- [First estimate](first-estimate.md) — `examples/00_first_estimate.py`.
  Instantiate a `dis_jax` estimator and run one estimate on a small
  image pair in memory. The cleanest place to see the public API loop
  before touching YAML or CLI orchestration.
- [Flow evaluation](flow-eval.md) — `examples/01_flow_eval.py`. The
  first config-backed repository workflow: a tiny synthetic dataset
  plus `src/main.py --mode eval`.
- [Supervised training](supervised-training.md) —
  `examples/02_supervised_training.py`. Tiny synthetic training run
  with validation and checkpoints.
- [Density evaluation](density-eval.md) — `examples/03_density_eval.py`.
  The same evaluation workflow using a density estimator.
- [Caching examples](caching.md) — `examples/10_caching.py` and
  `examples/11_caching.py`. Cold/warm cache reuse with the DIS and RAFT
  demos. Treat as an advanced optimization topic after the core
  eval/training paths make sense.

`src/main.py` is the CLI entrypoint these workflows invoke.

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:

first-estimate
flow-eval
supervised-training
density-eval
caching
```
