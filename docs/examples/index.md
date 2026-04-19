# Example workflows

This section collects small, runnable examples of the main FlowGym
workflows.

They cover complete paths through the code, from a single in-memory
estimate to configuration-driven evaluation, training, and caching runs.

## How to use this section

- Start with [First estimate](first-estimate.md) when you want one concrete
  estimator run before touching YAML or CLI orchestration.
- Move to [Flow evaluation](flow-eval.md) when you want the first
  config-backed repository workflow.
- Use [Supervised training](supervised-training.md) and
  [Density evaluation](density-eval.md) when you need the next real
  workflows, not just API snippets.
- Treat [Caching examples](caching.md) as an advanced optimization topic
  after the core eval/training paths make sense.

## Available example pages

- [First estimate](first-estimate.md):
  instantiate a `dis_jax` estimator and run one estimate on a small image
  pair in memory.
- [Flow evaluation](flow-eval.md):
  tiny synthetic dataset plus `src/main.py --mode eval`.
- [Supervised training](supervised-training.md):
  tiny synthetic training run with validation and checkpoints.
- [Density evaluation](density-eval.md):
  same evaluation workflow using a density estimator.
- [Caching examples](caching.md):
  cold/warm cache reuse with the existing DIS and RAFT demos.

## Repository scripts worth knowing

- `examples/00_first_estimate.py`:
  direct estimator API bridge between Getting Started and repo workflows.
- `examples/01_flow_eval.py`:
  first config-backed evaluation example.
- `examples/02_supervised_training.py`:
  smallest training walkthrough with validation and checkpoints.
- `examples/03_density_eval.py`:
  density workflow through the same CLI path.
- `examples/10_caching.py`:
  DIS-based cache reuse on temporary `.mat` data.
- `examples/11_caching.py`:
  RAFT-based cache reuse on temporary `.mat` data.
- `src/main.py`:
  the CLI entrypoint used by the workflow examples.

```{toctree}
:maxdepth: 1
:titlesonly:

first-estimate
flow-eval
supervised-training
density-eval
caching
```
