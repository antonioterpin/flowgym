# Density evaluation

This example shows that FlowGym's orchestration layer is not only for flow
fields. The same evaluation path also works for density estimators.

## What it does

- generates a tiny synthetic dataset locally
- uses the `simple` density estimator instead of a flow estimator
- runs `src/main.py --mode eval`
- prints the resulting density-oriented evaluation output

## Script

- `examples/03_density_eval.py`

Run it from the repository root:

```bash
uv run python examples/03_density_eval.py
```

## What to notice

- The repo-level flow stays the same: configs, CLI entrypoint, evaluation.
- The main difference is the estimator family and `estimate_type: density`.
- This page is the quickest way to see how density fits into the same mental
  model as flow evaluation.

## Related pages

- [Flow evaluation](flow-eval.md)
- [Estimator API overview](../user-guide/estimator-api.md)
- [Density estimators](../api/estimators.md#density-estimators)
