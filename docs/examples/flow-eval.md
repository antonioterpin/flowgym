# Flow evaluation

This example runs a small repository-backed flow evaluation end to end.

It uses local temporary files and the normal `src/main.py --mode eval`
entrypoint, so you can see the full evaluation path without depending on
shared datasets.

## What it does

- generates a tiny synthetic `.npy` flow dataset in a temporary directory
- writes local dataset and estimator YAML files
- runs `src/main.py --mode eval`
- prints the command and the evaluation output

## Script

- `examples/01_flow_eval.py`

Run it from the repository root:

```bash
uv run python examples/01_flow_eval.py
```

## What to notice

- It uses the same `src/main.py` entrypoint as real evaluation runs.
- The dataset config and estimator config are the main interface between repo
  workflows and estimator code.

## Related pages

- [First estimate](first-estimate.md)
- [Training and evaluation workflows](../user-guide/training-and-evaluation.md)
- [Configuration and data flow](../user-guide/configuration-and-data.md)
