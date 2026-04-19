# Flow evaluation

This example is the first full repository-backed workflow in the ladder.
After the direct estimator example, this is the next step when you want to
run FlowGym the same way the repository does.

## What it does

- generates a tiny synthetic `.npy` flow dataset in a temporary directory
- writes local dataset and model YAML files
- runs `src/main.py --mode eval`
- prints the command and the evaluation output

## Script

- `examples/01_flow_eval.py`

Run it from the repository root:

```bash
uv run python examples/01_flow_eval.py
```

## What to notice

- The example is self-contained and does not depend on `/shared/...` paths.
- It uses the same `src/main.py` entrypoint as real evaluation runs.
- The dataset config and model config are the main interface between repo
  workflows and estimator code.

## Related pages

- [First estimate](first-estimate.md)
- [Training and evaluation workflows](../user-guide/training-and-evaluation.md)
- [Configuration and data flow](../user-guide/configuration-and-data.md)
