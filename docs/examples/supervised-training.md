# Supervised training

This example runs a small supervised-training workflow through the normal
repository entrypoint.

It includes validation and checkpointing, so the run exercises the parts of
the training path that matter most in practice.

## What it does

- generates tiny synthetic training and validation datasets locally
- writes dataset and estimator YAML files
- includes `validation` settings in the dataset config
- runs `src/main.py --mode train-supervised`
- prints the checkpoint directory written during the run

## Script

- `examples/02_supervised_training.py`

Run it from the repository root:

```bash
uv run python examples/02_supervised_training.py
```

## What to notice

- This walkthrough uses the lightweight `dummy` estimator so the example can
  focus on the supervised-training workflow itself: validation cadence,
  checkpoint writing, and config wiring.
- Validation is wired through the dataset config, not a separate CLI flag.
- Checkpoints are written under the configured `out_dir`.
- This is the best example to study before changing training orchestration,
  validation cadence, or checkpoint behavior.

## Related pages

- [Flow evaluation](flow-eval.md)
- [Architecture guide](../guides/architecture.md)
- [Contributing guide](../guides/contributing.md)
