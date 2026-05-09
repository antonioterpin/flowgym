# example_experiment

Compact reference for `experiments/run.py`: one experiment folder, one
self-contained synthetic dataset, two model configs, one seed.

## Layout

```text
experiments/example_experiment/
├── exp.yaml
├── dataset.yaml
├── models/
│   ├── dummy_a.yaml
│   └── dummy_b.yaml
└── _generated/   # populated automatically, gitignored
```

## Run

From the repository root:

```bash
uv run python experiments/run.py --exp example_experiment
```

This expands the matrix and calls `src/main.py` once per `(seed, model)`
pair. Run names are composed from `study.tags` plus seed; the W&B run
records the commit and `diff.patch` (untracked-but-not-ignored files
are included via a temporary git index).
