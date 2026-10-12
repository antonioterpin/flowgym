# example_experiment

Compact reference for `experiments/run.py`: one experiment folder, one
self-contained synthetic dataset (random flow fields generated in memory by
synthpix's `kinematic` source), two model configs, one seed.

## Layout

```text
experiments/example_experiment/
├── exp.yaml
├── dataset.yaml
├── models/
│   ├── dummy_a.yaml
│   └── dummy_b.yaml
├── results.json  # written after the runs
└── _generated/   # populated automatically, gitignored
```

## Run

From the repository root:

```bash
uv run --extra wandb python experiments/run.py --exp example_experiment
```

This expands the matrix and calls `src/main.py` once per `(seed, model)`
pair. Run names are composed from `study.tags` plus seed; the W&B run
records the commit and `diff.patch` (untracked-but-not-ignored files
are included via a temporary git index).

Each run writes its final summary to `_generated/summaries/<model>/seed<N>.json`
(`--summary-out`); `results.json` keeps the metrics listed under `metrics:`
in `exp.yaml` (default `[mean_epe]`; summaries also have `median_epe`,
`p95_epe`, `p99_epe`, `max_epe`, `min_epe` over frames) per model, with the
mean and population std over seeds.

## Per-model overrides

A model config may set `dataset:` (a path relative to the experiment
folder) or `mode:` (e.g. `eval`) to override the `exp.yaml` value for
that model's runs, e.g. to evaluate baselines in a training experiment.
`dataset_overrides:` is deep-merged into the model's dataset config, e.g.
`{batch_size: 64, validation: {dataset: {batch_size: 64}}}`.
