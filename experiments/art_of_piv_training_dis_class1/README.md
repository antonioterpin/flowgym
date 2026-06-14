# art_of_piv_training_modes

Compare ArtOfPIV training modes with a shared estimator set from
`infra_smoke_v0`:

- classification: softmax cross-entropy
- regression: huber, l2
- bandits:
  - loss: huber, l2
  - exploration:
    - epsilon_greedy + linear(1.0 -> 0.0, 450 steps)
    - softmax + linear(10.0 -> 0.0, 450 steps)

## Run

From repository root:

```bash
uv run python experiments/run.py --exp art_of_piv_training_dis_class1
```

Use debug mode to skip clean-git commit checks:

```bash
uv run python experiments/run.py --exp art_of_piv_training_dis_class1 --debug
```

## Reuse choices

- Copies `dataset.yaml` and `validation_dataset.yaml` from
  `experiments/infra_smoke_v0` into this folder.
- Copies DIS sub-estimator list into
  `experiments/art_of_piv_training_dis_class1/models/dis_models.yaml`.
- Reuses router and optimizer settings from `infra_smoke_v0` model configs.
