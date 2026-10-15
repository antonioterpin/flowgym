# art_of_piv: main-table results

Reproduces the main-table rows of the ART of PIV paper. Each row trains the
router on a fixed library of estimators and compares it with the best single
estimator under the same latency budget.

| Row | Paper key | Library | Paper AEPE (best single) |
|---|---|---|---|
| `dis_20hz` | `disTwentyHz` | 10 DIS configs | 0.1271 (0.1498) |
| `dis_1khz` | `disOneKHz` | 10 DIS configs | 0.1383 (0.1577) |
| `dis_1khz_large` | `disOneKHzLarge` | same, standard router | 0.1383 (0.1577) |
| `openpiv_1khz` | `openpivOneKHz` | 10 OpenPIV configs | 0.1978 (0.2544) |
| `openpiv_1khz_large` | `openpivOneKHzLarge` | same, standard router | 0.1956 (0.2544) |
| `openpiv_4khz` | `openpivFourKHz` | 10 OpenPIV configs | 0.1975 (0.2544) |
| `openpiv_4khz_large` | `openpivFourKHzLarge` | same, standard router | 0.1956 (0.2544) |

## Layout

- `exp.yaml`: the models to run, the seeds (0, 1, 2) and the metrics kept in
  `results.json`.
- `models/`: one file per model. `router_<row>.yaml` is a row's ART of PIV
  router: it extends `router/standard.yaml` or `router/fast.yaml` (the 1 kHz
  rows) and sets its library. `dis/base.yaml` holds the settings shared by all DIS models; each
  DIS model `extends` it and sets its knobs, named after them
  (`ps9_st2_gd4_lv3.yaml`: patch size 9, stride 2, 4 gradient-descent
  iterations, 3 levels). A row's best single estimator is one of these files,
  run on its own in eval mode (it sets `mode`, `dataset` and `study`).
- `libraries/<row>.yaml`: the row's library, a list of model files (fixed;
  not re-selected).
- `datasets/train.yaml`: the router's training run, shared by all rows: train
  split, recipe, labels (the library's cached per-frame errors), and which
  files to validate (`datasets/val.yaml`) and test (`datasets/test.yaml`) on.
  The best single estimators run on `datasets/test.yaml` directly.
  `train_bs64.yaml` and `val_bs64.yaml` extend them with batches of 64 (the
  1 kHz rows).
- `results.json`: written by `run.py`; per model, the mean, std and per-seed
  values over the test frames of `mean_epe` (AEPE, what the paper reports),
  `median_epe`, `p95_epe` and `p99_epe`.

Data come from the private Hub datasets `activefluidcontrol/piv-class1`
(images and flow fields) and `activefluidcontrol/piv-class1-caches` (error
caches), both pinned to a revision. Log in with `hf auth login` first.

## Run

From the repository root, on a GPU:

```bash
uv run --extra cuda12 --extra wandb --extra hf python experiments/run.py --exp art_of_piv
```

## Protocol

Router training: 5000 steps on `train` (batch 16, shuffled), validation on
`val` every 500 steps (80 batches), and the best-on-validation checkpoint is
scored on the full `test` split (450 frames). The best single estimator is
scored on the same 450 frames.
