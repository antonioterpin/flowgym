# Table 1: main results

Each row trains the router on a fixed library of estimators and compares it
with the best single estimator under the same latency budget.

| Row | Paper key | Library | Paper AEPE (best single) |
|---|---|---|---|
| `dis_20hz` | `disTwentyHz` | 10 DIS configs | 0.1271 (0.1498) |
| `dis_1khz` | `disOneKHz` | 10 DIS configs | 0.1383 (0.1577) |
| `openpiv_1khz` | `openpivOneKHz` | 10 OpenPIV configs | 0.1978 (0.2544) |
| `openpiv_4khz` | `openpivFourKHz` | 10 OpenPIV configs | 0.1975 (0.2544) |
| `raft32` | `raftThirtyTwo` | 6 per-class RAFT32 specialists | 0.0096 (0.0120) |

## Files

- `exp.yaml`: the models to run, the seeds (0, 1, 2) and the metrics kept in
  `results.json`.
- `models/router_<row>.yaml`: a row's router. It extends
  `experiments/art_of_piv/models/router/standard.yaml` (or `fast.yaml` for the
  1 kHz rows) and sets its library from `experiments/art_of_piv/libraries/`.
- `models/best_single_<row>.yaml`: a row's best single estimator, one of the
  shared model files run on its own on the test split. Both OpenPIV rows share
  one.
- `results.json`: written by `run.py`; per model, the mean, std and per-seed
  values over the test frames of `mean_epe` (AEPE, what the paper reports),
  `median_epe`, `p95_epe` and `p99_epe`.

## Run

From the repository root, on a GPU:

```bash
uv run --extra cuda12 --extra wandb --extra hf python experiments/run.py --exp art_of_piv/table_1
```

## Protocol

Router training: 5000 steps on `train` (batch 16, shuffled; 64 for the 1 kHz
rows), validation on `val` every 500 steps (80 batches), and the
best-on-validation checkpoint is scored on the full `test` split (450 frames).
The best single estimator is scored on the same 450 frames.
