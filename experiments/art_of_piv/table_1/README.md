# Table 1: main results

Each row trains the router on a fixed library of estimators and compares it
with the best single estimator under the same latency budget.

| Row | Paper key | Library |
|---|---|---|
| `dis_20hz` | `disTwentyHz` | 10 DIS configs |
| `dis_1khz` | `disOneKHz` | 10 DIS configs |
| `openpiv_1khz` | `openpivOneKHz` | 10 OpenPIV configs |
| `openpiv_4khz` | `openpivFourKHz` | 10 OpenPIV configs |
| `raft32` | `raftThirtyTwo` | 6 per-class RAFT32 specialists |

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
  `std_epe`, `median_epe`, `p90_epe`, `p95_epe`, `p99_epe` and `max_epe`.
  Each run's summary in `_generated/summaries/` also lists every test frame's
  EPE.

## Run

Set up the environment as in [the ART of PIV README](../README.md#run), then,
from the repository root, on a GPU:

```bash
uv run python experiments/run.py --exp art_of_piv/table_1
```

## Protocol

Router training: 5000 steps on `train` (batch 16, shuffled; 64 for the 1 kHz
rows), validation on `val` every 500 steps (80 batches), and the
best-on-validation checkpoint is scored on the full `test` split (450 frames).
The best single estimator is scored on the same 450 frames.

## Results

Test AEPE on the 450 test frames, mean ± std over seeds 0, 1, 2 (RTX 5090),
generated from `results.json` (also per seed); the paper reads the means.

<!-- BEGIN results.json -->
| model | mean_epe | std_epe | median_epe | p90_epe | p95_epe | p99_epe | max_epe |
|---|---|---|---|---|---|---|---|
| `models/router_dis_20hz.yaml` | 0.12682 ± 0.00023 | 0.09675 ± 0.00050 | 0.09436 ± 0.00011 | 0.23840 ± 0.00106 | 0.32098 ± 0.00154 | 0.51303 | 0.67329 |
| `models/best_single_dis_20hz.yaml` | 0.14979 | 0.12704 ± 0.00000 | 0.10546 | 0.27669 | 0.37928 | 0.62749 ± 0.00000 | 1.43244 |
| `models/router_dis_1khz.yaml` | 0.13851 ± 0.00011 | 0.10577 ± 0.00035 | 0.10228 ± 0.00015 | 0.26481 ± 0.00000 | 0.34988 ± 0.00494 | 0.59728 ± 0.00590 | 0.78542 |
| `models/best_single_dis_1khz.yaml` | 0.15769 | 0.12189 ± 0.00000 | 0.11332 | 0.30161 | 0.38822 | 0.63073 | 1.00529 |
| `models/router_openpiv_1khz.yaml` | 0.19790 ± 0.00046 | 0.15746 ± 0.00030 | 0.14226 ± 0.00041 | 0.39104 ± 0.00129 | 0.49048 | 0.83173 | 1.00574 |
| `models/router_openpiv_4khz.yaml` | 0.19737 ± 0.00077 | 0.15579 ± 0.00205 | 0.14276 ± 0.00052 | 0.38988 | 0.48823 ± 0.00318 | 0.82226 ± 0.01340 | 0.97949 ± 0.03713 |
| `models/best_single_openpiv.yaml` | 0.25441 | 0.25589 | 0.16564 | 0.49927 | 0.85567 | 1.42072 | 1.95452 |
| `models/router_raft32.yaml` | 0.00956 ± 0.00000 | 0.00730 ± 0.00001 | 0.00768 ± 0.00001 | 0.01793 ± 0.00005 | 0.02268 | 0.03648 | 0.04237 |
| `models/best_single_raft32.yaml` | 0.01199 | 0.00896 | 0.01032 | 0.01998 ± 0.00000 | 0.02700 | 0.04508 ± 0.00000 | 0.07897 |
<!-- END results.json -->

## Library selection

Each DIS and OpenPIV row's library and best single estimator come from its
candidate pool, `experiments/art_of_piv/pools/{dis,openpiv}.yaml` (565 DIS and
1080 OpenPIV configs; each candidate's cached per-frame EPEs and timing are in
the pinned `piv-class1-caches` repo). For a row with nominal rate f, the
candidates with mean latency (`mean_ms` in the cache's `timing.json`, measured
on the RTX 5090) within 1/f are kept, minus those that diverge on some frame
or lack a tune frame or a timing. On the 1320 tune frames, the library is the
10 configs minimizing the mean oracle EPE (paper eq. library), solved exactly,
and the best single estimator is the K = 1 solution. `scripts/select_ensemble.py`
does both, from the repository root (CPU, under a minute each), with the pins
of `experiments/art_of_piv/datasets/train.yaml`:

```bash
CACHES=hf://activefluidcontrol/piv-class1-caches@7068a49525da1d8c1b7bdf4027a3c3a74da5d02a
TUNE=hf://activefluidcontrol/piv-class1@ccc784ebbe86f16121bdc51ccb20b4cdf6b6a150:tune
# Library of dis_20hz; add --out <file>.json to keep the picks
uv run python scripts/select_ensemble.py --cache-root $CACHES \
    --keys-from $TUNE --cache-id-prefix piv-class1-dis --time-limit 50 --K 10 --exact
# Its best single: the same without --exact and with --K 1
```

| Row | `--cache-id-prefix` | `--time-limit` (ms) |
|---|---|---|
| `dis_20hz` | `piv-class1-dis` | 50 |
| `dis_1khz` | `piv-class1-dis` | 1 |
| `openpiv_1khz` | `piv-class1-openpiv` | 1 |
| `openpiv_4khz` | `piv-class1-openpiv` | 0.25 |

The picks are `experiments/art_of_piv/libraries/<row>.yaml` (the exact
solution's order) and `models/best_single_<row>.yaml`.
`tests/integration/art_of_piv_paper/test_libraries.py` checks this: its fast
tests that the committed files and pools hash to the paper's caches, and,
with `uv run pytest -m run_explicitly tests/integration/art_of_piv_paper`,
that every selection above reproduces them (it downloads the caches).

The OpenPIV best single ties exactly with the same config without
`sig2noise_threshold` (identical errors on every frame). K = 1 is solved
greedily, which keeps the first cache in order, as the paper's selection did.
The raft32 row's library is the six per-class specialists (no selection).
