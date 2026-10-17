# Appendix A.1: learning signal

Table 3 and Figure 8 compare three ways to train the router on greedy DIS
libraries of 2 to 20 models, by validation AEPE: classification (the label is
the best member), regression (predict every member's error) and a bandit
(DQN with discount 0, which only sees the error of the member it ran). The
oracle picks the best member per frame using the ground truth; the baseline is
the single DIS config the libraries start from (the 1-model point).

## Files

- `exp.yaml`: the baseline, then per library size the oracle and the three
  routers; seeds 0, 1, 2.
- `models/{classification,regression,bandit}.yaml`: one file per signal,
  extending the shared router of the same name (`standard.yaml` for
  classification) in `experiments/art_of_piv/models/router/`; they score the
  best checkpoint on the val split (`dataset_overrides`).
- `models/oracle.yaml`: the `oracle_art_of_piv` estimator, which picks
  the member with the lowest cached EPE on each val frame.
- `models/n<K>/<signal>.yaml`: a signal on the library
  `experiments/art_of_piv/libraries/abl_n<K>.yaml` (the first K of 20
  DIS configs picked greedily on the tune split from the whole DIS pool:
  `scripts/select_ensemble.py --cache-id-prefix piv-class1-dis --K 20` with
  the pins of `table_1/README.md`, checked by
  `tests/integration/art_of_piv_paper/test_libraries.py`).
- `models/baseline.yaml`: the oracle on `abl_n1`, i.e. the single DIS config.
- `results.json`: written by `run.py`; per model, the mean, std and per-seed
  AEPE over the 1320 val frames.

## Run

```bash
uv run --extra cuda12 --extra wandb --extra hf python experiments/run.py --exp art_of_piv/appendix/A.1_learning_signal
```

## Protocol

Same router training as `table_1` (5000 steps, batch 16, validation every 500
steps, keep the best checkpoint), then the full val split in batches of 20.
The oracle and the baseline read the same cached per-frame EPEs as the
routers, so they are identical across seeds.

## Results

Validation AEPE, mean ± std over seeds 0, 1, 2 (per-seed values in
`results.json`). The oracle and the baseline are deterministic. The runs on
`abl_n5` to `abl_n20` are from 4× RTX 4090 (October 2026); those on
`abl_n2` to `abl_n4`, whose libraries did not change when the pool was
completed, are from one RTX 5090 (October 2026).

| models | oracle | classification | regression | bandit |
|---|---|---|---|---|
| 1 (baseline) | 0.1370 ± 0.0000 | | | |
| 2 | 0.1196 ± 0.0000 | 0.1208 ± 0.0004 | 0.1211 ± 0.0000 | 0.1219 ± 0.0005 |
| 3 | 0.1073 ± 0.0000 | 0.1086 ± 0.0001 | 0.1090 ± 0.0003 | 0.1096 ± 0.0001 |
| 4 | 0.1064 ± 0.0000 | 0.1081 ± 0.0001 | 0.1081 ± 0.0001 | 0.1101 ± 0.0002 |
| 5 | 0.1056 ± 0.0000 | 0.1079 ± 0.0003 | 0.1080 ± 0.0001 | 0.1102 ± 0.0008 |
| 10 | 0.1041 ± 0.0000 | 0.1075 ± 0.0003 | 0.1068 ± 0.0000 | 0.1101 ± 0.0008 |
| 15 | 0.1036 ± 0.0000 | 0.1076 ± 0.0001 | 0.1066 ± 0.0002 | 0.1097 ± 0.0006 |
| 20 | 0.1034 ± 0.0000 | 0.1076 ± 0.0001 | 0.1067 ± 0.0001 | 0.1106 ± 0.0007 |

Every router beats the baseline. Regression leads from 10 models on, and the
bandit trails both signals at every size.
