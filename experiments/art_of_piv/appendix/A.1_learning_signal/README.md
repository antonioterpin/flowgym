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

Validation AEPE, mean ± std over seeds 0, 1, 2, generated from
`results.json` (also per seed; models `n<K>/<signal>`). The oracle and the
baseline are deterministic. The runs on `abl_n5` to `abl_n20` are from 4×
RTX 4090 (October 2026); those on `abl_n2` to `abl_n4`, whose libraries did
not change when the pool was completed, are from one RTX 5090 (October 2026).

<!-- BEGIN results.json -->
| model | mean_epe |
|---|---|
| `models/baseline.yaml` | 0.13703 |
| `models/n2/oracle.yaml` | 0.11960 |
| `models/n2/classification.yaml` | 0.12082 ± 0.00036 |
| `models/n2/regression.yaml` | 0.12106 ± 0.00003 |
| `models/n2/bandit.yaml` | 0.12191 ± 0.00051 |
| `models/n3/oracle.yaml` | 0.10729 |
| `models/n3/classification.yaml` | 0.10863 ± 0.00005 |
| `models/n3/regression.yaml` | 0.10903 ± 0.00027 |
| `models/n3/bandit.yaml` | 0.10964 ± 0.00010 |
| `models/n4/oracle.yaml` | 0.10636 |
| `models/n4/classification.yaml` | 0.10805 ± 0.00010 |
| `models/n4/regression.yaml` | 0.10806 ± 0.00011 |
| `models/n4/bandit.yaml` | 0.11005 ± 0.00018 |
| `models/n5/oracle.yaml` | 0.10558 |
| `models/n5/classification.yaml` | 0.10786 ± 0.00031 |
| `models/n5/regression.yaml` | 0.10800 ± 0.00012 |
| `models/n5/bandit.yaml` | 0.11023 ± 0.00079 |
| `models/n10/oracle.yaml` | 0.10408 |
| `models/n10/classification.yaml` | 0.10751 ± 0.00027 |
| `models/n10/regression.yaml` | 0.10678 ± 0.00004 |
| `models/n10/bandit.yaml` | 0.11012 ± 0.00084 |
| `models/n15/oracle.yaml` | 0.10360 |
| `models/n15/classification.yaml` | 0.10757 ± 0.00015 |
| `models/n15/regression.yaml` | 0.10660 ± 0.00016 |
| `models/n15/bandit.yaml` | 0.10973 ± 0.00062 |
| `models/n20/oracle.yaml` | 0.10341 |
| `models/n20/classification.yaml` | 0.10763 ± 0.00008 |
| `models/n20/regression.yaml` | 0.10668 ± 0.00010 |
| `models/n20/bandit.yaml` | 0.11059 ± 0.00066 |
<!-- END results.json -->
