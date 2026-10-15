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
  greedily chosen DIS configs).
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
