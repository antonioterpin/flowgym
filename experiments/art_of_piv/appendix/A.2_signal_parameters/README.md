# Appendix A.2: signal parameters

Figure 9 and Table 4 tune the two learning signals of A.1 on the fixed
20-model greedy DIS library (`experiments/art_of_piv/libraries/abl_n20.yaml`),
by validation AEPE:

- Figure 9: the bandit router's exploration. Epsilon-greedy starts at ε and
  decays linearly to 0 over 4000 steps, for ε in {0, 0.02, 0.05, 0.1, 0.25,
  0.5, 1}.
- Table 4: the regression router's target: the raw cached EPEs, each
  frame's EPEs divided by their mean over the library (the paper's per-pair
  mean, target transform `per_frame_mean`), or their log.

## Files

- `exp.yaml`: the seven bandit routers, then the three regression routers;
  seeds 0, 1, 2.
- `models/bandit.yaml`, `models/regression.yaml`: the shared bandit and
  regression routers (`experiments/art_of_piv/models/router/`) on
  `abl_n20`, scoring the best checkpoint on the val split
  (`dataset_overrides`).
- `models/bandit/eps<ε>.yaml`: the bandit router with initial exploration
  ε (`eps1.yaml` is the shared router's schedule).
- `models/regression/{raw,mean,log}.yaml`: the regression router with each
  target (`log.yaml` is the shared router's target).
- `results.json`: written by `run.py`; per model, the mean, std and
  per-seed AEPE over the 1320 val frames.

## Run

```bash
uv run --extra cuda12 --extra wandb --extra hf python experiments/run.py --exp art_of_piv/appendix/A.2_signal_parameters
```

## Protocol

Same router training as A.1 and `table_1` (5000 steps, batch 16,
validation every 500 steps, keep the best checkpoint), then the full val
split in batches of 20. The 30 runs take about an hour on one RTX 4090.

## Results

Validation AEPE, mean ± std over seeds 0, 1, 2, 4× RTX 4090 (October 2026);
the paper's Figure 9 and Table 4 read these means from `results.json`.
Router training is not deterministic from run to run.

| model | AEPE | per seed |
|---|---|---|
| bandit, ε = 0 | 0.1337 ± 0.0070 | 0.1339, 0.1251, 0.1422 |
| bandit, ε = 0.02 | 0.1132 ± 0.0015 | 0.1140, 0.1111, 0.1144 |
| bandit, ε = 0.05 | 0.1144 ± 0.0018 | 0.1162, 0.1150, 0.1120 |
| bandit, ε = 0.1 | 0.1126 ± 0.0030 | 0.1115, 0.1167, 0.1096 |
| bandit, ε = 0.25 | 0.1097 ± 0.0001 | 0.1097, 0.1097, 0.1099 |
| bandit, ε = 0.5 | 0.1114 ± 0.0010 | 0.1103, 0.1127, 0.1114 |
| bandit, ε = 1 | 0.1106 ± 0.0007 | 0.1103, 0.1099, 0.1115 |
| regression, raw | 0.1084 ± 0.0004 | 0.1089, 0.1080, 0.1082 |
| regression, per-pair mean | 0.1072 ± 0.0001 | 0.1072, 0.1074, 0.1071 |
| regression, log | 0.1067 ± 0.0001 | 0.1068, 0.1066, 0.1066 |

The log target is best, then the per-pair mean, then raw. Without
exploration the bandit is clearly worst; for every ε ≥ 0.02 it lies within
0.110–0.115, lowest at ε = 0.25. The seeds spread by up to 0.007 px for
ε ≥ 0.02 and 0.017 px at ε = 0.
