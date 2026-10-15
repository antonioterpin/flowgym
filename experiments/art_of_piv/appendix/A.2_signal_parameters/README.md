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

Mean ± std over seeds 0, 1, 2 on an RTX 4090 (October 2026), against the
paper's `figures/ablation_bandit_eps.csv` and
`figures/ablation_regression_targets.csv`. Those hold one number per
configuration, from the June 2026 runs on the RTX 5090. Router training is
not deterministic from run to run.

| model | AEPE | per seed | paper | diff |
|---|---|---|---|---|
| bandit, ε = 0 | 0.1359 ± 0.0081 | 0.1385, 0.1249, 0.1444 | 0.1353 | +0.5 % |
| bandit, ε = 0.02 | 0.1165 ± 0.0026 | 0.1152, 0.1201, 0.1142 | 0.1213 | −4.0 % |
| bandit, ε = 0.05 | 0.1158 ± 0.0018 | 0.1151, 0.1183, 0.1139 | 0.1116 | +3.7 % |
| bandit, ε = 0.1 | 0.1119 ± 0.0023 | 0.1106, 0.1100, 0.1151 | 0.1128 | −0.8 % |
| bandit, ε = 0.25 | 0.1095 ± 0.0004 | 0.1099, 0.1095, 0.1090 | 0.1104 | −0.9 % |
| bandit, ε = 0.5 | 0.1104 ± 0.0006 | 0.1098, 0.1112, 0.1101 | 0.1119 | −1.3 % |
| bandit, ε = 1 | 0.1113 ± 0.0008 | 0.1113, 0.1103, 0.1122 | 0.1118 | −0.5 % |
| regression, raw | 0.1088 ± 0.0005 | 0.1094, 0.1087, 0.1082 | 0.1091 | −0.3 % |
| regression, per-pair mean | 0.1072 ± 0.0003 | 0.1070, 0.1070, 0.1075 | 0.1078 | −0.6 % |
| regression, log | 0.1064 ± 0.0001 | 0.1063, 0.1065, 0.1065 | 0.1066 | −0.1 % |

Table 4 reproduces within 0.6 % with the same ordering (log, then
per-pair mean, then raw). Figure 9 keeps its shape: no exploration is
clearly worst, and from ε = 0.1 on the AEPE is flat around 0.110–0.112. At
ε ≤ 0.05 the seeds spread by up to 0.02 px, so the paper's single runs at
0.02 and 0.05 sit within that spread.
