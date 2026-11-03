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

Set up the environment as in [the ART of PIV README](../../README.md#run), then:

```bash
uv run python experiments/run.py --exp art_of_piv/appendix/A.2_signal_parameters
```

## Protocol

Same router training as A.1 and `table_1` (5000 steps, batch 16,
validation every 500 steps, keep the best checkpoint), then the full val
split in batches of 20. The 30 runs take about an hour on one RTX 4090.

## Results

Validation AEPE, mean ± std over seeds 0, 1, 2, 4× RTX 4090 (October 2026),
generated from `results.json` (also per seed); the paper's Figure 9 and
Table 4 read the means from it.
Router training is not deterministic from run to run.

<!-- BEGIN results.json -->
| model | mean_epe |
|---|---|
| `models/bandit/eps0.yaml` | 0.13374 ± 0.00698 |
| `models/bandit/eps0.02.yaml` | 0.11316 ± 0.00149 |
| `models/bandit/eps0.05.yaml` | 0.11440 ± 0.00179 |
| `models/bandit/eps0.1.yaml` | 0.11260 ± 0.00302 |
| `models/bandit/eps0.25.yaml` | 0.10975 ± 0.00009 |
| `models/bandit/eps0.5.yaml` | 0.11144 ± 0.00096 |
| `models/bandit/eps1.yaml` | 0.11059 ± 0.00066 |
| `models/regression/raw.yaml` | 0.10837 ± 0.00035 |
| `models/regression/mean.yaml` | 0.10723 ± 0.00011 |
| `models/regression/log.yaml` | 0.10668 ± 0.00010 |
<!-- END results.json -->
