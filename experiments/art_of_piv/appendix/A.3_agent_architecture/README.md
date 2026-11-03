# Appendix A.3: agent architecture

Table 5 compares the deployed router of the three high-rate rows of Table 1
with a larger one: the standard router (three convolutional blocks, one head
per estimator) instead of the fast one, trained the same way (batch 64).
This ablation justifies a design choice, so like A.1 and A.2 it is scored on
the val split, not test.

## Files

- `exp.yaml`: per row, the deployed router, the larger router and the best
  single estimator; seeds 0, 1, 2.
- `models/router_<row>.yaml`: the deployed router of `table_1` (extends its
  file there); `models/router_<row>_large.yaml`: the larger router. Both
  score their best checkpoint on the full val split (1320 frames,
  `dataset_overrides`).
- `models/best_single_<row>.yaml`: the row's best single estimator on the
  val split. It is the oracle on a library holding only that estimator
  (`experiments/art_of_piv/libraries/best_single_<row>.yaml`), so its
  per-frame EPEs come from the same caches the routers train on.
- `results.json`: written by `run.py`; per model, the mean, std and
  per-seed AEPE over the val split.

Set up the environment as in [the ART of PIV README](../../README.md#run), then:

```bash
uv run python experiments/run.py --exp art_of_piv/appendix/A.3_agent_architecture
```

## Results

AEPE on the val split (1320 frames), mean ± std over seeds 0, 1, 2, RTX
5090 (October 2026), generated from `results.json`; the paper's Table 5
computes rAEPE against the row's best single estimator.

<!-- BEGIN results.json -->
| model | mean_epe | median_epe | p95_epe | p99_epe |
|---|---|---|---|---|
| `models/router_dis_1khz.yaml` | 0.12787 ± 0.00008 | 0.09851 ± 0.00008 | 0.28712 ± 0.00066 | 0.49516 ± 0.00758 |
| `models/router_dis_1khz_large.yaml` | 0.12716 ± 0.00018 | 0.09830 ± 0.00005 | 0.28469 ± 0.00009 | 0.42359 ± 0.01038 |
| `models/best_single_dis_1khz.yaml` | 0.15061 | 0.11031 | 0.34182 | 0.62324 |
| `models/router_openpiv_1khz.yaml` | 0.18269 ± 0.00008 | 0.13709 ± 0.00014 | 0.43978 ± 0.00047 | 0.69563 |
| `models/router_openpiv_1khz_large.yaml` | 0.18227 ± 0.00012 | 0.13714 ± 0.00018 | 0.44005 ± 0.00076 | 0.69563 |
| `models/router_openpiv_4khz.yaml` | 0.18275 ± 0.00016 | 0.13806 ± 0.00041 | 0.44014 ± 0.00008 | 0.69563 |
| `models/router_openpiv_4khz_large.yaml` | 0.18237 ± 0.00014 | 0.13798 ± 0.00013 | 0.44008 ± 0.00008 | 0.69563 |
| `models/best_single_openpiv.yaml` | 0.23133 | 0.15379 | 0.58439 | 1.34944 |
<!-- END results.json -->
## Agent execution time

`agent_time.py` times the agent of the deployed router (`router/fast.yaml`)
and of the larger one (`router/standard.yaml`) on the DIS and OpenPIV 1 kHz
libraries (the OpenPIV time is reported for both OpenPIV rows). For each
library member it forces the router to pick that member, times the full
jitted ART of PIV call on one 256x256 frame (warm-up excluded, best of 5
repetitions of 100 calls), and subtracts the member alone; the agent time is
the mean over the members. It writes `agent_time.json`. Run it on an idle GPU:

```bash
uv run python experiments/art_of_piv/appendix/A.3_agent_architecture/agent_time.py
```

Each key `<family>_<agent>_ms` feeds the paper macro
`agentOverhead<Family><Agent>Ms` (e.g. `dis_deployed_ms` →
`\agentOverheadDisDeployedMs`):

<!-- BEGIN agent_time.json -->
| key | value |
|---|---|
| `dis_deployed_ms` | 0.1632 |
| `dis_large_ms` | 0.2685 |
| `openpiv_deployed_ms` | 0.1112 |
| `openpiv_large_ms` | 0.2108 |
| `device` | NVIDIA GeForce RTX 5090 |
<!-- END agent_time.json -->

Timings vary slightly from run to run.
