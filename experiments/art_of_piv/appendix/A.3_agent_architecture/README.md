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

```bash
uv run --extra cuda12 --extra wandb --extra hf python experiments/run.py --exp art_of_piv/appendix/A.3_agent_architecture
```

## Results

Mean ± std over seeds 0, 1, 2 on the val split (1320 frames), RTX 5090
(October 2026); rAEPE against the row's best single estimator on val.

| Row | Agent | AEPE | rAEPE |
|---|---|---|---|
| `dis_1khz` | deployed | 0.12787 ± 0.00008 | 15.1 % |
| | larger | 0.12716 ± 0.00018 | 15.6 % |
| | best single | 0.15061 | |
| `openpiv_1khz` | deployed | 0.18269 ± 0.00008 | 21.0 % |
| | larger | 0.18227 ± 0.00012 | 21.2 % |
| `openpiv_4khz` | deployed | 0.18275 ± 0.00016 | 21.0 % |
| | larger | 0.18237 ± 0.00014 | 21.2 % |
| OpenPIV rows | best single | 0.23133 | |

The larger router is better by 0.0004–0.0007 px in every row, so the
deployed one loses little accuracy for its shorter execution time.

## Agent execution time

`agent_time.py` times the agent of the deployed router (`router/fast.yaml`)
and of the larger one (`router/standard.yaml`) on the DIS and OpenPIV 1 kHz
libraries (the OpenPIV time is reported for both OpenPIV rows). For each
library member it forces the router to pick that member, times the full
jitted ART of PIV call on one 256x256 frame (warm-up excluded, best of 5
repetitions of 100 calls), and subtracts the member alone; the agent time is
the mean over the members. It writes `agent_time.json`. Run it on an idle GPU:

```bash
uv run --extra cuda12 python experiments/art_of_piv/appendix/A.3_agent_architecture/agent_time.py
```

| Key | Paper macro | `agent_time.json` (ms, RTX 5090) |
|---|---|---|
| `dis_deployed_ms` | `agentOverheadDisDeployedMs` | 0.163 |
| `dis_large_ms` | `agentOverheadDisLargeMs` | 0.269 |
| `openpiv_deployed_ms` | `agentOverheadOpenpivDeployedMs` | 0.111 |
| `openpiv_large_ms` | `agentOverheadOpenpivLargeMs` | 0.211 |

Timings vary by about 0.01 ms between runs.
