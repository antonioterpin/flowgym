# Appendix A.3: agent architecture

Table 5 compares the deployed router of the three high-rate rows of Table 1
with a larger one: the standard router (three convolutional blocks, one head
per estimator) instead of the fast one, trained the same way (batch 64).

| Row | Paper key | Paper AEPE |
|---|---|---|
| `dis_1khz_large` | `disOneKHzLarge` | 0.1383 |
| `openpiv_1khz_large` | `openpivOneKHzLarge` | 0.1956 |
| `openpiv_4khz_large` | `openpivFourKHzLarge` | 0.1956 |

The deployed routers and the best single estimators are in
`experiments/art_of_piv/table_1/`.

```bash
uv run --extra cuda12 --extra wandb --extra hf python experiments/run.py --exp art_of_piv/appendix/A.3_agent_architecture
```

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

| Key | Paper macro | Paper (ms) | `agent_time.json` (ms, RTX 5090) |
|---|---|---|---|
| `dis_deployed_ms` | `agentOverheadDisDeployedMs` | 0.17 | 0.163 |
| `dis_large_ms` | `agentOverheadDisLargeMs` | 0.27 | 0.269 |
| `openpiv_deployed_ms` | `agentOverheadOpenpivDeployedMs` | 0.10 | 0.111 |
| `openpiv_large_ms` | `agentOverheadOpenpivLargeMs` | 0.20 | 0.211 |

The paper's numbers came from an earlier version of this script, which
subtracted each member's time recorded with its cache instead of timing the
member in the same run; the rest of the protocol is the same. Timings vary
by about 0.01 ms between runs.
