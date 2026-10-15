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
The table's agent execution times are not produced here yet.

```bash
uv run --extra cuda12 --extra wandb --extra hf python experiments/run.py --exp art_of_piv/appendix/A.3_agent_architecture
```
