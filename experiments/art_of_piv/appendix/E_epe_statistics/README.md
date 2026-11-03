# Appendix E: per-frame EPE statistics

Appendix E reports the spread and tails of the per-frame test EPE of Table 1
(`std_epe`, `median_epe`, `p90_epe`, ... in `table_1/results.json`) and how
often each router is better than, ties with or is worse than its row's best
single estimator on the same frame (`results.json` here; a tie is a frame
where the router picks the best single).

## Run

After `table_1`, whose summaries in `table_1/_generated/summaries/` list every
test frame's EPE (CPU only, a few seconds):

```bash
uv run python experiments/art_of_piv/appendix/E_epe_statistics/stats.py
```

## Results

Share of the 450 test frames (%), mean ± std over seeds 0, 1, 2, generated
from `results.json`:

<!-- BEGIN results.json -->
| model | better_pct | tie_pct | worse_pct |
|---|---|---|---|
| `models/router_dis_20hz.yaml` | 68.88889 ± 1.01024 | 13.70370 ± 0.89504 | 17.40741 ± 0.45662 |
| `models/router_dis_1khz.yaml` | 78.51852 ± 0.20951 | 0.22222 | 21.25926 ± 0.20951 |
| `models/router_openpiv_1khz.yaml` | 65.62963 ± 1.16652 | 21.92593 ± 1.40936 | 12.44444 ± 1.18981 |
| `models/router_openpiv_4khz.yaml` | 61.48148 ± 0.93110 | 27.33333 ± 1.78701 | 11.18519 ± 0.85747 |
| `models/router_raft32.yaml` | 98.07407 ± 0.20951 | 0.00000 | 1.92593 ± 0.20951 |
<!-- END results.json -->
