# Appendix B: greedy versus exact library selection

Appendix B compares greedy library selection with the exact optimum on the
DIS pool (the libraries of Table 1 are exact optima, those of A.1 greedy;
see `table_1/README.md`). For every library size K from 1 to 20,
`select.py` picks the K DIS configs that minimize the mean oracle EPE
(per frame, the error of the best member) over the 1320 tune frames, once
greedily and once exactly with the mixed-integer program of
`scripts/select_ensemble.py` (HiGHS), and times the exact solve.

## Files

- `select.py`: the experiment. It reads the cached per-frame EPEs of every
  DIS config from `hf://activefluidcontrol/piv-class1-caches` and the tune
  frames from `hf://activefluidcontrol/piv-class1`, both pinned as in
  `experiments/art_of_piv/datasets/`, and calls the greedy and MILP solvers
  of `scripts/select_ensemble.py`. The selection trains and evaluates no
  estimator, so it does not fit `run.py`.
- `results.json`: per K, the greedy and exact mean oracle EPE, the gap in
  pixels and relative to the optimum, the MILP's wall time and HiGHS
  status (the script fails unless every K is proven optimal), and both
  libraries (cache ids).

## Run

CPU only, about 3.5 min:

```bash
uv run --extra hf python experiments/art_of_piv/appendix/B_library_selection/select.py
```

## Results

Pool: 477 DIS configs (the 565 cached ones, minus 10 not run on every tune
frame and 78 that diverge on some frame). Greedy on this pool is the A.1
library `abl_n20`, in order (checked by
`tests/integration/art_of_piv_paper/test_libraries.py`).
The original run (June 2026) also dropped the 25 configs without a timing
file, leaving 452.

| K | greedy | exact | gap (px) | gap (%) | MILP (s) |
|---|---|---|---|---|---|
| 1 | 0.14273 | 0.14273 | 0.00000 | 0.00 | 41.7 |
| 2 | 0.12547 | 0.11762 | 0.00784 | 6.67 | 13.1 |
| 3 | 0.11475 | 0.11468 | 0.00007 | 0.06 | 9.8 |
| 4 | 0.11374 | 0.11354 | 0.00020 | 0.18 | 10.4 |
| 5 | 0.11300 | 0.11253 | 0.00046 | 0.41 | 8.5 |
| 10 | 0.11120 | 0.11104 | 0.00017 | 0.15 | 7.0 |
| 15 | 0.11070 | 0.11066 | 0.00004 | 0.03 | 5.9 |
| 20 | 0.11050 | 0.11049 | 0.00002 | 0.01 | 6.2 |

All sizes are in `results.json`. Times are wall clock on the CPU of the
RTX 4090 machine (64 cores).

The paper reports the gap at K = 2 (6.7 %), the largest gap for K ≥ 3
(0.41 %, at K = 5) and the MILP time range for K ≥ 2 (5.6–16.8 s) from
`results.json`. K = 1 takes 42 s.
