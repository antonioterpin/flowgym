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

The pool is every cached DIS config with errors on all tune frames, minus
those that diverge on some frame. Greedy on this pool is the A.1 library
`abl_n20`, in order (checked by
`tests/integration/art_of_piv_paper/test_libraries.py`). Generated from
`results.json` (which also has both libraries per K and the HiGHS status);
MILP times are wall clock on the CPU of the RTX 4090 machine (64 cores):

<!-- BEGIN results.json -->
Pool: 477 configs (78 diverging dropped), 1320 tune frames.

| K | greedy | exact | gap (px) | gap (%) | MILP (s) |
|---|---|---|---|---|---|
| 1 | 0.14273 | 0.14273 | 0.00000 | 0.00 | 41.7 |
| 2 | 0.12547 | 0.11762 | 0.00784 | 6.67 | 13.1 |
| 3 | 0.11475 | 0.11468 | 0.00007 | 0.06 | 9.8 |
| 4 | 0.11374 | 0.11354 | 0.00020 | 0.18 | 10.4 |
| 5 | 0.11300 | 0.11253 | 0.00046 | 0.41 | 8.5 |
| 6 | 0.11240 | 0.11209 | 0.00031 | 0.27 | 9.1 |
| 7 | 0.11199 | 0.11179 | 0.00020 | 0.18 | 16.8 |
| 8 | 0.11172 | 0.11150 | 0.00022 | 0.20 | 8.2 |
| 9 | 0.11145 | 0.11125 | 0.00019 | 0.17 | 8.0 |
| 10 | 0.11120 | 0.11104 | 0.00017 | 0.15 | 7.0 |
| 11 | 0.11108 | 0.11091 | 0.00017 | 0.15 | 6.9 |
| 12 | 0.11096 | 0.11080 | 0.00016 | 0.14 | 6.6 |
| 13 | 0.11085 | 0.11075 | 0.00010 | 0.09 | 6.1 |
| 14 | 0.11077 | 0.11070 | 0.00007 | 0.06 | 5.6 |
| 15 | 0.11070 | 0.11066 | 0.00004 | 0.03 | 5.9 |
| 16 | 0.11064 | 0.11062 | 0.00002 | 0.02 | 5.9 |
| 17 | 0.11060 | 0.11058 | 0.00002 | 0.02 | 6.3 |
| 18 | 0.11057 | 0.11055 | 0.00002 | 0.02 | 5.9 |
| 19 | 0.11054 | 0.11052 | 0.00002 | 0.02 | 5.7 |
| 20 | 0.11050 | 0.11049 | 0.00002 | 0.01 | 6.2 |
<!-- END results.json -->
