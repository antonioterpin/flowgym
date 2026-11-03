# 🎨 ART of PIV examples

These two example scripts show how to wire up an ART of PIV estimator
from configuration and reuse cached per-estimator EPE values across
runs:

- `examples/12_caching_art_of_piv.py` — ART of PIV routing across
  three `dis_jax` sub-estimators of different sizes.
- `examples/13_caching_art_of_piv_mixed.py` — ART of PIV routing
  across a mix of `dis_jax` and `raft_jax` sub-estimators.

Both follow the same two-pass pattern as the
[plain caching examples](caching.md): a cold-cache run that computes
and caches per-estimator EPEs, followed by a warm-cache run that
reuses them. With ART of PIV the cache is doing more work — the
estimator aggregates per-sub-estimator EPE so the training/eval path
can short-circuit dispatch when EPEs are already known.

For the conceptual model see the
[ART of PIV estimator page](../api/estimators/art-of-piv.md); this page focuses
on the runnable example.

## How to run them

```bash
uv sync --group dev
uv run python examples/12_caching_art_of_piv.py
uv run python examples/13_caching_art_of_piv_mixed.py
```

Both scripts create their own temporary `.mat` data and YAML configs
under a `tempfile.TemporaryDirectory()`; nothing is written into the
repo. The runs hit GPU 0 by default (`CUDA_VISIBLE_DEVICES=0`) and
disable W&B (`WANDB_MODE=disabled`).

## What the scripts do

Each script writes three temporary YAML files and then invokes
`src.main --mode eval` twice — once cold, once warm — printing a
speedup at the end. The three YAML files are:

1. **Dataset config.** Built by overriding a template
   (`src/flowgym/config/piv_dataset_class1_eval_original.yaml`) with
   smaller batch sizes, the list of generated `.mat` files, and a
   `caching:` block. The cache `spec` declares per-sample storage for
   the aggregated `epe_all` (one float per sub-estimator) plus a
   bitmask `epe_mask`. The first run uses `warm_start: index` (build
   the cache); the second run is rewritten to `warm_start: all` (load
   and reuse).
2. **Estimators list.** A YAML file listing the sub-estimators Art of
   PIV is allowed to route between. For example 12, three `dis_jax`
   estimators with different patch sizes; for example 13, a mix of
   `dis_jax` and `raft_jax`. The ART of PIV model config points at
   this file via `estimators_list_path`.
3. **Model config.** Sets `estimator: art_of_piv` and the full
   estimator config (router architecture, training mode, loss,
   exploration, etc.). See
   [docs/api/art-of-piv.md § "Minimal estimator config"](../api/estimators/art-of-piv.md#minimal-estimator-config)
   for a walk-through of the fields.

## What to look for in the output

When the scripts run successfully, you should see:

- output from the first (cold) evaluation run,
- output from the second (warm) evaluation run,
- the wall-clock duration of each,
- a final speedup line comparing the two.

The point of the comparison is the same as for the plain caching
examples — ART of PIV's training and evaluation loops can skip
running each sub-estimator when EPE is already cached, so the warm
run should be noticeably faster once the sub-estimators are
non-trivial. On very small inputs cache overhead can dominate; the
script prints a note in that case rather than failing.

## Related docs

- [ART of PIV estimator](../api/estimators/art-of-piv.md) — concept and module
  layout.
- [Caching examples](caching.md) — same two-pass pattern for plain
  `dis_jax` and `raft_jax`.
- [Evaluation and caching API](../api/evaluation.md) — the
  `CacheManager` machinery the cache config drives.
- [Configuration and data flow](../user-guide/configuration-and-data.md) —
  how dataset, model, and estimators-list YAMLs fit together.
