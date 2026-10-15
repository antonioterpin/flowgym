# 🎨 ART of PIV

## Goal

Different flow estimators win on different image pairs. A correlation
method may be the best choice on smooth, well-seeded particle fields,
while a learned model can dominate on harder, sparser, or noisier
inputs. Picking a single estimator up front and applying it
everywhere leaves accuracy on the table whenever the data drifts
outside that estimator's comfort zone.

**ART of PIV learns how to dynamically select the best estimator for
the current image pair.** Given a pool of sub-estimators (any
combination of correlation, optical-flow, and neural methods) and an
image pair, a small *router* network predicts which sub-estimator is
expected to produce the lowest-error flow on that pair. The router is
trained from supervision computed against ground truth (or
EPE-per-estimator pulled from a cache, when available), and at
inference time it routes each pair to the chosen sub-estimator and
returns its flow.

The router is intentionally cheap: it operates on the raw image pair
and emits a Q-value per estimator. The expensive part — running a
flow estimator — happens only for the selected one(s).

![Schematic overview of ART of PIV: image pair feeds the router, which selects one sub-estimator from the configured pool.](art_of_piv_graphics.png)

## How it works

1. **Router.** A small CNN+MLP that takes `(prev, curr)` images and
   emits a Q-value per sub-estimator. The action (i.e., which
   sub-estimator to dispatch) comes from an exploration policy on top
   of those Q-values (e.g. epsilon-greedy with a schedule during
   training, greedy at eval).

2. **Sub-estimator dispatch.** At inference, the action picks a single
   sub-estimator, which is run as usual to produce the final flow.
   Outputs are merged back into the estimator history so downstream
   consumers see a normal `FlowFieldEstimator`.

3. **Cache short-circuit.** Training a router is much cheaper than
   running every sub-estimator on every batch. When per-estimator EPE
   values are available from a cache, the inference path skips
   running the sub-estimators entirely during the train step and
   computes the loss directly against the cached EPEs.

4. **Training modes.** Targets come from immediate ground-truth EPE
   (or cached EPE) per estimator. Three modes are supported:
   - `bandits` (contextual-bandit-like): reward for the selected
     action only.
   - `regression`: predict per-estimator Q-values matching their
     (negated) EPEs.
   - `classification`: predict the index of the lowest-EPE estimator
     (logits).

## Training objectives

Notation: image pair $s = (\text{prev}, \text{curr})$; configurable pool
of $K$ sub-estimators $\{f_1, \dots, f_K\}$; ground-truth flow $\varphi$;
per-sub-estimator endpoint error
$e_k(s, \varphi) = \| f_k(s) - \varphi \|$; router output
$Q_\theta(s) \in \mathbb{R}^K$; optional target transform $\tau$
(default identity; can be set to e.g. $\log$).

All three modes share the same supervision signal — the EPE vector
$\mathbf{e}(s, \varphi) \in \mathbb{R}^K$ — and differ only in how the
router head is asked to *use* it.

**Regression.** Treat each $Q_\theta(s)_k$ as a prediction of the
(transformed, negated) per-sub-estimator error. Loss is a per-arm
regression:

$$\mathcal{L}_{\text{reg}}(\theta) = \frac{1}{K}\sum_{k=1}^{K} \ell\bigl(Q_\theta(s)_k,\; \tau(-e_k(s,\varphi))\bigr)$$

with $\ell$ configurable (`l2`, `huber`, …; see `loss_config`).

**Bandits.** Same predicting-the-error paradigm, but the loss is
applied only to the chosen arm $a \sim \pi(\,\cdot \mid s; Q_\theta)$
(e.g. $\epsilon$-greedy on $Q_\theta$):

$$\mathcal{L}_{\text{bandit}}(\theta) = \ell\bigl(Q_\theta(s)_a,\; \tau(-e_a(s,\varphi))\bigr).$$

Contextual-bandit-like: a single action is committed and a single
scalar reward $-e_a$ supervises it.

**Classification.** Treat $Q_\theta(s)$ as logits over the $K$ arms and
ask the router to pick the best arm. The target is the index of the
lowest-error sub-estimator $a^\star = \arg\min_k e_k(s, \varphi)$:

$$\mathcal{L}_{\text{cls}}(\theta) = -\log \operatorname{softmax}\bigl(Q_\theta(s)\bigr)_{a^\star}.$$

## When to use it

- You have several heterogeneous flow estimators available and no
  single one dominates the others on your data distribution.
- You can produce ground-truth flow (synthetic or otherwise) or have
  cached EPE-per-estimator for the training data.
- You're willing to pay the router's small forward cost in exchange
  for picking a better sub-estimator per pair.

End-to-end usage examples live in:

- `examples/12_caching_art_of_piv.py`
- `examples/13_caching_art_of_piv_mixed.py`

## Folder structure

This folder is split along architectural seams rather than by file
size, so each module owns one concern.

- `estimator.py` — the public `ArtOfPIVEstimator` class. Owns
  configuration validation, lazy sub-estimator construction, and
  dependency wiring; exposes `_estimate`, `_run_router`, `q_estimate`,
  and `create_train_step`.
- `router.py` — pure routing math: builds router input from
  `(prev, curr)`, runs `q_estimate`, and applies the exploration
  policy to produce actions. Knows nothing about sub-estimators,
  caches, or metrics.
- `inference.py` — runtime orchestration: reads history
  (`images`, `estimates`, `keys`), calls the router, short-circuits
  on cached EPE, dispatches selected sub-estimators, and emits
  inference metrics.
- `training_steps.py` — supervised train-step factory. Receives
  `q_estimate_fn`, `run_router_fn`, and `estimate_fn` as injected
  callables so the same routing code is reused at train and eval.
- `caching.py` — unified EPE-cache preload / indexing / enrichment.
- `metrics.py` — inference and training metrics (Q-value summaries,
  per-estimator selection counts, etc.).
- `oracle_selector.py` — `ArtOfPIVOracleEstimator`: a non-learned
  `ArtOfPIVEstimator` that picks the lowest-EPE sub-estimator, from the
  cached errors when the batch has them and from the ground truth
  otherwise. Useful as an upper bound on what a perfect router could
  achieve.
- `types.py` — protocol types (`RouterQEstimateFn`,
  `RouterOutputs`, `EstimateFn`, …) that keep module boundaries
  explicit.
- `validation/` — config-validation helpers (router config, training
  config, basic schema checks).

### How the modules interact

Inference / evaluation path:

1. `ArtOfPIVEstimator.__call__` (in `estimator.py`) delegates to the
   base estimator.
2. The base calls `ArtOfPIVEstimator._estimate`.
3. `_estimate` calls `inference.estimate(...)`.
4. `inference.estimate` calls `self._run_router(...)`, which calls
   `router.run_router(...)` → `router.q_estimate(...)`.
5. `inference.estimate` then either short-circuits on
   `cache_payload.epe_all` or calls `inference.run_subestimators(...)`.
6. Flow + extras (`choices`) are merged into history by the base.

Training path:

1. `create_train_step()` → `create_supervised_train_step()` in
   `training_steps.py`.
2. The step receives injected callables (`q_estimate_fn`,
   `run_router_fn`, `estimate_fn`).
3. The mode (`bandits` / `regression` / `classification`) and cache
   availability decide how targets are built from `epe_all` or
   ground truth.
4. Loss is computed against router outputs, gradients are applied,
   and per-mode metrics are emitted.

### Key data contracts

- Router input images `prev`, `curr`: `(B, H, W)`.
- Router output `q_values`: `(B, num_estimators)`. In `classification`
  mode these are treated as logits.
- Router output `actions`: `(B,)`.
- History state fields used here:
  - `state["images"]`: `(B, T, H, W)`,
  - `state["estimates"]`: `(B, T, H, W, 2)`,
  - `state["keys"]`: `(B, 2)`.
