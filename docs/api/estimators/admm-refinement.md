# 🤝 ADMM refinement

Different tunings — or different algorithms — often quantify the flow best
in different regions of the same image pair. **ADMM refinement** parallelizes
the instantaneous flow quantification with multiple algorithms and reconciles
them in a consensus framework based on the alternating direction method of
multipliers, seamlessly incorporating priors such as smoothness and
incompressibility. On a dense-inverse-search estimator it lowers the
end-point error by up to 20% at a 60 Hz inference rate, with further gains
from outlier rejection.

## How it works

Let $x_i$ be the flow field of estimator $i$ and $z$ the shared consensus
field. In scaled form, ADMM iterates

$$x_i \leftarrow \arg\min_{x_i}\; f_i(x_i)
  + \tfrac{\rho}{2}\lVert x_i - z + u_i \rVert^2,$$

$$z \leftarrow \arg\min_{z}\; g(z)
  + \tfrac{\rho}{2}\sum_i \lVert x_i - z + u_i \rVert^2,$$

$$u_i \leftarrow u_i + x_i - z,$$

where $f_i$ keeps each estimate close to its estimator's output, $g$ applies
the smoothness/incompressibility priors, $u_i$ are the (scaled) dual
variables, and $\rho$ is the penalty weight. Iteration stops once the primal
and dual residuals fall below tolerance. See the paper for the full
formulation.

## Notes

- **Method paper:** A. Bonomi, F. Banelli, A. Terpin, *Particle Image
  Velocimetry Refinement via Consensus ADMM*, 2025.
  [arXiv:2512.11695](https://arxiv.org/abs/2512.11695)
- **ADMM background:** S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein,
  *Distributed Optimization and Statistical Learning via the Alternating
  Direction Method of Multipliers*, 2011.

## Minimal estimator config

The consensus estimator wraps a pool of sub-estimators (the
`estimators_list`) and refines their outputs with ADMM. The
ADMM-specific fields live under `consensus_config`:

```yaml
estimator: consensus
estimate_type: flow
config:
  jit: true
  consensus_algorithm: admm
  estimators_list_path: <sub-estimators>.yaml
  consensus_config:
    solver_flows: closed_form_l1   # per-estimator (x) update
    solver_consensus: adam         # consensus (z) update
    rho: 2.0                       # augmented-Lagrangian penalty
    max_admm_iterations: 30
    regularizer_list: [smoothness, divergence, laplacian]
    regularizer_weights: { smoothness: 0.0, divergence: 0.0, laplacian: 0.0 }
    eps_abs_stopping: 1e-6
    eps_rel_stopping: 1e-6
```

For the dataset side see
[Configuration and data flow](../../user-guide/configuration-and-data.md);
runnable end-to-end setups live under `experiments/piv-admm/`.

## API reference

The estimator class is documented under
[Consensus estimator](consensus.md); the ADMM solver itself lives in
`flowgym.flow.consensus.admm`:

```{eval-rst}
.. automodule:: flowgym.flow.consensus.admm
   :members:
```
