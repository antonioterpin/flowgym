# 🎨 ART of PIV estimator

```{include} ../../../src/flowgym/flow/art_of_piv/README.md
:start-line: 2
:relative-images:
```

## Minimal estimator config

A complete ART of PIV configuration touches three YAML files: a model
config, an *estimators list* (the pool of sub-estimators the router
chooses between), and a dataset config with a `caching:` block. The
fragments below show the ART of PIV-specific fields; for the dataset
side see
[Configuration and data flow](../../user-guide/configuration-and-data.md)
and [the caching examples](../../examples/caching.md).

Model config (`model.yaml`):

```yaml
estimator: art_of_piv
estimate_type: flow
config:
  jit: true

  # Router architecture (CNN feature extractor + MLP head).
  router_config:
    features: [32, 64]      # conv-block output channels
    kernel_sizes: [3, 3]    # one per conv block
    strides: [2, 2]         # one per conv block
    norm_fn: none           # "none" | "batch" | "instance" | "group"
    mlp_hidden_dim: 128

  # Pool of sub-estimators the router can dispatch to.
  estimators_list_path: estimators_list.yaml

  # Training mode.
  training_mode: bandits    # "bandits" | "regression" | "classification"

  # Loss + optimizer for the router.
  loss_config: { name: l2 }
  optimizer_config:
    name: adam
    learning_rate: 0.0001

  # Exploration over actions during training.
  exploration_config:
    name: epsilon_greedy
    schedule: { name: constant, value: 0.1 }

  output_full_res: true
```

Estimators list (`estimators_list.yaml`) — every entry is a
fully-formed estimator config; the router learns to pick between
them:

```yaml
estimators:
  - name: DIS_Small
    estimator: dis_jax
    config: { patch_size: 7,  patch_stride: 2, grad_desc_iters: 3,
              start_level: 0, levels: 2, jit: true,
              output_full_res: true, use_temporal_propagation: false }
  - name: DIS_Medium
    estimator: dis_jax
    config: { patch_size: 11, patch_stride: 3, grad_desc_iters: 3,
              start_level: 0, levels: 2, jit: true,
              output_full_res: true, use_temporal_propagation: false }
  - name: DIS_Large
    estimator: dis_jax
    config: { patch_size: 13, patch_stride: 4, grad_desc_iters: 3,
              start_level: 0, levels: 2, jit: true,
              output_full_res: true, use_temporal_propagation: false }
```

To use a heterogeneous pool, replace any entry's `estimator:` with a
different registered estimator name (e.g. `raft_jax`) and adapt its
`config:` block — see
[`examples/13_caching_art_of_piv_mixed.py`](../../examples/art-of-piv.md)
for a DIS+RAFT setup.

Runnable end-to-end versions of these configs live in
[`examples/12_caching_art_of_piv.py`](../../examples/art-of-piv.md) and
its mixed-pool counterpart.

## Router preprocessing

`router_preprocess:` takes the same steps as `preprocess` and applies them
only to the frames the router sees, e.g. to run the router on a crop while
the sub-estimators use the full frame. It also applies in training, and is not
supported with `use_last_estimate`.

```yaml
config:
  router_preprocess:
    - {name: crop_special, target_height: 512, target_width: 512,
       fraction_v: 0.15, fraction_h: 0.85}
```

## API reference

For the nested encoder/head router format, `encoder.first_conv_stride`
sets the stride of the first convolution. It defaults to `1` and accepts
positive integers; values above one require `use_residual: false`.
This downsamples camera images within the learned encoder before the
remaining convolution blocks and pooling stages.

```{eval-rst}
.. automodule:: flowgym.flow.art_of_piv.estimator
   :members:

.. automodule:: flowgym.flow.art_of_piv.router
   :members:

.. automodule:: flowgym.flow.art_of_piv.inference
   :members:

.. automodule:: flowgym.flow.art_of_piv.training_steps
   :members:

.. automodule:: flowgym.flow.art_of_piv.caching
   :members:

.. automodule:: flowgym.flow.art_of_piv.types
   :members:
```
