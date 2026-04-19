# First estimate

This is the smallest meaningful FlowGym example in the repository. It sits
between the minimal snippets in [Getting started](../getting-started/index.md) and
the config-driven workflows later in this section.

It answers one question: "Can I instantiate an estimator and run it on a
pair of images without setting up dataset YAML files yet?"

## What it does

- builds a `dis_jax` estimator with `flowgym.make.make_estimator`
- creates two tiny grayscale images in memory
- initializes estimator state from the first image
- runs one estimate on the second image
- prints the returned tensor shape, metric keys, and a small flow summary

## Script

- `examples/00_first_estimate.py`

Run it from the repository root:

```bash
uv run python examples/00_first_estimate.py
```

## What to notice

- The image input shape is `(B, H, W)`.
- `make_estimator(...)` returns four pieces:
  trained state, state-construction function, estimate function, and model.
- The computed flow estimate lives in `state["estimates"][:, -1]`.
- This is the clearest place to inspect the public estimator API before
  moving to CLI workflows.

## Where to go next

- Go back to [Quick overview](../getting-started/quick-overview.md) when you
  want the shortest API recap.
- Continue to [Flow evaluation](flow-eval.md) when you want the first
  config-backed repository workflow.
