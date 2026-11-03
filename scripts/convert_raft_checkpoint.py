"""Convert a pretrained RAFT-PIV PyTorch checkpoint into a FlowGym artifact.

Writes ``<out-root>/{pytorch/<ckpt>, jax/checkpoints/0, metadata.json}``.
The Orbax tree under ``jax/checkpoints/0`` is what training configs point
``load_from`` at (with ``load_mode: params_only``); the original checkpoint
is copied under ``pytorch/`` so the artifact is self-contained.

Sources: RAFT32-PIV and RAFT256-PIV from Lagemann et al., Nature Machine
Intelligence 3, 2021 (Code Ocean capsule 7226151).

Example:

    uv run --extra other_methods python scripts/convert_raft_checkpoint.py \\
        --variant raft256 \\
        --ckpt ckpt.tar --out-root ~/raft_checkpoints/pretrained_raft256
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from flowgym.flow.raft.raft256_jax import RaftJax256Estimator
from flowgym.flow.raft.raft_jax import RaftJaxEstimator
from flowgym.make import load_model, save_model
from flowgym.nn.raft_torch_nn.flowNetsRAFT import RAFT
from flowgym.nn.raft_torch_nn.flowNetsRAFT256 import RAFT256
from flowgym.nn.raft_torch_to_jax import convert_raft_torch_state_dict

# variant -> (PyTorch module, FlowGym estimator, FlowGym registry name)
VARIANTS = {
    "raft32": (RAFT, RaftJaxEstimator, "raft_jax"),
    "raft256": (RAFT256, RaftJax256Estimator, "raft_jax_256"),
}


def main() -> None:
    """Convert, validate and save the checkpoint.

    Raises:
        ValueError: If the converted params do not fit the Flax model.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    torch_cls, estimator_cls, estimator_name = VARIANTS[args.variant]
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    # ``warp.grid`` is a non-learned buffer the upstream RAFT32 file carries.
    state_dict = {
        k: v
        for k, v in raw["model_state_dict"].items()
        if not k.startswith("warp.")
    }
    # Strict load: fails on anything but a RAFT-PIV state dict. RAFT32 and
    # RAFT256 share keys and shapes, so --variant must still be right.
    torch_cls().load_state_dict(state_dict, strict=True)

    estimator = estimator_cls(train=False)
    template = estimator.create_trainable_state(
        jnp.zeros((1, estimator.patch_size, estimator.patch_size)),
        jax.random.PRNGKey(0),
    )
    params = jax.tree_util.tree_map(
        jnp.asarray, convert_raft_torch_state_dict(state_dict)
    )
    if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(
        template.params
    ) or any(
        a.shape != b.shape
        for a, b in zip(
            jax.tree_util.tree_leaves(params),
            jax.tree_util.tree_leaves(template.params),
            strict=True,
        )
    ):
        raise ValueError("Converted params do not match the Flax model.")
    state = template.replace(params=params)

    step_dir = save_model(state, args.out_root / "jax", step=0)
    restored = load_model(step_dir, template, mode="params_only")
    for a, b in zip(
        jax.tree_util.tree_leaves(restored.params),
        jax.tree_util.tree_leaves(params),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)

    (args.out_root / "pytorch").mkdir(exist_ok=True)
    pt_copy = args.out_root / "pytorch" / args.ckpt.name
    if not pt_copy.exists():
        shutil.copy2(args.ckpt, pt_copy)
    metadata = {
        "variant": args.variant,
        "estimator": estimator_name,
        "source_pytorch_checkpoint": pt_copy.name,
        "source_epoch": raw.get("epoch"),
        "load_with": f"estimator: {estimator_name}, load_from: "
        f"{step_dir}, load_mode: params_only",
    }
    (args.out_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Saved {step_dir}")


if __name__ == "__main__":
    main()
