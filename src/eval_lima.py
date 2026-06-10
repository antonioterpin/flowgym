"""Standalone EPE evaluation for LIMA on a synthpix dataset.

Builds the estimator (optionally restoring a trained checkpoint via
``--load_from``) and reports endpoint-error (EPE) statistics over the dataset,
both full-field and excluding a 16 px border (the convention used in the LIMA
papers: Manickathan, Mucignat & Lunati, Exp. Fluids 64, 161, 2023,
https://doi.org/10.1007/s00348-023-03695-8, and Mucignat, Zdybał & Lunati,
Phys. Fluids 37, 105112, 2025, https://doi.org/10.1063/5.0283779). Reports a
per-scenario breakdown when batch file names are available.
"""

import argparse
import itertools
import re
from collections import defaultdict

import goggles as gg
import jax
import jax.numpy as jnp
import numpy as np
import synthpix
import yaml

from flowgym.make import make_estimator, select_gt
from flowgym.utils import setup_logging

logger = setup_logging(debug=False, use_wandb=False)


def _scenario(name: str) -> str:
    """Map a flow-field filename to its scenario label.

    Args:
        name: Flow-field file name or path.

    Returns:
        The scenario label (filename stem without the trailing index).
    """
    stem = str(name).split("/")[-1].replace(".mat", "")
    return re.sub(r"_\d+$", "", stem) or "unknown"


def main() -> None:
    """Evaluate LIMA and print EPE statistics."""
    parser = argparse.ArgumentParser(description="LIMA EPE evaluation.")
    parser.add_argument("--estimator", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--num_batches", type=int, default=None)
    parser.add_argument("--label", type=str, default="LIMA")
    args = parser.parse_args()

    with open(args.estimator) as f:
        ecfg = yaml.safe_load(f)
    with open(args.dataset) as f:
        dcfg = yaml.safe_load(f)
    if args.load_from:
        ecfg["load_from"] = args.load_from

    key = jax.random.PRNGKey(dcfg.get("seed", 0))
    key, subkey = jax.random.split(key)

    sampler = synthpix.make(dcfg, load_from=dcfg.get("load_from"))
    batch0 = next(sampler)
    gt0 = select_gt(ecfg["estimate_type"], batch0)

    trained_state, create_state_fn, compute_estimate_fn, _estimator = (
        make_estimator(
            ecfg,
            image_shape=(batch0.images1.shape[0], *dcfg["image_shape"]),
            estimate_shape=gt0.shape,
            load_from=ecfg.get("load_from"),
            rng=subkey,
        )
    )

    num_batches = args.num_batches or dcfg.get("num_batches", 45)
    epe_full: list[float] = []
    epe_crop: list[float] = []
    by_scen: dict[str, list[float]] = defaultdict(list)

    def all_batches():
        yield batch0
        yield from sampler

    for i, batch in enumerate(itertools.islice(all_batches(), num_batches)):
        key, sub = jax.random.split(key)
        gt = batch.flow_fields
        state = create_state_fn(batch.images1, sub)
        state, _metrics = compute_estimate_fn(
            batch.images2, state, trained_state, cache_payload=None
        )
        flow = state["estimates"][:, -1]
        epe = jnp.linalg.norm(flow - gt, axis=-1)  # (B, H, W)
        ef = np.asarray(jnp.mean(epe, axis=(1, 2)))
        ec = np.asarray(jnp.mean(epe[:, 16:-16, 16:-16], axis=(1, 2)))
        epe_full.extend(ef.tolist())
        epe_crop.extend(ec.tolist())
        files = getattr(batch, "files", None)
        if files is not None:
            for f, e in zip(files, ec, strict=False):
                by_scen[_scenario(f)].append(float(e))
        if (i + 1) % 10 == 0:
            logger.info(
                f"batch {i + 1}/{num_batches} running mean EPE "
                f"{np.mean(epe_crop):.4f}"
            )

    ef_arr = np.array(epe_full)
    ec_arr = np.array(epe_crop)
    print(f"\n===== {args.label} on {len(ec_arr)} samples =====")
    print(f"  Mean EPE (full field)   : {ef_arr.mean():.4f} px")
    print(f"  Mean EPE (excl. 16px)   : {ec_arr.mean():.4f} px")
    print(f"  Median EPE (excl. 16px) : {np.median(ec_arr):.4f} px")
    print(f"  Std EPE (excl. 16px)    : {ec_arr.std():.4f} px")
    if by_scen:
        print("  Per-scenario mean EPE (excl. 16px):")
        for scen in sorted(by_scen):
            vals = by_scen[scen]
            print(f"    {scen:26s} {np.mean(vals):.4f}  (n={len(vals)})")

    sampler.shutdown()
    gg.finish()


if __name__ == "__main__":
    main()
