import pytest
import jax.numpy as jnp
import numpy as np
import jax
import timeit
import flowgym.flow.postprocess as ip  # module under test
from flowgym.utils import load_configuration

config = load_configuration("src/flowgym/config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_POST_PROCESS"]


@pytest.mark.parametrize(
    "method, kwargs",
    [
        (ip.tile_average_interpolation, {"radius": 1}),
        (ip.tile_average_interpolation, {"radius": 3}),
        (ip.laplace_interpolation, {"num_iter": 4}),
        (ip.laplace_interpolation, {"num_iter": 32}),
        # (ip.laplace_interpolation, {"num_iter": 256}),
    ],
)
@pytest.mark.parametrize(
    "B, H, W",
    [
        (1, 16, 16),
        (32, 16, 16),
    ],
)
@pytest.mark.parametrize("p", [0.95, 0.9])
@pytest.mark.parametrize("seed", [0, 42])
def test_algorithms(method, kwargs, B, H, W, p, seed):
    key = jax.random.PRNGKey(seed)
    valid = jax.random.bernoulli(key, p, (B, H, W))
    flow = jnp.ones((B, H, W, 2))  # Uniform flow field
    flow = jnp.where(valid[..., None], flow, jnp.zeros_like(flow))
    out, new_valid, _ = method(flow, valid=valid, state=None, **kwargs)
    epe = jnp.mean(jnp.linalg.norm(out - 1, axis=-1))
    assert epe < 3e-3, f"Mean EPE {epe:.6f} is too high for {method.__name__}"
    assert jnp.all(new_valid), "Output validity mask should be all True."


def tile_average_interpolation_naive(
    flow: np.ndarray, valid: np.ndarray, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy baseline: triple-nested loops, easy to read, slow."""
    flow_out = flow.copy()
    valid_out = valid.copy()
    B, H, W, C = flow.shape
    for b in range(B):
        for y in range(H):
            for x in range(W):
                if not valid[b, y, x]:
                    y0, y1 = max(0, y - radius), min(H, y + radius + 1)
                    x0, x1 = max(0, x - radius), min(W, x + radius + 1)

                    mask = valid[b, y0:y1, x0:x1]
                    if mask.any():
                        patch = flow[b, y0:y1, x0:x1][mask]
                        flow_out[b, y, x] = patch.mean(axis=0)
                        valid_out[b, y, x] = True
    return flow_out, valid_out


def _jacobi_step(curr, fixed):
    B, H, W, C = curr.shape
    out = curr.copy()
    for b in range(B):
        for y in range(H):
            for x in range(W):
                if not fixed[b, y, x]:
                    for c in range(C):
                        up = curr[b, y - 1 if y > 0 else y, x, c]
                        down = curr[b, y + 1 if y < H - 1 else y, x, c]
                        left = curr[b, y, x - 1 if x > 0 else x, c]
                        right = curr[b, y, x + 1 if x < W - 1 else x, c]
                        out[b, y, x, c] = 0.25 * (up + down + left + right)
    return out


def laplace_interpolation_naive(flow, valid, num_iter):
    curr = flow.copy()
    for _ in range(num_iter):
        curr = _jacobi_step(curr, valid)
    return curr, np.ones_like(valid, dtype=bool)


@pytest.mark.parametrize(
    "method,method_naive,kwargs",
    [
        (
            ip.tile_average_interpolation,
            tile_average_interpolation_naive,
            {"radius": 1},
        ),
        (
            ip.tile_average_interpolation,
            tile_average_interpolation_naive,
            {"radius": 3},
        ),
        (ip.laplace_interpolation, laplace_interpolation_naive, {"num_iter": 4}),
        (ip.laplace_interpolation, laplace_interpolation_naive, {"num_iter": 32}),
        (ip.laplace_interpolation, laplace_interpolation_naive, {"num_iter": 256}),
    ],
)
@pytest.mark.parametrize(
    "B, H, W",
    [
        (1, 64, 64),
        (32, 64, 64),
    ],
)
@pytest.mark.parametrize("p", [0.95, 0.9, 0.8, 0.1])
@pytest.mark.parametrize("seed", [0])
def test_algorithms_naive(method, method_naive, kwargs, B, H, W, p, seed):
    key = jax.random.PRNGKey(seed)
    valid = jax.random.bernoulli(key, p, (B, H, W))
    flow = jax.random.normal(key, (B, H, W, 2))
    out, new_valid, _ = method(flow, valid=valid, state=None, **kwargs)
    out2, new_valid2 = method_naive(np.asarray(flow), np.asarray(valid), **kwargs)
    assert new_valid.shape == (B, H, W), "Output validity mask shape mismatch."
    assert new_valid2.shape == (B, H, W), "Naive output validity mask shape mismatch."
    assert jnp.all(new_valid == new_valid2), "Validity masks do not match."
    out = out * new_valid[..., None]
    out2 = jnp.asarray(out2 * new_valid2[..., None])
    delta = jnp.mean(jnp.linalg.norm(out - out2, axis=-1))
    assert delta < 1e-5, f"Mean discrepancy from reference {delta:.6f} is too high."


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize(
    "method,kwargs,B,H,W,time_limit",
    [
        (ip.tile_average_interpolation, {"radius": 3}, 1, 64, 64, 5.7e-5),
        (ip.tile_average_interpolation, {"radius": 3}, 512, 64, 64, 1.7e-4),
        (ip.tile_average_interpolation, {"radius": 3}, 1, 512, 512, 6.5e-5),
        (ip.tile_average_interpolation, {"radius": 3}, 64, 512, 512, 1.3e-3),
        (ip.tile_average_interpolation, {"radius": 3}, 1, 1024, 1024, 1.5e-4),
        (ip.tile_average_interpolation, {"radius": 3}, 16, 1024, 1024, 1.3e-3),
        (ip.laplace_interpolation, {"num_iter": 32}, 1, 64, 64, 1e-4),
        (ip.laplace_interpolation, {"num_iter": 32}, 512, 64, 64, 4.5e-4),
        (ip.laplace_interpolation, {"num_iter": 32}, 1, 512, 512, 2e-4),
        (ip.laplace_interpolation, {"num_iter": 32}, 64, 512, 512, 1e-2),
        (ip.laplace_interpolation, {"num_iter": 32}, 1, 1024, 1024, 3.2e-4),
        (ip.laplace_interpolation, {"num_iter": 32}, 16, 1024, 1024, 1e-2),
    ],
)
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("p", [0.1, 0.2, 0.5])
def test_algorithms_speed(method, kwargs, B, H, W, time_limit, seed, p):
    # Generate random flow field with a single outlier
    key = jax.random.PRNGKey(seed)
    flow = jax.random.normal(key, (B, H, W, 2))
    mask = jax.random.bernoulli(key, p, (B, H, W))

    fn_jit = jax.jit(lambda f, m, s: method(f, valid=m, state=s, **kwargs))

    def run():
        res, res_valid, _ = fn_jit(flow, mask, None)
        res.block_until_ready()
        res_valid.block_until_ready()

    time_jit = timeit.repeat(
        run,
        repeat=REPETITIONS,
        number=NUMBER_OF_EXECUTIONS,
    )
    jax_time = min(time_jit) / NUMBER_OF_EXECUTIONS

    assert jax_time < time_limit, f"{method.__name__} took too long: {jax_time:.6f}s"
