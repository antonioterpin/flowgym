"""Module that implements Farneback for use in the Estimator framework."""

from pyoptflow import HornSchunck
import numpy as np
import jax.numpy as jnp
import cv2

from goggles.history.types import History
from flowgym.flow.base import FlowFieldEstimator


def _gaussian(img: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma)


def _warp(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + u).astype(np.float32)
    map_y = (y + v).astype(np.float32)
    return cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
    )


def _hs_multiscale(
    img1: np.ndarray,
    img2: np.ndarray,
    levels: int = 6,
    alpha: float = 100,
    iters: int = 200,
    sigma: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    Pyr0, Pyr1 = [img1], [img2]
    for _ in range(1, levels):
        Pyr0.insert(0, cv2.pyrDown(Pyr0[0]))
        Pyr1.insert(0, cv2.pyrDown(Pyr1[0]))

    u = np.zeros(img1.shape, np.float32)
    v = np.zeros(img2.shape, np.float32)
    for A, B in zip(Pyr0, Pyr1):
        H, W = A.shape
        u = cv2.resize(u * 2, (W, H), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v * 2, (W, H), interpolation=cv2.INTER_LINEAR)

        B_warp = _warp(B, u, v)
        du, dv = HornSchunck(
            _gaussian(A, sigma),
            _gaussian(B_warp, sigma),
            alpha=alpha,
            Niter=iters,
            verbose=False,
        )
        u += du
        v += dv
    return u, v


class HornSchunckEstimator(FlowFieldEstimator):
    """HornSchunck flow field estimator."""

    def __init__(
        self,
        alpha: float = 100,
        n_iterations: int = 200,
        levels: int = 8,
        sigma: float = 0.1,
        **kwargs,
    ):
        """Initialize the HornSchunck estimator.

        Args:
            alpha: Regularization parameter for the flow field.
            n_iterations: Number of iterations for the algorithm.
            levels: Number of pyramid levels to use.
            sigma: Standard deviation for Gaussian smoothing.
            **kwargs: Additional keyword arguments for the base class.
        """
        if alpha <= 0.0:
            raise ValueError(f"alpha {alpha} must be > 0.0.")

        if n_iterations <= 0 or not isinstance(n_iterations, int):
            raise ValueError(f"n_iterations {n_iterations} must be a positive integer.")

        if levels <= 0 or not isinstance(levels, int):
            raise ValueError(f"levels {levels} must be a positive integer.")

        if sigma <= 0.0:
            raise ValueError(f"sigma {sigma} must be > 0.0.")

        self.alpha = alpha
        self.n_iterations = n_iterations
        self.levels = levels
        self.sigma = sigma

        super().__init__(**kwargs)

    def _estimate(
        self, image: jnp.ndarray, state: History, _, __
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the flow field using Farneback.

        Args:
            image: Input image.
            state: Current state of the estimator.
            _: Unused parameter.
            __: Unused parameter.

        Returns:
            Computed flow field.
            placeholder for additional outputs.
            placeholder for metrics.
        """
        # Note: Host callbacks, pure_functions calls, etc. seem
        # to be not particularly more efficient and also they are not as supported.
        # The support depends on the version of JAX, on the GPUs, etc.
        # After experimenting with them, I think it is ok to stick to a for loop...
        flows = np.zeros(image.shape + (2,), dtype=np.float32)
        for i in range(image.shape[0]):
            img1 = state["images"][i, -1, ...]
            img2 = image[i, ...]
            # Convert to numpy array
            img1 = np.asarray(img1)
            img2 = np.asarray(img2)
            u, v = _hs_multiscale(
                img1,
                img2,
                levels=self.levels,
                alpha=self.alpha,
                iters=self.n_iterations,
                sigma=self.sigma,
            )
            flows[i, ...] = np.stack((u, v), axis=-1)

        return jnp.asarray(flows), {}, {}
