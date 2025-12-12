"""Utility functions and classes for logging, timing."""

import csv
import importlib
from pathlib import Path
from types import ModuleType
from typing import Literal, overload

import goggles as gg
import jax
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = gg.get_logger(__name__, with_metrics=True)

DEBUG = False


@overload
def setup_logging(
    debug: bool = False,
    log_dir: str | Path | None = None,
    use_wandb: Literal[False] = False,
) -> gg.TextLogger: ...


@overload
def setup_logging(
    debug: bool = False,
    log_dir: str | Path | None = None,
    use_wandb: Literal[True] = True,
) -> gg.GogglesLogger: ...


def setup_logging(
    debug: bool = False,
    log_dir: str | Path | None = None,
    use_wandb: bool | None = False,
) -> gg.TextLogger | gg.GogglesLogger:
    """Initialize structured logging via Goggles.

    Args:
        debug: Enable DEBUG-level logging.
        log_dir: Optional directory to persist logs (JSONL).
        use_wandb: Whether to log to Weights & Biases.

    Returns:
        Root structured logger for the current process.
    """
    # 1. Attach console handler (pretty printing)
    gg.attach(
        gg.ConsoleHandler(
            name="flowgym.console",
            level=gg.DEBUG if debug else gg.INFO,
        ),
        scopes=["global"],
    )

    # 2. Optionally attach local storage for JSONL persistence
    if log_dir is not None:
        gg.attach(
            gg.LocalStorageHandler(
                path=Path(log_dir),
                name="flowgym.jsonl",
            ),
            scopes=["global"],
        )

    # 3. Optionally attach WandB if available and desired
    if use_wandb:
        gg.attach(
            gg.WandBHandler(project="flowgym"),
            scopes=["global"],
        )

    logger = gg.get_logger("flowgym", with_metrics=True)
    logger.info("Goggles logging initialized.")
    return logger


GracefulShutdown = gg.GracefulShutdown

load_configuration = gg.load_configuration


def clean_for_logging(obj):
    """Recursively cleans an object for logging.

    This function converts complex objects into simple types that can be logged.
    It handles dictionaries, lists, tuples, and JAX/NumPy 0d arrays.

    Args:
        obj: The object to clean for logging.

    Returns:
        The cleaned objectin a human-readable format for logging.
    """
    if isinstance(obj, dict):
        return {k: clean_for_logging(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(clean_for_logging(v) for v in obj)
    elif hasattr(obj, "item") and callable(obj.item):  # JAX/NumPy 0d arrays
        try:
            return obj.item()
        except Exception:
            return obj
    else:
        return obj


def write_dicts_to_csv(filename, all_rows):
    """Write a CSV with all keys appearing in any dict in all_rows as columns.

    Args:
        filename: str, the name of the file to write to
        all_rows: list of dicts
    """
    # Compute the union of all keys
    fieldnames = set()
    for row in all_rows:
        fieldnames.update(row.keys())
    fieldnames = sorted(fieldnames)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def viz(
    flow,
    img=None,
    scalar_field=None,
    scalar_label=None,
    downsample=1,
    bilinear=False,
    title="Flow Field",
):
    """Visualizes the flow field overlaid on the image.

    Args:
        flow: Flow field of shape (H, W, 2).
        img: Image of shape (H, W) or (H, W, 3).
        scalar_field: Scalar field of shape (H, W).
        scalar_label: Name of the scalar field for colorbar label.
        downsample: Factor by which to downsample.
        bilinear: Whether to use bilinear interpolation for downsampling.
        title: Title of the plot.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for viz function."
        "Please install it with: uv sync --extra extra")
    def _downsample(arr, factor):
        """Downsample an array by a given factor."""
        if bilinear:
            return cv2.resize(
                arr,
                (arr.shape[1] // factor, arr.shape[0] // factor),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            return arr[::factor, ::factor]

    u = _downsample(flow[..., 0], downsample)
    v = _downsample(flow[..., 1], downsample)
    H, W = u.shape

    # optionally downsample image
    if img is not None:
        img = _downsample(img, downsample)

    # optionally downsample scalar field
    if scalar_field is not None:
        scalar_field = _downsample(scalar_field, downsample)

    # 2) build grid in the downsampled coordinate system
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    fig, ax = plt.subplots(figsize=(8, 8))

    # show image as background if provided
    if img is not None:
        ax.imshow(img, cmap="gray", alpha=0.5, origin="upper")

    # 3) plot quiver, coloring by scalar_field if given
    if scalar_field is not None:
        Q = ax.quiver(
            X,
            Y,
            u,
            v,
            scalar_field,
            scale_units="xy",
            cmap="viridis",
            pivot="mid",
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(Q, cax=cax)
        if scalar_label:
            cbar.set_label(scalar_label)
    else:
        ax.quiver(X, Y, u, v, scale_units="xy", color="r", pivot="mid")

    # 4) clean up axes
    ax.set_aspect("equal")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # invert y-axis so origin is top-left
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.tight_layout()
    return fig


def flow_magnitude_heatmap(
    flow: np.ndarray, maxrad: float | None = None, minrad: float | None = None
) -> np.ndarray:
    """Create a heatmap from optical flow magnitude.

    Convert a flow field to an RGB image where color represents magnitude
    from blue (low) to red (high).

    Args:
        flow: Array (H, W, 2), the optical flow field.
        maxrad: Optional float, maximum value for normalization. If None, uses flow max.
        minrad: Optional float, minimum value for normalization. If None, uses flow min.

    Returns:
        RGB image (H, W, 3), dtype float32, values in [0, 1].
    """
    mag = np.linalg.norm(flow, axis=-1)  # (H, W)

    if maxrad is None:
        maxrad = np.max(mag) + 1e-6  # avoid divide-by-zero

    if minrad is None:
        minrad = np.min(mag) + 1e-6  # avoid divide-by-zero

    if not isinstance(maxrad, (float, int)):
        raise ValueError("maxrad must be a float or None.")

    if not isinstance(minrad, (float, int)):
        raise ValueError("minrad must be a float or None.")

    norm_mag = np.clip(
        (mag - minrad) / (maxrad - minrad), 0.0, 1.0
    )  # Normalize to [0, 1]

    # Use a colormap: 'jet' (blue to red)
    cmap = plt.get_cmap(
        "jet"
    )  # alternative: 'plasma', 'inferno', 'coolwarm', etc.
    rgb = cmap(norm_mag)[..., :3]  # Discard alpha

    return rgb.astype(np.float32)


def fig_to_np(fig, dtype=np.float32, drop_alpha=True) -> np.ndarray:
    """Convert a Matplotlib figure to an image array.

    Args:
        fig: matplotlib.figure.Figure
        dtype: desired numpy dtype (e.g. np.float32 or np.uint8)
        drop_alpha: if True, return RGB instead of RGBA

    Returns:
        np.ndarray of shape (H, W, C) with C = 3 or 4.
    """
    # Make sure the figure is rendered
    fig.canvas.draw()

    # buffer_rgba() works in new Matplotlib versions
    buf = np.asarray(fig.canvas.buffer_rgba())  # uint8, shape (H, W, 4)

    if drop_alpha and buf.shape[-1] == 4:
        buf = buf[..., :3]  # drop alpha → RGB

    if np.issubdtype(dtype, np.floating):
        buf = buf.astype(np.float32) / 255.0
        if dtype is not np.float32:
            buf = buf.astype(dtype)
    else:
        buf = buf.astype(dtype)

    return buf


def block_until_ready_dict(tree: dict) -> None:
    """Block until ready for all jax.Array leaves in a (possibly nested) dict.

    Args:
        tree: dict containing jax.Array leaves.
    """
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if isinstance(x, jax.Array) else x,
        tree,
    )


def log_flow_estimate(
    step_idx: int,
    log_every: int,
    flow: np.ndarray,
    old_gt: np.ndarray,
    episode_idx: int,
    choices: np.ndarray | None = None,
) -> None:
    """Log flow estimate and ground truth as images to wandb.

    TODO: check if this works correctly after the refactor.

    Args:
        step_idx: Current training step index.
        log_every: Frequency of logging.
        flow: Estimated flow field, shape (B, H, W, 2).
        old_gt: Ground truth flow field, shape (B, H, W, 2).
        episode_idx: Current episode index.
        choices: Optional array of algorithm choices, shape (B,).
    """
    if step_idx % log_every != 0:
        return
    image_counter = (step_idx - 1) // log_every
    errors = np.linalg.norm(flow - old_gt, axis=-1)
    epe_per_flow = np.mean(errors, axis=(1, 2))
    mag = np.linalg.norm(old_gt[0], axis=-1)
    maxrad = np.max(mag) + 1e-6
    downsample = flow[0].shape[0] // 32

    if choices is not None and choices.shape[0] > 0:
        title = "Flow Field Estimate with Algorithm "
        f"{choices[0]}"
        f" and EPE {epe_per_flow[0]:.3f}"
    else:
        title = "Flow Field Estimate with EPE "
        f"{epe_per_flow[0]:.3f}"
    fig1 = viz(
        np.asarray(flow[0]),
        scalar_field=np.asarray(np.linalg.norm(flow[0], axis=-1)),
        scalar_label="Magnitude",
        downsample=downsample,
        bilinear=True,
        title=title,
    )

    fig2 = viz(
        np.asarray(old_gt[0]),
        scalar_field=np.asarray(np.linalg.norm(old_gt[0], axis=-1)),
        scalar_label="Magnitude",
        downsample=downsample,
        bilinear=True,
        title="gt",
    )

    fig3 = flow_magnitude_heatmap(np.asarray(flow[0]), maxrad=float(maxrad))
    fig4 = flow_magnitude_heatmap(np.asarray(old_gt[0]), maxrad=float(maxrad))

    # ---- Log the figure to wandb ----
    logger.image(
        fig_to_np(fig1),
        name="Flow Estimate",
        step=episode_idx,
        image_counter=image_counter,
    )
    logger.image(
        fig_to_np(fig2),
        name="Flow GT",
        step=episode_idx,
        image_counter=image_counter,
    )
    logger.image(
        np.asarray(fig3),
        name="Flow Estimate Heatmap",
        step=episode_idx,
        image_counter=image_counter,
    )
    logger.image(
        np.asarray(fig4),
        name="Flow GT Heatmap",
        step=episode_idx,
        image_counter=image_counter,
    )

    plt.close(fig1)
    plt.close(fig2)


def optional_import(module_path: str) -> ModuleType | None:
    """Attempt to import an optional module.

    This function never raises ImportError. If the module cannot be imported,
    it simply returns None.

    Args:
        module_path (str): The full dotted-path of the module to import.

    Returns:
        ModuleType | None: The imported module, or None if unavailable.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError:
        return None


class MissingDependency:
    """Placeholder for objects that require an optional dependency.

    Calling an instance of this class raises ImportError with a helpful message.
    This allows registry entries or class factories to always exist, even when
    optional extras are not installed.
    """

    def __init__(self, name: str, extras: list[str] | str) -> None:
        """Initialize the MissingDependency placeholder.

        Args:
            name: Name of the component.
            extras: List of optional dependency groups.
        """
        self.name = name
        if isinstance(extras, str):
            extras = [extras]
        self.extras = extras

    def __call__(self) -> None:
        """Raise ImportError when attempting to instantiate the missing component.

        Raises:
            ImportError: Always raised to indicate the missing dependency.
        """
        raise ImportError(
            f"The component '{self.name}' requires one of the optional dependencies: "
            f"{self.extras}. Please install it via 'uv sync --extra [{'|'.join(self.extras)}]'. "
            f"Example: 'uv sync --extra {self.extras[0]}'. "
            "If you have forked the repo, consider using 'uv sync --all-groups'."
        )
