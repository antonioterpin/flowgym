"""LimaPivEstimator: the LIMA flow-field estimator for Flow Gym.

Wraps :class:`flowgym.nn.lima_model.LimaModel` as a trainable
:class:`flowgym.flow.base.FlowFieldEstimator`, with the multi-level
Jacobian-penalised L1 loss of Manickathan, Mucignat & Lunati (Exp. Fluids
2023) and the replicate-padding / reduced-search-range defaults of Mucignat,
Zdybał & Lunati (Phys. Fluids 2025).

References:
    Manickathan, Mucignat & Lunati, Exp. Fluids 64, 161 (2023),
    https://doi.org/10.1007/s00348-023-03695-8 (architecture + loss, Eq. 7,
    Table 2 weights); Mucignat, Zdybał & Lunati, Phys. Fluids 37, 105112
    (2025), https://doi.org/10.1063/5.0283779 (padding + search range);
    Manickathan, Mucignat & Lunati, Meas. Sci. Technol. 33, 124006 (2022),
    https://doi.org/10.1088/1361-6501/ac8fae (kinematic training). See the
    ``flowgym.flow.lima`` package docstring for the full reference list.
"""

from collections.abc import Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from goggles.history.types import History

from flowgym.common.base import NNEstimatorTrainableState
from flowgym.flow.base import FlowFieldEstimator
from flowgym.flow.lima.process import _PAD_MODES
from flowgym.nn.lima_model import (
    DEFAULT_DECODER_CHANNELS,
    DEFAULT_DECODER_DILATIONS,
    DEFAULT_ENCODER_CHANNELS,
    LimaModel,
)
from flowgym.types import PRNGKey, SupervisedExperience, SupervisedTrainStep

# Per-level loss weights from Table 2 of Manickathan et al. (2023),
# ordered coarse -> fine, keyed by number of refinement levels.
_PAPER_LEVEL_WEIGHTS: dict[int, tuple[float, ...]] = {
    6: (0.0025, 0.005, 0.01, 0.02, 0.08, 0.32),
    4: (0.01, 0.02, 0.08, 0.32),
}
_PADDING_MODES: tuple[str, ...] = ("zeros", "replicate", "reflect", "circular")


def _as_int_tuple(value: Sequence[int], name: str) -> tuple[int, ...]:
    """Coerce a sequence to a tuple of positive ints, validating entries.

    Args:
        value: Sequence of integers (e.g. channel counts).
        name: Parameter name for error messages.

    Returns:
        The values as a tuple of ints.

    Raises:
        TypeError: If ``value`` is not a non-string sequence of ints.
        ValueError: If it is empty or contains a non-positive entry.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of ints, got {value!r}.")
    out = tuple(value)
    if not out:
        raise ValueError(f"{name} must be non-empty.")
    for entry in out:
        if not isinstance(entry, int) or isinstance(entry, bool):
            raise TypeError(f"{name} entries must be ints, got {entry!r}.")
        if entry <= 0:
            raise ValueError(f"{name} entries must be positive, got {entry}.")
    return out


class LimaPivEstimator(FlowFieldEstimator):
    """LIMA flow-field estimator (lightweight image matching for PIV)."""

    def __init__(
        self,
        encoder_channels: Sequence[int] = DEFAULT_ENCODER_CHANNELS,
        decoder_channels: Sequence[int] = DEFAULT_DECODER_CHANNELS,
        decoder_dilations: Sequence[int] = DEFAULT_DECODER_DILATIONS,
        refine_levels: int | None = None,
        search_range: int = 2,
        padding_mode: str = "replicate",
        activation_slope: float = 0.1,
        lambda_u: float = 0.91,
        lambda_j: float = 0.09,
        level_weights: Sequence[float] | None = None,
        use_temporal_propagation: bool = False,
        grad_clip_norm: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the LIMA estimator.

        Args:
            encoder_channels: Output channels per encoder level (Table I); its
                length sets the pyramid depth.
            decoder_channels: Output channels per decoder layer (Table III).
            decoder_dilations: Dilation per decoder layer (Table III).
            refine_levels: Number of coarsest pyramid levels to refine over.
                Defaults to the full depth (LIMA-6).
            search_range: Local-correlation search range R.
            padding_mode: Convolution padding scheme (``zeros`` = LIMA0,
                ``replicate`` = LIMAR, also ``reflect``/``circular``).
            activation_slope: Negative slope of the LeakyReLU activations.
            lambda_u: Weight of the data (displacement) loss term.
            lambda_j: Weight of the Jacobian smoothness penalty.
            level_weights: Per-level loss weights (coarse to fine). Defaults to
                the LIMA-1 Table 2 values (Manickathan et al., Exp. Fluids 64,
                161, 2023) when available, else a geometric schedule.
            use_temporal_propagation: Warm-start from the previous estimate.
            grad_clip_norm: Global-norm gradient clipping threshold.
            **kwargs: Forwarded to :class:`FlowFieldEstimator` (e.g.
                ``optimizer_config``, ``preprocessing_steps``,
                ``postprocessing_steps``).

        Raises:
            TypeError: If a parameter has the wrong type.
            ValueError: If a parameter value is out of range or inconsistent.
        """
        self.encoder_channels = _as_int_tuple(
            encoder_channels, "encoder_channels"
        )
        self.decoder_channels = _as_int_tuple(
            decoder_channels, "decoder_channels"
        )
        self.decoder_dilations = _as_int_tuple(
            decoder_dilations, "decoder_dilations"
        )
        if len(self.decoder_channels) != len(self.decoder_dilations):
            raise ValueError(
                "decoder_channels and decoder_dilations must have equal "
                f"length, got {len(self.decoder_channels)} and "
                f"{len(self.decoder_dilations)}."
            )

        self.levels = len(self.encoder_channels)
        self._mult = 2**self.levels

        if refine_levels is None:
            refine_levels = self.levels
        if not isinstance(refine_levels, int) or isinstance(
            refine_levels, bool
        ):
            raise TypeError(
                f"refine_levels must be an int, got {refine_levels!r}."
            )
        if not 1 <= refine_levels <= self.levels:
            raise ValueError(
                f"refine_levels must be in [1, {self.levels}], got "
                f"{refine_levels}."
            )
        self.refine_levels = refine_levels

        if not isinstance(search_range, int) or isinstance(search_range, bool):
            raise TypeError(
                f"search_range must be an int, got {search_range!r}."
            )
        if search_range <= 0:
            raise ValueError(
                f"search_range must be positive, got {search_range}."
            )
        self.search_range = search_range

        if padding_mode not in _PADDING_MODES:
            raise ValueError(
                f"padding_mode must be one of {_PADDING_MODES}, got "
                f"{padding_mode!r}."
            )
        self.padding_mode = padding_mode

        if activation_slope < 0:
            raise ValueError(
                "activation_slope must be non-negative, got "
                f"{activation_slope}."
            )
        self.activation_slope = float(activation_slope)

        if lambda_u < 0 or lambda_j < 0:
            raise ValueError(
                "lambda_u and lambda_j must be non-negative, got "
                f"{lambda_u} and {lambda_j}."
            )
        self.lambda_u = float(lambda_u)
        self.lambda_j = float(lambda_j)

        if level_weights is None:
            level_weights = _PAPER_LEVEL_WEIGHTS.get(refine_levels)
        if level_weights is None:
            # Geometric fallback: finest level weighted 0.32, halving per step.
            level_weights = tuple(
                0.32 * 0.5 ** (refine_levels - 1 - i)
                for i in range(refine_levels)
            )
        level_weights = tuple(float(w) for w in level_weights)
        if len(level_weights) != refine_levels:
            raise ValueError(
                f"level_weights must have length refine_levels="
                f"{refine_levels}, got {len(level_weights)}."
            )
        self.level_weights = level_weights

        if not isinstance(use_temporal_propagation, bool):
            raise TypeError(
                "use_temporal_propagation must be a bool, got "
                f"{use_temporal_propagation!r}."
            )
        self.use_temporal_propagation = use_temporal_propagation

        if grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive, got {grad_clip_norm}."
            )
        self.grad_clip_norm = float(grad_clip_norm)

        self.model = LimaModel(
            encoder_channels=self.encoder_channels,
            decoder_channels=self.decoder_channels,
            decoder_dilations=self.decoder_dilations,
            search_range=self.search_range,
            refine_levels=self.refine_levels,
            padding_mode=self.padding_mode,
            activation_slope=self.activation_slope,
        )

        super().__init__(**kwargs)

    def _pad_to_multiple(self, n: int) -> int:
        """Round ``n`` up to the nearest multiple of the pyramid downsampling.

        Args:
            n: A spatial dimension size.

        Returns:
            The smallest multiple of ``2**levels`` that is >= ``n``.
        """
        return ((n + self._mult - 1) // self._mult) * self._mult

    def _jacobian_l1(self, flow: jnp.ndarray) -> jnp.ndarray:
        """Mean over pixels of the L1 norm of the displacement Jacobian.

        For a field ``(u, v)`` this is the spatial mean of
        ``|du/dx|+|dv/dx|+|du/dy|+|dv/dy|`` (an anisotropic total-variation
        smoothness penalty).

        Args:
            flow: Displacement field of shape (B, H, W, 2).

        Returns:
            Scalar Jacobian penalty (0 on degenerate singleton levels).
        """
        penalty: jnp.ndarray | float = 0.0
        if flow.shape[2] > 1:
            d_dx = flow[:, :, 1:, :] - flow[:, :, :-1, :]
            penalty = penalty + jnp.mean(jnp.sum(jnp.abs(d_dx), axis=-1))
        if flow.shape[1] > 1:
            d_dy = flow[:, 1:, :, :] - flow[:, :-1, :, :]
            penalty = penalty + jnp.mean(jnp.sum(jnp.abs(d_dy), axis=-1))
        return jnp.asarray(penalty)

    def create_trainable_state(
        self,
        dummy_input: jnp.ndarray,
        key: PRNGKey,
    ) -> NNEstimatorTrainableState:
        """Initialize model parameters and the optimizer state.

        Args:
            dummy_input: Dummy input of shape (B, H, W) used only for shapes.
            key: JAX PRNG key for parameter initialization.

        Returns:
            The initial trainable state of the estimator.

        Raises:
            ValueError: If ``dummy_input`` is not rank 3.
        """
        if dummy_input.ndim != 3:
            raise ValueError(
                "Dummy input must have 3 dimensions (B, H, W), got "
                f"{dummy_input.ndim}."
            )
        _, height, width = dummy_input.shape
        height_p = self._pad_to_multiple(height)
        width_p = self._pad_to_multiple(width)
        dummy = jnp.zeros((1, height_p, width_p, 2), dtype=jnp.float32)
        params = self.model.init(key, dummy)["params"]
        return NNEstimatorTrainableState.from_config(
            apply_fn=self.model.apply,
            params=params,
            optimizer_config=self.optimizer_config,
        )

    def _estimate(
        self,
        image: jnp.ndarray,
        state: History,
        trainable_state: NNEstimatorTrainableState,
        extras: dict,
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Estimate the flow between the two most recent frames.

        Args:
            image: Current batch of frames, shape (B, H, W).
            state: History containing previous images (and estimates).
            trainable_state: Current model parameters and optimizer state.
            extras: Additional data from history fields (unused here).

        Returns:
            A tuple of the full-resolution flow field (B, H, W, 2), an empty
            extras dict, and an empty metrics dict.
        """
        B, H, W = image.shape
        prev = state["images"][:, -1, ...]
        images = jnp.stack([prev, image], axis=-1)

        height_p = self._pad_to_multiple(H)
        width_p = self._pad_to_multiple(W)
        pad = ((0, 0), (0, height_p - H), (0, width_p - W), (0, 0))
        # Honor the estimator's padding_mode for the size-to-multiple padding
        # so the border behaviour matches the convolution padding (e.g. a
        # `zeros`/`reflect`/`circular` model no longer silently edge-pads here).
        pad_mode = _PAD_MODES[self.padding_mode]
        images = jnp.pad(images, pad, mode=pad_mode)

        flow_init = None
        if self.use_temporal_propagation:
            flow_init = jnp.pad(
                state["estimates"][:, -1, ...], pad, mode=pad_mode
            )

        per_level = cast(
            list[jnp.ndarray],
            self.model.apply(
                {"params": trainable_state.params}, images, flow_init
            ),
        )
        finest = per_level[-1]
        flow = jax.image.resize(
            finest, (B, height_p, width_p, 2), method="bilinear"
        )
        return flow[:, :H, :W, :], {}, {}

    def create_train_step(self) -> SupervisedTrainStep:
        """Build the supervised training step (multi-level Jacobian L1 loss).

        Returns:
            A function mapping ``(trainable_state, experience)`` to
            ``(loss, new_trainable_state, metrics)``.
        """
        mult = self._mult

        def train_step(
            trainable_state: NNEstimatorTrainableState,
            experience: SupervisedExperience,
        ) -> tuple[jnp.ndarray, NNEstimatorTrainableState, dict]:
            """Run one optimizer step.

            Args:
                trainable_state: Current model parameters and optimizer state.
                experience: Supervised experience (state, image pair, GT flow).

            Returns:
                The loss, the updated trainable state, and a metrics dict.

            Raises:
                ValueError: If the images are not a multiple of ``2**levels``.
            """
            state = experience.state
            images1, images2 = experience.obs
            ground_truth = experience.ground_truth
            B, H, W = images1.shape
            if H % mult or W % mult:
                raise ValueError(
                    "LIMA training images must have height and width that are "
                    f"multiples of {mult}; got {(H, W)}."
                )

            images = jnp.stack([images1, images2], axis=-1)
            flow_init = (
                state["estimates"][:, -1, ...]
                if self.use_temporal_propagation
                else None
            )

            def loss_fn(params: Any) -> tuple[jnp.ndarray, dict]:
                """Multi-level Jacobian-penalised L1 loss.

                Args:
                    params: Model parameters.

                Returns:
                    The scalar loss and an auxiliary metrics dict.
                """
                per_level = cast(
                    list[jnp.ndarray],
                    self.model.apply({"params": params}, images, flow_init),
                )
                total: jnp.ndarray | float = 0.0
                data_acc: jnp.ndarray | float = 0.0
                jac_acc: jnp.ndarray | float = 0.0
                for weight, flow_l in zip(
                    self.level_weights, per_level, strict=True
                ):
                    _, level_h, level_w, _ = flow_l.shape
                    gt_l = jax.image.resize(
                        ground_truth,
                        (B, level_h, level_w, 2),
                        method="bilinear",
                    )
                    data = jnp.mean(jnp.sum(jnp.abs(flow_l - gt_l), axis=-1))
                    jac = self._jacobian_l1(flow_l)
                    total = total + weight * (
                        self.lambda_u * data + self.lambda_j * jac
                    )
                    data_acc = data_acc + data
                    jac_acc = jac_acc + jac
                return jnp.asarray(total), {
                    "data_loss": jnp.asarray(data_acc),
                    "jacobian_loss": jnp.asarray(jac_acc),
                }

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                trainable_state.params
            )

            grad_norm = optax.global_norm(grads)
            scale = jnp.minimum(1.0, self.grad_clip_norm / (grad_norm + 1e-8))
            clipped = jax.tree_util.tree_map(lambda g: g * scale, grads)

            updates, new_opt_state = trainable_state.tx.update(
                clipped, trainable_state.opt_state, trainable_state.params
            )
            new_params = optax.apply_updates(trainable_state.params, updates)
            trainable_state = trainable_state.replace(
                params=new_params,
                opt_state=new_opt_state,
                step=trainable_state.step + 1,
            )

            metrics = {
                "loss": loss,
                "grad_norm": grad_norm,
                "clip_scale": scale,
                **aux,
            }
            return loss, trainable_state, metrics

        return train_step
