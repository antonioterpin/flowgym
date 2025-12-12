"""RAFT_torch class."""

import jax
import jax.numpy as jnp
import numpy as np
import torch


import torch.nn.functional as F

import scipy.signal

from flowgym.nn.raft_torch_nn.flowNetsRAFT import RAFT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from flowgym.flow.base import FlowFieldEstimator
from goggles.history.types import History
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RaftTorchEstimator(FlowFieldEstimator):
    """RAFT flow field estimator using two-frame history."""

    def __init__(self, **kwargs):
        """Initialize the RAFT estimator in pytorch."""
        # Validate RAFT specific parameters
        self.raft = RAFT().to(device)
        super().__init__(**kwargs)

    def _estimate(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state: None,
        extras: None,
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the flow field between the two most recent frames.

        Args:
            images: Current batch of frames, shape (B, H, W).
            state: Contains history_images of shape (B, T, H, W).
            trainable_state: Unused parameter.
            extras: Unused parameter.

        Returns:
            jnp.ndarray: Flow field of shape (B, H, W, 2) as float32.
            dict: placeholder for additional output.
            dict: placeholder for additional output.
        """
        self.raft.eval()
        # Convert to pytorch for RAFT
        prev = torch.utils.dlpack.from_dlpack(
            state["images"][:, 0, ...]
        )  # (B, H, W)
        curr = torch.utils.dlpack.from_dlpack(images)  # (B, H, W)
        prev = prev.unsqueeze(1).to(device) / 256  # (B, 1, H, W)
        curr = curr.unsqueeze(1).to(device) / 256  # (B, 1, H, W)
        images = torch.cat([prev, curr], dim=1)  # (B, 2, H, W)
        B, C, H, W = images.shape

        class Args:
            pass

        args = Args()
        args.amp = True
        args.iters = 12  # typical RAFT number of iterations
        args.offset = 32  # patch size
        args.shift = 8  # patch shift
        args.split_size = 50  # process 50 patches at a time

        # Example: zero ground truth and args config
        predicted_flows = torch.zeros_like(images)  # (B, 2, H, W)
        flows = torch.zeros_like(
            images
        )  # (B, 2, H, W) # zero ground truth for RAFT input

        # compute b-spline window
        WINDOW_SPLINE_2D = torch.from_numpy(
            np.squeeze(self._window_2D(window_size=args.offset, power=2))
        )

        # create patches of image and flow
        patches = (
            images.unfold(3, args.offset, args.shift)
            .unfold(2, args.offset, args.shift)
            .permute(0, 2, 3, 1, 5, 4)
        )
        patches = patches.reshape((-1, 2, args.offset, args.offset))
        flow_patches = (
            flows.unfold(3, args.offset, args.shift)
            .unfold(2, args.offset, args.shift)
            .permute(0, 2, 3, 1, 5, 4)
        )
        flow_patches = flow_patches.reshape((-1, 2, args.offset, args.offset))
        splitted_patches = torch.split(patches, args.split_size, dim=0)
        splitted_flow_patches = torch.split(flow_patches, args.split_size, dim=0)

        # Forward pass
        with autocast(enabled=args.amp):
            # unfold flow
            predicted_flow_patches = (
                predicted_flows.unfold(3, args.offset, args.shift)
                .unfold(2, args.offset, args.shift)
                .permute(0, 2, 3, 1, 5, 4)
            )
            predicted_flow_patches = predicted_flow_patches.reshape(
                (-1, 2, args.offset, args.offset)
            )
            # split flow patch tensor in batches for evaluation
            splitted_predicted_flow_patches = torch.split(
                predicted_flow_patches, args.split_size, dim=0
            )
            splitted_flow_output_patches = []

            with torch.no_grad():
                for split in range(len(splitted_patches)):
                    pred_flows = self.raft(
                        splitted_patches[split],
                        splitted_flow_patches[split],
                        flow_init=splitted_predicted_flow_patches[split],
                        args=args,
                    )
                    all_flow_iters = pred_flows[0]
                    splitted_flow_output_patches.append(all_flow_iters[-1])

            NUM_Yvectors, NUM_Xvectors = (
                int(H / args.shift - (args.offset / args.shift - 1)),
                int(W / args.shift - (args.offset / args.shift - 1)),
            )

            # fold and weight predicted flow patches
            flow_output_patches = torch.cat(splitted_flow_output_patches, dim=0)
            flow_output_patches = flow_output_patches * WINDOW_SPLINE_2D.cuda()
            flow_output_patches = flow_output_patches.reshape(
                (B, NUM_Yvectors, NUM_Xvectors, 2, args.offset, args.offset)
            ).permute(0, 3, 1, 2, 4, 5)
            flow_output_patches = flow_output_patches.contiguous().view(
                B, C, -1, args.offset * args.offset
            )
            flow_output_patches = flow_output_patches.permute(0, 1, 3, 2)
            flow_output_patches = flow_output_patches.contiguous().view(
                B, C * args.offset * args.offset, -1
            )
            predicted_flows_iter = F.fold(
                flow_output_patches,
                output_size=(H, W),
                kernel_size=args.offset,
                stride=args.shift,
            )

            folding_mask = torch.ones_like(images)  # (B, C, H, W)

            # compute folding mask
            mask_patches = folding_mask.unfold(3, args.offset, args.shift).unfold(
                2, args.offset, args.shift
            )
            mask_patches = mask_patches.contiguous().view(
                B, C, -1, args.offset, args.offset
            )
            mask_patches = mask_patches * WINDOW_SPLINE_2D.cuda()
            mask_patches = mask_patches.view(B, C, -1, args.offset * args.offset)
            mask_patches = mask_patches.permute(0, 1, 3, 2)
            mask_patches = mask_patches.contiguous().view(
                B, C * args.offset * args.offset, -1
            )
            folding_mask = F.fold(
                mask_patches,
                output_size=(H, W),
                kernel_size=args.offset,
                stride=args.shift,
            )

            predicted_flows += predicted_flows_iter / folding_mask

        # PyTorch tensor device
        torch_device = predicted_flows.device.type   # "cuda" or "cpu"

        # Choose matching JAX device if available
        if torch_device == "cuda" and any(d.platform == "gpu" for d in jax.devices()):
            target_device = next(d for d in jax.devices() if d.platform == "gpu")
        else:
            target_device = next(d for d in jax.devices() if d.platform == "cpu")

        # Convert: torch to numpy to jax
        np_flow = predicted_flows.permute(0, 2, 3, 1).detach().cpu().numpy()

        # Place onto the selected device
        with jax.default_device(target_device):
            jax_flow = jnp.asarray(np_flow)

        return jax_flow, {}, {}

    # spline windowing
    cached_2d_windows = dict()

    def _window_2D(self, window_size, power=2):
        """Make a 1D window function, then infer and return a 2D window function.
        Done with an augmentation, and self multiplication with its transpose.
        Could be generalized to more dimensions.
        """
        # Memoization
        global cached_2d_windows
        key = f"{window_size}_{power}"
        if key in self.cached_2d_windows:
            wind = self.cached_2d_windows[key]
        else:
            wind = self._spline_window(window_size, power)
            wind = np.expand_dims(np.expand_dims(wind, -1), -1)
            wind = wind * wind.transpose(1, 0, 2)
            self.cached_2d_windows[key] = wind
        return wind

    ### spline windowing
    def _spline_window(self, window_size, power=2):
        """Squared spline window function."""
        intersection = int(window_size / 4)
        wind_outer = (abs(2 * (scipy.signal.windows.triang(window_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = (
            1 - (abs(2 * (scipy.signal.windows.triang(window_size) - 1)) ** power) / 2
        )
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.average(wind)
        return wind


