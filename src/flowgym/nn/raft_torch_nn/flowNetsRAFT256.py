"""Copyright (c) 2020-2021, Christian Lagemann.

Portions of this code copyright 2020, princeton-vl
In the framework of:
Teed, Zachary, and Jia Deng. "Raft: Recurrent all-pairs field transforms for optical flow." European Conference on Computer Vision. Springer, Cham, 2020.
URL: https://github.com/princeton-vl/RAFT

RAFT256-PIV variant from the reference capsule
https://codeocean.com/capsule/7226151/tree/v1 (files ``flowNetsRAFT256.py``,
``submodules_RAFT_extractor256.py`` and ``submodules_RAFT_GRU256.py``).

Compared with the RAFT32-PIV variant in ``flowNetsRAFT.py`` the only
architectural differences are:

* the feature/context encoder downsamples the input by 1/8 (stride-2 in each
  of the three residual stages) instead of staying at full resolution, and
* the final flow is recovered with the learned *convex* upsampling head
  (``upsample_flow``), the published RAFT256-PIV default.

Every other building block (``ResidualBlock``, ``BasicUpdateBlock``,
``CorrBlock`` and the coordinate/sampling helpers) is byte-for-byte identical
to the RAFT32-PIV modules, so they are imported and reused here. Only the
strided ``BasicEncoder256`` is new.
"""

import torch
import torch.nn.functional as F
from torch import nn

from flowgym.nn.raft_torch_nn.flowNetsRAFT import (
    CorrBlock,
    coords_grid,
    sequence_loss,
)
from flowgym.nn.raft_torch_nn.submodules_RAFT_extractor import ResidualBlock
from flowgym.nn.raft_torch_nn.submodules_RAFT_GRU import BasicUpdateBlock

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


class BasicEncoder256(nn.Module):
    """Feature encoder that downsamples the input by 1/8.

    Identical to the RAFT32-PIV ``BasicEncoder`` except that each of the three
    residual stages uses stride 2, so a 256x256 patch is encoded at 32x32.
    """

    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0):
        super().__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=2)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(
                m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)
            ):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class RAFT256(nn.Module):
    """RAFT256-PIV.

    Uses the 1/8-resolution ``BasicEncoder256`` and recovers the full
    resolution flow with the learned convex upsampling head.
    """

    def __init__(self):
        super().__init__()

        self.hidden_dim = 128
        self.context_dim = 128
        self.corr_levels = 4
        self.corr_radius = 4

        self.fnet = BasicEncoder256(
            output_dim=256, norm_fn="instance", dropout=0.0
        )
        self.cnet = BasicEncoder256(
            output_dim=self.hidden_dim + self.context_dim,
            norm_fn="instance",
            dropout=0.0,
        )
        self.update_block = BasicUpdateBlock(
            hidden_dim=self.hidden_dim,
            corr_levels=self.corr_levels,
            corr_radius=self.corr_radius,
        )

    def initialize_flow(self, img):
        """Flow is the difference of two 1/8-resolution coordinate grids."""
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow [H/8, W/8, 2] -> [H, W, 2] via convex combination."""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, input, flowl0, args, flow_init=None, upsample=True):
        img1 = torch.unsqueeze(input[:, 0, :, :], dim=1)
        img2 = torch.unsqueeze(input[:, 1, :, :], dim=1)

        with autocast(enabled=args.amp):
            fmap1, fmap2 = self.fnet([img1, img2])

        corr_fn = CorrBlock(
            fmap1, fmap2, radius=self.corr_radius, num_levels=self.corr_levels
        )

        with autocast(enabled=args.amp):
            cnet = self.cnet(img1)
            net, inp = torch.split(
                cnet, [self.hidden_dim, self.context_dim], dim=1
            )
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(img1)

        if flow_init is not None:
            flow_init = F.interpolate(
                flow_init,
                [coords1.size()[2], coords1.size()[3]],
                mode="bilinear",
            )
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(args.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=args.amp):
                net, up_mask, delta_flow = self.update_block(
                    net, inp, corr, flow
                )

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # convex upsampling (published RAFT256-PIV default)
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)

        loss = sequence_loss(flow_predictions, flowl0)

        return flow_predictions, loss
