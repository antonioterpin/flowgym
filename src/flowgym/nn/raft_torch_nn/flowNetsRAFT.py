"""Copyright (c) 2020-2021, Christian Lagemann

Portions of this code copyright 2020, princeton-vl
In the framework of:
Teed, Zachary, and Jia Deng. "Raft: Recurrent all-pairs field transforms for optical flow." European Conference on Computer Vision. Springer, Cham, 2020.
URL: https://github.com/princeton-vl/RAFT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowgym.nn.raft_torch_nn.submodules_RAFT_extractor import BasicEncoder
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


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img.float(), grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.half()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


def sequence_loss(flow_preds, flow_gt):
    """Loss function defined over sequence of flow predictions"""
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = 0.8 ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


class RAFT(nn.Module):
    """RAFT"""

    def __init__(self):
        super().__init__()

        self.hidden_dim = 128
        self.context_dim = 128
        self.corr_levels = 4
        self.corr_radius = 4

        self.fnet = BasicEncoder(output_dim=256, norm_fn="instance", dropout=0.0)
        self.cnet = BasicEncoder(
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
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

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
            net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(img1)

        flow_predictions = []
        for itr in range(args.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            if itr == 0 and flow_init is not None:
                flow = flow_init
            else:
                flow = coords1 - coords0
            with autocast(enabled=args.amp):
                net, _, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            flow = coords1 - coords0
            flow_predictions.append(flow)

        loss = sequence_loss(flow_predictions, flowl0)

        return flow_predictions, loss
