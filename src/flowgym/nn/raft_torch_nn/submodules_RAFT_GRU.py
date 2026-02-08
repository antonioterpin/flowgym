"""Copyright (c) 2020-2021, Christian Lagemann.

Portions of this code copyright 2020, princeton-vl
In the framework of:
Teed, Zachary, and Jia Deng. "Raft: Recurrent all-pairs field transforms for optical flow." European Conference on Computer Vision. Springer, Cham, 2020.
URL: https://github.com/princeton-vl/RAFT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    """Flow head for RAFT optical flow estimation."""

    def __init__(self, input_dim=128, hidden_dim=256):
        """Initialize the flow head with input dimension and hidden dimension."""
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass through the flow head."""
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    """Convolutional GRU for RAFT optical flow estimation."""

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        """Initialize the convolutional GRU with hidden dimension and input dimension."""
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        """Forward pass through the convolutional GRU."""
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    """Separable convolutional GRU for RAFT optical flow estimation."""

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        """Initialize the separable convolutional GRU with hidden dimension and input dimension."""
        super().__init__()
        self.convz1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convr1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convq1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )

        self.convz2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convr2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convq2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )

    def forward(self, h, x):
        """Forward pass through the separable convolutional GRU."""
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class SmallMotionEncoder(nn.Module):
    """Small motion encoder for RAFT optical flow estimation."""

    def __init__(self, args):
        """Initialize the small motion encoder with arguments."""
        super().__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        """Forward pass through the small motion encoder."""
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicMotionEncoder(nn.Module):
    """Basic motion encoder for RAFT optical flow estimation."""

    def __init__(self, corr_levels, corr_radius):
        """Initialize the basic motion encoder with correlation levels and radius."""
        super().__init__()
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        cor_planes = self.corr_levels * (2 * self.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        """Forward pass through the basic motion encoder."""
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):
    """Small update block for RAFT optical flow estimation."""

    def __init__(self, args, hidden_dim=96):
        """Initialize the small update block with arguments and hidden dimension."""
        super().__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + 64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        """Forward pass through the small update block."""
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow


class BasicUpdateBlock(nn.Module):
    """Basic update block for RAFT optical flow estimation."""

    def __init__(self, corr_levels=4, corr_radius=4, hidden_dim=128, input_dim=128):
        """Initialize the basic update block with correlation levels, radius, hidden dimension, and input dimension."""
        super().__init__()
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.encoder = BasicMotionEncoder(self.corr_levels, self.corr_radius)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, upsample=True):
        """Forward pass through the basic update block."""
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balance gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow
