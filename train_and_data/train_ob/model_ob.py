#!/usr/bin/env python3
"""Tiny single-image segmenter for the unusable (optical-black / shaded) border.

Unlike the pair model, this sees ONE frame (no pose, no A/B pairing): the OB /
shading region is a per-frame spatial property.  The network is a small
stride-4 encoder with a light decoder, outputting a per-cell mask on the same
64x64 lattice as the dataset GT.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OBNet(nn.Module):
    """(B,1,256,256) -> (B,1,64,64) unusable-region logits."""

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        b = int(base)
        self.enc = nn.Sequential(
            ConvBlock(1, b, stride=2),        # 128
            ConvBlock(b, b * 2, stride=2),    # 64
            ConvBlock(b * 2, b * 4, stride=2),  # 32
            ConvBlock(b * 4, b * 4, stride=1),  # 32
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(b * 4, b * 2, 4, stride=2, padding=1),  # 64
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(b * 2, 1, 1),
        )
        nn.init.constant_(self.dec[-1].bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))
