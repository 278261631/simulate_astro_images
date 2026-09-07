#!/usr/bin/env python3
"""Compact CNN that predicts the relative pose (dx, dy, droll) between two
consecutive exposures A and B of the same sky field.

Design notes
------------
A and B are stacked as 2 input channels and pushed through a convolutional
body with *no* global average pooling: the network must read the spatial
"phase" of the star field (a pure translation/rotation changes every feature
map's layout, and global pooling would destroy that information). A moderate
adaptive-pooled feature grid is flattened and regressed through a small MLP
head (the same family of architecture used by deep homography estimators).

Outputs (raw head values):
    dx, dy   - predicted pixel offset (A centre w.r.t. B), floats
    c, s     - cos/sin of the roll difference (normalised in the decoder)
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, stride=stride),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairRegNet(nn.Module):
    def __init__(self, in_ch: int = 2, feat: int = 32, drop: float = 0.15,
                 pool_grid: int = 16) -> None:
        super().__init__()
        ch = in_ch
        body = []
        for i in range(3):
            cout = feat * (1 << i)
            body.append(ConvBlock(ch, cout, stride=2))
            ch = cout
        self.body = nn.Sequential(*body)  # final stride = 8
        self.reduce = nn.AdaptiveAvgPool2d((pool_grid, pool_grid))
        f = ch * pool_grid * pool_grid
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(f, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(512, 4),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """pair: (B, 2, H, W) float; returns (B, 4): dx, dy, cos_roll, sin_roll."""
        f = self.reduce(self.body(pair))
        return self.head(f)


def decode(y_raw: torch.Tensor) -> torch.Tensor:
    """(B, 4) raw -> (B, 3): dx, dy, droll_deg (roll via atan2 of c/s)."""
    dx = y_raw[:, 0]
    dy = y_raw[:, 1]
    c = y_raw[:, 2]
    s = y_raw[:, 3]
    roll = torch.atan2(s, c) * 180.0 / math.pi
    return torch.stack([dx, dy, roll], dim=1)


def encode_target(labels: np.ndarray) -> torch.Tensor:
    """(N, 3) float [dx, dy, droll_deg] -> (N, 4) [dx, dy, cos, sin]."""
    lab = torch.as_tensor(labels, dtype=torch.float32)
    r = torch.deg2rad(lab[:, 2])
    return torch.stack(
        [lab[:, 0], lab[:, 1], torch.cos(r), torch.sin(r)], dim=1
    )


def split_target(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a (N, 4) encoded target tensor into dx, dy and (cos, sin)."""
    return y[:, 0], y[:, 1], y[:, 2:4]


# ---------------------------------------------------------------------------
# Preprocessing shared by training and inference
# ---------------------------------------------------------------------------


def preprocess(a_u8: np.ndarray, b_u8: np.ndarray) -> torch.Tensor:
    """uint8 (H, W) frames -> (2, H, W) float in ~[0, 1].

    Per-frame percentile stretch normalises the background and makes the model
    insensitive to absolute brightness / gain, while stars (bright top end)
    keep their relative ranking.
    """
    outs = []
    for im in (a_u8, b_u8):
        x = im.astype(np.float32)
        lo, hi = np.percentile(x, (1.0, 99.5))
        if hi - lo > 1e-6:
            x = (x - lo) / (hi - lo)
        outs.append(torch.as_tensor(np.clip(x, 0.0, 1.0)))
    return torch.stack(outs)  # (2, H, W)
