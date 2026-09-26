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

#: stride (input px per heatmap cell) of the detection head output grid
DET_STRIDE = 4


def _warp_b_to_a(pair: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor,
                 roll: torch.Tensor) -> torch.Tensor:
    """Differentiably warp channel 1 (B) onto channel 0 (A) using pose.

    Content geometry (see validate_model_gui.warp_b_onto_a): a point at B
    pixel q appears in A at p = R(+roll)·q + (-dx, -dy), so to produce the
    aligned image at A-pixel p we sample B at q = R(-roll)·(p + (dx, dy)).

    ``dx/dy`` are in pixels, ``roll`` in radians; returns (B, 1, H, W).
    """
    b, _, h, w = pair.shape
    dev = pair.device
    roll = roll.detach()
    ang = -roll
    co = torch.cos(ang)
    si = torch.sin(ang)
    # normalised offset: norm px = 2 * px / (size - 1)
    tx = 2.0 * dx.detach() / float(w - 1)
    ty = 2.0 * dy.detach() / float(h - 1)
    a = co
    b_ = -si
    c = si
    d = co
    # q = R(-roll) @ (p + d_off)
    bx = a * tx + b_ * ty
    by = c * tx + d * ty
    theta = torch.stack([torch.stack([a, b_, bx], dim=1),
                         torch.stack([c, d, by], dim=1)], dim=1)  # (B,2,3)
    grid = nn.functional.affine_grid(theta, (b, 1, h, w),
                                     align_corners=True)
    src = pair[:, 1:2]
    warped = nn.functional.grid_sample(src, grid, align_corners=True,
                                       mode="bilinear",
                                       padding_mode="border")
    return warped


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
                 pool_grid: int = 16, n_cls: int = 3, n_mask: int = 0,
                 n_size: int = 0) -> None:
        """Pose regression + transient heatmap detection + OB mask segmentation.

        ``n_cls > 0`` enables the transient heatmap head (per-class Gaussian
        peaks).  ``n_size > 0`` adds a parallel per-detection size regression
        (apparent FWHM in pixels, one channel per class).  ``n_mask > 0``
        enables a full-resolution segmenter that marks regions of interest /
        to avoid; the channel layout is dataset-defined (e.g.
        [A-unusable, B-unusable, B-satellite]).  All default off, keeping
        pose-only / detection-only checkpoints fully compatible.
        """
        super().__init__()
        self.n_cls = int(n_cls)
        self.n_mask = int(n_mask)
        self.n_size = int(n_size)
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
        self.det = None
        self.det_size = None
        self.det_trunk = None
        self.ob = None
        self.res_enc = None
        if self.n_cls > 0 or self.n_mask > 0:
            # Residual encoder: turns (A, aligned-B, |A - aligned-B|) into a
            # stride-8 feature map. Ordinary stars cancel in the residual so
            # they do not become detection candidates; the dark border bands
            # survive in both frames' channels for the OB segmenter.
            self.res_enc = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                ConvBlock(32, 64, stride=2),
                ConvBlock(64, feat * 2, stride=2),
                ConvBlock(feat * 2, feat * 2, stride=2),
            )
            din = ch + feat * 2
        if self.n_cls > 0:
            # Shared detection decoder on the concatenated stride-8 features,
            # upsampled to stride-4.  Two 1x1 heads hang off its 64-ch output:
            # per-class heatmap logits and an optional per-class size (FWHM).
            self.det_trunk = nn.Sequential(
                nn.Conv2d(din, 96, 3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, 3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.det = nn.Conv2d(64, self.n_cls, 1)
            # start from a quiet background (sigmoid ~ 0.007) so the focal
            # negatives are cheap and positives have logit headroom; avoids
            # collapsing the whole map to the sigmoid's steep region.
            nn.init.constant_(self.det.bias, -5.0)
            if self.n_size > 0:
                self.det_size = nn.Conv2d(64, self.n_size, 1)
                # regress log(FWHM): start at ~3 px (e**1.1)
                nn.init.constant_(self.det_size.bias, math.log(3.0))
        if self.n_mask > 0:
            # Full-resolution segmenter: three transposed-conv upsamples take
            # the stride-8 feature back to stride-1, then a 1x1 classifier.
            self.ob = nn.Sequential(
                nn.Conv2d(din, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                ConvBlock(64, 64),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, self.n_mask, 1),
            )
            for m in self.ob.modules():
                if isinstance(m, nn.Conv2d) and m.out_channels == self.n_mask:
                    nn.init.constant_(m.bias, -1.0)

    def forward(self, pair: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor | None,
                           torch.Tensor | None, torch.Tensor | None]:
        """pair: (B, 2, H, W) float -> (pose, det, ob, size).

        pose : (B, 4)                 global pose regression
        det  : (B, n_cls, H/4, W/4)   per-class transient heatmap logits
        ob   : (B, n_mask, H, W)      per-frame region-mask logits
        size : (B, n_size, H/4, W/4)  per-class log(FWHM, px) regression

        Det/OB/size share the stride-8 fused features; B is warped onto A with
        the *predicted, detached* pose so their gradients never damage the pose
        head.  Any head can be disabled (returns None).
        """
        f = self.body(pair)
        pose = self.head(self.reduce(f))
        det = None
        ob = None
        size = None
        if self.res_enc is not None:
            dx = pose[:, 0]
            dy = pose[:, 1]
            roll = torch.atan2(pose[:, 3], pose[:, 2])
            wb = _warp_b_to_a(pair, dx, dy, roll)
            res = (pair[:, 0:1] - wb).abs()
            rfeat = self.res_enc(torch.cat([pair[:, 0:1], wb, res], dim=1))
            feat = torch.cat([f, rfeat], dim=1)
            if self.det is not None:
                f4 = self.det_trunk(feat)
                det = self.det(f4)
                if self.det_size is not None:
                    size = self.det_size(f4)
            if self.ob is not None:
                ob = self.ob(feat)
                if ob.shape[-2:] != pair.shape[-2:]:
                    ob = nn.functional.interpolate(
                        ob, size=pair.shape[-2:], mode="bilinear",
                        align_corners=False)
        return pose, det, ob, size


def decode(y_raw: torch.Tensor) -> torch.Tensor:
    """(B, 4) raw -> (B, 3): dx, dy, droll_deg (roll via atan2 of c/s)."""
    dx = y_raw[:, 0]
    dy = y_raw[:, 1]
    c = y_raw[:, 2]
    s = y_raw[:, 3]
    roll = torch.atan2(s, c) * 180.0 / math.pi
    return torch.stack([dx, dy, roll], dim=1)


def _infer_head_out(sd: dict, prefix: str) -> int:
    """Infer a head's output channel count from a state dict.

    The output layer (n_cls / n_mask) is far smaller than any intermediate
    channel, so the minimum over all 4-D weight dimensions is the head width;
    this is agnostic to Conv2d (out = shape[0]) vs ConvTranspose2d (out =
    shape[1]).
    """
    cands: list[int] = []
    for k, v in sd.items():
        if k.startswith(prefix) and k.endswith("weight") and v.ndim == 4:
            cands.extend((int(v.shape[0]), int(v.shape[1])))
    return min(cands) if cands else 0


def build_model_from_state(sd: dict, n_cls: int | None = None,
                           n_mask: int | None = None,
                           n_size: int | None = None) -> PairRegNet:
    """Instantiate PairRegNet matching an old/new state dict.

    Detection/size/OB weights exist only in checkpoints trained with those
    heads; pose-only checkpoints get all zeros.  When the heads are present
    their sizes are inferred from the output layers, so 1/3/4-class,
    1/2/3-channel-mask and 0/1-size checkpoints all load correctly.
    """
    inferred_cls = _infer_head_out(sd, "det.")
    inferred_mask = _infer_head_out(sd, "ob.")
    inferred_size = _infer_head_out(sd, "det_size.")
    net = PairRegNet(
        n_cls=n_cls if n_cls is not None else inferred_cls,
        n_mask=n_mask if n_mask is not None else inferred_mask,
        n_size=n_size if n_size is not None else inferred_size,
    )
    net.load_state_dict(sd)
    net.eval()
    return net


def cluster_points(points, radius: float = 24.0) -> list[list[int]]:
    """Greedy single-linkage clustering of 2-D points; returns index groups.

    Used to treat a satellite trail's centreline points as one instance for
    evaluation (a hit on any point counts as detecting the whole trail).
    """
    n = len(points)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            if dx * dx + dy * dy <= radius * radius:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def heat_to_peaks(prob: torch.Tensor, thresh: float = 0.35,
                  stride: int = DET_STRIDE) -> list[list[tuple[int, float, float, float]]]:
    """Per-batch peak extraction from (B, C, H, W) class probabilities.

    Non-maximum suppression on a 3x3 neighbourhood; returns per sample a list
    of (cls, x_px, y_px, score) where x_px/y_px are model-input-grid pixel
    coordinates (cell centre * stride). Coordinates are floats in the input
    image grid.
    """
    if prob is None or prob.dim() != 4:
        return [[] for _ in range(prob.shape[0])] if prob is not None else []
    b, c, h, w = prob.shape
    out: list[list[tuple[int, float, float, float]]] = []
    for bi in range(b):
        peaks: list[tuple[int, float, float, float]] = []
        for ci in range(c):
            m = prob[bi, ci]
            maxp = nn.functional.max_pool2d(m.unsqueeze(0).unsqueeze(0), 3,
                                            stride=1, padding=1)[0, 0]
            is_peak = (m >= maxp) & (m >= thresh)
            idxs = is_peak.nonzero(as_tuple=False)
            for idx in idxs:
                yy, xx = int(idx[0]), int(idx[1])
                s = float(m[yy, xx])
                if not any(abs(pp[1] - (xx + 0.5) * stride) < stride
                           and abs(pp[2] - (yy + 0.5) * stride) < stride
                           for pp in peaks):
                    peaks.append((ci, (xx + 0.5) * stride, (yy + 0.5) * stride, s))
        peaks.sort(key=lambda p: -p[3])
        out.append(peaks)
    return out


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
