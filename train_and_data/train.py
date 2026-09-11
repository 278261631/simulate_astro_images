#!/usr/bin/env python3
"""Train PairRegNet on the generated (A, B) -> (dx, dy, droll) dataset.

The train split is read *streaming* (memmap + per-batch fetch), so dataset
sizes of hundreds of thousands of pairs fit in RAM; per-frame normalisation
constants are computed once and cached as ``<data>/train_norm_<n>.npz``.

Usage:
    python train.py --data data --epochs 15 --batch 32

Writes ``models/best.pt`` (state dict) + ``models/checkpoint_best.pt`` (full
dict incl. metrics) and prints train/val/test statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (  # noqa: E402
    DET_STRIDE,
    PairRegNet,
    cluster_points,
    decode,
    encode_target,
    heat_to_peaks,
    preprocess,
)

#: class index of the satellite trail (appear=0, dim=1, satellite=2)
SAT_CLS = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Dataset loading (streaming for the big train split)
# ---------------------------------------------------------------------------
#
# A 500k-pair train set is ~35 GB of uint8 frames: it must never be fully
# materialised in RAM. Frames are opened as *memmaps* (only pages that are
# actually read consume RAM) and each training batch gathers + percentile-
# stretches just its own indices on the fly. The per-frame (lo, span)
# normalisation constants are tiny ((N,) float32 each) and are computed once,
# then cached to ``<data>/<split>_norm_<n>.npz`` so the expensive pass over
# the frames only happens on the first run.
#
# val/test are small and keep the original load-everything path.


def load_split(data_dir: Path, split: str):
    a = np.load(data_dir / f"{split}_a.npy")
    b = np.load(data_dir / f"{split}_b.npy")
    lab = np.load(data_dir / f"{split}_labels.npy")
    return a, b, lab


def open_split_mmap(data_dir: Path, split: str):
    """Open ``split``'s frames as memmaps + load its (small) labels fully."""
    a = np.load(data_dir / f"{split}_a.npy", mmap_mode="r")
    b = np.load(data_dir / f"{split}_b.npy", mmap_mode="r")
    lab = np.load(data_dir / f"{split}_labels.npy")
    return a, b, lab


def to_pair_tensor(a_u8: np.ndarray, b_u8: np.ndarray) -> torch.Tensor:
    """Stack all preprocessed pairs -> (N, 2, H, W) float32.

    Only used for small splits (val/test); for big splits see the memmap path.
    """
    n = a_u8.shape[0]
    pre = [preprocess(a_u8[i], b_u8[i]) for i in range(n)]
    return torch.stack(pre)


def resize_pair_tensor(pair: torch.Tensor, dst: int) -> torch.Tensor:
    """(..., 2, H, W) -> (..., 2, dst, dst) via area-ish bilinear resize.

    ``antialias`` is used when available (avoids moire/aliasing when
    down-sampling noisy frames); silently falls back otherwise.
    """
    if pair.shape[-1] == dst and pair.shape[-2] == dst:
        return pair
    try:
        return F.interpolate(pair, size=(dst, dst), mode="bilinear",
                             align_corners=False, antialias=True)
    except TypeError:  # pragma: no cover - older torch without antialias
        return F.interpolate(pair, size=(dst, dst), mode="bilinear",
                             align_corners=False)


def to_pair_tensor_resized(a_u8: np.ndarray, b_u8: np.ndarray, dst: int) -> torch.Tensor:
    """Percentile-normalise (like preprocess), then full-frame resize to ``dst``.

    This is the deterministic counterpart of the training-time ROI sampling with
    c == src: the label stays in original pixels and ``evaluate`` converts the
    prediction back via ``scale = src / dst``.
    """
    pair = to_pair_tensor(a_u8, b_u8)          # (N, 2, src, src)
    return resize_pair_tensor(pair, dst)


def roisample_pair(pair: torch.Tensor, y: torch.Tensor, src: int, dst: int,
                   cmin: int, cmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random-resolution ROI augmentation on a (2, src, src) float pair.

    Samples a square ROI of side ``c`` at a random location (identical for A
    and B, so it is a pure translation + scale: droll unchanged) and resizes it
    back to ``dst``. Because content is magnified by ``ke = dst / c``, the
    pixel-space offset dx/dy scales by ``ke``; cos/sin of the roll do not.

    With a 512px source and dst=192 this covers c in [96, 512] ->
    content-scale factors in [0.375, 2], i.e. deployment inputs between roughly
    64px and 512px after predict-style normalisation.
    """
    c = int(np.random.randint(cmin, min(cmax, src) + 1))
    y0 = int(np.random.randint(0, src - c + 1))
    x0 = int(np.random.randint(0, src - c + 1))
    roi = pair[:, y0 : y0 + c, x0 : x0 + c].unsqueeze(0)   # (1, 2, c, c)
    pair = resize_pair_tensor(roi, dst)[0]
    ke = dst / c
    y = y.clone()
    y[0] = y[0] * ke
    y[1] = y[1] * ke
    return pair, y


def frame_stretch_constants(imgs, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (1.0, 99.5) percentile -> (lo, span) (n,) float32 each.

    Mirrors ``model.preprocess`` exactly: a frame whose hi - lo <= 1e-6 is not
    rescaled (kept raw), which is encoded as lo=0, span=1.
    """
    lo = np.zeros(n, dtype=np.float32)
    span = np.ones(n, dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(n):
        vlo, vhi = np.percentile(imgs[i], (1.0, 99.5))
        d = float(vhi - vlo)
        if d > 1e-6:
            lo[i] = float(vlo)
            span[i] = d
        if (i + 1) % 50_000 == 0 or i == n - 1:
            el = time.perf_counter() - t0
            print(f"    norm {i + 1}/{n}   {el:6.0f}s elapsed  "
                  f"{(el / (i + 1)) * 1e3:4.1f} ms/frame", flush=True)
    return lo, span


def split_norm(data_dir: Path, split: str, a, b, n: int, src: int = 0):
    """(lo_a, span_a, lo_b, span_b) for ``split``; cached per (split, n, src).

    ``src`` is baked into the cache filename so datasets generated at different
    resolutions never silently reuse each other's percentile constants.
    """
    tag = f"_s{src}" if src else ""
    cache = data_dir / f"{split}_norm_{n}{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        if len(z["lo_a"]) == n:
            # np.savez arrays may stay lazily mmapped; materialise the copies.
            return tuple(np.array(z[k], copy=True) for k in
                         ("lo_a", "span_a", "lo_b", "span_b"))
    print(f"  computing per-frame stretch constants for {split} ({n} pairs) ...",
          flush=True)
    lo_a, span_a = frame_stretch_constants(a, n)
    lo_b, span_b = frame_stretch_constants(b, n)
    np.savez(cache, lo_a=lo_a, span_a=span_a, lo_b=lo_b, span_b=span_b)
    print(f"  cached norm constants to {cache.name}")
    return lo_a, span_a, lo_b, span_b


def fetch_norm_batch(a, b, norm, idx) -> np.ndarray:
    """Gather frames at ``idx`` (int array), percentile-stretch, stack.

    Returns a fresh writable float32 array of shape (B, 2, H, W) matching
    ``to_pair_tensor`` semantics: channel 0 = A, channel 1 = B.
    """
    la, sa, lb, sb = norm
    ia = np.asarray(idx, dtype=np.int64)
    la, sa = la[ia][:, None, None], sa[ia][:, None, None]
    lb, sb = lb[ia][:, None, None], sb[ia][:, None, None]
    xa = np.clip((a[ia].astype(np.float32) - la) / sa, 0.0, 1.0)
    xb = np.clip((b[ia].astype(np.float32) - lb) / sb, 0.0, 1.0)
    return np.stack((xa, xb), axis=1)


def horizontal_flip(pair: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pair = torch.flip(pair, dims=[2])  # mirror columns
    y = y.clone()
    y[0] = -y[0]  # dx
    y[3] = -y[3]  # sin(roll)
    return pair, y


def vertical_flip(pair: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pair = torch.flip(pair, dims=[1])  # mirror rows
    y = y.clone()
    y[1] = -y[1]  # dy
    y[3] = -y[3]  # sin(roll)
    return pair, y


def rotate_180(pair: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pair = torch.flip(pair, dims=[1, 2])
    y = y.clone()
    y[0] = -y[0]
    y[1] = -y[1]
    return pair, y


_AUGS = (horizontal_flip, vertical_flip, rotate_180)
_AUG_DIMS = {horizontal_flip: [2], vertical_flip: [1], rotate_180: [1, 2]}


def augment_pair(pair: torch.Tensor, y: torch.Tensor, crop: int | None = None,
                 p_crop: float = 0.0, hmap: torch.Tensor | None = None):
    """Augment one (2, H, W) pair (+ its (4,) encoded target) on the CPU.

    ``crop``: fixed-size random-location crop applied identically to both
    frames (a pure translation, so dx/dy/droll are unchanged). Mirrors and
    180 rotations flip the dx/dy/roll signs accordingly; photometric jitter
    and light noise are per channel/frame.

    ``hmap``: optional per-class heatmap target (C, gy, gx) which is spatially
    flipped *together* with the frames so transient GT stays consistent (crop
    and heatmaps are mutually exclusive: only full-frame flips are supported).
    Returns ``(pair, y)`` or ``(pair, y, hmap)``.
    """
    h, w = pair.shape[1:]
    if hmap is None and crop is not None and 0 < crop < h and np.random.rand() < p_crop:
        y0 = int(np.random.randint(0, h - crop + 1))
        x0 = int(np.random.randint(0, w - crop + 1))
        pair = pair[:, y0 : y0 + crop, x0 : x0 + crop]
    # mirror / 180 rotations, adjusting dx, dy and roll accordingly
    if np.random.rand() < 0.5:
        aug = _AUGS[int(np.random.randint(0, len(_AUGS)))]
        pair, y = aug(pair, y)
        if hmap is not None:
            hmap = torch.flip(hmap, dims=_AUG_DIMS[aug])
    # photometric: per-image brightness/contrast jitter + slight noise
    for c in range(2):
        s = float(np.random.uniform(0.7, 1.3))
        pair[c] = pair[c] * s
        if np.random.rand() < 0.5:
            pair[c] = pair[c] + float(np.random.uniform(-0.03, 0.03))
        if np.random.rand() < 0.3:
            pair[c] = pair[c] + torch.randn_like(pair[c]) * 0.01
    if hmap is not None:
        return pair, y, hmap
    return pair, y


# ---------------------------------------------------------------------------
# Loss + metrics
# ---------------------------------------------------------------------------


class RollAngleLoss(nn.Module):
    def __init__(self, w_deg: float = 1.0) -> None:
        super().__init__()
        # weight so that 1 degree of roll error costs about as much as
        # (w_deg) pixels of shift error
        self.w_rad2 = w_deg * w_deg * (math.pi / 180.0) ** 2 * 2.0

    def forward(self, raw: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # raw[:, 2:4] are un-normalised (c, s); angular error via dot product
        c = raw[:, 2]
        s = raw[:, 3]
        yc = y[:, 2]
        ys = y[:, 3]
        dot = c * yc + s * ys
        denom = torch.sqrt(c * c + s * s).clamp_min(1e-6) * torch.sqrt(yc * yc + ys * ys).clamp_min(1e-6)
        return torch.mean(1.0 - dot / denom)


def regression_loss(raw: torch.Tensor, y: torch.Tensor, w_shift: float = 1.0, w_roll: float = 1.0) -> torch.Tensor:
    dx = nn.functional.smooth_l1_loss(raw[:, 0], y[:, 0], beta=1.0)
    dy = nn.functional.smooth_l1_loss(raw[:, 1], y[:, 1], beta=1.0)
    ang = RollAngleLoss(w_roll)(raw, y)
    return w_shift * (dx + dy) + ang


def metrics(pred: np.ndarray, lab: np.ndarray) -> dict:
    """pred/lab: (N, 3) = [dx, dy, droll_deg]. Error magnitude stats."""
    e = np.abs(pred - lab)
    mag = np.hypot(e[:, 0], e[:, 1])
    ang = np.abs(((pred[:, 2] - lab[:, 2] + 180.0) % 360.0) - 180.0)
    return {
        "mae_dx": float(np.mean(e[:, 0])),
        "mae_dy": float(np.mean(e[:, 1])),
        "mae_mag": float(np.mean(mag)),
        "med_mag": float(np.median(mag)),
        "p90_mag": float(np.percentile(mag, 90)),
        "mae_roll": float(np.mean(ang)),
        "med_roll": float(np.median(ang)),
        "p90_roll": float(np.percentile(ang, 90)),
        "frac<1px": float(np.mean(mag < 1.0)),
        "frac<2px": float(np.mean(mag < 2.0)),
        "frac<0.5d": float(np.mean(ang < 0.5)),
        "frac<1.0d": float(np.mean(ang < 1.0)),
    }


def fmt(m: dict) -> str:
    return (
        f"dx {m['mae_dx']:.2f} dy {m['mae_dy']:.2f} |shift| {m['med_mag']:.2f}px"
        f" (p90 {m['p90_mag']:.2f}) | roll {m['med_roll']:.2f}\u00b0 (p90 {m['p90_roll']:.2f})"
        f" | <1px {m['frac<1px']:.1%} <1\u00b0 {m['frac<1.0d']:.1%}"
    )


# ---------------------------------------------------------------------------
# Transient detection: heatmap targets / loss / metrics
# ---------------------------------------------------------------------------
#
# GT lives in the dataset's native pixels (``trans_x/y`` in *_meta.npz).  The
# detection head works on the stride-8 model-input grid, so targets are built
# by binning native pixel coords /8 into a (C, gy, gx) tensor of soft Gaussian
# blobs.  Only supported in native mode (model_in == src); with ROI resampling
# the detection loss is simply disabled (head stays untrained).


def build_heat_targets(meta: dict, n: int, grid: int, sigma: float = 1.2,
                       n_cls: int = 4) -> torch.Tensor | None:
    """(n, C, grid, grid) int8 target tensor from meta GT arrays.

    CenterNet-style supervision: +1 at each source centre cell (radius ~1),
    -1 in the ignore ring around it, 0 elsewhere.  Point transients
    (new/brighten/dim) come from ``trans_*``; satellite trails are stored as
    multiple centreline points (class 3) in ``sat_*``.
    """
    if "trans_n" not in meta and "sat_n" not in meta:
        return None
    tn = np.asarray(meta["trans_n"], dtype=np.int64)[:n] if "trans_n" in meta else np.zeros(n, np.int64)
    tx = np.asarray(meta["trans_x"]) if "trans_x" in meta else np.zeros((n, 1))
    ty = np.asarray(meta["trans_y"]) if "trans_y" in meta else np.zeros((n, 1))
    tc = np.asarray(meta["trans_cls"]) if "trans_cls" in meta else np.zeros((n, 1))
    sn = np.asarray(meta["sat_n"], dtype=np.int64)[:n] if "sat_n" in meta else np.zeros(n, np.int64)
    sx = np.asarray(meta["sat_x"]) if "sat_x" in meta else np.zeros((n, 1))
    sy = np.asarray(meta["sat_y"]) if "sat_y" in meta else np.zeros((n, 1))
    out = np.zeros((n, n_cls, grid, grid), dtype=np.int8)

    def stamp(i: int, cl: int, cx: float, cy: float) -> None:
        if not (0 <= cl < n_cls) or not (0 <= cx < grid) or not (0 <= cy < grid):
            return
        x0 = max(0, int(math.floor(cy)) - 2)
        x1 = min(grid, int(math.ceil(cy)) + 3)
        y0 = max(0, int(math.floor(cx)) - 2)
        y1 = min(grid, int(math.ceil(cx)) + 3)
        for yy in range(x0, x1):
            for xx in range(y0, y1):
                d = math.hypot(yy - cy, xx - cx)
                if d <= 0.8:
                    out[i, cl, yy, xx] = 1
                elif d <= 1.8 and out[i, cl, yy, xx] == 0:
                    out[i, cl, yy, xx] = -1

    for i in range(n):
        for j in range(int(tn[i])):
            if j >= tx.shape[1]:
                break
            stamp(i, int(round(float(tc[i, j]))),
                  float(tx[i, j]) / DET_STRIDE, float(ty[i, j]) / DET_STRIDE)
        for j in range(int(sn[i])):
            if j >= sx.shape[1]:
                break
            stamp(i, SAT_CLS, float(sx[i, j]) / DET_STRIDE,
                  float(sy[i, j]) / DET_STRIDE)
    return torch.as_tensor(out)


class TransientHeatLoss(nn.Module):
    """Focal loss on binary heatmaps with an ignore ring (target -1)."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 neg_w: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.neg_w = neg_w

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
        y = target
        pos = y > 0.5
        ign = y < -0.5   # ignore ring
        lpos = -self.alpha * ((1.0 - p) ** self.gamma) * torch.log(p)
        lneg = -(1.0 - self.alpha) * (p ** self.gamma) * torch.log(1.0 - p)
        n_cls = logits.shape[1]
        # Per-class balanced terms: satellite trails contribute dozens of GT
        # points each, so a global positive mean would swamp the sparse
        # point-source classes (new/brighten/dim).
        pos_acc = torch.zeros((), device=logits.device)
        neg_acc = torch.zeros((), device=logits.device)
        n_pos = 0
        n_neg = 0
        for c in range(n_cls):
            pc = pos[:, c]
            if pc.any():
                pos_acc = pos_acc + lpos[:, c][pc].mean()
                n_pos += 1
            negc = (~pos[:, c]) & (~ign[:, c])
            if negc.any():
                vals = lneg[:, c][negc]
                k = min(256, vals.numel())
                neg_acc = neg_acc + vals.topk(k).values.mean()
                n_neg += 1
        loss = pos_acc / max(1, n_pos) + self.neg_w * (neg_acc / max(1, n_neg))
        return loss


def det_metrics(model_pred_peaks: list, gt: np.ndarray, tol_px: float = 8.0,
                n_cls: int = 3, sat_cls: int = SAT_CLS,
                sat_radius: float = 24.0) -> dict | None:
    """Peak-matching precision / recall per class.

    Point classes (new/brighten/dim) are matched one-to-one.  Satellite GT is
    stored as many centreline points; those are clustered into trail
    *instances* and a hit on any point detects the whole trail (so recall is
    not diluted by trail length).
    """
    n = len(model_pred_peaks)
    tp = np.zeros(n_cls); fp = np.zeros(n_cls); fn = np.zeros(n_cls)
    for i in range(n):
        gts = [(float(g[0]), float(g[1]), int(g[2])) for g in gt[i]]
        used_g = set()
        sat_pts = [(g[0], g[1]) for g in gts if g[2] == sat_cls]
        sat_inst = cluster_points(sat_pts, sat_radius) if sat_pts else []
        matched_inst: set[int] = set()
        # cluster_points returns indices into sat_pts; keep mapping for points
        sat_origin = [(g[0], g[1]) for g in gts if g[2] == sat_cls]

        for (cl, x, y, s) in model_pred_peaks[i]:
            if cl == sat_cls:
                hit = None
                for ii, grp in enumerate(sat_inst):
                    for k in grp:
                        gx, gy = sat_origin[k]
                        if math.hypot(x - gx, y - gy) <= tol_px:
                            hit = ii
                            break
                    if hit is not None:
                        break
                if hit is None:
                    fp[sat_cls] += 1
                elif hit not in matched_inst:
                    tp[sat_cls] += 1
                    matched_inst.add(hit)
                # duplicate detection on an already-found trail: ignored
                continue
            # point classes
            best = None
            for gi, (gx, gy, gcl) in enumerate(gts):
                if gi in used_g or gcl != cl:
                    continue
                d = math.hypot(x - gx, y - gy)
                if d <= tol_px and (best is None or d < best[0]):
                    best = (d, gi)
            if best is not None:
                tp[cl] += 1
                used_g.add(best[1])
            else:
                fp[cl] += 1
        # misses
        for gi, (gx, gy, gcl) in enumerate(gts):
            if gcl == sat_cls:
                continue
            if gi not in used_g:
                fn[gcl] += 1
        fn[sat_cls] += len(sat_inst) - len(matched_inst)

    tot_g = int(tp.sum() + fn.sum())
    tot_p = int(tp.sum() + fp.sum())
    tp_s, fp_s, fn_s = float(tp.sum()), float(fp.sum()), float(fn.sum())
    prec = tp_s / max(1e-9, tp_s + fp_s)
    rec = tp_s / max(1e-9, tp_s + fn_s)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    out = {"det_prec": prec, "det_rec": rec, "det_f1": f1,
           "det_tp": int(tp_s), "det_fp": int(fp_s), "det_fn": int(fn_s),
           "det_gt": tot_g, "det_pred": tot_p}
    for c in range(n_cls):
        p = tp[c] / max(1e-9, tp[c] + fp[c])
        r = tp[c] / max(1e-9, tp[c] + fn[c])
        out[f"det_c{c}_p"] = float(p)
        out[f"det_c{c}_r"] = float(r)
        out[f"det_c{c}_f1"] = float(2 * p * r / max(1e-9, p + r))
    return out


def fmt_det(d: dict) -> str:
    if d is None:
        return "det: -"
    return (f"det P {d['det_prec']:.2f} R {d['det_rec']:.2f} F1 {d['det_f1']:.2f}"
            f" ({d['det_tp']}/{d['det_gt']})")


@torch.no_grad()
def evaluate(model: nn.Module, pairs: torch.Tensor, lab: np.ndarray, device,
             batch: int, scale: float = 1.0, det_gt: list | None = None,
             det_scale: float = 1.0, det_n_cls: int = 4) -> dict:
    """pairs: (N,2,H,W) preprocessed; lab: (N,3) [dx, dy, droll_deg].

    ``scale`` converts the model's pixel predictions (valid in the *model
    input* coordinate grid, e.g. a down-sampled 192px view) back to the
    dataset's original pixel coordinates: multiply dx/dy by it.

    If ``det_gt`` (per-sample list of (x, y, cls) native px) is given, peak
    detection metrics are computed as well (det peaks converted to native px
    by ``det_scale``).
    """
    model.eval()
    preds = []
    peaks_all: list | None = [] if det_gt is not None else None
    for i in range(0, len(pairs), batch):
        pb = pairs[i : i + batch].to(device)
        pose, det = model(pb)
        out = decode(pose)
        if scale != 1.0:
            out = out.clone()
            out[:, 0] = out[:, 0] * scale
            out[:, 1] = out[:, 1] * scale
        preds.append(out.cpu().numpy())
        if peaks_all is not None and det is not None:
            prob = torch.sigmoid(det).cpu()
            for pk in heat_to_peaks(prob):
                if det_scale != 1.0:
                    pk = [(cl, x * det_scale, y * det_scale, s)
                          for (cl, x, y, s) in pk]
                peaks_all.append(pk)
    pred = np.concatenate(preds, axis=0)
    m = metrics(pred, lab)
    if peaks_all is not None and det_gt is not None:
        m.update(det_metrics(peaks_all, det_gt, tol_px=scale * 8.0, n_cls=det_n_cls))
    return m


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _write_best_info(model_dir: Path, dst: int, src: int, metric_scale: float,
                     vm: dict) -> None:
    """Sidecar JSON next to best.pt describing the model's input grid.

    ``best.pt`` is a bare state dict (no metadata), so inference tools
    (predict.py / validate_model_gui.py) read this file to know the pixel size
    the network was trained on instead of hard-coding 192.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "best_info.json").open("w") as f:
        json.dump(
            {"model_in": int(dst), "native": int(src),
             "metric_scale": float(metric_scale),
             "best_val": vm},
            f,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "models")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--crop", type=int, default=0,
                   help="train on fixed-size random crops of this size "
                        "(0 = full frame)")
    p.add_argument("--roi-min", type=int, default=0,
                   help="random-resolution ROI augmentation: min ROI side (px); "
                        "0 disables the augmentation")
    p.add_argument("--roi-max", type=int, default=0,
                   help="max ROI side (px); use <= the native data size, e.g. "
                        "--roi-min 96 --roi-max 512 for 512px data (content "
                        "scale ~64..512px after predict normalisation)")
    p.add_argument("--det-w", type=float, default=0.5,
                   help="weight of the transient heatmap loss (0 disables "
                        "detection training even if the data has GT)")
    p.add_argument("--det-neg-w", type=float, default=1.0,
                   help="weight of the hard-negative term in the detection "
                        "loss (lower = less background suppression)")
    p.add_argument("--max-train", type=int, default=0,
                   help="cap the number of train samples actually used "
                        "(0 = all); handy to smoke-test on a huge dataset")
    p.add_argument("--resume", type=Path, default=None,
                   help="state dict / checkpoint to initialise the model from "
                        "(two-stage: train pose first, then resume with "
                        "--det-w and a low --lr to tune detection)")
    p.add_argument("--freeze-pose", action="store_true",
                   help="freeze every parameter except the detection head "
                        "(resume from a converged pose checkpoint)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    print(f"device: {device}")

    t0 = time.perf_counter()
    # train split is streamed (memmap + per-batch fetch) so 100k+ samples fit
    # in RAM; val/test are tiny and keep the load-everything path.
    tr_a, tr_b, tr_y = open_split_mmap(args.data, "train")
    va_a, va_b, va_y = load_split(args.data, "val")
    te_a, te_b, te_y = load_split(args.data, "test")
    src = int(tr_a.shape[1])          # native dataset resolution
    n = int(tr_a.shape[0])
    if 0 < args.max_train < n:
        n = args.max_train
        tr_a, tr_b = tr_a[:n], tr_b[:n]
        tr_y = tr_y[:n]

    # Random-resolution ROI augmentation: the network always sees ``dst`` and
    # each label is rescaled by ke = dst / c, so one model covers a wide range
    # of deployment resolutions (predict.py normalises them to dst anyway).
    roi_active = 0 < args.roi_min <= args.roi_max
    if roi_active and args.roi_max > src:
        print(f"note: roi-max {args.roi_max} > native {src}; clamping to {src}")
    dst = 192 if (roi_active and src >= 192) else src
    metric_scale = float(src / dst) if dst != src else 1.0
    print(f"load: train {n} val {len(va_a)} test {len(te_a)}  "
          f"native {src}  model-in {dst}  roi {args.roi_min}-{min(args.roi_max, src) if roi_active else 0}"
          f"  ({time.perf_counter() - t0:.1f}s)")

    tr_norm = split_norm(args.data, "train", tr_a, tr_b, n, src)
    train_ys = encode_target(tr_y)
    if dst != src:
        val_pairs = to_pair_tensor_resized(va_a, va_b, dst)
        test_pairs = to_pair_tensor_resized(te_a, te_b, dst)
    else:
        val_pairs = to_pair_tensor(va_a, va_b)
        test_pairs = to_pair_tensor(te_a, te_b)
    val_ys = encode_target(va_y)
    test_ys = encode_target(te_y)
    print(f"preprocess done ({time.perf_counter() - t0:.1f}s)")

    lab = np.load(args.data / "test_labels.npy")
    zero = np.zeros_like(lab)
    print("baseline (predict dx=dy=roll=0):", fmt(metrics(zero, lab)))

    # ---- transient detection setup (native mode only) ----------------------
    det_active = False
    n_cls = 0
    det_w = 0.0
    grid = dst // DET_STRIDE
    det_loss = None
    val_det_gt = None
    val_det_targets = None
    test_det_gt = None
    test_det_targets = None
    tr_tn = tr_tx = tr_ty = tr_tc = None
    tr_sn = tr_sx = tr_sy = None

    if (args.det_w > 0 and not roi_active and args.crop <= 0
            and (args.data / "train_meta.npz").exists()):
        zt = np.load(args.data / "train_meta.npz")
        if "trans_n" in zt.files or "sat_n" in zt.files:
            det_active = True
            has_sat = "sat_n" in zt.files
            cls_max = -1
            if "trans_cls" in zt.files and "trans_n" in zt.files:
                tn_ = np.asarray(zt["trans_n"])
                tc_ = np.asarray(zt["trans_cls"])
                valid = [int(round(tc_[i, j])) for i in range(len(tn_))
                         for j in range(int(tn_[i]))]
                cls_max = max(valid) if valid else -1
            n_cls = max(cls_max + 1, (SAT_CLS + 1) if has_sat else 0, 1)
            det_w = float(args.det_w)
            tr_tn = np.asarray(zt["trans_n"]) if "trans_n" in zt.files else None
            tr_tx = np.asarray(zt["trans_x"]) if "trans_x" in zt.files else None
            tr_ty = np.asarray(zt["trans_y"]) if "trans_y" in zt.files else None
            tr_tc = np.asarray(zt["trans_cls"]) if "trans_cls" in zt.files else None
            tr_sn = np.asarray(zt["sat_n"]) if has_sat else None
            tr_sx = np.asarray(zt["sat_x"]) if has_sat else None
            tr_sy = np.asarray(zt["sat_y"]) if has_sat else None
            det_loss = TransientHeatLoss(neg_w=float(args.det_neg_w))

            def _load_trans_gt(path: Path, nn: int):
                zp = np.load(path)
                arr = {k: np.asarray(zp[k]) for k in zp.files
                       if k in ("trans_n", "trans_x", "trans_y", "trans_cls",
                                "sat_n", "sat_x", "sat_y")}
                tgt = build_heat_targets(arr, nn, grid, n_cls=n_cls)
                gt = []
                for i in range(nn):
                    row = []
                    for j in range(int(arr.get("trans_n", np.zeros(nn))[i])):
                        row.append((float(arr["trans_x"][i, j]),
                                    float(arr["trans_y"][i, j]),
                                    int(round(float(arr["trans_cls"][i, j])))))
                    for j in range(int(arr.get("sat_n", np.zeros(nn))[i])):
                        row.append((float(arr["sat_x"][i, j]),
                                    float(arr["sat_y"][i, j]), SAT_CLS))
                    gt.append(row)
                return tgt, gt

            val_det_targets, val_det_gt = _load_trans_gt(
                args.data / "val_meta.npz", len(va_y))
            test_det_targets, test_det_gt = _load_trans_gt(
                args.data / "test_meta.npz", len(te_y))
    if det_active:
        print(f"transient detection on: {n_cls} classes, heat grid {grid}")

    crop = args.crop if 0 < args.crop < dst else None
    model = PairRegNet(n_cls=n_cls).to(device)
    if args.resume is not None:
        robj = torch.load(args.resume, map_location=device)
        rsd = robj["model"] if isinstance(robj, dict) and "model" in robj else robj
        # lenient: allows resuming a pose-only checkpoint into a detection run
        # (detection weights stay random) or architecture evolution of det.
        missing, unexpected = model.load_state_dict(rsd, strict=False)
        print(f"resumed from {args.resume}  (missing {len(missing)} tensors, "
              f"unused {len(unexpected)})")
    if args.freeze_pose:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("det.")
        nf = sum(1 for p in model.parameters() if not p.requires_grad)
        print(f"froze {nf} parameter tensors outside the detection head")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = None

    for ep in range(args.epochs):
        model.train()
        t_ep = time.perf_counter()
        order = np.random.permutation(n)
        tot_loss = 0.0
        tot_reg = 0.0
        tot_det = 0.0
        nb = 0
        for i in range(0, n, args.batch):
            idx = order[i : i + args.batch]
            pb = torch.from_numpy(fetch_norm_batch(tr_a, tr_b, tr_norm, idx))
            yb = train_ys[torch.from_numpy(np.asarray(idx, dtype=np.int64))].clone()
            hb_raw = None
            if det_active:
                meta_b = {}
                if tr_tn is not None:
                    meta_b.update({"trans_n": tr_tn[idx], "trans_x": tr_tx[idx],
                                   "trans_y": tr_ty[idx], "trans_cls": tr_tc[idx]})
                if tr_sn is not None:
                    meta_b.update({"sat_n": tr_sn[idx], "sat_x": tr_sx[idx],
                                   "sat_y": tr_sy[idx]})
                hb_raw = build_heat_targets(meta_b, len(idx), grid, n_cls=n_cls)
            pairs, ys, hmaps = [], [], []
            for j in range(len(pb)):
                p, y = pb[j], yb[j]
                hj = hb_raw[j] if hb_raw is not None else None
                if roi_active:
                    p, y = roisample_pair(p, y, src, dst,
                                          args.roi_min, min(args.roi_max, src))
                out3 = augment_pair(p, y, crop=crop,
                                    p_crop=1.0 if crop else 0.0, hmap=hj)
                if hj is not None:
                    p, y, hj = out3
                    hmaps.append(hj)
                else:
                    p, y = out3
                pairs.append(p)
                ys.append(y)
            pb = torch.stack(pairs).to(device)
            yb = torch.stack(ys).to(device)
            opt.zero_grad()
            pose, det = model(pb)
            rloss = regression_loss(pose, yb)
            loss = rloss
            dl = None
            if det_active and det is not None:
                dl = det_loss(det, torch.stack(hmaps).to(device))
                loss = loss + det_w * dl
            loss.backward()
            opt.step()
            tot_loss += float(loss) * len(pb)
            tot_reg += float(rloss) * len(pb)
            if dl is not None:
                tot_det += float(dl) * len(pb)
            nb += len(pb)
        sched.step()
        vm = evaluate(model, val_pairs, va_y, device, args.batch,
                      scale=metric_scale, det_gt=val_det_gt if det_active else None,
                      det_n_cls=n_cls)
        det_str = fmt_det(vm) if det_active else ""
        print(
            f"ep {ep + 1:02d}/{args.epochs}  loss {tot_loss / nb:.4f}  "
            f"lr {sched.get_last_lr()[0]:.1e}  ({time.perf_counter() - t_ep:.0f}s)  "
            f"val: {fmt(vm)}  {det_str}"
        )
        score = vm["med_mag"] + 0.25 * vm["med_roll"]
        if best is None or score < best[0]:
            best = (score, ep, vm)
            args.model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.model_dir / "best.pt")
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "metrics": vm,
                 "model_in": dst, "native": src, "metric_scale": metric_scale},
                args.model_dir / "checkpoint_best.pt",
            )
            _write_best_info(args.model_dir, dst, src, metric_scale, vm)

    if best is None:
        raise RuntimeError("no epoch ran")
    _, ep_best, vm_best = best
    print(f"\nbest val at epoch {ep_best + 1}: {fmt(vm_best)}"
          + (f"  {fmt_det(vm_best)}" if det_active else ""))

    model.load_state_dict(torch.load(args.model_dir / "best.pt"))
    for split, pairs, labn, dgt in (
        ("val", val_pairs, va_y, val_det_gt if det_active else None),
        ("test", test_pairs, te_y, test_det_gt if det_active else None),
    ):
        m = evaluate(model, pairs, labn, device, args.batch, scale=metric_scale,
                     det_gt=dgt, det_n_cls=n_cls)
        line = f"{split} final: {fmt(m)}"
        if dgt is not None:
            line += f"  {fmt_det(m)}"
        print(line)

    with (args.model_dir / "train_summary.json").open("w") as f:
        json.dump(
            {
                "best_epoch": ep_best,
                "val": vm_best,
                "model_in": dst,
                "native": src,
                "metric_scale": metric_scale,
                "det_classes": n_cls,
                "det_weight": det_w,
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
