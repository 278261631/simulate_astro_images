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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import PairRegNet, decode, encode_target, preprocess  # noqa: E402


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


def split_norm(data_dir: Path, split: str, a, b, n: int):
    """(lo_a, span_a, lo_b, span_b) for ``split``; cached to disk per (split, n)."""
    cache = data_dir / f"{split}_norm_{n}.npz"
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


def augment_pair(pair: torch.Tensor, y: torch.Tensor, crop: int | None = None,
                 p_crop: float = 0.0):
    """Augment one (2, H, W) pair (+ its (4,) encoded target) on the CPU.

    ``crop``: fixed-size random-location crop applied identically to both
    frames (a pure translation, so dx/dy/droll are unchanged). Mirrors and
    180 rotations flip the dx/dy/roll signs accordingly; photometric jitter
    and light noise are per channel/frame.
    """
    h, w = pair.shape[1:]
    if crop is not None and 0 < crop < h and np.random.rand() < p_crop:
        y0 = int(np.random.randint(0, h - crop + 1))
        x0 = int(np.random.randint(0, w - crop + 1))
        pair = pair[:, y0 : y0 + crop, x0 : x0 + crop]
    # mirror / 180 rotations, adjusting dx, dy and roll accordingly
    if np.random.rand() < 0.5:
        aug = _AUGS[int(np.random.randint(0, len(_AUGS)))]
        pair, y = aug(pair, y)
    # photometric: per-image brightness/contrast jitter + slight noise
    for c in range(2):
        s = float(np.random.uniform(0.7, 1.3))
        pair[c] = pair[c] * s
        if np.random.rand() < 0.5:
            pair[c] = pair[c] + float(np.random.uniform(-0.03, 0.03))
        if np.random.rand() < 0.3:
            pair[c] = pair[c] + torch.randn_like(pair[c]) * 0.01
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


@torch.no_grad()
def evaluate(model: nn.Module, pairs: torch.Tensor, lab: np.ndarray, device, batch: int) -> dict:
    """pairs: (N,2,H,W) preprocessed; lab: (N,3) [dx, dy, droll_deg]."""
    model.eval()
    preds = []
    for i in range(0, len(pairs), batch):
        pb = pairs[i : i + batch].to(device)
        out = decode(model(pb)).cpu().numpy()
        preds.append(out)
    pred = np.concatenate(preds, axis=0)
    return metrics(pred, lab)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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
    p.add_argument("--max-train", type=int, default=0,
                   help="cap the number of train samples actually used "
                        "(0 = all); handy to smoke-test on a huge dataset")
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
    size = tr_a.shape[1]
    n = int(tr_a.shape[0])
    if 0 < args.max_train < n:
        n = args.max_train
        tr_a, tr_b = tr_a[:n], tr_b[:n]
        tr_y = tr_y[:n]
    print(f"load: train {n} val {len(va_a)} test {len(te_a)}  size {size}  "
          f"({time.perf_counter() - t0:.1f}s)")

    tr_norm = split_norm(args.data, "train", tr_a, tr_b, n)
    train_ys = encode_target(tr_y)
    val_pairs = to_pair_tensor(va_a, va_b)
    val_ys = encode_target(va_y)
    test_pairs = to_pair_tensor(te_a, te_b)
    test_ys = encode_target(te_y)
    print(f"preprocess done ({time.perf_counter() - t0:.1f}s)")

    lab = np.load(args.data / "test_labels.npy")
    zero = np.zeros_like(lab)
    print("baseline (predict dx=dy=roll=0):", fmt(metrics(zero, lab)))

    crop = args.crop if 0 < args.crop < size else None
    model = PairRegNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = None

    for ep in range(args.epochs):
        model.train()
        t_ep = time.perf_counter()
        order = np.random.permutation(n)
        tot_loss = 0.0
        nb = 0
        for i in range(0, n, args.batch):
            idx = order[i : i + args.batch]
            pb = torch.from_numpy(fetch_norm_batch(tr_a, tr_b, tr_norm, idx))
            yb = train_ys[torch.from_numpy(np.asarray(idx, dtype=np.int64))].clone()
            pairs, ys = [], []
            for j in range(len(pb)):
                p, y = augment_pair(pb[j], yb[j], crop=crop, p_crop=1.0 if crop else 0.0)
                pairs.append(p)
                ys.append(y)
            pb = torch.stack(pairs).to(device)
            yb = torch.stack(ys).to(device)
            opt.zero_grad()
            out = model(pb)
            loss = regression_loss(out, yb)
            loss.backward()
            opt.step()
            tot_loss += float(loss) * len(pb)
            nb += len(pb)
        sched.step()
        vm = evaluate(model, val_pairs, va_y, device, args.batch)
        print(
            f"ep {ep + 1:02d}/{args.epochs}  loss {tot_loss / nb:.4f}  "
            f"lr {sched.get_last_lr()[0]:.1e}  ({time.perf_counter() - t_ep:.0f}s)  "
            f"val: {fmt(vm)}"
        )
        score = vm["med_mag"] + 0.25 * vm["med_roll"]
        if best is None or score < best[0]:
            best = (score, ep, vm)
            args.model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.model_dir / "best.pt")
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "metrics": vm},
                args.model_dir / "checkpoint_best.pt",
            )

    if best is None:
        raise RuntimeError("no epoch ran")
    _, ep_best, vm_best = best
    print(f"\nbest val at epoch {ep_best + 1}: {fmt(vm_best)}")

    model.load_state_dict(torch.load(args.model_dir / "best.pt"))
    for split, pairs, lab in (("val", val_pairs, va_y), ("test", test_pairs, te_y)):
        m = evaluate(model, pairs, lab, device, args.batch)
        print(f"{split} final: {fmt(m)}")

    with (args.model_dir / "train_summary.json").open("w") as f:
        json.dump(
            {
                "best_epoch": ep_best,
                "val": vm_best,
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
