#!/usr/bin/env python3
"""Train PairRegNet on the generated (A, B) -> (dx, dy, droll) dataset.

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
# Dataset loading
# ---------------------------------------------------------------------------


def load_split(data_dir: Path, split: str):
    a = np.load(data_dir / f"{split}_a.npy")
    b = np.load(data_dir / f"{split}_b.npy")
    lab = np.load(data_dir / f"{split}_labels.npy")
    return a, b, lab


def to_pair_tensor(a_u8: np.ndarray, b_u8: np.ndarray) -> torch.Tensor:
    """Stack all preprocessed pairs -> (N, 2, H, W) float32."""
    n = a_u8.shape[0]
    pre = [preprocess(a_u8[i], b_u8[i]) for i in range(n)]
    return torch.stack(pre)


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


def augment_pair(pair: torch.Tensor, y: torch.Tensor, crop_min: int | None = None,
                 p_crop: float = 0.0):
    h, w = pair.shape[1:]
    # identical random crop on both frames; offsets are unchanged by it
    if crop_min is not None and p_crop > 0.0 and h > crop_min:
        if np.random.rand() < p_crop:
            cs = int(np.random.randint(crop_min, h))
            y0 = int(np.random.randint(0, h - cs + 1))
            x0 = int(np.random.randint(0, w - cs + 1))
            pair = pair[:, y0 : y0 + cs, x0 : x0 + cs]
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
                   help="train on random crops of this size (0 = full frame)")
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
    tr_a, tr_b, tr_y = load_split(args.data, "train")
    va_a, va_b, va_y = load_split(args.data, "val")
    te_a, te_b, te_y = load_split(args.data, "test")
    size = tr_a.shape[1]
    print(f"load: train {len(tr_a)} val {len(va_a)} test {len(te_a)}  size {size}  "
          f"({time.perf_counter() - t0:.1f}s)")

    train_pairs = to_pair_tensor(tr_a, tr_b)
    train_ys = encode_target(tr_y)
    val_pairs = to_pair_tensor(va_a, va_b)
    val_ys = encode_target(va_y)
    test_pairs = to_pair_tensor(te_a, te_b)
    test_ys = encode_target(te_y)
    print(f"preprocess done ({time.perf_counter() - t0:.1f}s)")

    lab = np.load(args.data / "test_labels.npy")
    zero = np.zeros_like(lab)
    print("baseline (predict dx=dy=roll=0):", fmt(metrics(zero, lab)))

    crop = args.crop or size
    model = PairRegNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n = len(train_pairs)
    best = None

    for ep in range(args.epochs):
        model.train()
        t_ep = time.perf_counter()
        order = torch.randperm(n)
        tot_loss = 0.0
        nb = 0
        for i in range(0, n, args.batch):
            idx = order[i : i + args.batch]
            pb = train_pairs[idx].clone()
            yb = train_ys[idx].clone()
            for j in range(len(pb)):
                pb[j], yb[j] = augment_pair(
                    pb[j], yb[j],
                    crop_min=(crop if crop < size else None),
                    p_crop=0.9 if crop < size else 0.0,
                )
            pb = pb.to(device)
            yb = yb.to(device)
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
