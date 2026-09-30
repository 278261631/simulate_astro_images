#!/usr/bin/env python3
"""Train the tiny single-image OB/unusable-region segmenter on data_256.

Uses both A and B frames as independent single-image samples (their borders
were sampled per-exposure).  No pose, no pairing: a frame -> 64x64 mask.

Usage:
    python train_ob.py --data ../data_256 --out ./models_ob --epochs 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from model_ob import OBNet  # noqa: E402


class DiceBCE(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum()
        dice = 1.0 - (2.0 * inter + 1.0) / (p.sum() + target.sum() + 1.0)
        return bce + dice


def load_split(data: Path, split: str):
    a = np.load(data / f"{split}_a.npy", mmap_mode="r")
    b = np.load(data / f"{split}_b.npy", mmap_mode="r")
    z = np.load(data / f"{split}_meta.npz")
    return a, b, np.asarray(z["ob_a"]), np.asarray(z["ob_b"])


def gather(a, b, ma, mb, idx: np.ndarray):
    """(imgs uint8, masks uint8) for a batch of virtual indices (A then B)."""
    N = a.shape[0]
    isb = idx >= N
    imgs = np.empty((len(idx), a.shape[1], a.shape[2]), dtype=np.uint8)
    masks = np.empty((len(idx), ma.shape[1], ma.shape[2]), dtype=np.uint8)
    if (~isb).any():
        ii = idx[~isb]
        imgs[~isb] = a[ii]
        masks[~isb] = ma[ii]
    if isb.any():
        ii = idx[isb] - N
        imgs[isb] = b[ii]
        masks[isb] = mb[ii]
    return imgs, masks


def augment_batch(imgs: np.ndarray, masks: np.ndarray, rng: np.random.RandomState) -> None:
    """In-place random flip (none/h/v/180) per sample; borders stay borders."""
    mode = rng.randint(0, 4, size=len(imgs))
    sh = (mode == 1) | (mode == 3)
    sv = (mode == 2) | (mode == 3)
    if sh.any():
        imgs[sh] = imgs[sh][:, :, ::-1]
        masks[sh] = masks[sh][:, :, ::-1]
    if sv.any():
        imgs[sv] = imgs[sv][:, ::-1, :]
        masks[sv] = masks[sv][:, ::-1, :]


@torch.no_grad()
def evaluate(model, a, b, ma, mb, device, batch, max_n=0):
    model.eval()
    N = a.shape[0]
    n = 2 * N if max_n <= 0 else min(max_n, 2 * N)
    inter = union = 0.0
    for i in range(0, n, batch):
        idx = np.arange(i, min(i + batch, n))
        imgs, masks = gather(a, b, ma, mb, idx)
        x = torch.from_numpy(imgs.astype(np.float32).reshape(-1, 1, *imgs.shape[1:]) / 255.0).to(device)
        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()[:, 0]
        gt = masks.astype(bool)
        inter += float((pred & gt).sum())
        union += float((pred | gt).sum())
    return inter / max(1.0, union)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=HERE.parent / "data_256")
    p.add_argument("--out", type=Path, default=HERE / "models_ob")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    print("device:", device)

    tr_a, tr_b, tr_ma, tr_mb = load_split(args.data, "train")
    va_a, va_b, va_ma, va_mb = load_split(args.data, "val")
    N = tr_a.shape[0]
    n = 2 * N if args.max_train <= 0 else min(args.max_train, 2 * N)
    print(f"train frames {n} (A+B of {N})  val frames {2*va_a.shape[0]}")

    model = OBNet(base=args.base).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model params: {nparam/1e3:.1f} K")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = DiceBCE()
    best = -1.0

    for ep in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        order = rng.permutation(n)
        tot = 0.0; nb = 0
        for i in range(0, n, args.batch):
            idx = order[i:i + args.batch]
            imgs, masks = gather(tr_a, tr_b, tr_ma, tr_mb, idx)
            augment_batch(imgs, masks, rng)
            x = torch.from_numpy(imgs.astype(np.float32)[:, None] / 255.0).to(device)
            y = torch.from_numpy(masks.astype(np.float32)[:, None]).to(device)
            opt.zero_grad()
            logits = model(x)
            loss = lossf(logits, y)
            loss.backward(); opt.step()
            tot += float(loss) * len(idx); nb += len(idx)
        sched.step()
        viou = evaluate(model, va_a, va_b, va_ma, va_mb, device, args.batch)
        print(f"ep {ep+1:02d}/{args.epochs}  loss {tot/nb:.4f}  val OB IoU {viou:.4f}"
              f"  ({time.perf_counter()-t0:.0f}s)")
        if viou > best:
            best = viou
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out / "best.pt")
            torch.save({"model": model.state_dict(), "base": args.base,
                        "val_iou": viou, "args": vars(args)}, args.out / "checkpoint_best.pt")
    print(f"best val OB IoU {best:.4f}  -> {args.out/'best.pt'}")


if __name__ == "__main__":
    main()
