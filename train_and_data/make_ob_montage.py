#!/usr/bin/env python3
"""Visualise the predicted unusable-region (OB/shaded border) mask.

Shows, for a few frames that have a sensor border:
    B (with GT contour) | B (with predicted contour + IoU)

Usage:
    python make_ob_montage.py <data_dir> <model.pt> [out.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import build_model_from_state  # noqa: E402


def stretch(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, (0.5, 99.5))
    return np.clip((img.astype(np.float32) - lo) / max(1e-6, hi - lo), 0.0, 1.0)


def main() -> None:
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    ckpt = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("models/best.pt")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else (
        HERE / "inspect_samples" / "ob_pred_montage.png")

    a = np.load(data / "test_a.npy")
    b = np.load(data / "test_b.npy")
    meta = np.load(data / "test_meta.npz")
    ob_b = meta["ob_b"]                       # (N, grid, grid) uint8

    sd = torch.load(ckpt, map_location="cpu")
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    net = build_model_from_state(sd).cpu().eval()

    have = [i for i in range(len(a)) if ob_b[i].any()]
    picks = (have or list(range(len(a))))[:5]

    with torch.no_grad():
        fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 7))
        for col, i in enumerate(picks):
            x = torch.from_numpy(
                np.stack([stretch(a[i]), stretch(b[i])])).unsqueeze(0)
            _pose, _det, ob = net(x)
            grid = ob_b.shape[-1]
            pred = torch.sigmoid(
                torch.nn.functional.adaptive_avg_pool2d(ob, (grid, grid)))[0, 1].numpy()
            gt = ob_b[i].astype(np.float32)
            pbin = (pred > 0.5).astype(np.float32)
            iou = float((pbin * gt).sum()) / max(1.0, float(((pbin + gt) > 0).sum()))

            bb = stretch(b[i])
            xx = np.arange(grid)
            axes[0, col].imshow(bb, cmap="gray")
            axes[0, col].contour(gt, levels=[0.5], colors="red", linewidths=1.4)
            axes[0, col].set_title(f"B #{i}\nGT border", fontsize=9)
            axes[1, col].imshow(bb, cmap="gray")
            axes[1, col].contour(pbin, levels=[0.5], colors="lime", linewidths=1.4)
            axes[1, col].set_title(f"pred border  IoU {iou:.2f}", fontsize=9)
            for r in range(2):
                axes[r, col].axis("off")

        fig.suptitle("OB / shaded-border prediction "
                     "(red = GT, green = predicted)", fontsize=12)
        plt.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
