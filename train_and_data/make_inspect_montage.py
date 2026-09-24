#!/usr/bin/env python3
"""Build a visual inspection montage of appear-transient samples.

Reads the small dataset in train_and_data/inspect_data/ and writes:
    train_and_data/inspect_samples/appear_montage.png
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

DATA_DIR = Path(__file__).resolve().parent / "inspect_data"
OUT_DIR = Path(__file__).resolve().parent / "inspect_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def stretch(img: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(img, (lo_pct, hi_pct))
    x = (img.astype(np.float32) - lo) / max(1e-6, hi - lo)
    return np.clip(x, 0.0, 1.0)


def annotate(img: np.ndarray, x: float, y: float, radius: int = 6) -> np.ndarray:
    """Draw a red circle around the transient on a stretched RGB image."""
    rgb = (img * 255).astype(np.uint8)
    rgb = np.stack([rgb, rgb, rgb], axis=-1)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    ix, iy = int(round(x)), int(round(y))
    draw.ellipse([ix - radius, iy - radius, ix + radius, iy + radius],
                 outline=(255, 40, 40), width=2)
    # crosshair
    draw.line([(ix - radius - 3, iy), (ix - radius, iy)], fill=(255, 40, 40), width=1)
    draw.line([(ix + radius, iy), (ix + radius + 3, iy)], fill=(255, 40, 40), width=1)
    draw.line([(ix, iy - radius - 3), (ix, iy - radius)], fill=(255, 40, 40), width=1)
    draw.line([(ix, iy + radius), (ix, iy + radius + 3)], fill=(255, 40, 40), width=1)
    return np.asarray(pil, dtype=np.float32) / 255.0


def main() -> None:
    a = np.load(DATA_DIR / "train_a.npy")
    b = np.load(DATA_DIR / "train_b.npy")
    meta = np.load(DATA_DIR / "train_meta.npz")

    trans_n = meta["trans_n"]
    trans_x = meta["trans_x"]
    trans_y = meta["trans_y"]
    trans_cls = meta["trans_cls"]

    # collect all appear samples with their B-frame pixel value
    records = []
    for i in range(len(a)):
        for j in range(int(trans_n[i])):
            if int(round(trans_cls[i, j])) == 0:  # appear
                x, y = trans_x[i, j], trans_y[i, j]
                if x < 0 or y < 0:
                    continue
                # sample a small aperture to estimate brightness
                ix, iy = int(round(x)), int(round(y))
                r = 2
                patch = b[i][max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1]
                brightness = float(patch.mean())
                records.append({
                    "idx": i,
                    "x": x,
                    "y": y,
                    "brightness": brightness,
                })

    if not records:
        raise RuntimeError("No appear transients found in the dataset")

    records.sort(key=lambda r: r["brightness"])
    n = len(records)
    # pick representative samples across the brightness range
    picks = [
        records[0],                       # dimmest
        records[n // 5],                  # dim
        records[n // 2],                  # medium
        records[3 * n // 4],              # bright
        records[-2],                      # very bright
        records[-1],                      # brightest
    ]

    cols = 3  # A, B annotated, B stretched
    rows = len(picks)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.8 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for row, rec in enumerate(picks):
        i, x, y, br = rec["idx"], rec["x"], rec["y"], rec["brightness"]
        a_img = stretch(a[i])
        b_img = stretch(b[i])
        b_annotated = annotate(b_img, x, y)
        b_stretched = stretch(b[i], lo_pct=0.0, hi_pct=99.9)
        b_stretch_annotated = annotate(b_stretched, x, y)

        axes[row, 0].imshow(a_img, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"A  (#{i})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(b_annotated, vmin=0, vmax=1)
        axes[row, 1].set_title(f"B  appear marked  (raw br={br:.1f})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(b_stretch_annotated, vmin=0, vmax=1)
        axes[row, 2].set_title(f"B  stretched")
        axes[row, 2].axis("off")

    fig.suptitle("Appear transients across the new 0-9 mag range "
                 "(top=dimmest, bottom=brightest)", fontsize=12)
    plt.tight_layout()
    out_path = OUT_DIR / "appear_montage.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved montage: {out_path}")
    print(f"Found {n} appear samples in {len(a)} pairs")


if __name__ == "__main__":
    main()
