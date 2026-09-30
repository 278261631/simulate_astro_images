#!/usr/bin/env python3
"""Run the single-image OB/unusable-region segmenter on one frame.

Usage:
    python predict_ob.py models_ob/best.pt frame.npy
    python predict_ob.py models_ob/best.pt frame.png --save mask.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from model_ob import OBNet  # noqa: E402


def load_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        im = np.load(path)
        if im.ndim == 3:
            im = im.mean(2)
        return im.astype(np.uint8)
    from PIL import Image
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.uint8)


def to_model_size(img: np.ndarray, size: int = 256) -> np.ndarray:
    if img.shape == (size, size):
        return img
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("image", type=Path)
    ap.add_argument("--save", type=Path, default=None, help="save the mask PNG")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    obj = torch.load(args.checkpoint, map_location="cpu")
    base = int(obj.get("base", 32)) if isinstance(obj, dict) else 32
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    net = OBNet(base=base).eval()
    net.load_state_dict(sd)

    img = to_model_size(load_gray(args.image))
    x = torch.from_numpy(img.astype(np.float32)[None, None] / 255.0)
    with torch.no_grad():
        prob = torch.sigmoid(net(x))[0, 0].numpy()
    mask = prob > args.thresh
    print(f"{args.image}: unusable region {100*mask.mean():.1f}% of cells "
          f"({mask.shape[0]}x{mask.shape[1]})")
    if args.save is not None:
        from PIL import Image
        Image.fromarray((mask * 255).astype(np.uint8)).save(args.save)
        print("saved", args.save)


if __name__ == "__main__":
    main()
