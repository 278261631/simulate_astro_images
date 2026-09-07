#!/usr/bin/env python3
"""Run the trained model on one (A, B) pair -> dx, dy, droll.

Inputs can be single-channel uint8 numpy arrays (``.npy``), 8/16-bit or float
grayscale images (via OpenCV), or the raw PNG/BMP saved by the GUI. Both are
normalised to the model's native size by area interpolation; predicted pixel
offsets refer to that normalised grid and are converted back to the original
image pixel scale via the scale factor s = model_size / original_width.

Usage:
    python predict.py model.pt frame_a.png frame_b.png
    python predict.py --checkpoint models/best.pt --a ../python/test_outputs/x.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import PairRegNet, decode, preprocess  # noqa: E402

MODEL_SIZE = 192


def load_gray(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        im = np.load(path)
        if im.ndim == 3:
            im = im.mean(axis=2)
        return im.astype(np.uint8)
    import cv2  # local import keeps training/generation light

    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.dtype != np.uint8:
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return im


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", nargs="?", type=Path,
                   default=Path(__file__).resolve().parent / "models" / "best.pt")
    p.add_argument("a", nargs="?", type=Path)
    p.add_argument("b", nargs="?", type=Path)
    args = p.parse_args()
    if args.a is None or args.b is None:
        p.error("provide --checkpoint model and two image paths")

    model = PairRegNet()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    ia = load_gray(args.a)
    ib = load_gray(args.b)
    if ia.shape != ib.shape:
        print(f"warning: shape mismatch A {ia.shape} vs B {ib.shape}", file=sys.stderr)

    # resize both to the model input size, keeping aspect distortion minimal
    import cv2

    h, w = ia.shape
    s = MODEL_SIZE / max(w, h)  # scale so max side == model size
    new_w, new_h = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if (new_w, new_h) != (w, h):
        ia = cv2.resize(ia, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ib = cv2.resize(ib, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # pad to square MODEL_SIZE
    ph, pw = MODEL_SIZE - ia.shape[0], MODEL_SIZE - ia.shape[1]
    pad = ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2))
    ia = np.pad(ia, pad, mode="edge")
    ib = np.pad(ib, pad, mode="edge")

    pair = preprocess(ia, ib).unsqueeze(0)  # (1, 2, H, W)
    with torch.no_grad():
        dx, dy, roll = decode(model(pair))[0].tolist()
    dx /= s
    dy /= s
    print(f"A center in B frame:  dx {dx:+.2f} px   dy {dy:+.2f} px   droll {roll:+.2f}\u00b0")
    print(f"shift B by {dx:+.2f}, {dy:+.2f} px and {-roll:+.2f}\u00b0 to align B onto A")


if __name__ == "__main__":
    main()
