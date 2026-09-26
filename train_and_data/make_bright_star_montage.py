#!/usr/bin/env python3
"""Before/after of raising the bright-star flux ceiling (FLUX_MAX).

OLD: flux clipped to 80  -> all stars brighter than mag ~3.2 flatten to the
                           same saturated size.
NEW: flux clipped to 1000 -> bright stars carry more signal, so their
                           saturated disc grows and magnitude ordering returns.

Writes: train_and_data/inspect_samples/bright_star_montage.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

import render_sky_patch as r  # noqa: E402
from render_sky_patch import (  # noqa: E402
    angular_distance_deg, apply_roll, gnomonic_project, load_catalog,
)

OUT = HERE / "inspect_samples" / "bright_star_montage.png"
CATALOG = HERE.parent / "data" / "hip_catalog.csv"


def render(mags, x, y, half, size, sigma, gain, flux_max):
    orig = r.mag_to_flux
    orig_max = r.FLUX_MAX
    r.FLUX_MAX = flux_max
    r.mag_to_flux = lambda m: np.clip(np.power(10.0, -0.4 * (np.asarray(m) - 8.0)),
                                      r.FLUX_MIN, flux_max)
    try:
        return r.render_psf_image(x, y, mags, half, size, size, sigma, gain)[..., 2]
    finally:
        r.mag_to_flux = orig
        r.FLUX_MAX = orig_max


def main() -> None:
    ra_all, dec_all, mag_all = load_catalog(CATALOG)
    ra0, dec0, fov = 83.82, -5.3875, 2.0
    max_mag, sigma, gain, size = 7.0, 1.2, 2.5, 256
    m = mag_all <= max_mag
    ra, dec, mag = ra_all[m], dec_all[m], mag_all[m]
    keep = angular_distance_deg(ra, dec, ra0, dec0) <= fov * 0.75
    ra, dec, mag = ra[keep], dec[keep], mag[keep]
    x, y, vis = gnomonic_project(ra, dec, ra0, dec0)
    x, y, mag = x[vis], y[vis], mag[vis]
    half = math.tan(math.radians(fov / 2.0))
    inside = (np.abs(x) <= half) & (np.abs(y) <= half)
    x, y, mag = x[inside], y[inside], mag[inside]
    xr, yr = apply_roll(x, y, 0.0)

    old = render(mag, xr, yr, half, size, sigma, gain, 80.0)
    new = render(mag, xr, yr, half, size, sigma, gain, 1000.0)

    bi = int(np.argmin(mag))
    xp = (xr[bi] + half) / (2 * half) * (size - 1)
    yp = (half - yr[bi]) / (2 * half) * (size - 1)
    cx, cy = int(round(xp)), int(round(yp))
    c = 24
    box = (slice(max(0, cy - c), cy + c), slice(max(0, cx - c), cx + c))

    test_mags = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    zeros = np.zeros_like(test_mags)
    o_size = [int((render(np.array([mm]), zeros[:1], zeros[:1], 0.5, 256, sigma, gain, 80.0) > 0.9).sum())
              for mm in test_mags]
    n_size = [int((render(np.array([mm]), zeros[:1], zeros[:1], 0.5, 256, sigma, gain, 1000.0) > 0.9).sum())
              for mm in test_mags]

    fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
    ax[0, 0].imshow(old, cmap="gray", vmin=0, vmax=1); ax[0, 0].set_title("OLD (FLUX_MAX=80)")
    ax[0, 1].imshow(new, cmap="gray", vmin=0, vmax=1); ax[0, 1].set_title("NEW (FLUX_MAX=1000)")
    im = ax[0, 2].imshow(np.abs(new - old), cmap="inferno"); ax[0, 2].set_title("|NEW − OLD|")
    fig.colorbar(im, ax=ax[0, 2], fraction=0.046)
    ax[1, 0].imshow(old[box], cmap="gray", vmin=0, vmax=1)
    ax[1, 0].set_title(f"OLD zoom (mag {mag[bi]:.1f})")
    ax[1, 1].imshow(new[box], cmap="gray", vmin=0, vmax=1)
    ax[1, 1].set_title("NEW zoom (same star)")
    ax[1, 2].bar(test_mags - 0.18, o_size, width=0.36, label="FLUX_MAX=80")
    ax[1, 2].bar(test_mags + 0.18, n_size, width=0.36, label="FLUX_MAX=1000")
    ax[1, 2].set_title("saturated pixels vs mag")
    ax[1, 2].set_xlabel("magnitude"); ax[1, 2].set_ylabel("pixels > 0.9"); ax[1, 2].legend()
    for a in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1]):
        a.axis("off")
    for a in (ax[1, 0], ax[1, 1]):
        a.axis("on"); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Raising the bright-star flux ceiling: bigger saturated discs", fontsize=12)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")
    print("saturated px  mag:", list(test_mags.astype(int)))
    print("  FLUX_MAX=80  :", o_size)
    print("  FLUX_MAX=1000:", n_size)


if __name__ == "__main__":
    main()
