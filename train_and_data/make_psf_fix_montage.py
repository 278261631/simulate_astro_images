#!/usr/bin/env python3
"""Before/after of the faint-star PSF fix in python/render_sky_patch.py.

The old renderer mixed an *unblurred* sqrt(flux) core into the image, which
dominates at low flux and made faint stars single, sub-PSF, dim pixels.  The
fix convolves the core with a narrower Gaussian (peak-normalised), so faint
stars keep their brightness but get a proper PSF profile.

Writes: train_and_data/inspect_samples/psf_fix_montage.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

from render_sky_patch import (  # noqa: E402
    angular_distance_deg,
    apply_roll,
    gaussian_kernel1d,
    convolve_along_axis,
    gnomonic_project,
    load_catalog,
    render_psf_image,
)

OUT = HERE / "inspect_samples" / "psf_fix_montage.png"
CATALOG = HERE.parent / "data" / "hip_catalog.csv"


def render_old(x, y, mag, half_extent, width_px, height_px, psf_sigma, gain):
    """Previous behaviour: unblurred sqrt(flux) core."""
    star_field = np.zeros((height_px, width_px), dtype=np.float32)
    flux = np.power(10.0, -0.4 * (mag - 8.0)).astype(np.float32)
    flux = np.clip(flux, 0.03, 80.0)
    xp = (x + half_extent) / (2.0 * half_extent) * (width_px - 1)
    yp = (half_extent - y) / (2.0 * half_extent) * (height_px - 1)
    xi = np.rint(xp).astype(np.int32)
    yi = np.rint(yp).astype(np.int32)
    v = (xi >= 0) & (xi < width_px) & (yi >= 0) & (yi < height_px)
    np.add.at(star_field, (yi[v], xi[v]), flux[v])
    k = gaussian_kernel1d(psf_sigma)
    blurred = convolve_along_axis(convolve_along_axis(star_field, k, 1), k, 0)
    core = np.sqrt(np.clip(star_field, 0.0, None))
    signal = np.clip(blurred + 0.12 * core, 0.0, None)
    lum = np.clip(1.0 - np.exp(-gain * signal), 0.0, 1.0)
    rgb = np.zeros((height_px, width_px, 3), dtype=np.float32)
    rgb[..., 0] = lum * 0.95
    rgb[..., 1] = lum * 0.97
    rgb[..., 2] = lum
    return rgb


def main() -> None:
    ra_all, dec_all, mag_all = load_catalog(CATALOG)
    ra0, dec0, fov = 83.82, -5.3875, 2.5
    max_mag, sigma, gain, size = 12.0, 1.2, 2.5, 256
    m = mag_all <= max_mag
    ra, dec, mag = ra_all[m], dec_all[m], mag_all[m]
    keep = angular_distance_deg(ra, dec, ra0, dec0) <= fov * 0.75
    ra, dec, mag = ra[keep], dec[keep], mag[keep]
    x, y, vis = gnomonic_project(ra, dec, ra0, dec0)
    x, y, mag = x[vis], y[vis], mag[vis]
    half = math.tan(math.radians(fov / 2.0))
    inside = (np.abs(x) <= half) & (np.abs(y) <= half)
    x, y, mag = x[inside], y[inside], mag[inside]

    old = render_old(x, y, mag, half, size, size, sigma, gain)[..., 2]
    new = render_psf_image(x, y, mag, half, size, size, sigma, gain)[..., 2]

    # a faint crop: window around the dimmest star
    dim_i = int(np.argmax(mag))
    xp = (x[dim_i] + half) / (2 * half) * (size - 1)
    yp = (half - y[dim_i]) / (2 * half) * (size - 1)
    cx, cy = int(round(xp)), int(round(yp))
    c = 20
    box = (slice(max(0, cy - c), cy + c), slice(max(0, cx - c), cx + c))

    fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
    ax[0, 0].imshow(old, cmap="gray", vmin=0, vmax=1); ax[0, 0].set_title("OLD full field")
    ax[0, 1].imshow(new, cmap="gray", vmin=0, vmax=1); ax[0, 1].set_title("NEW full field")
    im = ax[0, 2].imshow(np.abs(new - old), cmap="inferno"); ax[0, 2].set_title("|NEW − OLD|")
    fig.colorbar(im, ax=ax[0, 2], fraction=0.046)
    ax[1, 0].imshow(old[box], cmap="gray", vmin=0, vmax=max(0.3, old[box].max()))
    ax[1, 0].set_title(f"OLD zoom (mag {mag[dim_i]:.1f} star)")
    ax[1, 1].imshow(new[box], cmap="gray", vmin=0, vmax=max(0.3, new[box].max()))
    ax[1, 1].set_title("NEW zoom (same star)")
    for a in ax.ravel():
        a.axis("off")
    for a in ax[1, :2]:
        a.axis("on"); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Faint-star PSF fix: same peak brightness, proper PSF-shaped profile", fontsize=12)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
