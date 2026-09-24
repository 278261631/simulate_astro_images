#!/usr/bin/env python3
"""Visual comparison of the sensor-framing effects (crop model).

The frame is a crop of a larger sensor, so it can touch the physical border on
at most two *adjacent* sides.  This script renders one star field and shows:

    interior crop | one border | two adjacent borders (corner) | A vs B (independent)

Writes: train_and_data/inspect_samples/cmos_framing_montage.png
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
    gnomonic_project,
    load_catalog,
    render_psf_image,
)
import sky_effects  # noqa: E402

OUT_DIR = HERE / "inspect_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATALOG = HERE.parent / "data" / "hip_catalog.csv"


def base_render(size: int = 320) -> np.ndarray:
    ra_all, dec_all, mag_all = load_catalog(CATALOG)
    ra0, dec0, fov, roll = 83.82, -5.3875, 8.0, 0.0
    max_mag = 11.5
    mask = mag_all <= max_mag
    ra, dec, mag = ra_all[mask], dec_all[mask], mag_all[mask]
    radius = fov * math.sqrt(2.0) * 0.5
    keep = angular_distance_deg(ra, dec, ra0, dec0) <= radius
    ra, dec, mag = ra[keep], dec[keep], mag[keep]
    x, y, vis = gnomonic_project(ra, dec, ra0, dec0)
    x, y, mag = x[vis], y[vis], mag[vis]
    x, y = apply_roll(x, y, roll)
    half = math.tan(math.radians(fov / 2.0))
    inside = (np.abs(x) <= half) & (np.abs(y) <= half)
    x, y, mag = x[inside], y[inside], mag[inside]
    return render_psf_image(x=x, y=y, mag=mag, half_extent=half,
                            width_px=size, height_px=size,
                            psf_sigma=1.2, gain=2.5)


def stretch(img: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(img, (lo_pct, hi_pct))
    return np.clip((img - lo) / max(1e-6, hi - lo), 0.0, 1.0)


def noise(img: np.ndarray, seed: int, frame: int) -> np.ndarray:
    out = img.copy()
    sky_effects.add_sensor_noise(out, 0.02, 0.05, seed=seed, frame=frame)
    return out


def main() -> None:
    base = base_render(320)

    # interior crop: no sensor border touched
    interior = noise(base, 7, 0)

    # one border: left side optical black + top shading
    one = noise(base, 7, 1)
    sky_effects.apply_sensor_shading(one, top=70, drop=0.35)
    sky_effects.apply_optical_black(one, left=48, level=0.02, noise_sigma=0.006,
                                    seed=7, frame=1)

    # corner crop: two adjacent borders (top + left)
    corner = noise(base, 7, 2)
    sky_effects.apply_sensor_shading(corner, top=64, left=64, drop=0.4)
    sky_effects.apply_optical_black(corner, top=40, left=32, level=0.02,
                                    noise_sigma=0.006, seed=7, frame=2)

    # independent A/B (different cameras, different borders)
    a = noise(base, 7, 10)
    sky_effects.apply_sensor_shading(a, bottom=80, drop=0.3)
    sky_effects.apply_optical_black(a, right=40, level=0.02, noise_sigma=0.006,
                                    seed=7, frame=10)
    b = noise(base, 7, 11)
    sky_effects.apply_sensor_shading(b, top=56, left=56, drop=0.4)
    sky_effects.apply_optical_black(b, top=36, left=28, level=0.02,
                                    noise_sigma=0.006, seed=7, frame=11)

    panels = [
        ("interior crop (no border)", interior),
        ("one border (left OB, top shading)", one),
        ("corner crop (top+left)", corner),
        ("A: right OB, bottom shading", a),
        ("B: top+left, independent", b),
    ]

    fig, axes = plt.subplots(2, len(panels), figsize=(4 * len(panels), 8.5))

    ref = panels[0][1]
    lo, hi = np.percentile(ref, (0.5, 99.5))
    common = lambda im: np.clip((im - lo) / max(1e-6, hi - lo), 0.0, 1.0)  # noqa: E731

    for col, (title, img) in enumerate(panels):
        axes[0, col].imshow(common(img), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].axis("off")
        axes[1, col].imshow(stretch(img, 0.0, 99.9), cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("strong stretch", fontsize=10)
        axes[1, col].axis("off")

    fig.suptitle("Sensor framing (crop model): optical black + edge shading, "
                 "independent per exposure", fontsize=13)
    plt.tight_layout()
    out = OUT_DIR / "cmos_framing_montage.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved montage: {out}")


if __name__ == "__main__":
    main()
