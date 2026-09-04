#!/usr/bin/env python3
"""Render a sky patch image from HIP catalog data.

Usage example:
    python render_sky_patch.py --ra 83.63 --dec 22.01 --fov 10 --out m42.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_catalog = script_dir.parent / "data" / "hip_catalog.csv"

    parser = argparse.ArgumentParser(
        description="Generate a sky patch image from HIP catalog."
    )
    parser.add_argument(
        "--ra",
        type=float,
        default=83.82,
        help="Center RA in degrees (default: 83.82, M42).",
    )
    parser.add_argument(
        "--dec",
        type=float,
        default=-5.3875,
        help="Center Dec in degrees (default: -5.3875, M42).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=30.0,
        help="Field of view in degrees (image width/height, default: 30.0).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output image file path (e.g. output.png).",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=default_catalog,
        help=f"HIP catalog CSV path (default: {default_catalog}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI (default: 200).",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=6.0,
        help="Output image size in inches (default: 6.0).",
    )
    parser.add_argument(
        "--max-mag",
        type=float,
        default=10.0,
        help="Render stars with magnitude <= this value (default: 10.0).",
    )
    parser.add_argument(
        "--roll",
        type=float,
        default=0.0,
        help="Camera roll angle in degrees, positive is clockwise (default: 0.0).",
    )
    parser.add_argument(
        "--psf-sigma",
        type=float,
        default=1.2,
        help="Gaussian PSF sigma in pixels (default: 1.2).",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=2.0,
        help="Image gain for tone mapping (default: 2.0).",
    )
    return parser.parse_args()


def load_catalog(catalog_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    data = np.genfromtxt(
        catalog_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    return (
        np.asarray(data["ra"], dtype=np.float64),
        np.asarray(data["dec"], dtype=np.float64),
        np.asarray(data["mag"], dtype=np.float64),
    )


def angular_distance_deg(
    ra_deg: np.ndarray, dec_deg: np.ndarray, ra0_deg: float, dec0_deg: float
) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)

    cos_d = (
        np.sin(dec) * math.sin(dec0)
        + np.cos(dec) * math.cos(dec0) * np.cos(ra - ra0)
    )
    cos_d = np.clip(cos_d, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_d))


def gnomonic_project(
    ra_deg: np.ndarray, dec_deg: np.ndarray, ra0_deg: float, dec0_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)

    delta_ra = ra - ra0
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(delta_ra)
    visible = cosc > 0

    x = np.cos(dec) * np.sin(delta_ra) / cosc
    y = (
        np.cos(dec0) * np.sin(dec)
        - np.sin(dec0) * np.cos(dec) * np.cos(delta_ra)
    ) / cosc

    return x, y, visible


def mag_to_flux(mag: np.ndarray) -> np.ndarray:
    # Relative flux scale for point source rendering.
    flux = np.power(10.0, -0.4 * (mag - 8.0))
    return np.clip(flux, 0.03, 80.0)


def apply_roll(x: np.ndarray, y: np.ndarray, roll_deg: float) -> tuple[np.ndarray, np.ndarray]:
    if roll_deg == 0.0:
        return x, y

    # Positive roll is defined as clockwise on image plane.
    theta = -math.radians(roll_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    xr = x * cos_t - y * sin_t
    yr = x * sin_t + y * cos_t
    return xr, yr


def format_value(v: float) -> str:
    return f"{v:.3f}".replace("-", "m").replace(".", "p")


def build_output_path(base_out: Path, ra: float, dec: float, fov: float, roll: float) -> Path:
    suffix = (
        f"_ra{format_value(ra)}"
        f"_dec{format_value(dec)}"
        f"_fov{format_value(fov)}"
        f"_roll{format_value(roll)}"
    )
    ext = base_out.suffix if base_out.suffix else ".png"
    stem = base_out.stem if base_out.stem else "sky"
    return base_out.with_name(f"{stem}{suffix}{ext}")


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    sigma = max(0.1, float(sigma))
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def convolve_along_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis, image)


def render_psf_image(
    x: np.ndarray,
    y: np.ndarray,
    mag: np.ndarray,
    half_extent: float,
    width_px: int,
    height_px: int,
    psf_sigma: float,
    gain: float,
) -> np.ndarray:
    star_field = np.zeros((height_px, width_px), dtype=np.float32)
    flux = mag_to_flux(mag).astype(np.float32)

    xp = (x + half_extent) / (2.0 * half_extent) * (width_px - 1)
    yp = (half_extent - y) / (2.0 * half_extent) * (height_px - 1)
    xi = np.rint(xp).astype(np.int32)
    yi = np.rint(yp).astype(np.int32)

    valid = (xi >= 0) & (xi < width_px) & (yi >= 0) & (yi < height_px)
    xi = xi[valid]
    yi = yi[valid]
    flux = flux[valid]
    np.add.at(star_field, (yi, xi), flux)

    kernel = gaussian_kernel1d(psf_sigma)
    blurred = convolve_along_axis(star_field, kernel, axis=1)
    blurred = convolve_along_axis(blurred, kernel, axis=0)

    # Mix a small sharp core with blurred halo to keep stars point-like.
    core = np.sqrt(np.clip(star_field, 0.0, None))
    signal = np.clip(blurred + 0.12 * core, 0.0, None)
    luminance = 1.0 - np.exp(-gain * signal)
    luminance = np.clip(luminance, 0.0, 1.0)

    rgb = np.zeros((height_px, width_px, 3), dtype=np.float32)
    rgb[..., 0] = luminance * 0.95
    rgb[..., 1] = luminance * 0.97
    rgb[..., 2] = luminance
    return rgb


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.ra < 360.0):
        raise ValueError("--ra must be in [0, 360).")
    if not (-90.0 <= args.dec <= 90.0):
        raise ValueError("--dec must be in [-90, 90].")
    if not (0.0 < args.fov <= 170.0):
        raise ValueError("--fov must be in (0, 170].")
    if args.psf_sigma <= 0.0:
        raise ValueError("--psf-sigma must be > 0.")
    if args.gain <= 0.0:
        raise ValueError("--gain must be > 0.")

    ra, dec, mag = load_catalog(args.catalog.resolve())

    mask_mag = mag <= args.max_mag
    ra, dec, mag = ra[mask_mag], dec[mask_mag], mag[mask_mag]

    # Pre-filter by angular radius to keep plotting fast.
    radius = args.fov * math.sqrt(2.0) * 0.5
    dist = angular_distance_deg(ra, dec, args.ra, args.dec)
    mask_radius = dist <= radius
    ra, dec, mag = ra[mask_radius], dec[mask_radius], mag[mask_radius]

    x, y, visible = gnomonic_project(ra, dec, args.ra, args.dec)
    x, y, mag = x[visible], y[visible], mag[visible]
    x, y = apply_roll(x, y, args.roll)

    half_extent = math.tan(math.radians(args.fov / 2.0))
    in_frame = (np.abs(x) <= half_extent) & (np.abs(y) <= half_extent)
    x, y, mag = x[in_frame], y[in_frame], mag[in_frame]

    width_px = max(64, int(round(args.size * args.dpi)))
    height_px = max(64, int(round(args.size * args.dpi)))
    image = render_psf_image(
        x=x,
        y=y,
        mag=mag,
        half_extent=half_extent,
        width_px=width_px,
        height_px=height_px,
        psf_sigma=args.psf_sigma,
        gain=args.gain,
    )

    out_path = build_output_path(args.out, args.ra, args.dec, args.fov, args.roll)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415  (lazy: keep module import-light for the GUI)

    plt.imsave(out_path, image)

    print(f"Saved image: {out_path.resolve()}")
    print(f"Stars rendered: {len(mag)}")


if __name__ == "__main__":
    main()
