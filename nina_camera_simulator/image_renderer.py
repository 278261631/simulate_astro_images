#!/usr/bin/env python3
"""Render sky images for the simulated Alpaca camera.

Reuses the projection / PSF kernels from ../python/render_sky_patch.py so the
two implementations stay consistent.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from render_sky_patch import (  # noqa: E402
    angular_distance_deg,
    apply_roll,
    convolve_along_axis,
    gaussian_kernel1d,
    gnomonic_project,
    load_catalog,
    mag_to_flux,
)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required (pip install Pillow)") from exc


_CATALOG_CACHE: dict = {}


def resolve_catalog(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return p


def get_catalog(render_cfg: dict):
    """Load (ra, dec, mag) arrays once per (path, max_mag) and cache them."""
    cat_path = resolve_catalog(str(render_cfg.get("catalog", "../data/hip_catalog.csv")))
    max_mag = float(render_cfg.get("max_mag", 12.0))
    key = (str(cat_path), max_mag)
    if key not in _CATALOG_CACHE:
        if not cat_path.exists():
            raise FileNotFoundError(f"Catalog not found: {cat_path}")
        ra, dec, mag = load_catalog(cat_path)
        keep = mag <= max_mag
        _CATALOG_CACHE[key] = (ra[keep], dec[keep], mag[keep])
    return _CATALOG_CACHE[key]


def compute_fov_deg(
    numx: int, numy: int, pixel_size_um: float, focal_length_mm: float
) -> tuple[float, float]:
    """Sensor-based field of view: fov = 2 * atan(sensor_half / focal_length)."""
    fov_x = 2.0 * math.degrees(math.atan2(numx * pixel_size_um / 2000.0, focal_length_mm))
    fov_y = 2.0 * math.degrees(math.atan2(numy * pixel_size_um / 2000.0, focal_length_mm))
    return fov_x, fov_y


def focus_to_psf_sigma(
    focus: int, ideal: int, span: int, sigma_min: float, sigma_max: float
) -> float:
    """Map focuser position to a Gaussian PSF sigma (pixels).

    Out-of-focus (far from `ideal`) stars grow quadratically blurred up to
    sigma_max; at ideal focus the image is sharpest (sigma_min).
    """
    t = abs(focus - ideal) / max(1, span)
    t = min(t, 1.0)
    return sigma_min + t * t * (sigma_max - sigma_min)


def render_luminance(
    ra_cat: np.ndarray,
    dec_cat: np.ndarray,
    mag_cat: np.ndarray,
    ra0_deg: float,
    dec0_deg: float,
    fov_x_deg: float,
    fov_y_deg: float,
    width_px: int,
    height_px: int,
    psf_sigma: float,
    tone_gain: float,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """Project catalog stars into the frame and render a PSF-convolved sky.

    Returns a float luminance image with values in [0, 1].
    """
    radius = max(fov_x_deg, fov_y_deg) * math.sqrt(2.0) * 0.5
    dist = angular_distance_deg(ra_cat, dec_cat, ra0_deg, dec0_deg)
    mask = dist <= radius
    ra = ra_cat[mask]
    dec = dec_cat[mask]
    mag = mag_cat[mask]

    x, y, visible = gnomonic_project(ra, dec, ra0_deg, dec0_deg)
    x, y, mag = x[visible], y[visible], mag[visible]
    x, y = apply_roll(x, y, roll_deg)

    hx = math.tan(math.radians(fov_x_deg / 2.0))
    hy = math.tan(math.radians(fov_y_deg / 2.0))
    in_frame = (np.abs(x) <= hx) & (np.abs(y) <= hy)
    x, y, mag = x[in_frame], y[in_frame], mag[in_frame]

    star_field = np.zeros((height_px, width_px), dtype=np.float32)
    flux = mag_to_flux(mag).astype(np.float32)

    xp = (x + hx) / (2.0 * hx) * (width_px - 1)
    yp = (hy - y) / (2.0 * hy) * (height_px - 1)
    xi = np.rint(xp).astype(np.int32)
    yi = np.rint(yp).astype(np.int32)
    valid = (xi >= 0) & (xi < width_px) & (yi >= 0) & (yi < height_px)
    xi, yi, flux = xi[valid], yi[valid], flux[valid]
    np.add.at(star_field, (yi, xi), flux)

    kernel = gaussian_kernel1d(psf_sigma)
    blurred = convolve_along_axis(star_field, kernel, 1)
    blurred = convolve_along_axis(blurred, kernel, 0)

    core = np.sqrt(np.clip(star_field, 0.0, None))
    signal = np.clip(blurred + 0.12 * core, 0.0, None)
    lum = 1.0 - np.exp(-tone_gain * signal)
    return np.clip(lum, 0.0, 1.0)


def to_raw16(
    lum: np.ndarray,
    brightness: float = 40000.0,
    bias: float = 1200.0,
    noise_sigma: float = 40.0,
    vignetting: float = 0.35,
    simulate_noise: bool = False,
    seed=None,
) -> np.ndarray:
    """Turn a [0,1] luminance image into a 16-bit sensor frame.

    Applies read noise and a few hot pixels when simulate_noise is enabled,
    plus optional radial vignetting.
    """
    height, width = lum.shape
    img = lum.astype(np.float64) * brightness + bias

    if vignetting > 0.0:
        yy, xx = np.mgrid[0:height, 0:width]
        cx = max(1.0, (width - 1) / 2.0)
        cy = max(1.0, (height - 1) / 2.0)
        r2 = ((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2
        img *= 1.0 - vignetting * np.clip(r2, 0.0, 1.0)

    if simulate_noise:
        rng = np.random.default_rng(seed)
        img += rng.normal(0.0, noise_sigma, size=img.shape)
        n_hot = max(1, int(img.size * 0.00004))
        ys = rng.integers(0, height, size=n_hot)
        xs = rng.integers(0, width, size=n_hot)
        img[ys, xs] = 60000.0

    img = np.clip(np.rint(img), 0.0, 65535.0)
    return img.astype(np.uint16)


def _load_font(size: int):
    candidates = ("arial.ttf", "segoeui.ttf", "calibri.ttf", "DejaVuSans.ttf")
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap(text: str, width_px: int, font) -> list[str]:
    if not text:
        return []
    approx = max(1, int(width_px / max(1.0, 0.62 * font.size)))
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= approx:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:6]


def render_error_image(
    width_px: int, height_px: int, message: str, detail: str = "", background: int = 2000
) -> np.ndarray:
    """A mostly-blank 16-bit frame with a bright error text overlay.

    Used when NINA data cannot be read, so NINA still downloads a frame and the
    problem is visible on the image itself.
    """
    width = max(64, int(width_px))
    height = max(64, int(height_px))
    base_8bit = max(0, (background * 255 + 32767) // 65535)

    base = Image.new("L", (width, height), color=base_8bit)
    draw = ImageDraw.Draw(base)

    font_main = _load_font(max(20, width // 30))
    font_detail = _load_font(max(12, width // 52))
    lines = _wrap(detail, width, font_detail) if detail else []

    center_y = height / 2.0 - len(lines) * (height // 26) / 2.0
    draw.text((width / 2.0, center_y), message, fill=255, font=font_main, anchor="mm")
    for i, line in enumerate(lines):
        yy = height / 2.0 + (i + 1) * (height // 24)
        draw.text((width / 2.0, yy), line, fill=200, font=font_detail, anchor="mm")

    arr = np.asarray(base, dtype=np.uint16)
    return (arr * 250).astype(np.uint16)
