#!/usr/bin/env python3
"""Simulated imaging artifacts, applied as post-processing on the rendered sky.

Each effect works on a float RGB image (H, W, 3) in [0, 1] and is deterministic
given (seed, frame): enable/disable via the ``apply_effects`` config dict.

Effects:
    sensor noise   - bias/read noise floor (additive, random per exposure)
    cosmic rays    - short bright CCD/CMOS hit streaks
    meteor         - a bright transient trail crossing the field
    dust           - fixed dark CMOS dust specks (sensor-plane shadows)
    seeing         - atmospheric turbulence (extra Gaussian blur)
    spikes         - diffraction spikes on the brightest stars
    ghost          - faint offset "ghost" reflections of bright stars
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_DUST_SIZE = 0.03  # each speck radius as a fraction of the short image side


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    sigma = max(0.15, float(sigma))
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def blur_image(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable edge-clamped Gaussian blur; works for (H, W) or (H, W, C)."""
    if sigma <= 0.0:
        return image
    kernel = gaussian_kernel1d(sigma)
    radius = len(kernel) // 2
    result = image.astype(np.float32, copy=True)

    pad0 = [(radius, radius), (0, 0)] + [(0, 0)] * (image.ndim - 2)
    padded = np.pad(result, pad0, mode="edge")
    acc = np.zeros_like(result)
    for i, w in enumerate(kernel):
        acc += w * padded[i : i + result.shape[0], :]
    result = acc

    pad1 = [(0, 0), (radius, radius)] + [(0, 0)] * (image.ndim - 2)
    padded = np.pad(result, pad1, mode="edge")
    acc = np.zeros_like(result)
    for i, w in enumerate(kernel):
        acc += w * padded[:, i : i + result.shape[1]]
    result = acc
    return result


def _transient_rng(seed: int, frame: int) -> np.random.RandomState:
    state = (seed * 1000003 + frame * 1103515245) & 0xFFFFFFFF
    return np.random.RandomState(state)


def _fixed_rng(seed: int, salt: int) -> np.random.RandomState:
    return np.random.RandomState((seed * 7919 + salt) & 0xFFFFFFFF)


def _additive_mask_to_rgb(image: np.ndarray, mask: np.ndarray, weights: tuple[float, float, float]) -> None:
    """Add a scalar mask (H, W) to every RGB channel with per-channel weights."""
    image[..., 0] += mask * weights[0]
    image[..., 1] += mask * weights[1]
    image[..., 2] += mask * weights[2]


# ---------------------------------------------------------------------------
# Individual effects
# ---------------------------------------------------------------------------


def add_sensor_noise(image: np.ndarray, sigma: float, bias: float, seed: int, frame: int) -> None:
    """Bias background + random (read) noise. Bias is per-pixel offset too."""
    rng = _transient_rng(seed, frame)
    h, w = image.shape[:2]
    if bias > 0.0:
        noise = rng.normal(bias, sigma, size=(h, w)).astype(np.float32)
    else:
        noise = rng.normal(0.0, sigma, size=(h, w)).astype(np.float32)
    _additive_mask_to_rgb(image, noise, (1.0, 1.0, 1.0))


def add_cosmic_rays(image: np.ndarray, count: int, seed: int, frame: int) -> None:
    """Random short bright streaks typical of CCD/CMOS cosmic-ray hits."""
    rng = _transient_rng(seed * 3 + 1, frame)
    h, w = image.shape[:2]
    lengths = np.array([0, 0, 1, 1, 2, 3, 5, 7], dtype=np.int64)
    for _ in range(count):
        y0 = rng.uniform(0, h - 1)
        x0 = rng.uniform(0, w - 1)
        length = int(rng.choice(lengths))
        amp = float(rng.uniform(0.55, 1.0))
        if length == 0:
            yy = int(round(y0))
            xx = int(round(x0))
            if 0 <= yy < h and 0 <= xx < w:
                image[yy, xx] = np.clip(image[yy, xx] + amp, 0.0, 1.2)
            continue
        angle = rng.uniform(0.0, 2.0 * math.pi)
        sy = math.sin(angle)
        sx = math.cos(angle)
        rows = np.arange(length + 1)
        yy = np.rint(y0 + rows * sy).astype(np.int64)
        xx = np.rint(x0 + rows * sx).astype(np.int64)
        valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        yy = yy[valid]
        xx = xx[valid]
        if len(yy) == 0:
            continue
        decay = (1.0 - rows[: len(yy)] / (length + 1.0)) ** 1.5
        image[yy, xx] += (amp * decay[:, None]).astype(np.float32) * np.array(
            [1.0, 0.92, 1.05], dtype=np.float32
        )[None, :]


def add_meteor_trail(image: np.ndarray, count: int, seed: int, frame: int) -> None:
    """Bright tapered trails across the field (meteor transients)."""
    rng = _transient_rng(seed * 7 + 2, frame)
    h, w = image.shape[:2]
    diag = math.hypot(w, h)
    for _ in range(count):
        p0, p1 = _random_edge_points(rng, w, h)
        seg_x = p1[0] - p0[0]
        seg_y = p1[1] - p0[1]
        if math.hypot(seg_x, seg_y) < 0.4 * diag:
            continue
        mask = np.zeros((h, w), dtype=np.float32)
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        dx = gx - p0[0]
        dy = gy - p0[1]
        denom = seg_x * seg_x + seg_y * seg_y
        t = np.clip((dx * seg_x + dy * seg_y) / denom, 0.0, 1.0)
        px = dx - t * seg_x
        py = dy - t * seg_y
        dist2 = px * px + py * py
        sigma_w = max(0.8, math.hypot(seg_x, seg_y) / 900.0 + 0.6)
        fall = (1.0 - t) ** 1.6
        brightness = float(rng.uniform(0.35, 0.85))
        mask = brightness * fall * np.exp(-0.5 * dist2 / (sigma_w * sigma_w))
        _additive_mask_to_rgb(image, mask.astype(np.float32), (1.0, 1.0, 1.0))


def _random_edge_points(rng: np.random.RandomState, w: int, h: int) -> tuple[tuple[float, float], tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for _ in range(2):
        edge = int(rng.randint(0, 4))
        if edge == 0:
            points.append((rng.uniform(0, w - 1), 0.0))
        elif edge == 1:
            points.append((rng.uniform(0, w - 1), float(h - 1)))
        elif edge == 2:
            points.append((0.0, rng.uniform(0, h - 1)))
        else:
            points.append((float(w - 1), rng.uniform(0, h - 1)))
    return points[0], points[1]


def add_dust_specks(image: np.ndarray, count: int, size_frac: float, seed: int) -> None:
    """Dark CMOS dust donuts fixed in the sensor plane (same each exposure)."""
    rng = _fixed_rng(seed, 5)
    h, w = image.shape[:2]
    short_side = min(w, h)
    for _ in range(count):
        radius = float(short_side * max(0.004, size_frac) * rng.uniform(0.6, 1.5))
        cy = rng.uniform(radius, h - radius)
        cx = rng.uniform(radius, w - radius)
        darken = float(rng.uniform(0.35, 0.6))
        pad = max(2, int(math.ceil(radius * 2.5)))
        y0 = int(max(0, cy - pad))
        y1 = int(min(h, cy + pad))
        x0 = int(max(0, cx - pad))
        x1 = int(min(w, cx + pad))
        gy, gx = np.mgrid[y0:y1, x0:x1]
        dy = gy - cy
        dx = gx - cx
        r2 = (dx * dx + dy * dy) / (radius * radius)
        # dark core with a thin bright rim, then a soft halo edge.
        transmission = 1.0 - darken * np.exp(-r2 * 1.4)
        halo = 0.5 * darken * np.exp(-((r2 - 1.0) * 2.0) ** 2)
        transmission = np.clip(transmission + halo, 0.05, 1.0).astype(np.float32)
        img_block = image[y0:y1, x0:x1]
        img_block[..., 0] *= transmission
        img_block[..., 1] *= transmission
        img_block[..., 2] *= transmission


def add_diffraction_spikes(
    image: np.ndarray,
    stars: dict,
    length_frac: float,
    intensity: float,
    seed: int,
) -> None:
    """Cross spikes on the brightest stars (2 axes: vertical + horizontal)."""
    if stars is None or len(stars.get("x", [])) == 0:
        return
    rng = _fixed_rng(seed, 11)
    h, w = image.shape[:2]
    short_side = min(w, h)
    x = np.asarray(stars["x"], dtype=np.float64)
    y = np.asarray(stars["y"], dtype=np.float64)
    flux = np.asarray(stars["flux"], dtype=np.float64)
    mag = np.asarray(stars["mag"], dtype=np.float64)

    bright = (flux >= 4.0) & (mag <= 7.5)
    if not bright.any():
        return
    order = np.argsort(-flux[bright])[:60]
    xb = x[bright][order]
    yb = y[bright][order]
    fluxb = flux[bright][order]
    rel = np.clip(fluxb / 80.0, 0.0, 1.0)
    length = short_side * max(0.0, length_frac) * (0.35 + 0.65 * rel)
    length = np.clip(length, 3.0, short_side * 0.45)
    angles = [0.0, math.pi / 2.0]  # spikes along sensor axes
    for cxx, cyy, llen, relv in zip(xb, yb, length, rel):
        amp = intensity * (0.12 + 0.55 * math.sqrt(relv))
        for ang in angles:
            ux = math.cos(ang)
            uy = math.sin(ang)
            for sign in (1.0, -1.0):
                n = max(1, int(round(llen)))
                tt = np.arange(1, n + 1, dtype=np.float64)
                px = cxx + sign * tt * ux
                py = cyy + sign * tt * uy
                rows = np.rint(py).astype(np.int64)
                cols = np.rint(px).astype(np.int64)
                # give the line a sub-pixel width
                fall = np.exp(-1.8 * tt / n)
                for off in (-1, 0, 1):
                    rr = rows + off
                    cc = cols
                    valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                    val = (amp * fall)[valid]
                    if len(val):
                        image[rr[valid], cc[valid]] += val[:, None] * np.array(
                            [1.0, 0.98, 1.0], dtype=np.float32
                        )[None, :]
        # a tiny cross-jitter flavor is ignored; sample a value to keep rng stable
        _ = rng.rand()


def add_ghost_reflections(
    image: np.ndarray,
    stars: dict,
    intensity: float,
    seed: int,
) -> None:
    """Faint out-of-focus ghosts of the brightest stars, mirrored about center."""
    if stars is None or len(stars.get("x", [])) == 0 or intensity <= 0.0:
        return
    h, w = image.shape[:2]
    x = np.asarray(stars["x"], dtype=np.float64)
    y = np.asarray(stars["y"], dtype=np.float64)
    flux = np.asarray(stars["flux"], dtype=np.float64)
    bright = flux >= 10.0
    if not bright.any():
        return
    order = np.argsort(-flux[bright])[:5]
    cx, cy = w / 2.0, h / 2.0
    sigma = max(2.0, min(w, h) * 0.06)
    for idx in order:
        star_x = float(x[bright][idx])
        star_y = float(y[bright][idx])
        gx = 2.0 * cx - star_x
        gy = 2.0 * cy - star_y
        rel = float(flux[bright][idx]) / 80.0
        amp = intensity * 0.12 * (0.3 + 0.7 * math.sqrt(rel))
        add_gaussian_blob(image, gx, gy, sigma * (0.8 + 0.5 * rel), amp)


def add_gaussian_blob(
    image: np.ndarray, cx: float, cy: float, sigma: float, amplitude: float
) -> None:
    h, w = image.shape[:2]
    if amplitude <= 0.0:
        return
    pad = max(2, int(math.ceil(4.0 * sigma)))
    x0 = int(max(0, cx - pad))
    x1 = int(min(w, cx + pad))
    y0 = int(max(0, cy - pad))
    y1 = int(min(h, cy + pad))
    if x1 <= x0 or y1 <= y0:
        return
    gy, gx = np.mgrid[y0:y1, x0:x1]
    r2 = ((gx - cx) ** 2 + (gy - cy) ** 2) / (2.0 * sigma * sigma)
    blob = (amplitude * np.exp(-r2)).astype(np.float32)
    _additive_mask_to_rgb(image[y0:y1, x0:x1], blob, (0.9, 0.95, 1.0))


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def apply_effects(
    image: np.ndarray,
    stars: dict | None,
    art: dict,
) -> np.ndarray:
    """Apply the enabled artifact simulations to a float RGB image.

    ``image`` is (H, W, 3) float32 in ~[0, 1]; returns a new array.
    ``stars`` maps to arrays x/y (pixels), flux, mag.
    ``art`` is a flat config dict with ``enabled_*`` / parameter keys plus
    ``seed`` and ``frame``.
    """
    out = np.clip(image, 0.0, 1.0).astype(np.float32, copy=True)
    seed = int(art.get("seed", 0))
    frame = int(art.get("frame", 0))

    # --- optical / atmosphere (before detector effects) ---
    if art.get("spike", False):
        add_diffraction_spikes(
            out,
            stars,
            float(art.get("spike_len", 0.1)),
            float(art.get("spike_int", 1.0)),
            seed,
        )
    if art.get("ghost", False):
        add_ghost_reflections(out, stars, float(art.get("ghost_int", 0.5)), seed)

    seeing_sigma = float(art.get("seeing_sigma", 0.0))
    if art.get("seeing", False) and seeing_sigma > 0.0:
        out = blur_image(out, seeing_sigma)

    # --- sensor-plane shadows ---
    if art.get("dust", False):
        add_dust_specks(
            out,
            int(art.get("dust_count", 6)),
            float(art.get("dust_size", DEFAULT_DUST_SIZE)),
            seed,
        )

    # --- transient detector events ---
    if art.get("meteor", False):
        add_meteor_trail(out, int(art.get("meteor_count", 1)), seed, frame)
    if art.get("cr", False):
        add_cosmic_rays(out, int(art.get("cr_count", 10)), seed, frame)
    if art.get("noise", False):
        add_sensor_noise(
            out,
            float(art.get("noise_sigma", 0.02)),
            float(art.get("noise_bias", 0.05)),
            seed,
            frame,
        )

    return np.clip(out, 0.0, 1.0)
