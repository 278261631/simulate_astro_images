#!/usr/bin/env python3
"""Headless batch generator of (A, B, dx/dy/droll) training pairs.

For every pair it renders two exposures through the exact same pipeline the GUI
uses (``python/render_sky_patch.py`` projection/PSF + ``python/sky_effects.py``
artifacts):

    frame A - "commanded" pointing (ra, dec, roll) at an arbitrary sky centre
    frame B - same target, but with a *pointing error*: the centre is nudged
              within the field and the roll is perturbed (tracking / field
              rotation error). Transient artifacts (noise, cosmic rays,
              meteors, satellites) re-roll per frame and dust speckles are
              randomised per exposure, so A and B never share a stationary
              pixel pattern; the only consistent structure between the frames
              is the moving star field.

Label per pair (convention identical to ``sky_patch_gui._pixel_offset``):

    dx, dy    - pixel position of A's centre inside B minus B's image centre
                (sub-pixel, floats; positive right / down).
    droll     - B roll - A roll in degrees (deg).

So "A is obtained from B by translating the field by (dx, dy) and rotating by
droll" - a 2D similarity transform between the two exposures.

Output (as raw numpy files so no image I/O library is required):

    <out>/<split>_a.npy        (N, H, W) uint8   frame A grayscale
    <out>/<split>_b.npy        (N, H, W) uint8   frame B grayscale
    <out>/<split>_labels.npy   (N, 3)   float64  [dx, dy, droll_deg]
    <out>/<split>_meta.npz     fov, roll_a, stars_a, stars_b arrays
    <out>/params.json          dataset hyper-parameters

With ``--smoke`` an additional small independent smoke-test dataset is written
to ``--smoke-out`` (default ``data_smoke/``) as *directly usable PNG images +
text labels* (one A/B PNG pair and one TXT per sample, per split sub-folder;
no numpy files), using a separate RNG stream so the main dataset is unaffected.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PYTHON_DIR = HERE.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from render_sky_patch import (  # noqa: E402
    angular_distance_deg,
    apply_roll,
    gnomonic_project,
    load_catalog,
    render_psf_image,
)
import sky_effects  # noqa: E402

DEFAULT_CATALOG = HERE.parent / "data" / "hip_catalog.csv"

# luminance weights used when storing the RGB (H, W, 3) frames as grayscale
_LUM = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)

# transient-source bookkeeping
MAX_TRANSIENTS = 8                     # point sources per pair stored in *_meta.npz
MAX_SAT_POINTS = 96                    # satellite-trail GT points per pair
# Classes: appear = new OR brightening (A absent/faint -> B bright), dim =
# A bright -> B faint/absent, satellite = tumbling trail.
T_CLS = {"appear": 0, "dim": 1, "satellite": 2}
SAT_CLS = T_CLS["satellite"]
SAT_PROFILES = ("flat", "center", "glint")


def _tangent_xy_in_frame(ra_t: float, dec_t: float, geo: dict, fov: float,
                         margin: float = 0.98):
    """Project one physical (ra, dec) point into ``geo``'s frame; return
    post-roll tangent-plane (x, y) if inside the FOV, else None."""
    half = math.tan(math.radians(float(fov) / 2.0))
    x, y, vis = gnomonic_project(
        np.asarray([ra_t]), np.asarray([dec_t]), geo["ra"], geo["dec"])
    if not bool(vis[0]):
        return None
    x, y = apply_roll(x, y, geo["roll"])
    if abs(x[0]) > half * margin or abs(y[0]) > half * margin:
        return None
    return float(x[0]), float(y[0])


def _tangent_to_px(x: float, y: float, fov: float, size: int) -> tuple[float, float]:
    """Tangent-plane (x, y) (post-roll) -> pixel centre in a ``size`` square."""
    half = math.tan(math.radians(float(fov) / 2.0))
    px = ((x + half) / (2.0 * half)) * (size - 1)
    py = ((half - y) / (2.0 * half)) * (size - 1)
    return float(px), float(py)


def gnomonic_inverse_pt(x: float, y: float, ra0_deg: float, dec0_deg: float):
    """(x, y) tangent plane offset (radian-ish units, =tan of angle) -> ra/dec."""
    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)
    srx, crx = math.sin(ra0), math.cos(ra0)
    sdc, cdc = math.sin(dec0), math.cos(dec0)

    cx, cy, cz = cdc * crx, cdc * srx, sdc
    ex, ey, ez = -srx, crx, 0.0
    nx, ny, nz = -sdc * crx, -sdc * srx, cdc

    vx = cx + x * ex + y * nx
    vy = cy + x * ey + y * ny
    vz = cz + x * ez + y * nz
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    vx, vy, vz = vx / norm, vy / norm, vz / norm
    ra = math.degrees(math.atan2(vy, vx)) % 360.0
    dec = math.degrees(math.asin(max(-1.0, min(1.0, vz))))
    return ra, dec


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Randomized scene / artifact parameters (domain randomization)
# ---------------------------------------------------------------------------


def random_snap(rng: np.random.RandomState, args) -> dict:
    """Sample render options; GUI defaults are used as the centre of ranges.

    ``psf_sigma`` bounds come from ``args`` so the generator can be aimed at a
    particular native resolution: the same star field rendered at a higher
    pixel size keeps its pixel-space PSF (a camera is a camera), but after
    down-sampling the PSF appears sharper. Use tighter/smaller bounds for large
    ``--size`` outputs, wider bounds for small ones.
    """
    return {
        "fov": float(rng.uniform(1.5, 8.0)),
        "max_mag": float(rng.uniform(9.0, 11.0)),
        "psf_sigma": float(rng.uniform(args.psf_sigma_min, args.psf_sigma_max)),
        "gain": float(rng.uniform(1.2, 4.5)),
    }


def random_art(rng: np.random.RandomState) -> dict:
    """Sample the artifact configuration shared by A and B of one pair.

    Everything is deterministic given (seed, frame) exactly like sky_effects.
    Effects that must differ between the two exposures (noise / CR / meteor /
    satellite) depend on ``frame``. Sensor-plane dust is also randomised per
    frame, so the two exposures share no stationary pattern for a network to
    latch onto; the only consistent content is the moving star field.
    """
    art = {}
    art["noise"] = True
    art["noise_sigma"] = float(rng.uniform(0.004, 0.05))
    art["noise_bias"] = float(rng.uniform(0.01, 0.10))

    art["dust"] = rng.rand() < 0.9
    art["dust_count"] = int(rng.randint(0, 14))
    art["dust_size"] = float(rng.uniform(0.005, 0.06))
    art["dust_per_frame"] = True

    art["cr"] = rng.rand() < 0.85
    art["cr_count"] = int(rng.randint(0, 40))

    art["meteor"] = rng.rand() < 0.3
    art["meteor_count"] = int(rng.randint(1, 2))

    art["satellite"] = False   # satellites are now sampled explicitly (with GT)

    art["seeing"] = rng.rand() < 0.8
    art["seeing_sigma"] = float(rng.uniform(0.0, 2.5))

    art["spike"] = rng.rand() < 0.8
    art["spike_len"] = float(rng.uniform(0.01, 0.20))
    art["spike_int"] = float(rng.uniform(0.1, 1.6))

    art["ghost"] = rng.rand() < 0.5
    art["ghost_int"] = float(rng.uniform(0.05, 0.7))
    return art


# ---------------------------------------------------------------------------
# Rendering core (mirrors GUI SkyPatchGui._compute_image)
# ---------------------------------------------------------------------------


def project_stars(ra_all, dec_all, mag_all, snap):
    """Filter + gnomonic-project the catalog. Returns dict or None if < min."""
    max_mag = snap["max_mag"]
    ra0, dec0, fov = snap["ra"], snap["dec"], snap["fov"]
    w, h = snap["width"], snap["height"]

    mask = mag_all <= max_mag
    ra, dec, mag = ra_all[mask], dec_all[mask], mag_all[mask]

    radius = fov * math.sqrt(2.0) * 0.5
    dist = angular_distance_deg(ra, dec, ra0, dec0)
    keep = dist <= radius
    ra, dec, mag = ra[keep], dec[keep], mag[keep]

    x, y, visible = gnomonic_project(ra, dec, ra0, dec0)
    x, y, mag = x[visible], y[visible], mag[visible]
    x, y = apply_roll(x, y, snap["roll"])

    half = math.tan(math.radians(fov / 2.0))
    inside = (np.abs(x) <= half) & (np.abs(y) <= half)
    x, y, mag = x[inside], y[inside], mag[inside]
    if len(mag) < snap["min_stars"]:
        return None

    xp = (x + half) / (2.0 * half) * (w - 1)
    yp = (half - y) / (2.0 * half) * (h - 1)
    on = (xp >= 0) & (xp < w) & (yp >= 0) & (yp < h)
    flux = np.clip(np.power(10.0, -0.4 * (mag - 8.0)), 0.03, 80.0)
    return {
        "x": x,
        "y": y,
        "mag": mag,
        "xp": np.asarray(xp[on], dtype=np.float64),
        "yp": np.asarray(yp[on], dtype=np.float64),
        "flux": np.asarray(flux[on], dtype=np.float64),
        "count": int(len(mag)),
    }


def render_frame(snap: dict, art: dict, stars: dict,
                 extra: tuple | None = None,
                 satellites: list | None = None) -> np.ndarray:
    """(H, W, 3) float RGB in [0, 1] after PSF rendering + sky effects.

    ``extra`` = (x, y, mag) tangent-plane arrays of additional point sources
    (transients) rendered through the exact same PSF/tone-map as catalogue
    stars, but excluded from the spike/ghost bookkeeping (``eff_stars``).
    ``satellites`` = list of trail parameter dicts drawn before the effects
    (so they receive the same seeing/background treatment).
    """
    if extra is not None and len(extra[0]) > 0:
        xs = np.concatenate([stars["x"], extra[0]])
        ys = np.concatenate([stars["y"], extra[1]])
        ms = np.concatenate([stars["mag"], extra[2]])
    else:
        xs, ys, ms = stars["x"], stars["y"], stars["mag"]
    rgb = render_psf_image(
        x=xs,
        y=ys,
        mag=ms,
        half_extent=math.tan(math.radians(float(snap["fov"]) / 2.0)),
        width_px=int(snap["width"]),
        height_px=int(snap["height"]),
        psf_sigma=snap["psf_sigma"],
        gain=snap["gain"],
    )
    for s in satellites or []:
        sky_effects.draw_satellite_trail(
            rgb, s["p0"], s["p1"], s["width"], s["brightness"],
            bend_amp=s["bend_amp"], bend_waves=s["bend_waves"],
            profile=s["profile"], phase=s["phase"])
    eff_stars = {
        "x": stars["xp"],
        "y": stars["yp"],
        "flux": stars["flux"],
        "mag": stars["mag"],
    }
    return sky_effects.apply_effects(rgb, eff_stars, art)


def to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    lum = rgb @ _LUM
    return (np.clip(lum, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-pair sampling
# ---------------------------------------------------------------------------


class PairSampler:
    def __init__(self, catalog: tuple, args) -> None:
        self.ra_all, self.dec_all, self.mag_all = catalog
        self.args = args
        self.rng = np.random.RandomState(args.seed)
        self.roll_max = args.max_roll
        self.frac_lo = args.offset_frac_min
        self.frac_hi = args.offset_frac_max

    def sample_transients(self, rng: np.random.RandomState, size: int, fov: float,
                          geo_a: dict, geo_b: dict) -> dict | None:
        """Sample point-source transient sources for one A/B pair.

        Each source lives at a fixed sky position (defined relative to A's
        centre) and is rendered in both frames through the frame geometry.
        The ground truth (for the detection head) is the source position in
        frame B's pixels + class:
            0 appear - absent/faint in A, clearly bright in B (merges the old
                       'new' and 'brighten' cases; the A-side may be empty or a
                       faint visible source)
            1 dim    - clearly bright in A, much fainter (or gone) in B
        Only strong changes are generated (no mild variations).
        """
        rate = float(getattr(self.args, "transient_rate", 0.0))
        if rate <= 0.0:
            return None
        n = int(rng.poisson(rate))
        if n > MAX_TRANSIENTS:
            n = MAX_TRANSIENTS
        if n <= 0:
            return None
        half = math.tan(math.radians(float(fov) / 2.0))
        xa, ya, ma, xb, yb, mb = [], [], [], [], [], []
        gt: list[tuple[float, float, float]] = []
        for _ in range(n):
            cls = int(rng.choice([T_CLS["appear"], T_CLS["dim"]], p=[0.62, 0.38]))
            # position inside ~70% of the field so B's pointing offset/roll
            # keeps the source inside both frames.
            rho = half * 0.70 * math.sqrt(rng.uniform(0.0, 1.0))
            phi = rng.uniform(0.0, 2.0 * math.pi)
            u, v = rho * math.cos(phi), rho * math.sin(phi)
            ra, dec = gnomonic_inverse_pt(u, v, geo_a["ra"], geo_a["dec"])
            mag_a, mag_b = 99.0, 99.0
            if cls == T_CLS["appear"]:
                # half the time absent in A ('new'), half a faint visible
                # source ('brighten'); B is clearly bright either way
                if rng.rand() < 0.5:
                    mag_a = float(rng.uniform(9.5, 11.5))
                mag_b = float(rng.uniform(0.0, 4.0))
            else:  # dim: clearly bright in A, much fainter (or gone) in B
                mag_a = float(rng.uniform(0.0, 4.5))
                if rng.rand() < 0.4:
                    mag_b = 99.0                       # vanished
                else:
                    mag_b = float(rng.uniform(9.5, 13.0))

            pA = _tangent_xy_in_frame(ra, dec, geo_a, fov)
            pB = _tangent_xy_in_frame(ra, dec, geo_b, fov)
            if pA is None or pB is None:
                continue
            px_b, py_b = _tangent_to_px(pB[0], pB[1], fov, size)
            if not (0.0 <= px_b < size and 0.0 <= py_b < size):
                continue
            if mag_a < 90.0:
                xa.append(pA[0]); ya.append(pA[1]); ma.append(mag_a)
            if mag_b < 90.0:
                xb.append(pB[0]); yb.append(pB[1]); mb.append(mag_b)
            gt.append((px_b, py_b, float(cls)))

        if not gt:
            return None
        ex_a = (np.asarray(xa), np.asarray(ya), np.asarray(ma)) if xa else None
        ex_b = (np.asarray(xb), np.asarray(yb), np.asarray(mb)) if xb else None
        return {"extra_a": ex_a, "extra_b": ex_b,
                "gt": np.asarray(gt, dtype=np.float64)}

    def sample_satellites(self, rng: np.random.RandomState, size: int, fov: float,
                          geo_a: dict, geo_b: dict) -> dict | None:
        """Sample 0-2 tumbling satellite trails for one A/B pair.

        Appearance modes (mixed): 'single' = only visible in B (detectable),
        'both' = a trail in each frame at different positions.  The GT is the
        set of centreline points of the B-frame trail(s), class 'satellite'.
        """
        rate = float(getattr(self.args, "satellite_rate", 0.0))
        if rate <= 0.0:
            return None
        n = int(rng.poisson(rate))
        if n <= 0:
            return None
        n = min(n, 2)

        def _trail() -> dict:
            # random chord with endpoints on/near the frame border
            p0 = (rng.uniform(-0.1, 1.1) * size, rng.uniform(-0.1, 1.1) * size)
            p1 = (rng.uniform(-0.1, 1.1) * size, rng.uniform(-0.1, 1.1) * size)
            return {
                "p0": p0, "p1": p1,
                "width": float(rng.uniform(0.6, 2.0)),
                "brightness": float(rng.uniform(0.15, 0.6)),
                "bend_amp": float(rng.uniform(0.0, 6.0)),
                "bend_waves": float(rng.uniform(0.5, 2.5)),
                "profile": str(rng.choice(SAT_PROFILES)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }

        b_trails: list[dict] = []
        a_trails: list[dict] = []
        gt: list[tuple[float, float]] = []
        for _ in range(n):
            both = rng.rand() < 0.5
            tb = _trail()
            # reject degenerate (too short) chords
            if math.hypot(tb["p1"][0] - tb["p0"][0], tb["p1"][1] - tb["p0"][1]) < 0.4 * size:
                continue
            b_trails.append(tb)
            for (gx, gy) in sky_effects.satellite_centerline(
                    tb["p0"], tb["p1"], tb["bend_amp"], tb["bend_waves"],
                    tb["phase"], spacing_px=8.0):
                if 0.0 <= gx < size and 0.0 <= gy < size:
                    gt.append((gx, gy))
            if both:
                a_trails.append(_trail())
        if not gt:
            return None
        return {"a": a_trails, "b": b_trails,
                "gt": np.asarray(gt, dtype=np.float64)}


    def next_pair(self, frame_no: int, seed: int, size: int | None = None) -> dict:
        """Build one pair. Returns record with A/B geometry + images.

        ``size`` overrides ``args.size`` for the pair (used by the smoke
        dataset to mix different frame resolutions); ``None`` keeps the
        dataset-wide value.
        """
        args = self.args
        rng = self.rng
        if size is None:
            size = args.size
        fov = float(rng.uniform(args.fov_min, args.fov_max))
        snap = random_snap(rng, args)
        snap.update(
            {
                "width": size,
                "height": size,
                "min_stars": args.min_stars,
                "fov": fov,
            }
        )
        art = random_art(rng)
        art["seed"] = seed

        stars_a = None
        attempts = 0
        while stars_a is None and attempts < 60:
            ra0 = rng.uniform(0.0, 360.0)
            dec0 = rng.uniform(args.dec_min, args.dec_max)
            roll_a = rng.uniform(0.0, 360.0)
            geo_a = {"ra": ra0, "dec": dec0, "roll": roll_a}
            probe = dict(snap)
            probe.update(geo_a)
            stars_a = project_stars(self.ra_all, self.dec_all, self.mag_all, probe)
            attempts += 1
        if stars_a is None:
            # Extremely sparse sky: accept whatever we have.
            probe = dict(snap)
            probe.update(geo_a)
            stars_a = project_stars(self.ra_all, self.dec_all, self.mag_all, probe)

        # pointing error for B: radial offset in tangent plane + roll error
        frac = rng.uniform(self.frac_lo, self.frac_hi)
        max_ang = math.radians(fov * frac)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        rho = math.tan(max_ang) * math.sqrt(rng.uniform(0.0, 1.0))
        ra_b, dec_b = gnomonic_inverse_pt(
            rho * math.cos(phi), rho * math.sin(phi), geo_a["ra"], geo_a["dec"]
        )
        droll = float(rng.uniform(-self.roll_max, self.roll_max))
        roll_b = wrap_deg(geo_a["roll"] + droll)
        geo_b = {"ra": float(ra_b), "dec": float(dec_b), "roll": roll_b}

        trans = self.sample_transients(rng, size, fov, geo_a, geo_b)
        sats = self.sample_satellites(rng, size, fov, geo_a, geo_b)
        bg_lo, bg_hi = getattr(self.args, "bg_level_min", 0.0), getattr(self.args, "bg_level_max", 0.12)

        art_a = dict(art)
        art_a["frame"] = frame_no
        art_a["bg_level"] = float(rng.uniform(bg_lo, bg_hi))
        art_b = dict(art)
        art_b["frame"] = frame_no + 1
        art_b["bg_level"] = float(rng.uniform(bg_lo, bg_hi))

        snap_a = dict(snap)
        snap_a.update(geo_a)
        rgb_a = render_frame(snap_a, art_a, stars_a,
                             extra=trans["extra_a"] if trans else None,
                             satellites=sats["a"] if sats else None)

        probe_b = dict(snap)
        probe_b.update(geo_b)
        stars_b = project_stars(self.ra_all, self.dec_all, self.mag_all, probe_b)
        if stars_b is None:
            probe_b["min_stars"] = 0
            stars_b = project_stars(self.ra_all, self.dec_all, self.mag_all, probe_b)
        rgb_b = render_frame(snap, art_b, stars_b,
                             extra=trans["extra_b"] if trans else None,
                             satellites=sats["b"] if sats else None)

        dx, dy = self.a_center_in_b_px(geo_a, geo_b, fov, size)
        rec = {
            "img_a": to_gray_u8(rgb_a),
            "img_b": to_gray_u8(rgb_b),
            "label": (float(dx), float(dy), wrap_deg(roll_b - geo_a["roll"])),
            "ra0": float(geo_a["ra"]),
            "dec0": float(geo_a["dec"]),
            "roll_a": float(roll_a),
            "roll_b": float(roll_b),
            "offset_frac": float(frac),
            "seed": int(seed),
            "fov": fov,
            "droll": wrap_deg(roll_b - geo_a["roll"]),
            "stars_a": stars_a["count"],
            "stars_b": stars_b["count"] if stars_b is not None else 0,
            "bg_a": float(art_a["bg_level"]),
            "bg_b": float(art_b["bg_level"]),
        }
        if trans is not None:
            rec["gt_trans"] = trans["gt"]     # (K, 3) pxB_x, pxB_y, cls
        if sats is not None:
            rec["gt_sat"] = sats["gt"]        # (M, 2) pxB_x, pxB_y (class 3)
        return rec

    @staticmethod
    def a_center_in_b_px(geo_a: dict, geo_b: dict, fov: float, size: int) -> tuple[float, float]:
        """Sub-pixel position of A's centre inside B, minus B's centre."""
        half = math.tan(math.radians(float(fov) / 2.0))
        x, y, _ = gnomonic_project(
            np.asarray([geo_a["ra"]]), np.asarray([geo_a["dec"]]),
            geo_b["ra"], geo_b["dec"],
        )
        xr, yr = apply_roll(x, y, geo_b["roll"])
        xp = float((((xr + half) / (2.0 * half) * (size - 1)) - (size - 1) / 2.0)[0])
        yp = float((((half - yr) / (2.0 * half) * (size - 1)) - (size - 1) / 2.0)[0])
        return xp, yp


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _check_indices(meta: dict, count: int, k: int, rng: np.random.RandomState) -> list[int]:
    """Stratified sample indices: try to cover every transient class + satellite."""
    if count <= 0 or k <= 0:
        return []
    k = min(k, count)
    buckets: dict[str, list[int]] = {"appear": [], "dim": [],
                                     "satellite": [], "none": []}
    tn = meta.get("trans_n")
    tx = meta.get("trans_cls")
    sn = meta.get("sat_n")
    for i in range(count):
        tags = set()
        if tn is not None:
            for j in range(int(tn[i])):
                tags.add(["appear", "dim", "satellite"][int(round(tx[i, j]))])
        if sn is not None and int(sn[i]) > 0:
            tags.add("satellite")
        if not tags:
            buckets["none"].append(i)
        for t in tags:
            buckets[t].append(i)
    chosen: list[int] = []
    order = ["appear", "dim", "satellite", "none"]
    rng.shuffle(order)
    for key in order:
        if len(chosen) >= k:
            break
        pool = buckets[key]
        if pool:
            chosen.append(int(pool[rng.randint(len(pool))]))
    if len(chosen) < k:
        rest = [i for i in range(count) if i not in set(chosen)]
        rng.shuffle(rest)
        chosen.extend(rest[: k - len(chosen)])
    return sorted(chosen)


_CHECK_COLORS = {0: (255, 60, 60), 1: (60, 200, 255), 2: (200, 80, 255)}


def _annotate(u8: np.ndarray, pts, cls=None, radius: int = 3) -> np.ndarray:
    """Draw small coloured crosses on a grayscale image copy (for check PNGs)."""
    h, w = u8.shape[:2]
    out = np.dstack([u8, u8, u8]).astype(np.uint8) if u8.ndim == 2 else u8.copy()
    for n, pt in enumerate(pts):
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        c = _CHECK_COLORS.get(int(cls[n]) if cls is not None else 0, (255, 255, 255))
        for d in range(-radius, radius + 1):
            for xx, yy in ((x + d, y), (x, y + d)):
                if 0 <= yy < h and 0 <= xx < w:
                    out[yy, xx] = c
    return out


def export_check_samples(
    split: str,
    idxs: list[int],
    im_a: np.ndarray,
    im_b: np.ndarray,
    labels: np.ndarray,
    meta: dict,
    size: int,
    check_dir: Path,
) -> int:
    """Dump a handful of randomly picked pairs as PNGs + a text description.

    Saved per pair ``pair_<split>_<global-index>``:
        <stem>_a.png / <stem>_b.png   display-stretched grayscale frames
        <stem>_ab.png                 A | B side by side for quick eyeballing
        <stem>.txt                    geometry + star counts + the label
    Returns the number of pairs written.
    """
    check_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except Exception:  # pragma: no cover - optional dependency
        print("  check export skipped: Pillow not installed")
        return 0

    written = 0
    for gi in idxs:
        stem = check_dir / f"pair_{split}_{gi:06d}"
        dx, dy, droll = labels[gi]
        txt = (
            f"split            {split}\n"
            f"global index     {gi}\n"
            f"image size       {size} x {size}\n"
            f"A center RA/Dec  {meta['ra0'][gi]:.4f} deg  {meta['dec0'][gi]:.4f} deg\n"
            f"commanded roll A {meta['roll_a'][gi]:.2f} deg\n"
            f"actual roll B    {meta['roll_b'][gi]:.2f} deg   (droll {droll:+.3f})\n"
            f"FOV              {meta['fov'][gi]:.3f} deg\n"
            f"pointing offset  up to {meta['offset_frac'][gi] * 100.0:.1f}% of FOV\n"
            f"stars in frame   A {meta['stars_a'][gi]}   B {meta['stars_b'][gi]}\n"
            f"seed             {int(meta['seed'][gi])}\n"
            f"\n"
            f"label  dx  {dx:+.3f} px\n"
            f"       dy  {dy:+.3f} px\n"
            f"       droll {droll:+.3f} deg\n"
            f"(A's centre appears at B's centre + (dx, dy); shift B by "
            f"({-dx:+.3f}, {-dy:+.3f}) px and rotate {-droll:+.3f} deg to align "
            f"B onto A)\n"
        )

        def stretch(img: np.ndarray) -> np.ndarray:
            lo, hi = np.percentile(img, (0.5, 99.5))
            x = (img.astype(np.float32) - lo) / max(1e-6, hi - lo)
            return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        sa, sb = stretch(im_a[gi]), stretch(im_b[gi])
        # annotate B with the GT (point classes + satellite centreline points)
        p_pts, p_cls, s_pts = [], [], []
        tn = int(meta["trans_n"][gi]) if "trans_n" in meta else 0
        for j in range(tn):
            p_pts.append((meta["trans_x"][gi, j], meta["trans_y"][gi, j]))
            p_cls.append(int(round(meta["trans_cls"][gi, j])))
        sn = int(meta["sat_n"][gi]) if "sat_n" in meta else 0
        for j in range(sn):
            s_pts.append((meta["sat_x"][gi, j], meta["sat_y"][gi, j]))
        sb_ann = _annotate(sb, p_pts, p_cls) if p_pts else sb
        if s_pts:
            sb_ann = _annotate(sb_ann, s_pts, [SAT_CLS] * len(s_pts), radius=1)
        sb3 = np.dstack([sb_ann, sb_ann, sb_ann]) if sb_ann.ndim == 2 else sb_ann
        names = ["appear", "dim", "satellite"]
        txt += (
            f"transients       {[names[c] for c in p_cls]}\n"
            f"satellite points {sn}\n"
            f"sky background   A {meta.get('bg_a', np.zeros(1))[gi]:.3f}   "
            f"B {meta.get('bg_b', np.zeros(1))[gi]:.3f}\n"
        )
        Image.fromarray(sa).save(f"{stem}_a.png")
        Image.fromarray(sb3).save(f"{stem}_b.png")
        sa3 = np.dstack([sa, sa, sa]) if sa.ndim == 2 else sa
        gap = np.zeros((size, 4, 3), dtype=np.uint8) + 255
        Image.fromarray(np.concatenate([sa3, gap, sb3], axis=1)).save(f"{stem}_ab.png")
        stem.with_suffix(".txt").write_text(txt)
        written += 1
    return written


def export_pair_images(split: str, gi: int, rec: dict, size: int, split_dir: Path) -> None:
    """Write one pair as directly viewable PNGs (A/B) + a text label file.

    Used by the image/text smoke dataset: no .npy, just files a human (or any
    other tool) can consume:
        <split_dir>/<gi:06d>_a.png   display-stretched frame A
        <split_dir>/<gi:06d>_b.png   display-stretched frame B
        <split_dir>/<gi:06d>.txt     geometry + star counts + the label
    """
    try:
        from PIL import Image
    except Exception:
        raise SystemExit(
            "Pillow is required for the image/text smoke dataset:\n"
            "  python -m pip install pillow"
        )
    stem = split_dir / f"{gi:06d}"
    dx, dy, droll = rec["label"]
    txt = (
        f"split            {split}\n"
        f"index            {gi}\n"
        f"image size       {size} x {size}\n"
        f"A center RA/Dec  {rec['ra0']:.4f} deg  {rec['dec0']:.4f} deg\n"
        f"commanded roll A {rec['roll_a']:.2f} deg\n"
        f"actual roll B    {rec['roll_b']:.2f} deg   (droll {rec['droll']:+.3f})\n"
        f"FOV              {rec['fov']:.3f} deg\n"
        f"pointing offset  up to {rec['offset_frac'] * 100.0:.1f}% of FOV\n"
        f"stars in frame   A {rec['stars_a']}   B {rec['stars_b']}\n"
        f"seed             {int(rec['seed'])}\n"
        f"\n"
        f"label  dx  {dx:+.3f} px\n"
        f"       dy  {dy:+.3f} px\n"
        f"       droll {droll:+.3f} deg\n"
        f"(A's centre appears at B's centre + (dx, dy); shift B by "
        f"({-dx:+.3f}, {-dy:+.3f}) px and rotate {-droll:+.3f} deg to align "
        f"B onto A)\n"
    )
    stem.with_suffix(".txt").write_text(txt)

    lo, hi = np.percentile(rec["img_a"], (0.5, 99.5))
    x = (rec["img_a"].astype(np.float32) - lo) / max(1e-6, hi - lo)
    Image.fromarray((np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(
        f"{stem}_a.png")
    lo, hi = np.percentile(rec["img_b"], (0.5, 99.5))
    x = (rec["img_b"].astype(np.float32) - lo) / max(1e-6, hi - lo)
    Image.fromarray((np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(
        f"{stem}_b.png")


def run_split_images(sampler: "PairSampler", count: int, split: str,
                     out_dir: Path) -> dict:
    """Generate ``count`` pairs of ``split`` straight to PNG + TXT files.

    Returns the same summary dict as ``run_split`` (stats over the labels),
    without writing any numpy arrays.
    """
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    base_size = sampler.args.size
    mix = getattr(sampler.args, "smoke_sizes", None) or []

    def pair_size(i: int) -> int:
        # round-robin over the requested sizes so every resolution is covered
        return int(mix[i % len(mix)]) if mix else base_size

    labels = np.zeros((count, 3), dtype=np.float64)
    stars_a = np.zeros(count)
    fov = np.zeros(count)
    t0 = time.perf_counter()
    seed = sampler.args.seed + 10_000_000 * (0 if split == "train" else 1)

    for i in range(count):
        frame_no = 1 + i * 2 + (10_000_003 if split != "train" else 0)
        sz = pair_size(i)
        rec = sampler.next_pair(frame_no, seed + i * 7919, size=sz)
        export_pair_images(split, i, rec, sz, split_dir)
        labels[i] = rec["label"]
        stars_a[i] = rec["stars_a"]
        fov[i] = rec["fov"]
        if (i + 1) % 25 == 0 or i == count - 1:
            el = time.perf_counter() - t0
            print(
                f"  {split} {i + 1}/{count}   {el:6.1f}s elapsed  "
                f"avg {(el / (i + 1)) * 1e3:.0f} ms/pair",
                flush=True,
            )

    return {
        "split": split,
        "count": int(count),
        "mean_abs_dx": float(np.mean(np.abs(labels[:, 0]))),
        "mean_abs_dy": float(np.mean(np.abs(labels[:, 1]))),
        "mean_abs_droll": float(np.mean(np.abs(labels[:, 2]))),
        "max_abs_dx": float(np.max(np.abs(labels[:, 0]))),
        "max_abs_droll": float(np.max(np.abs(labels[:, 2]))),
        "stars_a": float(stars_a.mean()),
        "fov": float(fov.mean()),
    }


def run_split(
    sampler: PairSampler,
    count: int,
    split: str,
    out_dir: Path,
    t0: float,
    check_dir: Path | None = None,
    check_count: int = 0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    size = sampler.args.size
    im_a = np.zeros((count, size, size), dtype=np.uint8)
    im_b = np.zeros((count, size, size), dtype=np.uint8)
    labels = np.zeros((count, 3), dtype=np.float64)
    meta = {
        "fov": np.zeros(count),
        "roll_a": np.zeros(count),
        "roll_b": np.zeros(count),
        "ra0": np.zeros(count),
        "dec0": np.zeros(count),
        "offset_frac": np.zeros(count),
        "seed": np.zeros(count, dtype=np.int64),
        "stars_a": np.zeros(count, dtype=np.int32),
        "stars_b": np.zeros(count, dtype=np.int32),
    }
    # transient ground truth (padded with -1 to MAX_TRANSIENTS per sample)
    meta["trans_n"] = np.zeros(count, dtype=np.int32)
    meta["trans_x"] = np.full((count, MAX_TRANSIENTS), -1.0)
    meta["trans_y"] = np.full((count, MAX_TRANSIENTS), -1.0)
    meta["trans_cls"] = np.full((count, MAX_TRANSIENTS), -1.0)
    # satellite-trail GT (centreline points, class 3, padded with -1)
    meta["sat_n"] = np.zeros(count, dtype=np.int32)
    meta["sat_x"] = np.full((count, MAX_SAT_POINTS), -1.0)
    meta["sat_y"] = np.full((count, MAX_SAT_POINTS), -1.0)
    meta["bg_a"] = np.zeros(count)
    meta["bg_b"] = np.zeros(count)
    seed = sampler.args.seed + 10_000_000 * (0 if split == "train" else 1)

    for i in range(count):
        frame_no = 1 + i * 2 + (10_000_003 if split != "train" else 0)
        rec = sampler.next_pair(frame_no, seed + i * 7919)
        im_a[i] = rec["img_a"]
        im_b[i] = rec["img_b"]
        labels[i, 0] = rec["label"][0]
        labels[i, 1] = rec["label"][1]
        labels[i, 2] = rec["label"][2]
        meta["fov"][i] = rec["fov"]
        meta["roll_a"][i] = rec["roll_a"]
        meta["roll_b"][i] = rec["roll_b"]
        meta["ra0"][i] = rec["ra0"]
        meta["dec0"][i] = rec["dec0"]
        meta["offset_frac"][i] = rec["offset_frac"]
        meta["seed"][i] = rec["seed"]
        meta["stars_a"][i] = rec["stars_a"]
        meta["stars_b"][i] = rec["stars_b"]
        meta["bg_a"][i] = rec.get("bg_a", 0.0)
        meta["bg_b"][i] = rec.get("bg_b", 0.0)
        gt = rec.get("gt_trans")
        if gt is not None and len(gt):
            k = min(len(gt), MAX_TRANSIENTS)
            meta["trans_n"][i] = k
            meta["trans_x"][i, :k] = gt[:k, 0]
            meta["trans_y"][i, :k] = gt[:k, 1]
            meta["trans_cls"][i, :k] = gt[:k, 2]
        sg = rec.get("gt_sat")
        if sg is not None and len(sg):
            k = min(len(sg), MAX_SAT_POINTS)
            meta["sat_n"][i] = k
            meta["sat_x"][i, :k] = sg[:k, 0]
            meta["sat_y"][i, :k] = sg[:k, 1]
        if (i + 1) % 25 == 0 or i == count - 1:
            el = time.perf_counter() - t0
            print(
                f"  {split} {i + 1}/{count}   {el:6.1f}s elapsed  "
                f"avg {(el / (i + 1)) * 1e3:.0f} ms/pair",
                flush=True,
            )

    np.save(out_dir / f"{split}_a.npy", im_a)
    np.save(out_dir / f"{split}_b.npy", im_b)
    np.save(out_dir / f"{split}_labels.npy", labels)
    np.savez(out_dir / f"{split}_meta.npz", **meta)

    if check_dir is not None and check_count > 0 and count > 0:
        idxs = _check_indices(meta, count, check_count, sampler.rng)
        n_written = export_check_samples(
            split, idxs, im_a, im_b, labels, meta, size, check_dir
        )
        print(f"  check export: wrote {n_written} pairs to {check_dir}")
    return {
        "split": split,
        "count": int(count),
        "mean_abs_dx": float(np.mean(np.abs(labels[:, 0]))),
        "mean_abs_dy": float(np.mean(np.abs(labels[:, 1]))),
        "mean_abs_droll": float(np.mean(np.abs(labels[:, 2]))),
        "max_abs_dx": float(np.max(np.abs(labels[:, 0]))),
        "max_abs_droll": float(np.max(np.abs(labels[:, 2]))),
        "stars_a": float(meta["stars_a"].mean()),
        "fov": float(meta["fov"].mean()),
    }


def add_smoke_group(p: argparse.ArgumentParser) -> None:
    """Small independent test dataset in its own folder (never touches --out)."""
    g = p.add_argument_group("smoke test dataset")
    g.add_argument("--smoke", action="store_true",
                   help="also generate a small independent smoke-test dataset "
                        "into --smoke-out as PNG images + text labels "
                        "(separate RNG stream, so the main dataset is unaffected)")
    g.add_argument("--smoke-out", type=Path, default=HERE / "data_smoke",
                   help="folder for the image/text smoke-test dataset "
                        "(default: data_smoke)")
    g.add_argument("--smoke-size", type=int, default=96,
                   help="pixel size of the smoke frames (used when "
                        "--smoke-sizes is empty)")
    g.add_argument("--smoke-sizes", type=str, default="",
                   help="comma-separated frame sizes to MIX into the smoke set, "
                        "e.g. 64,128,192,384,512 (sizes are round-robined per "
                        "split so each resolution appears); empty = every "
                        "sample uses --smoke-size")
    g.add_argument("--smoke-train", type=int, default=16,
                   help="smoke train pairs")
    g.add_argument("--smoke-val", type=int, default=8,
                   help="smoke val pairs")
    g.add_argument("--smoke-test", type=int, default=8,
                   help="smoke test pairs")


def run_smoke(catalog: tuple, args) -> None:
    """Generate the independent smoke-test dataset into args.smoke_out.

    Data is stored as *directly usable images + text* (one PNG pair + one TXT
    label file per sample, per split sub-folder) - no numpy arrays. Uses its
    own RNG stream (seed + fixed offset) and a fresh sampler, so toggling
    --smoke never perturbs the main dataset's samples.
    """
    out = args.smoke_out
    if out.exists():
        shutil.rmtree(out)  # drop any stale numpy layout from previous runs
    out.mkdir(parents=True, exist_ok=True)
    sa = argparse.Namespace(**vars(args))
    sa.size = args.smoke_size
    sa.smoke_sizes = [int(x) for x in str(args.smoke_sizes).split(",") if x.strip()]
    sa.seed = args.seed + 8_000_017
    sampler = PairSampler(catalog, sa)
    t0 = time.perf_counter()
    summaries = []
    for split, n in (("train", args.smoke_train),
                     ("val", args.smoke_val),
                     ("test", args.smoke_test)):
        if n <= 0:
            continue
        s = run_split_images(sampler, n, split, out)
        summaries.append(s)
    print(f"\nsmoke total {time.perf_counter() - t0:.1f}s")

    mix = sa.smoke_sizes
    params_txt = (
        "smoke test dataset (images + text labels)\n"
        + (f"mixed sizes     {','.join(str(x) for x in mix)}\n" if mix
           else f"pixel size      {sa.size}\n")
        + f"fov range       {sa.fov_min:.1f} - {sa.fov_max:.1f} deg\n"
        + f"offset fraction {sa.offset_frac_min} - {sa.offset_frac_max} of FOV\n"
        + f"max |roll error| {sa.max_roll} deg\n"
        + f"seed            {sa.seed}\n"
        + f"\n"
        + f"label convention (per <index>.txt):\n"
        + f"  dx, dy   A's centre appears at B's centre + (dx, dy) px\n"
        + f"  droll    B roll - A roll, deg\n"
        + f"\n"
    )
    for s in summaries:
        print(s)
        params_txt += (
            f"{s['split']:<6} {s['count']:>4} pairs   "
            f"mean |dx| {s['mean_abs_dx']:.2f}  mean |dy| {s['mean_abs_dy']:.2f}  "
            f"mean |droll| {s['mean_abs_droll']:.2f} deg   "
            f"mean stars {s['stars_a']:.1f}   mean FOV {s['fov']:.2f} deg\n"
        )
    (out / "params.txt").write_text(params_txt)
    print(f"smoke dataset written to {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=HERE / "data")
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--train", type=int, default=600)
    p.add_argument("--val", type=int, default=80)
    p.add_argument("--test", type=int, default=80)
    add_smoke_group(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fov-min", type=float, default=1.5)
    p.add_argument("--fov-max", type=float, default=8.0)
    p.add_argument("--offset-frac-min", type=float, default=0.02,
                   help="min pointing offset as a fraction of the FOV")
    p.add_argument("--offset-frac-max", type=float, default=0.08)
    p.add_argument("--max-roll", type=float, default=8.0,
                   help="max |roll error| between A and B in degrees")
    p.add_argument("--transient-rate", type=float, default=1.2,
                   help="mean number of point-source transients per pair "
                        "(Poisson); 0 disables. Classes: new / brighten / dim, "
                        "GT in *_meta.npz trans_x/trans_y/trans_cls (B-frame px)")
    p.add_argument("--satellite-rate", type=float, default=0.8,
                   help="mean number of tumbling satellite trails per pair "
                        "(Poisson, capped at 2); 0 disables. GT in "
                        "*_meta.npz sat_x/sat_y (B-frame centreline points)")
    p.add_argument("--bg-level-min", type=float, default=0.0,
                   help="min uniform sky-background offset added per frame")
    p.add_argument("--bg-level-max", type=float, default=0.12,
                   help="max uniform sky-background offset (A/B sampled "
                        "independently)")
    p.add_argument("--min-stars", type=int, default=5)
    p.add_argument("--psf-sigma-min", type=float, default=0.7,
                   help="lower PSF sigma bound (px) sampled per frame")
    p.add_argument("--psf-sigma-max", type=float, default=2.2,
                   help="upper PSF sigma bound (px) sampled per frame")
    p.add_argument("--dec-min", type=float, default=-75.0)
    p.add_argument("--dec-max", type=float, default=75.0)
    p.add_argument("--check-dir", type=Path, default=HERE / "check_data",
                   help="folder for a few random pairs exported as PNG + txt "
                        "(for manual inspection)")
    p.add_argument("--check-count", type=int, default=6,
                   help="random pairs exported per split (0 to disable)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.catalog.exists():
        raise FileNotFoundError(args.catalog)
    if not (0 < args.offset_frac_min <= args.offset_frac_max <= 0.5):
        raise ValueError("offset fraction must be within (0, 0.5]")
    if not (0 < args.psf_sigma_min <= args.psf_sigma_max):
        raise ValueError("psf_sigma_min must be <= psf_sigma_max")
    ra, dec, mag = load_catalog(args.catalog)
    print(f"catalog: {len(ra)} stars")

    sampler = PairSampler((ra, dec, mag), args)
    t0 = time.perf_counter()
    summaries = []
    for split, n in (("train", args.train), ("val", args.val), ("test", args.test)):
        if n <= 0:
            continue
        s = run_split(sampler, n, split, args.out, t0, args.check_dir, args.check_count)
        summaries.append(s)

    if summaries:
        print(f"\ntotal {time.perf_counter() - t0:.1f}s")
        with (args.out / "params.json").open("w") as f:
            json.dump(
                {
                    "size": args.size,
                    "splits": summaries,
                    "label": "dx,dy = A-centre in B frame minus B-centre (px); droll = B-A (deg)",
                    "transient_classes": {k: v for k, v in T_CLS.items()},
                    "gt": ("trans_x/trans_y/trans_cls (points, B-frame px) and "
                           "sat_x/sat_y (satellite centreline points, B-frame px)"),
                    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                },
                f,
                indent=2,
            )
        for s in summaries:
            print(s)

    if args.smoke:
        run_smoke((ra, dec, mag), args)


if __name__ == "__main__":
    main()
