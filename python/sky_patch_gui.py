#!/usr/bin/env python3
"""PySide6 GUI for render_sky_patch.

Lets you pick a pointing center directly on a sky map (mouse click/drag),
choose FOV / roll / output pixel size, and live-render the sky patch into a
preview pane using the exact same projection / PSF kernels as
``render_sky_patch.py`` / ``cpp/render_sky_patch.cpp``.

Run:
    python sky_patch_gui.py
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_sky_patch import (  # noqa: E402
    angular_distance_deg,
    apply_roll,
    gnomonic_project,
    load_catalog,
    render_psf_image,
)
import sky_effects  # noqa: E402

DEFAULT_CATALOG = SCRIPT_DIR.parent / "data" / "hip_catalog.csv"
DEFAULT_RA = 83.82  # M42
DEFAULT_DEC = -5.3875
DEFAULT_FOV = 30.0

OVERVIEW_MAG = 6.5  # stars shown on the all-sky map


def _make_ispin(parent, lo: int, hi: int, step: int, val: int) -> QSpinBox:
    s = QSpinBox(parent)
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(val)
    return s


def _make_fspin(
    parent, lo: float, hi: float, decimals: int, step: float, val: float
) -> QDoubleSpinBox:
    s = QDoubleSpinBox(parent)
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(val)
    return s

# --------------------------------------------------------------------------
# Small astro / coordinate helpers
# --------------------------------------------------------------------------


def gnomonic_inverse_pt(x: float, y: float, ra0_deg: float, dec0_deg: float) -> tuple[float, float]:
    """Invert the gnomonic projection: tangent-plane (x, y) -> sky (ra, dec)."""
    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)
    srx, crx = math.sin(ra0), math.cos(ra0)
    sdc, cdc = math.sin(dec0), math.cos(dec0)

    # Local basis: center, east (increasing RA), north (increasing Dec).
    cx, cy, cz = cdc * crx, cdc * srx, sdc
    ex, ey, ez = -srx, crx, 0.0
    nx, ny, nz = -sdc * crx, -sdc * srx, cdc

    vx = cx + x * ex + y * nx
    vy = cy + x * ey + y * ny
    vz = cz + x * ez + y * nz
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    vx /= norm
    vy /= norm
    vz /= norm

    ra = math.degrees(math.atan2(vy, vx)) % 360.0
    dec = math.degrees(math.asin(max(-1.0, min(1.0, vz))))
    return ra, dec


def gnomonic_inverse_vec(
    x: np.ndarray, y: np.ndarray, ra0_deg: float, dec0_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised gnomonic inverse (used for frame registration)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)
    srx, crx = math.sin(ra0), math.cos(ra0)
    sdc, cdc = math.sin(dec0), math.cos(dec0)

    cx, cy, cz = cdc * crx, cdc * srx, sdc
    ex, ey = -srx, crx
    nx, ny, nz = -sdc * crx, -sdc * srx, cdc

    vx = cx + x * ex + y * nx
    vy = cy + x * ey + y * ny
    vz = cz + x * 0.0 + y * nz
    norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    vx /= norm
    vy /= norm
    vz /= norm
    ra = np.degrees(np.arctan2(vy, vx)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(vz, -1.0, 1.0)))
    return ra, dec


def ra_to_hms(ra_deg: float) -> str:
    ra = ra_deg % 360.0
    hours = ra / 15.0
    h = int(hours)
    m = int((hours - h) * 60.0)
    s = (hours - h - m / 60.0) * 3600.0
    return f"{h:02d}h{m:02d}m{s:04.1f}s"


def dec_to_dms(dec_deg: float) -> str:
    sign = "+" if dec_deg >= 0 else "-"
    dec = abs(dec_deg)
    d = int(dec)
    m = int((dec - d) * 60.0)
    s = (dec - d - m / 60.0) * 3600.0
    return f"{sign}{d:02d}\u00b0{m:02d}'{s:04.1f}\""


def fmt_angle(v: float) -> str:
    return f"{v:.3f}".replace("-", "m").replace(".", "p")


def default_save_name(ra: float, dec: float, fov: float, roll: float) -> str:
    return (
        f"sky_ra{fmt_angle(ra)}_dec{fmt_angle(dec)}_fov{fmt_angle(fov)}"
        f"_roll{fmt_angle(roll)}.png"
    )


class _SignalBridge(QtCore.QObject):
    """Small QObject living on the GUI thread so background threads can emit."""

    done = Signal(object)


# --------------------------------------------------------------------------
# All-sky map: click / drag to choose the pointing center
# --------------------------------------------------------------------------


class SkyMapWidget(QWidget):
    centerChanged = Signal(float, float)  # ra_deg, dec_deg
    hoverMoved = Signal(object)  # (ra_deg, dec_deg) or None

    _M_LEFT, _M_RIGHT, _M_TOP, _M_BOTTOM = 50.0, 10.0, 12.0, 22.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ra0 = DEFAULT_RA
        self._dec0 = DEFAULT_DEC
        self._fov = DEFAULT_FOV
        self._stars_ra = np.empty(0)
        self._stars_dec = np.empty(0)
        self._stars_mag = np.empty(0)
        self._scene: QPixmap | None = None
        self._drag = False
        self._hover: QPointF | None = None
        self.setMinimumHeight(220)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)

    # -- external state -----------------------------------------------------
    def set_catalog(self, ra: np.ndarray, dec: np.ndarray, mag: np.ndarray) -> None:
        mask = mag <= OVERVIEW_MAG
        self._stars_ra = np.asarray(ra[mask], dtype=np.float64)
        self._stars_dec = np.asarray(dec[mask], dtype=np.float64)
        self._stars_mag = np.asarray(mag[mask], dtype=np.float64)
        self._scene = None
        self.update()

    def set_center(self, ra: float, dec: float, emit: bool = False) -> None:
        ra = min(360.0, max(0.0, ra)) % 360.0
        dec = min(90.0, max(-90.0, dec))
        changed = (ra, dec) != (self._ra0, self._dec0)
        self._ra0, self._dec0 = ra, dec
        if changed:
            self.update()
        if emit and changed:
            self.centerChanged.emit(ra, dec)

    def set_fov(self, fov: float) -> None:
        self._fov = float(fov)
        self.update()

    # -- chart mapping ------------------------------------------------------
    def _plot_rect(self) -> QRectF:
        r = self.rect()
        w = max(1.0, r.width() - self._M_LEFT - self._M_RIGHT)
        h = max(1.0, r.height() - self._M_TOP - self._M_BOTTOM)
        return QRectF(self._M_LEFT, self._M_TOP, w, h)

    def _chart_to_widget(self, ra_deg: float, dec_deg: float) -> QPointF:
        plot = self._plot_rect()
        x = plot.left() + (ra_deg / 360.0) * plot.width()
        y = plot.top() + ((90.0 - dec_deg) / 180.0) * plot.height()
        return QPointF(x, y)

    def _widget_to_chart(self, p: QPointF) -> tuple[float, float]:
        plot = self._plot_rect()
        frac_x = (p.x() - plot.left()) / plot.width()
        frac_y = (p.y() - plot.top()) / plot.height()
        ra = min(360.0, max(0.0, frac_x * 360.0))
        dec = 90.0 - min(1.0, max(0.0, frac_y)) * 180.0
        return ra, dec

    # -- user interaction ---------------------------------------------------
    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = True
            self._pick(e.position())
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._hover = e.position()
        self.update()
        if self._drag:
            self._pick(e.position())
        else:
            ra, dec = self._widget_to_chart(e.position())
            self.hoverMoved.emit((ra, dec))
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._drag = False
        super().mouseReleaseEvent(e)

    def leaveEvent(self, e) -> None:  # noqa: N802
        self._hover = None
        self.hoverMoved.emit(None)
        self.update()
        super().leaveEvent(e)

    def _pick(self, pos: QPointF) -> None:
        ra, dec = self._widget_to_chart(pos)
        self.set_center(ra, dec, emit=True)

    def resizeEvent(self, e) -> None:  # noqa: N802
        self._scene = None
        super().resizeEvent(e)

    # -- rendering ----------------------------------------------------------
    def paintEvent(self, e) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._scene is None:
            self._scene = self._build_scene()
        if self._scene is not None:
            painter.drawPixmap(0, 0, self._scene)
        self._draw_overlay(painter)
        painter.end()

    def _build_scene(self) -> QPixmap | None:
        w, h = self.width(), self.height()
        if w < 10 or h < 10:
            return None
        dpr = self.devicePixelRatioF() or 1.0
        pixmap = QPixmap(max(1, int(round(w * dpr))), max(1, int(round(h * dpr))))
        pixmap.setDevicePixelRatio(dpr)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(dpr, dpr)

        painter.fillRect(self.rect(), QColor(16, 20, 32))
        plot = self._plot_rect()
        painter.fillRect(plot, QColor(10, 13, 22))

        # Grid lines + labels (RA every 2h, Dec every 15 deg).
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.5))
        painter.setFont(font)
        pen = QPen(QColor(52, 66, 96), 0.0)
        painter.setPen(pen)
        for ra in range(0, 360, 30):
            x = self._chart_to_widget(ra, 0).x()
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(QColor(150, 165, 195))
            painter.drawText(
                QRectF(x - 18, plot.top() - 11, 36, 11),
                Qt.AlignmentFlag.AlignCenter,
                f"{int(round(ra / 15.0)):02d}h",
            )
            painter.setPen(pen)
        for dec in range(-90, 91, 15):
            y = self._chart_to_widget(0, dec).y()
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor(150, 165, 195))
            painter.drawText(
                QRectF(plot.left() - self._M_LEFT + 2, y - 7, self._M_LEFT - 4, 14),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{dec:+d}\u00b0",
            )
            painter.setPen(pen)

        # Celestial equator highlight.
        y_eq = self._chart_to_widget(0, 0).y()
        painter.setPen(QPen(QColor(70, 96, 150), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(plot.left(), y_eq), QPointF(plot.right(), y_eq))

        # Stars.
        n = len(self._stars_mag)
        if n:
            xs = (self._stars_ra / 360.0) * plot.width() + plot.left()
            ys = ((90.0 - self._stars_dec) / 180.0) * plot.height() + plot.top()
            for i in range(n):
                mag = float(self._stars_mag[i])
                alpha = int(255 * min(1.0, (OVERVIEW_MAG - mag) * 0.45 + 0.35))
                radius = 0.45 + max(0.0, (4.5 - mag)) * 0.22
                col = QColor(230, 238, 255, alpha)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(col)
                painter.drawEllipse(QPointF(xs[i], ys[i]), radius, radius)

        painter.end()
        return pixmap

    def _draw_overlay(self, painter: QPainter) -> None:
        # FOV footprint.
        polys = self._footprint_polys()
        if polys:
            painter.setBrush(QColor(255, 214, 90, 26))
            painter.setPen(QPen(QColor(255, 214, 90, 210), 1.0))
            for poly in polys:
                painter.drawPolygon(poly)

        # Center crosshair.
        c = self._chart_to_widget(self._ra0, self._dec0)
        pen = QPen(QColor(255, 120, 120, 235), 1.2)
        painter.setPen(pen)
        painter.drawLine(QPointF(c.x() - 8, c.y()), QPointF(c.x() - 3, c.y()))
        painter.drawLine(QPointF(c.x() + 3, c.y()), QPointF(c.x() + 8, c.y()))
        painter.drawLine(QPointF(c.x(), c.y() - 8), QPointF(c.x(), c.y() - 3))
        painter.drawLine(QPointF(c.x(), c.y() + 3), QPointF(c.x(), c.y() + 8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 120, 120, 170), 1.0))
        painter.drawEllipse(c, 4.0, 4.0)

        # Hover pointer marker.
        if self._hover is not None and self.rect().contains(self._hover.toPoint()):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(120, 210, 255, 120))
            painter.drawEllipse(self._hover, 2.2, 2.2)

    def _footprint_polys(self) -> list[QPolygonF]:
        fov = max(0.001, self._fov)
        he = math.tan(math.radians(fov / 2.0))
        ra0, dec0 = self._ra0, self._dec0
        samples: list[tuple[float, float]] = []
        n = 24
        # Walk the gnomonic "square" edges in sky coordinates.
        for x in np.linspace(-he, he, n):
            samples.append(gnomonic_inverse_pt(float(x), -he, ra0, dec0))
        for y in np.linspace(-he, he, n):
            samples.append(gnomonic_inverse_pt(he, float(y), ra0, dec0))
        for x in np.linspace(he, -he, n):
            samples.append(gnomonic_inverse_pt(float(x), he, ra0, dec0))
        for y in np.linspace(he, -he, n):
            samples.append(gnomonic_inverse_pt(-he, float(y), ra0, dec0))

        # Split into polylines when RA wraps around the 0/360 edge.
        chunks: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        for pt in samples:
            if current and abs(pt[0] - current[-1][0]) > 180.0:
                if len(current) > 1:
                    chunks.append(current)
                current = []
            current.append(pt)
        if len(current) > 1:
            chunks.append(current)

        return [QPolygonF([self._chart_to_widget(ra, dec) for ra, dec in ch]) for ch in chunks]


# --------------------------------------------------------------------------
# Preview canvas: shows the rendered patch, supports zoom / pan / re-center
# --------------------------------------------------------------------------


class ImageCanvas(QWidget):
    recentered = Signal(float, float)  # ra_deg, dec_deg
    hoverMoved = Signal(object)  # (ra_deg, dec_deg) or None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._geo_ra = DEFAULT_RA
        self._geo_dec = DEFAULT_DEC
        self._geo_fov = DEFAULT_FOV
        self._geo_roll = 0.0
        self._scale = 1.0
        self._vcx = 0.0  # image pixel at widget center
        self._vcy = 0.0
        self._press = None  # QPointF
        self._press_vc = None  # (x, y)
        self._panning = False
        self._last_hover_sky = None
        self.setMouseTracking(True)
        self.setMinimumSize(320, 220)

    # -- external -----------------------------------------------------------
    def set_image(self, qimage: QImage, geometry: dict) -> None:
        new_dims = self._image is None or (
            qimage.width() != self._image.width() or qimage.height() != self._image.height()
        )
        self._image = qimage
        self._geo_ra = geometry["ra"]
        self._geo_dec = geometry["dec"]
        self._geo_fov = geometry["fov"]
        self._geo_roll = geometry["roll"]
        if new_dims:
            self._vcx = qimage.width() / 2.0
            self._vcy = qimage.height() / 2.0
            self._scale = self._fit_scale()
        self.update()

    def image_size(self) -> tuple[int, int] | None:
        if self._image is None:
            return None
        return self._image.width(), self._image.height()

    def clear_image(self) -> None:
        self._image = None
        self.update()

    def get_image(self) -> QImage | None:
        return self._image

    def _fit_scale(self) -> float:
        if self._image is None:
            return 1.0
        r = self.rect()
        if r.width() < 10 or r.height() < 10:
            return 1.0
        iw, ih = self._image.width(), self._image.height()
        return min(r.width() / iw, r.height() / ih)

    # -- coordinate mapping -------------------------------------------------
    def _img_to_widget(self, ix: float, iy: float) -> QPointF:
        c = self.rect().center()
        return QPointF(c.x() + (ix - self._vcx) * self._scale, c.y() + (iy - self._vcy) * self._scale)

    def _widget_to_img(self, p: QPointF) -> tuple[float, float]:
        c = self.rect().center()
        return (p.x() - c.x()) / self._scale + self._vcx, (p.y() - c.y()) / self._scale + self._vcy

    def _sky_from_img(self, ix: float, iy: float) -> tuple[float, float]:
        img = self._image
        if img is None:
            return self._geo_ra, self._geo_dec
        w, h = img.width(), img.height()
        half_extent = math.tan(math.radians(self._geo_fov / 2.0))
        x = (ix / max(1, w - 1)) * 2.0 * half_extent - half_extent
        y = half_extent - (iy / max(1, h - 1)) * 2.0 * half_extent
        # Remove camera roll (forward rotates by -roll, so invert with +roll).
        theta = math.radians(self._geo_roll)
        xr = x * math.cos(theta) - y * math.sin(theta)
        yr = x * math.sin(theta) + y * math.cos(theta)
        return gnomonic_inverse_pt(xr, yr, self._geo_ra, self._geo_dec)

    # -- user interaction ---------------------------------------------------
    def wheelEvent(self, e: QtGui.QWheelEvent) -> None:  # noqa: N802
        if self._image is None:
            return
        steps = e.angleDelta().y() / 120.0
        if steps == 0:
            return
        factor = 1.2 ** steps
        new_scale = self._scale * factor
        new_scale = min(64.0, max(0.02, new_scale))
        if abs(new_scale - self._scale) < 1e-9:
            return
        pos = e.position()
        c = self.rect().center()
        ix, iy = self._widget_to_img(pos)
        self._scale = new_scale
        self._vcx = ix - (pos.x() - c.x()) / self._scale
        self._vcy = iy - (pos.y() - c.y()) / self._scale
        self.update()

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._press = e.position()
            self._press_vc = (self._vcx, self._vcy)
            self._panning = False
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:  # noqa: N802
        pos = e.position()
        if self._press is not None and self._image is not None:
            dx = pos.x() - self._press.x()
            dy = pos.y() - self._press.y()
            if self._panning or (dx * dx + dy * dy) > 25.0:
                self._panning = True
                self._vcx = self._press_vc[0] - dx / self._scale
                self._vcy = self._press_vc[1] - dy / self._scale
                self.update()
        if self._image is not None:
            ix, iy = self._widget_to_img(pos)
            if 0 <= ix < self._image.width() and 0 <= iy < self._image.height():
                ra, dec = self._sky_from_img(ix, iy)
                self._last_hover_sky = (ra, dec)
                self.hoverMoved.emit((ra, dec))
            else:
                self._last_hover_sky = None
                self.hoverMoved.emit(None)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton and self._press is not None:
            if not self._panning:
                pos = e.position()
                if self._image is not None:
                    ix, iy = self._widget_to_img(pos)
                    if 0 <= ix < self._image.width() and 0 <= iy < self._image.height():
                        ra, dec = self._sky_from_img(ix, iy)
                        self.recentered.emit(ra, dec)
            self._press = None
            self._panning = False
        super().mouseReleaseEvent(e)

    def leaveEvent(self, e) -> None:  # noqa: N802
        self._last_hover_sky = None
        self.hoverMoved.emit(None)
        super().leaveEvent(e)

    # -- painting -----------------------------------------------------------
    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        if self._image is not None:
            self.update()

    def paintEvent(self, e) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(6, 8, 14))
        painter.setPen(QPen(QColor(70, 82, 108), 1.0))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        if self._image is None:
            painter.setPen(QColor(110, 120, 140))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image yet")
            painter.end()
            return

        iw, ih = self._image.width(), self._image.height()
        c = self.rect().center()
        x0, y0 = c.x() - self._vcx * self._scale, c.y() - self._vcy * self._scale

        # Visible source sub-rect of the image (crop for cheap painting).
        sx0 = max(0.0, (self.rect().left() - x0) / self._scale)
        sy0 = max(0.0, (self.rect().top() - y0) / self._scale)
        sx1 = min(float(iw), (self.rect().right() - x0) / self._scale)
        sy1 = min(float(ih), (self.rect().bottom() - y0) / self._scale)
        if sx1 > sx0 and sy1 > sy0:
            src = QRectF(sx0, sy0, sx1 - sx0, sy1 - sy0)
            dst = QRectF(x0 + sx0 * self._scale, y0 + sy0 * self._scale,
                         (sx1 - sx0) * self._scale, (sy1 - sy0) * self._scale)
            painter.drawImage(dst, self._image, src)

        # Reticle at image center (current pointing).
        reticle = self._img_to_widget(iw / 2.0, ih / 2.0)
        cx, cy = reticle.x(), reticle.y()
        if self.rect().adjusted(-4, -4, 4, 4).contains(QPointF(cx, cy).toPoint()):
            painter.setPen(QPen(QColor(255, 160, 90, 220), 1.2))
            painter.drawLine(QPointF(cx - 9, cy), QPointF(cx - 3, cy))
            painter.drawLine(QPointF(cx + 3, cy), QPointF(cx + 9, cy))
            painter.drawLine(QPointF(cx, cy - 9), QPointF(cx, cy - 3))
            painter.drawLine(QPointF(cx, cy + 3), QPointF(cx, cy + 9))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), 3.2, 3.2)

        painter.end()


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------


class SkyPatchGui(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sky Patch Renderer")
        self.resize(1180, 760)

        self._catalog = None  # (ra, dec, mag)
        self._token = 0
        self._busy = False
        self._need_render = False
        self._frame = 0  # exposure counter for per-frame random artifacts
        self._current_image: QImage | None = None
        self._error_image: QImage | None = None
        self._b_rgb: np.ndarray | None = None  # raw B frame as float RGB
        self._b_geo_n: dict | None = None  # A (commanded) geometry
        self._b_geo_b: dict | None = None  # B (pointing-error) geometry
        self._art_check: dict = {}
        self._art_param_ids: dict = {}

        self._bridge = _SignalBridge()
        self._bridge.done.connect(self._on_render_done)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._launch_render)

        self._build_ui()
        self._load_catalog()
        self._connect_signals()
        self._sync_center_ui()
        self._schedule_render()

    # -- UI construction ----------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # --- left control panel (scrollable) -------------------------------
        left = QWidget()
        left.setMinimumWidth(330)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # Center / pointing group
        center_box = QGroupBox("Pointing (click or drag the map)", left)
        cv = QVBoxLayout(center_box)
        self.map_widget = SkyMapWidget(center_box)
        cv.addWidget(self.map_widget)

        coord_row = QHBoxLayout()
        coord_row.addWidget(QLabel("RA", center_box))
        self.spin_ra = QDoubleSpinBox(center_box)
        self.spin_ra.setRange(0.0, 360.0)
        self.spin_ra.setDecimals(4)
        self.spin_ra.setSingleStep(0.1)
        self.spin_ra.setWrapping(True)
        self.spin_ra.setValue(DEFAULT_RA)
        coord_row.addWidget(self.spin_ra, 1)
        coord_row.addWidget(QLabel("Dec", center_box))
        self.spin_dec = QDoubleSpinBox(center_box)
        self.spin_dec.setRange(-90.0, 90.0)
        self.spin_dec.setDecimals(4)
        self.spin_dec.setSingleStep(0.1)
        self.spin_dec.setValue(DEFAULT_DEC)
        coord_row.addWidget(self.spin_dec, 1)
        cv.addLayout(coord_row)

        self.label_center_hms = QLabel(center_box)
        self.label_center_hms.setStyleSheet("color: #8fa3c8;")
        cv.addWidget(self.label_center_hms)
        left_layout.addWidget(center_box)

        # Camera / output group
        cam_box = QGroupBox("Camera & output image", left)
        form = QtWidgets.QFormLayout(cam_box)

        self.spin_fov = QDoubleSpinBox(cam_box)
        self.spin_fov.setRange(0.1, 170.0)
        self.spin_fov.setDecimals(2)
        self.spin_fov.setSingleStep(0.5)
        self.spin_fov.setValue(DEFAULT_FOV)
        form.addRow("FOV \u00b0", self.spin_fov)

        self.spin_roll = QDoubleSpinBox(cam_box)
        self.spin_roll.setRange(-180.0, 180.0)
        self.spin_roll.setDecimals(1)
        self.spin_roll.setSingleStep(5.0)
        self.spin_roll.setWrapping(True)
        form.addRow("Roll \u00b0 (CW +)", self.spin_roll)

        px_row = QHBoxLayout()
        self.spin_w = QSpinBox(cam_box)
        self.spin_w.setRange(64, 8192)
        self.spin_w.setSingleStep(32)
        self.spin_w.setValue(1024)
        px_row.addWidget(self.spin_w, 1)
        px_row.addWidget(QLabel("\u00d7", cam_box))
        self.spin_h = QSpinBox(cam_box)
        self.spin_h.setRange(64, 8192)
        self.spin_h.setSingleStep(32)
        self.spin_h.setValue(1024)
        px_row.addWidget(self.spin_h, 1)
        form.addRow("Pixels X \u00d7 Y", px_row)
        left_layout.addWidget(cam_box)

        # Rendering / appearance group
        opt_box = QGroupBox("Render options", left)
        oform = QtWidgets.QFormLayout(opt_box)

        self.spin_maxmag = QDoubleSpinBox(opt_box)
        self.spin_maxmag.setRange(2.0, 13.0)
        self.spin_maxmag.setDecimals(1)
        self.spin_maxmag.setSingleStep(0.5)
        self.spin_maxmag.setValue(10.0)
        oform.addRow("Max magnitude", self.spin_maxmag)

        self.spin_psf = QDoubleSpinBox(opt_box)
        self.spin_psf.setRange(0.2, 8.0)
        self.spin_psf.setDecimals(2)
        self.spin_psf.setSingleStep(0.05)
        self.spin_psf.setValue(1.2)
        oform.addRow("PSF sigma (px)", self.spin_psf)

        self.spin_gain = QDoubleSpinBox(opt_box)
        self.spin_gain.setRange(0.2, 8.0)
        self.spin_gain.setDecimals(2)
        self.spin_gain.setSingleStep(0.1)
        self.spin_gain.setValue(2.0)
        oform.addRow("Gain", self.spin_gain)
        left_layout.addWidget(opt_box)

        # Artifact simulations group
        self._build_artifact_group(left)

        # Actions
        btn_row = QHBoxLayout()
        self.check_live = QCheckBox("Live update", left)
        self.check_live.setChecked(True)
        btn_row.addWidget(self.check_live)
        self.btn_render = QPushButton("Render now", left)
        btn_row.addWidget(self.btn_render)
        left_layout.addLayout(btn_row)

        self.btn_save = QPushButton("Save image \u2026", left)
        left_layout.addWidget(self.btn_save)

        tip = QLabel(
            "Tip: wheel zooms the preview, drag pans, click re-points. "
            "Transient artifacts (noise / cosmic rays / meteors) re-roll on "
            "every new exposure; dust / spikes / ghosts stay fixed per seed.",
            left,
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #7c8aa6; font-size: 11px;")
        left_layout.addWidget(tip)
        left_layout.addStretch(1)

        scroll = QScrollArea(central)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(352)
        scroll.setMaximumWidth(430)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(left)
        root.addWidget(scroll)

        # --- preview panel -------------------------------------------------
        right = QWidget(central)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 8, 8, 8)

        split = QSplitter(Qt.Orientation.Vertical, right)

        self.frame_nominal = QGroupBox("A \u00b7 original frame (commanded pointing)", right)
        fn = QVBoxLayout(self.frame_nominal)
        fn.setContentsMargins(4, 4, 4, 4)
        self.lbl_nominal = QLabel(self.frame_nominal)
        self.lbl_nominal.setStyleSheet("color: #9fb4d8;")
        fn.addWidget(self.lbl_nominal)
        self.canvas = ImageCanvas(self.frame_nominal)
        fn.addWidget(self.canvas)
        split.addWidget(self.frame_nominal)

        self.frame_error = QGroupBox("B \u00b7 extra frame (pointing error)", right)
        fe = QVBoxLayout(self.frame_error)
        fe.setContentsMargins(4, 4, 4, 4)
        fe_top = QHBoxLayout()
        fe_top.addStretch(1)
        self.check_align = QCheckBox("Align B to A", self.frame_error)
        fe_top.addWidget(self.check_align)
        fe.addLayout(fe_top)
        self.lbl_error = QLabel(self.frame_error)
        self.lbl_error.setStyleSheet("color: #d8b89f;")
        fe.addWidget(self.lbl_error)
        self.lbl_error_offset = QLabel(self.frame_error)
        self.lbl_error_offset.setStyleSheet("color: #b9c4d8;")
        fe.addWidget(self.lbl_error_offset)
        self.canvas_error = ImageCanvas(self.frame_error)
        fe.addWidget(self.canvas_error)
        split.addWidget(self.frame_error)
        self.frame_error.setVisible(True)

        rv.addWidget(split, 1)
        root.addWidget(right, 1)

        # Status bar
        self.status = self.statusBar()
        self._s_center = QLabel()
        self._s_pointer = QLabel()
        self._s_render = QLabel()
        for w in (self._s_center, self._s_pointer, self._s_render):
            w.setContentsMargins(8, 0, 8, 0)
            self.status.addPermanentWidget(w)
        self._s_render.setText("catalog: loading \u2026")

    def _build_artifact_group(self, left: QWidget) -> None:
        """Checkboxes + parameter spins for the simulated imaging artifacts."""
        group = QGroupBox("Simulated artifacts", left)
        grid = QGridLayout(group)
        grid.setColumnStretch(1, 1)

        grid.addWidget(QLabel("Random seed", group), 0, 0)
        self.spin_seed = _make_ispin(group, 0, 999999, 1, 0)
        grid.addWidget(self.spin_seed, 0, 1)
        self.btn_roll_seed = QPushButton("Roll", group)
        self.btn_roll_seed.setToolTip("Pick a new random seed (re-rolls fixed artifacts)")
        grid.addWidget(self.btn_roll_seed, 0, 2)

        row = 1

        def effect(key: str, title: str, controls) -> None:
            nonlocal row
            check = QCheckBox(title, group)
            check.setChecked(True)  # all simulations on by default
            self._art_check[key] = check
            grid.addWidget(check, row, 0)
            row_params = []
            lay = QHBoxLayout()
            for label_text, widget in controls:
                lay.addWidget(QLabel(label_text, group))
                lay.addWidget(widget)
                row_params.append(widget)
            lay.addStretch(1)
            grid.addLayout(lay, row, 1, 1, 2)
            self._art_param_ids[key] = row_params
            row += 1

        effect(
            "pointing",
            "Extra pointing-error frame",
            [("\u2264 % fov", _make_ispin(group, 0, 10, 1, 10))],
        )
        effect(
            "noise",
            "Noise (bias + random)",
            [
                ("\u03c3", _make_fspin(group, 0.0, 0.15, 3, 0.005, 0.02)),
                ("bias", _make_fspin(group, 0.0, 0.5, 3, 0.01, 0.05)),
            ],
        )
        effect(
            "cr",
            "Cosmic rays",
            [("count", _make_ispin(group, 0, 300, 5, 10))],
        )
        effect(
            "meteor",
            "Meteor trail",
            [("count", _make_ispin(group, 0, 5, 1, 1))],
        )
        effect(
            "satellite",
            "Satellite trail",
            [
                ("count", _make_ispin(group, 0, 5, 1, 1)),
                ("width", _make_fspin(group, 0.3, 4.0, 1, 0.1, 0.9)),
                ("int", _make_fspin(group, 0.0, 1.2, 2, 0.05, 0.3)),
            ],
        )
        effect(
            "dust",
            "CMOS dust specks",
            [
                ("count", _make_ispin(group, 0, 60, 1, 6)),
                ("size%", _make_fspin(group, 0.0, 12.0, 1, 0.5, 3.0)),
            ],
        )
        effect(
            "seeing",
            "Atmospheric seeing",
            [("px", _make_fspin(group, 0.0, 8.0, 2, 0.1, 1.5))],
        )
        effect(
            "spike",
            "Diffraction spikes",
            [
                ("len%", _make_fspin(group, 0.0, 45.0, 0, 1.0, 3.0)),
                ("int", _make_fspin(group, 0.0, 3.0, 1, 0.1, 0.8)),
            ],
        )
        effect(
            "ghost",
            "Ghost reflections",
            [("int", _make_fspin(group, 0.0, 1.5, 2, 0.05, 0.5))],
        )

        left_layout = left.layout()
        left_layout.addWidget(group)
        self._refresh_art_widget_enabled()

    def _artifact_settings(self) -> dict:
        """Read the artifact UI into the flat config dict used by sky_effects."""
        def widget(key: str):
            return self._art_check[key]

        def param_widgets(key: str):
            return self._art_param_ids[key]

        def is_on(key: str) -> bool:
            return widget(key).isChecked()

        def fval(spin: QDoubleSpinBox) -> float:
            return float(spin.value())

        def ival(spin: QSpinBox) -> int:
            return int(spin.value())

        art = {
            "pointing": is_on("pointing"),
            "pointing_offset": ival(param_widgets("pointing")[0]) / 100.0,
            "noise": is_on("noise"),
            "noise_sigma": fval(param_widgets("noise")[0]),
            "noise_bias": fval(param_widgets("noise")[1]),
            "cr": is_on("cr"),
            "cr_count": ival(param_widgets("cr")[0]),
            "meteor": is_on("meteor"),
            "meteor_count": ival(param_widgets("meteor")[0]),
            "satellite": is_on("satellite"),
            "sat_count": ival(param_widgets("satellite")[0]),
            "sat_width": fval(param_widgets("satellite")[1]),
            "sat_brightness": fval(param_widgets("satellite")[2]),
            "dust": is_on("dust"),
            "dust_count": ival(param_widgets("dust")[0]),
            "dust_size": fval(param_widgets("dust")[1]) / 100.0,
            "seeing": is_on("seeing"),
            "seeing_sigma": fval(param_widgets("seeing")[0]),
            "spike": is_on("spike"),
            "spike_len": fval(param_widgets("spike")[0]) / 100.0,
            "spike_int": fval(param_widgets("spike")[1]),
            "ghost": is_on("ghost"),
            "ghost_int": fval(param_widgets("ghost")[0]),
            "seed": int(self.spin_seed.value()),
        }
        return art

    def _refresh_art_widget_enabled(self) -> None:
        if not hasattr(self, "_art_check"):
            return
        for key, params in self._art_param_ids.items():
            enabled = self._art_check[key].isChecked()
            for spin in params:
                spin.setEnabled(enabled)

    def _on_artifact_toggled(self, *_args) -> None:
        self._refresh_art_widget_enabled()
        pointing = self._art_check["pointing"].isChecked()
        self.frame_error.setVisible(pointing)
        self.check_align.setEnabled(pointing)
        self._schedule_render()

    def _on_artifact_changed(self, *_args) -> None:
        self._schedule_render()

    def _roll_seed(self) -> None:
        import random as _random

        self.spin_seed.setValue(_random.randrange(0, 1000000))
        self._force_render()

    # -- data & signals -----------------------------------------------------
    def _load_catalog(self) -> None:
        try:
            ra, dec, mag = load_catalog(DEFAULT_CATALOG)
        except Exception as exc:  # pragma: no cover - dialog path
            QMessageBox.critical(
                self, "Catalog error", f"Could not load {DEFAULT_CATALOG}:\n{exc}"
            )
            ra, dec, mag = np.empty(0), np.empty(0), np.empty(0)
        self._catalog = (ra, dec, mag)
        self.map_widget.set_catalog(ra, dec, mag)
        self.status.showMessage(f"HIP catalog: {len(ra)} stars", 6000)
        self._s_render.setText(f"catalog: {len(ra)} stars")

    def _connect_signals(self) -> None:
        self.map_widget.centerChanged.connect(self._on_map_center)
        self.map_widget.hoverMoved.connect(self._on_pointer_hover)
        self.canvas.recentered.connect(self._on_map_center)
        self.canvas.hoverMoved.connect(self._on_pointer_hover)
        self.canvas_error.recentered.connect(self._on_map_center)
        self.canvas_error.hoverMoved.connect(self._on_pointer_hover)

        for spin in (
            self.spin_ra,
            self.spin_dec,
            self.spin_fov,
            self.spin_roll,
            self.spin_w,
            self.spin_h,
            self.spin_maxmag,
            self.spin_psf,
            self.spin_gain,
        ):
            spin.valueChanged.connect(self._on_param_changed)

        # artifact simulation widgets
        for key, check in self._art_check.items():
            check.toggled.connect(self._on_artifact_toggled)
        for params in self._art_param_ids.values():
            for spin in params:
                spin.valueChanged.connect(self._on_artifact_changed)
        self.spin_seed.valueChanged.connect(self._on_artifact_changed)
        self.btn_roll_seed.clicked.connect(self._roll_seed)

        self.check_live.toggled.connect(self._on_live_toggled)
        self.btn_render.clicked.connect(self._force_render)
        self.btn_save.clicked.connect(self._save_image)
        self.check_align.toggled.connect(self._on_align_toggled)

    def _on_live_toggled(self, checked: bool) -> None:
        if checked:
            self._schedule_render()

    def _on_param_changed(self, *_args) -> None:
        self._sync_map_from_spins()
        self._sync_center_ui()
        self._schedule_render()

    def _on_map_center(self, ra: float, dec: float) -> None:
        self._set_center_spins(ra, dec)
        self._sync_map_from_spins()
        self._sync_center_ui()
        self._schedule_render()

    def _set_center_spins(self, ra: float, dec: float) -> None:
        for spin in (self.spin_ra, self.spin_dec):
            spin.blockSignals(True)
        self.spin_ra.setValue(ra % 360.0)
        self.spin_dec.setValue(min(90.0, max(-90.0, dec)))
        for spin in (self.spin_ra, self.spin_dec):
            spin.blockSignals(False)

    def _sync_map_from_spins(self) -> None:
        self.map_widget.set_center(self.spin_ra.value(), self.spin_dec.value(), emit=False)
        self.map_widget.set_fov(self.spin_fov.value())

    def _sync_center_ui(self) -> None:
        ra, dec = self.spin_ra.value(), self.spin_dec.value()
        self.label_center_hms.setText(
            f"center  {ra_to_hms(ra)}   {dec_to_dms(dec)}   ({ra:.4f}\u00b0, {dec:.4f}\u00b0)"
        )

    def _on_pointer_hover(self, sky) -> None:
        if sky is None:
            self._s_pointer.setText("pointer: -")
        else:
            ra, dec = sky
            self._s_pointer.setText(f"pointer: {ra_to_hms(ra)} {dec_to_dms(dec)}")

    def _params_snapshot(self) -> dict:
        return {
            "ra": self.spin_ra.value(),
            "dec": self.spin_dec.value(),
            "fov": self.spin_fov.value(),
            "roll": self.spin_roll.value(),
            "width": self.spin_w.value(),
            "height": self.spin_h.value(),
            "max_mag": self.spin_maxmag.value(),
            "psf_sigma": self.spin_psf.value(),
            "gain": self.spin_gain.value(),
        }

    def _resolve_pointing(self, snap: dict) -> dict:
        """Apply the pointing-error simulation -> the actual rendered RA/Dec/roll.

        When enabled, the output image centre is nudged by up to
        ``pointing_offset * fov`` degrees in a random direction and the camera
        roll is re-randomised, each exposure (tracking / field-rotation error).
        """
        ra, dec, fov = snap["ra"], snap["dec"], snap["fov"]
        art = snap.get("art") or {}
        if not art.get("pointing", False):
            return {
                "ra": ra,
                "dec": dec,
                "fov": fov,
                "roll": snap["roll"],
            }
        frame = int(art.get("frame", 0))
        seed = int(art.get("seed", 0))
        rng = np.random.RandomState((seed * 37 + frame * 131071) & 0xFFFFFFFF)
        max_angle = math.radians(fov * float(art.get("pointing_offset", 0.1)))
        if max_angle > 0.0:
            phi = rng.uniform(0.0, 2.0 * math.pi)
            rho = math.tan(max_angle) * math.sqrt(rng.uniform(0.0, 1.0))
            ra, dec = gnomonic_inverse_pt(
                rho * math.cos(phi), rho * math.sin(phi), ra, dec
            )
        roll = float(rng.uniform(0.0, 360.0))
        return {"ra": float(ra), "dec": float(dec), "fov": float(fov), "roll": roll}

    def _pixel_offset(self, geo_a: dict, geo_b: dict, w: int, h: int) -> tuple[int, int, float]:
        """Where A's centre appears inside B's frame, as a signed pixel offset.

        Returns (dx, dy, droll) in B's image coordinates (right/down positive,
        rounded to whole pixels): ``dx/dy`` = pixel position of A's centre
        minus B's image centre; ``droll`` (deg) = B roll - A roll.
        """
        half = math.tan(math.radians(float(geo_a["fov"]) / 2.0))
        x, y, _ = gnomonic_project(
            np.asarray([geo_a["ra"]]), np.asarray([geo_a["dec"]]),
            geo_b["ra"], geo_b["dec"],
        )
        xr, yr = apply_roll(x, y, geo_b["roll"])
        xp = float(((xr + half) / (2.0 * half) * (w - 1))[0])
        yp = float(((half - yr) / (2.0 * half) * (h - 1))[0])
        dx = int(round(xp - (w - 1) / 2.0))
        dy = int(round(yp - (h - 1) / 2.0))
        droll = (float(geo_b["roll"]) - float(geo_a["roll"]) + 180.0) % 360.0 - 180.0
        return dx, dy, droll

    def _align_b_to_a(self, rgb_b: np.ndarray, geo_a: dict, geo_b: dict) -> np.ndarray:
        """Resample frame B onto A's grid (translate + de-rotate) -> aligned B.

        Every output pixel (A's commanded pointing grid) is mapped to the sky
        through A's geometry and then back-projected into B, sampling B with
        bilinear interpolation. Result matches A in centre and orientation.
        """
        h, w = rgb_b.shape[:2]
        half = math.tan(math.radians(float(geo_a["fov"]) / 2.0))

        col = np.arange(w, dtype=np.float64)
        row = np.arange(h, dtype=np.float64)
        pxv, pyv = np.meshgrid(col, row)

        # pixel -> A's tangent-plane coords (undo A roll -> east/north frame)
        x = (pxv / max(1, w - 1)) * 2.0 * half - half
        y = half - (pyv / max(1, h - 1)) * 2.0 * half
        xu, yu = apply_roll(x, y, -geo_a["roll"])
        ra, dec = gnomonic_inverse_vec(xu, yu, geo_a["ra"], geo_a["dec"])

        # sky -> B frame pixel
        xb, yb, _ = gnomonic_project(ra, dec, geo_b["ra"], geo_b["dec"])
        xb, yb = apply_roll(xb, yb, geo_b["roll"])
        px_b = (xb + half) / (2.0 * half) * (w - 1)
        py_b = (half - yb) / (2.0 * half) * (h - 1)

        # bilinear sample
        x0 = np.floor(px_b).astype(np.int64)
        y0 = np.floor(py_b).astype(np.int64)
        wx = px_b - x0
        wy = py_b - y0
        ok = (x0 >= 0) & (x0 + 1 < w) & (y0 >= 0) & (y0 + 1 < h)
        out = np.zeros_like(rgb_b)
        if ok.any():
            iy = y0[ok]
            ix = x0[ok]
            wxl = wx[ok, None]
            wyl = wy[ok, None]
            c00 = rgb_b[iy, ix] * ((1.0 - wxl) * (1.0 - wyl))
            c01 = rgb_b[iy, np.clip(ix + 1, 0, w - 1)] * (wxl * (1.0 - wyl))
            c10 = rgb_b[np.clip(iy + 1, 0, h - 1), ix] * ((1.0 - wxl) * wyl)
            c11 = rgb_b[np.clip(iy + 1, 0, h - 1), np.clip(ix + 1, 0, w - 1)] * (wxl * wyl)
            out[ok] = c00 + c01 + c10 + c11
        return np.clip(out, 0.0, 1.0)

    # -- rendering control --------------------------------------------------
    def _force_render(self) -> None:
        self._debounce.stop()
        self._launch_render()

    def _schedule_render(self) -> None:
        if not self.check_live.isChecked():
            return
        self._debounce.start()

    def _launch_render(self) -> None:
        if self._catalog is None or len(self._catalog[0]) == 0:
            return
        self._token += 1
        self._need_render = False
        if self._busy:
            self._need_render = True  # newest request will win when we are free
            return
        snap = self._params_snapshot()
        # reserve one exposure id per output frame (nominal + optional error)
        n_frames = 2 if self._art_check["pointing"].isChecked() else 1
        self._frame += n_frames
        art = self._artifact_settings()
        art["frame"] = self._frame - n_frames + 1
        snap["art"] = art
        w, h = snap["width"], snap["height"]
        self._s_render.setText(f"rendering {w}\u00d7{h} \u2026")
        self.status.showMessage("Rendering \u2026")
        self._busy = True
        thread = threading.Thread(
            target=self._render_worker, args=(snap, self._token), daemon=True
        )
        thread.start()

    def _render_frame(self, snap: dict) -> tuple[QImage, int]:
        rgb, stars = self._compute_image(snap)
        rgb = sky_effects.apply_effects(rgb, stars, snap.get("art") or {})
        return rgb_to_qimage(rgb), stars["count"]

    def _render_worker(self, snap: dict, token: int) -> None:
        try:
            art = snap.get("art") or {}
            pointing = bool(art.get("pointing", False))
            nominal_frame = int(art.get("frame", 0))

            t0 = time.perf_counter()
            qnominal, stars_n = self._render_frame(snap)

            error_image = None
            error_geo = None
            stars_e = 0
            if pointing:
                snap_err = dict(snap)
                art_err = dict(art)
                art_err["frame"] = nominal_frame + 1
                snap_err["art"] = art_err
                error_geo = self._resolve_pointing(snap_err)
                snap_err.update(error_geo)
                error_image, stars_e = self._render_frame(snap_err)
            dt = time.perf_counter() - t0

            qimage = qnominal
        except Exception as exc:  # noqa: BLE001 - report back to GUI thread
            self._bridge.done.emit({"ok": False, "token": token, "error": str(exc)})
            return
        nominal_geo = {
            "ra": snap["ra"],
            "dec": snap["dec"],
            "fov": snap["fov"],
            "roll": snap["roll"],
        }
        self._bridge.done.emit(
            {
                "ok": True,
                "token": token,
                "image": qimage,
                "stars": stars_n,
                "stars_error": stars_e,
                "dt": dt,
                "geo": nominal_geo,
                "error_image": error_image,
                "error_geo": error_geo,
                "pointing_on": pointing,
            }
        )

    def _compute_image(self, snap: dict) -> tuple[np.ndarray, dict]:
        """Pure-numpy render; mirrors render_sky_patch.main() exactly.

        Returns the float RGB image plus the projected star pixel info that
        the artifact simulator needs (spikes / ghosts use star positions).
        """
        ra_all, dec_all, mag_all = self._catalog
        max_mag = snap["max_mag"]
        ra0, dec0, fov = snap["ra"], snap["dec"], snap["fov"]
        width_px = int(snap["width"])
        height_px = int(snap["height"])

        mask = mag_all <= max_mag
        ra, dec, mag = ra_all[mask], dec_all[mask], mag_all[mask]

        radius = fov * math.sqrt(2.0) * 0.5
        dist = angular_distance_deg(ra, dec, ra0, dec0)
        keep = dist <= radius
        ra, dec, mag = ra[keep], dec[keep], mag[keep]

        x, y, visible = gnomonic_project(ra, dec, ra0, dec0)
        x, y, mag = x[visible], y[visible], mag[visible]
        x, y = apply_roll(x, y, snap["roll"])

        half_extent = math.tan(math.radians(fov / 2.0))
        inside = (np.abs(x) <= half_extent) & (np.abs(y) <= half_extent)
        x, y, mag = x[inside], y[inside], mag[inside]

        rgb = render_psf_image(
            x=x,
            y=y,
            mag=mag,
            half_extent=half_extent,
            width_px=width_px,
            height_px=height_px,
            psf_sigma=snap["psf_sigma"],
            gain=snap["gain"],
        )

        # Pixel-space star coordinates + flux (for artifact overlays).
        xp = (x + half_extent) / (2.0 * half_extent) * (width_px - 1)
        yp = (half_extent - y) / (2.0 * half_extent) * (height_px - 1)
        on_frame = (
            (xp >= 0) & (xp < width_px) & (yp >= 0) & (yp < height_px)
        )
        flux = np.clip(np.power(10.0, -0.4 * (mag - 8.0)), 0.03, 80.0)
        stars = {
            "x": np.asarray(xp[on_frame], dtype=np.float64),
            "y": np.asarray(yp[on_frame], dtype=np.float64),
            "flux": np.asarray(flux[on_frame], dtype=np.float64),
            "mag": np.asarray(mag[on_frame], dtype=np.float64),
            "count": int(len(mag)),
        }
        return rgb, stars

    def _on_render_done(self, payload: dict) -> None:
        self._busy = False
        stale = payload.get("token") != self._token
        if payload.get("ok") and not stale:
            image = payload["image"]
            self._current_image = image
            self.canvas.set_image(image, payload.get("geo") or self._params_snapshot())
            self._s_render.setText(
                f"{image.width()}\u00d7{image.height()}  \u00b7  "
                f"{payload['stars']} stars  \u00b7  {payload['dt'] * 1000:.0f} ms"
            )
            self.status.showMessage("Ready", 3000)

            geo_n = payload.get("geo") or {}
            self.lbl_nominal.setText(
                f"commanded center  {ra_to_hms(geo_n.get('ra', 0.0))}  "
                f"{dec_to_dms(geo_n.get('dec', 0.0))}  \u00b7  roll {geo_n.get('roll', 0.0):.1f}\u00b0"
            )

            error_image = payload.get("error_image")
            error_geo = payload.get("error_geo")
            self._error_image = error_image
            show_error = bool(payload.get("pointing_on")) and error_image is not None
            self.frame_error.setVisible(show_error)
            if show_error and error_geo is not None:
                # Keep raw B (QImage + float RGB) so "Align B to A" can be
                # applied instantly later without re-rendering.
                self._b_geo_n = dict(payload.get("geo") or {})
                self._b_geo_b = dict(error_geo)
                self._b_rgb = qimage_to_float_rgb(error_image)
                self._refresh_b_display()
            else:
                self._b_rgb = None
                self._b_geo_n = None
                self._b_geo_b = None
                self.lbl_error_offset.setText("")
                self.lbl_error.setText("")
                self.canvas_error.clear_image()
        elif not payload.get("ok") and not stale:
            self.status.showMessage(f"Render error: {payload.get('error', '?')}", 8000)
            self._s_render.setText("render failed")
        if self._need_render and not self._busy:
            self._launch_render()

    def _on_align_toggled(self, *_args) -> None:
        # Registration only re-uses the current B frame - no re-render.
        self._refresh_b_display()

    def _refresh_b_display(self) -> None:
        if self._b_rgb is None or not self._b_geo_n or not self._b_geo_b:
            return
        geo_n, geo_b = self._b_geo_n, self._b_geo_b
        h, w = self._b_rgb.shape[:2]
        dx, dy, droll = self._pixel_offset(geo_n, geo_b, w, h)
        if self.check_align.isChecked():
            aligned = self._align_b_to_a(self._b_rgb, geo_n, geo_b)
            self.canvas_error.set_image(rgb_to_qimage(aligned), geo_n)
            self.lbl_error.setText(
                f"B \u2192 A aligned (on A grid)  \u00b7  shift ({-dx:+d},{-dy:+d}) px, "
                f"rotation {-droll:+.1f}\u00b0"
            )
            self.lbl_error_offset.setText(
                f"residual after registration \u2248 0 px / 0\u00b0  \u00b7  "
                f"original offset: \u0394x {dx:+d} px,  \u0394y {dy:+d} px,  "
                f"\u0394roll {droll:+.1f}\u00b0"
            )
        else:
            self.canvas_error.set_image(self._error_image, geo_b)
            self.lbl_error.setText(
                f"actual center  {ra_to_hms(geo_b['ra'])}  {dec_to_dms(geo_b['dec'])}  "
                f"\u00b7  roll {geo_b['roll']:.1f}\u00b0"
            )
            self.lbl_error_offset.setText(
                f"A centre in B frame:  \u0394x {dx:+d} px,  \u0394y {dy:+d} px   \u00b7   "
                f"\u0394roll {droll:+.1f}\u00b0"
                f"   (shift B by {-dx:+d}, {-dy:+d} px to align A/B)"
            )

    # -- save ---------------------------------------------------------------
    def _save_image(self) -> None:
        if self._current_image is None:
            QMessageBox.information(self, "Save image", "No image rendered yet.")
            return
        snap = self._params_snapshot()
        out_dir = SCRIPT_DIR / "test_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        default = default_save_name(snap["ra"], snap["dec"], snap["fov"], snap["roll"])
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save rendered image",
            str(out_dir / default),
            "PNG image (*.png);;BMP image (*.bmp);;JPEG image (*.jpg)",
        )
        if not path:
            return
        ok = self._current_image.save(path)
        if not ok:
            QMessageBox.warning(self, "Save image", f"Failed to write:\n{path}")
            return
        saved = [path]
        if self._error_image is not None:
            p = Path(path)
            err_path = str(p.with_name(f"{p.stem}_err{p.suffix}"))
            if self._error_image.save(err_path):
                saved.append(err_path)
        if len(saved) > 1:
            self.status.showMessage(
                f"Saved nominal + pointing-error frames:\n{saved[0]}\n{saved[1]}", 6000
            )
        else:
            self.status.showMessage(f"Saved {saved[0]}", 5000)


def rgb_to_qimage(rgb: np.ndarray) -> QImage:
    """Float RGB (H, W, 3) in [0, 1] -> deep-copied RGB32 QImage."""
    u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w = u8.shape[:2]
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 0] = u8[..., 2]  # B
    arr[..., 1] = u8[..., 1]  # G
    arr[..., 2] = u8[..., 0]  # R
    arr[..., 3] = 255
    qi = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGB32)
    return qi.copy()


def qimage_to_float_rgb(qi: QImage) -> np.ndarray:
    """RGB32 QImage -> float RGB (H, W, 3) in [0, 1]."""
    img = qi.convertToFormat(QImage.Format.Format_RGB32)
    h, w = img.height(), img.width()
    bgra = np.frombuffer(bytes(img.constBits()), dtype=np.uint8).reshape((h, w, 4))
    out = np.empty((h, w, 3), dtype=np.float32)
    out[..., 0] = bgra[..., 2] / 255.0  # R
    out[..., 1] = bgra[..., 1] / 255.0  # G
    out[..., 2] = bgra[..., 0] / 255.0  # B
    return out


def apply_dark_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QColor(24, 28, 40))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QColor(210, 218, 232))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QColor(14, 17, 26))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QColor(28, 32, 46))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QColor(210, 218, 232))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QColor(42, 48, 66))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QColor(210, 218, 232))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QColor(80, 110, 200))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QColor(120, 128, 145))
    app.setPalette(palette)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Sky Patch Renderer")
    apply_dark_theme(app)
    window = SkyPatchGui()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
