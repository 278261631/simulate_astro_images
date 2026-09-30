#!/usr/bin/env python3
"""Visual inspector for the single-image OB / unusable-region segmenter.

Loads models_ob/best.pt and browses the data_256 splits (or any external
image), showing for each frame:  input | GT mask | predicted mask | overlay.

Run:
    python train_and_data/train_ob/validate_ob_gui.py
    python train_and_data/train_ob/validate_ob_gui.py --ckpt path/to/best.pt --data ../data_256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSlider, QSpinBox, QVBoxLayout, QWidget,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model_ob import OBNet  # noqa: E402


def stretch(img: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(img, (lo_pct, hi_pct))
    return np.clip((img.astype(np.float32) - lo) / max(1e-6, hi - lo), 0.0, 1.0)


def gray_qimage(u8: np.ndarray) -> QImage:
    u8 = np.ascontiguousarray(u8)
    return QImage(u8.data, u8.shape[1], u8.shape[0], u8.strides[0],
                  QImage.Format.Format_Grayscale8)


def rgb_qimage(rgb_u8: np.ndarray) -> QImage:
    rgb_u8 = np.ascontiguousarray(rgb_u8)
    return QImage(rgb_u8.data, rgb_u8.shape[1], rgb_u8.shape[0],
                  rgb_u8.strides[0], QImage.Format.Format_RGB888)


def mask_pixmap(mask: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(gray_qimage((mask.astype(np.uint8) * 255)))


class OBGui(QMainWindow):
    def __init__(self, ckpt: Path, data: Path, base: int) -> None:
        super().__init__()
        self.setWindowTitle("OB / unusable-region segmenter — inspector")
        self.resize(1180, 460)
        self.data = data
        self.thresh = 0.5
        self.splits = {}
        self.split = "val"
        self.side = 0  # 0 = A, 1 = B
        self.index = 0
        self.ext_frame = None  # external image override

        obj = torch.load(ckpt, map_location="cpu")
        base = int(obj.get("base", base)) if isinstance(obj, dict) else base
        sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        self.net = OBNet(base=base).eval()
        self.net.load_state_dict(sd)
        self._load_split("val")
        self._load_split("test")
        self._build_ui()
        self._update()

    # -- data ---------------------------------------------------------------
    def _load_split(self, split: str) -> None:
        a = np.load(self.data / f"{split}_a.npy", mmap_mode="r")
        b = np.load(self.data / f"{split}_b.npy", mmap_mode="r")
        z = np.load(self.data / f"{split}_meta.npz")
        self.splits[split] = (a, b, np.asarray(z["ob_a"]), np.asarray(z["ob_b"]))

    def _frame_gt(self):
        a, b, ma, mb = self.splits[self.split]
        if self.side == 0:
            return a[self.index], ma[self.index]
        return b[self.index], mb[self.index]

    # -- ui -----------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("split"))
        self.cb_split = QComboBox(); self.cb_split.addItems(["val", "test"])
        self.cb_split.currentTextChanged.connect(self._on_split)
        bar.addWidget(self.cb_split)
        bar.addWidget(QLabel("frame"))
        self.cb_side = QComboBox(); self.cb_side.addItems(["A", "B"])
        self.cb_side.currentIndexChanged.connect(self._on_side)
        bar.addWidget(self.cb_side)
        self.btn_prev = QPushButton("◀ prev"); self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next = QPushButton("next ▶"); self.btn_next.clicked.connect(lambda: self._step(1))
        self.btn_rand = QPushButton("random"); self.btn_rand.clicked.connect(self._random)
        bar.addWidget(self.btn_prev); bar.addWidget(self.btn_next); bar.addWidget(self.btn_rand)
        bar.addWidget(QLabel("index"))
        self.spin = QSpinBox(); self.spin.setRange(0, 10**6)
        self.spin.valueChanged.connect(self._on_index)
        bar.addWidget(self.spin)
        bar.addWidget(QLabel("thresh"))
        self.sl = QSlider(Qt.Orientation.Horizontal); self.sl.setRange(5, 95); self.sl.setValue(50)
        self.sl.setFixedWidth(120); self.sl.valueChanged.connect(self._on_thresh)
        bar.addWidget(self.sl)
        self.btn_load = QPushButton("load image…"); self.btn_load.clicked.connect(self._load_image)
        bar.addWidget(self.btn_load)
        bar.addStretch(1)
        root.addLayout(bar)

        panels = QHBoxLayout()
        self.lbl_frame = self._panel("input")
        self.lbl_gt = self._panel("GT mask")
        self.lbl_pred = self._panel("pred mask")
        self.lbl_ov = self._panel("overlay (GT green / pred red)")
        for w in (self.lbl_frame, self.lbl_gt, self.lbl_pred, self.lbl_ov):
            panels.addWidget(w[1])
        root.addLayout(panels, 1)

        self.status = self.statusBar()
        self._s = QLabel()
        self.status.addPermanentWidget(self._s)

        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Left), self, lambda: self._step(-1))
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Right), self, lambda: self._step(1))

    def _panel(self, title: str):
        box = QWidget()
        v = QVBoxLayout(box); v.setContentsMargins(4, 4, 4, 4)
        t = QLabel(title); t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img = QLabel(); img.setMinimumSize(260, 260)
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img.setStyleSheet("background:#101018;")
        img.setScaledContents(True)
        v.addWidget(t); v.addWidget(img, 1)
        return t, box, img

    # -- events -------------------------------------------------------------
    def _on_split(self, s): self.split = s; self.ext_frame = None; self._clamp(); self._update()
    def _on_side(self, i): self.side = i; self.ext_frame = None; self._clamp(); self._update()
    def _on_index(self, v): self.index = v; self.ext_frame = None; self._update()
    def _on_thresh(self, v): self.thresh = v / 100.0; self._update()

    def _n(self):
        a = self.splits[self.split][0]
        return a.shape[0]

    def _clamp(self):
        self.index = max(0, min(self.index, self._n() - 1))
        self.spin.blockSignals(True); self.spin.setValue(self.index); self.spin.blockSignals(False)

    def _step(self, d):
        self.ext_frame = None
        self.index = (self.index + d) % self._n()
        self.spin.blockSignals(True); self.spin.setValue(self.index); self.spin.blockSignals(False)
        self._update()

    def _random(self):
        self.ext_frame = None
        a, b, ma, mb = self.splits[self.split]
        m = ma if self.side == 0 else mb
        # prefer frames that actually have a border
        cand = np.where(m.reshape(len(m), -1).any(1))[0]
        pool = cand if (cand is not None and len(cand) and np.random.rand() < 0.7) else np.arange(self._n())
        self.index = int(pool[np.random.randint(len(pool))])
        self.spin.blockSignals(True); self.spin.setValue(self.index); self.spin.blockSignals(False)
        self._update()

    def _load_image(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open frame", str(self.data),
                                           "Images (*.npy *.png *.jpg *.bmp *.tif)")
        if not f:
            return
        p = Path(f)
        if p.suffix.lower() == ".npy":
            im = np.load(p)
            im = im.mean(2) if im.ndim == 3 else im
            im = im.astype(np.uint8)
        else:
            from PIL import Image
            im = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        self.ext_frame = im
        self._update()

    # -- inference + render -------------------------------------------------
    def _predict(self, frame: np.ndarray) -> np.ndarray:
        f = frame
        if f.shape != (256, 256):
            from PIL import Image
            f = np.asarray(Image.fromarray(f).resize((256, 256), Image.BILINEAR))
        x = torch.from_numpy(f.astype(np.float32)[None, None] / 255.0)
        with torch.no_grad():
            prob = torch.sigmoid(self.net(x))[0, 0].numpy()
        return prob > self.thresh

    def _update(self):
        if self.ext_frame is not None:
            frame = self.ext_frame; gt = None
            title = "external image"
            self.lbl_gt[2].setText("n/a")
            self.lbl_gt[0].setText("GT mask (n/a)")
        else:
            frame, gt = self._frame_gt()
            title = f"{self.split} {'A' if self.side==0 else 'B'} #{self.index}"
            self.lbl_gt[2].setText("")
            self.lbl_gt[0].setText("GT mask")

        pred = self._predict(frame)
        disp = (stretch(frame) * 255).astype(np.uint8)
        self.lbl_frame[2].setPixmap(QPixmap.fromImage(gray_qimage(disp)))

        if gt is not None:
            self.lbl_gt[2].setPixmap(mask_pixmap(gt))
            inter = int((pred & gt.astype(bool)).sum()); uni = int((pred | gt.astype(bool)).sum())
            iou = inter / uni if uni > 0 else 1.0
            msg = (f"{title}   IoU {iou:.3f}   GT {100*gt.mean():.1f}%   "
                   f"pred {100*pred.mean():.1f}%")
        else:
            self.lbl_gt[2].setText("—")
            msg = f"{title}   pred {100*pred.mean():.1f}% (no GT)"

        self.lbl_pred[2].setPixmap(mask_pixmap(pred))

        # overlay: input in gray, GT green, pred red, both yellow (4x upsample)
        base = (stretch(frame) * 255).astype(np.float32)
        rgb = np.stack([base, base, base], axis=-1)
        if gt is not None:
            g = np.repeat(np.repeat(gt.astype(bool), 4, 0), 4, 1)
        else:
            g = np.zeros((256, 256), bool)
        p = np.repeat(np.repeat(pred, 4, 0), 4, 1)
        rgb[g & ~p] = [0, 220, 0]
        rgb[p & ~g] = [240, 40, 40]
        rgb[p & g] = [240, 220, 0]
        self.lbl_ov[2].setPixmap(QPixmap.fromImage(rgb_qimage(np.clip(rgb, 0, 255).astype(np.uint8))))

        self._s.setText(msg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=HERE / "models_ob" / "best.pt")
    ap.add_argument("--data", type=Path, default=HERE.parent / "data_256")
    ap.add_argument("--base", type=int, default=32)
    args = ap.parse_args()
    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")
    if not (args.data / "val_meta.npz").exists():
        raise SystemExit(f"data not found: {args.data}")

    app = QtWidgets.QApplication(sys.argv)
    win = OBGui(args.ckpt, args.data, args.base)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
