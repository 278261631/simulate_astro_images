#!/usr/bin/env python3
"""PySide6 GUI to validate a trained PairRegNet on generated (A, B) pairs.

Workflow
--------
1. Point at a trained checkpoint (a plain ``best.pt`` state dict or a full
   ``checkpoint_best.pt`` dict produced by ``train.py``) and at a dataset
   directory produced by ``generate_dataset.py`` (defaults to ``models/`` and
   ``data/`` next to this script).
2. Pick a split (train / val / test) and press *Run split*. Every pair is
   scored in a background thread and an error table + summary metrics are
   shown (same statistics ``train.py`` prints).
3. Click a table row to inspect that pair:
       A             frame A, display stretched
       B             frame B, display stretched
       GT align      B warped with the *ground truth* (dx, dy, droll), shown
                     in red blended over A in green
       Pred align    same but with the *predicted* transform
   Where the two channels coincide the blend looks neutral/yellow; residual
   mis-registration shows up as red or green fringes on star edges, so you can
   judge alignment quality at a glance.

Run:
    python validate_model_gui.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model import PairRegNet, decode, preprocess  # noqa: E402

DEFAULT_MODEL = HERE / "models" / "best.pt"
DEFAULT_DATA = HERE / "data"


# ---------------------------------------------------------------------------
# Small shared helpers (duplicated from train.py so the GUI is self-contained)
# ---------------------------------------------------------------------------


def wrap180(a: np.ndarray) -> np.ndarray:
    return ((a + 180.0) % 360.0) - 180.0


def error_metrics(pred: np.ndarray, lab: np.ndarray) -> dict:
    e = np.abs(pred - lab)
    mag = np.hypot(e[:, 0], e[:, 1])
    ang = np.abs(((pred[:, 2] - lab[:, 2] + 180.0) % 360.0) - 180.0)
    return {
        "mae_dx": float(np.mean(e[:, 0])),
        "mae_dy": float(np.mean(e[:, 1])),
        "mae_mag": float(np.mean(mag)),
        "med_mag": float(np.median(mag)),
        "p90_mag": float(np.percentile(mag, 90)),
        "mae_roll": float(np.mean(ang)),
        "med_roll": float(np.median(ang)),
        "p90_roll": float(np.percentile(ang, 90)),
        "frac<1px": float(np.mean(mag < 1.0)),
        "frac<2px": float(np.mean(mag < 2.0)),
        "frac<0.5d": float(np.mean(ang < 0.5)),
        "frac<1.0d": float(np.mean(ang < 1.0)),
    }


def fmt_metrics(m: dict) -> str:
    return (
        f"dx {m['mae_dx']:.2f} dy {m['mae_dy']:.2f} |shift| {m['med_mag']:.2f}px"
        f" (p90 {m['p90_mag']:.2f}) | roll {m['med_roll']:.2f}\u00b0 (p90 {m['p90_roll']:.2f})"
        f" | <1px {m['frac<1px']:.1%} <1\u00b0 {m['frac<1.0d']:.1%}"
    )


def display_stretch(u8: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(u8, (0.5, 99.5))
    x = (u8.astype(np.float32) - lo) / max(1e-6, hi - lo)
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load_checkpoint(path: Path) -> tuple[PairRegNet, dict | None]:
    """Load either a raw state dict or the full checkpoint dict train.py writes."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        sd = obj["model"]
        info = obj
    else:
        sd = obj
        info = None
    net = PairRegNet()
    net.load_state_dict(sd)
    net.eval()
    return net, info


# ---------------------------------------------------------------------------
# Image / warping helpers
# ---------------------------------------------------------------------------


def warp_b_onto_a(b_u8: np.ndarray, dx: float, dy: float, droll: float) -> np.ndarray:
    """Warp frame B so its star field aligns with frame A.

    ``cv2.warpAffine`` maps *source -> destination* (content is pushed by the
    matrix).  Given the generation label convention (A's centre content sits at
    B's centre + (dx, dy), B roll = A roll + droll), the content transform that
    brings B onto A was chosen empirically on real pairs as: rotate the content
    by +droll about the frame centre, then push it by (-dx, -dy).  This is the
    sign combination that maximises A/aligned-B correlation on the dataset.
    """
    import cv2  # optional heavy dependency, required only for this view

    h, w = b_u8.shape
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), droll, 1.0)
    m[0, 2] += -dx
    m[1, 2] += -dy
    return cv2.warpAffine(b_u8, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def blend_ab(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    """A in green, warped-B in red -> colour fringes reveal mis-registration."""
    out = np.zeros(a_u8.shape + (3,), dtype=np.uint8)
    out[..., 0] = b_u8
    out[..., 1] = a_u8
    out[..., 2] = np.maximum(a_u8, b_u8) >> 1
    return out


def to_pixmap(arr, max_side: int) -> QPixmap:
    if arr.ndim == 2:
        h, w = arr.shape
        img = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    else:
        h, w, _ = arr.shape
        img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    pm = QPixmap.fromImage(img)
    return pm.scaled(max_side, max_side, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ---------------------------------------------------------------------------
# Background evaluation worker
# ---------------------------------------------------------------------------


class EvalWorker(QThread):
    progress = Signal(int, int)          # current, total
    done = Signal(object)                # list[dict] per-sample records
    failed = Signal(str)

    def __init__(self, net: PairRegNet, a: np.ndarray, b: np.ndarray,
                 labels: np.ndarray, meta: dict | None, batch: int = 64,
                 parent=None) -> None:
        super().__init__(parent)
        self.net = net
        self.a = a
        self.b = b
        self.labels = labels
        self.meta = meta
        self.batch = batch

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            n = len(self.a)
            pairs = torch.stack([preprocess(self.a[i], self.b[i]) for i in range(n)])
            pred = np.zeros((n, 3), dtype=np.float64)
            with torch.no_grad():
                for s in range(0, n, self.batch):
                    if self.isInterruptionRequested():
                        return
                    e = min(s + self.batch, n)
                    out = decode(self.net(pairs[s:e])).numpy()
                    pred[s:e] = out
                    self.progress.emit(e, n)
            gt = self.labels
            records = []
            for i in range(n):
                e = pred[i] - gt[i]
                records.append(
                    {
                        "i": int(i),
                        "gt_dx": float(gt[i, 0]),
                        "gt_dy": float(gt[i, 1]),
                        "gt_roll": float(gt[i, 2]),
                        "dx": float(pred[i, 0]),
                        "dy": float(pred[i, 1]),
                        "roll": float(pred[i, 2]),
                        "err_dx": float(e[0]),
                        "err_dy": float(e[1]),
                        "err_roll": float(((e[2] + 180.0) % 360.0) - 180.0),
                        "mag": float(math.hypot(e[0], e[1])),
                        "elapsed": time.perf_counter() - t0,
                    }
                )
            self.done.emit(records)
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class ValidatorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PairRegNet validator  (A, B) -> dx, dy, droll")
        self.resize(1320, 920)

        self.net: PairRegNet | None = None
        self.split_a: np.ndarray | None = None
        self.split_b: np.ndarray | None = None
        self.split_lab: np.ndarray | None = None
        self.split_meta: dict | None = None
        self.records: list[dict] = []
        self.worker: EvalWorker | None = None

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.addLayout(self._build_top())
        self.status = QLabel("load a model, pick a split, then press Run split")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        split = QSplitter(Qt.Vertical, central)
        split.addWidget(self._build_image_panel())
        split.addWidget(self._build_result_panel())
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([560, 340])
        root.addWidget(split, 1)

        self._refresh_controls()

    # -- construction --------------------------------------------------------

    def _build_top(self) -> QHBoxLayout:
        top = QHBoxLayout()

        def label(text: str) -> QLabel:
            lb = QLabel(text)
            return lb

        top.addWidget(label("Model"))
        self.ed_model = QLineEdit(str(DEFAULT_MODEL))
        self.ed_model.setMinimumWidth(360)
        top.addWidget(self.ed_model)
        bt_model = QPushButton("Browse\u2026")
        bt_model.clicked.connect(self._browse_model)
        top.addWidget(bt_model)

        bt_load = QPushButton("Load model")
        bt_load.clicked.connect(self._load_model)
        top.addWidget(bt_load)

        top.addSpacing(16)
        top.addWidget(label("Data dir"))
        self.ed_data = QLineEdit(str(DEFAULT_DATA))
        self.ed_data.setMinimumWidth(280)
        top.addWidget(self.ed_data)
        bt_data = QPushButton("Browse\u2026")
        bt_data.clicked.connect(self._browse_data)
        top.addWidget(bt_data)

        top.addSpacing(16)
        self.cmb_split = QComboBox()
        self.cmb_split.addItems(["train", "val", "test"])
        top.addWidget(self.cmb_split)

        self.bt_run = QPushButton("Run split")
        self.bt_run.clicked.connect(self._run_split)
        top.addWidget(self.bt_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setMaximumWidth(220)
        top.addWidget(self.progress)
        top.addStretch(1)
        return top

    def _build_image_panel(self) -> QWidget:
        box = QWidget()
        grid = QGridLayout(box)

        def pane(title: str) -> tuple[QGroupBox, QLabel]:
            gb = QGroupBox(title)
            lb = QLabel("--")
            lb.setAlignment(Qt.AlignCenter)
            lb.setMinimumSize(360, 220)
            lay = QVBoxLayout(gb)
            lay.addWidget(lb)
            return gb, lb

        titles = ["A", "B", "B aligned to A  (ground truth)", "B aligned to A  (prediction)"]
        self._panes = {}
        pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for t, (r, c) in zip(titles, pos):
            gb, lb = pane(t)
            self._panes[t] = lb
            grid.addWidget(gb, r, c)
        self.pair_info = QLabel("pair --")
        self.pair_info.setWordWrap(True)
        grid.addWidget(self.pair_info, 2, 0, 1, 2)
        return box

    def _build_result_panel(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)

        row = QHBoxLayout()
        self.lbl_summary = QLabel("no run yet")
        self.lbl_summary.setWordWrap(True)
        row.addWidget(self.lbl_summary, 1)
        bt_sort = QPushButton("Worst 10 by |shift|")
        bt_sort.clicked.connect(self._show_worst)
        row.addWidget(bt_sort)
        lay.addLayout(row)

        cols = ["idx", "gt dx", "gt dy", "gt roll", "pred dx", "pred dy", "pred roll",
                "err dx", "err dy", "err roll", "|shift| px"]
        self.table = QTableWidget(0, len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self._on_select_row)
        lay.addWidget(self.table, 1)
        return box

    # -- actions ---------------------------------------------------------------

    def _browse_model(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "model checkpoint", str(DEFAULT_MODEL),
            "checkpoint (*.pt *.pth);;all files (*)")
        if p:
            self.ed_model.setText(p)

    def _browse_data(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "dataset directory", str(DEFAULT_DATA))
        if p:
            self.ed_data.setText(p)

    def _load_model(self) -> None:
        path = Path(self.ed_model.text())
        if not path.exists():
            QMessageBox.warning(self, "model", f"file not found:\n{path}")
            return
        try:
            self.net, info = load_checkpoint(path)
        except Exception as exc:
            QMessageBox.critical(self, "model", f"failed to load {path.name}:\n{exc}")
            return
        n_params = sum(p.numel() for p in self.net.parameters())
        extra = ""
        if info is not None:
            m = info.get("metrics")
            if m:
                extra = f"\nembedded val metrics from training: {fmt_metrics(m)}"
        self.status.setText(
            f"loaded {path.name} ({n_params / 1e6:.1f}M params){extra}")
        self._refresh_controls()

    def _run_split(self) -> None:
        if self.net is None:
            QMessageBox.information(self, "model", "load a model first")
            return
        data_dir = Path(self.ed_data.text())
        split = self.cmb_split.currentText()
        for suffix in ("_a", "_b", "_labels"):
            if not (data_dir / f"{split}{suffix}.npy").exists():
                QMessageBox.warning(
                    self, "dataset",
                    f"missing {split}{suffix}.npy in:\n{data_dir}")
                return
        if self.worker is not None and self.worker.isRunning():
            return

        a = np.load(data_dir / f"{split}_a.npy")
        b = np.load(data_dir / f"{split}_b.npy")
        lab = np.load(data_dir / f"{split}_labels.npy")
        meta = {}
        meta_path = data_dir / f"{split}_meta.npz"
        if meta_path.exists():
            meta = {k: np.load(meta_path)[k] for k in
                    ("ra0", "dec0", "roll_a", "roll_b", "fov", "offset_frac",
                     "seed", "stars_a", "stars_b")}

        self.split_a, self.split_b, self.split_lab, self.split_meta = a, b, lab, meta
        self.status.setText(
            f"evaluating {split} split: {len(a)} pairs, size {a.shape[1]} px \u2026")
        self.progress.setRange(0, len(a))
        self.progress.setValue(0)
        self.bt_run.setEnabled(False)
        self._clear_table()

        self.worker = EvalWorker(self.net, a, b, lab, meta, parent=self)
        self.worker.progress.connect(lambda cur, tot: self.progress.setValue(cur))
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_done(self, records: list[dict]) -> None:
        self.bt_run.setEnabled(True)
        self.records = records
        self._fill_table(records)
        n = len(records)
        if n == 0:
            return
        pred = np.asarray([[r["dx"], r["dy"], r["roll"]] for r in records])
        lab = np.asarray([[r["gt_dx"], r["gt_dy"], r["gt_roll"]] for r in records])
        m = error_metrics(pred, lab)
        zero = np.zeros_like(lab)
        base = error_metrics(zero, lab)
        el = records[0]["elapsed"]
        self.lbl_summary.setText(
            f"{self.cmb_split.currentText()}  {n} pairs  ({el:.1f}s)\n"
            f"baseline (predict zeros): {fmt_metrics(base)}\n"
            f"model:                    {fmt_metrics(m)}")
        self.status.setText("done")

    def _on_failed(self, msg: str) -> None:
        self.bt_run.setEnabled(True)
        self.status.setText("run failed")
        QMessageBox.critical(self, "evaluation", msg)

    def _fill_table(self, records: list[dict]) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(records))
        for row, r in enumerate(records):
            vals = [
                f"{r['i']}",
                f"{r['gt_dx']:+.2f}", f"{r['gt_dy']:+.2f}", f"{r['gt_roll']:+.2f}",
                f"{r['dx']:+.2f}", f"{r['dy']:+.2f}", f"{r['roll']:+.2f}",
                f"{r['err_dx']:+.2f}", f"{r['err_dy']:+.2f}", f"{r['err_roll']:+.2f}",
                f"{r['mag']:.3f}",
            ]
            for c, v in enumerate(vals):
                self.table.setItem(row, c, QTableWidgetItem(v))
        self.table.setSortingEnabled(True)
        if records:
            self.table.selectRow(0)

    def _clear_table(self) -> None:
        self.table.setRowCount(0)
        self.records = []
        self.lbl_summary.setText("running\u2026")
        for lb in self._panes.values():
            lb.setText("--")

    def _show_worst(self) -> None:
        if not self.records:
            return
        self.table.setSortingEnabled(False)
        rows = sorted(range(len(self.records)),
                      key=lambda i: self.records[i]["mag"], reverse=True)[:10]
        self.table.clearSelection()
        for r in rows:
            self.table.selectRow(r)
        self.table.setSortingEnabled(True)
        if rows:
            self.table.scrollToItem(self.table.item(rows[0], 0))

    def _on_select_row(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows or not self.records:
            return
        r = self.records[rows[0].row()]
        self._show_pair(r["i"], r)

    def _show_pair(self, idx: int, r: dict) -> None:
        a = display_stretch(self.split_a[idx])
        b = display_stretch(self.split_b[idx])
        gt_b = warp_b_onto_a(b, r["gt_dx"], r["gt_dy"], r["gt_roll"])
        pr_b = warp_b_onto_a(b, r["dx"], r["dy"], r["roll"])
        self._panes["A"].setPixmap(to_pixmap(a, 420))
        self._panes["B"].setPixmap(to_pixmap(b, 420))
        self._panes["B aligned to A  (ground truth)"].setPixmap(
            to_pixmap(blend_ab(a, gt_b), 420))
        self._panes["B aligned to A  (prediction)"].setPixmap(
            to_pixmap(blend_ab(a, pr_b), 420))

        meta = self.split_meta or {}
        has = {k: k in meta and len(meta[k]) > idx for k in
               ("ra0", "dec0", "roll_a", "roll_b", "fov", "offset_frac",
                "seed", "stars_a", "stars_b")}
        info = (
            f"pair {idx}/{len(self.split_a) - 1}"
            + (f"   RA {meta['ra0'][idx]:.4f}\u00b0  Dec {meta['dec0'][idx]:.4f}\u00b0"
               f"  FOV {meta['fov'][idx]:.3f}\u00b0" if has["fov"] else "")
            + (f"  roll A {meta['roll_a'][idx]:.2f}\u00b0 -> B {meta['roll_b'][idx]:.2f}\u00b0"
               if has["roll_b"] else "")
            + (f"  offset up to {meta['offset_frac'][idx] * 100:.1f}% FOV"
               if has["offset_frac"] else "")
            + (f"  stars {meta['stars_a'][idx]} / {meta['stars_b'][idx]}"
               if has["stars_b"] else "")
            + (f"  seed {int(meta['seed'][idx])}" if has["seed"] else "")
            + "\n"
            + f"GT   dx {r['gt_dx']:+.3f}  dy {r['gt_dy']:+.3f}  roll {r['gt_roll']:+.3f}\u00b0\n"
            + f"pred dx {r['dx']:+.3f}  dy {r['dy']:+.3f}  roll {r['roll']:+.3f}\u00b0"
            + f"    err {r['err_dx']:+.3f} / {r['err_dy']:+.3f} px, {r['err_roll']:+.3f}\u00b0,"
            + f" |shift| {r['mag']:.3f} px"
        )
        self.pair_info.setText(info)

    def _refresh_controls(self) -> None:
        self.bt_run.setEnabled(self.net is not None)

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(5000)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    win = ValidatorWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
