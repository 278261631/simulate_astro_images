#!/usr/bin/env python3
"""PySide6 GUI to validate a trained PairRegNet on generated (A, B) pairs.

Workflow
--------
1. Point at a trained checkpoint (a plain ``best.pt`` state dict or a full
   ``checkpoint_best.pt`` dict produced by ``train.py``) and at a dataset
   directory.  Two layouts are auto-detected per split:

   * numpy dataset from ``generate_dataset.py`` (default ``data/``):
     ``<split>_a.npy`` + ``<split>_b.npy`` + ``<split>_labels.npy``
   * image/text smoke dataset from ``generate_smoke_data.bat`` (default
     ``data_smoke/``): sub-folder ``<split>/`` holding one ``<n>_a.png`` /
     ``<n>_b.png`` pair plus ``<n>.txt`` label per sample.

2. Pick a split (train / val / test) and press *Run split*. Every pair is
   scored in a background thread and an error table + summary metrics are
   shown (same statistics ``train.py`` prints).  Frames of any size (and
   mixed sizes within one split, e.g. the default multi-resolution smoke set)
   are resampled to the model's native resolution (read from the checkpoint,
   fallback 192 px) for inference; predicted pixel
   offsets are scaled back to the original image scale so the GT comparison
   stays in the dataset's pixels.  When sizes are mixed the summary also
   lists one metric line per resolution, and the table's *px* column sorts
   them.

3. Click a table row to inspect that pair:
       A             frame A, display stretched
       B             frame B, display stretched
       GT align      B warped with the *ground truth* (dx, dy, droll), shown
                     in red blended over A in green
       Pred align    same but with the *predicted* transform
   Where the two channels coincide the blend looks neutral/yellow; residual
   mis-registration shows up as red or green fringes on star edges.

Run:
    python validate_model_gui.py
"""

from __future__ import annotations

import math
import re
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
DEFAULT_DATA = HERE / "data_smoke"
DEFAULT_MODEL_SIZE = 192  # pixel size fallback when the checkpoint has no metadata


def resolve_model_size(path: Path, obj) -> int:
    """Pixel size of the model input grid for a checkpoint.

    Priority: full checkpoint dict (``model_in`` written by train.py) ->
    sibling ``best_info.json`` sidecar -> legacy 192.
    """
    if isinstance(obj, dict) and isinstance(obj.get("model_in"), int):
        return int(obj["model_in"])
    side = Path(path).with_name("best_info.json")
    if side.exists():
        try:
            import json
            return int(json.loads(side.read_text(encoding="utf-8"))["model_in"])
        except Exception:
            pass
    return DEFAULT_MODEL_SIZE


# ---------------------------------------------------------------------------
# Metrics / formatting (duplicated from train.py so the GUI is self-contained)
# ---------------------------------------------------------------------------


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


def load_checkpoint(path: Path) -> tuple[PairRegNet, dict | None, int]:
    """Load a state dict / full checkpoint; return (net, info, model_input_size)."""
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
    return net, info, resolve_model_size(path, obj)


# ---------------------------------------------------------------------------
# Dataset loading (numpy layout OR data_smoke image/text layout)
# ---------------------------------------------------------------------------


class SplitData:
    """Uniform handle over a split, whichever layout produced it.

    ``a``/``b`` are either a single (n, H, W) uint8 array (numpy layout,
    uniform resolution) or a sequence of per-sample (H, W) arrays (image/text
    layout, which may mix resolutions). ``sizes`` always holds one entry per
    sample.
    """

    def __init__(self, a, b, labels: np.ndarray, meta_rows: list[dict | None]) -> None:
        self.labels = np.asarray(labels, dtype=np.float64)
        self.meta_rows = meta_rows
        self.n = len(self.labels)
        if isinstance(a, np.ndarray):
            self.a = a
            self.b = b
            sizes = np.full(self.n, int(a.shape[1]), dtype=int)
        else:
            self.a = [np.asarray(x, dtype=np.uint8) for x in a]
            self.b = [np.asarray(x, dtype=np.uint8) for x in b]
            sizes = np.asarray([x.shape[0] for x in self.a], dtype=int)
        self.sizes = np.asarray(sizes)

    @property
    def uniform_size(self) -> int | None:
        return int(self.sizes[0]) if (self.sizes == self.sizes[0]).all() else None

    def images(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.a[i], self.b[i]

    def meta(self, i: int) -> dict:
        row = self.meta_rows[i]
        return dict(row) if row else {}


def _numpy_meta_rows(n: int, npz: dict) -> list[dict]:
    """Per-sample meta dicts from a generate_dataset *_meta.npz file."""
    rows = []
    for i in range(n):
        m = {}
        for k in ("ra0", "dec0", "roll_a", "roll_b", "fov", "offset_frac",
                  "seed", "stars_a", "stars_b"):
            arr = npz.get(k)
            if arr is not None and i < len(arr):
                v = arr[i]
                m[k] = float(v) if k in ("fov", "offset_frac") else (
                    int(v) if k in ("seed", "stars_a", "stars_b") else float(v))
        rows.append(m)
    return rows


def load_numpy_split(data_dir: Path, split: str) -> SplitData:
    data_dir = Path(data_dir)
    a = np.load(data_dir / f"{split}_a.npy")
    b = np.load(data_dir / f"{split}_b.npy")
    lab = np.load(data_dir / f"{split}_labels.npy")
    meta = {}
    meta_path = data_dir / f"{split}_meta.npz"
    if meta_path.exists():
        meta = {k: np.load(meta_path)[k] for k in np.load(meta_path).files}
    return SplitData(a, b, lab, _numpy_meta_rows(len(a), meta))


_TXT_LABEL_RE = re.compile(
    r"label\s+dx\s+([+-]?[0-9.]+)\s+px\s+dy\s+([+-]?[0-9.]+)\s+px"
    r"\s+droll\s+([+-]?[0-9.]+)\s+deg",
    re.IGNORECASE,
)


def _txt_meta(path: Path) -> tuple[dict, tuple[float, float, float]]:
    """Parse one data_smoke <n>.txt label/geometry file -> (meta, (dx, dy, droll))."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    m = {}

    def grab(pattern: str, key: str, conv=float) -> None:
        r = re.search(pattern, txt, re.IGNORECASE)
        if r:
            m[key] = conv(r.group(1))

    r2 = re.search(r"A center RA/Dec\s+([-0-9.]+)\s+deg\s+([-0-9.]+)\s+deg",
                   txt, re.IGNORECASE)
    if r2:
        m["ra0"], m["dec0"] = float(r2.group(1)), float(r2.group(2))
    grab(r"commanded roll A\s+([-0-9.]+)", "roll_a")
    grab(r"actual roll B\s+([-0-9.]+)", "roll_b")
    grab(r"FOV\s+([0-9.]+)", "fov")
    grab(r"up to\s+([0-9.]+)\s*% of FOV", "offset_frac")
    if "offset_frac" in m:
        m["offset_frac"] = m["offset_frac"] / 100.0
    rs = re.search(r"stars in frame\s+A\s+(\d+)\s+B\s+(\d+)", txt, re.IGNORECASE)
    if rs:
        m["stars_a"], m["stars_b"] = int(rs.group(1)), int(rs.group(2))
    grab(r"seed\s+(\d+)", "seed", int)
    lm = _TXT_LABEL_RE.search(txt)
    if lm is None:
        raise ValueError(f"no 'label dx dy droll' line in {path}")
    label = (float(lm.group(1)), float(lm.group(2)), float(lm.group(3)))
    return m, label


def load_image_split(split_dir: Path) -> SplitData:
    """Load a data_smoke/<split>/ folder of <n>_a.png / <n>_b.png / <n>.txt."""
    split_dir = Path(split_dir)
    try:
        from PIL import Image
    except Exception:
        raise SystemExit(
            "Pillow is required to read the image/text dataset:\n"
            "  python -m pip install pillow"
        )

    a_files = sorted(split_dir.glob("*_a.png"))
    if not a_files:
        raise FileNotFoundError(f"no *_a.png files in {split_dir}")
    idxs, arrays_a, arrays_b = [], [], []
    meta_rows, labels = [], []
    for fa in a_files:
        stem = fa.name[: -len("_a.png")]
        fb = split_dir / f"{stem}_b.png"
        ft = split_dir / f"{stem}.txt"
        if not fb.exists() or not ft.exists():
            continue
        ia = np.asarray(Image.open(fa).convert("L"), dtype=np.uint8)
        ib = np.asarray(Image.open(fb).convert("L"), dtype=np.uint8)
        meta, label = _txt_meta(ft)
        idxs.append(int(stem))
        arrays_a.append(ia)
        arrays_b.append(ib)
        meta_rows.append(meta)
        labels.append(label)
    if not arrays_a:
        raise FileNotFoundError(f"no complete A/B/TXT pairs in {split_dir}")
    order = np.argsort(idxs)
    # kept as a list: --smoke-sizes may mix several frame resolutions in one split
    a = [arrays_a[i] for i in order]
    b = [arrays_b[i] for i in order]
    labels = [labels[i] for i in order]
    meta_rows = [meta_rows[i] for i in order]
    return SplitData(a, b, labels, meta_rows)


def load_split(data_dir: Path, split: str) -> SplitData:
    """numpy layout first, then the image/text data_smoke layout."""
    data_dir = Path(data_dir)
    if (data_dir / f"{split}_a.npy").exists():
        return load_numpy_split(data_dir, split)
    split_dir = data_dir / split
    if split_dir.is_dir():
        try:
            return load_image_split(split_dir)
        except FileNotFoundError:
            pass
    raise FileNotFoundError(
        f"no dataset found for split '{split}' in {data_dir}\n"
        f"expected either <split>_a.npy (numpy dataset) or "
        f"{split}/<n>_a.png + <n>_b.png + <n>.txt (image/text dataset)")


# ---------------------------------------------------------------------------
# Inference pre-processing (resample any size to the model's pixel grid)
# ---------------------------------------------------------------------------


def _to_model_scale(ia: np.ndarray, ib: np.ndarray, model_size: int):
    """Return (ia, ib resized to model_size, px_scale).  px_scale = model px
    per original px; predicted offsets in the original grid = prediction / it."""
    import cv2  # noqa: PLC0415 - heavy import, kept local like predict.py

    h, w = ia.shape
    s = model_size / max(w, h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if (nw, nh) != (w, h):
        ia = cv2.resize(ia, (nw, nh), interpolation=cv2.INTER_AREA)
        ib = cv2.resize(ib, (nw, nh), interpolation=cv2.INTER_AREA)
    ph, pw = model_size - nh, model_size - nw
    pad = ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2))
    ia = np.pad(ia, pad, mode="edge")
    ib = np.pad(ib, pad, mode="edge")
    return ia, ib, s


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

    def __init__(self, net: PairRegNet, src: SplitData, model_size: int,
                 batch: int = 64, parent=None) -> None:
        super().__init__(parent)
        self.net = net
        self.src = src
        self.model_size = model_size
        self.batch = batch

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            src = self.src
            n = len(src.labels)
            ms = self.model_size
            pairs = []
            scales = np.ones(n)
            for i in range(n):
                ia, ib = src.images(i)
                ia, ib, s = _to_model_scale(ia, ib, ms)
                scales[i] = s
                pairs.append(preprocess(ia, ib))
            pairs = torch.stack(pairs)
            pred = np.zeros((n, 3), dtype=np.float64)
            with torch.no_grad():
                for st in range(0, n, self.batch):
                    if self.isInterruptionRequested():
                        return
                    en = min(st + self.batch, n)
                    out = decode(self.net(pairs[st:en])).numpy()
                    pred[st:en] = out
                    self.progress.emit(en, n)
            # convert model-grid offsets back to the dataset's pixel scale
            pred[:, 0] /= scales
            pred[:, 1] /= scales
            gt = src.labels
            records = []
            for i in range(n):
                e = pred[i] - gt[i]
                records.append(
                    {
                        "i": int(i),
                        "size": int(src.sizes[i]),
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
        self.src: SplitData | None = None
        self.records: list[dict] = []
        self.worker: EvalWorker | None = None
        self.model_size: int = DEFAULT_MODEL_SIZE

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

        top.addWidget(QLabel("Model"))
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
        top.addWidget(QLabel("Data dir"))
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
        for t, (r, c) in zip(titles, [(0, 0), (0, 1), (1, 0), (1, 1)]):
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

        cols = ["idx", "px", "gt dx", "gt dy", "gt roll", "pred dx", "pred dy",
                "pred roll", "err dx", "err dy", "err roll", "|shift| px"]
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
            self.net, info, self.model_size = load_checkpoint(path)
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
            f"loaded {path.name} ({n_params / 1e6:.1f}M params, "
            f"model input {self.model_size}px){extra}")
        self._refresh_controls()

    def _run_split(self) -> None:
        if self.net is None:
            QMessageBox.information(self, "model", "load a model first")
            return
        data_dir = Path(self.ed_data.text())
        split = self.cmb_split.currentText()
        if not data_dir.is_dir():
            QMessageBox.warning(self, "dataset", f"folder not found:\n{data_dir}")
            return
        try:
            self.src = load_split(data_dir, split)
        except Exception as exc:
            QMessageBox.warning(self, "dataset", str(exc))
            return
        if self.worker is not None and self.worker.isRunning():
            return

        src = self.src
        sz = src.uniform_size
        sdesc = f"{sz} px" if sz is not None else \
            f"{int(src.sizes.min())}-{int(src.sizes.max())} px (mixed)"
        self.status.setText(
            f"evaluating {split} split: {src.n} pairs, {sdesc} "
            f"({data_dir}) \u2026")
        self.progress.setRange(0, src.n)
        self.progress.setValue(0)
        self.bt_run.setEnabled(False)
        self._clear_table()

        self.worker = EvalWorker(self.net, src, self.model_size, parent=self)
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
        lines = [f"{self.cmb_split.currentText()}  {n} pairs  ({el:.1f}s)"]
        ms = self.model_size
        if self.src is not None:
            sz = self.src.uniform_size
            if sz is None:
                lines[0] += (f"\n{sorted(set(self.src.sizes.tolist()))} px mixed; "
                             f"every frame resampled to {ms}px for inference")
            elif sz != ms:
                lines[0] += f"\n{sz}px frames resampled to {ms}px for inference"
        lines.append(f"baseline (predict zeros): {fmt_metrics(base)}")
        lines.append(f"model:                    {fmt_metrics(m)}")
        # per-resolution breakdown when sizes are mixed
        if self.src is not None and self.src.uniform_size is None:
            sizes = np.asarray([r["size"] for r in records])
            for z in np.unique(sizes):
                mask = sizes == z
                mz = error_metrics(pred[mask], lab[mask])
                lines.append(f"  {int(z)}px x {int(mask.sum()):<3}: {fmt_metrics(mz)}")
        self.lbl_summary.setText("\n".join(lines))
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
                f"{r['size']}",
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
        if self.src is None or not (0 <= idx < self.src.n):
            return
        ia, ib = self.src.images(idx)
        a = display_stretch(ia)
        b = display_stretch(ib)
        gt_b = warp_b_onto_a(b, r["gt_dx"], r["gt_dy"], r["gt_roll"])
        pr_b = warp_b_onto_a(b, r["dx"], r["dy"], r["roll"])
        self._panes["A"].setPixmap(to_pixmap(a, 420))
        self._panes["B"].setPixmap(to_pixmap(b, 420))
        self._panes["B aligned to A  (ground truth)"].setPixmap(
            to_pixmap(blend_ab(a, gt_b), 420))
        self._panes["B aligned to A  (prediction)"].setPixmap(
            to_pixmap(blend_ab(a, pr_b), 420))

        m = self.src.meta(idx)
        info = (
            f"pair {idx}/{self.src.n - 1}"
            + (f"   RA {m['ra0']:.4f}\u00b0  Dec {m['dec0']:.4f}\u00b0"
               f"  FOV {m['fov']:.3f}\u00b0" if "fov" in m and "ra0" in m else "")
            + (f"  roll A {m['roll_a']:.2f}\u00b0 -> B {m['roll_b']:.2f}\u00b0"
               if "roll_a" in m and "roll_b" in m else "")
            + (f"  offset up to {m['offset_frac'] * 100:.1f}% FOV"
               if "offset_frac" in m else "")
            + (f"  stars {m['stars_a']} / {m['stars_b']}"
               if "stars_a" in m and "stars_b" in m else "")
            + (f"  seed {m['seed']}" if "seed" in m else "")
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
