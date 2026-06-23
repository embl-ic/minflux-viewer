"""
minflux_viewer.ui.curvilinear_segmentation_dialog
==================================================
Interactive **curvilinear-structure segmentation** tool
(Analyze › Segmentation › Curvilinear Structures…).

Reproduces (and refines) Z. Huang's Fiji microtubule filament tracer: render the
XY histogram, optionally pre-enhance with an oriented line-kernel bank, run a
multi-scale **Hessian ridge filter** (Frangi / Sato), threshold, skeletonize and
trace each branch into a centre-line poly-line. A live preview shows the data,
the ridge map, the skeleton and the traced centre lines; **Add to Manager** files
the selected centre lines into the ROI Manager as open-line ROIs.

Two-phase update: geometry / scale / kernel changes rebuild the ridge map
(debounced); threshold / min-length changes only re-threshold + re-trace.

Modeless / non-owned per the project window convention.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..analysis import curvilinear_segmentation as cv

_MAX_IMAGE_PX = 40_000_000
_RIDGE_CMAP = "inferno"
_PATH_COLOR = (0, 229, 255)        # cyan centre lines
_SEL_COLOR = (255, 235, 0)         # yellow for selected centre lines
_SKEL_COLOR = (80, 255, 120)       # green skeleton overlay
_ROI_STROKE = "#00e5ff"


class CurvilinearSegmentationWindow(QDialog):
    """Modeless interactive Hessian/Frangi filament-tracing tool."""

    def __init__(self, state, dataset_idx: int, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._idx = dataset_idx
        self._owner = owner
        self.setWindowTitle("Curvilinear Structure Segmentation")
        self.resize(1180, 720)

        self._xy = self._load_xy()
        self._H = np.zeros((0, 0))
        self._ridge = np.zeros((0, 0))
        self._skeleton = np.zeros((0, 0), dtype=bool)
        self._xedge = np.zeros(0)
        self._yedge = np.zeros(0)
        self._paths: list[np.ndarray] = []
        self._lengths = np.empty(0)
        self._labels: list[str] = []
        self._suspend = False
        self._fitted = False
        self._image_lut = self._alpha_white_lut()

        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(220)
        self._recompute_timer.timeout.connect(self._recompute)
        self._retrace_timer = QTimer(self)
        self._retrace_timer.setSingleShot(True)
        self._retrace_timer.setInterval(160)
        self._retrace_timer.timeout.connect(self._retrace)

        self._build_ui()
        self._recompute()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.addWidget(self._build_controls())
        root.addWidget(self._build_center(), 1)
        root.addWidget(self._build_segments())

    def _build_controls(self) -> QWidget:
        left = QVBoxLayout()
        ds = self._dataset()
        title = QLabel(f"<b>{ds.name if ds else '(no dataset)'}</b>")
        title.setWordWrap(True)
        left.addWidget(title)
        self._info_label = QLabel(f"{int(self._xy.shape[0]):,} localizations")
        self._info_label.setStyleSheet("color:#888;")
        self._info_label.setWordWrap(True)
        left.addWidget(self._info_label)

        form = QFormLayout()
        self._pixel_spin = self._spin(0.5, 200.0, cv.DEFAULT_PIXEL_NM, 1, 0.5, " nm")
        self._pixel_spin.valueChanged.connect(self._schedule_recompute)
        form.addRow("Pixel size:", self._pixel_spin)
        self._method_combo = QComboBox()
        self._method_combo.addItem("Frangi (vesselness)", "frangi")
        self._method_combo.addItem("Sato (tubeness)", "sato")
        self._method_combo.currentIndexChanged.connect(self._schedule_recompute)
        form.addRow("Ridge filter:", self._method_combo)
        self._scale_min = self._spin(1.0, 1000.0, cv.DEFAULT_SCALE_MIN_NM, 1, 1.0, " nm")
        self._scale_min.setToolTip("Smallest filament half-width (Hessian scale) to detect.")
        self._scale_min.valueChanged.connect(self._schedule_recompute)
        form.addRow("Width min:", self._scale_min)
        self._scale_max = self._spin(1.0, 1000.0, cv.DEFAULT_SCALE_MAX_NM, 1, 1.0, " nm")
        self._scale_max.setToolTip("Largest filament half-width (Hessian scale) to detect.")
        self._scale_max.valueChanged.connect(self._schedule_recompute)
        form.addRow("Width max:", self._scale_max)
        self._n_scales = QSpinBox()
        self._n_scales.setRange(1, 12)
        self._n_scales.setValue(cv.DEFAULT_N_SCALES)
        self._n_scales.setToolTip("Number of Hessian scales between width min and max.")
        self._n_scales.valueChanged.connect(self._schedule_recompute)
        form.addRow("Scales:", self._n_scales)
        self._beta = self._spin(0.05, 5.0, cv.DEFAULT_BETA, 2, 0.05, "")
        self._beta.setToolTip("Frangi blobness sensitivity β (lower = stricter on line-likeness).")
        self._beta.valueChanged.connect(self._schedule_recompute)
        form.addRow("Frangi β:", self._beta)
        self._sqrt = QCheckBox("√ counts (stabilize)")
        self._sqrt.setToolTip("Convolve the square root of the histogram (down-weight aggregates).")
        self._sqrt.toggled.connect(self._schedule_recompute)
        form.addRow("", self._sqrt)
        left.addLayout(form)

        left.addWidget(self._hline())

        # directional pre-filter (the original oriented line-kernel bank)
        self._dir_enable = QCheckBox("Directional pre-filter")
        self._dir_enable.setToolTip(
            "Pre-enhance with a bank of oriented line kernels (max over angles), "
            "the original Fiji macro's structuring step. Usually unnecessary — the "
            "multi-scale Frangi filter already detects every orientation.")
        self._dir_enable.toggled.connect(self._schedule_recompute)
        left.addWidget(self._dir_enable)
        dirf = QFormLayout()
        self._dir_len = self._spin(4.0, 2000.0, 60.0, 1, 2.0, " nm")
        self._dir_len.valueChanged.connect(self._schedule_recompute)
        dirf.addRow("Kernel length:", self._dir_len)
        self._dir_width = self._spin(1.0, 500.0, 12.0, 1, 1.0, " nm")
        self._dir_width.valueChanged.connect(self._schedule_recompute)
        dirf.addRow("Kernel width:", self._dir_width)
        self._dir_angles = QSpinBox()
        self._dir_angles.setRange(2, 90)
        self._dir_angles.setValue(18)
        self._dir_angles.valueChanged.connect(self._schedule_recompute)
        dirf.addRow("Angles:", self._dir_angles)
        left.addLayout(dirf)

        left.addWidget(self._hline())

        detect = QFormLayout()
        self._otsu = QCheckBox("Otsu (auto threshold)")
        self._otsu.toggled.connect(self._on_otsu_toggled)
        detect.addRow("", self._otsu)
        self._threshold = self._spin(0.0, 1.0, cv.DEFAULT_THRESHOLD, 2, 0.01, "")
        self._threshold.setToolTip("Ridge-map threshold as a fraction of its max.")
        self._threshold.valueChanged.connect(lambda *_: self._schedule_retrace())
        detect.addRow("Threshold:", self._threshold)
        self._min_length = self._spin(0.0, 100000.0, cv.DEFAULT_MIN_LENGTH_NM, 0, 50.0, " nm")
        self._min_length.setToolTip("Discard traced centre lines shorter than this.")
        self._min_length.valueChanged.connect(lambda *_: self._schedule_retrace())
        detect.addRow("Min length:", self._min_length)
        left.addLayout(detect)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color:#1a8;")
        self._status_label.setWordWrap(True)
        left.addWidget(self._status_label)
        left.addStretch(1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        left.addWidget(close_btn)

        panel = QWidget()
        panel.setLayout(left)
        panel.setFixedWidth(300)
        return panel

    def _build_center(self) -> QWidget:
        center = QVBoxLayout()
        layers = QHBoxLayout()
        layers.addWidget(QLabel("Show:"))
        self._show_detection = QCheckBox("Detection")
        self._show_detection.setChecked(True)
        self._show_detection.toggled.connect(self._update_layer_visibility)
        self._show_ridge = QCheckBox("Ridge")
        self._show_ridge.setChecked(True)
        self._show_ridge.toggled.connect(self._update_layer_visibility)
        self._show_skeleton = QCheckBox("Skeleton")
        self._show_skeleton.setChecked(False)
        self._show_skeleton.toggled.connect(self._update_layer_visibility)
        self._show_image = QCheckBox("Data")
        self._show_image.setChecked(False)
        self._show_image.toggled.connect(self._update_layer_visibility)
        for w in (self._show_detection, self._show_ridge, self._show_skeleton, self._show_image):
            layers.addWidget(w)
        layers.addStretch(1)
        center.addLayout(layers)

        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "X (nm)")
        self._plot.setLabel("left", "Y (nm)")

        self._ridge_img = pg.ImageItem()
        self._ridge_img.setZValue(0)
        try:
            self._ridge_img.setLookupTable(
                pg.colormap.get(_RIDGE_CMAP).getLookupTable(0.0, 1.0, 256))
        except Exception:
            pass
        self._plot.addItem(self._ridge_img)

        self._hist_img = pg.ImageItem()
        self._hist_img.setZValue(3)
        self._hist_img.setLookupTable(self._image_lut)
        self._hist_img.setVisible(False)
        self._plot.addItem(self._hist_img)

        self._skel_img = pg.ImageItem()
        self._skel_img.setZValue(5)
        self._skel_img.setLookupTable(self._skeleton_lut())
        self._skel_img.setVisible(False)
        self._plot.addItem(self._skel_img)

        self._paths_item = pg.PlotDataItem(pen=pg.mkPen(_PATH_COLOR, width=2), connect="finite")
        self._paths_item.setZValue(10)
        self._plot.addItem(self._paths_item)
        self._sel_item = pg.PlotDataItem(pen=pg.mkPen(_SEL_COLOR, width=3), connect="finite")
        self._sel_item.setZValue(11)
        self._plot.addItem(self._sel_item)

        center.addWidget(self._plot, 1)
        panel = QWidget()
        panel.setLayout(center)
        return panel

    def _build_segments(self) -> QWidget:
        right = QVBoxLayout()
        self._seg_header = QLabel("Segments (0)")
        self._seg_header.setStyleSheet("font-weight:bold;")
        right.addWidget(self._seg_header)
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._on_list_selection)
        right.addWidget(self._list, 1)
        hint = QLabel("Select rows to add a subset (none selected = all).")
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        right.addWidget(hint)
        self._add_btn = QPushButton("Add to Manager")
        self._add_btn.clicked.connect(self._add_to_manager)
        right.addWidget(self._add_btn)

        right.addWidget(self._hline())
        wrow = QHBoxLayout()
        wrow.addWidget(QLabel("Profile ½-width:"))
        self._profile_halfwidth = self._spin(8.0, 2000.0, 80.0, 0, 5.0, " nm")
        self._profile_halfwidth.setToolTip(
            "Half the perpendicular scan range used for the width (FWHM) profile.")
        wrow.addWidget(self._profile_halfwidth)
        right.addLayout(wrow)
        self._width_btn = QPushButton("Measure Width (FWHM)")
        self._width_btn.setToolTip(
            "Average the rendered image perpendicular to the selected centre lines "
            "(or all) and report the full-width-at-half-maximum.")
        self._width_btn.clicked.connect(self._measure_width)
        right.addWidget(self._width_btn)

        panel = QWidget()
        panel.setLayout(right)
        panel.setFixedWidth(260)
        return panel

    @staticmethod
    def _spin(lo, hi, val, decimals, step, suffix) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setValue(val)
        if suffix:
            s.setSuffix(suffix)
        return s

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#444;")
        return line

    @staticmethod
    def _alpha_white_lut() -> np.ndarray:
        lut = np.zeros((256, 4), dtype=np.ubyte)
        lut[:, :3] = 255
        lut[:, 3] = (255.0 * (np.linspace(0.0, 1.0, 256) ** 0.5)).astype(np.ubyte)
        return lut

    @staticmethod
    def _skeleton_lut() -> np.ndarray:
        lut = np.zeros((256, 4), dtype=np.ubyte)
        lut[1:, 0], lut[1:, 1], lut[1:, 2] = _SKEL_COLOR
        lut[1:, 3] = 220
        return lut

    # ----------------------------------------------------------- data path
    def _dataset(self):
        if 0 <= self._idx < len(self._state.datasets):
            return self._state.datasets[self._idx]
        return None

    def _load_xy(self) -> np.ndarray:
        ds = self._dataset()
        if ds is None:
            return np.empty((0, 2))
        try:
            loc = np.asarray(ds.loc_nm, dtype=float)
        except Exception:
            return np.empty((0, 2))
        if loc.ndim != 2 or loc.shape[0] == 0 or loc.shape[1] < 2:
            return np.empty((0, 2))
        mask = np.asarray(ds.filter_mask, dtype=bool)
        if mask.shape[0] == loc.shape[0]:
            loc = loc[mask]
        xy = loc[:, :2]
        return xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])]

    # ----------------------------------------------------------- compute
    def _schedule_recompute(self) -> None:
        if not self._suspend:
            self._recompute_timer.start()

    def _schedule_retrace(self) -> None:
        if not self._suspend:
            self._retrace_timer.start()

    def _on_otsu_toggled(self, on: bool) -> None:
        self._threshold.setEnabled(not on)
        self._schedule_retrace()

    def _directional(self) -> dict | None:
        if not self._dir_enable.isChecked():
            return None
        return {"length_nm": self._dir_len.value(), "width_nm": self._dir_width.value(),
                "n_angles": self._dir_angles.value(), "gaussian": True}

    def _recompute(self) -> None:
        """Full update: render → optional directional pre-filter → ridge filter."""
        px = float(self._pixel_spin.value())
        xy = self._xy
        if xy.shape[0] < 3:
            self._reset("Need at least 3 localizations.")
            return
        nx = int((xy[:, 0].max() - xy[:, 0].min()) / px) + 2
        ny = int((xy[:, 1].max() - xy[:, 1].min()) / px) + 2
        if nx * ny > _MAX_IMAGE_PX:
            self._reset("")
            self._info_label.setText(
                f"Image {nx}×{ny} too large at {px:g} nm/px — increase the pixel size.")
            return

        self._H, self._xedge, self._yedge = cv.render_histogram_2d(xy[:, 0], xy[:, 1], px)
        image = cv.stabilize_histogram(self._H, self._sqrt.isChecked())
        directional = self._directional()
        if directional:
            image = cv.directional_filter(
                image, int(directional["length_nm"] / px),
                max(directional["width_nm"] / px, 1.0),
                int(directional["n_angles"]), True)
        sigmas = cv.scales_from_nm(self._scale_min.value(), self._scale_max.value(),
                                   self._n_scales.value(), px)
        self._ridge = cv.ridge_filter(image, sigmas, method=self._method_combo.currentData(),
                                      beta=float(self._beta.value()))
        self._info_label.setText(
            f"{xy.shape[0]:,} localizations · {self._H.shape[0]}×{self._H.shape[1]} px")
        self._draw_images()
        self._retrace()

    def _retrace(self) -> None:
        """Cheap update: threshold the cached ridge map, skeletonize, trace."""
        if self._suspend or self._ridge.size == 0:
            return
        px = float(self._pixel_spin.value())
        if self._otsu.isChecked():
            pos = self._ridge[self._ridge > 0]
            thr = cv.otsu_threshold(pos) if pos.size else 0.0
        else:
            thr = float(self._threshold.value())
        binary = self._ridge >= thr if thr > 0 else self._ridge > 0
        self._skeleton = cv.zhang_suen_skeleton(binary)
        paths_rc = cv.skeleton_to_paths(self._skeleton, int(max(self._min_length.value() / px, 1.0)))
        xc = 0.5 * (self._xedge[:-1] + self._xedge[1:])
        yc = 0.5 * (self._yedge[:-1] + self._yedge[1:])
        self._paths = [np.column_stack([xc[p[:, 0].astype(int)], yc[p[:, 1].astype(int)]])
                       for p in paths_rc]
        self._lengths = np.array([self._path_length(p) for p in self._paths])
        self._labels = [f"Segment {i}" for i in range(1, len(self._paths) + 1)]
        self._draw_skeleton()
        self._draw_paths()
        self._populate_list()
        self._status_label.setText("")
        has_paths = len(self._paths) > 0
        self._add_btn.setEnabled(has_paths)
        self._width_btn.setEnabled(has_paths)

    def _reset(self, status: str) -> None:
        self._H = np.zeros((0, 0))
        self._ridge = np.zeros((0, 0))
        self._skeleton = np.zeros((0, 0), dtype=bool)
        self._paths = []
        self._lengths = np.empty(0)
        self._labels = []
        for item in (self._ridge_img, self._hist_img, self._skel_img):
            item.clear()
        self._paths_item.clear()
        self._sel_item.clear()
        self._populate_list()
        self._status_label.setText(status)
        self._add_btn.setEnabled(False)
        self._width_btn.setEnabled(False)

    @staticmethod
    def _path_length(path: np.ndarray) -> float:
        if path.shape[0] < 2:
            return 0.0
        return float(np.sum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))))

    # ----------------------------------------------------------- drawing
    def _image_rect(self) -> QRectF:
        return QRectF(float(self._xedge[0]), float(self._yedge[0]),
                      float(self._xedge[-1] - self._xedge[0]),
                      float(self._yedge[-1] - self._yedge[0]))

    def _draw_images(self) -> None:
        if self._ridge.size == 0:
            return
        rect = self._image_rect()
        self._ridge_img.setImage(self._oriented(self._ridge), levels=(0.0, 1.0))
        self._ridge_img.setRect(rect)
        pos = self._H[self._H > 0]
        vmax = float(np.percentile(pos, 99.0)) if pos.size else 1.0
        self._hist_img.setImage(self._oriented(self._H), levels=(0.0, max(vmax, 1.0)))
        self._hist_img.setRect(rect)
        self._update_layer_visibility()
        if not self._fitted:
            self._plot.getViewBox().autoRange(padding=0.02)
            self._fitted = True

    def _draw_skeleton(self) -> None:
        if self._skeleton.size == 0:
            self._skel_img.clear()
            return
        self._skel_img.setImage(self._oriented(self._skeleton.astype(float)), levels=(0.0, 1.0))
        self._skel_img.setRect(self._image_rect())
        self._update_layer_visibility()

    def _draw_paths(self) -> None:
        x, y = self._concat(self._paths)
        if x.size:
            self._paths_item.setData(x=x, y=y)
        else:
            self._paths_item.clear()
        self._sel_item.clear()

    @staticmethod
    def _concat(paths: list[np.ndarray]):
        """Concatenate poly-lines into one (x, y) pair with NaN separators."""
        if not paths:
            return np.empty(0), np.empty(0)
        xs, ys = [], []
        for p in paths:
            xs.append(p[:, 0]); xs.append([np.nan])
            ys.append(p[:, 1]); ys.append([np.nan])
        return np.concatenate(xs), np.concatenate(ys)

    def _update_layer_visibility(self) -> None:
        self._ridge_img.setVisible(self._show_ridge.isChecked() and self._ridge.size > 0)
        self._hist_img.setVisible(self._show_image.isChecked() and self._H.size > 0)
        self._skel_img.setVisible(self._show_skeleton.isChecked() and self._skeleton.size > 0)
        det = self._show_detection.isChecked()
        self._paths_item.setVisible(det)
        self._sel_item.setVisible(det)

    @staticmethod
    def _oriented(img: np.ndarray) -> np.ndarray:
        try:
            row_major = pg.getConfigOption("imageAxisOrder") == "row-major"
        except Exception:
            row_major = True
        return img.T if row_major else img

    # ----------------------------------------------------------- segments list
    def _populate_list(self) -> None:
        self._suspend_list = True
        self._list.clear()
        for i, path in enumerate(self._paths):
            length = self._lengths[i] if i < self._lengths.size else 0.0
            item = QListWidgetItem(f"{self._labels[i]}   {path.shape[0]} px   L={length:.0f} nm")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._list.addItem(item)
        self._suspend_list = False
        self._seg_header.setText(f"Segments ({len(self._paths)})")
        self._sel_item.clear()

    def _on_list_selection(self) -> None:
        if getattr(self, "_suspend_list", False):
            return
        idx = [it.data(Qt.ItemDataRole.UserRole) for it in self._list.selectedItems()]
        x, y = self._concat([self._paths[i] for i in idx]) if idx else (np.empty(0), np.empty(0))
        if x.size:
            self._sel_item.setData(x=x, y=y)
        else:
            self._sel_item.clear()

    def _selected_indices(self) -> list[int]:
        idx = sorted(it.data(Qt.ItemDataRole.UserRole) for it in self._list.selectedItems())
        return idx if idx else list(range(len(self._paths)))

    # ----------------------------------------------------------- commit
    def _add_to_manager(self) -> None:
        sel = self._selected_indices()
        if not sel or self._owner is None or not hasattr(self._owner, "add_polyline_rois"):
            return
        ds = self._dataset()
        paths = [self._paths[i] for i in sel]
        names = [self._labels[i] for i in sel]
        opts = []
        if self._sqrt.isChecked():
            opts.append("sqrt-stabilized")
        if self._dir_enable.isChecked():
            opts.append("directional pre-filter")
        opts_str = (", " + ", ".join(opts)) if opts else ""
        log = (
            f"Curvilinear segmentation: added {len(paths)} of {len(self._paths)} "
            f"centre line(s) on '{ds.name if ds else '?'}' "
            f"({self._method_combo.currentData()} ridge, pixel={self._pixel_spin.value():.1f} nm, "
            f"width {self._scale_min.value():.0f}-{self._scale_max.value():.0f} nm × "
            f"{self._n_scales.value()} scales, "
            f"threshold={'Otsu' if self._otsu.isChecked() else f'{self._threshold.value():.2f}'}, "
            f"min length={self._min_length.value():.0f} nm{opts_str}); added poly-line ROIs.")
        n = self._owner.add_polyline_rois(
            self._idx, paths, name_prefix="Segment", source="curvilinear_segmentation_2d",
            stroke_color=_ROI_STROKE, names=names, log_message=log)
        if n:
            self._status_label.setText(f"Added {n} centre line(s) to the ROI Manager.")

    def _measure_width(self) -> None:
        """Measure FWHM of the perpendicular intensity profile of the selected
        (or all) centre lines against the rendered image."""
        sel = self._selected_indices()
        if not sel or self._H.size == 0:
            return
        paths = [self._paths[i] for i in sel]
        px = float(self._pixel_spin.value())
        res = cv.measure_filament_width(
            self._H, self._xedge, self._yedge, paths,
            max_dist_nm=float(self._profile_halfwidth.value()), step_nm=px,
            pixel_size_nm=px)
        ds = self._dataset()
        name = ds.name if ds else "?"
        valid = res["per_path_fwhm"][np.isfinite(res["per_path_fwhm"])]
        per = (f"; per-segment {valid.mean():.1f} ± {valid.std():.1f} nm (n={valid.size})"
               if valid.size else "")
        fwhm_txt = f"{res['fwhm']:.1f} nm" if np.isfinite(res["fwhm"]) else "n/a"
        self._status_label.setText(f"FWHM = {fwhm_txt}")
        if self._owner is not None and hasattr(self._owner, "_state"):
            self._owner._state.log(
                f"Curvilinear width (FWHM) on '{name}': combined FWHM = {fwhm_txt} "
                f"over {len(paths)} centre line(s) "
                f"(half-width {self._profile_halfwidth.value():.0f} nm){per}.")
        from .modeless import show_modeless
        from .width_profile_window import WidthProfileWindow
        win = WidthProfileWindow(res, title=f"{name} — {len(paths)} centre line(s)",
                                 owner=self._owner)
        show_modeless(win, self._owner)
