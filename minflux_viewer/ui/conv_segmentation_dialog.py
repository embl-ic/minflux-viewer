"""
minflux_viewer.ui.conv_segmentation_dialog
===========================================
Interactive **convolution segmentation** tool
(Analyze › Segmentation › Convolution…).

Pick a *geometry model* (ring / disk / gaussian / …); the tool builds a matched
convolution kernel from its parameters, FFT-convolves it with the rendered XY
histogram and runs non-maximum suppression to detect structures. A live preview
shows the **kernel**, the **response heatmap** and (optionally) the **original
rendered image** with detected centres, so the parameters can be tuned by eye.

Detections are listed in a selectable table; **Add to Manager** files the selected
detections (or all, if none are selected) into the ROI Manager as rectangle ROIs.

Two-phase update keeps tuning responsive:
  * geometry / pixel-size / zero-mean changes rebuild the response map
    (the FFT convolution) — debounced;
  * threshold / separation / cap changes only re-run the cheap NMS on the cached
    response map.

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

from ..analysis import conv_segmentation as cs
from .layer_brightness import LayerBrightness, percentile_levels
from .roi_overlay import PointMarkerItem

_MAX_IMAGE_PX = 40_000_000      # guard: skip convolution above this histogram size
_RESPONSE_CMAP = "inferno"
_SEL_COLOR = (255, 235, 0)      # yellow ring on the selected detections
_ROI_STROKE = "#00e5ff"
_MARKER_PX = 26                 # on-screen diameter of the point-ROI-style marker


class ConvSegmentationWindow(QDialog):
    """Modeless interactive geometry-kernel convolution segmentation tool."""

    def __init__(self, state, dataset_idx: int, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._idx = dataset_idx
        self._owner = owner
        self.setWindowTitle("Convolution Segmentation")
        self.resize(1180, 700)

        self._xy = self._load_xy()
        self._param_spins: dict[str, QDoubleSpinBox] = {}
        self._H = np.zeros((0, 0))
        self._response = np.zeros((0, 0))
        self._xedge = np.zeros(0)
        self._yedge = np.zeros(0)
        self._centers = np.empty((0, 2))
        self._values = np.empty(0)
        self._inside = np.empty(0)        # inside-ring ratio per detection (ring validation)
        self._labels: list[str] = []
        self._suspend = False
        self._fitted = False       # auto-range the response view only on first draw
        self._data_vmax = 1.0      # current Data-layer default high level
        self._bc = LayerBrightness()
        self._image_lut = self._alpha_white_lut()
        self._model_params: dict[str, dict] = {}   # per-model param cache (persisted)
        self._active_model_key: str | None = None

        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(180)
        self._recompute_timer.timeout.connect(self._recompute)

        self._build_ui()
        self._register_bc_layers()
        self._restore_prefs()      # restore last-session model + params (builds param spins)
        self._recompute()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.addWidget(self._build_controls())
        root.addWidget(self._build_center(), 1)
        root.addWidget(self._build_detections())

    def _build_controls(self) -> QWidget:
        left = QVBoxLayout()
        ds = self._dataset()
        title = QLabel(f"<b>{ds.name if ds else '(no dataset)'}</b>")
        title.setWordWrap(True)
        left.addWidget(title)
        n_loc = int(self._xy.shape[0])
        self._info_label = QLabel(f"{n_loc:,} localizations")
        self._info_label.setStyleSheet("color: #888;")
        self._info_label.setWordWrap(True)
        left.addWidget(self._info_label)

        left.addWidget(QLabel("Geometry model:"))
        self._model_combo = QComboBox()
        for key, model in cs.GEOMETRY_MODELS.items():
            self._model_combo.addItem(model.label, key)
            self._model_combo.setItemData(self._model_combo.count() - 1,
                                          model.description, Qt.ItemDataRole.ToolTipRole)
        self._model_combo.currentIndexChanged.connect(lambda *_: self._rebuild_params())
        left.addWidget(self._model_combo)

        # kernel preview
        self._kernel_plot = pg.PlotWidget()
        self._kernel_plot.setFixedHeight(160)
        self._kernel_plot.setAspectLocked(True)
        self._kernel_plot.hideAxis("bottom")
        self._kernel_plot.hideAxis("left")
        self._kernel_plot.setMenuEnabled(False)
        self._kernel_plot.setMouseEnabled(False, False)
        self._kernel_img = pg.ImageItem()
        self._kernel_plot.addItem(self._kernel_img)
        left.addWidget(QLabel("Kernel:"))
        left.addWidget(self._kernel_plot)

        # per-model parameter form (rebuilt on model change)
        self._param_form = QFormLayout()
        left.addLayout(self._param_form)

        common = QFormLayout()
        self._pixel_spin = self._spin(0.5, 200.0, 4.0, 1, 0.5, " nm")
        self._pixel_spin.valueChanged.connect(self._schedule_recompute)
        common.addRow("Pixel size:", self._pixel_spin)
        self._zero_mean = QCheckBox("Zero-mean kernel")
        self._zero_mean.setChecked(cs.DEFAULT_ZERO_MEAN)
        self._zero_mean.setToolTip(
            "Remove the kernel DC component so the convolution responds to the local "
            "pattern/contrast rather than raw localization density.")
        self._zero_mean.toggled.connect(self._schedule_recompute)
        common.addRow("", self._zero_mean)
        self._sqrt = QCheckBox("√ counts (stabilize)")
        self._sqrt.setChecked(cs.DEFAULT_SQRT)
        self._sqrt.setToolTip(
            "Convolve the square root of the histogram, down-weighting bright "
            "aggregates that would otherwise dominate the response (SMAP-style).")
        self._sqrt.toggled.connect(self._schedule_recompute)
        common.addRow("", self._sqrt)
        self._dog = QCheckBox("DoG band-pass")
        self._dog.setChecked(cs.DEFAULT_DOG)
        self._dog.setToolTip(
            "Difference-of-Gaussians band-pass on the response at the structure "
            "scale, suppressing slow background (SMAP segmentNPC).")
        self._dog.toggled.connect(self._schedule_recompute)
        common.addRow("", self._dog)
        left.addLayout(common)

        left.addWidget(self._hline())

        detect = QFormLayout()
        self._response_spin = self._spin(0.0, 1.0, cs.DEFAULT_MIN_RESPONSE, 2, 0.02, "")
        self._response_spin.setToolTip("Acceptance threshold as a fraction of the peak "
                                       "response (0 keeps every local maximum).")
        self._response_spin.valueChanged.connect(lambda *_: self._rethreshold())
        detect.addRow("Min response:", self._response_spin)
        self._sep_spin = self._spin(1.0, 5000.0, 100.0, 1, 5.0, " nm")
        self._sep_spin.setToolTip("Minimum centre-to-centre separation between detections.")
        self._sep_spin.valueChanged.connect(lambda *_: self._rethreshold())
        detect.addRow("Min separation:", self._sep_spin)
        self._max_spin = QSpinBox()
        self._max_spin.setRange(0, 100000)
        self._max_spin.setValue(0)
        self._max_spin.setToolTip("Cap on the number of detections (0 = unlimited).")
        self._max_spin.valueChanged.connect(lambda *_: self._rethreshold())
        detect.addRow("Max detections:", self._max_spin)
        left.addLayout(detect)

        # ring validation (reject solid blobs) — shown only for ring-like models
        self._ring_group = QWidget()
        rv = QFormLayout(self._ring_group)
        rv.setContentsMargins(0, 0, 0, 0)
        self._ring_validate = QCheckBox("Reject solid blobs (ring)")
        self._ring_validate.setToolTip(
            "Filter detections whose centre is filled, by the inside/outside-ring "
            "localization-count ratios (SMAP NPCsegmentCleanup).")
        self._ring_validate.toggled.connect(lambda *_: self._rethreshold())
        rv.addRow(self._ring_validate)
        self._max_inside = self._spin(0.0, 5.0, cs.DEFAULT_MAX_INSIDE, 2, 0.05, "")
        self._max_inside.setToolTip("Max inside/ring localization-count ratio (lower = stricter).")
        self._max_inside.valueChanged.connect(lambda *_: self._rethreshold())
        rv.addRow("Max inside ratio:", self._max_inside)
        self._max_outside = self._spin(0.0, 20.0, cs.DEFAULT_MAX_OUTSIDE, 2, 0.1, "")
        self._max_outside.setToolTip("Max outside/ring localization-count ratio (background rejection).")
        self._max_outside.valueChanged.connect(lambda *_: self._rethreshold())
        rv.addRow("Max outside ratio:", self._max_outside)
        left.addWidget(self._ring_group)

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
        self._show_response = QCheckBox("Response")
        self._show_response.setChecked(True)
        self._show_response.toggled.connect(self._update_layer_visibility)
        self._show_image = QCheckBox("Data")
        self._show_image.setChecked(False)
        self._show_image.toggled.connect(self._update_layer_visibility)
        layers.addWidget(self._show_detection)
        layers.addWidget(self._show_response)
        layers.addWidget(self._show_image)
        layers.addStretch(1)
        layers.addWidget(QLabel("B/C:"))
        self._bc_target = QComboBox()
        self._bc_target.addItems(["Response", "Data"])
        self._bc_target.setToolTip("Layer the Brightness/Contrast dialog adjusts.")
        self._bc_target.currentTextChanged.connect(self._bc.set_target)
        layers.addWidget(self._bc_target)
        self._bc_btn = QPushButton("B/C…")
        self._bc_btn.setToolTip("Open the Brightness/Contrast dialog for the selected layer.")
        self._bc_btn.clicked.connect(self._open_bc)
        layers.addWidget(self._bc_btn)
        center.addLayout(layers)

        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "X (nm)")
        self._plot.setLabel("left", "Y (nm)")

        self._resp_img = pg.ImageItem()
        self._resp_img.setZValue(0)
        try:
            self._resp_img.setLookupTable(
                pg.colormap.get(_RESPONSE_CMAP).getLookupTable(0.0, 1.0, 256))
        except Exception:
            pass
        self._plot.addItem(self._resp_img)

        self._hist_img = pg.ImageItem()
        self._hist_img.setZValue(5)
        self._hist_img.setLookupTable(self._image_lut)
        self._hist_img.setVisible(False)
        self._plot.addItem(self._hist_img)

        # Detections rendered like the data-viewer point ROIs: translucent
        # concentric jet rings at constant on-screen size (one pxMode scatter per
        # ring, outer→inner), so the data underneath stays visible.
        self._ring_scatters: list[pg.ScatterPlotItem] = []
        for i, (frac, rgba) in enumerate(PointMarkerItem._RINGS):
            sc = pg.ScatterPlotItem(size=_MARKER_PX * frac, symbol="o",
                                    pen=pg.mkPen(None), brush=pg.mkBrush(*rgba),
                                    pxMode=True)
            sc.setZValue(10 + i)                       # inner rings drawn on top
            self._plot.addItem(sc)
            self._ring_scatters.append(sc)
        self._sel_peaks = pg.ScatterPlotItem(
            size=_MARKER_PX + 8, pen=pg.mkPen(_SEL_COLOR, width=2.5), brush=None, pxMode=True)
        self._sel_peaks.setZValue(20)                  # selection ring above the markers
        self._plot.addItem(self._sel_peaks)
        # crop-box preview (data-coordinate squares, so they scale with zoom)
        self._box_item = pg.PlotDataItem(
            pen=pg.mkPen(_ROI_STROKE, width=1, style=Qt.PenStyle.DashLine), connect="finite")
        self._box_item.setZValue(15)
        self._box_item.setVisible(False)
        self._plot.addItem(self._box_item)

        center.addWidget(self._plot, 1)
        panel = QWidget()
        panel.setLayout(center)
        return panel

    def _build_detections(self) -> QWidget:
        right = QVBoxLayout()
        self._det_header = QLabel("Detections (0)")
        self._det_header.setStyleSheet("font-weight:bold;")
        right.addWidget(self._det_header)
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._on_list_selection)
        right.addWidget(self._list, 1)

        hint = QLabel("Select rows to add a subset (none selected = all).")
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        right.addWidget(hint)

        brow = QHBoxLayout()
        brow.addWidget(QLabel("Box size:"))
        self._box_size = self._spin(1.0, 100000.0, 100.0, 0, 5.0, " nm")
        self._box_size.setToolTip(
            "Side length of the square ROI placed (centred) on each detection — the "
            "crop region used for downstream particle averaging.")
        brow.addWidget(self._box_size)
        right.addLayout(brow)
        self._box_preview = QCheckBox("Preview boxes")
        self._box_preview.setToolTip("Outline the crop box around each detection in the view.")
        self._box_preview.toggled.connect(self._draw_boxes)
        self._box_size.valueChanged.connect(lambda *_: self._draw_boxes())
        right.addWidget(self._box_preview)

        self._add_btn = QPushButton("Add to Manager")
        self._add_btn.clicked.connect(self._add_to_manager)
        right.addWidget(self._add_btn)

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

    # ----------------------------------------------------- brightness/contrast
    def _register_bc_layers(self) -> None:
        """Register the Response and Data image layers with the shared B/C dialog."""
        self._bc.add_layer(
            "Response",
            get_pixels=lambda: self._response if self._response.size else None,
            apply=lambda lo, hi: self._resp_img.setLevels((lo, hi)),
            auto=lambda: percentile_levels(self._response, 0.0, 99.5),
            default=lambda: (0.0, 1.0))
        self._bc.add_layer(
            "Data",
            get_pixels=lambda: (self._H[self._H > 0] if self._H.size else None),
            apply=lambda lo, hi: self._hist_img.setLevels((lo, hi)),
            auto=lambda: percentile_levels(self._H, 0.0, 99.5, positive_only=True),
            default=lambda: (0.0, self._data_vmax))

    def _open_bc(self) -> None:
        self._bc.set_target(self._bc_target.currentText())
        self._bc.open(self)

    @staticmethod
    def _alpha_white_lut() -> np.ndarray:
        """White RGBA LUT whose alpha ramps with intensity (gamma 0.5), so the
        rendered image overlays the response with the empty background transparent."""
        lut = np.zeros((256, 4), dtype=np.ubyte)
        lut[:, :3] = 255
        lut[:, 3] = (255.0 * (np.linspace(0.0, 1.0, 256) ** 0.5)).astype(np.ubyte)
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

    def _model_key(self) -> str:
        return self._model_combo.currentData()

    def _params(self) -> dict:
        return {name: spin.value() for name, spin in self._param_spins.items()}

    # ----------------------------------------------------- parameter widgets
    def _rebuild_params(self, *, initial: bool = False) -> None:
        """Replace the per-model parameter spin boxes when the model changes.

        The leaving model's param values are cached (per-model, persisted across
        sessions) and the entering model's saved/cached values are restored."""
        if self._param_spins and self._active_model_key:    # cache the leaving model
            self._model_params[self._active_model_key] = {
                n: s.value() for n, s in self._param_spins.items()}
        while self._param_form.rowCount():
            self._param_form.removeRow(0)
        self._param_spins.clear()
        model = cs.get_model(self._model_key())
        saved = self._model_params.get(self._model_key(), {})
        for ps in model.params:
            spin = self._spin(ps.lo, ps.hi, float(saved.get(ps.name, ps.default)),
                              ps.decimals, ps.step, ps.suffix)
            if ps.tooltip:
                spin.setToolTip(ps.tooltip)
            spin.valueChanged.connect(self._schedule_recompute)
            self._param_form.addRow(ps.label + ":", spin)
            self._param_spins[ps.name] = spin
        self._active_model_key = self._model_key()
        # default the separation + crop-box size to the model's characteristic size
        self._suspend = True
        self._sep_spin.setValue(model.size(model.defaults()))
        self._box_size.setValue(model.size(model.defaults()))
        self._suspend = False
        self._ring_group.setVisible(model.ringlike)   # ring validation: ring-like only
        if not initial:
            self._recompute()

    # ----------------------------------------------------- persistence
    def _restore_prefs(self) -> None:
        """Restore the last-session parameters from ``state.prefs`` (or fall back
        to the built-in defaults), then build the param spins for the model."""
        cfg = dict(self._state.prefs.get("conv_segmentation", {}) or {})
        self._model_params = dict(cfg.get("model_params", {}) or {})
        mk = cfg.get("model")
        self._model_combo.blockSignals(True)
        if mk in cs.GEOMETRY_MODELS:
            i = self._model_combo.findData(mk)
            if i >= 0:
                self._model_combo.setCurrentIndex(i)
        self._model_combo.blockSignals(False)
        self._rebuild_params(initial=True)          # builds spins (restores per-model params)
        if not cfg:
            return
        self._suspend = True
        try:
            for spin, key in ((self._pixel_spin, "pixel_size"),
                              (self._response_spin, "min_response"),
                              (self._sep_spin, "min_separation"),
                              (self._box_size, "box_size"),
                              (self._max_inside, "max_inside"),
                              (self._max_outside, "max_outside")):
                if key in cfg:
                    spin.setValue(float(cfg[key]))
            if "max_detections" in cfg:
                self._max_spin.setValue(int(cfg["max_detections"]))
            for chk, key in ((self._zero_mean, "zero_mean"), (self._sqrt, "sqrt"),
                             (self._dog, "dog"), (self._ring_validate, "ring_validate")):
                if key in cfg:
                    chk.setChecked(bool(cfg[key]))
        finally:
            self._suspend = False

    def _save_prefs(self) -> None:
        """Persist the current parameters to ``state.prefs`` (across sessions)."""
        if self._active_model_key:
            self._model_params[self._active_model_key] = self._params()
        self._state.prefs["conv_segmentation"] = {
            "model": self._model_key(),
            "pixel_size": self._pixel_spin.value(),
            "zero_mean": self._zero_mean.isChecked(),
            "sqrt": self._sqrt.isChecked(),
            "dog": self._dog.isChecked(),
            "min_response": self._response_spin.value(),
            "min_separation": self._sep_spin.value(),
            "max_detections": self._max_spin.value(),
            "ring_validate": self._ring_validate.isChecked(),
            "max_inside": self._max_inside.value(),
            "max_outside": self._max_outside.value(),
            "box_size": self._box_size.value(),
            "model_params": self._model_params,
        }
        try:
            self._state.save_prefs()
        except Exception:
            pass

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        try:
            self._save_prefs()
        except Exception:
            pass
        try:
            self._bc.close()
        except Exception:
            pass
        super().closeEvent(event)

    # ----------------------------------------------------------- compute
    def _schedule_recompute(self) -> None:
        if not self._suspend:
            self._recompute_timer.start()

    def _recompute(self) -> None:
        """Full update: build kernel, render histogram, FFT-convolve."""
        px = float(self._pixel_spin.value())
        kernel = cs.build_kernel(self._model_key(), self._params(), px,
                                 zero_mean=self._zero_mean.isChecked())
        self._draw_kernel(kernel)

        xy = self._xy
        if xy.shape[0] < 3:
            self._reset_maps("Need at least 3 localizations.")
            return
        nx = int((xy[:, 0].max() - xy[:, 0].min()) / px) + 2
        ny = int((xy[:, 1].max() - xy[:, 1].min()) / px) + 2
        if nx * ny > _MAX_IMAGE_PX:
            self._reset_maps("")
            self._info_label.setText(
                f"Image {nx}×{ny} too large at {px:g} nm/px — increase the pixel size.")
            return

        # self._H stays the raw counts (for the rendered-image layer); the
        # convolution uses an optionally √-stabilized copy.
        self._H, self._xedge, self._yedge = cs.render_histogram_2d(xy[:, 0], xy[:, 1], px)
        H_conv = cs.stabilize_histogram(self._H, self._sqrt.isChecked())
        size_nm = cs.get_model(self._model_key()).size(self._params())
        dog_radius_px = (0.5 * size_nm / px) if self._dog.isChecked() else None
        self._response = cs.convolve_response(H_conv, kernel, dog_radius_px=dog_radius_px)
        self._info_label.setText(
            f"{xy.shape[0]:,} localizations · {self._H.shape[0]}×{self._H.shape[1]} px")
        self._draw_layers()
        self._rethreshold()

    def _reset_maps(self, status: str) -> None:
        self._H = np.zeros((0, 0))
        self._response = np.zeros((0, 0))
        self._resp_img.clear()
        self._hist_img.clear()
        self._set_rings(None)
        self._sel_peaks.clear()
        self._centers = np.empty((0, 2))
        self._values = np.empty(0)
        self._inside = np.empty(0)
        self._labels = []
        self._populate_list()
        self._status_label.setText(status)
        self._add_btn.setEnabled(False)

    def _rethreshold(self) -> None:
        """Cheap update: re-run NMS on the cached response map, then optional
        ring validation (reject solid blobs)."""
        if self._suspend or self._response.size == 0:
            return
        px = float(self._pixel_spin.value())
        centers, vals = cs.detections_from_response(
            self._response, self._xedge, self._yedge,
            min_response=float(self._response_spin.value()),
            min_distance_px=float(self._sep_spin.value()) / px,
            max_detections=(self._max_spin.value() or None))
        inside = np.full(centers.shape[0], np.nan)
        model = cs.get_model(self._model_key())
        if (model.ringlike and model.ring_geom is not None
                and self._ring_validate.isChecked() and centers.shape[0]):
            radius_nm, rim_nm = model.ring_geom(self._params())
            keep, ins, _outs = cs.ring_validation_mask(
                self._xy, centers, radius_nm, rim_nm,
                max_inside=float(self._max_inside.value()),
                max_outside=float(self._max_outside.value()))
            centers, vals, inside = centers[keep], vals[keep], ins[keep]
        self._centers = centers
        self._values = vals
        self._inside = inside
        prefix = self._name_prefix()
        self._labels = [f"{prefix} {i}" for i in range(1, centers.shape[0] + 1)]
        self._set_rings(centers)
        self._populate_list()
        self._status_label.setText("")
        self._add_btn.setEnabled(centers.shape[0] > 0)

    def _name_prefix(self) -> str:
        return cs.get_model(self._model_key()).label.split(" ")[0]

    def _set_rings(self, centers: np.ndarray) -> None:
        """Position (or clear) every concentric-ring scatter at the detections."""
        if centers is not None and centers.shape[0]:
            for sc in self._ring_scatters:
                sc.setData(x=centers[:, 0], y=centers[:, 1])
        else:
            for sc in self._ring_scatters:
                sc.clear()
        self._draw_boxes()

    def _draw_boxes(self) -> None:
        """Outline the square crop box (``Box size``) around each detection."""
        show = self._box_preview.isChecked() and self._show_detection.isChecked()
        if not show or self._centers.shape[0] == 0:
            self._box_item.clear()
            self._box_item.setVisible(False)
            return
        half = float(self._box_size.value()) / 2.0
        xs, ys = [], []
        for cx, cy in self._centers:
            xs += [cx - half, cx + half, cx + half, cx - half, cx - half, np.nan]
            ys += [cy - half, cy - half, cy + half, cy + half, cy - half, np.nan]
        self._box_item.setData(x=np.asarray(xs), y=np.asarray(ys))
        self._box_item.setVisible(True)

    # ----------------------------------------------------------- detections list
    def _populate_list(self) -> None:
        self._suspend_list = True
        self._list.clear()
        for i, (cx, cy) in enumerate(self._centers):
            r = float(self._values[i]) if i < self._values.size else float("nan")
            text = f"{self._labels[i]}   ({cx:.0f}, {cy:.0f})   r={r:.2f}"
            if i < self._inside.size and np.isfinite(self._inside[i]):
                text += f"   in={self._inside[i]:.2f}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._list.addItem(item)
        self._suspend_list = False
        self._det_header.setText(f"Detections ({self._centers.shape[0]})")
        self._sel_peaks.clear()

    def _on_list_selection(self) -> None:
        if getattr(self, "_suspend_list", False):
            return
        idx = [it.data(Qt.ItemDataRole.UserRole) for it in self._list.selectedItems()]
        if idx and self._centers.shape[0]:
            sel = self._centers[np.asarray(idx, dtype=int)]
            self._sel_peaks.setData(x=sel[:, 0], y=sel[:, 1])
        else:
            self._sel_peaks.clear()

    def _selected_indices(self) -> np.ndarray:
        idx = [it.data(Qt.ItemDataRole.UserRole) for it in self._list.selectedItems()]
        if idx:
            return np.sort(np.asarray(idx, dtype=int))
        return np.arange(self._centers.shape[0])      # none selected → all

    # ----------------------------------------------------------- drawing
    def _draw_kernel(self, kernel: np.ndarray) -> None:
        if kernel.size == 0:
            self._kernel_img.clear()
            return
        m = float(np.max(np.abs(kernel))) or 1.0
        self._kernel_img.setImage(self._oriented(kernel), levels=(-m, m))
        self._kernel_plot.getViewBox().autoRange(padding=0.02)

    def _draw_layers(self) -> None:
        if self._response.size == 0:
            self._resp_img.clear()
            self._hist_img.clear()
            return
        rect = QRectF(float(self._xedge[0]), float(self._yedge[0]),
                      float(self._xedge[-1] - self._xedge[0]),
                      float(self._yedge[-1] - self._yedge[0]))
        pos = self._H[self._H > 0]
        self._data_vmax = float(np.percentile(pos, 99.0)) if pos.size else 1.0
        self._data_vmax = max(self._data_vmax, 1.0)

        resp_lv = self._bc.effective_levels("Response") or (0.0, 1.0)
        self._resp_img.setImage(self._oriented(self._response), levels=resp_lv)
        self._resp_img.setRect(rect)

        data_lv = self._bc.effective_levels("Data") or (0.0, self._data_vmax)
        self._hist_img.setImage(self._oriented(self._H), levels=data_lv)
        self._hist_img.setRect(rect)
        self._update_layer_visibility()
        self._bc.refresh_if_open()

        if not self._fitted:        # keep the user's pan/zoom on later recomputes
            self._plot.getViewBox().autoRange(padding=0.02)
            self._fitted = True

    def _update_layer_visibility(self) -> None:
        self._resp_img.setVisible(self._show_response.isChecked()
                                  and self._response.size > 0)
        self._hist_img.setVisible(self._show_image.isChecked() and self._H.size > 0)
        det = self._show_detection.isChecked()
        for sc in self._ring_scatters:
            sc.setVisible(det)
        self._sel_peaks.setVisible(det)
        self._draw_boxes()

    @staticmethod
    def _oriented(img: np.ndarray) -> np.ndarray:
        """Account for the app's global pyqtgraph image axis order.

        Our arrays are indexed ``[x_bin, y_bin]``; row-major ImageItems expect
        ``[row=y, col=x]`` so they need a transpose (the app runs row-major)."""
        try:
            row_major = pg.getConfigOption("imageAxisOrder") == "row-major"
        except Exception:
            row_major = True
        return img.T if row_major else img

    # ----------------------------------------------------------- commit
    def _add_to_manager(self) -> None:
        sel = self._selected_indices()
        if sel.size == 0 or self._owner is None:
            return
        if not hasattr(self._owner, "add_segmentation_rois"):
            return
        ds = self._dataset()
        model = cs.get_model(self._model_key())
        params = self._params()
        side = max(float(self._box_size.value()), float(self._pixel_spin.value()) * 2.0)
        centers = self._centers[sel]
        names = [self._labels[i] for i in sel]
        opts = []
        if self._zero_mean.isChecked():
            opts.append("zero-mean kernel")
        if self._sqrt.isChecked():
            opts.append("sqrt-stabilized")
        if self._dog.isChecked():
            opts.append("DoG band-pass")
        if model.ringlike and self._ring_validate.isChecked():
            opts.append(f"ring validation (max inside={self._max_inside.value():.2f}, "
                        f"max outside={self._max_outside.value():.2f})")
        opts_str = ", ".join(opts) if opts else "raw kernel"
        log = (
            f"Convolution segmentation: added {centers.shape[0]} of "
            f"{self._centers.shape[0]} {model.label} detection(s) on "
            f"'{ds.name if ds else '?'}' "
            f"(pixel={self._pixel_spin.value():.1f} nm, "
            f"min response={self._response_spin.value():.2f}, "
            f"min separation={self._sep_spin.value():.0f} nm, "
            f"box={side:.0f} nm; "
            f"{opts_str}; {self._param_summary(model, params)}); added rectangle ROIs.")
        n = self._owner.add_segmentation_rois(
            self._idx, centers, side_nm=side,
            name_prefix=self._name_prefix(), source="conv_segmentation_2d",
            stroke_color=_ROI_STROKE, names=names, log_message=log)
        if n:
            self._status_label.setText(f"Added {n} ROI(s) to the ROI Manager.")

    @staticmethod
    def _param_summary(model, params: dict) -> str:
        return ", ".join(f"{ps.label.lower()}={params[ps.name]:g}{ps.suffix.strip()}"
                         for ps in model.params)
