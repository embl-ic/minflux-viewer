"""
minflux_viewer.ui.analysis.conv_segmentation_dialog
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
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from minflux_viewer.analysis import conv_segmentation as cs
from minflux_viewer.colormaps import colormap_lut
from minflux_viewer.core.computation_snapshot import input_signature
from minflux_viewer.core.cuda_fft import CudaFftSettings
from minflux_viewer.ui.common.background_tasks import (
    BackgroundTask,
    detach_task_signals,
    retire_background_tasks,
    shared_thread_pool,
)
from minflux_viewer.ui.common.dataset_binding import DatasetIndex, window_dataset_key
from minflux_viewer.ui.common.qt_lifecycle import qobject_alive
from minflux_viewer.ui.plots.plot_format import apply_spatial_y_direction
from minflux_viewer.ui.roi.roi_overlay import PointMarkerItem
from minflux_viewer.ui.widgets.layer_brightness import LayerBrightness, percentile_levels

_RESPONSE_CMAP = "inferno"
_SEL_COLOR = (255, 235, 0)      # yellow ring on the selected detections
_ROI_STROKE = "#00e5ff"
_MARKER_PX = 26                 # on-screen diameter of the point-ROI-style marker


class ConvSegmentationWindow(QDialog):
    """Modeless interactive geometry-kernel convolution segmentation tool."""

    # The window's dataset key, read and set as a position (None once the
    # dataset has closed). See ui/common/dataset_binding.py.
    _idx = DatasetIndex()

    def __init__(self, state, dataset_idx: int, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._idx = dataset_idx
        self._owner = owner
        self.setWindowTitle("Convolution Segmentation")
        self.resize(1180, 700)

        self._xy = self._load_xy()
        self._xy.flags.writeable = False
        self._closed = False
        self._generation = 0
        self._task: BackgroundTask | None = None
        self._pending = False
        self._map_key = None
        self._data_key: tuple | None = None
        self._preview_error: str | None = None
        self._map_error: str | None = None
        self._required_budgets: tuple[int, int] | None = None
        self._fft_provenance: dict = {}
        self._kernel = np.empty((0, 0))
        self._param_spins: dict[str, QDoubleSpinBox] = {}
        self._H = np.zeros((0, 0))
        self._response = np.zeros((0, 0))
        self._xedge = np.zeros(0)
        self._yedge = np.zeros(0)
        self._centers = np.empty((0, 2))
        self._values = np.empty(0)
        self._inside = np.empty(0)        # inside-ring ratio per detection (ring validation)
        self._support = np.empty(0)       # ring support score per detection (ring validation)
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
        for signal in (state.dataset_events.filter_changed, state.dataset_events.attributes_changed,
                       state.dataset_events.calibration_changed, state.dataset_events.overlay_transform_changed,
                       state.dataset_events.removed):
            signal.connect(self._source_changed)
        self._schedule_recompute()

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
        self._backend_combo = QComboBox()
        self._backend_combo.addItem("CPU (SciPy)", "cpu")
        self._backend_combo.addItem("CUDA FFT (CuPy)", "cuda")
        self._backend_combo.setToolTip(
            "Accelerates the FFT only, using float64. No automatic CPU fallback. "
            "Requires a separate trusted Python environment with CuPy and CUDA.")
        self._backend_combo.currentIndexChanged.connect(self._backend_changed)
        common.addRow("FFT backend:", self._backend_combo)
        self._cuda_controls = QWidget()
        cuda_form = QFormLayout(self._cuda_controls)
        cuda_form.setContentsMargins(0, 0, 0, 0)
        self._cuda_python = QLineEdit()
        self._cuda_python.setPlaceholderText("External Python executable")
        self._cuda_python.setToolTip("Absolute path to a trusted external Python; not site-packages or the viewer executable.")
        self._cuda_python.textChanged.connect(self._schedule_recompute)
        browse = QPushButton("…")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._choose_cuda_python)
        python_row = QHBoxLayout()
        python_row.addWidget(self._cuda_python)
        python_row.addWidget(browse)
        cuda_form.addRow("Python:", python_row)
        self._cuda_device = QSpinBox()
        self._cuda_device.setRange(0, 63)
        self._cuda_device.valueChanged.connect(self._schedule_recompute)
        cuda_form.addRow("Device index:", self._cuda_device)
        self._cuda_memory = QSpinBox()
        self._cuda_memory.setRange(64, 65536)
        self._cuda_memory.setValue(512)
        self._cuda_memory.setSuffix(" MiB")
        self._cuda_memory.setToolTip("Conservative device estimate cap; also refuses estimates over 80% of free VRAM. Not total VRAM usage.")
        self._cuda_memory.valueChanged.connect(self._schedule_recompute)
        cuda_form.addRow("GPU budget:", self._cuda_memory)
        self._cuda_controls.hide()
        common.addRow(self._cuda_controls)
        self._pixel_spin = self._spin(0.5, 200.0, 4.0, 1, 0.5, " nm")
        self._pixel_spin.valueChanged.connect(self._schedule_recompute)
        common.addRow("Pixel size:", self._pixel_spin)
        self._memory_spin = QSpinBox()
        self._memory_spin.setRange(32, 16384)
        self._memory_spin.setValue(cs.DEFAULT_MEMORY_BUDGET // 1024**2)
        self._memory_spin.setSuffix(" MiB")
        self._memory_spin.setToolTip("Conservative per-operation allocation estimate, not a total application memory limit.")
        self._memory_spin.valueChanged.connect(self._schedule_recompute)
        common.addRow("Memory budget:", self._memory_spin)
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
            "aggregates that would otherwise dominate the response.")
        self._sqrt.toggled.connect(self._schedule_recompute)
        common.addRow("", self._sqrt)
        self._dog = QCheckBox("DoG band-pass")
        self._dog.setChecked(cs.DEFAULT_DOG)
        self._dog.setToolTip(
            "Difference-of-Gaussians band-pass on the response at the structure "
            "scale, suppressing slow background.")
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
            "localization-count ratios.")
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
        self._min_support = self._spin(0.0, 1.0, cs.DEFAULT_MIN_SUPPORT, 2, 0.05, "")
        self._min_support.setToolTip(
            "Min ring support score: annulus angular coverage × radial fit. Unlike the "
            "inside/outside ratios this sees whether the ring is populated all the way "
            "round, so it rejects partial arcs. 0 = criterion off.")
        self._min_support.valueChanged.connect(lambda *_: self._rethreshold())
        rv.addRow("Min ring support:", self._min_support)
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(320)
        scroll.setWidget(panel)
        return scroll

    def _build_center(self) -> QWidget:
        center = QVBoxLayout()
        layers = QHBoxLayout()
        layers.addWidget(QLabel("Show:"))
        # Default layers: Detection + Data on, Response off (restored from prefs if
        # the user changed them — see _restore_prefs / _save_prefs).
        self._show_detection = QCheckBox("Detection")
        self._show_detection.setChecked(True)
        self._show_detection.toggled.connect(self._update_layer_visibility)
        self._show_response = QCheckBox("Response")
        self._show_response.setChecked(False)
        self._show_response.toggled.connect(self._update_layer_visibility)
        self._show_image = QCheckBox("Data")
        self._show_image.setChecked(True)
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

        notice = QHBoxLayout()
        self._preview_notice = QLabel()
        self._preview_notice.setTextFormat(Qt.TextFormat.PlainText)
        self._preview_notice.setWordWrap(True)
        self._preview_notice.setStyleSheet("font-weight: bold;")
        self._preview_notice.hide()
        notice.addWidget(self._preview_notice, 1)
        self._budget_btn = QPushButton("Use required budgets")
        self._budget_btn.setToolTip("Explicitly raise only the estimated RAM/GPU budgets; keep the pixel size and backend unchanged.")
        self._budget_btn.clicked.connect(self._use_required_budgets)
        self._budget_btn.hide()
        notice.addWidget(self._budget_btn)
        center.addLayout(notice)

        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "X (nm)")
        self._plot.setLabel("left", "Y (nm)")
        apply_spatial_y_direction(
            self._plot,
            vertical_coordinate="Y",
            prefs=self._state.prefs,
            preference_key="render_xy_origin",
        )

        self._resp_img = pg.ImageItem()
        self._resp_img.setZValue(0)
        try:
            self._resp_img.setLookupTable(colormap_lut(_RESPONSE_CMAP, alpha=False))
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
        if self._idx is not None and 0 <= self._idx < len(self._state.datasets):
            return self._state.datasets[self._idx]
        return None

    def _load_xy(self) -> np.ndarray:
        ds = self._dataset()
        if ds is None:
            return np.empty((0, 2))
        # Detect in display coordinates (loc_nm + overlay transform) — the frame
        # ROIs live in — so detections land correctly on an overlay channel.
        from minflux_viewer.core.roi_crop import display_xy_filtered
        return display_xy_filtered(ds)

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
            self._schedule_recompute()

    # ----------------------------------------------------- persistence
    def _backend_changed(self, *_args) -> None:
        self._cuda_controls.setVisible(self._backend_combo.currentData() == "cuda")
        self._schedule_recompute()

    def _choose_cuda_python(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Select trusted CUDA Python executable")
        if path:
            self._cuda_python.setText(path)

    def _cuda_settings(self) -> CudaFftSettings | None:
        if self._backend_combo.currentData() == "cpu":
            return None
        return CudaFftSettings(self._cuda_python.text().strip(), self._cuda_device.value(),
                               self._cuda_memory.value() * 1024**2)

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
            self._cuda_python.setText(str(cfg.get("cuda_python", "")))
            self._cuda_device.setValue(int(cfg.get("cuda_device", 0)))
            self._cuda_memory.setValue(int(cfg.get("cuda_memory_mib", 512)))
            self._backend_combo.setCurrentIndex(max(0, self._backend_combo.findData(cfg.get("fft_backend", "cpu"))))
            for spin, key in ((self._pixel_spin, "pixel_size"),
                              (self._response_spin, "min_response"),
                              (self._sep_spin, "min_separation"),
                              (self._box_size, "box_size"),
                              (self._max_inside, "max_inside"),
                              (self._max_outside, "max_outside"),
                              (self._min_support, "min_support")):
                if key in cfg:
                    spin.setValue(float(cfg[key]))
            if "max_detections" in cfg:
                self._max_spin.setValue(int(cfg["max_detections"]))
            if "memory_budget_mib" in cfg:
                self._memory_spin.setValue(int(cfg["memory_budget_mib"]))
            for chk, key in ((self._zero_mean, "zero_mean"), (self._sqrt, "sqrt"),
                             (self._dog, "dog"), (self._ring_validate, "ring_validate"),
                             (self._show_detection, "show_detection"),
                             (self._show_response, "show_response"),
                             (self._show_image, "show_data")):
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
            "fft_backend": self._backend_combo.currentData(),
            "cuda_python": self._cuda_python.text().strip(),
            "cuda_device": self._cuda_device.value(),
            "cuda_memory_mib": self._cuda_memory.value(),
            "pixel_size": self._pixel_spin.value(),
            "memory_budget_mib": self._memory_spin.value(),
            "zero_mean": self._zero_mean.isChecked(),
            "sqrt": self._sqrt.isChecked(),
            "dog": self._dog.isChecked(),
            "min_response": self._response_spin.value(),
            "min_separation": self._sep_spin.value(),
            "max_detections": self._max_spin.value(),
            "ring_validate": self._ring_validate.isChecked(),
            "max_inside": self._max_inside.value(),
            "max_outside": self._max_outside.value(),
            "min_support": self._min_support.value(),
            "box_size": self._box_size.value(),
            "model_params": self._model_params,
            "show_detection": self._show_detection.isChecked(),
            "show_response": self._show_response.isChecked(),
            "show_data": self._show_image.isChecked(),
        }
        try:
            self._state.save_prefs()
        except (OSError, TypeError, ValueError):
            import logging
            logging.getLogger(__name__).exception("Could not save segmentation preferences")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        from minflux_viewer.ui.common.qt_lifecycle import cleanup_callback, dispose_plot_widgets

        self._closed = True
        self._generation += 1
        self._pending = False
        self._recompute_timer.stop()
        if self._task is not None:
            retire_background_tasks([self._task])
            self._task = None
        cleanup_callback(self._save_prefs, context="save segmentation preferences on exit")
        cleanup_callback(self._bc.close, context="close segmentation brightness dialog")

        # Retire the pyqtgraph plots last, after this window's own
        # teardown, so nothing here touches an already-inert plot.
        dispose_plot_widgets(self)
        super().closeEvent(event)

    # ----------------------------------------------------------- compute
    def _schedule_recompute(self) -> None:
        if not self._suspend and not self._closed:
            self._preview_error = self._map_error = None
            self._required_budgets = None
            self._preview_notice.hide()
            self._budget_btn.hide()
            self._map_key = None
            self._generation += 1
            self._pending = True
            if self._task is not None:
                self._task.cancel()
            self._reset_maps("Updating convolution preview…")
            self._kernel_img.clear()
            self._recompute_timer.start()

    def _source_stamp(self):
        ds = self._dataset()
        key = window_dataset_key(self)
        if ds is None or key is None:
            return None
        return (self._state.data_revision(key), input_signature(ds),
                repr(ds.state.get("overlay_transform")), repr(ds.state.get("render_transform_2d")))

    def _source_changed(self, event) -> None:
        if self._closed or event.key != window_dataset_key(self):
            return
        if self._dataset() is None:
            self._generation += 1
            self._pending = False
            self._recompute_timer.stop()
            if self._task is not None:
                self._task.cancel()
            self._reset_maps("Dataset is closed.")
            self._required_budgets = None
            self._budget_btn.hide()
            self._show_map_error("Dataset is closed.")
            return
        self._xy = self._load_xy()
        self._xy.flags.writeable = False
        self._fitted = False
        self._schedule_recompute()

    def _recompute(self) -> None:
        """Dispatch the latest captured request; one active job per window."""
        if self._closed or self._dataset() is None:
            return
        if self._task is not None:
            self._pending = True
            return
        self._pending = False
        px = float(self._pixel_spin.value())
        key, params = self._model_key(), self._params()
        generation, stamp, xy = self._generation, self._source_stamp(), self._xy
        zero, sqrt, dog = self._zero_mean.isChecked(), self._sqrt.isChecked(), self._dog.isChecked()
        budget = int(self._memory_spin.value()) * 1024**2
        cuda_settings = self._cuda_settings()
        map_key = (stamp, key, tuple(params.items()), px, zero, sqrt, dog, budget, cuda_settings)
        cached = (dict(kernel=self._kernel, histogram=self._H, response_map=self._response,
                       xedge=self._xedge, yedge=self._yedge, fft_provenance=self._fft_provenance)
                  if self._map_key == map_key else None)
        error, self._preview_error = self._preview_error, None
        estimated_bytes = 0
        try:
            estimate = cs.estimate_convolution(xy, key, params, px)
            self._offer_budgets(estimate.bytes_required,
                                cuda_settings.estimate(estimate.image_shape, estimate.kernel_shape)
                                if cuda_settings is not None else 0)
            if error is None:
                cs.preflight_convolution(xy, key, params, px, budget_bytes=budget)
                if cuda_settings is not None:
                    cuda_settings.preflight(estimate.image_shape, estimate.kernel_shape)
                estimated_bytes = estimate.bytes_required
        except (ValueError, MemoryError) as exc:
            error = str(exc)
        data_only = error is not None
        if error is not None:
            self._show_map_error(error)
            if self._data_key == (stamp, px):
                self._draw_layers()
                return
            try:
                estimated_bytes = cs.preflight_histogram(xy, px, budget_bytes=budget)
            except (ValueError, MemoryError) as exc:
                self._show_map_error(f"{error}\nData preview also unavailable: {exc}")
                return
        threshold = dict(model_key=key, params=params, pixel_size_nm=px,
                         min_response=float(self._response_spin.value()),
                         min_distance_nm=float(self._sep_spin.value()),
                         max_detections=self._max_spin.value() or None,
                         ring_validation=(dict(max_inside=float(self._max_inside.value()),
                                               max_outside=float(self._max_outside.value()),
                                               min_support=float(self._min_support.value()))
                                          if self._ring_validate.isChecked() else None))

        def work(report):
            if data_only:
                return cs.histogram_maps(xy, pixel_size_nm=px, budget_bytes=budget,
                                          report=report), None
            maps = cached if cached is not None else cs.convolution_maps(
                xy, model_key=key, params=params, pixel_size_nm=px, zero_mean=zero,
                sqrt_stabilize=sqrt, dog=dog, budget_bytes=budget, report=report,
                cuda_settings=cuda_settings)
            detections = cs.threshold_response_maps(xy, maps, **threshold, report=report)
            report("Convolution preview ready")
            return maps, detections

        task = BackgroundTask(work, description=("Convolution input preview (FFT unavailable)" if data_only
                                                else "Convolution segmentation preview"), category="analysis",
                              estimated_bytes=estimated_bytes)
        self._task = task
        task.signals.done.connect(lambda result: self._computed(task, generation, stamp,
                                                               None if data_only else map_key, result))
        task.signals.failed.connect(lambda message: self._compute_failed(task, generation, stamp, message,
                                                                        data_only=data_only))
        task.signals.finished.connect(lambda: self._compute_finished(task))
        shared_thread_pool("convolution-segmentation").start(task)

    def _delivery_current(self, task: BackgroundTask, generation, stamp) -> bool:
        return (not self._closed and qobject_alive(self) and not task.is_cancelled
                and generation == self._generation and stamp == self._source_stamp()
                and self._task is task)

    def _computed(self, task, generation, stamp, map_key, result) -> None:
        if not self._delivery_current(task, generation, stamp):
            return
        maps, detections = result
        self._H = maps["histogram"]
        self._xedge, self._yedge = maps["xedge"], maps["yedge"]
        self._data_key = (stamp, float(self._pixel_spin.value()))
        if map_key is None:
            self._draw_layers()
            self._info_label.setText(f"{len(self._xy):,} localizations · Data only · {self._map_error}")
            return
        self._map_key = map_key
        self._kernel = maps["kernel"]
        self._response = maps["response_map"]
        self._fft_provenance = maps["fft_provenance"]
        self._map_error = None
        self._required_budgets = None
        self._preview_notice.hide()
        self._budget_btn.hide()
        self._draw_kernel(self._kernel)
        self._info_label.setText(
            f"{self._xy.shape[0]:,} localizations · {self._H.shape[0]}×{self._H.shape[1]} px"
            f" · {self._fft_provenance['backend']}")
        import json
        self._info_label.setToolTip(json.dumps(self._fft_provenance, indent=2))
        self._centers, self._values = detections["centers"], detections["peak_values"]
        self._inside, self._support = detections["inside_ratio"], detections["support"]
        self._labels = [f"{self._name_prefix()} {i}" for i in range(1, len(self._centers) + 1)]
        self._draw_layers()
        self._set_rings(self._centers)
        self._populate_list()
        self._status_label.setText("" if len(self._xy) >= 3 else "Need at least 3 localizations.")
        self._add_btn.setEnabled(len(self._centers) > 0)

    def _compute_failed(self, task, generation, stamp, message, *, data_only=False) -> None:
        if self._delivery_current(task, generation, stamp):
            self._map_key = None
            self._reset_maps(message)
            self._show_map_error(message)
            if not data_only and self._data_key != (stamp, float(self._pixel_spin.value())):
                self._preview_error = message
                self._pending = True

    def _show_map_error(self, message: str) -> None:
        self._map_key = None
        self._map_error = message
        self._info_label.setText(message)
        summary = " ".join(message.split())
        if len(summary) > 600:
            summary = summary[:600] + "… (full details in tooltip)"
        self._preview_notice.setText(f"Convolution unavailable — {summary}\n"
                                     "Input Data is independent; no response or detections have been computed.")
        self._preview_notice.setToolTip(message)
        self._preview_notice.show()
        self._status_label.setText(message)
        self._add_btn.setEnabled(False)

    def _offer_budgets(self, host_bytes: int, device_bytes: int) -> None:
        mib = 1024**2
        host = max(self._memory_spin.value(), (host_bytes + mib - 1) // mib)
        device = max(self._cuda_memory.value(), (device_bytes + mib - 1) // mib)
        needed = host > self._memory_spin.value() or device > self._cuda_memory.value()
        possible = host <= self._memory_spin.maximum() and device <= self._cuda_memory.maximum()
        self._required_budgets = (host, device) if needed and possible else None
        self._budget_btn.setVisible(self._required_budgets is not None)
        self._budget_btn.setToolTip(f"Explicitly use RAM {host:,} MiB and GPU {device:,} MiB estimate limits. "
                                    "Does not change pixel size/backend or override shared admission/free VRAM.")

    def _use_required_budgets(self) -> None:
        budgets = self._required_budgets
        if budgets is None or self._closed:
            return
        self._suspend = True
        try:
            self._memory_spin.setValue(budgets[0])
            self._cuda_memory.setValue(budgets[1])
        finally:
            self._suspend = False
        self._schedule_recompute()

    def _compute_finished(self, task) -> None:
        # Signals own callbacks capturing the task/window/maps. Break that
        # Qt/Python cycle on the GUI thread after terminal delivery.
        detach_task_signals(task)
        if self._closed or not qobject_alive(self) or self._task is not task:
            return
        self._task = None
        if self._pending and not self._recompute_timer.isActive():
            self._recompute()

    def _reset_maps(self, status: str) -> None:
        self._fft_provenance = {}
        self._info_label.setToolTip("")
        if self._data_key != (self._source_stamp(), float(self._pixel_spin.value())):
            self._H = np.zeros((0, 0))
            self._xedge = self._yedge = np.zeros(0)
            self._data_key = None
        self._response = np.zeros((0, 0))
        self._resp_img.clear()
        self._hist_img.clear()
        self._set_rings(None)
        self._sel_peaks.clear()
        self._centers = np.empty((0, 2))
        self._values = np.empty(0)
        self._inside = np.empty(0)
        self._support = np.empty(0)
        self._labels = []
        self._populate_list()
        self._status_label.setText(status)
        self._add_btn.setEnabled(False)
        self._draw_layers()

    def _rethreshold(self) -> None:
        """Coalesce detection-only changes; reuse the captured response map."""
        if self._suspend or self._closed:
            return
        if self._map_key is None and self._map_error is not None:
            return  # Thresholds cannot repair unavailable FFTs; keep the input preview.
        self._generation += 1
        self._pending = True
        if self._task is not None:
            self._task.cancel()
        self._add_btn.setEnabled(False)
        self._recompute_timer.start()

    def _name_prefix(self) -> str:
        return cs.get_model(self._model_key()).label.split(" ")[0]

    def _set_rings(self, centers: np.ndarray | None) -> None:
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
            if i < self._support.size and np.isfinite(self._support[i]):
                text += f"   sup={self._support[i]:.2f}"
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
        if self._H.size == 0:
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
        if self._response.size:
            self._resp_img.setImage(self._oriented(self._response), levels=resp_lv)
            self._resp_img.setRect(rect)
        else:
            self._resp_img.clear()

        data_lv = self._bc.effective_levels("Data") or (0.0, self._data_vmax)
        self._hist_img.setImage(self._oriented(self._H), levels=data_lv)
        self._hist_img.setRect(rect)
        self._update_layer_visibility()
        self._bc.refresh_if_open()

        if not self._fitted:        # keep the user's pan/zoom on later recomputes
            # Hidden layers have no childrenBounds. Fit the captured image's
            # world rectangle even when the user initially hid all three layers.
            self._plot.getViewBox().setRange(rect, padding=0.02)
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
        if self._closed or self._pending or self._task is not None or self._map_key is None:
            return
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
        import json
        opts.append("FFT " + json.dumps(self._fft_provenance, sort_keys=True))
        if self._zero_mean.isChecked():
            opts.append("zero-mean kernel")
        if self._sqrt.isChecked():
            opts.append("sqrt-stabilized")
        if self._dog.isChecked():
            opts.append("DoG band-pass")
        if model.ringlike and self._ring_validate.isChecked():
            rv = (f"max inside={self._max_inside.value():.2f}, "
                  f"max outside={self._max_outside.value():.2f}")
            if self._min_support.value() > 0:
                rv += f", min support={self._min_support.value():.2f}"
            opts.append(f"ring validation ({rv})")
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
            names=names, log_message=log)   # stroke_color defaults to the system ROI color
        if n:
            self._status_label.setText(f"Added {n} ROI(s) to the ROI Manager.")

    @staticmethod
    def _param_summary(model, params: dict) -> str:
        return ", ".join(f"{ps.label.lower()}={params[ps.name]:g}{ps.suffix.strip()}"
                         for ps in model.params)
