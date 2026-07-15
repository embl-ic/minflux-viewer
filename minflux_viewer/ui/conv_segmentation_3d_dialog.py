"""
minflux_viewer.ui.conv_segmentation_3d_dialog
==============================================
Interactive **3-D convolution segmentation** tool
(Analyze › Segmentation › Convolution (3D)…).

The volumetric counterpart of :mod:`conv_segmentation_dialog`. The data and the
kernel are genuinely 3-D (see :mod:`minflux_viewer.analysis.conv_segmentation_3d`);
only the *viewing* is reduced to orthogonal slices. The centre of the window is a
**2×2 orthogonal viewer**:

    ┌────────────┬────────────┐
    │  XY        │  YZ        │   linked crosshairs — moving the cursor in any
    ├────────────┼────────────┤   pane re-slices the others through the same
    │  XZ        │  3-D view  │   volume point (x0, y0, z0)
    └────────────┴────────────┘

The bottom-right pane is an (optional, experimental) real 3-D OpenGL render of the
localizations + detections + the three current slice planes, so the orthoslice
position is visible in volumetric context.

The controls / detections panels deliberately mirror the 2-D dialog so the two can
later merge behind one UI. Two-phase update: geometry / voxel-size changes rebuild
the response volume (debounced); threshold / separation / cap changes only re-run
the cheap 3-D NMS on the cached volume.

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
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..analysis import conv_segmentation_3d as cs3
from .layer_brightness import LayerBrightness, percentile_levels

_RESPONSE_CMAP = "inferno"
_SEL_COLOR = (255, 235, 0)        # yellow ring on the selected detections
_DET_COLOR = (0, 229, 255)        # cyan detection markers
_ROI_STROKE = "#00e5ff"
_CROSS_PEN = pg.mkPen((120, 220, 120), width=1, style=Qt.PenStyle.DashLine)
_PLANE_COLORS = {                 # crosshair-plane wireframe colour in the 3-D pane
    "xy": (0.4, 1.0, 0.4, 0.9),
    "yz": (1.0, 0.5, 0.4, 0.9),
    "xz": (0.5, 0.7, 1.0, 0.9),
}
_GL_MAX_POINTS = 60_000           # subsample cap for the experimental 3-D pane


class ConvSegmentation3DWindow(QDialog):
    """Modeless interactive 3-D geometry-kernel convolution segmentation tool."""

    def __init__(self, state, dataset_idx: int, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._idx = dataset_idx
        self._owner = owner
        self.setWindowTitle("Convolution Segmentation (3D)")
        self.resize(1320, 820)

        self._xyz = self._load_xyz()
        self._param_spins: dict[str, QDoubleSpinBox] = {}
        self._response = np.zeros((0, 0, 0))
        self._hist_vol = np.zeros((0, 0, 0))    # raw counts volume (Data layer)
        self._data_vmax = 1.0
        self._image_lut = self._alpha_white_lut()
        self._bc = LayerBrightness()
        self._xedge = np.zeros(0)
        self._yedge = np.zeros(0)
        self._zedge = np.zeros(0)
        self._centers = np.empty((0, 3))
        self._values = np.empty(0)
        self._labels: list[str] = []
        self._suspend = False
        self._suspend_cross = False
        self._fitted = False
        self._voxel_note = ""          # persistent note (auto-set voxel size, etc.)
        self._model_params: dict[str, dict] = {}
        self._active_model_key: str | None = None
        # crosshair position in nm (volume slice point)
        self._cursor = np.array([0.0, 0.0, 0.0])

        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(200)
        self._recompute_timer.timeout.connect(self._recompute)

        self._build_ui()
        self._register_bc_layers()
        self._restore_prefs()
        self._apply_feasible_default()
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
        n_loc = int(self._xyz.shape[0])
        self._info_label = QLabel(f"{n_loc:,} localizations")
        self._info_label.setStyleSheet("color: #888;")
        self._info_label.setWordWrap(True)
        left.addWidget(self._info_label)

        left.addWidget(QLabel("Geometry model:"))
        self._model_combo = QComboBox()
        for key, model in cs3.GEOMETRY_MODELS_3D.items():
            self._model_combo.addItem(model.label, key)
            self._model_combo.setItemData(self._model_combo.count() - 1,
                                          model.description, Qt.ItemDataRole.ToolTipRole)
        self._model_combo.currentIndexChanged.connect(lambda *_: self._rebuild_params())
        left.addWidget(self._model_combo)

        # kernel preview (central XY slice of the 3-D kernel)
        self._kernel_plot = pg.PlotWidget()
        self._kernel_plot.setFixedHeight(150)
        self._kernel_plot.setAspectLocked(True)
        self._kernel_plot.hideAxis("bottom")
        self._kernel_plot.hideAxis("left")
        self._kernel_plot.setMenuEnabled(False)
        self._kernel_plot.setMouseEnabled(False, False)
        self._kernel_img = pg.ImageItem()
        self._kernel_plot.addItem(self._kernel_img)
        left.addWidget(QLabel("Kernel (central XY slice):"))
        left.addWidget(self._kernel_plot)

        self._param_form = QFormLayout()
        left.addLayout(self._param_form)

        common = QFormLayout()
        self._pixel_spin = self._spin(0.5, 500.0, cs3.DEFAULT_PIXEL_NM, 1, 0.5, " nm")
        self._pixel_spin.setToolTip("XY voxel size of the histogram volume.")
        self._pixel_spin.valueChanged.connect(self._on_voxel_changed)
        common.addRow("Pixel size (XY):", self._pixel_spin)
        self._zpixel_spin = self._spin(0.5, 1000.0, cs3.DEFAULT_Z_PIXEL_NM, 1, 1.0, " nm")
        self._zpixel_spin.setToolTip("Z voxel depth of the histogram volume "
                                     "(usually coarser than the XY pixel).")
        self._zpixel_spin.valueChanged.connect(self._on_voxel_changed)
        common.addRow("Pixel size (Z):", self._zpixel_spin)
        self._zero_mean = QCheckBox("Zero-mean kernel")
        self._zero_mean.setChecked(cs3.DEFAULT_ZERO_MEAN)
        self._zero_mean.setToolTip(
            "Remove the kernel DC component so the convolution responds to the local "
            "pattern/contrast rather than raw localization density.")
        self._zero_mean.toggled.connect(self._schedule_recompute)
        common.addRow("", self._zero_mean)
        self._sqrt = QCheckBox("√ counts (stabilize)")
        self._sqrt.setChecked(cs3.DEFAULT_SQRT)
        self._sqrt.setToolTip(
            "Convolve the square root of the histogram, down-weighting bright "
            "aggregates that would otherwise dominate the response.")
        self._sqrt.toggled.connect(self._schedule_recompute)
        common.addRow("", self._sqrt)
        self._dog = QCheckBox("DoG band-pass")
        self._dog.setChecked(cs3.DEFAULT_DOG)
        self._dog.setToolTip(
            "Difference-of-Gaussians band-pass on the response at the structure "
            "scale, suppressing slow background.")
        self._dog.toggled.connect(self._schedule_recompute)
        common.addRow("", self._dog)
        left.addLayout(common)

        left.addWidget(self._hline())

        detect = QFormLayout()
        self._response_spin = self._spin(0.0, 1.0, cs3.DEFAULT_MIN_RESPONSE, 2, 0.02, "")
        self._response_spin.setToolTip("Acceptance threshold as a fraction of the peak "
                                       "response (0 keeps every local maximum).")
        self._response_spin.valueChanged.connect(lambda *_: self._rethreshold())
        detect.addRow("Min response:", self._response_spin)
        self._sep_spin = self._spin(1.0, 5000.0, 100.0, 1, 5.0, " nm")
        self._sep_spin.setToolTip("Minimum centre-to-centre separation between detections "
                                  "(physical 3-D distance).")
        self._sep_spin.valueChanged.connect(lambda *_: self._rethreshold())
        detect.addRow("Min separation:", self._sep_spin)
        self._max_spin = QSpinBox()
        self._max_spin.setRange(0, 100000)
        self._max_spin.setValue(0)
        self._max_spin.setToolTip("Cap on the number of detections (0 = unlimited).")
        self._max_spin.valueChanged.connect(lambda *_: self._rethreshold())
        detect.addRow("Max detections:", self._max_spin)
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
        self._show_detection.setToolTip("Show detections in the ortho slices and the 3-D view.")
        self._show_detection.toggled.connect(self._on_layer_toggle)
        self._show_response = QCheckBox("Response")
        self._show_response.setChecked(True)
        self._show_response.toggled.connect(self._refresh_slices)
        self._show_data = QCheckBox("Data")
        self._show_data.setChecked(False)
        self._show_data.setToolTip("Show the raw localizations: count slices in the "
                                   "ortho views and the point cloud in the 3-D view.")
        self._show_data.toggled.connect(self._on_layer_toggle)
        self._show_3d = QCheckBox("3-D view")
        self._show_3d.setChecked(True)
        self._show_3d.setToolTip("Experimental real 3-D render of localizations, "
                                 "detections and the three current slice planes.")
        self._show_3d.toggled.connect(self._on_toggle_3d)
        layers.addWidget(self._show_detection)
        layers.addWidget(self._show_response)
        layers.addWidget(self._show_data)
        layers.addWidget(self._show_3d)
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

        grid = QGridLayout()
        grid.setSpacing(4)
        # three ortho-slice panes: (plane key, h-axis label, v-axis label)
        self._panes: dict[str, dict] = {}
        self._make_pane(grid, "xy", "X (nm)", "Y (nm)", 0, 0)
        self._make_pane(grid, "yz", "Z (nm)", "Y (nm)", 0, 1)
        self._make_pane(grid, "xz", "X (nm)", "Z (nm)", 1, 0)
        # share axes so the orthoviewer panes line up under pan/zoom
        self._panes["yz"]["plot"].setYLink(self._panes["xy"]["plot"])   # Y
        self._panes["xz"]["plot"].setXLink(self._panes["xy"]["plot"])   # X
        self._build_3d_pane(grid, 1, 1)
        center.addLayout(grid, 1)

        panel = QWidget()
        panel.setLayout(center)
        return panel

    def _make_pane(self, grid: QGridLayout, key: str, hlabel: str, vlabel: str,
                   row: int, col: int) -> None:
        plot = pg.PlotWidget()
        plot.setAspectLocked(True)
        plot.setLabel("bottom", hlabel)
        plot.setLabel("left", vlabel)
        plot.setTitle(key.upper())

        resp = pg.ImageItem()
        resp.setZValue(0)
        try:
            resp.setLookupTable(pg.colormap.get(_RESPONSE_CMAP).getLookupTable(0.0, 1.0, 256))
        except Exception:
            pass
        plot.addItem(resp)

        data = pg.ImageItem()
        data.setZValue(5)
        data.setLookupTable(self._image_lut)
        data.setVisible(False)
        plot.addItem(data)

        det = pg.ScatterPlotItem(size=14, symbol="o", pxMode=True,
                                 pen=pg.mkPen(_DET_COLOR, width=1.6),
                                 brush=pg.mkBrush(*_DET_COLOR, 60))
        det.setZValue(10)
        plot.addItem(det)
        sel = pg.ScatterPlotItem(size=20, symbol="o", pxMode=True,
                                 pen=pg.mkPen(_SEL_COLOR, width=2.2), brush=None)
        sel.setZValue(20)
        plot.addItem(sel)

        vline = pg.InfiniteLine(angle=90, movable=True, pen=_CROSS_PEN)
        hline = pg.InfiniteLine(angle=0, movable=True, pen=_CROSS_PEN)
        vline.setZValue(30)
        hline.setZValue(30)
        plot.addItem(vline)
        plot.addItem(hline)
        vline.sigPositionChanged.connect(lambda *_a, k=key: self._on_cross_drag(k))
        hline.sigPositionChanged.connect(lambda *_a, k=key: self._on_cross_drag(k))
        plot.scene().sigMouseClicked.connect(lambda ev, k=key: self._on_pane_click(ev, k))

        grid.addWidget(plot, row, col)
        self._panes[key] = {"plot": plot, "resp": resp, "data": data, "det": det,
                            "sel": sel, "vline": vline, "hline": hline}

    def _build_3d_pane(self, grid: QGridLayout, row: int, col: int) -> None:
        self._gl_view = None
        self._gl_items: dict = {}
        self._gl_center = np.zeros(3)
        try:
            import pyqtgraph.opengl as gl
            self._gl_view = gl.GLViewWidget()
            self._gl_view.setBackgroundColor(pg.mkColor(10, 10, 14))
            grid.addWidget(self._gl_view, row, col)
        except Exception as exc:  # OpenGL unavailable → graceful placeholder
            ph = QLabel(f"3-D view unavailable\n({exc})")
            ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ph.setStyleSheet("color:#888; border:1px solid #333;")
            ph.setWordWrap(True)
            grid.addWidget(ph, row, col)
            self._show_3d.setChecked(False)
            self._show_3d.setEnabled(False)
            self._gl_placeholder = ph

    def _build_detections(self) -> QWidget:
        right = QVBoxLayout()
        self._det_header = QLabel("Detections (0)")
        self._det_header.setStyleSheet("font-weight:bold;")
        right.addWidget(self._det_header)
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._on_list_selection)
        right.addWidget(self._list, 1)

        hint = QLabel("Select a row to centre the orthoviewer on it. "
                      "Selection adds a subset (none selected = all).")
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        right.addWidget(hint)

        self._add_btn = QPushButton("Add to Manager")
        self._add_btn.setToolTip("Add each detection as a 3-D point ROI to the ROI Manager.")
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

    @staticmethod
    def _alpha_white_lut() -> np.ndarray:
        """White RGBA LUT whose alpha ramps with intensity (gamma 0.5), so the Data
        slices overlay the inferno response with the empty background transparent."""
        lut = np.zeros((256, 4), dtype=np.ubyte)
        lut[:, :3] = 255
        lut[:, 3] = (255.0 * (np.linspace(0.0, 1.0, 256) ** 0.5)).astype(np.ubyte)
        return lut

    @staticmethod
    def _sample(arr: np.ndarray, n: int = 1_000_000) -> np.ndarray:
        """A bounded random sample of an array's values, for a snappy B/C histogram."""
        a = np.asarray(arr).ravel()
        if a.size > n:
            a = a[np.random.default_rng(0).integers(0, a.size, n)]
        return a

    # ----------------------------------------------------- brightness/contrast
    def _register_bc_layers(self) -> None:
        """Register the Response and Data slice layers with the shared B/C dialog;
        levels apply to all three ortho panes at once."""
        self._bc.add_layer(
            "Response",
            get_pixels=lambda: (self._sample(self._response) if self._response.size else None),
            apply=lambda lo, hi: [self._panes[k]["resp"].setLevels((lo, hi))
                                  for k in self._panes],
            auto=lambda: percentile_levels(self._sample(self._response), 0.0, 99.5),
            default=lambda: (0.0, 1.0))
        self._bc.add_layer(
            "Data",
            get_pixels=lambda: (self._sample(self._hist_vol[self._hist_vol > 0])
                                if self._hist_vol.size else None),
            apply=lambda lo, hi: [self._panes[k]["data"].setLevels((lo, hi))
                                  for k in self._panes],
            auto=lambda: percentile_levels(self._sample(self._hist_vol), 0.0, 99.5,
                                           positive_only=True),
            default=lambda: (0.0, self._data_vmax))

    def _open_bc(self) -> None:
        self._bc.set_target(self._bc_target.currentText())
        self._bc.open(self)

    # ----------------------------------------------------------- data path
    def _dataset(self):
        if 0 <= self._idx < len(self._state.datasets):
            return self._state.datasets[self._idx]
        return None

    def _load_xyz(self) -> np.ndarray:
        ds = self._dataset()
        if ds is None:
            return np.empty((0, 3))
        # Detect in display coordinates (loc_nm + overlay transform) so 3-D point
        # ROIs land correctly on an overlay channel.
        from ..core.roi_crop import display_xyz_filtered
        return display_xyz_filtered(ds)

    def _model_key(self) -> str:
        return self._model_combo.currentData()

    def _params(self) -> dict:
        return {name: spin.value() for name, spin in self._param_spins.items()}

    # ----------------------------------------------------- parameter widgets
    def _rebuild_params(self, *, initial: bool = False) -> None:
        if self._param_spins and self._active_model_key:
            self._model_params[self._active_model_key] = {
                n: s.value() for n, s in self._param_spins.items()}
        while self._param_form.rowCount():
            self._param_form.removeRow(0)
        self._param_spins.clear()
        model = cs3.get_model(self._model_key())
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
        self._suspend = True
        self._sep_spin.setValue(model.size(model.defaults()))
        self._suspend = False
        if not initial:
            self._recompute()

    # ----------------------------------------------------- persistence
    def _restore_prefs(self) -> None:
        cfg = dict(self._state.prefs.get("conv_segmentation_3d", {}) or {})
        self._model_params = dict(cfg.get("model_params", {}) or {})
        mk = cfg.get("model")
        self._model_combo.blockSignals(True)
        if mk in cs3.GEOMETRY_MODELS_3D:
            i = self._model_combo.findData(mk)
            if i >= 0:
                self._model_combo.setCurrentIndex(i)
        self._model_combo.blockSignals(False)
        self._rebuild_params(initial=True)
        if not cfg:
            return
        self._suspend = True
        try:
            for spin, key in ((self._pixel_spin, "pixel_size"),
                              (self._zpixel_spin, "z_pixel_size"),
                              (self._response_spin, "min_response"),
                              (self._sep_spin, "min_separation")):
                if key in cfg:
                    spin.setValue(float(cfg[key]))
            if "max_detections" in cfg:
                self._max_spin.setValue(int(cfg["max_detections"]))
            for chk, key in ((self._zero_mean, "zero_mean"), (self._sqrt, "sqrt"),
                             (self._dog, "dog")):
                if key in cfg:
                    chk.setChecked(bool(cfg[key]))
        finally:
            self._suspend = False

    def _save_prefs(self) -> None:
        if self._active_model_key:
            self._model_params[self._active_model_key] = self._params()
        self._state.prefs["conv_segmentation_3d"] = {
            "model": self._model_key(),
            "pixel_size": self._pixel_spin.value(),
            "z_pixel_size": self._zpixel_spin.value(),
            "zero_mean": self._zero_mean.isChecked(),
            "sqrt": self._sqrt.isChecked(),
            "dog": self._dog.isChecked(),
            "min_response": self._response_spin.value(),
            "min_separation": self._sep_spin.value(),
            "max_detections": self._max_spin.value(),
            "model_params": self._model_params,
        }
        try:
            self._state.save_prefs()
        except Exception:
            pass

    def _apply_feasible_default(self) -> None:
        """MINFLUX fields of view are often several µm wide, so the saved/default
        voxel size can blow past the cap and leave the viewer blank. If the current
        voxel size is over the cap for this dataset, coarsen it to the smallest size
        that fits so the tool shows something on open."""
        if self._xyz.shape[0] < 3:
            return
        px = float(self._pixel_spin.value())
        pz = float(self._zpixel_spin.value())
        if cs3.estimate_voxel_count(self._xyz, px, pz) <= cs3.DEFAULT_MAX_VOXELS:
            return
        ratio = pz / px if px > 0 else 2.0
        npx, npz = cs3.suggest_voxel_sizes(
            self._xyz, max_voxels=cs3.DEFAULT_MAX_VOXELS, z_ratio=ratio)
        self._suspend = True
        try:
            self._pixel_spin.setValue(npx)
            self._zpixel_spin.setValue(npz)
        finally:
            self._suspend = False
        self._voxel_note = (
            f"Voxel size set to {npx:g}/{npz:g} nm to fit the "
            f"{cs3.DEFAULT_MAX_VOXELS/1e6:.0f} M voxel cap (large field of view).")
        self._status_label.setText(self._voxel_note)

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

    def _on_voxel_changed(self) -> None:
        """A user voxel-size edit clears the auto-set note, then recomputes."""
        if not self._suspend:
            self._voxel_note = ""
        self._schedule_recompute()

    def _recompute(self) -> None:
        """Full update: build kernel, voxelise, 3-D convolve."""
        px = float(self._pixel_spin.value())
        pz = float(self._zpixel_spin.value())
        kernel = cs3.build_kernel_3d(self._model_key(), self._params(), px, pz,
                                     zero_mean=self._zero_mean.isChecked())
        self._draw_kernel(kernel)

        xyz = self._xyz
        if xyz.shape[0] < 3:
            self._reset_maps("Need at least 3 localizations with finite XYZ.")
            return
        n_vox = cs3.estimate_voxel_count(xyz, px, pz)
        if n_vox > cs3.DEFAULT_MAX_VOXELS:
            ratio = pz / px if px > 0 else 2.0
            spx, spz = cs3.suggest_voxel_sizes(
                xyz, max_voxels=cs3.DEFAULT_MAX_VOXELS, z_ratio=ratio)
            self._reset_maps("")
            self._info_label.setText(
                f"Volume ≈ {n_vox/1e6:.0f} M voxels at {px:g}/{pz:g} nm — over the "
                f"{cs3.DEFAULT_MAX_VOXELS/1e6:.0f} M cap. Increase the voxel size to "
                f"≥ {spx:g}/{spz:g} nm.")
            return

        H, self._xedge, self._yedge, self._zedge = cs3.render_histogram_3d(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], px, pz)
        self._hist_vol = H                                  # raw counts (Data layer)
        pos = H[H > 0]
        self._data_vmax = max(float(np.percentile(pos, 99.0)) if pos.size else 1.0, 1.0)
        H = cs3.stabilize_histogram(H, self._sqrt.isChecked())
        if self._dog.isChecked():
            rad = 0.5 * cs3.get_model(self._model_key()).size(self._params())
            dog_radius_px = (rad / px, rad / px, rad / pz)
        else:
            dog_radius_px = None
        self._response = cs3.convolve_response_3d(H, kernel, dog_radius_px=dog_radius_px)
        nx, ny, nz = self._response.shape
        self._info_label.setText(
            f"{xyz.shape[0]:,} localizations · {nx}×{ny}×{nz} voxels "
            f"({n_vox/1e6:.1f} M)")
        if not self._fitted:
            self._cursor = np.array([
                0.5 * (self._xedge[0] + self._xedge[-1]),
                0.5 * (self._yedge[0] + self._yedge[-1]),
                0.5 * (self._zedge[0] + self._zedge[-1])])
        self._update_gl_static()
        self._rethreshold()
        self._bc.refresh_if_open()
        if not self._fitted:
            for p in self._panes.values():
                p["plot"].getViewBox().autoRange(padding=0.02)
            self._fitted = True

    def _reset_maps(self, status: str) -> None:
        self._response = np.zeros((0, 0, 0))
        self._hist_vol = np.zeros((0, 0, 0))
        for p in self._panes.values():
            p["resp"].clear()
            p["data"].clear()
            p["det"].clear()
            p["sel"].clear()
        self._centers = np.empty((0, 3))
        self._values = np.empty(0)
        self._labels = []
        self._populate_list()
        self._status_label.setText(status)
        self._add_btn.setEnabled(False)

    def _rethreshold(self) -> None:
        """Cheap update: re-run 3-D NMS on the cached response volume."""
        if self._suspend or self._response.size == 0:
            return
        centers, vals = cs3.detections_from_response_3d(
            self._response, self._xedge, self._yedge, self._zedge,
            min_response=float(self._response_spin.value()),
            min_distance_nm=float(self._sep_spin.value()),
            max_detections=(self._max_spin.value() or None))
        self._centers = centers
        self._values = vals
        prefix = self._name_prefix()
        self._labels = [f"{prefix} {i}" for i in range(1, centers.shape[0] + 1)]
        self._populate_list()
        self._refresh_slices()
        self._update_gl_detections()
        self._status_label.setText(self._voxel_note)
        self._add_btn.setEnabled(centers.shape[0] > 0)

    def _name_prefix(self) -> str:
        return cs3.get_model(self._model_key()).label.split(" ")[0]

    # ----------------------------------------------------------- slicing
    def _voxel_nm(self) -> tuple[float, float, float]:
        vx = float(self._xedge[1] - self._xedge[0]) if self._xedge.size > 1 else 1.0
        vy = float(self._yedge[1] - self._yedge[0]) if self._yedge.size > 1 else 1.0
        vz = float(self._zedge[1] - self._zedge[0]) if self._zedge.size > 1 else 1.0
        return vx, vy, vz

    @staticmethod
    def _index_for(value: float, edges: np.ndarray) -> int:
        if edges.size < 2:
            return 0
        return int(np.clip(np.searchsorted(edges, value) - 1, 0, edges.size - 2))

    def _refresh_slices(self) -> None:
        """Re-slice the cached response/data volumes at the cursor and redraw."""
        if self._response.size == 0:
            for p in self._panes.values():
                p["resp"].clear()
                p["data"].clear()
                p["det"].clear()
            self._position_crosshairs()
            return
        show_resp = self._show_response.isChecked()
        show_data = self._show_data.isChecked() and self._hist_vol.size > 0
        resp_lv = self._bc.effective_levels("Response") or (0.0, 1.0)
        data_lv = self._bc.effective_levels("Data") or (0.0, self._data_vmax)
        x0, y0, z0 = self._cursor
        ix = self._index_for(x0, self._xedge)
        iy = self._index_for(y0, self._yedge)
        iz = self._index_for(z0, self._zedge)
        vol = self._response
        dvol = self._hist_vol

        # slice arrays indexed [H, V]; _oriented transposes for row-major ImageItem
        for key, rslice, dslice, hedge, vedge in (
                ("xy", vol[:, :, iz], dvol[:, :, iz] if show_data else None,
                 self._xedge, self._yedge),
                ("yz", vol[ix, :, :].T, dvol[ix, :, :].T if show_data else None,
                 self._zedge, self._yedge),
                ("xz", vol[:, iy, :], dvol[:, iy, :] if show_data else None,
                 self._xedge, self._zedge)):
            self._set_slice(key, "resp", rslice, hedge, vedge, show_resp, resp_lv)
            self._set_slice(key, "data", dslice, hedge, vedge, show_data, data_lv)

        self._draw_detection_slices()
        self._position_crosshairs()

    def _set_slice(self, key: str, which: str, slice_hv, hedge: np.ndarray,
                   vedge: np.ndarray, show: bool, levels) -> None:
        img = self._panes[key][which]
        if not show or slice_hv is None:
            img.setVisible(False)
            return
        img.setVisible(True)
        img.setImage(self._oriented(np.ascontiguousarray(slice_hv)), levels=levels)
        img.setRect(QRectF(float(hedge[0]), float(vedge[0]),
                           float(hedge[-1] - hedge[0]), float(vedge[-1] - vedge[0])))

    def _draw_detection_slices(self) -> None:
        """Show detections near each current slice plane (within the model's extent)."""
        show = self._show_detection.isChecked() and self._centers.shape[0] > 0
        tol = max(0.5 * cs3.get_model(self._model_key()).size(self._params()),
                  *self._voxel_nm())
        x0, y0, z0 = self._cursor
        c = self._centers
        # plane key -> (off-axis values, cursor on that axis, (h, v) columns)
        plane_cfg = {
            "xy": (c[:, 2], z0, (0, 1)),
            "yz": (c[:, 0], x0, (2, 1)),
            "xz": (c[:, 1], y0, (0, 2)),
        }
        for key, (off, cur, (hc, vc)) in plane_cfg.items():
            pane = self._panes[key]
            if not show:
                pane["det"].clear()
                continue
            near = np.abs(off - cur) <= tol
            if near.any():
                pane["det"].setData(x=c[near, hc], y=c[near, vc])
            else:
                pane["det"].clear()
        self._update_selection_overlay()

    def _position_crosshairs(self) -> None:
        self._suspend_cross = True
        try:
            x0, y0, z0 = self._cursor
            self._panes["xy"]["vline"].setPos(x0)
            self._panes["xy"]["hline"].setPos(y0)
            self._panes["yz"]["vline"].setPos(z0)
            self._panes["yz"]["hline"].setPos(y0)
            self._panes["xz"]["vline"].setPos(x0)
            self._panes["xz"]["hline"].setPos(z0)
        finally:
            self._suspend_cross = False
        self._update_gl_planes()

    def _on_cross_drag(self, key: str) -> None:
        if self._suspend_cross:
            return
        pane = self._panes[key]
        h = float(pane["vline"].value())
        v = float(pane["hline"].value())
        x0, y0, z0 = self._cursor
        if key == "xy":
            x0, y0 = h, v
        elif key == "yz":
            z0, y0 = h, v
        elif key == "xz":
            x0, z0 = h, v
        self._cursor = np.array([x0, y0, z0])
        self._refresh_slices()

    def _on_pane_click(self, ev, key: str) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        vb = self._panes[key]["plot"].getViewBox()
        try:
            pt = vb.mapSceneToView(ev.scenePos())
        except Exception:
            return
        h, v = float(pt.x()), float(pt.y())
        x0, y0, z0 = self._cursor
        if key == "xy":
            x0, y0 = h, v
        elif key == "yz":
            z0, y0 = h, v
        elif key == "xz":
            x0, z0 = h, v
        self._cursor = np.array([x0, y0, z0])
        self._refresh_slices()

    # ----------------------------------------------------------- detections list
    def _populate_list(self) -> None:
        self._suspend_list = True
        self._list.clear()
        for i, (cx, cy, cz) in enumerate(self._centers):
            r = float(self._values[i]) if i < self._values.size else float("nan")
            text = f"{self._labels[i]}   ({cx:.0f}, {cy:.0f}, {cz:.0f})   r={r:.2f}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._list.addItem(item)
        self._suspend_list = False
        self._det_header.setText(f"Detections ({self._centers.shape[0]})")

    def _on_list_selection(self) -> None:
        if getattr(self, "_suspend_list", False):
            return
        idx = [it.data(Qt.ItemDataRole.UserRole) for it in self._list.selectedItems()]
        if idx and self._centers.shape[0]:
            # centre the orthoviewer on the most recently selected detection
            self._cursor = self._centers[int(idx[-1])].copy()
            self._refresh_slices()
        else:
            self._update_selection_overlay()

    def _update_selection_overlay(self) -> None:
        idx = [it.data(Qt.ItemDataRole.UserRole) for it in self._list.selectedItems()]
        if not idx or self._centers.shape[0] == 0 or not self._show_detection.isChecked():
            for p in self._panes.values():
                p["sel"].clear()
            return
        sel = self._centers[np.asarray(idx, dtype=int)]
        self._panes["xy"]["sel"].setData(x=sel[:, 0], y=sel[:, 1])
        self._panes["yz"]["sel"].setData(x=sel[:, 2], y=sel[:, 1])
        self._panes["xz"]["sel"].setData(x=sel[:, 0], y=sel[:, 2])

    def _selected_indices(self) -> np.ndarray:
        idx = [it.data(Qt.ItemDataRole.UserRole) for it in self._list.selectedItems()]
        if idx:
            return np.sort(np.asarray(idx, dtype=int))
        return np.arange(self._centers.shape[0])

    # ----------------------------------------------------------- drawing
    def _draw_kernel(self, kernel: np.ndarray) -> None:
        if kernel.ndim != 3 or kernel.size == 0:
            self._kernel_img.clear()
            return
        mid_z = kernel.shape[2] // 2
        sl = kernel[:, :, mid_z]                       # [x, y]
        m = float(np.max(np.abs(sl))) or 1.0
        self._kernel_img.setImage(self._oriented(sl), levels=(-m, m))
        self._kernel_plot.getViewBox().autoRange(padding=0.02)

    @staticmethod
    def _oriented(img: np.ndarray) -> np.ndarray:
        """Array indexed ``[h_bin, v_bin]`` → orientation expected by ImageItem
        (row-major needs a transpose so rows are the vertical axis)."""
        try:
            row_major = pg.getConfigOption("imageAxisOrder") == "row-major"
        except Exception:
            row_major = True
        return img.T if row_major else img

    # ----------------------------------------------------------- 3-D pane
    def _on_layer_toggle(self) -> None:
        """Detection / Data checkboxes drive both the ortho slices and the 3-D view."""
        self._refresh_slices()
        self._update_gl_visibility()

    def _on_toggle_3d(self) -> None:
        if self._gl_view is None:
            return
        self._gl_view.setVisible(self._show_3d.isChecked())
        if self._show_3d.isChecked():
            self._update_gl_static()
            self._update_gl_detections()
            self._update_gl_planes()
            self._update_gl_visibility()

    def _gl_ok(self) -> bool:
        return self._gl_view is not None and self._show_3d.isChecked()

    def _update_gl_visibility(self) -> None:
        """Honour the Detection / Data checkboxes in the 3-D pane: the raw point
        cloud follows Data, the detection markers follow Detection."""
        if self._gl_view is None:
            return
        raw = self._gl_items.get("raw")
        if raw is not None:
            raw.setVisible(self._show_data.isChecked())
        det = self._gl_items.get("det")
        if det is not None:
            det.setVisible(self._show_detection.isChecked())

    def _update_gl_static(self) -> None:
        """Subsampled raw localizations (gray) — rebuilt when the data/extent changes."""
        if not self._gl_ok():
            return
        import pyqtgraph.opengl as gl
        if self._xedge.size > 1:
            self._gl_center = np.array([
                0.5 * (self._xedge[0] + self._xedge[-1]),
                0.5 * (self._yedge[0] + self._yedge[-1]),
                0.5 * (self._zedge[0] + self._zedge[-1])])
        pts = self._xyz
        if pts.shape[0] > _GL_MAX_POINTS:
            sel = np.random.default_rng(0).choice(pts.shape[0], _GL_MAX_POINTS, replace=False)
            pts = pts[sel]
        local = pts - self._gl_center
        item = self._gl_items.get("raw")
        if item is None:
            item = gl.GLScatterPlotItem(pos=local, size=2.0,
                                        color=(0.6, 0.6, 0.6, 0.35), pxMode=True)
            self._gl_view.addItem(item)
            self._gl_items["raw"] = item
        else:
            item.setData(pos=local)
        item.setVisible(self._show_data.isChecked())
        if self._xedge.size > 1:
            span = max(float(self._xedge[-1] - self._xedge[0]),
                       float(self._yedge[-1] - self._yedge[0]),
                       float(self._zedge[-1] - self._zedge[0]), 1.0)
            self._gl_view.setCameraPosition(distance=span * 1.6)

    def _update_gl_detections(self) -> None:
        if not self._gl_ok():
            return
        import pyqtgraph.opengl as gl
        item = self._gl_items.get("det")
        if self._centers.shape[0] == 0:
            if item is not None:
                item.setData(pos=np.empty((0, 3)))
            return
        local = self._centers - self._gl_center
        color = (_DET_COLOR[0] / 255.0, _DET_COLOR[1] / 255.0, _DET_COLOR[2] / 255.0, 1.0)
        if item is None:
            item = gl.GLScatterPlotItem(pos=local, size=12.0, color=color, pxMode=True)
            self._gl_view.addItem(item)
            self._gl_items["det"] = item
        else:
            item.setData(pos=local, color=color)
        item.setVisible(self._show_detection.isChecked())

    def _update_gl_planes(self) -> None:
        """Draw the three current slice planes as wireframe rectangles in 3-D."""
        if not self._gl_ok() or self._xedge.size < 2:
            return
        import pyqtgraph.opengl as gl
        x0, y0, z0 = self._cursor
        xa, xb = float(self._xedge[0]), float(self._xedge[-1])
        ya, yb = float(self._yedge[0]), float(self._yedge[-1])
        za, zb = float(self._zedge[0]), float(self._zedge[-1])
        rects = {
            "xy": [(xa, ya, z0), (xb, ya, z0), (xb, yb, z0), (xa, yb, z0), (xa, ya, z0)],
            "yz": [(x0, ya, za), (x0, yb, za), (x0, yb, zb), (x0, ya, zb), (x0, ya, za)],
            "xz": [(xa, y0, za), (xb, y0, za), (xb, y0, zb), (xa, y0, zb), (xa, y0, za)],
        }
        for key, corners in rects.items():
            pts = np.asarray(corners, dtype=float) - self._gl_center
            item = self._gl_items.get(f"plane_{key}")
            if item is None:
                item = gl.GLLinePlotItem(pos=pts, color=_PLANE_COLORS[key],
                                         width=1.5, mode="line_strip", antialias=True)
                self._gl_view.addItem(item)
                self._gl_items[f"plane_{key}"] = item
            else:
                item.setData(pos=pts, color=_PLANE_COLORS[key])

    # ----------------------------------------------------------- commit
    def _add_to_manager(self) -> None:
        sel = self._selected_indices()
        if sel.size == 0 or self._owner is None:
            return
        if not hasattr(self._owner, "add_point_rois_3d"):
            self._status_label.setText("ROI output not available.")
            return
        ds = self._dataset()
        model = cs3.get_model(self._model_key())
        centers = self._centers[sel]
        names = [self._labels[i] for i in sel]
        opts = []
        if self._zero_mean.isChecked():
            opts.append("zero-mean kernel")
        if self._sqrt.isChecked():
            opts.append("sqrt-stabilized")
        if self._dog.isChecked():
            opts.append("DoG band-pass")
        opts_str = ", ".join(opts) if opts else "raw kernel"
        log = (
            f"Convolution segmentation (3D): added {centers.shape[0]} of "
            f"{self._centers.shape[0]} {model.label} detection(s) on "
            f"'{ds.name if ds else '?'}' "
            f"(voxel={self._pixel_spin.value():.1f}/{self._zpixel_spin.value():.1f} nm, "
            f"min response={self._response_spin.value():.2f}, "
            f"min separation={self._sep_spin.value():.0f} nm; "
            f"{opts_str}; {self._param_summary(model, self._params())}); "
            f"added 3-D point ROIs.")
        n = self._owner.add_point_rois_3d(
            self._idx, centers, name_prefix=self._name_prefix(),
            source="conv_segmentation_3d",   # stroke_color defaults to the system ROI colour
            names=names, log_message=log)
        if n:
            self._status_label.setText(f"Added {n} point ROI(s) to the ROI Manager.")

    @staticmethod
    def _param_summary(model, params: dict) -> str:
        return ", ".join(f"{ps.label.lower()}={params[ps.name]:g}{ps.suffix.strip()}"
                         for ps in model.params)
