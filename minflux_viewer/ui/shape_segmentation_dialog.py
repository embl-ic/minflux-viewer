"""
minflux_viewer.ui.shape_segmentation_dialog
===========================================
Interactive **shape-model segmentation** tool
(Analyze › Segmentation › Shape Model…).

Fits objects of a *known geometry* — capsule (obround, e.g. an E. coli cell),
curved capsule, ellipse, rectangle, disk — to the rendered XY density, and files
each fitted object into the ROI Manager as an editable **polygon contour**.

The engine is :mod:`minflux_viewer.analysis.shape_segmentation`; this window is
only its front end. Every control is built from the shape registry
(``ShapeModel.size_specs``), so registering a new geometry there makes it appear
here with its own min/max controls — no change in this file.

Why a polygon and not the parametric shape: a polygon is both a *region* (masks,
crop, in-ROI highlighting all work) and a *vertex-editable* ROI, so a fitted
contour can be corrected by hand and then behaves exactly like a hand-drawn ROI.

Detection is not interactive-fast (a fit takes ~0.5-8 s depending on field size),
so it runs on an explicit **Detect** press rather than live on every edit.

Modeless / non-owned per the project window convention.
"""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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

from ..analysis import shape_segmentation as ss

_PREF_KEY = "shape_segmentation"
_DENSITY_CMAP = "inferno"
_CONTOUR_COLOR = "#00e5ff"       # preview only; stored ROIs use the system colour
_SEL_COLOR = (255, 235, 0)
_CLIPPED_COLOR = "#ff9030"
_LOC_COLOR = (120, 190, 255, 110)   # translucent, so density shows through
_MAX_SCATTER_POINTS = 100_000       # display-only thinning cap


class ShapeSegmentationWindow(QDialog):
    """Modeless interactive known-geometry (shape-model) segmentation tool."""

    def __init__(self, state, dataset_idx: int, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._idx = dataset_idx
        self._owner = owner
        self.setWindowTitle("Shape Model Segmentation")
        self.resize(1220, 720)

        self._xy = self._load_xy()
        self._result: ss.ShapeSegmentationResult | None = None
        self._contours: list[np.ndarray] = []
        self._curves: list[pg.PlotCurveItem] = []
        self._range_spins: dict[str, tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}
        self._model_ranges: dict[str, dict[str, tuple[float, float]]] = {}
        self._active_model_key: str | None = None
        self._suspend = False
        self._fitted_view = False
        self._points_drawn = False

        self._build_ui()
        self._restore_prefs()
        # The localizations are known before any fit, so show them at once —
        # the window is useful (and the prior can be judged) before Detect.
        self._draw_points()
        self._update_layer_visibility()
        self._plot.getPlotItem().getViewBox().autoRange()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.addWidget(self._build_controls())
        root.addWidget(self._build_center(), 1)
        root.addWidget(self._build_objects())

    def _build_controls(self) -> QWidget:
        left = QVBoxLayout()
        ds = self._dataset()
        left.addWidget(QLabel(f"<b>{ds.name if ds else '(no dataset)'}</b>"))
        left.addWidget(QLabel(f"{self._xy.shape[0]:,} localizations"))

        self._model_combo = QComboBox()
        for key, model in ss.SHAPE_MODELS.items():
            self._model_combo.addItem(model.label, key)
            self._model_combo.setItemData(
                self._model_combo.count() - 1, model.description,
                Qt.ItemDataRole.ToolTipRole)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        form = QFormLayout()
        form.addRow("Shape", self._model_combo)
        left.addLayout(form)

        left.addWidget(self._hline())
        left.addWidget(QLabel("<b>Expected size</b>"))
        left.addWidget(QLabel(
            "<i>The prior. It is what separates objects that<br>"
            "touch without a visible waist, so set it to the<br>"
            "real spread rather than opening it up.</i>"))
        self._size_form = QFormLayout()
        left.addLayout(self._size_form)

        left.addWidget(self._hline())
        left.addWidget(QLabel("<b>Detection</b>"))
        tuning = QFormLayout()
        self._pixel_spin = self._spin(1.0, 500.0, 20.0, 1, 1.0, " nm")
        self._pixel_spin.setToolTip(
            "Pixel size of the density raster the fit runs on.")
        tuning.addRow("Pixel size", self._pixel_spin)

        self._smooth_spin = self._spin(0.0, 5000.0, 0.0, 0, 10.0, " nm")
        self._smooth_spin.setToolTip(
            "Foreground smoothing. 0 = automatic (one sixth of the nominal "
            "width), which closes gaps inside an object without merging "
            "neighbours.")
        tuning.addRow("Smoothing", self._smooth_spin)

        self._cost_spin = self._spin(0.0, 3.0, 0.25, 2, 0.05, "")
        self._cost_spin.setToolTip(
            "Price of one extra object, in nominal object areas. Raise it if "
            "touching objects are over-split into fragments; lower it if they "
            "stay merged into one oversized object.")
        tuning.addRow("Instance cost", self._cost_spin)

        self._max_per_comp = QSpinBox()
        self._max_per_comp.setRange(1, 20)
        self._max_per_comp.setValue(6)
        self._max_per_comp.setToolTip(
            "Most objects to consider inside one connected blob.")
        tuning.addRow("Max per blob", self._max_per_comp)

        self._vertices = QSpinBox()
        self._vertices.setRange(8, 256)
        self._vertices.setValue(48)
        self._vertices.setToolTip(
            "Vertices of the polygon contour handed to the ROI Manager. Fewer "
            "vertices are easier to correct by hand; more follow the fitted "
            "shape more closely.")
        tuning.addRow("Contour vertices", self._vertices)
        left.addLayout(tuning)

        self._use_roi = QCheckBox("Use acquisition ROI as the field")
        self._use_roi.setChecked(True)
        bounds = self._acquisition_bounds()
        self._use_roi.setEnabled(bounds is not None)
        self._use_roi.setToolTip(
            "Judge clipping against the scanned acquisition ROI recorded in the "
            "source .msr. Without it the field is spanned by the localizations "
            "themselves, so no object can be found to leave it and nothing is "
            "ever reported as clipped."
            if bounds is not None else
            "No acquisition ROI is available for this dataset (it did not come "
            "from a .msr, or the file records none), so clipping cannot be "
            "determined.")
        left.addWidget(self._use_roi)

        left.addWidget(self._hline())
        self._detect_btn = QPushButton("Detect")
        self._detect_btn.setDefault(True)
        self._detect_btn.clicked.connect(self._detect)
        left.addWidget(self._detect_btn)

        self._status_label = QLabel("Press Detect to fit.")
        self._status_label.setWordWrap(True)
        left.addWidget(self._status_label)
        left.addStretch(1)

        holder = QWidget()
        holder.setLayout(left)
        holder.setFixedWidth(310)
        return holder

    def _build_center(self) -> QWidget:
        center = QVBoxLayout()
        layers = QHBoxLayout()
        layers.addWidget(QLabel("Show:"))
        self._show_density = QCheckBox("Density")
        self._show_density.setChecked(True)
        self._show_density.setToolTip(
            "The rendered density the fit actually runs on.")
        self._show_locs = QCheckBox("Localizations")
        self._show_locs.setChecked(True)
        self._show_locs.setToolTip(
            f"The raw localization point cloud. Thinned to "
            f"{_MAX_SCATTER_POINTS:,} points for display when there are more; "
            f"thinning is display-only and deterministic, and never affects the "
            f"fit.")
        self._show_contours = QCheckBox("Contours")
        self._show_contours.setChecked(True)
        self._show_contours.setToolTip("The fitted object outlines.")
        for box in (self._show_density, self._show_locs, self._show_contours):
            box.toggled.connect(self._update_layer_visibility)
            layers.addWidget(box)
        layers.addStretch(1)
        self._loc_count_label = QLabel("")
        layers.addWidget(self._loc_count_label)
        center.addLayout(layers)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("k")
        self._plot.setAspectLocked(True)
        # Match the render view's XY convention rather than hard-coding one, so
        # this dialog and the render window agree about which way is up.
        self._plot.getPlotItem().invertY(self._y_inverted())
        self._plot.setLabel("bottom", "X", units="nm")
        self._plot.setLabel("left", "Y", units="nm")

        self._image = pg.ImageItem()
        self._image.setLookupTable(self._density_lut())
        self._image.setZValue(0)
        self._plot.addItem(self._image)

        self._scatter = pg.ScatterPlotItem(
            size=2.0, symbol="o", pen=pg.mkPen(None),
            brush=pg.mkBrush(*_LOC_COLOR), pxMode=True)
        self._scatter.setZValue(5)
        self._plot.addItem(self._scatter)

        center.addWidget(self._plot, 1)
        holder = QWidget()
        holder.setLayout(center)
        return holder

    def _y_inverted(self) -> bool:
        """Follow the render view's ``plot.render_xy_origin`` preference."""
        prefs = getattr(self._state, "prefs", None) or {}
        origin = (prefs.get("plot", {}) or {}).get("render_xy_origin", "top_left")
        return str(origin) == "top_left"

    @staticmethod
    def _oriented(img: np.ndarray) -> np.ndarray:
        """Account for the app's global pyqtgraph image axis order.

        The engine's rasters are indexed ``[row=y, col=x]``, which is what a
        row-major ImageItem wants; a col-major one needs the transpose. The
        order is a *runtime global* — opening a render window switches the app
        to row-major — so it has to be read per draw, not assumed.
        """
        try:
            row_major = pg.getConfigOption("imageAxisOrder") == "row-major"
        except Exception:
            row_major = True
        return img if row_major else img.T

    def _build_objects(self) -> QWidget:
        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Objects</b>"))
        self._list = QListWidget()
        self._list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._on_selection)
        right.addWidget(self._list, 1)

        self._add_btn = QPushButton("Add to ROI Manager")
        self._add_btn.setEnabled(False)
        self._add_btn.setToolTip(
            "File the selected objects (or all, if none are selected) as "
            "polygon ROIs. Their vertices can then be dragged, nudged with the "
            "arrow keys, or reshaped from the ROI right-click menu.")
        self._add_btn.clicked.connect(self._add_to_manager)
        right.addWidget(self._add_btn)

        holder = QWidget()
        holder.setLayout(right)
        holder.setFixedWidth(280)
        return holder

    @staticmethod
    def _spin(lo, hi, val, decimals, step, suffix) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setSuffix(suffix)
        box.setValue(val)
        return box

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    @staticmethod
    def _density_lut() -> np.ndarray:
        from ..colormaps import colormap_lut
        return colormap_lut(_DENSITY_CMAP)

    # -------------------------------------------------------- size controls
    def _rebuild_size_controls(self) -> None:
        """Replace the min/max spin pairs when the shape model changes.

        Built from the model's own ``size_specs``, so a newly registered
        geometry gets its controls for free.
        """
        if self._active_model_key is not None:
            self._model_ranges[self._active_model_key] = self._size_ranges()
        while self._size_form.rowCount():
            self._size_form.removeRow(0)
        self._range_spins.clear()

        key = self._model_key()
        model = ss.get_shape_model(key)
        remembered = self._model_ranges.get(key, {})
        for spec in model.size_specs:
            lo_val, hi_val = remembered.get(
                spec.name, (spec.default_lo, spec.default_hi))
            lo = self._spin(spec.limit_lo, spec.limit_hi, lo_val,
                            spec.decimals, spec.step, spec.suffix)
            hi = self._spin(spec.limit_lo, spec.limit_hi, hi_val,
                            spec.decimals, spec.step, spec.suffix)
            tip = spec.tooltip or spec.label
            lo.setToolTip(f"Smallest plausible {spec.label.lower()}. {tip}")
            hi.setToolTip(f"Largest plausible {spec.label.lower()}. {tip}")
            row = QWidget()
            box = QHBoxLayout(row)
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(lo)
            box.addWidget(QLabel("to"))
            box.addWidget(hi)
            self._size_form.addRow(spec.label, row)
            self._range_spins[spec.name] = (lo, hi)
        self._active_model_key = key

    def _size_ranges(self) -> dict[str, tuple[float, float]]:
        return {name: (lo.value(), hi.value())
                for name, (lo, hi) in self._range_spins.items()}

    def _on_model_changed(self) -> None:
        if self._suspend:
            return
        self._rebuild_size_controls()

    # ----------------------------------------------------------- data path
    def _dataset(self):
        if 0 <= self._idx < len(self._state.datasets):
            return self._state.datasets[self._idx]
        return None

    def _load_xy(self) -> np.ndarray:
        ds = self._dataset()
        if ds is None:
            return np.empty((0, 2))
        # Detect in display coordinates (loc_nm + overlay transform) — the frame
        # ROIs live in — so contours land correctly on an overlay channel.
        from ..core.roi_crop import display_xy_filtered
        return display_xy_filtered(ds)

    def _acquisition_bounds(self):
        """``(x0, y0, x1, y1)`` nm of this dataset's own acquisition ROI."""
        ds = self._dataset()
        if ds is None:
            return None
        path = (ds.metadata or {}).get("msr_source_path")
        did = (ds.metadata or {}).get("msr_dataset_did")
        if not path or not did:
            return None
        try:
            from ..msr.acquisition_roi import (
                group_by_dataset, read_acquisition_rois, union_bounds)
            groups = group_by_dataset(read_acquisition_rois(str(path))) or {}
            group = groups.get(str(did))
            if not group:
                return None
            x0, y0, width, height = union_bounds(group)
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None
        return (x0 * 1e9, y0 * 1e9, (x0 + width) * 1e9, (y0 + height) * 1e9)

    def _model_key(self) -> str:
        return self._model_combo.currentData()

    def _prior(self) -> ss.ShapePrior:
        ranges = self._size_ranges()
        model = ss.get_shape_model(self._model_key())
        lo, hi = [], []
        for spec in model.size_specs:
            low, high = ranges.get(spec.name, (spec.default_lo, spec.default_hi))
            lo.append(min(low, high))
            hi.append(max(low, high))
        return ss.ShapePrior(self._model_key(), tuple(lo), tuple(hi))

    # -------------------------------------------------------------- detect
    def _detect(self) -> None:
        if self._xy.shape[0] < 10:
            self._status_label.setText(
                "Not enough localizations in the current view to fit.")
            return
        prior = self._prior()
        try:
            prior.validate()
        except ValueError as exc:
            self._status_label.setText(f"<span style='color:#d33'>{exc}</span>")
            return
        cfg = ss.ShapeSegmentationConfig(
            detection_pixel_nm=float(self._pixel_spin.value()),
            smoothing_nm=(float(self._smooth_spin.value())
                          if self._smooth_spin.value() > 0 else None),
            instance_cost=float(self._cost_spin.value()),
            max_instances_per_component=int(self._max_per_comp.value()))
        bounds = self._acquisition_bounds() if self._use_roi.isChecked() else None

        self._status_label.setText("Fitting…")
        self._detect_btn.setEnabled(False)
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        started = time.time()
        try:
            result = ss.segment_shapes_in_points(
                self._xy[:, 0], self._xy[:, 1], prior=prior, cfg=cfg,
                bounds_nm=bounds)
        except Exception as exc:
            self._status_label.setText(
                f"<span style='color:#d33'>{type(exc).__name__}: {exc}</span>")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()
            self._detect_btn.setEnabled(True)
        elapsed = time.time() - started

        self._result = result
        vertices = int(self._vertices.value())
        self._contours = [ss.instance_outline(item, n_points=vertices)
                          for item in result.instances]
        self._draw()
        self._populate_list()
        self._add_btn.setEnabled(bool(result.instances))
        clipped = result.stats.get("n_clipped", 0)
        note = ("" if bounds is not None else
                " — no acquisition ROI, so clipping is not determined")
        reason = result.stats.get("reason")
        if reason:
            # Finding nothing is a legitimate result, but a bare zero is not
            # actionable; the engine says which stage dropped everything.
            self._status_label.setText(
                f"<b>No object found</b> ({elapsed:.1f} s): {reason}.")
        else:
            self._status_label.setText(
                f"{len(result.instances)} object(s), {clipped} clipped "
                f"({elapsed:.1f} s){note}.")

    # ------------------------------------------------------------- drawing
    def _draw(self) -> None:
        result = self._result
        for curve in self._curves:
            self._plot.removeItem(curve)
        self._curves.clear()
        if result is None:
            return
        density = np.asarray(result.detection_field, dtype=float)
        if density.size:
            hi = float(np.percentile(density[density > 0], 99.5)) \
                if np.any(density > 0) else 1.0
            self._image.setImage(self._oriented(density),
                                 levels=(0.0, max(hi, 1e-9)),
                                 autoLevels=False)
            px = float(result.detection_pixel_nm)
            origin = result.detection_origin_nm
            self._image.setRect(QRectF(
                origin[0], origin[1], density.shape[1] * px, density.shape[0] * px))
        for index, contour in enumerate(self._contours):
            clipped = result.instances[index].clipped
            curve = pg.PlotCurveItem(
                contour[:, 0], contour[:, 1],
                pen=pg.mkPen(_CLIPPED_COLOR if clipped else _CONTOUR_COLOR,
                             width=2))
            self._plot.addItem(curve)
            self._curves.append(curve)
        self._update_layer_visibility()
        if not self._fitted_view and self._contours:
            self._plot.getPlotItem().getViewBox().autoRange()
            self._fitted_view = True

    def _draw_points(self) -> None:
        """Show the localizations, thinned for display only."""
        self._points_drawn = True
        total = int(self._xy.shape[0])
        if total == 0:
            self._scatter.setData([], [])
            self._loc_count_label.setText("")
            return
        if total > _MAX_SCATTER_POINTS:
            # Deterministic stride, so the same points are shown on every
            # redraw instead of the cloud flickering between draws.
            step = int(np.ceil(total / _MAX_SCATTER_POINTS))
            shown = self._xy[::step]
            self._loc_count_label.setText(
                f"{shown.shape[0]:,} of {total:,} localizations shown")
        else:
            shown = self._xy
            self._loc_count_label.setText(f"{total:,} localizations")
        self._scatter.setData(shown[:, 0], shown[:, 1])

    def _update_layer_visibility(self) -> None:
        self._image.setVisible(self._show_density.isChecked())
        want_points = self._show_locs.isChecked()
        if want_points and not self._points_drawn:
            self._draw_points()
        self._scatter.setVisible(want_points)
        visible = self._show_contours.isChecked()
        for curve in self._curves:
            curve.setVisible(visible)

    def _populate_list(self) -> None:
        self._list.clear()
        result = self._result
        if result is None:
            return
        for index, item in enumerate(result.instances, 1):
            sizes = " ".join(f"{key.replace('_nm', '').replace('_deg', '')}"
                             f"={value:.0f}"
                             for key, value in item.size().items())
            flag = "  clipped" if item.clipped else ""
            entry = QListWidgetItem(
                f"{index}. {sizes}  iou={item.iou:.2f}{flag}")
            entry.setData(Qt.ItemDataRole.UserRole, index - 1)
            self._list.addItem(entry)

    def _on_selection(self) -> None:
        """Highlight the selected contours; with no selection every one is live
        (that is also what Add to Manager files)."""
        if self._result is None:
            return
        chosen = set(self._selected_indices().tolist())
        for index, curve in enumerate(self._curves):
            clipped = self._result.instances[index].clipped
            base = _CLIPPED_COLOR if clipped else _CONTOUR_COLOR
            if not chosen:
                curve.setPen(pg.mkPen(base, width=2))
            elif index in chosen:
                curve.setPen(pg.mkPen(_SEL_COLOR, width=3))
            else:
                curve.setPen(pg.mkPen(color=(140, 140, 140), width=1))

    def _selected_indices(self) -> np.ndarray:
        rows = [item.data(Qt.ItemDataRole.UserRole)
                for item in self._list.selectedItems()]
        return np.asarray(sorted(int(r) for r in rows), dtype=int)

    # ----------------------------------------------------------- ROI Manager
    def _add_to_manager(self) -> None:
        result = self._result
        if result is None or not result.instances or self._owner is None:
            return
        if not hasattr(self._owner, "add_polygon_rois"):
            self._state.log("[warn] cannot add ROIs: the owner window is not a "
                            "main window.")
            return
        chosen = self._selected_indices()
        if chosen.size == 0:
            chosen = np.arange(len(result.instances), dtype=int)
        model = ss.get_shape_model(self._model_key())
        prior = self._prior()
        polygons = [self._contours[i] for i in chosen]
        names = [f"{model.label.split(' (')[0]} {i + 1}" for i in chosen]
        ds = self._dataset()
        ranges = ", ".join(
            f"{spec.label.lower()} {prior.size_lo[j]:g}-{prior.size_hi[j]:g}"
            f"{spec.suffix.strip()}"
            for j, spec in enumerate(model.size_specs))
        log = (
            f"Shape-model segmentation: added {chosen.size} of "
            f"{len(result.instances)} {model.label} object(s) on "
            f"'{ds.name if ds else '?'}' "
            f"(pixel={self._pixel_spin.value():.1f} nm, "
            f"instance cost={self._cost_spin.value():.2f}, {ranges}; "
            f"{result.stats.get('n_clipped', 0)} clipped); "
            f"added polygon ROIs with {self._vertices.value()} vertices.")
        added = self._owner.add_polygon_rois(
            self._idx, polygons, name_prefix=model.label.split(" (")[0],
            source="shape_segmentation_2d", names=names, log_message=log)
        if added:
            self._status_label.setText(
                f"Added {added} polygon ROI(s). Drag their vertices, or use the "
                f"arrow keys, to correct them.")

    # ----------------------------------------------------------- preferences
    def _restore_prefs(self) -> None:
        prefs = (self._state.prefs or {}).get(_PREF_KEY, {}) \
            if hasattr(self._state, "prefs") else {}
        self._suspend = True
        try:
            self._model_ranges = {
                key: {name: (float(v[0]), float(v[1]))
                      for name, v in (value or {}).items()}
                for key, value in (prefs.get("ranges") or {}).items()}
            key = prefs.get("model")
            index = self._model_combo.findData(key) if key else -1
            self._model_combo.setCurrentIndex(index if index >= 0 else 0)
            self._pixel_spin.setValue(float(prefs.get("pixel_nm", 20.0)))
            self._smooth_spin.setValue(float(prefs.get("smoothing_nm", 0.0)))
            self._cost_spin.setValue(float(prefs.get("instance_cost", 0.25)))
            self._max_per_comp.setValue(int(prefs.get("max_per_component", 6)))
            self._vertices.setValue(int(prefs.get("vertices", 48)))
            if self._use_roi.isEnabled():
                self._use_roi.setChecked(bool(prefs.get("use_acquisition_roi", True)))
            layers = prefs.get("layers") or {}
            self._show_density.setChecked(bool(layers.get("density", True)))
            self._show_locs.setChecked(bool(layers.get("localizations", True)))
            self._show_contours.setChecked(bool(layers.get("contours", True)))
        except Exception:
            pass
        finally:
            self._suspend = False
        self._rebuild_size_controls()

    def _save_prefs(self) -> None:
        if not hasattr(self._state, "prefs") or self._state.prefs is None:
            return
        if self._active_model_key is not None:
            self._model_ranges[self._active_model_key] = self._size_ranges()
        self._state.prefs[_PREF_KEY] = {
            "model": self._model_key(),
            "ranges": {key: {name: list(value) for name, value in ranges.items()}
                       for key, ranges in self._model_ranges.items()},
            "pixel_nm": float(self._pixel_spin.value()),
            "smoothing_nm": float(self._smooth_spin.value()),
            "instance_cost": float(self._cost_spin.value()),
            "max_per_component": int(self._max_per_comp.value()),
            "vertices": int(self._vertices.value()),
            "use_acquisition_roi": bool(self._use_roi.isChecked()),
            "layers": {
                "density": bool(self._show_density.isChecked()),
                "localizations": bool(self._show_locs.isChecked()),
                "contours": bool(self._show_contours.isChecked()),
            },
        }
        save = getattr(self._state, "save_prefs", None)
        if callable(save):
            save()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._save_prefs()
        super().closeEvent(event)
