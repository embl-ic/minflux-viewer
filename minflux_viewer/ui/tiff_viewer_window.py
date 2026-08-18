"""
Standalone TIFF image viewer.

TIFF files are intentionally not MINFLUX datasets.  This window reuses the
render-view style while reading selected TIFF planes lazily from disk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import resource_path
from ..colormaps import (
    make_colormap,
    named_colormap_names,
)
from ..colors import solid_color_names
from .metadata_viewer import MetadataDocumentView
from .render_window import DepthRangeSlider

if TYPE_CHECKING:
    from ..core.obf_image_source import ObfImageSource
    from ..core.tiff_source import TiffImageSource

    # Any source with metadata / axis_size / read_plane / close (+ series helpers).
    ImageSource = TiffImageSource | ObfImageSource

_IMAGEJ_AUTO_THRESHOLD = 5000
_IMAGEJ_AUTO_RESET_THRESHOLD = 10
_IMAGEJ_AUTO_HIST_BINS = 256


def _z_sum_dtype(dtype) -> np.dtype:
    """Return a safe accumulator dtype for an image Z projection."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.bool_) or np.issubdtype(dtype, np.unsignedinteger):
        return np.dtype(np.uint64)
    if np.issubdtype(dtype, np.signedinteger):
        return np.dtype(np.int64)
    if np.issubdtype(dtype, np.floating):
        return np.dtype(np.float64)
    if np.issubdtype(dtype, np.complexfloating):
        return np.dtype(np.complex128)
    raise TypeError(f"Cannot sum image planes with dtype {dtype}")


def _sum_z_planes(source, *, t: int, c: int, z_start: int, z_stop: int) -> np.ndarray:
    """Read and sum an inclusive, zero-based Z range without integer overflow."""
    if z_stop < z_start:
        raise ValueError("Z range end must be greater than or equal to its start")
    first = np.asarray(source.read_plane(t=t, c=c, z=z_start))
    if z_start == z_stop:
        return first
    accumulator = np.array(first, dtype=_z_sum_dtype(first.dtype), copy=True)
    for z_index in range(z_start + 1, z_stop + 1):
        plane = np.asarray(source.read_plane(t=t, c=c, z=z_index))
        if plane.shape != first.shape:
            raise ValueError(
                f"Z plane {z_index + 1} has shape {plane.shape}, expected {first.shape}"
            )
        np.add(accumulator, plane, out=accumulator, casting="unsafe")
    return accumulator


class TiffViewerWindow(QWidget):
    """Fiji-like single-file TIFF viewer backed by lazy plane reads."""

    def __init__(self, source: ImageSource, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source = source
        self._plane: np.ndarray | None = None
        self._manual_levels: tuple[float, float] | None = None
        self._auto_bc = True
        self._bc_auto_threshold = 0
        self._bc_dialog = None
        self._info_window: TiffInfoWindow | None = None
        self._active_cmap = "gray"
        self._show_axes = False          # axes/ticks hidden by default (right-click › Axis)
        self._roi_overlay = None         # single active ROI (ImageJ-style)
        self._acquisition_roi_visible = True
        self._z_range_values = (1, 1)

        self.setWindowTitle(self._title_text())
        self.setWindowIcon(QIcon(str(resource_path("icons", "minflux_viewer_logo.png"))))
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(880, 920)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._build_ui()
        self._install_shortcuts()
        self._refresh_controls()
        self._load_current_plane(fit_view=True)
        self._init_roi_overlay()

    def closeEvent(self, event) -> None:
        if self._roi_overlay is not None:
            try:
                self._roi_overlay.detach()
            except Exception:
                pass
            self._roi_overlay = None
        if self._info_window is not None:
            try:
                self._info_window.close()
            except Exception:
                pass
        if self._bc_dialog is not None:
            try:
                self._bc_dialog.close()
            except Exception:
                pass
        self._source.close()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")
        self._image_view = pg.ImageView(view=pg.PlotItem(enableMenu=False))
        self._image_view.ui.histogram.hide()
        self._image_view.ui.roiBtn.hide()
        self._image_view.ui.menuBtn.hide()
        try:
            self._image_view.view.hideButtons()
            self._image_view.view.autoBtn.hide()
            self._image_view.view.setMenuEnabled(False)
        except Exception:
            pass
        self._view_box = self._image_view.view.vb
        self._view_box.setAspectLocked(True)
        # Fiji / render-view convention: low Y on top, high Y at the bottom.
        self._view_box.invertY(True)
        self._set_axes_visible(self._show_axes)
        self._image_view.ui.graphicsView.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._image_view.ui.graphicsView.customContextMenuRequested.connect(
            self._show_context_menu
        )
        root.addWidget(self._image_view, stretch=1)

        self._control_row = QWidget()
        control = QHBoxLayout(self._control_row)
        control.setContentsMargins(0, 0, 0, 0)
        control.setSpacing(8)

        # Series picker (multi-series TIFF, or OBF stacks in a .msr) — switch
        # series in-place without reopening the file.
        self._series_label = QLabel("Series:")
        # Keep this selector anchored at the left edge when the Z controls are
        # hidden.  QLabel/QComboBox otherwise expand into the unused row width,
        # which makes the combo appear on the right for a 2-D series.
        self._series_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred
        )
        self._series_combo = QComboBox()
        self._series_combo.setMinimumWidth(180)
        self._series_combo.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        for name in self._source.series_names():
            self._series_combo.addItem(str(name))
        self._series_combo.setCurrentIndex(int(self._source.metadata.series_index))
        self._series_combo.currentIndexChanged.connect(self._on_series_changed)
        control.addWidget(self._series_label)
        control.addWidget(self._series_combo)

        self._acquisition_roi_check = QCheckBox(
            str(getattr(self._source, "active_roi_label", "acquisition ROI"))
        )
        self._acquisition_roi_check.setChecked(True)
        self._acquisition_roi_check.setToolTip(
            "Show the read-only MINFLUX acquisition area from the MSR metadata"
        )
        self._acquisition_roi_check.toggled.connect(
            self._on_acquisition_roi_toggled
        )
        control.addWidget(self._acquisition_roi_check)

        self._t_spin = self._make_axis_spin("T")
        self._c_spin = self._make_axis_spin("C")
        self._t_label = QLabel("T:")
        self._c_label = QLabel("C:")
        control.addWidget(self._t_label)
        control.addWidget(self._t_spin)
        control.addWidget(self._c_label)
        control.addWidget(self._c_spin)

        # Z as an inclusive range; the selected planes are displayed as a sum
        # projection. Keep the slider and its readout in one widget so hiding
        # Z for a 2-D series also removes its stretch space from the row.
        self._z_controls = QWidget()
        z_control = QHBoxLayout(self._z_controls)
        z_control.setContentsMargins(0, 0, 0, 0)
        z_control.setSpacing(8)
        self._z_label = QLabel("Z:")
        self._z_slider = DepthRangeSlider()
        self._z_slider.set_scroll_options(1.0, False)
        self._z_slider.setToolTip(
            "Drag either edge to select an inclusive Z range; all selected slices are summed"
        )
        self._z_slider.rangeChanged.connect(self._on_z_slider_changed)
        self._z_value = QLabel("1-1 / 1")
        z_control.addWidget(self._z_label)
        z_control.addWidget(self._z_slider, 1)
        z_control.addWidget(self._z_value)
        control.addWidget(self._z_controls, 1)
        # When Z is hidden, give the unused width to a trailing spacer so the
        # fixed Series selector remains anchored at the left edge. When Z is
        # visible, the spacer yields that width back to the range slider.
        control.addStretch(0)
        self._z_control_layout_index = control.indexOf(self._z_controls)
        self._trailing_spacer_layout_index = control.count() - 1
        root.addWidget(self._control_row)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info_label)

    def _make_axis_spin(self, axis: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setMinimum(1)
        spin.setMaximum(1)
        spin.setKeyboardTracking(False)
        spin.valueChanged.connect(lambda _value, a=axis: self._on_axis_changed(a))
        return spin

    def _install_shortcuts(self) -> None:
        info = QShortcut(QKeySequence("I"), self)
        info.setContext(Qt.ShortcutContext.WindowShortcut)
        info.activated.connect(self._show_info_window)

        # Brightness/Contrast (Shift+C) is an application-wide shortcut handled by
        # the main window, which delegates to this window's own _show_brightness_contrast
        # when it is focused. A local Shift+C QShortcut here would be an ambiguous
        # overload with the global one, so it is intentionally not installed.

    def _refresh_controls(self) -> None:
        axes = self._source.metadata.axes
        multi_series = int(self._source.metadata.series_count) > 1
        acquisition_roi = self._source_acquisition_roi()
        has_acquisition_roi = acquisition_roi is not None
        self._acquisition_roi_check.setVisible(has_acquisition_roi)
        any_visible = multi_series or has_acquisition_roi
        for axis, spin, lbl in (("T", self._t_spin, self._t_label),
                                ("C", self._c_spin, self._c_label)):
            size = self._source.axis_size(axis)
            spin.blockSignals(True)
            spin.setRange(1, max(size, 1))
            spin.setValue(1)
            spin.blockSignals(False)
            visible = axis in axes and size > 1
            spin.setVisible(visible)
            lbl.setVisible(visible)
            any_visible = any_visible or visible
        # Inclusive Z range, reset to the first plane when the series changes.
        z_size = max(self._source.axis_size("Z"), 1)
        self._z_slider.blockSignals(True)
        self._z_slider.set_limits(1, max(z_size, 2), reset_range=True)
        self._z_slider.set_range(1, 1)
        self._z_slider.blockSignals(False)
        self._z_range_values = (1, 1)
        self._z_value.setText(f"1-1 / {z_size}")
        z_visible = "Z" in axes and z_size > 1
        self._z_label.setVisible(z_visible)
        self._z_slider.setVisible(z_visible)
        self._z_value.setVisible(z_visible)
        self._z_controls.setVisible(z_visible)
        control = self._control_row.layout()
        control.setStretch(self._z_control_layout_index, int(z_visible))
        control.setStretch(self._trailing_spacer_layout_index, int(not z_visible))
        any_visible = any_visible or z_visible

        # Keep the image selector in the fixed, left-most control position for
        # every image that has navigable axes.  A single-series XYZ image still
        # gets the same selector slot (disabled, with its one image name), so
        # the Z/T/C controls do not jump left when the user changes images.
        show_series_slot = any_visible
        self._series_label.setVisible(show_series_slot)
        self._series_combo.setVisible(show_series_slot)
        self._series_combo.setEnabled(multi_series)
        self._control_row.setVisible(any_visible)

    def _set_axes_visible(self, show: bool) -> None:
        """Show/hide the plot axes + ticks (right-click › Axis). Hidden by default."""
        self._show_axes = bool(show)
        view = self._image_view.view
        for ax in ("left", "bottom"):
            try:
                view.showAxis(ax, self._show_axes)
            except Exception:
                pass
        if self._show_axes:
            try:
                view.setLabel("bottom", "X (nm)")
                view.setLabel("left", "Y (nm)")
            except Exception:
                pass

    def _title_text(self) -> str:
        meta = self._source.metadata
        if int(meta.series_count) > 1:
            return f"{self._source.path.name} — {meta.image_name}"
        return self._source.path.name

    def _on_series_changed(self, index: int) -> None:
        try:
            self._source.set_series(int(index))
        except Exception:
            return
        # Fresh series → drop manual levels and re-auto, refit the view.
        self._manual_levels = None
        self._auto_bc = True
        self._bc_auto_threshold = 0
        self.setWindowTitle(self._title_text())
        self._refresh_controls()
        self._load_current_plane(fit_view=True)
        if self._info_window is not None:
            self._info_window.refresh(self._source)
        if self._roi_overlay is not None:
            # A different series means a different calibration and a different
            # ROI — carrying the old one over would place it arbitrarily.
            meta = self._source.metadata
            self._roi_overlay.clear()
            self._roi_overlay.set_pixel_size(meta.pixel_size_x_nm, meta.pixel_size_y_nm)
            self._set_source_roi_overlay()

    def set_series_index(self, index: int) -> None:
        """Select a series through the same path as the visible dropdown.

        The MSR reader reuses one image window for several selected OBF stacks.
        Keeping this small public entry point here lets the main-window registry
        switch an already-open viewer without constructing a second window.
        """
        index = int(index)
        if not 0 <= index < self._series_combo.count():
            raise IndexError(f"Image series index {index} is out of range")
        self._series_combo.setCurrentIndex(index)

    def _on_axis_changed(self, axis: str) -> None:
        self._bc_auto_threshold = 0
        self._load_current_plane(fit_view=False)

    def _on_z_slider_changed(self, lo: float, hi: float) -> None:
        # The shared render range slider is continuous. Snap it to image-plane
        # indices before reading so the TIFF/OBF source always receives integers.
        lo_int = int(np.floor(float(lo) + 0.5))
        hi_int = int(np.floor(float(hi) + 0.5))
        self._set_z_range(lo_int, hi_int, reload=True)

    def _set_z_range(self, lo: int, hi: int, *, reload: bool) -> None:
        z_size = max(self._source.axis_size("Z"), 1)
        lo = int(np.clip(lo, 1, z_size))
        hi = int(np.clip(hi, 1, z_size))
        if hi < lo:
            lo, hi = hi, lo
        changed = (lo, hi) != self._z_range_values
        self._z_range_values = (lo, hi)
        self._z_slider.blockSignals(True)
        self._z_slider.set_range(lo, hi)
        self._z_slider.blockSignals(False)
        self._z_value.setText(f"{lo}-{hi} / {z_size}")
        if reload and changed:
            self._bc_auto_threshold = 0
            self._load_current_plane(fit_view=False)

    def _load_current_plane(self, *, fit_view: bool) -> None:
        z_lo, z_hi = self._z_range_values
        self._plane = _sum_z_planes(
            self._source,
            t=self._t_spin.value() - 1,
            c=self._c_spin.value() - 1,
            z_start=z_lo - 1,
            z_stop=z_hi - 1,
        )
        if self._auto_bc and not self._is_color_plane(self._plane):
            levels = self._compute_auto_levels(self._plane)
            if levels is not None:
                self._manual_levels = levels
        self._show_plane(fit_view=fit_view)
        if self._bc_dialog is not None and self._bc_dialog.isVisible():
            pixels = self._bc_pixels()
            if pixels is not None:
                self._bc_dialog.set_data(pixels)
                if self._manual_levels is not None:
                    self._bc_dialog.set_levels(*self._manual_levels)

    def _show_plane(self, *, fit_view: bool = False) -> None:
        if self._plane is None:
            self._info_label.setText("No readable TIFF plane.")
            return
        sx = self._source.metadata.pixel_size_x_nm
        sy = self._source.metadata.pixel_size_y_nm
        plane = np.asarray(self._plane)
        auto_levels = self._manual_levels is None and not self._is_color_plane(plane)
        self._image_view.setImage(
            plane,
            autoRange=False,
            autoLevels=auto_levels,
            pos=[0.0, 0.0],
            scale=[sx, sy],
        )
        if self._manual_levels is not None and not self._is_color_plane(plane):
            self._image_view.setLevels(*self._manual_levels)
        self._apply_colormap()
        if fit_view:
            self._fit_view()

        h, w = plane.shape[:2]
        meta = self._source.metadata
        position = self._position_text()
        self._info_label.setText(
            f"{meta.axes}  |  {w} x {h} px  |  dtype={plane.dtype}  |  "
            f"px=({sx:.4g}, {sy:.4g}) nm{position}"
        )
        self._update_roi_label()

    def _fit_view(self) -> None:
        if self._plane is None:
            return
        sx = self._source.metadata.pixel_size_x_nm
        sy = self._source.metadata.pixel_size_y_nm
        h, w = self._plane.shape[:2]
        self._view_box.setRange(xRange=(0.0, w * sx), yRange=(0.0, h * sy), padding=0)

    def _position_text(self) -> str:
        parts = []
        for axis, w in (("T", self._t_spin), ("C", self._c_spin)):
            if w.isVisible():
                parts.append(f"{axis}={w.value()}/{w.maximum()}")
        if self._z_slider.isVisible():
            lo, hi = self._z_range_values
            parts.append(f"Z={lo}-{hi}/{max(self._source.axis_size('Z'), 1)}")
        return "  |  " + ", ".join(parts) if parts else ""

    @staticmethod
    def _is_color_plane(plane: np.ndarray | None) -> bool:
        return plane is not None and plane.ndim == 3 and plane.shape[-1] in (3, 4)

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        cmap_menu = menu.addMenu("Colormap")
        for name in named_colormap_names():
            action = cmap_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(self._active_cmap == name)
            action.triggered.connect(lambda _checked=False, value=name: self._on_cmap_changed(value))
        pure_menu = cmap_menu.addMenu("Solid color")
        for name in solid_color_names():
            action = pure_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(self._active_cmap == name)
            action.triggered.connect(lambda _checked=False, value=name: self._on_cmap_changed(value))
        menu.addAction("Brightness/Contrast", self._show_brightness_contrast)

        # One active ROI per image, ImageJ-style: drawing replaces it, and it is
        # written into / read from the TIFF itself. No ROI Manager here.
        roi_menu = menu.addMenu("ROI")
        armed = self._roi_overlay.tool if self._roi_overlay is not None else None
        roi_editable = self._roi_overlay is not None and self._roi_overlay.editable
        for tool, label in (("rectangle", "Draw Rectangle"), ("oval", "Draw Oval")):
            action = roi_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(armed == tool)
            action.setEnabled(roi_editable)
            action.setToolTip("Drag on the image to draw; pick again to stop drawing.")
            action.triggered.connect(lambda _c=False, t=tool: self._set_roi_tool(t))
        roi_menu.addSeparator()
        delete = roi_menu.addAction("Delete ROI")
        delete.setEnabled(
            roi_editable
            and self._roi_overlay is not None
            and self._roi_overlay.has_roi()
        )
        delete.triggered.connect(self._delete_roi)
        roi_menu.setToolTipsVisible(True)

        menu.addAction("Save As TIFF…", self._save_as_tiff)
        axis_action = menu.addAction("Axis")
        axis_action.setCheckable(True)
        axis_action.setChecked(self._show_axes)
        axis_action.triggered.connect(self._set_axes_visible)
        menu.addAction("Show Info...", self._show_info_window)
        menu.addAction("Reset View", self._reset_view)
        menu.exec(self._image_view.ui.graphicsView.mapToGlobal(pos))

    def _on_cmap_changed(self, name: str) -> None:
        self._active_cmap = name
        self._apply_colormap()

    def _apply_colormap(self) -> None:
        if self._plane is None or self._is_color_plane(self._plane):
            return
        try:
            cmap = make_colormap(self._active_cmap)
        except (KeyError, ValueError) as exc:
            print(f"Unknown TIFF colormap '{self._active_cmap}'; using gray: {exc}")
            cmap = make_colormap("gray")
        self._image_view.setColorMap(cmap)

    def _reset_view(self) -> None:
        self._manual_levels = None
        self._auto_bc = True
        self._bc_auto_threshold = 0
        if self._bc_dialog is not None:
            self._bc_dialog.set_auto_state(True)
        self._load_current_plane(fit_view=True)

    def _bc_pixels(self) -> np.ndarray | None:
        if self._plane is None:
            return None
        if self._is_color_plane(self._plane):
            arr = np.asarray(self._plane, dtype=float)
            return np.nanmean(arr[..., :3], axis=-1)
        return np.asarray(self._plane, dtype=float)

    def _show_brightness_contrast(self) -> None:
        if self._is_color_plane(self._plane):
            return
        if self._bc_dialog is None:
            from .brightness_contrast_dialog import BrightnessContrastDialog
            self._bc_dialog = BrightnessContrastDialog(
                on_levels_changed=self._on_levels_changed,
                on_auto=self._on_bc_auto,
                on_reset=self._on_bc_reset,
                parent=self,
            )
            self._bc_dialog.set_auto_state(self._auto_bc)
        pixels = self._bc_pixels()
        if pixels is not None:
            self._bc_dialog.set_data(pixels)
            if self._manual_levels is not None:
                self._bc_dialog.set_levels(*self._manual_levels)
        self._bc_dialog.show()
        self._bc_dialog.raise_()
        self._bc_dialog.activateWindow()

    def _on_levels_changed(self, lo: float, hi: float) -> None:
        self._auto_bc = False
        self._bc_auto_threshold = 0
        self._manual_levels = (float(lo), float(hi))
        if self._bc_dialog is not None:
            self._bc_dialog.set_auto_state(False)
        self._show_plane(fit_view=False)

    def _on_bc_auto(self) -> None:
        pixels = self._bc_pixels()
        if pixels is None:
            return
        levels = self._compute_auto_levels(pixels, advance_auto_threshold=True)
        if levels is None:
            return
        self._auto_bc = True
        self._manual_levels = levels
        self._show_plane(fit_view=False)
        if self._bc_dialog is not None:
            self._bc_dialog.set_data(pixels)
            self._bc_dialog.set_levels(*levels)
            self._bc_dialog.set_auto_state(True)

    def _on_bc_reset(self) -> None:
        pixels = self._bc_pixels()
        if pixels is None:
            return
        values = np.asarray(pixels, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        lo = float(values.min())
        hi = float(values.max())
        if hi <= lo:
            hi = lo + 1.0
        self._manual_levels = (lo, hi)
        self._auto_bc = False
        self._bc_auto_threshold = 0
        self._show_plane(fit_view=False)
        if self._bc_dialog is not None:
            self._bc_dialog.set_data(pixels)
            self._bc_dialog.set_levels(lo, hi)
            self._bc_dialog.set_auto_state(False)

    def _compute_auto_levels(
        self,
        pixels: np.ndarray,
        *,
        advance_auto_threshold: bool = False,
    ) -> tuple[float, float] | None:
        values = np.asarray(pixels, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        data_min = float(values.min())
        data_max = float(values.max())
        if data_max <= data_min:
            return (data_min, data_min + 1.0)
        if advance_auto_threshold:
            if self._bc_auto_threshold < _IMAGEJ_AUTO_RESET_THRESHOLD:
                self._bc_auto_threshold = _IMAGEJ_AUTO_THRESHOLD
            else:
                self._bc_auto_threshold //= 2
        auto_threshold = self._bc_auto_threshold
        if auto_threshold < _IMAGEJ_AUTO_RESET_THRESHOLD:
            auto_threshold = _IMAGEJ_AUTO_THRESHOLD

        hist, _ = np.histogram(values, bins=_IMAGEJ_AUTO_HIST_BINS, range=(data_min, data_max))
        pixel_count = int(values.size)
        limit = pixel_count // 10
        threshold = pixel_count // int(auto_threshold)

        found = False
        i = -1
        while not found and i < _IMAGEJ_AUTO_HIST_BINS - 1:
            i += 1
            count = int(hist[i])
            if count > limit:
                count = 0
            found = count > threshold
        hmin = i

        found = False
        i = _IMAGEJ_AUTO_HIST_BINS
        while not found and i > 0:
            i -= 1
            count = int(hist[i])
            if count > limit:
                count = 0
            found = count > threshold
        hmax = i
        if hmax < hmin:
            return (data_min, data_max)
        bin_size = (data_max - data_min) / float(_IMAGEJ_AUTO_HIST_BINS)
        lo = data_min + hmin * bin_size
        hi = data_min + hmax * bin_size
        if hi <= lo:
            lo, hi = data_min, data_max
        return (float(lo), float(hi if hi > lo else lo + 1.0))

    # ------------------------------------------------------------------
    # Active ROI (ImageJ-style: one per image, stored in the file)
    # ------------------------------------------------------------------

    def _init_roi_overlay(self) -> None:
        """Attach the overlay and show the ROI the file already carries."""
        from .image_roi_overlay import ImageRoiOverlay

        meta = self._source.metadata
        self._roi_overlay = ImageRoiOverlay(
            self._view_box,
            pixel_size=(meta.pixel_size_x_nm, meta.pixel_size_y_nm),
            on_changed=self._on_roi_changed,
        )
        self._set_source_roi_overlay()

    def _source_roi(self):
        getter = getattr(self._source, "active_roi", None)
        try:
            return getter() if callable(getter) else None
        except Exception:
            return None

    def _source_acquisition_roi(self):
        if getattr(self._source, "active_roi_role", None) != "acquisition":
            return None
        return self._source_roi()

    def _set_source_roi_overlay(self) -> None:
        if self._roi_overlay is None:
            return
        roi = self._source_roi()
        is_acquisition = (
            roi is not None
            and getattr(self._source, "active_roi_role", None) == "acquisition"
        )
        read_only = is_acquisition and bool(
            getattr(self._source, "active_roi_read_only", False)
        )
        self._roi_overlay.set_roi(roi, notify=False, editable=not read_only)
        self._roi_overlay.set_visible(
            not is_acquisition or self._acquisition_roi_visible
        )
        self._update_roi_label()

    def _on_acquisition_roi_toggled(self, checked: bool) -> None:
        self._acquisition_roi_visible = bool(checked)
        if self._roi_overlay is not None:
            self._roi_overlay.set_visible(self._acquisition_roi_visible)

    def _on_roi_changed(self) -> None:
        self._update_roi_label()

    def _update_roi_label(self) -> None:
        """Append the ROI read-out to the info line (rebuilt by ``_show_plane``)."""
        if self._roi_overlay is None:
            return
        roi = self._roi_overlay.current_roi() if self._roi_overlay.visible else None
        base = self._info_label.text().split("  |  ROI: ")[0]
        self._info_label.setText(base + (f"  |  ROI: {roi.summary()}" if roi else ""))

    def _set_roi_tool(self, tool: str | None) -> None:
        if self._roi_overlay is None or not self._roi_overlay.editable:
            return
        # Re-picking the armed tool disarms it, so a drag pans again.
        self._roi_overlay.set_tool(None if self._roi_overlay.tool == tool else tool)

    def _delete_roi(self) -> None:
        if self._roi_overlay is not None and self._roi_overlay.editable:
            self._roi_overlay.clear(notify=True)

    def _save_as_tiff(self) -> None:
        """Write the current series — with its active ROI — as an OME-TIFF."""
        from ..core.tiff_export import export_image_series_to_tiff

        suggested = f"{self._source.metadata.image_name or self._source.path.stem}.tif"
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save image as OME-TIFF", suggested, "TIFF image (*.tif *.tiff)")
        if not path:
            return
        roi = self._roi_overlay.current_roi() if self._roi_overlay is not None else None
        try:
            export_image_series_to_tiff(self._source, path, roi=roi)
        except Exception as exc:                                   # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        note = "with its active ROI" if roi else "without a ROI (none is set)"
        QMessageBox.information(self, "Image saved", f"Saved {note}:\n{path}")

    def _show_info_window(self) -> None:
        if self._info_window is None:
            self._info_window = TiffInfoWindow(self._source)
            self._info_window.destroyed.connect(lambda *_: setattr(self, "_info_window", None))
        self._info_window.show()
        self._info_window.raise_()
        self._info_window.activateWindow()


class TiffInfoWindow(QDialog):
    """Structured, series-specific metadata dialog for an image source."""

    def __init__(self, source: ImageSource, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Image Information")
        self.resize(980, 760)
        self._root = QVBoxLayout(self)
        self._summary = QWidget()
        self._grid = QGridLayout(self._summary)
        self._grid.setColumnStretch(1, 1)
        self._root.addWidget(self._summary)
        self._documents = MetadataDocumentView(parent=self)
        self._root.addWidget(self._documents, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close_btn)
        self._root.addLayout(row)
        self.refresh(source)

    def refresh(self, source: ImageSource) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for row, (key, value) in enumerate(source.metadata.raw_summary):
            key_label = QLabel(key)
            key_label.setStyleSheet("color: gray; font-size: 11px;")
            value_label = QLabel(value)
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value_label.setWordWrap(True)
            self._grid.addWidget(key_label, row, 0)
            self._grid.addWidget(value_label, row, 1)
        self._documents.set_documents(source.metadata.documents)
