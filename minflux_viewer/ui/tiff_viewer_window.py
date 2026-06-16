"""
Standalone TIFF image viewer.

TIFF files are intentionally not MINFLUX datasets.  This window reuses the
render-view style while reading selected TIFF planes lazily from disk.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import resource_path
from ..core.tiff_source import TiffImageSource
from .render_window import _CHANNEL_COLORS, _COLORMAPS, _PURE_COLOR_LUTS

_IMAGEJ_AUTO_THRESHOLD = 5000
_IMAGEJ_AUTO_RESET_THRESHOLD = 10
_IMAGEJ_AUTO_HIST_BINS = 256


class TiffViewerWindow(QWidget):
    """Fiji-like single-file TIFF viewer backed by lazy plane reads."""

    def __init__(self, source: TiffImageSource, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source = source
        self._plane: np.ndarray | None = None
        self._manual_levels: tuple[float, float] | None = None
        self._auto_bc = True
        self._bc_auto_threshold = 0
        self._bc_dialog = None
        self._info_window: TiffInfoWindow | None = None
        self._active_cmap = "gray"

        self.setWindowTitle(source.path.name)
        self.setWindowIcon(QIcon(str(resource_path("icons", "minflux_viewer_logo.png"))))
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(880, 920)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._build_ui()
        self._install_shortcuts()
        self._refresh_controls()
        self._load_current_plane(fit_view=True)

    def closeEvent(self, event) -> None:
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
        self._view_box.invertY(False)
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

        self._t_spin = self._make_axis_spin("T")
        self._c_spin = self._make_axis_spin("C")
        self._z_spin = self._make_axis_spin("Z")
        for label, spin in (("T:", self._t_spin), ("C:", self._c_spin), ("Z:", self._z_spin)):
            lbl = QLabel(label)
            lbl.setProperty("axis_label", label[0])
            control.addWidget(lbl)
            control.addWidget(spin)
        control.addStretch()
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

        bc = QShortcut(QKeySequence("Shift+C"), self)
        bc.setContext(Qt.ShortcutContext.WindowShortcut)
        bc.activated.connect(self._show_brightness_contrast)

    def _refresh_controls(self) -> None:
        axes = self._source.metadata.axes
        any_visible = False
        for axis, spin in (("T", self._t_spin), ("C", self._c_spin), ("Z", self._z_spin)):
            size = self._source.axis_size(axis)
            spin.blockSignals(True)
            spin.setRange(1, max(size, 1))
            spin.setValue(1)
            spin.blockSignals(False)
            visible = axis in axes and size > 1
            spin.setVisible(visible)
            for label in self._control_row.findChildren(QLabel):
                if label.property("axis_label") == axis:
                    label.setVisible(visible)
            any_visible = any_visible or visible
        self._control_row.setVisible(any_visible)

    def _on_axis_changed(self, _axis: str) -> None:
        self._bc_auto_threshold = 0
        self._load_current_plane(fit_view=False)

    def _load_current_plane(self, *, fit_view: bool) -> None:
        self._plane = self._source.read_plane(
            t=self._t_spin.value() - 1,
            c=self._c_spin.value() - 1,
            z=self._z_spin.value() - 1,
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
            f"{meta.axes}  |  {w} x {h} px  |  dtype={meta.dtype}  |  "
            f"px=({sx:.4g}, {sy:.4g}) nm{position}"
        )

    def _fit_view(self) -> None:
        if self._plane is None:
            return
        sx = self._source.metadata.pixel_size_x_nm
        sy = self._source.metadata.pixel_size_y_nm
        h, w = self._plane.shape[:2]
        self._view_box.setRange(xRange=(0.0, w * sx), yRange=(0.0, h * sy), padding=0)

    def _position_text(self) -> str:
        parts = []
        for axis, spin in (("T", self._t_spin), ("C", self._c_spin), ("Z", self._z_spin)):
            if spin.isVisible():
                parts.append(f"{axis}={spin.value()}/{spin.maximum()}")
        return "  |  " + ", ".join(parts) if parts else ""

    @staticmethod
    def _is_color_plane(plane: np.ndarray | None) -> bool:
        return plane is not None and plane.ndim == 3 and plane.shape[-1] in (3, 4)

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        cmap_menu = menu.addMenu("Colormap")
        for name in _COLORMAPS:
            action = cmap_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(self._active_cmap == name)
            action.triggered.connect(lambda _checked=False, value=name: self._on_cmap_changed(value))
        pure_menu = cmap_menu.addMenu("Pure color")
        for name in _PURE_COLOR_LUTS:
            action = pure_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(self._active_cmap == name)
            action.triggered.connect(lambda _checked=False, value=name: self._on_cmap_changed(value))
        menu.addAction("Brightness/Contrast", self._show_brightness_contrast)
        menu.addAction("Show Info", self._show_info_window)
        menu.addAction("Reset View", self._reset_view)
        menu.exec(self._image_view.ui.graphicsView.mapToGlobal(pos))

    def _on_cmap_changed(self, name: str) -> None:
        self._active_cmap = name
        self._apply_colormap()

    def _apply_colormap(self) -> None:
        if self._plane is None or self._is_color_plane(self._plane):
            return
        try:
            cmap = pg.colormap.get(self._active_cmap)
        except Exception:
            cmap = None
        if cmap is None and self._active_cmap in _CHANNEL_COLORS:
            color = np.asarray(_CHANNEL_COLORS[self._active_cmap], dtype=float)
            rgba = np.array(
                [
                    [0, 0, 0, 255],
                    [
                        int(round(color[0] * 255)),
                        int(round(color[1] * 255)),
                        int(round(color[2] * 255)),
                        255,
                    ],
                ],
                dtype=np.ubyte,
            )
            cmap = pg.ColorMap(np.array([0.0, 1.0]), rgba)
        if cmap is None:
            try:
                import matplotlib as mpl
                mpl_cmap = mpl.colormaps[self._active_cmap].resampled(256)
                rgba = mpl_cmap(np.linspace(0.0, 1.0, 256), bytes=True)
                cmap = pg.ColorMap(np.linspace(0.0, 1.0, 256), rgba)
            except Exception:
                cmap = pg.colormap.get("gray")
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

    def _show_info_window(self) -> None:
        if self._info_window is None:
            self._info_window = TiffInfoWindow(self._source)
            self._info_window.destroyed.connect(lambda *_: setattr(self, "_info_window", None))
        self._info_window.show()
        self._info_window.raise_()
        self._info_window.activateWindow()


class TiffInfoWindow(QDialog):
    """Small metadata dialog for a standalone TIFF image source."""

    def __init__(self, source: TiffImageSource, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("TIFF Image Information")
        self.resize(640, 520)
        root = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        root.addLayout(grid)
        for row, (key, value) in enumerate(source.metadata.raw_summary):
            key_label = QLabel(key)
            key_label.setStyleSheet("color: gray; font-size: 11px;")
            value_label = QLabel(value)
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value_label.setWordWrap(True)
            grid.addWidget(key_label, row, 0)
            grid.addWidget(value_label, row, 1)

        if source.metadata.ome_xml:
            root.addWidget(QLabel("OME-XML"))
            text = QPlainTextEdit()
            text.setReadOnly(True)
            text.setPlainText(source.metadata.ome_xml)
            root.addWidget(text, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close_btn)
        root.addLayout(row)
