"""
minflux_viewer.ui.render_window
================================
Fast interactive render window.

Displays a localization dataset as a rendered 2-D image:
1. Build a 2-D histogram over the currently visible region.
2. Apply a Gaussian blur whose sigma adapts to the current zoom.
3. Show the result in a pyqtgraph ImageView with an interactive colormap.

Interactions
------------
* **Drag** the image to pan.
* **Scroll** on the image to zoom in/out around the cursor.
* **Orientation dropdown** — switch between XY (default), XZ, YZ.
* **Z slider** (3-D data) — step through Z slabs; toggle "All Z" for
  projection across the full depth range.
* **Sigma** spin box to override the automatic blur.
* **Reset view** resets orientation (XY), zoom, B&C, and Z to centre.
* Focus on the window to make its dataset the active one (Fiji-style).

Performance
-----------
* Rebinning uses ``numpy.histogram2d`` on the (filtered) loc coordinates
  of the currently visible region — not the full dataset.
* Rebinds are **debounced** to the next Qt idle tick so fast drag-pan
  doesn't queue up work.
* Render texture size is fixed (default 800 × 800 pixels); image quality
  stays constant regardless of how many localizations are loaded.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import affine_transform, gaussian_filter, zoom

from ..core.app_state import AppState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RENDER_SIZE   = 800     # target image dimensions in pixels
_DEBOUNCE_MS   = 25      # delay between view change and re-render
_Z_SLAB_COUNT  = 30      # number of Z slabs for 3-D slider
_COLORMAPS     = ["hot", "inferno", "viridis", "magma", "plasma", "cividis", "gray"]
_ORIENTATIONS  = ["XY", "XZ", "YZ"]
_CHANNEL_LUTS  = ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Gray", *_COLORMAPS]
_CHANNEL_COLORS = {
    "Red": (1.0, 0.0, 0.0),
    "Green": (0.0, 1.0, 0.0),
    "Blue": (0.0, 0.0, 1.0),
    "Cyan": (0.0, 1.0, 1.0),
    "Magenta": (1.0, 0.0, 1.0),
    "Yellow": (1.0, 1.0, 0.0),
    "Gray": (1.0, 1.0, 1.0),
}
_CACHE_LIMIT = 16


class ManualAlignDialog(QDialog):
    """Modal keyboard-driven alignment helper for one rendered channel."""

    def __init__(self, render_window: "RenderWindow", ch_idx: int) -> None:
        super().__init__(render_window)
        self._render_window = render_window
        self._ch_idx = ch_idx
        ch = render_window._channels[ch_idx]
        self._original = dict(ch.get("transform") or {"dx": 0.0, "dy": 0.0, "angle": 0.0})
        self._ensure_world_transform()

        self.setWindowTitle("Manual channel alignment")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setModal(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.addWidget(QLabel(f"Apply manual translation / rotation to channel:\n{ch['name']}"))
        root.addWidget(QLabel("Keep this message window open and focused while pressing keys."))
        root.addWidget(QLabel("Arrow keys: move up/down, left/right"))

        move_row = QHBoxLayout()
        move_row.addWidget(QLabel("Each key press moves"))
        self._move_spin = QDoubleSpinBox()
        self._move_spin.setDecimals(3)
        self._move_spin.setRange(0.001, 1000.0)
        self._move_spin.setValue(0.5)
        self._move_spin.setSuffix(" pixel")
        move_row.addWidget(self._move_spin)
        move_row.addStretch()
        root.addLayout(move_row)

        root.addWidget(QLabel("Ctrl+Right: rotate clockwise; Ctrl+Left: rotate counterclockwise"))

        rotate_row = QHBoxLayout()
        rotate_row.addWidget(QLabel("Each key press rotates"))
        self._rotate_spin = QDoubleSpinBox()
        self._rotate_spin.setDecimals(3)
        self._rotate_spin.setRange(0.001, 45.0)
        self._rotate_spin.setValue(0.5)
        self._rotate_spin.setSuffix(" degree")
        rotate_row.addWidget(self._rotate_spin)
        rotate_row.addStretch()
        root.addLayout(rotate_row)

        self._status = QLabel("")
        self._status.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._status)

        button_row = QHBoxLayout()
        reset_btn = QPushButton("reset")
        reset_btn.clicked.connect(self._reset_transform)
        cancel_btn = QPushButton("cancel")
        cancel_btn.clicked.connect(self._cancel)
        apply_btn = QPushButton("apply")
        apply_btn.clicked.connect(self.accept)
        button_row.addStretch()
        button_row.addWidget(reset_btn)
        button_row.addWidget(cancel_btn)
        button_row.addWidget(apply_btn)
        root.addLayout(button_row)

        self._update_status()

    def keyPressEvent(self, event) -> None:
        transform = self._render_window._channels[self._ch_idx]["transform"]
        step = float(self._move_spin.value())
        rotate_step = float(self._rotate_spin.value())
        pixel_size_nm = self._render_window._current_render_pixel_size_nm()
        key = event.key()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Right:
            transform["angle"] -= rotate_step
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Left:
            transform["angle"] += rotate_step
        elif key == Qt.Key.Key_Left:
            transform["dx_nm"] -= step * pixel_size_nm
        elif key == Qt.Key.Key_Right:
            transform["dx_nm"] += step * pixel_size_nm
        elif key == Qt.Key.Key_Up:
            transform["dy_nm"] += step * pixel_size_nm
        elif key == Qt.Key.Key_Down:
            transform["dy_nm"] -= step * pixel_size_nm
        else:
            super().keyPressEvent(event)
            return
        self._update_status()
        self._render_window._compose_from_cache()
        event.accept()

    def reject(self) -> None:
        self._restore_original()
        super().reject()

    def closeEvent(self, event) -> None:
        self._restore_original()
        super().closeEvent(event)

    def _reset_transform(self) -> None:
        transform = self._render_window._channels[self._ch_idx]["transform"]
        transform.update({"dx_nm": 0.0, "dy_nm": 0.0, "angle": 0.0})
        self._update_status()
        self._render_window._compose_from_cache()
        self.setFocus()

    def _cancel(self) -> None:
        self._restore_original()
        self.reject()

    def _restore_original(self) -> None:
        self._render_window._channels[self._ch_idx]["transform"] = dict(self._original)
        self._render_window._compose_from_cache()

    def _update_status(self) -> None:
        transform = self._render_window._channels[self._ch_idx]["transform"]
        pixel_size_nm = self._render_window._current_render_pixel_size_nm()
        dx_px = float(transform.get("dx_nm", 0.0)) / pixel_size_nm
        dy_px = float(transform.get("dy_nm", 0.0)) / pixel_size_nm
        self._status.setText(
            f"Current transform: dx={dx_px:.3f} px, "
            f"dy={dy_px:.3f} px, rotation={transform['angle']:.3f} deg"
        )

    def _ensure_world_transform(self) -> None:
        ch = self._render_window._channels[self._ch_idx]
        transform = ch.setdefault("transform", {})
        pixel_size_nm = self._render_window._current_render_pixel_size_nm()
        if "dx_nm" not in transform:
            transform["dx_nm"] = float(transform.get("dx", 0.0)) * pixel_size_nm
        if "dy_nm" not in transform:
            transform["dy_nm"] = float(transform.get("dy", 0.0)) * pixel_size_nm
        if "angle" not in transform:
            transform["angle"] = 0.0
        if "anchor_x_nm" not in transform or "anchor_y_nm" not in transform:
            (x0, x1), (y0, y1) = self._render_window._view_box.viewRange()
            transform["anchor_x_nm"] = (x0 + x1) / 2.0
            transform["anchor_y_nm"] = (y0 + y1) / 2.0


class RenderWindow(QWidget):
    """
    Fast interactive 2-D Gaussian-blur render window.

    Reacts live to pan / zoom / filter changes; uses a Z slider for 3-D data.
    """

    TAG = "render_window"

    def __init__(
        self,
        state: AppState,
        dataset_idx: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._idx   = dataset_idx if dataset_idx is not None else state.active_idx

        self._locs_nm: np.ndarray | None = None  # (N, 3) filtered
        self._xy: np.ndarray | None = None       # (N, 2) histogram axes
        self._depth: np.ndarray | None = None    # (N,)   remaining axis
        self._orientation: str = "XY"
        self._has_depth: bool = False
        self._bounds_xy: tuple[float, float, float, float] = (0, 1, 0, 1)
        self._bounds_depth: tuple[float, float] = (0, 0)
        self._render_mode: str = "localizations"
        self._image_data: np.ndarray | None = None
        self._dataset_dim_label: str = "2D"
        self._channels: list[dict] = []
        self._channel_rows: list[tuple[QCheckBox, QComboBox]] = []
        self._tile_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._last_scalar_tile: np.ndarray | None = None
        self._last_tile_geometry: tuple[float, float, float, float] | None = None

        # Manual levels override (set by B&C). None → pyqtgraph autoLevels.
        self._manual_levels: tuple[float, float] | None = None
        self._auto_bc: bool = True   # True → recompute levels each render
        self._bc_dialog = None
        self._roi_overlay = None

        self.setWindowTitle("Render")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(880, 920)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(_DEBOUNCE_MS)
        self._redraw_timer.timeout.connect(self._render)

        self._build_ui()
        self._refresh_from_dataset()

        state.active_changed.connect(self._on_active_changed)
        state.filter_changed.connect(self._on_filter_changed)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(8)

        top.addWidget(QLabel("View:"))
        self._orient_combo = QComboBox()
        self._orient_combo.addItems(_ORIENTATIONS)
        self._orient_combo.setCurrentText("XY")
        self._orient_combo.currentTextChanged.connect(self._on_orientation_changed)
        top.addWidget(self._orient_combo)

        top.addSpacing(12)

        top.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(_COLORMAPS)
        self._cmap_combo.setCurrentText("hot")
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        top.addWidget(self._cmap_combo)

        top.addSpacing(12)

        top.addWidget(QLabel("Sigma (nm):"))
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setDecimals(2)
        self._sigma_spin.setRange(0.0, 10000.0)
        self._sigma_spin.setSingleStep(1.0)
        self._sigma_spin.setValue(0.0)
        self._sigma_spin.setSpecialValueText("auto")
        self._sigma_spin.setToolTip("0 = auto-scale with zoom")
        self._sigma_spin.editingFinished.connect(self._schedule_render)
        top.addWidget(self._sigma_spin)

        top.addSpacing(12)

        bc_btn = QPushButton("B&&C…")
        bc_btn.setToolTip("Brightness & Contrast")
        bc_btn.clicked.connect(self._show_brightness_contrast)
        top.addWidget(bc_btn)

        self._axis_check = QCheckBox("Axis")
        self._axis_check.setChecked(False)
        self._axis_check.toggled.connect(self._set_axes_visible)
        top.addWidget(self._axis_check)

        reset_btn = QPushButton("Reset view")
        reset_btn.clicked.connect(self._reset_view)
        top.addWidget(reset_btn)

        top.addStretch()
        root.addLayout(top)

        pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")
        self._image_view = pg.ImageView(view=pg.PlotItem())
        self._image_view.ui.histogram.hide()
        self._image_view.ui.roiBtn.hide()
        self._image_view.ui.menuBtn.hide()
        self._image_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        try:
            self._image_view.view.hideButtons()
            self._image_view.view.autoBtn.hide()
        except Exception:
            pass
        self._view_box = self._image_view.view.vb
        self._view_box.setAspectLocked(True)
        self._view_box.invertY(False)
        self._view_box.sigRangeChanged.connect(self._on_range_changed)
        self._image_view.wheelEvent = self._wheel_zoom  # type: ignore[assignment]
        self._set_axes_visible(False)

        root.addWidget(self._image_view, stretch=1)
        from .roi_overlay import RoiOverlayController
        self._roi_overlay = RoiOverlayController(
            self._state.rois,
            self,
            self._image_view.ui.graphicsView,
            self._image_view.view,
            coordinate_space="plot",
        )

        self._channel_area = QScrollArea()
        self._channel_area.setWidgetResizable(True)
        self._channel_area.setMaximumHeight(120)
        self._channel_widget = QWidget()
        self._channel_layout = QVBoxLayout(self._channel_widget)
        self._channel_layout.setContentsMargins(4, 4, 4, 4)
        self._channel_layout.setSpacing(2)
        self._channel_area.setWidget(self._channel_widget)
        root.addWidget(self._channel_area)

        # Depth slider row — XY→Z, XZ→Y, YZ→X
        self._depth_row = QWidget()
        z_lay = QHBoxLayout(self._depth_row)
        z_lay.setContentsMargins(0, 0, 0, 0)
        z_lay.setSpacing(6)

        self._all_depth_check = QCheckBox("All")
        self._all_depth_check.setChecked(True)
        self._all_depth_check.setToolTip("Project across the full depth range")
        self._all_depth_check.toggled.connect(self._on_all_depth_toggled)
        z_lay.addWidget(self._all_depth_check)

        self._depth_axis_label = QLabel("Z:")
        self._depth_axis_label.setMinimumWidth(22)
        z_lay.addWidget(self._depth_axis_label)

        self._depth_slider = QSlider(Qt.Orientation.Horizontal)
        self._depth_slider.setRange(0, _Z_SLAB_COUNT - 1)
        self._depth_slider.setValue(_Z_SLAB_COUNT // 2)
        self._depth_slider.setEnabled(False)
        self._depth_slider.valueChanged.connect(self._on_depth_changed)
        z_lay.addWidget(self._depth_slider, stretch=1)

        self._depth_label = QLabel("all")
        self._depth_label.setMinimumWidth(180)
        self._depth_label.setStyleSheet("color: gray; font-size: 11px;")
        z_lay.addWidget(self._depth_label)

        root.addWidget(self._depth_row)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info_label)

    # ------------------------------------------------------------------
    # Dataset binding
    # ------------------------------------------------------------------

    def _refresh_from_dataset(self) -> None:
        ds = self._state.datasets[self._idx] if self._idx is not None else None
        if ds is None:
            self._locs_nm = self._xy = self._depth = self._image_data = None
            self.setWindowTitle("Render")
            self._info_label.setText("No dataset.")
            return

        self.setWindowTitle(f"Render  —  {ds.name}")
        self._dataset_dim_label = f"{ds.prop.num_dim}D"
        self._build_channels()
        self._rebuild_channel_ui()
        self._tile_cache.clear()

        if ds.image_data is not None and not ds.has_localizations:
            self._render_mode = "image"
            self._locs_nm = self._xy = self._depth = None
            self._image_data = self._prepare_image_payload(ds.image_data)
            self._depth_row.setVisible(False)
            self._bounds_xy = self._image_bounds(ds, self._image_data)
            self._fit_view()
            self._schedule_render()
            return

        self._render_mode = "localizations"
        self._image_data = None
        self._locs_nm = self._channel_locs(self._channels[0]) if self._channels else np.empty((0, 3))

        if self._locs_nm.shape[0] == 0:
            self._info_label.setText("No finite localisations pass the current filter.")
            return

        self._apply_orientation()
        self._schedule_render()

    def _build_channels(self) -> None:
        """Build lightweight channel descriptors; scalar tiles are computed lazily."""
        previous_transforms = {
            ch.get("dataset_idx"): dict(ch.get("transform") or {"dx": 0.0, "dy": 0.0, "angle": 0.0})
            for ch in self._channels
        }
        self._channels = []
        color_cycle = ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Gray"]
        active_recent = None
        if self._idx is not None and 0 <= self._idx < len(self._state.datasets):
            ds0 = self._state.datasets[self._idx]
            active_recent = ds0.file.recent_path or ds0.file.path
        for idx, ds in enumerate(self._state.datasets):
            if not (ds.has_localizations or ds.image_data is not None):
                continue
            source = ds.file.recent_path or ds.file.path
            same_source = active_recent is not None and source == active_recent
            visible = idx == self._idx or same_source
            self._channels.append({
                "dataset_idx": idx,
                "name": ds.name,
                "kind": "image" if ds.image_data is not None and not ds.has_localizations else "localizations",
                "visible": visible,
                "lut": color_cycle[len(self._channels) % len(color_cycle)],
                "levels": None,
                "transform": previous_transforms.get(idx, {"dx_nm": 0.0, "dy_nm": 0.0, "angle": 0.0}),
            })
        if len(self._channels) == 1:
            self._channels[0]["lut"] = self._cmap_combo.currentText()
        if not self._channels and self._idx is not None:
            ds = self._state.datasets[self._idx]
            self._channels.append({
                "dataset_idx": self._idx,
                "name": ds.name,
                "kind": "localizations",
                "visible": True,
                "lut": "Red",
                "levels": None,
                "transform": previous_transforms.get(self._idx, {"dx_nm": 0.0, "dy_nm": 0.0, "angle": 0.0}),
            })

    def _manual_align_channel(self, ch_idx: int) -> None:
        if not (0 <= ch_idx < len(self._channels)):
            return
        dialog = ManualAlignDialog(self, ch_idx)
        dialog.exec()

    def _rebuild_channel_ui(self) -> None:
        while self._channel_layout.count():
            item = self._channel_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._channel_rows = []
        for ch_idx, ch in enumerate(self._channels):
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox(f"{ch_idx + 1}: {ch['name']}")
            cb.setChecked(bool(ch["visible"]))
            cb.toggled.connect(lambda checked, i=ch_idx: self._on_channel_visible(i, checked))
            lay.addWidget(cb, stretch=1)
            lut = QComboBox()
            lut.addItems(_CHANNEL_LUTS)
            lut.setCurrentText(ch["lut"])
            lut.currentTextChanged.connect(lambda text, i=ch_idx: self._on_channel_lut(i, text))
            lay.addWidget(lut)
            align_btn = QPushButton("manual align")
            align_btn.clicked.connect(lambda _checked=False, i=ch_idx: self._manual_align_channel(i))
            lay.addWidget(align_btn)
            self._channel_layout.addWidget(row)
            self._channel_rows.append((cb, lut))
        self._channel_area.setVisible(len(self._channels) > 1)

    def _on_channel_visible(self, ch_idx: int, visible: bool) -> None:
        if 0 <= ch_idx < len(self._channels):
            self._channels[ch_idx]["visible"] = bool(visible)
            self._compose_from_cache()

    def _on_channel_lut(self, ch_idx: int, lut: str) -> None:
        if 0 <= ch_idx < len(self._channels):
            self._channels[ch_idx]["lut"] = lut
            self._compose_from_cache()

    def _channel_locs(self, ch: dict) -> np.ndarray:
        ds = self._state.datasets[ch["dataset_idx"]]
        return self._dataset_locs(ds)

    def _dataset_locs(self, ds) -> np.ndarray:
        try:
            locs = np.asarray(ds.loc_nm, dtype=np.float64)
        except Exception:
            return np.empty((0, 3), dtype=np.float64)
        if locs.ndim != 2 or locs.shape[1] < 2:
            return np.empty((0, 3), dtype=np.float64)
        if locs.shape[1] == 2:
            locs = np.column_stack([locs, np.zeros(locs.shape[0], dtype=np.float64)])
        mask = np.asarray(ds.filter_mask, dtype=bool)
        if mask.shape[0] == locs.shape[0]:
            locs = locs[mask]
        finite = np.all(np.isfinite(locs[:, :3]), axis=1)
        return locs[finite, :3]

    def _oriented_locs(self, ds) -> np.ndarray:
        locs = self._dataset_locs(ds)
        if locs.shape[0] == 0:
            return locs
        if self._orientation == "XY":
            return locs[:, [0, 1, 2]]
        if self._orientation == "XZ":
            return locs[:, [0, 2, 1]]
        if self._orientation == "YZ":
            return locs[:, [1, 2, 0]]
        return locs

    def _image_bounds(self, ds, image: np.ndarray) -> tuple[float, float, float, float]:
        ox, oy = ds.image_origin_nm
        sx, sy = ds.image_pixel_size_nm
        height, width = image.shape[:2]
        return (ox, ox + width * sx, oy, oy + height * sy)

    def _prepare_image_payload(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float64)
        if arr.ndim == 2:
            return arr
        if arr.ndim >= 3:
            axes = tuple(range(arr.ndim - 2))
            return np.nanmax(arr, axis=axes)
        raise ValueError("Image payload must be at least 2D")

    def _show_image_dataset(self, ds, *, fit_view: bool = False) -> None:
        if self._image_data is None:
            self._info_label.setText("No image data.")
            return
        ox, oy = ds.image_origin_nm
        sx, sy = ds.image_pixel_size_nm
        height, width = self._image_data.shape[:2]
        self._bounds_xy = (ox, ox + width * sx, oy, oy + height * sy)
        self._image_view.setImage(
            self._image_data,
            autoRange=False,
            autoLevels=self._manual_levels is None,
            pos=[ox, oy],
            scale=[sx, sy],
        )
        if self._manual_levels is not None:
            self._image_view.setLevels(*self._manual_levels)
        self._on_cmap_changed(self._cmap_combo.currentText())
        if fit_view:
            self._fit_view()
        self._info_label.setText(
            f"{self._dataset_dim_label} image  |  {width} × {height} px  |  "
            f"px=({sx:.1f}, {sy:.1f}) nm"
        )

    def _apply_orientation(self) -> None:
        """Split (N,3) locs into (xy_pair, depth) based on current orientation."""
        if self._locs_nm is None or self._locs_nm.shape[0] == 0:
            return

        o = self._orientation
        if o == "XY":
            xy_idx, depth_idx, depth_name = (0, 1), 2, "Z"
        elif o == "XZ":
            xy_idx, depth_idx, depth_name = (0, 2), 1, "Y"
        elif o == "YZ":
            xy_idx, depth_idx, depth_name = (1, 2), 0, "X"
        else:
            return

        self._xy    = self._locs_nm[:, list(xy_idx)]
        self._depth = self._locs_nm[:, depth_idx]

        x, y = self._xy[:, 0], self._xy[:, 1]
        pad_x = (x.max() - x.min()) * 0.02 if x.max() > x.min() else 1.0
        pad_y = (y.max() - y.min()) * 0.02 if y.max() > y.min() else 1.0
        self._bounds_xy = (
            float(x.min() - pad_x), float(x.max() + pad_x),
            float(y.min() - pad_y), float(y.max() + pad_y),
        )
        for ch in self._channels:
            if not ch["visible"] or ch["dataset_idx"] == self._idx:
                continue
            ds_ch = self._state.datasets[ch["dataset_idx"]]
            if ch["kind"] == "image" and ds_ch.image_data is not None:
                image = self._prepare_image_payload(ds_ch.image_data)
                bx0, bx1, by0, by1 = self._image_bounds(ds_ch, image)
            else:
                locs_ch = self._oriented_locs(ds_ch)
                if locs_ch.shape[0] == 0:
                    continue
                bx0, bx1 = float(locs_ch[:, 0].min()), float(locs_ch[:, 0].max())
                by0, by1 = float(locs_ch[:, 1].min()), float(locs_ch[:, 1].max())
            self._bounds_xy = (
                min(self._bounds_xy[0], bx0),
                max(self._bounds_xy[1], bx1),
                min(self._bounds_xy[2], by0),
                max(self._bounds_xy[3], by1),
            )
        self._bounds_depth = (float(self._depth.min()), float(self._depth.max()))
        self._has_depth = (self._bounds_depth[1] - self._bounds_depth[0]) > 1.0

        ds = self._state.datasets[self._idx]
        dataset_is_3d = ds.prop.num_dim == 3
        self._depth_row.setVisible(dataset_is_3d)
        self._depth_axis_label.setText(f"{depth_name}:")
        self._depth_slider.setEnabled(
            self._has_depth and not self._all_depth_check.isChecked()
        )

        self._fit_view()

    def _fit_view(self) -> None:
        x0, x1, y0, y1 = self._bounds_xy
        self._view_box.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)

    def _reset_view(self) -> None:
        """Reset orientation (XY), zoom, B&C levels, and depth to centre."""
        self._manual_levels = None
        self._auto_bc = True
        if self._bc_dialog is not None:
            self._bc_dialog.set_auto_state(True)
        if self._render_mode == "image":
            ds = self._state.datasets[self._idx] if self._idx is not None else None
            if ds is not None:
                self._image_data = self._prepare_image_payload(ds.image_data)
                self._bounds_xy = self._image_bounds(ds, self._image_data)
                self._fit_view()
                self._schedule_render()
            return
        self._sigma_spin.setValue(0.0)
        self._all_depth_check.setChecked(True)
        self._depth_slider.setValue(_Z_SLAB_COUNT // 2)
        if self._orient_combo.currentText() != "XY":
            self._orient_combo.setCurrentText("XY")
        else:
            self._apply_orientation()
            self._schedule_render()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _schedule_render(self) -> None:
        self._redraw_timer.start()

    def _render(self) -> None:
        (x0, x1), (y0, y1) = self._view_box.viewRange()
        if x1 <= x0 or y1 <= y0:
            return

        view_w, view_h = x1 - x0, y1 - y0
        pixel_size_nm = max(view_w, view_h) / _RENDER_SIZE
        n_bins_x = max(int(round(view_w / pixel_size_nm)), 8)
        n_bins_y = max(int(round(view_h / pixel_size_nm)), 8)

        tiles = []
        counts = []
        for ch_idx, ch in enumerate(self._channels):
            tile, count = self._tile_for_channel(ch_idx, ch, x0, x1, y0, y1, n_bins_y, n_bins_x, pixel_size_nm)
            tiles.append(tile)
            counts.append(count if ch["visible"] else 0)
        if not tiles:
            self._image_view.clear()
            return
        scalar = np.stack(tiles, axis=0).astype(np.float32, copy=False)
        self._last_scalar_tile = scalar
        self._last_tile_geometry = (x0, x1, y0, y1)

        if self._auto_bc and scalar.shape[0] == 1:
            levels = self._compute_auto_levels(scalar[0])
            if levels is not None:
                self._manual_levels = levels
                if self._bc_dialog is not None and self._bc_dialog.isVisible():
                    self._bc_dialog.set_levels(*levels)

        rgba = self._compose_rgba(scalar)
        self._image_view.setImage(rgba, autoRange=False, autoLevels=False, pos=[x0, y0], scale=[(x1 - x0) / n_bins_x, (y1 - y0) / n_bins_y])

        if self._bc_dialog is not None and self._bc_dialog.isVisible():
            first = scalar[0] if scalar.size else np.zeros((1, 1))
            self._bc_dialog.set_data_range(float(first.min()), float(first.max()))

        self._info_label.setText(
            f"{self._dataset_dim_label}  |  "
            f"{sum(counts):,} samples in view  |  {len([c for c in self._channels if c['visible']])} channel(s)  |  "
            f"{n_bins_x} × {n_bins_y} px  |  "
            f"px={pixel_size_nm:.1f} nm"
        )

    def _bc_pixels(self) -> np.ndarray | None:
        if self._last_scalar_tile is not None:
            for idx, ch in enumerate(self._channels):
                if ch["visible"] and idx < self._last_scalar_tile.shape[0]:
                    return self._last_scalar_tile[idx]
        img = self._image_view.imageItem.image
        return None if img is None else np.asarray(img)

    def _compose_from_cache(self, *_args) -> None:
        if self._last_scalar_tile is None:
            self._schedule_render()
            return
        if self._last_tile_geometry is None:
            (x0, x1), (y0, y1) = self._view_box.viewRange()
        else:
            x0, x1, y0, y1 = self._last_tile_geometry
        _, h, w = self._last_scalar_tile.shape
        rgba = self._compose_rgba(self._last_scalar_tile)
        self._image_view.setImage(
            rgba,
            autoRange=False,
            autoLevels=False,
            pos=[x0, y0],
            scale=[(x1 - x0) / max(w, 1), (y1 - y0) / max(h, 1)],
        )

    def _current_render_pixel_size_nm(self) -> float:
        if self._last_tile_geometry is not None and self._last_scalar_tile is not None:
            x0, x1, y0, y1 = self._last_tile_geometry
            _, h, w = self._last_scalar_tile.shape
            return max((x1 - x0) / max(w, 1), (y1 - y0) / max(h, 1), 1e-12)
        (x0, x1), (y0, y1) = self._view_box.viewRange()
        return max((x1 - x0) / _RENDER_SIZE, (y1 - y0) / _RENDER_SIZE, 1e-12)

    def _compose_rgba(self, scalar: np.ndarray) -> np.ndarray:
        if scalar.ndim != 3 or scalar.shape[0] == 0:
            return np.zeros((1, 1, 4), dtype=np.float32)
        c, h, w = scalar.shape
        visible = [i for i, ch in enumerate(self._channels[:c]) if ch["visible"]]
        if not visible:
            rgba = np.zeros((h, w, 4), dtype=np.float32)
            rgba[..., 3] = 1.0
            return rgba

        rgb = np.zeros((h, w, 3), dtype=np.float32)
        for idx in visible:
            tile = self._transformed_tile(scalar[idx], self._channels[idx])
            norm = self._normalized_tile(tile, self._channels[idx])
            rgb += self._map_norm_to_rgb(norm, self._channels[idx]["lut"])
        np.clip(rgb, 0.0, 1.0, out=rgb)

        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[..., :3] = rgb
        return rgba

    def _transformed_tile(self, tile: np.ndarray, ch: dict) -> np.ndarray:
        transform = ch.get("transform") or {}
        pixel_size_nm = self._current_render_pixel_size_nm()
        dx_nm = float(transform.get("dx_nm", float(transform.get("dx", 0.0)) * pixel_size_nm))
        dy_nm = float(transform.get("dy_nm", float(transform.get("dy", 0.0)) * pixel_size_nm))
        angle = float(transform.get("angle", 0.0))
        if abs(dx_nm) < 1e-9 and abs(dy_nm) < 1e-9 and abs(angle) < 1e-9:
            return tile
        if self._last_tile_geometry is None:
            return tile
        x0, x1, y0, y1 = self._last_tile_geometry
        h, w = tile.shape
        sx = (x1 - x0) / max(w, 1)
        sy = (y1 - y0) / max(h, 1)
        if sx == 0.0 or sy == 0.0:
            return tile
        theta = np.deg2rad(angle)
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
        inv_rot_world = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)
        tile_to_world = np.array([[0.0, sx], [sy, 0.0]], dtype=np.float64)
        world_to_tile = np.array([[0.0, 1.0 / sy], [1.0 / sx, 0.0]], dtype=np.float64)
        world_origin = np.array([x0, y0], dtype=np.float64)
        anchor = np.array([
            float(transform.get("anchor_x_nm", (x0 + x1) / 2.0)),
            float(transform.get("anchor_y_nm", (y0 + y1) / 2.0)),
        ], dtype=np.float64)
        shift = np.array([dx_nm, dy_nm], dtype=np.float64)
        matrix = world_to_tile @ inv_rot_world @ tile_to_world
        offset = world_to_tile @ (inv_rot_world @ (world_origin - anchor - shift) + anchor - world_origin)
        return affine_transform(
            tile,
            matrix,
            offset=offset,
            output_shape=tile.shape,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        ).astype(np.float32, copy=False)

    def _normalized_tile(self, tile: np.ndarray, ch: dict) -> np.ndarray:
        levels = ch.get("levels")
        if levels is None:
            levels = self._manual_levels if len(self._channels) == 1 else self._compute_auto_levels(tile)
        if levels is None:
            lo, hi = float(np.nanmin(tile)), float(np.nanmax(tile))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0
        else:
            lo, hi = levels
        norm = (tile.astype(np.float32, copy=False) - float(lo)) / max(float(hi) - float(lo), 1e-12)
        return np.clip(norm, 0.0, 1.0)

    def _map_norm_to_rgb(self, norm: np.ndarray, lut: str) -> np.ndarray:
        if lut in _CHANNEL_COLORS:
            return norm[..., None] * np.asarray(_CHANNEL_COLORS[lut], dtype=np.float32)
        try:
            cmap = pg.colormap.get(lut)
        except Exception:
            cmap = None
        if cmap is None:
            try:
                cmap = self._pg_colormap_from_matplotlib(lut)
            except Exception:
                cmap = self._pg_colormap_from_matplotlib("hot")
        table = cmap.getLookupTable(0.0, 1.0, 256)
        table = np.asarray(table, dtype=np.float32)
        if table.max() > 1.0:
            table /= 255.0
        idx = np.clip((norm * 255).astype(np.int16), 0, 255)
        return table[idx, :3]

    def _tile_for_channel(self, ch_idx: int, ch: dict, x0: float, x1: float, y0: float, y1: float, h: int, w: int, pixel_size_nm: float) -> tuple[np.ndarray, int]:
        sigma_nm = self._sigma_spin.value()
        if sigma_nm <= 0.0:
            sigma_nm = pixel_size_nm * 1.2
        depth_key = ("all",)
        if self._has_depth and not self._all_depth_check.isChecked():
            depth_key = ("slab", int(self._depth_slider.value()))
        key = (
            ch["dataset_idx"], ch["kind"], self._orientation,
            round(x0, 3), round(x1, 3), round(y0, 3), round(y1, 3),
            h, w, round(sigma_nm, 3), depth_key,
        )
        cached = self._tile_cache.get(key)
        if cached is not None:
            self._tile_cache.move_to_end(key)
            return cached, int(np.count_nonzero(cached))
        ds = self._state.datasets[ch["dataset_idx"]]
        if ch["kind"] == "image":
            tile = self._image_tile(ds, x0, x1, y0, y1, h, w)
            count = int(np.count_nonzero(tile))
        else:
            tile, count = self._localization_tile(ds, x0, x1, y0, y1, h, w, sigma_nm, pixel_size_nm)
        self._tile_cache[key] = tile
        if len(self._tile_cache) > _CACHE_LIMIT:
            self._tile_cache.popitem(last=False)
        return tile, count

    def _localization_tile(self, ds, x0, x1, y0, y1, h, w, sigma_nm, pixel_size_nm) -> tuple[np.ndarray, int]:
        locs = self._oriented_locs(ds)
        if locs.shape[0] == 0:
            return np.zeros((h, w), dtype=np.float32), 0
        xy = locs[:, :2]
        depth = locs[:, 2]
        x, y = xy[:, 0], xy[:, 1]
        in_view = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if self._has_depth and not self._all_depth_check.isChecked():
            d0_all, d1_all = self._bounds_depth
            idx = self._depth_slider.value()
            d_lo = d0_all + (d1_all - d0_all) * idx / _Z_SLAB_COUNT
            d_hi = d0_all + (d1_all - d0_all) * (idx + 1) / _Z_SLAB_COUNT
            in_view &= (depth >= d_lo) & (depth < d_hi)
            self._depth_label.setText(f"slab [{d_lo:.0f}, {d_hi:.0f}] nm")
        elif self._has_depth:
            self._depth_label.setText(f"all [{self._bounds_depth[0]:.0f}, {self._bounds_depth[1]:.0f}] nm")
        else:
            self._depth_label.setText("—")
        xv, yv = x[in_view], y[in_view]
        if xv.size == 0:
            return np.zeros((h, w), dtype=np.float32), 0
        hist, _, _ = np.histogram2d(yv, xv, bins=[h, w], range=[[y0, y1], [x0, x1]])
        sigma_px = max(sigma_nm / pixel_size_nm, 0.5)
        if sigma_px < 30.0:
            hist = gaussian_filter(hist, sigma=sigma_px, mode="constant")
        return hist.astype(np.float32, copy=False), int(xv.size)

    def _image_tile(self, ds, x0, x1, y0, y1, h, w) -> np.ndarray:
        image = self._prepare_image_payload(ds.image_data)
        ox, oy = ds.image_origin_nm
        sx, sy = ds.image_pixel_size_nm
        ix0 = int(np.floor((x0 - ox) / sx))
        ix1 = int(np.ceil((x1 - ox) / sx))
        iy0 = int(np.floor((y0 - oy) / sy))
        iy1 = int(np.ceil((y1 - oy) / sy))
        src = np.zeros((max(iy1 - iy0, 1), max(ix1 - ix0, 1)), dtype=np.float32)
        x0c, x1c = max(ix0, 0), min(ix1, image.shape[1])
        y0c, y1c = max(iy0, 0), min(iy1, image.shape[0])
        if x1c > x0c and y1c > y0c:
            src_y0, src_y1 = y0c - iy0, y1c - iy0
            src_x0, src_x1 = x0c - ix0, x1c - ix0
            src[src_y0:src_y1, src_x0:src_x1] = image[y0c:y1c, x0c:x1c]
        zy = h / max(src.shape[0], 1)
        zx = w / max(src.shape[1], 1)
        return zoom(src, (zy, zx), order=1).astype(np.float32, copy=False)[:h, :w]

    # ------------------------------------------------------------------
    # Colormap resolution with graceful fallback
    # ------------------------------------------------------------------

    def _pg_colormap_from_matplotlib(self, name: str, n: int = 256) -> pg.ColorMap:
        import matplotlib as mpl
        mpl_cmap = mpl.colormaps[name].resampled(n)
        rgba = mpl_cmap(np.linspace(0.0, 1.0, n), bytes=True)
        pos  = np.linspace(0.0, 1.0, n)
        return pg.ColorMap(pos, rgba)

    def _on_cmap_changed(self, name: str) -> None:
        if self._channels:
            target = next((i for i, ch in enumerate(self._channels) if ch["visible"]), 0)
            self._channels[target]["lut"] = name
            if 0 <= target < len(self._channel_rows):
                combo = self._channel_rows[target][1]
                if combo.findText(name) >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentText(name)
                    combo.blockSignals(False)
            self._compose_from_cache()
            return

        # 1. pyqtgraph built-ins
        try:
            cmap = pg.colormap.get(name)
            if cmap is not None:
                self._image_view.setColorMap(cmap)
                return
        except Exception:
            pass

        # 2. Matplotlib (covers 'hot', 'gray', and everything else)
        try:
            cmap = self._pg_colormap_from_matplotlib(name)
            self._image_view.setColorMap(cmap)
            return
        except Exception as e:
            print(f"Failed to load matplotlib colormap '{name}': {e!r}")

        # 3. colorcet
        try:
            cmap = pg.colormap.get(name, source="colorcet")
            if cmap is not None:
                self._image_view.setColorMap(cmap)
                return
        except Exception:
            pass

        # 4. Final fallback — CET-L3 (perceptually uniform)
        try:
            self._image_view.setColorMap(pg.colormap.get("CET-L3"))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Brightness & Contrast
    # ------------------------------------------------------------------

    def _compute_auto_levels(self, hist: np.ndarray) -> tuple[float, float] | None:
        """Fiji-style auto: lo=0, hi=99.65th percentile of non-zero bins.

        Returns None when there are fewer than 10 non-zero bins (empty view).
        """
        nonzero = hist.ravel()
        nonzero = nonzero[nonzero > 0]
        if nonzero.size < 10:
            return None
        lo = 0.0
        hi = float(np.percentile(nonzero, 99.65))
        if hi <= lo:
            hi = lo + 1.0
        return (lo, hi)

    def _show_brightness_contrast(self) -> None:
        if self._bc_dialog is None:
            from .brightness_contrast_dialog import BrightnessContrastDialog
            self._bc_dialog = BrightnessContrastDialog(
                on_levels_changed=self._on_levels_changed,
                on_auto=self._on_bc_auto,
                on_reset=self._on_bc_reset,
                parent=self,
            )
            img = self._image_view.imageItem.image
            if img is not None:
                pixels = self._bc_pixels()
                self._bc_dialog.set_data_range(float(pixels.min()), float(pixels.max()))
                if self._manual_levels is not None:
                    self._bc_dialog.set_levels(*self._manual_levels)
            self._bc_dialog.set_auto_state(self._auto_bc)

        self._bc_dialog.show()
        self._bc_dialog.raise_()
        self._bc_dialog.activateWindow()

    def _on_levels_changed(self, lo: float, hi: float) -> None:
        # Manual slider adjustment: disable dynamic auto
        if self._auto_bc:
            self._auto_bc = False
            if self._bc_dialog is not None:
                self._bc_dialog.set_auto_state(False)
        self._manual_levels = (lo, hi)
        self._image_view.setLevels(lo, hi)
        target = next((i for i, ch in enumerate(self._channels) if ch["visible"]), 0)
        if 0 <= target < len(self._channels):
            self._channels[target]["levels"] = (lo, hi)
            self._compose_from_cache()

    def _on_bc_auto(self, enabled: bool) -> None:
        self._auto_bc = enabled
        if enabled:
            img = self._bc_pixels()
            if img is not None:
                levels = self._compute_auto_levels(img)
                if levels is not None:
                    self._manual_levels = levels
                    for ch in self._channels:
                        if ch["visible"]:
                            ch["levels"] = levels
                            break
                    self._compose_from_cache()
                    if self._bc_dialog is not None:
                        self._bc_dialog.set_levels(*levels)

    def _on_bc_reset(self) -> None:
        self._manual_levels = None
        self._auto_bc = True
        for ch in self._channels:
            ch["levels"] = None
        if self._bc_dialog is not None:
            self._bc_dialog.set_auto_state(True)
        self._schedule_render()

    # ------------------------------------------------------------------
    # LUT dialog (Fiji-style B&C + colormap)
    # ------------------------------------------------------------------

    def open_lut_dialog(self) -> None:
        """Open the rich LUT editor for this render window."""
        from .lut_dialog import LutDialog, make_colormap

        # Lazily build, then refresh state on every open
        if getattr(self, "_lut_dialog", None) is None:
            self._lut_dialog = LutDialog(
                on_levels_changed=self._on_levels_changed,
                on_cmap_changed=self._on_lut_cmap_changed,
                on_invert_changed=self._on_lut_invert_changed,
                on_reset=self._on_bc_reset,
                parent=self,
            )

        img = self._bc_pixels()
        if img is None:
            self._lut_dialog.show(); self._lut_dialog.raise_()
            return

        data_lo = float(img.min())
        data_hi = float(img.max() if img.max() > img.min() else img.min() + 1.0)
        if self._manual_levels is not None:
            lo, hi = self._manual_levels
        else:
            lo, hi = data_lo, data_hi

        self._lut_invert = bool(getattr(self, "_lut_invert", False))
        self._lut_dialog.load_image(
            pixels=img, data_lo=data_lo, data_hi=data_hi,
            lo=float(lo), hi=float(hi),
            cmap_name=self._active_channel_lut(),
            invert=self._lut_invert,
        )
        self._lut_dialog.show()
        self._lut_dialog.raise_()
        self._lut_dialog.activateWindow()

    def _on_lut_cmap_changed(self, name: str, invert: bool) -> None:
        """Apply the LUT-dialog's colormap choice to the image."""
        from .lut_dialog import make_colormap
        self._lut_invert = invert
        # Update the toolbar combo if the name is one it knows; suppress
        # signals so we don't recurse into _on_cmap_changed.
        idx = self._cmap_combo.findText(name)
        if idx >= 0:
            self._cmap_combo.blockSignals(True)
            self._cmap_combo.setCurrentIndex(idx)
            self._cmap_combo.blockSignals(False)
        if self._channels:
            target = next((i for i, ch in enumerate(self._channels) if ch["visible"]), 0)
            self._channels[target]["lut"] = name
            if 0 <= target < len(self._channel_rows) and self._channel_rows[target][1].findText(name) >= 0:
                combo = self._channel_rows[target][1]
                combo.blockSignals(True)
                combo.setCurrentText(name)
                combo.blockSignals(False)
            self._compose_from_cache()
            return
        try:
            self._image_view.setColorMap(make_colormap(name, invert=invert))
        except Exception as exc:
            print(f"LUT cmap change failed: {exc}")

    def _on_lut_invert_changed(self, invert: bool) -> None:
        self._lut_invert = invert
        self._on_lut_cmap_changed(self._active_channel_lut(), invert)

    def _active_channel_lut(self) -> str:
        for ch in self._channels:
            if ch["visible"]:
                return str(ch["lut"])
        return self._cmap_combo.currentText()

    # ------------------------------------------------------------------
    # View interaction
    # ------------------------------------------------------------------

    def _on_range_changed(self, *_args) -> None:
        self._schedule_render()

    def _set_axes_visible(self, visible: bool) -> None:
        plot_item = self._image_view.view
        for axis_name in ("left", "bottom"):
            plot_item.showAxis(axis_name, show=visible)

    def _on_orientation_changed(self, text: str) -> None:
        if text not in _ORIENTATIONS:
            return
        self._orientation = text
        self._apply_orientation()
        self._schedule_render()

    def _on_all_depth_toggled(self, checked: bool) -> None:
        self._depth_slider.setEnabled(self._has_depth and not checked)
        self._schedule_render()

    def _on_depth_changed(self, _value: int) -> None:
        self._schedule_render()

    def _wheel_zoom(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        factor = 0.85 if delta > 0 else 1.0 / 0.85
        scene_pos = self._image_view.view.mapSceneToView(event.position())
        (x0, x1), (y0, y1) = self._view_box.viewRange()
        cx, cy = scene_pos.x(), scene_pos.y()
        new_w = (x1 - x0) * factor
        new_h = (y1 - y0) * factor
        fx = (cx - x0) / (x1 - x0) if x1 > x0 else 0.5
        fy = (cy - y0) / (y1 - y0) if y1 > y0 else 0.5
        self._view_box.setRange(
            xRange=(cx - fx * new_w, cx - fx * new_w + new_w),
            yRange=(cy - fy * new_h, cy - fy * new_h + new_h),
            padding=0, update=True,
        )
        event.accept()

    # ------------------------------------------------------------------
    # Fiji-style active-dataset-follows-focus (requirement #2)
    # ------------------------------------------------------------------

    def focusInEvent(self, event) -> None:
        if self._idx is not None and 0 <= self._idx < len(self._state.datasets):
            self._state.set_active(self._idx)
        if self._roi_overlay is not None:
            self._roi_overlay.activate()
        self._adopt_visible_lut_dialog()
        super().focusInEvent(event)

    def changeEvent(self, event) -> None:
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            if self._idx is not None and 0 <= self._idx < len(self._state.datasets):
                self._state.set_active(self._idx)
            if self._roi_overlay is not None:
                self._roi_overlay.activate()
            self._adopt_visible_lut_dialog()
        super().changeEvent(event)

    def _adopt_visible_lut_dialog(self) -> None:
        """If the floating LUT dialog is open, retarget it to this render window."""
        app = QApplication.instance()
        if app is None:
            return
        for widget in app.topLevelWidgets():
            if widget is self:
                continue
            if widget.__class__.__name__ == "LutDialog" and widget.isVisible():
                if getattr(self, "_lut_dialog", None) is widget:
                    self.open_lut_dialog()
                else:
                    widget.close()
                    self.open_lut_dialog()
                return

    # ------------------------------------------------------------------
    # State signals
    # ------------------------------------------------------------------

    def _on_active_changed(self, idx: int) -> None:
        # Each render window is pinned to one dataset; other views follow active.
        pass

    def _on_filter_changed(self, idx: int) -> None:
        if any(ch["dataset_idx"] == idx for ch in self._channels):
            self._refresh_from_dataset()
