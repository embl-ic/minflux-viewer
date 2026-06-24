"""
minflux_viewer.ui.render_window
================================
Fast interactive render window — pyramid / lazy-load edition.

Displays a localization dataset as a rendered 2-D image using a three-layer
pipeline:

1. **SpatialGrid** — O(k) bounding-box queries replace the old O(N) scan.
2. **PhysicalTileCache** — fixed-tile LRU cache keyed by (dataset, mask_version,
   orientation, LOD, tile_row, tile_col).  Tiles survive pan/zoom and are
   naturally invalidated when the mask changes (mask_version in key).
3. **RenderScheduler** — QThreadPool workers with a generation counter so stale
   results are silently discarded.  Tiles are composited progressively as they
   arrive; coarser-LOD placeholders fill blank regions immediately.

LOD levels (nm/pixel → behaviour):
    LOD 0 >= 100 nm/px :  50 × 50 px histogram tile
    LOD 1  20–100 nm/px: 100 × 100 px histogram tile
    LOD 2   5–20 nm/px : 256 × 256 px histogram tile
    LOD 3   1–5  nm/px : 512 × 512 px per-loc Gaussian tile
    LOD 4   < 1  nm/px :1024 ×1024 px per-loc Gaussian tile

Interactions (unchanged from pre-pyramid version)
--------------------------------------------------
* **Drag** the image to pan.
* **Scroll** on the image to zoom in/out around the cursor.
* **Orientation dropdown** — switch between XY (default), XZ, YZ.
* **Z slider** (3-D data) — step through Z slabs.
* **Sigma** spin box to override the automatic blur.
* **Reset view** resets orientation (XY), zoom, B&C, and Z to centre.
* Focus on the window to make its dataset the active one (Fiji-style).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QSignalBlocker, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import affine_transform, gaussian_filter, zoom

from .. import resource_path
from ..core.app_state import AppState
from ..core.overlay import (
    apply_display_transform_nm,
    dataset_group_id,
    identity_matrix4,
    matrix4_to_xy3,
    transform_key,
)
from ..core.roi_selection import active_roi_mask, rectangle_mask, roi_region_mask
from ..core.spatial_grid import SpatialGrid
from .render_config import (
    DIRECT_RENDER_THRESHOLD_NM,
    PER_LOC_SWITCH_COUNT,
    PHYSICAL_TILE_NM,
    actual_pixel_size_nm,
    lod_for_pixel_size,
    render_tile_px,
)
from .render_scheduler import RenderScheduler
from .tile_cache import PhysicalTileCache, TileKey

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RENDER_SIZE   = 800     # fallback target image dimension for image-mode render
_DEBOUNCE_MS   = 25      # delay between view change and re-render
_COLORMAPS     = ["hot", "inferno", "viridis", "magma", "plasma", "cividis", "gray"]
_ORIENTATIONS  = ["XY", "XZ", "YZ", "3D"]
_RENDER_ORIENTATIONS = {"XY", "XZ", "YZ"}
_PURE_COLOR_LUTS = ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow"]
_CHANNEL_LUTS  = ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Gray", *_COLORMAPS]
_IMAGEJ_AUTO_THRESHOLD = 5000
_IMAGEJ_AUTO_RESET_THRESHOLD = 10
_IMAGEJ_AUTO_HIST_BINS = 256
_CHANNEL_COLORS = {
    "Red": (1.0, 0.0, 0.0),
    "Green": (0.0, 1.0, 0.0),
    "Blue": (0.0, 0.0, 1.0),
    "Cyan": (0.0, 1.0, 1.0),
    "Magenta": (1.0, 0.0, 1.0),
    "Yellow": (1.0, 1.0, 0.0),
    "Gray": (1.0, 1.0, 1.0),
}


class DepthRangeSlider(QWidget):
    """Small horizontal floating-point range slider for depth gating."""

    rangeChanged = pyqtSignal(float, float)
    doubleClicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._min_value = 0.0
        self._max_value = 1.0
        self._lo = 0.0
        self._hi = 1.0
        self._drag_handle: str | None = None
        self._drag_start_x = 0.0
        self._drag_start_range: tuple[float, float] = (0.0, 1.0)
        self._handle_radius = 6
        self._wheel_step_nm = 1.0
        self._reverse_wheel = False
        self.setMinimumHeight(24)
        self.setMinimumWidth(160)
        self.setMouseTracking(True)
        self.setToolTip("Drag range edges; wheel shifts the range; double-click to type values")

    def set_limits(self, lo: float, hi: float, *, reset_range: bool = True) -> None:
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            hi = lo + 1.0
        self._min_value = lo
        self._max_value = hi
        if reset_range:
            self._lo, self._hi = lo, hi
        else:
            self._lo = min(max(self._lo, lo), hi)
            self._hi = min(max(self._hi, lo), hi)
            if self._hi < self._lo:
                self._lo, self._hi = self._hi, self._lo
        self.update()

    def set_range(self, lo: float, hi: float, *, emit: bool = False) -> None:
        lo = min(max(float(lo), self._min_value), self._max_value)
        hi = min(max(float(hi), self._min_value), self._max_value)
        if hi < lo:
            lo, hi = hi, lo
        changed = not np.isclose([self._lo, self._hi], [lo, hi]).all()
        self._lo, self._hi = lo, hi
        self.update()
        if emit and changed:
            self.rangeChanged.emit(self._lo, self._hi)

    def range(self) -> tuple[float, float]:
        return self._lo, self._hi

    def set_scroll_options(self, step_nm: float, reverse: bool) -> None:
        self._wheel_step_nm = max(float(step_nm), 0.0)
        self._reverse_wheel = bool(reverse)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        y = self.height() // 2
        x0, x1 = self._track_bounds()
        enabled = self.isEnabled()

        base_color = QColor(170, 170, 170) if enabled else QColor(205, 205, 205)
        active_color = QColor(0, 120, 215) if enabled else QColor(165, 165, 165)
        handle_color = QColor("white") if enabled else QColor(235, 235, 235)
        outline_color = QColor(70, 70, 70) if enabled else QColor(170, 170, 170)

        painter.setPen(QPen(base_color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(x0, y, x1, y)
        lo_x = self._value_to_pos(self._lo)
        hi_x = self._value_to_pos(self._hi)
        painter.setPen(QPen(active_color, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(lo_x, y, hi_x, y)
        painter.setPen(QPen(outline_color, 1))
        painter.setBrush(handle_color)
        painter.drawEllipse(lo_x - self._handle_radius, y - self._handle_radius, self._handle_radius * 2, self._handle_radius * 2)
        painter.drawEllipse(hi_x - self._handle_radius, y - self._handle_radius, self._handle_radius * 2, self._handle_radius * 2)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self.isEnabled():
            super().mousePressEvent(event)
            return
        x = float(event.position().x())
        lo_x = self._value_to_pos(self._lo)
        hi_x = self._value_to_pos(self._hi)
        handle_hit_radius = self._handle_radius + 3
        if abs(x - lo_x) <= handle_hit_radius:
            self._drag_handle = "lo"
        elif abs(x - hi_x) <= handle_hit_radius:
            self._drag_handle = "hi"
        elif min(lo_x, hi_x) < x < max(lo_x, hi_x):
            self._drag_handle = "range"
            self._drag_start_x = x
            self._drag_start_range = (self._lo, self._hi)
        else:
            self._drag_handle = "lo" if abs(x - lo_x) <= abs(x - hi_x) else "hi"
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_handle is None or not self.isEnabled():
            super().mouseMoveEvent(event)
            return
        x = float(event.position().x())
        if self._drag_handle == "range":
            delta = self._pos_to_value(x) - self._pos_to_value(self._drag_start_x)
            self._shift_range(delta, base_range=self._drag_start_range)
        else:
            self._set_drag_value(x)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_handle = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self.doubleClicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:
        if not self.isEnabled():
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        steps = max(1, int(round(abs(delta) / 120.0)))
        direction = 1.0 if delta > 0 else -1.0
        if self._reverse_wheel:
            direction *= -1.0
        self._shift_range(direction * steps * self._wheel_step_nm)
        event.accept()

    def _shift_range(
        self,
        delta: float,
        *,
        base_range: tuple[float, float] | None = None,
    ) -> None:
        base_lo, base_hi = base_range if base_range is not None else (self._lo, self._hi)
        width = base_hi - base_lo
        lo = base_lo + float(delta)
        hi = base_hi + float(delta)
        if lo < self._min_value:
            lo = self._min_value
            hi = lo + width
        if hi > self._max_value:
            hi = self._max_value
            lo = hi - width
        self.set_range(lo, hi, emit=True)

    def _set_drag_value(self, x: float) -> None:
        value = self._pos_to_value(x)
        if self._drag_handle == "lo":
            self.set_range(value, self._hi, emit=True)
        elif self._drag_handle == "hi":
            self.set_range(self._lo, value, emit=True)

    def _track_bounds(self) -> tuple[int, int]:
        margin = self._handle_radius + 2
        return margin, max(margin, self.width() - margin)

    def _value_to_pos(self, value: float) -> int:
        x0, x1 = self._track_bounds()
        span = self._max_value - self._min_value
        frac = 0.0 if span <= 0 else (float(value) - self._min_value) / span
        return int(round(x0 + np.clip(frac, 0.0, 1.0) * (x1 - x0)))

    def _pos_to_value(self, x: float) -> float:
        x0, x1 = self._track_bounds()
        frac = np.clip((float(x) - x0) / max(x1 - x0, 1), 0.0, 1.0)
        return self._min_value + frac * (self._max_value - self._min_value)


class DepthRangeDialog(QDialog):
    """Manual editor for the active depth range."""

    def __init__(
        self,
        axis_name: str,
        limits: tuple[float, float],
        current: tuple[float, float],
        inclusive: tuple[bool, bool],
        scroll_step_nm: float,
        reverse_scroll: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{axis_name} range view option")
        self.setModal(True)

        limit_lo, limit_hi = limits
        self._limit_lo = float(limit_lo)
        self._limit_hi = float(limit_hi)
        self._last_bound_edited = "lo"
        self._syncing = False

        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        self._lo_spin = QDoubleSpinBox()
        self._lo_spin.setDecimals(3)
        self._lo_spin.setRange(limit_lo, limit_hi)
        self._lo_spin.setValue(float(current[0]))
        self._lo_spin.setSuffix(" nm")
        min_row = QHBoxLayout()
        min_row.addWidget(self._lo_spin)
        min_row.addWidget(QLabel(f"dataset min: {limit_lo:.1f} nm"))
        form.addRow("min:", min_row)

        self._lo_inclusive = QCheckBox("left inclusive")
        self._lo_inclusive.setChecked(bool(inclusive[0]))
        self._range_spin = QDoubleSpinBox()
        self._range_spin.setDecimals(3)
        self._range_spin.setRange(0.0, max(float(limit_hi) - float(limit_lo), 0.0))
        self._range_spin.setValue(max(float(current[1]) - float(current[0]), 0.0))
        self._range_spin.setSuffix(" nm")
        self._hi_inclusive = QCheckBox("right inclusive")
        self._hi_inclusive.setChecked(bool(inclusive[1]))
        range_row = QHBoxLayout()
        range_row.addWidget(self._lo_inclusive)
        range_row.addWidget(self._range_spin)
        range_row.addWidget(self._hi_inclusive)
        form.addRow("range:", range_row)

        self._hi_spin = QDoubleSpinBox()
        self._hi_spin.setDecimals(3)
        self._hi_spin.setRange(limit_lo, limit_hi)
        self._hi_spin.setValue(float(current[1]))
        self._hi_spin.setSuffix(" nm")
        max_row = QHBoxLayout()
        max_row.addWidget(self._hi_spin)
        max_row.addWidget(QLabel(f"dataset max: {limit_hi:.1f} nm"))
        form.addRow("max:", max_row)

        self._scroll_step_spin = QDoubleSpinBox()
        self._scroll_step_spin.setDecimals(3)
        self._scroll_step_spin.setRange(0.001, max(float(limit_hi) - float(limit_lo), 0.001))
        self._scroll_step_spin.setValue(max(float(scroll_step_nm), 0.001))
        self._scroll_step_spin.setSuffix(" nm")
        self._reverse_scroll = QCheckBox("reverse mouse scroll direction")
        self._reverse_scroll.setChecked(bool(reverse_scroll))
        scroll_row = QHBoxLayout()
        scroll_row.addWidget(self._scroll_step_spin)
        scroll_row.addWidget(self._reverse_scroll)
        form.addRow("mouse scroll =", scroll_row)

        self._lo_spin.valueChanged.connect(self._on_lo_changed)
        self._hi_spin.valueChanged.connect(self._on_hi_changed)
        self._range_spin.valueChanged.connect(self._on_range_changed)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def values(self) -> tuple[tuple[float, float], tuple[bool, bool]]:
        lo = float(self._lo_spin.value())
        hi = float(self._hi_spin.value())
        if hi < lo:
            lo, hi = hi, lo
        return (lo, hi), (self._lo_inclusive.isChecked(), self._hi_inclusive.isChecked())

    def scroll_options(self) -> tuple[float, bool]:
        return float(self._scroll_step_spin.value()), self._reverse_scroll.isChecked()

    def _on_lo_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._last_bound_edited = "lo"
        lo = float(value)
        hi = max(float(self._hi_spin.value()), lo)
        self._set_values(lo, hi)

    def _on_hi_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._last_bound_edited = "hi"
        hi = float(value)
        lo = min(float(self._lo_spin.value()), hi)
        self._set_values(lo, hi)

    def _on_range_changed(self, value: float) -> None:
        if self._syncing:
            return
        requested = max(float(value), 0.0)
        lo = float(self._lo_spin.value())
        hi = float(self._hi_spin.value())
        if self._last_bound_edited == "hi":
            lo = hi - requested
            if lo < self._limit_lo:
                lo = self._limit_lo
        else:
            hi = lo + requested
            if hi > self._limit_hi:
                hi = self._limit_hi
        self._set_values(lo, hi)

    def _set_values(self, lo: float, hi: float) -> None:
        lo = min(max(float(lo), self._limit_lo), self._limit_hi)
        hi = min(max(float(hi), self._limit_lo), self._limit_hi)
        if hi < lo:
            hi = lo
        self._syncing = True
        try:
            blockers = (
                QSignalBlocker(self._lo_spin),
                QSignalBlocker(self._range_spin),
                QSignalBlocker(self._hi_spin),
            )
            self._lo_spin.setValue(lo)
            self._hi_spin.setValue(hi)
            self._range_spin.setValue(max(hi - lo, 0.0))
            del blockers
        finally:
            self._syncing = False


class SigmaDialog(QDialog):
    """Small modal dialog for render-convolution sigma in physical units."""

    def __init__(self, values_xyz: tuple[float, float, float], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Render Sigma")
        self.setModal(True)
        self.resize(260, 120)

        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        self._spins: list[QDoubleSpinBox] = []
        for label, value in zip(("X sigma", "Y sigma", "Z sigma"), values_xyz):
            spin = QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(0.0, 10000.0)
            spin.setSingleStep(1.0)
            spin.setValue(float(value))
            spin.setSpecialValueText("auto")
            spin.setSuffix(" nm")
            form.addRow(label, spin)
            self._spins.append(spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def values_xyz(self) -> tuple[float, float, float]:
        return tuple(float(spin.value()) for spin in self._spins)


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

    def accept(self) -> None:
        self._render_window._apply_manual_channel_transform(self._ch_idx)
        super().accept()

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
        axis0, axis1 = self._render_window._orientation_axes(self._render_window._orientation)
        axis_names = "XYZ"
        self._status.setText(
            f"Current transform: d{axis_names[axis0]}={dx_px:.3f} px, "
            f"d{axis_names[axis1]}={dy_px:.3f} px, rotation={transform['angle']:.3f} deg"
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
    Fast interactive 2-D render window with pyramid / lazy-load pipeline.

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

        self._locs_nm: np.ndarray | None = None  # (N, 3) filtered, for depth setup
        self._xy: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._orientation: str = "XY"
        self._has_depth: bool = False
        self._depth_axis_name: str = "Z"
        self._bounds_xy: tuple[float, float, float, float] = (0, 1, 0, 1)
        self._fit_view_size: tuple[float, float] = (1.0, 1.0)
        self._suppress_zoom_limit: bool = False
        self._bounds_depth: tuple[float, float] = (0, 0)
        self._depth_range: tuple[float, float] = (0, 0)
        self._depth_inclusive: tuple[bool, bool] = (True, True)
        self._depth_range_initialized: bool = False
        self._depth_scroll_step_nm: float = 1.0
        self._depth_reverse_scroll: bool = False
        self._render_mode: str = "localizations"
        self._image_data: np.ndarray | None = None
        self._dataset_dim_label: str = "2D"
        self._active_cmap: str = "hot"
        self._axis_visible: bool = False
        self._sigma_nm_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._channels: list[dict] = []
        self._channel_rows: list[tuple[QLabel, QComboBox]] = []
        self._export_workers: list = []  # live TIFF-export QThreads (kept from GC)

        # Pyramid render pipeline state
        self._phys_tile_cache: PhysicalTileCache = PhysicalTileCache()
        self._scheduler: RenderScheduler = RenderScheduler(parent=self)
        self._channel_grids: dict[int, SpatialGrid | None] = {}
        self._channel_locs_xyz: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._mask_versions: dict[int, int] = {}
        self._tile_grid_x0: float = 0.0
        self._tile_grid_y0: float = 0.0
        self._pending_tile_keys: set[TileKey] = set()
        self._last_lod: int = 0
        self._last_px_nm: float = 10.0

        self._last_scalar_tile: np.ndarray | None = None
        self._last_tile_geometry: tuple[float, float, float, float] | None = None

        # Manual levels override (set by B&C). None → pyqtgraph autoLevels.
        self._manual_levels: tuple[float, float] | None = None
        self._auto_bc: bool = True
        self._bc_auto_threshold: int = 0
        self._bc_dialog = None
        self._roi_overlay = None
        self._roi_highlight_item = None
        self._volume_window = None

        self.setWindowTitle("Render")
        self.setWindowIcon(QIcon(str(resource_path("icons", "minflux_viewer_logo.png"))))
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(880, 920)
        # Keep pyqtgraph ImageView/ViewBox objects alive after close. On
        # Windows, deleting them while more render windows are being created can
        # crash inside pyqtgraph's ViewBox cleanup path.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(_DEBOUNCE_MS)
        self._redraw_timer.timeout.connect(self._render)

        self._build_ui()
        self._info_shortcut = QShortcut(QKeySequence("I"), self)
        self._info_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._info_shortcut.activated.connect(self._show_data_info_window)
        self._refresh_from_dataset()

        state.active_changed.connect(self._on_active_changed)
        state.filter_changed.connect(self._on_filter_changed)
        # RIMF / z-scaling change: re-pull loc_nm and re-render (same as a
        # filter change — it busts tile caches and refreshes depth).
        state.calibration_changed.connect(self._on_filter_changed)
        state.roi_selection_changed.connect(self._on_roi_selection_changed)
        state.rois.selection_changed.connect(self._redraw_roi_highlight)
        self._scheduler.tile_ready.connect(self._on_tile_ready)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")
        self._image_view = pg.ImageView(view=pg.PlotItem(enableMenu=False))
        self._image_view.ui.histogram.hide()
        self._image_view.ui.roiBtn.hide()
        self._image_view.ui.menuBtn.hide()
        self._image_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        try:
            self._image_view.view.hideButtons()
            self._image_view.view.autoBtn.hide()
            self._image_view.view.setMenuEnabled(False)
        except Exception:
            pass
        self._view_box = self._image_view.view.vb
        try:
            self._view_box.setMenuEnabled(False)
        except Exception:
            pass
        self._view_box.setAspectLocked(True)
        self._view_box.invertY(False)
        self._view_box.sigRangeChanged.connect(self._on_range_changed)
        self._image_view.ui.graphicsView.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._image_view.ui.graphicsView.customContextMenuRequested.connect(
            self._show_context_menu
        )
        self._set_axes_visible(False)

        root.addWidget(self._image_view, stretch=1)
        self._roi_highlight_item = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen(255, 210, 0, 235, width=1.6),
            brush=pg.mkBrush(255, 230, 0, 65),
        )
        self._image_view.view.addItem(self._roi_highlight_item)
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

        self._depth_slider = DepthRangeSlider()
        self._depth_slider.setEnabled(False)
        self._depth_slider.rangeChanged.connect(self._on_depth_range_changed)
        self._depth_slider.doubleClicked.connect(self._show_depth_range_dialog)
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

        self.setWindowTitle(ds.name)
        self._dataset_dim_label = f"{ds.prop.num_dim}D"
        self._build_channels()
        self._rebuild_channel_ui()
        self._update_overlay_title()
        self._scheduler.cancel()

        if ds.image_data is not None and not ds.has_localizations:
            self._render_mode = "image"
            self._locs_nm = self._xy = self._depth = None
            self._configure_image_depth(ds.image_data)
            self._image_data = self._prepare_image_payload(ds.image_data)
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
        self._rebuild_all_grids()
        self._schedule_render()

    def _build_channels(self) -> None:
        """Build lightweight channel descriptors; scalar tiles are computed lazily."""
        previous_transforms = {
            ch.get("dataset_idx"): dict(ch.get("transform") or {"dx": 0.0, "dy": 0.0, "angle": 0.0})
            for ch in self._channels
        }
        self._channels = []
        color_cycle = ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Gray"]
        active_group = None
        if self._idx is not None and 0 <= self._idx < len(self._state.datasets):
            ds0 = self._state.datasets[self._idx]
            active_group = dataset_group_id(ds0)
        for idx, ds in enumerate(self._state.datasets):
            if not (ds.has_localizations or ds.image_data is not None):
                continue
            same_group = active_group is not None and dataset_group_id(ds) == active_group
            if idx != self._idx and not same_group:
                continue
            visible = idx == self._idx or same_group
            # A channel may opt to start hidden (e.g. the DCR "unassigned" channel).
            if visible and ds.state.get("overlay_default_hidden"):
                visible = False
            self._channels.append({
                "dataset_idx": idx,
                "name": ds.name,
                "kind": "image" if ds.image_data is not None and not ds.has_localizations else "localizations",
                "visible": visible,
                "lut": ds.state.get("render_channel_lut") or color_cycle[len(self._channels) % len(color_cycle)],
                "levels": None,
                "loc_transform": ds.state.get("overlay_transform") or ds.state.get("render_transform_2d"),
                "transform": previous_transforms.get(idx, {"dx_nm": 0.0, "dy_nm": 0.0, "angle": 0.0}),
            })
        if len(self._channels) == 1 and not self._state.datasets[self._channels[0]["dataset_idx"]].state.get("render_channel_lut"):
            self._channels[0]["lut"] = self._active_cmap
        if not self._channels and self._idx is not None:
            ds = self._state.datasets[self._idx]
            self._channels.append({
                "dataset_idx": self._idx,
                "name": ds.name,
                "kind": "localizations",
                "visible": True,
                "lut": ds.state.get("render_channel_lut") or "Red",
                "levels": None,
                "loc_transform": ds.state.get("overlay_transform") or ds.state.get("render_transform_2d"),
                "transform": previous_transforms.get(self._idx, {"dx_nm": 0.0, "dy_nm": 0.0, "angle": 0.0}),
            })

    def _manual_align_channel(self, ch_idx: int) -> None:
        if not (0 <= ch_idx < len(self._channels)):
            return
        dialog = ManualAlignDialog(self, ch_idx)
        dialog.exec()

    def _apply_manual_channel_transform(self, ch_idx: int) -> None:
        if not (0 <= ch_idx < len(self._channels)):
            return
        ch = self._channels[ch_idx]
        ds_idx = ch.get("dataset_idx")
        if ds_idx is None or not (0 <= ds_idx < len(self._state.datasets)):
            return
        ds = self._state.datasets[ds_idx]
        manual = self._manual_transform_matrix(ch.get("transform") or {})
        base_transform = ch.get("loc_transform") or ds.state.get("overlay_transform") or ds.state.get("render_transform_2d")
        base = self._transform_matrix4(base_transform)
        matrix = manual @ base
        record = self._updated_transform_record(base_transform, matrix)
        ds.state["overlay_transform"] = record
        ds.state["render_transform_2d"] = record
        ch["loc_transform"] = record
        ch["transform"] = {"dx_nm": 0.0, "dy_nm": 0.0, "angle": 0.0}
        self._rebuild_all_grids()
        self._phys_tile_cache.clear()
        self._scheduler.cancel()
        self._schedule_render()

    def _manual_transform_matrix(self, transform: dict) -> np.ndarray:
        matrix = identity_matrix4()
        orientation = self._orientation
        axes = self._orientation_axes(orientation)
        dx_nm = float(transform.get("dx_nm", 0.0))
        dy_nm = float(transform.get("dy_nm", 0.0))
        angle = float(transform.get("angle", 0.0))

        translation = np.zeros(3, dtype=np.float64)
        translation[axes[0]] = dx_nm
        translation[axes[1]] = dy_nm

        anchor = np.zeros(3, dtype=np.float64)
        anchor[axes[0]] = float(transform.get("anchor_x_nm", 0.0))
        anchor[axes[1]] = float(transform.get("anchor_y_nm", 0.0))

        rot = identity_matrix4()
        if abs(angle) > 1e-12:
            theta = np.deg2rad(angle)
            c, s = float(np.cos(theta)), float(np.sin(theta))
            a0, a1 = axes
            rot[a0, a0] = c
            rot[a0, a1] = -s
            rot[a1, a0] = s
            rot[a1, a1] = c

        to_anchor = identity_matrix4()
        to_anchor[:3, 3] = -anchor
        from_anchor = identity_matrix4()
        from_anchor[:3, 3] = anchor + translation
        matrix = from_anchor @ rot @ to_anchor
        return matrix

    def _transform_matrix4(self, transform: dict | None) -> np.ndarray:
        if isinstance(transform, dict):
            matrix_4 = transform.get("matrix_4x4")
            if matrix_4 is not None:
                arr = np.asarray(matrix_4, dtype=np.float64)
                if arr.shape == (4, 4):
                    return arr
            matrix_3 = transform.get("matrix_3x3")
            if matrix_3 is not None:
                arr = np.asarray(matrix_3, dtype=np.float64)
                if arr.shape == (3, 3):
                    out = identity_matrix4()
                    out[:2, :2] = arr[:2, :2]
                    out[:2, 3] = arr[:2, 2]
                    return out
        return identity_matrix4()

    def _updated_transform_record(self, previous: dict | None, matrix: np.ndarray) -> dict:
        record = dict(previous or {})
        record["matrix_4x4"] = np.asarray(matrix, dtype=np.float64).tolist()
        record["matrix_3x3"] = matrix4_to_xy3(matrix).tolist()
        record["alignment_mode"] = record.get("alignment_mode") or "manual"
        provenance = dict(record.get("provenance") or {})
        provenance["manual_alignment"] = {
            "orientation": self._orientation,
            "method": "keyboard translation/rotation",
        }
        record["provenance"] = provenance
        return record

    @staticmethod
    def _orientation_axes(orientation: str) -> tuple[int, int]:
        if orientation == "XZ":
            return (0, 2)
        if orientation == "YZ":
            return (1, 2)
        return (0, 1)

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
            row.mousePressEvent = lambda event, i=ch_idx, r=row: self._on_channel_row_pressed(r, event, i)
            vis_cb = QCheckBox()
            vis_cb.setChecked(bool(ch["visible"]))
            vis_cb.toggled.connect(lambda checked, i=ch_idx: self._on_channel_visible(i, checked))
            lay.addWidget(vis_cb)
            name_lbl = QLabel(f"{ch_idx + 1}: {ch['name']}")
            lay.addWidget(name_lbl, stretch=1)
            lut = QComboBox()
            lut.addItems(_CHANNEL_LUTS)
            lut.setCurrentText(ch["lut"])
            lut.currentTextChanged.connect(lambda text, i=ch_idx: self._on_channel_lut(i, text))
            lay.addWidget(lut)
            align_btn = QPushButton("manual align")
            align_btn.clicked.connect(lambda _checked=False, i=ch_idx: self._manual_align_channel(i))
            lay.addWidget(align_btn)
            self._channel_layout.addWidget(row)
            self._channel_rows.append((name_lbl, lut))
        self._channel_area.setVisible(len(self._channels) > 1)

    def _on_channel_row_pressed(self, row: QWidget, event, ch_idx: int) -> None:
        if event.button() == Qt.MouseButton.LeftButton and 0 <= ch_idx < len(self._channels):
            ds_idx = self._channels[ch_idx].get("dataset_idx")
            if ds_idx is not None and 0 <= ds_idx < len(self._state.datasets):
                self._idx = ds_idx
                self._state.set_active(ds_idx)
                self._update_overlay_title()
        QWidget.mousePressEvent(row, event)

    def _update_overlay_title(self) -> None:
        if self._idx is None or not (0 <= self._idx < len(self._state.datasets)):
            return
        ds = self._state.datasets[self._idx]
        overlay_idx = ds.state.get("overlay_index")
        if overlay_idx and len(self._channels) > 1:
            self.setWindowTitle(f"Overlay {overlay_idx} - {ds.name}")
        else:
            self.setWindowTitle(ds.name)

    def _on_channel_visible(self, ch_idx: int, visible: bool) -> None:
        if 0 <= ch_idx < len(self._channels):
            self._channels[ch_idx]["visible"] = bool(visible)
            self._compose_from_cache()

    def _on_channel_lut(self, ch_idx: int, lut: str) -> None:
        if 0 <= ch_idx < len(self._channels):
            self._channels[ch_idx]["lut"] = lut
            try:
                ds_idx = self._channels[ch_idx]["dataset_idx"]
                self._state.datasets[ds_idx].state["render_channel_lut"] = lut
            except Exception:
                pass
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
        locs = locs[finite, :3]
        return self._apply_dataset_render_transform(ds, locs)

    def _apply_dataset_render_transform(self, ds, locs: np.ndarray) -> np.ndarray:
        transform = ds.state.get("overlay_transform") or ds.state.get("render_transform_2d")
        if not transform:
            return locs
        return apply_display_transform_nm(locs, transform)

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
        arr = np.asarray(image)
        if arr.ndim == 2:
            return arr.astype(np.float64, copy=False)
        if self._image_is_color_2d(arr):
            return arr
        if self._image_is_stack(arr):
            return self._project_image_stack(arr)
        if arr.ndim >= 3:
            axes = tuple(range(arr.ndim - 2))
            return np.nanmax(arr.astype(np.float64, copy=False), axis=axes)
        raise ValueError("Image payload must be at least 2D")

    @staticmethod
    def _image_is_color_2d(image: np.ndarray) -> bool:
        return image.ndim == 3 and image.shape[-1] in (3, 4)

    @classmethod
    def _image_is_stack(cls, image: np.ndarray) -> bool:
        return (
            (image.ndim == 3 and not cls._image_is_color_2d(image))
            or (image.ndim == 4 and image.shape[-1] in (3, 4))
        )

    @classmethod
    def _image_depth_count(cls, image: np.ndarray) -> int:
        arr = np.asarray(image)
        return int(arr.shape[0]) if cls._image_is_stack(arr) else 1

    def _project_image_stack(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        depth = self._image_depth_count(arr)
        if depth <= 1:
            return np.asarray(arr[0] if arr.ndim >= 3 else arr, dtype=np.float64)
        if self._all_depth_check.isChecked() or not self._has_depth:
            selected = arr
        else:
            lo, hi = sorted(self._depth_range)
            slice_numbers = np.arange(1, depth + 1, dtype=float)
            left_inc, right_inc = self._depth_inclusive
            lo_mask = slice_numbers >= lo if left_inc else slice_numbers > lo
            hi_mask = slice_numbers <= hi if right_inc else slice_numbers < hi
            mask = lo_mask & hi_mask
            if not np.any(mask):
                center = int(np.clip(round((lo + hi) / 2.0), 1, depth))
                mask[center - 1] = True
            selected = arr[mask]
        if selected.shape[0] == 1:
            return selected[0]
        return np.nanmax(selected.astype(np.float64, copy=False), axis=0)

    def _configure_image_depth(self, image: np.ndarray) -> None:
        depth = self._image_depth_count(image)
        if depth <= 1:
            self._has_depth = False
            self._bounds_depth = (0.0, 0.0)
            self._depth_range = (0.0, 0.0)
            self._depth_axis_name = "Z"
            self._depth_row.setVisible(False)
            self._update_depth_label()
            return
        self._has_depth = True
        self._bounds_depth = (1.0, float(depth))
        self._depth_axis_name = "Slice"
        self._depth_scroll_step_nm = 1.0
        self._depth_reverse_scroll = False
        self._depth_slider.set_limits(*self._bounds_depth, reset_range=True)
        self._depth_slider.set_scroll_options(self._depth_scroll_step_nm, self._depth_reverse_scroll)
        self._depth_row.setVisible(True)
        self._depth_axis_label.setText("Z:")
        self._depth_slider.setEnabled(not self._all_depth_check.isChecked())
        if not self._all_depth_check.isChecked():
            self._set_default_depth_range()
        else:
            self._depth_range = self._bounds_depth
            self._depth_inclusive = (True, True)
            self._depth_range_initialized = False
        self._update_depth_label()

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
        self._clear_roi_highlight()
        if self._manual_levels is not None:
            self._image_view.setLevels(*self._manual_levels)
        self._on_cmap_changed(self._active_cmap)
        if fit_view:
            self._fit_view()
        depth = self._image_depth_count(ds.image_data)
        stack_note = f"  |  {depth} slices" if depth > 1 else ""
        self._info_label.setText(
            f"{self._dataset_dim_label} image{stack_note}  |  {width} × {height} px  |  "
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
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        if x_max <= x_min:
            x_min -= 1.0
            x_max += 1.0
        if y_max <= y_min:
            y_min -= 1.0
            y_max += 1.0
        self._bounds_xy = (
            x_min, x_max,
            y_min, y_max,
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
        self._depth_axis_name = depth_name
        self._depth_range = self._bounds_depth
        self._depth_inclusive = (True, True)
        self._depth_range_initialized = False
        self._depth_scroll_step_nm = max((self._bounds_depth[1] - self._bounds_depth[0]) / 20.0, 0.001)
        self._depth_reverse_scroll = False
        self._depth_slider.set_limits(*self._bounds_depth, reset_range=True)
        self._depth_slider.set_scroll_options(self._depth_scroll_step_nm, self._depth_reverse_scroll)

        ds = self._state.datasets[self._idx]
        dataset_is_3d = ds.prop.num_dim == 3
        self._depth_row.setVisible(dataset_is_3d)
        self._depth_axis_label.setText(f"{depth_name}:")
        self._depth_slider.setEnabled(
            self._has_depth and not self._all_depth_check.isChecked()
        )
        if self._has_depth and not self._all_depth_check.isChecked():
            self._set_default_depth_range()
        self._update_depth_label()

        self._fit_view()

    def _fit_view(self) -> None:
        x0, x1, y0, y1 = self._bounds_xy
        self._set_zoom_limits()
        self._suppress_zoom_limit = True
        try:
            self._view_box.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
        finally:
            self._suppress_zoom_limit = False
        QTimer.singleShot(0, self._remember_fit_view_size)

    def _remember_fit_view_size(self) -> None:
        """Store the aspect-adjusted fitted data view as the zoom-out baseline."""
        try:
            (x0, x1), (y0, y1) = self._view_box.viewRange()
        except Exception:
            bx0, bx1, by0, by1 = self._bounds_xy
            self._fit_view_size = (max(float(bx1 - bx0), 1.0), max(float(by1 - by0), 1.0))
            return
        self._fit_view_size = (
            max(float(x1 - x0), 1.0),
            max(float(y1 - y0), 1.0),
        )

    # Maximum zoom-out as a multiple of the fitted data view. The fitted view
    # may be aspect-adjusted by PyQtGraph, so this baseline avoids clipping
    # data on wide/tall render windows.
    _ZOOM_OUT_LIMIT: float = 2.0

    def _set_zoom_limits(self) -> None:
        """Clear ViewBox hard limits — zoom-out is enforced after range changes.

        ViewBox.setLimits(maxXRange, maxYRange) causes PyQtGraph to clamp
        the range by adjusting the center without knowing where the cursor
        is, which makes the view drift sideways while zooming out.
        We instead let PyQtGraph perform its cursor-centred zoom, then clamp
        the resulting range in _on_range_changed.
        """
        try:
            self._view_box.setLimits(
                xMin=None, xMax=None,
                yMin=None, yMax=None,
                maxXRange=None, maxYRange=None,
                minXRange=None, minYRange=None,
            )
        except Exception:
            pass

    def _reset_view(self) -> None:
        """Reset orientation (XY), zoom, B&C levels, and depth to centre."""
        self._manual_levels = None
        self._auto_bc = True
        self._bc_auto_threshold = 0
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
        self._sigma_nm_xyz = (0.0, 0.0, 0.0)
        self._all_depth_check.setChecked(True)
        self._depth_range = self._bounds_depth
        self._depth_range_initialized = False
        self._depth_slider.set_range(*self._bounds_depth)
        self._update_depth_label()
        self._scheduler.cancel()
        if self._orientation != "XY":
            self._set_orientation("XY")
        else:
            self._apply_orientation()
            self._rebuild_all_grids()
            self._schedule_render()

    # ------------------------------------------------------------------
    # Rendering — pyramid pipeline
    # ------------------------------------------------------------------

    def _schedule_render(self) -> None:
        self._redraw_timer.start()

    def _render(self) -> None:
        """Timer callback: dispatch to tiled or direct render based on zoom."""
        if self._render_mode == "image":
            self._render_image_mode()
            return

        (x0, x1), (y0, y1) = self._view_box.viewRange()
        if x1 <= x0 or y1 <= y0:
            return

        view_w, view_h = x1 - x0, y1 - y0
        viewport_px_nm = max(view_w, view_h) / _RENDER_SIZE

        if viewport_px_nm < DIRECT_RENDER_THRESHOLD_NM:
            self._render_direct(x0, x1, y0, y1, viewport_px_nm)
            return

        lod = lod_for_pixel_size(viewport_px_nm)
        px_nm = actual_pixel_size_nm(lod)

        self._last_lod = lod
        self._last_px_nm = px_nm

        # Increment generation — workers from before this call are stale.
        self._scheduler.new_generation()
        self._pending_tile_keys = set()

        sigma_yx_nm = self._sigma_yx_for_orientation(px_nm)
        depth_range_active: tuple[float, float] | None = None
        if self._has_depth and not self._all_depth_check.isChecked():
            depth_range_active = self._depth_range

        tiles: list[np.ndarray] = []
        all_missing: list[TileKey] = []

        for ch in self._channels:
            if ch["kind"] == "image":
                ds = self._state.datasets[ch["dataset_idx"]]
                n_bins_x = max(int(round(view_w / px_nm)), 8)
                n_bins_y = max(int(round(view_h / px_nm)), 8)
                canvas = self._image_tile(ds, x0, x1, y0, y1, n_bins_y, n_bins_x)
                tiles.append(canvas)
            else:
                ds_idx = ch["dataset_idx"]
                mask_ver = self._mask_versions.get(ds_idx, 0)
                tr_key = self._channel_loc_transform_key(ch)
                depth_key = (round(depth_range_active[0], 1), round(depth_range_active[1], 1)) if depth_range_active else None
                canvas, missing = self._composite_loc_channel(
                    ds_idx, mask_ver, self._orientation, tr_key, depth_key,
                    x0, x1, y0, y1, lod, px_nm,
                )
                tiles.append(canvas)
                all_missing.extend(missing)

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
                for ch in self._channels:
                    if ch["visible"]:
                        ch["levels"] = None
                        break
                if self._bc_dialog is not None and self._bc_dialog.isVisible():
                    self._bc_dialog.set_levels(*levels)

        rgba = self._compose_rgba(scalar)
        self._image_view.setImage(
            rgba, autoRange=False, autoLevels=False,
            pos=[x0, y0], scale=[px_nm, px_nm],
        )
        self._redraw_roi_highlight()

        self._pending_tile_keys = set(all_missing)

        # Submit missing tile jobs to the worker pool
        if all_missing:
            for ch in self._channels:
                if ch["kind"] != "localizations":
                    continue
                ds_idx = ch["dataset_idx"]
                mask_ver = self._mask_versions.get(ds_idx, 0)
                tr_key = self._channel_loc_transform_key(ch)
                depth_key = (round(depth_range_active[0], 1), round(depth_range_active[1], 1)) if depth_range_active else None
                ch_missing = [
                    k for k in all_missing
                    if k.dataset_id == ds_idx
                    and k.mask_version == mask_ver
                    and k.transform_key == tr_key
                    and k.depth_range == depth_key
                ]
                if not ch_missing:
                    continue
                xyz = self._channel_locs_xyz.get(ds_idx)
                grid = self._channel_grids.get(ds_idx)
                if xyz is None or grid is None:
                    continue
                xnm, ynm, znm = xyz
                self._scheduler.request(
                    ch_missing, xnm, ynm, znm, grid,
                    sigma_yx_nm, depth_range_active,
                    self._tile_grid_x0, self._tile_grid_y0,
                )

        n_vis = len([c for c in self._channels if c["visible"]])
        suffix = f"  |  {len(all_missing)} tiles loading…" if all_missing else ""
        self._info_label.setText(
            f"{self._dataset_dim_label}  |  {n_vis} ch  |  "
            f"LOD {lod}  |  px={px_nm:.1f} nm{suffix}"
        )

        if self._bc_dialog is not None and self._bc_dialog.isVisible():
            first = scalar[0] if scalar.size else np.zeros((1, 1))
            self._bc_dialog.set_data(first)

    def _render_image_mode(self) -> None:
        if self._idx is None or self._image_data is None:
            return
        ds = self._state.datasets[self._idx]
        self._image_data = self._prepare_image_payload(ds.image_data)
        self._show_image_dataset(ds)

    def _render_direct(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        px_nm: float,
    ) -> None:
        """Render the viewport at its exact pixel size for fine zoom.

        Bypasses the LOD tile cache completely.  Uses the SpatialGrid for
        fast O(k) loc lookup then renders per-localization (sparse) or
        histogram (dense) at the viewport's own nm/px resolution.

        This ensures the image stays sharp at any zoom level — pixelation
        only appears when individual localizations are so sparse that blank
        pixels are physically correct.
        """
        from scipy.ndimage import gaussian_filter

        self._scheduler.cancel()
        self._pending_tile_keys.clear()
        self._last_lod = -1
        self._last_px_nm = px_nm

        sigma_yx_nm = self._sigma_yx_for_orientation(px_nm)
        depth_range_active: tuple[float, float] | None = None
        if self._has_depth and not self._all_depth_check.isChecked():
            depth_range_active = self._depth_range

        n_bins_x = max(int(round((x1 - x0) / px_nm)), 1)
        n_bins_y = max(int(round((y1 - y0) / px_nm)), 1)

        tiles: list[np.ndarray] = []
        total_count = 0

        for ch in self._channels:
            if ch["kind"] == "image":
                canvas = self._image_tile(
                    self._state.datasets[ch["dataset_idx"]],
                    x0, x1, y0, y1, n_bins_y, n_bins_x,
                )
                tiles.append(canvas)
                continue

            ds_idx = ch["dataset_idx"]
            xyz = self._channel_locs_xyz.get(ds_idx)
            grid = self._channel_grids.get(ds_idx)
            if xyz is None or grid is None or len(xyz[0]) == 0:
                tiles.append(np.zeros((n_bins_y, n_bins_x), dtype=np.float32))
                continue

            xnm, ynm, znm = xyz
            indices = grid.query(x0, x1, y0, y1)

            if len(indices) > 0:
                xv = xnm[indices]
                yv = ynm[indices]
                in_view = (xv >= x0) & (xv <= x1) & (yv >= y0) & (yv <= y1)
                if depth_range_active is not None:
                    z = znm[indices]
                    d_lo, d_hi = depth_range_active
                    in_view &= (z >= d_lo) & (z <= d_hi)
                xv = xv[in_view]
                yv = yv[in_view]
            else:
                xv = np.empty(0, dtype=np.float64)
                yv = np.empty(0, dtype=np.float64)

            count = len(xv)
            total_count += count

            if count == 0:
                canvas = np.zeros((n_bins_y, n_bins_x), dtype=np.float32)

            elif count < PER_LOC_SWITCH_COUNT:
                # Per-localization Gaussian — accurate at any pixel size
                canvas = np.zeros((n_bins_y, n_bins_x), dtype=np.float32)
                sigma_y_px = max(sigma_yx_nm[0] / px_nm, 0.3)
                sigma_x_px = max(sigma_yx_nm[1] / px_nm, 0.3)
                r_y = int(np.ceil(3.0 * sigma_y_px))
                r_x = int(np.ceil(3.0 * sigma_x_px))
                px_col = (xv - x0) / px_nm
                px_row = (yv - y0) / px_nm
                for cx, cy in zip(px_col, px_row):
                    c0 = int(round(cx)) - r_x
                    c1 = int(round(cx)) + r_x + 1
                    r0 = int(round(cy)) - r_y
                    r1 = int(round(cy)) + r_y + 1
                    dc0 = max(0, c0);  dc1 = min(n_bins_x, c1)
                    dr0 = max(0, r0);  dr1 = min(n_bins_y, r1)
                    if dc1 > dc0 and dr1 > dr0:
                        kc = np.arange(dc0, dc1, dtype=np.float32) - cx
                        kr = np.arange(dr0, dr1, dtype=np.float32) - cy
                        gauss = np.exp(
                            -0.5 * (
                                kr[:, None] ** 2 / sigma_y_px ** 2
                                + kc[None, :] ** 2 / sigma_x_px ** 2
                            )
                        )
                        canvas[dr0:dr1, dc0:dc1] += gauss

            else:
                # Histogram render at viewport resolution
                hist, _, _ = np.histogram2d(
                    yv, xv,
                    bins=[n_bins_y, n_bins_x],
                    range=[[y0, y1], [x0, x1]],
                )
                sig_y_px = sigma_yx_nm[0] / px_nm
                sig_x_px = sigma_yx_nm[1] / px_nm
                if max(sig_y_px, sig_x_px) >= 0.3:
                    hist = gaussian_filter(
                        hist.astype(np.float32),
                        sigma=(max(sig_y_px, 0.3), max(sig_x_px, 0.3)),
                        mode="constant",
                    )
                canvas = hist.astype(np.float32, copy=False)

            tiles.append(canvas)

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
                for ch in self._channels:
                    if ch["visible"]:
                        ch["levels"] = None
                        break
                if self._bc_dialog is not None and self._bc_dialog.isVisible():
                    self._bc_dialog.set_levels(*levels)

        rgba = self._compose_rgba(scalar)
        self._image_view.setImage(
            rgba, autoRange=False, autoLevels=False,
            pos=[x0, y0], scale=[px_nm, px_nm],
        )
        self._redraw_roi_highlight()

        n_vis = len([c for c in self._channels if c["visible"]])
        self._info_label.setText(
            f"{self._dataset_dim_label}  |  {total_count:,} locs in view  |  {n_vis} ch  |  "
            f"direct {px_nm:.2f} nm/px"
        )
        if self._bc_dialog is not None and self._bc_dialog.isVisible():
            first = scalar[0] if scalar.size else np.zeros((1, 1))
            self._bc_dialog.set_data(first)

    # Maximum canvas edge in pixels — prevents OOM when zoom limits are
    # bypassed mid-scroll before pyqtgraph can enforce them.
    _MAX_CANVAS_PX: int = 3000
    # Maximum physical tiles per axis in one composite call.
    _MAX_TILES_PER_AXIS: int = 64

    def _composite_loc_channel(
        self,
        ds_idx: int,
        mask_ver: int,
        orientation: str,
        tr_key: tuple,
        depth_key: tuple | None,
        vp_x0: float,
        vp_x1: float,
        vp_y0: float,
        vp_y1: float,
        lod: int,
        px_nm: float,
    ) -> tuple[np.ndarray, list[TileKey]]:
        """Composite physical tiles into a viewport canvas; return missing keys."""
        canvas_w = min(max(int(round((vp_x1 - vp_x0) / px_nm)), 1), self._MAX_CANVAS_PX)
        canvas_h = min(max(int(round((vp_y1 - vp_y0) / px_nm)), 1), self._MAX_CANVAS_PX)
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        missing: list[TileKey] = []

        col0 = int(np.floor((vp_x0 - self._tile_grid_x0) / PHYSICAL_TILE_NM))
        col1 = int(np.floor((vp_x1 - self._tile_grid_x0) / PHYSICAL_TILE_NM))
        row0 = int(np.floor((vp_y0 - self._tile_grid_y0) / PHYSICAL_TILE_NM))
        row1 = int(np.floor((vp_y1 - self._tile_grid_y0) / PHYSICAL_TILE_NM))

        # Cap tile range so the nested loop can never stall the main thread.
        if (col1 - col0) > self._MAX_TILES_PER_AXIS:
            mid = (col0 + col1) // 2
            col0 = mid - self._MAX_TILES_PER_AXIS // 2
            col1 = mid + self._MAX_TILES_PER_AXIS // 2
        if (row1 - row0) > self._MAX_TILES_PER_AXIS:
            mid = (row0 + row1) // 2
            row0 = mid - self._MAX_TILES_PER_AXIS // 2
            row1 = mid + self._MAX_TILES_PER_AXIS // 2

        tile_px = render_tile_px(lod)

        for tile_row in range(row0, row1 + 1):
            for tile_col in range(col0, col1 + 1):
                key = TileKey(
                    dataset_id=ds_idx,
                    mask_version=mask_ver,
                    orientation=orientation,
                    lod=lod,
                    tile_row=tile_row,
                    tile_col=tile_col,
                    transform_key=tr_key,
                    depth_range=depth_key,
                )

                tile_array = self._phys_tile_cache.get(key)
                if tile_array is None:
                    placeholder = self._phys_tile_cache.get_placeholder(key, tile_px)
                    missing.append(key)
                    tile_array = placeholder

                if tile_array is not None:
                    tx0 = self._tile_grid_x0 + tile_col * PHYSICAL_TILE_NM
                    ty0 = self._tile_grid_y0 + tile_row * PHYSICAL_TILE_NM
                    px_col = int(round((tx0 - vp_x0) / px_nm))
                    px_row = int(round((ty0 - vp_y0) / px_nm))
                    th, tw = tile_array.shape
                    dc0 = max(px_col, 0)
                    dc1 = min(px_col + tw, canvas_w)
                    dr0 = max(px_row, 0)
                    dr1 = min(px_row + th, canvas_h)
                    sc0 = dc0 - px_col
                    sr0 = dr0 - px_row
                    if dc1 > dc0 and dr1 > dr0:
                        canvas[dr0:dr1, dc0:dc1] = tile_array[
                            sr0 : sr0 + (dr1 - dr0),
                            sc0 : sc0 + (dc1 - dc0),
                        ]

        return canvas, missing

    @pyqtSlot(object, object)
    def _on_tile_ready(self, key: TileKey, array: np.ndarray) -> None:
        """Called on the main thread when a worker finishes a tile."""
        self._phys_tile_cache.put(key, array)
        self._pending_tile_keys.discard(key)

        # Re-composite only if the tile is still relevant
        if (
            key.orientation == self._orientation
            and key.mask_version == self._mask_versions.get(key.dataset_id, 0)
            and self._last_tile_geometry is not None
        ):
            self._recomposite_from_tiles()

    def _recomposite_from_tiles(self) -> None:
        """Re-assemble the display image from whatever is in the tile cache."""
        if self._last_tile_geometry is None:
            return
        x0, x1, y0, y1 = self._last_tile_geometry
        lod = self._last_lod
        px_nm = self._last_px_nm

        depth_range_active: tuple[float, float] | None = None
        if self._has_depth and not self._all_depth_check.isChecked():
            depth_range_active = self._depth_range

        tiles: list[np.ndarray] = []
        for ch in self._channels:
            if ch["kind"] == "image":
                ds = self._state.datasets[ch["dataset_idx"]]
                n_bins_x = max(int(round((x1 - x0) / px_nm)), 8)
                n_bins_y = max(int(round((y1 - y0) / px_nm)), 8)
                canvas = self._image_tile(ds, x0, x1, y0, y1, n_bins_y, n_bins_x)
            else:
                ds_idx = ch["dataset_idx"]
                mask_ver = self._mask_versions.get(ds_idx, 0)
                tr_key = self._channel_loc_transform_key(ch)
                depth_key = (round(depth_range_active[0], 1), round(depth_range_active[1], 1)) if depth_range_active else None
                canvas, _ = self._composite_loc_channel(
                    ds_idx, mask_ver, self._orientation, tr_key, depth_key,
                    x0, x1, y0, y1, lod, px_nm,
                )
            tiles.append(canvas)

        if not tiles:
            return

        scalar = np.stack(tiles, axis=0).astype(np.float32, copy=False)
        self._last_scalar_tile = scalar

        if self._auto_bc and scalar.shape[0] == 1:
            levels = self._compute_auto_levels(scalar[0])
            if levels is not None:
                self._manual_levels = levels

        rgba = self._compose_rgba(scalar)
        self._image_view.setImage(
            rgba, autoRange=False, autoLevels=False,
            pos=[x0, y0], scale=[px_nm, px_nm],
        )
        self._redraw_roi_highlight()

        pending = len(self._pending_tile_keys)
        n_vis = len([c for c in self._channels if c["visible"]])
        suffix = f"  |  {pending} tiles loading…" if pending else ""
        self._info_label.setText(
            f"{self._dataset_dim_label}  |  {n_vis} ch  |  "
            f"LOD {lod}  |  px={px_nm:.1f} nm{suffix}"
        )

        if self._bc_dialog is not None and self._bc_dialog.isVisible():
            first = scalar[0] if scalar.size else np.zeros((1, 1))
            self._bc_dialog.set_data(first)

    # ------------------------------------------------------------------
    # Spatial grid management
    # ------------------------------------------------------------------

    def _build_channel_grid(self, ch: dict) -> None:
        """Build SpatialGrid for one channel from its oriented filtered locs."""
        ds_idx = ch["dataset_idx"]
        ds = self._state.datasets[ds_idx]
        oriented = self._oriented_locs(ds)
        if oriented.shape[0] == 0:
            self._channel_grids[ds_idx] = None
            self._channel_locs_xyz[ds_idx] = (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
            return
        xnm = np.ascontiguousarray(oriented[:, 0], dtype=np.float64)
        ynm = np.ascontiguousarray(oriented[:, 1], dtype=np.float64)
        znm = np.ascontiguousarray(oriented[:, 2], dtype=np.float64)
        self._channel_locs_xyz[ds_idx] = (xnm, ynm, znm)
        self._channel_grids[ds_idx] = SpatialGrid(xnm, ynm)

    def _compute_tile_grid_origin(self) -> None:
        """Set _tile_grid_x0/_y0 aligned to PHYSICAL_TILE_NM boundaries."""
        all_x = [v[0] for v in self._channel_locs_xyz.values() if len(v[0]) > 0]
        all_y = [v[1] for v in self._channel_locs_xyz.values() if len(v[0]) > 0]
        if not all_x:
            self._tile_grid_x0 = 0.0
            self._tile_grid_y0 = 0.0
            return
        xmin = float(min(a.min() for a in all_x))
        ymin = float(min(a.min() for a in all_y))
        self._tile_grid_x0 = float(np.floor(xmin / PHYSICAL_TILE_NM) * PHYSICAL_TILE_NM)
        self._tile_grid_y0 = float(np.floor(ymin / PHYSICAL_TILE_NM) * PHYSICAL_TILE_NM)

    def _rebuild_all_grids(self) -> None:
        """Rebuild spatial grids for every localization channel."""
        for ch in self._channels:
            if ch["kind"] == "localizations":
                self._build_channel_grid(ch)
        self._compute_tile_grid_origin()

    def _refresh_depth_state(self) -> None:
        """Update depth axis bounds and slider from _locs_nm without resetting the viewport.

        Called after a filter change so the depth slider reflects the new
        filtered range while the pan/zoom position is preserved.
        """
        if self._locs_nm is None or self._locs_nm.shape[0] == 0:
            return
        o = self._orientation
        if o == "XY":
            xy_idx, depth_idx = [0, 1], 2
        elif o == "XZ":
            xy_idx, depth_idx = [0, 2], 1
        elif o == "YZ":
            xy_idx, depth_idx = [1, 2], 0
        else:
            return
        self._xy    = self._locs_nm[:, xy_idx]
        self._depth = self._locs_nm[:, depth_idx]
        self._bounds_depth = (float(self._depth.min()), float(self._depth.max()))
        self._has_depth = (self._bounds_depth[1] - self._bounds_depth[0]) > 1.0
        # Clamp the active depth range to the new data bounds without resetting it
        self._depth_slider.set_limits(*self._bounds_depth, reset_range=False)
        self._update_depth_label()

    def _increment_mask_version(self, dataset_id: int) -> None:
        self._mask_versions[dataset_id] = self._mask_versions.get(dataset_id, 0) + 1

    # ------------------------------------------------------------------
    # Brightness & Contrast helpers
    # ------------------------------------------------------------------

    def _bc_pixels(self) -> np.ndarray | None:
        if self._last_scalar_tile is not None:
            for idx, ch in enumerate(self._channels):
                if ch["visible"] and idx < self._last_scalar_tile.shape[0]:
                    return self._last_scalar_tile[idx]
        img = self._image_view.imageItem.image
        return None if img is None else np.asarray(img)

    def _compose_from_cache(self, *_args) -> None:
        """Recompose RGBA from the last scalar tile without re-rendering."""
        if self._last_scalar_tile is None:
            self._schedule_render()
            return
        if self._last_tile_geometry is None:
            (x0, x1), (y0, y1) = self._view_box.viewRange()
            x0, x1, y0, y1 = x0, x1, y0, y1
        else:
            x0, x1, y0, y1 = self._last_tile_geometry
        px_nm = self._last_px_nm
        rgba = self._compose_rgba(self._last_scalar_tile)
        self._image_view.setImage(
            rgba,
            autoRange=False,
            autoLevels=False,
            pos=[x0, y0],
            scale=[px_nm, px_nm],
        )
        self._redraw_roi_highlight()

    def _current_render_pixel_size_nm(self) -> float:
        if self._last_px_nm and self._last_px_nm > 0:
            return self._last_px_nm
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
        out = affine_transform(
            tile,
            matrix,
            offset=offset,
            output_shape=tile.shape,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        ).astype(np.float32, copy=False)
        out[~np.isfinite(out)] = 0.0
        out[out < 0.0] = 0.0
        vmax = float(np.nanmax(out)) if out.size else 0.0
        if vmax > 0.0:
            out[out < vmax * 1e-6] = 0.0
        return out

    def _normalized_tile(self, tile: np.ndarray, ch: dict) -> np.ndarray:
        if tile.size == 0 or not np.any(np.isfinite(tile)):
            return np.zeros(tile.shape, dtype=np.float32)
        positive = np.isfinite(tile) & (tile > 0.0)
        if not np.any(positive):
            return np.zeros(tile.shape, dtype=np.float32)
        levels = ch.get("levels")
        if levels is None:
            levels = self._manual_levels if len(self._channels) == 1 else self._compute_auto_levels(tile[positive])
        if levels is None:
            lo, hi = float(np.nanmin(tile[positive])), float(np.nanmax(tile[positive]))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0
        else:
            lo, hi = levels
        safe = np.nan_to_num(tile.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        norm = (safe - float(lo)) / max(float(hi) - float(lo), 1e-12)
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

    # ------------------------------------------------------------------
    # Tile helpers
    # ------------------------------------------------------------------

    def _channel_loc_transform_key(self, ch: dict) -> tuple:
        return transform_key(ch.get("loc_transform"))

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
        self._active_cmap = name
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

        try:
            cmap = pg.colormap.get(name)
            if cmap is not None:
                self._image_view.setColorMap(cmap)
                return
        except Exception:
            pass

        try:
            cmap = self._pg_colormap_from_matplotlib(name)
            self._image_view.setColorMap(cmap)
            return
        except Exception as e:
            print(f"Failed to load matplotlib colormap '{name}': {e!r}")

        try:
            cmap = pg.colormap.get(name, source="colorcet")
            if cmap is not None:
                self._image_view.setColorMap(cmap)
                return
        except Exception:
            pass

        try:
            self._image_view.setColorMap(pg.colormap.get("CET-L3"))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Brightness & Contrast
    # ------------------------------------------------------------------

    def _compute_auto_levels(
        self,
        hist: np.ndarray,
        *,
        advance_auto_threshold: bool = False,
    ) -> tuple[float, float] | None:
        """ImageJ/Fiji-style auto levels with repeated-press contrast boost."""
        values = np.asarray(hist, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size < 10:
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

        histogram, _edges = np.histogram(
            values,
            bins=_IMAGEJ_AUTO_HIST_BINS,
            range=(data_min, data_max),
        )
        pixel_count = int(values.size)
        limit = pixel_count // 10
        threshold = pixel_count // int(auto_threshold)

        found = False
        i = -1
        while not found and i < _IMAGEJ_AUTO_HIST_BINS - 1:
            i += 1
            count = int(histogram[i])
            if count > limit:
                count = 0
            found = count > threshold
        hmin = i

        found = False
        i = _IMAGEJ_AUTO_HIST_BINS
        while not found and i > 0:
            i -= 1
            count = int(histogram[i])
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
        if hi <= lo:
            hi = lo + 1.0
        return (float(lo), float(hi))

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
                if pixels is not None:
                    self._bc_dialog.set_data(pixels)
                    if self._manual_levels is not None:
                        self._bc_dialog.set_levels(*self._manual_levels)
            self._bc_dialog.set_auto_state(self._auto_bc)

        self._bc_dialog.show()
        self._bc_dialog.raise_()
        self._bc_dialog.activateWindow()

    def _on_levels_changed(self, lo: float, hi: float) -> None:
        if self._auto_bc:
            self._auto_bc = False
            if self._bc_dialog is not None:
                self._bc_dialog.set_auto_state(False)
        self._bc_auto_threshold = 0
        self._manual_levels = (lo, hi)
        target = next((i for i, ch in enumerate(self._channels) if ch["visible"]), 0)
        if 0 <= target < len(self._channels):
            self._channels[target]["levels"] = (lo, hi)
            self._compose_from_cache()

    def _on_bc_auto(self) -> None:
        img = self._bc_pixels()
        if img is None:
            return
        levels = self._compute_auto_levels(img, advance_auto_threshold=True)
        if levels is None:
            return
        self._auto_bc = True
        self._manual_levels = levels
        for ch in self._channels:
            if ch["visible"]:
                ch["levels"] = None
                break
        self._compose_from_cache()
        if self._bc_dialog is not None:
            self._bc_dialog.set_data(img)
            self._bc_dialog.set_levels(*levels)
            self._bc_dialog.set_auto_state(True)

    def _on_bc_reset(self) -> None:
        img = self._bc_pixels()
        if img is None:
            return
        values = np.asarray(img, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        levels = (float(values.min()), float(values.max() if values.max() > values.min() else values.min() + 1.0))
        self._manual_levels = levels
        self._auto_bc = False
        self._bc_auto_threshold = 0
        for ch in self._channels:
            if ch["visible"]:
                ch["levels"] = levels
                break
        if self._bc_dialog is not None:
            self._bc_dialog.set_data(img)
            self._bc_dialog.set_levels(*levels)
        self._compose_from_cache()

    # ------------------------------------------------------------------
    # LUT dialog (Fiji-style B&C + colormap)
    # ------------------------------------------------------------------

    def open_lut_dialog(self) -> None:
        """Open the rich LUT editor for this render window."""
        from .lut_dialog import LutDialog, make_colormap

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
        from .lut_dialog import make_colormap
        self._lut_invert = invert
        self._active_cmap = name
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
        return self._active_cmap

    # ------------------------------------------------------------------
    # View interaction
    # ------------------------------------------------------------------

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)

        view_menu = menu.addMenu("View")
        for orientation in _ORIENTATIONS:
            action = view_menu.addAction(orientation)
            action.setCheckable(True)
            if orientation == "3D":
                action.setChecked(self._volume_window is not None and self._volume_window.isVisible())
                action.setEnabled(self._render_mode == "localizations" and self._has_depth)
                action.triggered.connect(self._show_3d_volume_window)
            else:
                action.setChecked(self._orientation == orientation)
                action.setEnabled(orientation in _RENDER_ORIENTATIONS)
                action.triggered.connect(
                    lambda _checked=False, value=orientation: self._set_orientation(value)
                )

        cmap_menu = menu.addMenu("Colormap")
        for name in _COLORMAPS:
            action = cmap_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(self._active_channel_lut() == name)
            action.triggered.connect(lambda _checked=False, value=name: self._on_cmap_changed(value))
        pure_menu = cmap_menu.addMenu("Pure color")
        for name in _PURE_COLOR_LUTS:
            action = pure_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(self._active_channel_lut() == name)
            action.triggered.connect(lambda _checked=False, value=name: self._on_cmap_changed(value))

        bc_action = menu.addAction("Brightness/Contrast", self._show_brightness_contrast)
        bc_shortcut = str(self._state.prefs.get("shortcuts", {}).get("brightness_contrast", "") or "")
        if bc_shortcut:
            bc_action.setShortcut(QKeySequence(bc_shortcut))
            try:
                bc_action.setShortcutVisibleInContextMenu(False)
            except Exception:
                pass
        menu.addAction("Sigma", self._show_sigma_dialog)

        axis_action = menu.addAction("Axis")
        axis_action.setCheckable(True)
        axis_action.setChecked(self._axis_visible)
        axis_action.triggered.connect(self._set_axis_visible_from_menu)

        export_action = menu.addAction("Export to TIFF…", self._export_to_tiff)
        export_action.setEnabled(self._render_mode == "localizations")

        menu.addAction("Reset View", self._reset_view)
        menu.exec(self._image_view.ui.graphicsView.mapToGlobal(pos))

    def _set_orientation(self, text: str) -> None:
        if text not in _RENDER_ORIENTATIONS:
            return
        self._orientation = text
        self._apply_orientation()
        self._rebuild_all_grids()
        self._scheduler.cancel()
        self._schedule_render()
        if self._axis_visible:
            self._update_axis_labels()      # keep X/Y/Z labels in sync with the plane
        # Re-project point markers onto the new view plane.
        if self._roi_overlay is not None:
            self._roi_overlay.refresh()

    def _set_axis_visible_from_menu(self, checked: bool) -> None:
        self._set_axes_visible(bool(checked))

    def _show_3d_volume_window(self) -> None:
        if self._idx is None or not (0 <= self._idx < len(self._state.datasets)):
            return
        self._state.set_active(self._idx)
        if self._volume_window is None:
            from .volume_window import VolumeRenderWindow
            self._volume_window = VolumeRenderWindow(
                self._state,
                self._idx,
                sigma_nm_xyz=self._sigma_nm_xyz,
                parent=self,
            )
            self._volume_window.destroyed.connect(lambda *_: setattr(self, "_volume_window", None))
        else:
            self._volume_window._sigma_nm_xyz = self._sigma_nm_xyz
            self._volume_window.refresh_from_dataset()
        self._volume_window.show()
        self._volume_window.raise_()
        self._volume_window.activateWindow()

    def _show_sigma_dialog(self) -> None:
        dialog = SigmaDialog(self._sigma_nm_xyz, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._sigma_nm_xyz = dialog.values_xyz()
        self._phys_tile_cache.clear()
        self._scheduler.cancel()
        self._schedule_render()
        if self._volume_window is not None and self._volume_window.isVisible():
            self._volume_window._sigma_nm_xyz = self._sigma_nm_xyz
            self._volume_window.refresh_from_dataset()

    def _active_rectangle_xy_bounds(self) -> tuple[float, float, float, float] | None:
        """X/Y bounds (nm, display coords) of the active/selected XY-plane
        rectangle ROI, or None. Used to restrict the TIFF export to the ROI."""
        from ..core.roi_selection import rectangle_bounds

        candidates = []
        ctrl = getattr(self, "_roi_overlay", None)
        if ctrl is not None:
            try:
                rec = ctrl.current_record()
            except Exception:
                rec = None
            if rec is not None:
                candidates.append(rec)
        try:
            selected = set(self._state.rois.selected_ids)
            candidates.extend(r for r in self._state.rois.records if r.id in selected)
        except Exception:
            pass
        for rec in candidates:
            if getattr(rec, "type", None) != "rectangle":
                continue
            plane = (getattr(rec, "context", {}) or {}).get("view_plane") or self._orientation
            if plane != "XY":
                continue
            bounds = rectangle_bounds(rec)
            if bounds is not None:
                return bounds  # (x0, x1, y0, y1)
        return None

    def _gather_export_channels(self) -> list:
        """Snapshot the visible localization channels as TiffExportChannel(s)."""
        from ..core.tiff_export import TiffExportChannel

        out = []
        for ch in self._channels:
            if not ch.get("visible") or ch.get("kind") != "localizations":
                continue
            ds = self._state.datasets[ch["dataset_idx"]]
            xyz = self._dataset_locs(ds)
            if xyz.shape[0] > 0:
                out.append(TiffExportChannel(name=str(ch.get("name") or ds.name), xyz=xyz))
        return out

    def _export_to_tiff(self) -> None:
        """Export the visible localization channels to a multipage (OME-)TIFF.

        A modal dialog collects XY pixel size, Z voxel depth, the export ranges
        (XY pre-filled from the active rectangle ROI when present, else the data
        extent) and — for 3-D data — the editable RIMF z-scaling. Each visible
        channel becomes a Z-sliced 2-D-histogram stack with physical calibration
        in OME metadata. The binning/writing runs on a background worker so the
        viewer stays responsive; progress and completion go to the Log (no
        progress bar / pop-up).
        """
        from .tiff_export_dialog import TiffExportDialog, TiffExportWorker

        if self._render_mode != "localizations":
            QMessageBox.information(
                self, "Export to TIFF", "TIFF export is only available for localization renders."
            )
            return

        channels = self._gather_export_channels()
        if not channels:
            QMessageBox.information(
                self, "Export to TIFF", "No visible localizations pass the current filter."
            )
            return

        all_xyz = np.vstack([c.xyz for c in channels])
        is_3d = float(all_xyz[:, 2].max() - all_xyz[:, 2].min()) > 1.0

        roi_bounds = self._active_rectangle_xy_bounds()
        if roi_bounds is not None:
            x_span = (roi_bounds[0], roi_bounds[1])
            y_span = (roi_bounds[2], roi_bounds[3])
        else:
            x_span = (float(all_xyz[:, 0].min()), float(all_xyz[:, 0].max()))
            y_span = (float(all_xyz[:, 1].min()), float(all_xyz[:, 1].max()))
        z_span = (float(all_xyz[:, 2].min()), float(all_xyz[:, 2].max()))

        ds0 = self._state.datasets[self._idx] if self._idx is not None else None
        rimf = None
        if is_3d and ds0 is not None:
            try:
                rimf = float(getattr(ds0.cali, "RIMF", 1.0) or 1.0)
            except Exception:
                rimf = None

        stem = "render"
        folder = ""
        if ds0 is not None:
            stem = Path(str(getattr(ds0, "name", "") or "render")).stem or "render"
            folder = str(getattr(getattr(ds0, "file", None), "folder", "") or "")
        default_path = str(Path(folder) / f"{stem}.ome.tif") if folder else f"{stem}.ome.tif"
        default_px = max(1.0, round(self._current_render_pixel_size_nm(), 1))

        dialog = TiffExportDialog(
            default_path=default_path,
            default_pixel_nm=default_px,
            default_voxel_nm=default_px,
            is_3d=is_3d,
            x_span=x_span,
            y_span=y_span,
            z_span=z_span,
            rimf=rimf,
            xy_from_roi=roi_bounds is not None,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        params = dialog.params()
        if not params["path"]:
            QMessageBox.warning(self, "Export to TIFF", "Choose an output file path.")
            return

        # Apply an edited RIMF to the active dataset (single source of truth — the
        # render and all views update too), then re-snapshot so the exported z
        # and the rescaled Z range agree.
        new_rimf = params.get("rimf")
        if new_rimf is not None and rimf is not None and abs(new_rimf - rimf) > 1e-6 and ds0 is not None:
            ds0.set_rimf(float(new_rimf), source="manual (tiff export)")
            if self._idx is not None:
                self._state.notify_calibration_changed(self._idx)
            self._state.log(
                f"RIMF set to {new_rimf:.4f} (manual, TIFF export)", dataset_idx=self._idx
            )
            channels = self._gather_export_channels()
            if not channels:
                QMessageBox.information(
                    self, "Export to TIFF", "No visible localizations pass the current filter."
                )
                return

        worker = TiffExportWorker(
            channels,
            params["path"],
            pixel_size_nm=params["pixel_size_nm"],
            voxel_depth_nm=params["voxel_depth_nm"],
            is_3d=is_3d,
            x_range=params["x_range"],
            y_range=params["y_range"],
            z_range=params["z_range"],
            parent=self,
        )
        self._export_workers.append(worker)
        worker.progress.connect(lambda msg: self._state.log(msg, dataset_idx=self._idx))
        worker.completed.connect(self._on_tiff_export_done)
        worker.failed.connect(self._on_tiff_export_failed)
        worker.completed.connect(lambda *_: self._forget_export_worker(worker))
        worker.failed.connect(lambda *_: self._forget_export_worker(worker))
        n_ch = len(channels)
        self._state.log(
            f"Export to TIFF started: {n_ch} channel(s), pixel {params['pixel_size_nm']:g} nm"
            + (f", voxel {params['voxel_depth_nm']:g} nm" if is_3d else " (2-D)")
            + f" → {params['path']}",
            dataset_idx=self._idx,
        )
        worker.start()

    def _on_tiff_export_done(self, result) -> None:
        self._state.log(
            f"Export to TIFF done: {result.axes} {result.shape} {result.dtype} "
            f"(max count {result.max_count}) → {result.path}",
            dataset_idx=self._idx,
        )

    def _on_tiff_export_failed(self, message: str) -> None:
        self._state.log(f"Export to TIFF failed: {message}", level="ERROR", dataset_idx=self._idx)

    def _forget_export_worker(self, worker) -> None:
        try:
            worker.wait(50)
        except Exception:
            pass
        if worker in self._export_workers:
            self._export_workers.remove(worker)

    def _show_data_info_window(self) -> None:
        if self._idx is None or not (0 <= self._idx < len(self._state.datasets)):
            return
        self._state.set_active(self._idx)
        win = self._find_data_info_window(self._idx)
        if win is None:
            from .data_window import DataWindow
            win = DataWindow(self._state.datasets[self._idx], self._idx, self._state)
        if win.isMinimized():
            win.setWindowState(win.windowState() & ~Qt.WindowState.WindowMinimized)
        win.show()
        win.raise_()
        win.activateWindow()

    def _find_data_info_window(self, dataset_idx: int) -> QWidget | None:
        app = QApplication.instance()
        if app is None:
            return None
        for widget in app.topLevelWidgets():
            if widget.__class__.__name__ == "DataWindow" and getattr(widget, "_idx", None) == dataset_idx:
                return widget
        return None

    def _sigma_yx_for_orientation(self, pixel_size_nm: float) -> tuple[float, float]:
        sx, sy, sz = self._sigma_nm_xyz
        if self._orientation == "XZ":
            display_x, display_y = sx, sz
        elif self._orientation == "YZ":
            display_x, display_y = sy, sz
        else:
            display_x, display_y = sx, sy
        # Default: 0.5 px anti-aliasing, not 1.2.  MINFLUX precision is 1–5 nm;
        # spreading each loc over a full pixel blurs away all spatial detail.
        auto_sigma = pixel_size_nm * 0.5
        sigma_x = display_x if display_x > 0.0 else auto_sigma
        sigma_y = display_y if display_y > 0.0 else auto_sigma
        return float(sigma_y), float(sigma_x)

    def _on_range_changed(self, *_args) -> None:
        if self._suppress_zoom_limit:
            self._remember_fit_view_size()
            self._schedule_render()
            return
        if self._enforce_zoom_out_limit():
            return
        self._schedule_render()

    def _enforce_zoom_out_limit(self) -> bool:
        """Clamp over-zoomed view ranges after PyQtGraph updates the ViewBox."""
        try:
            (x0, x1), (y0, y1) = self._view_box.viewRange()
        except Exception:
            return False
        w = float(x1 - x0)
        h = float(y1 - y0)
        if not np.isfinite(w) or not np.isfinite(h) or w <= 0.0 or h <= 0.0:
            return False

        fit_w, fit_h = self._fit_view_size
        max_w = max(float(fit_w), 1.0) * self._ZOOM_OUT_LIMIT
        max_h = max(float(fit_h), 1.0) * self._ZOOM_OUT_LIMIT
        if w <= max_w and h <= max_h:
            return False

        aspect = max(w / h, 1e-12)
        target_w = min(w, max_w)
        target_h = target_w / aspect
        if target_h > max_h:
            target_h = max_h
            target_w = target_h * aspect
        bx0, bx1, by0, by1 = self._bounds_xy
        dcx = (float(bx0) + float(bx1)) / 2.0
        dcy = (float(by0) + float(by1)) / 2.0
        blocker = QSignalBlocker(self._view_box)
        try:
            self._view_box.setRange(
                xRange=(dcx - target_w / 2.0, dcx + target_w / 2.0),
                yRange=(dcy - target_h / 2.0, dcy + target_h / 2.0),
                padding=0,
                update=True,
            )
        finally:
            del blocker
        self._schedule_render()
        return True

    def _set_axes_visible(self, visible: bool) -> None:
        self._axis_visible = bool(visible)
        plot_item = self._image_view.view
        for axis_name in ("left", "bottom"):
            plot_item.showAxis(axis_name, show=visible)
        if visible:
            self._update_axis_labels()

    def _axis_label_names(self) -> tuple[str, str]:
        """Bottom/left axis labels for the current orientation, so the user can
        tell which axis is which."""
        unit = "px" if self._image_data is not None else "nm"
        names = {"XY": ("X", "Y"), "XZ": ("X", "Z"), "YZ": ("Y", "Z")}.get(self._orientation, ("X", "Y"))
        return f"{names[0]} ({unit})", f"{names[1]} ({unit})"

    def _update_axis_labels(self) -> None:
        bottom, left = self._axis_label_names()
        view = self._image_view.view
        view.setLabel("bottom", bottom)
        view.setLabel("left", left)

    def _on_orientation_changed(self, text: str) -> None:
        self._set_orientation(text)

    def _on_all_depth_toggled(self, checked: bool) -> None:
        self._depth_slider.setEnabled(self._has_depth and not checked)
        if self._has_depth and not checked and not self._depth_range_initialized:
            self._set_default_depth_range()
        self._update_depth_label()
        self._scheduler.cancel()
        self._schedule_render()

    def _on_depth_range_changed(self, lo: float, hi: float) -> None:
        self._depth_range = (float(lo), float(hi))
        self._depth_range_initialized = True
        self._update_depth_label()
        self._scheduler.cancel()
        self._schedule_render()

    def _set_default_depth_range(self) -> None:
        d_lo, d_hi = self._bounds_depth
        if self._render_mode == "image":
            center = float(round((d_lo + d_hi) / 2.0))
            lo = hi = min(max(center, d_lo), d_hi)
            self._depth_range = (lo, hi)
            self._depth_inclusive = (True, True)
            self._depth_range_initialized = True
            self._depth_slider.set_range(lo, hi)
            return
        width = max((d_hi - d_lo) / 10.0, 0.0)
        center = (d_lo + d_hi) / 2.0
        lo = max(d_lo, center - width / 2.0)
        hi = min(d_hi, center + width / 2.0)
        self._depth_range = (lo, hi)
        self._depth_inclusive = (True, True)
        self._depth_range_initialized = True
        self._depth_slider.set_range(lo, hi)

    def _show_depth_range_dialog(self) -> None:
        if not self._has_depth or self._all_depth_check.isChecked():
            return
        dialog = DepthRangeDialog(
            self._depth_axis_name,
            self._bounds_depth,
            self._depth_range,
            self._depth_inclusive,
            self._depth_scroll_step_nm,
            self._depth_reverse_scroll,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._depth_range, self._depth_inclusive = dialog.values()
        self._depth_scroll_step_nm, self._depth_reverse_scroll = dialog.scroll_options()
        self._depth_range_initialized = True
        self._depth_slider.set_range(*self._depth_range)
        self._depth_slider.set_scroll_options(self._depth_scroll_step_nm, self._depth_reverse_scroll)
        self._update_depth_label()
        self._scheduler.cancel()
        self._schedule_render()

    def _update_depth_label(self) -> None:
        if not self._has_depth:
            self._depth_label.setText("—")
            return
        if self._all_depth_check.isChecked():
            self._depth_label.setText(self._format_depth_range(self._bounds_depth, (True, True)))
            return
        self._depth_label.setText(self._format_depth_range(self._depth_range, self._depth_inclusive))

    def _format_depth_range(
        self,
        values: tuple[float, float],
        inclusive: tuple[bool, bool],
    ) -> str:
        left = "[" if inclusive[0] else "("
        right = "]" if inclusive[1] else ")"
        if self._render_mode == "image" and self._depth_axis_name == "Slice":
            lo, hi = int(round(values[0])), int(round(values[1]))
            if lo == hi:
                return f"slice {lo}"
            return f"slices {left}{lo}, {hi}{right}"
        return f"{left}{values[0]:.1f}, {values[1]:.1f}{right} nm"

    def _clear_roi_highlight(self) -> None:
        if self._roi_highlight_item is not None:
            self._roi_highlight_item.setData([], [])

    def _roi_masks_for_dataset(self, ds) -> list[tuple[object, np.ndarray]]:
        records = [r for r in self._state.rois.records if r.id in set(self._state.rois.selected_ids)]
        draft_id = ds.state.get("active_roi_draft_id")
        if draft_id:
            draft_meta = ds.state.get("roi_masks", {}).get(draft_id, {})
            draft_record = next((r for r in records if r.id == draft_id), None)
            if draft_record is None and isinstance(draft_meta, dict):
                draft_record = type("_RoiHighlight", (), {
                    "id": draft_id,
                    "stroke_color": draft_meta.get("stroke_color", "#ffff00"),
                })()
            if draft_record is not None and all(r.id != draft_id for r in records):
                records.append(draft_record)
        out: list[tuple[object, np.ndarray]] = []
        ftr = np.asarray(ds.filter_mask, dtype=bool).ravel()
        for record in records:
            mask = active_roi_mask(ds, selected_ids=[record.id], include_active_draft=False)
            if mask is None and record.id == draft_id:
                mask = active_roi_mask(ds, selected_ids=[], include_active_draft=True)
            if mask is None:
                continue
            mask = np.asarray(mask, dtype=bool).ravel()
            if ftr.size == mask.size:
                mask &= ftr
            out.append((record, mask))
        return out

    @staticmethod
    def _roi_highlight_brushes(record, count: int) -> list:
        color = pg.mkColor(getattr(record, "stroke_color", "#ffff00") or "#ffff00")
        fill = pg.mkColor(color)
        fill.setAlpha(75)
        return [pg.mkBrush(fill)] * int(count)

    def _owns_active_roi_draft(self) -> bool:
        """True when an ROI is currently being drawn in *this* render view."""
        ctrl = getattr(self, "_roi_overlay", None)
        if ctrl is None:
            return False
        try:
            return ctrl.current_record() is not None
        except Exception:
            return getattr(ctrl, "draft", None) is not None

    def _roi_highlight_enabled(self) -> bool:
        from ..core.roi_selection import roi_highlight_enabled
        return roi_highlight_enabled(
            self._state.prefs, is_source=self._owns_active_roi_draft())

    def _redraw_roi_highlight(self) -> None:
        if not self._roi_highlight_enabled():
            self._clear_roi_highlight()
            return
        if self._roi_highlight_item is None or self._render_mode == "image":
            self._clear_roi_highlight()
            return
        if self._orientation == "XY":
            axes, depth_axis = (0, 1), 2
        elif self._orientation == "XZ":
            axes, depth_axis = (0, 2), 1
        elif self._orientation == "YZ":
            axes, depth_axis = (1, 2), 0
        else:
            self._clear_roi_highlight()
            return

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        brushes: list = []
        channels = self._channels or [{"dataset_idx": self._idx, "visible": True, "kind": "localizations"}]
        per_channel_max = max(1, 200_000 // max(len(channels), 1))
        for ch in channels:
            if not ch.get("visible", True) or ch.get("kind") == "image":
                continue
            ds_idx = ch.get("dataset_idx")
            if ds_idx is None or not (0 <= ds_idx < len(self._state.datasets)):
                continue
            ds = self._state.datasets[ds_idx]
            locs = self._raw_render_locs(ds)
            if locs.shape[0] == 0:
                continue
            for record, mask in self._roi_masks_for_dataset(ds):
                n = min(mask.size, locs.shape[0])
                visible = mask[:n] & np.all(np.isfinite(locs[:n, :3]), axis=1)
                if self._has_depth and not self._all_depth_check.isChecked():
                    lo, hi = self._depth_range
                    left_inc, right_inc = self._depth_inclusive
                    depth = locs[:n, depth_axis]
                    lo_mask = depth >= lo if left_inc else depth > lo
                    hi_mask = depth <= hi if right_inc else depth < hi
                    visible &= lo_mask & hi_mask
                indices = np.flatnonzero(visible)
                if indices.size > per_channel_max:
                    step = int(np.ceil(indices.size / per_channel_max))
                    indices = indices[::step]
                if indices.size:
                    xs.append(locs[indices, axes[0]])
                    ys.append(locs[indices, axes[1]])
                    brushes.extend(self._roi_highlight_brushes(record, indices.size))
        if not xs:
            self._clear_roi_highlight()
            return
        self._roi_highlight_item.setData(
            x=np.concatenate(xs),
            y=np.concatenate(ys),
            brush=brushes,
            pen=None,
            size=7,
        )

    def roi_view_plane(self) -> str | None:
        """Current view orientation for ROI 3-D placement (XY/XZ/YZ)."""
        return self._orientation if self._orientation in {"XY", "XZ", "YZ"} else None

    def roi_depth_center(self) -> float | None:
        """Centre of the current viewing range of the out-of-plane (depth) axis,
        i.e. the fallback value a newly drawn ROI gets in the dimension not shown
        on screen when there is no data at that XY.  ``None`` for 2-D datasets."""
        if not self._has_depth:
            return None
        lo, hi = self._depth_range
        return 0.5 * (float(lo) + float(hi))

    def roi_depths_at(self, points):
        """Data-aware out-of-plane value for each drawn in-plane vertex.

        For every ``(a, b)`` in *points* (current-plane coordinates), return the
        XY-proximity-weighted median of the localizations' depth coordinate at
        that column — so a vertex lands on the actual 3-D structure rather than
        the static range midpoint. ``None`` per point where the column is empty
        (caller falls back to ``roi_depth_center``). ``None`` list for 2-D data."""
        plane = self.roi_view_plane()
        if plane is None or not self._has_depth or not points:
            return [None] * len(points)
        locs = self._depth_profile_locs()
        if locs is None:
            return [None] * len(points)
        from ..core.roi_depth import weighted_depths
        i, j = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}[plane]
        k = {"XY": 2, "XZ": 1, "YZ": 0}[plane]
        return weighted_depths(points, locs[:, i], locs[:, j], locs[:, k])

    def _depth_profile_locs(self):
        """Active dataset's localizations in display nm (all Z, unfiltered) for
        sampling the out-of-plane structure under a drawn vertex."""
        if self._idx is None or not (0 <= self._idx < len(self._state.datasets)):
            return None
        try:
            locs = self._raw_render_locs(self._state.datasets[self._idx])
        except Exception:
            return None
        return locs if locs.ndim == 2 and locs.shape[1] >= 3 and locs.shape[0] else None

    def snap_to_density(self, x_nm: float, y_nm: float, *,
                        window_nm: float = 60.0, iterations: int = 3):
        """Snap an in-plane cursor (nm) to the local high-density centre by
        mean-shifting on the **rendered density raster** (the magnetic-lasso
        snap). Returns snapped ``(x_nm, y_nm)`` or ``None`` when no raster is
        available (caller then keeps the raw cursor)."""
        tile = self._last_scalar_tile
        geom = self._last_tile_geometry
        px = self._last_px_nm
        if tile is None or geom is None or not px or px <= 0:
            return None
        tile = np.asarray(tile)
        if tile.ndim != 3 or tile.shape[0] == 0:
            return None
        visible = [i for i, ch in enumerate(self._channels[:tile.shape[0]]) if ch.get("visible")]
        if not visible:
            visible = list(range(tile.shape[0]))
        density = tile[visible].sum(axis=0)
        if density.ndim != 2 or density.size == 0:
            return None
        from ..core.magnetic_snap import snap_density_centroid
        x0, _x1, y0, _y1 = geom
        col = (float(x_nm) - x0) / px
        row = (float(y_nm) - y0) / px
        win = int(np.clip(round(float(window_nm) / px), 3, 51))
        sc, sr = snap_density_centroid(density, col, row, window=win, iterations=iterations)
        return (x0 + sc * px, y0 + sr * px)

    def normalize_roi_record(self, record):
        """Tag a drawn ROI with the view plane it was drawn in and the centre of
        the current viewing depth range, so its out-of-plane position is defined
        (rather than defaulting to a meaningless in-plane coordinate)."""
        plane = self.roi_view_plane()
        if plane is None:
            return record
        ctx = dict(record.context)
        ctx.setdefault("view_plane", plane)
        ctx.setdefault("depth_axis", {"XY": "Z", "XZ": "Y", "YZ": "X"}[plane])
        if record.type != "point":
            center = self.roi_depth_center()
            if center is not None:
                ctx.setdefault("depth_value", float(center))
        record.context = ctx
        return record

    def compute_roi_selection(self, record):
        if record.type not in {"rectangle", "oval", "polygon", "freehand"} or self._idx is None:
            return None
        if not (0 <= self._idx < len(self._state.datasets)):
            return None
        ds = self._state.datasets[self._idx]
        locs = self._raw_render_locs(ds)
        if locs.shape[0] == 0:
            return None

        if self._orientation == "XY":
            axes, depth_axis, depth_name = (0, 1), 2, "Z"
        elif self._orientation == "XZ":
            axes, depth_axis, depth_name = (0, 2), 1, "Y"
        elif self._orientation == "YZ":
            axes, depth_axis, depth_name = (1, 2), 0, "X"
        else:
            return None

        base = np.asarray(ds.filter_mask, dtype=bool)
        if base.shape[0] != locs.shape[0]:
            base = np.ones(locs.shape[0], dtype=bool)
        base &= np.all(np.isfinite(locs[:, :3]), axis=1)
        if self._has_depth and not self._all_depth_check.isChecked():
            lo, hi = self._depth_range
            left_inc, right_inc = self._depth_inclusive
            depth = locs[:, depth_axis]
            lo_mask = depth >= lo if left_inc else depth > lo
            hi_mask = depth <= hi if right_inc else depth < hi
            base &= lo_mask & hi_mask

        mask = roi_region_mask(locs[:, axes[0]], locs[:, axes[1]], record, base_mask=base)
        context = {
            "source_view": "render",
            "dataset_idx": self._idx,
            "orientation": self._orientation,
            "x_axis": "XYZ"[axes[0]],
            "y_axis": "XYZ"[axes[1]],
            "depth_axis": depth_name,
            "depth_range": list(self._depth_range) if self._has_depth and not self._all_depth_check.isChecked() else None,
            "depth_inclusive": list(self._depth_inclusive),
        }
        return ds, mask, context

    def _raw_render_locs(self, ds) -> np.ndarray:
        try:
            locs = np.asarray(ds.loc_nm, dtype=np.float64)
        except Exception:
            return np.empty((0, 3), dtype=np.float64)
        if locs.ndim != 2 or locs.shape[1] < 2:
            return np.empty((0, 3), dtype=np.float64)
        if locs.shape[1] == 2:
            locs = np.column_stack([locs, np.zeros(locs.shape[0], dtype=np.float64)])
        return self._apply_dataset_render_transform(ds, locs[:, :3])

    def closeEvent(self, event) -> None:
        if self._volume_window is not None:
            try:
                self._volume_window.close()
            except Exception:
                pass
            self._volume_window = None
        # Let any in-flight TIFF export finish so its QThread is not destroyed
        # while running (the file write is short relative to the binning).
        for worker in list(self._export_workers):
            try:
                if worker.isRunning():
                    worker.wait()
            except Exception:
                pass
        self._export_workers.clear()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Fiji-style active-dataset-follows-focus
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
        pass

    def _on_filter_changed(self, idx: int) -> None:
        if any(ch["dataset_idx"] == idx for ch in self._channels):
            self._increment_mask_version(idx)
            for ch in self._channels:
                if ch["dataset_idx"] == idx and ch["kind"] == "localizations":
                    self._build_channel_grid(ch)
            self._compute_tile_grid_origin()
            # Update depth slider range without touching the viewport
            if self._channels and self._channels[0]["dataset_idx"] == idx:
                self._locs_nm = self._channel_locs(self._channels[0])
                self._refresh_depth_state()
            self._scheduler.cancel()
            self._schedule_render()
            if self._volume_window is not None and self._volume_window.isVisible():
                self._volume_window.refresh_from_dataset()

    def _on_roi_selection_changed(self, idx: int) -> None:
        if any(ch["dataset_idx"] == idx for ch in self._channels):
            self._redraw_roi_highlight()
