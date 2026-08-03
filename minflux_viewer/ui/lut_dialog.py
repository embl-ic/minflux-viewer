"""
minflux_viewer.ui.lut_dialog
=============================
Fiji-style LUT (Look-Up Table) editor.

Shows for the currently active render image:

* a **mini histogram** of the currently rendered pixel values
  (log-y, simplified — just enough to pick sensible min/max)
* a **colormap dropdown** including 8 matplotlib colormaps plus the
  seven single-colour channels (Red, Green, Blue, Cyan, Magenta,
  Yellow, Gray)
* four **sliders** — Minimum, Maximum, Brightness, Contrast — matching
  Fiji's B&C behaviour
* **Auto** — compute min/max that leave 0.35 % of pixels saturated at
  each end, like Fiji's ``ImagePlus::IJ_Auto`` method
* **Invert LUT** — flip the colormap
* **Reset** — discard all edits and restore the state that was active
  when the dialog was opened

The dialog is callback-driven (like :mod:`brightness_contrast_dialog`):
the render window plugs in `on_levels_changed`, `on_cmap_changed`, and
`on_invert_changed` closures that actually apply changes to the image.

Notes
-----
Brightness and Contrast sliders are *derived* from (min, max):

    midpoint  = (min + max) / 2        <-> Brightness (inverted)
    half_span = (max - min) / 2        <-> Contrast   (inverted)

so moving Brightness shifts both limits up/down together, and moving
Contrast squeezes/stretches them symmetrically around the midpoint.
This is exactly how Fiji's B&C dialog works.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Colormap options — 8 common mpl maps + 7 single-colour channels.
# The channel names correspond to pure colour ramps (black → colour).
# ---------------------------------------------------------------------------

_SINGLE_COLOURS = ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Gray"]

_MPL_CMAPS = [
    "glasbey", "jet", "HiLo", "parula",
    "viridis", "inferno", "magma", "plasma", "cividis",
    "hot", "gray", "turbo",
]

ALL_COLORMAPS: list[str] = _MPL_CMAPS + _SINGLE_COLOURS


_SLIDER_RES = 1000


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------

_CHANNEL_RGB = {
    "Red":     (1.0, 0.0, 0.0),
    "Green":   (0.0, 1.0, 0.0),
    "Blue":    (0.0, 0.0, 1.0),
    "Cyan":    (0.0, 1.0, 1.0),
    "Magenta": (1.0, 0.0, 1.0),
    "Yellow":  (1.0, 1.0, 0.0),
    "Gray":    (1.0, 1.0, 1.0),
}


def apply_gamma(cmap: pg.ColorMap, gamma: float) -> pg.ColorMap:
    """Gamma-warp a colormap: the displayed colour at normalised value ``t`` is
    the original colormap sampled at ``t**gamma``.

    ``gamma < 1`` brightens the mid-tones (bows the transfer curve up),
    ``gamma > 1`` darkens them. A no-op at ``gamma == 1``.
    """
    try:
        g = float(gamma)
    except (TypeError, ValueError):
        return cmap
    if not np.isfinite(g) or g <= 0.0 or abs(g - 1.0) < 1e-6:
        return cmap
    try:
        lut = cmap.getLookupTable(0.0, 1.0, 256).astype(np.float64)
        pts = np.linspace(0.0, 1.0, 256)
        warped = pts ** g
        out = np.empty_like(lut)
        for c in range(lut.shape[1]):
            out[:, c] = np.interp(warped, pts, lut[:, c])
        return pg.ColorMap(pts.astype(np.float32), out.astype(np.uint8))
    except Exception:
        return cmap


def make_colormap(name: str, *, invert: bool = False, gamma: float = 1.0) -> pg.ColorMap:
    """
    Build a ``pg.ColorMap`` by name.

    Tries pyqtgraph built-ins first, falls back to matplotlib, then to
    single-colour channel ramps, then to CET-L3 as a last resort. ``gamma``
    gamma-warps the resulting LUT (see :func:`apply_gamma`).
    """
    pts = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    key = name.lower().replace(" ", "_")

    # 1. Single-colour channels (black → channel colour)
    if name in _CHANNEL_RGB:
        r, g, b = _CHANNEL_RGB[name]
        rgba = np.zeros((256, 4), dtype=np.uint8)
        rgba[:, 0] = (pts * r * 255).astype(np.uint8)
        rgba[:, 1] = (pts * g * 255).astype(np.uint8)
        rgba[:, 2] = (pts * b * 255).astype(np.uint8)
        rgba[:, 3] = 255
        cmap = pg.ColorMap(pts, rgba)
    elif key == "glasbey":
        try:
            import colorcet as cc
            colors = cc.glasbey
            rgba = np.array([
                tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
                for c in colors[:256]
            ], dtype=np.ubyte)
            cmap = pg.ColorMap(np.linspace(0.0, 1.0, len(rgba)), rgba)
        except Exception:
            rgba = np.array([
                [230, 25, 75, 255], [60, 180, 75, 255], [255, 225, 25, 255],
                [0, 130, 200, 255], [245, 130, 48, 255], [145, 30, 180, 255],
                [70, 240, 240, 255], [240, 50, 230, 255], [210, 245, 60, 255],
                [250, 190, 190, 255], [0, 128, 128, 255], [230, 190, 255, 255],
            ], dtype=np.ubyte)
            cmap = pg.ColorMap(np.linspace(0.0, 1.0, len(rgba)), rgba)
    elif key == "hilo":
        rgba = np.array([[0, 0, 255, 255], [35, 35, 35, 255], [255, 255, 255, 255], [255, 0, 0, 255]], dtype=np.ubyte)
        cmap = pg.ColorMap(np.linspace(0.0, 1.0, len(rgba)), rgba)
    elif key == "parula":
        rgba = np.array([
            [53, 42, 135, 255], [15, 92, 221, 255], [18, 125, 216, 255],
            [7, 156, 207, 255], [21, 177, 180, 255], [89, 189, 140, 255],
            [165, 190, 107, 255], [225, 185, 82, 255], [252, 206, 46, 255],
            [249, 251, 14, 255],
        ], dtype=np.ubyte)
        cmap = pg.ColorMap(np.linspace(0.0, 1.0, len(rgba)), rgba)
    else:
        # 2. pyqtgraph built-ins
        cmap = None
        try:
            cmap = pg.colormap.get(name)
        except Exception:
            cmap = None
        # 3. matplotlib fallback
        if cmap is None:
            try:
                import matplotlib as mpl
                rgba = mpl.colormaps[name].resampled(256)(pts, bytes=True)
                cmap = pg.ColorMap(pts, rgba)
            except Exception:
                cmap = None
        # 4. last-resort
        if cmap is None:
            cmap = pg.colormap.get("CET-L3")

    if invert:
        cmap = _invert_cmap(cmap)
    cmap = apply_gamma(cmap, gamma)
    return cmap


def _invert_cmap(cmap: pg.ColorMap) -> pg.ColorMap:
    """Reverse a colormap's lookup table."""
    try:
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        lut = lut[::-1].copy()
        pts = np.linspace(0.0, 1.0, lut.shape[0], dtype=np.float32)
        return pg.ColorMap(pts, lut)
    except Exception:
        return cmap


# ---------------------------------------------------------------------------
# Gamma-draggable histogram viewbox
# ---------------------------------------------------------------------------

class _GammaViewBox(pg.ViewBox):
    """Histogram ViewBox where a left-drag **anywhere** re-fits gamma.

    Child items (the min/max level lines, the mid-value dot) get first crack at
    the drag — pyqtgraph offers a drag to the topmost item under the cursor and
    only falls back to the ViewBox when none claims it — so those keep working;
    a drag on empty plot area lands here and tilts the gamma transfer curve.
    """

    def __init__(self, on_drag: Callable[[float, float], None], **kwargs) -> None:
        super().__init__(enableMouse=False, **kwargs)
        self._on_drag = on_drag
        self.setMouseEnabled(x=False, y=False)

    def mouseDragEvent(self, ev, axis=None) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            p = self.mapSceneToView(ev.scenePos())
            self._on_drag(float(p.x()), float(p.y()))
        else:
            super().mouseDragEvent(ev, axis=axis)


# ---------------------------------------------------------------------------
# LutDialog
# ---------------------------------------------------------------------------

class LutDialog(QDialog):
    """
    Fiji-style LUT editor.

    Parameters
    ----------
    on_levels_changed : (lo, hi) -> None
    on_cmap_changed   : (name, invert) -> None
    on_invert_changed : (invert) -> None    # optional fast path
    on_reset          : () -> None
    """

    def __init__(
        self,
        on_levels_changed: Callable[[float, float], None],
        on_cmap_changed:   Callable[[str, bool], None],
        on_invert_changed: Callable[[bool], None] | None = None,
        on_reset:          Callable[[], None] | None = None,
        on_gamma_changed:  Callable[[float], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cb_levels  = on_levels_changed
        self._cb_cmap    = on_cmap_changed
        self._cb_invert  = on_invert_changed
        self._cb_reset   = on_reset
        self._cb_gamma   = on_gamma_changed
        self._gamma: float = 1.0

        self.setWindowTitle("LUT")
        # Non-modal so the user can adjust levels while watching the image
        self.setModal(False)
        self.resize(380, 420)

        # Data range (the two extremes the sliders can reach)
        self._data_lo: float = 0.0
        self._data_hi: float = 1.0
        # Current (lo, hi) values
        self._lo: float = 0.0
        self._hi: float = 1.0
        # State captured on open for Reset
        self._initial_state: dict | None = None
        # Current histogram values (for Auto computation)
        self._pixel_values: np.ndarray | None = None
        # Invert flag
        self._invert: bool = False
        # Histogram Y extent (for the gamma tilt line / control-dot placement)
        self._hist_ymax: float = 1.0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # ── Histogram preview ────────────────────────────────────
        # Custom viewbox so a drag on empty plot area re-fits gamma (the level
        # lines / mid dot still claim their own drags first).
        self._gamma_vb = _GammaViewBox(self._on_curve_dragged)
        self._hist_plot = pg.PlotWidget(background="#222", viewBox=self._gamma_vb)
        self._hist_plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._hist_plot.setFixedHeight(120)
        self._hist_plot.setMouseEnabled(x=False, y=False)
        self._hist_plot.setMenuEnabled(False)
        self._hist_plot.hideButtons()
        self._hist_plot.getPlotItem().hideAxis("left")
        self._hist_plot.getPlotItem().getAxis("bottom").setPen("#888")
        self._hist_curve = pg.PlotCurveItem(
            pen=pg.mkPen("#ccc", width=1), fillLevel=0,
            brush=pg.mkBrush(180, 180, 180, 180),
        )
        self._hist_plot.addItem(self._hist_curve)

        # Draggable vertical markers for the current min/max levels (like the
        # histogram filter bounds): drag them on the histogram to set the levels.
        self._lo_line = pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen("#3af", width=2),
            hoverPen=pg.mkPen("#8cf", width=3))
        self._hi_line = pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen("#f83", width=2),
            hoverPen=pg.mkPen("#fb7", width=3))
        self._lo_line.sigPositionChanged.connect(self._on_lo_line_moved)
        self._hi_line.sigPositionChanged.connect(self._on_hi_line_moved)
        # Gamma "tilt line": the transfer curve from (lo, 0) to (hi, top). Drag
        # anywhere on the plot to re-fit gamma (handled by _GammaViewBox).
        self._tf_curve = pg.PlotCurveItem(
            pen=pg.mkPen("#6c6", width=2, style=Qt.PenStyle.DashLine))
        self._tf_curve.setZValue(5)
        self._hist_plot.addItem(self._tf_curve)
        self._lo_line.setZValue(10)
        self._hi_line.setZValue(10)
        self._hist_plot.addItem(self._lo_line)
        self._hist_plot.addItem(self._hi_line)

        root.addWidget(self._hist_plot)

        # ── Colormap dropdown ────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(ALL_COLORMAPS)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        row.addWidget(self._cmap_combo, stretch=1)
        root.addLayout(row)

        # ── Sliders ──────────────────────────────────────────────
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        # Min
        grid.addWidget(QLabel("Minimum"), 0, 0)
        self._min_slider = QSlider(Qt.Orientation.Horizontal)
        self._min_slider.setRange(0, _SLIDER_RES)
        self._min_slider.valueChanged.connect(self._on_min_changed)
        grid.addWidget(self._min_slider, 0, 1)
        self._min_spin = QDoubleSpinBox()
        self._min_spin.setDecimals(3)
        self._min_spin.valueChanged.connect(self._on_min_spin)
        grid.addWidget(self._min_spin, 0, 2)

        # Max
        grid.addWidget(QLabel("Maximum"), 1, 0)
        self._max_slider = QSlider(Qt.Orientation.Horizontal)
        self._max_slider.setRange(0, _SLIDER_RES)
        self._max_slider.valueChanged.connect(self._on_max_changed)
        grid.addWidget(self._max_slider, 1, 1)
        self._max_spin = QDoubleSpinBox()
        self._max_spin.setDecimals(3)
        self._max_spin.valueChanged.connect(self._on_max_spin)
        grid.addWidget(self._max_spin, 1, 2)

        # Brightness
        grid.addWidget(QLabel("Brightness"), 2, 0)
        self._br_slider = QSlider(Qt.Orientation.Horizontal)
        self._br_slider.setRange(0, _SLIDER_RES)
        self._br_slider.setValue(_SLIDER_RES // 2)
        self._br_slider.valueChanged.connect(self._on_br_changed)
        grid.addWidget(self._br_slider, 2, 1, 1, 2)

        # Contrast
        grid.addWidget(QLabel("Contrast"), 3, 0)
        self._co_slider = QSlider(Qt.Orientation.Horizontal)
        self._co_slider.setRange(0, _SLIDER_RES)
        self._co_slider.setValue(_SLIDER_RES // 2)
        self._co_slider.valueChanged.connect(self._on_co_changed)
        grid.addWidget(self._co_slider, 3, 1, 1, 2)

        root.addLayout(grid)

        # ── Gamma ────────────────────────────────────────────────
        # Non-linear intensity mapping: drag the green dot on the histogram to
        # tilt the transfer curve, or set it precisely here.
        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Gamma"))
        self._gamma_spin = QDoubleSpinBox()
        self._gamma_spin.setDecimals(2)
        self._gamma_spin.setRange(0.10, 10.0)
        self._gamma_spin.setSingleStep(0.05)
        self._gamma_spin.setValue(1.0)
        self._gamma_spin.setToolTip(
            "Gamma correction. <1 brightens mid-tones, >1 darkens them. "
            "Drag the green dot on the histogram to tilt the transfer curve.")
        self._gamma_spin.valueChanged.connect(self._on_gamma_spin)
        gamma_row.addWidget(self._gamma_spin)
        gamma_reset = QPushButton("γ=1")
        gamma_reset.setToolTip("Reset gamma to 1 (linear).")
        gamma_reset.clicked.connect(lambda: self._set_gamma(1.0))
        gamma_row.addWidget(gamma_reset)
        gamma_row.addStretch()
        root.addLayout(gamma_row)

        # ── Action buttons ────────────────────────────────────────
        btn_row = QHBoxLayout()
        for label, cb in [
            ("Auto",   self._on_auto),
            ("Invert LUT", self._on_invert),
            ("Reset",  self._on_reset_clicked),
        ]:
            b = QPushButton(label)
            b.clicked.connect(cb)
            btn_row.addWidget(b)
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public: called from the render window
    # ------------------------------------------------------------------

    def load_image(
        self,
        pixels: np.ndarray,
        data_lo: float,
        data_hi: float,
        lo: float,
        hi: float,
        cmap_name: str,
        invert: bool,
        gamma: float = 1.0,
        capture_baseline: bool = True,
    ) -> None:
        """
        Populate the dialog with the current render image state.

        When *capture_baseline* is True (opening the dialog) this also captures
        the state as the "reset baseline". Live syncs from the owning window
        (colormap / brightness / color-by changed outside the dialog) pass
        ``capture_baseline=False`` so the Reset target isn't overwritten.
        """
        self._pixel_values = np.asarray(pixels).ravel()
        self._data_lo = float(data_lo)
        self._data_hi = float(data_hi if data_hi > data_lo else data_lo + 1.0)

        # Update spin-box ranges
        for spin in (self._min_spin, self._max_spin):
            spin.blockSignals(True)
            spin.setRange(self._data_lo - abs(self._data_lo),
                          self._data_hi + abs(self._data_hi) + 1.0)
            spin.blockSignals(False)
        # The draggable level lines are confined to the histogram's data range.
        # Block their signals: setBounds() re-clamps a line sitting outside the new
        # range and would emit sigPositionChanged with a STALE level, firing the
        # owner's level callback and corrupting its manual levels (opening the LUT
        # dialog then visibly re-scaled the plot — the "everything turns one colour"
        # bug). Levels are applied silently just below via _set_levels_silent.
        for _line in (self._lo_line, self._hi_line):
            _line.blockSignals(True)
            _line.setBounds((self._data_lo, self._data_hi))
            _line.blockSignals(False)

        # Record baseline for Reset (only when the dialog is (re)opened).
        if capture_baseline or self._initial_state is None:
            self._initial_state = dict(
                lo=float(lo), hi=float(hi),
                cmap=str(cmap_name), invert=bool(invert), gamma=float(gamma),
            )

        # Apply to widgets
        self._gamma = float(max(0.1, min(10.0, gamma)))
        self._gamma_spin.blockSignals(True)
        self._gamma_spin.setValue(self._gamma)
        self._gamma_spin.blockSignals(False)
        self._set_levels_silent(lo, hi)
        self._set_combo_silent(cmap_name)
        self._invert = bool(invert)
        self._update_histogram()

    # ------------------------------------------------------------------
    # Level / slider logic
    # ------------------------------------------------------------------

    def _set_levels_silent(self, lo: float, hi: float) -> None:
        """Set (lo, hi) and update sliders/spinboxes without firing callbacks."""
        self._lo = float(lo)
        self._hi = float(hi)
        self._sync_widgets_from_levels()

    def _sync_widgets_from_levels(self) -> None:
        blocked = (self._min_slider, self._max_slider, self._min_spin, self._max_spin,
                   self._br_slider, self._co_slider, self._lo_line, self._hi_line)
        for w in blocked:
            w.blockSignals(True)
        try:
            self._min_slider.setValue(self._level_to_slider(self._lo))
            self._max_slider.setValue(self._level_to_slider(self._hi))
            self._min_spin.setValue(self._lo)
            self._max_spin.setValue(self._hi)
            # Map (lo, hi) → (brightness, contrast) sliders
            mid  = 0.5 * (self._lo + self._hi)
            span = max(1e-12, self._hi - self._lo)
            data_span = max(1e-12, self._data_hi - self._data_lo)
            # Brightness: midpoint mapped to [0, slider_res], inverted
            #   slider=0    → midpoint = data_hi
            #   slider=res  → midpoint = data_lo
            br_frac = 1.0 - (mid - self._data_lo) / data_span
            br_frac = max(0.0, min(1.0, br_frac))
            self._br_slider.setValue(int(br_frac * _SLIDER_RES))
            # Contrast: span mapped [0, data_span] -> slider, inverted
            #   slider=0    → span = data_span   (low contrast, whole range)
            #   slider=res  → span = 0           (high contrast, zero width)
            co_frac = 1.0 - min(1.0, span / data_span)
            self._co_slider.setValue(int(co_frac * _SLIDER_RES))

            self._lo_line.setPos(self._lo)
            self._hi_line.setPos(self._hi)
        finally:
            for w in blocked:
                w.blockSignals(False)
        self._redraw_transfer_curve()                 # curve endpoints follow lo/hi

    def _level_to_slider(self, v: float) -> int:
        span = max(1e-12, self._data_hi - self._data_lo)
        frac = (v - self._data_lo) / span
        return int(max(0, min(_SLIDER_RES, frac * _SLIDER_RES)))

    def _slider_to_level(self, s: int) -> float:
        frac = s / _SLIDER_RES
        return self._data_lo + frac * (self._data_hi - self._data_lo)

    def _emit_levels(self) -> None:
        self._lo_line.blockSignals(True)
        self._hi_line.blockSignals(True)
        self._lo_line.setPos(self._lo)
        self._hi_line.setPos(self._hi)
        self._lo_line.blockSignals(False)
        self._hi_line.blockSignals(False)
        self._redraw_transfer_curve()
        self._cb_levels(self._lo, self._hi)

    # -- Draggable min/max lines (like the histogram filter bounds) --

    def _on_lo_line_moved(self) -> None:
        lo = float(self._lo_line.value())
        if lo > self._hi:
            lo = self._hi
        self._lo = max(self._data_lo, lo)
        self._sync_widgets_from_levels()              # re-clamps line, updates sliders/spins
        self._cb_levels(self._lo, self._hi)

    def _on_hi_line_moved(self) -> None:
        hi = float(self._hi_line.value())
        if hi < self._lo:
            hi = self._lo
        self._hi = min(self._data_hi, hi)
        self._sync_widgets_from_levels()
        self._cb_levels(self._lo, self._hi)

    # -- Gamma tilt line -------------------------------------------

    def _redraw_transfer_curve(self) -> None:
        """Draw the (lo,0)→(hi,top) transfer curve for the current gamma and put
        the draggable control dot at its mid-value."""
        ymax = max(1e-9, float(self._hist_ymax))
        lo, hi = float(self._lo), float(self._hi)
        if hi <= lo:
            self._tf_curve.setData([], [])
            return
        xs = np.linspace(lo, hi, 64)
        t = np.clip((xs - lo) / (hi - lo), 0.0, 1.0)
        self._tf_curve.setData(xs, ymax * np.power(t, self._gamma))

    def _on_curve_dragged(self, x: float, y: float) -> None:
        """Whole-line drag: re-fit gamma so the curve passes through (x, y)."""
        ymax = max(1e-9, float(self._hist_ymax))
        lo, hi = float(self._lo), float(self._hi)
        if hi <= lo:
            return
        t = (x - lo) / (hi - lo)
        t = min(max(t, 0.02), 0.98)                       # ignore the pinned ends
        frac = min(max(y / ymax, 0.02), 0.98)
        gamma = float(np.log(frac) / np.log(t))           # frac = t**gamma
        self._set_gamma(gamma)

    def _on_gamma_spin(self, value: float) -> None:
        self._set_gamma(float(value))

    def _set_gamma(self, gamma: float) -> None:
        gamma = float(max(0.1, min(10.0, gamma)))
        self._gamma = gamma
        self._gamma_spin.blockSignals(True)
        self._gamma_spin.setValue(gamma)
        self._gamma_spin.blockSignals(False)
        self._redraw_transfer_curve()                  # snaps the dot's x back to mid
        if self._cb_gamma is not None:
            self._cb_gamma(self._gamma)

    # -- Slider/spin handlers ---------------------------------------

    def _on_min_changed(self, s: int) -> None:
        lo = self._slider_to_level(s)
        if lo > self._hi:
            lo = self._hi
        self._lo = lo
        self._min_spin.blockSignals(True); self._min_spin.setValue(lo); self._min_spin.blockSignals(False)
        self._emit_levels()

    def _on_min_spin(self, v: float) -> None:
        self._lo = min(v, self._hi)
        self._min_slider.blockSignals(True); self._min_slider.setValue(self._level_to_slider(self._lo)); self._min_slider.blockSignals(False)
        self._emit_levels()

    def _on_max_changed(self, s: int) -> None:
        hi = self._slider_to_level(s)
        if hi < self._lo:
            hi = self._lo
        self._hi = hi
        self._max_spin.blockSignals(True); self._max_spin.setValue(hi); self._max_spin.blockSignals(False)
        self._emit_levels()

    def _on_max_spin(self, v: float) -> None:
        self._hi = max(v, self._lo)
        self._max_slider.blockSignals(True); self._max_slider.setValue(self._level_to_slider(self._hi)); self._max_slider.blockSignals(False)
        self._emit_levels()

    def _on_br_changed(self, s: int) -> None:
        """Move midpoint up/down keeping span."""
        data_span = max(1e-12, self._data_hi - self._data_lo)
        br_frac = s / _SLIDER_RES
        mid = self._data_lo + (1.0 - br_frac) * data_span
        span = self._hi - self._lo
        self._lo = mid - span / 2
        self._hi = mid + span / 2
        self._sync_widgets_from_levels()
        self._emit_levels()

    def _on_co_changed(self, s: int) -> None:
        """Change span keeping midpoint."""
        data_span = max(1e-12, self._data_hi - self._data_lo)
        co_frac = s / _SLIDER_RES
        span = (1.0 - co_frac) * data_span
        mid = 0.5 * (self._lo + self._hi)
        self._lo = mid - span / 2
        self._hi = mid + span / 2
        self._sync_widgets_from_levels()
        self._emit_levels()

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _on_auto(self) -> None:
        """Fiji-style auto: leave 0.35 % of pixels saturated at each end."""
        if self._pixel_values is None or self._pixel_values.size == 0:
            return
        # Drop zeros from histogram estimation (typical for empty pixels
        # in a rendered SMLM image)
        vals = self._pixel_values
        vals = vals[vals > 0] if np.any(vals > 0) else vals
        if vals.size == 0:
            return
        lo, hi = np.percentile(vals, [0.35, 99.65])
        self._lo, self._hi = float(lo), float(hi)
        if self._hi <= self._lo:
            self._hi = self._lo + 1.0
        self._sync_widgets_from_levels()
        self._emit_levels()

    def _on_invert(self) -> None:
        self._invert = not self._invert
        if self._cb_invert is not None:
            self._cb_invert(self._invert)
        else:
            # Fallback — re-emit cmap with invert flag
            self._cb_cmap(self._cmap_combo.currentText(), self._invert)

    def _on_reset_clicked(self) -> None:
        if self._initial_state is None:
            return
        # Reset display range to the current image's full data range + linear gamma.
        self._set_levels_silent(self._data_lo, self._data_hi)
        self._set_gamma(1.0)
        self._emit_levels()

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _on_cmap_changed(self, name: str) -> None:
        self._cb_cmap(name, self._invert)

    def _set_combo_silent(self, name: str) -> None:
        self._cmap_combo.blockSignals(True)
        i = self._cmap_combo.findText(name, Qt.MatchFlag.MatchFixedString)
        if i >= 0:
            self._cmap_combo.setCurrentIndex(i)
        self._cmap_combo.blockSignals(False)

    def _update_histogram(self) -> None:
        """Draw a simplified histogram on the preview plot."""
        if self._pixel_values is None or self._pixel_values.size == 0:
            self._hist_curve.setData([], [])
            self._hist_ymax = 1.0
            self._redraw_transfer_curve()
            return
        vals = self._pixel_values
        nz = vals[vals > 0] if np.any(vals > 0) else vals
        if nz.size == 0:
            nz = vals
        h, edges = np.histogram(nz, bins=128,
                                range=(self._data_lo, self._data_hi))
        xs = 0.5 * (edges[:-1] + edges[1:])
        # Log-compress the counts so the histogram shape is readable
        ys = np.log1p(h)
        self._hist_curve.setData(xs, ys)
        self._hist_plot.setXRange(self._data_lo, self._data_hi, padding=0.02)
        self._hist_ymax = max(1.0, float(ys.max()) * 1.05)
        self._hist_plot.setYRange(0, self._hist_ymax)
        self._redraw_transfer_curve()                 # gamma tilt line follows the histogram
