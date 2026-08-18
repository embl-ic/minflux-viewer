"""
minflux_viewer.ui.lut_dialog
=============================
Fiji-style LUT (Look-Up Table) editor.

Shows for the currently active render image:

* a **mini histogram** of the currently rendered pixel values
  (log-y, simplified — just enough to pick sensible min/max)
* an application-owned **colormap dropdown**, including persistent custom maps
* four **sliders** — Minimum, Maximum, Brightness, Contrast — matching
  Fiji's B&C behaviour
* **Auto** — ImageJ-style repeated-click cycle that progressively increases
  contrast and restarts from the full-range automatic threshold when exhausted
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

from typing import TYPE_CHECKING, Callable

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
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..colormaps import (
    channel_colormap_names,
    custom_colormap_stops,
    delete_custom_colormap,
    is_custom_colormap,
    make_colormap,
    store_custom_colormap,
)

if TYPE_CHECKING:
    from ..core.app_state import AppState


# ---------------------------------------------------------------------------
# Colormap options are resolved dynamically so newly saved custom maps appear
# without restarting the application.
# ---------------------------------------------------------------------------

ALL_COLORMAPS: list[str] = channel_colormap_names()


_SLIDER_RES = 1000
_IMAGEJ_AUTO_THRESHOLD = 5000
_IMAGEJ_AUTO_RESET_THRESHOLD = 10
_IMAGEJ_AUTO_HIST_BINS = 256


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
    on_auto           : () -> (lo, hi) | None  # optional owner-managed cycle
    """

    def __init__(
        self,
        on_levels_changed: Callable[[float, float], None],
        on_cmap_changed:   Callable[[str, bool], None],
        on_invert_changed: Callable[[bool], None] | None = None,
        on_reset:          Callable[[], None] | None = None,
        on_gamma_changed:  Callable[[float], None] | None = None,
        parent: QWidget | None = None,
        on_auto: Callable[[], tuple[float, float] | None] | None = None,
        state: "AppState | None" = None,
    ) -> None:
        super().__init__(parent)
        self._cb_levels  = on_levels_changed
        self._cb_cmap    = on_cmap_changed
        self._cb_invert  = on_invert_changed
        self._cb_reset   = on_reset
        self._cb_gamma   = on_gamma_changed
        self._cb_auto    = on_auto
        self._state      = state
        self._gamma: float = 1.0
        self._auto_threshold: int = 0

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

    def rebind(
        self,
        on_levels_changed: Callable[[float, float], None],
        on_cmap_changed:   Callable[[str, bool], None],
        on_invert_changed: Callable[[bool], None] | None = None,
        on_reset:          Callable[[], None] | None = None,
        on_gamma_changed:  Callable[[float], None] | None = None,
        on_auto: Callable[[], tuple[float, float] | None] | None = None,
        state: "AppState | None" = None,
    ) -> None:
        """Point this dialog at a different view without rebuilding it.

        There is one LUT dialog application-wide (see :func:`shared_lut_dialog`)
        and it follows the focused view, so only the callbacks change — the
        widgets, and the window's position on screen, are kept.
        """
        self._cb_levels = on_levels_changed
        self._cb_cmap   = on_cmap_changed
        self._cb_invert = on_invert_changed
        self._cb_reset  = on_reset
        self._cb_gamma  = on_gamma_changed
        self._cb_auto   = on_auto
        if state is not None:
            self._state = state

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
        self._cmap_combo.addItems(channel_colormap_names())
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        row.addWidget(self._cmap_combo, stretch=1)
        self._custom_cmap_button = QToolButton()
        self._custom_cmap_button.setText("Custom")
        self._custom_cmap_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )
        custom_menu = QMenu(self._custom_cmap_button)
        self._create_custom_cmap_action = custom_menu.addAction(
            "Create custom colormap…"
        )
        self._create_custom_cmap_action.triggered.connect(
            self._create_custom_colormap
        )
        self._edit_custom_cmap_action = custom_menu.addAction(
            "Edit current custom colormap…"
        )
        self._edit_custom_cmap_action.triggered.connect(
            self._edit_custom_colormap
        )
        self._delete_custom_cmap_action = custom_menu.addAction(
            "Delete current custom colormap…"
        )
        self._delete_custom_cmap_action.triggered.connect(
            self._delete_custom_colormap
        )
        custom_menu.aboutToShow.connect(self._update_custom_colormap_actions)
        self._custom_cmap_button.setMenu(custom_menu)
        self._custom_cmap_button.setEnabled(self._state is not None)
        row.addWidget(self._custom_cmap_button)
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
        self._auto_btn = QPushButton("Auto")
        self._auto_btn.setCheckable(True)
        self._auto_btn.setToolTip(
            "ImageJ-style automatic display range. Repeated clicks increase "
            "clipping; when no valid range remains, the sequence resets and "
            "starts again."
        )
        self._auto_btn.clicked.connect(self._on_auto)
        btn_row.addWidget(self._auto_btn)
        invert_btn = QPushButton("Invert LUT")
        invert_btn.clicked.connect(self._on_invert)
        btn_row.addWidget(invert_btn)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._on_reset_clicked)
        btn_row.addWidget(reset_btn)
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
        # dialog then visibly re-scaled the plot — the "everything turns one color"
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
        if capture_baseline and self._cb_auto is None:
            self._auto_threshold = 0
            self.set_auto_state(False)

    def set_auto_state(self, checked: bool) -> None:
        """Reflect whether automatic level selection is currently active."""
        self._auto_btn.blockSignals(True)
        self._auto_btn.setChecked(bool(checked))
        self._auto_btn.blockSignals(False)

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

    def _emit_levels(self, *, manual: bool = True) -> None:
        if manual:
            self._auto_threshold = 0
            self.set_auto_state(False)
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
        self._auto_threshold = 0
        self.set_auto_state(False)
        self._cb_levels(self._lo, self._hi)

    def _on_hi_line_moved(self) -> None:
        hi = float(self._hi_line.value())
        if hi < self._lo:
            hi = self._lo
        self._hi = min(self._data_hi, hi)
        self._sync_widgets_from_levels()
        self._auto_threshold = 0
        self.set_auto_state(False)
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
        """Run the same repeated-click ImageJ Auto cycle as the B/C palette."""
        if self._cb_auto is not None:
            levels = self._cb_auto()
            if levels is None:
                self.set_auto_state(False)
                return
            self._set_levels_silent(*levels)
            self.set_auto_state(True)
            return

        levels = self._compute_imagej_auto_levels()
        if levels is None:
            self.set_auto_state(False)
            return
        self._lo, self._hi = levels
        self._sync_widgets_from_levels()
        self.set_auto_state(True)
        self._emit_levels(manual=False)

    def _compute_imagej_auto_levels(self) -> tuple[float, float] | None:
        if self._pixel_values is None or self._pixel_values.size == 0:
            return None
        values = np.asarray(self._pixel_values, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size < 10:
            return None
        data_min = float(values.min())
        data_max = float(values.max())
        if data_max <= data_min:
            return data_min, data_min + 1.0

        if self._auto_threshold < _IMAGEJ_AUTO_RESET_THRESHOLD:
            self._auto_threshold = _IMAGEJ_AUTO_THRESHOLD
        else:
            self._auto_threshold //= 2

        histogram, _edges = np.histogram(
            values,
            bins=_IMAGEJ_AUTO_HIST_BINS,
            range=(data_min, data_max),
        )
        pixel_count = int(values.size)
        limit = pixel_count // 10
        threshold = pixel_count // self._auto_threshold

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
            self._auto_threshold = 0
            return data_min, data_max

        bin_size = (data_max - data_min) / float(_IMAGEJ_AUTO_HIST_BINS)
        lo = data_min + hmin * bin_size
        hi = data_min + hmax * bin_size
        if hi <= lo:
            lo, hi = data_min, data_max
        if hi <= lo:
            hi = lo + 1.0
        return float(lo), float(hi)

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
        self._auto_threshold = 0
        self.set_auto_state(False)
        # Reset display range to the current image's full data range + linear gamma.
        self._set_levels_silent(self._data_lo, self._data_hi)
        self._set_gamma(1.0)
        self._emit_levels()

    # ------------------------------------------------------------------
    # Persistent custom colormaps
    # ------------------------------------------------------------------

    def _update_custom_colormap_actions(self) -> None:
        editable = is_custom_colormap(self._cmap_combo.currentText())
        self._edit_custom_cmap_action.setEnabled(editable)
        self._delete_custom_cmap_action.setEnabled(editable)

    def _refresh_colormap_combo(self, select: str | None = None) -> None:
        current = select or self._cmap_combo.currentText() or "hot"
        names = channel_colormap_names()
        # Hidden compatibility maps are not offered to new users, but an old
        # saved selection must remain visible while it is active.
        if current not in names:
            try:
                make_colormap(current)
            except (KeyError, ValueError):
                current = "hot"
            else:
                names.append(current)
        self._cmap_combo.blockSignals(True)
        self._cmap_combo.clear()
        self._cmap_combo.addItems(names)
        self._cmap_combo.setCurrentText(current)
        self._cmap_combo.blockSignals(False)

    def _save_custom_colormap_dialog(self, dialog) -> None:
        if self._state is None:
            return
        try:
            name = store_custom_colormap(
                self._state.prefs,
                dialog.result_name(),
                dialog.result_stops(),
                replacing=dialog.replacing_name,
            )
            self._state.save_prefs()
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, dialog.windowTitle(), str(exc))
            return
        self._refresh_colormap_combo(name)
        self._cb_cmap(name, self._invert)

    def _create_custom_colormap(self) -> None:
        if self._state is None:
            return
        from .custom_colormap_dialog import CustomColormapDialog

        dialog = CustomColormapDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._save_custom_colormap_dialog(dialog)

    def _edit_custom_colormap(self) -> None:
        current = self._cmap_combo.currentText()
        if self._state is None or not is_custom_colormap(current):
            return
        from .custom_colormap_dialog import CustomColormapDialog

        dialog = CustomColormapDialog(
            self, name=current, stops=custom_colormap_stops(current)
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._save_custom_colormap_dialog(dialog)

    def _delete_custom_colormap(self) -> None:
        current = self._cmap_combo.currentText()
        if self._state is None or not is_custom_colormap(current):
            return
        answer = QMessageBox.question(
            self,
            "Delete custom colormap",
            f"Delete the custom colormap ‘{current}’ from the application?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if delete_custom_colormap(self._state.prefs, current):
            self._state.save_prefs()
        self._refresh_colormap_combo("hot")
        self._cb_cmap("hot", self._invert)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _on_cmap_changed(self, name: str) -> None:
        self._cb_cmap(name, self._invert)

    def _set_combo_silent(self, name: str) -> None:
        if self._cmap_combo.findText(name, Qt.MatchFlag.MatchFixedString) < 0:
            self._refresh_colormap_combo(name)
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


# ---------------------------------------------------------------------------
# One LUT dialog, application-wide
# ---------------------------------------------------------------------------

def shared_lut_dialog(owner, **callbacks) -> "LutDialog":
    """The single application-wide LUT dialog, rebound to *owner*.

    One instance rather than one per view: the previous design left a hidden
    dialog behind for every view that had ever opened one (they outlived the
    main window), and keeping only one visible meant hiding the rest — which
    could not guarantee the invariant, because ``close()`` may be refused.

    The dialog is deliberately **parentless**, matching the project's modeless
    convention, so it is not destroyed with whichever view happened to create
    it.  ``MainWindow`` closes it on shutdown.
    """
    state = getattr(owner, "_state", None)
    dialog = getattr(state, "_shared_lut_dialog", None)
    try:
        alive = dialog is not None and dialog.objectName() is not None
    except RuntimeError:                      # C++ side already gone
        alive = False
        dialog = None

    if not alive:
        dialog = LutDialog(**callbacks)
        if state is not None:
            state._shared_lut_dialog = dialog
    else:
        dialog.rebind(**callbacks)

    # Only the current owner may hold the reference the sync helpers read.
    previous = getattr(state, "_shared_lut_owner", None)
    if previous is not None and previous is not owner:
        try:
            previous._lut_dialog = None
        except (AttributeError, RuntimeError):
            pass
    if state is not None:
        state._shared_lut_owner = owner
    owner._lut_dialog = dialog
    return dialog


def close_shared_lut_dialog(state) -> None:
    """Shut the one LUT dialog down (application exit)."""
    dialog = getattr(state, "_shared_lut_dialog", None)
    if dialog is None:
        return
    try:
        dialog.close()
        dialog.deleteLater()
    except RuntimeError:
        pass
    state._shared_lut_dialog = None
    owner = getattr(state, "_shared_lut_owner", None)
    if owner is not None:
        try:
            owner._lut_dialog = None
        except (AttributeError, RuntimeError):
            pass
    state._shared_lut_owner = None
