"""The time row: a range gate on ``tim``, built to be the depth row's sibling.

Render and Scatter gate the drawn rows by the out-of-plane coordinate with
:class:`~minflux_viewer.ui.render_window.DepthRangeSlider`.  Time is the same
kind of thing on a different per-localization scalar, so it reuses that exact
widget rather than inventing a second slider idiom: the two rows look alike,
drag alike and wheel alike, and a user who has used one has used both.

What time adds over depth is *playback*.  A depth slab is normally placed once;
a time window is walked, so the row carries previous / play / next and an
options dialog holding the window width, the step, the rate and the axis mode.

The widget owns no data: the host supplies a
:class:`~minflux_viewer.core.tracking_time.TimeAxis` and asks for the gate back.
``row_mask(values)`` is the only way the gate reaches a render path, so every
view applies it identically.

⚠ The row asks for the axis's *interval statistics* separately, through
:attr:`TimeAxisRow.diagnosticsRequested`, and only once the gate is engaged.
Measuring them is a lexsort over every row -- 4.50 s of the 5.92 s a full axis
costs at 20,000,000 -- so a view whose time row is never touched must not pay
it.  Until then the step falls back to ``span / 200``.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFontInfo
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.tracking_time import (
    TIME_MODE_LABELS,
    TIME_MODE_SHORT,
    TIME_MODE_TIPS,
    TIME_MODES,
    TimeAxis,
    format_time_seconds,
)

_BUTTON_SIZE = (24, 24)
_PLAY_GLYPH = "▶"       # BLACK RIGHT-POINTING TRIANGLE
_PAUSE_GLYPH = "■"      # BLACK SQUARE -- one solid block, like the rotation pane
# The step buttons are CHEVRONS, not triangles. With all three drawn as filled
# triangles, play and next were indistinguishable at 24 px -- and the middle
# button is the only latching one of the three, so it has to look different.
_PREV_GLYPH = "‹"       # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
_NEXT_GLYPH = "›"       # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
_GLYPH_SCALE = {
    _PLAY_GLYPH: 0.85, _PAUSE_GLYPH: 0.72, _PREV_GLYPH: 1.7, _NEXT_GLYPH: 1.7,
}

#: Default playback rate, in window steps per second.
DEFAULT_RATE_HZ = 10.0
#: The window opens this fraction of the axis wide.
DEFAULT_WINDOW_FRACTION = 0.05
# The row shares its window with the plot, so it asks for as little width as it
# can: on the scatter's ortho grid every pixel it takes comes out of the
# rotating pane's controls in the cell below.
_MODE_COMBO_MAX_WIDTH = 96
_READ_OUT_MIN_WIDTH = 116


def _scaled_glyph_font(button: QToolButton, base: float, glyph: str):
    font = button.font()
    font.setPointSizeF(max(5.0, base * _GLYPH_SCALE.get(glyph, 1.0)))
    return font


class TimeRangeDialog(QDialog):
    """Options behind the time slider: extent, window, step, rate and mode.

    The axis extent is shown but not editable: unlike a depth slab, the bounds
    are the acquisition's own, and typing a wider one would claim time the data
    does not cover.  What is editable is how the window moves through them.
    """

    def __init__(
        self,
        axis: TimeAxis,
        current: tuple[float, float],
        *,
        step: float,
        rate_hz: float,
        loop: bool,
        grow: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Time range view option")
        self.setModal(True)
        self._axis = axis

        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        decimals = 0 if axis.is_index else 9
        # The editable fields are raw axis values, so they carry their unit: the
        # compact "12 ms" form the read-out uses is not something to type back in.
        suffix = "" if axis.is_index else " s"
        span = max(axis.span, 1e-12)

        self._mode_combo = QComboBox()
        for mode in TIME_MODES:
            self._mode_combo.addItem(TIME_MODE_LABELS[mode], mode)
            self._mode_combo.setItemData(
                self._mode_combo.count() - 1,
                TIME_MODE_TIPS[mode],
                Qt.ItemDataRole.ToolTipRole,
            )
        index = self._mode_combo.findData(axis.mode)
        if index >= 0:
            self._mode_combo.setCurrentIndex(index)
        form.addRow("Axis:", self._mode_combo)

        extent = QLabel(axis.format_range(axis.lo, axis.hi))
        extent.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("Data extent:", extent)

        self._lo_spin = QDoubleSpinBox()
        self._lo_spin.setDecimals(decimals)
        self._lo_spin.setSuffix(suffix)
        self._lo_spin.setRange(axis.lo, axis.hi)
        self._lo_spin.setValue(float(current[0]))
        form.addRow("Window start:", self._lo_spin)

        self._width_spin = QDoubleSpinBox()
        self._width_spin.setDecimals(decimals)
        self._width_spin.setSuffix(suffix)
        self._width_spin.setRange(0.0, span)
        self._width_spin.setValue(max(float(current[1]) - float(current[0]), 0.0))
        self._width_spin.setToolTip(
            "How much of the axis is drawn at once. Set it to the full extent "
            "with 'All' on the row instead of typing the span here."
        )
        form.addRow("Window width:", self._width_spin)

        self._step_spin = QDoubleSpinBox()
        self._step_spin.setDecimals(decimals)
        self._step_spin.setSuffix(suffix)
        self._step_spin.setRange(0.0, span)
        self._step_spin.setValue(max(float(step), 0.0))
        measured = (
            "n/a" if axis.interval is None
            else (f"{axis.interval:.0f}" if axis.is_index
                  else format_time_seconds(axis.interval))
        )
        self._step_spin.setToolTip(
            "How far one previous/next click, one wheel notch or one playback "
            f"tick moves the window. Measured modal interval: {measured}."
        )
        form.addRow("Step:", self._step_spin)

        self._rate_spin = QDoubleSpinBox()
        self._rate_spin.setDecimals(1)
        self._rate_spin.setRange(0.1, 120.0)
        self._rate_spin.setValue(max(float(rate_hz), 0.1))
        self._rate_spin.setSuffix(" steps/s")
        form.addRow("Playback rate:", self._rate_spin)

        self._loop_check = QCheckBox("Return to the start at the end")
        self._loop_check.setChecked(bool(loop))
        form.addRow("", self._loop_check)

        self._grow_check = QCheckBox("Grow from the start instead of sliding")
        self._grow_check.setChecked(bool(grow))
        self._grow_check.setToolTip(
            "Hold the window's start where it is and move only its end, so the "
            "drawn set accumulates -- the cumulative trail of a whole track "
            "rather than a moving slice of it."
        )
        form.addRow("", self._grow_check)

        note = QLabel(axis.describe())
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def values(self) -> dict:
        lo = float(self._lo_spin.value())
        hi = min(lo + float(self._width_spin.value()), float(self._axis.hi))
        return {
            "mode": str(self._mode_combo.currentData() or self._axis.mode),
            "range": (lo, hi),
            "step": float(self._step_spin.value()),
            "rate_hz": float(self._rate_spin.value()),
            "loop": bool(self._loop_check.isChecked()),
            "grow": bool(self._grow_check.isChecked()),
        }


class TimeAxisRow(QWidget):
    """``[All] T: [====slider====] [prev][play][next] [mode] read-out``.

    The slider sits immediately after the label so it starts at the same x as
    the depth row's -- the two then read as one control on two axes, which is
    the whole argument for reusing the widget.

    Emits :attr:`rangeChanged` whenever the gate changes (including when ``All``
    is toggled, where the gate becomes "everything"), and :attr:`modeChanged`
    when the axis itself changes, which the host answers by rebuilding its
    per-row time values.
    """

    rangeChanged = pyqtSignal()
    modeChanged = pyqtSignal(str)
    #: The row now wants the measured interval statistics, which it cannot
    #: compute itself. The host answers by re-supplying the axis with them.
    diagnosticsRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._axis: TimeAxis | None = None
        #: A window restored before any axis was supplied. The slider's limits
        #: are still the placeholder (0, 1) then, so applying it directly would
        #: clamp it into that range and the first set_axis would reset it.
        self._pending_range: tuple[float, float] | None = None
        self._step = 0.0
        self._rate_hz = DEFAULT_RATE_HZ
        self._loop = True
        self._grow = False
        self._range_initialized = False
        self._unavailable_reason = ""
        #: Whether the gate has ever been engaged. Until it has, the axis is
        #: built without its interval statistics -- 4.5 s of the 5.9 s total on a
        #: 20 M-row acquisition -- and the step falls back to span/200.
        self._want_diagnostics = False

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._all_check = QCheckBox("All")
        self._all_check.setChecked(True)
        self._all_check.setToolTip("Draw the whole time axis at once")
        self._all_check.toggled.connect(self._on_all_toggled)
        row.addWidget(self._all_check)

        self._axis_label = QLabel("T:")
        self._axis_label.setMinimumWidth(22)
        row.addWidget(self._axis_label)

        from .render_window import DepthRangeSlider

        self._slider = DepthRangeSlider()
        self._slider.setEnabled(False)
        self._slider.setToolTip(
            "Drag the window edges; wheel moves it by one step; double-click "
            "for the axis, window width, step and playback rate"
        )
        self._slider.rangeChanged.connect(self._on_slider_range_changed)
        self._slider.doubleClicked.connect(self.show_options_dialog)
        row.addWidget(self._slider, stretch=1)

        self._prev_button = self._glyph_button(
            _PREV_GLYPH, "Step the window back", self._step_back
        )
        row.addWidget(self._prev_button)

        self._play_button = self._glyph_button(_PLAY_GLYPH, "Play", None)
        self._play_button.setCheckable(True)
        self._play_button.toggled.connect(self.set_playing)
        row.addWidget(self._play_button)

        self._next_button = self._glyph_button(
            _NEXT_GLYPH, "Step the window forward", self._step_forward
        )
        row.addWidget(self._next_button)

        self._mode_combo = QComboBox()
        for mode in TIME_MODES:
            self._mode_combo.addItem(TIME_MODE_SHORT[mode], mode)
            self._mode_combo.setItemData(
                self._mode_combo.count() - 1,
                TIME_MODE_TIPS[mode],
                Qt.ItemDataRole.ToolTipRole,
            )
        self._mode_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._mode_combo.setMaximumWidth(_MODE_COMBO_MAX_WIDTH)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_combo_changed)
        row.addWidget(self._mode_combo)

        self._read_out = QLabel("all")
        self._read_out.setMinimumWidth(_READ_OUT_MIN_WIDTH)
        self._read_out.setStyleSheet("color: gray; font-size: 11px;")
        row.addWidget(self._read_out)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._sync_enabled()

    # ------------------------------------------------------------------ build

    def _glyph_button(self, glyph: str, tip: str, slot) -> QToolButton:
        button = QToolButton()
        button.setFixedSize(*_BUTTON_SIZE)
        base = button.font().pointSizeF()
        if base <= 0.0:                      # a font set in pixels reports -1
            base = QFontInfo(button.font()).pointSizeF()
        button._mfv_base_point_size = float(base) if base > 0.0 else 9.0
        button.setFont(_scaled_glyph_font(button, button._mfv_base_point_size, glyph))
        button.setText(glyph)
        button.setToolTip(tip)
        button.setAccessibleName(tip)
        if slot is not None:
            button.clicked.connect(slot)
        return button

    # ------------------------------------------------------------------- axis

    def set_axis(self, axis: TimeAxis | None, *, reason: str = "") -> None:
        """Adopt *axis*, or disable the row and say why when it is ``None``.

        The window width and position are kept across a re-supply where the new
        extent allows it, so a filter change or a re-render does not throw away
        where the user was looking -- the same contract the depth row keeps.
        """
        previous = self._axis
        self._axis = axis
        self._unavailable_reason = str(reason or "")
        if axis is None:
            self.stop()
            self._range_initialized = False
            self._sync_enabled()
            self._update_read_out()
            return

        self._sync_mode_combo(axis.mode)
        measurement_arrived = (
            previous is not None
            and previous.interval is None
            and axis.interval is not None
        )
        if (previous is None or previous.mode != axis.mode or self._step <= 0.0
                or measurement_arrived):
            # A new axis is in new units, so a step carried over from the old one
            # would be meaningless -- unlike the rate, loop and grow settings,
            # which are about how the window moves rather than how far.
            #
            # ⚠ ``measurement_arrived`` is the case that engaging the gate
            # creates: the axis was built without its interval statistics and the
            # step was the span/200 fallback, and the measured interval has just
            # replaced it. Keeping the fallback would mean the user asked for the
            # measurement and went on stepping by a made-up number. A step typed
            # into the options dialog is safe, because that dialog forces the
            # measurement first, so ``previous.interval`` is already set.
            self._step = float(axis.step)
        restored = self._pending_range
        self._pending_range = None
        lo, hi = restored if restored is not None else self._slider.range()
        keep = restored is not None or (
            self._range_initialized
            and previous is not None
            and previous.mode == axis.mode
        )
        self._slider.set_limits(axis.lo, axis.hi, reset_range=not keep)
        self._slider.set_scroll_options(self._step, False)
        if keep:
            self._range_initialized = True
            width = min(max(hi - lo, 0.0), axis.span)
            lo = min(max(lo, axis.lo), axis.hi - width)
            self._slider.set_range(lo, lo + width)
        elif not self._all_check.isChecked():
            self._set_default_window()
        self._sync_enabled()
        self._update_read_out()

    def set_available_modes(self, modes) -> None:
        """Enable only *modes*, and move off a mode this dataset cannot supply.

        An unavailable mode is disabled with its reason rather than removed, so
        the row always lists the same three entries and a dataset without ``tim``
        shows *why* absolute time is not on offer instead of appearing not to
        have the feature.
        """
        allowed = tuple(mode for mode in TIME_MODES if mode in set(modes or ()))
        model = self._mode_combo.model()
        for index in range(self._mode_combo.count()):
            mode = str(self._mode_combo.itemData(index) or "")
            enabled = mode in allowed
            item = model.item(index) if hasattr(model, "item") else None
            if item is not None:
                item.setEnabled(enabled)
            if not enabled:
                self._mode_combo.setItemData(
                    index,
                    f"{TIME_MODE_TIPS.get(mode, '')} "
                    "Unavailable: this dataset does not carry the attribute it "
                    "needs.".strip(),
                    Qt.ItemDataRole.ToolTipRole,
                )
            else:
                self._mode_combo.setItemData(
                    index, TIME_MODE_TIPS.get(mode, ""), Qt.ItemDataRole.ToolTipRole
                )
        if allowed and self.mode() not in allowed:
            self._sync_mode_combo(allowed[0])
            self._range_initialized = False

    def axis(self) -> TimeAxis | None:
        return self._axis

    def wants_diagnostics(self) -> bool:
        """Whether the host should measure the interval when supplying the axis."""
        return self._want_diagnostics

    def _request_diagnostics(self) -> None:
        if self._want_diagnostics:
            return
        self._want_diagnostics = True
        self.diagnosticsRequested.emit()

    def mode(self) -> str:
        return str(self._mode_combo.currentData() or TIME_MODES[0])

    def is_available(self) -> bool:
        return self._axis is not None

    def is_active(self) -> bool:
        """The gate narrows the drawn rows (an axis exists and ``All`` is off)."""
        return self._axis is not None and not self._all_check.isChecked()

    def range(self) -> tuple[float, float] | None:
        """The active window, or ``None`` when the whole axis is drawn."""
        if not self.is_active():
            return None
        lo, hi = self._slider.range()
        return float(lo), float(hi)

    def range_key(self) -> tuple[float, float] | None:
        """A rounded, hashable form of :meth:`range` for a tile-cache key."""
        window = self.range()
        if window is None:
            return None
        return (round(window[0], 12), round(window[1], 12))

    def set_window(self, lo: float, hi: float) -> None:
        """Engage the gate on ``[lo, hi]``, as an opening state rather than a nudge.

        Used by a host that has a meaningful window to start from -- the Tracking
        View opens on a comet-length tail rather than on the whole axis. It asks
        for the interval statistics like any other engagement, because the step
        it hands to playback has to be the measured one.
        """
        if self._axis is None:
            self._pending_range = (float(lo), float(hi))
            return
        self._request_diagnostics()
        if self._all_check.isChecked():
            self._all_check.blockSignals(True)
            self._all_check.setChecked(False)
            self._all_check.blockSignals(False)
        self._slider.set_range(float(lo), float(hi))
        self._range_initialized = True
        self._sync_enabled()
        self._update_read_out()
        self.rangeChanged.emit()

    def row_mask(self, values: np.ndarray | None) -> np.ndarray | None:
        """*values* inside the window, or ``None`` when the gate is inactive.

        Non-finite values are **excluded** by the comparison, which is the right
        answer: a localization with no timestamp is not inside any time window,
        and silently keeping it would put untimed rows in every frame.
        """
        window = self.range()
        if window is None or values is None:
            return None
        array = np.asarray(values, dtype=float)
        lo, hi = window
        return (array >= lo) & (array <= hi)

    # -------------------------------------------------------------- interaction

    def _sync_mode_combo(self, mode: str) -> None:
        index = self._mode_combo.findData(mode)
        if index < 0 or index == self._mode_combo.currentIndex():
            return
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(index)
        self._mode_combo.blockSignals(False)

    def _on_mode_combo_changed(self, _index: int) -> None:
        self.stop()
        # The window lives in the new axis's units, so the old one cannot be
        # carried across -- keeping the numbers would silently mis-gate.
        self._range_initialized = False
        self.modeChanged.emit(self.mode())

    def _on_all_toggled(self, checked: bool) -> None:
        if checked:
            self.stop()
        else:
            # Engaging the gate is when the measured interval starts to matter:
            # it is the step, and it is what the options dialog reports.
            self._request_diagnostics()
            if self._axis is not None and not self._range_initialized:
                self._set_default_window()
        self._sync_enabled()
        self._update_read_out()
        self.rangeChanged.emit()

    def _on_slider_range_changed(self, _lo: float, _hi: float) -> None:
        self._range_initialized = True
        self._update_read_out()
        self.rangeChanged.emit()

    def _set_default_window(self) -> None:
        axis = self._axis
        if axis is None:
            return
        width = max(axis.span * DEFAULT_WINDOW_FRACTION, self._step)
        width = min(width, axis.span)
        self._slider.set_range(axis.lo, axis.lo + width)
        self._range_initialized = True

    def _sync_enabled(self) -> None:
        available = self._axis is not None
        active = self.is_active()
        self._all_check.setEnabled(available)
        self._mode_combo.setEnabled(available)
        self._slider.setEnabled(active)
        for button in (self._prev_button, self._play_button, self._next_button):
            button.setEnabled(active)
        if not active:
            self._set_play_glyph(False)
        tip = self._unavailable_reason or ""
        self._axis_label.setToolTip(tip)
        self._read_out.setToolTip(tip)

    def show_options_dialog(self) -> None:
        # The dialog reports the measured interval, so ask for it before reading
        # the axis -- the request is answered synchronously by the host.
        self._request_diagnostics()
        axis = self._axis
        if axis is None:
            return
        lo, hi = self._slider.range()
        dialog = TimeRangeDialog(
            axis,
            (lo, hi),
            step=self._step,
            rate_hz=self._rate_hz,
            loop=self._loop,
            grow=self._grow,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        self._step = max(float(values["step"]), 0.0)
        self._rate_hz = float(values["rate_hz"])
        self._loop = bool(values["loop"])
        self._grow = bool(values["grow"])
        self._slider.set_scroll_options(self._step, False)
        if values["mode"] != axis.mode:
            self._sync_mode_combo(values["mode"])
            self._on_mode_combo_changed(self._mode_combo.currentIndex())
            return
        if self._all_check.isChecked():
            self._all_check.setChecked(False)      # emits through _on_all_toggled
        self._slider.set_range(*values["range"])
        self._range_initialized = True
        self._update_read_out()
        self.rangeChanged.emit()

    # ----------------------------------------------------------------- playback

    def _effective_step(self) -> float:
        axis = self._axis
        if axis is None:
            return 0.0
        if self._step > 0.0:
            return self._step
        return max(axis.span / 200.0, 1e-12)

    def _step_back(self) -> None:
        self.stop()
        self._shift(-self._effective_step())

    def _step_forward(self) -> None:
        self.stop()
        self._shift(self._effective_step())

    def _shift(self, delta: float) -> bool:
        """Move the window by *delta*; ``False`` when it was already at the end."""
        axis = self._axis
        if axis is None or not self.is_active():
            return False
        lo, hi = self._slider.range()
        moved = max(abs(delta), axis.span) * 1e-9
        if self._grow:
            new_hi = min(hi + delta, axis.hi)
            new_lo = min(lo, new_hi)
            if abs(new_hi - hi) <= moved and delta > 0:
                return False
            self._slider.set_range(new_lo, new_hi, emit=True)
            return True
        width = hi - lo
        new_lo = min(max(lo + delta, axis.lo), max(axis.hi - width, axis.lo))
        if abs(new_lo - lo) <= moved and delta != 0.0:
            return False
        self._slider.set_range(new_lo, new_lo + width, emit=True)
        return True

    def _rewind(self) -> None:
        axis = self._axis
        if axis is None:
            return
        lo, hi = self._slider.range()
        width = hi - lo
        if self._grow:
            self._slider.set_range(axis.lo, axis.lo + max(self._effective_step(), 0.0),
                                   emit=True)
            return
        self._slider.set_range(axis.lo, axis.lo + width, emit=True)

    def set_playing(self, playing: bool) -> None:
        if playing and self.is_active():
            self._timer.setInterval(max(1, int(round(1000.0 / max(self._rate_hz, 0.1)))))
            self._timer.start()
            self._set_play_glyph(True)
            return
        self._timer.stop()
        self._set_play_glyph(False)
        if self._play_button.isChecked():
            self._play_button.blockSignals(True)
            self._play_button.setChecked(False)
            self._play_button.blockSignals(False)

    def stop(self) -> None:
        self.set_playing(False)

    def is_playing(self) -> bool:
        return self._timer.isActive()

    def _advance(self) -> None:
        if self._shift(self._effective_step()):
            return
        if self._loop:
            self._rewind()
            return
        self.stop()

    def _set_play_glyph(self, playing: bool) -> None:
        glyph = _PAUSE_GLYPH if playing else _PLAY_GLYPH
        base = float(getattr(self._play_button, "_mfv_base_point_size", 9.0) or 9.0)
        self._play_button.setFont(
            _scaled_glyph_font(self._play_button, base, glyph)
        )
        self._play_button.setText(glyph)
        tip = "Pause" if playing else "Play"
        self._play_button.setToolTip(tip)
        self._play_button.setAccessibleName(tip)

    # ------------------------------------------------------------------ read-out

    def _update_read_out(self) -> None:
        axis = self._axis
        if axis is None:
            self._read_out.setText(self._unavailable_reason or "no time axis")
            return
        if not self._all_check.isChecked():
            lo, hi = self._slider.range()
            text = axis.format_range(lo, hi)
            if self._grow:
                text = f"{text} (growing)"
        else:
            text = f"all ({axis.format_range(axis.lo, axis.hi)})"
        self._read_out.setText(text)

    def status_suffix(self) -> str:
        """What the host's info line adds while the gate is narrowing the view."""
        axis = self._axis
        window = self.range()
        if axis is None or window is None:
            return ""
        return f"  |  {TIME_MODE_LABELS.get(axis.mode, axis.mode)} {axis.format_range(*window)}"

    # -------------------------------------------------------------------- state

    def state(self) -> dict:
        lo, hi = self._slider.range()
        return {
            "mode": self.mode(),
            "all": bool(self._all_check.isChecked()),
            "range": [float(lo), float(hi)] if self._range_initialized else None,
            "step": float(self._step),
            "rate_hz": float(self._rate_hz),
            "loop": bool(self._loop),
            "grow": bool(self._grow),
        }

    def restore_state(self, state: dict | None) -> None:
        """Adopt a saved row state.  Playback is never restored as *running*."""
        if not isinstance(state, dict):
            return
        mode = str(state.get("mode") or "")
        if mode in TIME_MODES:
            self._sync_mode_combo(mode)
        self._step = float(state.get("step") or 0.0)
        self._rate_hz = float(state.get("rate_hz") or DEFAULT_RATE_HZ)
        self._loop = bool(state.get("loop", True))
        self._grow = bool(state.get("grow", False))
        window = state.get("range")
        if isinstance(window, (list, tuple)) and len(window) == 2:
            if self._axis is None:
                self._pending_range = (float(window[0]), float(window[1]))
            else:
                self._slider.set_range(float(window[0]), float(window[1]))
                self._range_initialized = True
        self._all_check.blockSignals(True)
        self._all_check.setChecked(bool(state.get("all", True)))
        self._all_check.blockSignals(False)
        self._sync_enabled()
        self._update_read_out()
