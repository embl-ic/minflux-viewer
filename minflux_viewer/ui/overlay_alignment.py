"""Temporary keyboard controls for multi-layer overlay alignment."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


ALIGNMENT_HELP_TEXT = (
    "mouse drag in the view, or use arrow keys ↔↕ to move horizontally/vertically; "
    "comma ⸴ = ↺ period · = ↻ to rotate"
)


def alignment_help_label(parent: QWidget | None = None) -> QLabel:
    """Return the common, uniformly sized manual-alignment instruction label."""
    label = QLabel(ALIGNMENT_HELP_TEXT, parent)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setStyleSheet("font-size: 12pt;")
    label.setContentsMargins(0, 2, 0, 3)
    label.setWordWrap(True)
    return label


class OverlayAlignmentPanel(QWidget):
    """Transactional alignment controls embedded below an overlay view.

    The owner supplies the actual transform and redraw operations.  This keeps
    the same temporary UI usable by the render and scatter overlay windows.
    """

    def __init__(self, owner, channels: list[dict], initial_index: int = 0) -> None:
        super().__init__(owner)
        self._owner = owner
        self._channels = channels
        self._selected_index = max(0, min(int(initial_index), len(channels) - 1))
        config_fn = getattr(owner, "_overlay_alignment_control_config", None)
        config = config_fn() if callable(config_fn) else {}
        self._translation_unit = str(config.get("translation_unit", "nm"))
        translation_value = float(config.get("translation_step", 1.0))
        translation_maximum = float(config.get("translation_maximum", 1000.0))
        rotation_value = float(config.get("rotation_step", 0.1))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet("QLabel#overlayAlignmentStatus { color: gray; font-size: 11px; }")

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(3)

        title = QLabel("Overlay alignment")
        title.setStyleSheet("font-weight: bold;")
        root.addWidget(title)

        self._channel_rows: list[QCheckBox] = []
        for index, channel in enumerate(channels):
            check = QCheckBox(f"{index + 1}: {channel.get('name', '')}")
            check.setChecked(bool(channel.get("visible", True)))
            check.toggled.connect(
                lambda checked, i=index: self._owner._overlay_alignment_visibility(i, checked)
            )
            root.addWidget(check)
            self._channel_rows.append(check)
        root.addSpacing(8)

        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Manual align channel"))
        self._channel_combo = QComboBox(self)
        self._channel_combo.addItems([str(ch.get("name", "")) for ch in channels])
        self._channel_combo.setCurrentIndex(self._selected_index)
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        select_row.addWidget(self._channel_combo, stretch=1)
        self._reset_button = QPushButton("Reset", self)
        self._reset_button.clicked.connect(self._owner._overlay_alignment_reset)
        select_row.addWidget(self._reset_button)
        root.addLayout(select_row)

        self._help_label = alignment_help_label(self)
        root.addWidget(self._help_label)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Each key press moves"))
        self._translation_spin = self._step_spin(translation_value, translation_maximum)
        # Compatibility alias for the first version of this shared panel.
        self._pixel_spin = self._translation_spin
        step_row.addWidget(self._translation_spin)
        step_row.addWidget(QLabel(f"{self._translation_unit},"))
        self._degree_spin = self._step_spin(rotation_value, 45.0)
        step_row.addWidget(self._degree_spin)
        step_row.addWidget(QLabel("degree"))
        step_row.addSpacing(12)
        step_row.addWidget(QLabel("|"))
        self._status = QLabel("", objectName="overlayAlignmentStatus")
        step_row.addWidget(self._status, stretch=1)
        root.addLayout(step_row)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        cancel = QPushButton("Cancel", self)
        cancel.clicked.connect(self._owner._overlay_alignment_cancel)
        apply = QPushButton("Apply", self)
        apply.clicked.connect(self._owner._overlay_alignment_apply)
        button_row.addWidget(cancel)
        button_row.addWidget(apply)
        root.addLayout(button_row)
        self._translation_spin.valueChanged.connect(self._save_steps)
        self._degree_spin.valueChanged.connect(self._save_steps)

        self._drag_view = None
        self._drag_last: QPointF | None = None
        drag_view_fn = getattr(owner, "_overlay_alignment_drag_view", None)
        if callable(drag_view_fn):
            self._drag_view = drag_view_fn()
        if self._drag_view is not None:
            self._drag_view.installEventFilter(self)
            self._drag_view.setMouseTracking(True)
            self._drag_view.setCursor(Qt.CursorShape.OpenHandCursor)
            self._drag_view.setToolTip("Drag to move the selected alignment channel")
        self.refresh_status()

    @staticmethod
    def _step_spin(value: float, maximum: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.01, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        spin.setFixedWidth(108)
        return spin

    @property
    def selected_index(self) -> int:
        return self._selected_index

    def _on_channel_changed(self, index: int) -> None:
        if 0 <= index < len(self._channels):
            self._selected_index = int(index)
            self._owner._overlay_alignment_set_channel(index)
            self.refresh_status()
            self.setFocus()

    def refresh_status(self) -> None:
        self._status.setText(self._owner._overlay_alignment_status(self._selected_index))

    def _save_steps(self, _value: float) -> None:
        callback = getattr(self._owner, "_overlay_alignment_steps_changed", None)
        if callable(callback):
            callback(self._translation_spin.value(), self._degree_spin.value())

    def eventFilter(self, watched, event) -> bool:
        if watched is not self._drag_view:
            return super().eventFilter(watched, event)
        event_type = event.type()
        if event_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._drag_last = QPointF(event.position())
            watched.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return True
        if event_type == QEvent.Type.MouseMove and self._drag_last is not None:
            current = QPointF(event.position())
            delta_fn = getattr(self._owner, "_overlay_alignment_view_delta", None)
            drag_fn = getattr(self._owner, "_overlay_alignment_drag", None)
            if callable(delta_fn) and callable(drag_fn):
                dx_nm, dy_nm = delta_fn(self._drag_last, current)
                if dx_nm or dy_nm:
                    drag_fn(self._selected_index, float(dx_nm), float(dy_nm))
                    self.refresh_status()
            self._drag_last = current
            event.accept()
            return True
        if event_type == QEvent.Type.MouseButtonRelease and self._drag_last is not None:
            self._drag_last = None
            watched.setCursor(Qt.CursorShape.OpenHandCursor)
            self.setFocus()
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def detach(self) -> None:
        if self._drag_view is not None:
            self._drag_view.removeEventFilter(self)
            self._drag_view.unsetCursor()
            self._drag_view.setToolTip("")
            self._drag_view = None
        self._drag_last = None

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Left:
            delta = (-1.0, 0.0, 0.0)
        elif key == Qt.Key.Key_Right:
            delta = (1.0, 0.0, 0.0)
        elif key == Qt.Key.Key_Up:
            delta = (0.0, -1.0, 0.0)
        elif key == Qt.Key.Key_Down:
            delta = (0.0, 1.0, 0.0)
        elif key in {Qt.Key.Key_Comma, Qt.Key.Key_Period} or event.text() in {",", "."}:
            # The stored matrix uses Cartesian-positive angles. In the default
            # top-left XY view, Y is inverted, so that mathematical direction
            # appears reversed on screen. Ask the owner for the current view's
            # conversion so the labels always describe the visible rotation.
            sign_fn = getattr(self._owner, "_overlay_alignment_rotation_sign", None)
            counter_clockwise_sign = float(sign_fn()) if callable(sign_fn) else 1.0
            is_comma = key == Qt.Key.Key_Comma or event.text() == ","
            semantic_sign = 1.0 if is_comma else -1.0
            delta = (0.0, 0.0, semantic_sign * counter_clockwise_sign)
        else:
            super().keyPressEvent(event)
            return
        self._owner._overlay_alignment_nudge(
            self._selected_index,
            delta[0] * self._translation_spin.value(),
            delta[1] * self._translation_spin.value(),
            delta[2] * self._degree_spin.value(),
        )
        self.refresh_status()
        event.accept()
