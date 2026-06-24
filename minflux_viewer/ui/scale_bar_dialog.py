"""Options dialog for Analyze › Measure › Scale Bar."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QVBoxLayout,
)

#: Pure colour name → RGB (dropdown order per the spec).
_PURE_COLORS: dict[str, tuple[int, int, int]] = {
    "White": (255, 255, 255), "Black": (0, 0, 0), "Red": (255, 0, 0),
    "Green": (0, 200, 0), "Blue": (0, 0, 255), "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255), "Yellow": (255, 255, 0), "Gray": (128, 128, 128),
}

LOCATIONS = ("Lower Right", "Lower Left", "Upper Right", "Upper Left", "At Selection")


class _ColorCombo(QComboBox):
    """A pure-colour dropdown (+ optional 'None' first, + 'Custom…' last)."""

    def __init__(self, *, default: str, allow_none: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._custom: QColor | None = None
        if allow_none:
            self.addItem("None")
        for name in _PURE_COLORS:
            self.addItem(name)
        self.addItem("Custom…")
        i = self.findText(default)
        if i >= 0:
            self.setCurrentIndex(i)
        self.activated.connect(self._on_activated)

    def _on_activated(self, _index: int) -> None:
        if self.currentText() == "Custom…":
            initial = self._custom or QColor(255, 255, 255)
            color = QColorDialog.getColor(initial, self, "Choose colour")
            if color.isValid():
                self._custom = color

    def color(self) -> QColor | None:
        """The selected QColor, or ``None`` (only when 'None' is selectable)."""
        text = self.currentText()
        if text == "None":
            return None
        if text == "Custom…":
            return self._custom or QColor(255, 255, 255)
        r, g, b = _PURE_COLORS.get(text, (255, 255, 255))
        return QColor(r, g, b)

    def set_color(self, value: QColor | None) -> None:
        """Select the dropdown entry matching *value* (None / pure name / Custom)."""
        if value is None:
            i = self.findText("None")
            if i >= 0:
                self.setCurrentIndex(i)
            return
        rgb = (value.red(), value.green(), value.blue())
        for name, pure in _PURE_COLORS.items():
            if pure == rgb:
                i = self.findText(name)
                if i >= 0:
                    self.setCurrentIndex(i)
                return
        self._custom = QColor(value)
        i = self.findText("Custom…")
        if i >= 0:
            self.setCurrentIndex(i)


class ScaleBarDialog(QDialog):
    """Collect scale-bar parameters (width/height/font/colour/location/orientation)."""

    def __init__(self, parent=None, *, default_width_nm: float = 100.0,
                 default_height_nm: float = 10.0, has_selection: bool = False,
                 initial: dict | None = None, edit_mode: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scale Bar Properties" if edit_mode else "Scale Bar")
        self.setModal(True)
        self._edit_mode = edit_mode

        root = QVBoxLayout(self)
        form = QFormLayout()

        self._width = QDoubleSpinBox()
        self._width.setRange(0.01, 1.0e9)
        self._width.setDecimals(2)
        self._width.setValue(float(default_width_nm))
        self._width.setSuffix(" nm")
        form.addRow("Width:", self._width)

        self._height = QDoubleSpinBox()
        self._height.setRange(0.01, 1.0e9)
        self._height.setDecimals(2)
        self._height.setValue(float(default_height_nm))
        self._height.setSuffix(" nm")
        form.addRow("Height:", self._height)

        self._font = QSpinBox()
        self._font.setRange(4, 200)
        self._font.setValue(12)
        form.addRow("Font size:", self._font)

        self._color = _ColorCombo(default="White")
        form.addRow("Color:", self._color)

        self._bg = _ColorCombo(default="None", allow_none=True)
        form.addRow("Background:", self._bg)

        # Location only matters for the initial placement; in edit mode the bar
        # is already placed (and freely draggable), so the row is hidden.
        self._location = QComboBox()
        self._location.addItems(LOCATIONS)
        if not edit_mode:
            form.addRow("Location:", self._location)

        self._orientation = QComboBox()
        self._orientation.addItems(["Horizontal", "Vertical"])
        form.addRow("Orientation:", self._orientation)

        if initial:
            self._apply_initial(initial)

        root.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _apply_initial(self, initial: dict) -> None:
        if "width_nm" in initial:
            self._width.setValue(float(initial["width_nm"]))
        if "height_nm" in initial:
            self._height.setValue(float(initial["height_nm"]))
        if "font_size" in initial:
            self._font.setValue(int(initial["font_size"]))
        if "color" in initial:
            self._color.set_color(initial["color"])
        if "bg_color" in initial:
            self._bg.set_color(initial["bg_color"])
        if "horizontal" in initial:
            self._orientation.setCurrentText(
                "Horizontal" if initial["horizontal"] else "Vertical")

    def values(self) -> dict:
        return {
            "width_nm": float(self._width.value()),
            "height_nm": float(self._height.value()),
            "font_size": int(self._font.value()),
            "color": self._color.color(),
            "bg_color": self._bg.color(),
            "location": self._location.currentText(),
            "horizontal": self._orientation.currentText() == "Horizontal",
        }
