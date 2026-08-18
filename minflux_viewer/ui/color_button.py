"""Small RGBA swatch button shared by Color and Preferences dialogs."""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QColorDialog, QDialog, QPushButton, QWidget

from ..colors import DEFAULT_COLOR_PREFS, color_preferences, normalize_rgba


def qcolor_from_rgba(value) -> QColor:
    r, g, b, a = normalize_rgba(value)
    return QColor(r, g, b, a)


def rgba_from_qcolor(color: QColor) -> tuple[int, int, int, int]:
    return color.red(), color.green(), color.blue(), color.alpha()


class ColorSwatchButton(QPushButton):
    """Compact checkerboard-backed color tile with an RGBA value."""

    def __init__(self, rgba=(0, 0, 0, 255), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rgba = normalize_rgba(rgba)
        self.setFixedSize(32, 28)
        self.setIconSize(QSize(22, 18))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._refresh_icon()

    def rgba(self) -> tuple[int, int, int, int]:
        return self._rgba

    def set_rgba(self, rgba) -> None:
        value = normalize_rgba(rgba)
        if value == self._rgba:
            return
        self._rgba = value
        self._refresh_icon()

    def _refresh_icon(self) -> None:
        width, height, cell = 22, 18, 5
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        for y in range(0, height, cell):
            for x in range(0, width, cell):
                shade = 220 if ((x // cell) + (y // cell)) % 2 == 0 else 160
                painter.fillRect(x, y, cell, cell, QColor(shade, shade, shade))
        painter.fillRect(0, 0, width, height, qcolor_from_rgba(self._rgba))
        painter.setPen(QColor(85, 85, 85))
        painter.drawRect(0, 0, width - 1, height - 1)
        painter.end()
        self.setIcon(QIcon(pixmap))
        r, g, b, a = self._rgba
        self.setToolTip(f"RGBA: {r}, {g}, {b}, {a}")


def choose_rgba(parent: QWidget, current, title: str = "Choose color"):
    """Open the shared non-native Qt palette with an Alpha control."""
    dialog = QColorDialog(qcolor_from_rgba(current), parent)
    dialog.setWindowTitle(title)
    dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
    dialog.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, True)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    color = dialog.selectedColor()
    return rgba_from_qcolor(color) if color.isValid() else None


def sync_custom_palette(prefs) -> None:
    """Load the app palette into QColorDialog's process-wide custom slots."""
    stored = color_preferences(prefs).get("custom_palette", [])
    defaults = DEFAULT_COLOR_PREFS["custom_palette"]
    for index in range(QColorDialog.customCount()):
        value = stored[index] if index < len(stored) else defaults[index % len(defaults)]
        QColorDialog.setCustomColor(index, qcolor_from_rgba(value))


def custom_palette_rgba() -> list[list[int]]:
    """Capture QColorDialog's process-wide custom slots for persistence."""
    return [
        list(rgba_from_qcolor(QColorDialog.customColor(index)))
        for index in range(QColorDialog.customCount())
    ]
