"""Shared marker-style dialog for scatter plots."""

from __future__ import annotations

from typing import Any

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class PlotStyleDialog(QDialog):
    """Edit the marker shape, size, transparency, and colour of a plot layer."""

    def __init__(self, layer: dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Plot style: {layer.get('label', '')}")
        self.resize(360, 180)
        self._color = tuple(layer.get("color", (30, 90, 180)))
        self._color_changed = False

        layout = QVBoxLayout(self)
        grid = QGridLayout()
        layout.addLayout(grid)

        grid.addWidget(QLabel("Shape:"), 0, 0)
        self.symbol_combo = QComboBox()
        symbols = [
            ("circle", "o"),
            ("square", "s"),
            ("triangle", "t"),
            ("diamond", "d"),
            ("plus", "+"),
            ("x", "x"),
            ("star", "star"),
        ]
        for label, value in symbols:
            self.symbol_combo.addItem(label, value)
        idx = self.symbol_combo.findData(layer.get("symbol", "o"))
        if idx >= 0:
            self.symbol_combo.setCurrentIndex(idx)
        grid.addWidget(self.symbol_combo, 0, 1)

        grid.addWidget(QLabel("Size:"), 1, 0)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 50)
        self.size_spin.setValue(int(layer.get("size", 6)))
        grid.addWidget(self.size_spin, 1, 1)

        grid.addWidget(QLabel("Transparency:"), 2, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 100.0)
        self.alpha_spin.setSuffix(" %")
        self.alpha_spin.setValue(
            round(100.0 * (1.0 - int(layer.get("alpha", 120)) / 255.0))
        )
        grid.addWidget(self.alpha_spin, 2, 1)

        grid.addWidget(QLabel("Color:"), 3, 0)
        self.color_button = QPushButton(self._color_label())
        self.color_button.clicked.connect(self._choose_color)
        grid.addWidget(self.color_button, 3, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def color_changed(self) -> bool:
        """Whether the user explicitly selected a new colour."""
        return self._color_changed

    def _color_label(self) -> str:
        return f"RGB {self._color[0]}, {self._color[1]}, {self._color[2]}"

    def _choose_color(self) -> None:
        color = QColorDialog.getColor(QColor(*self._color), self, "Choose plot color")
        if color.isValid():
            self._color = (color.red(), color.green(), color.blue())
            self._color_changed = True
            self.color_button.setText(self._color_label())

    def result_payload(self) -> dict[str, Any]:
        transparency = float(self.alpha_spin.value()) / 100.0
        alpha = int(round(255.0 * (1.0 - transparency)))
        return {
            "symbol": self.symbol_combo.currentData(),
            "size": int(self.size_spin.value()),
            "color": self._color,
            "alpha": max(0, min(255, alpha)),
        }
