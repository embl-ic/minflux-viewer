"""Parameter dialog for NPC segmentation (Analyze › Segmentation › NPC › 2D)."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)


class NpcSegmentationDialog(QDialog):
    """Ask for NPC ring geometry; the rest use sensible defaults."""

    def __init__(self, parent=None, *, defaults: dict | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NPC Segmentation (2D)")
        d = defaults or {}

        root = QVBoxLayout(self)
        intro = QLabel("Detect NPC centres by convolving the rendered XY histogram "
                       "with a ring kernel of the given geometry.")
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        self._diameter = self._spin(10.0, 2000.0, d.get("diameter_nm", 100.0), 1, 1.0, " nm")
        self._rim = self._spin(1.0, 500.0, d.get("rim_nm", 20.0), 1, 1.0, " nm")
        self._pixel = self._spin(0.5, 50.0, d.get("pixel_size_nm", 4.0), 1, 0.5, " nm")
        self._support = self._spin(0.0, 1.0, d.get("min_support", 0.25), 2, 0.05, "")
        form.addRow("NPC diameter:", self._diameter)
        form.addRow("Rim size:", self._rim)
        form.addRow("Pixel size:", self._pixel)
        self._support.setToolTip("Minimum ring support score (angular coverage × "
                                 "radial-fit) to accept a candidate. 0 = keep all peaks.")
        form.addRow("Min support score:", self._support)
        root.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @staticmethod
    def _spin(lo, hi, val, decimals, step, suffix) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setValue(val)
        if suffix:
            s.setSuffix(suffix)
        return s

    def params(self) -> dict:
        return {
            "diameter_nm": self._diameter.value(),
            "rim_nm": self._rim.value(),
            "pixel_size_nm": self._pixel.value(),
            "min_support": self._support.value(),
        }
