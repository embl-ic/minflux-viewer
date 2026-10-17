"""Shared widgets for editing a dataset's **Z scaling factor**.

The factor is a dimensionless multiplier applied to raw ``loc_z`` as a *view*
(see ``core/dataset.py::set_z_scaling_factor``), and useful values sit in a
narrow band around 0.6–1.0. Two steps therefore serve different intents and are
deliberately different sizes:

* an **arrow click / arrow key** is a deliberate fine adjustment — 0.01;
* a **mouse-wheel notch** is a coarse sweep through the range — 0.1.

``QAbstractSpinBox`` drives both from ``singleStep``, so the wheel step is
applied by swapping it for the duration of the wheel event.

Four decimals are shown and accepted throughout: an estimate such as 0.6375 (the
Gaussian fit on the reference Octahedron file) and a project value such as
0.6667 must survive a round trip through any of these editors unrounded.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QLabel, QVBoxLayout

#: Decimals shown and accepted by every Z scaling factor editor.
Z_SCALING_DECIMALS = 4
#: Step for an arrow click / Up-Down key press.
Z_SCALING_ARROW_STEP = 0.01
#: Step for one mouse-wheel notch.
Z_SCALING_WHEEL_STEP = 0.1


class ZScalingFactorSpinBox(QDoubleSpinBox):
    """Z scaling factor editor: 0.01 per arrow click, 0.1 per wheel notch."""

    def __init__(self, parent=None, *, minimum: float = 0.0001,
                 maximum: float = 100.0) -> None:
        super().__init__(parent)
        self.setDecimals(Z_SCALING_DECIMALS)
        self.setRange(float(minimum), float(maximum))
        self.setSingleStep(Z_SCALING_ARROW_STEP)
        self.setKeyboardTracking(False)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.setSingleStep(Z_SCALING_WHEEL_STEP)
        try:
            super().wheelEvent(event)
        finally:
            self.setSingleStep(Z_SCALING_ARROW_STEP)


def format_z_scaling_factor(value: float) -> str:
    """The factor as text, keeping up to four decimals but never fewer than two.

    ``0.6667`` reads as ``0.6667`` rather than being rounded to ``0.67``, while
    a plain ``1.0`` still reads as ``1.00`` instead of a bare ``1.``.
    """
    text = f"{float(value):.{Z_SCALING_DECIMALS}f}"
    whole, _, decimals = text.partition(".")
    decimals = decimals.rstrip("0")
    return f"{whole}.{decimals.ljust(2, '0')}"


class ZScalingFactorDialog(QDialog):
    """Modal prompt for one Z scaling factor.

    Replaces ``QInputDialog.getDouble`` so the two step sizes above apply; the
    return convention (``value, accepted``) is kept identical so call sites read
    the same.
    """

    def __init__(self, parent, current: float, *, title: str = "Set Z Scaling Factor",
                 label: str | None = None, minimum: float = 0.0001,
                 maximum: float = 100.0) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        root = QVBoxLayout(self)
        text = QLabel(label or (
            "Dimensionless multiplier in\n"
            "z_calibrated = z_raw × z_scaling_factor:"
        ))
        root.addWidget(text)
        self._spin = ZScalingFactorSpinBox(self, minimum=minimum, maximum=maximum)
        self._spin.setValue(float(current))
        self._spin.setToolTip(
            "Arrow buttons and Up/Down step by "
            f"{Z_SCALING_ARROW_STEP:g}; the mouse wheel steps by "
            f"{Z_SCALING_WHEEL_STEP:g}."
        )
        root.addWidget(self._spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._spin.setFocus()
        self._spin.selectAll()

    def value(self) -> float:
        return float(self._spin.value())

    @staticmethod
    def ask(parent, current: float, **kwargs) -> tuple[float, bool]:
        dlg = ZScalingFactorDialog(parent, current, **kwargs)
        accepted = dlg.exec() == QDialog.DialogCode.Accepted
        return dlg.value(), accepted
