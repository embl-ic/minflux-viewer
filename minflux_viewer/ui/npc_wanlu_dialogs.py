"""Parameter dialogs for the Wanlu NPC detection + averaging commands
(Analyze › Segmentation › NPC detection wanlu / NPC average wanlu)."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


def _dspin(lo, hi, val, decimals=1, step=1.0, suffix=" nm") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    return s


class NpcDetectWanluDialog(QDialog):
    """Ring-convolution NPC detection + sub-unit clustering parameters."""

    def __init__(self, parent=None, *, defaults: dict | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NPC detection (Wanlu)")
        d = defaults or {}
        root = QVBoxLayout(self)
        intro = QLabel("Detect NPC centres by ring convolution, cluster MINFLUX "
                       "traces into sub-units (corners), and keep NPCs with enough "
                       "assigned sub-units. Produces the per-NPC table consumed by "
                       "'NPC average wanlu'.")
        intro.setWordWrap(True)
        root.addWidget(intro)
        form = QFormLayout()
        self._pixel = _dspin(0.5, 50.0, d.get("pixel_size", 4.0), 1, 0.5)
        self._diam = _dspin(10.0, 500.0, d.get("diameter", 75.0), 1, 1.0)
        self._rim = _dspin(1.0, 200.0, d.get("rim", 20.0), 1, 1.0)
        self._min_units = QSpinBox()
        self._min_units.setRange(1, 16)
        self._min_units.setValue(int(d.get("min_units", 3)))
        self._min_units.setToolTip("Minimum assigned sub-units to accept an NPC.")
        self._z_scale = _dspin(0.1, 2.0, d.get("z_scale", 1.0), 2, 0.01, "")
        self._z_scale.setToolTip("Z flattening factor (MATLAB 'rimf'). Use < 1 (e.g. "
                                 "0.67) for un-z-corrected 'noScaling' data; 1.0 keeps "
                                 "the dataset's RIMF-applied Z.")
        form.addRow("Pixel size:", self._pixel)
        form.addRow("NPC diameter:", self._diam)
        form.addRow("Rim size:", self._rim)
        form.addRow("Min sub-units:", self._min_units)
        form.addRow("Z scale (rimf):", self._z_scale)
        root.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def params(self) -> dict:
        return {"pixel_size": self._pixel.value(), "diameter": self._diam.value(),
                "rim": self._rim.value(), "min_units": self._min_units.value(),
                "z_scale": self._z_scale.value()}


class NpcAverageWanluDialog(QDialog):
    """Two-ring NPC averaging parameters."""

    def __init__(self, parent=None, *, defaults: dict | None = None, n_npc: int = 0) -> None:
        super().__init__(parent)
        self.setWindowTitle("NPC average (Wanlu)")
        d = defaults or {}
        root = QVBoxLayout(self)
        intro = QLabel(f"Average {n_npc} detected NPC(s): split each into two Z rings, "
                       "fit a constrained two-ring model, align by tilt + 8-fold "
                       "symmetry, and pool into one averaged NPC.")
        intro.setWordWrap(True)
        root.addWidget(intro)
        form = QFormLayout()
        self._diam_min = _dspin(20.0, 200.0, d.get("diam_min", 70.0), 1, 1.0)
        self._diam_max = _dspin(20.0, 300.0, d.get("diam_max", 90.0), 1, 1.0)
        self._inter_min = _dspin(5.0, 100.0, d.get("inter_min", 20.0), 1, 1.0)
        self._inter_max = _dspin(5.0, 150.0, d.get("inter_max", 40.0), 1, 1.0)
        self._inter_exp = _dspin(0.0, 150.0, d.get("inter_exp", 0.0), 1, 1.0)
        self._inter_exp.setToolTip("Expected inter-ring distance for the Z fallback "
                                   "(0 = use the midpoint of the bounds).")
        self._sym = QSpinBox()
        self._sym.setRange(2, 16)
        self._sym.setValue(int(d.get("symmetry", 8)))
        self._sym.setToolTip("Rotational symmetry used for phase alignment (8 for NPC).")
        form.addRow("Diameter min:", self._diam_min)
        form.addRow("Diameter max:", self._diam_max)
        form.addRow("Inter-ring min:", self._inter_min)
        form.addRow("Inter-ring max:", self._inter_max)
        form.addRow("Inter-ring expected:", self._inter_exp)
        form.addRow("Symmetry:", self._sym)
        root.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def params(self) -> dict:
        exp = self._inter_exp.value()
        return {"diameter_bounds": (self._diam_min.value(), self._diam_max.value()),
                "inter_ring_bounds": (self._inter_min.value(), self._inter_max.value()),
                "expected_inter_ring": (exp if exp > 0 else None),
                "symmetry": self._sym.value()}
