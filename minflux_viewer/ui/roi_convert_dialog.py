"""Dialogs for ROI conversion, resize and property editing.

* ``RoiResizeDialog`` — enlarge/shrink amount (Process › ROI › Enlarge / Shrink,
  and the right-click *Enlarge / Shrink…*).
* ``RoiSizeDialog`` — width/height for size-needing conversions (point→box/oval,
  line→region).
* ``RoiConvertDialog`` — target type + size for the ROI Manager batch *Convert*.
* ``RoiPropertyDialog`` — editable name / type / geometry for a single
  rectangle / oval / point ROI.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from ..core.roi import RoiRecord, _bounds
from ..core.roi_convert import TARGET_LABELS, available_conversions

_TYPE_LABELS = {
    "point": "Point", "rectangle": "Rectangle", "oval": "Oval",
    "polygon": "Polygon", "convex_hull": "Convex hull", "region": "Region",
    "line": "Line", "freehand": "Freehand", "polyline": "Polyline",
    "freehand_line": "Freehand line", "magnetic_lasso": "Magnetic lasso",
    "angle": "Angle",
}


def _coord_spin(value: float = 0.0) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(-1.0e9, 1.0e9)
    s.setDecimals(1)
    s.setSingleStep(10.0)
    s.setSuffix(" nm")
    s.setValue(float(value))
    return s


class RoiResizeDialog(QDialog):
    """Enlarge / shrink an ROI by an amount in nm."""

    def __init__(self, parent=None, *, allow_shrink: bool = True, default: float = 10.0) -> None:
        super().__init__(parent)
        self.setWindowTitle("Enlarge / Shrink ROI")
        self.setModal(True)
        root = QVBoxLayout(self)
        row = QHBoxLayout()
        self._mode = QComboBox()
        self._mode.addItem("Enlarge", "enlarge")
        if allow_shrink:
            self._mode.addItem("Shrink", "shrink")
        self._value = QDoubleSpinBox()
        self._value.setRange(0.1, 1.0e7)
        self._value.setDecimals(1)
        self._value.setSingleStep(1.0)
        self._value.setValue(float(default))
        row.addWidget(self._mode)
        row.addWidget(self._value, 1)
        row.addWidget(QLabel("nm"))
        root.addLayout(row)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def values(self) -> tuple[str, float]:
        return self._mode.currentData(), float(self._value.value())


class RoiSizeDialog(QDialog):
    """Ask for Width / Height (nm) — point→box/oval and line→region."""

    def __init__(self, parent=None, *, title: str = "ROI size", need_height: bool = True,
                 default: float = 100.0, width_label: str = "Width") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        root = QVBoxLayout(self)
        form = QFormLayout()
        self._w = QDoubleSpinBox()
        self._w.setRange(0.1, 1.0e7)
        self._w.setDecimals(1)
        self._w.setValue(float(default))
        self._w.setSuffix(" nm")
        form.addRow(f"{width_label}:", self._w)
        self._h = None
        if need_height:
            self._h = QDoubleSpinBox()
            self._h.setRange(0.1, 1.0e7)
            self._h.setDecimals(1)
            self._h.setValue(float(default))
            self._h.setSuffix(" nm")
            form.addRow("Height:", self._h)
        root.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def values(self) -> tuple[float, float]:
        w = float(self._w.value())
        return w, (float(self._h.value()) if self._h is not None else w)


class RoiConvertDialog(QDialog):
    """Batch convert: pick a target type (+ optional size) for selected ROIs."""

    def __init__(self, parent=None, *, targets: list[str] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Convert ROIs")
        self.setModal(True)
        root = QVBoxLayout(self)
        intro = QLabel("Convert all selected ROIs to the target type. Conversions that "
                       "do not apply to a given ROI are skipped.")
        intro.setWordWrap(True)
        root.addWidget(intro)
        form = QFormLayout()
        self._target = QComboBox()
        for tok in (targets or ["point", "bounding_box", "rectangle", "oval", "ellipse",
                                "polygon", "convex_hull", "region", "line"]):
            self._target.addItem(TARGET_LABELS.get(tok, tok), tok)
        form.addRow("Convert to:", self._target)
        self._w = QDoubleSpinBox()
        self._w.setRange(0.1, 1.0e7)
        self._w.setDecimals(1)
        self._w.setValue(100.0)
        self._w.setSuffix(" nm")
        self._h = QDoubleSpinBox()
        self._h.setRange(0.1, 1.0e7)
        self._h.setDecimals(1)
        self._h.setValue(100.0)
        self._h.setSuffix(" nm")
        form.addRow("Width:", self._w)
        form.addRow("Height:", self._h)
        root.addLayout(form)
        hint = QLabel("Width/Height apply to point→box/oval and line→region (uses Width).")
        hint.setStyleSheet("color: #666;")
        hint.setWordWrap(True)
        root.addWidget(hint)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def values(self) -> tuple[str, float, float]:
        return self._target.currentData(), float(self._w.value()), float(self._h.value())


#: ROI types the editable property dialog supports (X/Y[/W/H] numeric fields).
EDITABLE_PROPERTY_TYPES = {"rectangle", "oval", "point"}


class RoiPropertyDialog(QDialog):
    """Editable name / type / geometry for a single rectangle, oval or point ROI.

    Name is renamable, Type is a dropdown that converts on accept, and geometry
    is shown as a 2×2 grid of X / Y / W / H numeric fields (W/H drive the size
    when a point is converted to a box/oval).
    """

    def __init__(self, record: RoiRecord, parent=None) -> None:
        super().__init__(parent)
        self._record = record
        self.setWindowTitle("ROI Properties")
        self.setModal(True)

        root = QVBoxLayout(self)
        form = QFormLayout()
        self._name = QLineEdit(record.name)
        form.addRow("Name:", self._name)

        self._type = QComboBox()
        self._type.addItem(_TYPE_LABELS.get(record.type, record.type), record.type)
        for tok in available_conversions(record):
            # Only offer simple, geometry-defined targets in the inline dropdown.
            if tok in {"rectangle", "oval", "point"}:
                self._type.addItem(_TYPE_LABELS.get(tok, tok), tok)
        form.addRow("Type:", self._type)
        root.addLayout(form)

        x, y, w, h = _bounds(record.geometry)
        grid = QGridLayout()
        self._x = _coord_spin(x)
        self._y = _coord_spin(y)
        self._w = _coord_spin(w)
        self._h = _coord_spin(h)
        if record.type == "point" and (w == 0 and h == 0):
            self._w.setValue(100.0)
            self._h.setValue(100.0)
        grid.addWidget(QLabel("X ="), 0, 0)
        grid.addWidget(self._x, 0, 1)
        grid.addWidget(QLabel("Y ="), 0, 2)
        grid.addWidget(self._y, 0, 3)
        grid.addWidget(QLabel("W ="), 1, 0)
        grid.addWidget(self._w, 1, 1)
        grid.addWidget(QLabel("H ="), 1, 2)
        grid.addWidget(self._h, 1, 3)
        self._angle = None
        if record.type in {"rectangle", "oval"}:
            self._angle = QDoubleSpinBox()
            self._angle.setRange(-360.0, 360.0)
            self._angle.setDecimals(1)
            self._angle.setSingleStep(1.0)
            self._angle.setSuffix(" °")
            self._angle.setValue(float(record.geometry.get("angle", 0.0) or 0.0))
            grid.addWidget(QLabel("Angle ="), 2, 0)
            grid.addWidget(self._angle, 2, 1)
        root.addLayout(grid)
        self._wh_note = QLabel("W/H define the size when converting a point to a box/oval.")
        self._wh_note.setStyleSheet("color: #666;")
        self._wh_note.setWordWrap(True)
        self._wh_note.setVisible(record.type == "point")
        root.addWidget(self._wh_note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def result_record(self) -> RoiRecord:
        """Build the updated record from the edited fields (applies geometry edits,
        then a type conversion if the type changed). Keeps the same ``id``."""
        import copy

        from ..core.roi_convert import convert_roi

        new_type = self._type.currentData()
        x, y = float(self._x.value()), float(self._y.value())
        w, h = float(self._w.value()), float(self._h.value())

        edited = copy.deepcopy(self._record)
        if edited.type in {"rectangle", "oval"}:
            geom = {"bounds": [x, y, w, h]}
            angle = float(self._angle.value()) if self._angle is not None else float(
                edited.geometry.get("angle", 0.0) or 0.0)
            if abs(angle) > 1e-9:
                geom["angle"] = angle
            edited.geometry = geom
        elif edited.type == "point":
            pt = list(edited.geometry.get("point", [0.0, 0.0]))
            depth = pt[2] if len(pt) > 2 else 0.0
            edited.geometry = {"point": [x, y, depth]}

        if new_type == edited.type:
            edited.name = self._name.text().strip() or edited.name
            edited.selection_dirty = True
            return edited

        out = convert_roi(edited, new_type, width=w, height=h)
        out.id = edited.id
        out.name = self._name.text().strip() or edited.name
        return out


def roi_property_text(record: RoiRecord) -> str:
    """Read-only multi-line property text for one ROI (name / type / geometry)."""
    from ..core.roi import record_to_points

    def f(v):
        return f"{int(round(float(v)))}" if np.isfinite(float(v)) else ""

    lines = [f"Name: {record.name}", f"Type: {record.type}"]
    g = record.geometry
    if record.type in {"rectangle", "oval"}:
        x, y, w, h = _bounds(g)
        geo = f"X={f(x)}, Y={f(y)}, W={f(w)}, H={f(h)}"
        if float(g.get("angle", 0.0) or 0.0):
            geo += f", angle={float(g['angle']):.1f}°"
        lines.append(f"Geometry: {geo}")
    elif record.type == "point":
        pt = g.get("point", [0.0, 0.0])
        lines.append("Geometry: (" + ", ".join(f(c) for c in pt[:3]) + ")")
    else:
        pts = record_to_points(record)
        x, y, w, h = _bounds(g)
        lines.append(f"Geometry: {len(pts)} vertices, bbox X={f(x)}, Y={f(y)}, W={f(w)}, H={f(h)}")
    return "\n".join(lines)
