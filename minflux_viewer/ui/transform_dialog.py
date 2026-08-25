"""Inspect and edit a dataset's overlay transform."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.overlay import identity_matrix4, matrix4_to_xy3, transform_to_matrix4


def updated_transform_record(previous, matrix_4x4: np.ndarray) -> dict:
    """Return a live-state transform record with an edited canonical matrix."""
    matrix = np.asarray(matrix_4x4, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("The transform must be a finite 4 × 4 matrix.")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-12):
        raise ValueError("The last row of a homogeneous transform must be [0, 0, 0, 1].")

    record = dict(previous) if isinstance(previous, dict) else {}
    record["matrix_4x4"] = matrix.tolist()
    record["matrix_3x3"] = matrix4_to_xy3(matrix).tolist()
    record["alignment_mode"] = "manual matrix"
    provenance = dict(record.get("provenance") or {})
    provenance["manual_matrix_edit"] = {
        "method": "matrix editor in Dataset Information",
        "coordinate_unit": "nm",
        "matrix_convention": "column vector; transformed_xyz = matrix_4x4 @ [x, y, z, 1]",
    }
    record["provenance"] = provenance
    return record


def _xyz_euler_degrees(rotation: np.ndarray) -> np.ndarray:
    """XYZ Euler angles for ``R = Rz @ Ry @ Rx`` (column-vector convention)."""
    matrix = np.asarray(rotation, dtype=np.float64)
    horizontal = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if horizontal > 1e-10:
        x = np.arctan2(matrix[2, 1], matrix[2, 2])
        y = np.arctan2(-matrix[2, 0], horizontal)
        z = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        # Gimbal lock: choose Z = 0 and retain a valid equivalent solution.
        x = np.arctan2(-matrix[1, 2], matrix[1, 1])
        y = np.arctan2(-matrix[2, 0], horizontal)
        z = 0.0
    return np.rad2deg((x, y, z))


def _translation_direction(axis: str, value: float, *, xy_origin_top_left: bool) -> str:
    if abs(value) <= 1e-12:
        return "no translation"
    sign = "+" if value > 0.0 else "−"
    if axis == "X":
        screen = "right" if value > 0.0 else "left"
        return f"toward {sign}X ({screen})"
    if axis == "Y":
        if xy_origin_top_left:
            screen = "down in a top-left-origin XY view" if value > 0.0 else "up in a top-left-origin XY view"
        else:
            screen = "up in a bottom-left-origin XY view" if value > 0.0 else "down in a bottom-left-origin XY view"
        return f"toward {sign}Y ({screen})"
    return f"toward {sign}Z ({'positive' if value > 0.0 else 'negative'} axial direction)"


def _rotation_direction(axis: str, angle: float, *, xy_origin_top_left: bool) -> str:
    if abs(angle) <= 1e-10:
        return "no rotation"
    mathematical = "counter-clockwise" if angle > 0.0 else "clockwise"
    note = f"{mathematical} when viewed from +{axis} toward the origin (right-hand convention)"
    if axis == "Z":
        if xy_origin_top_left:
            screen = "clockwise" if angle > 0.0 else "counter-clockwise"
            note += f"; appears {screen} in a top-left-origin XY view"
        else:
            note += f"; appears {mathematical} in a bottom-left-origin XY view"
    return note


def transform_description(matrix_4x4: np.ndarray, *, xy_origin_top_left: bool) -> str:
    """Human-readable translation and rotation decomposition for a 4×4 matrix."""
    matrix = np.asarray(matrix_4x4, dtype=np.float64)
    translation = matrix[:3, 3]
    linear = matrix[:3, :3]
    rigid = bool(
        np.allclose(linear.T @ linear, np.eye(3), rtol=0.0, atol=1e-6)
        and np.isclose(np.linalg.det(linear), 1.0, rtol=0.0, atol=1e-6)
    )
    if rigid:
        rotation = linear
        rotation_intro = "Rotation — XYZ Euler decomposition (R = Rz · Ry · Rx):"
    else:
        # A polar decomposition supplies a useful orientation while explicitly
        # avoiding the claim that a scaled/sheared affine matrix is purely rigid.
        u, _singular_values, vh = np.linalg.svd(linear)
        rotation = u @ vh
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vh
        rotation_intro = (
            "Rotation — closest rigid XYZ Euler component (the matrix also contains scale, shear, or reflection):"
        )
    angles = _xyz_euler_degrees(rotation)

    lines = [
        "Translation — matrix column in nm (where the coordinate origin is mapped):",
    ]
    for axis, value in zip("XYZ", translation, strict=True):
        lines.append(
            f"  {axis}  {float(value):+.6g} nm — "
            f"{_translation_direction(axis, float(value), xy_origin_top_left=xy_origin_top_left)}"
        )
    lines.append("")
    lines.append(rotation_intro)
    for axis, angle in zip("XYZ", angles, strict=True):
        lines.append(
            f"  {axis}  {float(angle):+.6g}° — "
            f"{_rotation_direction(axis, float(angle), xy_origin_top_left=xy_origin_top_left)}"
        )
    lines.extend(
        (
            "",
            "Euler angles are a readable decomposition of the stored matrix, not a history of separate alignment steps.",
        )
    )
    return "\n".join(lines)


class TransformDialog(QDialog):
    """Modal editor for the canonical 4×4 dataset display transform."""

    def __init__(
        self,
        transform,
        *,
        dataset_name: str,
        xy_origin_top_left: bool,
        manual_align_enabled: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._previous = transform
        self._xy_origin_top_left = bool(xy_origin_top_left)
        self.manual_alignment_requested = False
        matrix = transform_to_matrix4(transform)
        self._initial_matrix = matrix.copy() if matrix is not None else identity_matrix4()
        self._matrix_spins: list[list[QDoubleSpinBox | None]] = []

        self.setWindowTitle("Dataset Transform")
        self.setMinimumWidth(760)
        root = QVBoxLayout(self)

        heading = QLabel(f"Transform for {dataset_name}")
        heading_font = heading.font()
        heading_font.setBold(True)
        heading.setFont(heading_font)
        root.addWidget(heading)

        intro = QLabel(
            "The app stores an XYZ homogeneous transform. It maps a raw coordinate "
            "column [x, y, z, 1]ᵀ in nm into the overlay reference coordinate system. "
            "The 4 × 4 form is shown because the stored 3 × 3 form is only an XY projection."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        matrix_group = QGroupBox("Current transform matrix (editable)")
        matrix_layout = QGridLayout(matrix_group)
        headers = ("x", "y", "z", "1")
        rows = ("x′", "y′", "z′", "1")
        for column, text in enumerate(headers, start=1):
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            matrix_layout.addWidget(label, 0, column)
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        for row, row_name in enumerate(rows):
            matrix_layout.addWidget(QLabel(row_name), row + 1, 0)
            spin_row: list[QDoubleSpinBox | None] = []
            for column in range(4):
                if row == 3:
                    value = QLabel(f"{self._initial_matrix[row, column]:.0f}")
                    value.setFont(mono)
                    value.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    value.setStyleSheet("color: gray;")
                    matrix_layout.addWidget(value, row + 1, column + 1)
                    spin_row.append(None)
                    continue
                spin = QDoubleSpinBox()
                spin.setRange(-1.0e12, 1.0e12)
                spin.setDecimals(9)
                spin.setSingleStep(1.0 if column == 3 else 0.001)
                spin.setValue(float(self._initial_matrix[row, column]))
                spin.setFont(mono)
                spin.setMinimumWidth(135)
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(self._refresh_description)
                matrix_layout.addWidget(spin, row + 1, column + 1)
                spin_row.append(spin)
            self._matrix_spins.append(spin_row)
        root.addWidget(matrix_group)

        description_group = QGroupBox("Transform description")
        description_layout = QVBoxLayout(description_group)
        self._description = QLabel()
        self._description.setWordWrap(True)
        self._description.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._description.setFont(mono)
        description_layout.addWidget(self._description)
        root.addWidget(description_group)

        buttons = QDialogButtonBox()
        self._manual_button = QPushButton("Manual align…")
        self._manual_button.setEnabled(bool(manual_align_enabled))
        if manual_align_enabled:
            self._manual_button.setToolTip(
                "Close this dialog and start the same interactive mode as channel-row right-click → Manual align. "
                "Unapplied matrix edits are discarded."
            )
        else:
            self._manual_button.setToolTip("Manual alignment requires at least two datasets in this overlay.")
        self._manual_button.clicked.connect(self._request_manual_alignment)
        buttons.addButton(self._manual_button, QDialogButtonBox.ButtonRole.ActionRole)
        apply_button = buttons.addButton("Apply matrix", QDialogButtonBox.ButtonRole.AcceptRole)
        cancel_button = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        apply_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        root.addWidget(buttons)

        self._refresh_description()

    def matrix(self) -> np.ndarray:
        matrix = identity_matrix4()
        for row in range(3):
            for column in range(4):
                spin = self._matrix_spins[row][column]
                if spin is not None:
                    matrix[row, column] = spin.value()
        return matrix

    def updated_record(self) -> dict:
        return updated_transform_record(self._previous, self.matrix())

    def _refresh_description(self, *_args) -> None:
        self._description.setText(
            transform_description(
                self.matrix(), xy_origin_top_left=self._xy_origin_top_left
            )
        )

    def _request_manual_alignment(self) -> None:
        self.manual_alignment_requested = True
        self.reject()
