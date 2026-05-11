"""ImageJ-style Analyze > Set Measurements dialog."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from ..core.app_state import AppState, DEFAULT_PREFS


_MEASUREMENT_OPTIONS = [
    ("area", "Area"),
    ("mean", "Mean gray value"),
    ("standard_deviation", "Standard deviation"),
    ("modal", "Modal gray value"),
    ("min_max", "Min & max gray value"),
    ("centroid", "Centroid"),
    ("center_of_mass", "Center of mass"),
    ("perimeter", "Perimeter"),
    ("bounding_rectangle", "Bounding rectangle"),
    ("fit_ellipse", "Fit ellipse"),
    ("shape_descriptors", "Shape descriptors"),
    ("feret", "Feret's diameter"),
    ("integrated_density", "Integrated density"),
    ("median", "Median"),
    ("skewness", "Skewness"),
    ("kurtosis", "Kurtosis"),
    ("area_fraction", "Area fraction"),
    ("stack_position", "Stack position"),
    ("limit_to_threshold", "Limit to threshold"),
    ("display_label", "Display label"),
    ("invert_y", "Invert Y coordinates"),
    ("scientific_notation", "Scientific notation"),
    ("add_to_overlay", "Add to overlay"),
    ("nan_empty_cells", "NaN empty cells"),
]


class SetMeasurementsDialog(QDialog):
    """Persist measurement columns/options used by Measure and ROI Manager."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._checks: dict[str, QCheckBox] = {}

        self.setWindowTitle("Set Measurements")
        self.setModal(True)
        self.resize(460, 430)
        self._build_ui()
        self._load()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(22)
        grid.setVerticalSpacing(6)
        for idx, (key, label) in enumerate(_MEASUREMENT_OPTIONS):
            cb = QCheckBox(label)
            self._checks[key] = cb
            grid.addWidget(cb, idx % 12, idx // 12)
        root.addLayout(grid)

        redirect_row = QHBoxLayout()
        redirect_row.addWidget(QLabel("Redirect to:"))
        self._redirect_combo = QComboBox()
        self._redirect_combo.addItem("None")
        for ds in self._state.datasets:
            self._redirect_combo.addItem(ds.name)
        redirect_row.addWidget(self._redirect_combo, stretch=1)
        root.addLayout(redirect_row)

        decimal_row = QHBoxLayout()
        decimal_row.addWidget(QLabel("Decimal places (0-9):"))
        self._decimal_spin = QSpinBox()
        self._decimal_spin.setRange(0, 9)
        decimal_row.addWidget(self._decimal_spin)
        decimal_row.addStretch()
        root.addLayout(decimal_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Help
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        buttons.helpRequested.connect(self._show_help)
        root.addWidget(buttons)

    def _load(self) -> None:
        prefs = self._state.prefs.get("measurements", {})
        defaults = DEFAULT_PREFS["measurements"]
        for key, cb in self._checks.items():
            cb.setChecked(bool(prefs.get(key, defaults.get(key, False))))
        redirect_to = str(prefs.get("redirect_to", "None") or "None")
        idx = self._redirect_combo.findText(redirect_to)
        self._redirect_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._decimal_spin.setValue(int(prefs.get("decimal_places", 3)))

    def _accept(self) -> None:
        prefs = self._state.prefs.setdefault("measurements", {})
        for key, cb in self._checks.items():
            prefs[key] = bool(cb.isChecked())
        prefs["redirect_to"] = self._redirect_combo.currentText()
        prefs["decimal_places"] = int(self._decimal_spin.value())
        self._state.save_prefs()
        self._state.log("Measurement settings saved.")
        self.accept()

    def _show_help(self) -> None:
        self._state.log(
            "Set Measurements controls which result columns are recorded by Analyze > Measure and ROI Manager measurements.",
            "INFO",
        )
