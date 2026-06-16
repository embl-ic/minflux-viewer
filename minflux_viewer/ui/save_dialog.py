"""
Save-processed-data dialog.

Collects the export format and a few options; the actual writing is done by
:func:`minflux_viewer.core.save.save_dataset`.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QLabel,
    QVBoxLayout,
)

# (label, format key)
_FORMATS = (
    ("MATLAB (.mat)", "mat"),
    ("NumPy (.npy)", "npy"),
    ("JSON (.json)", "json"),
    ("CSV (.csv)", "csv"),
)


class SaveProcessedDataDialog(QDialog):
    """Pick an export format and what to include when saving a dataset."""

    def __init__(self, dataset_name: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save processed data")
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)
        if dataset_name:
            root.addWidget(QLabel(f"Dataset: <b>{dataset_name}</b>"))

        form = QFormLayout()
        form.setVerticalSpacing(8)

        self._format = QComboBox()
        for label, key in _FORMATS:
            self._format.addItem(label, key)
        form.addRow("Export format", self._format)
        root.addLayout(form)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(line)

        self._only_valid = QCheckBox("Save only valid localizations")
        self._only_valid.setChecked(True)
        self._only_valid.setToolTip("Export only rows where vld is True.")

        self._all_iters = QCheckBox("Save all iterations")
        self._all_iters.setChecked(False)
        self._all_iters.setToolTip(
            "Export every iteration row instead of just the final (last-valid) "
            "localization."
        )

        self._ftr = QCheckBox("Save filter state (ftr column)")
        self._ftr.setChecked(True)
        self._ftr.setToolTip(
            "Add an 'ftr' column holding the current combined filter mask "
            "(the gated/visible state). No data is deleted, so the filter can "
            "be re-applied later."
        )

        self._derived = QCheckBox("Save derived attributes")
        self._derived.setChecked(True)
        self._derived.setToolTip(
            "Include computed attributes (den, siz, dst, dur, len, spd, dt, "
            "tim_trace, idx) where available."
        )

        self._metadata = QCheckBox("Save metadata (sidecar .json)")
        self._metadata.setChecked(True)
        self._metadata.setToolTip(
            "Write a '<name>_metadata.json' alongside the data with RIMF, "
            "transform, filter specs and provenance. It is tagged so it is "
            "distinguishable from filter-preset and data JSON files."
        )

        for chk in (self._only_valid, self._all_iters, self._ftr,
                    self._derived, self._metadata):
            root.addWidget(chk)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def options(self) -> dict:
        """Return the chosen export format and inclusion flags."""
        return {
            "fmt": self._format.currentData(),
            "only_valid": self._only_valid.isChecked(),
            "all_iterations": self._all_iters.isChecked(),
            "include_ftr": self._ftr.isChecked(),
            "include_derived": self._derived.isChecked(),
            "include_metadata": self._metadata.isChecked(),
        }
