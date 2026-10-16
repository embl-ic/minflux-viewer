'Viewer alignment dialog'

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)


class ViewerAlignmentDialog(QDialog):
    """Choose the reference dataset and alignment mode for a multi-channel open.

    ``mbm_check(ref_name) -> (available, reason)`` re-evaluates bead-based
    alignment availability whenever the reference choice changes — matched
    bead counts depend on which dataset is the reference. Cancelling this
    dialog aborts the whole "Open in MINFLUX viewer" action.
    """

    def __init__(
        self,
        selected: list[str],
        mbm_check,
        parent=None,
        *,
        confocal_candidates=None,
        reserved_names=(),
    ):
        super().__init__(parent)
        self.setWindowTitle("Open in MINFLUX viewer")
        self.setMinimumWidth(650 if confocal_candidates else 520)
        self._mbm_check = mbm_check
        self.confocal_widget = None

        layout = QVBoxLayout(self)
        if confocal_candidates:
            from minflux_viewer.ui.confocal_mapping_dialog import ConfocalMappingOptionsWidget

            self.confocal_widget = ConfocalMappingOptionsWidget(
                list(confocal_candidates),
                self,
                reserved_names=reserved_names,
            )
            layout.addWidget(self.confocal_widget)

        label = QLabel(
            "This MSR file contains multiple datasets.\n"
            "Choose how to align them when opening as a multi-channel render overlay, "
            "or open each as an individual dataset."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        # Open each selected dataset standalone (no overlay, no alignment). When
        # checked, the reference/alignment controls below are irrelevant.
        self.individual_check = QCheckBox(
            "Open each as an individual dataset (no overlay / no alignment)", self)
        self.individual_check.toggled.connect(self._on_individual_toggled)
        layout.addWidget(self.individual_check)

        ref_row = QHBoxLayout()
        self._ref_label = QLabel("Reference dataset:")
        ref_row.addWidget(self._ref_label)
        self.ref_combo = QComboBox(self)
        self.ref_combo.addItems(selected)
        self.ref_combo.setCurrentIndex(0)
        self.ref_combo.currentTextChanged.connect(self._on_reference_changed)
        ref_row.addWidget(self.ref_combo, 1)
        layout.addLayout(ref_row)

        row = QHBoxLayout()
        self._align_label = QLabel("align with:")
        row.addWidget(self._align_label)
        self.align_combo = QComboBox(self)
        self.align_combo.addItems(["mbm info", "stage origin", "data centroid"])
        row.addWidget(self.align_combo, 1)
        layout.addLayout(row)

        self._mbm_note = QLabel("")
        self._mbm_note.setWordWrap(True)
        self._mbm_note.setStyleSheet("color: #666;")
        self._mbm_note.setVisible(False)
        layout.addWidget(self._mbm_note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_reference_changed(self.ref_combo.currentText())
        if len(selected) < 2:
            for widget in (
                label,
                self.individual_check,
                self._ref_label,
                self.ref_combo,
                self._align_label,
                self.align_combo,
                self._mbm_note,
            ):
                widget.setVisible(False)
        self.adjustSize()

    def _accept_if_valid(self) -> None:
        if self.confocal_widget is not None:
            valid, reason = self.confocal_widget.validate(require_selection=False)
            if not valid:
                QMessageBox.warning(self, "Open in MINFLUX viewer", reason)
                return
        self.accept()

    def _on_individual_toggled(self, checked: bool) -> None:
        """Disable the overlay alignment controls when opening individually."""
        self.ref_combo.setEnabled(not checked)
        self.align_combo.setEnabled(not checked)
        if checked:
            self._mbm_note.setVisible(False)

    def open_individual(self) -> bool:
        return self.individual_check.isChecked()

    def _on_reference_changed(self, ref_name: str) -> None:
        try:
            available, reason = self._mbm_check(ref_name)
        except Exception as exc:
            available, reason = False, str(exc)
        idx = self.align_combo.findText("mbm info")
        item = getattr(self.align_combo.model(), "item", lambda *_: None)(idx) if idx >= 0 else None
        if item is not None:
            item.setEnabled(bool(available))
        if available:
            # "mbm info" is the first item, so it is the default when usable;
            # do not override an explicit user choice on reference changes.
            self._mbm_note.setVisible(False)
        else:
            if self.align_combo.currentText() == "mbm info":
                self.align_combo.setCurrentText("stage origin")
            self._mbm_note.setText(f"mbm info unavailable: {reason}" if reason else "mbm info unavailable")
            self._mbm_note.setVisible(True)

    def alignment_mode(self) -> str:
        return self.align_combo.currentText().strip().lower()

    def reference_dataset(self) -> str:
        return self.ref_combo.currentText()

    def confocal_options(self):
        return None if self.confocal_widget is None else self.confocal_widget.options()
