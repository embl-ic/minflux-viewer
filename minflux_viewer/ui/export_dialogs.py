"""Focused dialogs for spreadsheet and OME-Zarr exports."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.save import decode_csv_separator


def _path_row(dialog: QDialog, path_edit: QLineEdit, callback) -> QWidget:
    row = QWidget(dialog)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(path_edit, 1)
    browse = QPushButton(row)
    browse.setIcon(dialog.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
    browse.setToolTip("Choose output path")
    browse.setFixedWidth(32)
    browse.clicked.connect(callback)
    layout.addWidget(browse)
    return row


class CsvExportDialog(QDialog):
    """Choose processed attributes, headers, delimiter, and CSV path."""

    def __init__(
        self,
        attribute_names: list[str],
        default_path: str | Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save Spreadsheet")
        self.resize(700, 520)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.path_edit = QLineEdit(str(default_path))
        form.addRow("File:", _path_row(self, self.path_edit, self._browse))
        self.separator_edit = QLineEdit(",")
        self.separator_edit.setMaxLength(2)
        self.separator_edit.setToolTip(r"Enter one character, or \t for a tab.")
        form.addRow("Separator:", self.separator_edit)
        layout.addLayout(form)

        self.table = QTableWidget(len(attribute_names), 3, self)
        self.table.setHorizontalHeaderLabels(["Include", "Attribute", "Column header"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        for row, name in enumerate(attribute_names):
            include = QTableWidgetItem()
            include.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            include.setCheckState(Qt.CheckState.Checked)
            self.table.setItem(row, 0, include)

            attribute = QTableWidgetItem(str(name))
            attribute.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            self.table.setItem(row, 1, attribute)
            self.table.setItem(row, 2, QTableWidgetItem(str(name)))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

        selection_row = QHBoxLayout()
        select_all = QPushButton("Select All")
        select_all.clicked.connect(lambda: self._set_all_checked(True))
        select_none = QPushButton("Select None")
        select_none.clicked.connect(lambda: self._set_all_checked(False))
        reset_headers = QPushButton("Reset Headers")
        reset_headers.clicked.connect(self._reset_headers)
        selection_row.addWidget(select_all)
        selection_row.addWidget(select_none)
        selection_row.addWidget(reset_headers)
        selection_row.addStretch(1)
        layout.addLayout(selection_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Save Spreadsheet",
            self.path_edit.text(),
            "Spreadsheet (*.csv)",
        )
        if path:
            self.path_edit.setText(path)

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(state)

    def _reset_headers(self) -> None:
        for row in range(self.table.rowCount()):
            self.table.item(row, 2).setText(self.table.item(row, 1).text())

    def selected_columns(self) -> list[tuple[str, str]]:
        selected = []
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).checkState() != Qt.CheckState.Checked:
                continue
            selected.append(
                (
                    self.table.item(row, 1).text(),
                    self.table.item(row, 2).text().strip(),
                )
            )
        return selected

    def accept(self) -> None:
        if not self.path_edit.text().strip():
            QMessageBox.warning(self, "Save Spreadsheet", "Choose an output file path.")
            return
        selected = self.selected_columns()
        if not selected:
            QMessageBox.warning(self, "Save Spreadsheet", "Select at least one attribute.")
            return
        if any(not header for _name, header in selected):
            QMessageBox.warning(self, "Save Spreadsheet", "Column headers cannot be empty.")
            return
        try:
            decode_csv_separator(self.separator_edit.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Save Spreadsheet", str(exc))
            return
        super().accept()


class OmeZarrExportDialog(QDialog):
    """Collect the small set of choices needed by the OME-NGFF export."""

    def __init__(
        self,
        default_path: str | Path,
        *,
        pixel_size_nm: float,
        z_voxel_nm: float,
        is_3d: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save OME-NGFF 0.5 / Zarr v3")
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.path_edit = QLineEdit(str(default_path))
        form.addRow("Package:", _path_row(self, self.path_edit, self._browse))

        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setDecimals(3)
        self.pixel_size_spin.setRange(0.001, 1_000_000.0)
        self.pixel_size_spin.setValue(max(0.001, float(pixel_size_nm)))
        self.pixel_size_spin.setSuffix(" nm")
        form.addRow("XY pixel size:", self.pixel_size_spin)

        self.z_voxel_spin = QDoubleSpinBox()
        self.z_voxel_spin.setDecimals(3)
        self.z_voxel_spin.setRange(0.001, 1_000_000.0)
        self.z_voxel_spin.setValue(max(0.001, float(z_voxel_nm)))
        self.z_voxel_spin.setSuffix(" nm")
        form.addRow("Z voxel depth:", self.z_voxel_spin)
        self.z_voxel_spin.setVisible(bool(is_3d))
        z_label = form.labelForField(self.z_voxel_spin)
        if z_label is not None:
            z_label.setVisible(bool(is_3d))
        self.is_3d = bool(is_3d)

        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(1, 10)
        self.levels_spin.setValue(6)
        form.addRow("Maximum pyramid levels:", self.levels_spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Save OME-NGFF 0.5 / Zarr v3",
            self.path_edit.text(),
            "OME-Zarr (*.ome.zarr)",
        )
        if path:
            self.path_edit.setText(path)

    def accept(self) -> None:
        if not self.path_edit.text().strip():
            QMessageBox.warning(self, "Save OME-Zarr", "Choose an output package path.")
            return
        super().accept()
