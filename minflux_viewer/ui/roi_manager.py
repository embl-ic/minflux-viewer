"""ImageJ-inspired ROI Manager window."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QVBoxLayout,
    QWidget,
)

from ..core.app_state import AppState


class RoiManagerWindow(QWidget):
    TAG = "roi_manager"

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._store = state.rois
        self._refreshing = False

        self.setWindowTitle("ROI Manager")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(330, 430)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._build_ui()
        self._connect()
        self._refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        root.addWidget(self._list, stretch=1)

        grid = QGridLayout()
        buttons = [
            ("Add", self._add),
            ("Update", self._update),
            ("Delete", self._delete),
            ("Rename...", self._rename),
            ("Deselect", self._deselect),
            ("Open...", self._open),
            ("Save...", self._save),
            ("Move Up", lambda: self._move(-1)),
            ("Move Down", lambda: self._move(1)),
            ("Combine", self._combine),
        ]
        for idx, (label, slot) in enumerate(buttons):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            grid.addWidget(btn, idx // 2, idx % 2)
        root.addLayout(grid)

        row = QHBoxLayout()
        self._show_all = QCheckBox("Show All")
        self._labels = QCheckBox("Labels")
        row.addWidget(self._show_all)
        row.addWidget(self._labels)
        row.addStretch()
        root.addLayout(row)

    def _connect(self) -> None:
        self._list.itemSelectionChanged.connect(self._on_list_selection)
        self._show_all.toggled.connect(self._store.set_show_all)
        self._labels.toggled.connect(self._store.set_show_labels)
        self._store.changed.connect(self._refresh)
        self._store.selection_changed.connect(self._refresh_selection)

    def _refresh(self) -> None:
        self._refreshing = True
        self._list.clear()
        for idx, record in enumerate(self._store.records):
            item = QListWidgetItem(f"{idx + 1:04d}  {record.name}")
            item.setData(Qt.ItemDataRole.UserRole, record.id)
            self._list.addItem(item)
        self._show_all.blockSignals(True)
        self._labels.blockSignals(True)
        self._show_all.setChecked(self._store.show_all)
        self._labels.setChecked(self._store.show_labels)
        self._show_all.blockSignals(False)
        self._labels.blockSignals(False)
        self._refreshing = False
        self._refresh_selection()

    def _refresh_selection(self) -> None:
        self._refreshing = True
        selected = set(self._store.selected_ids)
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setSelected(item.data(Qt.ItemDataRole.UserRole) in selected)
        self._refreshing = False

    def _on_list_selection(self) -> None:
        if self._refreshing:
            return
        ids = [item.data(Qt.ItemDataRole.UserRole) for item in self._list.selectedItems()]
        self._store.select(ids)

    def _add(self) -> None:
        adapter = self._store.active_adapter
        if adapter is None or adapter.current_record() is None:
            QMessageBox.information(self, "Add ROI", "Draw an ROI on a compatible window first.")
            return
        self._store.add(adapter.consume_draft())

    def _update(self) -> None:
        adapter = self._store.active_adapter
        selected = self._store.selected_records()
        if len(selected) != 1:
            QMessageBox.information(self, "Update ROI", "Select exactly one ROI to update.")
            return
        if adapter is None:
            QMessageBox.information(self, "Update ROI", "Activate a compatible ROI window first.")
            return
        record_for_update = getattr(adapter, "record_for_update", None)
        if callable(record_for_update):
            record = record_for_update(selected[0])
        else:
            record = adapter.replace_selected_from_draft() if adapter.current_record() is not None else None
        if record is None:
            QMessageBox.information(self, "Update ROI", "Select an ROI visible in the active compatible window, or draw a replacement ROI first.")
            return
        record.name = selected[0].name
        self._store.update(selected[0].id, record)

    def _delete(self) -> None:
        self._store.delete_selected()

    def _rename(self) -> None:
        selected = self._store.selected_records()
        if len(selected) != 1:
            QMessageBox.information(self, "Rename ROI", "Select exactly one ROI to rename.")
            return
        name, ok = QInputDialog.getText(self, "Rename ROI", "Name:", text=selected[0].name)
        if ok and name.strip():
            self._store.rename(selected[0].id, name.strip())

    def _deselect(self) -> None:
        self._store.deselect()

    def _move(self, direction: int) -> None:
        self._store.move_selected(direction)

    def _combine(self) -> None:
        self._store.combine_selected()

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open ROI set",
            "",
            "ROI files (*.roi *.zip *.json);;ImageJ ROI (*.roi);;ImageJ RoiSet (*.zip);;Native ROI set (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            records = self._store.load(path)
            self._state.log(f"Loaded {len(records)} ROI(s) from {Path(path).name}.")
        except Exception as exc:
            self._state.log(f"Failed to load ROI file: {exc}", "ERROR")
            QMessageBox.critical(self, "Open ROI set", str(exc))

    def _save(self) -> None:
        records = self._store.selected_records() or self._store.records
        if not records:
            QMessageBox.information(self, "Save ROI set", "No ROIs to save.")
            return
        default_filter = "ImageJ RoiSet (*.zip)" if len(records) > 1 else "ImageJ ROI (*.roi)"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save ROI set",
            "",
            "ImageJ ROI (*.roi);;ImageJ RoiSet (*.zip);;Native ROI set (*.json)",
            default_filter,
        )
        if not path:
            return
        path_obj = Path(path)
        if not path_obj.suffix:
            if "json" in selected_filter.lower():
                path_obj = path_obj.with_suffix(".json")
            elif len(records) == 1 and "roi" in selected_filter.lower() and "zip" not in selected_filter.lower():
                path_obj = path_obj.with_suffix(".roi")
            else:
                path_obj = path_obj.with_suffix(".zip")
        if path_obj.suffix.lower() in {".roi", ".zip"} and any(r.coordinate_space != "pixel" for r in records):
            reply = QMessageBox.warning(
                self,
                "ImageJ ROI export",
                "Some ROIs are in plot coordinates, not ImageJ pixel coordinates. "
                "Use native JSON for exact round-trip, or continue to write the current coordinates.",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Ok:
                return
        try:
            self._store.save(path_obj, records)
            self._state.log(f"Saved {len(records)} ROI(s) to {path_obj.name}.")
        except Exception as exc:
            self._state.log(f"Failed to save ROI file: {exc}", "ERROR")
            QMessageBox.critical(self, "Save ROI set", str(exc))
