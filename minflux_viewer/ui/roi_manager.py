"""ImageJ-inspired ROI Manager window."""

from __future__ import annotations

from pathlib import Path

import html

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
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
        self.setAcceptDrops(True)

        self._build_ui()
        self._connect()
        self._refresh()

    # -- drag & drop (ROI-set files) ------------------------------------
    _ROI_DROP_EXTS = {".json", ".roi", ".zip"}

    def _has_roi_url(self, event) -> bool:
        md = event.mimeData()
        return md.hasUrls() and any(
            Path(u.toLocalFile()).suffix.lower() in self._ROI_DROP_EXTS
            for u in md.urls())

    def dragEnterEvent(self, event) -> None:
        if self._has_roi_url(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if self._has_roi_url(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in self._ROI_DROP_EXTS:
                self._load_roi_path(path)
        event.acceptProposedAction()

    def _load_roi_path(self, path: str) -> None:
        try:
            records = self._store.load(path)
            self._state.log(f"Loaded {len(records)} ROI(s) from {Path(path).name}.")
        except Exception as exc:
            self._state.log(f"Failed to load ROI file: {exc}", "ERROR")
            QMessageBox.critical(self, "Open ROI set", str(exc))

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
            ("Property", self._property),
            ("Combine", self._combine),
            ("Convert", self._convert),
            ("Move Up", lambda: self._move(-1)),
            ("Move Down", lambda: self._move(1)),
            ("Select All", self._select_all),
            ("Deselect", self._deselect),
            ("Save", self._save),
            ("Open", self._open),
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
        self._store.deselect()        # filed into the Manager; leaves the viewer (Show-all reveals)

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

    def _notify_roi_changed(self, record) -> None:
        idx = record.context.get("dataset_idx") if isinstance(record.context, dict) else None
        self._state.notify_roi_selection_changed(idx if isinstance(idx, int) else None)

    def _property(self) -> None:
        from .roi_convert_dialog import EDITABLE_PROPERTY_TYPES, RoiPropertyDialog, roi_property_text

        selected = self._store.selected_records()
        if not selected:
            QMessageBox.information(self, "ROI Properties", "Select an ROI first.")
            return
        if len(selected) == 1 and selected[0].type in EDITABLE_PROPERTY_TYPES:
            record = selected[0]
            dlg = RoiPropertyDialog(record, self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            new_record = dlg.result_record()
            self._store.update(record.id, new_record)
            self._store.select([record.id])
            self._notify_roi_changed(record)
            return
        # Multiple selection or a type without simple geometry editing → read-only.
        text = "\n\n".join(roi_property_text(r) for r in selected)
        box = QMessageBox(self)
        box.setWindowTitle("ROI Properties")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText("<pre style='font-family:monospace; margin:0'>"
                    + html.escape(text) + "</pre>")
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                    | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        box.exec()

    def _convert(self) -> None:
        from ..core.roi_convert import available_conversions, convert_roi
        from .roi_convert_dialog import RoiConvertDialog

        selected = self._store.selected_records()
        if not selected:
            QMessageBox.information(self, "Convert ROIs", "Select one or more ROIs first.")
            return
        dlg = RoiConvertDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        target, width, height = dlg.values()
        converted_ids = []
        skipped = 0
        last_record = None
        for record in selected:
            if target not in available_conversions(record):
                skipped += 1
                continue
            kwargs = {}
            if record.type == "point" and target in {"rectangle", "oval"}:
                kwargs = {"width": width, "height": height}
            elif target == "region":
                kwargs = {"width": width}
            try:
                new_record = convert_roi(record, target, **kwargs)
            except Exception:
                skipped += 1
                continue
            new_record.name = (
                f"{new_record.type}-{self._store.next_type_index(new_record.type)}"
                if self._store._is_auto_name(record) else record.name)
            self._store.update(record.id, new_record)
            converted_ids.append(record.id)
            last_record = record
        if converted_ids:
            self._store.select(converted_ids)
            if last_record is not None:
                self._notify_roi_changed(last_record)
        self._state.log(
            f"Convert ROIs → {target}: {len(converted_ids)} converted, {skipped} skipped.")
        if not converted_ids:
            QMessageBox.information(
                self, "Convert ROIs", "None of the selected ROIs can convert to that type.")

    def _select_all(self) -> None:
        self._store.select([r.id for r in self._store.records])

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
        self._load_roi_path(path)

    def _save(self) -> None:
        records = self._store.selected_records() or self._store.records
        if not records:
            QMessageBox.information(self, "Save ROI set", "No ROIs to save.")
            return
        has_non_pixel = any(r.coordinate_space != "pixel" for r in records)
        default_filter = (
            "Native ROI set (*.json)"
            if has_non_pixel
            else ("ImageJ RoiSet (*.zip)" if len(records) > 1 else "ImageJ ROI (*.roi)")
        )
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
        if path_obj.suffix.lower() in {".roi", ".zip"} and has_non_pixel:
            # Don't block: write the plot/attribute (nm) geometry directly as
            # ImageJ pixel coordinates. The spatial calibration is lost, but the
            # shape/region is preserved (matches a 1 nm/pixel viewer TIFF export);
            # use native JSON for an exact, calibrated round-trip.
            self._state.log(
                "Saving plot/attribute-coordinate ROI(s) to ImageJ format: their nm "
                "geometry is written as pixel coordinates (calibration not preserved). "
                "Use native JSON (*.json) for an exact round-trip.", "WARN")
        try:
            self._store.save(path_obj, records)
            self._state.log(f"Saved {len(records)} ROI(s) to {path_obj.name}.")
        except Exception as exc:
            self._state.log(f"Failed to save ROI file: {exc}", "ERROR")
            QMessageBox.critical(self, "Save ROI set", str(exc))
