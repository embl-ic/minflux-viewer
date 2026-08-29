"""Modeless, reusable table window for script and plugin results."""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PyQt6.QtGui import QFontDatabase, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .text_select import make_labels_selectable


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return format(number, ".12g")
    return str(value)


def _sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isnan(number):
            return (0, number)
    if value is None:
        return (2, "")
    return (1, str(value).casefold())


class ResultsTableModel(QAbstractTableModel):
    """Read-only model over a handle-owned list of row mappings."""

    def __init__(
        self,
        columns: list[str],
        rows: list[dict[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._columns = columns
        self._rows = rows

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._columns)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._rows[index.row()].get(self._columns[index.column()])
        if role == Qt.ItemDataRole.DisplayRole:
            return _display_value(value)
        if role == Qt.ItemDataRole.UserRole:
            return value
        if role == Qt.ItemDataRole.TextAlignmentRole:
            horizontal = (
                Qt.AlignmentFlag.AlignRight
                if isinstance(value, numbers.Real) and not isinstance(value, bool)
                else Qt.AlignmentFlag.AlignLeft
            )
            return horizontal | Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(  # noqa: N802 - Qt API
        self,
        section: int,
        orientation: Qt.Orientation,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._columns[section]
        return section + 1

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()


class _NumericSortProxy(QSortFilterProxyModel):
    """Sort numerals by value while keeping strings case-insensitive."""

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        role = Qt.ItemDataRole.UserRole
        return _sort_key(left.data(role)) < _sort_key(right.data(role))


class _ResultsView(QTableView):
    """Table view with spreadsheet-style copy of the selected cells."""

    def copy_selection(self) -> None:
        indexes = self.selectedIndexes()
        if not indexes:
            return
        rows = range(min(i.row() for i in indexes), max(i.row() for i in indexes) + 1)
        cols = range(min(i.column() for i in indexes), max(i.column() for i in indexes) + 1)
        selected = {(i.row(), i.column()): i for i in indexes}
        lines = []
        for row in rows:
            lines.append(
                "\t".join(
                    str(selected[(row, col)].data() or "") if (row, col) in selected else ""
                    for col in cols
                )
            )
        from PyQt6.QtWidgets import QApplication

        QApplication.clipboard().setText("\n".join(lines))


class ResultsTableWindow(QWidget):
    """Sortable, selectable presentation for one named results table."""

    def __init__(
        self,
        name: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        save_csv: Callable[[str], str],
    ) -> None:
        super().__init__(None)
        self._name = name
        self._save_csv = save_csv
        self.setWindowTitle(name)
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(760, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._summary = QLabel()
        root.addWidget(self._summary)

        self._model = ResultsTableModel(columns, rows, self)
        self._proxy = _NumericSortProxy(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setDynamicSortFilter(True)

        self.table = _ResultsView(self)
        self.table.setObjectName("resultsTable")
        self.table.setModel(self._proxy)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self.table)
        copy_shortcut.activated.connect(self.table.copy_selection)
        root.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        save = QPushButton("Save CSV…")
        save.clicked.connect(self._choose_csv)
        buttons.addWidget(save)
        buttons.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        root.addLayout(buttons)

        make_labels_selectable(self)
        self.refresh()

    def refresh(self) -> None:
        self._model.refresh()
        rows = self._model.rowCount()
        columns = self._model.columnCount()
        self._summary.setText(
            f"{rows:,} row{'s' if rows != 1 else ''} · "
            f"{columns:,} column{'s' if columns != 1 else ''}"
        )

    def _choose_csv(self) -> None:
        suggested = "_".join(self._name.split()) or "results"
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save results as CSV",
            str(Path.home() / f"{suggested}.csv"),
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            self._save_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save results", str(exc))
