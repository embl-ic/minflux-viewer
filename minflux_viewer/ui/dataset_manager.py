"""
minflux_viewer.ui.dataset_manager
====================================
Dataset manager dialog — port of ``dialog_info.mlapp``.

A table listing all loaded datasets with columns:
  Active | Name | Dims | Locs | Traces | Loaded | View

Supports selecting the active dataset, removing datasets, and shows a
summary row at the bottom.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.app_state import AppState

# Column indices
_COL_ACTIVE  = 0
_COL_NAME    = 1
_COL_DIMS    = 2
_COL_LOCS    = 3
_COL_ITR     = 4
_COL_TRACES  = 5
_COL_LOADED  = 6
_COL_VIEW    = 7
_NCOLS       = 8


class DatasetManager(QDialog):
    """Non-modal dialog listing all loaded datasets."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        # Non-owned top-level window: passing the main window as the QWidget
        # parent (with the Qt.Window flag) makes the OS pin this above its owner
        # in Z-order. Keep the main window only as an explicit owner reference
        # (lifetime is held by MainWindow._ds_manager).
        super().__init__(None)
        self._owner   = parent
        self._state   = state
        self._updating = False

        self.setWindowTitle("Dataset manager")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(800, 240)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._positioned = False

        self._build_ui()
        self._rebuild_table()

        state.dataset_added.connect(self._on_dataset_added)
        state.dataset_removed.connect(self._on_dataset_removed)
        state.active_changed.connect(self._on_active_changed)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Non-owned top-level window: place it fully on the active screen on
        # first show so it never opens partly off the screen edge.
        if not self._positioned:
            self._positioned = True
            from .modeless import ensure_on_screen
            ensure_on_screen(self, self._owner)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Table ─────────────────────────────────────────────────
        self._table = QTableWidget(0, _NCOLS)
        self._table.setHorizontalHeaderLabels(
            ["Active", "Name", "Dims", "Locs", "Itr", "Traces", "Loaded", "View"]
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(_COL_ACTIVE, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(_COL_ACTIVE,  52)
        self._table.setColumnWidth(_COL_NAME,   200)
        self._table.setColumnWidth(_COL_DIMS,    52)
        self._table.setColumnWidth(_COL_LOCS,    80)
        self._table.setColumnWidth(_COL_ITR,     72)
        self._table.setColumnWidth(_COL_TRACES,  72)
        self._table.setColumnWidth(_COL_LOADED, 160)
        self._table.setColumnWidth(_COL_VIEW,    90)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._table.doubleClicked.connect(self._on_double_click)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)
        root.addWidget(self._table)

        # ── Button row ────────────────────────────────────────────
        bar = QHBoxLayout()

        self._activate_btn = QPushButton("Set active")
        self._activate_btn.clicked.connect(self._set_selected_active)
        bar.addWidget(self._activate_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_selected)
        bar.addWidget(self._remove_btn)

        bar.addStretch()

        self._info = QLabel("")
        self._info.setStyleSheet("color: gray; font-size: 11px;")
        bar.addWidget(self._info)

        root.addLayout(bar)

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def _rebuild_table(self) -> None:
        self._updating = True
        self._table.setRowCount(0)
        for idx, ds in enumerate(self._state.datasets):
            self._insert_row(idx, ds)
        self._update_info()
        self._updating = False

    def _insert_row(self, idx: int, ds) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        is_active = (idx == self._state.active_idx)

        items = [
            ("✓" if is_active else "",    Qt.AlignmentFlag.AlignCenter),
            (ds.file.name,                Qt.AlignmentFlag.AlignLeft),
            (f"{ds.prop.num_dim}D",       Qt.AlignmentFlag.AlignCenter),
            (f"{ds.prop.num_loc:,}",      Qt.AlignmentFlag.AlignRight),
            (self._itr_text(ds),          Qt.AlignmentFlag.AlignCenter),
            (f"{ds.prop.num_traces:,}",   Qt.AlignmentFlag.AlignRight),
            (ds.file.datetime or "—",     Qt.AlignmentFlag.AlignLeft),
            (self._view_text(idx),         Qt.AlignmentFlag.AlignCenter),
        ]
        is_3d = int(getattr(ds.prop, "num_dim", 2)) >= 3
        for col, (text, align) in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            if is_active:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            if col == _COL_DIMS and is_3d:
                item.setToolTip("Double-click to estimate anisotropy / set RIMF")
            self._table.setItem(row, col, item)

    def _refresh_row(self, row: int, idx: int) -> None:
        """Update the active-column cell for one row."""
        is_active = (idx == self._state.active_idx)
        item = self._table.item(row, _COL_ACTIVE)
        if item:
            item.setText("✓" if is_active else "")
            font = item.font()
            font.setBold(is_active)
            item.setFont(font)
        # Bold other columns too
        for col in range(1, _NCOLS):
            it = self._table.item(row, col)
            if it:
                if col == _COL_VIEW:
                    it.setText(self._view_text(idx))
                font = it.font()
                font.setBold(is_active)
                it.setFont(font)

    def _itr_text(self, ds) -> str:
        total = max(1, int(ds.metadata.get("raw_num_itr", ds.prop.num_itr or 1)))
        mode  = str(ds.metadata.get("iteration_load_mode", "")).lower()
        if not mode:
            mode = "last"
        prefix = "all" if mode == "all" else "last"
        return f"{prefix}/{total}"

    def _view_text(self, idx: int) -> str:
        provider = getattr(self._owner, "dataset_view_status", None)
        if callable(provider):
            return provider(idx)
        return "None"

    def refresh_views(self) -> None:
        for row in range(self._table.rowCount()):
            self._refresh_row(row, row)

    def _update_info(self) -> None:
        n = len(self._state.datasets)
        self._info.setText(
            f"{n} dataset{'s' if n != 1 else ''} loaded."
            if n else "No data loaded."
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _set_selected_active(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if rows:
            self._state.set_active(rows[0].row())

    def _remove_selected(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        self._state.remove_dataset(idx)

    def _show_context_menu(self, pos) -> None:
        """Right-click menu on a dataset row: Save as… (save that dataset to disk)."""
        index = self._table.indexAt(pos)
        row = index.row()
        if not (0 <= row < len(self._state.datasets)):
            return
        menu = QMenu(self)
        save_action = menu.addAction("Save as…")
        chosen = menu.exec(self._table.viewport().mapToGlobal(pos))
        if chosen is save_action:
            self._save_dataset(row)

    def _save_dataset(self, row: int) -> None:
        """Open the main window's Save / Export dialog for the row's dataset (same
        as File › Save Processed Data, but targeting the selected dataset)."""
        if not (0 <= row < len(self._state.datasets)):
            return
        ds = self._state.datasets[row]
        saver = getattr(self._owner, "save_dataset", None)
        if callable(saver):
            saver(ds)
        else:
            self._state.log("Save as…: no save handler is available.", "WARN")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_double_click(self, index) -> None:
        row = index.row()
        if not (0 <= row < len(self._state.datasets)):
            return
        ds = self._state.datasets[row]
        self._state.set_active(row)
        # Double-clicking the "Dims" cell of a 3-D dataset opens the Estimate
        # Anisotropy Factor (RIMF) dialog — same as Analyze > Trace > Estimate
        # Anisotropy — so the user can review / manually set and apply RIMF.
        if index.column() == _COL_DIMS and int(getattr(ds.prop, "num_dim", 2)) >= 3:
            try:
                from ..analysis.trace_analysis import show_anisotropy_dialog
                owner = self._owner if isinstance(self._owner, QWidget) else self
                show_anisotropy_dialog(owner, ds, self._state)
            except Exception as exc:
                self._state.log(f"Estimate anisotropy failed: {exc}", "ERROR")

    def _on_dataset_added(self, idx: int) -> None:
        ds = self._state.datasets[idx]
        self._insert_row(idx, ds)
        self._update_info()

    def _on_dataset_removed(self, _idx: int) -> None:
        self._rebuild_table()

    def _on_active_changed(self, active_idx: int) -> None:
        for row in range(self._table.rowCount()):
            self._refresh_row(row, row)
