"""
minflux_viewer.ui.dataset_manager
====================================
Dataset manager dialog — port of ``dialog_info.mlapp``.

A table listing all loaded datasets with columns:
  Active | Name | Dims | Locs | Traces | Loaded | View

Supports selecting the active dataset, closing datasets, and shows a
summary row at the bottom.

Rows are multi-selectable (Ctrl / Shift click).  Selecting is not activating:
the active dataset changes only on double-click or "Set active".  A right-click
inside a multi-selection offers the batch actions — close, duplicate, or combine
the selected datasets into one multi-channel overlay; a right-click on a single
row offers the per-dataset ones (reset / save / close / duplicate, view its MBM
beads or image series, map a confocal signal).

Dropping a file **on a row** applies it to that dataset, which is a different
verb from dropping on the main window ("open this"): a filter preset, a ROI set,
a processing-metadata sidecar or a TIFF to map as a confocal signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
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
        # Multi-selection (Ctrl / Shift click) drives the batch context-menu
        # actions.  Selecting is *not* activating — the active dataset changes
        # only on double-click or "Set active", so a multi-selection never
        # silently retargets the rest of the app.
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._table.doubleClicked.connect(self._on_double_click)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)
        # Dropping a file ON a row applies it to that dataset (see _on_row_drop).
        self._table.setAcceptDrops(True)
        self._table.viewport().setAcceptDrops(True)
        self._table.viewport().installEventFilter(self)
        root.addWidget(self._table)

        # ── Button row ────────────────────────────────────────────
        bar = QHBoxLayout()

        self._activate_btn = QPushButton("Set active")
        self._activate_btn.clicked.connect(self._set_selected_active)
        bar.addWidget(self._activate_btn)

        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self._close_selected)
        bar.addWidget(self._close_btn)

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

    def _selected_rows(self) -> list[int]:
        """Selected table rows, ascending, clipped to existing datasets."""
        model = self._table.selectionModel()
        if model is None:
            return []
        n = len(self._state.datasets)
        return sorted({i.row() for i in model.selectedRows() if 0 <= i.row() < n})

    def _set_selected_active(self) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        current = self._table.currentRow()
        self._state.set_active(current if current in rows else rows[0])

    def _close_selected(self) -> None:
        """*Close* button — close every selected dataset, then keep a row
        highlighted so the button can be pressed repeatedly."""
        self._close_rows(self._selected_rows())

    def _close_rows(self, rows: list[int]) -> None:
        rows = [r for r in rows if 0 <= r < len(self._state.datasets)]
        if not rows:
            return
        closer = getattr(self._owner, "close_datasets", None)
        if callable(closer):
            closer(rows)
        else:                                   # no main window (headless use)
            for idx in sorted(rows, reverse=True):
                self._state.remove_dataset(idx)
        # Each removal rebuilt the table (clearing the selection), so re-highlight
        # only once everything is gone.
        self._highlight_after_close(rows[0])

    def _highlight_after_close(self, removed_row: int) -> None:
        """Highlight the next close candidate so ``Close`` can be pressed
        repeatedly without going back to the table.

        Closing the **top** row keeps the highlight at the top, walking *down*
        the list; closing any other row moves it to the entry **above** the
        closed one — which for the bottom row means walking *up*.  Both are
        ``removed_row - 1`` clamped into the now-shorter table.
        """
        n = self._table.rowCount()
        if n <= 0:
            return
        row = max(0, min(removed_row - 1, n - 1))
        self._table.clearSelection()
        self._table.setCurrentCell(row, _COL_NAME)
        self._table.selectRow(row)

    def _duplicate_rows(self, rows: list[int]) -> None:
        datasets = [self._state.datasets[r] for r in rows
                    if 0 <= r < len(self._state.datasets)]
        if not datasets:
            return
        duplicator = getattr(self._owner, "duplicate_datasets", None)
        if callable(duplicator):
            duplicator(datasets)
        else:
            self._state.log("Duplicate all: no duplicate handler is available.", "WARN")

    def _combine_rows(self, rows: list[int]) -> None:
        combiner = getattr(self._owner, "combine_datasets_as_overlay", None)
        if callable(combiner):
            combiner(rows)
        else:
            self._state.log("Combine as overlay: no combine handler is available.", "WARN")

    def _show_context_menu(self, pos) -> None:
        """Right-click actions for the clicked row, or for the whole
        multi-selection when the click lands inside it."""
        index = self._table.indexAt(pos)
        row = index.row()
        if not (0 <= row < len(self._state.datasets)):
            return
        selected = self._selected_rows()
        if len(selected) > 1 and row in selected:
            self._show_multi_context_menu(selected, pos)
            return

        ds = self._state.datasets[row]
        menu = QMenu(self)
        reset_action = menu.addAction("Reset")
        reset_action.setToolTip("Restore filters, ROI masks, RIMF and view state to as-loaded")
        save_action = menu.addAction("Save as…")
        close_action = menu.addAction("Close")
        dup_action = menu.addAction("Duplicate")

        menu.addSeparator()
        mbm_action = menu.addAction("View mbm info…")
        mbm_action.setEnabled(self._has_mbm(ds))
        if not mbm_action.isEnabled():
            mbm_action.setToolTip("This dataset carries no beam-monitoring (MBM) bead data")
        images_action = menu.addAction("View image series")
        images_action.setEnabled(self._has_msr_source(ds))
        if not images_action.isEnabled():
            images_action.setToolTip("Available for datasets imported from an MSR file")

        menu.addSeparator()
        map_action = menu.addAction("Map confocal signal…")
        map_action.setEnabled(self._has_msr_source(ds))
        if not map_action.isEnabled():
            map_action.setToolTip("Available for datasets imported from an MSR file")

        chosen = menu.exec(self._table.viewport().mapToGlobal(pos))
        if chosen is reset_action:
            self._reset_row(row)
        elif chosen is save_action:
            self._save_dataset(row)
        elif chosen is close_action:
            self._close_rows([row])
        elif chosen is dup_action:
            self._duplicate_rows([row])
        elif chosen is mbm_action:
            self._view_mbm(row)
        elif chosen is images_action:
            self._view_image_series(row)
        elif chosen is map_action:
            self._map_confocal_signal(row)

    @staticmethod
    def _has_msr_source(ds) -> bool:
        """True when the dataset's source ``.msr`` is still on disk."""
        source_path = str(ds.metadata.get("msr_source_path", "") or "")
        return bool(source_path) and Path(source_path).suffix.lower() == ".msr"

    @staticmethod
    def _has_mbm(ds) -> bool:
        """True when the dataset carries beam-monitoring bead points."""
        try:
            from ..core.overlay import mbm_points_array
            points = mbm_points_array(ds)
        except Exception:
            return False
        return points is not None and bool(getattr(points, "size", 0))

    def _reset_row(self, row: int) -> None:
        resetter = getattr(self._owner, "reset_dataset", None)
        if callable(resetter):
            resetter(row)
        else:
            self._state.log("Reset: no reset handler is available.", "WARN")

    def _view_mbm(self, row: int) -> None:
        viewer = getattr(self._owner, "view_dataset_mbm", None)
        if callable(viewer):
            viewer(row)
        else:
            self._state.log("View mbm info: no handler is available.", "WARN")

    def _view_image_series(self, row: int) -> None:
        viewer = getattr(self._owner, "view_dataset_image_series", None)
        if callable(viewer):
            viewer(row)
        else:
            self._state.log("View image series: no handler is available.", "WARN")

    # ------------------------------------------------------------------
    # Drop a file ON a row → apply it to that dataset
    # ------------------------------------------------------------------

    #: Fallback when the owner is not a main window (headless / test use).
    _DROP_EXTS = (".json", ".roi", ".zip", ".tif", ".tiff")

    def _drop_exts(self) -> tuple[str, ...]:
        return tuple(getattr(self._owner, "DROP_ON_DATASET_EXTS", self._DROP_EXTS))

    def _droppable_paths(self, event) -> list[str]:
        """Local file paths in *event* this manager can apply to a dataset."""
        md = event.mimeData()
        if md is None or not md.hasUrls():
            return []
        exts = self._drop_exts()
        out = []
        for url in md.urls():
            path = url.toLocalFile()
            if path and Path(path).suffix.lower() in exts:
                out.append(path)
        return out

    def _row_at_event(self, event) -> int | None:
        """The dataset row under the drag cursor, or None if not over one."""
        try:
            point = event.position().toPoint()
        except AttributeError:                       # Qt5-style QDropEvent
            point = event.pos()
        row = self._table.indexAt(point).row()
        return row if 0 <= row < len(self._state.datasets) else None

    def eventFilter(self, obj, event):
        """Handle drag/drop over the table viewport.

        The filter runs before ``QAbstractItemView``'s own drag-drop machinery
        (which is ``NoDragDrop`` here and would reject the drop), so accepted
        events are consumed and never reach it.
        """
        if obj is self._table.viewport():
            kind = event.type()
            if kind in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
                row = self._row_at_event(event)
                paths = self._droppable_paths(event) if row is not None else []
                if paths:
                    self._show_drop_hint(row, paths)
                    event.acceptProposedAction()
                else:
                    self._clear_drop_hint()
                    event.ignore()
                return True
            if kind == QEvent.Type.DragLeave:
                self._clear_drop_hint()
                return False
            if kind == QEvent.Type.Drop:
                self._clear_drop_hint()
                row = self._row_at_event(event)
                paths = self._droppable_paths(event)
                if row is not None and paths:
                    event.acceptProposedAction()
                    self._on_row_drop(row, paths)
                else:
                    event.ignore()
                return True
        return super().eventFilter(obj, event)

    def _show_drop_hint(self, row: int, paths: list[str]) -> None:
        """Name the target dataset in the info line while a drag hovers a row."""
        name = self._state.datasets[row].name
        what = Path(paths[0]).name if len(paths) == 1 else f"{len(paths)} files"
        self._info.setText(f"Drop '{what}' on '{name}'")

    def _clear_drop_hint(self) -> None:
        self._update_info()

    def _on_row_drop(self, row: int, paths: list[str]) -> None:
        """Apply each dropped file to the dataset at *row*."""
        handler = getattr(self._owner, "drop_file_on_dataset", None)
        if not callable(handler):
            self._state.log("Drop on dataset: no handler is available.", "WARN")
            return
        for path in paths:
            try:
                handler(row, path)
            except Exception as exc:
                self._state.log(
                    f"Drop '{Path(path).name}' on dataset: {exc}", "ERROR")

    def _show_multi_context_menu(self, rows: list[int], pos) -> None:
        """Batch actions over a multi-selection."""
        menu = QMenu(self)
        close_action = menu.addAction("Close all")
        dup_action = menu.addAction("Duplicate all")
        combine_action = menu.addAction("Combine as multi-channel overlay")
        chosen = menu.exec(self._table.viewport().mapToGlobal(pos))
        if chosen is close_action:
            self._close_rows(rows)
        elif chosen is dup_action:
            self._duplicate_rows(rows)
        elif chosen is combine_action:
            self._combine_rows(rows)

    def _map_confocal_signal(self, row: int) -> None:
        """Discover geometrically matching MSR images and map user selections."""
        if not (0 <= row < len(self._state.datasets)):
            return
        dataset = self._state.datasets[row]
        source_path = Path(str(dataset.metadata.get("msr_source_path", "") or ""))
        if not source_path.is_file():
            QMessageBox.warning(
                self,
                "Map confocal signal",
                f"The source MSR file is not available:\n\n{source_path}",
            )
            return

        selected = [{
            "dataset_key": str(dataset.metadata.get("msr_dataset_key", dataset.name)),
            "did": str(dataset.metadata.get("msr_dataset_did", "") or ""),
        }]
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            from ..core.confocal_mapping import discover_confocal_candidates

            candidates = discover_confocal_candidates(source_path, selected)
        except Exception as exc:
            QMessageBox.critical(self, "Map confocal signal", f"Candidate detection failed:\n\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        if not candidates:
            QMessageBox.information(
                self,
                "Map confocal signal",
                "No scalar image stack has calibrated X and Y bounds matching this "
                "dataset's acquisition ROI within 1%.",
            )
            return

        from .confocal_mapping_dialog import (
            ConfocalMappingOptionsDialog,
            apply_confocal_mapping_options,
        )

        dialog = ConfocalMappingOptionsDialog(
            candidates,
            self,
            reserved_names=(
                set(dataset.attr.keys())
                | set(dataset.mfx_raw.keys())
                | {"xnm", "ynm", "znm"}
            ),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            results = apply_confocal_mapping_options(
                dataset, source_path, dialog.options(), self,
            )
        except Exception as exc:
            self._state.log(f"Confocal signal mapping failed for '{dataset.name}': {exc}", "ERROR")
            QMessageBox.warning(self, "Map confocal signal", str(exc))
            return

        if results:
            self._state.notify_attributes_changed(row)
            for result in results:
                self._state.log(
                    f"Mapped confocal signal '{result.attribute_name}' onto '{dataset.name}': "
                    f"{result.finite_count:,}/{result.total_count:,} localizations in bounds.",
                    dataset_idx=row,
                )
            names = ", ".join(result.attribute_name for result in results)
            self._state.status_message.emit(f"Mapped confocal attribute(s): {names}")

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
