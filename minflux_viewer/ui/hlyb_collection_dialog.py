"""
minflux_viewer.ui.hlyb_collection_dialog
========================================
Accumulator for the **pooled** HlyB/D subunit pair analysis.

The workflow it supports, one dataset at a time:

1. open a dataset and delineate each *E. coli* with a region ROI — by hand, or
   with *Analyze › Segmentation › Shape Model…* — and file them in the ROI
   Manager;
2. press **Collect from active dataset** here, which cuts one cell out of the
   dataset per region ROI and appends it to the pool;
3. open the next dataset, redraw/repopulate the ROI Manager, collect again;
4. when enough cells are pooled, **Run pooled analysis**.

Each collected cell becomes one spatial component, so the pair-distance profile
is accumulated *within* cells and never across cells or acquisitions. That is
the same rule the single-dataset workflow enforces; here the segmentation is
supplied by the operator instead of inferred.

Modeless / non-owned per the project window convention.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.cell_collection import (
    CellCollection,
    extract_cells,
    load_cell_collection,
    region_roi_records,
    save_cell_collection,
)

TITLE = "HlyB/D Pooled Pair Analysis"
_COLUMNS = ("Dataset", "ROI", "Localizations", "Traces")


class HlyBCollectionWindow(QDialog):
    """Collect ROI-delimited cells across datasets, then analyse them together."""

    def __init__(self, state, owner=None) -> None:
        super().__init__(None)
        self._state = state
        self._owner = owner
        self.setWindowTitle(TITLE)
        self.resize(760, 520)
        self._collection: CellCollection = getattr(
            state, "_hlyb_cell_collection", None) or CellCollection()
        # Survives this window closing, so a long pooling session is not lost to
        # an accidental close.
        state._hlyb_cell_collection = self._collection
        self._build_ui()
        self._refresh()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "<b>Pool cells across datasets.</b> Draw a region ROI around each "
            "cell (by hand, or with <i>Analyze › Segmentation › Shape "
            "Model…</i>), file them in the ROI Manager, then collect. Repeat "
            "for as many datasets as you need."))

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self._table, 1)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)

        collect_row = QHBoxLayout()
        self._collect_btn = QPushButton("Collect from active dataset")
        self._collect_btn.setToolTip(
            "Cut one cell out of the active dataset per region ROI currently in "
            "the ROI Manager, and append them to the pool.")
        self._collect_btn.clicked.connect(self._collect)
        collect_row.addWidget(self._collect_btn)
        self._remove_btn = QPushButton("Remove selected")
        self._remove_btn.clicked.connect(self._remove_selected)
        collect_row.addWidget(self._remove_btn)
        self._clear_btn = QPushButton("Clear pool")
        self._clear_btn.clicked.connect(self._clear)
        collect_row.addWidget(self._clear_btn)
        root.addLayout(collect_row)

        io_row = QHBoxLayout()
        self._save_btn = QPushButton("Save pool…")
        self._save_btn.setToolTip(
            "Write the pooled cells to one HDF5 file, so a collection can be "
            "continued in a later session without reopening every dataset.")
        self._save_btn.clicked.connect(self._save)
        io_row.addWidget(self._save_btn)
        self._load_btn = QPushButton("Load pool…")
        self._load_btn.setToolTip("Append a saved pool to the current one.")
        self._load_btn.clicked.connect(self._load)
        io_row.addWidget(self._load_btn)
        io_row.addStretch(1)
        self._run_btn = QPushButton("Run pooled analysis…")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._run)
        io_row.addWidget(self._run_btn)
        root.addLayout(io_row)

    # ------------------------------------------------------------- helpers
    def _dataset(self):
        idx = getattr(self._state, "active_idx", None)
        datasets = getattr(self._state, "datasets", []) or []
        if idx is None or not (0 <= idx < len(datasets)):
            return None, None
        return idx, datasets[idx]

    def _refresh(self) -> None:
        cells = list(self._collection)
        self._table.setRowCount(len(cells))
        for row, cell in enumerate(cells):
            for column, value in enumerate(
                    (cell.dataset, cell.roi, f"{cell.n_locs:,}",
                     f"{cell.n_traces:,}")):
                item = QTableWidgetItem(str(value))
                if column >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(row, column, item)
        info = self._collection.summary()
        self._summary.setText(
            f"<b>{info['n_cells']}</b> cell(s) pooled from "
            f"<b>{info['n_datasets']}</b> dataset(s) — "
            f"{info['n_locs']:,} localizations, {info['n_traces']:,} traces."
            if info["n_cells"] else
            "<i>Nothing pooled yet.</i>")
        has = bool(cells)
        self._run_btn.setEnabled(has)
        self._save_btn.setEnabled(has)
        self._remove_btn.setEnabled(has)
        self._clear_btn.setEnabled(has)

    # ------------------------------------------------------------ collect
    def _collect(self) -> None:
        idx, ds = self._dataset()
        if ds is None:
            QMessageBox.information(
                self, TITLE, "Open a dataset and make it active first.")
            return
        store = getattr(self._state, "rois", None)
        records = region_roi_records(store, idx) if store is not None else []
        if not records:
            QMessageBox.information(
                self, TITLE,
                "No region ROI for this dataset is in the ROI Manager.\n\n"
                "Draw one around each cell — or run Analyze › Segmentation › "
                "Shape Model… and add its contours — then collect again.")
            return

        fresh = [r for r in records
                 if not self._collection.has(str(getattr(ds, "name", "") or ""),
                                             str(getattr(r, "name", "") or ""))]
        repeats = len(records) - len(fresh)
        if not fresh:
            QMessageBox.information(
                self, TITLE,
                f"All {len(records)} ROI(s) of '{ds.name}' are already pooled.")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            cells, skipped = extract_cells(ds, fresh, dataset_idx=idx)
        except Exception as exc:                                  # noqa: BLE001
            QMessageBox.warning(self, TITLE, f"Could not collect: {exc}")
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

        added = self._collection.extend(cells)
        self._refresh()
        notes = []
        if repeats:
            notes.append(f"{repeats} already pooled")
        if skipped:
            notes.append(f"{len(skipped)} too small")
        detail = f" ({'; '.join(notes)})" if notes else ""
        self._log(f"HlyB/D pooled analysis: collected {added} cell(s) from "
                  f"'{ds.name}'{detail}; pool now holds "
                  f"{len(self._collection)} cell(s) from "
                  f"{len(self._collection.datasets)} dataset(s).")
        if skipped:
            QMessageBox.information(
                self, TITLE,
                f"Collected {added} cell(s).\n\nSkipped:\n· "
                + "\n· ".join(skipped))

    def _remove_selected(self) -> None:
        rows = sorted({item.row() for item in self._table.selectedItems()})
        if not rows:
            return
        self._collection.remove(rows)
        self._refresh()

    def _clear(self) -> None:
        if not len(self._collection):
            return
        if QMessageBox.question(
                self, TITLE,
                f"Discard all {len(self._collection)} pooled cell(s)?"
        ) != QMessageBox.StandardButton.Yes:
            return
        self._collection.clear()
        self._refresh()

    # --------------------------------------------------------------- I/O
    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save pooled cells", "", "Cell collection (*.h5)")
        if not path:
            return
        try:
            save_cell_collection(path, self._collection)
        except Exception as exc:                                  # noqa: BLE001
            QMessageBox.warning(self, TITLE, f"Could not save: {exc}")
            return
        self._log(f"HlyB/D pooled analysis: saved {len(self._collection)} "
                  f"pooled cell(s) to '{path}'.")

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load pooled cells", "", "Cell collection (*.h5)")
        if not path:
            return
        try:
            loaded = load_cell_collection(path)
        except Exception as exc:                                  # noqa: BLE001
            QMessageBox.warning(self, TITLE, f"Could not load: {exc}")
            return
        added = self._collection.extend(list(loaded))
        self._refresh()
        self._log(f"HlyB/D pooled analysis: loaded {added} cell(s) from "
                  f"'{path}'; pool now holds {len(self._collection)} cell(s).")

    # --------------------------------------------------------------- run
    def _run(self) -> None:
        from ..analysis.hlyb_staged import Staged3DConfig, analyze_hlyb_staged_pooled
        from ..plugins.hlyb_pair_analysis.runner import PROJECT_Z_SCALING_FACTOR
        from .hlyb_staged_dialog import HlyBStagedDialog, HlyBStagedWindow
        from .modeless import show_modeless

        if not len(self._collection):
            return
        defaults = getattr(self._state, "_hlyb_staged_cfg", None)
        defaults = (Staged3DConfig(z_scaling_factor=PROJECT_Z_SCALING_FACTOR)
                    if defaults is None else
                    Staged3DConfig(**{**vars(defaults),
                    "z_scaling_factor": PROJECT_Z_SCALING_FACTOR}))
        dlg = HlyBStagedDialog(self, defaults=defaults)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cfg = dlg.config()
        self._state._hlyb_staged_cfg = cfg

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = analyze_hlyb_staged_pooled(self._collection.as_cells(), cfg)
        except Exception as exc:                                  # noqa: BLE001
            QMessageBox.warning(self, TITLE, f"Analysis failed: {exc}")
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

        owner = self._owner or self
        title = (f"{result['n_cells_analysed']} cell(s) from "
                 f"{max(result['n_datasets'], 1)} dataset(s)")
        win = HlyBStagedWindow(result, title=title, owner=owner,
                               prefs=getattr(self._state, "prefs", None))
        show_modeless(win, owner)
        self._log_result(cfg, result)

    def _log_result(self, cfg, result) -> None:
        from ..plugins.hlyb_pair_analysis.runner import pooled_log_line, pooled_payload
        log = getattr(self._state, "log", None)
        if not callable(log):
            return
        log(pooled_log_line(cfg, result),
            method_data=pooled_payload(cfg, result))

    def _log(self, message: str) -> None:
        log = getattr(self._state, "log", None)
        if callable(log):
            log(message)
