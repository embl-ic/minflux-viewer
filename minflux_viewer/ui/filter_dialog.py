"""
minflux_viewer.ui.filter_dialog
==================================
Filter dialog — port of ``dialog_filter.mlapp``.

A table of filter rows.  Each row specifies:
  attribute | aggregation mode | min | max | enabled

Filters are combined with logical AND.  The result is written to
``dataset.filter_mask`` and ``state.filter_changed`` is emitted.

Filters can be saved to / loaded from JSON (same format as the MATLAB app).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.app_state import AppState

_AGG_MODES = ["per loc", "trace mean", "trace stdev", "trace max", "trace min", "trace range"]

# Column indices
_COL_ENABLED = 0
_COL_ATTR    = 1
_COL_MODE    = 2
_COL_MIN     = 3
_COL_MAX     = 4
_COL_DELETE  = 5
_NCOLS       = 6


class FilterDialog(QDialog):
    """Non-modal filter management dialog."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._applying = False

        self.setWindowTitle("Filter")
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint
        )
        self.resize(720, 360)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # Single-shot timer — debounces rapid edits (spinners) before applying.
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.timeout.connect(self._apply_all)

        self._build_ui()
        self._populate_attr_lists()

        # Accept JSON filter files dropped anywhere on the dialog.
        # The table widget must not intercept drops, or they won't reach here.
        self.setAcceptDrops(True)
        self._table.setAcceptDrops(False)
        self._table.viewport().setAcceptDrops(False)

        state.active_changed.connect(self._on_active_changed)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Dataset label ─────────────────────────────────────────
        self._ds_label = QLabel("No data loaded.")
        self._ds_label.setStyleSheet("font-weight: bold;")
        root.addWidget(self._ds_label)

        # ── Filter table ──────────────────────────────────────────
        self._table = QTableWidget(0, _NCOLS)
        self._table.setHorizontalHeaderLabels(
            ["On", "Attribute", "Mode", "Min", "Max", ""]
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(_COL_ENABLED, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(_COL_DELETE,  QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(_COL_ENABLED, 36)
        self._table.setColumnWidth(_COL_ATTR,   140)
        self._table.setColumnWidth(_COL_MODE,   120)
        self._table.setColumnWidth(_COL_MIN,    100)
        self._table.setColumnWidth(_COL_MAX,    100)
        self._table.setColumnWidth(_COL_DELETE,  36)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._table)

        # ── Button row ────────────────────────────────────────────
        bar = QHBoxLayout()

        add_btn = QPushButton("Add filter row")
        add_btn.clicked.connect(self._add_row)
        bar.addWidget(add_btn)

        apply_btn = QPushButton("Apply all")
        apply_btn.clicked.connect(self._apply_all)
        bar.addWidget(apply_btn)

        reset_btn = QPushButton("Reset all filters")
        reset_btn.clicked.connect(self._reset_all)
        bar.addWidget(reset_btn)

        bar.addStretch()

        save_btn = QPushButton("Save JSON…")
        save_btn.clicked.connect(self._save_json)
        bar.addWidget(save_btn)

        load_btn = QPushButton("Load JSON…")
        load_btn.clicked.connect(self._load_json)
        bar.addWidget(load_btn)

        root.addLayout(bar)

        # ── Info label ────────────────────────────────────────────
        self._info = QLabel("")
        self._info.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info)

    # ------------------------------------------------------------------
    # Row management
    # ------------------------------------------------------------------

    def _populate_attr_lists(self) -> None:
        ds = self._state.active_dataset
        if ds is None:
            self._ds_label.setText("No data loaded.")
            self._numeric_attrs: list[str] = []
            return

        self._ds_label.setText(f"Dataset: {ds.name}")
        self._numeric_attrs = [
            k for k in ds.prop.attr_names
            if k not in ("ftr", "idx")
            and np.issubdtype(np.asarray(ds.attr[k]).dtype, np.number)
            and np.asarray(ds.attr[k]).ndim == 1
        ]
        # Refresh combos in existing rows
        for row in range(self._table.rowCount()):
            combo = self._table.cellWidget(row, _COL_ATTR)
            if isinstance(combo, QComboBox):
                old = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(self._numeric_attrs)
                if old in self._numeric_attrs:
                    combo.setCurrentText(old)
                combo.blockSignals(False)

    def _add_row(self, attr: str = "", mode: str = "per loc",
                 lo: float = 0.0, hi: float = 1.0, enabled: bool = True) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        # Enabled checkbox — block signals during setup to avoid spurious applies
        chk = QCheckBox()
        chk.setStyleSheet("margin-left: 8px;")
        chk.blockSignals(True)
        chk.setChecked(enabled)
        chk.blockSignals(False)
        chk.stateChanged.connect(self._apply_all)
        self._table.setCellWidget(row, _COL_ENABLED, chk)

        # Attribute combo
        attr_combo = QComboBox()
        attr_combo.blockSignals(True)
        attr_combo.addItems(self._numeric_attrs)
        if attr in self._numeric_attrs:
            attr_combo.setCurrentText(attr)
        elif self._numeric_attrs:
            attr_combo.setCurrentIndex(0)
        attr_combo.blockSignals(False)
        attr_combo.currentTextChanged.connect(self._auto_fill_range)
        attr_combo.currentTextChanged.connect(self._apply_all)
        self._table.setCellWidget(row, _COL_ATTR, attr_combo)

        # Mode combo
        mode_combo = QComboBox()
        mode_combo.blockSignals(True)
        mode_combo.addItems(_AGG_MODES)
        mode_combo.setCurrentText(mode)
        mode_combo.blockSignals(False)
        mode_combo.currentTextChanged.connect(self._apply_all)
        self._table.setCellWidget(row, _COL_MODE, mode_combo)

        # Min / Max spinners — use debounce timer so dragging doesn't hammer the renderer
        min_spin = _make_spin(lo)
        max_spin = _make_spin(hi)
        min_spin.valueChanged.connect(lambda _: self._apply_timer.start(400))
        max_spin.valueChanged.connect(lambda _: self._apply_timer.start(400))
        self._table.setCellWidget(row, _COL_MIN, min_spin)
        self._table.setCellWidget(row, _COL_MAX, max_spin)

        # Delete button
        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(30)
        del_btn.clicked.connect(lambda _, r=row: self._delete_row(r))
        self._table.setCellWidget(row, _COL_DELETE, del_btn)

        self._auto_fill_range(attr_combo.currentText(), row)

    def _delete_row(self, row: int) -> None:
        self._table.removeRow(row)
        self._apply_all()  # views update immediately when a row is removed

    def _auto_fill_range(self, attr: str, row: int | None = None) -> None:
        """Fill Min/Max spinners with the attribute's actual range."""
        ds = self._state.active_dataset
        if ds is None or attr not in ds.attr:
            return
        vals = np.asarray(ds.attr[attr]).ravel().astype(float)
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))

        if row is None:
            # Find the row that triggered this (by sender)
            for r in range(self._table.rowCount()):
                if self._table.cellWidget(r, _COL_ATTR) is self.sender():
                    row = r
                    break
        if row is None:
            return

        for col, val in ((_COL_MIN, lo), (_COL_MAX, hi)):
            spin = self._table.cellWidget(row, col)
            if isinstance(spin, QDoubleSpinBox):
                spin.blockSignals(True)
                spin.setRange(lo - abs(lo), hi + abs(hi) + 1)
                spin.setValue(val)
                spin.blockSignals(False)

    # ------------------------------------------------------------------
    # Apply / Reset
    # ------------------------------------------------------------------

    def _apply_all(self) -> None:
        ds = self._state.active_dataset
        if ds is None:
            return

        mask = np.ones(ds.prop.num_loc, dtype=bool)

        for row in range(self._table.rowCount()):
            chk  = self._table.cellWidget(row, _COL_ENABLED)
            if not (isinstance(chk, QCheckBox) and chk.isChecked()):
                continue

            attr_combo = self._table.cellWidget(row, _COL_ATTR)
            mode_combo = self._table.cellWidget(row, _COL_MODE)
            min_spin   = self._table.cellWidget(row, _COL_MIN)
            max_spin   = self._table.cellWidget(row, _COL_MAX)
            if not all([attr_combo, mode_combo, min_spin, max_spin]):
                continue

            attr = attr_combo.currentText()
            mode = mode_combo.currentText()
            lo   = min_spin.value()
            hi   = max_spin.value()

            if attr not in ds.attr:
                continue

            raw  = np.asarray(ds.attr[attr]).ravel().astype(float)
            row_mask = _compute_mask(raw, mode, lo, hi, ds)
            mask &= row_mask

        self._applying = True
        ds.filter_mask = mask
        self._state.notify_filter_changed()
        self._applying = False

        n_pass = int(mask.sum())
        self._info.setText(
            f"{n_pass:,} / {ds.prop.num_loc:,} localisations pass all enabled filters."
        )

    def _reset_all(self) -> None:
        ds = self._state.active_dataset
        if ds is None:
            return
        ds.filter_mask = np.ones(ds.prop.num_loc, dtype=bool)
        self._state.notify_filter_changed()
        self._info.setText("All filters reset.")

    # ------------------------------------------------------------------
    # JSON save / load
    # ------------------------------------------------------------------

    def _save_json(self) -> None:
        ds = self._state.active_dataset
        default_dir = self._state.prefs["file"].get("default_folder", str(Path.home()))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save filter", default_dir, "Filter JSON (*.json)"
        )
        if not path:
            return

        rows = []
        for row in range(self._table.rowCount()):
            chk  = self._table.cellWidget(row, _COL_ENABLED)
            attr = self._table.cellWidget(row, _COL_ATTR)
            mode = self._table.cellWidget(row, _COL_MODE)
            lo   = self._table.cellWidget(row, _COL_MIN)
            hi   = self._table.cellWidget(row, _COL_MAX)
            rows.append({
                "apply":     isinstance(chk, QCheckBox) and chk.isChecked(),
                "attribute": attr.currentText() if isinstance(attr, QComboBox) else "",
                "value_as":  mode.currentText() if isinstance(mode, QComboBox) else "per loc",
                "min":       lo.value() if isinstance(lo, QDoubleSpinBox) else 0.0,
                "max":       hi.value() if isinstance(hi, QDoubleSpinBox) else 1.0,
            })
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
        self._info.setText(f"Saved: {Path(path).name}")

    def _load_json(self, path: str | None = None) -> None:
        """Load a JSON filter file and APPEND its rows to the table.

        If *path* is None a file-open dialog is shown.  Existing rows are
        kept; new rows are added at the bottom.
        """
        if path is None:
            default_dir = self._state.prefs["file"].get("default_folder", str(Path.home()))
            path, _ = QFileDialog.getOpenFileName(
                self, "Load filter", default_dir, "Filter JSON (*.json)"
            )
        if not path:
            return
        try:
            with open(path) as f:
                rows = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        added = 0
        for r in rows:
            self._add_row(
                attr    = r.get("attribute", ""),
                mode    = r.get("value_as", "per loc"),
                lo      = float(r.get("min", 0.0)),
                hi      = float(r.get("max", 1.0)),
                enabled = bool(r.get("apply", True)),
            )
            added += 1
        self._info.setText(f"Appended {added} row(s) from {Path(path).name}")
        if added:
            self._apply_all()

    # ------------------------------------------------------------------
    # Drag-and-drop  (JSON filter files)
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(".json"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(".json"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".json"):
                self._load_json(path)
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_active_changed(self, _idx: int) -> None:
        self._populate_attr_lists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spin(value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setDecimals(4)
    spin.setRange(-1e15, 1e15)
    spin.setValue(value)
    spin.setSingleStep(abs(value) * 0.01 if value != 0 else 0.01)
    return spin


def _compute_mask(
    raw: np.ndarray,
    mode: str,
    lo: float,
    hi: float,
    ds,
) -> np.ndarray:
    """Return a per-loc boolean mask for a single filter row."""
    from ..utils.filters import compute_filter_mask
    return compute_filter_mask(
        raw, mode, lo, hi,
        ds.prop.trace_idx,
        ds.prop.num_loc_per_trace,
        ds.prop.num_traces,
    )
