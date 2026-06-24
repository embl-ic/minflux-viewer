from __future__ import annotations

import csv
import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _is_numeric_dtype(dt) -> bool:
    try:
        return np.issubdtype(dt, np.integer) or np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.bool_)
    except Exception:
        return False


class _StateLogAdapter:
    """Callable log adapter used by the MSR parser and older tests."""

    def __init__(self, state) -> None:
        self._state = state

    def __call__(self, message: str) -> None:
        text = str(message)
        lowered = text.lower().lstrip()
        if lowered.startswith("[error]") or lowered.startswith("error"):
            level = "ERROR"
        elif lowered.startswith("[warn]") or lowered.startswith("warn"):
            level = "WARN"
        else:
            level = "INFO"
        self._state.log(text, level)


def _json_compact(node):
    """Readability transform for Imspector's XML-derived metadata dicts.

    Imspector OME metadata packs every XML element as a single-item list of
    dicts whose attributes are named keys and whose child content sits under
    an empty-string key. This unwraps single-element lists and hoists the
    ``""`` content next to the attributes, so ``{"Name": [{"": "x"}]}``
    reads as ``Name: x``. Generic JSON without those patterns passes through
    unchanged.
    """
    if isinstance(node, dict):
        if set(node.keys()) == {""}:
            return _json_compact(node[""])
        merged = {k: _json_compact(v) for k, v in node.items() if k != ""}
        if "" in node:
            content = _json_compact(node[""])
            if isinstance(content, dict):
                for key, val in content.items():
                    merged[key if key not in merged else f"{key} (content)"] = val
            else:
                merged["(text)" if "(text)" not in merged else "(text content)"] = content
        return merged
    if isinstance(node, (list, tuple)):
        items = [_json_compact(v) for v in node]
        return items[0] if len(items) == 1 else items
    return node


def _json_type_name(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (list, tuple)):
        return "list"
    return type(value).__name__


def _json_leaf_text(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _populate_metadata_items(parent_item, node, *, _budget: list[int] | None = None) -> None:
    """Hierarchical key/value items for (compacted) JSON metadata in the
    main reader tree.

    Main-tree columns are reused as: leaf -> "key: value" | type;
    container -> key | object/list | item count. ``_budget`` caps the total
    number of created items so a pathological metadata blob cannot freeze
    the tree.
    """
    if _budget is None:
        _budget = [5000]
    pairs = node.items() if isinstance(node, dict) else ((f"[{i}]", v) for i, v in enumerate(node))
    for key, value in pairs:
        if _budget[0] <= 0:
            parent_item.addChild(QTreeWidgetItem(["… (truncated)", "", "", ""]))
            return
        _budget[0] -= 1
        if isinstance(value, (dict, list, tuple)):
            n = len(value)
            count = f"{n} key{'s' if n != 1 else ''}" if isinstance(value, dict) else f"{n} item{'s' if n != 1 else ''}"
            item = QTreeWidgetItem([str(key), _json_type_name(value), count, ""])
            parent_item.addChild(item)
            _populate_metadata_items(item, value, _budget=_budget)
        else:
            item = QTreeWidgetItem([f"{key}: {_json_leaf_text(value)}", _json_type_name(value), "", ""])
            item.setToolTip(0, f"{key} = {_json_leaf_text(value)}")
            parent_item.addChild(item)


class JsonViewDialog(QDialog):
    """Browsable JSON / metadata viewer.

    Tree tab: key | value | type columns with a filter box, compact-view
    toggle (unwraps Imspector's XML packing), expand/collapse, and copy actions.
    Raw tab: the plain ``json.dumps`` text as before.
    """

    def __init__(self, title: str, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 680)
        self._data = data

        layout = QVBoxLayout(self)

        bar = QHBoxLayout()
        self._filter_edit = QLineEdit(self)
        self._filter_edit.setPlaceholderText("Filter keys and values…")
        self._filter_edit.setClearButtonEnabled(True)
        self._filter_edit.textChanged.connect(self._apply_filter)
        bar.addWidget(self._filter_edit, 1)
        self._compact_chk = QCheckBox("Compact", self)
        self._compact_chk.setChecked(True)
        self._compact_chk.setToolTip(
            "Unwrap single-element lists and XML content wrappers for readability."
        )
        self._compact_chk.toggled.connect(self._rebuild)
        bar.addWidget(self._compact_chk)
        expand_btn = QPushButton("Expand all", self)
        expand_btn.clicked.connect(lambda: self.tree.expandAll())
        bar.addWidget(expand_btn)
        collapse_btn = QPushButton("Collapse all", self)
        collapse_btn.clicked.connect(lambda: self.tree.collapseAll())
        bar.addWidget(collapse_btn)
        layout.addLayout(bar)

        self._tabs = QTabWidget(self)
        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Key", "Value", "Type"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_tree_menu)
        self._tabs.addTab(self.tree, "Tree")
        self._raw_edit = QPlainTextEdit(self)
        self._raw_edit.setReadOnly(True)
        self._raw_edit.setPlainText(self._dumps(data))
        self._tabs.addTab(self._raw_edit, "Raw JSON")
        layout.addWidget(self._tabs, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy_btn = buttons.addButton("Copy JSON", QDialogButtonBox.ButtonRole.ActionRole)
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self._dumps(self._data)))
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
        layout.addWidget(buttons)

        self._rebuild()

    @staticmethod
    def _dumps(data) -> str:
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)

    def _display_data(self):
        return _json_compact(self._data) if self._compact_chk.isChecked() else self._data

    def _rebuild(self) -> None:
        self.tree.clear()
        data = self._display_data()
        root = self.tree.invisibleRootItem()
        if isinstance(data, (dict, list, tuple)):
            self._populate(root, data, path="")
        else:
            self._add_leaf(root, "(value)", data, path="")
        self.tree.expandToDepth(2)
        for col in range(3):
            self.tree.resizeColumnToContents(col)
        self.tree.setColumnWidth(0, min(self.tree.columnWidth(0), 380))
        self._apply_filter(self._filter_edit.text())

    def _populate(self, parent, node, path: str) -> None:
        if isinstance(node, dict):
            pairs = node.items()
        else:
            pairs = ((f"[{i}]", v) for i, v in enumerate(node))
        for key, value in pairs:
            child_path = f"{path}/{key}" if path else str(key)
            if isinstance(value, (dict, list, tuple)):
                n = len(value)
                summary = f"{{{n} key{'s' if n != 1 else ''}}}" if isinstance(value, dict) else f"[{n} item{'s' if n != 1 else ''}]"
                item = QTreeWidgetItem([str(key), summary, _json_type_name(value)])
                item.setForeground(1, Qt.GlobalColor.gray)
                item.setForeground(2, Qt.GlobalColor.gray)
                item.setData(0, Qt.ItemDataRole.UserRole, child_path)
                item.setData(0, Qt.ItemDataRole.UserRole + 1, value)
                parent.addChild(item)
                self._populate(item, value, child_path)
            else:
                self._add_leaf(parent, str(key), value, child_path)

    def _add_leaf(self, parent, key: str, value, path: str) -> None:
        item = QTreeWidgetItem([key, _json_leaf_text(value), _json_type_name(value)])
        item.setForeground(2, Qt.GlobalColor.gray)
        item.setToolTip(1, _json_leaf_text(value))
        item.setData(0, Qt.ItemDataRole.UserRole, path)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, value)
        parent.addChild(item)

    # -- filtering ------------------------------------------------------

    def _apply_filter(self, text: str) -> None:
        needle = (text or "").strip().lower()

        def visit(item) -> bool:
            self_match = bool(needle) and (
                needle in item.text(0).lower() or needle in item.text(1).lower()
            )
            child_match = False
            for i in range(item.childCount()):
                if visit(item.child(i)):
                    child_match = True
            visible = not needle or self_match or child_match
            item.setHidden(not visible)
            if needle and child_match:
                item.setExpanded(True)
            return self_match or child_match

        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            visit(root.child(i))

    # -- context menu ---------------------------------------------------

    def _show_tree_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        copy_key = menu.addAction("Copy key")
        copy_value = menu.addAction("Copy value")
        copy_path = menu.addAction("Copy path")
        action = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if action is copy_key:
            QApplication.clipboard().setText(item.text(0))
        elif action is copy_value:
            value = item.data(0, Qt.ItemDataRole.UserRole + 1)
            text = self._dumps(value) if isinstance(value, (dict, list, tuple)) else _json_leaf_text(value)
            QApplication.clipboard().setText(text)
        elif action is copy_path:
            QApplication.clipboard().setText(str(item.data(0, Qt.ItemDataRole.UserRole) or ""))


class ArrayPreviewDialog(QDialog):
    def __init__(self, title: str, array, preview_rows: int | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 500)

        layout = QVBoxLayout(self)
        table = QTableWidget(self)
        layout.addWidget(table)

        arr = np.asarray(array)
        total_rows = arr.shape[0] if arr.ndim >= 1 else 0
        n = total_rows if preview_rows is None else min(preview_rows, total_rows)
        if arr.ndim == 1:
            arr2 = arr[:n].reshape(-1, 1)
        elif arr.ndim > 2:
            arr2 = arr[:n].reshape(n, -1)
        else:
            arr2 = arr[:n]

        table.setRowCount(n)
        table.setColumnCount(arr2.shape[1] if n else 1)
        table.setHorizontalHeaderLabels([str(i) for i in range(table.columnCount())])
        for row in range(n):
            for col in range(arr2.shape[1]):
                table.setItem(row, col, QTableWidgetItem(str(arr2[row, col])))
        table.resizeColumnsToContents()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
        layout.addWidget(buttons)


class PlotWindow(QDialog):
    def __init__(self, data, title: str = "Plot", parent=None, label: str | None = None, mode: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(950, 700)
        self._datasets: list[dict[str, Any]] = []
        self._owner = parent
        self._controls: dict[int, dict[str, Any]] = {}

        import pyqtgraph as pg

        self.plot = pg.PlotWidget(background="w")
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.getAxis("bottom").setPen("k")
        self.plot.getAxis("left").setPen("k")
        self.plot.getAxis("bottom").setTextPen("k")
        self.plot.getAxis("left").setTextPen("k")
        self.mode = QComboBox(self)
        self.mode.addItems(["line", "scatter", "histogram"])
        if mode in {"line", "scatter", "histogram"}:
            self.mode.setCurrentText(mode)
        self.mode.currentIndexChanged.connect(self._draw)

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Mode:"))
        top.addWidget(self.mode)
        top.addStretch(1)
        layout.addLayout(top)
        layout.addWidget(self.plot)
        self.layer_controls = QGroupBox("Plot layers")
        self.layer_controls_layout = QVBoxLayout(self.layer_controls)
        self.layer_controls_layout.setContentsMargins(6, 6, 6, 6)
        self.layer_controls_layout.setSpacing(4)
        layout.addWidget(self.layer_controls)
        self.add_data(data, label or title, redraw=True)

    @staticmethod
    def _to_series_matrix(a: np.ndarray) -> np.ndarray:
        if a.ndim == 1:
            return a.reshape(-1, 1)
        if a.ndim > 2:
            return a.reshape(a.shape[0], -1)
        r, c = a.shape
        if r in (2, 3) and c not in (2, 3):
            return a.T
        return a

    @staticmethod
    def _finite_bounds(series: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray] | None:
        m = np.asarray(series, dtype=float)
        if m.size == 0:
            return None
        if mode == "scatter" and m.ndim == 2 and m.shape[1] >= 2:
            vals = m[:, :2]
        elif mode == "line":
            vals = m.reshape(m.shape[0], -1)
            vals = np.column_stack([np.arange(vals.shape[0]), np.nanmin(vals, axis=1)])
        else:
            vals = m.reshape(-1, 1)
        vals = vals[np.all(np.isfinite(vals), axis=1)]
        if vals.size == 0:
            return None
        return vals.min(axis=0), vals.max(axis=0)

    def add_data(self, data, label: str, redraw: bool = True) -> None:
        import pyqtgraph as pg

        series = self._to_series_matrix(np.asarray(data))
        idx = len(self._datasets)
        color = pg.intColor(idx, hues=max(idx + 2, 2))
        self._datasets.append({
            "label": label,
            "series": series,
            "visible": True,
            "symbol": "o",
            "size": 6,
            "color": (color.red(), color.green(), color.blue()),
            "alpha": 120,
        })
        self._add_layer_control(idx)
        if redraw:
            self._draw()

    def _add_layer_control(self, idx: int) -> None:
        ds = self._datasets[idx]
        row = QWidget(self.layer_controls)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        visible = QCheckBox(ds["label"])
        visible.setChecked(True)
        visible.toggled.connect(lambda checked, i=idx: self._set_layer_visible(i, checked))
        layout.addWidget(visible, stretch=1)
        style_button = QPushButton("plot style")
        style_button.clicked.connect(lambda _checked=False, i=idx: self._edit_layer_style(i))
        layout.addWidget(style_button)
        self.layer_controls_layout.addWidget(row)
        self._controls[idx] = {"row": row, "visible": visible, "style": style_button}

    def _set_layer_visible(self, idx: int, visible: bool) -> None:
        if 0 <= idx < len(self._datasets):
            self._datasets[idx]["visible"] = bool(visible)
            self._draw()

    def _edit_layer_style(self, idx: int) -> None:
        if not (0 <= idx < len(self._datasets)):
            return
        dlg = PlotStyleDialog(self._datasets[idx], self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._datasets[idx].update(dlg.result_payload())
            self._draw()

    def can_overlay(self, data) -> tuple[bool, str]:
        """Return whether *data* is visually compatible with this plot window."""
        if not self._datasets:
            return True, ""
        mode = self.mode.currentText()
        new_series = self._to_series_matrix(np.asarray(data))
        existing_bounds = [
            self._finite_bounds(ds["series"], mode)
            for ds in self._datasets
        ]
        existing_bounds = [b for b in existing_bounds if b is not None]
        new_bounds = self._finite_bounds(new_series, mode)
        if not existing_bounds or new_bounds is None:
            return True, ""
        old_min = np.min(np.vstack([b[0] for b in existing_bounds]), axis=0)
        old_max = np.max(np.vstack([b[1] for b in existing_bounds]), axis=0)
        new_min, new_max = new_bounds
        old_span = np.maximum(old_max - old_min, 1e-12)
        combined_min = np.minimum(old_min, new_min)
        combined_max = np.maximum(old_max, new_max)
        combined_span = combined_max - combined_min
        if np.any(combined_span > old_span * 100.0):
            return False, "new data range is more than 100x wider than the active plot range"
        new_center = (new_min + new_max) / 2.0
        old_center = (old_min + old_max) / 2.0
        if np.any(np.abs(new_center - old_center) > old_span * 50.0):
            return False, "new data range is far away from the active plot range"
        return True, ""

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if hasattr(self._owner, "_on_plot_window_focused"):
            self._owner._on_plot_window_focused(self)

    def closeEvent(self, event):
        if hasattr(self._owner, "_on_plot_window_closed"):
            self._owner._on_plot_window_closed(self)
        super().closeEvent(event)

    def _draw(self):
        import pyqtgraph as pg

        self.plot.clear()
        legend = getattr(self.plot.plotItem, "legend", None)
        if legend is None:
            self.plot.addLegend(offset=(10, 10))
        else:
            legend.clear()
        self.plot.setAspectLocked(False)
        mode = self.mode.currentText()
        if mode == "line":
            for ds_idx, ds in enumerate(self._datasets):
                if not ds.get("visible", True):
                    continue
                m = ds["series"]
                cols = m.shape[1] if m.ndim == 2 else 1
                x = np.arange(m.shape[0])
                for i in range(cols):
                    r, g, b = ds["color"]
                    color = pg.mkColor(r, g, b, min(int(ds["alpha"]) + 70, 255))
                    name = ds["label"] if cols == 1 else f"{ds['label']} [{i}]"
                    self.plot.plot(x, m[:, i], pen=pg.mkPen(color, width=1.5), name=name)
            return
        if mode == "scatter":
            self.plot.setLabel("bottom", "x")
            self.plot.setLabel("left", "y")
            self.plot.setAspectLocked(True)
            for ds_idx, ds in enumerate(self._datasets):
                if not ds.get("visible", True):
                    continue
                m = ds["series"]
                r, g, b = ds["color"]
                brush = pg.mkBrush(r, g, b, int(ds["alpha"]))
                pen = pg.mkPen(r, g, b, min(int(ds["alpha"]) + 70, 255))
                if m.shape[1] >= 2:
                    x, y = m[:, 0], m[:, 1]
                else:
                    x, y = np.arange(m.shape[0]), m[:, 0]
                self.plot.plot(
                    x, y, pen=None, symbol=ds["symbol"], symbolSize=int(ds["size"]),
                    symbolBrush=brush, symbolPen=pen, name=ds["label"],
                )
            return
        self.plot.setLabel("bottom", "value")
        self.plot.setLabel("left", "count")
        for ds_idx, ds in enumerate(self._datasets):
            if not ds.get("visible", True):
                continue
            m = ds["series"]
            cols = m.shape[1] if m.ndim == 2 else 1
            for i in range(cols):
                values = m[:, i]
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                h, edges = np.histogram(values, bins=50)
                x = 0.5 * (edges[:-1] + edges[1:])
                r, g, b = ds["color"]
                color = pg.mkColor(r, g, b, int(ds["alpha"]))
                name = ds["label"] if cols == 1 else f"{ds['label']} [{i}]"
                self.plot.plot(
                    x, h, stepMode=False, fillLevel=0,
                    brush=pg.mkBrush(color), pen=pg.mkPen(color, width=1.2), name=name,
                )


class PlotStyleDialog(QDialog):
    def __init__(self, layer: dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Plot style: {layer.get('label', '')}")
        self.resize(360, 180)
        self._color = tuple(layer.get("color", (30, 90, 180)))

        layout = QVBoxLayout(self)
        grid = QGridLayout()
        layout.addLayout(grid)

        grid.addWidget(QLabel("Shape:"), 0, 0)
        self.symbol_combo = QComboBox()
        symbols = [("circle", "o"), ("square", "s"), ("triangle", "t"), ("diamond", "d"), ("plus", "+"), ("x", "x"), ("star", "star")]
        for label, value in symbols:
            self.symbol_combo.addItem(label, value)
        idx = self.symbol_combo.findData(layer.get("symbol", "o"))
        if idx >= 0:
            self.symbol_combo.setCurrentIndex(idx)
        grid.addWidget(self.symbol_combo, 0, 1)

        grid.addWidget(QLabel("Size:"), 1, 0)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 50)
        self.size_spin.setValue(int(layer.get("size", 6)))
        grid.addWidget(self.size_spin, 1, 1)

        grid.addWidget(QLabel("Transparency:"), 2, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 100.0)
        self.alpha_spin.setSuffix(" %")
        self.alpha_spin.setValue(round(100.0 * (1.0 - int(layer.get("alpha", 120)) / 255.0)))
        grid.addWidget(self.alpha_spin, 2, 1)

        grid.addWidget(QLabel("Color:"), 3, 0)
        self.color_button = QPushButton(self._color_label())
        self.color_button.clicked.connect(self._choose_color)
        grid.addWidget(self.color_button, 3, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _color_label(self) -> str:
        return f"RGB {self._color[0]}, {self._color[1]}, {self._color[2]}"

    def _choose_color(self) -> None:
        from PyQt6.QtGui import QColor
        color = QColorDialog.getColor(QColor(*self._color), self, "Choose plot color")
        if color.isValid():
            self._color = (color.red(), color.green(), color.blue())
            self.color_button.setText(self._color_label())

    def result_payload(self) -> dict[str, Any]:
        transparency = float(self.alpha_spin.value()) / 100.0
        alpha = int(round(255.0 * (1.0 - transparency)))
        return {
            "symbol": self.symbol_combo.currentData(),
            "size": int(self.size_spin.value()),
            "color": self._color,
            "alpha": max(0, min(255, alpha)),
        }


class AlignmentSaveDialog(QDialog):
    def __init__(self, initial_paths: dict[str, str], initial_options: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save channel alignment outputs")
        self.resize(820, 240)

        # Remembered checkbox states from the last run (persisted in settings).
        opts = initial_options or {}
        self.save_beads = QCheckBox("Average beads CSV (all aligned channels)")
        self.save_beads.setChecked(bool(opts.get("save_beads", True)))
        self.save_transform = QCheckBox("Rigid transform CSV (all aligned channels)")
        self.save_transform.setChecked(bool(opts.get("save_transform", True)))
        self.save_localizations = QCheckBox("Aligned localizations CSV (all aligned channels)")
        self.save_localizations.setChecked(bool(opts.get("save_localizations", False)))
        self.show_plot = QCheckBox("Show beads and alignment result")
        self.show_plot.setChecked(bool(opts.get("show_plot", False)))

        self.path_beads = QLineEdit(initial_paths["beads"])
        self.path_transform = QLineEdit(initial_paths["transform"])
        self.path_localizations = QLineEdit(initial_paths["localizations"])

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Use <b>Show beads drift</b> to manually check bead drift and select "
            "beads for alignment — otherwise all common beads are used to compute "
            "the alignment transformation.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        layout.addWidget(hint)
        grid = QGridLayout()
        layout.addLayout(grid)
        self._add_path_row(grid, 0, self.save_beads, self.path_beads)
        self._add_path_row(grid, 1, self.save_transform, self.path_transform)
        self._add_path_row(grid, 2, self.save_localizations, self.path_localizations)
        layout.addWidget(self.show_plot)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_path_row(self, grid, row: int, checkbox, lineedit):
        button = QPushButton("Browse…")
        button.clicked.connect(lambda: self._browse(lineedit))
        grid.addWidget(checkbox, row, 0)
        grid.addWidget(lineedit, row, 1)
        grid.addWidget(button, row, 2)

    def _browse(self, lineedit):
        current = lineedit.text().strip()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose output CSV",
            current or "",
            "CSV (*.csv);;All files (*.*)",
        )
        if path:
            lineedit.setText(path)

    def result_payload(self):
        return {
            "save_beads": self.save_beads.isChecked(),
            "save_transform": self.save_transform.isChecked(),
            "save_localizations": self.save_localizations.isChecked(),
            "show_plot": self.show_plot.isChecked(),
            "path_beads": self.path_beads.text().strip(),
            "path_transform": self.path_transform.text().strip(),
            "path_localizations": self.path_localizations.text().strip(),
        }


class FieldDialog(QDialog):
    def __init__(self, datasets: list[dict], prechecked: Optional[dict[str, dict]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Datasets / Fields included…")
        self.resize(max(500, 320 * max(1, len(datasets))), 520)
        self._vars: dict[str, dict[str, Any]] = {}

        root = QVBoxLayout(self)
        outer = QHBoxLayout()
        root.addLayout(outer)

        for ds in datasets:
            prev = prechecked.get(ds["key"], {}) if prechecked else {}
            panel = QGroupBox(ds["name"])
            panel_layout = QVBoxLayout(panel)
            include = QCheckBox("Include dataset")
            include.setChecked(prev.get("checked", True))
            panel_layout.addWidget(include)

            mfx_all = QCheckBox("All mfx fields")
            mfx_all.setChecked(prev.get("mfx") is None if prechecked else True)
            mfx_box = QGroupBox("mfx fields")
            mfx_layout = QVBoxLayout(mfx_box)
            mfx_layout.addWidget(mfx_all)
            mfx_checks = {}
            pre_mfx = set(prev.get("mfx") or []) if prechecked else set()
            for name in ds.get("mfx_fields", []):
                cb = QCheckBox(name)
                cb.setChecked(True if (prev.get("mfx") is None if prechecked else True) else name in pre_mfx)
                mfx_layout.addWidget(cb)
                mfx_checks[name] = cb
            mfx_all.toggled.connect(lambda checked, checks=mfx_checks: [cb.setChecked(checked) for cb in checks.values()])

            mbm_all = QCheckBox("All mbm fields")
            mbm_all.setChecked(prev.get("mbm") is None if prechecked else True)
            mbm_box = QGroupBox("mbm fields")
            mbm_layout = QVBoxLayout(mbm_box)
            mbm_layout.addWidget(mbm_all)
            mbm_checks = {}
            pre_mbm = set(prev.get("mbm") or []) if prechecked else set()
            for name in ds.get("mbm_fields", []):
                cb = QCheckBox(name)
                cb.setChecked(True if (prev.get("mbm") is None if prechecked else True) else name in pre_mbm)
                mbm_layout.addWidget(cb)
                mbm_checks[name] = cb
            mbm_all.toggled.connect(lambda checked, checks=mbm_checks: [cb.setChecked(checked) for cb in checks.values()])

            panel_layout.addWidget(mfx_box)
            panel_layout.addWidget(mbm_box)
            panel_layout.addStretch(1)
            outer.addWidget(panel)
            self._vars[ds["key"]] = {
                "ds": include,
                "mfx_all": mfx_all,
                "mfx": mfx_checks,
                "mbm_all": mbm_all,
                "mbm": mbm_checks,
            }

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def result_payload(self):
        out = {}
        for key, vv in self._vars.items():
            out[key] = {
                "checked": vv["ds"].isChecked(),
                "mfx": None if vv["mfx_all"].isChecked() else {name for name, cb in vv["mfx"].items() if cb.isChecked()},
                "mbm": None if vv["mbm_all"].isChecked() else {name for name, cb in vv["mbm"].items() if cb.isChecked()},
            }
        return out


# Sentinel: user cancelled the alignment dialog -> abort the whole
# "Open in MINFLUX viewer" action (distinct from None = "not applicable").
_ALIGN_CANCELLED = object()


class ViewerAlignmentDialog(QDialog):
    """Choose the reference dataset and alignment mode for a multi-channel open.

    ``mbm_check(ref_name) -> (available, reason)`` re-evaluates bead-based
    alignment availability whenever the reference choice changes — matched
    bead counts depend on which dataset is the reference. Cancelling this
    dialog aborts the whole "Open in MINFLUX viewer" action.
    """

    def __init__(self, selected: list[str], mbm_check, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open in MINFLUX viewer")
        self.resize(520, 180)
        self._mbm_check = mbm_check

        layout = QVBoxLayout(self)
        label = QLabel(
            "This MSR file contains multiple datasets.\n"
            "Choose how to align them when opening as a multi-channel render overlay."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference dataset:"))
        self.ref_combo = QComboBox(self)
        self.ref_combo.addItems(selected)
        self.ref_combo.setCurrentIndex(0)
        self.ref_combo.currentTextChanged.connect(self._on_reference_changed)
        ref_row.addWidget(self.ref_combo, 1)
        layout.addLayout(ref_row)

        row = QHBoxLayout()
        row.addWidget(QLabel("align with:"))
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
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_reference_changed(self.ref_combo.currentText())

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


class AlignmentPlotWindow(QDialog):
    def __init__(self, results, parent=None, data_bounds_nm=None):
        super().__init__(parent)
        self.setWindowTitle("Beads and alignment result")
        self.resize(1150, 760)
        self.results = results
        self.datasets = self._build_plot_datasets(results)
        # (min_xyz, max_xyz) nm bounding box of all datasets' raw localizations,
        # drawn as a view-aware yellow data-range rectangle. None = no box.
        self.data_bounds_nm = data_bounds_nm
        self.visibility = {}
        self._items = {}
        self._axis_indices = (0, 1)

        import pyqtgraph as pg

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Visible channels:"))
        for key, meta in self.datasets.items():
            cb = QCheckBox(meta["label"])
            cb.setChecked(True)
            cb.toggled.connect(self._update_visibility)
            self.visibility[key] = cb
            controls.addWidget(cb)
        controls.addWidget(QLabel("View:"))
        self.view_mode = QComboBox()
        self.view_mode.addItems(["XY", "XZ", "YZ"])
        self.view_mode.currentTextChanged.connect(self._on_view_changed)
        controls.addWidget(self.view_mode)
        controls.addStretch(1)
        self.hover_label = QLabel("Hover near a point to see bead ID, channel, and coordinates (nm).")
        controls.addWidget(self.hover_label)
        layout.addLayout(controls)

        self.plot = pg.PlotWidget(background="w")
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.setAspectLocked(True)
        self.plot.addLegend()
        self.plot.getAxis("bottom").setPen("k")
        self.plot.getAxis("left").setPen("k")
        self.plot.getAxis("bottom").setTextPen("k")
        self.plot.getAxis("left").setTextPen("k")
        layout.addWidget(self.plot, stretch=1)
        self.plot.scene().sigMouseMoved.connect(self._on_hover)
        self._draw()

    def _build_plot_datasets(self, results):
        cmap = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        reference = results[0]
        datasets = {
            "reference_original": {
                "label": f"{reference.channel1_name} original",
                "color": cmap[0],
                "xyz_nm": np.asarray(reference.channel1_xyz, dtype=np.float64),
                "bead_ids": np.asarray(reference.bead_ids),
                "symbol": "o",
            }
        }
        for idx, result in enumerate(results, start=1):
            color = cmap[idx % len(cmap)]
            base = f"channel_{idx+1}"
            datasets[f"{base}_original"] = {
                "label": f"{result.channel2_name} original",
                "color": color,
                "xyz_nm": np.asarray(result.channel2_xyz, dtype=np.float64),
                "bead_ids": np.asarray(result.bead_ids),
                "symbol": "o",
            }
            datasets[f"{base}_aligned"] = {
                "label": f"{result.channel2_name} aligned",
                "color": color,
                "xyz_nm": np.asarray(result.channel2_aligned_xyz, dtype=np.float64),
                "bead_ids": np.asarray(result.bead_ids),
                "symbol": "star",
            }
        return datasets

    def _draw(self):
        import pyqtgraph as pg

        self.plot.clear()
        if getattr(self.plot.plotItem, "legend", None) is not None:
            self.plot.plotItem.legend.clear()
        self._items = {}
        x_idx, y_idx = self._axis_indices
        for key, meta in self.datasets.items():
            xyz = meta["xyz_nm"]
            brush = pg.mkBrush(meta["color"] + "80")
            pen = pg.mkPen(meta["color"], width=1)
            item = pg.ScatterPlotItem(
                xyz[:, x_idx],
                xyz[:, y_idx],
                size=9,
                symbol=meta["symbol"],
                brush=brush,
                pen=pen,
                name=meta["label"],
            )
            item.setVisible(self.visibility[key].isChecked())
            self.plot.addItem(item)
            self._items[key] = item
        self._draw_data_range_box()
        labels = self._axis_labels()
        self.plot.setLabel("bottom", labels[0], units="nm")
        self.plot.setLabel("left", labels[1], units="nm")
        self.plot.setTitle("Bead positions before and after channel alignment")
        self._set_equal_range()

    def _data_box_xy(self):
        """(x0, y0, x1, y1) of the data-range box for the current view, or None."""
        if self.data_bounds_nm is None:
            return None
        mins, maxs = self.data_bounds_nm
        x_idx, y_idx = self._axis_indices
        if x_idx >= len(mins) or y_idx >= len(mins):
            return None
        return (float(mins[x_idx]), float(mins[y_idx]),
                float(maxs[x_idx]), float(maxs[y_idx]))

    def _draw_data_range_box(self):
        import pyqtgraph as pg

        box = self._data_box_xy()
        if box is None:
            return
        x0, y0, x1, y1 = box
        self.plot.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            pen=pg.mkPen((230, 200, 0), width=2),
            name="data range",
        )

    def _update_visibility(self):
        if not self._items:
            return
        for key, item in self._items.items():
            item.setVisible(self.visibility[key].isChecked())

    def _axis_labels(self):
        mode = self.view_mode.currentText().upper()
        if mode == "XZ":
            return "x", "z"
        if mode == "YZ":
            return "y", "z"
        return "x", "y"

    def _on_view_changed(self):
        mode = self.view_mode.currentText().upper()
        if mode == "XZ":
            self._axis_indices = (0, 2)
        elif mode == "YZ":
            self._axis_indices = (1, 2)
        else:
            self._axis_indices = (0, 1)
        self._draw()

    def _set_equal_range(self):
        visible_xyz = [meta["xyz_nm"] for key, meta in self.datasets.items() if self.visibility[key].isChecked()]
        if not visible_xyz:
            visible_xyz = [meta["xyz_nm"] for meta in self.datasets.values()]
        xyz = np.vstack(visible_xyz)
        x_idx, y_idx = self._axis_indices
        xy = xyz[:, [x_idx, y_idx]]
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        box = self._data_box_xy()
        if box is not None:
            corners = np.array([[box[0], box[1]], [box[2], box[3]]], dtype=np.float64)
            xy = np.vstack([xy, corners]) if xy.size else corners
        if xy.size == 0:
            return
        mins = xy.min(axis=0)
        maxs = xy.max(axis=0)
        centers = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
        self.plot.setXRange(centers[0] - radius, centers[0] + radius, padding=0)
        self.plot.setYRange(centers[1] - radius, centers[1] + radius, padding=0)

    def _best_hover_target(self, scene_pos):
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            return None
        point = self.plot.plotItem.vb.mapSceneToView(scene_pos)
        mouse_xy = np.array([point.x(), point.y()], dtype=np.float64)
        view_range = self.plot.plotItem.vb.viewRange()
        sx = max(abs(view_range[0][1] - view_range[0][0]), 1.0)
        sy = max(abs(view_range[1][1] - view_range[1][0]), 1.0)
        best = None
        x_idx, y_idx = self._axis_indices
        for key in self.datasets:
            if not self.visibility[key].isChecked():
                continue
            xyz = self.datasets[key]["xyz_nm"][:, [x_idx, y_idx]]
            scaled = np.column_stack([(xyz[:, 0] - mouse_xy[0]) / sx, (xyz[:, 1] - mouse_xy[1]) / sy])
            dist2 = np.sum(scaled ** 2, axis=1)
            idx = int(np.argmin(dist2))
            best_dist2 = float(dist2[idx])
            if best is None or best_dist2 < best["dist2"]:
                best = {"key": key, "index": idx, "dist2": best_dist2}
        return best

    def _on_hover(self, scene_pos):
        target = self._best_hover_target(scene_pos)
        if target is None or target["dist2"] > 0.0004:
            self.hover_label.setText("Hover near a point to see bead ID, channel, and coordinates (nm).")
            return
        key = target["key"]
        idx = target["index"]
        xyz = self.datasets[key]["xyz_nm"][idx]
        bead_id = int(self.datasets[key]["bead_ids"][idx])
        self.hover_label.setText(
            f"bead ID {bead_id} | {self.datasets[key]['label']} | "
            f"x {xyz[0]:.3f} nm, y {xyz[1]:.3f} nm, z {xyz[2]:.3f} nm"
        )


class _ParseWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, msr_path: str, tmp_dir: str):
        super().__init__()
        self._msr_path = msr_path
        self._tmp_dir = tmp_dir

    def run(self):
        try:
            from ...msr import parse_msr_general
            result = parse_msr_general(
                self._msr_path,
                self._tmp_dir,
                log=lambda msg: self.progress.emit(msg),
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class MsrReaderDialog(QWidget):
    """
    MINFLUX .msr parser & exporter.  Near-verbatim PyQt6 port of
    D:/Git/MINFLUX_msr_reader/ui_qt/app.py with viewer integration.
    """

    def __init__(
        self,
        state=None,
        msr_path: str | None = None,
        parent: QWidget | None = None,
    ):
        # Deliberately a NON-owned top-level window: passing the main window as
        # the QWidget parent (with the Qt.Window flag) makes the OS keep this
        # dialog permanently above its owner in Z-order, so it sits in front of
        # every viewer window. The viewer plot windows avoid this by having no
        # parent; mirror that here and keep an explicit owner reference for the
        # few main-window lookups below (``self._owner``).
        super().__init__(None)
        self._owner = parent
        self.setWindowTitle("MINFLUX .msr Reader & Converter")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(980, 860)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._positioned = False

        self._state = state
        self._worker: _ParseWorker | None = None

        self.parsed: dict[str, Any] = {}
        self.field_selection = {}
        # gri-IDs the user excluded via "Show Beads Drift". None until the user
        # applies a selection (None/empty = use all beads, as before).
        self._bead_unchecked_gris = None
        # Remembered "Align channel" save-dialog checkbox states (persisted).
        self._alignment_save_options: dict = {}
        self.nodeinfo: dict[int, dict[str, Any]] = {}
        self.fullpath_by_item: dict[int, str] = {}
        self.datasetnode_info: dict[int, dict[str, Any]] = {}
        self._plot_windows: list[PlotWindow] = []
        self._active_plot_window: PlotWindow | None = None
        self._child_dialogs: list[QDialog] = []

        settings = self._load_settings(tempfile.gettempdir())
        self._build_ui(settings)

        if msr_path:
            self.input_path_edit.setText(msr_path)
            self._last_input_dir = str(Path(msr_path).parent)
            self._save_settings()
            self._start_parse_worker(msr_path, self._temp_dir())

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _settings_path(self) -> Path:
        return Path.home() / ".minflux_viewer_msr_settings.json"

    def _temp_dir(self) -> str:
        """App-wide temp folder (Preferences > File > Temporary folder), else the
        system temp dir. This is the single place a temp folder is defined."""
        prefs = getattr(self._state, "prefs", None)
        folder = ""
        if isinstance(prefs, dict):
            folder = str((prefs.get("file") or {}).get("temp_folder", "") or "").strip()
        return folder or tempfile.gettempdir()

    def _load_settings(self, system_tmp: str) -> dict[str, Any]:
        path = self._settings_path()
        defaults = {
            "out_dir": system_tmp,
            "last_input_dir": str(Path.home()),
            "preview_rows": 100,
            "preview_all_rows": False,
            "mode_folder": False,
            "plot_new_window": True,
            "formats": {"mat": True, "npy": False, "json": False, "csv": False, "zarr": False},
            "field_selection": {},
        }
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    merged = dict(defaults)
                    merged.update(data)
                    if not isinstance(merged.get("formats"), dict):
                        merged["formats"] = defaults["formats"]
                    if not isinstance(merged.get("field_selection"), dict):
                        merged["field_selection"] = {}
                    return merged
        except Exception:
            pass
        return defaults

    def _save_settings(self):
        def clean_selection(selection):
            out = {}
            for key, value in (selection or {}).items():
                out[key] = {
                    "checked": bool(value.get("checked", True)),
                    "mfx": None if value.get("mfx") is None else sorted(value.get("mfx") or []),
                    "mbm": None if value.get("mbm") is None else sorted(value.get("mbm") or []),
                }
            return out

        input_path = self.input_path_edit.text().strip()
        last_input_dir = input_path
        if input_path and Path(input_path).is_file():
            last_input_dir = str(Path(input_path).parent)
        elif not input_path:
            last_input_dir = getattr(self, "_last_input_dir", str(Path.home()))
        data = {
            "out_dir": self.out_dir_edit.text().strip() or self._temp_dir(),
            "last_input_dir": last_input_dir,
            "preview_rows": int(self.preview_rows_spin.value()),
            "preview_all_rows": bool(self.preview_all_rows_check.isChecked()),
            "mode_folder": bool(self.mode_folder.isChecked()),
            "plot_new_window": bool(self.plot_new_window_check.isChecked()),
            "formats": {
                "mat": bool(self.fmt_mat.isChecked()),
                "npy": bool(self.fmt_npy.isChecked()),
                "json": bool(self.fmt_json.isChecked()),
                "csv": bool(self.fmt_csv.isChecked()),
                "zarr": bool(self.fmt_zarr.isChecked()),
            },
            "field_selection": clean_selection(self.field_selection),
            "alignment_save": dict(self._alignment_save_options or {}),
        }
        try:
            self._settings_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._last_input_dir = last_input_dir
        except Exception as exc:
            self.log(f"[warn] could not save settings: {exc}")

    def showEvent(self, event):
        super().showEvent(event)
        # Non-owned top-level window: place it fully on the active screen on
        # first show (it is tall — 860 px — and otherwise opens with its title
        # bar above the screen top on shorter / scaled displays).
        if not self._positioned:
            self._positioned = True
            from ...ui.modeless import ensure_on_screen
            ensure_on_screen(self, self._owner)

    def closeEvent(self, event):
        self._save_settings()
        for win in list(self._plot_windows):
            try:
                win.close()
            except Exception:
                pass
        self._plot_windows.clear()
        self._active_plot_window = None
        for dlg in list(self._child_dialogs):
            try:
                dlg.close()
            except Exception:
                pass
        self._child_dialogs.clear()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, settings):
        root = QVBoxLayout(self)

        input_type_row = QHBoxLayout()
        input_type_row.addWidget(QLabel("Input type:"))
        self.mode_file = QRadioButton("Single .msr file")
        self.mode_folder = QRadioButton("Folder (batch)")
        self.mode_folder.setChecked(bool(settings.get("mode_folder", False)))
        self.mode_file.setChecked(not self.mode_folder.isChecked())
        input_type_row.addWidget(self.mode_file)
        input_type_row.addWidget(self.mode_folder)
        input_type_row.addStretch(1)
        root.addLayout(input_type_row)

        self.input_path_edit = QLineEdit()
        self._last_input_dir = settings.get("last_input_dir") or str(Path.home())
        root.addLayout(self._browse_row("MSR file or folder:", self.input_path_edit, self.browse_input))

        action_row = QHBoxLayout()
        self._parse_button = QPushButton("Parse MSR file")
        self._parse_button.clicked.connect(self.on_parse)
        action_row.addWidget(self._parse_button)
        action_row.addWidget(QLabel("Preview"))
        self.preview_all_rows_check = QCheckBox("all rows")
        self.preview_all_rows_check.setChecked(bool(settings.get("preview_all_rows", False)))
        action_row.addWidget(self.preview_all_rows_check)
        self.preview_rows_spin = QSpinBox()
        self.preview_rows_spin.setRange(1, 1_000_000)
        self.preview_rows_spin.setValue(int(settings.get("preview_rows", 100) or 100))
        action_row.addWidget(self.preview_rows_spin)
        self.preview_all_rows_check.toggled.connect(self._on_preview_all_rows_toggled)
        self.preview_rows_spin.valueChanged.connect(lambda _value: self._save_settings())
        self._on_preview_all_rows_toggled(self.preview_all_rows_check.isChecked())
        self.plot_new_window_check = QCheckBox("plot in new window")
        self.plot_new_window_check.setChecked(bool(settings.get("plot_new_window", True)))
        self.plot_new_window_check.toggled.connect(lambda _checked: self._save_settings())
        action_row.addWidget(self.plot_new_window_check)
        action_row.addStretch(1)
        root.addLayout(action_row)

        tree_group = QGroupBox("Parsed content")
        tree_layout = QVBoxLayout(tree_group)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Path / Info", "Kind", "Shape", "DType"])
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        tree_layout.addWidget(self.tree)
        root.addWidget(tree_group, 1)

        self.out_dir_edit = QLineEdit(settings["out_dir"])
        root.addLayout(self._browse_row("Output folder:", self.out_dir_edit, self.browse_out))

        export_row = QHBoxLayout()
        field_button = QPushButton("Datasets / Fields included…")
        field_button.clicked.connect(self.on_fields_dialog)
        export_row.addWidget(field_button)
        export_row.addWidget(QLabel("Formats:"))
        self.fmt_mat = QCheckBox("MATLAB (.mat)")
        self.fmt_npy = QCheckBox("NumPy (.npy)")
        self.fmt_json = QCheckBox("JSON (.json)")
        self.fmt_csv = QCheckBox("(.csv)")
        self.fmt_zarr = QCheckBox("Zarr (.zarr)")
        self.fmt_zarr.setToolTip(
            "Write the dataset's embedded zarr store to a .zarr directory "
            "(the full MINFLUX container: mfx + beads)."
        )
        formats = settings.get("formats") or {}
        self.fmt_mat.setChecked(bool(formats.get("mat", True)))
        self.fmt_npy.setChecked(bool(formats.get("npy", False)))
        self.fmt_json.setChecked(bool(formats.get("json", False)))
        self.fmt_csv.setChecked(bool(formats.get("csv", False)))
        self.fmt_zarr.setChecked(bool(formats.get("zarr", False)))
        export_row.addWidget(self.fmt_mat)
        export_row.addWidget(self.fmt_npy)
        export_row.addWidget(self.fmt_json)
        export_row.addWidget(self.fmt_csv)
        export_row.addWidget(self.fmt_zarr)
        export_row.addStretch(1)
        root.addLayout(export_row)

        self.logbox = None
        self.field_selection = settings.get("field_selection") or {}
        self._alignment_save_options = settings.get("alignment_save") or {}

        bottom = QHBoxLayout()
        self._beads_drift_btn = QPushButton("Show beads drift")
        self._beads_drift_btn.setToolTip(
            "Inspect MBM/bead drift per dataset and manually select beads for "
            "alignment.")
        self._beads_drift_btn.clicked.connect(self._on_show_beads_drift)
        align_button = QPushButton("Align channel")
        align_button.clicked.connect(self.on_align_channel)
        self._open_viewer_btn = QPushButton("Open in MINFLUX viewer")
        self._open_viewer_btn.clicked.connect(self._on_open_in_viewer)
        self._open_viewer_btn.setEnabled(self._state is not None)
        export_button = QPushButton("OK (export)")
        export_button.clicked.connect(self.on_ok)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.close)
        bottom.addWidget(self._beads_drift_btn)
        bottom.addWidget(align_button)
        bottom.addWidget(self._open_viewer_btn)
        bottom.addStretch(1)
        bottom.addWidget(cancel_button)
        bottom.addWidget(export_button)
        root.addLayout(bottom)

    def _browse_row(self, label: str, lineedit: QLineEdit, callback):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(lineedit, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(callback)
        row.addWidget(button)
        return row

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, msg: str):
        if self._state is not None:
            level = "ERROR" if "[error]" in msg.lower() or "[ERROR]" in msg else "INFO"
            self._state.log(msg, level)
            return
        if self.logbox is not None:
            self.logbox.appendPlainText(msg)
        print(msg)

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def browse_input(self):
        start_dir = self._last_input_dir if os.path.exists(str(self._last_input_dir)) else str(Path.home())
        if self.mode_file.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "Choose an .msr file", start_dir, "Imspector .msr (*.msr);;All files (*.*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Choose a folder of .msr files", start_dir)
        if path:
            self.input_path_edit.setText(path)
            self._last_input_dir = str(Path(path).parent if Path(path).is_file() else Path(path))
            self._save_settings()

    def browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output folder", self.out_dir_edit.text().strip() or self._temp_dir())
        if path:
            self.out_dir_edit.setText(path)
            self._save_settings()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _item_id(self, item) -> int:
        return id(item)

    def _ordinal(self, value: int) -> str:
        if 10 <= (value % 100) <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
        return f"{value}{suffix}"

    def _set_item_tooltip(self, item: QTreeWidgetItem, text: str):
        for column in range(self.tree.columnCount()):
            item.setToolTip(column, text)

    def _on_preview_all_rows_toggled(self, checked: bool) -> None:
        self.preview_rows_spin.setEnabled(not checked)
        self.preview_rows_spin.setVisible(not checked)
        if hasattr(self, "fmt_mat"):
            self._save_settings()

    def _preview_rows_limit(self) -> int | None:
        return None if self.preview_all_rows_check.isChecked() else int(self.preview_rows_spin.value())

    def _preview_limited_array(self, data):
        rows = self._preview_rows_limit()
        arr = np.asarray(data)
        if rows is None or arr.ndim == 0:
            return arr
        return arr[: min(rows, arr.shape[0])]

    def _plot_label_for_path(self, item, path: str) -> str:
        ds_item = self._dataset_item_of(item)
        if ds_item is None:
            return path
        return f"{ds_item.text(0)}. {path}"

    def _safe_export_stem(self, item, path: str) -> str:
        label = self._plot_label_for_path(item, path)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
        return safe.strip("._") or "export"

    def _show_child_dialog(self, dlg: QDialog) -> None:
        self._child_dialogs.append(dlg)
        dlg.destroyed.connect(lambda _obj=None, d=dlg: self._child_dialogs.remove(d) if d in self._child_dialogs else None)
        dlg.show()
        dlg.raise_()

    # ------------------------------------------------------------------
    # Parse (async)
    # ------------------------------------------------------------------

    def on_parse(self):
        self.tree.clear()
        self.parsed = {}
        self.nodeinfo.clear()
        self.fullpath_by_item.clear()
        self.datasetnode_info.clear()

        ipath = self.input_path_edit.text().strip()
        tmp = self._temp_dir()
        if not ipath or not os.path.exists(ipath):
            self.log("[error] set an input .msr file (or folder) first.")
            return
        msr = ipath if os.path.isfile(ipath) else next((str(p) for p in Path(ipath).glob("*.msr")), None)
        if not msr:
            self.log("no msr file found!")
            return

        self.log(f"[parse] using: {msr}")
        self._save_settings()
        self._start_parse_worker(msr, tmp)

    def _start_parse_worker(self, msr: str, tmp: str):
        self._parse_button.setEnabled(False)
        self._worker = _ParseWorker(msr, tmp)
        self._worker.progress.connect(self.log)
        self._worker.finished.connect(self._on_parse_done)
        self._worker.failed.connect(self._on_parse_failed)
        self._worker.start()

    def _on_parse_done(self, result: dict):
        from ...msr import state as MFSTATE

        self.parsed = result
        self._parse_button.setEnabled(True)
        self._build_tree_from_result(result)

        if getattr(MFSTATE, "mfx_map", {}):
            self.log(f"[global] mfx datasets: {len(MFSTATE.mfx_map)}")
            for key, arr in MFSTATE.mfx_map.items():
                self.log(f"          - {key}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            self.log("[global] mfx is empty")
        if getattr(MFSTATE, "mbm_map", {}):
            self.log(f"[global] mbm datasets: {len(MFSTATE.mbm_map)}")
            for key, arr in MFSTATE.mbm_map.items():
                self.log(f"          - {key}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            self.log("[global] mbm is empty")

    def _on_parse_failed(self, error: str):
        self.log(f"[ERROR] {error}")
        self._parse_button.setEnabled(True)

    def _build_tree_from_result(self, result: dict):
        from ...msr.descriptions import describe_path

        msr = result.get("msr", "unknown.msr")
        root_item = QTreeWidgetItem([os.path.basename(msr), "", "", ""])
        self._set_item_tooltip(root_item, "MSR file used to store measurement data generated by iMSPECTOR software.")
        self.tree.addTopLevelItem(root_item)
        self.nodeinfo[self._item_id(root_item)] = {"type": "msr_root", "description": root_item.toolTip(0)}

        if result.get("mode") == "modern":
            for ds_index, ds in enumerate(result.get("datasets", []), start=1):
                ds_item = QTreeWidgetItem([str(ds.get("display_name") or ds.get("did")), "dataset", "", str(ds.get("did", ""))])
                self._set_item_tooltip(ds_item, f"{self._ordinal(ds_index)} dataset from MINFLUX measurement.")
                root_item.addChild(ds_item)
                self.datasetnode_info[self._item_id(ds_item)] = {"did": ds.get("did"), "zroot": ds.get("zroot")}
                self.fullpath_by_item[self._item_id(ds_item)] = ""
                self.nodeinfo[self._item_id(ds_item)] = {"type": "dataset", "description": ds_item.toolTip(0), "zroot": ds.get("zroot"), "path": ""}

                node_map = {"": ds_item}
                for fld in ds.get("fields") or []:
                    path = fld["path"]
                    parts = path.split("/")
                    cur = ""
                    parent = ds_item
                    for i, part in enumerate(parts):
                        cur = part if i == 0 else f"{cur}/{part}"
                        if cur in node_map:
                            parent = node_map[cur]
                            continue
                        is_last = i == len(parts) - 1
                        kind = fld["kind"] if is_last else "group"
                        shape = "" if kind != "array" else str(fld.get("shape"))
                        dtype = "" if kind != "array" else fld.get("dtype", "")
                        item = QTreeWidgetItem([part, kind, shape, dtype])
                        self._set_item_tooltip(item, fld.get("description") if is_last else describe_path(cur, is_array=False))
                        parent.addChild(item)
                        node_map[cur] = item
                        self.fullpath_by_item[self._item_id(item)] = cur
                        if not is_last:
                            self.nodeinfo[self._item_id(item)] = {"type": "modern_group", "zroot": ds["zroot"], "path": cur, "description": item.toolTip(0)}
                        if is_last and kind == "array":
                            self.nodeinfo[self._item_id(item)] = {
                                "type": "modern_array",
                                "zroot": ds["zroot"],
                                "path": path,
                                "description": item.toolTip(0),
                                "dtype_raw": fld.get("dtype_raw") or fld.get("dtype"),
                                "show_raw_dtype": path not in {"grd/mbm/points", "grd/search_0/points", "mfx"},
                            }
                            for sub in fld.get("dtype_fields") or []:
                                self._add_dtype_field_nodes(item, path, [sub])
                        parent = item
        else:
            meta = result.get("metadata")
            series_root = QTreeWidgetItem(["Series", "", "", ""])
            root_item.addChild(series_root)
            for s in result.get("legacy_series_tree") or []:
                name = s.get("display_name", f"Series {int(s.get('index', 0)) + 1}")
                item = QTreeWidgetItem([f"- Series {int(s.get('index', 0)) + 1}: {name}", "", s.get("shape_str", ""), s.get("dtype", "")])
                series_root.addChild(item)
            meta_item = QTreeWidgetItem(["<metadata>", "", "", ""])
            self._set_item_tooltip(
                meta_item,
                "OME metadata of this legacy MSR file. Right-click > Show metadata "
                "opens it in a searchable JSON viewer.",
            )
            root_item.addChild(meta_item)
            if isinstance(meta, dict):
                self.nodeinfo[self._item_id(meta_item)] = {"type": "legacy_metadata", "data": meta}
                _populate_metadata_items(meta_item, _json_compact(meta))
            else:
                meta_item.addChild(QTreeWidgetItem(["(metadata unavailable)", "", "", ""]))

        root_item.setExpanded(True)
        for i in range(root_item.childCount()):
            ds_item = root_item.child(i)
            ds_item.setExpanded(True)
            for j in range(ds_item.childCount()):
                child = ds_item.child(j)
                child.setExpanded(child.text(0) == "grd")
                for k in range(child.childCount()):
                    child.child(k).setExpanded(False)
        for column in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(column)

    def _add_dtype_field_nodes(self, parent_item, path_prefix: str, fields: list):
        from ...msr.descriptions import describe_path

        for sub in fields or []:
            text = sub.get("name", "")
            kind = "struct" if sub.get("children") else sub.get("kind", "field")
            shape = sub.get("logical_shape") or str(sub.get("shape") or ())
            dtype = sub.get("dtype", "")
            fullpath = f"{path_prefix}/{text}" if path_prefix else text
            if sub.get("children") and fullpath == "mfx/itr":
                dtype = "Structured per-iteration MINFLUX table"
            item = QTreeWidgetItem([text, kind, shape, dtype])
            self._set_item_tooltip(item, sub.get("description") or describe_path(fullpath, is_array=False))
            parent_item.addChild(item)
            self.fullpath_by_item[self._item_id(item)] = fullpath
            self.nodeinfo[self._item_id(item)] = {"type": "modern_field", "description": item.toolTip(0)}
            if sub.get("children"):
                self._add_dtype_field_nodes(item, fullpath, sub["children"])

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------

    def _current_item(self):
        items = self.tree.selectedItems()
        return items[0] if items else None

    def _selected_items(self) -> list[QTreeWidgetItem]:
        return self.tree.selectedItems() or ([self._current_item()] if self._current_item() is not None else [])

    def _selected_array_items(self) -> list[QTreeWidgetItem]:
        return [item for item in self._selected_items() if item.text(1) in {"field", "array"}]

    def _dataset_item_of(self, item):
        while item is not None:
            if item.text(1) == "dataset":
                return item
            item = item.parent()
        return None

    def _zroot_for_dataset_item(self, item):
        info = self.datasetnode_info.get(self._item_id(item))
        return (info or {}).get("zroot"), (info or {}).get("did")

    def _load_array_for_path(self, zroot: str, path: str):
        import zarr
        arch = zarr.open(zroot, mode="r")
        if path in arch:
            return np.asarray(arch[path])

        parts = [part for part in path.split("/") if part]
        for split_at in range(len(parts) - 1, 0, -1):
            parent = "/".join(parts[:split_at])
            if parent not in arch:
                continue

            data = np.asarray(arch[parent])
            for field in parts[split_at:]:
                names = getattr(data.dtype, "names", None)
                if not names or field not in names:
                    raise KeyError(f"field not found: {parent}/{'/'.join(parts[split_at:])}")
                data = data[field]
            return np.asarray(data)
        raise KeyError(f"path not found: {path}")

    def _resolve_context_array(self, item=None):
        item = item or self._current_item()
        if item is None:
            return None, None, "Choose a field or array node."
        ds_item = self._dataset_item_of(item)
        if ds_item is None:
            return None, None, "Choose a field or array node."
        zroot, _ = self._zroot_for_dataset_item(ds_item)
        if not zroot:
            return None, None, "Dataset has no zarr root."
        path = self.fullpath_by_item.get(self._item_id(item))
        if not path:
            return None, None, "Choose a field or array node."
        try:
            arr = self._load_array_for_path(zroot, path)
        except Exception as exc:
            return None, None, f"Cannot load {path}:\n{exc}"
        return path, arr, None

    # ------------------------------------------------------------------
    # Plot window handling
    # ------------------------------------------------------------------

    def _on_plot_window_focused(self, window: PlotWindow):
        if window in self._plot_windows:
            self._active_plot_window = window

    def _on_plot_window_closed(self, window: PlotWindow):
        if window in self._plot_windows:
            self._plot_windows.remove(window)
        if self._active_plot_window is window:
            self._active_plot_window = self._plot_windows[-1] if self._plot_windows else None

    def _visible_plot_windows(self) -> list[PlotWindow]:
        self._plot_windows = [win for win in self._plot_windows if win is not None and win.isVisible()]
        return self._plot_windows

    def _target_plot_window(self) -> PlotWindow | None:
        windows = self._visible_plot_windows()
        if not windows:
            return None
        active = QApplication.activeWindow()
        if isinstance(active, PlotWindow) and active in windows:
            return active
        if self._active_plot_window in windows:
            return self._active_plot_window
        return windows[-1]

    def _default_plot_mode(self, path: str, data) -> str:
        name = path.rsplit("/", 1)[-1].lower()
        arr = np.asarray(data)
        if name in {"xyz", "loc", "lnc", "ext"} or (arr.ndim == 2 and arr.shape[1] in (2, 3)):
            return "scatter"
        if np.issubdtype(arr.dtype, np.bool_):
            return "histogram"
        if name in {"tim", "time", "timestamp", "tid"}:
            return "line"
        if name in {"efo", "cfr", "dcr"}:
            return "histogram"
        if np.issubdtype(arr.dtype, np.integer) and np.unique(arr[: min(arr.size, 10000)]).size <= 16:
            return "histogram"
        return "line" if arr.ndim == 1 else "scatter"

    def _show_plot_window(self, data, title: str, label: str, mode: str | None = None) -> PlotWindow:
        win = PlotWindow(data, title=title, parent=self, label=label, mode=mode)
        self._plot_windows.append(win)
        self._active_plot_window = win
        win.show()
        win.raise_()
        win.activateWindow()
        return win

    def _plot_data(self, data, title: str, label: str, mode: str | None = None) -> None:
        if self.plot_new_window_check.isChecked():
            self._show_plot_window(data, title, label, mode)
            return

        target = self._target_plot_window()
        if target is None:
            self._show_plot_window(data, title, label, mode)
            return

        can_overlay, reason = target.can_overlay(data)
        if not can_overlay:
            self.log(
                f"[warn] plot range for '{label}' does not fit the active plot window "
                f"({reason}); opening a new plot window."
            )
            self._show_plot_window(data, title, label, mode)
            return

        target.add_data(data, label, redraw=True)
        target.raise_()
        target.activateWindow()

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _show_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        if not item.isSelected():
            self.tree.clearSelection()
            item.setSelected(True)
        self.tree.setCurrentItem(item)
        menu = QMenu(self)
        selected_arrays = self._selected_array_items()
        can_export = bool(selected_arrays)
        preview = menu.addAction("Preview")
        plot = menu.addAction("Plot…")
        export_csv = menu.addAction("Export to CSV…")
        export_npy = menu.addAction("Export to NumPy (.npy)…")
        export_json = menu.addAction("Export to JSON…")
        show_meta = menu.addAction("Show metadata")
        plot.setEnabled(can_export)
        export_csv.setEnabled(can_export)
        export_npy.setEnabled(can_export)
        export_json.setEnabled(can_export)
        show_meta.setEnabled(any(self._can_show_metadata(selected) for selected in self._selected_items()))

        action = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if action == preview:
            self._ctx_preview()
        elif action == plot:
            self._ctx_plot()
        elif action == export_csv:
            self._ctx_export("csv")
        elif action == export_npy:
            self._ctx_export("npy")
        elif action == export_json:
            self._ctx_export("json")
        elif action == show_meta:
            self._ctx_show_metadata()

    def _node_metadata_target(self, item):
        info = self.nodeinfo.get(self._item_id(item), {})
        if info.get("type") in {"modern_group", "modern_array", "dataset"}:
            return info.get("zroot"), info.get("path") or ""
        ds_item = self._dataset_item_of(item)
        if ds_item is not None:
            zroot, _ = self._zroot_for_dataset_item(ds_item)
            return zroot, self.fullpath_by_item.get(self._item_id(item), "") or ""
        return None, None

    def _legacy_metadata_of(self, item):
        """The parsed legacy OME metadata dict if *item* is (inside) the
        ``<metadata>`` node of a legacy file, else None."""
        while item is not None:
            info = self.nodeinfo.get(self._item_id(item), {})
            if info.get("type") == "legacy_metadata":
                return info.get("data")
            item = item.parent()
        return None

    def _can_show_metadata(self, item) -> bool:
        from ...msr.io import read_zarr_attrs
        if self._legacy_metadata_of(item) is not None:
            return True
        zroot, path = self._node_metadata_target(item)
        if not zroot:
            return False
        try:
            return bool(read_zarr_attrs(zroot, path or ""))
        except Exception:
            return False

    def _ctx_show_metadata(self):
        from ...msr.io import read_zarr_attrs
        shown = 0
        legacy_shown = False
        for item in self._selected_items():
            legacy_meta = self._legacy_metadata_of(item)
            if legacy_meta is not None:
                if not legacy_shown:
                    name = Path(self.parsed.get("msr", "")).name or "legacy file"
                    self._show_child_dialog(JsonViewDialog(f"Metadata: {name}", legacy_meta, self))
                    legacy_shown = True
                    shown += 1
                continue
            zroot, path = self._node_metadata_target(item)
            if not zroot:
                continue
            try:
                attrs = read_zarr_attrs(zroot, path or "")
            except Exception as exc:
                self.log(f"[metadata] failed for {item.text(0)}: {exc}")
                continue
            if not attrs:
                continue
            self._show_child_dialog(JsonViewDialog(f"Metadata: {path or '<dataset root>'}", attrs, self))
            shown += 1
        if shown == 0:
            QMessageBox.information(self, "Metadata", "No Zarr metadata is available for the selected node(s).")

    def _ctx_preview(self):
        shown = 0
        for item in self._selected_array_items():
            path, arr, err = self._resolve_context_array(item)
            if err:
                self.log(f"[preview] {err}")
                continue
            label = self._plot_label_for_path(item, path)
            self._show_child_dialog(ArrayPreviewDialog(f"Preview: {label}", arr, self._preview_rows_limit(), self))
            shown += 1
        if shown == 0:
            QMessageBox.information(self, "Preview", "Choose one or more field or array nodes.")

    def _ctx_plot(self):
        entries = []
        for item in self._selected_array_items():
            path, arr, err = self._resolve_context_array(item)
            if err:
                self.log(f"[plot] {err}")
                continue
            data = np.asarray(self._preview_limited_array(arr))
            if data.size == 0:
                self.log(f"[plot] skip empty selection: {path}")
                continue
            if not _is_numeric_dtype(data.dtype):
                self.log(f"[plot] skip non-numeric selection: {path}")
                continue
            entries.append((item, path, data, self._plot_label_for_path(item, path)))
        if not entries:
            QMessageBox.information(self, "Plot", "Choose one or more numeric field or array nodes.")
            return
        if self.plot_new_window_check.isChecked() or len(entries) == 1:
            for _item, path, data, label in entries:
                self._plot_data(data, f"Plot: {label}", label, self._default_plot_mode(path, data))
            return
        groups: dict[str, list[tuple[Any, str, np.ndarray, str]]] = {}
        for entry in entries:
            _item, path, _data, _label = entry
            groups.setdefault(path, []).append(entry)
        for path, group in groups.items():
            win = None
            for _item, _path, data, label in group:
                if win is None:
                    win = self._show_plot_window(data, f"Plot: {path}", label, self._default_plot_mode(path, data))
                else:
                    can_overlay, reason = win.can_overlay(data)
                    if can_overlay:
                        win.add_data(data, label, redraw=True)
                    else:
                        self.log(f"[warn] plot range for '{label}' does not fit grouped plot ({reason}); opening a new plot window.")
                        self._show_plot_window(data, f"Plot: {label}", label, self._default_plot_mode(path, data))

    def _ctx_export(self, fmt: str):
        items = self._selected_array_items()
        if not items:
            QMessageBox.information(self, "Export", "Choose one or more field or array nodes.")
            return
        if len(items) == 1:
            item = items[0]
            path = self.fullpath_by_item.get(self._item_id(item), item.text(0))
            filters = {"csv": "CSV (*.csv);;All files (*.*)", "npy": "NumPy (*.npy);;All files (*.*)", "json": "JSON (*.json);;All files (*.*)"}
            out, _ = QFileDialog.getSaveFileName(self, f"Export to {fmt.upper()}", f"{self._safe_export_stem(item, path)}.{fmt}", filters[fmt])
            if not out:
                return
            targets = [(item, out)]
        else:
            folder = QFileDialog.getExistingDirectory(self, f"Choose folder for {len(items)} {fmt.upper()} exports", self.out_dir_edit.text().strip() or str(Path.home()))
            if not folder:
                return
            targets = []
            for item in items:
                path = self.fullpath_by_item.get(self._item_id(item), item.text(0))
                targets.append((item, str(Path(folder) / f"{self._safe_export_stem(item, path)}.{fmt}")))
        for item, out in targets:
            path, data, err = self._resolve_context_array(item)
            if err:
                self.log(f"[export] {err}")
                continue
            try:
                if fmt == "csv":
                    np.savetxt(out, self._to_2d(data), delimiter=",")
                elif fmt == "npy":
                    np.save(out, data, allow_pickle=False)
                else:
                    self._save_array_json(out, data)
                self.log(f"[export] wrote {out}")
            except Exception as exc:
                self.log(f"[export] failed {path}: {exc}")

    def _save_array_json(self, fname: str, arr):
        a = np.asarray(arr)
        names = getattr(getattr(a, "dtype", None), "names", None)
        if names:
            out = []
            for row in a:
                d = {}
                for k in names:
                    v = row[k]
                    if isinstance(v, np.ndarray):
                        d[k] = v.tolist()
                    elif isinstance(v, np.generic):
                        d[k] = v.item()
                    else:
                        d[k] = v
                out.append(d)
        else:
            out = a.tolist()
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)

    def _to_2d(self, a):
        a = np.asarray(a)
        if a.ndim == 1:
            return a.reshape(-1, 1)
        if a.ndim > 2:
            return a.reshape(a.shape[0], -1)
        return a

    # ------------------------------------------------------------------
    # Fields / export
    # ------------------------------------------------------------------

    def _gather_datasets_for_dialog(self) -> list[dict]:
        from ...msr import state as MFSTATE
        datasets = []
        if self.parsed.get("mode") == "modern":
            for ds in self.parsed.get("datasets") or []:
                key = ds.get("display_name") or ds.get("did") or "dataset"
                mfx_arr = MFSTATE.mfx_map.get(key)
                mbm_arr = MFSTATE.mbm_map.get(key)
                datasets.append({
                    "key": key,
                    "name": key,
                    "mfx_fields": list(getattr(getattr(mfx_arr, "dtype", None), "names", []) or []),
                    "mbm_fields": list(getattr(getattr(mbm_arr, "dtype", None), "names", []) or []),
                })
        else:
            key = "legacy"
            mfx_arr = MFSTATE.mfx_map.get(key) or MFSTATE.mfx
            mbm_arr = MFSTATE.mbm_map.get(key) or MFSTATE.mbm
            datasets.append({
                "key": key,
                "name": key,
                "mfx_fields": list(getattr(getattr(mfx_arr, "dtype", None), "names", []) or []),
                "mbm_fields": list(getattr(getattr(mbm_arr, "dtype", None), "names", []) or []),
            })
        return datasets

    def on_fields_dialog(self):
        ds_list = self._gather_datasets_for_dialog()
        if not ds_list:
            QMessageBox.information(self, "No datasets", "No datasets loaded to select fields from.")
            return
        dlg = FieldDialog(ds_list, self.field_selection or None, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.field_selection = dlg.result_payload()
            self._save_settings()
            self.log("[select] updated datasets/fields selection.")

    def _derive_global_field_filters(self):
        sel = self.field_selection or {}
        if not sel:
            return None, None
        active = [v for v in sel.values() if v.get("checked", True)]
        if not active:
            return set(), set()
        any_all_mfx = any(v.get("mfx") is None for v in active)
        mfx_sel = None if any_all_mfx else set().union(*(set(v.get("mfx") or []) for v in active))
        any_all_mbm = any(v.get("mbm") is None for v in active)
        mbm_sel = None if any_all_mbm else set().union(*(set(v.get("mbm") or []) for v in active))
        return mfx_sel, mbm_sel

    def _subset_struct_fields(self, arr, names_sel):
        if arr is None:
            return None
        names = getattr(getattr(arr, "dtype", None), "names", None)
        if names and names_sel is not None:
            keep = [n for n in names if n in names_sel]
            if not keep:
                return None
            try:
                return arr[keep]
            except Exception:
                pass
        return arr

    def _export_current_parsed(self, out_dir: str, formats: list[str], mfx_sel_global=None, mbm_sel_global=None):
        from ...msr import state as MFSTATE
        os.makedirs(out_dir, exist_ok=True)

        # dataset display-name -> in-memory zarr store (for the .zarr export)
        store_by_name = {
            (ds.get("display_name") or ds.get("did") or "dataset"): ds.get("zroot")
            for ds in (self.parsed.get("datasets") or [])
        }

        def subset_struct(arr, names_sel):
            return self._subset_struct_fields(arr, names_sel)

        def save_zarr(key: str):
            from ...msr.mfxdta import unpack_zarr_store_to_dir
            store = store_by_name.get(key)
            if not store:
                self.log(f"[zarr] skipping '{key}' (no embedded store).")
                return None
            path = Path(out_dir) / f"{key}.zarr"
            unpack_zarr_store_to_dir(store, path)
            self.log(f"[zarr] wrote {path}")

        def save_mat(fn_base: str, arr):
            if arr is None:
                return None
            try:
                from scipy.io import savemat
            except Exception as exc:
                self.log(f"[warn] SciPy not available; skip .mat ({exc})")
                return None
            def as_col(a):
                a = np.asarray(a)
                if a.ndim == 1:
                    return a.reshape(-1, 1)
                if a.ndim == 2:
                    return a
                return a.reshape(a.shape[0], -1)
            payload = {}
            names = getattr(getattr(arr, "dtype", None), "names", None)
            if names:
                for k in names:
                    col = arr[k]
                    subnames = getattr(col.dtype, "names", None)
                    if subnames:
                        for sk in subnames:
                            payload[f"{k}_{sk}"] = as_col(col[sk])
                    else:
                        payload[k] = as_col(col)
            else:
                payload["data"] = as_col(arr)
            path = Path(out_dir) / f"{fn_base}.mat"
            savemat(str(path), payload, do_compression=False, oned_as="column", long_field_names=True)
            self.log(f"[mat] wrote {path}")

        def save_npy(fn_base: str, arr):
            if arr is None:
                return None
            path = Path(out_dir) / f"{fn_base}.npy"
            np.save(str(path), arr, allow_pickle=False)
            self.log(f"[npy] wrote {path}")

        def save_json(fn_base: str, arr):
            if arr is None:
                return None
            names = getattr(getattr(arr, "dtype", None), "names", None)
            if not names:
                self.log("[json] skipping (no named fields).")
                return None
            path = Path(out_dir) / f"{fn_base}.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(arr.tolist(), fh, ensure_ascii=False)
            self.log(f"[json] wrote {path}")

        def save_csv(fn_base: str, arr):
            if arr is None:
                return None
            names = getattr(getattr(arr, "dtype", None), "names", None)
            if not names:
                self.log("[csv] skipping (no named fields).")
                return None
            path = Path(out_dir) / f"{fn_base}.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(list(names))
                for row in arr:
                    writer.writerow([np.asarray(row[k]).tolist() if hasattr(row[k], "shape") else row[k] for k in names])
            self.log(f"[csv] wrote {path}")

        if self.parsed.get("mode") == "modern":
            datasets = [(ds.get("display_name") or ds.get("did") or "dataset") for ds in self.parsed.get("datasets") or []]
        else:
            datasets = list(getattr(MFSTATE, "mfx_map", {}).keys() | getattr(MFSTATE, "mbm_map", {}).keys()) or ["legacy"]

        for key in datasets:
            mfx_arr = MFSTATE.mfx_map.get(key)
            mbm_arr = MFSTATE.mbm_map.get(key)
            if mfx_sel_global is not None or mbm_sel_global is not None:
                mfx_sel = mfx_sel_global
                mbm_sel = mbm_sel_global
            else:
                sel = self.field_selection.get(key, {"checked": True, "mfx": None, "mbm": None})
                if not sel.get("checked", True):
                    self.log(f"[export] skip dataset '{key}' (unchecked).")
                    continue
                mfx_sel = sel.get("mfx")
                mbm_sel = sel.get("mbm")
            mfx_sub = subset_struct(mfx_arr, mfx_sel)
            mbm_sub = subset_struct(mbm_arr, mbm_sel)
            kept = (list(getattr(getattr(mfx_sub, "dtype", None), "names", []) or [])
                    if mfx_sel is not None else "all")
            self.log(f"[export] '{key}': mfx fields = "
                     f"{', '.join(kept) if isinstance(kept, list) else kept}"
                     + ("" if "zarr" not in formats else "  (.zarr writes the full store)"))
            if "mat" in formats:
                save_mat(f"{key}_mfx", mfx_sub)
                save_mat(f"{key}_mbm", mbm_sub)
            if "npy" in formats:
                save_npy(f"{key}_mfx", mfx_sub)
                save_npy(f"{key}_mbm", mbm_sub)
            if "json" in formats:
                save_json(f"{key}_mfx", mfx_sub)
                save_json(f"{key}_mbm", mbm_sub)
            if "csv" in formats:
                save_csv(f"{key}_mfx", mfx_sub)
                save_csv(f"{key}_mbm", mbm_sub)
            if "zarr" in formats:
                save_zarr(key)

    def _find_msr_files(self, path_str: str):
        p = Path(path_str)
        folder = p.parent if p.is_file() else p
        return sorted(list(folder.glob("*.msr")) + list(folder.glob("*.MSR")))

    def _viewer_import_plan(self) -> dict[str, set[str] | None]:
        """Return dataset -> selected mfx fields, or None for all fields."""
        datasets_info = self.parsed.get("datasets", [])
        all_keys = [(ds.get("display_name") or ds.get("did") or "dataset") for ds in datasets_info]
        selected = self._selected_items()
        plan: dict[str, set[str] | None] = {}
        used_tree_selection = False
        for item in selected:
            ds_item = self._dataset_item_of(item)
            if ds_item is None:
                continue
            key = ds_item.text(0)
            path = self.fullpath_by_item.get(self._item_id(item), "")
            if item.text(1) == "dataset":
                plan[key] = None
                used_tree_selection = True
            elif path == "mfx":
                plan[key] = None
                used_tree_selection = True
            elif path.startswith("mfx/"):
                field = path.split("/", 1)[1].split("/", 1)[0]
                if key in plan and plan[key] is None:
                    pass
                else:
                    plan.setdefault(key, set()).add(field)
                used_tree_selection = True
        if used_tree_selection:
            return plan
        if self.field_selection:
            for key in all_keys:
                sel = self.field_selection.get(key, {"checked": True, "mfx": None})
                if not sel.get("checked", True):
                    continue
                fields = sel.get("mfx")
                plan[key] = None if fields is None else set(fields)
            return plan
        return {key: None for key in all_keys}

    def _viewer_selected_dataset_names(self, import_plan: dict[str, set[str] | None]) -> list[str]:
        return [
            ds.get("display_name") or ds.get("did") or "dataset"
            for ds in self.parsed.get("datasets", [])
            if (ds.get("display_name") or ds.get("did") or "dataset") in import_plan
        ]

    def _viewer_zroot_for_dataset_name(self, name: str) -> str | None:
        for ds in self.parsed.get("datasets", []) or []:
            key = ds.get("display_name") or ds.get("did") or "dataset"
            if key == name:
                return ds.get("zroot")
        return None

    def _ask_viewer_channel_alignment(self, import_plan: dict[str, set[str] | None]):
        """Ask for reference dataset + alignment mode.

        Returns ``(mode, ref_name)``, ``None`` when no alignment question
        applies (legacy file or fewer than two datasets), or
        ``_ALIGN_CANCELLED`` when the user cancelled — which must abort the
        whole open, not just skip the transforms.
        """
        if self.parsed.get("mode") != "modern":
            return None
        selected = self._viewer_selected_dataset_names(import_plan)
        if len(selected) < 2:
            return None

        def mbm_check(ref_name: str) -> tuple[bool, str]:
            ordered = [ref_name] + [n for n in selected if n != ref_name]
            return self._viewer_mbm_alignment_available(ordered)

        dlg = ViewerAlignmentDialog(selected, mbm_check, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.log("[viewer align] cancelled; open aborted")
            return _ALIGN_CANCELLED
        mode = dlg.alignment_mode()
        ref_name = dlg.reference_dataset()
        self.log(f"[viewer align] selected mode: {mode}; reference: {ref_name}")
        return mode, ref_name

    def _viewer_mbm_alignment_available(self, selected: list[str]) -> tuple[bool, str]:
        from ...msr import state as MFSTATE
        from ...msr.alignment import MIN_BEADS_FOR_ALIGNMENT, extract_bead_summaries

        if len(selected) < 2:
            return False, "fewer than two selected datasets"
        ref_name = selected[0]
        ref_mbm = self._aligned_mbm_points(ref_name)
        if ref_mbm is None:
            return False, f"no bead points for reference dataset {ref_name}"
        # 2 matched beads suffice (translational XYZ); the transform mode is
        # downgraded to what the matched-bead count supports — see
        # alignment.effective_transform_type.
        min_beads = MIN_BEADS_FOR_ALIGNMENT
        try:
            ref_ids = set(extract_bead_summaries(ref_mbm))
        except Exception as exc:
            return False, f"reference bead points are not parseable ({exc})"
        ref_used = self._viewer_used_bead_gri_for_dataset(ref_name)
        if ref_used is not None:
            ref_ids &= ref_used
            if not ref_ids:
                return False, f"reference dataset {ref_name} has no metadata-marked used beads"
        if len(ref_ids) < min_beads:
            return False, f"reference dataset has {len(ref_ids)} usable bead(s); need at least {min_beads}"
        for moving_name in selected[1:]:
            moving_mbm = self._aligned_mbm_points(moving_name)
            if moving_mbm is None:
                return False, f"no bead points for {moving_name}"
            try:
                moving_ids = set(extract_bead_summaries(moving_mbm))
            except Exception as exc:
                return False, f"bead points for {moving_name} are not parseable ({exc})"
            moving_used = self._viewer_used_bead_gri_for_dataset(moving_name)
            if moving_used is not None:
                moving_ids &= moving_used
                if not moving_ids:
                    return False, f"{moving_name} has no metadata-marked used beads"
            matched = len(ref_ids & moving_ids)
            if matched < min_beads:
                return False, f"{moving_name} has {matched} usable bead(s) matching the reference; need at least {min_beads}"
        return True, ""

    def _viewer_used_bead_gri_for_dataset(self, name: str) -> set[int] | None:
        zroot = self._viewer_zroot_for_dataset_name(name)
        if not zroot:
            return None
        mbm_attrs = self._read_zarr_attrs_safe(zroot, "mbm") or self._read_zarr_attrs_safe(zroot, "grd/mbm")
        points_attrs = self._read_zarr_attrs_safe(zroot, "grd/mbm/points")

        used_names = self._extract_used_bead_names(mbm_attrs)
        name_to_gri, used_gri_from_points = self._extract_point_metadata_beads(points_attrs)
        if used_names and name_to_gri:
            used_gri = {name_to_gri[name] for name in used_names if name in name_to_gri}
            missing = sorted(name for name in used_names if name not in name_to_gri)
            if missing:
                preview = ", ".join(missing[:5])
                suffix = "..." if len(missing) > 5 else ""
                self.log(f"[viewer align] {name}: {len(missing)} used bead name(s) not mapped to gri: {preview}{suffix}")
            if used_gri:
                self.log(f"[viewer align] {name}: using {len(used_gri)} metadata-marked used bead(s)")
                return used_gri
        if used_gri_from_points:
            self.log(f"[viewer align] {name}: using {len(used_gri_from_points)} bead(s) marked used in points metadata")
            return used_gri_from_points
        return None

    def _read_zarr_attrs_safe(self, zroot: str, path: str) -> dict:
        try:
            from ...msr.io import read_zarr_attrs

            attrs = read_zarr_attrs(zroot, path)
            return attrs if isinstance(attrs, dict) else {}
        except Exception:
            return {}

    def _extract_used_bead_names(self, attrs: dict) -> set[str]:
        if not isinstance(attrs, dict):
            return set()
        used = attrs.get("used")
        if not isinstance(used, (list, tuple, set)):
            return set()
        return {str(name).strip() for name in used if str(name).strip()}

    def _extract_point_metadata_beads(self, attrs: dict) -> tuple[dict[str, int], set[int]]:
        name_to_gri: dict[str, int] = {}
        used_gri: set[int] = set()

        def visit(node) -> None:
            if isinstance(node, dict):
                if "name" in node and "gri" in node:
                    try:
                        gri = int(node.get("gri"))
                    except Exception:
                        gri = None
                    name = str(node.get("name", "")).strip()
                    if gri is not None and name:
                        name_to_gri[name] = gri
                        if bool(node.get("used", False)):
                            used_gri.add(gri)
                for value in node.values():
                    visit(value)
            elif isinstance(node, (list, tuple)):
                for value in node:
                    visit(value)

        visit(attrs)
        return name_to_gri, used_gri

    def _filter_mbm_points_to_used_beads(self, name: str, points) -> np.ndarray:
        arr = np.asarray(points)
        used_gri = self._viewer_used_bead_gri_for_dataset(name)
        if used_gri is None:
            return arr
        names = set(getattr(getattr(arr, "dtype", None), "names", ()) or ())
        if "gri" not in names:
            return arr
        mask = np.isin(np.asarray(arr["gri"], dtype=np.uint32), np.asarray(sorted(used_gri), dtype=np.uint32))
        filtered = arr[mask]
        self.log(f"[viewer align] {name}: filtered mbm points to {len(filtered)} row(s) from used bead metadata")
        return filtered

    def _viewer_dataset_looks_3d(self, name: str) -> bool:
        from ...msr import state as MFSTATE

        mfx = MFSTATE.mfx_map.get(name)
        try:
            from ...core.loader import _normalize_mfx_attrs

            data_prefs = (self._state.prefs if self._state is not None else {}).get("data", {}) or {}
            attrs, num_raw, _num_itr, _version, _detail = _normalize_mfx_attrs(
                mfx,
                load_all_itr=data_prefs.get("iter_load", "last") == "all",
            )
            loc_z = np.asarray(attrs.get("loc_z", np.zeros(num_raw)), dtype=np.float64)
            finite = np.isfinite(loc_z)
            if np.any(finite) and np.nanmax(np.abs(loc_z[finite])) > 1e-12:
                return True
        except Exception:
            pass
        mbm = MFSTATE.mbm_map.get(name)
        try:
            names = set(getattr(getattr(mbm, "dtype", None), "names", ()) or ())
            if "xyz" in names:
                xyz = np.asarray(mbm["xyz"], dtype=np.float64)
                if xyz.ndim >= 2 and xyz.shape[-1] >= 3:
                    z = xyz[..., 2]
                    finite = np.isfinite(z)
                    return bool(np.any(finite) and np.nanmax(np.abs(z[finite])) > 1e-12)
        except Exception:
            pass
        return False

    def _viewer_channel_alignment_transforms(
        self,
        import_plan: dict[str, set[str] | None],
        mode: str,
        ref_name: str | None = None,
    ) -> dict[str, dict]:
        from ...msr import state as MFSTATE
        from ...msr.alignment import align_channels_from_arrays

        selected = self._viewer_selected_dataset_names(import_plan)
        if len(selected) < 2:
            return {}
        # The reference leads the list; both the mbm and fallback paths treat
        # selected[0] as the reference channel.
        if ref_name and ref_name in selected:
            selected = [ref_name] + [name for name in selected if name != ref_name]
        if mode in {"stage origin", "data centroid"}:
            return self._viewer_fallback_alignment_transforms(selected, mode)
        if mode != "mbm info":
            return {}
        ref_name = selected[0]
        ref_mbm = self._aligned_mbm_points(ref_name)
        if ref_mbm is None:
            raise ValueError(f"Could not find bead points (`grd/mbm/points`) for reference channel {ref_name}.")
        ref_mbm_for_alignment = self._filter_mbm_points_to_used_beads(ref_name, ref_mbm)

        transforms: dict[str, dict] = {}
        transforms[ref_name] = self._viewer_transform_record(
            ref_name,
            ref_name,
            np.eye(4, dtype=np.float64),
            method="reference identity",
        )
        gri_to_rid = self._bead_gri_to_rid(selected)
        excluded = sorted(int(g) for g in (self._bead_unchecked_gris or set()))
        self.log(
            f"[viewer align] user-excluded {len(excluded)} bead(s): "
            f"{self._format_bead_ids(excluded, gri_to_rid)}")
        for moving_name in selected[1:]:
            moving_mbm = self._aligned_mbm_points(moving_name)
            if moving_mbm is None:
                raise ValueError(f"Could not find bead points (`grd/mbm/points`) for {moving_name}.")
            moving_mbm_for_alignment = self._filter_mbm_points_to_used_beads(moving_name, moving_mbm)
            result = align_channels_from_arrays(
                ref_name, ref_mbm_for_alignment, moving_name, moving_mbm_for_alignment,
                transform_type=self._mbm_transform_type())
            transforms[moving_name] = {
                "reference_channel": ref_name,
                "moving_channel": moving_name,
                "matrix_3x3": result.matrix_3x3.tolist(),
                "matrix_4x4": result.matrix_4x4.tolist(),
                "translation_nm": result.translation.tolist(),
                "z_translation_nm": float(result.z_translation),
                "rotation_2x2": result.rotation.tolist(),
                "matched_bead_count": int(result.bead_ids.size),
                "rmse_xy_nm": float(result.rmse),
                "method": "mbm xy rigid plus z translation",
            }
            used = [int(b) for b in result.bead_ids]
            self.log(
                f"[viewer align] {moving_name} -> {ref_name}; used {len(used)} bead(s): "
                f"{self._format_bead_ids(used, gri_to_rid)}; "
                f"translation_nm=({result.translation[0]:.3f}, {result.translation[1]:.3f}); "
                f"rmse_xy_nm={result.rmse:.3f}"
            )
        return transforms

    def _ask_viewer_alignment_fallback(self, reason: str) -> str | None:
        box = QMessageBox(self)
        box.setWindowTitle("Open in MINFLUX viewer")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText("Bead-based channel alignment is not available for this MSR file.")
        box.setInformativeText(
            f"{reason}\n\n"
            "Choose a fallback alignment method for the viewer overlay:"
        )
        stage_button = box.addButton("Stage origin", QMessageBox.ButtonRole.AcceptRole)
        centroid_button = box.addButton("Data centroid", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is stage_button:
            self.log("[viewer align] using fallback: stage origin")
            return "stage origin"
        if clicked is centroid_button:
            self.log("[viewer align] using fallback: data centroid")
            return "data centroid"
        self.log("[viewer align] fallback alignment cancelled; opening without transforms")
        return None

    def _viewer_fallback_alignment_transforms(self, selected: list[str], mode: str) -> dict[str, dict]:
        from ...msr import state as MFSTATE

        if not selected:
            return {}
        ref_name = selected[0]
        ref_centroid = self._mfx_centroid_nm(MFSTATE.mfx_map.get(ref_name))
        transforms: dict[str, dict] = {}
        for name in selected:
            matrix = np.eye(4, dtype=np.float64)
            if name != ref_name and mode == "data centroid":
                moving_centroid = self._mfx_centroid_nm(MFSTATE.mfx_map.get(name))
                matrix[:3, 3] = ref_centroid - moving_centroid
            method = "stage origin" if mode == "stage origin" else "median localization centroid"
            transforms[name] = self._viewer_transform_record(ref_name, name, matrix, method=method)
            t = matrix[:3, 3]
            self.log(
                f"[viewer align] fallback {name} -> {ref_name}; method={method}; "
                f"translation_nm=({t[0]:.3f}, {t[1]:.3f})"
            )
        return transforms

    def _viewer_transform_record(self, ref_name: str, moving_name: str, matrix_4x4: np.ndarray, *, method: str) -> dict:
        matrix = np.asarray(matrix_4x4, dtype=np.float64)
        matrix_3x3 = np.eye(3, dtype=np.float64)
        matrix_3x3[:2, :2] = matrix[:2, :2]
        matrix_3x3[:2, 2] = matrix[:2, 3]
        return {
            "reference_channel": ref_name,
            "moving_channel": moving_name,
            "matrix_3x3": matrix_3x3.tolist(),
            "matrix_4x4": matrix.tolist(),
            "translation_nm": matrix[:2, 3].astype(float).tolist(),
            "z_translation_nm": float(matrix[2, 3]),
            "rotation_2x2": matrix[:2, :2].astype(float).tolist(),
            "matched_bead_count": 0,
            "rmse_xy_nm": 0.0,
            "method": method,
        }

    def _mfx_centroid_nm(self, mfx) -> np.ndarray:
        if mfx is None:
            return np.zeros(3, dtype=np.float64)
        try:
            from ...core.loader import _normalize_mfx_attrs

            data_prefs = (self._state.prefs if self._state is not None else {}).get("data", {}) or {}
            attrs, num_raw, _num_itr, _version, _detail = _normalize_mfx_attrs(
                mfx,
                load_all_itr=data_prefs.get("iter_load", "last") == "all",
            )
            loc_x = np.asarray(attrs.get("loc_x", np.full(num_raw, np.nan)), dtype=np.float64).ravel()
            loc_y = np.asarray(attrs.get("loc_y", np.full(num_raw, np.nan)), dtype=np.float64).ravel()
            loc_z = np.asarray(attrs.get("loc_z", np.zeros_like(loc_x)), dtype=np.float64).ravel()
            mask = np.isfinite(loc_x) & np.isfinite(loc_y) & np.isfinite(loc_z)
            if "vld" in attrs:
                vld = np.asarray(attrs["vld"], dtype=bool).ravel()
                if vld.size == mask.size:
                    mask &= vld
            if not np.any(mask):
                return np.zeros(3, dtype=np.float64)
            return np.nanmedian(np.column_stack([loc_x[mask], loc_y[mask], loc_z[mask]]) * 1e9, axis=0)
        except Exception:
            return np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def _ensure_single_file_parsed(self) -> bool:
        from ...msr import pick_one_msr, parse_msr_general
        ipath = self.input_path_edit.text().strip()
        tmp = self._temp_dir()
        if self.mode_folder.isChecked():
            QMessageBox.information(self, "Align channel", "Channel alignment is currently available only in single-file mode.")
            return False
        if self.parsed:
            return True
        msr = pick_one_msr(ipath)
        if not msr:
            self.log("[align] no msr file found.")
            QMessageBox.critical(self, "Align channel", "Choose a valid .msr file first.")
            return False
        self.log(f"[align] parsing before alignment: {msr}")
        self.parsed = parse_msr_general(msr, tmp, log=self.log)
        return True

    def _channel_alignment_defaults(self):
        from ...msr import pick_one_msr
        out_dir = Path(self.out_dir_edit.text().strip() or self._temp_dir())
        msr = pick_one_msr(self.input_path_edit.text().strip())
        stem = Path(msr).stem if msr else "channel_alignment"
        return {
            "beads": str(out_dir / f"{stem}_aligned_beads.csv"),
            "transform": str(out_dir / f"{stem}_rigid_transform.csv"),
            "localizations": str(out_dir / f"{stem}_aligned_localizations.csv"),
        }

    def _write_alignment_beads_csv(self, out_path: str, results):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "bead_id", "reference_channel_name", "reference_count_used", "reference_x_nm", "reference_y_nm", "reference_z_nm",
                "moving_channel_name", "moving_count_used", "moving_x_nm", "moving_y_nm", "moving_z_nm",
                "moving_aligned_x_nm", "moving_aligned_y_nm", "moving_aligned_z_nm",
            ])
            for result in results:
                for idx, bead_id in enumerate(result.bead_ids):
                    writer.writerow([
                        int(bead_id), result.channel1_name, int(result.channel1_counts[idx]), *result.channel1_xyz[idx].tolist(),
                        result.channel2_name, int(result.channel2_counts[idx]), *result.channel2_xyz[idx].tolist(),
                        *result.channel2_aligned_xyz[idx].tolist(),
                    ])

    def _write_alignment_transform_csv(self, out_path: str, results):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "reference_channel_name", "moving_channel_name", "matched_bead_count",
                "translation_x_nm", "translation_y_nm", "rmse_xy_nm",
                "m00", "m01", "m02_nm", "m10", "m11", "m12_nm", "m20", "m21", "m22",
            ])
            for result in results:
                writer.writerow([
                    result.channel1_name, result.channel2_name, int(result.bead_ids.size), float(result.translation[0]), float(result.translation[1]), float(result.rmse),
                    float(result.matrix_3x3[0, 0]), float(result.matrix_3x3[0, 1]), float(result.matrix_3x3[0, 2]),
                    float(result.matrix_3x3[1, 0]), float(result.matrix_3x3[1, 1]), float(result.matrix_3x3[1, 2]),
                    float(result.matrix_3x3[2, 0]), float(result.matrix_3x3[2, 1]), float(result.matrix_3x3[2, 2]),
                ])

    def _build_aligned_localizations_table(self, channel_name, channel_mfx, matrix_4x4):
        from ...msr.alignment import apply_homogeneous_transform
        names = set(getattr(getattr(channel_mfx, "dtype", None), "names", ()) or ())
        if "loc" not in names:
            raise ValueError(f"{channel_name} mfx array has no 'loc' field")
        loc = np.asarray(channel_mfx["loc"], dtype=np.float64) * 1e9
        if loc.ndim != 2 or loc.shape[1] < 2:
            raise ValueError("mfx.loc must have shape (N, 2+) for alignment export")
        # Use the full 4x4 so Z is transformed too (3D datasets / 3D alignment).
        aligned = apply_homogeneous_transform(loc, matrix_4x4)
        rows = []
        for idx in range(loc.shape[0]):
            row = {"channel_name": channel_name, "index": idx, "loc_x_nm": float(loc[idx, 0]), "loc_y_nm": float(loc[idx, 1]), "aligned_x_nm": float(aligned[idx, 0]), "aligned_y_nm": float(aligned[idx, 1])}
            if loc.shape[1] >= 3:
                row["loc_z_nm"] = float(loc[idx, 2])
                row["aligned_z_nm"] = float(aligned[idx, 2])
            if "vld" in names:
                row["vld"] = bool(channel_mfx["vld"][idx])
            if "itr" in names:
                row["itr"] = int(channel_mfx["itr"][idx])
            rows.append(row)
        return rows

    def _write_dict_rows_csv(self, out_path: str, rows: list[dict]):
        if not rows:
            raise ValueError("no localization rows are available to export")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _aligned_mbm_points(self, name):
        """Bead points for *name* with beads excluded via "Show Beads Drift"
        removed. An empty selection (the default) returns all beads unchanged, so
        the alignment behaves exactly as before unless the user made a choice."""
        from ...msr import state as MFSTATE

        points = MFSTATE.mbm_map.get(name)
        if not self._bead_unchecked_gris or points is None:
            return points
        try:
            import numpy as np
            if "gri" in (getattr(points, "dtype", None).names or ()):
                gri = np.asarray(points["gri"]).ravel()
                return points[~np.isin(gri, list(self._bead_unchecked_gris))]
        except Exception:
            pass
        return points

    def _bead_gri_to_rid(self, names) -> dict[int, str]:
        """Map ``gri`` (numeric bead ID) → R-ID (the ``points_by_gri`` "name",
        e.g. ``R113``) across *names*, for human-readable bead reporting."""
        out: dict[int, str] = {}
        for name in names:
            zroot = self._viewer_zroot_for_dataset_name(name)
            if not zroot:
                continue
            attrs = self._read_zarr_attrs_safe(zroot, "grd/mbm/points")
            pbg = attrs.get("points_by_gri", {}) if isinstance(attrs, dict) else {}
            for key, info in (pbg or {}).items():
                if not isinstance(info, dict):
                    continue
                try:
                    gri = int(info.get("gri", key))
                except (TypeError, ValueError):
                    continue
                rid = info.get("name")
                if rid:
                    out.setdefault(gri, str(rid))
        return out

    @staticmethod
    def _format_bead_ids(gris, gri_to_rid: dict[int, str] | None = None) -> str:
        """``"12 (R113), 27 (R140)"`` — gri plus R-ID when known; ``(none)`` if empty."""
        gri_to_rid = gri_to_rid or {}
        parts = []
        for g in sorted({int(x) for x in (gris or [])}):
            rid = gri_to_rid.get(g)
            parts.append(f"{g} ({rid})" if rid else str(g))
        return ", ".join(parts) if parts else "(none)"

    def _on_show_beads_drift(self):
        from ...msr import state as MFSTATE
        from .beads_drift import gather_msr_bead_drift
        from .beads_drift_dialog import BeadsDriftDialog

        datasets = (self.parsed or {}).get("datasets") or []
        bead_data = gather_msr_bead_drift(datasets, getattr(MFSTATE, "mbm_map", {}))
        if not bead_data:
            QMessageBox.information(
                self, "Show beads drift",
                "The currently parsed .msr file has no MBM/bead data to display.\n\n"
                "Parse a modern multi-channel .msr that contains 'grd/mbm/points'.")
            return
        dlg = BeadsDriftDialog(bead_data, unchecked_gris=self._bead_unchecked_gris, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._bead_unchecked_gris = dlg.unchecked_gris()
            gri_to_rid = {int(b["gri"]): str(b.get("rid", "") or "")
                          for d in bead_data for b in d.get("beads", [])}
            gri_to_rid = {g: r for g, r in gri_to_rid.items() if r}
            kept = sorted(int(g) for g in dlg.selected_gris())
            excluded = sorted(int(g) for g in self._bead_unchecked_gris)
            self.log(f"[beads] selection applied: {len(kept)} bead(s) kept: "
                     f"{self._format_bead_ids(kept, gri_to_rid)}")
            self.log(f"[beads] {len(excluded)} bead(s) excluded from alignment: "
                     f"{self._format_bead_ids(excluded, gri_to_rid)}")

    def _mbm_transform_type(self) -> str:
        """The MBM-handling transform mode from Preferences (Preferences > MBM
        Handling > transform_type); falls back to the rigid-XY + translational-Z
        default when no app state / preference is available."""
        from ...msr.alignment import TRANSFORM_RIGID_XY_Z

        default = TRANSFORM_RIGID_XY_Z
        if self._state is None:
            return default
        return (self._state.prefs.get("mbm_handling", {}) or {}).get(
            "transform_type", default) or default

    def _combined_loc_bounds_nm(self, names) -> tuple[np.ndarray, np.ndarray] | None:
        """Combined raw-localization bounding box (nm) across *names*, for the
        alignment plot's data-range overlay.

        Surveys each dataset's raw ``mfx['loc']`` with no validity check: an
        ``(Nloc, Nitr, 3)`` grid uses the last iteration, a flat m2410
        ``(N, 2/3)`` array is used as-is. Returns ``(min_xyz, max_xyz)`` in nm
        (z padded with 0 for 2-D data), or ``None`` when no coordinates exist.
        """
        from ...msr import state as MFSTATE

        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        for name in names:
            mfx = MFSTATE.mfx_map.get(name)
            fields = set(getattr(getattr(mfx, "dtype", None), "names", ()) or ())
            if "loc" not in fields:
                continue
            loc = np.asarray(mfx["loc"], dtype=np.float64)
            if loc.ndim == 3 and loc.shape[-1] >= 2:
                loc = loc[:, -1, :]  # last iteration of an (Nloc, Nitr, 3) grid
            if loc.ndim != 2 or loc.shape[1] < 2:
                continue
            cols = min(3, loc.shape[1])
            xyz = np.zeros((loc.shape[0], 3), dtype=np.float64)
            xyz[:, :cols] = loc[:, :cols]
            xyz *= 1e9
            xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
            if xyz.size == 0:
                continue
            mins.append(xyz.min(axis=0))
            maxs.append(xyz.max(axis=0))
        if not mins:
            return None
        return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)

    def on_align_channel(self):
        from ...msr import state as MFSTATE
        from ...msr.alignment import MIN_BEADS_FOR_ALIGNMENT, align_channels_from_arrays

        if not self._ensure_single_file_parsed():
            return
        if self.parsed.get("mode") != "modern":
            QMessageBox.information(self, "Align channel", "Channel alignment is available only for modern MINFLUX datasets.")
            return
        datasets = self.parsed.get("datasets") or []
        if len(datasets) < 2:
            QMessageBox.critical(self, "Align channel", f"Expected at least 2 datasets for channel alignment, found {len(datasets)}.")
            return
        ref_name = datasets[0].get("display_name") or datasets[0].get("did") or "channel_1"
        ref_mbm = self._aligned_mbm_points(ref_name)
        if ref_mbm is None:
            QMessageBox.critical(self, "Align channel", "Could not find bead points (`grd/mbm/points`) for the reference channel.")
            return
        transform_type = self._mbm_transform_type()
        self.log(f"[align] MBM-handling transform: {transform_type}")
        try:
            results = []
            aligned_localizations = []
            loc_bound_names = [ref_name]
            for ds in datasets[1:]:
                moving_name = ds.get("display_name") or ds.get("did") or "channel"
                loc_bound_names.append(moving_name)
                moving_mbm = self._aligned_mbm_points(moving_name)
                moving_mfx = MFSTATE.mfx_map.get(moving_name)
                if moving_mbm is None:
                    raise ValueError(f"Could not find bead points (`grd/mbm/points`) for {moving_name}.")
                if moving_mfx is None:
                    raise ValueError(f"Could not find `mfx` localizations for {moving_name}.")
                result = align_channels_from_arrays(
                    ref_name, ref_mbm, moving_name, moving_mbm, transform_type=transform_type)
                if int(result.bead_ids.size) < MIN_BEADS_FOR_ALIGNMENT:
                    raise ValueError(
                        f"Only {int(result.bead_ids.size)} matched bead(s) between "
                        f"'{ref_name}' and '{moving_name}'; at least {MIN_BEADS_FOR_ALIGNMENT} "
                        f"are required for alignment "
                        f"(check the bead selection in 'Show beads drift').")
                results.append(result)
                aligned_localizations.extend(self._build_aligned_localizations_table(moving_name, moving_mfx, result.matrix_4x4))
        except Exception as exc:
            self.log(f"[align] failed: {exc}")
            QMessageBox.critical(self, "Align channel", f"Alignment failed:\n{exc}")
            return
        self.log(f"[align] reference channel: {ref_name}")
        align_names = {ref_name} | {r.channel2_name for r in results}
        gri_to_rid = self._bead_gri_to_rid(align_names)
        excluded = sorted(int(g) for g in (self._bead_unchecked_gris or set()))
        self.log(
            f"[align] user-excluded {len(excluded)} bead(s): "
            f"{self._format_bead_ids(excluded, gri_to_rid)}")
        for result in results:
            used = [int(b) for b in result.bead_ids]
            self.log(
                f"[align] {result.channel2_name} -> {result.channel1_name}; used "
                f"{len(used)} bead(s): {self._format_bead_ids(used, gri_to_rid)}; "
                f"transform={result.transform_type}; "
                f"translation_nm=({result.translation[0]:.3f}, {result.translation[1]:.3f}); "
                f"rmse_xy_nm={result.rmse:.3f}")

        dlg = AlignmentSaveDialog(
            self._channel_alignment_defaults(), self._alignment_save_options, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.log("[align] save cancelled.")
            return
        payload = dlg.result_payload()
        # Remember the checkbox states for next time.
        self._alignment_save_options = {
            k: bool(payload[k])
            for k in ("save_beads", "save_transform", "save_localizations", "show_plot")
        }
        self._save_settings()
        try:
            if payload["save_beads"]:
                self._write_alignment_beads_csv(payload["path_beads"], results)
                self.log(f"[align] wrote beads CSV: {payload['path_beads']}")
            if payload["save_transform"]:
                self._write_alignment_transform_csv(payload["path_transform"], results)
                self.log(f"[align] wrote transform CSV: {payload['path_transform']}")
            if payload["save_localizations"]:
                self._write_dict_rows_csv(payload["path_localizations"], aligned_localizations)
                self.log(f"[align] wrote aligned localizations CSV: {payload['path_localizations']}")
        except Exception as exc:
            self.log(f"[align] save failed: {exc}")
            QMessageBox.critical(self, "Align channel", f"Failed to save alignment outputs:\n{exc}")
            return
        if payload["show_plot"]:
            data_bounds = self._combined_loc_bounds_nm(loc_bound_names)
            AlignmentPlotWindow(results, self, data_bounds_nm=data_bounds).exec()
        summary = [f"Reference channel: {ref_name}"] + [
            f"{result.channel2_name}: matched {result.bead_ids.size}, translation ({result.translation[0]:.3f}, {result.translation[1]:.3f}) nm, RMSE XY {result.rmse:.3f} nm"
            for result in results
        ]
        QMessageBox.information(self, "Align channel", "Alignment completed.\n\n" + "\n".join(summary))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def on_ok(self):
        self._save_settings()
        out_dir = self.out_dir_edit.text().strip() or self._temp_dir()
        if not out_dir:
            QMessageBox.critical(self, "Missing output", "Please set an output folder.")
            return
        os.makedirs(out_dir, exist_ok=True)
        formats = []
        if self.fmt_mat.isChecked():
            formats.append("mat")
        if self.fmt_npy.isChecked():
            formats.append("npy")
        if self.fmt_json.isChecked():
            formats.append("json")
        if self.fmt_csv.isChecked():
            formats.append("csv")
        if self.fmt_zarr.isChecked():
            formats.append("zarr")
        if not formats:
            QMessageBox.critical(self, "No formats", "Pick at least one export format.")
            return
        tmp = self._temp_dir()
        ipath = self.input_path_edit.text().strip()
        if self.mode_folder.isChecked():
            from ...msr import parse_msr_general
            files = self._find_msr_files(ipath)
            if not files:
                self.log("[batch] no .msr files found in the folder.")
                return
            mfx_sel_global, mbm_sel_global = self._derive_global_field_filters()
            self.log(f"[batch] exporting {len(files)} files with global filters: mfx={'ALL' if mfx_sel_global is None else sorted(mfx_sel_global)}; mbm={'ALL' if mbm_sel_global is None else sorted(mbm_sel_global)}")
            for idx, msr_path in enumerate(files, start=1):
                try:
                    sub_out = Path(out_dir) / msr_path.stem
                    os.makedirs(sub_out, exist_ok=True)
                    self.log(f"[batch {idx}/{len(files)}] parse & export: {msr_path} -> {sub_out}")
                    self.parsed = parse_msr_general(str(msr_path), tmp, log=self.log)
                    self._export_current_parsed(str(sub_out), formats, mfx_sel_global, mbm_sel_global)
                    QApplication.processEvents()
                except Exception as exc:
                    self.log(f"[batch error] {msr_path}: {exc}")
            self.log("[done] Batch export complete.")
            return
        if not self.parsed:
            from ...msr import pick_one_msr, parse_msr_general
            msr = pick_one_msr(ipath)
            if not msr:
                self.log("no msr file found!")
                return
            self.parsed = parse_msr_general(msr, tmp, log=self.log)
        self._export_current_parsed(out_dir, formats)
        self.log("[done] Export complete.")

    # ------------------------------------------------------------------
    # Open in viewer (viewer-specific addition)
    # ------------------------------------------------------------------

    def _on_open_in_viewer(self):
        from ...msr import state as MFSTATE
        from ...core.loader import load_from_mfx_array

        if self._state is None:
            QMessageBox.warning(self, "Not connected", "Not connected to a viewer instance.")
            return

        datasets_info = self.parsed.get("datasets", [])
        if not datasets_info:
            QMessageBox.warning(self, "No datasets", "No modern datasets found. Parse an .msr file first.")
            return
        import_plan = self._viewer_import_plan()
        if not import_plan:
            QMessageBox.information(self, "Open in MINFLUX viewer", "No selected datasets or fields are available to open.")
            return

        msr_path = Path(self.parsed.get("msr", ""))
        imported = []
        imported_indices = []
        errors = []
        render_group_id = f"msr:{msr_path.resolve() if str(msr_path) else 'memory'}:{uuid.uuid4().hex}"
        overlay_index = 1
        parent = self._owner
        if parent is not None:
            try:
                overlay_index = int(getattr(parent, "_next_overlay_index", 1))
                setattr(parent, "_next_overlay_index", overlay_index + 1)
            except Exception:
                overlay_index = 1
        viewer_transforms: dict[str, dict] = {}
        align_choice = self._ask_viewer_channel_alignment(import_plan)
        if align_choice is _ALIGN_CANCELLED:
            self.log("[viewer] open in viewer aborted by user")
            return
        if align_choice:
            alignment_mode, reference_name = align_choice
            try:
                viewer_transforms = self._viewer_channel_alignment_transforms(
                    import_plan, alignment_mode, reference_name,
                )
            except Exception as exc:
                self.log(f"[viewer align] failed: {exc}")
                QMessageBox.warning(
                    self,
                    "Open in MINFLUX viewer",
                    f"Channel alignment could not be computed and will be skipped:\n\n{exc}",
                )

        previous_suspend_auto_render = getattr(self._state, "suspend_auto_render", False)
        self._state.suspend_auto_render = True
        try:
            for ds in datasets_info:
                key = ds.get("display_name") or ds.get("did") or "dataset"
                if key not in import_plan:
                    continue
                mfx = MFSTATE.mfx_map.get(key)

                if mfx is None:
                    errors.append(f"Dataset '{key}': mfx array not in memory — skipped.")
                    continue

                try:
                    field_sel = import_plan.get(key)
                    if field_sel is not None:
                        names = set(getattr(getattr(mfx, "dtype", None), "names", []) or [])
                        if "loc" in names and "loc" not in field_sel:
                            field_sel = set(field_sel)
                            field_sel.add("loc")
                            self.log(f"[viewer] added required 'loc' field for '{key}' so it can render.")
                        mfx_to_load = self._subset_struct_fields(mfx, field_sel)
                        if mfx_to_load is None:
                            errors.append(f"Dataset '{key}': selected fields contain no importable mfx data — skipped.")
                            continue
                    else:
                        mfx_to_load = mfx
                    display_name = f"{msr_path.name} | {key}" if msr_path.name else key
                    dataset = load_from_mfx_array(
                        mfx=mfx_to_load,
                        name=display_name,
                        folder=str(msr_path.parent),
                        datetime_str=datetime.now().strftime("%Y-%b-%d, %H:%M:%S"),
                        recent_path=str(msr_path),
                        prefs=self._state.prefs,
                    )
                    from ...core.dataset import AttributeComponent
                    dataset.state["overlay_id"] = render_group_id
                    dataset.state["render_group_id"] = render_group_id
                    dataset.state["overlay_index"] = overlay_index
                    dataset.state["overlay_order"] = len(imported_indices) + 1
                    dataset.state["overlay_lut"] = ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow"][len(imported_indices) % 6]
                    dataset.state["render_channel_lut"] = dataset.state["overlay_lut"]
                    dataset.metadata["msr_source_path"] = str(msr_path)
                    dataset.metadata["msr_dataset_key"] = key
                    dataset.metadata["msr_dataset_name"] = key
                    dataset.metadata["overlay_id"] = render_group_id
                    dataset.metadata["overlay_index"] = overlay_index
                    # The data version (m2410/m2205/legacy) is detected from the
                    # mfx structure by load_from_mfx_array — keep it. Record the
                    # obf/mfxdta CONTAINER separately (it's the transport, not the
                    # data version), and ensure the dataset reads as MINFLUX.
                    src_fmt = ds.get("source_format")
                    if src_fmt:
                        dataset.metadata["source_format"] = src_fmt
                        dataset.metadata["is_minflux"] = True
                        dataset.metadata["has_real_tid"] = True
                        if ds.get("mfxdta_version") is not None:
                            dataset.metadata["mfxdta_version"] = ds.get("mfxdta_version")
                    if key in MFSTATE.mbm_map:
                        dataset.mbm = AttributeComponent({"points": MFSTATE.mbm_map[key]})
                        dataset.metadata["mbm_points"] = MFSTATE.mbm_map[key]
                    if key in viewer_transforms:
                        transform = viewer_transforms[key]
                        dataset.state["overlay_transform"] = transform
                        dataset.state["render_transform_2d"] = transform
                        dataset.metadata["overlay_transform"] = transform
                        dataset.metadata["render_transform_2d"] = transform
                        dataset.metadata["transformed"] = bool(transform.get("moving_channel") != transform.get("reference_channel"))
                    idx = self._state.add_dataset(dataset)
                    loaded = self._state.datasets[idx]
                    loaded.state["overlay_id"] = render_group_id
                    loaded.state["render_group_id"] = render_group_id
                    loaded.state["overlay_index"] = overlay_index
                    loaded.state["overlay_order"] = len(imported_indices) + 1
                    loaded.state["overlay_lut"] = dataset.state.get("overlay_lut")
                    loaded.state["render_channel_lut"] = loaded.state["overlay_lut"]
                    loaded.metadata["msr_source_path"] = str(msr_path)
                    loaded.metadata["msr_dataset_key"] = key
                    loaded.metadata["msr_dataset_name"] = key
                    loaded.metadata["overlay_id"] = render_group_id
                    loaded.metadata["overlay_index"] = overlay_index
                    if key in MFSTATE.mbm_map:
                        loaded.mbm = AttributeComponent({"points": MFSTATE.mbm_map[key]})
                        loaded.metadata["mbm_points"] = MFSTATE.mbm_map[key]
                    if key in viewer_transforms:
                        transform = viewer_transforms[key]
                        loaded.state["overlay_transform"] = transform
                        loaded.state["render_transform_2d"] = transform
                        loaded.metadata["overlay_transform"] = transform
                        loaded.metadata["render_transform_2d"] = transform
                        loaded.metadata["transformed"] = bool(transform.get("moving_channel") != transform.get("reference_channel"))
                    imported_indices.append(idx)
                    imported.append(key)
                    self.log(f"[viewer] '{key}' → {loaded.prop.num_loc:,} loc")
                except Exception as exc:
                    errors.append(f"Dataset '{key}': {exc}")
                    self.log(f"[error] {key}: {exc}")
        finally:
            self._state.suspend_auto_render = previous_suspend_auto_render

        # A single imported dataset is standalone, not an overlay channel — drop
        # the assigned channel colour so it renders with the default LUT (hot).
        # The Red/Green/Blue cycle is only meaningful for a multi-channel overlay.
        if len(imported_indices) == 1:
            sole = self._state.datasets[imported_indices[0]]
            sole.state.pop("overlay_lut", None)
            sole.state.pop("render_channel_lut", None)

        # Record an overlay-level summary so the Method-text generator can
        # describe the whole multi-channel acquisition (source .msr, channels and
        # their properties, reference, bead selection, alignment matrix) — tagged
        # to the first channel so it auto-checks with the overlay. A single
        # imported dataset is standalone, not an overlay, so skip it.
        if len(imported_indices) >= 2:
            self._record_overlay_load_event(
                msr_path, imported, imported_indices, viewer_transforms,
                align_choice if (align_choice and align_choice is not _ALIGN_CANCELLED) else None,
            )

        if errors:
            QMessageBox.warning(
                self, "Import warnings",
                "Some datasets could not be imported:\n\n" + "\n".join(errors),
            )
        if imported:
            self.log(f"[viewer] imported {len(imported)} dataset(s): {', '.join(imported)}")
            parent = self._owner
            if parent is not None and hasattr(parent, "_show_render") and imported_indices:
                existing = getattr(parent, "_render_windows", {}).get(imported_indices[0])
                if existing is not None and hasattr(existing, "_refresh_from_dataset"):
                    existing._refresh_from_dataset()
                parent._show_render(imported_indices[0])
            self.close()

    def _record_overlay_load_event(self, msr_path, imported, imported_indices,
                                   viewer_transforms, align_choice):
        """Persist an overlay summary on each channel's metadata and emit one
        parseable, overlay-tagged Log event the Method-text generator can expand
        into a full description of the multi-channel load + alignment."""
        if self._state is None:
            return
        alignment_mode = None
        reference_key = imported[0] if imported else None
        if align_choice:
            alignment_mode, ref_name = align_choice
            if ref_name:
                reference_key = ref_name
        for transform in (viewer_transforms or {}).values():
            rc = transform.get("reference_channel")
            if rc:
                reference_key = rc
                break
        excluded = (sorted(int(g) for g in self._bead_unchecked_gris)
                    if self._bead_unchecked_gris else [])

        for idx in imported_indices:
            try:
                md = self._state.datasets[idx].metadata
            except Exception:
                continue
            md["overlay_channels"] = list(imported)
            md["overlay_reference"] = reference_key
            md["overlay_alignment_mode"] = alignment_mode or "none"
            md["overlay_bead_excluded"] = list(excluded)

        msg = (f"MSR overlay loaded from '{Path(msr_path).name}': {len(imported)} channel(s) "
               f"[{', '.join(imported)}]; reference channel '{reference_key}'; "
               f"channel alignment: {alignment_mode or 'none'}")
        if alignment_mode == "mbm info":
            if excluded:
                msg += f"; beads: all but {len(excluded)} user-excluded ({excluded})"
            else:
                msg += "; beads: all available beads used"
        msg += "."
        try:
            self._state.log(msg, dataset_idx=imported_indices[0])
        except TypeError:       # older AppState.log without dataset_idx
            self._state.log(msg)


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def msr_available() -> bool:
    try:
        import msr_reader  # noqa: F401
        return True
    except Exception:
        return False


def msr_unavailable_message() -> str:
    return (
        "The <b>msr-reader</b> library is not installed.<br><br>"
        "Opening <tt>.msr</tt> files requires it. It is a pure-Python, "
        "cross-platform dependency normally installed automatically:<br><br>"
        "<tt>poetry install</tt>  (or)  <tt>pip install msr-reader</tt>"
    )


def open_msr(
    msr_path: str,
    state=None,
    parent: QWidget | None = None,
) -> None:
    """Open the MSR reader dialog for *msr_path*."""
    if not msr_available():
        QMessageBox.warning(parent, "MSR reader not available", msr_unavailable_message())
        return
    dlg = MsrReaderDialog(state=state, msr_path=msr_path, parent=parent)
    # The dialog is an unowned top-level window (so it does not sit permanently
    # in front of the viewer); keep a reference on the owner so it is not
    # garbage-collected once this function returns.
    if parent is not None:
        parent._plugin_msr_reader_dialog = dlg
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication.instance() or QApplication([])
    window = MsrReaderDialog()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
