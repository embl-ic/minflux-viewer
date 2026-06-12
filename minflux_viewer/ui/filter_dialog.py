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

from pathlib import Path
import math

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence, QShortcut
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
from ..core.attributes import is_trace_wise_attribute, plot_attribute_names
from ..core.loader import attr_values_1d
from ..core.filter_io import (
    is_filter_json_file,
    is_filter_json_payload,
    load_filter_json as load_filter_json_file,
    save_filter_json as save_filter_json_file,
)

_AGG_MODES = ["per loc", "trace mean", "trace stdev", "trace max", "trace min", "trace range"]
_FILTER_ATTR_PRIORITY = ("efo", "cfr", "siz", "tim", "tid", "sta", "itr")

# Column indices
_COL_ENABLED = 0
_COL_ATTR    = 1
_COL_MODE    = 2
_COL_MIN     = 3
_COL_MAX     = 4
_COL_DISPLAY = 5
_COL_DELETE  = 6
_NCOLS       = 7


class SmartBoundsSpinBox(QDoubleSpinBox):
    """Filter-bound spin box with data-type-aware display and stepping."""

    _SCI_LOW = 1e-3
    _SCI_HIGH = 1e6

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._kind = "float"
        self._data_min = 0.0
        self._data_max = 1.0
        self.setRange(-1e15, 1e15)
        self.configure(value=0.0, values=None, mode="per loc")

    def configure(
        self,
        *,
        value: float | None = None,
        values: np.ndarray | None = None,
        data_min: float | None = None,
        data_max: float | None = None,
        mode: str = "per loc",
    ) -> None:
        if values is not None:
            arr = np.asarray(values)
            finite = arr[np.isfinite(arr.astype(float, copy=False))] if arr.size else np.asarray([], dtype=float)
            if finite.size:
                data_min = float(np.nanmin(finite.astype(float, copy=False)))
                data_max = float(np.nanmax(finite.astype(float, copy=False)))
            self._kind = self._infer_kind(arr, mode)
        else:
            self._kind = "float" if mode in {"trace mean", "trace stdev", "trace range"} else self._kind

        if data_min is not None:
            self._data_min = float(data_min)
        if data_max is not None:
            self._data_max = float(data_max)
        lo, hi = sorted((self._data_min, self._data_max))
        span = max(abs(hi - lo), abs(lo), abs(hi), 1.0)
        self.setRange(lo - span, hi + span)
        self._apply_display_rules()
        if value is not None:
            self.setValue(float(value))
        self.setSingleStep(self._smart_step())

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt API
        if self._kind == "bool":
            return str(int(round(max(0.0, min(1.0, value)))))
        if self._kind == "int":
            return str(int(round(value)))
        return self._format_float(float(value))

    def valueFromText(self, text: str) -> float:  # noqa: N802 - Qt API
        text = text.strip().replace(",", "")
        try:
            if self._kind in {"bool", "int"}:
                return float(int(round(float(text))))
            return float(text)
        except Exception:
            return float(self.value())

    def stepBy(self, steps: int) -> None:  # noqa: N802 - Qt API
        value = float(self.value()) + int(steps) * float(self.singleStep())
        if self._kind == "bool":
            value = max(0.0, min(1.0, round(value)))
        elif self._kind == "int":
            value = round(value)
        self.setValue(value)

    def _apply_display_rules(self) -> None:
        if self._kind == "bool":
            self.setDecimals(0)
            self.setRange(0.0, 1.0)
        elif self._kind == "int":
            self.setDecimals(0)
        else:
            # Keep enough decimals internally for metre-scale values while
            # textFromValue controls the visible representation.
            self.setDecimals(12)

    def _smart_step(self) -> float:
        if self._kind == "bool":
            return 1.0
        lo, hi = sorted((float(self._data_min), float(self._data_max)))
        span = abs(hi - lo)
        if not math.isfinite(span) or span <= 0.0:
            span = max(abs(float(self.value())), abs(lo), abs(hi), 1.0)
        exponent = math.floor(math.log10(span)) - 1 if span > 0.0 else -2
        step = 10.0 ** exponent
        if self._kind == "int":
            return float(max(1, int(round(step))))
        return float(step)

    @classmethod
    def _infer_kind(cls, values: np.ndarray, mode: str) -> str:
        if mode in {"trace mean", "trace median", "trace stdev", "trace range"}:
            return "float"
        if values.dtype.kind == "b":
            return "bool"
        if values.dtype.kind in {"i", "u"} and mode in {"per loc", "trace min", "trace max"}:
            return "int"
        return "float"

    @classmethod
    def _format_float(cls, value: float) -> str:
        if not math.isfinite(value):
            return str(value)
        abs_value = abs(value)
        if abs_value != 0.0 and (abs_value < cls._SCI_LOW or abs_value >= cls._SCI_HIGH):
            return f"{value:.3e}"
        return f"{value:.3f}"


class FilterDialog(QDialog):
    """Non-modal filter management dialog."""

    def __init__(
        self,
        state: AppState,
        parent: QWidget | None = None,
        *,
        dataset_idx: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._dataset_idx = dataset_idx if dataset_idx is not None else state.active_idx
        self._applying = False

        self.setWindowTitle("Filter")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(720, 360)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # Single-shot timer — debounces rapid edits (spinners) before applying.
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.timeout.connect(self._apply_all)
        self._view_shortcuts: dict[str, QShortcut] = {}

        self._build_ui()
        self._populate_attr_lists()
        self._install_view_shortcuts()

        # Accept JSON filter files dropped anywhere on the dialog.
        # The table widget must not intercept drops, or they won't reach here.
        self.setAcceptDrops(True)
        self._table.setAcceptDrops(False)
        self._table.viewport().setAcceptDrops(False)

        state.active_changed.connect(self._on_active_changed)

    @property
    def dataset_idx(self) -> int | None:
        return self._dataset_idx

    def _dataset(self):
        if self._dataset_idx is None:
            return None
        if not (0 <= self._dataset_idx < len(self._state.datasets)):
            return None
        return self._state.datasets[self._dataset_idx]

    def _install_view_shortcuts(self) -> None:
        """Open plot/render windows for the dataset owned by this filter dialog."""
        defaults = {
            "attribute_plot": "Ctrl+1",
            "attribute_histogram": "Ctrl+2",
            "scatter_plot": "Ctrl+3",
            "render": "Ctrl+R",
            "dataset_manager": "Ctrl+D",
        }
        shortcuts = self._state.prefs.get("shortcuts", {})
        for command, fallback in defaults.items():
            seq = str(shortcuts.get(command, fallback) or fallback)
            shortcut = QShortcut(QKeySequence(seq), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda cmd=command: self._open_associated_view(cmd))
            self._view_shortcuts[command] = shortcut

    def _open_associated_view(self, command: str) -> None:
        idx = self._dataset_idx
        if idx is None or not (0 <= idx < len(self._state.datasets)):
            return
        main = self._main_window()
        if main is None:
            return
        self._state.set_active(idx)
        if command == "dataset_manager":
            main._show_dataset_manager()
            return
        if command == "attribute_plot":
            win = main._show_attr_plot(idx)
        elif command == "attribute_histogram":
            win = main._show_histogram(idx)
        elif command == "scatter_plot":
            win = self._show_overlay_aware_window(
                main,
                idx,
                window_attr="_scatter_windows",
                show_method="_show_scatter",
            )
        elif command == "render":
            win = self._show_overlay_aware_window(
                main,
                idx,
                window_attr="_render_windows",
                show_method="_show_render",
            )
        else:
            return
        if win is not None:
            try:
                win.show()
                win.raise_()
                win.activateWindow()
            except Exception:
                pass

    def _show_overlay_aware_window(self, main, idx: int, *, window_attr: str, show_method: str):
        group_indices = self._overlay_group_indices(idx)
        if len(group_indices) <= 1:
            return getattr(main, show_method)(idx)

        win = self._find_group_window(main, group_indices, idx, window_attr)
        if win is None:
            win = getattr(main, show_method)(group_indices[0])
        self._activate_window_channel(win, idx)
        return win

    def _overlay_group_indices(self, idx: int) -> list[int]:
        try:
            from ..core.overlay import overlay_members

            members = overlay_members(self._state, idx)
        except Exception:
            members = []
        return [int(member_idx) for member_idx, _ds in members] or [idx]

    def _find_group_window(self, main, group_indices: list[int], target_idx: int, window_attr: str):
        group_set = set(group_indices)
        mapping = getattr(main, window_attr, {})
        for win in list(mapping.values()):
            try:
                channels = getattr(win, "_channels", None) or []
                if any(ch.get("dataset_idx") == target_idx for ch in channels):
                    return win
                win_idx = getattr(win, "_idx", getattr(win, "_dataset_idx", None))
                if type(win_idx) is int and win_idx in group_set:
                    return win
            except RuntimeError:
                continue
            except Exception:
                continue
        return None

    def _activate_window_channel(self, win, idx: int) -> None:
        if win is None or not (0 <= idx < len(self._state.datasets)):
            return
        try:
            channels = getattr(win, "_channels", None) or []
            if any(ch.get("dataset_idx") == idx for ch in channels):
                if hasattr(win, "_idx"):
                    win._idx = idx
                if hasattr(win, "_dataset_idx"):
                    win._dataset_idx = idx
                self._state.set_active(idx)
                if hasattr(win, "_update_overlay_title"):
                    win._update_overlay_title()
        except Exception:
            pass

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
            ["On", "Attribute", "Mode", "Min", "Max", "", ""]
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_ENABLED, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(_COL_DISPLAY, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(_COL_DELETE,  QHeaderView.ResizeMode.Fixed)
        hh.setStretchLastSection(False)
        self._table.setColumnWidth(_COL_ENABLED, 36)
        self._table.setColumnWidth(_COL_ATTR,   170)
        self._table.setColumnWidth(_COL_MODE,   130)
        self._table.setColumnWidth(_COL_MIN,    130)
        self._table.setColumnWidth(_COL_MAX,    130)
        self._table.setColumnWidth(_COL_DISPLAY, 36)
        self._table.setColumnWidth(_COL_DELETE,  36)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._table)

        # ── Button row ────────────────────────────────────────────
        bar = QHBoxLayout()

        add_btn = QPushButton("Add filter row")
        add_btn.setAutoDefault(False)
        add_btn.setDefault(False)
        add_btn.clicked.connect(lambda _checked=False: self._add_row())
        bar.addWidget(add_btn)

        apply_btn = QPushButton("Apply all")
        apply_btn.setAutoDefault(False)
        apply_btn.setDefault(False)
        apply_btn.clicked.connect(self._apply_all)
        bar.addWidget(apply_btn)

        reset_btn = QPushButton("Reset all filters")
        reset_btn.setAutoDefault(False)
        reset_btn.setDefault(False)
        reset_btn.clicked.connect(self._reset_all)
        bar.addWidget(reset_btn)

        bar.addStretch()

        save_btn = QPushButton("Save")
        save_btn.setAutoDefault(False)
        save_btn.setDefault(False)
        save_btn.clicked.connect(self._save_json)
        bar.addWidget(save_btn)

        load_btn = QPushButton("Load")
        load_btn.setAutoDefault(False)
        load_btn.setDefault(False)
        load_btn.clicked.connect(lambda _checked=False: self.load_filter_json())
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
        ds = self._dataset()
        if ds is None:
            self._ds_label.setText("No Dataset Loaded")
            self._numeric_attrs: list[str] = []
            return

        self._ds_label.setText(f"Dataset: {ds.name}")
        self._numeric_attrs = plot_attribute_names(ds, self._state.prefs, exclude=("ftr",))
        if not self._numeric_attrs:
            self._numeric_attrs = [
                key for key in ds.attr.keys()
                if key != "ftr"
                and np.asarray(ds.attr[key]).ndim == 1
                and np.issubdtype(np.asarray(ds.attr[key]).dtype, np.number)
            ]
        self._numeric_attrs = _prioritize_filter_attrs(self._numeric_attrs)
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
                self._enforce_trace_mode(row)

    def _add_row(self, attr: str = "efo", mode: str = "per loc",
                 lo: float = 0.0, hi: float = 1.0, enabled: bool = False,
                 min_inclusive: bool = True, max_inclusive: bool = True,
                 auto_range: bool = True) -> None:
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
        attr_combo.currentTextChanged.connect(lambda text, r=row: self._on_row_attr_changed(text, r))
        self._table.setCellWidget(row, _COL_ATTR, attr_combo)

        # Mode combo
        mode_combo = QComboBox()
        mode_combo.blockSignals(True)
        mode_combo.addItems(_AGG_MODES)
        mode_combo.setCurrentText(mode)
        mode_combo.blockSignals(False)
        mode_combo.currentTextChanged.connect(lambda _text, r=row: self._on_row_mode_changed(r))
        self._table.setCellWidget(row, _COL_MODE, mode_combo)

        # Min / Max spinners — use debounce timer so dragging doesn't hammer the renderer
        min_spin = _make_spin(lo)
        max_spin = _make_spin(hi)
        min_spin.setProperty("inclusive", bool(min_inclusive))
        max_spin.setProperty("inclusive", bool(max_inclusive))
        min_spin.editingFinished.connect(self._apply_all)
        max_spin.editingFinished.connect(self._apply_all)
        self._table.setCellWidget(row, _COL_MIN, min_spin)
        self._table.setCellWidget(row, _COL_MAX, max_spin)

        # Eye / display-in-histogram button
        eye_btn = QPushButton("👁")
        eye_btn.setFixedWidth(30)
        eye_btn.setToolTip("Show this filter in the histogram")
        eye_btn.setAutoDefault(False)
        eye_btn.setDefault(False)
        eye_btn.clicked.connect(lambda _, r=row: self._display_filter(r))
        self._table.setCellWidget(row, _COL_DISPLAY, eye_btn)

        # Delete button
        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(30)
        del_btn.setAutoDefault(False)
        del_btn.setDefault(False)
        del_btn.clicked.connect(lambda _, r=row: self._delete_row(r))
        self._table.setCellWidget(row, _COL_DELETE, del_btn)

        self._enforce_trace_mode(row)
        if auto_range:
            self._auto_fill_range(attr_combo.currentText(), row)
        else:
            self._configure_row_spinners(row, keep_values=True)
        self._table.resizeColumnsToContents()

    def _on_row_attr_changed(self, attr: str, row: int) -> None:
        self._enforce_trace_mode(row)
        self._auto_fill_range(attr, row)
        self._table.resizeColumnsToContents()

    def _on_row_mode_changed(self, row: int) -> None:
        self._enforce_trace_mode(row)
        self._configure_row_spinners(row, keep_values=True)

    def _enforce_trace_mode(self, row: int) -> None:
        attr_combo = self._table.cellWidget(row, _COL_ATTR)
        mode_combo = self._table.cellWidget(row, _COL_MODE)
        if not isinstance(attr_combo, QComboBox) or not isinstance(mode_combo, QComboBox):
            return
        trace_wise = is_trace_wise_attribute(attr_combo.currentText())
        mode_combo.blockSignals(True)
        if trace_wise:
            mode_combo.setCurrentText("trace mean")
            mode_combo.setEnabled(False)
            mode_combo.setToolTip("This is a track-level attribute expanded per localization; filtering uses trace mean.")
        else:
            mode_combo.setEnabled(True)
            mode_combo.setToolTip("")
        mode_combo.blockSignals(False)

    def _configure_row_spinners(self, row: int, *, keep_values: bool) -> None:
        ds = self._dataset()
        attr_combo = self._table.cellWidget(row, _COL_ATTR)
        mode_combo = self._table.cellWidget(row, _COL_MODE)
        min_spin = self._table.cellWidget(row, _COL_MIN)
        max_spin = self._table.cellWidget(row, _COL_MAX)
        if ds is None or not isinstance(attr_combo, QComboBox) or not isinstance(mode_combo, QComboBox):
            return
        if not isinstance(min_spin, SmartBoundsSpinBox) or not isinstance(max_spin, SmartBoundsSpinBox):
            return
        attr = attr_combo.currentText()
        if attr not in ds.attr:
            return
        raw_values, range_values = _filter_spinner_values(ds, attr, mode_combo.currentText())
        if raw_values.size == 0 or range_values.size == 0:
            return
        vals_float = range_values.astype(float, copy=False)
        finite = vals_float[np.isfinite(vals_float)]
        if finite.size == 0:
            return
        data_min = float(np.nanmin(finite))
        data_max = float(np.nanmax(finite))
        mode = mode_combo.currentText()
        for spin, fallback in ((min_spin, data_min), (max_spin, data_max)):
            spin.blockSignals(True)
            spin.configure(
                value=float(spin.value()) if keep_values else fallback,
                values=raw_values,
                data_min=data_min,
                data_max=data_max,
                mode=mode,
            )
            spin.blockSignals(False)

    def _delete_row(self, row: int) -> None:
        self._table.removeRow(row)
        self._apply_all()  # views update immediately when a row is removed

    def _auto_fill_range(self, attr: str, row: int | None = None) -> None:
        """Fill Min/Max spinners with the attribute's actual range."""
        ds = self._dataset()
        if ds is None or attr not in ds.attr:
            return
        mode_combo = self._table.cellWidget(row, _COL_MODE)
        mode = mode_combo.currentText() if isinstance(mode_combo, QComboBox) else "per loc"
        raw_vals, range_vals = _filter_spinner_values(ds, attr, mode)
        finite = range_vals.astype(float, copy=False)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))

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
            if isinstance(spin, SmartBoundsSpinBox):
                spin.blockSignals(True)
                spin.configure(value=val, values=raw_vals, data_min=lo, data_max=hi, mode=mode)
                spin.blockSignals(False)

    # ------------------------------------------------------------------
    # Apply / Reset
    # ------------------------------------------------------------------

    def _apply_all(self) -> None:
        ds = self._dataset()
        if ds is None:
            return

        mask = np.ones(ds.prop.num_loc, dtype=bool)
        specs: list[dict] = []

        for row in range(self._table.rowCount()):
            chk  = self._table.cellWidget(row, _COL_ENABLED)
            if not (isinstance(chk, QCheckBox) and chk.isChecked()):
                continue

            attr_combo = self._table.cellWidget(row, _COL_ATTR)
            mode_combo = self._table.cellWidget(row, _COL_MODE)
            min_spin   = self._table.cellWidget(row, _COL_MIN)
            max_spin   = self._table.cellWidget(row, _COL_MAX)
            if any(widget is None for widget in (attr_combo, mode_combo, min_spin, max_spin)):
                continue

            attr = attr_combo.currentText()
            mode = mode_combo.currentText()
            lo   = min_spin.value()
            hi   = max_spin.value()
            lo_inc = bool(min_spin.property("inclusive") if min_spin.property("inclusive") is not None else True)
            hi_inc = bool(max_spin.property("inclusive") if max_spin.property("inclusive") is not None else True)

            if attr not in ds.attr:
                continue

            # Persist the spec so it can be re-evaluated against the raw
            # all-iteration store (Stage B iteration/validity browsing).
            specs.append({
                "attribute": attr, "mode": mode,
                "lo": float(lo), "hi": float(hi),
                "lo_inc": lo_inc, "hi_inc": hi_inc,
            })

            raw = attr_values_1d(ds, attr)
            if raw is None:
                # No per-loc 1-D representation; the persisted spec still
                # applies on the raw store in the plot windows.
                continue
            raw = np.asarray(raw).ravel().astype(float)
            row_mask = _compute_mask(raw, mode, lo, hi, ds, lo_inc, hi_inc)
            mask &= row_mask

        ds.state["filter_specs"] = specs

        self._applying = True
        ds.filter_mask = mask
        self._state.notify_filter_changed(self._dataset_idx)
        self._applying = False

        n_pass = int(mask.sum())
        self._info.setText(
            f"{n_pass:,} / {ds.prop.num_loc:,} localisations pass all enabled filters."
        )

    def _reset_all(self) -> None:
        ds = self._dataset()
        if ds is None:
            return
        ds.state["filter_specs"] = []
        ds.filter_mask = np.ones(ds.prop.num_loc, dtype=bool)
        self._state.notify_filter_changed(self._dataset_idx)
        self._info.setText("All filters reset.")

    def _display_filter(self, row: int) -> None:
        ds = self._dataset()
        if ds is None or not (0 <= row < self._table.rowCount()):
            return
        attr_combo = self._table.cellWidget(row, _COL_ATTR)
        mode_combo = self._table.cellWidget(row, _COL_MODE)
        min_spin = self._table.cellWidget(row, _COL_MIN)
        max_spin = self._table.cellWidget(row, _COL_MAX)
        if not all(isinstance(w, (QComboBox, SmartBoundsSpinBox)) for w in (attr_combo, mode_combo, min_spin, max_spin)):
            return
        if self._dataset_idx is not None:
            self._state.set_active(self._dataset_idx)
        main = self._main_window()
        hist = None
        histogram_existed = False
        if main is not None:
            existing_hist = getattr(main, "_histogram_windows", {}).get(self._dataset_idx)
            histogram_existed = existing_hist is not None and existing_hist.isVisible()
            hist = main._show_histogram(self._dataset_idx)
        if hist is None:
            from .histogram_window import HistogramWindow
            hist = HistogramWindow(self._state, dataset_idx=self._dataset_idx)
            hist.show()
        hist.show()
        hist.raise_()
        hist.activateWindow()

        enabled_chk = self._table.cellWidget(row, _COL_ENABLED)
        snapshot = {
            "enabled": enabled_chk.isChecked() if isinstance(enabled_chk, QCheckBox) else False,
            "attr": attr_combo.currentText(),
            "mode": mode_combo.currentText(),
            "lo": float(min_spin.value()),
            "hi": float(max_spin.value()),
            "lo_inc": bool(min_spin.property("inclusive") if min_spin.property("inclusive") is not None else True),
            "hi_inc": bool(max_spin.property("inclusive") if max_spin.property("inclusive") is not None else True),
            "mask": ds.filter_mask.copy(),
        }

        def write_values(lo: float, hi: float, _lo_inc: bool = True, _hi_inc: bool = True) -> None:
            min_spin.setValue(min(lo, hi))
            max_spin.setValue(max(lo, hi))
            min_spin.setProperty("inclusive", bool(_lo_inc))
            max_spin.setProperty("inclusive", bool(_hi_inc))
            self._table.resizeColumnsToContents()
            chk = self._table.cellWidget(row, _COL_ENABLED)
            if isinstance(chk, QCheckBox):
                chk.setChecked(True)
            self._apply_all()

        def cancel() -> None:
            attr_combo.setCurrentText(snapshot["attr"])
            mode_combo.setCurrentText(snapshot["mode"])
            min_spin.setValue(snapshot["lo"])
            max_spin.setValue(snapshot["hi"])
            min_spin.setProperty("inclusive", snapshot["lo_inc"])
            max_spin.setProperty("inclusive", snapshot["hi_inc"])
            if isinstance(enabled_chk, QCheckBox):
                enabled_chk.setChecked(snapshot["enabled"])
            ds.filter_mask = snapshot["mask"]
            self._state.notify_filter_changed(self._dataset_idx)
            self._update_info()

        hist.start_filter_edit(
            attr=attr_combo.currentText(),
            mode=mode_combo.currentText(),
            lo=min_spin.value(),
            hi=max_spin.value(),
            on_update=write_values,
            on_finish=write_values,
            on_cancel=cancel,
            restore_view_on_finish=histogram_existed,
            restore_view_on_cancel=histogram_existed,
        )

    def _main_window(self):
        from PyQt6.QtWidgets import QApplication, QMainWindow
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QMainWindow) and hasattr(widget, "_show_histogram"):
                return widget
        return None

    # ------------------------------------------------------------------
    # JSON save / load
    # ------------------------------------------------------------------

    def _save_json(self) -> None:
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
                "min_inclusive": bool(lo.property("inclusive") if isinstance(lo, QDoubleSpinBox) and lo.property("inclusive") is not None else True),
                "max_inclusive": bool(hi.property("inclusive") if isinstance(hi, QDoubleSpinBox) and hi.property("inclusive") is not None else True),
            })
        try:
            save_filter_json_file(path, rows)
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))
            return
        self._info.setText(f"Saved: {Path(path).name}")

    def load_filter_json(self, path: str | None = None) -> None:
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
            rows = load_filter_json_file(path)
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
                enabled = bool(r.get("apply", False)),
                min_inclusive = bool(r.get("min_inclusive", True)),
                max_inclusive = bool(r.get("max_inclusive", True)),
                auto_range = False,
            )
            added += 1
        self._info.setText(f"Appended {added} row(s) from {Path(path).name}")

    # ------------------------------------------------------------------
    # Drag-and-drop  (JSON filter files)
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path.lower().endswith(".json") and is_filter_json_file(path):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path.lower().endswith(".json") and is_filter_json_file(path):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".json"):
                self.load_filter_json(path)
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_active_changed(self, _idx: int) -> None:
        if self._dataset_idx is None and _idx is not None:
            self._dataset_idx = _idx
        self._populate_attr_lists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spin(value: float) -> SmartBoundsSpinBox:
    spin = SmartBoundsSpinBox()
    spin.configure(value=value, data_min=value, data_max=value, mode="per loc")
    return spin


def _prioritize_filter_attrs(attrs: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in _FILTER_ATTR_PRIORITY:
        if name in attrs and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in attrs:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _filter_spinner_values(ds, attr: str, mode: str) -> tuple[np.ndarray, np.ndarray]:
    raw = attr_values_1d(ds, attr)
    raw = np.empty(0) if raw is None else np.asarray(raw).ravel()
    if raw.size == 0 or mode == "per loc":
        return raw, raw
    raw_float = raw.astype(float, copy=False)
    trace_idx = np.asarray(getattr(ds.prop, "trace_idx", np.empty((0, 2))), dtype=int)
    num_traces = int(getattr(ds.prop, "num_traces", 0) or trace_idx.shape[0])
    if trace_idx.ndim != 2 or trace_idx.shape[1] < 2 or num_traces <= 0:
        return raw, raw
    fn = {
        "trace mean": np.nanmean,
        "trace median": np.nanmedian,
        "trace min": np.nanmin,
        "trace max": np.nanmax,
        "trace stdev": np.nanstd,
        "trace range": lambda a: float(np.nanmax(a)) - float(np.nanmin(a)),
    }.get(mode)
    if fn is None:
        return raw, raw
    values: list[float] = []
    with np.errstate(all="ignore"):
        for start, stop in trace_idx[:num_traces]:
            start_i = max(0, int(start))
            stop_i = min(int(stop), raw_float.size - 1)
            if stop_i < start_i:
                continue
            values.append(float(fn(raw_float[start_i:stop_i + 1])))
    return raw, np.asarray(values, dtype=float)


def _compute_mask(
    raw: np.ndarray,
    mode: str,
    lo: float,
    hi: float,
    ds,
    lo_inclusive: bool = True,
    hi_inclusive: bool = True,
) -> np.ndarray:
    """Return a per-loc boolean mask for a single filter row."""
    from ..utils.filters import compute_filter_mask
    return compute_filter_mask(
        raw, mode, lo, hi,
        ds.prop.trace_idx,
        ds.prop.num_loc_per_trace,
        ds.prop.num_traces,
        lo_inclusive,
        hi_inclusive,
    )
