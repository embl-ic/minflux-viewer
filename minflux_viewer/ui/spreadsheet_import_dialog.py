"""
Spreadsheet column-mapping dialog.

Shown when :func:`minflux_viewer.core.spreadsheet_loader.auto_import` can't load
a table unattended (ambiguous columns, or camera-pixel coordinates without a
pixel size). The role combos, per-coordinate units, and pixel-size field are
**pre-filled** from the loader's best guess, and the preview shows a handful of
representative rows spanning the whole file.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..core.spreadsheet_loader import (
    COORD_ROLES,
    DEFAULT_PIXEL_SIZE_NM,
    ROLES,
    AutoImportAmbiguity,
    SpreadsheetTable,
    build_dataset_from_mapping,
    guess_mapping,
    guess_time_unit,
    guess_units,
    read_table,
    representative_row_indices,
    table_stats,
)

# Role → (display label, required?)
_ROLE_LABELS: dict[str, tuple[str, bool]] = {
    "x": ("x (→ xnm)", True),
    "y": ("y (→ ynm)", True),
    "z": ("z (→ znm)", False),
    "prec_xy": ("precision xy", False),
    "prec_z": ("precision z", False),
    "id": ("trace id (→ tid)", False),
    "frame": ("time / frame (→ tim)", False),
    "photons": ("photons", False),
}
# Display unit ↔ internal unit token.
_UNIT_CHOICES = (("nm", "nm"), ("µm", "um"), ("mm", "mm"), ("m", "m"), ("pixel", "px"))
# Time unit for the frame → tim column (None = frame index, kept as-is).
_TIME_CHOICES = (("frames", None), ("s", "s"), ("ms", "ms"))
_NONE = "<none>"


class SpreadsheetMappingDialog(QDialog):
    """Map spreadsheet columns to localization roles and build a dataset.

    Roles, coordinate/time units, and the pixel size are **pre-filled** from the
    loader's value-based best guess (headers first, then column statistics), so
    a headerless MINFLUX-like table opens with x/y/z/tid/tim already populated —
    the user only confirms or corrects, then imports.
    """

    def __init__(self, table: SpreadsheetTable,
                 mapping: dict[str, str | None] | None = None,
                 units: dict[str, str] | None = None,
                 *, time_unit: str | None = None,
                 pixel_size_nm: float | None = None,
                 prefs: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self._table = table
        self._stats = table_stats(table)                 # cheap per-column stats
        self._mapping0 = mapping or guess_mapping(table, use_values=True, stats_map=self._stats)
        self._units0 = units or guess_units(table, self._mapping0, stats_map=self._stats)
        self._time_unit0 = time_unit if time_unit is not None else \
            guess_time_unit(table, self._mapping0, stats_map=self._stats)
        pref_px = ((prefs or {}).get("data", {}) or {}).get("pixel_size_nm")
        self._pixel0 = pixel_size_nm or pref_px or DEFAULT_PIXEL_SIZE_NM
        self._prefs = prefs
        self._role_combos: dict[str, QComboBox] = {}
        self._unit_combos: dict[str, QComboBox] = {}
        self._time_combo: QComboBox | None = None

        self.setWindowTitle(f"Open spreadsheet — {Path(table.path).name}")
        self.resize(940, 620)
        self._build_ui()
        self._update_pixel_enabled()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            f"<b>{self._table.n_rows:,}</b> rows · detected tool: "
            f"<b>{self._table.detected_tool}</b>. Assign columns to roles "
            "(<b>x</b> and <b>y</b> are required); coordinate units are pre-filled."))

        col_names = [c.name for c in self._table.numeric_columns()]
        grp = QGroupBox("Column mapping")
        grid = QGridLayout(grp)
        grid.addWidget(QLabel("<b>role</b>"), 0, 0)
        grid.addWidget(QLabel("<b>column</b>"), 0, 1)
        grid.addWidget(QLabel("<b>unit</b>"), 0, 2)
        for r, role in enumerate(ROLES, start=1):
            label, required = _ROLE_LABELS[role]
            grid.addWidget(QLabel(label + (" *" if required else "")), r, 0)

            combo = QComboBox()
            combo.addItem(_NONE, None)
            for i, name in enumerate(col_names, start=1):
                combo.addItem(name, name)
                combo.setItemData(i, self._stats_tooltip(name), Qt.ItemDataRole.ToolTipRole)
            chosen = self._mapping0.get(role)
            idx = combo.findData(chosen) if chosen else 0
            combo.setCurrentIndex(max(0, idx))
            combo.currentIndexChanged.connect(lambda _i, cb=combo: self._sync_combo_tooltip(cb))
            self._sync_combo_tooltip(combo)
            grid.addWidget(combo, r, 1)
            self._role_combos[role] = combo

            if role in COORD_ROLES:
                ucombo = QComboBox()
                for disp, tok in _UNIT_CHOICES:
                    ucombo.addItem(disp, tok)
                uidx = ucombo.findData(self._units0.get(role, "nm"))
                ucombo.setCurrentIndex(max(0, uidx))
                ucombo.currentIndexChanged.connect(self._update_pixel_enabled)
                grid.addWidget(ucombo, r, 2)
                self._unit_combos[role] = ucombo
            elif role == "frame":
                tcombo = QComboBox()
                tcombo.setToolTip("Unit of the time column — 'ms' is rescaled to "
                                  "seconds; 'frames' keeps the raw index.")
                for disp, tok in _TIME_CHOICES:
                    tcombo.addItem(disp, tok)
                tidx = tcombo.findData(self._time_unit0)
                tcombo.setCurrentIndex(max(0, tidx))
                grid.addWidget(tcombo, r, 2)
                self._time_combo = tcombo
        grid.setColumnStretch(1, 1)
        root.addWidget(grp)

        # Pixel size (only relevant when a coordinate unit is 'pixel').
        px_row = QHBoxLayout()
        self._px_label = QLabel("pixel size (nm/px):")
        px_row.addWidget(self._px_label)
        self._px_spin = QDoubleSpinBox()
        self._px_spin.setRange(0.1, 100000.0)
        self._px_spin.setDecimals(2)
        self._px_spin.setValue(self._pixel0)
        px_row.addWidget(self._px_spin)
        px_row.addStretch()
        root.addLayout(px_row)

        root.addWidget(QLabel("Preview (representative rows across the file):"))
        root.addWidget(self._build_preview(), stretch=1)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        self._buttons = bb
        root.addWidget(bb)

    def _build_preview(self) -> QTableWidget:
        headers = self._table.headers
        rows = representative_row_indices(self._table.n_rows)
        table = QTableWidget(len(rows), len(headers) + 1)
        table.setHorizontalHeaderLabels(["row"] + headers)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        col_by_name = {c.name: c for c in self._table.columns}
        for vr, ri in enumerate(rows):
            item = QTableWidgetItem(str(ri + 1))            # 1-based row number
            item.setForeground(Qt.GlobalColor.gray)
            table.setItem(vr, 0, item)
            for c, h in enumerate(headers, start=1):
                col = col_by_name.get(h)
                val = ""
                if col is not None and ri < col.values.size:
                    v = col.values[ri]
                    val = "" if v != v else (f"{v:.4g}" if col.numeric else str(v))
                table.setItem(vr, c, QTableWidgetItem(val))
        table.resizeColumnsToContents()
        return table

    def _update_pixel_enabled(self) -> None:
        needs = any(cb.currentData() == "px" for cb in self._unit_combos.values())
        self._px_label.setEnabled(needs)
        self._px_spin.setEnabled(needs)

    def _stats_tooltip(self, name: str | None) -> str:
        """Human-readable value statistics for column *name* (dtype · range ·
        median step · unique count), used as a hover hint on the role dropdowns."""
        st = self._stats.get(name) if name else None
        if st is None or st.n_finite == 0:
            return ""
        parts = ["int" if st.is_integer else "float",
                 f"range [{st.vmin:.4g}, {st.vmax:.4g}]"]
        if st.median_abs_diff == st.median_abs_diff:      # not NaN
            parts.append(f"median step {st.median_abs_diff:.4g}")
        parts.append(f"{st.n_unique:,} unique")
        return " · ".join(parts)

    def _sync_combo_tooltip(self, cb: QComboBox) -> None:
        cb.setToolTip(self._stats_tooltip(cb.currentData()))

    # -------------------------------------------------------------- result
    def _current_mapping(self) -> dict[str, str | None]:
        return {role: cb.currentData() for role, cb in self._role_combos.items()}

    def _on_accept(self) -> None:
        m = self._current_mapping()
        if m.get("x") is None or m.get("y") is None:
            QMessageBox.warning(self, "Open spreadsheet",
                                "Please assign both the x and y columns.")
            return
        self.accept()

    def build_dataset(self):
        mapping = self._current_mapping()
        units = {role: cb.currentData() for role, cb in self._unit_combos.items()}
        pixel = float(self._px_spin.value()) if self._px_spin.isEnabled() else None
        time_unit = self._time_combo.currentData() if self._time_combo is not None else None
        return build_dataset_from_mapping(
            self._table, mapping, units=units, pixel_size_nm=pixel,
            time_unit=time_unit, prefs=self._prefs)


def import_spreadsheet(path, *, prefs: dict | None = None, parent=None):
    """Read *path*, pre-fill the mapping dialog from the value-based best guess,
    and build the dataset on OK. Always shows the dialog for confirmation (never a
    silent import); returns the dataset, or ``None`` if cancelled."""
    table = read_table(path)
    dlg = SpreadsheetMappingDialog(table, prefs=prefs, parent=parent)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return dlg.build_dataset()


def open_mapping_dialog(ambiguity: "AutoImportAmbiguity", parent=None):
    """Show the mapping dialog for an :class:`AutoImportAmbiguity` (from the
    headless :func:`auto_import`); return a dataset or ``None`` if cancelled."""
    dlg = SpreadsheetMappingDialog(
        ambiguity.table, ambiguity.mapping, ambiguity.units, parent=parent)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return dlg.build_dataset()
