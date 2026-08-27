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

from ..core.export_size import format_file_size
from ..core.spreadsheet_loader import (
    COORD_ROLES,
    DEFAULT_PIXEL_SIZE_NM,
    ROLES,
    AutoImportAmbiguity,
    SpreadsheetTable,
    build_dataset_from_mapping,
    guess_mapping,
    guess_time_unit,
    delimited_header_row,
    guess_units,
    is_canonical_minflux_table,
    minflux_table_kind,
    read_table,
    read_table_preview,
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
    "photons": ("photons (→ eco)", False),
    "itr": ("iteration (itr)", False),
    "vld": ("valid mask (vld)", False),
}
# Hover help for the two MINFLUX-only roles: they SELECT rows rather than adding
# an attribute, which is not guessable from the label alone.
_ROLE_TIPS: dict[str, str] = {
    "itr": ("MINFLUX iteration index. One row of a raw export is one "
            "(localization × iteration) event, so mapping this keeps only the "
            "last iteration — otherwise every iteration is imported as its own "
            "localization."),
    "vld": ("MINFLUX validity flag. Mapping this drops the invalid rows "
            "(failed probes), matching what the native loaders materialize."),
    "photons": ("Photon count per localization. In a MINFLUX export this is "
                "'eco'; it is stored under that name so the CRLB precision "
                "estimate finds it."),
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
        count_prefix = "approximately " if self._table.n_rows_is_estimate else ""
        root.addWidget(QLabel(
            f"<b>{count_prefix}{self._table.n_rows:,}</b> rows · detected tool: "
            f"<b>{self._table.detected_tool}</b>. Assign columns to roles "
            "(<b>x</b> and <b>y</b> are required); coordinate units are pre-filled."))
        if self._table.preview_only:
            size = format_file_size(Path(self._table.path).stat().st_size)
            warning = QLabel(
                f"<b>Large text table ({size}).</b> This dialog sampled only "
                f"{max((column.values.size for column in self._table.columns), default=0):,} "
                "rows from the beginning, middle and end; it did not parse the "
                "complete file. The complete table is read only after you click OK. "
                "CSV import can still take minutes; Zarr, MAT or NumPy is preferable "
                "for routine MINFLUX work."
            )
            warning.setWordWrap(True)
            warning.setStyleSheet(
                "QLabel { background: #fff4cc; border: 1px solid #d6b84b; "
                "padding: 6px; color: #4b3b00; }"
            )
            root.addWidget(warning)

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
            if role in _ROLE_TIPS:
                grid.itemAtPosition(r, 0).widget().setToolTip(_ROLE_TIPS[role])
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
        sampled = self._table.sample_row_indices
        if sampled is not None:
            rows = list(range(len(sampled)))
            displayed_rows = [int(value) for value in sampled]
        else:
            rows = representative_row_indices(self._table.n_rows)
            displayed_rows = rows
        table = QTableWidget(len(rows), len(headers) + 1)
        table.setHorizontalHeaderLabels(["row"] + headers)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        col_by_name = {c.name: c for c in self._table.columns}
        for vr, (ri, displayed_row) in enumerate(zip(rows, displayed_rows)):
            prefix = "≈" if self._table.n_rows_is_estimate and vr >= 12 else ""
            item = QTableWidgetItem(f"{prefix}{displayed_row + 1}")
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
        # The canonical MSR-reader table has richer semantics than a generic
        # spreadsheet: one row is an iteration event, and every raw field must
        # survive.  When the user accepted the untouched canonical mapping, use
        # its dedicated loader instead of reducing it to x/y/z/tid/tim.
        if (
            is_canonical_minflux_table(self._table)
            and mapping == self._mapping0
            and units == self._units0
            and time_unit == self._time_unit0
        ):
            from ..core.loader import load_csv

            return load_csv(self._table.path, prefs=self._prefs)
        table = read_table(self._table.path) if self._table.preview_only else self._table
        return build_dataset_from_mapping(
            table, mapping, units=units, pixel_size_nm=pixel,
            time_unit=time_unit, prefs=self._prefs)


#: Delimited-text suffixes ``core.loader.load_csv`` can read directly. An Excel
#: workbook cannot take the direct route however its columns are named.
_DELIMITED_SUFFIXES = frozenset({".csv", ".tsv", ".txt"})


def minflux_direct_load_kind(path) -> str | None:
    """``"raw"``/``"snapshot"`` when *path* is a MINFLUX table to load directly.

    Reads only the header row. A table this application wrote names its columns
    unambiguously, and putting it through the generic mapping actively loses
    information — the iteration axis and the validity mask have no role in a
    generic localization table — so it bypasses the confirmation dialog.
    """
    if Path(path).suffix.lower() not in _DELIMITED_SUFFIXES:
        return None
    try:
        headers = delimited_header_row(path)
    except Exception:                                    # noqa: BLE001
        return None
    return minflux_table_kind(headers) if headers else None


def import_spreadsheet(path, *, prefs: dict | None = None, parent=None, log=None,
                       apply_sidecar: bool = True):
    """Read *path* and return a dataset, or ``None`` if the user cancelled.

    A table this application wrote (recognised by :func:`minflux_table_kind`) is
    loaded straight through ``core.loader.load_csv``, which keeps every raw
    iteration field. Anything else opens the mapping dialog, pre-filled from the
    value-based best guess — never a silent generic import.
    """
    kind = minflux_direct_load_kind(path)
    if kind is not None:
        from ..core.loader import load_csv

        if log is not None:
            log(f"'{Path(path).name}': canonical MINFLUX {kind} table — "
                "loaded directly, no column mapping needed.")
        return load_csv(path, prefs=prefs, apply_sidecar=apply_sidecar)
    table = read_table_preview(path)
    dlg = SpreadsheetMappingDialog(table, prefs=prefs, parent=parent)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return dlg.build_dataset()


def open_mapping_dialog(ambiguity: AutoImportAmbiguity, parent=None):
    """Show the mapping dialog for an :class:`AutoImportAmbiguity` (from the
    headless :func:`auto_import`); return a dataset or ``None`` if cancelled."""
    dlg = SpreadsheetMappingDialog(
        ambiguity.table, ambiguity.mapping, ambiguity.units, parent=parent)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return dlg.build_dataset()
