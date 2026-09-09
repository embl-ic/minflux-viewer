"""
Spreadsheet column-mapping dialog.

Shown when :func:`minflux_viewer.core.spreadsheet_loader.auto_import` can't load
a table unattended (ambiguous columns, or camera-pixel coordinates without a
pixel size). The parameter combos, per-coordinate units, and pixel-size field
are **pre-filled** from the loader's best guess, and the preview shows a handful
of representative rows spanning the whole file.

The mapping area is progressive. It opens *simple* — the four parameters every
localization table needs (x, y, z, time/frame) — and *expands* to the rest:
precision, trace id, photons, iteration, valid mask, plus one row per raw
MINFLUX attribute (cfr, efo, dcr, …) added with *+ add parameter*. A parameter
the fold is hiding is counted in the group title, never silently dropped.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt
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
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.attributes import RAW_ATTRIBUTE_DESCRIPTIONS
from ..core.export_size import format_file_size
from ..core.spreadsheet_loader import (
    COORD_ROLES,
    DEFAULT_PIXEL_SIZE_NM,
    EXTRA_ATTR_ROLES,
    PREC_ROLES,
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

# Parameter → (display label, required?)
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
#: Shown before the mapping is expanded: what a plain localization table needs.
#: Everything else in :data:`ROLES` appears once the user expands.
SIMPLE_ROLES: tuple[str, ...] = ("x", "y", "z", "frame")
# Hover help where the label alone does not say what mapping the parameter does
# — itr / vld SELECT rows rather than adding an attribute, and a precision has a
# unit of its own. Every other parameter falls back to its attribute description.
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
    "prec_xy": ("Lateral localization precision. Its unit follows x / y unless "
                "you pick one here."),
    "prec_z": ("Axial localization precision. Its unit follows z unless you "
               "pick one here."),
}
# Display unit ↔ internal unit token.
_UNIT_CHOICES = (("nm", "nm"), ("µm", "um"), ("mm", "mm"), ("metre", "m"),
                 ("pixel", "px"))
# A precision is a length in the coordinate system, so it inherits the
# coordinate unit by default — an explicit choice overrides that.
_PREC_UNIT_CHOICES = {"prec_xy": (("as x / y", None),) + _UNIT_CHOICES,
                      "prec_z": (("as z", None),) + _UNIT_CHOICES}
# Time unit for the frame → tim column (None = frame index, kept as-is).
_TIME_CHOICES = (("frames", None), ("second", "s"), ("ms", "ms"))
_NONE = "<none>"
#: Rows of the preview the dialog opens tall enough to show. The preview holds
#: more (the head, a logarithmic spread and the last row); this is what fits
#: without scrolling, and what the dialog's opening height is measured from.
PREVIEW_ROWS = 10


def _role_label(role: str) -> str:
    """Display label for *role* — an added raw attribute is its own label."""
    return _ROLE_LABELS.get(role, (role, False))[0]


def _role_tip(role: str) -> str:
    """Hover help: the role-specific note, else the attribute's description."""
    return _ROLE_TIPS.get(role) or RAW_ATTRIBUTE_DESCRIPTIONS.get(role, "")


class _PreviewTable(QTableWidget):
    """The preview grid, asking to be :data:`PREVIEW_ROWS` rows tall.

    A scroll area's size hint is a fixed default whatever it holds, so a dialog
    sized from it opens on an arbitrary number of rows. This pins the opening
    height to a readable sample; the table still shrinks and grows with the
    dialog, and still holds more rows than it shows.
    """

    def _chrome(self) -> int:
        return (self.horizontalHeader().sizeHint().height()
                + 2 * self.frameWidth()
                + self.horizontalScrollBar().sizeHint().height())

    def sizeHint(self) -> QSize:
        row = self.verticalHeader().defaultSectionSize()
        return QSize(super().sizeHint().width(), self._chrome() + PREVIEW_ROWS * row)

    def minimumSizeHint(self) -> QSize:
        row = self.verticalHeader().defaultSectionSize()
        return QSize(super().minimumSizeHint().width(), self._chrome() + 3 * row)


class SpreadsheetMappingDialog(QDialog):
    """Map spreadsheet columns to localization parameters and build a dataset.

    Parameters, coordinate/time units, and the pixel size are **pre-filled** from
    the loader's value-based best guess (headers first, then column statistics),
    so a headerless MINFLUX-like table opens with x/y/z/tid/tim already populated
    — the user only confirms or corrects, then imports.

    The mapping grid has two depths (see :data:`SIMPLE_ROLES`) and grows by one
    row per *+ add parameter*, each row mapping a column to a further raw MINFLUX
    attribute (:data:`~minflux_viewer.core.spreadsheet_loader.EXTRA_ATTR_ROLES`).
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
        self._row_widgets: dict[str, list[QWidget]] = {}
        self._extra_rows: list[dict] = []
        self._time_combo: QComboBox | None = None
        # Opens folded: the four parameters that carry a localization table. A
        # guess the fold hides is counted in the group title (see _map_title),
        # so a pre-filled parameter is out of sight but never out of mind.
        self._advanced = False
        self._built = False

        self.setWindowTitle(f"Open spreadsheet — {Path(table.path).name}")
        self._build_ui()
        self._built = True
        self._update_pixel_enabled()
        # Exactly tall enough for the folded mapping and PREVIEW_ROWS of table.
        # The hints are measured after the rows this view hides are hidden.
        self._measure()
        self.resize(940, self.sizeHint().height())

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        count_prefix = "approximately " if self._table.n_rows_is_estimate else ""
        root.addWidget(QLabel(
            f"<b>{count_prefix}{self._table.n_rows:,}</b> rows · detected tool: "
            f"<b>{self._table.detected_tool}</b>. Assign columns to parameters "
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

        self._col_names = [c.name for c in self._table.numeric_columns()]
        root.addWidget(self._build_mapping_group())

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
        self._preview = self._build_preview()
        root.addWidget(self._preview, stretch=1)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        self._buttons = bb
        root.addWidget(bb)

    def _build_mapping_group(self) -> QGroupBox:
        """The mapping grid plus its *+ add parameter* / expand-fold controls."""
        self._map_group = QGroupBox("Column mapping")
        outer = QVBoxLayout(self._map_group)
        self._grid = QGridLayout()
        outer.addLayout(self._grid)
        self._grid.addWidget(QLabel("<b>parameter</b>"), 0, 0)
        self._grid.addWidget(QLabel("<b>column</b>"), 0, 1)
        self._grid.addWidget(QLabel("<b>unit</b>"), 0, 2)
        # The controls exist before the rows do: adding a row re-reads them.
        self._add_btn = QPushButton("+ add parameter")
        self._add_btn.setToolTip(
            "Map a column to a further raw MINFLUX attribute, imported under "
            "its canonical name.")
        self._add_btn.clicked.connect(lambda: self._add_extra_row())
        self._toggle_btn = QToolButton()
        self._toggle_btn.setAutoRaise(True)
        self._toggle_btn.clicked.connect(self._toggle_advanced)

        self._grid_row = 1
        for role in ROLES:
            self._add_role_row(role)
        # A raw attribute the guess already matched (a 'dcr' column, say) gets a
        # row of its own, so the import is never silently richer than the dialog.
        for role in EXTRA_ATTR_ROLES:
            if self._mapping0.get(role):
                self._add_extra_row(role)
        self._grid.setColumnStretch(1, 1)

        controls = QHBoxLayout()
        controls.addWidget(self._add_btn)
        controls.addStretch()
        controls.addWidget(self._toggle_btn)
        outer.addLayout(controls)
        self._apply_mode()
        return self._map_group

    def _column_combo(self, chosen: str | None) -> QComboBox:
        """A column dropdown over the numeric columns, pre-set to *chosen*."""
        combo = QComboBox()
        combo.addItem(_NONE, None)
        for i, name in enumerate(self._col_names, start=1):
            combo.addItem(name, name)
            combo.setItemData(i, self._stats_tooltip(name), Qt.ItemDataRole.ToolTipRole)
        idx = combo.findData(chosen) if chosen else 0
        combo.setCurrentIndex(max(0, idx))
        combo.currentIndexChanged.connect(lambda _i, cb=combo: self._sync_combo_tooltip(cb))
        self._sync_combo_tooltip(combo)
        return combo

    def _add_role_row(self, role: str) -> None:
        """One fixed row of the grid: label · column · unit (where it has one)."""
        r, self._grid_row = self._grid_row, self._grid_row + 1
        label, required = _ROLE_LABELS[role]
        name = QLabel(label + (" *" if required else ""))
        if _role_tip(role):
            name.setToolTip(_role_tip(role))
        self._grid.addWidget(name, r, 0)

        combo = self._column_combo(self._mapping0.get(role))
        self._grid.addWidget(combo, r, 1)
        self._role_combos[role] = combo
        widgets: list[QWidget] = [name, combo]

        unit: QComboBox | None = None
        if role in COORD_ROLES:
            unit = QComboBox()
            for disp, tok in _UNIT_CHOICES:
                unit.addItem(disp, tok)
            unit.setCurrentIndex(max(0, unit.findData(self._units0.get(role, "nm"))))
            unit.currentIndexChanged.connect(self._update_pixel_enabled)
            self._unit_combos[role] = unit
        elif role in PREC_ROLES:
            unit = QComboBox()
            choices = _PREC_UNIT_CHOICES[role]
            unit.setToolTip(f"Unit of the precision column — '{choices[0][0]}' "
                            "takes the coordinate unit, which is what most "
                            "tools export.")
            for disp, tok in choices:
                unit.addItem(disp, tok)
            unit.currentIndexChanged.connect(self._update_pixel_enabled)
            self._unit_combos[role] = unit
        elif role == "frame":
            unit = QComboBox()
            unit.setToolTip("Unit of the time column — 'ms' is rescaled to "
                            "seconds; 'frames' keeps the raw index.")
            for disp, tok in _TIME_CHOICES:
                unit.addItem(disp, tok)
            unit.setCurrentIndex(max(0, unit.findData(self._time_unit0)))
            self._time_combo = unit
        if unit is not None:
            self._grid.addWidget(unit, r, 2)
            widgets.append(unit)
        self._row_widgets[role] = widgets

    # ------------------------------------------------- added parameter rows
    def _available_extra_roles(self, exclude: dict | None = None) -> list[str]:
        """Raw attributes not already in the mapping, in canonical order."""
        taken = {e["role"] for e in self._extra_rows if e is not exclude}
        return [role for role in EXTRA_ATTR_ROLES if role not in taken]

    def _add_extra_row(self, role: str | None = None) -> None:
        """Append a row mapping a column to a further raw MINFLUX attribute."""
        available = self._available_extra_roles()
        role = role or (available[0] if available else None)
        if role is None:
            return
        before = self._measure()
        r, self._grid_row = self._grid_row, self._grid_row + 1
        param = QComboBox()
        column = self._column_combo(self._mapping0.get(role))
        remove = QToolButton()
        remove.setText("✕")
        remove.setAutoRaise(True)
        remove.setToolTip("Remove this parameter from the mapping.")
        entry = {"role": role, "param": param, "column": column,
                 "widgets": [param, column, remove]}
        self._extra_rows.append(entry)
        self._role_combos[role] = column
        self._grid.addWidget(param, r, 0)
        self._grid.addWidget(column, r, 1)
        self._grid.addWidget(remove, r, 3)
        param.currentIndexChanged.connect(
            lambda _i, e=entry: self._on_extra_role_changed(e))
        remove.clicked.connect(lambda _c=False, e=entry: self._remove_extra_row(e))
        self._refresh_extra_choices()
        for widget in entry["widgets"]:
            widget.setVisible(self._advanced)
        self._grow(before)

    def _refresh_extra_choices(self) -> None:
        """Re-fill every added row's parameter dropdown with what is still free
        (its own choice included), and disable *+ add* once nothing is."""
        for entry in self._extra_rows:
            param: QComboBox = entry["param"]
            choices = self._available_extra_roles(exclude=entry)
            blocked = param.blockSignals(True)
            param.clear()
            for role in choices:
                param.addItem(_role_label(role), role)
                if _role_tip(role):
                    param.setItemData(param.count() - 1, _role_tip(role),
                                      Qt.ItemDataRole.ToolTipRole)
            param.setCurrentIndex(max(0, param.findData(entry["role"])))
            param.blockSignals(blocked)
            param.setToolTip(_role_tip(entry["role"]))
        self._add_btn.setEnabled(bool(self._available_extra_roles()))

    def _on_extra_role_changed(self, entry: dict) -> None:
        role = entry["param"].currentData()
        if role is None or role == entry["role"]:
            return
        self._role_combos.pop(entry["role"], None)
        entry["role"] = role
        self._role_combos[role] = entry["column"]
        self._refresh_extra_choices()

    def _remove_extra_row(self, entry: dict) -> None:
        before = self._measure()
        self._extra_rows.remove(entry)
        self._role_combos.pop(entry["role"], None)
        for widget in entry["widgets"]:
            self._grid.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        self._refresh_extra_choices()
        self._grow(before)

    # ------------------------------------------------------- simple / expanded
    def _apply_mode(self) -> None:
        """Show the rows the current depth calls for, and label the toggle."""
        for role, widgets in self._row_widgets.items():
            visible = self._advanced or role in SIMPLE_ROLES
            for widget in widgets:
                widget.setVisible(visible)
        for entry in self._extra_rows:
            for widget in entry["widgets"]:
                widget.setVisible(self._advanced)
        self._add_btn.setVisible(self._advanced)
        self._toggle_btn.setText("▲ fold" if self._advanced else "▼ expand")
        self._toggle_btn.setToolTip(
            "Hide the parameters beyond x / y / z / time."
            if self._advanced else
            "Show precision, trace id, photons, iteration, the valid mask and "
            "any added attribute.")
        self._map_group.setTitle(self._map_title())

    def _map_title(self) -> str:
        """Title that owns up to mappings the folded view is hiding."""
        if self._advanced:
            return "Column mapping"
        hidden = sum(1 for role, combo in self._role_combos.items()
                     if role not in SIMPLE_ROLES and combo.currentData())
        return ("Column mapping" if not hidden else
                f"Column mapping — {hidden} more mapped, expand to see")

    def _toggle_advanced(self) -> None:
        before = self._measure()
        self._advanced = not self._advanced
        self._apply_mode()
        self._grow(before)

    def _measure(self) -> tuple[int, int]:
        """``(mapping-group height, dialog height)`` with the layouts re-measured.

        The cached hints are stale the instant a row is shown, hidden or added,
        and re-activating the dialog layout can already have resized the window
        to its new minimum — so both numbers must be read together.
        """
        for layout in (self._map_group.layout(), self.layout()):
            if layout is not None:
                layout.invalidate()
                layout.activate()
        return self._map_group.sizeHint().height(), self.height()

    def _grow(self, before: tuple[int, int]) -> None:
        """Give the mapping group the height it just gained (or hand it back),
        so expanding never eats the preview and folding never leaves a gap."""
        if not self._built:
            return
        map_before, dialog_before = before
        delta = self._measure()[0] - map_before
        if delta:
            # resize() is clamped by the layout's own minimum for a top-level
            # window, so this only ever asks; Qt keeps the dialog usable.
            self.resize(self.width(), dialog_before + delta)

    def _build_preview(self) -> QTableWidget:
        headers = self._table.headers
        sampled = self._table.sample_row_indices
        if sampled is not None:
            rows = list(range(len(sampled)))
            displayed_rows = [int(value) for value in sampled]
        else:
            rows = representative_row_indices(self._table.n_rows)
            displayed_rows = rows
        table = _PreviewTable(len(rows), len(headers) + 1)
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
        median step · unique count), used as a hover hint on the column dropdowns."""
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
        return build_spreadsheet_dataset(
            self._table, self.dataset_build_spec(), prefs=self._prefs
        )

    def dataset_build_spec(self) -> dict:
        """Capture widget choices as plain data safe to use on a worker thread."""
        mapping = self._current_mapping()
        # A precision left on 'as x / y' contributes no unit: the build inherits
        # the coordinate unit, which is what the loader did before this control.
        units = {role: combo.currentData()
                 for role, combo in self._unit_combos.items()
                 if combo.currentData() is not None}
        pixel = float(self._px_spin.value()) if self._px_spin.isEnabled() else None
        time_unit = (
            self._time_combo.currentData() if self._time_combo is not None else None
        )
        # The canonical MSR-reader table has richer semantics than a generic
        # spreadsheet: one row is an iteration event, and every raw field must
        # survive.  When the user accepted the untouched canonical mapping, use
        # its dedicated loader instead of reducing it to x/y/z/tid/tim.
        def mapped(m: dict) -> dict:
            return {role: name for role, name in m.items() if name}

        canonical = (
            is_canonical_minflux_table(self._table)
            and mapped(mapping) == mapped(self._mapping0)
            and all(units.get(a) == self._units0.get(a) for a in COORD_ROLES)
            and not any(role in units for role in PREC_ROLES)
            and time_unit == self._time_unit0
        )
        return {
            "mapping": mapping,
            "units": units,
            "pixel_size_nm": pixel,
            "time_unit": time_unit,
            "canonical": canonical,
        }


def build_spreadsheet_dataset(
    table: SpreadsheetTable,
    spec: dict,
    *,
    prefs: dict | None = None,
    apply_sidecar: bool = True,
):
    """Build from a dialog snapshot without reading any Qt widget state."""
    if spec.get("canonical"):
        from ..core.loader import load_csv

        return load_csv(
            table.path, prefs=prefs, apply_sidecar=apply_sidecar
        )
    complete = read_table(table.path) if table.preview_only else table
    return build_dataset_from_mapping(
        complete,
        dict(spec.get("mapping") or {}),
        units=dict(spec.get("units") or {}),
        pixel_size_nm=spec.get("pixel_size_nm"),
        time_unit=spec.get("time_unit"),
        prefs=prefs,
    )


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
