"""
Save / Export dialog.

Most decisions are pre-configured in **Preferences > Data > "When saving/exporting
data file:"** (enabled formats, data content, include flags, filter handling), so
this dialog stays short: it asks for the **path + format**, the **content** (only
when both raw and snapshot are enabled), and exposes the rest under **More
options…** for one-off overrides.

Data content:
- **Raw canonical** — the untouched all-iteration ``mfx`` (reloads + re-applies the
  recipe). For a file-backed dataset the raw is already on disk, so this offers a
  "recipe sidecar only" shortcut.
- **Processed snapshot** — the current view with Z scaling factor/transform/filter baked in.

**.csv means the custom column table here**, matching *File > Save As > Custom
table*: choose the attributes and name the columns. The canonical all-iteration
table -- the one the MSR reader writes and the one that reloads without the
column-mapping dialog -- is still reachable, as a checkbox under *More options*,
because it is the only CSV that can carry the iteration axis and the validity
mask. See :func:`options` (``csv_mode``).

The wording of the option labels is deliberately plain English rather than the
internal vocabulary; :mod:`minflux_viewer.ui.preferences_dialog` shows the same
options and must use the same words.

The writing is done by :func:`minflux_viewer.core.save.save_processed`.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import formats as _formats
from ..core.save import METADATA_SUFFIX

# All derived from the one registry (:mod:`minflux_viewer.core.formats`); this
# dialog used to keep its own copy of every table, which had to be hand-synced
# with core.save, Preferences and the router each time a format changed.
_FORMAT_LABELS = {spec.key: spec.label for spec in _formats.save_formats()}
#: ``.csv`` in THIS dialog is the custom column picker, not the canonical
#: writer the registry label names (which the MSR reader and the More-options
#: checkbox use). Overridden here only, so the registry stays truthful for them.
_FORMAT_LABELS["csv"] = "Custom table (.csv)"
_EXT = {spec.key: spec.extensions[0] for spec in _formats.save_formats()}
#: A Qt name filter is the label with a glob, e.g. "NumPy (*.npy)".
_FILTERS = {key: label.replace("(.", "(*.")
            for key, label in _FORMAT_LABELS.items()}
#: Every writable key (scripting, tests). The dropdown lists only the offered
#: ones -- ``zarr_zip`` keeps its writer but has left the menus.
_ALL_FORMATS = [spec.key for spec in _formats.save_formats()]
_OFFERED_FORMATS = [spec.key for spec in _formats.offered_save_formats()]

#: Not a data format, and deliberately not a :class:`FormatSpec`: it writes the
#: processing *about* a dataset, so content / attribute / filter choices do not
#: apply to it and ``save_processed`` never sees it. The dialog offers it last
#: and the caller routes it to ``MainWindow.save_metadata_only``.
METADATA_ONLY = "metadata_only"
_METADATA_ONLY_LABEL = "Viewer metadata only (.json)"

# --- option wording --------------------------------------------------------
# Plain English, because these are questions about the user's data, not about
# the implementation. Kept as constants so _sync_location (which rewrites two of
# them for Zarr) and Preferences cannot drift from the dialog.
LBL_ATTRS = "Include attribute columns (efo, cfr, dcr, tid, tim …), not only coordinates"
TIP_ATTRS = (
    "The measured per-localization values written beside the x/y/z coordinates: "
    "efo, cfr, dcr, eco, ecc, efc, fbg, tid, tim and the rest.\n\n"
    "Unticked, a processed snapshot holds coordinates only. This affects the "
    "processed snapshot alone — original acquisition data always carries "
    "everything the file contained."
)
LBL_DERIVED = ("Also save viewer-computed attributes "
               "(density, trace size, speed …) as fixed values")
TIP_DERIVED = (
    "Values this application computed rather than the microscope measuring "
    "them: den (local density — which depends on the radius and method you "
    "chose), siz, dst, spd, dt, dur, len, tim_trace.\n\n"
    "They are normally recomputed when the file is reopened, so they can come "
    "back different. Tick this to write today's numbers as data."
)
LBL_RECIPE = ("Save processing metadata beside the data file "
              f"(…_{METADATA_SUFFIX}.json)")
TIP_RECIPE = (
    "A small JSON file written next to the data, recording the processing this "
    "dataset carries: Z scaling factor, transforms, filters, ROIs and "
    "acquisition time.\n\n"
    "Opening the data file again offers to restore them. Without it the data "
    "loads exactly as saved, with no record of how it was produced."
)
LBL_FILTER = "Filtered-out localizations:"
LBL_FILTER_FLAG = "keep, marked in an 'ftr' column"
LBL_FILTER_APPLY = "remove from the file"
TIP_FILTER = (
    "What happens to localizations the active filter hides. Keeping them adds "
    "a boolean 'ftr' column so the filter can be undone later; removing them "
    "makes the file smaller and final.\n\n"
    "Processed snapshot only."
)
LBL_CSV_CANONICAL = "Write the canonical table instead (all iterations, reloads directly)"
TIP_CSV_CANONICAL = (
    "The custom table is the current view: filtered, last valid iteration, one "
    "row per localization, with the columns and headers you choose.\n\n"
    "The canonical table is the whole mfx node — every iteration and every "
    "validity state, with fixed column names (loc_x/loc_y/loc_z, itr, vld, "
    "dcr_0/dcr_1 …). It is the only CSV that can carry the iteration axis, and "
    "it reopens without the column-mapping dialog. This is what the MSR "
    "reader's .csv export writes."
)
#: Formats that only carry the canonical raw data (no processed snapshot).
_RAW_ONLY_FORMATS = _formats.raw_only_formats()
# A file-backed dataset can skip re-writing raw only for reloadable raw formats.
_RELOADABLE_RAW_EXT = (".mat", ".npy", ".json")


class SaveProcessedDataDialog(QDialog):
    """Collect where/how to save, defaulting from the export preferences."""

    def __init__(
        self,
        dataset_name: str = "",
        *,
        members=None,
        file_backed: bool = False,
        source_path: str | Path | None = None,
        default_dir: str | Path | None = None,
        prefs: dict | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save / export data")
        self.setMinimumWidth(520)

        self._file_backed = bool(file_backed)
        # Every channel of the overlay this dataset belongs to, or just it. The
        # one-dataset-per-file formats write a group as several files, so the
        # dialog has to name each of them and the folder they share.
        self._members = [str(name or "") for name in (members or [])]
        self._is_group = len(self._members) > 1
        self._chosen_path: Path | None = None
        # Zarr forces "derived" and "recipe" on (it stores both internally).
        # These remember what the user had, so switching away from Zarr restores
        # it instead of silently leaving the forced values behind.
        self._zarr_forced = False
        self._pref_derived = True
        self._pref_recipe = True
        self._default_dir = Path(default_dir or Path.home())
        data_prefs = (prefs or {}).get("data", {}) if prefs else {}

        # Enabled formats, in registry order (fall back to every offered one so
        # saving is never locked out by a stale preference).
        enabled = [k for k in _OFFERED_FORMATS
                   if k in set(data_prefs.get("export_formats", _OFFERED_FORMATS))]
        self._formats = enabled or list(_OFFERED_FORMATS)
        # content choices
        content = list(data_prefs.get("export_content", ["raw", "snapshot"])) or \
            ["raw", "snapshot"]
        self._contents = [c for c in ("raw", "snapshot") if c in content] or \
            ["raw", "snapshot"]

        root = QVBoxLayout(self)
        if self._is_group:
            root.addWidget(QLabel(
                f"Dataset: <i>{len(self._members)} channels of one overlay</i>"))
            for member in self._members:
                item = QLabel(f"        <b>{member}</b>")
                item.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                root.addWidget(item)
        elif dataset_name:
            root.addWidget(QLabel(f"Dataset: <b>{dataset_name}</b>"))

        # ── content ──────────────────────────────────────────────────
        content_row = QHBoxLayout()
        content_row.addWidget(QLabel("Content:"))
        self._content_combo = QComboBox()
        _labels = {"raw": "Raw canonical (all iterations)",
                   "snapshot": "Processed snapshot (current view)"}
        for c in self._contents:
            self._content_combo.addItem(_labels[c], c)
        self._content_combo.currentIndexChanged.connect(lambda *_: self._refresh())
        content_row.addWidget(self._content_combo, 1)
        root.addLayout(content_row)
        # Hide the row entirely when there's only one content choice.
        if len(self._contents) < 2:
            self._content_combo.parentWidget()  # no-op; keep ref
            for i in range(content_row.count()):
                w = content_row.itemAt(i).widget()
                if w is not None:
                    w.setVisible(False)

        # ── file-backed raw shortcut ─────────────────────────────────
        src = Path(source_path) if source_path else None
        self._recipe_only = QCheckBox("raw data already on disk — write recipe sidecar only")
        self._recipe_only.setChecked(True)
        self._recipe_only.toggled.connect(lambda *_: self._refresh())
        if self._file_backed and src is not None:
            note = QLabel(f"Source: <b>{src}</b>")
            note.setWordWrap(True)
            note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            root.addWidget(note)
            root.addWidget(self._recipe_only)

        # ── name / format / location ─────────────────────────────────
        self._path_box = QWidget()
        pbox = QVBoxLayout(self._path_box)
        pbox.setContentsMargins(0, 0, 0, 0)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        stem = Path(dataset_name or "dataset").stem or "dataset"
        self._name = QLineEdit(f"{stem}_export")
        # A group writes one file per channel, so the single Name field becomes
        # one per channel, pre-filled with the dataset's own name -- the file is
        # then recognisable as that channel rather than as "…_export_2".
        self._member_names: list[QLineEdit] = []
        if self._is_group:
            self._name.setVisible(False)
            first = QLineEdit(self._members[0])
            self._member_names.append(first)
            name_row.addWidget(first, 1)
        else:
            name_row.addWidget(self._name, 1)
        self._format = QComboBox()
        for key in self._formats:
            self._format.addItem(_FORMAT_LABELS[key], key)
        self._format.insertSeparator(self._format.count())
        self._format.addItem(_METADATA_ONLY_LABEL, METADATA_ONLY)
        self._format.currentIndexChanged.connect(lambda *_: self._sync_location())
        name_row.addWidget(self._format)
        pbox.addLayout(name_row)
        for member in self._members[1:]:
            extra_row = QHBoxLayout()
            spacer = QLabel("")
            spacer.setFixedWidth(self._name.fontMetrics().horizontalAdvance("Name:"))
            extra_row.addWidget(spacer)
            field = QLineEdit(member)
            self._member_names.append(field)
            extra_row.addWidget(field, 1)
            pbox.addLayout(extra_row)

        browse_row = QHBoxLayout()
        self._loc_lbl = QLabel(f"Location: {self._default_dir}")
        self._loc_lbl.setWordWrap(True)
        # A group goes into a folder of its own, and the folder is the user's to
        # name -- there is no overlay "number" worth inventing one from.
        self._folder_edit = QLineEdit(str(self._default_dir))
        self._folder_edit.setToolTip(
            "The folder the channels are written into. It is created if it does "
            "not exist; each channel becomes one data file plus its metadata.")
        browse_row.addWidget(QLabel("Location:"))
        browse_row.addWidget(self._folder_edit, 1)
        browse_row.addWidget(self._loc_lbl, 1)
        browse = QPushButton("Save as…")
        browse.clicked.connect(self._on_browse)
        browse_row.addWidget(browse)
        pbox.addLayout(browse_row)
        self._folder_edit.setVisible(self._is_group)
        self._loc_lbl.setVisible(not self._is_group)
        self._msr_note = QLabel(
            "<span style='color:gray'>.msr uses a custom writer — reopens in this "
            "viewer via the MSR reader; may not open in Abberior Imspector. Saves "
            "raw canonical data.</span>")
        self._msr_note.setWordWrap(True)
        self._msr_note.setVisible(False)
        pbox.addWidget(self._msr_note)
        root.addWidget(self._path_box)

        # ── More options… ────────────────────────────────────────────
        self._more_btn = QPushButton("More options ▸")
        self._more_btn.setCheckable(True)
        self._more_btn.setFlat(True)
        self._more_btn.toggled.connect(self._toggle_more)
        root.addWidget(self._more_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        self._more_box = QGroupBox()
        more = QVBoxLayout(self._more_box)
        self._inc_attrs = QCheckBox(LBL_ATTRS)
        self._inc_attrs.setToolTip(TIP_ATTRS)
        self._inc_attrs.setChecked(bool(data_prefs.get("export_include_attrs", True)))
        self._inc_derived = QCheckBox(LBL_DERIVED)
        self._inc_derived.setToolTip(TIP_DERIVED)
        self._inc_derived.setChecked(bool(data_prefs.get("export_include_derived", False)))
        self._inc_recipe = QCheckBox(LBL_RECIPE)
        self._inc_recipe.setToolTip(TIP_RECIPE)
        self._inc_recipe.setChecked(bool(data_prefs.get("export_include_recipe", True)))
        more.addWidget(self._inc_attrs)
        more.addWidget(self._inc_derived)
        more.addWidget(self._inc_recipe)
        # The one way to the all-iteration canonical CSV from the viewer; the
        # menus offer only the custom column picker under this extension.
        self._csv_canonical = QCheckBox(LBL_CSV_CANONICAL)
        self._csv_canonical.setToolTip(TIP_CSV_CANONICAL)
        self._csv_canonical.toggled.connect(lambda *_: self._refresh())
        more.addWidget(self._csv_canonical)
        filt_row = QHBoxLayout()
        self._filter_lbl = QLabel(LBL_FILTER)
        filt_row.addWidget(self._filter_lbl)
        self._filter_mode = QComboBox()
        self._filter_mode.setToolTip(TIP_FILTER)
        self._filter_mode.addItem(LBL_FILTER_FLAG, "flag")
        self._filter_mode.addItem(LBL_FILTER_APPLY, "apply")
        i = self._filter_mode.findData(data_prefs.get("export_filter_mode", "flag"))
        if i >= 0:
            self._filter_mode.setCurrentIndex(i)
        filt_row.addWidget(self._filter_mode, 1)
        more.addLayout(filt_row)
        self._more_box.setVisible(False)
        root.addWidget(self._more_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._sync_location()
        self._refresh()

    # ------------------------------------------------------------------
    def _current_content(self) -> str:
        return self._content_combo.currentData() or self._contents[0]

    def _is_recipe_only(self) -> bool:
        """Metadata-only: file-backed, raw content, shortcut checked."""
        return (self._file_backed and self._current_content() == "raw"
                and self._recipe_only.isChecked())

    def is_custom_csv(self) -> bool:
        """True when .csv means "pick the columns" rather than the canonical table."""
        canonical = getattr(self, "_csv_canonical", None)
        return (self._format.currentData() == "csv"
                and canonical is not None and not canonical.isChecked())

    def _refresh(self) -> None:
        # The recipe-only shortcut only applies to raw content on a file-backed ds.
        show_shortcut = self._file_backed and self._current_content() == "raw"
        self._recipe_only.setVisible(show_shortcut)
        self._path_box.setVisible(not self._is_recipe_only())
        self._sync_csv_options()
        self.adjustSize()

    def is_metadata_only(self) -> bool:
        """True when the dialog will write the recipe and no data file."""
        return self._format.currentData() == METADATA_ONLY

    def _sync_csv_options(self) -> None:
        """A custom table decides its own columns, so the snapshot options that
        would otherwise describe them do not apply."""
        canonical = getattr(self, "_csv_canonical", None)
        if canonical is None:                      # still constructing
            return
        fmt = self._format.currentData()
        canonical.setVisible(fmt == "csv")
        if fmt == METADATA_ONLY:
            # No data file is written, so none of these describe anything.
            for widget in (self._inc_attrs, self._inc_derived, self._filter_mode,
                           self._filter_lbl, self._content_combo, self._name):
                widget.setEnabled(False)
            self._inc_recipe.setChecked(True)
            self._inc_recipe.setEnabled(False)
            return
        self._name.setEnabled(True)
        custom = self.is_custom_csv()
        self._inc_attrs.setEnabled(not custom)
        self._filter_mode.setEnabled(not custom)
        self._filter_lbl.setEnabled(not custom)
        # ``derived`` belongs to _sync_location while Zarr is selected (it is
        # stored inside the store, so it is forced on there); this must not undo
        # that. Zarr is never .csv, so ``custom`` is False in that branch anyway.
        if fmt != "zarr":
            self._inc_derived.setEnabled(not custom)
        if fmt == "csv":
            self._content_combo.setEnabled(
                not custom and self._content_combo.count() > 1)

    def _toggle_more(self, on: bool) -> None:
        self._more_btn.setText("More options ▾" if on else "More options ▸")
        self._more_box.setVisible(on)
        self.adjustSize()

    def _sync_location(self) -> None:
        fmt = self._format.currentData()
        if fmt is None or fmt == METADATA_ONLY:
            # No data file, so nothing here has an extension or a location; the
            # separator row (data None) lands here too when Qt walks the list.
            self._sync_csv_options()
            return
        ext = _EXT[fmt]
        if self._chosen_path is not None:
            self._loc_lbl.setText(f"Location: {self._chosen_path.with_suffix(ext).parent}")
        # .msr and the self-contained Zarr schema retain canonical raw data;
        # processing is separate state inside Zarr rather than baked coordinates.
        note = getattr(self, "_msr_note", None)
        if note is not None:
            is_raw_only = fmt in _RAW_ONLY_FORMATS
            note.setVisible(is_raw_only)
            if fmt == "zarr":
                note.setText(
                    "<span style='color:gray'>Self-contained MINFLUX Viewer "
                    "Zarr v2: an active overlay is saved with all channels, "
                    "transforms and LUTs; viewer ROIs and linked MSR images are "
                    "included. An existing store can update processing only after "
                    "raw-data verification. No sibling metadata JSON is written.</span>"
                )
            elif fmt == "msr":
                note.setText(
                    "<span style='color:gray'>.msr uses a custom writer — "
                    "reopens in this viewer via the MSR reader; may not open in "
                    "Abberior Imspector. Saves raw canonical data.</span>"
                )
            if is_raw_only:
                i = self._content_combo.findData("raw")
                if i >= 0:
                    self._content_combo.setCurrentIndex(i)
                self._content_combo.setEnabled(False)
            else:
                self._content_combo.setEnabled(self._content_combo.count() > 1)
        recipe = getattr(self, "_inc_recipe", None)
        derived = getattr(self, "_inc_derived", None)
        if recipe is not None and derived is not None:
            if fmt == "zarr":
                if not self._zarr_forced:          # remember before overriding
                    self._pref_recipe = recipe.isChecked()
                    self._pref_derived = derived.isChecked()
                    self._zarr_forced = True
                recipe.setText("Processing metadata is stored inside the Zarr dataset")
                recipe.setChecked(True)
                recipe.setEnabled(False)
                derived.setText("Derived attributes are stored inside the Zarr dataset")
                derived.setChecked(True)
                derived.setEnabled(False)
            else:
                if self._zarr_forced:
                    recipe.setChecked(self._pref_recipe)
                    derived.setChecked(self._pref_derived)
                    self._zarr_forced = False
                recipe.setText(LBL_RECIPE)
                recipe.setEnabled(True)
                derived.setText(LBL_DERIVED)
                derived.setEnabled(True)
        self._sync_csv_options()

    def _on_browse(self) -> None:
        fmt = self._format.currentData()
        ext = _EXT[fmt]
        suggested = str(self._default_dir / f"{self._name.text() or 'dataset'}{ext}")
        if fmt == "zarr":
            from .zarr_save_dialog import choose_zarr_save_path

            p = choose_zarr_save_path(self, "Save / export data", suggested)
            if p is None:
                return
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save / export data", suggested, _FILTERS[fmt]
            )
            if not path:
                return
            p = Path(path)
        self._chosen_path = p
        self._name.setText(p.stem)
        self._loc_lbl.setText(f"Location: {p.parent}")
        key = p.suffix.lower().lstrip(".")
        i = self._format.findData(key)
        if i >= 0:
            self._format.setCurrentIndex(i)

    # ------------------------------------------------------------------
    def options(self) -> dict:
        """Return the full save options for :func:`core.save.save_processed`."""
        include = {
            "attrs": self._inc_attrs.isChecked(),
            "derived": self._inc_derived.isChecked(),
            "recipe": self._inc_recipe.isChecked(),
        }
        filter_mode = self._filter_mode.currentData() or "flag"
        content = self._current_content()
        if self._is_recipe_only():
            return {"data_path": None, "fmt": None, "content": "raw",
                    "include": {**include, "recipe": True}, "filter_mode": filter_mode}
        fmt = self._format.currentData()
        if fmt == METADATA_ONLY:
            # The path is asked for separately: a dataset with no file behind it
            # has no folder to be beside, and this dialog's Name/Location row is
            # about a data file that will not be written.
            return {"data_path": None, "fmt": METADATA_ONLY, "content": "raw",
                    "include": include, "filter_mode": filter_mode,
                    "csv_mode": None}
        if fmt in _RAW_ONLY_FORMATS:
            content = "raw"
        ext = _EXT[fmt]
        if self._chosen_path is not None:
            data_path = self._chosen_path.with_suffix(ext)
        else:
            name = self._name.text().strip() or "dataset"
            data_path = self._default_dir / f"{name}{ext}"
        if self._is_group:
            # A group is a folder plus one stem per channel, not a single path;
            # the caller writes it through core.overlay_save.save_overlay_group.
            return {"data_path": None, "fmt": fmt, "content": content,
                    "include": include, "filter_mode": filter_mode,
                    "csv_mode": ("custom" if self.is_custom_csv()
                                 else ("canonical" if fmt == "csv" else None)),
                    "group_folder": Path(self._folder_edit.text().strip()
                                         or self._default_dir),
                    "group_names": [field.text().strip() or name
                                    for field, name in zip(self._member_names,
                                                           self._members)]}
        # ``csv_mode`` is the caller's cue to open the column picker instead of
        # calling save_processed: a custom table is written by a different
        # function (core.save.write_spreadsheet_csv) and takes no content /
        # include / filter arguments.
        csv_mode = ("custom" if self.is_custom_csv()
                    else ("canonical" if fmt == "csv" else None))
        return {"data_path": data_path, "fmt": fmt, "content": content,
                "include": include, "filter_mode": filter_mode,
                "csv_mode": csv_mode}
