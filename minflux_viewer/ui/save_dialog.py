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
- **Processed snapshot** — the current view with RIMF/transform/filter baked in.

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

# (extension label, format key)
_FORMAT_LABELS = {
    "mat": "MATLAB (.mat)", "npy": "NumPy (.npy)", "npz": "NumPy zip (.npz)",
    "json": "JSON (.json)", "csv": "CSV (.csv)", "zarr": "Zarr (.zarr)",
    "msr": "MINFLUX (.msr)",
}
_EXT = {"mat": ".mat", "npy": ".npy", "npz": ".npz", "json": ".json",
        "csv": ".csv", "zarr": ".zarr", "msr": ".msr"}
_FILTERS = {
    "mat": "MATLAB (*.mat)", "npy": "NumPy (*.npy)", "npz": "NumPy zip (*.npz)",
    "json": "JSON (*.json)", "csv": "CSV (*.csv)", "zarr": "Zarr (*.zarr)",
    "msr": "MINFLUX (*.msr)",
}
_ALL_FORMATS = ["mat", "npy", "npz", "json", "csv", "zarr", "msr"]
#: Formats that only carry the canonical raw data (no processed snapshot).
_RAW_ONLY_FORMATS = {"msr"}
# A file-backed dataset can skip re-writing raw only for reloadable raw formats.
_RELOADABLE_RAW_EXT = (".mat", ".npy", ".json")


class SaveProcessedDataDialog(QDialog):
    """Collect where/how to save, defaulting from the export preferences."""

    def __init__(
        self,
        dataset_name: str = "",
        *,
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
        self._chosen_path: Path | None = None
        self._default_dir = Path(default_dir or Path.home())
        data_prefs = (prefs or {}).get("data", {}) if prefs else {}

        # enabled formats (fall back to all so saving is never locked out)
        enabled = [k for k in _ALL_FORMATS
                   if k in set(data_prefs.get("export_formats", _ALL_FORMATS))]
        self._formats = enabled or list(_ALL_FORMATS)
        # content choices
        content = list(data_prefs.get("export_content", ["raw", "snapshot"])) or \
            ["raw", "snapshot"]
        self._contents = [c for c in ("raw", "snapshot") if c in content] or \
            ["raw", "snapshot"]

        root = QVBoxLayout(self)
        if dataset_name:
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
        name_row.addWidget(self._name, 1)
        self._format = QComboBox()
        for key in self._formats:
            self._format.addItem(_FORMAT_LABELS[key], key)
        self._format.currentIndexChanged.connect(lambda *_: self._sync_location())
        name_row.addWidget(self._format)
        pbox.addLayout(name_row)

        browse_row = QHBoxLayout()
        self._loc_lbl = QLabel(f"Location: {self._default_dir}")
        self._loc_lbl.setWordWrap(True)
        browse_row.addWidget(self._loc_lbl, 1)
        browse = QPushButton("Save as…")
        browse.clicked.connect(self._on_browse)
        browse_row.addWidget(browse)
        pbox.addLayout(browse_row)
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
        self._inc_attrs = QCheckBox("include properties & attributes")
        self._inc_attrs.setChecked(bool(data_prefs.get("export_include_attrs", True)))
        self._inc_derived = QCheckBox("freeze derived attributes")
        self._inc_derived.setChecked(bool(data_prefs.get("export_include_derived", False)))
        self._inc_recipe = QCheckBox("write recipe sidecar")
        self._inc_recipe.setChecked(bool(data_prefs.get("export_include_recipe", True)))
        more.addWidget(self._inc_attrs)
        more.addWidget(self._inc_derived)
        more.addWidget(self._inc_recipe)
        filt_row = QHBoxLayout()
        filt_row.addWidget(QLabel("Filter handling:"))
        self._filter_mode = QComboBox()
        self._filter_mode.addItem("flag rows (ftr column, keep all)", "flag")
        self._filter_mode.addItem("apply (drop filtered rows)", "apply")
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

    def _refresh(self) -> None:
        # The recipe-only shortcut only applies to raw content on a file-backed ds.
        show_shortcut = self._file_backed and self._current_content() == "raw"
        self._recipe_only.setVisible(show_shortcut)
        self._path_box.setVisible(not self._is_recipe_only())
        self.adjustSize()

    def _toggle_more(self, on: bool) -> None:
        self._more_btn.setText("More options ▾" if on else "More options ▸")
        self._more_box.setVisible(on)
        self.adjustSize()

    def _sync_location(self) -> None:
        fmt = self._format.currentData()
        ext = _EXT[fmt]
        if self._chosen_path is not None:
            self._loc_lbl.setText(f"Location: {self._chosen_path.with_suffix(ext).parent}")
        # .msr is raw-only: show its disclaimer and pin the content to raw.
        note = getattr(self, "_msr_note", None)
        if note is not None:
            is_msr = fmt in _RAW_ONLY_FORMATS
            note.setVisible(is_msr)
            if is_msr:
                i = self._content_combo.findData("raw")
                if i >= 0:
                    self._content_combo.setCurrentIndex(i)
                self._content_combo.setEnabled(False)
            else:
                self._content_combo.setEnabled(self._content_combo.count() > 1)

    def _on_browse(self) -> None:
        fmt = self._format.currentData()
        ext = _EXT[fmt]
        suggested = str(self._default_dir / f"{self._name.text() or 'dataset'}{ext}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save / export data", suggested, _FILTERS[fmt])
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
        if fmt in _RAW_ONLY_FORMATS:
            content = "raw"                  # .msr carries only canonical raw data
        ext = _EXT[fmt]
        if self._chosen_path is not None:
            data_path = self._chosen_path.with_suffix(ext)
        else:
            name = self._name.text().strip() or "dataset"
            data_path = self._default_dir / f"{name}{ext}"
        return {"data_path": data_path, "fmt": fmt, "content": content,
                "include": include, "filter_mode": filter_mode}
