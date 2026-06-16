"""
Save-processed-data dialog.

The data is always written as **canonical raw** values; all gating, calibration
and filtering go into a ``<stem>_metadata.json`` sidecar. The dialog therefore
only decides *where*:

- **File-backed dataset** (already has a physical ``.mat`` / ``.npy`` / ``.json``):
  the raw data is already on disk, so only the metadata sidecar is written.
- **No source file** (``.msr`` extract, duplicate): the user picks a name, a
  format (``.mat`` / ``.npy`` / ``.json``) and, optionally, a full location.

The actual writing is done by :func:`minflux_viewer.core.save.save_processed`.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

_DESCRIPTION = (
    "The data is saved as <b>canonical raw data</b>, intentionally without "
    "touching the raw values. All gating, calibration and filtering are saved "
    "to and tracked from a <code>&lt;stem&gt;_metadata.json</code> file."
)

# (label, format key, extension)
_FORMATS = (
    ("MATLAB (.mat)", "mat", ".mat"),
    ("NumPy (.npy)", "npy", ".npy"),
    ("JSON (.json)", "json", ".json"),
)
_FILTERS = {
    "mat": "MATLAB (*.mat)", "npy": "NumPy (*.npy)", "json": "JSON (*.json)",
}


class SaveProcessedDataDialog(QDialog):
    """Decide where to save processed data (data file + metadata sidecar)."""

    def __init__(
        self,
        dataset_name: str = "",
        *,
        file_backed: bool = False,
        source_path: str | Path | None = None,
        default_dir: str | Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save processed data")
        self.setMinimumWidth(480)

        self._file_backed = bool(file_backed)
        self._chosen_path: Path | None = None
        self._default_dir = Path(default_dir or Path.home())

        root = QVBoxLayout(self)
        if dataset_name:
            root.addWidget(QLabel(f"Dataset: <b>{dataset_name}</b>"))

        desc = QLabel(_DESCRIPTION)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: palette(mid); margin: 4px 0;")
        root.addWidget(desc)

        if self._file_backed:
            self._build_file_backed(root, source_path)
        else:
            self._build_no_file(root, dataset_name)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ------------------------------------------------------------------
    def _build_file_backed(self, root: QVBoxLayout, source_path) -> None:
        src = Path(source_path) if source_path else None
        stem = (src.stem if src else "dataset")
        path_lbl = QLabel(f"Input data file exists:\n<b>{src}</b>")
        path_lbl.setWordWrap(True)
        path_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(path_lbl)
        note = QLabel(
            f"Only the operation will be saved into "
            f"<b>{stem}_metadata.json</b> (next to the data file)."
        )
        note.setWordWrap(True)
        root.addWidget(note)

    def _build_no_file(self, root: QVBoxLayout, dataset_name: str) -> None:
        stem = Path(dataset_name or "dataset").stem or "dataset"

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._name = QLineEdit(f"{stem}_processed")
        name_row.addWidget(self._name, 1)
        self._format = QComboBox()
        for label, key, _ext in _FORMATS:
            self._format.addItem(label, key)
        name_row.addWidget(self._format)
        root.addLayout(name_row)

        browse_row = QHBoxLayout()
        self._loc_lbl = QLabel(f"Location: {self._default_dir}")
        self._loc_lbl.setWordWrap(True)
        browse_row.addWidget(self._loc_lbl, 1)
        browse = QPushButton("Save as…")
        browse.clicked.connect(self._on_browse)
        browse_row.addWidget(browse)
        root.addLayout(browse_row)

    # ------------------------------------------------------------------
    def _on_browse(self) -> None:
        fmt = self._format.currentData()
        ext = dict((k, e) for _l, k, e in _FORMATS)[fmt]
        suggested = str(self._default_dir / f"{self._name.text() or 'dataset'}{ext}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save processed data", suggested, _FILTERS[fmt]
        )
        if not path:
            return
        p = Path(path)
        self._chosen_path = p
        # reflect the chosen file in the form
        self._name.setText(p.stem)
        self._loc_lbl.setText(f"Location: {p.parent}")
        if p.suffix.lower() in (".mat", ".npy", ".json"):
            key = p.suffix.lower().lstrip(".")
            i = self._format.findData(key)
            if i >= 0:
                self._format.setCurrentIndex(i)

    # ------------------------------------------------------------------
    def options(self) -> dict:
        """Return where/how to save.

        File-backed → ``{"data_path": None}`` (metadata only). Otherwise
        ``{"data_path": <Path>, "fmt": <key>}``.
        """
        if self._file_backed:
            return {"data_path": None, "fmt": None}
        fmt = self._format.currentData()
        ext = dict((k, e) for _l, k, e in _FORMATS)[fmt]
        if self._chosen_path is not None:
            data_path = self._chosen_path.with_suffix(ext)
        else:
            name = self._name.text().strip() or "dataset"
            data_path = self._default_dir / f"{name}{ext}"
        return {"data_path": data_path, "fmt": fmt}
