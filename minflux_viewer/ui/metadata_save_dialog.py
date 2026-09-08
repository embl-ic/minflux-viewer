"""Where to write a processing-metadata file, when it stands on its own.

Saving the recipe *alone* is a real thing to want: to look at what the current
processing amounts to before deciding which data format to commit it to, or to
get hold of it when the format in use keeps it internally (``.zarr`` and our
``.msr`` store it inside the file and write no sidecar at all).

That is why this asks rather than deriving a path. ``save_processed``'s
metadata-only branch takes the dataset's own folder, which is right when the
sidecar accompanies a file already on disk and wrong when there is none -- it
would put the file in whatever the working directory happens to be. A dataset
with no file behind it is also exactly the case the note in this dialog exists
to state, instead of leaving the user to discover it from a sidecar that pairs
with nothing.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.save import METADATA_SUFFIX

__all__ = ["MetadataSaveDialog", "ask_metadata_save_path", "suggested_metadata_path"]


def suggested_metadata_path(ds, default_dir) -> Path:
    """``<data stem>_viewer_metadata.json`` beside the data, or in *default_dir*.

    The stem always comes from the dataset, so the pair stays adjacent in a
    sorted folder even when the user redirects the file elsewhere.
    """
    handle = getattr(ds, "file", None)
    folder = Path(str(getattr(handle, "folder", "") or "") or default_dir)
    if not folder.is_dir():
        folder = Path(default_dir)
    stem = Path(str(getattr(ds, "name", "") or "dataset")).stem or "dataset"
    return folder / f"{stem}_{METADATA_SUFFIX}.json"


class MetadataSaveDialog(QDialog):
    """Path chooser for a standalone processing-metadata file."""

    def __init__(self, suggested, *, data_filename: str | None = None,
                 dataset_name: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save viewer metadata")
        self.setMinimumWidth(600)

        root = QVBoxLayout(self)
        if dataset_name:
            root.addWidget(QLabel(f"Dataset: <b>{dataset_name}</b>"))

        row = QHBoxLayout()
        row.addWidget(QLabel("Save to"))
        self._path = QLineEdit(str(suggested))
        self._path.setToolTip(
            "The processing this dataset carries — Z scaling factor, transform, "
            "filters, ROIs and acquisition time — as a JSON file that can be "
            "applied to a data file later."
        )
        row.addWidget(self._path, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._on_browse)
        row.addWidget(browse)
        root.addLayout(row)

        # State the pairing plainly. A recipe finds its dataset by DID, then by
        # data-file name, then by acquisition time -- so one written with no data
        # file is still useful, and the user should know that is what they have.
        if data_filename:
            note = QLabel(f"Accompanies <b>{data_filename}</b>.")
        else:
            note = QLabel(
                "This dataset has <b>no data file</b> on disk, so the metadata "
                "will not name one. It still records the processing, and can be "
                "applied to a data file you save later."
            )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        root.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _on_browse(self) -> None:
        current = self.path() or Path.home() / f"{METADATA_SUFFIX}.json"
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save viewer metadata", str(current),
            "Viewer metadata (*.json);;All files (*)")
        if chosen:
            self._path.setText(chosen)

    def path(self) -> Path | None:
        text = self._path.text().strip().strip('"')
        if not text:
            return None
        candidate = Path(text)
        if candidate.suffix.lower() != ".json":
            candidate = candidate.with_suffix(".json")
        return candidate

    def accept(self) -> None:  # noqa: D102 - behavior is the class contract
        target = self.path()
        if target is None:
            QMessageBox.warning(self, "Save viewer metadata",
                                "Enter a file path to save to.")
            return
        if not target.parent.is_dir():
            QMessageBox.warning(
                self, "Save viewer metadata",
                f"{target.parent}\ndoes not exist. Choose another folder.")
            return
        super().accept()


def ask_metadata_save_path(parent, ds, default_dir, *,
                           data_filename: str | None = None) -> Path | None:
    """Show :class:`MetadataSaveDialog`; the chosen path, or None if cancelled."""
    dialog = MetadataSaveDialog(
        suggested_metadata_path(ds, default_dir),
        data_filename=data_filename,
        dataset_name=str(getattr(ds, "name", "") or ""),
        parent=parent,
    )
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.path()
