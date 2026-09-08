"""Zarr-aware save-path chooser, and the File > Save dialog built on it.

Qt's ordinary save dialog treats an existing ``.zarr`` directory as a folder to
navigate into.  A Zarr directory is the file-format package in this application,
so Save must return that directory itself and let the caller offer update/replace.

:class:`ZarrQuickSaveDialog` is what **File > Save** (Ctrl+S) shows.  It asks for
one thing -- where -- because the format is always MINFLUX Viewer Zarr v2 and
that format has nothing else to decide: it is raw-canonical plus separate
processing state, so the content/attribute/derived/sidecar choices of the
Save / export dialog are all either fixed or meaningless for it.  Choosing among
formats is what *Save As* and the Dataset Manager's *Save / export data* are for.
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


class ZarrSaveFileDialog(QFileDialog):
    """A save dialog that accepts an existing ``.zarr`` directory as a file."""

    def __init__(
        self,
        parent: QWidget | None,
        title: str,
        suggested: str | Path,
    ) -> None:
        super().__init__(parent, title)
        self._accepted_path: Path | None = None
        self.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        self.setOption(QFileDialog.Option.DontConfirmOverwrite, True)
        self.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        self.setFileMode(QFileDialog.FileMode.AnyFile)
        self.setNameFilter("MINFLUX Viewer Zarr v2 (*.zarr)")
        self.setDefaultSuffix("zarr")

        proposed = Path(suggested)
        folder = proposed.parent
        if not folder.is_dir():
            folder = Path.home()
        self.setDirectory(str(folder))
        self.selectFile(proposed.name)

    def accept(self) -> None:  # noqa: D102 - behavior is the class contract
        selected = super().selectedFiles()
        if not selected:
            return
        candidate = Path(selected[0])
        if candidate.is_dir() and candidate.suffix.lower() != ".zarr":
            self.setDirectory(str(candidate))
            return
        if candidate.suffix.lower() != ".zarr":
            candidate = candidate.with_suffix(".zarr")
        if candidate.exists() and not candidate.is_dir():
            QMessageBox.warning(
                self,
                "Invalid Zarr destination",
                f"{candidate}\nexists but is not a Zarr directory.",
            )
            return

        self._accepted_path = candidate
        # QFileDialog.accept() navigates into an existing directory. Bypass that
        # implementation after validating the package path and close as a normal
        # accepted dialog instead.
        QDialog.accept(self)

    def selected_zarr_path(self) -> Path | None:
        return self._accepted_path


def choose_zarr_save_path(
    parent: QWidget | None,
    title: str,
    suggested: str | Path,
) -> Path | None:
    dialog = ZarrSaveFileDialog(parent, title, suggested)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.selected_zarr_path()


class ZarrQuickSaveDialog(QDialog):
    """File > Save: one path field, Browse, Save / Cancel.

    The path is pre-filled with where this dataset would go, so Save is usually
    two keystrokes.  Overwrite confirmation is deliberately NOT done here: an
    existing MINFLUX Viewer store offers *Update processing only* as well as
    *Replace complete store*, which is a decision for the caller
    (``MainWindow._zarr_overwrite_mode``), not a yes/no in a file chooser.
    """

    def __init__(self, suggested: str | Path, *, parent: QWidget | None = None,
                 dataset_name: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Save")
        self.setMinimumWidth(560)

        root = QVBoxLayout(self)
        if dataset_name:
            root.addWidget(QLabel(f"Dataset: <b>{dataset_name}</b>"))

        row = QHBoxLayout()
        row.addWidget(QLabel("Save to"))
        self._path = QLineEdit(str(suggested))
        self._path.setToolTip(
            "MINFLUX Viewer Zarr v2 (.zarr): the self-contained application "
            "format. Raw canonical data plus processing state, ROIs, overlay "
            "channels and linked images, with no sidecar file."
        )
        row.addWidget(self._path, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._on_browse)
        row.addWidget(browse)
        root.addLayout(row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _on_browse(self) -> None:
        chosen = choose_zarr_save_path(self, "Save", self.path() or Path.home())
        if chosen is not None:
            self._path.setText(str(chosen))

    def path(self) -> Path | None:
        """The entered destination, given a ``.zarr`` extension; None if blank."""
        text = self._path.text().strip().strip('"')
        if not text:
            return None
        from ..core import formats as _formats

        return _formats.normalize_path("zarr", Path(text))

    def accept(self) -> None:  # noqa: D102 - behavior is the class contract
        target = self.path()
        if target is None:
            QMessageBox.warning(self, "Save", "Enter a file path to save to.")
            return
        if not target.parent.is_dir():
            QMessageBox.warning(
                self, "Save",
                f"{target.parent}\ndoes not exist. Choose another folder.")
            return
        if target.exists() and not target.is_dir():
            QMessageBox.warning(
                self, "Save",
                f"{target}\nexists but is not a Zarr directory.")
            return
        super().accept()


def ask_zarr_save_path(parent, suggested, *, dataset_name: str = "") -> Path | None:
    """Show :class:`ZarrQuickSaveDialog`; the chosen path, or None if cancelled."""
    dialog = ZarrQuickSaveDialog(suggested, parent=parent, dataset_name=dataset_name)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.path()
