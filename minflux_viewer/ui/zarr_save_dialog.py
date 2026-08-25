"""Zarr-aware save-path chooser.

Qt's ordinary save dialog treats an existing ``.zarr`` directory as a folder to
navigate into.  A Zarr directory is the file-format package in this application,
so Save must return that directory itself and let the caller offer update/replace.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QDialog, QFileDialog, QMessageBox, QWidget


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
