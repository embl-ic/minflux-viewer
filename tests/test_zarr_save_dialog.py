"""The Zarr save chooser treats a directory store as one save target."""

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QDialog

from minflux_viewer.ui.zarr_save_dialog import ZarrSaveFileDialog


@pytest.fixture(scope="module")
def _app():
    yield QApplication.instance() or QApplication([])


def test_save_accepts_existing_zarr_directory_instead_of_entering_it(_app, tmp_path):
    target = tmp_path / "processed.zarr"
    target.mkdir()
    dialog = ZarrSaveFileDialog(None, "Save Zarr v2", target)
    try:
        dialog.selectFile(str(target))
        dialog.accept()
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert dialog.selected_zarr_path() == target
        assert Path(dialog.directory().absolutePath()) == tmp_path
    finally:
        dialog.close()


def test_save_adds_zarr_suffix_for_new_store(_app, tmp_path):
    target = tmp_path / "new-processed"
    dialog = ZarrSaveFileDialog(None, "Save Zarr v2", target)
    try:
        dialog.selectFile(str(target))
        dialog.accept()
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert dialog.selected_zarr_path() == target.with_suffix(".zarr")
    finally:
        dialog.close()
