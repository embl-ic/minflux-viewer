"""The render TIFF export dialog always hands back an OME-TIFF name.

Fiji routes a TIFF to Bio-Formats only for a ``.ome.tif`` name; any other name
opens a CZYX export as one flat stack (see ``core.tiff_export.ome_tiff_path``).
"""

from pathlib import Path

import pytest
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from minflux_viewer.ui.tiff_export_dialog import TiffExportDialog


def _dialog(qtbot, default_path):
    dlg = TiffExportDialog(
        default_path=str(default_path), default_pixel_nm=10.0, default_voxel_nm=10.0,
        is_3d=True, x_span=(0.0, 100.0), y_span=(0.0, 100.0), z_span=(0.0, 50.0),
        z_scaling_factor=1.0, xy_from_roi=False,
    )
    qtbot.addWidget(dlg)
    return dlg


@pytest.mark.parametrize("typed, expected", [
    ("run.tif", "run.ome.tif"),
    ("run.ome.tif", "run.ome.tif"),
    ("run", "run.ome.tif"),
])
def test_params_always_return_an_ome_tiff_name(qtbot, tmp_path, typed, expected):
    dlg = _dialog(qtbot, tmp_path / typed)
    assert Path(dlg.params()["path"]) == tmp_path / expected


def test_an_empty_path_stays_empty(qtbot):
    dlg = _dialog(qtbot, "")
    assert dlg.params()["path"] == ""


def test_browse_asks_before_a_renamed_target_replaces_an_existing_file(qtbot, tmp_path, monkeypatch):
    (tmp_path / "run.ome.tif").write_bytes(b"existing")
    dlg = _dialog(qtbot, tmp_path / "start.ome.tif")
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "run.tif"), "")))
    asked = []

    def refuse(*args, **kwargs):
        asked.append(args[2])
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(refuse))
    dlg._browse()
    assert asked and "run.ome.tif" in asked[0]
    assert Path(dlg.params()["path"]) == tmp_path / "start.ome.tif"   # unchanged

    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    dlg._browse()
    assert Path(dlg.params()["path"]) == tmp_path / "run.ome.tif"
