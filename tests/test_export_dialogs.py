"""Focused export-dialog behavior."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from minflux_viewer.ui.export_dialogs import OmeZarrExportDialog
from minflux_viewer.ui.main_window import _OmeZarrTask


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def test_ome_zarr_dialog_exposes_z_voxel_depth_for_3d(_app, tmp_path):
    dialog = OmeZarrExportDialog(
        tmp_path / "volume.ome.zarr",
        pixel_size_nm=4.0,
        z_voxel_nm=12.5,
        is_3d=True,
    )
    assert dialog.pixel_size_spin.value() == 4.0
    assert dialog.z_voxel_spin.value() == 12.5
    assert not dialog.z_voxel_spin.isHidden()
    dialog.close()


def test_ome_zarr_dialog_hides_z_voxel_depth_for_2d(_app, tmp_path):
    dialog = OmeZarrExportDialog(
        tmp_path / "image.ome.zarr",
        pixel_size_nm=5.0,
        z_voxel_nm=10.0,
        is_3d=False,
    )
    assert dialog.z_voxel_spin.isHidden()
    dialog.close()


def test_ome_zarr_task_relays_worker_progress(_app):
    updates = []
    results = []

    def export(report):
        report(0.25, "Writing voxel pyramid")
        return "complete"

    task = _OmeZarrTask(export)
    task.signals.progress.connect(
        lambda fraction, stage: updates.append((fraction, stage))
    )
    task.signals.done.connect(results.append)
    task.run()

    assert updates == [(0.25, "Writing voxel pyramid")]
    assert results == ["complete"]
