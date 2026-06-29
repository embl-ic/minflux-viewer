"""MSR channel alignment should only use metadata-marked used MBM beads."""

from __future__ import annotations

import numpy as np
import pytest


_POINT_DTYPE = np.dtype([("gri", "<i4"), ("xyz", "<f8", (3,))])


def _points(bead_ids: list[int]) -> np.ndarray:
    points = np.zeros(len(bead_ids), dtype=_POINT_DTYPE)
    points["gri"] = bead_ids
    points["xyz"] = np.column_stack([
        np.asarray(bead_ids, dtype=float),
        np.asarray(bead_ids, dtype=float) * 2.0,
        np.zeros(len(bead_ids), dtype=float),
    ])
    return points


def test_align_channel_filters_raw_mbm_points_to_metadata_used_beads():
    pytest.importorskip("PyQt6")
    from minflux_viewer.msr import state as msr_state
    from minflux_viewer.msr.alignment import align_channels_from_arrays
    from minflux_viewer.plugins.msr_reader.msr_reader_dialog import MsrReaderDialog

    old_meta = getattr(msr_state, "mbm_meta_map", None)
    try:
        msr_state.mbm_meta_map = {
            "ref": {
                "points_by_gri": {
                    str(g): {"gri": g, "name": f"R{g}"}
                    for g in (1, 2, 3, 4)
                },
                "used": ["R1", "R3"],
            },
            "moving": {
                "points_by_gri": {
                    str(g): {"gri": g, "name": f"R{g}"}
                    for g in (1, 2, 3, 4)
                },
                "used": ["R1", "R3"],
            },
        }
        dlg = MsrReaderDialog.__new__(MsrReaderDialog)
        dlg.parsed = {}
        logs: list[str] = []
        dlg.log = logs.append

        ref = dlg._filter_mbm_points_to_used_beads("ref", _points([1, 2, 3, 4]), log_prefix="[align]")
        moving = dlg._filter_mbm_points_to_used_beads("moving", _points([1, 2, 3, 4]), log_prefix="[align]")
        result = align_channels_from_arrays("ref", ref, "moving", moving)

        assert result.bead_ids.tolist() == [1, 3]
        assert {int(g) for g in np.unique(ref["gri"])} == {1, 3}
        assert {int(g) for g in np.unique(moving["gri"])} == {1, 3}
    finally:
        if old_meta is None:
            try:
                del msr_state.mbm_meta_map
            except AttributeError:
                pass
        else:
            msr_state.mbm_meta_map = old_meta


def test_alignment_plot_3d_builds_visible_gl_layers():
    pytest.importorskip("PyQt6")
    pytest.importorskip("pyqtgraph.opengl")
    from PyQt6.QtWidgets import QApplication

    from minflux_viewer.plugins.msr_reader.msr_reader_dialog import AlignmentPlotWindow

    app = QApplication.instance() or QApplication([])

    class Result:
        pass

    result = Result()
    result.channel1_name = "ref"
    result.channel2_name = "moving"
    result.bead_ids = np.array([1, 3], dtype=np.uint32)
    result.channel1_xyz = np.array([[0.0, 0.0, 0.0], [1000.0, 2000.0, 100.0]])
    result.channel2_xyz = np.array([[10.0, 0.0, 0.0], [1010.0, 2000.0, 100.0]])
    result.channel2_aligned_xyz = np.array([[0.0, 0.0, 0.0], [1000.0, 2000.0, 100.0]])

    win = AlignmentPlotWindow(
        [result],
        data_bounds_nm=(
            np.array([-500.0, -500.0, -50.0]),
            np.array([1500.0, 2500.0, 150.0]),
        ),
    )
    try:
        win._set_view_mode("3D")

        assert win._view_mode == "3D"
        assert len(win._items_3d) == 3
        assert win._grid_3d is not None
        assert win._axis_3d is not None
        assert win._axis_3d_items
        assert len(win._box_3d_items) == 12
        assert win._plot_xyz_um(visible_only=True, include_data_bounds=True).shape == (8, 3)

        win._set_3d_axis_visible(False)
        assert win._show_3d_axis is False
        assert not win._axis_3d_items
        assert len(win._items_3d) == 3

        win._set_3d_bounding_box_visible(False)
        assert win._show_3d_bounding_box is False
        assert not win._box_3d_items
        assert len(win._items_3d) == 3
    finally:
        win.close()
