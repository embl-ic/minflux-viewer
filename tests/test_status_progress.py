"""Status-bar progress for long computations + live ROI-draw read-out."""

from __future__ import annotations

import sys

import pytest


@pytest.fixture
def _app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def test_status_progress_and_task_drive_status_bar(_app):
    from minflux_viewer.core.app_state import AppState

    st = AppState()
    msgs: list[str] = []
    st.status_message.connect(msgs.append)

    st.status_progress("Computing X")                  # indeterminate
    assert msgs[-1] == "Computing X…"
    st.status_progress("Computing X…", 0.5)            # trailing … stripped, % appended
    assert msgs[-1] == "Computing X…  50%"

    with st.task("Doing thing…") as t:
        t.update(0.25)
        t.update(0.75)
    assert msgs[0 + 2] == "Doing thing…"               # header on task entry
    assert any(m == "Doing thing…  25%" for m in msgs)
    assert any(m == "Doing thing…  75%" for m in msgs)
    assert msgs[-1] == "Doing thing: done."            # completion


def test_task_failure_reports_status(_app):
    from minflux_viewer.core.app_state import AppState

    st = AppState()
    msgs: list[str] = []
    st.status_message.connect(msgs.append)
    with pytest.raises(RuntimeError):
        with st.task("Risky op"):
            raise RuntimeError("boom")
    assert msgs[-1] == "Risky op: failed."


def test_roi_status_readout_text(_app):
    pytest.importorskip("pyqtgraph")
    from minflux_viewer.ui.roi_overlay import RoiOverlayController
    f = RoiOverlayController._roi_status_text

    assert f("rectangle", {"bounds": [49, 183, 37, 33]}) == "Rectangle X=49, Y=183, W=37, H=33"
    assert "angle=12°" in f("rectangle", {"bounds": [0, 0, 100, 100], "angle": 12.0})
    assert f("oval", {"bounds": [49, 183, 37, 33]}) == "Oval X=49, Y=183, W=37, H=33"
    # straight line: start point + length + angle from the +X axis in (-180, 180]
    assert f("line", {"points": [[0, 0], [300, 400]]}) == \
        "Line 2 vertices, X=0, Y=0, length=500, angle=53.1°"
    # angle sign follows atan2(dy, dx): pointing down-left → -135°
    assert "angle=-135.0°" in f("line", {"points": [[0, 0], [-10, -10]]})
    # 3-D line vertices carry a depth — index [0]/[1], never unpack
    assert f("line", {"points": [[575, -149, 5], [658, -94, 5]]}).startswith(
        "Line 2 vertices, X=575, Y=-149, length=")
    # other multi-vertex shapes keep type + vertex count + bounding box (origin + size)
    poly = f("polygon", {"points": [[10, 20], [110, 20], [110, 120], [10, 120]]})
    assert "4 vertices" in poly and "Bounding Box X=10, Y=20, W=100, H=100" in poly
    assert "3 vertices" in f("polyline", {"points": [[0, 0], [100, 0], [100, 100]]})
    # a point geometry is 3-D — index [0]/[1], never unpack x,y = p; report X/Y
    assert f("point", {"point": [49, 183, 300]}) == "Point X=49, Y=183"
    assert f("angle", {"points": [[0, 0], [1, 0], [1, 1]]}) == ""   # angle uses its own read-out


def test_emit_status_pushes_to_owner_status_bar(_app):
    pytest.importorskip("pyqtgraph")
    from types import SimpleNamespace

    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.ui.roi_overlay import RoiOverlayController

    st = AppState()
    msgs: list[str] = []
    st.status_message.connect(msgs.append)
    fake = SimpleNamespace(owner=SimpleNamespace(_state=st))    # minimal controller stand-in

    RoiOverlayController._emit_status(fake, "Rectangle X=0, Y=0, W=100, H=200")
    assert msgs[-1] == "Rectangle X=0, Y=0, W=100, H=200"
    RoiOverlayController._emit_status(fake, "")                 # empty → no emit
    assert len(msgs) == 1
    RoiOverlayController._clear_status(fake)                    # blank the status bar
    assert msgs[-1] == "" and len(msgs) == 2
