"""Render-window ROI keyboard / right-click integration.

The render window is a ``pg.ImageView`` whose own ``keyPressEvent`` grabs the
arrow keys for its (unused) timeline, so before the fix arrow-key ROI editing
only worked while the inner GraphicsView held focus. The controller now accepts
key events from the ImageView (and the window), and the render View context menu
defers to a ROI right-click.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from minflux_viewer.core.app_state import AppState
from minflux_viewer.core.dataset import build_localization_dataset
from minflux_viewer.core.roi import RoiRecord
from minflux_viewer.ui.render_window import RenderWindow
from minflux_viewer.ui.roi_overlay import item_to_geometry


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _render_with_line(_app):
    st = AppState()
    rng = np.random.default_rng(0)
    st.add_dataset(build_localization_dataset(
        name="d", x_nm=rng.uniform(0, 1000, 2000), y_nm=rng.uniform(0, 1000, 2000)))
    w = RenderWindow(st, 0)
    w.resize(500, 500)
    w.show()
    _app.processEvents()
    rec = RoiRecord.create("line", {"points": [[300.0, 300.0], [600.0, 600.0]], "closed": False})
    st.rois.add(rec)
    st.rois.select([rec.id])
    w._roi_overlay.refresh()
    _app.processEvents()
    return w, rec


def _line_x0(ctrl, rec):
    item = ctrl.items.get(rec.id)
    return round(item_to_geometry("line", item)["points"][0][0], 1)


def test_imageview_is_registered_key_source(_app):
    w, _rec = _render_with_line(_app)
    assert w._image_view in w._roi_overlay._extra_key_sources
    assert w in w._roi_overlay._extra_key_sources
    w.close()


def test_arrow_moves_line_regardless_of_focus(_app):
    from PyQt6.QtTest import QTest
    w, rec = _render_with_line(_app)
    ctrl = w._roi_overlay
    for target in (w._image_view, w._image_view.ui.graphicsView, w):
        target.setFocus()
        _app.processEvents()
        before = _line_x0(ctrl, rec)
        QTest.keyClick(QApplication.focusWidget(), Qt.Key.Key_Right)
        _app.processEvents()
        assert _line_x0(ctrl, rec) == before + 1.0, (
            f"arrow did not reach the ROI when {type(target).__name__} had focus")
    w.close()


def test_view_menu_defers_to_roi_right_click(_app):
    """`_show_context_menu` returns without showing the View menu when the click
    is on a ROI (the ROI controller shows the ROI menu instead)."""
    import pyqtgraph as pg
    w, rec = _render_with_line(_app)
    ctrl = w._roi_overlay
    gv = w._image_view.ui.graphicsView
    vb = ctrl.view_box
    shown = {"view": False}
    # Wrap the actual menu build so we can tell if the View menu would show.
    from PyQt6.QtWidgets import QMenu
    QMenu.exec = lambda self, *a, **k: shown.__setitem__("view", True)
    # right-click ON the line (view (450,450) is on the segment) → deferred (no View menu)
    on_line = gv.mapFromScene(vb.mapViewToScene(pg.Point(450.0, 450.0)))
    w._show_context_menu(on_line)
    assert shown["view"] is False
    # right-click on empty space → View menu shows
    shown["view"] = False
    off = gv.mapFromScene(vb.mapViewToScene(pg.Point(50.0, 950.0)))
    w._show_context_menu(off)
    assert shown["view"] is True
    w.close()
