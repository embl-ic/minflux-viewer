"""Scale-bar anchor geometry + dialog/overlay construction."""

import pytest

from minflux_viewer.ui.scale_bar import format_nm, scale_bar_anchor


def test_format_nm_units():
    assert format_nm(100) == "100 nm"
    assert format_nm(1500) == "1.5 µm"
    assert format_nm(2_000_000) == "2 mm"


def test_anchor_corners_non_inverted():
    vr = ((0.0, 1000.0), (0.0, 800.0))   # data range
    pxs = (1.0, 1.0)                      # 1 nm/pixel → margin 18 nm
    bw, bh = 100.0, 10.0
    # Lower Left: inset from xmin and ymin
    bx, by = scale_bar_anchor(vr, pxs, bar_w=bw, bar_h=bh, location="Lower Left",
                              x_inv=False, y_inv=False)
    assert (round(bx), round(by)) == (18, 18)
    # Lower Right: bar's right edge inset from xmax
    bx, by = scale_bar_anchor(vr, pxs, bar_w=bw, bar_h=bh, location="Lower Right",
                              x_inv=False, y_inv=False)
    assert (round(bx + bw), round(by)) == (1000 - 18, 18)
    # Upper Right: inset from xmax and ymax
    bx, by = scale_bar_anchor(vr, pxs, bar_w=bw, bar_h=bh, location="Upper Right",
                              x_inv=False, y_inv=False)
    assert (round(bx + bw), round(by + bh)) == (1000 - 18, 800 - 18)


def test_anchor_respects_inversion():
    vr = ((0.0, 1000.0), (0.0, 800.0))
    pxs = (1.0, 1.0)
    # y inverted: screen-top is ymin → "Upper" hugs ymin
    bx, by = scale_bar_anchor(vr, pxs, bar_w=100.0, bar_h=10.0, location="Upper Left",
                              x_inv=False, y_inv=True)
    assert round(by) == 18


def test_anchor_at_selection_centres_on_roi():
    bx, by = scale_bar_anchor(((0, 1000), (0, 800)), (1.0, 1.0), bar_w=100.0, bar_h=10.0,
                              location="At Selection", x_inv=False, y_inv=False,
                              roi_center=(500.0, 400.0))
    assert (bx + 50.0, by + 5.0) == (500.0, 400.0)   # bar centred on the ROI centre


# -- Qt construction smoke ------------------------------------------------

@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("pyqtgraph")
    from PyQt6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _viewbox():
    import pyqtgraph as pg
    plot = pg.PlotWidget()
    plot.setXRange(0, 1000)
    plot.setYRange(0, 800)
    return plot, plot.getPlotItem().getViewBox()


def test_scale_bar_item_builds_and_is_draggable(_app):
    from PyQt6.QtGui import QColor

    from minflux_viewer.ui.scale_bar import ScaleBarItem
    _plot, vb = _viewbox()
    sb = ScaleBarItem(vb, width_nm=100, height_nm=10, font_size=12,
                      color=QColor(255, 255, 255), bg_color=QColor(0, 0, 0),
                      horizontal=True, center=(500.0, 400.0))
    assert sb in vb.allChildren()                      # added to the view
    assert (sb.pos().x(), sb.pos().y()) == (500.0, 400.0)
    # bar covers ~100 nm in x at its centre (data-coord physical size)
    rect = sb.boundingRect()
    assert rect.width() >= 100.0
    # params round-trip
    p = sb.params()
    assert p["width_nm"] == 100.0 and p["height_nm"] == 10.0 and p["horizontal"] is True
    sb.remove()
    assert sb not in vb.allChildren()


def test_scale_bar_item_set_params(_app):
    from PyQt6.QtGui import QColor

    from minflux_viewer.ui.scale_bar import ScaleBarItem
    _plot, vb = _viewbox()
    sb = ScaleBarItem(vb, width_nm=100, height_nm=10, font_size=12,
                      color=QColor(255, 255, 255), bg_color=None,
                      horizontal=True, center=(0.0, 0.0))
    sb.set_params({"width_nm": 250.0, "height_nm": 20.0, "font_size": 16,
                   "color": QColor(255, 0, 0), "bg_color": QColor(0, 0, 0),
                   "horizontal": False})
    p = sb.params()
    assert p["width_nm"] == 250.0 and p["horizontal"] is False
    assert p["color"].getRgb()[:3] == (255, 0, 0)
    assert p["bg_color"] is not None


def test_scale_bar_label_stays_above_bar_when_y_inverted(_app):
    from PyQt6.QtGui import QColor

    from minflux_viewer.ui.scale_bar import ScaleBarItem
    _plot, vb = _viewbox()
    sb = ScaleBarItem(vb, width_nm=1.0, height_nm=1.0, font_size=10,
                      color=QColor(255, 255, 255), bg_color=None,
                      horizontal=True, center=(500.0, 400.0))
    y_normal = sb._label.pos().y()
    assert y_normal > 0                                  # +y is up → label above the bar
    vb.invertY(True)
    sb._relayout()
    y_inverted = sb._label.pos().y()
    assert y_inverted < 0                                # +y now down → label flips to keep
    assert abs(y_inverted) == pytest.approx(abs(y_normal), rel=1e-6)   # same on-screen gap
    sb.remove()


def test_scale_bar_dialog_values(_app):
    from minflux_viewer.ui.scale_bar_dialog import ScaleBarDialog
    dlg = ScaleBarDialog(default_width_nm=200.0, default_height_nm=8.0)
    v = dlg.values()
    assert v["width_nm"] == 200.0 and v["height_nm"] == 8.0
    assert v["color"].getRgb()[:3] == (255, 255, 255)   # default White
    assert v["bg_color"] is None                        # default None
    assert v["location"] == "Lower Right" and v["horizontal"] is True


def test_scale_bar_dialog_edit_prefill(_app):
    from PyQt6.QtGui import QColor

    from minflux_viewer.ui.scale_bar_dialog import ScaleBarDialog
    initial = {"width_nm": 333.0, "height_nm": 12.0, "font_size": 18,
               "color": QColor(255, 0, 0), "bg_color": None, "horizontal": False}
    dlg = ScaleBarDialog(initial=initial, edit_mode=True)
    v = dlg.values()
    assert v["width_nm"] == 333.0 and v["font_size"] == 18
    assert v["color"].getRgb()[:3] == (255, 0, 0)       # Red prefilled
    assert v["bg_color"] is None
    assert v["horizontal"] is False                     # Vertical prefilled
    # custom (non-pure) colour round-trips through the Custom… entry
    dlg2 = ScaleBarDialog(initial={**initial, "color": QColor(17, 34, 51)}, edit_mode=True)
    assert dlg2.values()["color"].getRgb()[:3] == (17, 34, 51)
