from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication, QWidget

from minflux_viewer.core.roi import RoiStore
from minflux_viewer.ui.qt_lifecycle import close_view_boxes, image_view_boxes
from minflux_viewer.ui.roi_overlay import RoiOverlayController


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def test_image_view_boxes_are_closed_and_unregistered_idempotently(_app):
    import pyqtgraph as pg
    from pyqtgraph.graphicsItems.ViewBox.ViewBox import ViewBox

    image_view = pg.ImageView(view=pg.PlotItem(enableMenu=False))
    boxes = image_view_boxes(image_view)
    assert len(boxes) == 3
    assert all(box in ViewBox.AllViews for box in boxes)

    close_view_boxes(image_view)
    assert all(box not in ViewBox.AllViews for box in boxes)
    close_view_boxes(image_view)  # repeated shutdown paths are harmless
    image_view.close()


def test_plot_widget_is_fully_closed_after_viewbox_unregistration(_app):
    import pyqtgraph as pg

    from minflux_viewer.ui.qt_lifecycle import close_plot_widgets

    plot = pg.PlotWidget()
    box = plot.getViewBox()

    close_plot_widgets(plot)
    close_plot_widgets(plot)

    assert box not in pg.ViewBox.AllViews
    assert plot.plotItem is None


def test_roi_controller_dispose_detaches_store_and_queued_callbacks(_app):
    import pyqtgraph as pg

    store = RoiStore()
    owner = QWidget()
    plot = pg.PlotWidget()
    controller = RoiOverlayController(
        store, owner, plot, plot.getPlotItem(), source_view="render"
    )
    controller.add_key_event_source(owner)
    controller.activate()
    assert store.active_adapter is controller

    controller.dispose()
    controller.dispose()
    assert store.active_adapter is None
    assert controller._disposed is True
    assert controller.eventFilter(plot, QEvent(QEvent.Type.FocusIn)) is False

    # Long-lived store emissions after the window has gone must not call back.
    del controller.view_widget
    store.changed.emit()
    store.selection_changed.emit()
    store.restore_requested.emit()
    from minflux_viewer.ui.qt_lifecycle import close_plot_widgets

    close_plot_widgets(plot)
    owner.close()
