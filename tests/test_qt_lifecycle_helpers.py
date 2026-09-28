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


def test_plot_widget_is_retired_after_viewbox_unregistration(_app):
    import pyqtgraph as pg

    from minflux_viewer.ui.qt_lifecycle import close_plot_widgets

    plot = pg.PlotWidget()
    box = plot.getViewBox()

    close_plot_widgets(plot)
    close_plot_widgets(plot)

    assert box not in pg.ViewBox.AllViews
    assert plot.plotItem is not None
    assert plot.updatesEnabled() is False
    assert plot.isHidden()


def test_deleted_pyqtgraph_label_ignores_deferred_resize(_app):
    from PyQt6 import sip
    from pyqtgraph.graphicsItems.LabelItem import LabelItem

    label = LabelItem("queued layout")
    sip.delete(label)
    label.resizeEvent(None)


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


def test_pyqtgraph_version_gate_matches_the_lines_the_helpers_target():
    """The helpers reach into pyqtgraph internals, so the version is checked."""
    from minflux_viewer.ui.qt_lifecycle import (
        TESTED_PYQTGRAPH_VERSIONS,
        _pyqtgraph_version_is_tested,
    )

    assert _pyqtgraph_version_is_tested("0.14.0")
    assert _pyqtgraph_version_is_tested("0.13.7")
    # A future line must be reported, not silently assumed compatible.
    assert not _pyqtgraph_version_is_tested("0.15.0")
    assert not _pyqtgraph_version_is_tested("1.0.0")
    assert not _pyqtgraph_version_is_tested("garbage")

    # pyproject pins the same range, so a resolver cannot outrun this file.
    import pyqtgraph

    assert _pyqtgraph_version_is_tested(pyqtgraph.__version__), (
        f"installed pyqtgraph {pyqtgraph.__version__} is outside "
        f"{TESTED_PYQTGRAPH_VERSIONS}; review ui/qt_lifecycle.py"
    )


def test_label_guard_fails_open_when_pyqtgraph_renames_its_text_child(_app):
    """A renamed ``LabelItem.item`` must not silently disable label layout.

    Skipping every resize would leave labels mispositioned with nothing in the
    log to trace, so the guard defers to the original implementation instead.
    """
    import pyqtgraph as pg
    from pyqtgraph.graphicsItems.LabelItem import LabelItem

    from minflux_viewer.ui.qt_lifecycle import install_pyqtgraph_lifecycle_guards

    install_pyqtgraph_lifecycle_guards()
    assert getattr(LabelItem, "_mfv_deleted_object_guard", False)

    label = pg.LabelItem("hello")
    calls = []
    # Emulate a future pyqtgraph whose text child is no longer called ``item``.
    original_item = label.item
    try:
        type(label).resizeEvent  # guard is installed on the class
        del label.item
        with pytest.warns(RuntimeWarning, match="no 'item' attribute"):
            with pytest.raises(AttributeError):
                # Upstream itself raises without the attribute: proof the guard
                # delegated rather than swallowing the call.
                label.resizeEvent(None)
        calls.append("delegated")
    finally:
        label.item = original_item
    assert calls == ["delegated"]

    # A live label still lays out normally through the original implementation.
    label.resizeEvent(None)


def test_label_guard_skips_a_deleted_text_child(_app):
    """The guard's actual job: a deleted child is skipped, not raised on."""
    import pyqtgraph as pg
    from PyQt6 import sip

    from minflux_viewer.ui.qt_lifecycle import install_pyqtgraph_lifecycle_guards

    install_pyqtgraph_lifecycle_guards()
    label = pg.LabelItem("hello")
    sip.delete(label.item)
    assert label.resizeEvent(None) is None


def test_dispose_plot_widgets_unregisters_without_detaching(_app):
    """The broad disposer must not reparent a plot its window still holds.

    ``close_plot_widgets`` hides/reparents/deletes, which is right for a window
    being destroyed but crashes a QDialog that keeps the plot on an attribute
    and can still service a queued callback after ``closeEvent``.
    """
    import pyqtgraph as pg
    from PyQt6.QtWidgets import QVBoxLayout, QWidget
    from pyqtgraph.graphicsItems.ViewBox.ViewBox import ViewBox

    from minflux_viewer.ui.qt_lifecycle import dispose_plot_widgets

    window = QWidget()
    layout = QVBoxLayout(window)
    # Built in a loop with no attribute: the case that motivates a tree walk.
    plots = []
    for _ in range(3):
        plot = pg.PlotWidget()
        layout.addWidget(plot)
        plots.append(plot)
    boxes = [plot.getPlotItem().vb for plot in plots]
    assert all(box in ViewBox.AllViews for box in boxes)

    dispose_plot_widgets(window)

    assert all(box not in ViewBox.AllViews for box in boxes)
    # Still parented and still usable: Qt deletes them with the window.
    assert all(plot.parentWidget() is window for plot in plots)
    dispose_plot_widgets(window)  # idempotent
    window.close()


def test_dispose_plot_widgets_leaves_an_image_views_own_plot_alone(_app):
    """An ImageView's nested roiPlot is its own; only close_image_views owns it."""
    import pyqtgraph as pg
    from PyQt6.QtWidgets import QVBoxLayout, QWidget
    from pyqtgraph.graphicsItems.ViewBox.ViewBox import ViewBox

    from minflux_viewer.ui.qt_lifecycle import dispose_plot_widgets

    window = QWidget()
    layout = QVBoxLayout(window)
    image_view = pg.ImageView(view=pg.PlotItem(enableMenu=False))
    layout.addWidget(image_view)
    roi_plot = image_view.ui.roiPlot

    dispose_plot_widgets(window)

    assert roi_plot.parentWidget() is not None
    assert image_view.view.vb not in ViewBox.AllViews
    assert roi_plot.getPlotItem().vb not in ViewBox.AllViews
    window.close()


def test_an_infinite_line_that_outlived_its_view_does_not_raise_in_a_paint(_app):
    """Upstream guards the first ViewBox access in boundingRect, not the second.

    ``GraphicsItem.viewRect`` answers from ``_cachedView``, which is not
    invalidated when the item loses its ViewBox, so
    ``InfiniteLine._computeBoundingRect`` passes its ``if vr is None`` guard and
    then calls ``self.getViewBox().size()`` on ``None``. Inside a paint that
    AttributeError becomes a native abort rather than a traceback, which is how
    it reached the suite as ``Fatal Python error: Aborted`` with no Python
    stack.
    """
    import pyqtgraph as pg
    from PyQt6.QtCore import QRectF
    from pyqtgraph.graphicsItems.InfiniteLine import InfiniteLine

    import minflux_viewer.ui  # noqa: F401 - installing the guards is the point

    assert InfiniteLine._mfv_deleted_object_guard

    plot = pg.PlotWidget()
    line = pg.InfiniteLine(pos=5.0, angle=90)
    plot.addItem(line)
    plot.resize(300, 200)
    plot.show()
    _app.processEvents()
    assert line.boundingRect().width() != 0          # a real rect while shown

    plot.removeItem(line)
    line._boundingRect = None                         # force a recomputation
    # Exactly the inconsistent pair the guard exists for.
    assert line.viewRect() is not None
    assert line.getViewBox() is None

    assert line.boundingRect() == QRectF()            # empty, and no raise
    plot.close()


def test_the_infinite_line_guard_is_installed_once_and_defers_when_unsure():
    """Fails open: an unexpected pyqtgraph is warned about, not silently bent."""
    import warnings

    from pyqtgraph.graphicsItems.InfiniteLine import InfiniteLine

    from minflux_viewer.ui.qt_lifecycle import _install_infinite_line_guard

    installed = InfiniteLine._computeBoundingRect
    _install_infinite_line_guard(__import__("PyQt6.sip", fromlist=["sip"]),
                                 warnings)
    # Already guarded, so re-installing must not wrap it a second time.
    assert InfiniteLine._computeBoundingRect is installed


def test_the_guard_survives_the_view_dying_mid_call(_app):
    """``getViewBox`` is a weakref deref, so a pre-check cannot be the guarantee.

    It can answer with a ViewBox on one call and ``None`` on the very next, as
    the referent is collected in between -- which is exactly what happened
    after the first version of this guard shipped: it checked, passed, and
    upstream then got ``None`` a moment later.
    """
    import pyqtgraph as pg
    from PyQt6.QtCore import QRectF

    import minflux_viewer.ui  # noqa: F401

    plot = pg.PlotWidget()
    line = pg.InfiniteLine(pos=5.0, angle=90)
    plot.addItem(line)
    plot.resize(300, 200)
    plot.show()
    _app.processEvents()
    line._boundingRect = None

    answers = [plot.getPlotItem().vb, None, None, None]

    def dying_view_box():
        return answers.pop(0) if answers else None

    line.getViewBox = dying_view_box          # live once, then collected
    assert line.boundingRect() == QRectF()    # no raise out of the paint
    plot.close()
