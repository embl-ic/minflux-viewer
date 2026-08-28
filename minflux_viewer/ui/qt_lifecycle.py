"""Small, idempotent helpers for deterministic Qt/pyqtgraph teardown.

Qt owns the C++ children of a widget, while pyqtgraph also keeps process-wide
weak registries of ``ViewBox`` wrappers.  Letting Qt delete the child menus
before pyqtgraph unregisters the corresponding view leaves a live Python
wrapper around dead C++ controls.  Plot windows use these helpers *before*
their QWidget hierarchy is destroyed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def install_pyqtgraph_lifecycle_guards() -> None:
    """Ignore pyqtgraph label resizes only after SIP confirms deletion.

    PyQtGraph 0.14 can dispatch a deferred ``LabelItem.resizeEvent`` after Qt
    has deleted the graphics item or its text child. Every callback on a live
    object still runs through the original implementation and propagates its
    errors normally.
    """
    try:
        from PyQt6 import sip
        from pyqtgraph.graphicsItems.LabelItem import LabelItem
    except ImportError:  # pragma: no cover - non-GUI/minimal installations
        return
    if getattr(LabelItem, "_mfv_deleted_object_guard", False):
        return

    original_resize_event = LabelItem.resizeEvent

    def resize_event(label, event):
        if sip.isdeleted(label):
            return
        text_item = getattr(label, "item", None)
        if text_item is None or sip.isdeleted(text_item):
            return
        try:
            return original_resize_event(label, event)
        except RuntimeError as exc:
            if "has been deleted" in str(exc):
                return
            raise

    LabelItem.resizeEvent = resize_event
    LabelItem._mfv_deleted_object_guard = True


def qobject_alive(obj: Any) -> bool:
    """Return whether *obj* still wraps a live Qt object."""
    if obj is None:
        return False
    try:
        from PyQt6 import sip

        return not sip.isdeleted(obj)
    except (ImportError, TypeError, RuntimeError):
        return False


def disconnect_signal(signal: Any, slot: Any | None = None) -> None:
    """Disconnect a Qt signal without leaking teardown exceptions."""
    if signal is None:
        return
    try:
        if slot is None:
            signal.disconnect()
        else:
            signal.disconnect(slot)
    except (AttributeError, TypeError, RuntimeError):
        pass


def remove_event_filter(target: Any, event_filter: Any) -> None:
    """Remove an event filter when both wrappers are still usable."""
    if not qobject_alive(target) or not qobject_alive(event_filter):
        return
    try:
        target.removeEventFilter(event_filter)
    except (AttributeError, RuntimeError):
        pass


def _view_box(candidate: Any) -> Any | None:
    """Resolve common pyqtgraph wrappers to their underlying ViewBox."""
    if candidate is None:
        return None
    try:
        from pyqtgraph import ViewBox

        if isinstance(candidate, ViewBox):
            return candidate
        box = getattr(candidate, "vb", None)
        if isinstance(box, ViewBox):
            return box
    except (ImportError, RuntimeError):
        return None
    try:
        getter = getattr(candidate, "getViewBox", None)
        if callable(getter):
            box = getter()
            return box if isinstance(box, ViewBox) else None
    except RuntimeError:
        return None
    return None


def image_view_boxes(image_view: Any) -> tuple[Any, ...]:
    """Return every ViewBox owned by a pyqtgraph ImageView, without duplicates."""
    if image_view is None:
        return ()
    candidates: list[Any] = []
    try:
        candidates.append(getattr(image_view, "view", None))
        ui = getattr(image_view, "ui", None)
        candidates.append(getattr(ui, "roiPlot", None))
        histogram = getattr(ui, "histogram", None)
        candidates.append(getattr(histogram, "item", histogram))
    except RuntimeError:
        pass
    return _unique_view_boxes(candidates)


def _unique_view_boxes(candidates: Iterable[Any]) -> tuple[Any, ...]:
    boxes: list[Any] = []
    seen: set[int] = set()
    for candidate in candidates:
        box = _view_box(candidate)
        if box is None or id(box) in seen:
            continue
        seen.add(id(box))
        boxes.append(box)
    return tuple(boxes)


def _disconnect_widget_group(group: Any) -> None:
    """Empty a pyqtgraph WidgetGroup while its child widgets are still alive."""
    if group is None:
        return
    try:
        widgets = list(getattr(group, "widgetList", ()))
    except (RuntimeError, TypeError):
        widgets = []
    signal_names = (
        "toggled",
        "clicked",
        "valueChanged",
        "currentIndexChanged",
        "editingFinished",
        "stateChanged",
        "textChanged",
    )
    for widget in widgets:
        for name in signal_names:
            try:
                disconnect_signal(getattr(widget, name, None))
            except RuntimeError:
                pass
    for name in ("widgetList", "scales", "uncachedWidgets", "cache"):
        try:
            getattr(group, name).clear()
        except (AttributeError, RuntimeError):
            pass


def _close_view_box_menu(box: Any) -> None:
    """Break ViewBoxMenu/WidgetGroup signal cycles before QObject deletion."""
    try:
        menu = getattr(box, "menu", None)
    except RuntimeError:
        return
    if menu is None or not qobject_alive(menu):
        return
    for group in list(getattr(menu, "widgetGroups", ()) or ()):
        _disconnect_widget_group(group)
    try:
        menu.widgetGroups.clear()
    except (AttributeError, RuntimeError):
        pass
    # The generated menu owns many controls whose signals target the menu and
    # its WidgetGroups. Disconnect them before Qt removes the child hierarchy;
    # otherwise weak-key callbacks can run while wrapping half-deleted widgets.
    try:
        from PyQt6.QtCore import QObject

        children = menu.findChildren(QObject)
    except (ImportError, RuntimeError, TypeError):
        children = []
    for child in children:
        for name in (
            "toggled", "clicked", "valueChanged", "currentIndexChanged",
            "editingFinished", "stateChanged", "textChanged", "triggered",
        ):
            try:
                disconnect_signal(getattr(child, name, None))
            except RuntimeError:
                pass
    try:
        disconnect_signal(box.sigStateChanged)
    except (AttributeError, RuntimeError):
        pass
    try:
        menu.close()
        menu.deleteLater()
        box.menu = None
    except (AttributeError, RuntimeError):
        pass


def close_view_boxes(*candidates: Any) -> None:
    """Close/unregister ViewBoxes exactly once, before their menus are deleted."""
    expanded: list[Any] = []
    for candidate in candidates:
        expanded.extend(image_view_boxes(candidate))
        expanded.append(candidate)
    for box in _unique_view_boxes(expanded):
        if getattr(box, "_mfv_view_box_closed", False):
            continue
        try:
            box._mfv_view_box_closed = True
        except (AttributeError, RuntimeError):
            pass
        if not qobject_alive(box):
            continue
        _close_view_box_menu(box)
        # register() installs a destroyed-lambda that walks the process-wide
        # weak registries. We unregister explicitly below, so leaving that
        # callback connected only creates a second, GC-time traversal while
        # Qt may already be deleting other views.
        try:
            disconnect_signal(box.destroyed)
        except (AttributeError, RuntimeError):
            pass
        try:
            box.close()
        except (KeyError, RuntimeError):
            # ``ViewBox.close`` is not idempotent in pyqtgraph 0.14 because its
            # unregister step deletes from a WeakKeyDictionary unconditionally.
            pass


def close_plot_widgets(*plots: Any) -> None:
    """Retire PlotWidgets without clearing their scene during a close event.

    PyQtGraph's ``PlotWidget.close()`` clears the graphics scene immediately.
    With a deferred layout activation still queued, that can delete a
    ``LabelItem`` before its pending resize callback runs. Make the plot inert,
    detach it from the owner, and defer deletion until already-posted layout
    work has run.
    """
    for plot in plots:
        if plot is None or getattr(plot, "_mfv_plot_widget_closed", False):
            continue
        try:
            plot._mfv_plot_widget_closed = True
        except (AttributeError, RuntimeError):
            pass
        if not qobject_alive(plot):
            continue
        close_view_boxes(plot)
        _disconnect_widget_group(getattr(plot, "stateGroup", None))
        try:
            plot.setUpdatesEnabled(False)
            plot.hide()
            plot.setParent(None)
            plot.deleteLater()
        except (AttributeError, RuntimeError):
            pass


def close_image_views(*image_views: Any) -> None:
    """Dispose every PlotItem/ViewBox owned by a pyqtgraph ImageView."""
    for image_view in image_views:
        if image_view is None:
            continue
        close_view_boxes(image_view)
        try:
            ui = image_view.ui
            # Do not call PlotItem.close() here: ImageView can have queued
            # viewStateChanged callbacks that still expect its auto button.
            # The owning QWidget will delete PlotItems after the queue drains;
            # only their unsafe WidgetGroup weak maps need proactive cleanup.
            for plot in (image_view.view, getattr(ui, "roiPlot", None)):
                _disconnect_widget_group(getattr(plot, "stateGroup", None))
        except (AttributeError, RuntimeError):
            pass
