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
        try:
            box.close()
        except (KeyError, RuntimeError):
            # ``ViewBox.close`` is not idempotent in pyqtgraph 0.14 because its
            # unregister step deletes from a WeakKeyDictionary unconditionally.
            pass


def close_plot_widgets(*plots: Any) -> None:
    """Close complete PlotWidgets after unregistering their ViewBoxes.

    PlotWidget.close() also detaches axes, labels, proxy widgets, and its scene;
    closing only the ViewBox leaves those objects able to receive queued layout
    events during owner destruction.
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
        try:
            plot.close()
        except (AttributeError, RuntimeError):
            pass
