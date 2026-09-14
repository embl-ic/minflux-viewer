"""
minflux_viewer.ui.ortho_roi
===========================
ROI projections and controllers in the orthogonal view's **side** panes.

The primary pane keeps the window's own :class:`RoiOverlayController`, which
draws, hit-tests and edits. Each side pane has a peer controller for drawing and
editing geometry its axes can express. This module's outline layer supplies the
remaining display-only projections, so every ROI stays visible in all three
views. A pending volume draft is handed to whichever peer pane the user enters,
making its projected 2-D shape editable without filing it into the Manager.

⚠ **The side panes use the ORTHO column order, not the standalone one.** An ortho
YZ pane draws Z horizontally (``ORTHO_AXIS_COLUMNS["YZ"] == (2, 1)``) while a
standalone YZ projection draws Y horizontally (``AXIS_COLUMNS["YZ"] == (1, 2)``).
Both are right for their own view, and this module is the reason the geometry
layer is asked in **axis columns**: it hands `volume_silhouette` and
`project_flat_record` the pane's actual columns, so one stored geometry is drawn
correctly in either convention with nothing to get wrong.

A **flat** ROI has no extent on the axis it was drawn against, so in a side pane
it is genuinely a segment at its recorded depth. It is drawn **dashed** for
exactly that reason: solid would make it indistinguishable from a volume ROI of
zero thickness, and a user would reasonably read it as constraining Z when it
does not.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import QObject, Qt

from ..core.roi_projection import DEGENERATE_POINT, project_flat_record
from ..core.roi_selection import VOLUME_ROI_TYPES
from ..core.roi_volume import volume_silhouette
from .ortho_view import ORTHO_AXIS_COLUMNS, SIDE_PLANES

__all__ = ["pane_outline", "pane_owns_record", "OrthoRoiOutlines", "OrthoPaneOwner",
           "attach_pane_controllers"]

#: Drawn above the projected scatter but below the primary pane's ROI handles.
_Z_VALUE = 6


def pane_owns_record(record, plane: str) -> bool:
    """Whether *plane*'s own controller draws and edits *record*.

    ⚠ This is the rule that keeps a ROI from being read through the wrong two
    axes. ``roi_visible_in`` gates on family + dataset, **not plane**, so a
    controller on a side pane would otherwise draw a rectangle drawn in XY
    verbatim -- putting its Y coordinate on the pane's Z axis, off in a range
    the data never occupies, and making it draggable there so the next drag
    writes that nonsense back into the record.

    A **volume** ROI is safe in any pane: its geometry is in named data axes,
    so the pane reads it through its own columns. A **flat** ROI is only safe
    in the plane it was drawn in; elsewhere it is edge-on, and the display-only
    dashed outline is the honest way to show it.
    """
    kind = getattr(record, "type", None)
    if kind in VOLUME_ROI_TYPES:
        return True
    context = getattr(record, "context", None) or {}
    return str(context.get("view_plane") or "").upper() == str(plane).upper()


def pane_outline(record, plane: str):
    """``(kind, outline)`` for *record* on an ortho *plane*, or ``None``.

    ``kind`` is ``"full"`` for a shape with real extent on both of the pane's
    axes and ``"line"`` / ``"point"`` when the ROI is edge-on. The caller must
    style those differently -- see the module docstring.
    """
    columns = ORTHO_AXIS_COLUMNS.get(plane)
    if columns is None:
        return None
    h_axis, v_axis = columns

    if getattr(record, "type", None) in VOLUME_ROI_TYPES:
        outline = volume_silhouette(record, h_axis, v_axis)
        if outline is None or len(outline) < 3:
            return None
        return "full", [[float(a), float(b)] for a, b in outline]

    context = getattr(record, "context", None) or {}
    origin = str(context.get("view_plane") or "XY")
    depth = context.get("depth_value")
    if depth is None and record.type == "point":
        point = (getattr(record, "geometry", None) or {}).get("point") or []
        depth = float(point[2]) if len(point) >= 3 else None
    if depth is not None:
        depth = float(depth)
    return project_flat_record(
        record, h_axis, v_axis, origin_plane=origin, depth_value=depth)


class OrthoRoiOutlines:
    """Keeps one curve item per (pane, ROI) and refreshes them together.

    Items are reused across refreshes and only their data is rewritten, because
    a ROI moves far more often than it appears or disappears -- rebuilding the
    items each time is what makes a drag feel heavy.
    """

    def __init__(self, panes: dict, *, has_controller=None) -> None:
        self._panes = panes                       # plane -> PlotWidget
        self._items: dict[tuple[str, str], pg.PlotCurveItem] = {}
        # Callable(plane) -> bool: does that pane have its own ROI controller?
        # Asked per refresh rather than captured, because the controllers are
        # attached lazily when the mode is first entered.
        self._has_controller = has_controller

    def _owned_here(self, plane: str) -> bool:
        try:
            return bool(self._has_controller(plane)) if callable(self._has_controller) else False
        except Exception:
            return False

    def clear(self) -> None:
        for (plane, _rid), item in list(self._items.items()):
            plot = self._panes.get(plane)
            if plot is not None:
                try:
                    plot.getPlotItem().removeItem(item)
                except Exception:
                    pass
        self._items.clear()

    def refresh(self, records, *, color_of=None, visible: bool = True,
                drafts=(), draft_planes=None) -> None:
        """Draw *records* into every side pane; drop the items of any that went.

        A record a pane's **own** controller draws is skipped there, or it would
        appear twice -- once as a grabbable item and once as this decoration,
        which is a confusing picture even when the two agree. A draft is drawn
        everywhere except the pane named by ``draft_planes[id]``: that controller
        already owns the editable item, while this layer still supplies the two
        other projections during seeding and cross-pane editing.
        """
        if not visible:
            self.clear()
            return
        wanted: set[tuple[str, str]] = set()
        for plane in SIDE_PLANES:
            plot = self._panes.get(plane)
            if plot is None:
                continue
            owned = self._owned_here(plane)
            for record in list(records or []) + list(drafts or []):
                draft_plane = (draft_planes or {}).get(getattr(record, "id", ""))
                controller_owns = record not in (drafts or ()) or draft_plane == plane
                if owned and controller_owns and pane_owns_record(record, plane):
                    continue
                result = pane_outline(record, plane)
                if result is None:
                    continue
                kind, outline = result
                pts = np.asarray(outline, dtype=float)
                if pts.ndim != 2 or pts.shape[0] < 1:
                    continue
                if kind == "full":
                    pts = np.vstack([pts, pts[:1]])       # close the outline
                key = (plane, getattr(record, "id", ""))
                wanted.add(key)
                colour = color_of(record) if callable(color_of) else "#ffff00"
                # Dashed for an edge-on flat ROI: solid would be
                # indistinguishable from a volume ROI of zero thickness.
                style = pg.mkPen(
                    colour,
                    width=1.6 if kind == "full" else 1.2,
                    style=(Qt.PenStyle.SolidLine if kind == "full"
                           else Qt.PenStyle.DashLine),
                )
                item = self._items.get(key)
                if item is None:
                    item = pg.PlotCurveItem()
                    item.setZValue(_Z_VALUE)
                    plot.getPlotItem().addItem(item)
                    self._items[key] = item
                item.setPen(style)
                if kind == DEGENERATE_POINT or pts.shape[0] == 1:
                    # A single marker cannot be a curve; give it a short tick so
                    # it is visible at all rather than silently absent.
                    item.setData([pts[0, 0]], [pts[0, 1]])
                else:
                    item.setData(pts[:, 0], pts[:, 1])
        for key in [k for k in self._items if k not in wanted]:
            item = self._items.pop(key)
            plot = self._panes.get(key[0])
            if plot is not None:
                try:
                    plot.getPlotItem().removeItem(item)
                except Exception:
                    pass


class OrthoPaneOwner(QObject):
    """A side pane presented to :class:`RoiOverlayController` as its own view.

    ⚠ A ``QObject``, because the controller parents itself to its owner. It is
    in turn parented to the real window, so the whole chain is destroyed with
    the view rather than outliving it.

    Everything is delegated to the real window except the handful of answers
    that are about *which plane this pane shows* -- so one controller per pane
    gives the side panes drawing, hit-testing and editing with no second
    implementation of any of it.

    ⚠ ``roi_view_columns`` is the load-bearing one. The controller projects
    geometry through it, and an ortho YZ pane plots ``(2, 1)`` where a
    standalone YZ projection plots ``(1, 2)``; without it a shape drawn here
    would write Z data into a Y coordinate -- silently, and past every
    range-based test.
    """

    def __init__(self, owner, plane: str) -> None:
        super().__init__(owner)
        self._owner = owner
        self._plane = str(plane).upper()

    def __getattr__(self, name):                 # everything else is the window's
        # Only reached when normal lookup fails, so QObject's own attributes are
        # never shadowed and the window answers for everything else.
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.__dict__["_owner"], name)

    # -- what makes this pane itself -------------------------------------
    def roi_view_plane(self) -> str:
        return self._plane

    def roi_view_columns(self):
        return ORTHO_AXIS_COLUMNS[self._plane]

    def roi_controller_group(self):
        """Identity shared by the primary and both side-pane controllers.

        A pending volume ROI may move between controllers in this group when
        the user enters another pane.  Returning the real window keeps that
        hand-off local to one orthogonal view; a draft must never jump to a
        different render/scatter window merely because both use the same
        :class:`RoiStore`.
        """
        return self._owner

    def compute_roi_selection(self, record):
        """Rows inside *record*, measured on THIS pane's axes."""
        return self._owner.compute_roi_selection(record, columns=self.roi_view_columns())

    def roi_displays_record(self, record) -> bool:
        """Narrows the controller's scope to what this pane's axes can express."""
        return pane_owns_record(record, self._plane)

    def normalize_roi_record(self, record):
        """Stamp the plane/depth of THIS pane, not the window's primary one.

        ⚠ The owner's implementation reads ``self.roi_view_plane()``, and
        reaching it through ``__getattr__`` binds it to the *window* -- which
        answers "XY" for the whole ortho mode. A shape drawn in the XZ pane was
        therefore filed as an XY shape. Binding the same function to this
        adapter instead makes every ``self.`` in it resolve to the pane.
        """
        function = getattr(type(self._owner), "normalize_roi_record", None)
        if function is None:
            return record
        return function(self, record)

    def roi_depth_center(self):
        return _depth_of(self._owner, self._plane, centre=True)

    def roi_depth_range(self):
        return _depth_of(self._owner, self._plane, centre=False)

    def roi_depths_at(self, points):
        # Data-aware depth is the primary pane's business; a side pane falls
        # back to the view centre rather than inventing a second rule.
        return [None] * len(points or [])


def _depth_of(owner, plane: str, *, centre: bool):
    """The out-of-plane extent for a side pane, from the dataset's own coords."""
    import numpy as np

    from ..core.roi_volume import AXIS_INDEX, PLANE_NORMAL_AXIS

    column = AXIS_INDEX[PLANE_NORMAL_AXIS[plane]]
    getter = getattr(owner, "roi_pane_coords", None)
    coords = getter() if callable(getter) else None
    if coords is None or getattr(coords, "ndim", 0) != 2 or coords.shape[1] <= column:
        return None
    values = coords[:, column]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lo, hi = float(values.min()), float(values.max())
    return 0.5 * (lo + hi) if centre else (lo, hi)


def attach_pane_controllers(owner, panes: dict, *, source_view: str) -> dict:
    """One :class:`RoiOverlayController` per side pane; ``{plane: controller}``.

    They share the owner's ``RoiStore``, so a ROI drawn in any pane is the same
    record everywhere -- the panes differ only in the axes they read it through.
    """
    from .roi_overlay import RoiOverlayController

    controllers: dict = {}
    for plane in SIDE_PLANES:
        plot = panes.get(plane)
        if plot is None:
            continue
        controllers[plane] = RoiOverlayController(
            owner._state.rois,
            OrthoPaneOwner(owner, plane),
            plot,
            plot.getPlotItem(),
            coordinate_space="plot",
            source_view=source_view,
        )
    return controllers
