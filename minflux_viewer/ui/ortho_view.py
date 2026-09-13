"""
minflux_viewer.ui.ortho_view
=============================
Shared orthogonal-view plumbing for the coordinate views.

The layout is a 2x2 grid with the bottom-right cell empty::

            col 0            col 1
    row 0   [ XY ]           [ YZ ]
    row 1   [ XZ ]           (empty)

so the XZ pane sits *exactly* under XY and the YZ pane *exactly* beside it.
A grid is used rather than nested splitters because it is the arrangement
that guarantees that alignment: the panes in one row share a height and the
panes in one column share a width, by construction.

Axis roles, and what is linked to what:

* XY -- horizontal X, vertical Y.  The **primary** pane: it keeps the
  window's existing plot object, so everything that consumes it (ROI
  drawing, scale bar, plot profile, colorbar docking, manual alignment)
  is unaffected by this module.
* YZ -- to the right, **sharing the vertical Y axis** with XY, so its
  horizontal axis is Z.
* XZ -- below, **sharing the horizontal X axis** with XY, so its vertical
  axis is Z.

The two shared axes are native pyqtgraph links.  The third relationship --
XZ's *vertical* Z against YZ's *horizontal* Z -- is transposed, which
pyqtgraph has no API for, and is not a range link at all: the two panes get
their Z from :meth:`OrthoPanes.apply_depth_range`, which puts Z at the **XY
pane's own nm per pixel**.  Every pane is therefore isotropic, so a structure's
proportions on screen are its proportions in the sample and the dataset's Z
scaling factor is visible rather than merely applied.  See that method for why
a common *range* would have been the wrong contract.

This is computed rather than delegated to pyqtgraph's aspect lock, which
resolves a violated aspect by expanding the other axis -- and on a side pane
that axis is *linked to XY*, so the lock could push the primary pane's range
around.  Setting Z explicitly cannot feed back.

Side-pane input is an owner choice.  Scatter keeps them slaved to the primary
pane, while render opts into interaction and propagates a side pane's shared
axis and Z range to the other panes.  In either case every pane stays a
projection over the axis it does not show.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QGridLayout, QWidget

__all__ = [
    "ORTHO_AXIS",
    "AXIS_COLUMNS",
    "ORTHO_AXIS_COLUMNS",
    "axis_columns",
    "PANE_PLANES",
    "PANE_CELLS",
    "SIDE_PLANES",
    "axis_labels",
    "ortho_pane_labels",
    "grid_stretch",
    "plot_item_of",
    "OrthoPanes",
    "OrthoCrosshair",
    "PLACEMENTS",
    "FloatingPaneWindow",
    "floating_geometry",
    "fit_ortho_plot_rects",
]


#: Name of the orthogonal mode as it appears in a projection selector.
ORTHO_AXIS = "Ortho-View"

#: Which columns of an ``(N, 3)`` display-nm array each **standalone**
#: projection plots, as ``(horizontal, vertical)``.  One definition, shared by
#: every view.
AXIS_COLUMNS: dict[str, tuple[int, int]] = {
    "XY": (0, 1),
    "XZ": (0, 2),
    "YZ": (1, 2),
}

#: The same, for a pane of the orthogonal layout.
#:
#: ⚠ The YZ pane is **transposed** against the standalone YZ projection.  On
#: its own, YZ plots Y horizontally and Z vertically.  Sitting beside XY it has
#: to share XY's *vertical* Y axis, so Y becomes its vertical axis and Z its
#: horizontal one — otherwise the shared axis carries Z data inside a Y range,
#: which shows up as the pane's content squeezed into a narrow band rather than
#: as an error.  XZ, sharing XY's horizontal X, is the same either way.
ORTHO_AXIS_COLUMNS: dict[str, tuple[int, int]] = {
    "XY": (0, 1),
    "YZ": (2, 1),
    "XZ": (0, 2),
}

#: The three panes in grid order; the first is the primary one.
PANE_PLANES: tuple[str, str, str] = ("XY", "YZ", "XZ")

#: ``plane -> (row, column)`` in the 2x2 grid.
PANE_CELLS: dict[str, tuple[int, int]] = {
    "XY": (0, 0),
    "YZ": (0, 1),
    "XZ": (1, 0),
}

#: The two panes added by the orthogonal mode.
SIDE_PLANES: tuple[str, str] = ("YZ", "XZ")


def axis_columns(plane: str, *, ortho: bool = False) -> tuple[int, int]:
    """``(horizontal, vertical)`` columns for *plane* in the given mode."""
    table = ORTHO_AXIS_COLUMNS if ortho else AXIS_COLUMNS
    return table.get(plane, (0, 1))


def axis_labels(plane: str, *, ortho: bool = False) -> tuple[str, str]:
    """``(bottom, left)`` axis labels for one pane."""
    horizontal, vertical = axis_columns(plane, ortho=ortho)
    return f"{'XYZ'[horizontal]} (nm)", f"{'XYZ'[vertical]} (nm)"


def ortho_pane_labels(plane: str) -> tuple[str, str]:
    """``(bottom, left)`` labels for one ortho pane, shared axes named once.

    X is shared by XY and XZ and Y by XY and YZ, and a shared axis carries the
    same numbers in both panes, so labelling it twice puts a redundant title in
    the middle of the layout.  X is named at the bottom of the stack and Y down
    its left; the inner copy is blank.  Z is **not** shared -- XZ's is vertical
    and YZ's horizontal -- so both Z labels stay.
    """
    bottom, left = axis_labels(plane, ortho=True)
    if plane == "XY":
        return "", left        # X is named under the XZ pane instead
    if plane == "YZ":
        return bottom, ""      # Y is named beside the XY pane instead
    return bottom, left


def grid_stretch(active: bool, *, primary: int = 3, side: int = 2) -> tuple[
    tuple[int, int], tuple[int, int]
]:
    """``((row0, row1), (col0, col1))`` stretch weights for the pane grid.

    Leaving the orthogonal mode must zero the side weights, not merely hide
    the side panes: a hidden widget still holds its row/column open while a
    stretch factor is set, so the primary pane would keep only ``primary /
    (primary + side)`` of the window after the side panes disappeared.
    """
    if not active:
        return (1, 0), (1, 0)
    return (primary, side), (primary, side)


def plot_item_of(widget):
    """The ``PlotItem`` behind a ``PlotWidget`` or an ``ImageView`` built on one.

    The render view's primary pane is a ``pg.ImageView`` and the scatter
    view's is a ``pg.PlotWidget``; both expose a ``PlotItem``, by different
    names.  Returns ``None`` when the widget has neither.
    """
    getter = getattr(widget, "getPlotItem", None)
    if callable(getter):
        return getter()
    view = getattr(widget, "view", None)
    if view is not None and hasattr(view, "vb"):
        return view
    return None


def _view_box_of(widget):
    item = plot_item_of(widget)
    return getattr(item, "vb", None) if item is not None else None


class OrthoPanes(QObject):
    """Own the 2x2 pane grid, its visibility, and the inter-pane axis links.

    The owner keeps its own plot objects; this class only arranges them and
    wires their ranges together.  ``set_active`` is the single switch.
    """

    #: Stretch of the primary pane against a Z pane, applied on both axes:
    #: 3:2 gives XY 0.6 x 0.6 of the grid, YZ 0.4 x 0.6 and XZ 0.6 x 0.4.
    #: Because the panes are isotropic, a side pane's thickness *is* the Z
    #: range it can show -- 0.4 rather than 0.25 is what buys back the Z that
    #: would otherwise be clipped at a deep zoom.
    PRIMARY_STRETCH = 3
    SIDE_STRETCH = 2

    #: Gap between panes, in px.
    SPACING_PX = 3

    #: Pinned axis metrics, in px.  pyqtgraph maps a linked range through the
    #: ratio of the two panes' *screen* widths, so panes whose tick labels
    #: differ in width (Z's "200" against Y's "12000") get different plot
    #: rectangles and therefore different data ranges at the same nm/px --
    #: measured at a stable 36-74 nm, i.e. one feature at two different
    #: screen positions.  Pinning makes the rectangles identical and the
    #: disagreement exactly zero.
    AXIS_WIDTH_PX = 62
    AXIS_HEIGHT_PX = 30

    #: Slack around the depth range, as a fraction of its span, so points at
    #: the extremes are not drawn on the pane border.
    DEPTH_PADDING_FRAC = 0.04

    def __init__(
        self,
        container: QWidget,
        panes: dict[str, QWidget],
        *,
        parent: QObject | None = None,
        on_pane_closed=None,
        on_activated=None,
        on_side_range_changed=None,
        interactive_sides: bool = False,
    ) -> None:
        super().__init__(parent if parent is not None else container)
        self._container = container
        self._panes = dict(panes)
        self._active = False
        self._linked = False
        self._syncing = False
        self._depth_scale: float | None = None
        self._depth_request: tuple[float, float] | None = None
        self._depth_clipped = False
        self._depth_centre: float | None = None
        self._placement = "embedded"
        self._on_pane_closed = on_pane_closed
        self._on_activated = on_activated
        self._side_range_callback = on_side_range_changed
        # ⚠ Opt-in. An interactive side pane pushes its range onto the primary
        # with setXRange/setYRange, and an explicit range assignment *disables
        # auto-range* — which is the scatter view's "showing everything" state
        # and what its viewport crop keys on. Turning this on there made the
        # crop unable to lift and the ranges run away.
        self._interactive_sides = bool(interactive_sides)
        self._hosts: dict[str, QWidget] = {}
        self._owner_filter_installed = False
        self._sticky = True
        self._aligning = False

        # Reuse a grid the owner already put on the container, so the primary
        # pane can be placed at build time and the side panes added lazily on
        # first use -- a window that never enters the mode then pays nothing.
        layout = container.layout()
        if not isinstance(layout, QGridLayout):
            layout = QGridLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        for plane, widget in self._panes.items():
            row, column = PANE_CELLS[plane]
            layout.addWidget(widget, row, column)
        self._layout = layout

        for plane in SIDE_PLANES:
            view_box = self.view_box(plane)
            if view_box is not None:
                # With interaction on, the side panes pan and zoom like the
                # primary one and _on_side_range_changed routes the result to
                # the panes that share those axes. Off, they are slaved: the
                # range comes from the primary and the data, never the mouse.
                view_box.setMouseEnabled(x=self._interactive_sides,
                                         y=self._interactive_sides)
                view_box.setMenuEnabled(False)
                # ⚠ Never auto-range a side pane. Both of its axes are driven
                # (the shared one from the primary, Z from apply_depth_range),
                # and with interaction on, an auto-range triggered merely by
                # drawing the projection pushed a new range back onto the
                # primary — rendering the side panes moved the XY view.
                try:
                    view_box.disableAutoRange()
                except Exception:
                    pass
                # The shared Z scale is derived from these panes' pixel extents,
                # so it has to be re-derived whenever they change size --
                # including the relayout that follows entering the mode, where
                # the first apply necessarily ran against stale geometry.
                view_box.sigResized.connect(self._on_side_resized)

        self.set_active(False)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    def pane(self, plane: str) -> QWidget | None:
        return self._panes.get(plane)

    def side_panes(self) -> list[QWidget]:
        return [self._panes[p] for p in SIDE_PLANES if p in self._panes]

    def view_box(self, plane: str):
        widget = self._panes.get(plane)
        return _view_box_of(widget) if widget is not None else None

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    @property
    def placement(self) -> str:
        return self._placement

    def set_placement(self, placement: str) -> None:
        """Move the side panes between the grid and their own windows.

        The panes themselves are *moved*, not duplicated, so both arrangements
        share one projection path, one set of links and one isotropy rule.
        """
        placement = placement if placement in PLACEMENTS else "embedded"
        if placement == self._placement:
            return
        self._placement = placement
        if self._active:
            self.set_active(False)
            self.set_active(True)

    def set_sticky(self, sticky: bool) -> None:
        """Whether floating windows follow the owner when it moves."""
        self._sticky = bool(sticky)

    def set_active(self, active: bool) -> None:
        """Show or hide the side panes and (un)link the shared axes."""
        active = bool(active)
        floating = active and self._placement == "floating"
        if floating:
            self._float_panes()
        else:
            self._embed_panes()
        for plane in SIDE_PLANES:
            widget = self._panes.get(plane)
            if widget is not None:
                widget.setVisible(active)

        # Floating leaves the side cells empty, so the primary pane takes the
        # whole page exactly as it does outside the mode.
        (row0, row1), (col0, col1) = grid_stretch(
            active and not floating,
            primary=self.PRIMARY_STRETCH, side=self.SIDE_STRETCH,
        )
        self._layout.setRowStretch(0, row0)
        self._layout.setRowStretch(1, row1)
        self._layout.setColumnStretch(0, col0)
        self._layout.setColumnStretch(1, col1)
        self._layout.setSpacing(self.SPACING_PX if active else 0)

        if active:
            self._pin_axis_metrics()
            self._link()
        else:
            self._unlink()
        self._active = active
        if floating:
            # Fit on entry only. A later sticky re-align must not resize the
            # owner again -- that would fight a window the user just sized.
            #
            # ⚠ Deferred a turn: the grid stretch was just set to give the
            # primary pane the whole page, and Qt has not applied it yet.
            # Measuring now reports the pane at its old 0.6 share (577 px where
            # it is about to be 960), and the planner then shrinks the window
            # to fit a primary that was never that small.
            QTimer_singleShot(0, self._enter_floating)

    # ------------------------------------------------------------------
    # Floating hosts
    # ------------------------------------------------------------------

    def _owner_window(self) -> QWidget | None:
        try:
            return self._container.window()
        except Exception:
            return None

    def _float_panes(self) -> None:
        if self._hosts:
            return
        for plane in SIDE_PLANES:
            pane = self._panes.get(plane)
            if pane is None:
                continue
            self._layout.removeWidget(pane)
            host = FloatingPaneWindow(
                plane, pane,
                on_close=self._on_host_closed,
                on_activated=self._on_activated,
            )
            self._hosts[plane] = host
            host.show()
        owner = self._owner_window()
        if owner is not None and not self._owner_filter_installed:
            owner.installEventFilter(self)
            self._owner_filter_installed = True

    def _embed_panes(self) -> None:
        if not self._hosts:
            return
        owner = self._owner_window()
        if owner is not None and self._owner_filter_installed:
            owner.removeEventFilter(self)
            self._owner_filter_installed = False
        for plane, host in list(self._hosts.items()):
            pane = self._panes.get(plane)
            if pane is not None:
                # Release the fixed size the floating fit pinned on it, or the
                # pane would refuse to follow the grid's stretch factors.
                _release_fixed_size(pane)
                row, column = PANE_CELLS[plane]
                self._layout.addWidget(pane, row, column)
            host._on_close = None
            try:
                host.close()
                host.deleteLater()
            except Exception:
                pass
        self._hosts.clear()

    def hosts(self):
        """The floating windows currently in use, if any."""
        return list(self._hosts.values())

    def _on_host_closed(self, plane: str) -> None:
        """One side window was closed by the user -- hand it to the owner.

        A single floating window with no partner is a state nothing else has a
        name for, so the owner decides; the render view leaves the mode
        entirely, which is what Fiji's Orthogonal_Views does.
        """
        if self._aligning:
            return
        if self._on_pane_closed is not None:
            self._on_pane_closed(plane)
            return
        self._placement = "embedded"
        QTimer_singleShot(0, self._fold_back)

    def _fold_back(self) -> None:
        if self._active:
            self.set_active(False)
            self.set_active(True)

    def _chrome(self) -> dict[str, tuple[int, int, int, int]]:
        """Per-window overhead around the plot, as ``(left, top, right, bottom)``.

        Measured on each side rather than as a total: a side window carries its
        title bar above the plot and its out-of-plane slider below it, and a
        planner that treats the whole overhead as one number puts the bottom
        window past the screen edge by exactly the slider's height.
        """
        out: dict[str, tuple[int, int, int, int]] = {}
        for plane in PANE_PLANES:
            window = (self._owner_window() if plane == "XY"
                      else self._hosts.get(plane))
            plot = self._global_plot_rect(plane)
            if window is None or plot is None:
                continue
            frame = window.frameGeometry()
            out[plane] = (
                max(plot[0] - frame.x(), 0),
                max(plot[1] - frame.y(), 0),
                max(frame.x() + frame.width() - (plot[0] + plot[2]), 0),
                max(frame.y() + frame.height() - (plot[1] + plot[3]), 0),
            )
        return out

    def _available_rect(self) -> tuple[int, int, int, int] | None:
        """Usable area of the screen the owner window is on."""
        owner = self._owner_window()
        if owner is None:
            return None
        try:
            screen = owner.screen()
            if screen is None:
                from PyQt6.QtWidgets import QApplication
                screen = QApplication.primaryScreen()
            rect = screen.availableGeometry()
        except Exception:
            return None
        return rect.x(), rect.y(), rect.width(), rect.height()

    def fit_to_screen(self) -> bool:
        """Shrink the owner and place the side windows so all three fit.

        Floating needs more screen than embedded -- the same three plot areas,
        but each now with its own chrome and the side panes outside the primary
        rather than inside it -- so without this the arrangement simply runs off
        the bottom of the monitor (measured: the XZ window's plot ended 214 px
        below the available area).
        """
        if not self._hosts:
            return False
        available = self._available_rect()
        primary = self._global_plot_rect("XY")
        if available is None or primary is None:
            return False
        plan = fit_ortho_plot_rects(
            available, primary,
            ratio=self.SIDE_STRETCH / float(self.PRIMARY_STRETCH),
            chrome=self._chrome(),
        )
        self._aligning = True
        try:
            self._place_owner(plan["XY"])
            for plane, host in self._hosts.items():
                self._place_host(host, plane, plan[plane])
        finally:
            self._aligning = False
        # The owner moved and resized, so the exact relative placement is
        # re-derived from where it actually landed.
        self.align_floating()
        return True

    def _enter_floating(self) -> None:
        if not (self._active and self._placement == "floating" and self._hosts):
            return
        self.align_floating()
        self.fit_to_screen()

    def _place_owner(self, target) -> None:
        """Resize the owner so its plot rect is *target*, keeping the data view.

        ⚠ The view range is captured and restored around the resize. A
        pyqtgraph view is ranged in data units, so a smaller widget showing the
        same nm/px shows *less data* -- the user would lose their place the
        moment the mode is enabled. Restoring the range keeps the same region
        on screen at a coarser scale, which is what "keep the zoom" means for a
        window that just got smaller.
        """
        owner = self._owner_window()
        view_box = self.view_box("XY")
        if owner is None or view_box is None:
            return
        try:
            (x0, x1), (y0, y1) = view_box.viewRange()
        except Exception:
            x0 = x1 = y0 = y1 = None
        tx, ty, tw, th = target
        current = self._global_plot_rect("XY")
        for _pass in range(self.PLACEMENT_PASSES):
            if current is None:
                break
            cw, ch = current[2], current[3]
            if (cw, ch) != (tw, th):
                geo = owner.geometry()
                # ⚠ Never grow the owner. The plan is shrink-only by
                # construction, so a delta that asks for more means the plot
                # rect was measured mid-layout — and acting on it grew an
                # 980 px window to 1145 instead of shrinking it.
                owner.resize(
                    max(min(geo.width() + (tw - cw), geo.width()), 200),
                    max(min(geo.height() + (th - ch), geo.height()), 200),
                )
            current = self._global_plot_rect("XY") or current
            frame = owner.frameGeometry()
            owner.move(frame.x() + (tx - current[0]), frame.y() + (ty - current[1]))
            current = self._global_plot_rect("XY") or current
            if (current[2], current[3]) == (tw, th):
                break
        if x0 is not None:
            try:
                view_box.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
            except Exception:
                pass

    def align_floating(self) -> None:
        """Place each floating window so its PLOT AREA lines up with XY's.

        ⚠ Matching window *frames* would leave the data misaligned by whatever
        the axis gutters and the window decorations happen to be -- which is
        the whole difficulty of the floating arrangement, and why ImageJ can
        only approximate it (``arrangeWindows`` polls for the windows to exist,
        then positions frames). Here the target is computed for the plot
        rectangle and the frame offset is measured back out of the shown
        window, so the answer is exact rather than approximate.
        """
        if not self._hosts or self._aligning:
            return
        owner_plot = self._global_plot_rect("XY")
        if owner_plot is None:
            return
        self._aligning = True
        try:
            thickness = self._floating_thickness(owner_plot)
            for plane, host in self._hosts.items():
                target = floating_geometry(owner_plot, plane, thickness=thickness)
                self._place_host(host, plane, target)
        finally:
            self._aligning = False

    def _floating_thickness(self, owner_plot) -> int:
        """Z-pane plot thickness, from the same 0.6/0.4 ratio as the grid."""
        _x, _y, w, h = owner_plot
        ratio = self.SIDE_STRETCH / float(self.PRIMARY_STRETCH)
        return max(int(round(min(w, h) * ratio)), 80)

    #: Correction passes allowed when fitting a floating host's plot rect.
    #: ⚠ One pass is not enough: a resize does not move the plot rectangle by
    #: exactly the amount asked for (layout rounding, minimum sizes and the
    #: frame all absorb some of it), and a single pass left the YZ pane 2 px
    #: short of XY -- invisible on screen but a real misalignment, and it only
    #: showed up under full-suite timing. The loop converges to an exact match
    #: in a couple of iterations instead of sleeping like ImageJ's 2.5 s poll.
    PLACEMENT_PASSES = 4

    def _place_host(self, host, plane: str, target) -> None:
        """Move/resize *host* so its plot rect equals *target* exactly.

        ⚠ The **pane** is pinned, not the window. Resizing the host and hoping
        the plot rect follows lets the window manager and the layout have a
        vote, and they take some of it: a negotiated fit stalled 2 px short on
        the YZ pane under full-suite timing. 2 px sounds cosmetic and is not —
        YZ's height *is* the linked Y axis, so a 0.3 % height mismatch remaps
        the Y range and puts the same feature at different screen positions,
        the very drift the pinned axis metrics exist to prevent. A fixed size
        on the pane sets its min and max together, so the host has to adopt it.
        """
        tx, ty, tw, th = target
        pane = self._panes.get(plane)
        current = self._global_plot_rect(plane)
        if pane is None or current is None:
            host.setGeometry(tx, ty, tw, th)
            return
        for _pass in range(self.PLACEMENT_PASSES):
            cw, ch = current[2], current[3]
            if (cw, ch) != (tw, th):
                size = pane.size()
                pane.setFixedSize(max(size.width() + (tw - cw), 80),
                                  max(size.height() + (th - ch), 80))
                host.adjustSize()
            current = self._global_plot_rect(plane) or current
            frame = host.frameGeometry()
            host.move(frame.x() + (tx - current[0]), frame.y() + (ty - current[1]))
            current = self._global_plot_rect(plane) or current
            if (current[0], current[1], current[2], current[3]) == (tx, ty, tw, th):
                return

    def _global_plot_rect(self, plane: str):
        """One pane's plot rectangle in global screen pixels."""
        view_box = self.view_box(plane)
        widget = self._panes.get(plane)
        if view_box is None or widget is None or not widget.isVisible():
            return None
        try:
            rect = view_box.sceneBoundingRect()
            viewport = widget.viewport() if hasattr(widget, "viewport") else widget
            origin = viewport.mapToGlobal(_QPoint(int(rect.x()), int(rect.y())))
        except Exception:
            return None
        return origin.x(), origin.y(), int(rect.width()), int(rect.height())

    def eventFilter(self, obj, event):
        """Keep the floating windows with the owner while sticky."""
        try:
            kind = event.type()
        except Exception:
            return False
        if self._sticky and self._hosts and kind in _OWNER_FOLLOW_EVENTS():
            QTimer_singleShot(0, self.align_floating)
        return False

    def depth_pixels(self) -> dict[str, float]:
        """Each side pane's free-axis extent, in *plot-area* pixels.

        XZ carries Z vertically and YZ horizontally, so the two are different
        measurements of the same axis -- which is precisely why their scales
        drift apart unless they are equalised.
        """
        out: dict[str, float] = {}
        for plane, vertical in (("XZ", True), ("YZ", False)):
            view_box = self.view_box(plane)
            if view_box is None:
                continue
            try:
                rect = view_box.sceneBoundingRect()
            except Exception:
                continue
            extent = float(rect.height() if vertical else rect.width())
            if extent > 1.0:
                out[plane] = extent
        return out

    def primary_scale_nm_per_px(self) -> float | None:
        """The XY pane's nm per plot-area pixel, or ``None`` if unmeasurable.

        XY is aspect-locked, so its X and Y scales are equal and either does.
        """
        view_box = self.view_box("XY")
        if view_box is None:
            return None
        try:
            rect = view_box.sceneBoundingRect()
            (x0, x1), _ = view_box.viewRange()
        except Exception:
            return None
        if rect.width() <= 1.0 or x1 <= x0:
            return None
        return float(x1 - x0) / float(rect.width())

    def apply_depth_range(self, lo: float, hi: float) -> bool:
        """Centre both side panes on ``[lo, hi]`` of Z **at the XY scale**.

        The panes are fully isotropic: Z is given the same nm per pixel as X
        and Y, so a structure's proportions on screen are its proportions in
        the sample, and the Z scaling factor is visible in the picture.  Before
        this each pane auto-fitted Z to whatever it was handed, so halving
        ``cali.z_scaling_factor`` halved the data and changed the display not
        at all -- the factor was applied but not observable.

        ⚠ Equal *ranges* would not be equal *scales*, which is why the range
        is derived per pane rather than shared.  XZ shows Z down its height and
        YZ across its width, and those pixel extents are set by the window's
        proportions -- measured 210 px against 162 px on a square window, so
        one Z range rendered at 7.36 against 9.55 nm/px and a feature came out
        30% taller in one pane than it was wide in the other.  Each pane gets
        the range its own pixel extent earns at the common scale, about a
        common centre.

        The price of isotropy is that a pane shows only ``scale x pixels`` of
        Z, so a tall structure at a deep zoom is **clipped** rather than
        squeezed.  :attr:`depth_clipped` reports that, so a caller can say so
        rather than let a partial projection pass for the whole one.
        """
        pixels = self.depth_pixels()
        scale = self.primary_scale_nm_per_px()
        if len(pixels) < 2 or scale is None:
            return False
        lo, hi = float(lo), float(hi)
        if not (hi > lo):
            centre = 0.5 * (lo + hi)
            lo, hi = centre - 0.5, centre + 0.5
        span = (hi - lo) * (1.0 + 2.0 * self.DEPTH_PADDING_FRAC)
        # The crosshair is the view's anchor when there is one: zooming should
        # close in on the marked point, not drift back to the middle of the
        # data. Falls back to the data centre before one is placed.
        centre = self._depth_centre
        if centre is None:
            centre = 0.5 * (lo + hi)

        self._syncing = True
        try:
            half = 0.5 * scale * pixels["XZ"]
            self.view_box("XZ").setYRange(centre - half, centre + half, padding=0)
            half = 0.5 * scale * pixels["YZ"]
            self.view_box("YZ").setXRange(centre - half, centre + half, padding=0)
        except Exception:
            return False
        finally:
            self._syncing = False
        self._depth_scale = scale
        self._depth_request = (lo, hi)
        self._depth_clipped = span > scale * min(pixels["XZ"], pixels["YZ"])
        return True

    def set_depth_centre(self, centre: float | None) -> None:
        """Anchor the side panes' Z on *centre* (the crosshair), or the data."""
        self._depth_centre = None if centre is None else float(centre)

    @property
    def depth_centre(self) -> float | None:
        """The Z the side panes are centred on, or ``None`` for the data centre."""
        return self._depth_centre

    @property
    def depth_clipped(self) -> bool:
        """Whether the last applied Z range did not fit at the XY scale."""
        return self._depth_clipped

    def dispose_hosts(self) -> None:
        self._embed_panes()

    def dispose(self) -> None:
        """Disconnect every receiver this object owns, before the panes die.

        The project's lifecycle rule is that a closing UI disconnects its own
        receivers rather than relying on Qt ordering: a queued ``sigResized``
        or ``sigRangeChanged`` delivered while the window is tearing down would
        reach a half-dead object, which is the class of fault that surfaces
        later as a native abort somewhere unrelated.
        """
        self._unlink()
        for plane in SIDE_PLANES:
            view_box = self.view_box(plane)
            if view_box is None:
                continue
            try:
                view_box.sigResized.disconnect(self._on_side_resized)
            except TypeError:
                pass
        self._depth_request = None
        self._active = False
        self._embed_panes()

    def _copy_shared_ranges(self, *_args) -> None:
        """Give each side pane the primary pane's shared-axis range verbatim."""
        primary = self.view_box("XY")
        if primary is None or self._syncing:
            return
        self._syncing = True
        try:
            (x0, x1), (y0, y1) = primary.viewRange()
            xz = self.view_box("XZ")
            if xz is not None:
                xz.setXRange(x0, x1, padding=0)
            yz = self.view_box("YZ")
            if yz is not None:
                yz.setYRange(y0, y1, padding=0)
        except Exception:
            pass
        finally:
            self._syncing = False

    def _on_side_range_changed(self, *args) -> None:
        """Push a side pane's own pan/zoom out to the panes that share its axes.

        A side pane owns one shared axis (XZ owns X, YZ owns Y) and one Z axis
        that the other side pane also shows.  Zooming it therefore has to reach
        the primary pane on the shared axis and the opposite side pane on Z --
        the native links only run the other way, and are not used at all when
        floating.
        """
        if self._syncing or not self._active:
            return
        source = args[0] if args else None
        plane = next((p for p in SIDE_PLANES if self.view_box(p) is source), None)
        if plane is None:
            return
        primary = self.view_box("XY")
        other = "YZ" if plane == "XZ" else "XZ"
        other_box = self.view_box(other)
        side = self.view_box(plane)
        if primary is None or other_box is None or side is None:
            return
        self._syncing = True
        z_range = None
        try:
            (h0, h1), (v0, v1) = side.viewRange()
            if plane == "XZ":
                primary.setXRange(h0, h1, padding=0)      # shared X
                other_box.setXRange(v0, v1, padding=0)    # its Z -> YZ's Z
                z_range = (float(v0), float(v1))
            else:
                primary.setYRange(v0, v1, padding=0)      # shared Y
                other_box.setYRange(h0, h1, padding=0)    # its Z -> XZ's Z
                z_range = (float(h0), float(h1))
        except Exception:
            pass
        finally:
            self._syncing = False
        # The owner, rather than this layout controller, owns the semantic
        # depth marker. Keep the guard raised while it moves the drawn lines:
        # an InfiniteLine geometry update can itself produce a range signal,
        # and that decoration must not be mistaken for another mouse gesture.
        if z_range is not None and callable(self._side_range_callback):
            self._syncing = True
            try:
                self._side_range_callback(plane, z_range)
            except Exception:
                pass
            finally:
                self._syncing = False

    def _on_primary_range_changed(self, *_args) -> None:
        """Re-apply Z at the XY pane's new scale."""
        if self._syncing or not self._active:
            return
        if self._placement == "floating":
            self._copy_shared_ranges()
        if self._depth_request is not None:
            self.apply_depth_range(*self._depth_request)

    def _on_side_resized(self, *_args) -> None:
        """Re-apply the last depth range at the new pixel extents.

        Cheap -- it sets two ranges and redraws nothing -- so it runs inline
        rather than through the owner's debounce, which would leave the panes
        at visibly unequal scales until it fired.
        """
        if self._syncing or not self._active or self._depth_request is None:
            return
        self.apply_depth_range(*self._depth_request)

    @property
    def depth_scale_nm_per_px(self) -> float | None:
        """The common Z scale last applied, or ``None`` before the first."""
        return self._depth_scale

    # ------------------------------------------------------------------
    # Axis geometry and links
    # ------------------------------------------------------------------

    def _pin_axis_metrics(self) -> None:
        for widget in self._panes.values():
            item = plot_item_of(widget)
            if item is None:
                continue
            try:
                item.getAxis("left").setWidth(self.AXIS_WIDTH_PX)
                item.getAxis("bottom").setHeight(self.AXIS_HEIGHT_PX)
            except Exception:
                pass

    def _link(self) -> None:
        if self._linked:
            return
        primary = self.view_box("XY")
        xz = self.view_box("XZ")
        yz = self.view_box("YZ")
        if primary is None:
            return
        # ⚠ Across windows the shared axis is COPIED, not linked.
        # pyqtgraph maps a linked range through the ratio of the two panes'
        # plot rectangles, which is exact inside one layout (same row, same
        # height by construction) but only best-effort across top-level
        # windows -- the window manager gets a vote, and a 2 px height
        # difference remapped the Y range by 0.6 %, i.e. **27.9 nm of data
        # misalignment**, silently. Copying the range makes the data
        # agreement exact whatever the pixels do; the cost is that the side
        # pane's own scale may differ from XY's by that same pixel error, so
        # it is very slightly anisotropic instead of very slightly wrong about
        # where things are. For a view whose purpose is reading a feature's
        # position across panes, that is the right way round.
        if self._placement == "floating":
            primary.sigRangeChanged.connect(self._copy_shared_ranges)
            self._copy_shared_ranges()
        else:
            if xz is not None:
                xz.setXLink(primary)      # both horizontal axes are X
            if yz is not None:
                yz.setYLink(primary)      # both vertical axes are Y
        # Isotropy ties Z to the XY scale, so a zoom has to carry Z with it.
        # Inline, not through the owner's redraw debounce: the Z *scale* must
        # track the drag, while re-projecting the cropped *content* can wait
        # for it to settle. Neither side pane's Z feeds back into XY, so this
        # cannot loop.
        primary.sigRangeChanged.connect(self._on_primary_range_changed)
        if self._interactive_sides:
            for plane in SIDE_PLANES:
                side = self.view_box(plane)
                if side is not None:
                    side.sigRangeChanged.connect(self._on_side_range_changed)
        self._linked = True

    def _unlink(self) -> None:
        if not self._linked:
            return
        primary = self.view_box("XY")
        if primary is not None:
            for slot in (self._on_primary_range_changed, self._copy_shared_ranges):
                try:
                    primary.sigRangeChanged.disconnect(slot)
                except TypeError:
                    pass
        for plane in SIDE_PLANES:
            side = self.view_box(plane)
            if side is None:
                continue
            try:
                side.sigRangeChanged.disconnect(self._on_side_range_changed)
            except TypeError:
                pass
        for plane, setter in (("XZ", "setXLink"), ("YZ", "setYLink")):
            view_box = self.view_box(plane)
            if view_box is None:
                continue
            try:
                getattr(view_box, setter)(None)
            except Exception:
                pass
        self._linked = False


class OrthoCrosshair:
    """A 3-D position marker drawn across the ortho panes.

    ⚠ **It is an indicator, not a slice selector, and the difference is not
    cosmetic.**  In ImageJ's ``Orthogonal_Views`` the crosshair *navigates*
    precisely because the side views are single slices -- ``updateXZView()``
    copies one row of pixels across all depth slices -- so moving ``crossLoc``
    changes what those views show.  Our side panes are projections over the
    whole third axis, so moving this marker changes nothing in them.  What it
    does instead is answer the question a projection cannot: *where* along the
    collapsed axis a feature sits, by showing the same point in all three
    panes at once.  Turning it into a true navigator means adding a slab mode
    first, and then this class gains the callback that re-renders on move.

    Each pane gets one vertical and one horizontal line, positioned from the
    pane's own two columns of the shared ``(x, y, z)`` point -- so the lines
    cross on the same physical location in every pane.
    """

    def __init__(self, panes: dict[str, QWidget], *, color=(255, 90, 90, 190)) -> None:
        self._panes = dict(panes)
        self._color = color
        self._lines: dict[str, tuple] = {}
        self._point: tuple[float, float, float] | None = None
        self._visible = False

    # -- state ---------------------------------------------------------

    @property
    def point(self) -> tuple[float, float, float] | None:
        return self._point

    @property
    def visible(self) -> bool:
        return self._visible

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        if not self._visible:
            self._remove_lines()
        else:
            self.refresh()

    def set_point(self, point) -> None:
        """Place the marker at ``(x, y, z)`` in display nm."""
        if point is None:
            self._point = None
        else:
            self._point = (float(point[0]), float(point[1]), float(point[2]))
        self.refresh()

    def update_from_pane(self, plane: str, h: float, v: float) -> tuple[float, float, float] | None:
        """Move the two coordinates *plane* shows, keeping the third.

        A click in one pane fixes the two axes that pane draws and says nothing
        about the one it projects over, so the third coordinate is carried
        forward rather than reset.
        """
        if plane not in ORTHO_AXIS_COLUMNS:
            return self._point
        point = list(self._point if self._point is not None else (0.0, 0.0, 0.0))
        horizontal, vertical = ORTHO_AXIS_COLUMNS[plane]
        point[horizontal] = float(h)
        point[vertical] = float(v)
        self.set_point(point)
        return self._point

    # -- drawing -------------------------------------------------------

    def refresh(self) -> None:
        if not self._visible or self._point is None:
            self._remove_lines()
            return
        import pyqtgraph as pg

        for plane, widget in self._panes.items():
            item = plot_item_of(widget)
            if item is None or not widget.isVisible():
                continue
            horizontal, vertical = ORTHO_AXIS_COLUMNS[plane]
            lines = self._lines.get(plane)
            if lines is None:
                pen = pg.mkPen(self._color, width=1, style=_DASH_STYLE())
                v_line = pg.InfiniteLine(angle=90, movable=False, pen=pen)
                h_line = pg.InfiniteLine(angle=0, movable=False, pen=pen)
                for line in (v_line, h_line):
                    line.setZValue(20)
                    # ignoreBounds: a view-spanning decoration must never take
                    # part in an auto-range fit, or it grows the view it is
                    # drawn in (the same rule as the histogram's filter label).
                    item.addItem(line, ignoreBounds=True)
                lines = (v_line, h_line)
                self._lines[plane] = lines
            lines[0].setPos(self._point[horizontal])
            lines[1].setPos(self._point[vertical])
            for line in lines:
                line.setVisible(True)

    def _remove_lines(self) -> None:
        for plane, lines in list(self._lines.items()):
            for line in lines:
                try:
                    line.setVisible(False)
                except Exception:
                    pass

    def dispose(self) -> None:
        for plane, lines in list(self._lines.items()):
            item = plot_item_of(self._panes.get(plane))
            for line in lines:
                try:
                    if item is not None:
                        item.removeItem(line)
                except Exception:
                    pass
        self._lines.clear()


def _DASH_STYLE():
    from PyQt6.QtCore import Qt
    return Qt.PenStyle.DashLine


# ---------------------------------------------------------------------------
# Floating placement (the ImageJ Orthogonal_Views arrangement)
# ---------------------------------------------------------------------------

#: Where a side pane lives.  ``embedded`` is the 2x2 grid; ``floating`` puts
#: each side pane in its own top-level window beside and below the owner, the
#: way ImageJ's ``Orthogonal_Views`` arranges its ``xz_image``/``yz_image``.
PLACEMENTS: tuple[str, str] = ("embedded", "floating")


def fit_ortho_plot_rects(
    available: tuple[int, int, int, int],
    primary_plot: tuple[int, int, int, int],
    *,
    ratio: float,
    gap: int = 8,
    chrome: dict[str, tuple[int, int, int, int]] | None = None,
    min_side: int = 100,
) -> dict[str, tuple[int, int, int, int]]:
    """Plot rectangles for all three windows, fitted onto one monitor.

    Floating needs more screen than embedded: the three plot areas are the
    same, but each now carries its own window chrome, and the side panes sit
    *outside* the primary instead of inside it.  Left alone the arrangement
    simply runs off the bottom of the screen.  So the primary is scaled down
    until the whole set fits, keeping its shape and the side ratio.

    ``chrome`` is per window as ``{plane: (left, top, right, bottom)}``.
    ⚠ The split matters and a single width/height will not do: a side window
    carries its title bar **above** the plot and its out-of-plane slider
    **below** it, so treating the total as all-above puts the bottom window
    past the screen edge by exactly the slider's height.

    Scaling is **down only** -- enabling the mode should not grow a window the
    user sized.
    """
    ax, ay, aw, ah = available
    _px, _py, pw, ph = primary_plot
    chrome = chrome or {}

    def side(plane, index):
        box = chrome.get(plane) or (0, 0, 0, 0)
        return int(box[index])

    xy_l, xy_t, xy_r, xy_b = (side("XY", i) for i in range(4))
    yz_l, _yz_t, yz_r, _yz_b = (side("YZ", i) for i in range(4))
    _xz_l, xz_t, _xz_r, xz_b = (side("XZ", i) for i in range(4))

    width_budget = aw - gap - xy_l - xy_r - yz_l - yz_r
    height_budget = ah - gap - xy_t - xy_b - xz_t - xz_b

    thickness0 = max(ratio * min(pw, ph), 1.0)
    scale = 1.0
    if pw + thickness0 > 0:
        scale = min(scale, width_budget / float(pw + thickness0))
    if ph + thickness0 > 0:
        scale = min(scale, height_budget / float(ph + thickness0))
    scale = max(min(scale, 1.0), 0.05)

    p_w = max(int(round(pw * scale)), min_side)
    p_h = max(int(round(ph * scale)), min_side)
    thickness = max(int(round(thickness0 * scale)), min_side)

    x0 = ax + xy_l
    y0 = ay + xy_t
    return {
        "XY": (x0, y0, p_w, p_h),
        "YZ": (x0 + p_w + xy_r + gap + yz_l, y0, thickness, p_h),
        "XZ": (x0, y0 + p_h + xy_b + gap + xz_t, p_w, thickness),
    }


class FloatingPaneWindow(QWidget):
    """A top-level window hosting one side pane.

    Deliberately a bare host: the pane inside it is the *same* widget the
    embedded grid uses, moved rather than duplicated, so the projection, the
    axis links and the isotropy all keep working with no second code path.
    """

    def __init__(self, plane: str, pane: QWidget, *, on_close=None,
                 on_activated=None) -> None:
        super().__init__(None)
        from PyQt6.QtWidgets import QVBoxLayout

        self._plane = plane
        self._on_close = on_close
        self._on_activated = on_activated
        self.setWindowTitle(f"{plane} view")
        self.setWindowFlags(Qt_WindowType_Window())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(pane, 1)
        self._layout = layout

    def add_footer(self, widget) -> None:
        """Put an owner-supplied control strip under the pane."""
        self._layout.addWidget(widget, 0)

    @property
    def plane(self) -> str:
        return self._plane

    def changeEvent(self, event) -> None:
        """Raise the whole set when this window is activated.

        Three windows that belong to one view should not be separated by
        another window landing between them.
        """
        from PyQt6.QtCore import QEvent
        if (event.type() == QEvent.Type.ActivationChange
                and self.isActiveWindow() and self._on_activated is not None):
            self._on_activated()
        super().changeEvent(event)

    def closeEvent(self, event) -> None:
        # Closing one side window leaves the mode half-present, so it is
        # reported rather than swallowed: the owner decides what that means.
        if self._on_close is not None:
            self._on_close(self._plane)
        super().closeEvent(event)


def Qt_WindowType_Window():
    from PyQt6.QtCore import Qt
    return Qt.WindowType.Window


def floating_geometry(
    owner_plot: tuple[int, int, int, int],
    plane: str,
    *,
    thickness: int,
    gap: int = 8,
) -> tuple[int, int, int, int]:
    """Where one floating side window's *plot area* must sit, in screen px.

    ``owner_plot`` is the primary pane's plot rectangle in global coordinates.
    YZ goes to its right sharing the vertical extent, XZ below it sharing the
    horizontal one -- the same arrangement as the embedded grid, and the same
    as ImageJ's ``arrangeWindows``.

    Returned as the *plot area* rather than the window frame: matching the
    frames would leave the data misaligned by whatever the axis gutters and
    the window decorations happen to be, which is the whole difficulty of the
    floating arrangement.
    """
    x, y, w, h = owner_plot
    if plane == "YZ":
        return x + w + gap, y, thickness, h
    if plane == "XZ":
        return x, y + h + gap, w, thickness
    raise ValueError(f"not a side plane: {plane!r}")


def _QPoint(x: int, y: int):
    from PyQt6.QtCore import QPoint
    return QPoint(x, y)


def QTimer_singleShot(msec: int, callback) -> None:
    from PyQt6.QtCore import QTimer
    QTimer.singleShot(msec, callback)


def _OWNER_FOLLOW_EVENTS():
    from PyQt6.QtCore import QEvent
    return (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.WindowStateChange)


def _release_fixed_size(widget) -> None:
    """Undo ``setFixedSize`` so the widget can be laid out by a stretch again."""
    _MAX = 16777215
    try:
        widget.setMinimumSize(0, 0)
        widget.setMaximumSize(_MAX, _MAX)
    except Exception:
        pass
