"""
minflux_viewer.ui.scatter_window
=================================
Scatter plot window — supports both 2-D projections (XY / XZ / YZ) and
a true interactive 3-D scatter via OpenGL.

Mode is chosen with the **Axis** dropdown:
* ``XY``, ``XZ``, ``YZ``  — pyqtgraph 2-D ``ScatterPlotItem``
* ``3D``                  — pyqtgraph ``GLScatterPlotItem`` inside a
                            ``GLViewWidget``. Mouse-drag rotates,
                            middle-drag pans, scroll zooms.

Each point is colored by either local density (default) or any numeric
attribute selected from the **Color by** dropdown. The colormap matches
the render window's behaviour and falls back gracefully if a name isn't
available in pyqtgraph's built-in set.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..colormaps import (
    BUILTIN_COLORMAP_NAMES,
    channel_colormap_names,
    make_colormap,
    named_colormap_names,
)
from ..colors import (
    is_solid_color,
    solid_color_names,
    solid_color_rgba,
)
from ..core.app_state import AppState
from ..core.attributes import plot_attribute_names
from ..core.loader import attr_values_1d
from ..core.overlay import (
    apply_display_transform_nm,
    identity_matrix4,
    manual_alignment_matrix4,
    matrix4_to_xy3,
    overlay_members,
    transform_key,
    transform_to_matrix4,
)
from ..core.roi_selection import REGION_ROI_TYPES, roi_region_mask
from .attribute_help import apply_attribute_menu_tooltips, apply_attribute_tooltips
from .gl_3d_reference import nice_step, three_plane_grid_positions, tick_values
from .ortho_view import (
    AXIS_COLUMNS,
    ORTHO_AXIS,
    OrthoPanes,
    SIDE_PLANES,
    axis_columns,
    axis_labels,
    ortho_pane_labels,
)
from .plot_format import plot_widget

# ---------------------------------------------------------------------------
# Helpers — colormap loading shared with render_window
# ---------------------------------------------------------------------------

def _load_cmap(name: str) -> pg.ColorMap:
    """Compatibility wrapper around the application-owned registry."""
    return make_colormap(name)


# ---------------------------------------------------------------------------
# ScatterWindow
# ---------------------------------------------------------------------------

_AXIS_OPTIONS = ["XY", "XZ", "YZ", "3D", ORTHO_AXIS]
_MAX_DISPLAY_POINTS_2D = 100_000
#: Settle delay before the ortho side panes are re-projected after a pan/zoom.
_ORTHO_REDRAW_MS = 110
_MAX_DISPLAY_POINTS_3D = 150_000

_NAMED_CMAPS = list(BUILTIN_COLORMAP_NAMES)


class ScatterWindow(QWidget):
    """Interactive 2D / 3D scatter plot of MINFLUX localisations."""

    TAG = "scatter_window"

    def __init__(self, state: AppState, parent: QWidget | None = None, *, dataset_idx: int | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._dataset_idx = dataset_idx if dataset_idx is not None else state.active_idx
        self._3d_view = None         # built lazily on first 3D switch
        self._3d_scatter = None
        self._3d_grid = None
        self._3d_axis = None
        self._3d_axis_items: list = []
        self._3d_box_items: list = []
        self._show_2d_axis = True
        self._show_3d_axis = True
        self._show_2d_grid = True
        self._show_3d_grid = True
        self._show_3d_bounding_box = True
        self._show_colorbar = True
        self._colorbar_show_values = True
        self._colorbar_orientation = "vertical"
        self._colorbar_geometry: list[int] | None = None
        self._colorbar = None
        self._point_symbol = "o"
        self._point_size = 2
        self._point_alpha = 255
        self._manual_color_levels: tuple[float, float] | None = None
        self._last_color_values: np.ndarray = np.empty(0)
        self._lut_dialog = None
        self._lut_invert = False
        self._lut_gamma = 1.0
        self._roi_overlay = None
        self._view_state_key = "scatter_plot_state"
        self._cached_dataset_idx: int | None = None
        self._cached_locs_nm: np.ndarray | None = None
        self._color_cache_key: tuple | None = None
        self._color_cache: dict | None = None
        self._brush_lut_key: tuple | None = None
        self._brush_lut: list | None = None
        self._rgba_lut: np.ndarray | None = None
        self._3d_camera_initialised = False
        self._last_axis_text = "XY"
        self._channels: list[dict] = []
        self._channel_rows: list[tuple[QLabel, QLabel]] = []
        self._overlay_alignment_panel = None
        self._overlay_alignment_original: list[dict] | None = None
        self._overlay_alignment_original_visibility: list[bool] | None = None
        self._roi_highlight_2d = None
        self._roi_highlight_3d = None
        self._ortho_redrawing = False
        # ⚠ "The XY pane is showing everything, so do not crop" is an
        # EXPLICIT flag, not pyqtgraph's auto-range state. Inferring it was
        # what kept the side panes read-only: a side pane pushes its pan
        # back to the primary with setXRange, which *disables* auto-range,
        # so merely touching a side pane made the crop engage as though the
        # user had zoomed XY -- and the narrowed crop moved the shared Z,
        # which moved the side pane again. Ranges ran to +-100 um.
        self._ortho_show_all = True
        # Whether the colorbar was on when ortho took its gutter, so leaving
        # the mode gives it back rather than silently dropping it.
        self._ortho_colorbar_restore: bool | None = None
        self._ortho_colorbar_syncing = False

        self.setWindowTitle("Scatter Plot")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(720, 680)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._ortho_timer = QTimer(self)
        self._ortho_timer.setSingleShot(True)
        self._ortho_timer.setInterval(_ORTHO_REDRAW_MS)
        self._ortho_timer.timeout.connect(self._redraw_ortho_projection)

        self._build_ui()
        self._refresh()

        state.filter_changed.connect(self._on_filter_changed)
        state.attributes_changed.connect(self._on_attributes_changed)
        state.calibration_changed.connect(self._on_calibration_changed)
        state.overlay_transform_changed.connect(self._on_overlay_transform_changed)
        state.roi_selection_changed.connect(self._on_roi_selection_changed)
        state.rois.selection_changed.connect(self._redraw_roi_highlight)
        # The side panes' outlines follow the store, not the redraw debounce:
        # adding, deleting or re-selecting a ROI must show there immediately,
        # the same as it does in the primary pane.
        state.rois.changed.connect(self._refresh_ortho_roi_outlines)
        state.rois.selection_changed.connect(self._refresh_ortho_roi_outlines)

    def refresh_preferences(self) -> None:
        self._apply_y_axis_direction()
        if getattr(self, "_roi_overlay", None) is not None:
            self._roi_overlay.refresh()

    def refresh_global_colors(self, *, reset_overlay: bool = False) -> None:
        """Repaint solid colors and, when requested, reload overlay slots."""
        if reset_overlay and self._channels:
            from ..core.overlay import overlay_color_cycle
            cycle = overlay_color_cycle(self._state.prefs)
            for index, channel in enumerate(self._channels):
                lut = cycle[index % len(cycle)]
                channel["lut"] = lut
                ds_idx = channel.get("dataset_idx")
                if ds_idx is not None and 0 <= ds_idx < len(self._state.datasets):
                    self._state.datasets[ds_idx].state["overlay_lut"] = lut
                    self._state.datasets[ds_idx].state["render_channel_lut"] = lut
        try:
            self._cmap = _load_cmap(self._cmap_combo.currentText())
        except Exception:
            pass
        self._rebuild_channel_ui()
        self._invalidate_color_cache()
        self._redraw_current(save_state=False)

    @property
    def dataset_idx(self) -> int | None:
        return self._dataset_idx

    def _dataset(self):
        if self._dataset_idx is None:
            return None
        if not (0 <= self._dataset_idx < len(self._state.datasets)):
            return None
        return self._state.datasets[self._dataset_idx]

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self._root_layout = root
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Headless controls. The user-facing controls live in the right-click
        # menu, but QComboBox keeps the existing state/update code compact.
        self._cbar_combo = QComboBox(self)
        self._cbar_combo.setMinimumWidth(120)
        self._cbar_combo.currentTextChanged.connect(self._on_color_by_changed)
        self._cbar_combo.hide()

        self._cmap_combo = QComboBox(self)
        self._cmap_combo.setEditable(True)
        self._cmap_combo.addItems(named_colormap_names())
        self._cmap_combo.setCurrentText("jet")
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        self._cmap_combo.hide()

        self._black_bg_check = QCheckBox(self)
        self._black_bg_check.toggled.connect(self._on_background_changed)
        self._black_bg_check.hide()

        self._axis_combo = QComboBox(self)
        self._axis_combo.addItems(_AXIS_OPTIONS)
        self._axis_combo.currentTextChanged.connect(self._on_axis_changed)
        self._axis_combo.hide()

        # Stacked widget — switches between 2D plot and 3D GL view
        self._stack = QStackedWidget()
        self._stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self._stack, stretch=1)

        # ── 2D plot widget ─────────────────────────────────────────
        pg.setConfigOptions(antialias=False)
        self._plot_2d = plot_widget(background="w")
        self._plot_2d.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._plot_2d.customContextMenuRequested.connect(self._show_context_menu)
        self._plot_2d.getPlotItem().getViewBox().setMenuEnabled(False)
        self._plot_2d.setAspectLocked(True)
        self._apply_y_axis_direction()
        self._plot_2d.showGrid(
            x=self._show_2d_grid, y=self._show_2d_grid, alpha=0.2
        )
        self._scatter_2d = pg.ScatterPlotItem(
            size=self._point_size,
            symbol=self._point_symbol,
            pen=None,
            brush=pg.mkBrush(200, 200, 200, 180),
        )
        self._plot_2d.addItem(self._scatter_2d)
        self._roi_highlight_2d = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen(255, 210, 0, 230, width=1.5),
            brush=pg.mkBrush(255, 230, 0, 70),
        )
        self._plot_2d.addItem(self._roi_highlight_2d)

        self._cmap = _load_cmap("jet")
        # The 2-D page is the orthogonal pane grid with the side panes hidden,
        # so entering ortho mode never reparents _plot_2d: it stays the same
        # object, in the same cell, and everything keyed to it is unaffected.
        self._build_ortho_panes()
        self._stack.addWidget(self._plot_page)
        from .roi_overlay import RoiOverlayController
        self._roi_overlay = RoiOverlayController(
            self._state.rois,
            self,
            self._plot_2d,
            self._plot_2d.getPlotItem(),
            coordinate_space="plot",
            source_view="scatter",
        )
        # Also catch arrow-nudge / 't' when focus is on the window itself (not the
        # inner graphics view), so ROI keyboard editing works regardless of focus.
        self._roi_overlay.add_key_event_source(self)
        # 3D view added lazily by _ensure_3d_built()

        from .floating_colorbar import FloatingColorBar

        self._colorbar = FloatingColorBar(
            self._stack,
            on_visibility_changed=self._set_colorbar_visible,
            on_customize=self.open_lut_dialog,
            on_state_changed=self._on_colorbar_state_changed,
            attribute_names=self._colorbar_attribute_names,
            current_attribute=self._current_colorbar_attribute,
            on_attribute_changed=self._set_colorbar_attribute,
            plot_area=self._colorbar_plot_area,
            background_color=self._colorbar_background,
        )
        self._colorbar.set_bar_visible(False)

        # A docked bar aligns its gradient with the ViewBox, so repaint it
        # whenever that geometry changes (resize, axis shown/hidden).
        try:
            view_box = self._plot_2d.getPlotItem().getViewBox()
            view_box.sigResized.connect(lambda *_: self._colorbar.update())
            # The ortho crop and the shared Z scale both follow the XY view, so
            # they have to be refreshed when it moves and when it is resized
            # (the Z scale is derived from the side panes' pixel extents).
            view_box.sigRangeChanged.connect(self._on_ortho_view_changed)
            view_box.sigResized.connect(self._on_ortho_view_changed)
        except Exception:
            pass

        self._channel_area = QScrollArea()
        self._channel_area.setWidgetResizable(True)
        self._channel_area.setMaximumHeight(110)
        self._channel_widget = QWidget()
        self._channel_layout = QVBoxLayout(self._channel_widget)
        self._channel_layout.setContentsMargins(4, 4, 4, 4)
        self._channel_layout.setSpacing(2)
        self._channel_area.setWidget(self._channel_widget)
        root.addWidget(self._channel_area)
        self._channel_area_layout_index = root.indexOf(self._channel_area)

        # Status line
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info_label)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self._stack.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._stack.customContextMenuRequested.connect(self._show_context_menu)

    # ------------------------------------------------------------------
    # Orthogonal panes
    # ------------------------------------------------------------------

    def _build_ortho_panes(self) -> None:
        """Build the 2x2 pane grid holding XY plus the two side projections.

        The side panes are created up front but stay hidden until the user
        picks ``Ortho``; building them lazily would save a couple of empty
        pyqtgraph items and cost the mode a rebuild path.
        """
        self._plot_page = QWidget()
        self._pane_plots: dict[str, object] = {"XY": self._plot_2d}
        self._pane_scatters: dict[str, object] = {"XY": self._scatter_2d}
        self._pane_highlights: dict[str, object] = {"XY": self._roi_highlight_2d}

        for plane in SIDE_PLANES:
            plot = plot_widget(background="w")
            plot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            plot.customContextMenuRequested.connect(self._show_context_menu)
            # Aspect is deliberately NOT locked on a side pane: Z is stretched
            # to fill the panel, as in the MATLAB interactive render this mode
            # reproduces. Positions along the axis shared with XY stay exact;
            # only the Z direction is scaled, which is why it stays ticked.
            plot.showGrid(x=self._show_2d_grid, y=self._show_2d_grid, alpha=0.2)
            scatter = pg.ScatterPlotItem(
                size=self._point_size,
                symbol=self._point_symbol,
                pen=None,
                brush=pg.mkBrush(200, 200, 200, 180),
            )
            plot.addItem(scatter)
            highlight = pg.ScatterPlotItem(
                size=7,
                pen=pg.mkPen(255, 210, 0, 230, width=1.5),
                brush=pg.mkBrush(255, 230, 0, 70),
            )
            plot.addItem(highlight)
            self._pane_plots[plane] = plot
            self._pane_scatters[plane] = scatter
            self._pane_highlights[plane] = highlight

        self._plot_page.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._plot_page.customContextMenuRequested.connect(self._show_context_menu)
        self._ortho = OrthoPanes(
            self._plot_page, self._pane_plots, parent=self, interactive_sides=True)
        from .ortho_roi import OrthoRoiOutlines
        # The primary pane keeps the real ROI controller (drawing, hit-testing,
        # editing); the side panes show where each ROI lies on their own axes,
        # so a ROI is visible in all three views rather than only the one it was
        # drawn in.
        self._ortho_roi_outlines = OrthoRoiOutlines(self._pane_plots)
        self._wire_manual_range_gestures()

    def _wire_manual_range_gestures(self) -> None:
        """Clear the show-all flag on a real mouse gesture in any pane.

        ``sigRangeChangedManually`` is pyqtgraph's own answer to "did the *user*
        do this?" — a ViewBox emits it from ``mouseDragEvent`` and ``wheelEvent``
        and from nowhere else, so it distinguishes a pan or a zoom from every
        programmatic ``setXRange`` this window makes. All three panes are wired:
        zooming a side pane genuinely narrows what the primary shows (it owns
        one of XY's axes), so the crop should engage for that too.
        """
        for plane in ("XY", *SIDE_PLANES):
            plot = self._pane_plots.get(plane)
            if plot is None:
                continue
            try:
                view_box = plot.getPlotItem().getViewBox()
                view_box.sigRangeChangedManually.connect(self._on_manual_range_gesture)
            except Exception:
                pass

    def _on_manual_range_gesture(self, *_args) -> None:
        """The user panned or zoomed: stop showing everything and re-project."""
        if not self._ortho_active() or not self._ortho_show_all:
            return
        self._ortho_show_all = False
        self._on_ortho_view_changed()

    def _apply_page_background(self, black: bool) -> None:
        """Paint the pane page itself, so the empty bottom-right cell and the
        gaps between panes read as part of the plot rather than as a hole.

        A palette, not a stylesheet: a stylesheet cascades to every descendant,
        including the ROI controller's context menu, which is parented to the
        view widget.
        """
        page = getattr(self, "_plot_page", None)
        if page is None:
            return
        color = QColor(0, 0, 0) if black else QColor(255, 255, 255)
        palette = page.palette()
        palette.setColor(page.backgroundRole(), color)
        page.setAutoFillBackground(True)
        page.setPalette(palette)

    def _pane_labels(self, plane: str) -> tuple[str, str]:
        """``(bottom, left)`` for one pane, shared axes named once in ortho."""
        if self._ortho_active():
            return ortho_pane_labels(plane)
        return axis_labels(plane)

    def enter_ortho_mode(self) -> bool:
        """Switch this view to the orthogonal mode. True if it is now on.

        Public because selecting a 3-D ROI tool turns it on: a volume shape is
        drawn in one plane and bounded in the other two, so a single projection
        cannot show what is being made.
        """
        if not self._ortho_available():
            return False
        if not self._ortho_active():
            self._axis_combo.setCurrentText(ORTHO_AXIS)
        return self._ortho_active()

    def _ortho_active(self) -> bool:
        return self._axis_combo.currentText() == ORTHO_AXIS

    def _active_plane(self) -> str:
        """The plane the *interactive* 2-D pane shows.

        In orthogonal mode the primary pane is XY and is the only pane that
        takes ROI drawing, manual alignment or a scale bar, so every consumer
        of "which projection is this" resolves ``Ortho`` to ``XY``.
        """
        axis = self._axis_combo.currentText()
        return "XY" if axis == ORTHO_AXIS else axis

    def _drawn_planes(self) -> list[str]:
        """The planes to draw into for the current mode, primary first."""
        if self._ortho_active():
            return ["XY", *SIDE_PLANES]
        axis = self._axis_combo.currentText()
        return [axis] if axis in AXIS_COLUMNS else []

    def _pane_targets(self) -> list[tuple[str, object, object, object, tuple[int, int]]]:
        """``(plane, plot, scatter, highlight, columns)`` for each pane drawn.

        Outside ortho mode there is one entry, and it is the primary pane
        showing whichever projection the user selected — so the single-pane
        and three-pane paths are the same loop. The columns are resolved here
        so one place decides which pair a pane plots (the ortho YZ pane is
        transposed against the standalone YZ projection).
        """
        ortho = self._ortho_active()
        targets = []
        for plane in self._drawn_planes():
            key = plane if ortho else "XY"
            targets.append((
                plane,
                self._pane_plots[key],
                self._pane_scatters[key],
                self._pane_highlights[key],
                axis_columns(plane, ortho=ortho),
            ))
        return targets

    def _sync_ortho_colorbar(self, active: bool) -> None:
        """Hand the colorbar's gutter to the panes while ortho is on.

        A docked bar reserves a strip at the right of the stack, flush against
        the YZ pane -- 84 px, which at the default window size is most of a
        92 px-wide YZ plot area. Since the panes are isotropic, that strip *is*
        Z range, so the bar steps aside and the previous state is restored on
        the way out. Turning it back on while in ortho drops the memory, so an
        explicit choice is not overridden later.
        """
        if self._colorbar is None:
            return
        self._ortho_colorbar_syncing = True
        try:
            if active:
                if self._ortho_colorbar_restore is None and self._show_colorbar:
                    self._ortho_colorbar_restore = True
                    self._set_colorbar_visible(False)
            elif self._ortho_colorbar_restore:
                self._ortho_colorbar_restore = None
                self._set_colorbar_visible(True)
            else:
                self._ortho_colorbar_restore = None
        finally:
            self._ortho_colorbar_syncing = False

    def _ortho_available(self) -> bool:
        """Ortho is offered only for a 3-D dataset — two of its three panes
        would otherwise be an empty line."""
        ds = self._dataset()
        return ds is not None and getattr(ds.prop, "num_dim", 2) == 3

    def _clear_2d_panes(self) -> None:
        for scatter in self._pane_scatters.values():
            scatter.setData([], [])

    def _ortho_view_rect(self) -> tuple[float, float, float, float] | None:
        """The XY pane's current data rectangle, or ``None`` while showing all.

        ``None`` means "showing everything" — the state after a fit or a Reset
        View — and cropping to a range still being fitted would blank the side
        panes on the first draw.

        ⚠ That state is read from :attr:`_ortho_show_all`, **not** from
        ``autoRangeEnabled()``. Auto-range is pyqtgraph's own bookkeeping and any
        programmatic ``setXRange`` clears it, including the one a side pane makes
        when it pushes its pan onto the primary — so the inferred version turned
        the crop on for a gesture the user never made in this pane, and the
        narrowed crop then moved the shared Z, which moved the side pane again.
        The flag is cleared only by a genuine mouse gesture
        (``sigRangeChangedManually``), which is the thing actually being asked
        about.
        """
        if self._ortho_show_all:
            return None
        view_box = self._pane_plots["XY"].getPlotItem().getViewBox()
        try:
            (x0, x1), (y0, y1) = view_box.viewRange()
        except Exception:
            return None
        if not (x1 > x0 and y1 > y0):
            return None
        return float(x0), float(x1), float(y0), float(y1)

    def _ortho_visible_mask(self, locs: np.ndarray) -> np.ndarray | None:
        """Rows inside the XY viewport, so a side pane projects only what the
        XY pane is actually showing.

        Without this the XZ pane pools every localization at a given X, whatever
        its Y — so zooming onto one structure still showed the Z of everything
        behind and in front of it, which is the opposite of what an orthogonal
        view is for.
        """
        rect = self._ortho_view_rect()
        if rect is None or locs.ndim != 2 or locs.shape[1] < 2:
            return None
        x0, x1, y0, y1 = rect
        return (
            (locs[:, 0] >= x0) & (locs[:, 0] <= x1)
            & (locs[:, 1] >= y0) & (locs[:, 1] <= y1)
        )

    def _ortho_row_filter(self, locs: np.ndarray, ftr: np.ndarray) -> np.ndarray:
        """``ftr`` narrowed to the XY viewport while ortho is active.

        Applied **before** decimation, not after: the display budget must be
        spent on rows that are actually in view, or zooming into a sparse
        region would thin away the few points it contains.
        """
        if not self._ortho_active():
            return ftr
        mask = self._ortho_visible_mask(locs)
        if mask is None:
            return ftr
        ftr = np.asarray(ftr, dtype=bool)
        if mask.shape[0] != ftr.shape[0]:
            return ftr
        return ftr & mask

    def _ortho_axis_note(self, *, cropped: bool) -> str:
        """Status text for the ortho panes.

        It names the two things the picture cannot say for itself: that a
        clipped Z is not the whole projection, and which Z scaling factor the
        geometry on screen was computed with — isotropic panes are only worth
        reading as geometry when that number is known.
        """
        note = "XY · YZ · XZ"
        note += " (visible region" if cropped else " (full range"
        if self._ortho.depth_clipped:
            note += ", Z clipped"
        note += ")"
        ds = self._dataset()
        factor = float(getattr(getattr(ds, "cali", None), "z_scaling_factor", 1.0) or 1.0)
        if abs(factor - 1.0) > 1e-9:
            from .z_scaling_widgets import format_z_scaling_factor
            note += f"  |  Z scaling {format_z_scaling_factor(factor)}"
        return note

    def _sync_ortho_depth(self, locs_and_rows) -> None:
        """Centre both side panes on the Z of what is drawn, at the XY scale.

        ``locs_and_rows`` is the ``(locs, indices)`` actually plotted, so the Z
        follows the cropped projection: zoom onto a flat structure and the side
        panes centre on its Z rather than the whole stack's. The panes are
        isotropic, so the *range* they show is set by the XY zoom, not by the
        data — which is what makes the Z scaling factor visible.
        """
        if not self._ortho_active():
            return
        lows, highs = [], []
        for locs, indices in locs_and_rows:
            if locs.ndim != 2 or locs.shape[1] < 3 or indices.size == 0:
                continue
            z = locs[indices, 2]
            z = z[np.isfinite(z)]
            if z.size:
                lows.append(float(z.min()))
                highs.append(float(z.max()))
        if not lows:
            return
        self._ortho.apply_depth_range(min(lows), max(highs))

    def _on_ortho_view_changed(self, *_args) -> None:
        """Re-project the side panes after the XY view settles.

        Debounced rather than immediate: the crop changes on every step of a
        drag, and re-running two ``setData`` calls per step would make panning
        the cost of redrawing the whole plot twice.
        """
        if not self._ortho_active() or self._ortho_redrawing:
            return
        self._ortho_timer.start()

    def _redraw_ortho_projection(self) -> None:
        if not self._ortho_active() or self._dataset() is None:
            return
        self._ortho_redrawing = True
        try:
            self._redraw_current(save_state=False)
        finally:
            self._ortho_redrawing = False
        self._refresh_ortho_roi_outlines()

    def _refresh_ortho_roi_outlines(self) -> None:
        """Show every in-scope ROI in the two side panes as well.

        Includes the uncommitted draft, so a shape being drawn is visible in all
        three panes as it is made -- which is the whole point while a volume ROI
        is being seeded.
        """
        drawer = getattr(self, "_ortho_roi_outlines", None)
        if drawer is None:
            return
        if not self._ortho_active():
            drawer.clear()
            return
        controller = getattr(self, "_roi_overlay", None)
        records = []
        try:
            for record in self._state.rois.records:
                if controller is None or controller._record_in_scope(record):
                    records.append(record)
            draft = getattr(controller, "draft", None) if controller else None
            if draft is not None:
                records.append(draft)
        except Exception:
            return
        drawer.refresh(records, color_of=lambda rec: rec.stroke_color)

    def _set_info_text(self, text: str, ds=None) -> None:
        """Prefix Scatter status with the source dataset dimensionality."""
        if ds is None:
            ds = self._dataset()
        prefix = f"{ds.prop.num_dim}D  |  " if ds is not None else ""
        self._info_label.setText(f"{prefix}{text}")

    def _ensure_3d_built(self) -> None:
        """Construct the OpenGL widget on first 3D request."""
        if self._3d_view is not None:
            return
        try:
            import pyqtgraph.opengl as gl
        except ImportError as e:
            self._set_info_text(
                f"3D view unavailable: PyOpenGL is not installed ({e}). "
                "Run: poetry install"
            )
            return

        view = gl.GLViewWidget()
        view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        view.customContextMenuRequested.connect(self._show_context_menu)
        view.setBackgroundColor("k" if self._black_bg_check.isChecked() else "w")

        # Camera-rotating three-plane reference grid (XY + XZ + YZ).
        grid = gl.GLLinePlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=self._reference_color(),
            width=1.0,
            mode="lines",
            antialias=True,
        )
        grid.setGLOptions("translucent")
        grid.setVisible(self._show_3d_grid)
        view.addItem(grid)

        # Light XYZ axes
        axis = gl.GLAxisItem()
        axis.setSize(500, 500, 500)
        view.addItem(axis)
        try:
            axis.setVisible(False)
        except Exception:
            pass

        scatter = gl.GLScatterPlotItem(pxMode=True, size=self._point_size_3d())
        view.addItem(scatter)
        roi_scatter = gl.GLScatterPlotItem(pxMode=True, size=7.0)
        view.addItem(roi_scatter)

        self._3d_view    = view
        self._3d_scatter = scatter
        self._3d_grid = grid
        self._3d_axis = axis
        self._roi_highlight_3d = roi_scatter
        self._stack.addWidget(view)
        self._apply_3d_blend(self._black_bg_check.isChecked())

    def _on_background_changed(self, black: bool) -> None:
        self._apply_background(black)
        self._update_color()

    def _background_is_black(self) -> bool:
        return self._black_bg_check.isChecked()

    def _set_black_background(self, enabled: bool) -> None:
        self._black_bg_check.setChecked(bool(enabled))

    def _point_size_3d(self) -> float:
        """Keep tiny GL sprites visible while preserving the 2-D size control."""
        compensation = 2 if not self._background_is_black() else 1
        return float(self._point_size + compensation)

    def _current_style_color(self) -> tuple[int, int, int]:
        if len(self._channels) > 1:
            lut = str(self._channels[self._active_channel_index()].get("lut", "Gray"))
        else:
            lut = self._cmap_combo.currentText()
        r, g, b, _alpha = self._lut_color(lut, alpha=255)
        return r, g, b

    def _show_plot_style_dialog(self) -> None:
        from .plot_style_dialog import PlotStyleDialog

        dlg = PlotStyleDialog(
            {
                "label": self.windowTitle(),
                "symbol": self._point_symbol,
                "size": self._point_size,
                "alpha": self._point_alpha,
                "color": self._current_style_color(),
            },
            self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._apply_plot_style(
                dlg.result_payload(), color_changed=dlg.color_changed
            )

    def _apply_plot_style(
        self, payload: dict, *, color_changed: bool = False
    ) -> None:
        self._point_symbol = str(payload.get("symbol", self._point_symbol))
        self._point_size = max(1, min(50, int(payload.get("size", self._point_size))))
        self._point_alpha = max(
            0, min(255, int(payload.get("alpha", self._point_alpha)))
        )

        if color_changed:
            r, g, b = (
                max(0, min(255, int(v)))
                for v in payload.get("color", (128, 128, 128))
            )
            custom_lut = f"solid:custom:#{r:02x}{g:02x}{b:02x}"
            if len(self._channels) > 1:
                self._on_channel_lut(self._active_channel_index(), custom_lut)
            else:
                self._cmap_combo.setCurrentText(custom_lut)

        # Marker alpha is part of the cached brushes/RGBA LUT.
        self._brush_lut_key = None
        self._brush_lut = None
        self._rgba_lut = None
        self._redraw_current(save_state=True)

    def _apply_background(self, black: bool) -> None:
        for plot in getattr(self, "_pane_plots", {"XY": self._plot_2d}).values():
            plot.setBackground("k" if black else "w")
        self._apply_page_background(black)
        if self._3d_view is not None:
            self._3d_view.setBackgroundColor("k" if black else "w")
            self._refresh_3d_reference_items()
        self._apply_3d_blend(black)

    def _apply_3d_blend(self, black: bool) -> None:
        """Point blend mode for the 3-D view.

        ``additive`` glows on black but is invisible on white (it adds to the
        already-max white). ``translucent`` (alpha over) shows the point color
        on any background, so use it for the white background.
        """
        mode = "additive" if black else "translucent"
        # May be called from _apply_background before the 3-D view is built.
        for item in (getattr(self, "_3d_scatter", None),
                     getattr(self, "_roi_highlight_3d", None)):
            if item is not None:
                try:
                    item.setGLOptions(mode)
                except Exception:
                    pass

    def _xy_origin_top_left(self) -> bool:
        value = str(
            self._state.prefs.get("plot", {}).get("scatter_xy_origin", "top_left")
        ).lower()
        return value != "bottom_left"

    def _apply_y_axis_direction(self) -> None:
        try:
            invert = self._active_plane() == "XY" and self._xy_origin_top_left()
            self._plot_2d.getPlotItem().getViewBox().invertY(invert)
            if getattr(self, "_ortho", None) is not None and self._ortho.active:
                # YZ shares the vertical Y axis with XY, so it must be inverted
                # the same way: the link carries the range, not the direction,
                # and a mismatch would put the top of one pane at the bottom of
                # the other. XZ's vertical is Z, which is always natural (up).
                yz = self._ortho.view_box("YZ")
                if yz is not None:
                    yz.invertY(invert)
                xz = self._ortho.view_box("XZ")
                if xz is not None:
                    xz.invertY(False)
        except Exception:
            pass

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)

        view_menu = menu.addMenu("View")
        view_menu.setToolTipsVisible(True)
        for axis in _AXIS_OPTIONS:
            action = view_menu.addAction(axis)
            action.setCheckable(True)
            action.setChecked(axis == self._axis_combo.currentText())
            if axis == ORTHO_AXIS:
                action.setEnabled(self._ortho_available())
                action.setToolTip(
                    "XY, YZ and XZ at once, each a projection over the axis it "
                    "does not show"
                    if self._ortho_available()
                    else "Orthogonal views need a 3-D dataset"
                )
            action.triggered.connect(lambda _checked=False, value=axis: self._axis_combo.setCurrentText(value))
        view_menu.addSeparator()

        bg_action = view_menu.addAction("Black background")
        bg_action.setCheckable(True)
        bg_action.setChecked(self._background_is_black())
        bg_action.triggered.connect(self._set_black_background)

        axis_action = view_menu.addAction("Axis")
        axis_action.setCheckable(True)
        axis_action.setChecked(self._current_axis_visible())
        axis_action.triggered.connect(self._set_current_axis_visible)

        grid_action = view_menu.addAction("Grid lines")
        grid_action.setCheckable(True)
        grid_action.setChecked(self._current_grid_visible())
        grid_action.triggered.connect(self._set_current_grid_visible)

        view_menu.addAction("Plot style", self._show_plot_style_dialog)
        menu.addSeparator()

        is_overlay = len(self._channels) > 1
        active_ci = self._active_channel_index() if is_overlay else None

        if is_overlay and active_ci is not None:
            # Overlay: right-click actions target the **active** channel (click a
            # channel row to make it active). Its per-channel dropdown is gone.
            ch = self._channels[active_ci]
            active_color_by = ch.get("color_by")

            # Color by attribute (active channel); a solid-color pick reverts it.
            color_menu = menu.addMenu(f"Color by  (active channel {active_ci + 1})")
            solid_action = color_menu.addAction("Solid color")
            solid_action.setCheckable(True)
            solid_action.setChecked(not active_color_by)
            solid_action.triggered.connect(
                lambda _=False, ci=active_ci: self._set_channel_color_by(ci, None))
            color_menu.addSeparator()
            for i in range(self._cbar_combo.count()):
                text = self._cbar_combo.itemText(i)
                action = color_menu.addAction(text)
                action.setCheckable(True)
                action.setChecked(text == active_color_by)
                action.triggered.connect(
                    lambda _=False, value=text, ci=active_ci: self._set_channel_color_by(ci, value))
            apply_attribute_menu_tooltips(color_menu, self._colorbar_attribute_names())

            # Solid color of the active channel — same options as the render dropdown.
            current_lut = str(ch.get("lut", "Gray"))
            cmap_menu = menu.addMenu(f"Colormap  (active channel {active_ci + 1})")
            for lut_name in channel_colormap_names():
                action = cmap_menu.addAction(lut_name)
                action.setCheckable(True)
                action.setChecked(not active_color_by and lut_name == current_lut)
                action.triggered.connect(
                    lambda _=False, v=lut_name, ci=active_ci: self._on_channel_lut(ci, v))
        else:
            color_menu = menu.addMenu("Color by")
            for i in range(self._cbar_combo.count()):
                text = self._cbar_combo.itemText(i)
                action = color_menu.addAction(text)
                action.setCheckable(True)
                action.setChecked(text == self._cbar_combo.currentText())
                action.triggered.connect(lambda _=False, value=text: self._cbar_combo.setCurrentText(value))
            apply_attribute_menu_tooltips(color_menu, self._colorbar_attribute_names())

            cmap_menu = menu.addMenu("Colormap")
            current_cmap = self._cmap_combo.currentText()
            for text in named_colormap_names():
                action = cmap_menu.addAction(text)
                action.setCheckable(True)
                action.setChecked(text == current_cmap)
                action.triggered.connect(lambda _=False, v=text: self._cmap_combo.setCurrentText(v))
            cmap_menu.addSeparator()
            solid_menu = cmap_menu.addMenu("Solid color")
            for color_name in solid_color_names():
                action = solid_menu.addAction(color_name)
                action.setCheckable(True)
                action.setChecked(current_cmap == f"solid:{color_name}")
                action.triggered.connect(lambda _=False, cn=color_name: self._cmap_combo.setCurrentText(f"solid:{cn}"))
            # No 'Custom...' entry: this list is the global COLOR registry, so a
            # one-off color belongs there (add/rename it in the COLOR dialog)
            # rather than in an unnamed per-view override.  Existing saved
            # 'solid:custom:#rrggbb' values still resolve and render.

        colorbar_action = menu.addAction("Colorbar")
        colorbar_action.setCheckable(True)
        colorbar_action.setChecked(self._show_colorbar)
        colorbar_action.triggered.connect(self._set_colorbar_visible)

        if self._axis_combo.currentText() == "3D":
            menu.addSeparator()
            box_action = menu.addAction("Bounding Box")
            box_action.setCheckable(True)
            box_action.setChecked(self._show_3d_bounding_box)
            box_action.triggered.connect(self._set_3d_bounding_box_visible)

        menu.addSeparator()
        menu.addAction("Reset View", self._reset_view)

        sender = self.sender()
        if isinstance(sender, QWidget):
            menu.exec(sender.mapToGlobal(pos))
        else:
            menu.exec(self.mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Slots & lifecycle
    # ------------------------------------------------------------------

    def _on_axis_changed(self, _text: str) -> None:
        """Switch between 2D and 3D mode and re-draw.

        The background (black/white) is preserved across the switch — 3-D no
        longer forces black, since points are now visible on white too.
        """
        axis_text = self._axis_combo.currentText()
        is_3d = axis_text == "3D"
        # Leaving ortho must also drop the side panes' grid weight, or the
        # primary pane keeps only its share of the window (see grid_stretch).
        ortho_on = axis_text == ORTHO_AXIS and not is_3d
        if ortho_on and not self._ortho.active:
            # Entering the mode starts uncropped, whatever the previous
            # projection's zoom left behind: the side panes should open on the
            # whole dataset rather than on a rectangle the user set elsewhere.
            self._ortho_show_all = True
        self._ortho.set_active(ortho_on)
        self._sync_ortho_colorbar(ortho_on)
        if is_3d:
            self._ensure_3d_built()
            self._apply_background(self._black_bg_check.isChecked())
            if self._3d_view is not None:
                if self._3d_grid is not None:
                    self._3d_grid.setVisible(self._show_3d_grid)
                self._stack.setCurrentWidget(self._3d_view)
        else:
            self._stack.setCurrentWidget(self._plot_page)
            self._apply_2d_reference_visibility()
            self._apply_background(self._black_bg_check.isChecked())
        self._apply_y_axis_direction()
        self._update_colorbar_visibility()
        self._last_axis_text = axis_text
        self._save_view_state()
        self._refresh()
        # Re-project point markers onto the new projection plane.
        if self._roi_overlay is not None and not is_3d:
            self._roi_overlay.refresh()

    def refresh_colormap_list(self) -> None:
        """Rebuild the colormap choices after the shared list was reordered.

        The combo is editable and also carries ``solid:`` encodings, so the
        current text is restored verbatim rather than by index.
        """
        current = self._cmap_combo.currentText()
        self._cmap_combo.blockSignals(True)
        try:
            self._cmap_combo.clear()
            self._cmap_combo.addItems(named_colormap_names())
            self._cmap_combo.setCurrentText(current)
        finally:
            self._cmap_combo.blockSignals(False)

    def _on_cmap_changed(self, name: str) -> None:
        self._cmap = _load_cmap(name)
        self._update_colorbar_visibility()
        self._invalidate_color_cache()
        self._update_color()
        self.sync_lut_dialog()

    def _on_color_by_changed(self, _name: str) -> None:
        """The color-by attribute changed → its value range is different, so drop
        any manual levels and auto-scale to the new attribute (this re-tunes both
        the plot colors and the LUT dialog to the new range)."""
        self._manual_color_levels = None
        self._invalidate_color_cache()
        self._update_color()
        self.sync_lut_dialog()

    def _colorbar_attribute_names(self) -> list[str]:
        return [
            self._cbar_combo.itemText(index)
            for index in range(self._cbar_combo.count())
        ]

    def _current_colorbar_attribute(self) -> str:
        if len(self._channels) > 1:
            active = self._active_channel_index()
            if 0 <= active < len(self._channels):
                return str(self._channels[active].get("color_by") or "")
            return ""
        return self._cbar_combo.currentText()

    def _set_colorbar_attribute(self, name: str) -> None:
        if name not in self._colorbar_attribute_names():
            return
        if len(self._channels) > 1:
            self._set_channel_color_by(self._active_channel_index(), name)
        else:
            self._cbar_combo.setCurrentText(name)

    def _colorbar_plot_area(self) -> QRect | None:
        """The 2-D ViewBox rectangle in ``_stack`` coordinates.

        A docked colorbar aligns its gradient to this, so it spans exactly the
        plot's own data area the way the in-plot colorbar used to.  The 3-D view
        has no ViewBox, so it returns None and the bar falls back to its own
        height.
        """
        plot = getattr(self, "_plot_2d", None)
        if plot is None or self._stack.currentWidget() is not self._plot_page:
            return None
        try:
            scene_rect = plot.getPlotItem().getViewBox().sceneBoundingRect()
            offset = plot.viewport().mapTo(self._stack, QPoint(0, 0))
            return QRect(
                plot.mapFromScene(scene_rect.topLeft()) + offset,
                plot.mapFromScene(scene_rect.bottomRight()) + offset,
            )
        except Exception:
            return None

    def _colorbar_background(self) -> QColor:
        return QColor(0, 0, 0) if self._black_bg_check.isChecked() else QColor(255, 255, 255)

    def _set_colorbar_visible(self, visible: bool) -> None:
        self._show_colorbar = bool(visible)
        # A choice the user makes while ortho holds the bar's gutter is theirs
        # to keep: forget what the mode was going to restore, so leaving it
        # does not undo them.
        if self._ortho_active() and not self._ortho_colorbar_syncing:
            self._ortho_colorbar_restore = None
        self._update_colorbar_visibility()
        self._save_view_state()

    def _on_colorbar_state_changed(self) -> None:
        if self._colorbar is None:
            return
        self._colorbar_show_values = self._colorbar.show_values
        self._colorbar_orientation = self._colorbar.orientation
        self._colorbar_geometry = self._colorbar.serialized_geometry()
        self._save_view_state()

    def _update_colorbar_visibility(self) -> None:
        if self._colorbar is None or not self._show_colorbar:
            if self._colorbar is not None:
                self._colorbar.set_bar_visible(False)
            return
        if self._cmap_combo.currentText().startswith("solid:"):
            self._colorbar.set_bar_visible(False)
            return

        dataset = self._dataset()
        attribute = self._cbar_combo.currentText()
        if len(self._channels) > 1:
            active = self._active_channel_index()
            if not (0 <= active < len(self._channels)):
                self._colorbar.set_bar_visible(False)
                return
            channel = self._channels[active]
            attribute = str(channel.get("color_by") or "")
            dataset_idx = channel.get("dataset_idx")
            if not attribute or dataset_idx is None or not (
                0 <= dataset_idx < len(self._state.datasets)
            ):
                self._colorbar.set_bar_visible(False)
                return
            dataset = self._state.datasets[dataset_idx]
        if dataset is None or not attribute:
            self._colorbar.set_bar_visible(False)
            return

        cache = self._color_cache_for_dataset(dataset, attribute)
        if cache is None:
            self._colorbar.set_bar_visible(False)
            return
        values = np.asarray(cache["values"], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            self._colorbar.set_bar_visible(False)
            return
        self._last_color_values = values
        lut = np.asarray(
            self._cmap.getLookupTable(0.0, 1.0, 256, alpha=True),
            dtype=np.uint8,
        )
        self._colorbar.set_color_data(
            lut,
            float(cache["vmin"]),
            float(cache["vmax"]),
            attribute,
        )
        self._colorbar.set_bar_visible(True)

    def _update_color(self) -> None:
        self._redraw_current(save_state=True)

    def _on_filter_changed(self, idx: int) -> None:
        if idx == self._dataset_idx or any(ch.get("dataset_idx") == idx for ch in self._channels):
            self._redraw_current(save_state=False)

    def _on_attributes_changed(self, idx: int) -> None:
        if idx == self._dataset_idx or any(ch.get("dataset_idx") == idx for ch in self._channels):
            self._refresh()

    def _on_calibration_changed(self, idx: int) -> None:
        # Z scaling factor changed: the cached loc_nm (keyed only by dataset
        # index) is now stale on z — invalidate it before redrawing.
        if idx == self._dataset_idx or any(ch.get("dataset_idx") == idx for ch in self._channels):
            self._cached_dataset_idx = None
            self._cached_locs_nm = None
            self._redraw_current(save_state=False)

    def _on_overlay_transform_changed(self, idx: int) -> None:
        """Reload a changed live transform and redraw its overlay channel."""
        if not (0 <= idx < len(self._state.datasets)):
            return
        current = (
            self._state.datasets[idx].state.get("overlay_transform")
            or self._state.datasets[idx].state.get("render_transform_2d")
        )
        changed = False
        for channel in self._channels:
            if channel.get("dataset_idx") != idx:
                continue
            if transform_key(channel.get("loc_transform")) != transform_key(current):
                channel["loc_transform"] = current
                changed = True
        if changed:
            self._cached_dataset_idx = None
            self._cached_locs_nm = None
            self._redraw_current(save_state=False)

    def _on_roi_selection_changed(self, idx: int) -> None:
        if idx == self._dataset_idx or any(ch.get("dataset_idx") == idx for ch in self._channels):
            self._redraw_roi_highlight()

    def _reset_view(self) -> None:
        if self._axis_combo.currentText() == "3D" and self._3d_view is not None:
            self._reset_3d_camera()
        else:
            self._apply_y_axis_direction()
            if self._ortho.active:
                # ⚠ Lift the crop BEFORE fitting. A fit ranges to the items in
                # the view, and those are the cropped subset — so fitting first
                # re-fits to the very region the crop came from and Reset View
                # cannot escape it. Raising the flag makes _ortho_view_rect
                # report "showing everything", the redraw then puts every point
                # back, and auto-range fits to all of them.
                self._ortho_show_all = True
                self._plot_2d.getPlotItem().getViewBox().enableAutoRange(enable=True)
                self._redraw_ortho_projection()
                return
            self._plot_2d.autoRange()

    def _apply_2d_reference_visibility(self) -> None:
        for plot in getattr(self, "_pane_plots", {"XY": self._plot_2d}).values():
            plot_item = plot.getPlotItem()
            for axis_name in ("left", "bottom"):
                plot_item.showAxis(axis_name, show=self._show_2d_axis)
            plot.showGrid(x=self._show_2d_grid, y=self._show_2d_grid, alpha=0.2)

    def _current_axis_visible(self) -> bool:
        if self._axis_combo.currentText() == "3D":
            return self._show_3d_axis
        return self._show_2d_axis

    def _set_current_axis_visible(self, checked: bool) -> None:
        if self._axis_combo.currentText() == "3D":
            self._set_3d_axis_visible(checked)
            return
        self._show_2d_axis = bool(checked)
        self._apply_2d_reference_visibility()
        self._save_view_state()

    def _current_grid_visible(self) -> bool:
        if self._axis_combo.currentText() == "3D":
            return self._show_3d_grid
        return self._show_2d_grid

    def _set_current_grid_visible(self, checked: bool) -> None:
        if self._axis_combo.currentText() == "3D":
            self._show_3d_grid = bool(checked)
            if self._3d_grid is not None:
                self._3d_grid.setVisible(self._show_3d_grid)
        else:
            self._show_2d_grid = bool(checked)
            self._apply_2d_reference_visibility()
        self._save_view_state()

    def _reset_3d_camera(self, pos: np.ndarray | None = None) -> None:
        """Centre the 3D camera on the data."""
        ds = self._dataset()
        if ds is None or self._3d_view is None:
            return
        if pos is None:
            indices = self._visible_indices(ds.filter_mask, self._current_locs(ds).shape[0], _MAX_DISPLAY_POINTS_3D)
            pos = self._current_locs(ds)[indices, :3]
        pos = np.asarray(pos, dtype=float)
        if pos.ndim != 2 or pos.shape[1] < 3:
            return
        pos = pos[np.all(np.isfinite(pos[:, :3]), axis=1), :3]
        if pos.shape[0] == 0:
            return
        centre = pos.mean(axis=0)
        span = pos.max(axis=0) - pos.min(axis=0)
        extent = float(np.linalg.norm(span))
        if not np.isfinite(extent) or extent <= 0:
            extent = 100.0
        self._3d_view.opts["center"] = pg.Vector(*centre)
        self._3d_view.setCameraPosition(distance=max(extent * 1.5, 100.0))

    def _set_3d_axis_visible(self, checked: bool) -> None:
        self._show_3d_axis = bool(checked)
        self._refresh_3d_reference_items()
        self._save_view_state()

    def _set_3d_bounding_box_visible(self, checked: bool) -> None:
        self._show_3d_bounding_box = bool(checked)
        self._refresh_3d_reference_items()
        self._save_view_state()

    @staticmethod
    def _nice_3d_step(span: float, target: int = 6) -> float:
        return nice_step(span, target=target)

    @staticmethod
    def _tick_values(lo: float, hi: float, *, max_ticks: int = 5) -> list[float]:
        return tick_values(lo, hi, max_ticks=max_ticks)

    @staticmethod
    def _fmt_tick(value: float) -> str:
        if abs(value) >= 1000 or float(value).is_integer():
            return f"{value:.0f}"
        return f"{value:.3g}"

    def _reference_color(self) -> tuple[float, float, float, float]:
        return (0.72, 0.72, 0.72, 0.72) if self._background_is_black() else (0.28, 0.28, 0.28, 0.58)

    def _text_color(self) -> tuple[int, int, int, int]:
        return (230, 230, 230, 230) if self._background_is_black() else (35, 35, 35, 230)

    def _visible_xyz_3d(self) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self._channels:
            channels = self._channels
            for ch in channels:
                if not ch.get("visible", True):
                    continue
                ds_idx = ch.get("dataset_idx")
                if ds_idx is None or not (0 <= ds_idx < len(self._state.datasets)):
                    continue
                ds = self._state.datasets[ds_idx]
                locs = self._locs_for_dataset(ds)
                mask = np.asarray(ds.filter_mask, dtype=bool).ravel()
                if mask.size != locs.shape[0]:
                    mask = np.ones(locs.shape[0], dtype=bool)
                finite = np.all(np.isfinite(locs[:, :3]), axis=1)
                xyz = locs[mask & finite, :3]
                if xyz.size:
                    parts.append(xyz)
        else:
            ds = self._dataset()
            if ds is not None:
                locs = self._current_locs(ds)
                mask = np.asarray(ds.filter_mask, dtype=bool).ravel()
                if mask.size != locs.shape[0]:
                    mask = np.ones(locs.shape[0], dtype=bool)
                finite = np.all(np.isfinite(locs[:, :3]), axis=1)
                xyz = locs[mask & finite, :3]
                if xyz.size:
                    parts.append(xyz)
        if not parts:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack(parts).astype(np.float64, copy=False)

    @staticmethod
    def _expanded_bounds(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[0] == 0 or xyz.shape[1] < 3:
            return None
        xyz = xyz[np.all(np.isfinite(xyz[:, :3]), axis=1), :3]
        if xyz.size == 0:
            return None
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        spans = maxs - mins
        max_span = max(float(np.max(spans)), 1.0)
        for i in range(3):
            if spans[i] <= 0:
                mins[i] -= max_span * 0.05
                maxs[i] += max_span * 0.05
        spans = maxs - mins
        return mins, maxs, spans

    def _clear_3d_items(self, items: list) -> None:
        if self._3d_view is None:
            items.clear()
            return
        for item in list(items):
            try:
                self._3d_view.removeItem(item)
            except Exception:
                pass
        items.clear()

    def _add_3d_line(
        self,
        items: list,
        start: np.ndarray,
        end: np.ndarray,
        *,
        color: tuple[float, float, float, float],
        width: float = 1.2,
    ) -> None:
        if self._3d_view is None:
            return
        try:
            import pyqtgraph.opengl as gl
            item = gl.GLLinePlotItem(
                pos=np.vstack([start, end]).astype(np.float32),
                color=color,
                width=width,
                antialias=True,
            )
            self._3d_view.addItem(item)
            items.append(item)
        except Exception:
            pass

    def _add_3d_text(self, items: list, pos: np.ndarray, text: str, *, label: bool = False) -> None:
        if self._3d_view is None:
            return
        try:
            import pyqtgraph.opengl as gl
            font = QFont("Helvetica", 11 if label else 8)
            item = gl.GLTextItem(
                pos=np.asarray(pos, dtype=np.float64),
                text=str(text),
                color=self._text_color(),
                font=font,
                glOptions="translucent",
            )
            self._3d_view.addItem(item)
            items.append(item)
        except Exception:
            pass

    def _configure_3d_grid(self, mins: np.ndarray, maxs: np.ndarray, spans: np.ndarray) -> None:
        if self._3d_grid is None:
            return
        try:
            padding = np.maximum(spans * 0.04, max(float(np.max(spans)), 1.0) * 0.01)
            grid_mins = mins - padding
            grid_maxs = maxs + padding
            self._3d_grid.setData(
                pos=three_plane_grid_positions(grid_mins, grid_maxs, target=8),
                color=self._reference_color(),
                width=1.0,
                mode="lines",
            )
            self._3d_grid.setVisible(self._show_3d_grid)
        except Exception:
            pass

    def _draw_3d_axis(self, mins: np.ndarray, maxs: np.ndarray, spans: np.ndarray) -> None:
        if not self._show_3d_axis:
            return
        origin = mins.copy()
        pad = max(float(np.max(spans)) * 0.04, 1.0)
        axis_defs = (
            (0, np.array([maxs[0], origin[1], origin[2]]), (0.9, 0.15, 0.15, 0.95), "X (nm)"),
            (1, np.array([origin[0], maxs[1], origin[2]]), (0.15, 0.7, 0.25, 0.95), "Y (nm)"),
            (2, np.array([origin[0], origin[1], maxs[2]]), (0.15, 0.35, 0.95, 0.95), "Z (nm)"),
        )
        for dim, end, color, label in axis_defs:
            self._add_3d_line(self._3d_axis_items, origin, end, color=color, width=2.0)
            label_pos = end.copy()
            label_pos[dim] += pad
            self._add_3d_text(self._3d_axis_items, label_pos, label, label=True)
            for tick in self._tick_values(float(mins[dim]), float(maxs[dim])):
                tick_pos = origin.copy()
                tick_pos[dim] = tick
                tick_end = tick_pos.copy()
                side_dim = 1 if dim != 1 else 0
                tick_end[side_dim] += pad * 0.35
                self._add_3d_line(self._3d_axis_items, tick_pos, tick_end, color=color, width=1.0)
                text_pos = tick_end.copy()
                text_pos[side_dim] += pad * 0.12
                self._add_3d_text(self._3d_axis_items, text_pos, self._fmt_tick(tick))

    def _draw_3d_bounding_box(self, mins: np.ndarray, maxs: np.ndarray) -> None:
        if not self._show_3d_bounding_box:
            return
        x0, y0, z0 = mins
        x1, y1, z1 = maxs
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ], dtype=np.float64)
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        color = self._reference_color()
        for a, b in edges:
            self._add_3d_line(self._3d_box_items, corners[a], corners[b], color=color, width=1.4)

    def _refresh_3d_reference_items(self) -> None:
        if self._3d_view is None:
            return
        self._clear_3d_items(self._3d_axis_items)
        self._clear_3d_items(self._3d_box_items)
        bounds = self._expanded_bounds(self._visible_xyz_3d())
        if bounds is None:
            return
        mins, maxs, spans = bounds
        self._configure_3d_grid(mins, maxs, spans)
        self._draw_3d_axis(mins, maxs, spans)
        self._draw_3d_bounding_box(mins, maxs)

    def _build_channels(self) -> None:
        previous = {
            ch.get("dataset_idx"): {"visible": ch.get("visible", True),
                                    "lut": ch.get("lut"),
                                    "color_by": ch.get("color_by"),
                                    "transform": ch.get("transform")}
            for ch in self._channels
        }
        self._channels = []
        if self._dataset_idx is None:
            return
        channel_luts = channel_colormap_names()
        for pos, (idx, ds) in enumerate(overlay_members(self._state, self._dataset_idx)):
            prev = previous.get(idx, {})
            self._channels.append({
                "dataset_idx": idx,
                "name": ds.name,
                "visible": bool(prev.get("visible", True)),
                "lut": prev.get("lut") or ds.state.get("overlay_lut") or ds.state.get("render_channel_lut") or channel_luts[pos % len(channel_luts)],
                "color_by": prev.get("color_by"),   # None = solid; else attribute
                "loc_transform": ds.state.get("overlay_transform") or ds.state.get("render_transform_2d"),
                "transform": dict(prev.get("transform") or {}),
            })

    def _rebuild_channel_ui(self) -> None:
        while self._channel_layout.count():
            item = self._channel_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._channel_rows = []
        active_idx = self._active_channel_index()
        for ch_idx, ch in enumerate(self._channels):
            row = QWidget()
            row.mousePressEvent = lambda event, i=ch_idx, r=row: self._on_channel_row_pressed(r, event, i)
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            vis_cb = QCheckBox()
            vis_cb.setChecked(bool(ch["visible"]))
            vis_cb.toggled.connect(lambda checked, i=ch_idx: self._on_channel_visible(i, checked))
            lay.addWidget(vis_cb)
            # Color swatch replaces the per-channel colormap dropdown: the color
            # is defined in the render view (overlay_lut) and can be changed here
            # only for the ACTIVE channel via the right-click Colormap menu.
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            self._style_channel_swatch(swatch, str(ch["lut"]))
            lay.addWidget(swatch)
            name_lbl = QLabel(f"{ch_idx + 1}: {ch['name']}")
            font = name_lbl.font()
            font.setBold(ch_idx == active_idx)
            name_lbl.setFont(font)
            lay.addWidget(name_lbl, stretch=1)
            self._channel_layout.addWidget(row)
            self._channel_rows.append((name_lbl, swatch))
        self._channel_area.setVisible(len(self._channels) > 1)

    def _style_channel_swatch(self, swatch: QLabel, lut: str) -> None:
        r, g, b, _ = self._lut_color(lut)
        swatch.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #888;")

    def _locs_for_dataset_raw(self, ds_idx: int | None) -> np.ndarray:
        if ds_idx is None or not (0 <= ds_idx < len(self._state.datasets)):
            return np.empty((0, 3), dtype=float)
        locs = np.asarray(self._state.datasets[ds_idx].loc_nm, dtype=float)
        if locs.ndim == 2 and locs.shape[1] == 2:
            locs = np.column_stack([locs, np.zeros(locs.shape[0], dtype=float)])
        return locs

    def _manual_align_channel(self, ch_idx: int) -> None:
        if len(self._channels) < 2 or not (0 <= ch_idx < len(self._channels)):
            return
        if self._overlay_alignment_panel is not None:
            self._overlay_alignment_panel._channel_combo.setCurrentIndex(ch_idx)
            self._overlay_alignment_panel.setFocus()
            return
        self._overlay_alignment_original = [dict(ch.get("transform") or {}) for ch in self._channels]
        self._overlay_alignment_original_visibility = [bool(ch.get("visible", True)) for ch in self._channels]
        for ch in self._channels:
            self._ensure_channel_world_transform(ch)
        from .overlay_alignment import OverlayAlignmentPanel

        self._channel_area.hide()
        self._overlay_alignment_panel = OverlayAlignmentPanel(self, self._channels, ch_idx)
        self._root_layout.insertWidget(self._channel_area_layout_index, self._overlay_alignment_panel)
        self._overlay_alignment_panel.show()
        self._overlay_alignment_panel.setFocus()

    def start_manual_alignment_for_dataset(self, dataset_idx: int) -> bool:
        """Start the same mode as channel-row right-click → Manual align."""
        if len(self._channels) < 2:
            return False
        for channel_idx, channel in enumerate(self._channels):
            if channel.get("dataset_idx") == dataset_idx:
                self._manual_align_channel(channel_idx)
                return True
        return False

    def _overlay_alignment_control_config(self) -> dict:
        plot = self._state.prefs.setdefault("plot", {})
        return {
            "translation_unit": "nm",
            "translation_step": float(plot.get("scatter_alignment_translation_nm", 1.0)),
            "translation_maximum": 100000.0,
            "rotation_step": float(plot.get("scatter_alignment_rotation_deg", 0.1)),
        }

    def _overlay_alignment_steps_changed(
        self, translation_step: float, rotation_step: float
    ) -> None:
        plot = self._state.prefs.setdefault("plot", {})
        plot["scatter_alignment_translation_nm"] = float(translation_step)
        plot["scatter_alignment_rotation_deg"] = float(rotation_step)
        self._state.save_prefs()

    def _overlay_alignment_drag_view(self):
        if self._axis_combo.currentText() == "3D":
            return None
        return self._plot_2d.viewport()

    def _overlay_alignment_view_delta(self, start, end) -> tuple[float, float]:
        view_box = self._plot_2d.getPlotItem().getViewBox()
        start_view = view_box.mapSceneToView(self._plot_2d.mapToScene(start.toPoint()))
        end_view = view_box.mapSceneToView(self._plot_2d.mapToScene(end.toPoint()))
        return float(end_view.x() - start_view.x()), float(end_view.y() - start_view.y())

    def _overlay_alignment_rotation_sign(self) -> float:
        """Stored-angle sign that appears counter-clockwise in the current view."""
        invert_y = self._active_plane() == "XY" and self._xy_origin_top_left()
        return -1.0 if invert_y else 1.0

    def _ensure_channel_world_transform(self, ch: dict) -> None:
        transform = ch.setdefault("transform", {})
        if "anchor_x_nm" not in transform or "anchor_y_nm" not in transform:
            axis = self._active_plane()
            axes = (0, 2) if axis == "XZ" else (1, 2) if axis == "YZ" else (0, 1)
            if axis == "3D":
                ds_idx = ch.get("dataset_idx")
                locs = self._locs_for_dataset_raw(ds_idx)
                centre = np.nanmedian(locs[:, :3], axis=0) if locs.size else np.zeros(3)
                transform["anchor_x_nm"] = float(centre[axes[0]])
                transform["anchor_y_nm"] = float(centre[axes[1]])
            else:
                ranges = self._plot_2d.getPlotItem().getViewBox().viewRange()
                transform["anchor_x_nm"] = float(sum(ranges[0]) / 2.0)
                transform["anchor_y_nm"] = float(sum(ranges[1]) / 2.0)
        transform.setdefault("dx_nm", float(transform.get("dx", 0.0)))
        transform.setdefault("dy_nm", float(transform.get("dy", 0.0)))
        transform.setdefault("angle", 0.0)

    def _overlay_alignment_visibility(self, ch_idx: int, visible: bool) -> None:
        self._on_channel_visible(ch_idx, visible)

    def _overlay_alignment_set_channel(self, _ch_idx: int) -> None:
        self._redraw_current(save_state=False)

    def _overlay_alignment_nudge(
        self, ch_idx: int, dx_nm: float, dy_nm: float, rotation: float
    ) -> None:
        self._update_overlay_alignment_transform(ch_idx, dx_nm, dy_nm, rotation)

    def _overlay_alignment_drag(self, ch_idx: int, dx_nm: float, dy_nm: float) -> None:
        self._update_overlay_alignment_transform(ch_idx, dx_nm, dy_nm, 0.0)

    def _update_overlay_alignment_transform(
        self, ch_idx: int, dx_nm: float, dy_nm: float, rotation: float
    ) -> None:
        if not (0 <= ch_idx < len(self._channels)):
            return
        ch = self._channels[ch_idx]
        self._ensure_channel_world_transform(ch)
        transform = ch["transform"]
        transform["dx_nm"] = float(transform.get("dx_nm", 0.0)) + dx_nm
        transform["dy_nm"] = float(transform.get("dy_nm", 0.0)) + dy_nm
        transform["angle"] = float(transform.get("angle", 0.0)) + rotation
        self._cached_dataset_idx = None
        self._cached_locs_nm = None
        self._redraw_current(save_state=False)

    def _overlay_alignment_status(self, ch_idx: int) -> str:
        if not (0 <= ch_idx < len(self._channels)):
            return "X +0.0 nm | Y +0.0 nm | rotation +0.0°"
        transform = self._channels[ch_idx].get("transform") or {}
        dx = float(transform.get("dx_nm", 0.0))
        dy = float(transform.get("dy_nm", 0.0))
        return f"X {dx:+.1f} nm | Y {dy:+.1f} nm | rotation {float(transform.get('angle', 0.0)):+.1f}°"

    def _overlay_alignment_reset(self) -> None:
        panel = self._overlay_alignment_panel
        if panel is None:
            return
        ch_idx = panel.selected_index
        if 0 <= ch_idx < len(self._channels):
            self._channels[ch_idx]["transform"] = dict(
                self._overlay_alignment_original[ch_idx]
                if self._overlay_alignment_original is not None else {}
            )
        self._cached_dataset_idx = None
        self._cached_locs_nm = None
        self._redraw_current(save_state=False)
        panel.refresh_status()
        panel.setFocus()

    def _overlay_alignment_apply(self) -> None:
        if self._overlay_alignment_panel is None:
            return
        changed_dataset_indices: list[int] = []
        for ch in self._channels:
            transform = ch.get("transform") or {}
            if not any(abs(float(transform.get(key, 0.0))) > 1e-12 for key in ("dx_nm", "dy_nm", "angle")):
                continue
            ds_idx = ch.get("dataset_idx")
            if ds_idx is None or not (0 <= ds_idx < len(self._state.datasets)):
                continue
            ds = self._state.datasets[ds_idx]
            base_transform = ch.get("loc_transform") or ds.state.get("overlay_transform") or ds.state.get("render_transform_2d")
            base = transform_to_matrix4(base_transform)
            if base is None:
                base = identity_matrix4()
            matrix = manual_alignment_matrix4(transform, self._active_plane()) @ base
            record = dict(base_transform or {})
            record["matrix_4x4"] = matrix.tolist()
            record["matrix_3x3"] = matrix4_to_xy3(matrix).tolist()
            record["alignment_mode"] = "manual"
            provenance = dict(record.get("provenance") or {})
            provenance["manual_alignment"] = {
                "orientation": self._active_plane(),
                "method": "keyboard/drag translation and keyboard rotation",
            }
            record["provenance"] = provenance
            ds.state["overlay_transform"] = record
            ds.state["render_transform_2d"] = record
            ch["loc_transform"] = record
            ch["transform"] = {}
            changed_dataset_indices.append(ds_idx)
        self._cached_dataset_idx = None
        self._cached_locs_nm = None
        self._redraw_current(save_state=False)
        self._end_overlay_alignment()
        for ds_idx in changed_dataset_indices:
            self._state.notify_overlay_transform_changed(ds_idx)

    def _overlay_alignment_cancel(self) -> None:
        if self._overlay_alignment_original is not None:
            for ch, original in zip(self._channels, self._overlay_alignment_original, strict=True):
                ch["transform"] = dict(original)
        if self._overlay_alignment_original_visibility is not None:
            for ch, visible in zip(self._channels, self._overlay_alignment_original_visibility, strict=True):
                ch["visible"] = visible
        self._cached_dataset_idx = None
        self._cached_locs_nm = None
        self._redraw_current(save_state=False)
        self._end_overlay_alignment()

    def _end_overlay_alignment(self) -> None:
        panel = self._overlay_alignment_panel
        self._overlay_alignment_panel = None
        self._overlay_alignment_original = None
        self._overlay_alignment_original_visibility = None
        if panel is not None:
            panel.detach()
            self._root_layout.removeWidget(panel)
            panel.deleteLater()
        self._channel_area.show()

    def _active_channel_index(self) -> int | None:
        for i, ch in enumerate(self._channels):
            if ch.get("dataset_idx") == self._dataset_idx:
                return i
        return 0 if self._channels else None

    def _refresh_channel_highlight(self) -> None:
        """Bold the active channel's name (the one the right-click menu targets)."""
        active_idx = self._active_channel_index()
        for i, (name_lbl, _swatch) in enumerate(self._channel_rows):
            font = name_lbl.font()
            font.setBold(i == active_idx)
            name_lbl.setFont(font)

    def _on_channel_row_pressed(self, row: QWidget, event, ch_idx: int) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            menu = QMenu(row)
            action = menu.addAction("Manual align")
            action.setEnabled(len(self._channels) > 1)
            action.triggered.connect(lambda _checked=False: self._manual_align_channel(ch_idx))
            menu.exec(event.globalPosition().toPoint())
            return
        if event.button() == Qt.MouseButton.LeftButton and 0 <= ch_idx < len(self._channels):
            ds_idx = self._channels[ch_idx]["dataset_idx"]
            if 0 <= ds_idx < len(self._state.datasets):
                self._dataset_idx = ds_idx
                self._state.set_active(ds_idx)
                self._update_overlay_title()
                self._refresh_channel_highlight()
                self._update_colorbar_visibility()
        QWidget.mousePressEvent(row, event)

    def _on_channel_visible(self, ch_idx: int, visible: bool) -> None:
        if 0 <= ch_idx < len(self._channels):
            self._channels[ch_idx]["visible"] = bool(visible)
            self._redraw_current(save_state=False)

    def _on_channel_lut(self, ch_idx: int, lut: str) -> None:
        if 0 <= ch_idx < len(self._channels):
            self._channels[ch_idx]["lut"] = lut
            self._channels[ch_idx]["color_by"] = None    # solid color picked
            ds_idx = self._channels[ch_idx]["dataset_idx"]
            if 0 <= ds_idx < len(self._state.datasets):
                self._state.datasets[ds_idx].state["render_channel_lut"] = lut
                self._state.datasets[ds_idx].state["overlay_lut"] = lut
            if 0 <= ch_idx < len(self._channel_rows):
                self._style_channel_swatch(self._channel_rows[ch_idx][1], lut)
            self._redraw_current(save_state=False)

    def _set_channel_color_by(self, ch_idx: int, attr: str | None) -> None:
        """Color an overlay channel by an attribute (``attr``) or, when ``None``,
        revert it to its solid channel color."""
        if 0 <= ch_idx < len(self._channels):
            self._channels[ch_idx]["color_by"] = attr
            self._manual_color_levels = None          # auto-scale to the new attribute
            self._invalidate_color_cache()
            self._redraw_current(save_state=False)
            self.sync_lut_dialog()

    def _update_overlay_title(self) -> None:
        ds = self._dataset()
        if ds is None:
            return
        overlay_idx = ds.state.get("overlay_index")
        if overlay_idx and len(self._channels) > 1:
            self.setWindowTitle(f"Scatter Plot (overlay {overlay_idx}) - {ds.name}")
        else:
            self.setWindowTitle(f"Scatter Plot  -  {ds.name}")

    # ------------------------------------------------------------------
    # Drawing — dispatches to 2D or 3D path
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        if self._overlay_alignment_panel is not None:
            self._overlay_alignment_cancel()
        ds = self._dataset()
        if ds is None:
            self._clear_2d_panes()
            if self._3d_view is not None:
                self._3d_scatter.setData(pos=np.empty((0, 3)))
                self._refresh_3d_reference_items()
            self.setWindowTitle("Scatter Plot")
            return

        self._build_channels()
        self._rebuild_channel_ui()
        self._update_overlay_title()
        saved = ds.state.get(self._view_state_key, {})
        self._show_2d_axis = bool(saved.get("show_2d_axis", True))
        self._show_3d_axis = bool(saved.get("show_3d_axis", True))
        self._show_2d_grid = bool(saved.get("show_2d_grid", True))
        self._show_3d_grid = bool(saved.get("show_3d_grid", True))
        self._show_3d_bounding_box = bool(
            saved.get("show_3d_bounding_box", True)
        )
        self._show_colorbar = bool(saved.get("show_colorbar", True))
        self._colorbar_show_values = bool(
            saved.get("colorbar_show_values", True)
        )
        self._colorbar_orientation = (
            "horizontal"
            if saved.get("colorbar_orientation") == "horizontal"
            else "vertical"
        )
        stored_geometry = saved.get("colorbar_geometry")
        self._colorbar_geometry = (
            list(stored_geometry)
            if isinstance(stored_geometry, (list, tuple))
            else None
        )
        if self._colorbar is not None:
            self._colorbar.set_orientation(
                self._colorbar_orientation, notify=False
            )
            self._colorbar.set_show_values(
                self._colorbar_show_values, notify=False
            )
            self._colorbar.restore_geometry(self._colorbar_geometry)
        self._point_symbol = str(saved.get("point_symbol", "o"))
        self._point_size = max(1, min(50, int(saved.get("point_size", 2))))
        self._point_alpha = max(0, min(255, int(saved.get("point_alpha", 255))))

        self._axis_combo.blockSignals(True)
        axis_default = saved.get("axis", "XY")
        # A saved ortho state must not resurrect on a 2-D dataset, whose YZ/XZ
        # panes would be a line.
        if axis_default == ORTHO_AXIS and not self._ortho_available():
            axis_default = "XY"
        if self._axis_combo.findText(axis_default) >= 0:
            self._axis_combo.setCurrentText(axis_default)
        self._axis_combo.blockSignals(False)
        self._last_axis_text = self._axis_combo.currentText()
        self._apply_y_axis_direction()
        self._ortho.set_active(self._ortho_active())
        self._sync_ortho_colorbar(self._ortho_active())
        if self._axis_combo.currentText() == "3D":
            self._ensure_3d_built()
            if self._3d_view is not None:
                if self._3d_grid is not None:
                    self._3d_grid.setVisible(self._show_3d_grid)
                self._stack.setCurrentWidget(self._3d_view)
        else:
            self._stack.setCurrentWidget(self._plot_page)
        self._apply_2d_reference_visibility()
        self._update_colorbar_visibility()

        self._cmap_combo.blockSignals(True)
        prefs_cmap = self._state.prefs.get("plot", {}).get("scatter_cmap", "jet")
        cmap_default = saved.get("colormap") or prefs_cmap
        self._cmap_combo.setCurrentText(cmap_default)
        self._cmap_combo.blockSignals(False)
        self._cmap = _load_cmap(self._cmap_combo.currentText())

        self._black_bg_check.blockSignals(True)
        self._black_bg_check.setChecked(bool(saved.get("black_background", False)))
        self._black_bg_check.blockSignals(False)
        self._apply_background(self._black_bg_check.isChecked())

        # Populate color-by combo (preserve selection if possible)
        old = self._cbar_combo.currentText()
        self._cbar_combo.blockSignals(True)
        self._cbar_combo.clear()
        numeric_attrs = plot_attribute_names(ds, self._state.prefs, exclude=("ftr",))
        self._cbar_combo.addItems(numeric_attrs)
        apply_attribute_tooltips(self._cbar_combo)
        prefs_color_by = self._state.prefs.get("plot", {}).get("scatter_color_by", "tid")
        color_default = saved.get("color_by") or old or prefs_color_by
        if color_default in numeric_attrs:
            self._cbar_combo.setCurrentText(color_default)
        elif old in numeric_attrs:
            self._cbar_combo.setCurrentText(old)
        elif prefs_color_by in numeric_attrs:
            self._cbar_combo.setCurrentText(prefs_color_by)
        elif "tid" in numeric_attrs:
            self._cbar_combo.setCurrentText("tid")
        self._cbar_combo.blockSignals(False)

        self._redraw_current(save_state=True)

    def _redraw_current(self, *, save_state: bool) -> None:
        ds = self._dataset()
        if ds is None:
            return
        if len(self._channels) > 1:
            self._draw_overlay(save_state=save_state)
            self._redraw_roi_highlight()
            return
        self._draw(self._current_locs(ds), ds.filter_mask, ds, save_state=save_state)
        self._redraw_roi_highlight()

    def _current_locs(self, ds) -> np.ndarray:
        idx = self._dataset_idx
        if self._cached_dataset_idx != idx or self._cached_locs_nm is None:
            locs = np.asarray(ds.loc_nm, dtype=float)
            if locs.ndim == 2 and locs.shape[1] == 2:
                locs = np.column_stack([locs, np.zeros(locs.shape[0], dtype=float)])
            locs = apply_display_transform_nm(
                locs,
                ds.state.get("overlay_transform") or ds.state.get("render_transform_2d"),
            )
            self._cached_dataset_idx = idx
            self._cached_locs_nm = locs
        return self._cached_locs_nm

    def _locs_for_dataset(self, ds) -> np.ndarray:
        locs = np.asarray(ds.loc_nm, dtype=float)
        if locs.ndim == 2 and locs.shape[1] == 2:
            locs = np.column_stack([locs, np.zeros(locs.shape[0], dtype=float)])
        transform = ds.state.get("overlay_transform") or ds.state.get("render_transform_2d")
        ds_idx = next(
            (index for index, candidate in enumerate(self._state.datasets) if candidate is ds),
            None,
        )
        channel = next(
            (ch for ch in self._channels if ch.get("dataset_idx") == ds_idx),
            None,
        )
        if channel is not None:
            preview = channel.get("transform") or {}
            if any(abs(float(preview.get(key, 0.0))) > 1e-12 for key in ("dx_nm", "dy_nm", "angle")):
                base = transform_to_matrix4(transform)
                if base is None:
                    base = identity_matrix4()
                matrix = manual_alignment_matrix4(preview, self._active_plane()) @ base
                transform = {"matrix_4x4": matrix.tolist()}
        return apply_display_transform_nm(locs, transform)

    def _roi_masks_for_dataset(self, ds) -> list[tuple[object, np.ndarray]]:
        from .roi_highlight import highlight_masks
        return highlight_masks(self._state, ds)

    def _highlight_color(self):
        """COLOR ▸ ROI ▸ highlight data in ROI (one color for every ROI)."""
        from .roi_highlight import highlight_color
        return highlight_color(self._state.prefs)

    def _roi_highlight_brushes(self, record, count: int) -> list:
        from .roi_highlight import highlight_brushes
        return highlight_brushes(self._state.prefs, count)

    def _roi_highlight_rgba(self, record, count: int, alpha: float = 0.95) -> np.ndarray:
        from .roi_highlight import highlight_rgba
        return highlight_rgba(self._state.prefs, count, alpha)

    def _clear_roi_highlight(self) -> None:
        for highlight in getattr(self, "_pane_highlights", {}).values():
            highlight.setData([], [])
        if self._roi_highlight_3d is not None:
            self._roi_highlight_3d.setData(pos=np.empty((0, 3), dtype=np.float32))

    def _owns_active_roi_draft(self) -> bool:
        """True when an ROI is currently being drawn in *this* scatter view."""
        from .roi_highlight import owns_active_draft
        return owns_active_draft(getattr(self, "_roi_overlay", None))

    def _roi_highlight_enabled(self) -> bool:
        from .roi_highlight import roi_highlight_enabled
        return roi_highlight_enabled(
            self._state.prefs, is_source=self._owns_active_roi_draft())

    def _redraw_roi_highlight(self) -> None:
        if not self._roi_highlight_enabled():
            self._clear_roi_highlight()
            return
        if self._axis_combo.currentText() == "3D":
            self._redraw_roi_highlight_3d()
        else:
            self._redraw_roi_highlight_2d()

    def _redraw_roi_highlight_2d(self) -> None:
        if self._roi_highlight_2d is None:
            return
        targets = self._pane_targets()
        if not targets:
            self._clear_roi_highlight()
            return
        # The highlight is "these localizations", which projects truthfully into
        # any plane — unlike a ROI *shape*, whose in-plane geometry means nothing
        # in the other two panes. So the highlight is drawn in all three; only
        # the drawn ROI itself stays on the XY pane.
        picked: list[tuple[np.ndarray, np.ndarray]] = []   # (locs, indices)
        brushes: list = []
        channels = self._channels or [{"dataset_idx": self._dataset_idx, "visible": True}]
        per_channel_max = max(1, _MAX_DISPLAY_POINTS_2D // max(len(channels), 1))
        for ch in channels:
            if not ch.get("visible", True):
                continue
            ds_idx = ch.get("dataset_idx")
            if ds_idx is None or not (0 <= ds_idx < len(self._state.datasets)):
                continue
            ds = self._state.datasets[ds_idx]
            locs = self._locs_for_dataset(ds)
            if locs.ndim != 2 or locs.shape[0] == 0:
                continue
            for record, mask in self._roi_masks_for_dataset(ds):
                n = min(mask.size, locs.shape[0])
                if n == 0:
                    continue
                visible = mask[:n] & np.all(np.isfinite(locs[:n, :3]), axis=1)
                indices = self._visible_indices(visible, n, per_channel_max)
                if indices.size:
                    picked.append((locs, indices))
                    brushes.extend(self._roi_highlight_brushes(record, indices.size))
        if not picked:
            for *_head, highlight, _cols in targets:
                highlight.setData([], [])
            return
        for _plane, _plot, _scatter, highlight, (ci, cj) in targets:
            highlight.setData(
                x=np.concatenate([locs[idx, ci] for locs, idx in picked]),
                y=np.concatenate([locs[idx, cj] for locs, idx in picked]),
                brush=brushes,
                pen=None,
                size=7,
            )

    def _redraw_roi_highlight_3d(self) -> None:
        if self._roi_highlight_3d is None:
            return
        parts: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        channels = self._channels or [{"dataset_idx": self._dataset_idx, "visible": True}]
        per_channel_max = max(1, _MAX_DISPLAY_POINTS_3D // max(len(channels), 1))
        for ch in channels:
            if not ch.get("visible", True):
                continue
            ds_idx = ch.get("dataset_idx")
            if ds_idx is None or not (0 <= ds_idx < len(self._state.datasets)):
                continue
            ds = self._state.datasets[ds_idx]
            locs = self._locs_for_dataset(ds)
            if locs.ndim != 2 or locs.shape[0] == 0:
                continue
            for record, mask in self._roi_masks_for_dataset(ds):
                n = min(mask.size, locs.shape[0])
                visible = mask[:n] & np.all(np.isfinite(locs[:n, :3]), axis=1)
                indices = self._visible_indices(visible, n, per_channel_max)
                if indices.size:
                    parts.append(locs[indices, :3])
                    colors.append(self._roi_highlight_rgba(record, indices.size))
        if not parts:
            self._roi_highlight_3d.setData(pos=np.empty((0, 3), dtype=np.float32))
            return
        pos = np.vstack(parts).astype(np.float32, copy=False)
        rgba = np.vstack(colors).astype(np.float32, copy=False)
        self._roi_highlight_3d.setData(pos=pos, color=rgba, size=7.0, pxMode=True)

    def _lut_color(self, lut: str, alpha: int = 190) -> tuple[int, int, int, int]:
        alpha = int(round(int(alpha) * self._point_alpha / 255.0))
        if is_solid_color(lut):
            r, g, b, color_alpha = solid_color_rgba(lut)
            alpha = int(round(alpha * color_alpha / 255.0))
            return int(r), int(g), int(b), int(alpha)
        try:
            cmap = _load_cmap(lut)
            color = cmap.map(np.array([0.72]), mode="byte")[0]
            if len(color) >= 4:
                alpha = int(round(alpha * int(color[3]) / 255.0))
            return int(color[0]), int(color[1]), int(color[2]), int(alpha)
        except Exception:
            return 180, 180, 180, int(alpha)

    def _draw_overlay(self, *, save_state: bool) -> None:
        ds_active = self._dataset()
        if ds_active is not None and save_state:
            self._save_view_state(ds_active)
        axis = self._axis_combo.currentText()
        self._apply_y_axis_direction()
        if axis == "3D":
            self._draw_overlay_3d()
            return
        targets = self._pane_targets()
        if not targets:
            return
        primary_ci, primary_cj = targets[0][4]
        # Gather each channel's rows and brushes once; the panes differ only in
        # which two columns of the same rows they plot.
        picked: list[tuple[np.ndarray, np.ndarray]] = []   # (locs, indices)
        brushes: list = []
        total = 0
        for ch in self._channels:
            if not ch.get("visible", True):
                continue
            ds = self._state.datasets[ch["dataset_idx"]]
            locs = self._locs_for_dataset(ds)
            if locs.ndim != 2 or locs.shape[1] < 3:
                continue
            mask = np.asarray(ds.filter_mask, dtype=bool)
            if mask.shape[0] != locs.shape[0]:
                mask = np.ones(locs.shape[0], dtype=bool)
            mask &= np.all(np.isfinite(locs[:, :3]), axis=1)
            mask = self._ortho_row_filter(locs, mask)
            indices = self._visible_indices(mask, locs.shape[0], max(1, _MAX_DISPLAY_POINTS_2D // max(len(self._channels), 1)))
            if indices.size == 0:
                continue
            picked.append((locs, indices))
            color_by = ch.get("color_by")
            if color_by:
                _v, bins, _lbl, _lo, _hi = self._color_bins_for_points(
                    locs[indices, primary_ci], locs[indices, primary_cj],
                    None, ds, indices, attr=color_by)
                brushes.extend(self._brushes_for_bins(bins))
            else:
                color = self._lut_color(str(ch.get("lut", "Gray")))
                brushes.extend([pg.mkBrush(*color)] * indices.size)
            total += int(np.count_nonzero(mask))
        if not picked:
            self._clear_2d_panes()
            self._last_color_values = np.empty(0, dtype=float)
            self._update_colorbar_visibility()
            self._set_info_text("No localisations pass the current filters.")
            return
        for plane, plot, scatter, _highlight, (ci, cj) in targets:
            scatter.setData(
                x=np.concatenate([locs[idx, ci] for locs, idx in picked]),
                y=np.concatenate([locs[idx, cj] for locs, idx in picked]),
                brush=brushes,
                pen=None,
                size=self._point_size,
                symbol=self._point_symbol,
            )
            ax_x, ax_y = self._pane_labels(plane)
            plot.setLabel("bottom", ax_x)
            plot.setLabel("left", ax_y)
        self._sync_ortho_depth(picked)
        self._update_colorbar_visibility()
        self._set_info_text(f"{total:,} filtered localisations across {len([c for c in self._channels if c.get('visible', True)])} channel(s)")

    def _draw_overlay_3d(self) -> None:
        self._ensure_3d_built()
        if self._3d_view is None:
            return
        pos_parts: list[np.ndarray] = []
        rgba_parts: list[np.ndarray] = []
        total = 0
        for ch in self._channels:
            if not ch.get("visible", True):
                continue
            ds = self._state.datasets[ch["dataset_idx"]]
            locs = self._locs_for_dataset(ds)
            mask = np.asarray(ds.filter_mask, dtype=bool)
            if mask.shape[0] != locs.shape[0]:
                mask = np.ones(locs.shape[0], dtype=bool)
            mask &= np.all(np.isfinite(locs[:, :3]), axis=1)
            indices = self._visible_indices(mask, locs.shape[0], max(1, _MAX_DISPLAY_POINTS_3D // max(len(self._channels), 1)))
            if indices.size == 0:
                continue
            pos_parts.append(locs[indices, :3])
            color_by = ch.get("color_by")
            if color_by:
                p = locs[indices, :3]
                _v, bins, _lbl, _lo, _hi = self._color_bins_for_points(
                    p[:, 0], p[:, 1], p[:, 2], ds, indices, attr=color_by)
                rgba_parts.append(self._rgba_for_bins(bins, for_3d=True).astype(np.float32))
            else:
                r, g, b, a = self._lut_color(str(ch.get("lut", "Gray")), alpha=220)
                rgba = np.tile(np.array([[r / 255.0, g / 255.0, b / 255.0, a / 255.0]], dtype=np.float32), (indices.size, 1))
                rgba_parts.append(rgba)
            total += int(np.count_nonzero(mask))
        if not pos_parts:
            self._3d_scatter.setData(pos=np.empty((0, 3)))
            self._refresh_3d_reference_items()
            self._last_color_values = np.empty(0, dtype=float)
            self._update_colorbar_visibility()
            self._set_info_text("No finite XYZ localisations pass the current filters for 3D display.")
            return
        pos = np.vstack(pos_parts).astype(np.float32, copy=False)
        rgba = np.vstack(rgba_parts).astype(np.float32, copy=False)
        self._3d_scatter.setData(
            pos=pos, color=rgba, size=self._point_size_3d(), pxMode=True
        )
        self._refresh_3d_reference_items()
        if not self._3d_camera_initialised:
            self._reset_3d_camera(pos)
            self._3d_camera_initialised = True
        self._update_colorbar_visibility()
        self._set_info_text(f"{total:,} filtered localisations across {len([c for c in self._channels if c.get('visible', True)])} channel(s)")

    def _draw(self, locs: np.ndarray, ftr: np.ndarray, ds, *, save_state: bool = True) -> None:
        if save_state:
            self._save_view_state(ds)
        is_3d = self._axis_combo.currentText() == "3D"
        if is_3d:
            self._draw_3d(locs, ftr, ds)
        else:
            self._draw_2d(locs, ftr, ds)

    def _save_view_state(self, ds=None) -> None:
        ds = ds or self._dataset()
        if ds is None:
            return
        ds.state[self._view_state_key] = {
            "color_by": self._cbar_combo.currentText(),
            "colormap": self._cmap_combo.currentText(),
            "black_background": self._black_bg_check.isChecked(),
            "show_2d_axis": bool(self._show_2d_axis),
            "show_3d_axis": bool(self._show_3d_axis),
            "show_2d_grid": bool(self._show_2d_grid),
            "show_3d_grid": bool(self._show_3d_grid),
            "show_3d_bounding_box": bool(self._show_3d_bounding_box),
            "show_colorbar": bool(self._show_colorbar),
            "colorbar_show_values": bool(self._colorbar_show_values),
            "colorbar_orientation": self._colorbar_orientation,
            "colorbar_geometry": self._colorbar_geometry,
            "point_symbol": self._point_symbol,
            "point_size": int(self._point_size),
            "point_alpha": int(self._point_alpha),
            "axis": self._axis_combo.currentText(),
        }

    # -- 2D path -----------------------------------------------------

    def _draw_2d(self, locs: np.ndarray, ftr: np.ndarray, ds) -> None:
        axis = self._axis_combo.currentText()
        self._apply_y_axis_direction()
        targets = self._pane_targets()
        if not targets:
            return

        # One thinned index set for every pane, so the three projections show
        # the same localizations rather than three independent samples, and the
        # side panes project only what the XY pane is showing.
        n_passing = int(np.count_nonzero(np.asarray(ftr, dtype=bool)))
        ftr = self._ortho_row_filter(locs, ftr)
        indices = self._visible_indices(ftr, locs.shape[0], _MAX_DISPLAY_POINTS_2D)
        n_visible = int(np.count_nonzero(np.asarray(ftr, dtype=bool)))
        n_display = indices.size
        if n_display == 0:
            self._clear_2d_panes()
            self._last_color_values = np.empty(0, dtype=float)
            self._update_colorbar_visibility()
            self._set_info_text("No localisations pass the current filter.", ds)
            return

        # The colour bins depend only on the row indices, not on the projection,
        # so they are resolved once and the brushes reused across the panes.
        primary_ci, primary_cj = targets[0][4]
        c_vals, color_bins, c_label, vmin, vmax = self._color_bins_for_points(
            locs[indices, primary_ci], locs[indices, primary_cj], None, ds, indices
        )
        self._last_color_values = np.asarray(c_vals, dtype=float)
        brushes = self._brushes_for_bins(color_bins)

        for plane, plot, scatter, _highlight, (ci, cj) in targets:
            scatter.setData(
                x=locs[indices, ci], y=locs[indices, cj],
                brush=brushes,
                pen=None,
                size=self._point_size,
                symbol=self._point_symbol,
            )
            ax_x, ax_y = self._pane_labels(plane)
            plot.setLabel("bottom", ax_x)
            plot.setLabel("left", ax_y)

        self._sync_ortho_depth([(locs, indices)])
        self._update_colorbar_visibility()

        display_note = (
            f"showing {n_display:,} / {n_visible:,} passing"
            if n_display < n_visible
            else f"{n_visible:,}"
        )
        axis_note = axis
        if self._ortho_active():
            axis_note = self._ortho_axis_note(cropped=n_visible < n_passing)
        self._set_info_text(
            f"{display_note} / {ds.prop.num_loc:,} localisations  "
            f"({100*n_visible/ds.prop.num_loc:.1f} %)  |  axis: {axis_note}  |  "
            f"color: {c_label}",
            ds,
        )

    # -- 3D path -----------------------------------------------------

    def _draw_3d(self, locs: np.ndarray, ftr: np.ndarray, ds) -> None:
        if self._3d_view is None:
            return  # PyOpenGL not installed

        indices = self._visible_indices(ftr, locs.shape[0], _MAX_DISPLAY_POINTS_3D)
        n_visible = int(np.count_nonzero(np.asarray(ftr, dtype=bool)))
        raw_display = indices.size
        if raw_display == 0:
            self._3d_scatter.setData(pos=np.empty((0, 3)))
            self._refresh_3d_reference_items()
            self._last_color_values = np.empty(0, dtype=float)
            self._update_colorbar_visibility()
            self._set_info_text("No localisations pass the current filter.", ds)
            return

        pos = np.asarray(locs[indices, :3], dtype=float)
        finite_mask = np.all(np.isfinite(pos), axis=1)
        if not np.all(finite_mask):
            indices = indices[finite_mask]
            pos = pos[finite_mask]
        n_display = indices.size
        if n_display == 0:
            self._3d_scatter.setData(pos=np.empty((0, 3)))
            self._refresh_3d_reference_items()
            self._last_color_values = np.empty(0, dtype=float)
            self._update_colorbar_visibility()
            self._set_info_text(
                "No finite XYZ localisations pass the current filter for 3D display.",
                ds,
            )
            return

        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        c_vals, color_bins, c_label, vmin, vmax = self._color_bins_for_points(x, y, z, ds, indices)
        self._last_color_values = np.asarray(c_vals, dtype=float)
        rgba = self._rgba_for_bins(color_bins, for_3d=True)

        pos = pos.astype(np.float32, copy=False)
        self._3d_scatter.setData(
            pos=pos, color=rgba, size=self._point_size_3d(), pxMode=True
        )
        self._refresh_3d_reference_items()

        # First-time camera setup
        if not self._3d_camera_initialised:
            self._reset_3d_camera(pos)
            self._3d_camera_initialised = True
        self._update_colorbar_visibility()

        display_note = (
            f"showing {n_display:,} / {n_visible:,} passing"
            if n_display < n_visible
            else f"{n_visible:,}"
        )
        self._set_info_text(
            f"{display_note} / {ds.prop.num_loc:,} localisations  "
            f"({100*n_visible/ds.prop.num_loc:.1f} %)  |  axis: 3D  |  "
            f"color: {c_label} ∈ [{vmin:.3g}, {vmax:.3g}]",
            ds,
        )

    @staticmethod
    def _visible_indices(ftr: np.ndarray, total: int, max_points: int) -> np.ndarray:
        from .roi_highlight import decimate
        return decimate(ftr, total, max_points)

    # -- shared color helpers --------------------------------------

    def _color_bins_for_points(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray | None,
        ds, indices: np.ndarray, attr: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, str, float, float]:
        """Return values, uint8 color bins, label, and display levels.

        ``attr`` overrides the window's color-by combo (used to color one
        overlay channel by a chosen attribute)."""
        c_name = attr if attr is not None else self._cbar_combo.currentText()
        # Resolve through _color_cache_for_dataset (→ attr_values_1d), not a bare
        # ``c_name in ds.attr`` check: coordinate views xnm/ynm/znm are NOT keys
        # in ds.attr (the store holds loc_x/loc_y/loc_z), so that check wrongly
        # rejected them and colored every point with bin 0 (one flat color).
        cache = self._color_cache_for_dataset(ds, c_name)
        if cache is None:
            values = np.zeros(indices.size, dtype=float)
            bins, vmin, vmax = self._map_values_to_bins(values)
            return values, bins, c_name, vmin, vmax
        return (
            cache["values"][indices],
            cache["bins"][indices],
            c_name,
            cache["vmin"],
            cache["vmax"],
        )

    def _color_cache_for_dataset(self, ds, c_name: str) -> dict | None:
        key = (
            id(ds),
            c_name,
            self._cmap_combo.currentText(),
            self._lut_invert,
            self._manual_color_levels,
            id(self._cmap),
            ds.prop.num_loc,
        )
        if self._color_cache_key == key:
            return self._color_cache

        values = attr_values_1d(ds, c_name)
        values = np.empty(0) if values is None else np.asarray(values).ravel().astype(float)
        if values.size != ds.prop.num_loc:
            self._color_cache_key = key
            self._color_cache = None
            return None

        bins, vmin, vmax = self._map_values_to_bins(values)
        self._color_cache_key = key
        self._color_cache = {
            "values": values,
            "bins": bins,
            "vmin": vmin,
            "vmax": vmax,
        }
        return self._color_cache

    def _map_values_to_bins(self, c_vals: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Robust normalise values into 256 LUT indices."""
        c_vals = np.asarray(c_vals, dtype=float)
        finite = np.asarray(c_vals, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            c_vals = np.zeros_like(c_vals, dtype=float)
            vmin, vmax = 0.0, 1.0
        elif self._manual_color_levels is not None:
            vmin, vmax = self._manual_color_levels
        else:
            vmin, vmax = np.nanpercentile(finite, [1, 99])
        if vmax <= vmin:
            vmax = vmin + 1.0
        normed = np.clip((c_vals - vmin) / (vmax - vmin), 0, 1)
        normed = np.nan_to_num(normed, nan=0.0, posinf=1.0, neginf=0.0)
        bins = np.rint(normed * 255.0).astype(np.uint8)
        return bins, float(vmin), float(vmax)

    def _ensure_color_luts(self) -> None:
        key = (
            self._cmap_combo.currentText(),
            self._lut_invert,
            id(self._cmap),
            self._point_alpha,
        )
        if self._brush_lut_key == key and self._brush_lut is not None and self._rgba_lut is not None:
            return
        rgba = np.asarray(
            self._cmap.getLookupTable(0.0, 1.0, 256, alpha=True),
            dtype=np.uint8,
        ).copy()
        rgba[:, 3] = np.rint(
            rgba[:, 3].astype(float) * self._point_alpha / 255.0
        ).astype(np.uint8)
        self._brush_lut = [
            pg.mkBrush(*(int(channel) for channel in color)) for color in rgba
        ]
        self._rgba_lut = rgba.astype(np.float32) / 255.0
        self._brush_lut_key = key

    def _brushes_for_bins(self, bins: np.ndarray) -> list:
        self._ensure_color_luts()
        lut = self._brush_lut or []
        return [lut[int(i)] for i in np.asarray(bins, dtype=np.uint8)]

    def _rgba_for_bins(self, bins: np.ndarray, *, for_3d: bool = False) -> np.ndarray:
        self._ensure_color_luts()
        lut = self._rgba_lut
        if lut is None:
            return np.empty((0, 4), dtype=np.float32)
        rgba = lut[np.asarray(bins, dtype=np.uint8)]
        if for_3d:
            rgba = rgba.copy()
            # GL point sprites are tiny and can visually disappear when bright
            # LUT colors blend over a white clear color. Keep 2D colors exact,
            # but darken only the too-bright 3D colors on white backgrounds.
            if not self._background_is_black() and rgba.size:
                rgb = rgba[:, :3]
                luminance = (
                    0.2126 * rgb[:, 0]
                    + 0.7152 * rgb[:, 1]
                    + 0.0722 * rgb[:, 2]
                )
                bright = luminance > 0.58
                if np.any(bright):
                    scale = np.clip(0.58 / luminance[bright], 0.55, 1.0)
                    rgb[bright] *= scale[:, None]
                rgba[:, 3] = self._point_alpha / 255.0
            else:
                rgba[:, 3] = self._point_alpha / 255.0
        return rgba

    def _invalidate_color_cache(self) -> None:
        self._color_cache_key = None
        self._color_cache = None

    def open_lut_dialog(self) -> None:
        from .lut_dialog import shared_lut_dialog

        shared_lut_dialog(
            self,
            on_levels_changed=self._on_lut_levels_changed,
            on_cmap_changed=self._on_lut_cmap_changed,
            on_invert_changed=self._on_lut_invert_changed,
            on_gamma_changed=self._on_lut_gamma_changed,
            state=self._state,
        )
        if not self._refresh_lut_dialog(capture_baseline=True):
            self._set_info_text("LUT unavailable: no color values to display.")
            return
        self._lut_dialog.show()
        self._lut_dialog.raise_()
        self._lut_dialog.activateWindow()

    def _refresh_lut_dialog(self, *, capture_baseline: bool) -> bool:
        """(Re)load the LUT dialog from the current color values / colormap.
        Returns False when there is nothing to color."""
        dlg = self._lut_dialog
        if dlg is None:
            return False
        vals = np.asarray(self._last_color_values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            self._refresh()
            vals = np.asarray(self._last_color_values, dtype=float)
            vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return False
        data_lo = float(np.nanmin(vals))
        data_hi = float(np.nanmax(vals))
        if data_hi <= data_lo:
            data_hi = data_lo + 1.0
        lo, hi = self._manual_color_levels or tuple(np.nanpercentile(vals, [1, 99]))
        dlg.load_image(
            pixels=vals, data_lo=data_lo, data_hi=data_hi,
            lo=float(lo), hi=float(hi),
            cmap_name=self._cmap_combo.currentText(),
            invert=self._lut_invert,
            gamma=self._lut_gamma,
            capture_baseline=capture_baseline,
        )
        return True

    def sync_lut_dialog(self) -> None:
        """Push the current colormap / color-by / levels into an open LUT dialog,
        so external changes reflect in realtime. Skipped while the user is editing
        the dialog itself (it is the active window then)."""
        dlg = self._lut_dialog
        try:
            if dlg is None or not dlg.isVisible() or dlg.isActiveWindow():
                return
        except RuntimeError:
            return
        self._refresh_lut_dialog(capture_baseline=False)

    def roi_view_plane(self) -> str | None:
        """Current scatter projection for ROI 3-D placement (XY/XZ/YZ); ``None``
        in 3-D mode (ROIs are not drawn there)."""
        axis = self._active_plane()
        return axis if axis in {"XY", "XZ", "YZ"} else None

    def coordinate_view_box(self):
        """The 2-D coordinate ViewBox for overlays (e.g. a scale bar), or None
        in 3-D mode."""
        if self.roi_view_plane() is None:
            return None
        return self._plot_2d.getPlotItem().getViewBox()

    def _profile_channels(self):
        return self._channels or [
            {"dataset_idx": self._dataset_idx, "visible": True, "kind": "localizations"}]

    def profile_localizations(self):
        """``(M, 2)`` filtered, visible localizations projected into the current 2-D
        scatter projection (display nm), for the Plot Profile. ``None`` in 3-D mode."""
        if self.roi_view_plane() is None:
            return None
        from ..core.roi_crop import plane_localizations
        return plane_localizations(self._state, self._profile_channels(), self.roi_view_plane())

    def profile_locs_version(self):
        """Cheap token — changes only when :meth:`profile_localizations` would
        (dataset / filter / Z scaling factor / visibility / projection), never on zoom/pan."""
        if self.roi_view_plane() is None:
            return None
        from ..core.roi_crop import plane_localizations_version
        return plane_localizations_version(
            self._state, self._profile_channels(), self.roi_view_plane())

    def roi_depth_center(self) -> float | None:
        """Centre of the data extent of the out-of-plane (depth) axis — the
        value a drawn ROI gets in the dimension not shown in this projection.
        The scatter view has no depth slider, so the data extent is the natural
        'current viewing range' of that axis."""
        depth_map = {"XY": 2, "XZ": 1, "YZ": 0}
        axis = self._active_plane()
        if axis not in depth_map:
            return None
        ds = self._dataset()
        if ds is None:
            return None
        locs = self._current_locs(ds)
        k = depth_map[axis]
        if locs.ndim != 2 or locs.shape[1] <= k:
            return None
        col = locs[:, k]
        col = col[np.isfinite(col)]
        if col.size == 0:
            return None
        return 0.5 * (float(col.min()) + float(col.max()))

    def roi_depth_range(self):
        """Visible range of the out-of-plane axis. The scatter view has no depth
        slider, so the data extent of that axis is its viewing range -- the same
        reasoning ``roi_depth_center`` already uses for the centre."""
        depth_map = {"XY": 2, "XZ": 1, "YZ": 0}
        axis = self._active_plane()
        ds = self._dataset()
        if axis not in depth_map or ds is None:
            return None
        locs = self._current_locs(ds)
        k = depth_map[axis]
        if locs.ndim != 2 or locs.shape[1] <= k:
            return None
        col = locs[:, k]
        col = col[np.isfinite(col)]
        if col.size == 0:
            return None
        return float(col.min()), float(col.max())

    def roi_depths_at(self, points):
        """Data-aware out-of-plane value per drawn vertex (weighted median of the
        depth axis among localizations near that in-plane location); ``None`` per
        empty column so the caller falls back to ``roi_depth_center``."""
        axis = self._active_plane()
        if axis not in {"XY", "XZ", "YZ"} or not points:
            return [None] * len(points)
        ds = self._dataset()
        if ds is None:
            return [None] * len(points)
        locs = self._current_locs(ds)
        if locs.ndim != 2 or locs.shape[1] < 3:
            return [None] * len(points)
        from ..core.roi_depth import weighted_depths
        i, j = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}[axis]
        k = {"XY": 2, "XZ": 1, "YZ": 0}[axis]
        return weighted_depths(points, locs[:, i], locs[:, j], locs[:, k])

    def roi_dataset_indices(self) -> "set[int]":
        """Datasets this scatter displays — every overlay channel, not just the
        anchor it is keyed by."""
        indices = {ch["dataset_idx"] for ch in getattr(self, "_channels", [])
                   if isinstance(ch.get("dataset_idx"), int)}
        if isinstance(getattr(self, "_dataset_idx", None), int):
            indices.add(int(self._dataset_idx))
        return indices

    def normalize_roi_record(self, record):
        """Tag a drawn ROI with its view plane and the centre of the out-of-plane
        data range, so its third-dimension position is defined."""
        plane = self.roi_view_plane()
        if plane is None:
            return record
        ctx = dict(record.context)
        ctx.setdefault("view_plane", plane)
        ctx.setdefault("depth_axis", {"XY": "Z", "XZ": "Y", "YZ": "X"}[plane])
        # The Z scaling factor a ROI was DRAWN at. Recorded, never auto-applied:
        # a ROI's Z is frozen when drawn, and this is what makes it possible to
        # report one as stale and to pre-fill the exact ratio for the Manager's
        # Scale Z. Without it, "update with the Z scaling factor" is not
        # computable at all -- the old value cannot be recovered afterwards.
        _ds = self._dataset()
        _factor = getattr(getattr(_ds, "cali", None), "z_scaling_factor", None)
        if isinstance(_factor, (int, float)):
            ctx.setdefault("z_scaling_factor", float(_factor))
        if record.type != "point":
            center = self.roi_depth_center()
            if center is not None:
                ctx.setdefault("depth_value", float(center))
        record.context = ctx
        return record

    def compute_roi_selection(self, record):
        from ..core.roi_selection import VOLUME_ROI_TYPES

        if (record.type not in REGION_ROI_TYPES | VOLUME_ROI_TYPES
                or self._axis_combo.currentText() == "3D"):
            return None
        ds = self._dataset()
        if ds is None:
            return None
        locs = self._current_locs(ds)
        if locs.ndim != 2 or locs.shape[1] < 2:
            return None
        if locs.shape[1] == 2:
            locs = np.column_stack([locs, np.zeros(locs.shape[0], dtype=float)])

        axis = self._active_plane()
        if axis not in AXIS_COLUMNS:
            return None
        ci, cj = AXIS_COLUMNS[axis]
        base = np.asarray(ds.filter_mask, dtype=bool)
        if base.shape[0] != locs.shape[0]:
            base = np.ones(locs.shape[0], dtype=bool)
        base &= np.all(np.isfinite(locs[:, :3]), axis=1)
        if record.type in VOLUME_ROI_TYPES:
            from ..core.roi_volume import roi_volume_mask
            mask = roi_volume_mask(locs[:, 0], locs[:, 1], locs[:, 2],
                                   record, base_mask=base)
        else:
            mask = roi_region_mask(locs[:, ci], locs[:, cj], record, base_mask=base)
        context = {
            "source_view": "scatter",
            "dataset_idx": self._dataset_idx,
            "axis": axis,
            "x_axis": "XYZ"[ci],
            "y_axis": "XYZ"[cj],
        }
        return ds, mask, context

    def _on_lut_levels_changed(self, lo: float, hi: float) -> None:
        self._manual_color_levels = (float(lo), float(hi))
        self._invalidate_color_cache()
        self._update_color()

    def _on_lut_cmap_changed(self, name: str, invert: bool) -> None:
        self._lut_invert = bool(invert)
        self._cmap_combo.blockSignals(True)
        self._cmap_combo.setCurrentText(name)
        self._cmap_combo.blockSignals(False)
        self._cmap = make_colormap(name, invert=self._lut_invert, gamma=self._lut_gamma)
        self._update_colorbar_visibility()
        self._invalidate_color_cache()
        self._update_color()

    def _on_lut_invert_changed(self, invert: bool) -> None:
        self._on_lut_cmap_changed(self._cmap_combo.currentText(), invert)

    def _on_lut_gamma_changed(self, gamma: float) -> None:
        self._lut_gamma = float(gamma)
        self._on_lut_cmap_changed(self._cmap_combo.currentText(), self._lut_invert)

    def closeEvent(self, event) -> None:
        from .lut_dialog import release_shared_lut_owner
        from .qt_lifecycle import close_plot_widgets

        self._end_overlay_alignment()
        if self._roi_overlay is not None:
            self._roi_overlay.dispose()
            self._roi_overlay = None
        release_shared_lut_owner(self)
        # Drop the inter-pane links before the plots go inert: a queued range
        # change must not reach a half-torn-down pane.
        self._ortho.dispose()
        # ⚠ The side panes take the SAFE tier: they are children of _plot_page
        # and Qt deletes them with it, whereas close_plot_widgets also
        # reparents and deleteLater()s, leaving an orphan that outlives its
        # page — which reproduced as a Qt abort inside a later, unrelated test.
        # _plot_2d keeps the aggressive tier it has always had.
        from .qt_lifecycle import dispose_plot_widgets
        for _plane in SIDE_PLANES:
            dispose_plot_widgets(self._pane_plots[_plane])
        close_plot_widgets(self._plot_2d)
        super().closeEvent(event)

    def focusInEvent(self, event) -> None:
        if self._dataset_idx is not None and 0 <= self._dataset_idx < len(self._state.datasets):
            self._state.set_active(self._dataset_idx)
        if self._roi_overlay is not None and self._axis_combo.currentText() != "3D":
            self._roi_overlay.activate()
        super().focusInEvent(event)

    def changeEvent(self, event) -> None:
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            if self._dataset_idx is not None and 0 <= self._dataset_idx < len(self._state.datasets):
                self._state.set_active(self._dataset_idx)
            if self._roi_overlay is not None and self._axis_combo.currentText() != "3D":
                self._roi_overlay.activate()
        super().changeEvent(event)
