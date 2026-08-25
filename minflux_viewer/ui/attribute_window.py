"""
minflux_viewer.ui.attribute_window
=====================================
Attribute plot window — port of ``plot_attribute.m``.

Plots two to four numeric attributes against each other. X/Y/Z select a 2-D
projection or an interactive OpenGL 3-D scatter; an optional C dimension maps
linearly onto a perceptually uniform sequential colormap.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPoint, QRect, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QImage, QMatrix4x4
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..colormaps import (
    colormap_lut,
    custom_colormap_names,
    make_colormap,
)
from ..colors import component_colors, viewer_color
from ..core.app_state import AppState
from ..core.attributes import attribute_description, plot_attribute_names
from ..core.iteration import (
    FLATTEN_LABEL,
    STACKED_LABEL,
    iteration_bold_flags,
    iteration_labels,
    ordinal,
    parse_iteration_label,
)
from ..core.loader import (
    attr_matches_selection,
    attr_values_1d,
    effective_iterations_for_attr,
    is_value_pool_selector,
    mfx_filter_mask,
    mfx_get,
)
from ..core.roi_selection import rectangle_mask
from .attribute_cpu import (
    BulkScatterItem,
    aggregate_screen_points,
    joint_extent,
    spatial_representative_indices,
)
from .attribute_help import apply_attribute_menu_tooltips, apply_attribute_tooltips
from .gpu_capabilities import point_limit_from_memory
from .gl_3d_reference import three_plane_grid_positions, tick_values
from .plot_format import plot_widget

_VIEW_DIMENSIONS = {
    "XY": ("X", "Y"),
    "XZ": ("X", "Z"),
    "YZ": ("Y", "Z"),
    # The top row remains X/Y in 3-D; Z stays editable from the context menu.
    "3D": ("X", "Y"),
}
_VIEW_OPTIONS = tuple(_VIEW_DIMENSIONS)
# Perceptually uniform sequential maps suited to linearly normalized values.
_LINEAR_COLORMAPS = ("viridis", "cividis", "inferno", "magma", "plasma")
# How many markers one draw may put on screen. Measured on a 246k-localization
# MINFLUX file (900x700 window): a uniform-colour 2-D series costs ~0.2 s at
# 246k and ~1 s at 1.16 M, so 1 M is the edge of comfortable. A C-coloured
# series needs its own, far smaller budget: pyqtgraph keys a symbol per point
# when brushes vary, costing ~16 us/point (4.1 s at 246k) no matter how the
# brushes are built. Both are per *view*, not per dataset — see _series_data.
_MAX_DISPLAY_POINTS = 1_000_000
_MAX_COLOR_DISPLAY_POINTS = 50_000
# The GPU draws every point of a real acquisition (20.6 M in 2.7 s, 0.04 s per
# pan), so thinning is not applied there at all. This is only a guard against
# exhausting graphics memory, and the status line reports it if it ever bites.
# Fraction of the visible span kept as a margin, so a small pan does not
# immediately expose an unpainted edge.
_VIEW_THIN_MARGIN = 0.25
# Trailing debounce for re-thinning after the view range settles.
_RETHIN_DELAY_MS = 120


# Hover help for the per-window Thinning checkbox. Preferences > Appearance >
# Attribute Plot carries the long form of the same explanation.
THINNING_TOOLTIP = (
    "Keep spatial representatives from the points currently in view when there\n"
    "are more than one draw can paint responsively. No value is altered and no\n"
    "point is averaged away.\n"
    "\n"
    "The subsample is recomputed for the visible range, so zooming in restores\n"
    "the omitted points; the status line reports how many of how many points\n"
    "are drawn. In a dense region the marker density no longer reflects the\n"
    "data density. One row per occupied spatial cell is protected before the\n"
    "remaining capacity is filled, so isolated features are retained.\n"
    "\n"
    "Applies only to the legacy pyqtgraph CPU renderer. GPU mode is exact up to\n"
    "its startup memory-derived upload limit; the separate CPU-fix window uses\n"
    "bulk painting and complete screen-space aggregation instead.\n"
    "\n"
    "Uncheck for a faithful plot of every point. Expect seconds-long redraws\n"
    "on multi-million-row selections (~19 s at 20 M rows), repeated on every\n"
    "pan and zoom. The default comes from Preferences > Appearance."
)


class _DataBoundsItem(pg.GraphicsObject):
    """A rectangle that exists only to give the ViewBox something to fit.

    ⚠ Do **not** replace this with a `PlotDataItem` carrying the corner points:
    with no pen and no symbol it hides both of its children, and
    `ViewBox.childrenBounds` skips invisible items, so it contributes nothing —
    which is exactly the failure this class exists to fix.
    """

    def __init__(self, rect: QRectF) -> None:
        super().__init__()
        self._rect = QRectF(rect)

    def set_rect(self, rect: QRectF) -> None:
        self.prepareGeometryChange()
        self._rect = QRectF(rect)

    def boundingRect(self) -> QRectF:
        return QRectF(self._rect)

    def paint(self, *_args) -> None:
        return


def _row_selector(sel):
    """The row-identity selector for *sel*.

    ``all [sum]`` / ``all [average]`` produce one value per localization laid on
    the ``last`` rows, so anything that identifies a *row* (the synthetic ``idx``,
    ``tid``, a filter mask) must be fetched at ``last`` — pooling those would be
    meaningless. Every other selector is its own row selector.
    """
    return "last" if is_value_pool_selector(sel) else sel


def _iter_color(prefs: dict, k: int, n: int = 1) -> tuple[int, int, int, int]:
    """Colour of stacked iteration *k* of *n*, from the sequential ramp."""
    colors = list(component_colors(prefs, "functions", "Iteration series").values())
    if not colors:
        return (70, 130, 180, 255)
    if n <= 1 or len(colors) == 1:
        return colors[0]
    # Spread the ramp over the series actually drawn: with three iterations the
    # plot should still run end to end, not sit in the ramp's dark corner.
    position = min(k, n - 1) / (n - 1)
    return colors[min(len(colors) - 1, round(position * (len(colors) - 1)))]


class AttributeWindow(QWidget):
    """Plot two to four numeric attributes against each other."""

    TAG = "attribute_window"

    def __init__(
        self,
        state: AppState,
        parent: QWidget | None = None,
        *,
        dataset_idx: int | None = None,
        cpu_fix: bool = False,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._dataset_idx = dataset_idx if dataset_idx is not None else state.active_idx
        self._cpu_fix = bool(cpu_fix)
        self._window_label = "Attribute Plot (CPU fix)" if self._cpu_fix else "Attribute Plot"
        self._view_state_key = (
            "attribute_plot_cpu_state" if self._cpu_fix else "attribute_plot_state"
        )
        self._zoom_active = False
        self._zoom_mode = "unconstrained"
        self._zoom_preview = None
        self._zoom_drag_start = None
        self._view_box = None
        self._original_mouse_drag_event = None
        self._numeric_attrs: list[str] = []
        # Z and C are independent: a plot can be XY, XYZ, XYC or XYZC.
        self._has_z = False
        self._has_c = False
        # Thinning lives in Preferences and the View menu, not the top row.
        self._thinning = bool(
            state.prefs.get("plot", {}).get("attribute_thinning", True)
        )
        self._dimension_attrs = {"X": "", "Y": "", "Z": "", "C": ""}
        self._view_mode = "XY"
        self._c_mapping = _LINEAR_COLORMAPS[0]
        self._manual_color_levels: tuple[float, float] | None = None
        self._lut_invert = False
        self._lut_gamma = 1.0
        self._lut_dialog = None
        self._last_color_values = np.empty(0, dtype=float)
        self._black_background = False
        self._show_2d_axis = True
        self._show_3d_axis = True
        self._show_2d_grid = True
        self._show_3d_grid = True
        self._show_3d_bounding_box = True
        self._show_colorbar = True
        self._show_legend = True
        self._legend_geometry: list[int] | None = None
        self._view_restricted = False
        self._thinned = False
        self._thin_step = 1
        self._thin_method = "none"
        self._thin_drawable: int | None = None
        self._rethinking = False
        # One pre-thinning series, reused only while re-thinning after a pan or
        # zoom (see _rethin_now). Dropped by every ordinary draw, so it cannot
        # outlive the controls it was fetched for.
        self._series_cache: tuple | None = None
        self._colorbar_show_values = True
        self._colorbar_orientation = "vertical"
        self._colorbar_geometry: list[int] | None = None
        self._colorbar = None
        self._point_symbol = "o"
        # MATLAB line specifiers for the Lines option; the connecting curve is
        # a pyqtgraph item in both renderers, so this works on the GPU too.
        self._line_style = "-"
        self._line_width = 1.0
        self._point_size = 3
        self._point_color = tuple(viewer_color(state.prefs, "attribute_data"))
        self._point_alpha = int(self._point_color[3])
        self._plot_style_custom = False
        self._syncing_axis_combos = False
        self._3d_view = None
        self._gl_module = None
        self._gl_grid = None
        self._gl_axis = None
        self._gl_box = None
        self._gl_axis_items: list[object] = []
        self._gl_series_items: list[object] = []
        # Experimental GPU renderer for the 2-D projections: a GL canvas
        # behind the plot, driven by the plot's own ViewBox.
        self._gl2d_view = None
        self._gl2d_items: list[object] = []
        self._cpu_render_summary = ""
        self._cpu_aggregate_active = False
        # Data centre and span, so positions upload once as float32 and the
        # view change is a per-frame affine instead of a re-upload.
        # GPU rendering is the default: it is tried on every 2-D draw and
        # falls back to pyqtgraph when the machine cannot do it.
        self._use_gl_2d = not self._cpu_fix
        self._gl2d_origin = (0.0, 0.0)
        self._gl2d_span = (1.0, 1.0)
        self._gl_bounds_item = None
        self._gpu_fallback_note = ""
        self._gl2d_verified = False
        self._gl2d_blank_checks = 0
        # Extent of the *un-thinned* series, so the ViewBox fits the data even
        # when only part of it is on the canvas.
        self._data_extent: tuple[float, float, float, float] | None = None
        self._3d_camera_initialised = False
        # Data extent the unit cube was built from. 3-D coordinates are
        # normalized per axis, so re-deriving it from changed data would move
        # every point; the toggles freeze it instead.
        self._3d_extent: tuple[np.ndarray, np.ndarray] | None = None
        self._freeze_3d_extent = False

        self.setWindowTitle(self._window_label)
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(780, 400)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._roi_overlay = None

        self._build_ui()
        self._refresh()

        state.filter_changed.connect(self._on_filter_changed)
        state.attributes_changed.connect(self._on_attributes_changed)

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
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Control row ───────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(4)

        self._axis_a_label = QLabel("X:")
        bar.addWidget(self._axis_a_label)
        self._x_combo = QComboBox()
        self._x_combo.setMinimumWidth(56)
        self._x_combo.currentTextChanged.connect(
            lambda text: self._on_dimension_control_changed("X", text)
        )
        bar.addWidget(self._x_combo)

        self._axis_b_label = QLabel("Y:")
        bar.addWidget(self._axis_b_label)
        self._y_combo = QComboBox()
        self._y_combo.setMinimumWidth(56)
        self._y_combo.currentTextChanged.connect(
            lambda text: self._on_dimension_control_changed("Y", text)
        )
        bar.addWidget(self._y_combo)

        self._z_label = QLabel("Z:")
        bar.addWidget(self._z_label)
        self._z_combo = QComboBox()
        self._z_combo.setMinimumWidth(56)
        self._z_combo.currentTextChanged.connect(
            lambda text: self._on_dimension_control_changed("Z", text)
        )
        bar.addWidget(self._z_combo)

        self._c_label = QLabel("C:")
        bar.addWidget(self._c_label)
        self._c_combo = QComboBox()
        self._c_combo.setMinimumWidth(56)
        self._c_combo.currentTextChanged.connect(
            lambda text: self._on_dimension_control_changed("C", text)
        )
        bar.addWidget(self._c_combo)

        self._iter_label = QLabel("Iter:")
        bar.addWidget(self._iter_label)
        self._iter_combo = QComboBox()
        self._iter_combo.setMinimumWidth(76)
        self._iter_combo.currentTextChanged.connect(self._draw)
        bar.addWidget(self._iter_combo)

        self._valid_chk = QCheckBox("Valid only")
        self._valid_chk.setChecked(True)
        self._valid_chk.setToolTip("Show only vld=True localizations. Uncheck to include invalid ones.")
        self._valid_chk.stateChanged.connect(self._draw_keeping_view)
        bar.addWidget(self._valid_chk)

        self._lines_chk = QCheckBox("Lines")
        self._lines_chk.setChecked(False)
        self._lines_chk.stateChanged.connect(self._draw_keeping_view)
        bar.addWidget(self._lines_chk)

        self._filter_chk = QCheckBox("Filtered only")
        self._filter_chk.setChecked(True)
        self._filter_chk.stateChanged.connect(self._draw_keeping_view)
        bar.addWidget(self._filter_chk)

        bar.addStretch()

        self._zoom_btn = QPushButton("zoom")
        self._zoom_btn.setCheckable(True)
        self._zoom_btn.setToolTip("Left-click to enable plot zoom. Right-click to choose the zoom mode.")
        self._zoom_btn.toggled.connect(self._set_zoom_active)
        self._zoom_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._zoom_btn.customContextMenuRequested.connect(self._show_zoom_menu)
        bar.addWidget(self._zoom_btn)
        root.addLayout(bar)

        # ── Plot ─────────────────────────────────────────────────
        pg.setConfigOptions(antialias=True)
        self._plot = plot_widget(background="w")
        self._plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._plot.customContextMenuRequested.connect(self._show_context_menu)
        self._plot.getPlotItem().setMenuEnabled(False)
        self._view_box = self._plot.getPlotItem().vb
        try:
            self._view_box.setMenuEnabled(False)
        except Exception:
            pass
        self._original_mouse_drag_event = self._view_box.mouseDragEvent
        self._view_box.mouseDragEvent = self._zoom_mouse_drag_event
        # Thinning is resolved against the visible range, so a settled pan or
        # zoom has to re-run the draw. Parented to self, so it dies with the
        # window rather than firing into a torn-down widget.
        self._rethin_timer = QTimer(self)
        self._rethin_timer.setSingleShot(True)
        self._rethin_timer.timeout.connect(self._rethin_now)
        try:
            self._view_box.sigRangeChanged.connect(self._on_view_range_changed)
        except Exception:
            pass
        self._apply_plot_colors()

        # Series items are (curve, scatter) pairs, created on demand so the
        # "all iterations" overlay can show one colored series per iteration.
        self._series_items: list[tuple] = []
        self._legend = None
        from .roi_overlay import RoiOverlayController
        self._roi_overlay = RoiOverlayController(
            self._state.rois,
            self,
            self._plot,
            self._plot.getPlotItem(),
            coordinate_space="plot",
        )

        self._stack = QStackedWidget()
        # The plot sits in a one-cell grid so the GPU canvas can go in the
        # same cell *behind* it, leaving every 2-D feature working while the
        # points are drawn by OpenGL.
        self._plot_page = QWidget()
        page_layout = QGridLayout(self._plot_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(self._plot, 0, 0)
        self._stack.addWidget(self._plot_page)
        self._stack.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._stack.customContextMenuRequested.connect(self._show_context_menu)
        root.addWidget(self._stack)

        from .floating_colorbar import FloatingColorBar

        self._colorbar = FloatingColorBar(
            self._stack,
            on_visibility_changed=self._set_colorbar_visible,
            on_customize=self.open_lut_dialog,
            on_state_changed=self._on_colorbar_state_changed,
            attribute_names=lambda: list(self._numeric_attrs),
            current_attribute=lambda: self._dimension_attrs["C"],
            on_attribute_changed=lambda name: self._set_dimension_attribute(
                "C", name
            ),
            plot_area=self._colorbar_plot_area,
            background_color=lambda: QColor(*self._current_background_color()),
        )
        self._colorbar.set_bar_visible(False)

        from .floating_legend import FloatingLegend

        # pyqtgraph's own legend samples the *item*, so on the GPU path (empty,
        # hidden scatter items) every row drew an "invisible eye" instead of a
        # colour, and it could never appear over the 3-D view at all.
        self._legend = FloatingLegend(
            self._stack,
            on_visibility_changed=self._set_legend_visible,
            on_state_changed=self._on_legend_state_changed,
            plot_area=self._colorbar_plot_area,
            background_color=lambda: QColor(*self._current_background_color()),
        )

        # A docked bar aligns its gradient with the ViewBox, so repaint it
        # whenever that geometry changes (resize, axis shown/hidden).
        try:
            self._plot.getPlotItem().getViewBox().sigResized.connect(
                lambda *_: self._colorbar.update()
            )
        except Exception:
            pass

        # ── Info ─────────────────────────────────────────────────
        self._info = QLabel("")
        self._info.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    @property
    def _dimension_count(self) -> int:
        """Derived, for state files and the stacked-iteration gate."""
        return 2 + int(self._has_z) + int(self._has_c)

    def _visible_dimensions(self) -> tuple[str, str]:
        return _VIEW_DIMENSIONS.get(self._view_mode, ("X", "Y"))

    def _spatial_dimensions(self) -> tuple[str, ...]:
        if self._view_mode == "3D" and self._has_z:
            return ("X", "Y", "Z")
        return self._visible_dimensions()

    def _active_dimensions(self) -> tuple[str, ...]:
        dimensions = ["X", "Y"]
        if self._has_z:
            dimensions.append("Z")
        if self._has_c:
            dimensions.append("C")
        return tuple(dimensions)

    def _pick_attribute(
        self,
        preferred: tuple[str, ...],
        *,
        used: tuple[str, ...] = (),
    ) -> str:
        for name in (*preferred, *self._numeric_attrs):
            if name in self._numeric_attrs and name not in used:
                return name
        return self._numeric_attrs[0] if self._numeric_attrs else ""

    def _default_dimension_attribute(self, dimension: str) -> str:
        used = tuple(
            self._dimension_attrs[key]
            for key in ("X", "Y", "Z", "C")
            if key != dimension and self._dimension_attrs[key]
        )
        preferred = {
            "X": ("idx", "tim"),
            "Y": ("efo", "cfr", "tim"),
            "Z": ("efo", "cfr", "tim", "tid", "idx"),
            "C": ("cfr", "efo", "tim", "tid", "idx"),
        }[dimension]
        return self._pick_attribute(preferred, used=used)

    def _sync_visible_attribute_controls(self) -> None:
        controls = (
            ("X", self._axis_a_label, self._x_combo),
            ("Y", self._axis_b_label, self._y_combo),
            ("Z", self._z_label, self._z_combo),
            ("C", self._c_label, self._c_combo),
        )
        active_dimensions = set(self._active_dimensions())
        self._syncing_axis_combos = True
        try:
            for dimension, label, combo in controls:
                visible = dimension in active_dimensions
                label.setVisible(visible)
                combo.setVisible(visible)
                if not visible:
                    continue
                current_items = [combo.itemText(i) for i in range(combo.count())]
                combo.blockSignals(True)
                if current_items != self._numeric_attrs:
                    combo.clear()
                    combo.addItems(self._numeric_attrs)
                    self._apply_attribute_combo_tooltips(combo)
                value = self._dimension_attrs.get(dimension, "")
                if value in self._numeric_attrs:
                    combo.setCurrentText(value)
                combo.setToolTip(attribute_description(combo.currentText()))
                combo.blockSignals(False)
        finally:
            self._syncing_axis_combos = False

    def _on_dimension_control_changed(self, dimension: str, text: str) -> None:
        if self._syncing_axis_combos or not text:
            return
        if dimension not in self._active_dimensions():
            return
        self._set_dimension_attribute(dimension, text)

    def _dimension_present(self, dimension: str) -> bool:
        return self._has_z if dimension == "Z" else self._has_c

    def _set_thinning(self, enabled: bool) -> None:
        """View ▸ Thinning; the default for new windows is in Preferences."""
        enabled = bool(enabled)
        if enabled == self._thinning:
            return
        self._thinning = enabled
        self._persist_current_view_state()
        self._draw()

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)

        view_menu = menu.addMenu("View")
        view_options = _VIEW_OPTIONS if self._has_z else ("XY",)
        if self._cpu_fix:
            view_options = tuple(view for view in view_options if view != "3D")
        for view in view_options:
            action = view_menu.addAction(view)
            action.setCheckable(True)
            action.setChecked(view == self._view_mode)
            action.triggered.connect(
                lambda _checked=False, value=view: self._set_view_mode(value)
            )
        view_menu.addSeparator()

        background_action = view_menu.addAction("Black background")
        background_action.setCheckable(True)
        background_action.setChecked(self._black_background)
        background_action.triggered.connect(self._set_black_background)

        axis_action = view_menu.addAction("Axis")
        axis_action.setCheckable(True)
        axis_action.setChecked(self._current_axis_visible())
        axis_action.triggered.connect(self._set_current_axis_visible)

        grid_action = view_menu.addAction("Grid lines")
        grid_action.setCheckable(True)
        grid_action.setChecked(self._current_grid_visible())
        grid_action.triggered.connect(self._set_current_grid_visible)

        view_menu.addAction("Plot style", self._show_plot_style_dialog)

        view_menu.setToolTipsVisible(True)
        if self._cpu_fix:
            aggregation_action = view_menu.addAction("Automatic screen aggregation")
            aggregation_action.setCheckable(True)
            aggregation_action.setChecked(True)
            aggregation_action.setEnabled(False)
            aggregation_action.setToolTip(
                "When drawable markers outnumber display pixels, every visible "
                "row is reduced into a count grid (and mean C per cell). Zooming "
                "recomputes the grid; sparse views use exact bulk-painted markers."
            )
        else:
            thinning_action = view_menu.addAction("Thinning")
            thinning_action.setCheckable(True)
            thinning_action.setChecked(self._thinning)
            thinning_action.setToolTip(THINNING_TOOLTIP)
            thinning_action.triggered.connect(self._set_thinning)

        # The colorbar's own menu can hide it; without an entry here there
        # would be no way back once it is gone.
        legend_action = view_menu.addAction("Legend")
        legend_action.setCheckable(True)
        legend_action.setChecked(self._show_legend)
        legend_action.setToolTip(
            "Iteration colours of the 'all [stacked]' view; drag it, or "
            "right-click it to undock."
        )
        legend_action.triggered.connect(self._set_legend_visible)

        colorbar_action = view_menu.addAction("Colorbar")
        colorbar_action.setCheckable(True)
        colorbar_action.setChecked(self._show_colorbar)
        colorbar_action.setEnabled(self._has_c)
        colorbar_action.triggered.connect(self._set_colorbar_visible)
        menu.addSeparator()

        # Z and C are added, removed and exchanged from here; their attributes
        # are picked in the top row.
        for dimension in ("Z", "C"):
            if self._dimension_present(dimension):
                continue
            submenu = menu.addMenu(f"add new attribute as {dimension}")
            for name in self._numeric_attrs:
                action = submenu.addAction(name)
                action.triggered.connect(
                    lambda _checked=False, dim=dimension, value=name:
                    self._add_dimension(dim, value)
                )
            apply_attribute_menu_tooltips(submenu, self._numeric_attrs)
        for dimension in ("Z", "C"):
            if not self._dimension_present(dimension):
                continue
            action = menu.addAction(f"remove {dimension} attribute")
            action.triggered.connect(
                lambda _checked=False, dim=dimension: self._remove_dimension(dim)
            )
        if self._has_z or self._has_c:
            menu.addAction("swap Z / C", self._swap_z_and_c)

        if self._view_mode == "3D":
            menu.addSeparator()
            box_action = menu.addAction("Bounding Box")
            box_action.setCheckable(True)
            box_action.setChecked(self._show_3d_bounding_box)
            box_action.triggered.connect(self._set_3d_bounding_box_visible)

        menu.addSeparator()
        menu.addAction("Reset View", self._reset_view)

        source = self.sender()
        if not isinstance(source, QWidget):
            source = self
        menu.exec(source.mapToGlobal(pos))

    def _c_mapping_names(self) -> list[str]:
        """Quantitative defaults followed by application-owned custom maps."""
        names: list[str] = []
        for name in (*_LINEAR_COLORMAPS, *custom_colormap_names(), self._c_mapping):
            if name in names:
                continue
            try:
                make_colormap(name)
            except (KeyError, TypeError, ValueError):
                continue
            names.append(name)
        return names

    def _current_background_color(self) -> tuple[int, int, int, int]:
        if self._black_background:
            return 0, 0, 0, 255
        return tuple(viewer_color(self._state.prefs, "attribute_background"))

    def _reference_color(self) -> tuple[float, float, float, float]:
        background = self._current_background_color()
        luminance = (
            0.2126 * background[0]
            + 0.7152 * background[1]
            + 0.0722 * background[2]
        )
        return (
            (0.72, 0.72, 0.72, 0.58)
            if luminance < 128
            else (0.28, 0.28, 0.28, 0.48)
        )

    def _text_color(self) -> tuple[int, int, int, int]:
        return (
            (235, 235, 235, 240)
            if self._gl_blend_mode_for_background(
                self._current_background_color()
            ) == "additive"
            else (25, 25, 25, 240)
        )

    def _set_black_background(self, enabled: bool) -> None:
        self._black_background = bool(enabled)
        self._apply_plot_colors()
        self._draw()

    def _apply_2d_reference_visibility(self) -> None:
        plot_item = self._plot.getPlotItem()
        for axis_name in ("left", "bottom"):
            plot_item.showAxis(axis_name, show=self._show_2d_axis)
        self._plot.showGrid(
            x=self._show_2d_grid, y=self._show_2d_grid, alpha=0.2
        )

    def _current_axis_visible(self) -> bool:
        return self._show_3d_axis if self._view_mode == "3D" else self._show_2d_axis

    def _set_current_axis_visible(self, checked: bool) -> None:
        if self._view_mode == "3D":
            self._show_3d_axis = bool(checked)
            if self._gl_axis is not None:
                self._gl_axis.setVisible(self._show_3d_axis)
            for item in self._gl_axis_items:
                item.setVisible(self._show_3d_axis)
        else:
            self._show_2d_axis = bool(checked)
            self._apply_2d_reference_visibility()
        self._draw()

    def _current_grid_visible(self) -> bool:
        return self._show_3d_grid if self._view_mode == "3D" else self._show_2d_grid

    def _set_current_grid_visible(self, checked: bool) -> None:
        if self._view_mode == "3D":
            self._show_3d_grid = bool(checked)
            if self._gl_grid is not None:
                self._gl_grid.setVisible(self._show_3d_grid)
        else:
            self._show_2d_grid = bool(checked)
            self._apply_2d_reference_visibility()
        self._draw()

    def _set_3d_bounding_box_visible(self, checked: bool) -> None:
        self._show_3d_bounding_box = bool(checked)
        if self._gl_box is not None:
            self._gl_box.setVisible(self._show_3d_bounding_box)
        self._persist_current_view_state()

    def _current_style_color(self) -> tuple[int, int, int]:
        return tuple(int(channel) for channel in self._point_color[:3])

    def _show_plot_style_dialog(self) -> None:
        from .plot_style_dialog import PlotStyleDialog

        dialog = PlotStyleDialog(
            {
                "label": self.windowTitle(),
                "symbol": self._point_symbol,
                "line_style": self._line_style,
                "line_width": self._line_width,
                "size": self._point_size,
                "alpha": self._point_alpha,
                "color": self._current_style_color(),
            },
            self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_plot_style(
                dialog.result_payload(), color_changed=dialog.color_changed
            )

    def _apply_plot_style(
        self, payload: dict, *, color_changed: bool = False
    ) -> None:
        self._point_symbol = str(payload.get("symbol", self._point_symbol))
        self._line_style = str(payload.get("line_style", self._line_style))
        self._line_width = max(
            0.5, min(10.0, float(payload.get("line_width", self._line_width)))
        )
        self._point_size = max(
            1, min(50, int(payload.get("size", self._point_size)))
        )
        self._point_alpha = max(
            0, min(255, int(payload.get("alpha", self._point_alpha)))
        )
        if color_changed:
            raw_color = payload.get("color", self._point_color[:3])
            rgb = tuple(
                max(0, min(255, int(channel)))
                for channel in raw_color[:3]
            )
            if len(rgb) == 3:
                self._point_color = (*rgb, self._point_alpha)
        else:
            self._point_color = (*self._point_color[:3], self._point_alpha)
        self._plot_style_custom = True
        self._draw()

    def _reset_view(self) -> None:
        if self._view_mode == "3D" and self._3d_view is not None:
            self._3d_view.opts["center"] = pg.Vector(0.0, 0.0, 0.0)
            self._3d_view.setCameraPosition(distance=2.5)
            self._3d_camera_initialised = True
        else:
            self._plot.autoRange()

    def _persist_current_view_state(self) -> None:
        dataset = self._dataset()
        if dataset is None:
            return
        self._save_view_state(
            dataset,
            use_lines=self._lines_chk.isChecked(),
            filtered_only=self._filter_chk.isChecked(),
            vld_only=self._valid_chk.isChecked(),
        )

    def _colorbar_plot_area(self) -> QRect | None:
        """The 2-D ViewBox rectangle in ``_stack`` coordinates.

        A docked colorbar aligns its gradient to this so it spans exactly the
        plot's data area; the 3-D view has no ViewBox and returns None.
        """
        plot = getattr(self, "_plot", None)
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

    @property
    def view_mode(self) -> str:
        """The active projection: XY / XZ / YZ / 3D."""
        return self._view_mode

    @property
    def gpu_2d(self) -> bool:
        """True while the 2-D projection is drawn by the GPU renderer."""
        return self._use_gl_2d

    def _gpu_allowed(self) -> bool:
        """Whether startup probing permits an OpenGL 2-D renderer."""

        if self._cpu_fix:
            return False
        capabilities = getattr(self._state, "gpu_capabilities", None)
        return capabilities is None or bool(getattr(capabilities, "available", False))

    def _gpu_point_limit(self) -> int:
        """Current memory-derived marker upload limit."""

        capabilities = getattr(self._state, "gpu_capabilities", None)
        if capabilities is not None:
            return max(0, int(getattr(capabilities, "point_limit", 0)))
        # AppState can be constructed directly by tests/embedders without the
        # startup probe. Retain a dynamic RAM guard in that case.
        try:
            import psutil

            available = int(psutil.virtual_memory().available)
        except Exception:
            return 0
        return point_limit_from_memory(
            available_system_memory_bytes=available,
            free_gpu_memory_bytes=None,
        )

    def set_gpu_2d(self, enabled: bool) -> None:
        """Public entry point for View ▸ GPU rendering in the main menu."""
        self._set_use_gl_2d(enabled and self._gpu_allowed())

    def _set_use_gl_2d(self, enabled: bool) -> None:
        """Switch the 2-D projections between pyqtgraph and the GPU renderer."""
        enabled = bool(enabled) and self._gpu_allowed()
        if enabled == self._use_gl_2d:
            return
        self._use_gl_2d = enabled
        if enabled:
            self._gpu_fallback_note = ""
        self._persist_current_view_state()
        self._draw()

    def _set_legend_visible(self, visible: bool) -> None:
        self._show_legend = bool(visible)
        self._legend.set_legend_visible(self._show_legend)
        self._persist_current_view_state()

    def _on_legend_state_changed(self) -> None:
        self._legend_geometry = self._legend.serialized_geometry()
        self._persist_current_view_state()

    def _update_legend(self, records: list[dict]) -> None:
        """Show the iteration notation, in 2-D, on the GPU canvas and in 3-D.

        Only the stacked view colours by iteration; every other selection draws
        one series, which needs no legend.
        """
        entries = [
            (str(record.get("name") or ""), record.get("color") or (0, 0, 0, 255))
            for record in records
            if record.get("name")
        ]
        self._legend.set_entries(entries, title="Iteration")
        self._legend.set_legend_visible(bool(entries) and self._show_legend)

    def _set_colorbar_visible(self, visible: bool) -> None:
        self._show_colorbar = bool(visible)
        self._update_colorbar()
        self._persist_current_view_state()

    def _on_colorbar_state_changed(self) -> None:
        if self._colorbar is None:
            return
        self._colorbar_show_values = self._colorbar.show_values
        self._colorbar_orientation = self._colorbar.orientation
        self._colorbar_geometry = self._colorbar.serialized_geometry()
        self._persist_current_view_state()

    def _update_colorbar(
        self,
        color_lo: float | None = None,
        color_hi: float | None = None,
    ) -> None:
        if self._colorbar is None:
            return
        finite = np.asarray(self._last_color_values, dtype=float)
        finite = finite[np.isfinite(finite)]
        visible = self._has_c and self._show_colorbar and finite.size > 0
        if not visible:
            self._colorbar.set_bar_visible(False)
            return
        if (
            color_lo is None
            or color_hi is None
            or not np.isfinite(color_lo)
            or not np.isfinite(color_hi)
            or color_hi <= color_lo
        ):
            color_lo, color_hi = self._color_levels(
                self._last_color_values, self._manual_color_levels
            )
        lut = colormap_lut(
            self._c_mapping,
            n=256,
            invert=self._lut_invert,
            gamma=self._lut_gamma,
            alpha=True,
        )
        self._colorbar.set_color_data(
            lut,
            float(color_lo),
            float(color_hi),
            f"C: {self._dimension_attrs['C']}",
        )
        self._colorbar.set_bar_visible(True)

    def _set_dimension_attribute(self, dimension: str, name: str) -> None:
        if dimension not in self._dimension_attrs or name not in self._numeric_attrs:
            return
        self._dimension_attrs[dimension] = name
        # A plotted axis showing a different attribute is a different value
        # range — `efo` in the tens of thousands where `cfr` is 0..1 — so the
        # old range is meaningless and the view has to be re-fitted. C only
        # recolours the same points, and its levels re-derive on their own, so
        # the view is left where the user put it.
        refit_axis = self._axis_of_dimension(dimension)
        if dimension == "C":
            self._manual_color_levels = None
        self._sync_visible_attribute_controls()
        self._style_iteration_boldness()
        self._draw()
        if refit_axis:
            self._refit_view(refit_axis)
        if dimension == "C":
            self.sync_lut_dialog()

    def _axis_of_dimension(self, dimension: str) -> str:
        """"x" / "y" when this dimension is a plotted 2-D axis, else "".

        C only recolours the same points, and Z is not on screen in an XY view,
        so neither is a reason to move anything.
        """
        if self._view_mode == "3D":
            return ""
        x_dimension, y_dimension = self._visible_dimensions()
        if dimension == x_dimension:
            return "x"
        if dimension == y_dimension:
            return "y"
        return ""

    def _refit_view(self, axis: str) -> None:
        """Fit one 2-D axis to the data it is now showing.

        Only the changed axis: switching Y from `efo` to `cfr` moves from tens
        of thousands to 0..1, but the X window is still the region the user
        chose to look at. pyqtgraph turns auto-range off at the first zoom or
        pan (and `_draw_keeping_view` pins it deliberately), so it has to be
        turned back on rather than merely asked to fit once. 3-D needs nothing:
        its cube is normalized from the data on every ordinary draw.
        """
        if self._view_mode == "3D" or axis not in ("x", "y"):
            return
        try:
            self._view_box.enableAutoRange(axis=axis)
            self._view_box.updateAutoRange()
        except Exception:
            pass

    def _set_c_mapping(self, name: str) -> None:
        try:
            make_colormap(name)
        except (KeyError, TypeError, ValueError):
            return
        self._c_mapping = name
        self._draw()
        self.sync_lut_dialog()

    def _set_view_mode(self, view: str) -> None:
        if view not in _VIEW_OPTIONS or (
            not self._has_z and view != "XY"
        ) or (self._cpu_fix and view == "3D"):
            return
        if view == "3D" and not self._ensure_3d_built():
            return
        self._view_mode = view
        self._sync_visible_attribute_controls()
        self._zoom_btn.setEnabled(view != "3D")
        if view == "3D":
            if self._gl_axis is not None:
                self._gl_axis.setVisible(self._show_3d_axis)
            for item in self._gl_axis_items:
                item.setVisible(self._show_3d_axis)
            if self._gl_grid is not None:
                self._gl_grid.setVisible(self._show_3d_grid)
            self._stack.setCurrentWidget(self._3d_view)
        else:
            self._stack.setCurrentWidget(self._plot_page)
            self._apply_2d_reference_visibility()
        self._style_iteration_boldness()
        self._draw()

    def _add_dimension(self, dimension: str, attribute: str = "") -> None:
        """Give the plot a Z or a C dimension, showing the chosen attribute.

        Z and C are independent, so a plot can be XY, XYZ, XYC or XYZC; the
        menu offers whichever of the two is missing.
        """
        if dimension not in ("Z", "C") or not self._numeric_attrs:
            return
        if dimension == "Z" and self._has_z:
            return
        if dimension == "C" and self._has_c:
            return
        if attribute not in self._numeric_attrs:
            attribute = (
                self._dimension_attrs[dimension]
                if self._dimension_attrs[dimension] in self._numeric_attrs
                else self._default_dimension_attribute(dimension)
            )
        self._dimension_attrs[dimension] = attribute
        if dimension == "Z":
            self._has_z = True
        else:
            self._has_c = True
            self._manual_color_levels = None
        self._set_stacked_enabled(not self._has_c)
        self._sync_visible_attribute_controls()
        self._style_iteration_boldness()
        self._draw()

    def _remove_dimension(self, dimension: str) -> None:
        """Drop Z or C, leaving the other one and the XY pair untouched."""
        if dimension == "Z" and self._has_z:
            self._has_z = False
            if self._view_mode != "XY":
                self._view_mode = "XY"
                self._stack.setCurrentWidget(self._plot_page)
                self._zoom_btn.setEnabled(True)
                self._apply_2d_reference_visibility()
        elif dimension == "C" and self._has_c:
            self._has_c = False
            self._last_color_values = np.empty(0, dtype=float)
            try:
                if self._lut_dialog is not None:
                    self._lut_dialog.hide()
            except RuntimeError:
                self._lut_dialog = None
        else:
            return
        self._set_stacked_enabled(not self._has_c)
        self._sync_visible_attribute_controls()
        self._style_iteration_boldness()
        self._draw()

    def _swap_z_and_c(self) -> None:
        """Exchange the Z and C attributes — or move a lone one to the other.

        With only Z (or only C) present there is nothing to exchange it with,
        so the dimension itself moves: XYZ becomes XYC and back. A view that
        was showing Z has to return to XY when Z goes away.
        """
        if not (self._has_z or self._has_c):
            return
        z_attribute = self._dimension_attrs["Z"] if self._has_z else ""
        c_attribute = self._dimension_attrs["C"] if self._has_c else ""
        self._has_z, self._has_c = bool(c_attribute), bool(z_attribute)
        if c_attribute:
            self._dimension_attrs["Z"] = c_attribute
        if z_attribute:
            self._dimension_attrs["C"] = z_attribute
        if not self._has_z and self._view_mode != "XY":
            self._view_mode = "XY"
            self._stack.setCurrentWidget(self._plot_page)
            self._zoom_btn.setEnabled(True)
            self._apply_2d_reference_visibility()
        if self._has_c:
            self._manual_color_levels = None
        else:
            self._last_color_values = np.empty(0, dtype=float)
        self._set_stacked_enabled(not self._has_c)
        self._sync_visible_attribute_controls()
        self._style_iteration_boldness()
        self._draw()

    def _set_stacked_enabled(self, enabled: bool) -> None:
        index = self._iter_combo.findText(STACKED_LABEL)
        if index < 0:
            return
        item = self._iter_combo.model().item(index)
        if item is not None:
            item.setEnabled(bool(enabled))
            item.setToolTip(
                "" if enabled else "Unavailable while C uses color for the fourth dimension."
            )
        if not enabled and self._iter_combo.currentText() == STACKED_LABEL:
            replacement = (
                FLATTEN_LABEL
                if self._iter_combo.findText(FLATTEN_LABEL) >= 0
                else self._default_iter_label(self._dataset())
            )
            self._iter_combo.blockSignals(True)
            self._iter_combo.setCurrentText(replacement)
            self._iter_combo.blockSignals(False)

    @staticmethod
    def _gl_blend_mode_for_background(color) -> str:
        """Choose a visible OpenGL blend mode for the configured background."""
        red, green, blue = (float(channel) for channel in color[:3])
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return "additive" if luminance < 128.0 else "translucent"

    def _apply_3d_blend(self, item) -> None:
        """Keep OpenGL items visible on both light and dark backgrounds.

        Pyqtgraph's additive mode works well on black, but adding any color to
        white produces white and makes points appear to be missing entirely.
        """
        background = self._current_background_color()
        item.setGLOptions(self._gl_blend_mode_for_background(background))

    @staticmethod
    def _format_3d_tick(value: float) -> str:
        if abs(value) >= 1000 or float(value).is_integer():
            return f"{value:.0f}"
        return f"{value:.3g}"

    def _clear_3d_axis_items(self) -> None:
        if self._3d_view is not None:
            for item in self._gl_axis_items:
                try:
                    self._3d_view.removeItem(item)
                except Exception:
                    pass
        self._gl_axis_items = []

    def _add_3d_axis_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        color: tuple[float, float, float, float],
        *,
        width: float = 1.0,
    ) -> None:
        if self._3d_view is None:
            return
        item = self._gl_module.GLLinePlotItem(
            pos=np.vstack([start, end]).astype(np.float32),
            color=color,
            width=width,
            antialias=True,
        )
        self._apply_3d_blend(item)
        item.setVisible(self._show_3d_axis)
        self._3d_view.addItem(item)
        self._gl_axis_items.append(item)

    def _add_3d_axis_text(
        self,
        position: np.ndarray,
        text: str,
        *,
        label: bool = False,
    ) -> None:
        if self._3d_view is None:
            return
        item = self._gl_module.GLTextItem(
            pos=np.asarray(position, dtype=np.float64),
            text=str(text),
            color=self._text_color(),
            font=QFont("Helvetica", 11 if label else 8),
            glOptions="translucent",
        )
        item.setVisible(self._show_3d_axis)
        self._3d_view.addItem(item)
        self._gl_axis_items.append(item)

    def _update_3d_axis_items(
        self,
        raw_positions: np.ndarray,
        extent: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Axis lines, names and numeric ticks around the unit cube.

        ``extent`` is the data range the cube was built from; the ticks must
        come from the same one or they would disagree with where the points
        actually sit.
        """
        self._clear_3d_axis_items()
        if not self._show_3d_axis or self._3d_view is None:
            return
        if extent is not None and np.all(np.isfinite(extent[0])):
            raw_mins, raw_maxs = extent
        else:
            raw = np.asarray(raw_positions, dtype=float)
            raw = raw[np.all(np.isfinite(raw), axis=1)]
            if raw.size == 0:
                return
            raw_mins = np.min(raw, axis=0)
            raw_maxs = np.max(raw, axis=0)
        origin = np.full(3, -0.5, dtype=float)
        axis_colors = (
            (0.90, 0.15, 0.15, 0.95),
            (0.15, 0.70, 0.25, 0.95),
            (0.15, 0.35, 0.95, 0.95),
        )
        for dimension, color in enumerate(axis_colors):
            end = origin.copy()
            end[dimension] = 0.5
            label_position = end.copy()
            label_position[dimension] += 0.07
            dimension_name = "XYZ"[dimension]
            self._add_3d_axis_text(
                label_position,
                f"{dimension_name}: {self._dimension_attrs[dimension_name]}",
                label=True,
            )
            lo, hi = float(raw_mins[dimension]), float(raw_maxs[dimension])
            ticks = tick_values(lo, hi, max_ticks=5)
            if not ticks and np.isfinite(lo):
                ticks = [lo]
            side_dimension = 1 if dimension != 1 else 0
            for tick in ticks:
                normalized = (
                    0.0 if hi <= lo else (tick - lo) / (hi - lo) - 0.5
                )
                tick_position = origin.copy()
                tick_position[dimension] = normalized
                tick_end = tick_position.copy()
                tick_end[side_dimension] += 0.025
                self._add_3d_axis_line(
                    tick_position, tick_end, color, width=1.0
                )
                text_position = tick_end.copy()
                text_position[side_dimension] += 0.012
                self._add_3d_axis_text(
                    text_position, self._format_3d_tick(tick)
                )

    def _ensure_3d_built(self) -> bool:
        if self._3d_view is not None:
            return True
        try:
            import pyqtgraph.opengl as gl
        except ImportError as exc:
            self._info.setText(f"3D view unavailable: PyOpenGL is not installed ({exc}).")
            return False
        try:
            view = gl.GLViewWidget()
            view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            view.customContextMenuRequested.connect(self._show_context_menu)
            view.setBackgroundColor(QColor(*self._current_background_color()))
            grid = gl.GLLinePlotItem(
                pos=three_plane_grid_positions(
                    np.full(3, -0.5), np.full(3, 0.5), target=5
                ),
                color=self._reference_color(),
                width=1.0,
                mode="lines",
                antialias=True,
            )
            grid.setVisible(self._show_3d_grid)
            self._apply_3d_blend(grid)
            view.addItem(grid)
            axis = gl.GLAxisItem()
            axis.setSize(1.0, 1.0, 1.0)
            axis.translate(-0.5, -0.5, -0.5)
            axis.setVisible(self._show_3d_axis)
            self._apply_3d_blend(axis)
            view.addItem(axis)
            corners = np.array(
                [
                    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
                    [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
                    [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
                ],
                dtype=np.float32,
            )
            edges = (
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            )
            box = gl.GLLinePlotItem(
                pos=np.vstack([corners[[start, end]] for start, end in edges]),
                color=self._reference_color(),
                width=1.4,
                mode="lines",
                antialias=True,
            )
            box.setVisible(self._show_3d_bounding_box)
            self._apply_3d_blend(box)
            view.addItem(box)
        except Exception as exc:
            self._info.setText(f"3D view unavailable: {exc}")
            return False
        self._gl_module = gl
        self._3d_view = view
        self._gl_grid = grid
        self._gl_axis = axis
        self._gl_box = box
        self._stack.addWidget(view)
        self._apply_plot_colors()
        return True

    def _refresh(self) -> None:
        ds = self._dataset()
        if ds is None:
            self.setWindowTitle(self._window_label)
            self._clear_series()
            self._clear_gl_series()
            return

        self.setWindowTitle(f"{self._window_label}  —  {ds.name}")

        numeric = plot_attribute_names(ds, self._state.prefs)
        self._numeric_attrs = numeric
        saved = ds.state.get(self._view_state_key, {})

        # Z and C are stored independently; a state file written before they
        # could exist apart carries only the count (3 meant Z, 4 meant both).
        try:
            saved_count = int(saved.get("dimension_count", 2))
        except (TypeError, ValueError):
            saved_count = 2
        self._has_z = bool(saved.get("has_z", saved_count >= 3))
        self._has_c = bool(saved.get("has_c", saved_count >= 4))
        for dimension, state_key in (("X", "x"), ("Y", "y"), ("Z", "z"), ("C", "c")):
            candidate = saved.get(state_key, self._dimension_attrs[dimension])
            if candidate in numeric:
                self._dimension_attrs[dimension] = candidate
            else:
                self._dimension_attrs[dimension] = self._default_dimension_attribute(dimension)
        saved_view = str(saved.get("view", self._view_mode))
        allowed_views = (
            tuple(view for view in _VIEW_OPTIONS if view != "3D")
            if self._cpu_fix else _VIEW_OPTIONS
        )
        self._view_mode = (
            saved_view if self._has_z and saved_view in allowed_views else "XY"
        )
        saved_mapping = str(saved.get("c_mapping", self._c_mapping))
        try:
            make_colormap(saved_mapping)
            self._c_mapping = saved_mapping
        except (KeyError, TypeError, ValueError):
            self._c_mapping = _LINEAR_COLORMAPS[0]
        self._lut_invert = bool(saved.get("c_mapping_invert", False))
        try:
            self._lut_gamma = max(0.01, float(saved.get("c_mapping_gamma", 1.0)))
        except (TypeError, ValueError):
            self._lut_gamma = 1.0
        saved_levels = saved.get("c_mapping_levels")
        try:
            lo, hi = (float(value) for value in saved_levels)
            self._manual_color_levels = (lo, hi) if hi > lo else None
        except (TypeError, ValueError):
            self._manual_color_levels = None

        self._black_background = bool(saved.get("black_background", False))
        self._show_2d_axis = bool(saved.get("show_2d_axis", True))
        self._show_3d_axis = bool(saved.get("show_3d_axis", True))
        self._show_2d_grid = bool(saved.get("show_2d_grid", True))
        self._show_3d_grid = bool(saved.get("show_3d_grid", True))
        self._show_3d_bounding_box = bool(
            saved.get("show_3d_bounding_box", True)
        )
        # The normal window attempts GPU rendering only when the startup probe
        # succeeded. The separate CPU-fix window never constructs a GL view.
        self._use_gl_2d = (
            not self._cpu_fix
            and self._gpu_allowed()
            and bool(saved.get("gl_2d", True))
        )
        self._show_legend = bool(saved.get("show_legend", True))
        stored_legend = saved.get("legend_geometry")
        self._legend_geometry = (
            list(stored_legend) if isinstance(stored_legend, (list, tuple)) else None
        )
        if getattr(self, "_legend", None) is not None:
            self._legend.restore_geometry(self._legend_geometry)
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
        self._plot_style_custom = bool(saved.get("plot_style_custom", False))
        default_color = tuple(viewer_color(self._state.prefs, "attribute_data"))
        if self._plot_style_custom:
            stored_color = saved.get("point_color", default_color)
            try:
                rgba = tuple(
                    max(0, min(255, int(channel))) for channel in stored_color
                )
            except (TypeError, ValueError):
                rgba = default_color
            if len(rgba) == 3:
                try:
                    stored_alpha = int(
                        saved.get("point_alpha", default_color[3])
                    )
                except (TypeError, ValueError):
                    stored_alpha = int(default_color[3])
                rgba = (*rgba, max(0, min(255, stored_alpha)))
            self._point_color = rgba if len(rgba) == 4 else default_color
            try:
                stored_alpha = int(
                    saved.get("point_alpha", self._point_color[3])
                )
            except (TypeError, ValueError):
                stored_alpha = int(self._point_color[3])
            self._point_alpha = max(0, min(255, stored_alpha))
            self._point_color = (*self._point_color[:3], self._point_alpha)
        else:
            self._point_color = default_color
            self._point_alpha = int(default_color[3])
        self._point_symbol = str(saved.get("point_symbol", "o"))
        self._line_style = str(saved.get("line_style", "-"))
        try:
            self._line_width = max(0.5, min(10.0, float(saved.get("line_width", 1.0))))
        except (TypeError, ValueError):
            self._line_width = 1.0
        try:
            self._point_size = max(1, min(50, int(saved.get("point_size", 3))))
        except (TypeError, ValueError):
            self._point_size = 3

        # Iteration dropdown: last (Nth) · all [flatten] · all [stacked] · (N-1)th … 1st
        iter_opts = self._iter_labels(ds)
        self._eff_iter_cache = {}                    # dataset (re)loaded → drop cache
        self._iter_combo.blockSignals(True)
        self._iter_combo.clear()
        self._iter_combo.addItems(iter_opts)
        default_label = self._default_iter_label(ds)
        saved_label = str(saved.get("iter", "") or "")
        self._iter_combo.setCurrentText(saved_label if saved_label in iter_opts else default_label)
        self._iter_combo.blockSignals(False)
        self._set_stacked_enabled(not self._has_c)
        has_iters = bool(iter_opts)
        self._iter_combo.setVisible(has_iters)
        self._iter_label.setVisible(has_iters)
        self._style_iteration_boldness()             # bold the useful iterations

        self._lines_chk.blockSignals(True)
        self._filter_chk.blockSignals(True)
        self._valid_chk.blockSignals(True)
        self._lines_chk.setChecked(bool(saved.get("lines", False)))
        self._filter_chk.setChecked(bool(saved.get("filtered_only", True)))
        self._valid_chk.setChecked(bool(saved.get("valid_only", True)))
        self._thinning = bool(saved.get(
            "thinning",
            self._state.prefs.get("plot", {}).get("attribute_thinning", True),
        ))
        self._lines_chk.blockSignals(False)
        self._filter_chk.blockSignals(False)
        self._valid_chk.blockSignals(False)
        self._sync_visible_attribute_controls()
        self._apply_plot_colors()
        self._apply_2d_reference_visibility()
        if self._view_mode == "3D":
            if self._ensure_3d_built():
                self._zoom_btn.setEnabled(False)
                self._stack.setCurrentWidget(self._3d_view)
            else:
                self._view_mode = "XY"
                self._zoom_btn.setEnabled(True)
                self._sync_visible_attribute_controls()
                self._stack.setCurrentWidget(self._plot_page)
        else:
            self._zoom_btn.setEnabled(True)
            self._stack.setCurrentWidget(self._plot_page)

        self._draw()

    def _apply_plot_colors(self) -> None:
        background = self._current_background_color()
        # In GPU mode the plot must stay see-through: the background belongs to
        # the canvas behind it, or repainting it here would hide the points.
        self._plot.setBackground(None if self._use_gl_2d else QColor(*background))
        self._apply_page_background()
        if self._gl2d_view is not None:
            self._gl2d_view.setBackgroundColor(QColor(*background))
            for item in self._gl2d_items:
                self._apply_3d_blend(item)
        if self._3d_view is not None:
            self._3d_view.setBackgroundColor(QColor(*background))
            for item in (
                self._gl_grid,
                self._gl_axis,
                self._gl_box,
                *self._gl_axis_items,
                *self._gl_series_items,
            ):
                if item is not None:
                    self._apply_3d_blend(item)
        luminance = 0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
        foreground = QColor(25, 25, 25) if luminance >= 145 else QColor(235, 235, 235)
        if self._gl_grid is not None:
            self._gl_grid.setData(color=self._reference_color())
        if self._gl_box is not None:
            self._gl_box.setData(color=self._reference_color())
        for name in ("bottom", "left"):
            axis = self._plot.getPlotItem().getAxis(name)
            axis.setPen(foreground)
            axis.setTextPen(foreground)

    def refresh_preferences(self) -> None:
        """Adopt the Appearance default for thinning after Preferences OK."""
        wanted = bool(
            self._state.prefs.get("plot", {}).get("attribute_thinning", True)
        )
        if wanted == self._thinning:
            return
        self._thinning = wanted
        self._draw()

    def refresh_colors(self) -> None:
        if not self._plot_style_custom:
            self._point_color = tuple(
                viewer_color(self._state.prefs, "attribute_data")
            )
            self._point_alpha = int(self._point_color[3])
        self._apply_plot_colors()
        self._draw()

    @staticmethod
    def _apply_attribute_combo_tooltips(combo: QComboBox) -> None:
        # Kept as a thin name: the wording and the wiring are shared.
        apply_attribute_tooltips(combo)

    # ------------------------------------------------------------------
    # Iteration / validity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _num_itr(ds) -> int:
        return max(1, int(ds.metadata.get("raw_num_itr", ds.prop.num_itr or 1)))

    def _iter_labels(self, ds) -> list[str]:
        return iteration_labels(self._num_itr(ds))

    def _default_iter_label(self, ds) -> str:
        labels = self._iter_labels(ds)
        return f"last ({ordinal(self._num_itr(ds))})" if labels else ""

    def _style_iteration_boldness(self) -> None:
        """Bold the iteration-dropdown entries where **both** plotted attributes
        hold real values (intersection), so the useful iterations stand out."""
        from PyQt6.QtGui import QFont

        ds = self._dataset()
        if ds is None or self._iter_combo.count() == 0:
            return
        cache = getattr(self, "_eff_iter_cache", None)
        if cache is None:
            cache = self._eff_iter_cache = {}
        eff = None
        for dimension in self._active_dimensions():
            attr = self._dimension_attrs.get(dimension, "")
            if not attr:
                continue
            if attr not in cache:
                try:
                    cache[attr] = effective_iterations_for_attr(ds, attr)
                except Exception:
                    cache[attr] = None
            e = cache[attr]
            if e is None:                            # undetermined → no constraint
                continue
            eff = e.copy() if eff is None else (eff & e)
        labels = [self._iter_combo.itemText(i) for i in range(self._iter_combo.count())]
        flags = (iteration_bold_flags(labels, eff, self._num_itr(ds))
                 if eff is not None else [False] * len(labels))
        for i, bold in enumerate(flags):
            font = QFont(self._iter_combo.font())
            font.setBold(bool(bold))
            self._iter_combo.setItemData(i, font, Qt.ItemDataRole.FontRole)

    def _selection(self) -> tuple:
        """Return (itr_selector, render_mode) for the current label."""
        return parse_iteration_label(self._iter_combo.currentText())

    #: MATLAB line specifiers -> Qt pen styles.
    _LINE_STYLES = {
        "-": Qt.PenStyle.SolidLine,
        "--": Qt.PenStyle.DashLine,
        ":": Qt.PenStyle.DotLine,
        "-.": Qt.PenStyle.DashDotLine,
    }

    def _line_pen(self, rgba) -> pg.mkPen:
        """Pen for the connecting line, in the chosen MATLAB line style."""
        return pg.mkPen(
            rgba,
            width=float(self._line_width),
            style=self._LINE_STYLES.get(self._line_style, Qt.PenStyle.SolidLine),
        )

    def _clear_series(self) -> None:
        for curve, scatter in self._series_items:
            self._plot.removeItem(curve)
            self._plot.removeItem(scatter)
        self._series_items = []

    def _clear_gl_series(self) -> None:
        if self._3d_view is not None:
            for item in self._gl_series_items:
                try:
                    self._3d_view.removeItem(item)
                except Exception:
                    pass
        self._gl_series_items = []

    def _add_series(
        self,
        x,
        y,
        color,
        *,
        use_lines: bool,
        name: str | None = None,
        brushes=None,
        line_visible_mask: np.ndarray | None = None,
        markers: bool = True,
    ):
        """``markers=False`` leaves the points to the GPU canvas underneath.

        The scatter item is still created — ``_clear_series`` and the stacked
        legend expect one — but it is given **no data**: handing pyqtgraph the
        points and merely hiding them would still pay its ~1 us per point,
        which is the entire cost the GPU path exists to avoid.
        """
        rgba = tuple(int(channel) for channel in color[:3]) + (
            self._point_alpha,
        )
        # self._plot.plot(name=...) registers a legend sample reliably.
        line_x, line_y = x, y
        if use_lines and line_visible_mask is not None:
            visible = np.asarray(line_visible_mask, dtype=bool).ravel()
            line_x = np.asarray(x, dtype=float).copy()
            line_y = np.asarray(y, dtype=float).copy()
            if visible.size != line_x.size or line_y.size != line_x.size:
                raise ValueError("Color visibility mask must match series data")
            # A NaN gap makes pyqtgraph split the connecting line.  A point
            # without a C value is therefore absent from both the marker and
            # its connecting trace rather than merely receiving an endpoint
            # color from the LUT.
            line_x[~visible] = np.nan
            line_y[~visible] = np.nan
        curve = self._plot.plot(
            line_x if use_lines else [], line_y if use_lines else [],
            pen=self._line_pen(rgba) if use_lines else None,
            name=name if use_lines else None,
        )
        scatter = pg.ScatterPlotItem(
            size=self._point_size,
            symbol=self._point_symbol,
            pen=None,
            brush=pg.mkBrush(*rgba),
        )
        scatter.setVisible(markers)
        if markers:
            scatter.setData(
                x,
                y,
                size=self._point_size,
                symbol=self._point_symbol,
                brush=brushes if brushes is not None else pg.mkBrush(*rgba),
            )
        self._plot.addItem(scatter)
        self._series_items.append((curve, scatter))

    # ------------------------------------------------------------------
    # Zoom tool
    # ------------------------------------------------------------------

    def _set_zoom_active(self, active: bool) -> None:
        self._zoom_active = bool(active)
        self._zoom_btn.setText(f"zoom: {self._zoom_mode}" if active else "zoom")
        self._clear_zoom_preview()

    def _show_zoom_menu(self, pos) -> None:
        menu = QMenu(self)
        for mode, label in (
            ("unconstrained", "unconstrained"),
            ("horizontal", "horizontal only"),
            ("vertical", "vertical only"),
        ):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self._zoom_mode == mode)
            action.triggered.connect(lambda _checked=False, value=mode: self._set_zoom_mode(value))
        menu.exec(self._zoom_btn.mapToGlobal(pos))

    def _set_zoom_mode(self, mode: str) -> None:
        if mode not in {"unconstrained", "horizontal", "vertical"}:
            return
        self._zoom_mode = mode
        if self._zoom_active:
            self._zoom_btn.setText(f"zoom: {self._zoom_mode}")
        self._clear_zoom_preview()

    def _zoom_mouse_drag_event(self, event, axis=None) -> None:
        if not self._zoom_active or event.button() != Qt.MouseButton.LeftButton:
            self._original_mouse_drag_event(event, axis=axis)
            return
        event.accept()
        if event.isStart():
            self._zoom_drag_start = self._view_box.mapSceneToView(
                event.buttonDownScenePos(Qt.MouseButton.LeftButton)
            )
            self._clear_zoom_preview()
            self._zoom_preview = pg.PlotDataItem(
                pen=pg.mkPen((30, 120, 220), width=1.5, style=Qt.PenStyle.DashLine)
            )
            self._plot.addItem(self._zoom_preview)
        if self._zoom_drag_start is None:
            return
        current = self._view_box.mapSceneToView(event.scenePos())
        self._update_zoom_preview(self._zoom_drag_start, current)
        if event.isFinish():
            self._apply_zoom_drag(self._zoom_drag_start, current)
            self._zoom_drag_start = None
            self._clear_zoom_preview()

    def _update_zoom_preview(self, start, current) -> None:
        if self._zoom_preview is None:
            return
        x0, x1 = float(start.x()), float(current.x())
        y0, y1 = float(start.y()), float(current.y())
        (vx0, vx1), (vy0, vy1) = self._view_box.viewRange()
        if self._zoom_mode == "horizontal":
            ymid = (vy0 + vy1) / 2.0
            cap = (vy1 - vy0) * 0.08
            self._zoom_preview.setData(
                [x0, x1, np.nan, x0, x0, np.nan, x1, x1],
                [ymid, ymid, np.nan, ymid - cap, ymid + cap, np.nan, ymid - cap, ymid + cap],
            )
        elif self._zoom_mode == "vertical":
            xmid = (vx0 + vx1) / 2.0
            cap = (vx1 - vx0) * 0.08
            self._zoom_preview.setData(
                [xmid, xmid, np.nan, xmid - cap, xmid + cap, np.nan, xmid - cap, xmid + cap],
                [y0, y1, np.nan, y0, y0, np.nan, y1, y1],
            )
        else:
            self._zoom_preview.setData(
                [x0, x1, x1, x0, x0],
                [y0, y0, y1, y1, y0],
            )

    def _apply_zoom_drag(self, start, current) -> None:
        x0, x1 = sorted((float(start.x()), float(current.x())))
        y0, y1 = sorted((float(start.y()), float(current.y())))
        (vx0, vx1), (vy0, vy1) = self._view_box.viewRange()
        min_dx = abs(vx1 - vx0) * 1e-6
        min_dy = abs(vy1 - vy0) * 1e-6
        if self._zoom_mode == "horizontal":
            if (x1 - x0) > min_dx:
                self._view_box.setRange(xRange=(x0, x1), yRange=(vy0, vy1), padding=0.0)
        elif self._zoom_mode == "vertical":
            if (y1 - y0) > min_dy:
                self._view_box.setRange(xRange=(vx0, vx1), yRange=(y0, y1), padding=0.0)
        elif (x1 - x0) > min_dx and (y1 - y0) > min_dy:
            self._view_box.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.0)

    def _clear_zoom_preview(self) -> None:
        if self._zoom_preview is not None:
            try:
                self._plot.removeItem(self._zoom_preview)
            except Exception:
                pass
            self._zoom_preview = None

    def _axis_values(self, ds, name, sel, vld_only, default):
        """Values for one axis. ``idx`` is always the synthetic flattened index;
        other attributes use ds.attr in the default view, raw store otherwise."""
        if name == "idx":
            # In the default view (sel == "last", vld_only == the loaded
            # validity) this is exactly the materialized rows' index. A pooled
            # selection sits on the `last` rows, so it takes their index —
            # summing/averaging row positions would be meaningless.
            v = mfx_get(ds, "idx", itr=_row_selector(sel), vld_only=vld_only)
        elif default:
            # attr_values_1d falls back to the raw store for per-iteration
            # columns kept 2-D in ds.attr (m2205 all-iteration store).
            v = attr_values_1d(ds, name)
        else:
            v = mfx_get(ds, name, itr=sel, vld_only=vld_only)
        return None if v is None else np.asarray(v).ravel().astype(float)

    def _display_budget(self, values: dict[str, np.ndarray]) -> int | None:
        """Markers one draw may paint, or None when nothing is held back.

        Thinning exists because pyqtgraph costs ~1 µs of Python per point; the
        GPU has no such cost, so a GPU-drawn view (the 2-D canvas, or 3-D,
        which is always OpenGL) keeps only a memory guard. A C dimension on the
        **CPU** means per-point brushes, which pyqtgraph paints roughly twenty
        times slower than one shared brush, so that path gets its own budget.
        """
        if self._cpu_fix:
            # The CPU-fix path consumes the full selection in a chunked screen
            # reduction when exact bulk markers would overplot the viewport.
            return None
        if self._use_gl_2d or self._view_mode == "3D":
            limit = self._gpu_point_limit()
            return limit if limit > 0 else None
        if not self._thinning:
            return None
        if "C" in values:
            return _MAX_COLOR_DISPLAY_POINTS
        return _MAX_DISPLAY_POINTS

    def _visible_row_mask(
        self, values: dict[str, np.ndarray]
    ) -> np.ndarray | None:
        """Rows inside the current 2-D view plus a margin.

        Returns None when the view cannot be resolved (3-D, or no range yet),
        in which case thinning falls back to sampling the whole series.
        """
        if self._view_mode == "3D":
            return None
        x_dimension, y_dimension = self._visible_dimensions()
        if x_dimension not in values or y_dimension not in values:
            return None
        try:
            (x0, x1), (y0, y1) = self._view_box.viewRange()
        except Exception:
            return None
        bounds = np.array([x0, x1, y0, y1], dtype=float)
        if not np.all(np.isfinite(bounds)) or x1 <= x0 or y1 <= y0:
            return None
        # While an axis still auto-ranges, the range describes the *previous*
        # data (on the first draw, pyqtgraph's placeholder 0..1) and is about to
        # be refitted to whatever we draw. Restricting to it would drop nearly
        # everything and then fit the view to that remnant, so sample the whole
        # series until the user's own zoom or pan turns auto-range off.
        try:
            if any(self._view_box.autoRangeEnabled()):
                return None
        except Exception:
            return None
        pad_x = (x1 - x0) * _VIEW_THIN_MARGIN
        pad_y = (y1 - y0) * _VIEW_THIN_MARGIN
        x = np.asarray(values[x_dimension], dtype=float)
        y = np.asarray(values[y_dimension], dtype=float)
        # Non-finite rows compare False, so they never consume the budget.
        return (
            (x >= x0 - pad_x) & (x <= x1 + pad_x)
            & (y >= y0 - pad_y) & (y <= y1 + pad_y)
        )

    def _record_data_extent(self, values: dict[str, np.ndarray]) -> None:
        """Union the finite extent of the plotted dimensions into _data_extent."""
        try:
            x_dimension, y_dimension = self._visible_dimensions()
            x = np.asarray(values[x_dimension], dtype=float)
            y = np.asarray(values[y_dimension], dtype=float)
        except (KeyError, ValueError):
            return
        extent = joint_extent(x, y)
        if extent is None:
            return
        if self._data_extent is None:
            self._data_extent = extent
        else:
            previous = self._data_extent
            self._data_extent = (
                min(previous[0], extent[0]), max(previous[1], extent[1]),
                min(previous[2], extent[2]), max(previous[3], extent[3]),
            )

    def _drawable_row_mask(
        self, values: dict[str, np.ndarray]
    ) -> np.ndarray | None:
        """Rows finite in every plotted coordinate — the ones that can appear.

        A row with a NaN coordinate paints nothing (pyqtgraph skips it, and the
        3-D path filters it out before the upload), so letting it consume the
        display budget silently thins the plot. On a real m2410 file this is
        the difference between spending the budget on 246,437 drawable rows and
        spreading it over 2,196,618 rows of which 88.8 % are empty probes.
        """
        mask = None
        for dimension in self._spatial_dimensions():
            value = values.get(dimension)
            if value is None:
                continue
            finite = np.isfinite(np.asarray(value, dtype=float))
            mask = finite if mask is None else (mask & finite)
        return mask

    def _thin_for_view(
        self, values: dict[str, np.ndarray], n: int
    ) -> tuple[dict[str, np.ndarray], int]:
        """Reduce a series to what one draw can paint, honouring the zoom.

        Restricting to the visible rows first is what makes zooming in reveal
        detail instead of merely enlarging a fixed subsample, and is why the
        drawn set no longer depends on how large the *unseen* rest of the
        selection is.
        """
        self._view_restricted = False
        self._thin_step = 1
        self._thin_method = "none"
        self._thin_drawable = None
        budget = self._display_budget(values)
        if budget is None or n <= budget:
            return values, n
        if self._use_gl_2d:
            # About to hold rows back: record what the whole series covers, or
            # Reset View would only ever find the part that was drawn.
            self._record_data_extent(values)
        visible = self._visible_row_mask(values)
        drawable = self._drawable_row_mask(values)
        if drawable is not None:
            count = int(drawable.sum())
            if count < n:
                self._thin_drawable = count
            if count == 0:
                drawable = None
        keep = visible if drawable is None else (
            drawable if visible is None else (visible & drawable)
        )
        rows = None if keep is None else np.flatnonzero(keep)
        if rows is None or rows.size == 0:
            rows = (
                np.flatnonzero(drawable)
                if drawable is not None else np.arange(n, dtype=np.int64)
            )
        self._view_restricted = visible is not None
        if rows.size > budget:
            self._thin_step = int(np.ceil(rows.size / budget))
            x_dimension, y_dimension = self._visible_dimensions()
            rows = spatial_representative_indices(
                values[x_dimension],
                values[y_dimension],
                budget,
                candidate_indices=rows,
            )
            self._thin_method = "spatial cells"
        return {dim: value[rows] for dim, value in values.items()}, n

    def _on_view_range_changed(self, *_args) -> None:
        """Re-thin once the view settles, when the last draw held rows back."""
        if self._rethinking or not (self._view_restricted or self._thinned):
            return
        self._rethin_timer.start(_RETHIN_DELAY_MS)

    def _rethin_now(self) -> None:
        if self._rethinking:
            return
        self._rethinking = True
        try:
            self._draw(reuse_series=True)
        finally:
            self._rethinking = False

    def _series_data(
        self,
        ds,
        dimensions: tuple[str, ...],
        sel,
        vld_only: bool,
        filtered_only: bool,
        *,
        reuse: bool = False,
    ) -> tuple[dict[str, np.ndarray], int, list[str]]:
        """Return aligned, filtered, display-thinned values for dimensions.

        The one shared path serves every 2-D projection, 3-D XYZ, and the C
        values. It retains the original last-valid/raw-iteration semantics.

        ``reuse`` re-thins the rows fetched by the previous draw instead of
        reading the raw store again — worth 0.5 s per pan when browsing 20 M
        raw rows, and only ever set by the pan/zoom re-thin.
        """
        key = (
            id(ds),
            tuple(self._dimension_attrs[dim] for dim in dimensions),
            str(sel),
            bool(vld_only),
            bool(filtered_only),
        )
        if reuse and self._series_cache is not None and self._series_cache[0] == key:
            _key, values, n, missing = self._series_cache
            values, n = self._thin_for_view(values, n)
            return values, n, list(missing)

        default = (
            sel == "last"
            and attr_matches_selection(ds, itr="last", vld_only=vld_only)
        )
        values: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for dimension in dimensions:
            name = self._dimension_attrs[dimension]
            value = self._axis_values(ds, name, sel, vld_only, default)
            if value is None:
                missing.append(name)
            else:
                values[dimension] = value
        if missing:
            return {}, 0, missing

        non_index_sizes = [
            values[dimension].size
            for dimension in dimensions
            if self._dimension_attrs[dimension] != "idx"
        ]
        target_size = min(non_index_sizes) if non_index_sizes else min(
            values[dimension].size for dimension in dimensions
        )
        for dimension in dimensions:
            if (
                self._dimension_attrs[dimension] == "idx"
                and values[dimension].size != target_size
            ):
                values[dimension] = np.arange(1, target_size + 1, dtype=float)
        n = min(values[dimension].size for dimension in dimensions)
        values = {dimension: values[dimension][:n] for dimension in dimensions}

        if filtered_only and n:
            if default:
                fmask = np.asarray(ds.filter_mask, dtype=bool).ravel()
                filter_missing: list[str] = []
            else:
                result = mfx_filter_mask(
                    ds, itr=_row_selector(sel), vld_only=vld_only
                )
                if result is None:
                    fmask = np.empty(0, dtype=bool)
                    filter_missing = []
                else:
                    fmask, filter_missing = result
            missing.extend(filter_missing)
            if fmask.shape[0] == n:
                values = {
                    dimension: value[fmask]
                    for dimension, value in values.items()
                }
                n = next(iter(values.values())).size if values else 0

        self._series_cache = (key, values, n, list(missing))
        values, n = self._thin_for_view(values, n)
        return values, n, missing

    @staticmethod
    def _auto_color_levels(values: np.ndarray) -> tuple[float, float]:
        """Return a meaningful C range, including for a constant attribute."""
        raw_values = np.asarray(values).ravel()
        numeric_values = np.asarray(raw_values, dtype=float)
        finite = numeric_values[np.isfinite(numeric_values)]
        if finite.size == 0:
            return 0.0, 1.0

        data_lo, data_hi = float(np.min(finite)), float(np.max(finite))
        if data_hi > data_lo:
            return data_lo, data_hi

        # Boolean C values have their natural full domain even when the
        # filtered data contain just True or just False.  Integer 0/1 data
        # fall through to the same [0, 1] behaviour below.
        if np.issubdtype(raw_values.dtype, np.bool_):
            return 0.0, 1.0

        value = data_lo
        if value == 0.0:
            return 0.0, 1.0
        if value > 0.0:
            return 0.0, value
        return value, 0.0

    @classmethod
    def _color_levels(
        cls,
        values: np.ndarray,
        levels: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """Use valid manual LUT levels, otherwise the data's safe C range."""
        if levels is not None:
            lo, hi = (float(level) for level in levels)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return lo, hi
        return cls._auto_color_levels(values)

    @classmethod
    def _linear_color_bins(
        cls,
        values: np.ndarray,
        levels: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Linearly map C values over a non-degenerate range to 0..255."""
        raw_values = np.asarray(values).ravel()
        numeric_values = np.asarray(raw_values, dtype=float)
        lo, hi = cls._color_levels(raw_values, levels)
        normalized = np.clip((numeric_values - lo) / (hi - lo), 0.0, 1.0)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        return np.rint(normalized * 255.0).astype(np.uint8), lo, hi

    def _mapped_colors(self, values: np.ndarray) -> tuple[list, np.ndarray, float, float]:
        """Per-point brushes for the 2-D scatter, plus the RGBA the 3-D one uses."""
        rgba, lo, hi, bins, c_is_finite, lut = self._mapped_rgba(values)
        # The colours come from a 256-entry LUT, so at most 257 distinct brushes
        # exist (the last being the transparent marker of a missing C value).
        # Building one QBrush per point instead measured 137x slower at 246k
        # points, which was most of the cost of a C-coloured draw.
        cache = [pg.mkBrush(*(int(channel) for channel in entry)) for entry in lut]
        cache.append(pg.mkBrush(0, 0, 0, 0))
        # bins is uint8: the transparent slot (256) has to be indexed as a
        # wider integer or it wraps to LUT entry 0 and paints opaque.
        keys = np.where(c_is_finite, bins.astype(np.int32), len(cache) - 1)
        return [cache[key] for key in keys], rgba, lo, hi

    def _mapped_rgba(self, values: np.ndarray):
        """The C colours as a float RGBA array — no QBrush objects built.

        The 3-D view uploads colours as an array, so it must not pay for the
        per-point brush list the 2-D scatter needs.
        """
        bins, lo, hi = self._linear_color_bins(
            values, levels=self._manual_color_levels
        )
        lut = colormap_lut(
            self._c_mapping,
            n=256,
            invert=self._lut_invert,
            gamma=self._lut_gamma,
            alpha=True,
        ).copy()
        lut[:, 3] = self._point_alpha
        selected = lut[bins]
        # Missing C values do not represent a colormap endpoint.  Preserve
        # their coordinates for selection/series bookkeeping, but make their
        # plotted marker fully transparent in both the 2-D and OpenGL views.
        c_is_finite = np.isfinite(np.asarray(values, dtype=float).ravel())
        selected[~c_is_finite, 3] = 0
        return (
            selected.astype(np.float32) / 255.0,
            lo,
            hi,
            bins,
            c_is_finite,
            lut,
        )

    @staticmethod
    def _normalize_3d_positions(
        positions: np.ndarray,
        extent: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scale each attribute axis independently into a centred unit cube.

        Returns the extent it used, so a later draw can be pinned to it: the
        cube is relative to the data, and re-deriving it after the data changed
        would slide every point to a new place in the box.
        """
        positions = np.asarray(positions, dtype=float)
        out = np.zeros_like(positions, dtype=float)
        mins = np.full(3, np.nan)
        maxs = np.full(3, np.nan)
        for axis in range(3):
            values = positions[:, axis]
            if extent is not None and np.isfinite(extent[0][axis]):
                lo, hi = float(extent[0][axis]), float(extent[1][axis])
            else:
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    out[:, axis] = np.nan
                    continue
                lo, hi = float(np.min(finite)), float(np.max(finite))
            mins[axis], maxs[axis] = lo, hi
            if hi <= lo:
                out[:, axis] = 0.0
            else:
                out[:, axis] = (values - lo) / (hi - lo) - 0.5
        return out, mins, maxs

    def _draw_2d_series(
        self,
        records: list[dict],
        dimensions: tuple[str, str],
        *,
        use_lines: bool,
        stacked: bool,
    ) -> tuple[float | None, float | None]:
        if self._cpu_fix:
            return self._draw_cpu_2d_series(
                records, dimensions, use_lines=use_lines, stacked=stacked
            )
        self._stack.setCurrentWidget(self._plot_page)
        self._clear_series()
        self._clear_gl_series()
        gpu = self._use_gl_2d and self._ensure_gl_canvas()
        self._set_plot_transparent(gpu)
        if not gpu:
            self._show_gl_canvas(False)
            self._clear_gl_bounds()
        color_lo = color_hi = None
        if gpu:
            color_lo, color_hi = self._draw_gl_2d_series(records, dimensions)
        for record in records:
            values = record["values"]
            brushes = None
            line_visible_mask = None
            if "C" in values:
                line_visible_mask = np.isfinite(np.asarray(values["C"], dtype=float))
                if not gpu:
                    brushes, _rgba, color_lo, color_hi = self._mapped_colors(
                        values["C"]
                    )
            self._add_series(
                values[dimensions[0]],
                values[dimensions[1]],
                record["color"],
                use_lines=use_lines,
                name=record["name"],
                brushes=brushes,
                line_visible_mask=line_visible_mask,
                markers=not gpu,
            )
        self._plot.setLabel("bottom", self._dimension_attrs[dimensions[0]])
        self._plot.setLabel("left", self._dimension_attrs[dimensions[1]])
        return color_lo, color_hi

    def _cpu_view_geometry(
        self, records: list[dict], dimensions: tuple[str, str]
    ) -> tuple[tuple[float, float, float, float] | None, int, int, bool]:
        """Aggregation bounds, viewport pixels and whether the range is user-set."""

        width = max(32, int(self._plot.viewport().width()))
        height = max(32, int(self._plot.viewport().height()))
        manual_view = False
        try:
            manual_view = not any(self._view_box.autoRangeEnabled())
            (x0, x1), (y0, y1) = self._view_box.viewRange()
            bounds = (float(x0), float(x1), float(y0), float(y1))
            if manual_view and np.all(np.isfinite(bounds)) and x1 > x0 and y1 > y0:
                return bounds, width, height, True
        except Exception:
            pass

        extents = []
        for record in records:
            values = record.get("values") or {}
            if dimensions[0] in values and dimensions[1] in values:
                extent = joint_extent(values[dimensions[0]], values[dimensions[1]])
                if extent is not None:
                    extents.append(extent)
        if not extents:
            return None, width, height, manual_view
        bounds = (
            min(extent[0] for extent in extents),
            max(extent[1] for extent in extents),
            min(extent[2] for extent in extents),
            max(extent[3] for extent in extents),
        )
        x0, x1, y0, y1 = bounds
        if x1 <= x0:
            x0, x1 = x0 - 0.5, x1 + 0.5
        if y1 <= y0:
            y0, y1 = y0 - 0.5, y1 + 0.5
        return (x0, x1, y0, y1), width, height, False

    @staticmethod
    def _cpu_density_alpha(counts: np.ndarray, alpha: int) -> np.ndarray:
        """Log-density opacity: singletons stay visible, dense cells saturate."""

        counts = np.asarray(counts)
        out = np.zeros(counts.shape, dtype=np.uint8)
        occupied = counts > 0
        if not occupied.any():
            return out
        peak = int(np.max(counts))
        if peak <= 1:
            out[occupied] = max(1, int(alpha))
            return out
        density = np.log1p(counts[occupied].astype(float)) / np.log1p(peak)
        # Keep an isolated cell legible while still encoding count in opacity.
        density = 0.25 + 0.75 * density
        out[occupied] = np.rint(density * max(1, int(alpha))).astype(np.uint8)
        return out

    def _cpu_image(
        self,
        aggregation,
        *,
        color,
        c_values=None,
    ) -> tuple[np.ndarray, float | None, float | None]:
        counts = aggregation.counts
        image = np.zeros((*counts.shape, 4), dtype=np.uint8)
        color_lo = color_hi = None
        means = aggregation.mean_values()
        if means is None:
            image[..., :3] = np.asarray(color[:3], dtype=np.uint8)
        else:
            color_lo, color_hi = self._cpu_color_levels(c_values)
            bins, _lo, _hi = self._linear_color_bins(
                means, levels=(color_lo, color_hi)
            )
            lut = colormap_lut(
                self._c_mapping,
                n=256,
                invert=self._lut_invert,
                gamma=self._lut_gamma,
                alpha=True,
            )
            image[..., :3] = lut[bins, :3]
        image[..., 3] = self._cpu_density_alpha(counts, self._point_alpha)
        if means is not None:
            image[~np.isfinite(means), 3] = 0
        return image, color_lo, color_hi

    def _cpu_color_levels(self, values) -> tuple[float, float]:
        """C range without materialising another full finite-value array."""

        levels = self._manual_color_levels
        if levels is not None:
            lo, hi = (float(level) for level in levels)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return lo, hi
        source = np.asarray(values).ravel()
        data_lo = np.inf
        data_hi = -np.inf
        for start in range(0, source.size, 1_000_000):
            chunk = np.asarray(source[start:start + 1_000_000], dtype=float)
            finite = chunk[np.isfinite(chunk)]
            if finite.size:
                data_lo = min(data_lo, float(np.min(finite)))
                data_hi = max(data_hi, float(np.max(finite)))
        if not np.isfinite(data_lo):
            return 0.0, 1.0
        if data_hi > data_lo:
            return data_lo, data_hi
        if np.issubdtype(source.dtype, np.bool_) or data_lo == 0.0:
            return 0.0, 1.0
        return (0.0, data_lo) if data_lo > 0.0 else (data_lo, 0.0)

    def _cpu_line_values(
        self,
        x,
        y,
        *,
        bounds,
        budget: int,
        c_values=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spatially representative curve vertices with original gaps retained."""

        x_values = np.asarray(x).ravel()
        y_values = np.asarray(y).ravel()
        n = min(x_values.size, y_values.size)
        finite = np.isfinite(x_values[:n]) & np.isfinite(y_values[:n])
        if c_values is not None:
            colors = np.asarray(c_values, dtype=float).ravel()
            n = min(n, colors.size)
            finite = finite[:n] & np.isfinite(colors[:n])
        x0, x1, y0, y1 = bounds
        finite &= (
            (x_values[:n] >= x0) & (x_values[:n] <= x1)
            & (y_values[:n] >= y0) & (y_values[:n] <= y1)
        )
        candidates = np.flatnonzero(finite)
        if candidates.size == 0:
            return np.empty(0), np.empty(0)
        selected = spatial_representative_indices(
            x_values[:n], y_values[:n], budget, candidate_indices=candidates
        )
        if selected.size == 0:
            return np.empty(0), np.empty(0)
        line_x = x_values[selected]
        line_y = y_values[selected]
        # Selection skips ordinary valid vertices for LOD; those must remain
        # connected. Break only if the omitted source span contains an invalid
        # or out-of-view row (for example an original NaN separator).
        excluded_prefix = np.r_[0, np.cumsum(~finite, dtype=np.int64)]
        excluded_between = (
            excluded_prefix[selected[1:]]
            - excluded_prefix[selected[:-1] + 1]
        )
        breaks = np.flatnonzero(excluded_between > 0) + 1
        if breaks.size:
            line_x = np.insert(line_x, breaks, np.nan)
            line_y = np.insert(line_y, breaks, np.nan)
        return line_x, line_y

    def _draw_cpu_2d_series(
        self,
        records: list[dict],
        dimensions: tuple[str, str],
        *,
        use_lines: bool,
        stacked: bool,
    ) -> tuple[float | None, float | None]:
        """Exact bulk markers when sparse; complete screen reduction when dense."""

        del stacked  # the records already encode per-iteration series/colours
        self._stack.setCurrentWidget(self._plot_page)
        self._clear_series()
        self._clear_gl_series()
        self._show_gl_canvas(False)
        self._set_plot_transparent(False)
        geometry = self._cpu_view_geometry(records, dimensions)
        bounds, width, height, manual_view = geometry
        self._cpu_render_summary = "CPU: no drawable coordinate pairs"
        self._cpu_aggregate_active = False
        if bounds is None:
            self._clear_gl_bounds()
            return None, None

        # Reset View must always know the full drawable extent, even though a
        # zoomed aggregation image covers only the current viewport.
        full_extents = [
            joint_extent(
                record["values"][dimensions[0]],
                record["values"][dimensions[1]],
            )
            for record in records
        ]
        full_extents = [extent for extent in full_extents if extent is not None]
        if full_extents:
            self._set_gl_bounds(
                min(extent[0] for extent in full_extents),
                max(extent[1] for extent in full_extents),
                min(extent[2] for extent in full_extents),
                max(extent[3] for extent in full_extents),
            )

        pixels = max(1, width * height)
        total_visible = 0
        total_drawable = 0
        total_rendered = 0
        aggregated_cells = 0
        any_aggregate = False
        color_lo = color_hi = None
        for record in records:
            values = record["values"]
            x_values = np.asarray(values[dimensions[0]]).ravel()
            y_values = np.asarray(values[dimensions[1]]).ravel()
            c_values = np.asarray(values["C"]).ravel() if "C" in values else None
            n = min(x_values.size, y_values.size)
            if c_values is not None:
                n = min(n, c_values.size)
                c_values = c_values[:n]
            x_values = x_values[:n]
            y_values = y_values[:n]
            aggregation = aggregate_screen_points(
                x_values,
                y_values,
                bounds=bounds,
                width=width,
                height=height,
                values=c_values,
            )
            total_drawable += aggregation.drawable_count
            total_visible += aggregation.visible_count
            # Per-point C grouping costs more state changes than a shared pen;
            # both thresholds are derived from the actual display resolution.
            exact_limit = pixels if c_values is not None else pixels * 2
            aggregate = aggregation.visible_count > exact_limit
            any_aggregate |= aggregate

            if aggregate:
                image, lo, hi = self._cpu_image(
                    aggregation, color=record["color"], c_values=c_values
                )
                item = pg.ImageItem(image, axisOrder="row-major")
                x0, x1, y0, y1 = bounds
                item.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
                self._plot.addItem(item)
                rendered = aggregation.occupied_count
                aggregated_cells += rendered
                total_rendered += rendered
                if lo is not None and hi is not None:
                    color_lo = lo if color_lo is None else min(color_lo, lo)
                    color_hi = hi if color_hi is None else max(color_hi, hi)
            else:
                x0, x1, y0, y1 = bounds
                marker_mask = (
                    np.isfinite(x_values) & np.isfinite(y_values)
                    & (x_values >= x0) & (x_values <= x1)
                    & (y_values >= y0) & (y_values <= y1)
                )
                bins = lut = None
                if c_values is not None:
                    marker_mask &= np.isfinite(c_values)
                    bins, lo, hi = self._linear_color_bins(
                        c_values[marker_mask], levels=self._manual_color_levels
                    )
                    lut = colormap_lut(
                        self._c_mapping,
                        n=256,
                        invert=self._lut_invert,
                        gamma=self._lut_gamma,
                        alpha=True,
                    ).copy()
                    lut[:, 3] = self._point_alpha
                    color_lo = lo if color_lo is None else min(color_lo, lo)
                    color_hi = hi if color_hi is None else max(color_hi, hi)
                item = BulkScatterItem(
                    x_values[marker_mask],
                    y_values[marker_mask],
                    color=(*record["color"][:3], self._point_alpha),
                    size=self._point_size,
                    symbol=self._point_symbol,
                    color_bins=bins,
                    lut=lut,
                )
                self._plot.addItem(item)
                total_rendered += item.point_count

            line_x = line_y = np.empty(0)
            if use_lines:
                line_x, line_y = self._cpu_line_values(
                    x_values,
                    y_values,
                    bounds=bounds,
                    budget=pixels,
                    c_values=c_values,
                )
            curve = self._plot.plot(
                line_x,
                line_y,
                pen=self._line_pen(
                    (*record["color"][:3], self._point_alpha)
                ) if use_lines else None,
                name=record["name"] if use_lines else None,
            )
            self._series_items.append((curve, item))

        self._cpu_aggregate_active = any_aggregate
        # A user-set view stores only that viewport in an exact item too, so a
        # settled pan/zoom must rebuild even when it did not cross the LOD edge.
        self._view_restricted = manual_view
        self._thinned = any_aggregate
        if any_aggregate:
            reduction = "count + mean C" if any(
                "C" in record.get("values", {}) for record in records
            ) else "count"
            self._cpu_render_summary = (
                f"CPU screen aggregation ({reduction}): {total_visible:,} visible "
                f"drawable rows → {aggregated_cells:,} occupied cells"
            )
        else:
            self._cpu_render_summary = (
                f"CPU bulk painting: {total_rendered:,} visible markers "
                f"({total_drawable:,} drawable rows)"
            )
        self._plot.setLabel("bottom", self._dimension_attrs[dimensions[0]])
        self._plot.setLabel("left", self._dimension_attrs[dimensions[1]])
        return color_lo, color_hi

    # ------------------------------------------------------------------
    # Experimental GPU 2-D renderer
    #
    # The GL canvas sits *behind* the pyqtgraph plot, which stays on screen
    # with a transparent background. Only the markers move to the GPU, so the
    # axes, grid, zoom modes, Reset View, plot style, the ROI overlay and its
    # selection all keep working exactly as they do on the CPU path.
    # ------------------------------------------------------------------

    def _gpu_unavailable(self, reason: str) -> bool:
        """Fall back to pyqtgraph and say so, instead of drawing nothing.

        The toggle is switched off as well: a ticked menu entry over a plot
        that is being drawn on the CPU would be a lie, and every later draw
        would retry the same failing import or context.
        """
        self._use_gl_2d = False
        self._clear_gl_bounds()
        # Without the GPU the CPU renderer needs the display budget back, or a
        # multi-million-row selection would take seconds per redraw.
        self._thinning = True
        message = f"GPU rendering unavailable ({reason}); drawing on the CPU."
        # _draw rewrites the status line at the end, so carry the reason there
        # instead of losing it a few milliseconds after setting it.
        self._gpu_fallback_note = message
        self._info.setText(message)
        try:
            self._state.log(message, level="WARN")
        except Exception:
            pass
        return False

    def _ensure_gl_canvas(self) -> bool:
        """Create the GL canvas in the plot's own layout cell, behind it."""
        if not self._gpu_allowed():
            capabilities = getattr(self._state, "gpu_capabilities", None)
            reason = getattr(capabilities, "reason", "startup capability probe failed")
            return self._gpu_unavailable(str(reason))
        if self._gl2d_view is not None:
            return True
        try:
            import pyqtgraph.opengl as gl
        except Exception as exc:                       # noqa: BLE001 - report it
            # ImportError without PyOpenGL; other failures on a machine whose
            # OpenGL libraries cannot be loaded at all.
            return self._gpu_unavailable(f"PyOpenGL is not usable: {exc}")
        try:
            view = gl.GLViewWidget()
            view.setBackgroundColor(QColor(*self._current_background_color()))
            # Looking straight down -Z at a plane, so the projection is a plain
            # scale and the item transform can match the ViewBox exactly.
            view.opts["elevation"] = 90.0
            view.opts["azimuth"] = -90.0
            view.opts["fov"] = 60.0
            view.opts["distance"] = 1.0
            view.opts["center"] = pg.Vector(0.0, 0.0, 0.0)
            # The plot on top owns every interaction; the canvas must not
            # orbit, pan or zoom on its own.
            view.mousePressEvent = lambda event: event.ignore()
            view.mouseMoveEvent = lambda event: event.ignore()
            view.wheelEvent = lambda event: event.ignore()
            self._gl_module = gl
            # Deliberately NOT in the page's layout: its geometry is driven to
            # match the plot area, which a layout would immediately override.
            view.setParent(self._plot_page)
            view.lower()
            self._plot.raise_()
            self._gl2d_view = view
            view_box = self._plot.getPlotItem().getViewBox()
            view_box.sigRangeChanged.connect(self._sync_gl_canvas)
            view_box.sigResized.connect(self._sync_gl_canvas)
            # Judge the canvas only once it has presented a frame — checking
            # earlier reports an unpainted widget as broken and would switch a
            # perfectly good GPU off.
            view.frameSwapped.connect(self._on_gl_canvas_frame)
        except Exception as exc:                       # noqa: BLE001 - report it
            return self._gpu_unavailable(str(exc))
        return True

    @staticmethod
    def _canvas_rendered_nothing(view) -> bool:
        """True when the canvas painted one flat colour despite holding points.

        A context can report itself valid and still draw nothing — a forced
        software rasterizer does exactly that here, raising GL errors that
        pyqtgraph catches and prints rather than propagates. Checking the
        pixels is the only way to notice, and it is done once per canvas.
        """
        try:
            image = view.grabFramebuffer()
            if image.isNull() or image.width() < 2 or image.height() < 2:
                return True
            image = image.convertToFormat(QImage.Format.Format_RGB32)
            pointer = image.constBits()
            pointer.setsize(image.sizeInBytes())
            rows = np.frombuffer(pointer, dtype=np.uint8).reshape(
                image.height(), image.bytesPerLine()
            )
            pixels = rows[:, : image.width() * 4].reshape(-1, 4)
            return not bool(np.any(pixels != pixels[0]))
        except Exception:
            # Cannot tell — never take a working plot away on a guess.
            return False

    def _on_gl_canvas_frame(self) -> None:
        """First presented frame: schedule the one-shot capability check."""
        if self._gl2d_verified or not self._use_gl_2d:
            return
        self._gl2d_verified = True
        QTimer.singleShot(150, self._verify_gl_canvas)

    def _verify_gl_canvas(self) -> None:
        """Drop back to the CPU if the canvas cannot actually draw."""
        view = self._gl2d_view
        if view is None or not self._use_gl_2d or not view.isVisible():
            return
        if not self._gl2d_items:
            return
        try:
            valid = bool(view.isValid())
        except Exception:
            valid = False
        if valid and not self._canvas_rendered_nothing(view):
            self._gl2d_blank_checks = 0
            return
        if valid:
            # One blank frame can be a transient; a second, later one is not.
            self._gl2d_blank_checks += 1
            if self._gl2d_blank_checks < 2:
                QTimer.singleShot(400, self._verify_gl_canvas)
                return
            self._gpu_unavailable("the OpenGL canvas rendered nothing")
        else:
            self._gpu_unavailable("no OpenGL context on this display")
        self._show_gl_canvas(False)
        self._draw()

    def _set_plot_transparent(self, transparent: bool) -> None:
        """Let the GL canvas show through the plot, or paint over it again."""
        widgets = (self._plot, self._plot.viewport())
        for widget in widgets:
            widget.setAttribute(
                Qt.WidgetAttribute.WA_TranslucentBackground, transparent
            )
            widget.setAttribute(
                Qt.WidgetAttribute.WA_NoSystemBackground, transparent
            )
            widget.setAutoFillBackground(not transparent)
            widget.setStyleSheet(
                "background: transparent; border: none;" if transparent else ""
            )
        if transparent:
            self._plot.setBackground(None)
        else:
            self._plot.setBackground(QColor(*self._current_background_color()))
        self._apply_page_background()

    def _apply_page_background(self) -> None:
        """In GPU mode the page paints the plot background.

        The canvas covers only the data area, so without this the axis strips
        keep the window's pale colour while their tick text is drawn light for
        a dark plot — unreadable. A palette is used rather than a stylesheet so
        nothing cascades onto the transparent plot or the canvas.
        """
        page = getattr(self, "_plot_page", None)
        if page is None:
            return
        if not self._use_gl_2d:
            page.setAutoFillBackground(False)
            return
        palette = page.palette()
        palette.setColor(
            page.backgroundRole(), QColor(*self._current_background_color())
        )
        page.setPalette(palette)
        page.setAutoFillBackground(True)

    def _clear_gl_bounds(self) -> None:
        if self._gl_bounds_item is None:
            return
        try:
            self._plot.removeItem(self._gl_bounds_item)
        except Exception:
            pass
        self._gl_bounds_item = None

    def _set_gl_bounds(
        self, x_lo: float, x_hi: float, y_lo: float, y_hi: float
    ) -> None:
        """Publish the GPU data extent to the ViewBox as an invisible outline.

        ⚠ `ViewBox.childrenBounds` skips items that are not visible, and in GPU
        mode the scatter item is both empty and hidden — so without this the
        view has **no** data bounds. Auto-range then fits whatever else happens
        to be in the scene: during a zoom drag that is the rubber band itself,
        which collapses the view onto the band as it is dragged, and afterwards
        neither Reset View nor the `A` button can find the data again.

        The item paints nothing; it only carries the data rectangle, so
        auto-range, Reset View and the `A` button behave exactly as they do
        when pyqtgraph is drawing the points itself.
        """
        self._clear_gl_bounds()
        if not all(np.isfinite([x_lo, x_hi, y_lo, y_hi])):
            return
        item = _DataBoundsItem(
            QRectF(x_lo, y_lo, max(x_hi - x_lo, 1e-12), max(y_hi - y_lo, 1e-12))
        )
        item.setZValue(-1e6)
        self._plot.addItem(item)
        self._gl_bounds_item = item

    def _clear_gl_2d_items(self) -> None:
        view = self._gl2d_view
        if view is None:
            self._gl2d_items = []
            return
        for item in self._gl2d_items:
            try:
                view.removeItem(item)
            except Exception:
                pass
        self._gl2d_items = []

    def _show_gl_canvas(self, visible: bool) -> None:
        if not visible:
            self._clear_gl_bounds()
        if self._gl2d_view is None:
            return
        self._gl2d_view.setVisible(visible)
        if not visible:
            self._clear_gl_2d_items()

    def _sync_gl_canvas(self, *_args) -> None:
        """Map the ViewBox range onto the canvas with a per-item transform.

        The canvas spans the whole page while the ViewBox occupies only the
        area inside the axes, so the transform targets that sub-rectangle. It
        is an affine update per frame — the positions themselves are uploaded
        once, which is what keeps panning at a few tens of milliseconds even
        with millions of points.
        """
        view = self._gl2d_view
        if view is None or not view.isVisible():
            return
        try:
            view_box = self._plot.getPlotItem().getViewBox()
            (x0, x1), (y0, y1) = view_box.viewRange()
            rect = view_box.sceneBoundingRect()
        except Exception:
            return
        if x1 <= x0 or y1 <= y0 or rect.width() <= 1 or rect.height() <= 1:
            return
        # The canvas *is* the plot area, so anything outside the current range
        # falls outside the widget and is clipped rather than drawn over the
        # axes.
        offset = self._plot.viewport().mapTo(self._plot_page, QPoint(0, 0))
        view.setGeometry(
            int(round(rect.left())) + offset.x(),
            int(round(rect.top())) + offset.y(),
            max(1, int(round(rect.width()))),
            max(1, int(round(rect.height()))),
        )
        if not self._gl2d_items:
            return
        width, height = max(1, view.width()), max(1, view.height())
        half_w = np.tan(np.radians(view.opts["fov"] / 2.0)) * view.opts["distance"]
        half_h = half_w * height / width
        origin_x, origin_y = self._gl2d_origin
        span_x, span_y = self._gl2d_span
        # Data range -> the canvas' full extent, in normalized data units.
        scale_x = 2.0 * half_w / (x1 - x0) * span_x
        scale_y = 2.0 * half_h / (y1 - y0) * span_y
        shift_x = (origin_x - (x0 + x1) / 2.0) / span_x
        shift_y = (origin_y - (y0 + y1) / 2.0) / span_y
        matrix = QMatrix4x4()
        matrix.scale(scale_x, scale_y, 1.0)
        matrix.translate(shift_x, shift_y, 0.0)
        for item in self._gl2d_items:
            item.setTransform(matrix)
        view.update()

    def _draw_gl_2d_series(
        self, records: list[dict], dimensions: tuple[str, str]
    ) -> tuple[float | None, float | None]:
        """Upload the points to the canvas; the plot above draws everything else."""
        if not self._ensure_gl_canvas():
            return None, None
        self._clear_gl_2d_items()
        self._show_gl_canvas(True)
        x_dimension, y_dimension = dimensions
        blocks = [
            (
                np.asarray(record["values"][x_dimension], dtype=float),
                np.asarray(record["values"][y_dimension], dtype=float),
                record,
            )
            for record in records
            if record["values"]
        ]
        if not blocks:
            return None, None
        finite_pairs = [
            (x[mask], y[mask])
            for x, y, _record in blocks
            for mask in (np.isfinite(x) & np.isfinite(y),)
            if mask.any()
        ]
        finite_x = np.concatenate([pair[0] for pair in finite_pairs]) if finite_pairs else np.empty(0)
        finite_y = np.concatenate([pair[1] for pair in finite_pairs]) if finite_pairs else np.empty(0)
        if finite_x.size == 0 or finite_y.size == 0:
            return None, None
        # Centre and scale once so float32 keeps sub-pixel precision even for
        # idx values in the tens of millions.
        self._gl2d_origin = (
            float((np.min(finite_x) + np.max(finite_x)) / 2.0),
            float((np.min(finite_y) + np.max(finite_y)) / 2.0),
        )
        self._gl2d_span = (
            float(max(np.ptp(finite_x), 1e-12)),
            float(max(np.ptp(finite_y), 1e-12)),
        )
        # Prefer the un-thinned extent: what is on the canvas may be only the
        # rows inside the current view.
        self._set_gl_bounds(*(self._data_extent or (
            float(np.min(finite_x)), float(np.max(finite_x)),
            float(np.min(finite_y)), float(np.max(finite_y)),
        )))
        origin_x, origin_y = self._gl2d_origin
        span_x, span_y = self._gl2d_span

        color_lo = color_hi = None
        for x_values, y_values, record in blocks:
            finite = np.isfinite(x_values) & np.isfinite(y_values)
            if not finite.any():
                continue
            pos = np.column_stack([
                (x_values[finite] - origin_x) / span_x,
                (y_values[finite] - origin_y) / span_y,
                np.zeros(int(finite.sum())),
            ]).astype(np.float32, copy=False)
            if "C" in record["values"]:
                rgba, color_lo, color_hi, *_ = self._mapped_rgba(
                    record["values"]["C"]
                )
                rgba = rgba[finite]
            else:
                base = np.asarray(
                    (*record["color"][:3], self._point_alpha), dtype=float
                )
                rgba = np.tile(base / 255.0, (pos.shape[0], 1)).astype(np.float32)
            item = self._gl_module.GLScatterPlotItem(
                pos=pos,
                color=rgba,
                size=float(self._point_size),
                pxMode=True,
            )
            self._apply_3d_blend(item)
            self._gl2d_view.addItem(item)
            self._gl2d_items.append(item)
        self._sync_gl_canvas()
        return color_lo, color_hi

    def _draw_3d_series(
        self,
        records: list[dict],
        *,
        use_lines: bool,
    ) -> tuple[float | None, float | None]:
        if not self._ensure_3d_built():
            return None, None
        self._stack.setCurrentWidget(self._3d_view)
        self._clear_series()
        self._clear_gl_series()
        self._show_gl_canvas(False)
        raw_positions = [
            np.column_stack([
                record["values"]["X"],
                record["values"]["Y"],
                record["values"]["Z"],
            ])
            for record in records
            if record["values"]
        ]
        if not raw_positions:
            self._clear_3d_axis_items()
            return None, None
        combined = np.vstack(raw_positions)
        frozen = (
            self._3d_extent
            if self._freeze_3d_extent and self._3d_extent is not None
            else None
        )
        normalized, mins, maxs = self._normalize_3d_positions(combined, frozen)
        if np.all(np.isfinite(mins)):
            self._3d_extent = (mins, maxs)
        self._update_3d_axis_items(combined, self._3d_extent)
        offset = 0
        color_lo = color_hi = None
        all_finite_positions: list[np.ndarray] = []
        for record, raw in zip(records, raw_positions):
            pos = normalized[offset:offset + raw.shape[0]]
            offset += raw.shape[0]
            finite = np.all(np.isfinite(pos), axis=1)
            pos = pos[finite].astype(np.float32, copy=False)
            if pos.size == 0:
                continue
            all_finite_positions.append(pos)
            if "C" in record["values"]:
                rgba, color_lo, color_hi, *_ = self._mapped_rgba(
                    record["values"]["C"]
                )
                rgba = rgba[finite]
                line_visible = np.isfinite(
                    np.asarray(record["values"]["C"], dtype=float)
                )[finite]
            else:
                base = np.asarray(
                    (*record["color"][:3], self._point_alpha), dtype=float
                )
                rgba = np.tile(base / 255.0, (pos.shape[0], 1)).astype(np.float32)
                line_visible = np.ones(pos.shape[0], dtype=bool)
            scatter = self._gl_module.GLScatterPlotItem(
                pos=pos, color=rgba, size=float(self._point_size), pxMode=True
            )
            self._apply_3d_blend(scatter)
            self._3d_view.addItem(scatter)
            self._gl_series_items.append(scatter)
            if use_lines and pos.shape[0] > 1:
                # GL line strips do not recognise NaN vertices as 2-D
                # pyqtgraph curves do.  Draw each contiguous finite-C run so
                # missing C values leave a real gap in a 3-D connected plot.
                edges = np.flatnonzero(
                    np.diff(np.r_[False, line_visible, False].astype(np.int8))
                )
                for start, stop in zip(edges[::2], edges[1::2]):
                    if stop - start < 2:
                        continue
                    line = self._gl_module.GLLinePlotItem(
                        pos=pos[start:stop],
                        color=rgba[start:stop],
                        width=1.0,
                        mode="line_strip",
                    )
                    self._apply_3d_blend(line)
                    self._3d_view.addItem(line)
                    self._gl_series_items.append(line)
        if all_finite_positions and not self._3d_camera_initialised:
            self._3d_view.opts["center"] = pg.Vector(0.0, 0.0, 0.0)
            self._3d_view.setCameraPosition(distance=2.5)
            self._3d_camera_initialised = True
        return color_lo, color_hi

    def _save_view_state(self, ds, *, use_lines: bool, filtered_only: bool, vld_only: bool) -> None:
        ds.state[self._view_state_key] = {
            "x": self._dimension_attrs["X"],
            "y": self._dimension_attrs["Y"],
            "z": self._dimension_attrs["Z"],
            "c": self._dimension_attrs["C"],
            "dimension_count": self._dimension_count,
            "has_z": bool(self._has_z),
            "has_c": bool(self._has_c),
            "view": self._view_mode,
            "c_mapping": self._c_mapping,
            "c_mapping_invert": self._lut_invert,
            "c_mapping_gamma": self._lut_gamma,
            "c_mapping_levels": (
                list(self._manual_color_levels)
                if self._manual_color_levels is not None
                else None
            ),
            "black_background": self._black_background,
            "show_2d_axis": self._show_2d_axis,
            "show_3d_axis": self._show_3d_axis,
            "show_2d_grid": self._show_2d_grid,
            "show_3d_grid": self._show_3d_grid,
            "show_3d_bounding_box": self._show_3d_bounding_box,
            "show_colorbar": self._show_colorbar,
            "show_legend": bool(self._show_legend),
            "legend_geometry": self._legend_geometry,
            "colorbar_show_values": self._colorbar_show_values,
            "colorbar_orientation": self._colorbar_orientation,
            "colorbar_geometry": self._colorbar_geometry,
            "point_symbol": self._point_symbol,
            "line_style": self._line_style,
            "line_width": float(self._line_width),
            "point_size": self._point_size,
            "point_alpha": self._point_alpha,
            "point_color": list(self._point_color),
            "plot_style_custom": self._plot_style_custom,
            "lines": use_lines,
            "filtered_only": filtered_only,
            "iter": self._iter_combo.currentText() or "",
            "valid_only": vld_only,
            "thinning": bool(self._thinning),
            "gl_2d": bool(self._use_gl_2d),
        }

    def _point_count_text(self, records: list[dict], total: int) -> str:
        """Points drawn out of points selected, so thinning is never silent.

        The two reasons for a short draw read differently: off-screen rows were
        simply not painted (nothing about the visible plot is approximate),
        while a stride means the visible region itself is sampled.
        """
        drawn = 0
        drawable_drawn = 0
        for record in records:
            values = record.get("values") or {}
            if values:
                drawn += int(next(iter(values.values())).size)
                mask = self._drawable_row_mask(values)
                if mask is not None and "C" in values:
                    mask = mask & np.isfinite(np.asarray(values["C"], dtype=float))
                drawable_drawn += (
                    int(mask.sum()) if mask is not None
                    else int(next(iter(values.values())).size)
                )
        self._thinned = drawn < total
        if not self._thinned:
            if drawable_drawn < total:
                return f"{drawable_drawn:,} drawable of {total:,} selected rows"
            return f"{total:,} points"
        step = max(1, int(getattr(self, "_thin_step", 1)))
        method = getattr(self, "_thin_method", "none")
        drawable = getattr(self, "_thin_drawable", None)
        if self._view_restricted:
            scope = (
                "visible range" if step == 1
                else (
                    "spatially representative cells of the visible range"
                    if method == "spatial cells" else
                    f"1 in {step:,} of the visible range"
                )
            )
        elif drawable is not None:
            # Naming the drawable pool keeps the ratio checkable: the rest of
            # the selection has no finite coordinate and could never be drawn.
            scope = (
                f"all {drawable:,} with a finite value" if step == 1
                else (
                    f"spatially representative cells of the {drawable:,} "
                    "with finite coordinates"
                    if method == "spatial cells" else
                    f"1 in {step:,} of the {drawable:,} with a finite value"
                )
            )
        else:
            scope = (
                "spatially representative cells"
                if method == "spatial cells" else f"1 in {step:,}"
            )
        return f"{drawn:,} of {total:,} points ({scope})"

    def _draw_keeping_view(self) -> None:
        """Redraw without moving the view, in 2-D or in 3-D.

        Toggling Valid only / Lines / Filtered only is a comparison: the same
        window on the data has to be there before and after, so the range is
        put back even though the data extent may have changed under it — and
        even when the view was still auto-ranging, which would otherwise refit
        to the new extent and throw away what the user was looking at. In 3-D
        the equivalent is the cube's extent, pinned for the same reason.
        """
        keep = None
        if self._view_mode != "3D":
            try:
                (x0, x1), (y0, y1) = self._view_box.viewRange()
                if x1 > x0 and y1 > y0:
                    keep = ((x0, x1), (y0, y1))
            except Exception:
                keep = None
        # 3-D has no range to restore: its cube is normalized from the data, so
        # the extent is pinned instead and the points stay where they were.
        self._freeze_3d_extent = True
        try:
            self._draw()
        finally:
            self._freeze_3d_extent = False
        if keep is not None:
            self._view_box.setRange(
                xRange=keep[0], yRange=keep[1], padding=0.0
            )

    def _draw(self, *, reuse_series: bool = False) -> None:
        ds = self._dataset()
        if ds is None or not self._numeric_attrs:
            return
        if not reuse_series:
            self._series_cache = None
        self._data_extent = None

        itr_sel, render = self._selection()
        if self._has_c and render == "stacked":
            render = "flatten"
        vld_only = self._valid_chk.isChecked()
        filtered_only = self._filter_chk.isChecked()
        use_lines = self._lines_chk.isChecked()
        dimensions = self._spatial_dimensions()
        data_dimensions = (
            (*dimensions, "C") if self._has_c else dimensions
        )
        self._save_view_state(
            ds,
            use_lines=use_lines,
            filtered_only=filtered_only,
            vld_only=vld_only,
        )

        records: list[dict] = []
        total = 0
        missing: list[str] = []
        if render == "stacked":
            n_itr = self._num_itr(ds)
            for iteration in range(n_itr):
                values, n, miss = self._series_data(
                    ds, data_dimensions, iteration, vld_only, filtered_only,
                    reuse=reuse_series,
                )
                total += n
                missing = miss or missing
                if values:
                    records.append({
                        "values": values,
                        "color": _iter_color(self._state.prefs, iteration, n_itr),
                        "name": ordinal(iteration + 1),
                    })
            note = (
                f"{self._point_count_text(records, total)} across "
                f"{n_itr} iterations  |  all [stacked]"
            )
        else:
            selector = "all" if render == "flatten" else itr_sel
            values, total, missing = self._series_data(
                ds, data_dimensions, selector, vld_only, filtered_only,
                reuse=reuse_series,
            )
            if values:
                records.append({
                    "values": values,
                    "color": self._point_color,
                    "name": None,
                })
            note = (
                f"{self._point_count_text(records, total)}  |  "
                f"{self._iter_combo.currentText() or 'last'}"
            )

        self._update_legend(records if render == "stacked" else [])
        if self._has_c and records:
            color_arrays = [
                np.asarray(record["values"]["C"], dtype=float).ravel()
                for record in records
                if "C" in record["values"]
            ]
            self._last_color_values = (
                color_arrays[0] if len(color_arrays) == 1
                else np.concatenate(color_arrays)
            )
        else:
            self._last_color_values = np.empty(0, dtype=float)

        if self._view_mode == "3D":
            color_lo, color_hi = self._draw_3d_series(records, use_lines=use_lines)
            axis_note = " / ".join(
                f"{dimension}={self._dimension_attrs[dimension]}"
                for dimension in ("X", "Y", "Z")
            )
            note += f"  |  3D (independently scaled): {axis_note}"
        else:
            color_lo, color_hi = self._draw_2d_series(
                records,
                self._visible_dimensions(),
                use_lines=use_lines,
                stacked=render == "stacked",
            )
            note += f"  |  view: {self._view_mode}"
            if self._use_gl_2d:
                note += " (GPU)"
            elif self._cpu_fix and self._cpu_render_summary:
                note += f"  |  {self._cpu_render_summary}"

        requested_names = [self._dimension_attrs[dim] for dim in data_dimensions]
        if missing:
            axis_missing = [name for name in missing if name in requested_names]
            filter_missing = [name for name in missing if name not in requested_names]
            if axis_missing:
                note += f"  |  {', '.join(dict.fromkeys(axis_missing))} has no per-iteration values"
            if filter_missing:
                note += f"  |  filter on {', '.join(dict.fromkeys(filter_missing))} not applied"
        if self._has_c:
            note += f"  |  C={self._dimension_attrs['C']} [{self._c_mapping}]"
            if color_lo is not None and color_hi is not None:
                note += f" {color_lo:g}..{color_hi:g}"
        if not vld_only:
            note += "  |  incl. invalid"
        if self._gpu_fallback_note:
            note += f"  |  {self._gpu_fallback_note}"
        self._info.setText(note)
        self._update_colorbar(color_lo, color_hi)

    def open_lut_dialog(self) -> None:
        """Open the shared LUT editor for the fourth (C) dimension."""
        if not self._has_c:
            self._info.setText(
                "LUT unavailable: add a C attribute dimension first."
            )
            return
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
            self._info.setText("LUT unavailable: C has no finite values.")
            return
        self._lut_dialog.show()
        self._lut_dialog.raise_()
        self._lut_dialog.activateWindow()

    def _refresh_lut_dialog(self, *, capture_baseline: bool) -> bool:
        dialog = self._lut_dialog
        if dialog is None or not self._has_c:
            return False
        values = np.asarray(self._last_color_values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            self._draw()
            values = np.asarray(self._last_color_values, dtype=float)
            values = values[np.isfinite(values)]
        if values.size == 0:
            return False
        data_lo = float(np.min(values))
        data_hi = float(np.max(values))
        if data_hi <= data_lo:
            data_hi = data_lo + 1.0
        lo, hi = self._manual_color_levels or (data_lo, data_hi)
        dialog.load_image(
            pixels=values,
            data_lo=data_lo,
            data_hi=data_hi,
            lo=float(lo),
            hi=float(hi),
            cmap_name=self._c_mapping,
            invert=self._lut_invert,
            gamma=self._lut_gamma,
            capture_baseline=capture_baseline,
        )
        return True

    def sync_lut_dialog(self) -> None:
        dialog = self._lut_dialog
        try:
            if dialog is None or not dialog.isVisible() or dialog.isActiveWindow():
                return
        except RuntimeError:
            return
        self._refresh_lut_dialog(capture_baseline=False)

    def _on_lut_levels_changed(self, lo: float, hi: float) -> None:
        self._manual_color_levels = (float(lo), float(hi))
        self._draw()

    def _on_lut_cmap_changed(self, name: str, invert: bool) -> None:
        try:
            make_colormap(name)
        except (KeyError, TypeError, ValueError):
            return
        self._c_mapping = name
        self._lut_invert = bool(invert)
        self._draw()

    def _on_lut_invert_changed(self, invert: bool) -> None:
        self._on_lut_cmap_changed(self._c_mapping, invert)

    def _on_lut_gamma_changed(self, gamma: float) -> None:
        self._lut_gamma = max(0.01, float(gamma))
        self._draw()

    def compute_roi_selection(self, record):
        if record.type != "rectangle" or self._view_mode == "3D":
            return None
        ds = self._dataset()
        if ds is None:
            return None
        # ROI selection maps back onto the materialized (last-iteration, valid)
        # store. It cannot be mapped when the view shows a different iteration
        # or includes invalid localizations.
        itr_sel, render = self._selection()
        if (itr_sel != "last" or render != "single"
                or not attr_matches_selection(
                    ds, itr="last", vld_only=self._valid_chk.isChecked())):
            return None
        x_dimension, y_dimension = self._visible_dimensions()
        x_name = self._dimension_attrs[x_dimension]
        y_name = self._dimension_attrs[y_dimension]
        if not x_name or not y_name:
            return None
        x = attr_values_1d(ds, x_name)
        y = attr_values_1d(ds, y_name)
        if x is None or y is None:
            return None
        x = np.asarray(x).ravel().astype(float)
        y = np.asarray(y).ravel().astype(float)
        n = min(x.size, y.size, ds.prop.num_loc)
        if n == 0:
            return None
        base = np.ones(n, dtype=bool)
        if self._filter_chk.isChecked():
            ftr = np.asarray(ds.filter_mask, dtype=bool).ravel()
            if ftr.size == n:
                base = ftr.copy()
        mask = rectangle_mask(x[:n], y[:n], record, base_mask=base)
        if mask.size != ds.prop.num_loc:
            full = np.zeros(ds.prop.num_loc, dtype=bool)
            full[:mask.size] = mask
            mask = full
        context = {
            "source_view": "attribute",
            "dataset_idx": self._dataset_idx,
            "x_attr": x_name,
            "y_attr": y_name,
            "attribute_view": self._view_mode,
            "filtered_only": self._filter_chk.isChecked(),
        }
        return ds, mask, context

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_filter_changed(self, idx: int) -> None:
        if idx == self._dataset_idx:
            self._draw()

    def _on_attributes_changed(self, idx: int) -> None:
        if idx == self._dataset_idx:
            self._refresh()

    def focusInEvent(self, event) -> None:
        if self._dataset_idx is not None and 0 <= self._dataset_idx < len(self._state.datasets):
            self._state.set_active(self._dataset_idx)
        if self._roi_overlay is not None:
            self._roi_overlay.activate()
        super().focusInEvent(event)

    def changeEvent(self, event) -> None:
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            if self._dataset_idx is not None and 0 <= self._dataset_idx < len(self._state.datasets):
                self._state.set_active(self._dataset_idx)
            if self._roi_overlay is not None:
                self._roi_overlay.activate()
        super().changeEvent(event)
