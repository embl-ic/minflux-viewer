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
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
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
_MAX_DISPLAY_POINTS = 50_000


def _row_selector(sel):
    """The row-identity selector for *sel*.

    ``all [sum]`` / ``all [average]`` produce one value per localization laid on
    the ``last`` rows, so anything that identifies a *row* (the synthetic ``idx``,
    ``tid``, a filter mask) must be fetched at ``last`` — pooling those would be
    meaningless. Every other selector is its own row selector.
    """
    return "last" if is_value_pool_selector(sel) else sel


def _iter_color(prefs: dict, k: int) -> tuple[int, int, int, int]:
    colors = list(component_colors(prefs, "functions", "Iteration series").values())
    return colors[k % len(colors)] if colors else (70, 130, 180, 255)


class AttributeWindow(QWidget):
    """Plot two to four numeric attributes against each other."""

    TAG = "attribute_window"

    def __init__(self, state: AppState, parent: QWidget | None = None, *, dataset_idx: int | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._dataset_idx = dataset_idx if dataset_idx is not None else state.active_idx
        self._view_state_key = "attribute_plot_state"
        self._zoom_active = False
        self._zoom_mode = "unconstrained"
        self._zoom_preview = None
        self._zoom_drag_start = None
        self._view_box = None
        self._original_mouse_drag_event = None
        self._numeric_attrs: list[str] = []
        self._dimension_count = 2
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
        self._colorbar_show_values = True
        self._colorbar_orientation = "vertical"
        self._colorbar_geometry: list[int] | None = None
        self._colorbar = None
        self._point_symbol = "o"
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
        self._3d_camera_initialised = False

        self.setWindowTitle("Attribute Plot")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(700, 400)
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
        bar.setSpacing(8)

        self._axis_a_label = QLabel("X:")
        bar.addWidget(self._axis_a_label)
        self._x_combo = QComboBox()
        self._x_combo.setMinimumWidth(100)
        self._x_combo.currentTextChanged.connect(
            lambda text: self._on_visible_attribute_changed(0, text)
        )
        bar.addWidget(self._x_combo)

        self._axis_b_label = QLabel("Y:")
        bar.addWidget(self._axis_b_label)
        self._y_combo = QComboBox()
        self._y_combo.setMinimumWidth(100)
        self._y_combo.currentTextChanged.connect(
            lambda text: self._on_visible_attribute_changed(1, text)
        )
        bar.addWidget(self._y_combo)

        self._iter_label = QLabel("Iter:")
        bar.addWidget(self._iter_label)
        self._iter_combo = QComboBox()
        self._iter_combo.setMinimumWidth(96)
        self._iter_combo.currentTextChanged.connect(self._draw)
        bar.addWidget(self._iter_combo)

        self._valid_chk = QCheckBox("Valid only")
        self._valid_chk.setChecked(True)
        self._valid_chk.setToolTip("Show only vld=True localizations. Uncheck to include invalid ones.")
        self._valid_chk.stateChanged.connect(self._draw)
        bar.addWidget(self._valid_chk)

        self._lines_chk = QCheckBox("Lines")
        self._lines_chk.setChecked(False)
        self._lines_chk.stateChanged.connect(self._draw)
        bar.addWidget(self._lines_chk)

        self._filter_chk = QCheckBox("Filtered only")
        self._filter_chk.setChecked(True)
        self._filter_chk.stateChanged.connect(self._draw)
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
        self._stack.addWidget(self._plot)
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
        )
        self._colorbar.set_bar_visible(False)

        # ── Info ─────────────────────────────────────────────────
        self._info = QLabel("")
        self._info.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _visible_dimensions(self) -> tuple[str, str]:
        return _VIEW_DIMENSIONS.get(self._view_mode, ("X", "Y"))

    def _spatial_dimensions(self) -> tuple[str, ...]:
        if self._view_mode == "3D" and self._dimension_count >= 3:
            return ("X", "Y", "Z")
        return self._visible_dimensions()

    def _active_dimensions(self) -> tuple[str, ...]:
        dimensions = ["X", "Y"]
        if self._dimension_count >= 3:
            dimensions.append("Z")
        if self._dimension_count >= 4:
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
        dimensions = self._visible_dimensions()
        labels = (self._axis_a_label, self._axis_b_label)
        combos = (self._x_combo, self._y_combo)
        self._syncing_axis_combos = True
        try:
            for dimension, label, combo in zip(dimensions, labels, combos):
                label.setText(f"{dimension}:")
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

    def _on_visible_attribute_changed(self, slot: int, text: str) -> None:
        if self._syncing_axis_combos or not text:
            return
        dimensions = self._visible_dimensions()
        if not 0 <= slot < len(dimensions):
            return
        self._dimension_attrs[dimensions[slot]] = text
        combo = self._x_combo if slot == 0 else self._y_combo
        combo.setToolTip(attribute_description(text))
        self._style_iteration_boldness()
        self._draw()

    def _context_dimension(self) -> str:
        return {"XY": "Z", "XZ": "Y", "YZ": "X", "3D": "Z"}.get(
            self._view_mode, "Z"
        )

    def _add_attribute_submenu(self, menu: QMenu, dimension: str) -> None:
        current = self._dimension_attrs.get(dimension, "")
        submenu = menu.addMenu(f"{dimension}: {current}")
        for name in self._numeric_attrs:
            action = submenu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.setToolTip(attribute_description(name))
            action.triggered.connect(
                lambda _checked=False, dim=dimension, value=name:
                self._set_dimension_attribute(dim, value)
            )

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)

        view_menu = menu.addMenu("View")
        view_options = _VIEW_OPTIONS if self._dimension_count >= 3 else ("XY",)
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
        menu.addSeparator()

        if self._dimension_count >= 3:
            self._add_attribute_submenu(menu, self._context_dimension())

        if self._dimension_count >= 4:
            self._add_attribute_submenu(menu, "C")
            mapping_menu = menu.addMenu("C: mapping")
            for name in self._c_mapping_names():
                action = mapping_menu.addAction(name)
                action.setCheckable(True)
                action.setChecked(name == self._c_mapping)
                action.triggered.connect(
                    lambda _checked=False, value=name: self._set_c_mapping(value)
                )
            mapping_menu.addSeparator()
            colorbar_action = mapping_menu.addAction("Colorbar")
            colorbar_action.setCheckable(True)
            colorbar_action.setChecked(self._show_colorbar)
            colorbar_action.triggered.connect(self._set_colorbar_visible)

        add_action = menu.addAction("add new attribute as plot dimension")
        add_action.setEnabled(self._dimension_count < 4)
        add_action.triggered.connect(self._add_attribute_dimension)
        reduce_action = menu.addAction("reduce attribute dimension")
        reduce_action.setEnabled(self._dimension_count > 2)
        reduce_action.triggered.connect(self._reduce_attribute_dimension)

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
        visible = self._dimension_count >= 4 and self._show_colorbar and finite.size > 0
        if not visible:
            self._colorbar.set_bar_visible(False)
            return
        if color_lo is None or color_hi is None:
            if self._manual_color_levels is not None:
                color_lo, color_hi = self._manual_color_levels
            else:
                color_lo, color_hi = float(np.min(finite)), float(np.max(finite))
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
        if dimension == "C":
            self._manual_color_levels = None
        self._sync_visible_attribute_controls()
        self._style_iteration_boldness()
        self._draw()
        if dimension == "C":
            self.sync_lut_dialog()

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
            self._dimension_count < 3 and view != "XY"
        ):
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
            self._stack.setCurrentWidget(self._plot)
            self._apply_2d_reference_visibility()
        self._style_iteration_boldness()
        self._draw()

    def _add_attribute_dimension(self) -> None:
        if self._dimension_count >= 4 or not self._numeric_attrs:
            return
        self._dimension_count += 1
        dimension = "Z" if self._dimension_count == 3 else "C"
        if self._dimension_attrs[dimension] not in self._numeric_attrs:
            self._dimension_attrs[dimension] = self._default_dimension_attribute(dimension)
        self._set_stacked_enabled(self._dimension_count < 4)
        self._sync_visible_attribute_controls()
        self._style_iteration_boldness()
        self._draw()

    def _reduce_attribute_dimension(self) -> None:
        if self._dimension_count <= 2:
            return
        self._dimension_count -= 1
        if self._dimension_count < 3 and self._view_mode != "XY":
            self._view_mode = "XY"
            self._stack.setCurrentWidget(self._plot)
            self._zoom_btn.setEnabled(True)
            self._apply_2d_reference_visibility()
        if self._dimension_count < 4:
            self._last_color_values = np.empty(0, dtype=float)
            try:
                if self._lut_dialog is not None:
                    self._lut_dialog.hide()
            except RuntimeError:
                self._lut_dialog = None
        self._set_stacked_enabled(self._dimension_count < 4)
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

    def _update_3d_axis_items(self, raw_positions: np.ndarray) -> None:
        self._clear_3d_axis_items()
        if not self._show_3d_axis or self._3d_view is None:
            return
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
            self.setWindowTitle("Attribute Plot")
            self._clear_series()
            self._clear_gl_series()
            return

        self.setWindowTitle(f"Attribute Plot  —  {ds.name}")

        numeric = plot_attribute_names(ds, self._state.prefs)
        self._numeric_attrs = numeric
        saved = ds.state.get(self._view_state_key, {})

        try:
            saved_count = int(saved.get("dimension_count", self._dimension_count))
        except (TypeError, ValueError):
            saved_count = 2
        self._dimension_count = max(2, min(4, saved_count))
        for dimension, state_key in (("X", "x"), ("Y", "y"), ("Z", "z"), ("C", "c")):
            candidate = saved.get(state_key, self._dimension_attrs[dimension])
            if candidate in numeric:
                self._dimension_attrs[dimension] = candidate
            else:
                self._dimension_attrs[dimension] = self._default_dimension_attribute(dimension)
        saved_view = str(saved.get("view", self._view_mode))
        self._view_mode = (
            saved_view if self._dimension_count >= 3 and saved_view in _VIEW_OPTIONS else "XY"
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
        self._set_stacked_enabled(self._dimension_count < 4)
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
                self._stack.setCurrentWidget(self._plot)
        else:
            self._zoom_btn.setEnabled(True)
            self._stack.setCurrentWidget(self._plot)

        self._draw()

    def _apply_plot_colors(self) -> None:
        background = self._current_background_color()
        self._plot.setBackground(QColor(*background))
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
        for i in range(combo.count()):
            combo.setItemData(i, attribute_description(combo.itemText(i)), Qt.ItemDataRole.ToolTipRole)

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

    def _clear_series(self) -> None:
        for curve, scatter in self._series_items:
            self._plot.removeItem(curve)
            self._plot.removeItem(scatter)
        self._series_items = []
        if self._legend is not None:
            try:
                self._legend.scene().removeItem(self._legend)
            except Exception:
                pass
            self._legend = None
        # PlotItem caches its legend; clear it or a later addLegend() returns
        # the detached (invisible) old one.
        try:
            self._plot.getPlotItem().legend = None
        except Exception:
            pass

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
    ):
        rgba = tuple(int(channel) for channel in color[:3]) + (
            self._point_alpha,
        )
        # self._plot.plot(name=...) registers a legend sample reliably.
        curve = self._plot.plot(
            x if use_lines else [], y if use_lines else [],
            pen=pg.mkPen(rgba, width=1) if use_lines else None,
            name=name if use_lines else None,
        )
        scatter = pg.ScatterPlotItem(
            size=self._point_size,
            symbol=self._point_symbol,
            pen=None,
            brush=pg.mkBrush(*rgba),
        )
        scatter.setData(
            x,
            y,
            size=self._point_size,
            symbol=self._point_symbol,
            brush=brushes if brushes is not None else pg.mkBrush(*rgba),
        )
        self._plot.addItem(scatter)
        self._series_items.append((curve, scatter))
        if name is not None and not use_lines and self._legend is not None:
            self._legend.addItem(scatter, name)

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

    def _series_data(
        self,
        ds,
        dimensions: tuple[str, ...],
        sel,
        vld_only: bool,
        filtered_only: bool,
    ) -> tuple[dict[str, np.ndarray], int, list[str]]:
        """Return aligned, filtered, display-thinned values for dimensions.

        The one shared path serves every 2-D projection, 3-D XYZ, and the C
        values. It retains the original last-valid/raw-iteration semantics.
        """
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

        if n > _MAX_DISPLAY_POINTS:
            step = int(np.ceil(n / _MAX_DISPLAY_POINTS))
            values = {
                dimension: value[::step]
                for dimension, value in values.items()
            }
        return values, n, missing

    @staticmethod
    def _linear_color_bins(
        values: np.ndarray,
        levels: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Linearly map finite values over their full min/max range to 0..255."""
        values = np.asarray(values, dtype=float).ravel()
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros(values.size, dtype=np.uint8), 0.0, 1.0
        data_lo, data_hi = float(np.min(finite)), float(np.max(finite))
        lo, hi = levels or (data_lo, data_hi)
        lo, hi = float(lo), float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = data_lo, data_hi
        if hi <= lo:
            bins = np.full(values.size, 128, dtype=np.uint8)
            return bins, lo, hi
        normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        return np.rint(normalized * 255.0).astype(np.uint8), lo, hi

    def _mapped_colors(self, values: np.ndarray) -> tuple[list, np.ndarray, float, float]:
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
        brushes = [pg.mkBrush(*(int(channel) for channel in rgba)) for rgba in selected]
        return brushes, selected.astype(np.float32) / 255.0, lo, hi

    @staticmethod
    def _normalize_3d_positions(positions: np.ndarray) -> np.ndarray:
        """Scale each attribute axis independently into a centred unit cube."""
        positions = np.asarray(positions, dtype=float)
        out = np.zeros_like(positions, dtype=float)
        for axis in range(3):
            values = positions[:, axis]
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                out[:, axis] = np.nan
                continue
            lo, hi = float(np.min(finite)), float(np.max(finite))
            if hi <= lo:
                out[:, axis] = 0.0
            else:
                out[:, axis] = (values - lo) / (hi - lo) - 0.5
        return out

    def _draw_2d_series(
        self,
        records: list[dict],
        dimensions: tuple[str, str],
        *,
        use_lines: bool,
        stacked: bool,
    ) -> tuple[float | None, float | None]:
        self._stack.setCurrentWidget(self._plot)
        self._clear_series()
        self._clear_gl_series()
        if stacked:
            self._legend = self._plot.addLegend(offset=(-10, 10))
        color_lo = color_hi = None
        for record in records:
            values = record["values"]
            brushes = None
            if "C" in values:
                brushes, _rgba, color_lo, color_hi = self._mapped_colors(values["C"])
            self._add_series(
                values[dimensions[0]],
                values[dimensions[1]],
                record["color"],
                use_lines=use_lines,
                name=record["name"],
                brushes=brushes,
            )
        self._plot.setLabel("bottom", self._dimension_attrs[dimensions[0]])
        self._plot.setLabel("left", self._dimension_attrs[dimensions[1]])
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
        self._update_3d_axis_items(combined)
        normalized = self._normalize_3d_positions(combined)
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
                _brushes, rgba, color_lo, color_hi = self._mapped_colors(
                    record["values"]["C"]
                )
                rgba = rgba[finite]
            else:
                base = np.asarray(
                    (*record["color"][:3], self._point_alpha), dtype=float
                )
                rgba = np.tile(base / 255.0, (pos.shape[0], 1)).astype(np.float32)
            scatter = self._gl_module.GLScatterPlotItem(
                pos=pos, color=rgba, size=float(self._point_size), pxMode=True
            )
            self._apply_3d_blend(scatter)
            self._3d_view.addItem(scatter)
            self._gl_series_items.append(scatter)
            if use_lines and pos.shape[0] > 1:
                line = self._gl_module.GLLinePlotItem(
                    pos=pos, color=rgba, width=1.0, mode="line_strip"
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
            "colorbar_show_values": self._colorbar_show_values,
            "colorbar_orientation": self._colorbar_orientation,
            "colorbar_geometry": self._colorbar_geometry,
            "point_symbol": self._point_symbol,
            "point_size": self._point_size,
            "point_alpha": self._point_alpha,
            "point_color": list(self._point_color),
            "plot_style_custom": self._plot_style_custom,
            "lines": use_lines,
            "filtered_only": filtered_only,
            "iter": self._iter_combo.currentText() or "",
            "valid_only": vld_only,
        }

    def _draw(self) -> None:
        ds = self._dataset()
        if ds is None or not self._numeric_attrs:
            return

        itr_sel, render = self._selection()
        if self._dimension_count >= 4 and render == "stacked":
            render = "flatten"
        vld_only = self._valid_chk.isChecked()
        filtered_only = self._filter_chk.isChecked()
        use_lines = self._lines_chk.isChecked()
        dimensions = self._spatial_dimensions()
        data_dimensions = (
            (*dimensions, "C") if self._dimension_count >= 4 else dimensions
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
                    ds, data_dimensions, iteration, vld_only, filtered_only
                )
                total += n
                missing = miss or missing
                if values:
                    records.append({
                        "values": values,
                        "color": _iter_color(self._state.prefs, iteration),
                        "name": ordinal(iteration + 1),
                    })
            note = f"{total:,} points across {n_itr} iterations  |  all [stacked]"
        else:
            selector = "all" if render == "flatten" else itr_sel
            values, total, missing = self._series_data(
                ds, data_dimensions, selector, vld_only, filtered_only
            )
            if values:
                records.append({
                    "values": values,
                    "color": self._point_color,
                    "name": None,
                })
            note = f"{total:,} points  |  {self._iter_combo.currentText() or 'last'}"

        if self._dimension_count >= 4 and records:
            self._last_color_values = np.concatenate([
                np.asarray(record["values"]["C"], dtype=float).ravel()
                for record in records
                if "C" in record["values"]
            ])
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

        requested_names = [self._dimension_attrs[dim] for dim in data_dimensions]
        if missing:
            axis_missing = [name for name in missing if name in requested_names]
            filter_missing = [name for name in missing if name not in requested_names]
            if axis_missing:
                note += f"  |  {', '.join(dict.fromkeys(axis_missing))} has no per-iteration values"
            if filter_missing:
                note += f"  |  filter on {', '.join(dict.fromkeys(filter_missing))} not applied"
        if self._dimension_count >= 4:
            note += f"  |  C={self._dimension_attrs['C']} [{self._c_mapping}]"
            if color_lo is not None and color_hi is not None:
                note += f" {color_lo:g}..{color_hi:g}"
        if not vld_only:
            note += "  |  incl. invalid"
        self._info.setText(note)
        self._update_colorbar(color_lo, color_hi)

    def open_lut_dialog(self) -> None:
        """Open the shared LUT editor for the fourth (C) dimension."""
        if self._dimension_count < 4:
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
        if dialog is None or self._dimension_count < 4:
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
