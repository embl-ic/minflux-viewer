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

Each point is coloured by either local density (default) or any numeric
attribute selected from the **Colour by** dropdown. The colormap matches
the render window's behaviour and falls back gracefully if a name isn't
available in pyqtgraph's built-in set.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.app_state import AppState
from ..core.attributes import plot_attribute_names
from ..core.overlay import CHANNEL_LUTS, PURE_COLOR_RGB, apply_display_transform_nm, overlay_members
from ..core.roi_selection import rectangle_mask
from .plot_format import plot_widget

# ---------------------------------------------------------------------------
# Helpers — colormap loading shared with render_window
# ---------------------------------------------------------------------------

def _load_cmap(name: str) -> pg.ColorMap:
    """Load a colormap, trying pyqtgraph → matplotlib → colorcet → CET-L3."""
    key = name.lower().replace(" ", "_")
    if key == "glasbey":
        try:
            import colorcet as cc
            colors = cc.glasbey
            rgba = np.array([
                tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
                for c in colors[:256]
            ], dtype=np.ubyte)
            return pg.ColorMap(np.linspace(0.0, 1.0, len(rgba)), rgba)
        except Exception:
            base = np.array([
                [230, 25, 75, 255], [60, 180, 75, 255], [255, 225, 25, 255],
                [0, 130, 200, 255], [245, 130, 48, 255], [145, 30, 180, 255],
                [70, 240, 240, 255], [240, 50, 230, 255], [210, 245, 60, 255],
                [250, 190, 190, 255], [0, 128, 128, 255], [230, 190, 255, 255],
            ], dtype=np.ubyte)
            return pg.ColorMap(np.linspace(0.0, 1.0, len(base)), base)
    if key == "hilo":
        rgba = np.array([[0, 0, 255, 255], [35, 35, 35, 255], [255, 255, 255, 255], [255, 0, 0, 255]], dtype=np.ubyte)
        return pg.ColorMap(np.linspace(0.0, 1.0, len(rgba)), rgba)
    if key == "parula":
        rgba = np.array([
            [53, 42, 135, 255], [15, 92, 221, 255], [18, 125, 216, 255],
            [7, 156, 207, 255], [21, 177, 180, 255], [89, 189, 140, 255],
            [165, 190, 107, 255], [225, 185, 82, 255], [252, 206, 46, 255],
            [249, 251, 14, 255],
        ], dtype=np.ubyte)
        return pg.ColorMap(np.linspace(0.0, 1.0, len(rgba)), rgba)
    # 1. pyqtgraph built-ins
    try:
        c = pg.colormap.get(name)
        if c is not None:
            return c
    except Exception:
        pass
    # 2. matplotlib
    try:
        import matplotlib as mpl
        mpl_cmap = mpl.colormaps[name].resampled(256)
        rgba = mpl_cmap(np.linspace(0.0, 1.0, 256), bytes=True)
        pos  = np.linspace(0.0, 1.0, 256)
        return pg.ColorMap(pos, rgba)
    except Exception:
        pass
    # 3. colorcet
    try:
        c = pg.colormap.get(name, source="colorcet")
        if c is not None:
            return c
    except Exception:
        pass
    # 4. final fallback
    return pg.colormap.get("CET-L3")


# ---------------------------------------------------------------------------
# ScatterWindow
# ---------------------------------------------------------------------------

_AXIS_OPTIONS = ["XY", "XZ", "YZ", "3D"]
_MAX_DISPLAY_POINTS_2D = 100_000
_MAX_DISPLAY_POINTS_3D = 150_000


class ScatterWindow(QWidget):
    """Interactive 2D / 3D scatter plot of MINFLUX localisations."""

    TAG = "scatter_window"

    def __init__(self, state: AppState, parent: QWidget | None = None, *, dataset_idx: int | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._dataset_idx = dataset_idx if dataset_idx is not None else state.active_idx
        self._3d_view = None         # built lazily on first 3D switch
        self._manual_color_levels: tuple[float, float] | None = None
        self._last_color_values: np.ndarray = np.empty(0)
        self._lut_dialog = None
        self._lut_invert = False
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
        self._last_2d_black_background = False
        self._channels: list[dict] = []
        self._channel_rows: list[tuple[QLabel, QComboBox]] = []

        self.setWindowTitle("Scatter Plot")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(720, 680)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._build_ui()
        self._refresh()

        state.filter_changed.connect(self._on_filter_changed)

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

        # Headless controls. The user-facing controls live in the right-click
        # menu, but QComboBox keeps the existing state/update code compact.
        self._cbar_combo = QComboBox(self)
        self._cbar_combo.setMinimumWidth(120)
        self._cbar_combo.currentTextChanged.connect(self._update_color)
        self._cbar_combo.hide()

        self._cmap_combo = QComboBox(self)
        self._cmap_combo.addItems(["glasbey", "jet", "HiLo", "parula", "turbo", "hot", "gray"])
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
        self._plot_2d.showGrid(x=True, y=True, alpha=0.2)
        self._scatter_2d = pg.ScatterPlotItem(
            size=2, pen=None, brush=pg.mkBrush(200, 200, 200, 180),
        )
        self._plot_2d.addItem(self._scatter_2d)

        self._cmap = _load_cmap("jet")
        self._colorbar = pg.ColorBarItem(
            colorMap=self._cmap, label="", interactive=False,
        )
        plot_item = self._plot_2d.getPlotItem()
        plot_item.layout.addItem(self._colorbar, 2, 5)

        self._stack.addWidget(self._plot_2d)
        from .roi_overlay import RoiOverlayController
        self._roi_overlay = RoiOverlayController(
            self._state.rois,
            self,
            self._plot_2d,
            self._plot_2d.getPlotItem(),
            coordinate_space="plot",
        )
        # 3D view added lazily by _ensure_3d_built()

        self._channel_area = QScrollArea()
        self._channel_area.setWidgetResizable(True)
        self._channel_area.setMaximumHeight(110)
        self._channel_widget = QWidget()
        self._channel_layout = QVBoxLayout(self._channel_widget)
        self._channel_layout.setContentsMargins(4, 4, 4, 4)
        self._channel_layout.setSpacing(2)
        self._channel_area.setWidget(self._channel_widget)
        root.addWidget(self._channel_area)

        # Status line
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info_label)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self._stack.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._stack.customContextMenuRequested.connect(self._show_context_menu)

    def _ensure_3d_built(self) -> None:
        """Construct the OpenGL widget on first 3D request."""
        if self._3d_view is not None:
            return
        try:
            import pyqtgraph.opengl as gl
        except ImportError as e:
            self._info_label.setText(
                f"3D view unavailable: PyOpenGL is not installed ({e}). "
                "Run: poetry install"
            )
            return

        view = gl.GLViewWidget()
        view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        view.customContextMenuRequested.connect(self._show_context_menu)
        view.setBackgroundColor("k" if self._black_bg_check.isChecked() else "w")

        # Reference grid in the XY plane
        grid = gl.GLGridItem()
        grid.setSize(1000, 1000)
        grid.setSpacing(100, 100)
        view.addItem(grid)

        # Light XYZ axes
        axis = gl.GLAxisItem()
        axis.setSize(500, 500, 500)
        view.addItem(axis)

        scatter = gl.GLScatterPlotItem(pxMode=True, size=2.0)
        view.addItem(scatter)

        self._3d_view    = view
        self._3d_scatter = scatter
        self._stack.addWidget(view)

    def _on_background_changed(self, black: bool) -> None:
        self._apply_background(black)
        self._update_color()

    def _background_is_black(self) -> bool:
        return self._black_bg_check.isChecked()

    def _set_black_background(self, enabled: bool) -> None:
        self._black_bg_check.setChecked(bool(enabled))

    def _apply_background(self, black: bool) -> None:
        self._plot_2d.setBackground("k" if black else "w")
        if self._3d_view is not None:
            self._3d_view.setBackgroundColor("k" if black else "w")

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)

        colour_menu = menu.addMenu("Colour by")
        for i in range(self._cbar_combo.count()):
            text = self._cbar_combo.itemText(i)
            action = colour_menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(text == self._cbar_combo.currentText())
            action.triggered.connect(lambda _checked=False, value=text: self._cbar_combo.setCurrentText(value))

        cmap_menu = menu.addMenu("Colormap")
        for i in range(self._cmap_combo.count()):
            text = self._cmap_combo.itemText(i)
            action = cmap_menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(text == self._cmap_combo.currentText())
            action.triggered.connect(lambda _checked=False, value=text: self._cmap_combo.setCurrentText(value))

        bg_action = menu.addAction("Black background")
        bg_action.setCheckable(True)
        bg_action.setChecked(self._background_is_black())
        bg_action.triggered.connect(self._set_black_background)

        axis_menu = menu.addMenu("Axis")
        for axis in _AXIS_OPTIONS:
            action = axis_menu.addAction(axis)
            action.setCheckable(True)
            action.setChecked(axis == self._axis_combo.currentText())
            action.triggered.connect(lambda _checked=False, value=axis: self._axis_combo.setCurrentText(value))

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
        """Switch between 2D and 3D mode and re-draw."""
        axis_text = self._axis_combo.currentText()
        is_3d = axis_text == "3D"
        if is_3d:
            if self._last_axis_text != "3D":
                self._last_2d_black_background = self._black_bg_check.isChecked()
            self._black_bg_check.blockSignals(True)
            self._black_bg_check.setChecked(True)
            self._black_bg_check.blockSignals(False)
            self._ensure_3d_built()
            self._apply_background(True)
            if self._3d_view is not None:
                self._stack.setCurrentWidget(self._3d_view)
                self._colorbar.setVisible(False)
        else:
            if self._last_axis_text == "3D":
                self._black_bg_check.blockSignals(True)
                self._black_bg_check.setChecked(self._last_2d_black_background)
                self._black_bg_check.blockSignals(False)
                self._apply_background(self._last_2d_black_background)
            self._stack.setCurrentWidget(self._plot_2d)
            self._colorbar.setVisible(True)
        self._last_axis_text = axis_text
        self._save_view_state()
        self._refresh()

    def _on_cmap_changed(self, name: str) -> None:
        self._cmap = _load_cmap(name)
        self._colorbar.setColorMap(self._cmap)
        self._invalidate_color_cache()
        self._update_color()

    def _update_color(self) -> None:
        self._redraw_current(save_state=True)

    def _on_filter_changed(self, idx: int) -> None:
        if idx == self._dataset_idx or any(ch.get("dataset_idx") == idx for ch in self._channels):
            self._redraw_current(save_state=False)

    def _reset_view(self) -> None:
        if self._axis_combo.currentText() == "3D" and self._3d_view is not None:
            self._reset_3d_camera()
        else:
            self._plot_2d.autoRange()

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

    def _build_channels(self) -> None:
        previous = {
            ch.get("dataset_idx"): {"visible": ch.get("visible", True), "lut": ch.get("lut")}
            for ch in self._channels
        }
        self._channels = []
        if self._dataset_idx is None:
            return
        for pos, (idx, ds) in enumerate(overlay_members(self._state, self._dataset_idx)):
            prev = previous.get(idx, {})
            self._channels.append({
                "dataset_idx": idx,
                "name": ds.name,
                "visible": bool(prev.get("visible", True)),
                "lut": prev.get("lut") or ds.state.get("overlay_lut") or ds.state.get("render_channel_lut") or CHANNEL_LUTS[pos % len(CHANNEL_LUTS)],
            })

    def _rebuild_channel_ui(self) -> None:
        while self._channel_layout.count():
            item = self._channel_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._channel_rows = []
        for ch_idx, ch in enumerate(self._channels):
            row = QWidget()
            row.mousePressEvent = lambda event, i=ch_idx, r=row: self._on_channel_row_pressed(r, event, i)
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            vis_cb = QCheckBox()
            vis_cb.setChecked(bool(ch["visible"]))
            vis_cb.toggled.connect(lambda checked, i=ch_idx: self._on_channel_visible(i, checked))
            lay.addWidget(vis_cb)
            name_lbl = QLabel(f"{ch_idx + 1}: {ch['name']}")
            lay.addWidget(name_lbl, stretch=1)
            lut = QComboBox()
            lut.addItems(CHANNEL_LUTS)
            lut.setCurrentText(str(ch["lut"]))
            lut.currentTextChanged.connect(lambda text, i=ch_idx: self._on_channel_lut(i, text))
            lay.addWidget(lut)
            self._channel_layout.addWidget(row)
            self._channel_rows.append((name_lbl, lut))
        self._channel_area.setVisible(len(self._channels) > 1)

    def _on_channel_row_pressed(self, row: QWidget, event, ch_idx: int) -> None:
        if event.button() == Qt.MouseButton.LeftButton and 0 <= ch_idx < len(self._channels):
            ds_idx = self._channels[ch_idx]["dataset_idx"]
            if 0 <= ds_idx < len(self._state.datasets):
                self._dataset_idx = ds_idx
                self._state.set_active(ds_idx)
                self._update_overlay_title()
        QWidget.mousePressEvent(row, event)

    def _on_channel_visible(self, ch_idx: int, visible: bool) -> None:
        if 0 <= ch_idx < len(self._channels):
            self._channels[ch_idx]["visible"] = bool(visible)
            self._redraw_current(save_state=False)

    def _on_channel_lut(self, ch_idx: int, lut: str) -> None:
        if 0 <= ch_idx < len(self._channels):
            self._channels[ch_idx]["lut"] = lut
            ds_idx = self._channels[ch_idx]["dataset_idx"]
            if 0 <= ds_idx < len(self._state.datasets):
                self._state.datasets[ds_idx].state["render_channel_lut"] = lut
                self._state.datasets[ds_idx].state["overlay_lut"] = lut
            self._redraw_current(save_state=False)

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
        ds = self._dataset()
        if ds is None:
            self._scatter_2d.setData([], [])
            if self._3d_view is not None:
                self._3d_scatter.setData(pos=np.empty((0, 3)))
            self.setWindowTitle("Scatter Plot")
            return

        self._build_channels()
        self._rebuild_channel_ui()
        self._update_overlay_title()
        saved = ds.state.get(self._view_state_key, {})

        self._axis_combo.blockSignals(True)
        axis_default = saved.get("axis", "XY")
        if self._axis_combo.findText(axis_default) >= 0:
            self._axis_combo.setCurrentText(axis_default)
        self._axis_combo.blockSignals(False)
        self._last_axis_text = self._axis_combo.currentText()
        if self._axis_combo.currentText() == "3D":
            self._black_bg_check.blockSignals(True)
            self._black_bg_check.setChecked(True)
            self._black_bg_check.blockSignals(False)
            self._ensure_3d_built()
            self._apply_background(True)
            if self._3d_view is not None:
                self._stack.setCurrentWidget(self._3d_view)
                self._colorbar.setVisible(False)
        else:
            self._stack.setCurrentWidget(self._plot_2d)
            self._colorbar.setVisible(True)

        self._cmap_combo.blockSignals(True)
        cmap_default = saved.get("colormap", "jet")
        if self._cmap_combo.findText(cmap_default) >= 0:
            self._cmap_combo.setCurrentText(cmap_default)
        else:
            self._cmap_combo.setCurrentText("jet")
        self._cmap_combo.blockSignals(False)
        self._cmap = _load_cmap(self._cmap_combo.currentText())
        self._colorbar.setColorMap(self._cmap)

        force_black_3d = self._axis_combo.currentText() == "3D"
        self._black_bg_check.blockSignals(True)
        self._black_bg_check.setChecked(
            True if force_black_3d else bool(saved.get("black_background", False))
        )
        self._black_bg_check.blockSignals(False)
        self._apply_background(self._black_bg_check.isChecked())

        # Populate colour-by combo (preserve selection if possible)
        old = self._cbar_combo.currentText()
        self._cbar_combo.blockSignals(True)
        self._cbar_combo.clear()
        numeric_attrs = plot_attribute_names(ds, self._state.prefs, exclude=("ftr", "idx"))
        self._cbar_combo.addItems(["(density)"] + numeric_attrs)
        color_default = saved.get("color_by", old or "tid")
        if color_default in numeric_attrs or color_default == "(density)":
            self._cbar_combo.setCurrentText(color_default)
        elif old in numeric_attrs or old == "(density)":
            self._cbar_combo.setCurrentText(old)
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
            return
        self._draw(self._current_locs(ds), ds.filter_mask, ds, save_state=save_state)

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
        return apply_display_transform_nm(
            locs,
            ds.state.get("overlay_transform") or ds.state.get("render_transform_2d"),
        )

    def _lut_color(self, lut: str, alpha: int = 190) -> tuple[int, int, int, int]:
        if lut in PURE_COLOR_RGB:
            r, g, b = PURE_COLOR_RGB[lut]
            return int(r), int(g), int(b), int(alpha)
        try:
            cmap = _load_cmap(lut)
            color = cmap.map(np.array([0.72]), mode="byte")[0]
            return int(color[0]), int(color[1]), int(color[2]), int(alpha)
        except Exception:
            return 180, 180, 180, int(alpha)

    def _draw_overlay(self, *, save_state: bool) -> None:
        ds_active = self._dataset()
        if ds_active is not None and save_state:
            self._save_view_state(ds_active)
        axis = self._axis_combo.currentText()
        col_map = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
        if axis == "3D":
            self._draw_overlay_3d()
            return
        ci, cj = col_map.get(axis, (0, 1))
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
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
            indices = self._visible_indices(mask, locs.shape[0], max(1, _MAX_DISPLAY_POINTS_2D // max(len(self._channels), 1)))
            if indices.size == 0:
                continue
            x = locs[indices, ci]
            y = locs[indices, cj]
            xs.append(x)
            ys.append(y)
            color = self._lut_color(str(ch.get("lut", "Gray")))
            brushes.extend([pg.mkBrush(*color)] * indices.size)
            total += int(np.count_nonzero(mask))
        if not xs:
            self._scatter_2d.setData([], [])
            self._info_label.setText("No localisations pass the current filters.")
            return
        self._scatter_2d.setData(x=np.concatenate(xs), y=np.concatenate(ys), brush=brushes, pen=None, size=2)
        self._plot_2d.setLabel("bottom", "XYZ"[ci] + " (nm)")
        self._plot_2d.setLabel("left", "XYZ"[cj] + " (nm)")
        self._colorbar.setVisible(False)
        self._info_label.setText(f"{total:,} filtered localisations across {len([c for c in self._channels if c.get('visible', True)])} channel(s)")

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
            r, g, b, a = self._lut_color(str(ch.get("lut", "Gray")), alpha=220)
            rgba = np.tile(np.array([[r / 255.0, g / 255.0, b / 255.0, a / 255.0]], dtype=np.float32), (indices.size, 1))
            rgba_parts.append(rgba)
            total += int(np.count_nonzero(mask))
        if not pos_parts:
            self._3d_scatter.setData(pos=np.empty((0, 3)))
            self._info_label.setText("No finite XYZ localisations pass the current filters for 3D display.")
            return
        pos = np.vstack(pos_parts).astype(np.float32, copy=False)
        rgba = np.vstack(rgba_parts).astype(np.float32, copy=False)
        self._3d_scatter.setData(pos=pos, color=rgba, size=3.0, pxMode=True)
        if not self._3d_camera_initialised:
            self._reset_3d_camera(pos)
            self._3d_camera_initialised = True
        self._info_label.setText(f"{total:,} filtered localisations across {len([c for c in self._channels if c.get('visible', True)])} channel(s)")

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
            "axis": self._axis_combo.currentText(),
        }

    # -- 2D path -----------------------------------------------------

    def _draw_2d(self, locs: np.ndarray, ftr: np.ndarray, ds) -> None:
        axis = self._axis_combo.currentText()
        col_map  = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
        ax_text  = {"XY": ("X (nm)", "Y (nm)"),
                    "XZ": ("X (nm)", "Z (nm)"),
                    "YZ": ("Y (nm)", "Z (nm)")}
        ci, cj = col_map[axis]

        indices = self._visible_indices(ftr, locs.shape[0], _MAX_DISPLAY_POINTS_2D)
        n_visible = int(np.count_nonzero(np.asarray(ftr, dtype=bool)))
        n_display = indices.size
        if n_display == 0:
            self._scatter_2d.setData([], [])
            self._info_label.setText("No localisations pass the current filter.")
            return

        x = locs[indices, ci]
        y = locs[indices, cj]
        c_vals, color_bins, c_label, vmin, vmax = self._color_bins_for_points(x, y, None, ds, indices)
        self._last_color_values = np.asarray(c_vals, dtype=float)
        brushes = self._brushes_for_bins(color_bins)

        self._scatter_2d.setData(
            x=x, y=y,
            brush=brushes,
            pen=None, size=2,
        )

        self._colorbar.setLevels((vmin, vmax))
        self._colorbar.getAxis("right").setLabel(c_label)

        ax_x, ax_y = ax_text[axis]
        self._plot_2d.setLabel("bottom", ax_x)
        self._plot_2d.setLabel("left", ax_y)

        display_note = (
            f"showing {n_display:,} / {n_visible:,} passing"
            if n_display < n_visible
            else f"{n_visible:,}"
        )
        self._info_label.setText(
            f"{display_note} / {ds.prop.num_loc:,} localisations  "
            f"({100*n_visible/ds.prop.num_loc:.1f} %)  |  axis: {axis}  |  "
            f"colour: {c_label}"
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
            self._info_label.setText("No localisations pass the current filter.")
            return

        pos = np.asarray(locs[indices, :3], dtype=float)
        finite_mask = np.all(np.isfinite(pos), axis=1)
        if not np.all(finite_mask):
            indices = indices[finite_mask]
            pos = pos[finite_mask]
        n_display = indices.size
        if n_display == 0:
            self._3d_scatter.setData(pos=np.empty((0, 3)))
            self._info_label.setText(
                "No finite XYZ localisations pass the current filter for 3D display."
            )
            return

        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        c_vals, color_bins, c_label, vmin, vmax = self._color_bins_for_points(x, y, z, ds, indices)
        self._last_color_values = np.asarray(c_vals, dtype=float)
        rgba = self._rgba_for_bins(color_bins, for_3d=True)

        pos = pos.astype(np.float32, copy=False)
        point_size = 4.0 if not self._background_is_black() else 3.0
        self._3d_scatter.setData(pos=pos, color=rgba, size=point_size, pxMode=True)

        # First-time camera setup
        if not self._3d_camera_initialised:
            self._reset_3d_camera(pos)
            self._3d_camera_initialised = True

        display_note = (
            f"showing {n_display:,} / {n_visible:,} passing"
            if n_display < n_visible
            else f"{n_visible:,}"
        )
        self._info_label.setText(
            f"{display_note} / {ds.prop.num_loc:,} localisations  "
            f"({100*n_visible/ds.prop.num_loc:.1f} %)  |  axis: 3D  |  "
            f"colour: {c_label} ∈ [{vmin:.3g}, {vmax:.3g}]"
        )

    @staticmethod
    def _visible_indices(ftr: np.ndarray, total: int, max_points: int) -> np.ndarray:
        mask = np.asarray(ftr, dtype=bool).ravel()
        if mask.size != total:
            mask = np.ones(total, dtype=bool)
        indices = np.flatnonzero(mask)
        if indices.size > max_points:
            step = int(np.ceil(indices.size / max_points))
            indices = indices[::step]
        return indices

    # -- shared colour helpers --------------------------------------

    def _color_bins_for_points(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray | None,
        ds, indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str, float, float]:
        """Return values, uint8 color bins, label, and display levels."""
        c_name = self._cbar_combo.currentText()
        if c_name == "(density)" or c_name not in ds.attr:
            if z is None:
                from ..utils.filters import fast_density_2d
                values = fast_density_2d(x, y)
            else:
                from ..utils.filters import fast_density_3d
                values = fast_density_3d(x, y, z)
            bins, vmin, vmax = self._map_values_to_bins(values)
            return values, bins, "density", vmin, vmax

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
            self._dataset_idx,
            c_name,
            self._cmap_combo.currentText(),
            self._lut_invert,
            self._manual_color_levels,
            id(self._cmap),
            ds.prop.num_loc,
        )
        if self._color_cache_key == key:
            return self._color_cache

        values = np.asarray(ds.attr.get(c_name, np.empty(0))).ravel().astype(float)
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
        key = (self._cmap_combo.currentText(), self._lut_invert, id(self._cmap))
        if self._brush_lut_key == key and self._brush_lut is not None and self._rgba_lut is not None:
            return
        qcolors = self._cmap.mapToQColor(np.linspace(0.0, 1.0, 256))
        self._brush_lut = [pg.mkBrush(c) for c in qcolors]
        self._rgba_lut = np.asarray(
            [[c.redF(), c.greenF(), c.blueF(), c.alphaF()] for c in qcolors],
            dtype=np.float32,
        )
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
            # LUT colours blend over a white clear colour. Keep 2D colours exact,
            # but darken only the too-bright 3D colours on white backgrounds.
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
                rgba[:, 3] = 1.0
            else:
                rgba[:, 3] = np.maximum(rgba[:, 3], 0.9)
        return rgba

    def _invalidate_color_cache(self) -> None:
        self._color_cache_key = None
        self._color_cache = None

    def open_lut_dialog(self) -> None:
        vals = np.asarray(self._last_color_values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            self._refresh()
            vals = np.asarray(self._last_color_values, dtype=float)
            vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            self._info_label.setText("LUT unavailable: no colour values to display.")
            return

        from .lut_dialog import LutDialog

        if self._lut_dialog is None:
            self._lut_dialog = LutDialog(
                on_levels_changed=self._on_lut_levels_changed,
                on_cmap_changed=self._on_lut_cmap_changed,
                on_invert_changed=self._on_lut_invert_changed,
                parent=self,
            )

        data_lo = float(np.nanmin(vals))
        data_hi = float(np.nanmax(vals))
        if data_hi <= data_lo:
            data_hi = data_lo + 1.0
        lo, hi = self._manual_color_levels or tuple(np.nanpercentile(vals, [1, 99]))
        self._lut_dialog.load_image(
            pixels=vals,
            data_lo=data_lo,
            data_hi=data_hi,
            lo=float(lo),
            hi=float(hi),
            cmap_name=self._cmap_combo.currentText(),
            invert=self._lut_invert,
        )
        self._lut_dialog.show()
        self._lut_dialog.raise_()
        self._lut_dialog.activateWindow()

    def compute_roi_selection(self, record):
        if record.type != "rectangle" or self._axis_combo.currentText() == "3D":
            return None
        ds = self._dataset()
        if ds is None:
            return None
        locs = self._current_locs(ds)
        if locs.ndim != 2 or locs.shape[1] < 2:
            return None
        if locs.shape[1] == 2:
            locs = np.column_stack([locs, np.zeros(locs.shape[0], dtype=float)])

        axis = self._axis_combo.currentText()
        col_map = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
        if axis not in col_map:
            return None
        ci, cj = col_map[axis]
        base = np.asarray(ds.filter_mask, dtype=bool)
        if base.shape[0] != locs.shape[0]:
            base = np.ones(locs.shape[0], dtype=bool)
        base &= np.all(np.isfinite(locs[:, :3]), axis=1)
        mask = rectangle_mask(locs[:, ci], locs[:, cj], record, base_mask=base)
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
        from .lut_dialog import make_colormap
        self._lut_invert = bool(invert)
        idx = self._cmap_combo.findText(name)
        if idx >= 0:
            self._cmap_combo.blockSignals(True)
            self._cmap_combo.setCurrentIndex(idx)
            self._cmap_combo.blockSignals(False)
        self._cmap = make_colormap(name, invert=self._lut_invert)
        self._colorbar.setColorMap(self._cmap)
        self._invalidate_color_cache()
        self._update_color()

    def _on_lut_invert_changed(self, invert: bool) -> None:
        self._on_lut_cmap_changed(self._cmap_combo.currentText(), invert)

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
