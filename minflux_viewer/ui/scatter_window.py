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
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.app_state import AppState


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


class ScatterWindow(QWidget):
    """Interactive 2D / 3D scatter plot of MINFLUX localisations."""

    TAG = "scatter_window"

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._3d_view = None         # built lazily on first 3D switch
        self._manual_color_levels: tuple[float, float] | None = None
        self._last_color_values: np.ndarray = np.empty(0)
        self._lut_dialog = None
        self._lut_invert = False
        self._roi_overlay = None

        self.setWindowTitle("Scatter Plot")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(720, 680)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._build_ui()
        self._refresh()

        state.filter_changed.connect(self._on_filter_changed)
        state.active_changed.connect(self._on_active_changed)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Toolbar row
        bar = QHBoxLayout()
        bar.setSpacing(8)

        bar.addWidget(QLabel("Colour by:"))
        self._cbar_combo = QComboBox()
        self._cbar_combo.setMinimumWidth(120)
        self._cbar_combo.currentTextChanged.connect(self._update_color)
        bar.addWidget(self._cbar_combo)

        bar.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["glasbey", "jet", "HiLo", "parula", "turbo", "hot", "gray"])
        self._cmap_combo.setCurrentText("glasbey")
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        bar.addWidget(self._cmap_combo)

        self._white_bg_check = QCheckBox("white background")
        self._white_bg_check.toggled.connect(self._on_background_changed)
        bar.addWidget(self._white_bg_check)

        bar.addWidget(QLabel("Axis:"))
        self._axis_combo = QComboBox()
        self._axis_combo.addItems(_AXIS_OPTIONS)
        self._axis_combo.currentTextChanged.connect(self._on_axis_changed)
        bar.addWidget(self._axis_combo)

        self._reset_btn = QPushButton("Reset view")
        self._reset_btn.clicked.connect(self._reset_view)
        bar.addWidget(self._reset_btn)

        bar.addStretch()
        root.addLayout(bar)

        # Stacked widget — switches between 2D plot and 3D GL view
        self._stack = QStackedWidget()
        self._stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self._stack, stretch=1)

        # ── 2D plot widget ─────────────────────────────────────────
        pg.setConfigOptions(antialias=False)
        self._plot_2d = pg.PlotWidget(background="k")
        self._plot_2d.setAspectLocked(True)
        self._plot_2d.showGrid(x=True, y=True, alpha=0.2)
        self._scatter_2d = pg.ScatterPlotItem(
            size=2, pen=None, brush=pg.mkBrush(200, 200, 200, 180),
        )
        self._plot_2d.addItem(self._scatter_2d)

        self._cmap = _load_cmap("glasbey")
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

        # Status line
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info_label)

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
        view.setBackgroundColor("w" if self._white_bg_check.isChecked() else "k")

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

    def _on_background_changed(self, white: bool) -> None:
        self._plot_2d.setBackground("w" if white else "k")
        if self._3d_view is not None:
            self._3d_view.setBackgroundColor("w" if white else "k")
        self._update_color()

    # ------------------------------------------------------------------
    # Slots & lifecycle
    # ------------------------------------------------------------------

    def _on_axis_changed(self, _text: str) -> None:
        """Switch between 2D and 3D mode and re-draw."""
        is_3d = self._axis_combo.currentText() == "3D"
        if is_3d:
            self._ensure_3d_built()
            if self._3d_view is not None:
                self._stack.setCurrentWidget(self._3d_view)
                self._colorbar.setVisible(False)
        else:
            self._stack.setCurrentWidget(self._plot_2d)
            self._colorbar.setVisible(True)
        self._refresh()

    def _on_cmap_changed(self, name: str) -> None:
        self._cmap = _load_cmap(name)
        self._colorbar.setColorMap(self._cmap)
        self._update_color()

    def _update_color(self) -> None:
        ds = self._state.active_dataset
        if ds is None:
            return
        self._draw(ds.loc_nm, ds.filter_mask, ds)

    def _on_filter_changed(self, idx: int) -> None:
        if idx == self._state.active_idx:
            self._refresh()

    def _on_active_changed(self, _idx: int) -> None:
        self._refresh()

    def _reset_view(self) -> None:
        if self._axis_combo.currentText() == "3D" and self._3d_view is not None:
            self._reset_3d_camera()
        else:
            self._plot_2d.autoRange()

    def _reset_3d_camera(self) -> None:
        """Centre the 3D camera on the data."""
        ds = self._state.active_dataset
        if ds is None or self._3d_view is None:
            return
        locs = ds.loc_nm[ds.filter_mask]
        if locs.shape[0] == 0:
            return
        centre = locs.mean(axis=0)
        extent = float(np.linalg.norm(locs.max(axis=0) - locs.min(axis=0)))
        self._3d_view.opts["center"] = pg.Vector(*centre)
        self._3d_view.setCameraPosition(distance=max(extent * 1.5, 100.0))

    # ------------------------------------------------------------------
    # Drawing — dispatches to 2D or 3D path
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        ds = self._state.active_dataset
        if ds is None:
            self._scatter_2d.setData([], [])
            if self._3d_view is not None:
                self._3d_scatter.setData(pos=np.empty((0, 3)))
            self.setWindowTitle("Scatter Plot")
            return

        self.setWindowTitle(f"Scatter Plot  —  {ds.name}")

        # Populate colour-by combo (preserve selection if possible)
        old = self._cbar_combo.currentText()
        self._cbar_combo.blockSignals(True)
        self._cbar_combo.clear()
        numeric_attrs = [
            k for k in ds.prop.attr_names
            if k not in ("ftr", "idx")
            and np.issubdtype(np.asarray(ds.attr[k]).dtype, np.number)
            and ds.attr[k].ndim == 1
        ]
        self._cbar_combo.addItems(["(density)"] + numeric_attrs)
        if old in numeric_attrs or old == "(density)":
            self._cbar_combo.setCurrentText(old)
        self._cbar_combo.blockSignals(False)

        self._draw(ds.loc_nm, ds.filter_mask, ds)

    def _draw(self, locs: np.ndarray, ftr: np.ndarray, ds) -> None:
        is_3d = self._axis_combo.currentText() == "3D"
        if is_3d:
            self._draw_3d(locs, ftr, ds)
        else:
            self._draw_2d(locs, ftr, ds)

    # -- 2D path -----------------------------------------------------

    def _draw_2d(self, locs: np.ndarray, ftr: np.ndarray, ds) -> None:
        axis = self._axis_combo.currentText()
        col_map  = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
        ax_text  = {"XY": ("X (nm)", "Y (nm)"),
                    "XZ": ("X (nm)", "Z (nm)"),
                    "YZ": ("Y (nm)", "Z (nm)")}
        ci, cj = col_map[axis]

        x = locs[ftr, ci]
        y = locs[ftr, cj]
        n = x.size
        if n == 0:
            self._scatter_2d.setData([], [])
            self._info_label.setText("No localisations pass the current filter.")
            return

        c_vals, c_label = self._color_values(x, y, None, ds, ftr)
        self._last_color_values = np.asarray(c_vals, dtype=float)
        rgba, vmin, vmax = self._map_colors(c_vals)

        self._scatter_2d.setData(
            x=x, y=y,
            brush=[pg.mkBrush(c) for c in rgba],
            pen=None, size=2,
        )

        self._colorbar.setLevels((vmin, vmax))
        self._colorbar.getAxis("right").setLabel(c_label)

        ax_x, ax_y = ax_text[axis]
        self._plot_2d.setLabel("bottom", ax_x)
        self._plot_2d.setLabel("left", ax_y)

        self._info_label.setText(
            f"{n:,} / {ds.prop.num_loc:,} localisations  "
            f"({100*n/ds.prop.num_loc:.1f} %)  |  axis: {axis}  |  "
            f"colour: {c_label}"
        )

    # -- 3D path -----------------------------------------------------

    def _draw_3d(self, locs: np.ndarray, ftr: np.ndarray, ds) -> None:
        if self._3d_view is None:
            return  # PyOpenGL not installed

        x, y, z = locs[ftr, 0], locs[ftr, 1], locs[ftr, 2]
        n = x.size
        if n == 0:
            self._3d_scatter.setData(pos=np.empty((0, 3)))
            self._info_label.setText("No localisations pass the current filter.")
            return

        c_vals, c_label = self._color_values(x, y, z, ds, ftr)
        self._last_color_values = np.asarray(c_vals, dtype=float)
        rgba_qcolor, vmin, vmax = self._map_colors(c_vals)
        # GLScatterPlotItem needs (N, 4) float RGBA in [0, 1]
        rgba = np.empty((n, 4), dtype=np.float32)
        for i, qc in enumerate(rgba_qcolor):
            rgba[i, 0] = qc.redF()
            rgba[i, 1] = qc.greenF()
            rgba[i, 2] = qc.blueF()
            rgba[i, 3] = qc.alphaF()

        pos = np.column_stack([x, y, z]).astype(np.float32)
        self._3d_scatter.setData(pos=pos, color=rgba, size=2.0)

        # First-time camera setup
        if not getattr(self, "_3d_camera_initialised", False):
            self._reset_3d_camera()
            self._3d_camera_initialised = True

        self._info_label.setText(
            f"{n:,} / {ds.prop.num_loc:,} localisations  "
            f"({100*n/ds.prop.num_loc:.1f} %)  |  axis: 3D  |  "
            f"colour: {c_label} ∈ [{vmin:.3g}, {vmax:.3g}]"
        )

    # -- shared colour helpers --------------------------------------

    def _color_values(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray | None,
        ds, ftr: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """Return (per-point colour value, human label)."""
        c_name = self._cbar_combo.currentText()
        if c_name == "(density)" or c_name not in ds.attr:
            if z is None:
                from ..utils.filters import fast_density_2d
                return fast_density_2d(x, y), "density"
            from ..utils.filters import fast_density_3d
            return fast_density_3d(x, y, z), "density"
        return np.asarray(ds.attr[c_name])[ftr].astype(float), c_name

    def _map_colors(
        self, c_vals: np.ndarray
    ) -> tuple[list, float, float]:
        """Robust normalise + colormap → list of QColor."""
        if self._manual_color_levels is not None:
            vmin, vmax = self._manual_color_levels
        else:
            vmin, vmax = np.nanpercentile(c_vals, [1, 99])
        if vmax <= vmin:
            vmax = vmin + 1.0
        normed = np.clip((c_vals - vmin) / (vmax - vmin), 0, 1)
        rgba = self._cmap.mapToQColor(normed)
        return rgba, float(vmin), float(vmax)

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

    def _on_lut_levels_changed(self, lo: float, hi: float) -> None:
        self._manual_color_levels = (float(lo), float(hi))
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
        self._update_color()

    def _on_lut_invert_changed(self, invert: bool) -> None:
        self._on_lut_cmap_changed(self._cmap_combo.currentText(), invert)

    def focusInEvent(self, event) -> None:
        if self._roi_overlay is not None and self._axis_combo.currentText() != "3D":
            self._roi_overlay.activate()
        super().focusInEvent(event)

    def changeEvent(self, event) -> None:
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            if self._roi_overlay is not None and self._axis_combo.currentText() != "3D":
                self._roi_overlay.activate()
        super().changeEvent(event)
