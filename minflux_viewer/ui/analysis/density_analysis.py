"""Result dialogs and launchers for numerical analysis."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from minflux_viewer.analysis import local_density as _calculation
from minflux_viewer.colormaps import BUILTIN_COLORMAP_NAMES, make_colormap, named_colormap_names
from minflux_viewer.ui.plots.plot_format import apply_spatial_y_direction


def run_local_density(parent, state) -> None:
    """Ask for settings, compute local density, and show the density image."""
    ds = state.active_dataset
    if ds is None:
        return

    # The dialog seeds its fields from the Preferences (a sensible starting
    # point), but the plugin is for ad-hoc exploration at different radii — it
    # must NOT write its choices back. The `data.local_density_*` preferences are
    # user-owned and drive the on-load `den` derived attribute; they change only
    # via the Preferences dialog. (Previously each run overwrote + saved them.)
    dialog = LocalDensityOptionsDialog(ds, state.prefs, parent=parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return
    opts = dialog.options()

    try:
        density, method, detail, image, edges, pts_valid, dens_valid = _calculation.compute_local_density(ds, **opts)
    except Exception as exc:
        state.log(f"Local density failed: {exc}", "ERROR")
        QMessageBox.critical(parent, "Local Density", str(exc))
        return

    ds.attr["den"] = density
    ds.attr["local_density"] = density
    ds.derived["den"] = density
    ds.derived["local_density"] = density
    for name in ("den", "local_density"):
        if name not in ds.prop.attr_names:
            ds.prop.attr_names.append(name)

    finite = density[np.isfinite(density)]
    shape = "ball" if int(opts["dimensions"]) == 3 else "disk"
    msg = (
        f"Local density of '{ds.name}': neighbours within a {opts['radius_nm']:g} nm "
        f"{shape} ({method}, {detail}). "
        f"median={np.nanmedian(finite):.3g}, mean={np.nanmean(finite):.3g} "
        f"over {finite.size:,} localizations."
        if finite.size
        else f"Computed local density for '{ds.name}', but no finite values were produced."
    )
    state.log(msg)
    try:
        state.journal.add("analysis", msg)
    except Exception:
        pass
    win = LocalDensityImageWindow(
        image,
        edges,
        title=f"Local Density — {ds.name}",
        info=msg,
        points_nm=pts_valid,
        density_values=dens_valid,
        dimensions=int(opts["dimensions"]),
        prefs=state.prefs,
        parent=None,
    )
    # Modeless + non-owned, but registered on the owner so it is closed with the
    # main window on shutdown (close_modeless in _close_all_child_windows).
    from minflux_viewer.ui.common.modeless import show_modeless
    show_modeless(win, parent)


class LocalDensityOptionsDialog(QDialog):
    """Small ordered options dialog for Analyze > Local Density."""

    def __init__(self, ds, prefs: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Local Density")
        self.setModal(True)
        self.resize(360, 180)
        data_prefs = prefs.get("data", {}) if prefs else {}

        root = QVBoxLayout(self)
        form = QFormLayout()

        self._dimensions = QComboBox()
        self._dimensions.addItem("2D (XY)", 2)
        self._dimensions.addItem("3D (XYZ)", 3)
        default_dim = 3 if getattr(ds.prop, "num_dim", 2) == 3 else 2
        idx = self._dimensions.findData(int(data_prefs.get("local_density_dimensions", default_dim)))
        self._dimensions.setCurrentIndex(max(idx, 0))
        self._dimensions.currentIndexChanged.connect(self._update_method_state)
        form.addRow("Compute density in", self._dimensions)

        self._radius = QDoubleSpinBox()
        self._radius.setRange(0.1, 1_000_000.0)
        self._radius.setDecimals(2)
        self._radius.setValue(float(data_prefs.get("local_density_radius", 100.0)))
        self._radius.setSuffix(" nm")
        form.addRow("Local density radius", self._radius)

        self._method = QComboBox()
        self._method.addItem("KD-tree ball — exact, spherical (recommended)", "kdtree")
        self._method.addItem("Voxel ball radius count — fast, spherical", "voxel_radius")
        self._method.addItem("Voxel histogram — fastest, cubic voxel", "voxel_histogram")
        idx = self._method.findData(data_prefs.get("local_density_method", "kdtree"))
        self._method.setCurrentIndex(max(idx, 0))
        form.addRow("Method", self._method)

        root.addLayout(form)
        self._hint = QLabel("")
        self._hint.setStyleSheet("color: gray;")
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._update_method_state()

    def options(self) -> dict:
        return {
            "dimensions": int(self._dimensions.currentData()),
            "radius_nm": float(self._radius.value()),
            "method": str(self._method.currentData()),
            "voxel_size_nm": float(self._radius.value()),
            "smooth_sigma": 1.0,
        }

    def _update_method_state(self) -> None:
        shape = "ball (sphere)" if int(self._dimensions.currentData()) == 3 else "disk (circle)"
        self._hint.setText(
            f"Counts each localization's neighbours inside a {shape} of the given "
            "radius. KD-tree is exact + parallelised (recommended); voxel ball is a "
            "fast spherical approximation; voxel histogram is fastest but uses a "
            "cubic voxel (coarser). The result window shows the XY density heatmap "
            "(plus a 3-D scatter colored by density for 3-D data)."
        )


class LocalDensityImageWindow(QWidget):
    """Local-density result viewer: a 2-D XY heatmap and (for 3-D data) a 3-D
    scatter of the localizations colored by density, in the app's viewer style."""

    TAG = "local_density_image"
    _CMAPS = list(BUILTIN_COLORMAP_NAMES)
    _MAX_3D_POINTS = 400_000

    def __init__(
        self,
        image: np.ndarray,
        edges: tuple[np.ndarray, np.ndarray],
        *,
        title: str,
        info: str,
        points_nm: np.ndarray | None = None,
        density_values: np.ndarray | None = None,
        dimensions: int = 2,
        prefs: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowType.Window)
        try:
            from minflux_viewer import resource_path
            self.setWindowIcon(QIcon(str(resource_path("icons", "minflux_viewer_logo.png"))))
        except Exception:
            pass
        self.resize(820, 720)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._cmap = "hot"
        self._prefs = prefs or {}
        self._img = np.asarray(image, dtype=float)
        if self._img.size == 0:
            self._img = np.zeros((1, 1), dtype=float)
        self._edges = edges
        self._pts3d, self._dens3d, self._anchor = self._prepare_3d(points_nm, density_values, dimensions)
        self._gl_view = None
        self._gl_scatter = None

        root = QVBoxLayout(self)
        # View selector — a 3-D scatter is offered only when the data is 3-D.
        top = QHBoxLayout()
        self._view_label = QLabel("View:")
        top.addWidget(self._view_label)
        self._view_combo = QComboBox()
        self._view_combo.addItem("2D heatmap", "2d")
        if self._pts3d is not None:
            self._view_combo.addItem("3D scatter (colored by density)", "3d")
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        top.addWidget(self._view_combo)
        top.addSpacing(12)
        top.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        for name in named_colormap_names():
            self._cmap_combo.addItem(name)
        self._cmap_combo.setCurrentText(self._cmap)
        self._cmap_combo.currentTextChanged.connect(self._set_cmap)
        top.addWidget(self._cmap_combo)
        top.addStretch(1)
        root.addLayout(top)
        # hide the view selector (leave the colormap) when there is no 3-D data
        self._view_label.setVisible(self._pts3d is not None)
        self._view_combo.setVisible(self._pts3d is not None)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_2d_widget())        # index 0
        root.addWidget(self._stack, stretch=1)

        label = QLabel(info)
        label.setWordWrap(True)
        label.setStyleSheet("color: gray;")
        root.addWidget(label)

        self._apply_image(fit=True)

    # -- 3-D data prep -------------------------------------------------------
    def _prepare_3d(self, points_nm, density_values, dimensions):
        """(pts (M,3), dens (M,), anchor) for the 3-D scatter, or (None, None, None)
        when there is no usable 3-D data (2-D request or flat Z)."""
        if points_nm is None or density_values is None or int(dimensions) != 3:
            return None, None, None
        pts = np.asarray(points_nm, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
            return None, None, None
        dens = np.asarray(density_values, dtype=float).ravel()
        keep = np.all(np.isfinite(pts[:, :3]), axis=1) & np.isfinite(dens)
        pts, dens = pts[keep, :3], dens[keep]
        if pts.shape[0] == 0 or float(np.ptp(pts[:, 2])) <= 0:
            return None, None, None                            # flat Z ⇒ 2-D only
        if pts.shape[0] > self._MAX_3D_POINTS:                 # decimate for GL
            step = int(np.ceil(pts.shape[0] / self._MAX_3D_POINTS))
            pts, dens = pts[::step], dens[::step]
        return pts, dens, pts.mean(axis=0)

    # -- 2-D heatmap ---------------------------------------------------------
    def _build_2d_widget(self) -> QWidget:
        self._view = pg.ImageView(view=pg.PlotItem())
        self._view.ui.roiBtn.hide()
        self._view.ui.menuBtn.hide()
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._plot = self._view.getView()
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "X", units="nm")
        self._plot.setLabel("left", "Y", units="nm")
        apply_spatial_y_direction(
            self._plot,
            vertical_coordinate="Y",
            prefs=self._prefs,
            preference_key="render_xy_origin",
        )
        try:
            self._plot.getViewBox().setMenuEnabled(False)     # use our context menu
        except Exception:
            pass
        gv = self._view.ui.graphicsView
        gv.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        gv.customContextMenuRequested.connect(self._show_context_menu)
        return self._view

    def _apply_image(self, *, fit: bool) -> None:
        # autoRange=False so we can place the image at nm coordinates (setRect)
        # and THEN fit the view to it — otherwise the view stays ranged to the
        # pixel grid and the image sits off-screen (the old "need View All" bug).
        self._view.setImage(self._img.T, autoRange=False, autoLevels=True)
        self._view.setColorMap(make_colormap(self._cmap))
        x_edges, y_edges = self._edges
        if len(x_edges) > 1 and len(y_edges) > 1:
            item = self._view.getImageItem()
            item.setRect(float(x_edges[0]), float(y_edges[0]),
                         float(x_edges[-1] - x_edges[0]), float(y_edges[-1] - y_edges[0]))
        if fit:
            self._plot.getViewBox().autoRange()

    # -- 3-D scatter ---------------------------------------------------------
    def _scatter_colors(self) -> np.ndarray:
        """RGBA (float32) per point from the density values (1–99 % robust range)."""
        d = self._dens3d
        finite = np.isfinite(d)
        if finite.any():
            lo, hi = np.percentile(d[finite], [1.0, 99.0])
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        norm[~finite] = 0.0
        cols = make_colormap(self._cmap).map(norm.astype(float), mode="float")
        return np.asarray(cols, dtype=np.float32)

    def _build_3d_widget(self) -> QWidget:
        try:
            import pyqtgraph.opengl as gl
        except Exception as exc:                               # OpenGL unavailable
            fallback = QLabel(f"3-D view unavailable: {exc}")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setStyleSheet("color: gray;")
            return fallback
        pts = (self._pts3d - self._anchor).astype(np.float32)
        view = gl.GLViewWidget()
        view.setBackgroundColor("k")
        self._gl_scatter = gl.GLScatterPlotItem(
            pos=pts, color=self._scatter_colors(), size=3.0, pxMode=True)
        view.addItem(self._gl_scatter)
        span = float(np.ptp(pts)) or 100.0
        view.setCameraPosition(distance=span * 1.4)
        grid = gl.GLGridItem()
        grid.setSize(span * 1.5, span * 1.5)
        grid.setSpacing(max(span / 10.0, 1.0), max(span / 10.0, 1.0))
        view.addItem(grid)
        try:
            view.addItem(gl.GLAxisItem(size=pg.Vector(span / 2, span / 2, span / 2)))
        except Exception:
            pass
        self._gl_view = view
        return view

    # -- view switching / colormap ------------------------------------------
    def _on_view_changed(self, *_a) -> None:
        if self._view_combo.currentData() == "3d":
            if self._stack.count() < 2:                        # lazy-build on first use
                self._stack.addWidget(self._build_3d_widget())
            self._stack.setCurrentIndex(1)
        else:
            self._stack.setCurrentIndex(0)

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        cmap_menu = menu.addMenu("Colormap")
        for name in named_colormap_names():
            act = cmap_menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(self._cmap == name)
            act.triggered.connect(lambda _=False, n=name: self._set_cmap(n))
        menu.addAction("Reset view", lambda: self._plot.getViewBox().autoRange())
        menu.exec(self._view.ui.graphicsView.mapToGlobal(pos))

    def _set_cmap(self, name: str) -> None:
        self._cmap = name
        if self._cmap_combo.currentText() != name:             # keep the combo in sync
            self._cmap_combo.blockSignals(True)
            self._cmap_combo.setCurrentText(name)
            self._cmap_combo.blockSignals(False)
        self._view.setColorMap(make_colormap(name))
        if self._gl_scatter is not None:                       # recolor the 3-D scatter
            try:
                self._gl_scatter.setData(color=self._scatter_colors())
            except Exception:
                pass

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Retire this window's pyqtgraph plots before Qt deletes them."""
        from minflux_viewer.ui.common.qt_lifecycle import dispose_plot_widgets

        dispose_plot_widgets(self)
        super().closeEvent(event)
