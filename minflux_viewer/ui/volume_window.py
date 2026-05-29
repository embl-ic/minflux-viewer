"""
minflux_viewer.ui.volume_window
===============================
Experimental OpenGL volume preview for filtered 3-D localisation data.

This window intentionally lives beside the 2-D render window.  It voxelises
the currently filtered localisations into a bounded RGBA volume and displays
that volume with pyqtgraph's GLVolumeItem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import gaussian_filter

from .. import resource_path
from ..core.app_state import AppState

_DEFAULT_MAX_VOXELS = 8_000_000


@dataclass(frozen=True)
class VolumePayload:
    """Voxelised RGBA volume plus physical placement metadata."""

    scalar: np.ndarray
    norm: np.ndarray
    rgba: np.ndarray
    origin_nm: tuple[float, float, float]
    voxel_nm: tuple[float, float, float]
    counts: tuple[int, int, int]
    n_locs: int
    intensity_max: float


def _finite_filtered_locs(dataset) -> np.ndarray:
    """Return current filtered finite localisation coordinates in nm."""
    try:
        locs = np.asarray(dataset.loc_nm, dtype=np.float64)
    except Exception:
        return np.empty((0, 3), dtype=np.float64)
    if locs.ndim != 2 or locs.shape[1] < 2:
        return np.empty((0, 3), dtype=np.float64)
    if locs.shape[1] == 2:
        locs = np.column_stack([locs, np.zeros(locs.shape[0], dtype=np.float64)])
    mask = np.asarray(dataset.filter_mask, dtype=bool)
    if mask.shape[0] == locs.shape[0]:
        locs = locs[mask]
    finite = np.all(np.isfinite(locs[:, :3]), axis=1)
    return np.ascontiguousarray(locs[finite, :3], dtype=np.float64)


def _matplotlib_rgba(name: str, values: np.ndarray) -> np.ndarray:
    """Map normalised values to uint8 RGBA through matplotlib."""
    import matplotlib as mpl

    cmap = mpl.colormaps.get_cmap(name)
    return np.asarray(cmap(values, bytes=True), dtype=np.uint8)


def _surface_color(cmap_name: str, value: float, opacity: float) -> tuple[float, float, float, float]:
    rgba = _matplotlib_rgba(cmap_name, np.array([np.clip(value, 0.0, 1.0)], dtype=float))[0]
    return (
        float(rgba[0]) / 255.0,
        float(rgba[1]) / 255.0,
        float(rgba[2]) / 255.0,
        float(np.clip(opacity, 0.05, 1.0)),
    )


def make_volume_payload(
    locs_nm: np.ndarray,
    *,
    xy_voxel_nm: float | None = None,
    z_voxel_nm: float | None = None,
    voxel_nm: float | None = None,
    max_dim: int,
    max_voxels: int = 4_000_000,
    cmap_name: str = "hot",
    opacity: float = 0.45,
    sigma_nm_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> VolumePayload:
    """Voxelise localisation coordinates into a bounded RGBA volume."""
    locs = np.asarray(locs_nm, dtype=np.float64)
    if locs.ndim != 2 or locs.shape[1] != 3:
        raise ValueError("locs_nm must have shape (N, 3)")
    if locs.shape[0] == 0:
        raise ValueError("No localisations pass the current filter.")

    lo = np.nanmin(locs, axis=0)
    hi = np.nanmax(locs, axis=0)
    span = hi - lo
    if float(span[2]) <= 1.0:
        raise ValueError("The active dataset does not have a usable 3-D Z range.")
    pad = np.maximum(span * 0.02, 0.5)
    lo = lo - pad
    hi = hi + pad
    span = np.maximum(hi - lo, 1.0)

    if xy_voxel_nm is None:
        xy_voxel_nm = voxel_nm
    if z_voxel_nm is None:
        z_voxel_nm = voxel_nm
    xy_voxel = max(float(xy_voxel_nm if xy_voxel_nm is not None else 1.0), 0.001)
    z_voxel = max(float(z_voxel_nm if z_voxel_nm is not None else xy_voxel), 0.001)
    voxel_xyz_requested = np.array([xy_voxel, xy_voxel, z_voxel], dtype=np.float64)
    max_dim = max(int(max_dim), 8)
    max_voxels = max(int(max_voxels), 8)
    counts = np.maximum(np.ceil(span / voxel_xyz_requested).astype(int), 2)
    scale = max(
        float(counts.max()) / float(max_dim),
        float(np.prod(counts, dtype=np.float64) / max_voxels) ** (1.0 / 3.0),
        1.0,
    )
    if scale > 1.0:
        voxel_xyz_requested *= scale * 1.01
        counts = np.maximum(np.ceil(span / voxel_xyz_requested).astype(int), 2)

    edges = [
        np.linspace(float(lo[i]), float(hi[i]), int(counts[i]) + 1, dtype=np.float64)
        for i in range(3)
    ]
    hist, _ = np.histogramdd(locs, bins=edges)
    volume = hist.astype(np.float32, copy=False)

    voxel_xyz = tuple(float(edges[i][1] - edges[i][0]) for i in range(3))
    sigma_px = []
    for sigma_nm, step_nm in zip(sigma_nm_xyz, voxel_xyz):
        if sigma_nm > 0.0:
            sigma_px.append(max(float(sigma_nm) / max(step_nm, 1e-12), 0.0))
        else:
            sigma_px.append(0.75)
    if max(sigma_px) >= 0.1:
        volume = gaussian_filter(volume, sigma=tuple(sigma_px), mode="constant")

    nonzero = volume[volume > 0.0]
    if nonzero.size:
        vmax = float(np.percentile(nonzero, 99.7))
        if vmax <= 0.0:
            vmax = float(nonzero.max())
    else:
        vmax = 1.0
    vmax = max(vmax, 1e-12)
    norm = np.clip(volume / vmax, 0.0, 1.0)

    rgba = _matplotlib_rgba(cmap_name, norm.ravel()).reshape((*volume.shape, 4))
    alpha = (np.power(norm, 0.75) * np.clip(float(opacity), 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba[..., 3] = alpha
    rgba[norm <= 0.0, 3] = 0
    return VolumePayload(
        scalar=np.ascontiguousarray(volume, dtype=np.float32),
        norm=np.ascontiguousarray(norm, dtype=np.float32),
        rgba=np.ascontiguousarray(rgba, dtype=np.uint8),
        origin_nm=(float(edges[0][0]), float(edges[1][0]), float(edges[2][0])),
        voxel_nm=voxel_xyz,
        counts=(int(counts[0]), int(counts[1]), int(counts[2])),
        n_locs=int(locs.shape[0]),
        intensity_max=vmax,
    )


class VolumeRenderWindow(QWidget):
    """Experimental pyqtgraph GLVolumeItem render view."""

    TAG = "volume_render_window"

    def __init__(
        self,
        state: AppState,
        dataset_idx: int,
        *,
        sigma_nm_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._idx = int(dataset_idx)
        self._sigma_nm_xyz = sigma_nm_xyz
        self._view = None
        self._volume_item = None
        self._mip_items: list = []
        self._surface_item = None
        self._overlay_items: list = []
        self._payload: VolumePayload | None = None
        self._show_axis_system = True
        self._show_bounding_box = True
        self._camera_initialized = False
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(250)
        self._rebuild_timer.timeout.connect(self.refresh_from_dataset)

        self.setWindowTitle("3D Volume Preview")
        self.setWindowIcon(QIcon(str(resource_path("icons", "minflux_viewer_logo.png"))))
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(980, 780)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._build_ui()
        self.refresh_from_dataset()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        bar = QHBoxLayout()
        bar.setSpacing(8)

        bar.addWidget(QLabel("XY pixel"))
        self._xy_voxel_spin = QDoubleSpinBox()
        self._xy_voxel_spin.setRange(0.1, 10_000.0)
        self._xy_voxel_spin.setDecimals(1)
        self._xy_voxel_spin.setSuffix(" nm")
        self._xy_voxel_spin.setValue(5.0)
        self._xy_voxel_spin.valueChanged.connect(self._schedule_rebuild)
        bar.addWidget(self._xy_voxel_spin)

        bar.addWidget(QLabel("Z voxel"))
        self._z_voxel_spin = QDoubleSpinBox()
        self._z_voxel_spin.setRange(0.1, 10_000.0)
        self._z_voxel_spin.setDecimals(1)
        self._z_voxel_spin.setSuffix(" nm")
        self._z_voxel_spin.setValue(10.0)
        self._z_voxel_spin.valueChanged.connect(self._schedule_rebuild)
        bar.addWidget(self._z_voxel_spin)

        bar.addWidget(QLabel("Max dim"))
        self._max_dim_spin = QSpinBox()
        self._max_dim_spin.setRange(32, 1024)
        self._max_dim_spin.setValue(160)
        self._max_dim_spin.setToolTip(
            f"Maximum voxels along the longest axis. Total voxels are still capped at {_DEFAULT_MAX_VOXELS:,}."
        )
        self._max_dim_spin.valueChanged.connect(self._schedule_rebuild)
        bar.addWidget(self._max_dim_spin)

        bar.addWidget(QLabel("Mode"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Volume", "MIP", "Surface", "ISO surface"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        bar.addWidget(self._mode_combo)

        bar.addWidget(QLabel("Opacity"))
        self._opacity_spin = QDoubleSpinBox()
        self._opacity_spin.setRange(0.01, 1.0)
        self._opacity_spin.setDecimals(2)
        self._opacity_spin.setSingleStep(0.05)
        self._opacity_spin.setValue(0.45)
        self._opacity_spin.valueChanged.connect(self._schedule_rebuild)
        bar.addWidget(self._opacity_spin)

        bar.addWidget(QLabel("ISO"))
        self._iso_spin = QDoubleSpinBox()
        self._iso_spin.setRange(0.01, 0.99)
        self._iso_spin.setDecimals(2)
        self._iso_spin.setSingleStep(0.05)
        self._iso_spin.setValue(0.25)
        self._iso_spin.valueChanged.connect(self._schedule_rebuild)
        bar.addWidget(self._iso_spin)

        bar.addWidget(QLabel("Colormap"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["hot", "inferno", "viridis", "magma", "plasma", "cividis", "gray", "turbo"])
        self._cmap_combo.currentTextChanged.connect(self._schedule_rebuild)
        bar.addWidget(self._cmap_combo)

        self._rebuild_btn = QPushButton("Rebuild")
        self._rebuild_btn.clicked.connect(self.refresh_from_dataset)
        bar.addWidget(self._rebuild_btn)

        self._reset_btn = QPushButton("Reset camera")
        self._reset_btn.clicked.connect(self._reset_camera)
        bar.addWidget(self._reset_btn)
        bar.addStretch()
        root.addLayout(bar)

        try:
            import pyqtgraph.opengl as gl
        except Exception as exc:
            self._info_label = QLabel(f"3D OpenGL preview unavailable: {exc}")
            self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root.addWidget(self._info_label, stretch=1)
            return

        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor("k")
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self._show_context_menu)
        root.addWidget(self._view, stretch=1)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info_label)
        self._on_mode_changed(self._mode_combo.currentText())

    def refresh_from_dataset(self) -> None:
        if self._idx < 0 or self._idx >= len(self._state.datasets):
            self._info_label.setText("No active dataset.")
            return
        ds = self._state.datasets[self._idx]
        self.setWindowTitle(f"3D Volume Preview - {ds.name}")
        locs = _finite_filtered_locs(ds)
        if locs.shape[0] > 0:
            span = np.ptp(locs, axis=0)
            suggested_xy = max(float(ds.cali.pixel_size), float(max(span[0], span[1])) / 160.0, 0.5)
            suggested_z = max(float(ds.cali.pixel_size), float(span[2]) / 96.0, 0.5)
            if not getattr(self, "_voxel_initialised", False):
                self._xy_voxel_spin.setValue(float(suggested_xy))
                self._z_voxel_spin.setValue(float(suggested_z))
                self._voxel_initialised = True

        if self._view is None:
            self._info_label.setText("3D OpenGL preview unavailable.")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            payload = make_volume_payload(
                locs,
                xy_voxel_nm=float(self._xy_voxel_spin.value()),
                z_voxel_nm=float(self._z_voxel_spin.value()),
                max_dim=int(self._max_dim_spin.value()),
                max_voxels=_DEFAULT_MAX_VOXELS,
                cmap_name=self._cmap_combo.currentText(),
                opacity=float(self._opacity_spin.value()),
                sigma_nm_xyz=self._sigma_nm_xyz,
            )
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self._info_label.setText(str(exc))
            self._clear_render_items()
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

        self._payload = payload
        self._set_render_item(payload)
        if not self._camera_initialized:
            self._reset_camera()
        nx, ny, nz = payload.counts
        vx, vy, vz = payload.voxel_nm
        self._info_label.setText(
            f"{payload.n_locs:,} filtered locs  |  volume {nx} x {ny} x {nz}  |  "
            f"voxel=({vx:.1f}, {vy:.1f}, {vz:.1f}) nm  |  "
            f"mode={self._mode_combo.currentText()}  |  intensity max~{payload.intensity_max:.3g}"
        )

    def _schedule_rebuild(self, *_args) -> None:
        if not getattr(self, "_voxel_initialised", False):
            return
        self._rebuild_timer.start()

    def _on_mode_changed(self, mode: str) -> None:
        self._iso_spin.setEnabled(mode in {"Surface", "ISO surface"})
        self._schedule_rebuild()

    def _clear_render_items(self) -> None:
        if self._view is None:
            return
        if self._volume_item is not None:
            self._view.removeItem(self._volume_item)
            self._volume_item = None
        if self._surface_item is not None:
            self._view.removeItem(self._surface_item)
            self._surface_item = None
        for item in self._mip_items:
            self._view.removeItem(item)
        self._mip_items = []

    def _clear_overlay_items(self) -> None:
        if self._view is None:
            return
        for item in self._overlay_items:
            try:
                self._view.removeItem(item)
            except Exception:
                pass
        self._overlay_items = []

    def _set_render_item(self, payload: VolumePayload) -> None:
        mode = self._mode_combo.currentText()
        if mode == "MIP":
            self._set_mip_items(payload)
        elif mode in {"Surface", "ISO surface"}:
            self._set_surface_item(payload, smooth=(mode == "Surface"))
        else:
            self._set_volume_item(payload)
        self._update_overlays(payload)

    def _set_volume_item(self, payload: VolumePayload) -> None:
        if self._view is None:
            return
        import pyqtgraph.opengl as gl

        self._clear_render_items()
        self._volume_item = gl.GLVolumeItem(payload.rgba, sliceDensity=1, smooth=True, glOptions="translucent")
        self._volume_item.scale(*payload.voxel_nm)
        self._volume_item.translate(*self._scene_origin(payload), local=False)
        self._view.addItem(self._volume_item)

    def _set_mip_items(self, payload: VolumePayload) -> None:
        if self._view is None:
            return
        import pyqtgraph.opengl as gl

        self._clear_render_items()
        rgba = payload.rgba
        opacity = np.clip(float(self._opacity_spin.value()), 0.0, 1.0)
        image = rgba.max(axis=2)
        ox, oy, oz = self._scene_origin(payload)
        sz = payload.counts[2] * payload.voxel_nm[2]
        img = np.ascontiguousarray(image, dtype=np.uint8)
        img[..., 3] = np.minimum(img[..., 3].astype(np.float32) * max(opacity, 0.05), 255).astype(np.uint8)
        item = gl.GLImageItem(img)
        item.scale(payload.voxel_nm[0], payload.voxel_nm[1], 1.0)
        item.translate(ox, oy, oz + sz / 2.0, local=False)
        self._view.addItem(item)
        self._mip_items.append(item)

    def _set_surface_item(self, payload: VolumePayload, *, smooth: bool) -> None:
        if self._view is None:
            return
        import pyqtgraph.opengl as gl

        self._clear_render_items()
        level = float(self._iso_spin.value())
        data = np.ascontiguousarray(payload.norm, dtype=np.float32)
        if float(data.max()) <= level:
            self._info_label.setText("ISO threshold is above the rendered volume intensity.")
            return
        verts, faces = pg.isosurface(data, level)
        if len(verts) == 0 or len(faces) == 0:
            self._info_label.setText("No surface found at the current ISO threshold.")
            return
        verts = np.asarray(verts, dtype=np.float32)
        ox, oy, oz = self._scene_origin(payload)
        verts[:, 0] = ox + verts[:, 0] * payload.voxel_nm[0]
        verts[:, 1] = oy + verts[:, 1] * payload.voxel_nm[1]
        verts[:, 2] = oz + verts[:, 2] * payload.voxel_nm[2]
        color = _surface_color(self._cmap_combo.currentText(), level, float(self._opacity_spin.value()))
        mesh = gl.MeshData(vertexes=verts, faces=np.asarray(faces, dtype=np.uint32))
        self._surface_item = gl.GLMeshItem(
            meshdata=mesh,
            smooth=bool(smooth),
            color=color,
            shader="shaded",
            glOptions="translucent",
        )
        self._view.addItem(self._surface_item)

    def _scene_origin(self, payload: VolumePayload) -> tuple[float, float, float]:
        sx = payload.counts[0] * payload.voxel_nm[0]
        sy = payload.counts[1] * payload.voxel_nm[1]
        sz = payload.counts[2] * payload.voxel_nm[2]
        return (-sx / 2.0, -sy / 2.0, -sz / 2.0)

    def _volume_bounds(self, payload: VolumePayload) -> tuple[float, float, float, float, float, float]:
        x0, y0, z0 = self._scene_origin(payload)
        sx = payload.counts[0] * payload.voxel_nm[0]
        sy = payload.counts[1] * payload.voxel_nm[1]
        sz = payload.counts[2] * payload.voxel_nm[2]
        return x0, x0 + sx, y0, y0 + sy, z0, z0 + sz

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        axis_action = menu.addAction("Axis system")
        axis_action.setCheckable(True)
        axis_action.setChecked(self._show_axis_system)
        box_action = menu.addAction("Bounding box")
        box_action.setCheckable(True)
        box_action.setChecked(self._show_bounding_box)
        menu.addSeparator()
        menu.addAction("Reset camera", self._reset_camera)
        action = menu.exec(self._view.mapToGlobal(pos))
        if action is axis_action:
            self._show_axis_system = axis_action.isChecked()
            self._update_overlays(self._payload)
        elif action is box_action:
            self._show_bounding_box = box_action.isChecked()
            self._update_overlays(self._payload)

    def _update_overlays(self, payload: VolumePayload | None) -> None:
        self._clear_overlay_items()
        if self._view is None or payload is None:
            return
        if self._show_bounding_box:
            self._add_bounding_box(payload)
        if self._show_axis_system:
            self._add_axis_system(payload)

    def _add_line(
        self,
        points: list[tuple[float, float, float]],
        color: tuple[float, float, float, float],
        *,
        width: float = 1.0,
        mode: str = "line_strip",
    ) -> None:
        if self._view is None:
            return
        import pyqtgraph.opengl as gl

        item = gl.GLLinePlotItem(
            pos=np.asarray(points, dtype=np.float32),
            color=color,
            width=width,
            mode=mode,
            antialias=False,
        )
        self._view.addItem(item)
        self._overlay_items.append(item)

    def _add_text(
        self,
        text: str,
        pos: tuple[float, float, float],
        color: tuple[int, int, int, int] = (230, 230, 230, 255),
    ) -> None:
        if self._view is None:
            return
        try:
            import pyqtgraph.opengl as gl

            item = gl.GLTextItem(pos=np.asarray(pos, dtype=float), text=text, color=color)
        except Exception:
            return
        self._view.addItem(item)
        self._overlay_items.append(item)

    def _add_bounding_box(self, payload: VolumePayload) -> None:
        x0, x1, y0, y1, z0, z1 = self._volume_bounds(payload)
        corners = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        points = []
        for a, b in edges:
            points.extend([corners[a], corners[b]])
        self._add_line(points, (0.8, 0.8, 0.8, 0.55), mode="lines")

    def _add_axis_system(self, payload: VolumePayload) -> None:
        x0, x1, y0, y1, z0, z1 = self._volume_bounds(payload)
        sx, sy, sz = x1 - x0, y1 - y0, z1 - z0
        tick = max(min(sx, sy, sz) * 0.025, 1.0)
        self._add_line([(x0, y0, z0), (x1, y0, z0)], (1.0, 0.2, 0.2, 0.95), width=2.0)
        self._add_line([(x0, y0, z0), (x0, y1, z0)], (0.2, 1.0, 0.2, 0.95), width=2.0)
        self._add_line([(x0, y0, z0), (x0, y0, z1)], (0.2, 0.45, 1.0, 0.95), width=2.0)

        self._add_ticks("x", x0, x1, y0, z0, tick, (1.0, 0.35, 0.35, 0.85))
        self._add_ticks("y", y0, y1, x0, z0, tick, (0.35, 1.0, 0.35, 0.85))
        self._add_ticks("z", z0, z1, x0, y0, tick, (0.35, 0.55, 1.0, 0.85))
        self._add_text(f"X {sx:.0f} nm", (x1 + 2 * tick, y0, z0), (255, 110, 110, 255))
        self._add_text(f"Y {sy:.0f} nm", (x0, y1 + 2 * tick, z0), (110, 255, 110, 255))
        self._add_text(f"Z {sz:.0f} nm", (x0, y0, z1 + 2 * tick), (120, 160, 255, 255))

    def _add_ticks(
        self,
        axis: str,
        lo: float,
        hi: float,
        fixed_a: float,
        fixed_b: float,
        tick: float,
        color: tuple[float, float, float, float],
    ) -> None:
        span = hi - lo
        if span <= 0:
            return
        for value in np.linspace(lo, hi, 5):
            if axis == "x":
                self._add_line([(value, fixed_a, fixed_b), (value, fixed_a - tick, fixed_b)], color)
            elif axis == "y":
                self._add_line([(fixed_a, value, fixed_b), (fixed_a - tick, value, fixed_b)], color)
            else:
                self._add_line([(fixed_a, fixed_b, value), (fixed_a - tick, fixed_b, value)], color)

    def _reset_camera(self) -> None:
        if self._view is None or self._payload is None:
            return
        payload = self._payload
        sx = payload.counts[0] * payload.voxel_nm[0]
        sy = payload.counts[1] * payload.voxel_nm[1]
        sz = payload.counts[2] * payload.voxel_nm[2]
        extent = max(float(np.linalg.norm([sx, sy, sz])), 10.0)
        self._view.opts["center"] = pg.Vector(0.0, 0.0, 0.0)
        self._view.setCameraPosition(distance=extent * 1.6, elevation=25, azimuth=45)
        self._camera_initialized = True

    def closeEvent(self, event) -> None:
        self._rebuild_timer.stop()
        self._clear_render_items()
        self._clear_overlay_items()
        super().closeEvent(event)
