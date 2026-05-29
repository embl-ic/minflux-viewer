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
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import gaussian_filter

from .. import resource_path
from ..core.app_state import AppState


@dataclass(frozen=True)
class VolumePayload:
    """Voxelised RGBA volume plus physical placement metadata."""

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


def make_volume_payload(
    locs_nm: np.ndarray,
    *,
    voxel_nm: float,
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

    voxel = max(float(voxel_nm), 0.001)
    max_dim = max(int(max_dim), 8)
    max_voxels = max(int(max_voxels), 8)
    counts = np.maximum(np.ceil(span / voxel).astype(int), 2)
    scale = max(
        float(counts.max()) / float(max_dim),
        float(np.prod(counts, dtype=np.float64) / max_voxels) ** (1.0 / 3.0),
        1.0,
    )
    if scale > 1.0:
        voxel *= scale * 1.001
        counts = np.maximum(np.ceil(span / voxel).astype(int), 2)

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
        self._grid_item = None
        self._axis_item = None
        self._payload: VolumePayload | None = None

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

        bar.addWidget(QLabel("Voxel"))
        self._voxel_spin = QDoubleSpinBox()
        self._voxel_spin.setRange(0.1, 10_000.0)
        self._voxel_spin.setDecimals(2)
        self._voxel_spin.setSuffix(" nm")
        self._voxel_spin.setValue(5.0)
        bar.addWidget(self._voxel_spin)

        bar.addWidget(QLabel("Max dim"))
        self._max_dim_spin = QSpinBox()
        self._max_dim_spin.setRange(32, 256)
        self._max_dim_spin.setValue(160)
        bar.addWidget(self._max_dim_spin)

        bar.addWidget(QLabel("Opacity"))
        self._opacity_spin = QDoubleSpinBox()
        self._opacity_spin.setRange(0.01, 1.0)
        self._opacity_spin.setDecimals(2)
        self._opacity_spin.setSingleStep(0.05)
        self._opacity_spin.setValue(0.45)
        bar.addWidget(self._opacity_spin)

        bar.addWidget(QLabel("Colormap"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["hot", "inferno", "viridis", "magma", "plasma", "cividis", "gray", "turbo"])
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
        root.addWidget(self._view, stretch=1)

        self._grid_item = gl.GLGridItem()
        self._view.addItem(self._grid_item)
        self._axis_item = gl.GLAxisItem()
        self._view.addItem(self._axis_item)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._info_label)

    def refresh_from_dataset(self) -> None:
        if self._idx < 0 or self._idx >= len(self._state.datasets):
            self._info_label.setText("No active dataset.")
            return
        ds = self._state.datasets[self._idx]
        self.setWindowTitle(f"3D Volume Preview - {ds.name}")
        locs = _finite_filtered_locs(ds)
        if locs.shape[0] > 0:
            span = np.ptp(locs, axis=0)
            suggested = max(float(ds.cali.pixel_size), float(span.max()) / 160.0, 0.5)
            if not getattr(self, "_voxel_initialised", False):
                self._voxel_spin.setValue(float(suggested))
                self._voxel_initialised = True

        if self._view is None:
            self._info_label.setText("3D OpenGL preview unavailable.")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            payload = make_volume_payload(
                locs,
                voxel_nm=float(self._voxel_spin.value()),
                max_dim=int(self._max_dim_spin.value()),
                cmap_name=self._cmap_combo.currentText(),
                opacity=float(self._opacity_spin.value()),
                sigma_nm_xyz=self._sigma_nm_xyz,
            )
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self._info_label.setText(str(exc))
            if self._volume_item is not None:
                self._view.removeItem(self._volume_item)
                self._volume_item = None
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

        self._payload = payload
        self._set_volume_item(payload)
        self._reset_camera()
        nx, ny, nz = payload.counts
        vx, vy, vz = payload.voxel_nm
        self._info_label.setText(
            f"{payload.n_locs:,} filtered locs  |  volume {nx} x {ny} x {nz}  |  "
            f"voxel=({vx:.2f}, {vy:.2f}, {vz:.2f}) nm  |  intensity max~{payload.intensity_max:.3g}"
        )

    def _set_volume_item(self, payload: VolumePayload) -> None:
        if self._view is None:
            return
        import pyqtgraph.opengl as gl

        if self._volume_item is not None:
            self._view.removeItem(self._volume_item)
        self._volume_item = gl.GLVolumeItem(payload.rgba, sliceDensity=1, smooth=True, glOptions="translucent")
        self._volume_item.scale(*payload.voxel_nm)
        self._volume_item.translate(*payload.origin_nm, local=False)
        self._view.addItem(self._volume_item)

        sx = payload.counts[0] * payload.voxel_nm[0]
        sy = payload.counts[1] * payload.voxel_nm[1]
        sz = payload.counts[2] * payload.voxel_nm[2]
        if self._grid_item is not None:
            self._grid_item.setSize(x=sx, y=sy)
            self._grid_item.setSpacing(x=max(sx / 10.0, payload.voxel_nm[0]), y=max(sy / 10.0, payload.voxel_nm[1]))
            self._grid_item.resetTransform()
            self._grid_item.translate(payload.origin_nm[0] + sx / 2.0, payload.origin_nm[1] + sy / 2.0, payload.origin_nm[2])
        if self._axis_item is not None:
            axis_size = max(sx, sy, sz, 1.0) * 0.35
            self._axis_item.setSize(axis_size, axis_size, axis_size)
            self._axis_item.resetTransform()
            self._axis_item.translate(*payload.origin_nm)

    def _reset_camera(self) -> None:
        if self._view is None or self._payload is None:
            return
        payload = self._payload
        sx = payload.counts[0] * payload.voxel_nm[0]
        sy = payload.counts[1] * payload.voxel_nm[1]
        sz = payload.counts[2] * payload.voxel_nm[2]
        center = (
            payload.origin_nm[0] + sx / 2.0,
            payload.origin_nm[1] + sy / 2.0,
            payload.origin_nm[2] + sz / 2.0,
        )
        extent = max(float(np.linalg.norm([sx, sy, sz])), 10.0)
        self._view.opts["center"] = pg.Vector(*center)
        self._view.setCameraPosition(distance=extent * 1.6, elevation=25, azimuth=45)
