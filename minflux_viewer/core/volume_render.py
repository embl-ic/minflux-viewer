"""
minflux_viewer.core.volume_render
=================================
Cached histogram-volume rendering helpers for Fiji-style image views.

The renderer converts localisation coordinates into a regular 3-D voxel grid,
applies anisotropic Gaussian smoothing, and exposes 2-D slice/projection views
that can be shown in an image-style canvas.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class VolumeRenderConfig:
    """Physical rendering parameters expressed in nanometres."""

    xy_bin_nm: float
    z_bin_nm: float
    sigma_xy_nm: float
    sigma_z_nm: float


@dataclass(frozen=True)
class VolumeSlice:
    """A 2-D image plus spatial metadata for placement in the image canvas."""

    image: np.ndarray
    x_nm: float
    y_nm: float
    pixel_x_nm: float
    pixel_y_nm: float
    z_nm: float | None
    description: str


def _sanitise_step(step_nm: float, fallback_nm: float) -> float:
    if np.isfinite(step_nm) and step_nm > 0:
        return float(step_nm)
    return float(fallback_nm)


def _extent_with_padding(values_nm: np.ndarray) -> tuple[float, float]:
    lo = float(np.min(values_nm))
    hi = float(np.max(values_nm))
    if hi <= lo:
        hi = lo + 1.0
    pad = max((hi - lo) * 0.02, 0.5)
    return lo - pad, hi + pad


class HistogramVolume:
    """Cached voxelised representation of a MINFLUX localisation cloud."""

    def __init__(self, loc_nm: np.ndarray) -> None:
        if loc_nm.ndim != 2 or loc_nm.shape[1] != 3:
            raise ValueError("loc_nm must have shape (N, 3)")

        self._loc_nm = np.asarray(loc_nm, dtype=np.float64)
        self._config: VolumeRenderConfig | None = None
        self._raw_volume: np.ndarray | None = None
        self._smoothed_volume: np.ndarray | None = None
        self._edges_x: np.ndarray | None = None
        self._edges_y: np.ndarray | None = None
        self._edges_z: np.ndarray | None = None

    @property
    def has_data(self) -> bool:
        return self._loc_nm.shape[0] > 0

    @property
    def is_3d(self) -> bool:
        if not self.has_data:
            return False
        z = self._loc_nm[:, 2]
        return bool(np.ptp(z) > 1.0)

    def configure(self, config: VolumeRenderConfig) -> None:
        """Rebuild caches only when the physical render settings change."""
        if self._config == config and self._smoothed_volume is not None:
            return
        self._config = config
        self._build_volume()

    def _build_volume(self) -> None:
        if self._config is None:
            raise RuntimeError("configure() must be called before rendering")
        if not self.has_data:
            self._raw_volume = np.zeros((1, 1, 1), dtype=np.float32)
            self._smoothed_volume = self._raw_volume.copy()
            self._edges_x = np.array([0.0, 1.0], dtype=np.float64)
            self._edges_y = np.array([0.0, 1.0], dtype=np.float64)
            self._edges_z = np.array([0.0, 1.0], dtype=np.float64)
            return

        cfg = self._config
        x = self._loc_nm[:, 0]
        y = self._loc_nm[:, 1]
        z = self._loc_nm[:, 2]
        x0, x1 = _extent_with_padding(x)
        y0, y1 = _extent_with_padding(y)
        z0, z1 = _extent_with_padding(z)

        self._edges_x = np.linspace(
            x0, x1, max(int(ceil((x1 - x0) / cfg.xy_bin_nm)), 1) + 1, dtype=np.float64
        )
        self._edges_y = np.linspace(
            y0, y1, max(int(ceil((y1 - y0) / cfg.xy_bin_nm)), 1) + 1, dtype=np.float64
        )
        z_step = cfg.z_bin_nm if self.is_3d else max(z1 - z0, 1.0)
        self._edges_z = np.linspace(
            z0, z1, max(int(ceil((z1 - z0) / z_step)), 1) + 1, dtype=np.float64
        )

        hist, _ = np.histogramdd(
            self._loc_nm,
            bins=(self._edges_x, self._edges_y, self._edges_z),
        )
        self._raw_volume = hist.astype(np.float32, copy=False)

        sigma = (
            max(cfg.sigma_xy_nm / self.pixel_x_nm, 0.0),
            max(cfg.sigma_xy_nm / self.pixel_y_nm, 0.0),
            max(cfg.sigma_z_nm / self.pixel_z_nm, 0.0),
        )
        self._smoothed_volume = gaussian_filter(self._raw_volume, sigma=sigma, mode="constant")

    @property
    def shape(self) -> tuple[int, int, int]:
        if self._smoothed_volume is None:
            return (0, 0, 0)
        return self._smoothed_volume.shape

    @property
    def pixel_x_nm(self) -> float:
        assert self._edges_x is not None
        return float(self._edges_x[1] - self._edges_x[0])

    @property
    def pixel_y_nm(self) -> float:
        assert self._edges_y is not None
        return float(self._edges_y[1] - self._edges_y[0])

    @property
    def pixel_z_nm(self) -> float:
        assert self._edges_z is not None
        return float(self._edges_z[1] - self._edges_z[0])

    @property
    def z_centres_nm(self) -> np.ndarray:
        assert self._edges_z is not None
        return 0.5 * (self._edges_z[:-1] + self._edges_z[1:])

    def render(self, mode: str, z_index: int = 0, slab_bins: int = 2) -> VolumeSlice:
        """
        Render a Fiji-style image from the cached volume.

        Parameters
        ----------
        mode:
            One of ``slice``, ``sum``, or ``max``.
        z_index:
            Index of the current focal plane in the voxel grid.
        slab_bins:
            Radius, in z bins, around ``z_index`` used for slab projections.
        """
        if self._smoothed_volume is None or self._edges_x is None or self._edges_y is None:
            raise RuntimeError("configure() must be called before render()")

        nx, ny, nz = self.shape
        if nz == 0:
            image = np.zeros((1, 1), dtype=np.float32)
            return VolumeSlice(image, 0.0, 0.0, 1.0, 1.0, None, "empty")

        z_index = int(np.clip(z_index, 0, nz - 1))
        z_lo = max(z_index - slab_bins, 0)
        z_hi = min(z_index + slab_bins + 1, nz)
        volume = self._smoothed_volume

        if mode == "max":
            plane = np.max(volume[:, :, z_lo:z_hi], axis=2)
            desc = f"MIP over {z_hi - z_lo} z bins"
            z_nm = None
        elif mode == "sum":
            plane = np.sum(volume[:, :, z_lo:z_hi], axis=2)
            desc = f"slab sum over {z_hi - z_lo} z bins"
            z_nm = float(np.mean(self.z_centres_nm[z_lo:z_hi]))
        else:
            plane = volume[:, :, z_index]
            desc = f"single z slice #{z_index + 1}/{nz}"
            z_nm = float(self.z_centres_nm[z_index])

        return VolumeSlice(
            image=np.asarray(plane.T, dtype=np.float32),
            x_nm=float(self._edges_x[0]),
            y_nm=float(self._edges_y[0]),
            pixel_x_nm=self.pixel_x_nm,
            pixel_y_nm=self.pixel_y_nm,
            z_nm=z_nm,
            description=desc,
        )


def config_from_dataset(
    dataset,
    *,
    xy_bin_nm: float | None = None,
    z_bin_nm: float | None = None,
    sigma_xy_nm: float | None = None,
    sigma_z_nm: float | None = None,
) -> VolumeRenderConfig:
    """Build a reasonable default render configuration from a dataset."""
    base_xy = _sanitise_step(float(xy_bin_nm or dataset.cali.pixel_size), 4.0)
    base_z = _sanitise_step(float(z_bin_nm or max(base_xy * 2.0, 8.0)), 8.0)
    sig_xy = _sanitise_step(float(sigma_xy_nm or base_xy * 1.5), base_xy * 1.5)
    sig_z = _sanitise_step(float(sigma_z_nm or max(sig_xy, base_z)), max(sig_xy, base_z))
    return VolumeRenderConfig(
        xy_bin_nm=base_xy,
        z_bin_nm=base_z,
        sigma_xy_nm=sig_xy,
        sigma_z_nm=sig_z,
    )
