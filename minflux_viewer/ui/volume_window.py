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
from copy import deepcopy

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
    # Compact per-channel normalized volumes let overlay LUT/visibility changes
    # recompose without repeating the expensive histogram/blur pass.
    channel_norms: tuple[np.ndarray, ...] | None = None


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


def _clip_region(locs, sigma, region):
    """Keep only localizations inside the 3-D box (xlo,xhi,ylo,yhi,zlo,zhi nm),
    carrying an aligned per-loc ``sigma`` (or None) along with them."""
    xlo, xhi, ylo, yhi, zlo, zhi = region
    keep = (
        (locs[:, 0] >= xlo) & (locs[:, 0] <= xhi)
        & (locs[:, 1] >= ylo) & (locs[:, 1] <= yhi)
        & (locs[:, 2] >= zlo) & (locs[:, 2] <= zhi)
    )
    return locs[keep], (None if sigma is None else sigma[keep])


def _matplotlib_rgba(
    name: str, values: np.ndarray, *, invert: bool = False
) -> np.ndarray:
    """Map normalised values to uint8 RGBA through matplotlib."""
    try:
        import matplotlib as mpl
        cmap = mpl.colormaps.get_cmap(name)
        mapper = lambda v: np.asarray(cmap(v, bytes=True), dtype=np.uint8)
    except Exception:
        # The LUT dialog also exposes pyqtgraph/colorcet and pure-colour ramps
        # that are not necessarily registered in matplotlib.
        from .lut_dialog import make_colormap
        cmap = make_colormap(name)
        table = np.asarray(cmap.getLookupTable(0.0, 1.0, 256), dtype=np.uint8)
        if table.ndim != 2 or table.shape[1] not in (3, 4):
            table = np.tile(np.array([[255, 255, 255, 255]], dtype=np.uint8), (256, 1))
        elif table.shape[1] == 3:
            table = np.column_stack([table, np.full(256, 255, dtype=np.uint8)])
        mapper = lambda v: table[np.clip(np.rint(v * 255.0), 0, 255).astype(np.int64)]
    if invert:
        values = 1.0 - np.asarray(values, dtype=float)
    return mapper(np.asarray(values, dtype=float))


def _normalize_volume(
    volume: np.ndarray, black_pct: float, white_pct: float
) -> tuple[np.ndarray, float]:
    """Normalize a voxel scalar using nonzero-voxel percentile B/C."""
    nonzero = np.asarray(volume)[np.asarray(volume) > 0.0]
    wp = float(np.clip(white_pct, 0.0, 100.0))
    bp = float(np.clip(black_pct, 0.0, max(wp - 1e-3, 0.0)))
    if nonzero.size:
        vhi = float(np.percentile(nonzero, wp))
        if vhi <= 0.0:
            vhi = float(nonzero.max())
        vlo = 0.0 if bp <= 0.0 else float(np.percentile(nonzero, bp))
    else:
        vhi, vlo = 1.0, 0.0
    vhi = max(vhi, vlo + 1e-12)
    norm = np.clip((np.asarray(volume, dtype=np.float32) - vlo) / (vhi - vlo), 0.0, 1.0)
    return np.ascontiguousarray(norm, dtype=np.float32), vhi


def _compose_multichannel_rgba(
    channel_norms: list[np.ndarray] | tuple[np.ndarray, ...],
    channel_rgb: list | tuple,
    opacity: float,
) -> np.ndarray:
    """Compose normalized overlay channels into one additive RGBA volume."""
    if not channel_norms:
        return np.empty((0, 0, 0, 4), dtype=np.uint8)
    shape = np.asarray(channel_norms[0]).shape
    rgb_accum = np.zeros((*shape, 3), dtype=np.float32)
    alpha_accum = np.zeros(shape, dtype=np.float32)
    for norm_uint8, rgb in zip(channel_norms, channel_rgb):
        norm = np.asarray(norm_uint8, dtype=np.float32) / 255.0
        colour = np.asarray(rgb, dtype=np.float32).ravel()[:3]
        for k in range(3):
            rgb_accum[..., k] += float(colour[k]) * norm
        alpha_accum = np.maximum(alpha_accum, np.power(norm, 0.75))
    rgba = np.zeros((*shape, 4), dtype=np.uint8)
    rgba[..., :3] = (np.clip(rgb_accum, 0.0, 1.0) * 255.0).astype(np.uint8)
    alpha = (
        alpha_accum * np.clip(float(opacity), 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    alpha[alpha_accum <= 0.0] = 0
    rgba[..., 3] = alpha
    return np.ascontiguousarray(rgba, dtype=np.uint8)


def _surface_color(
    cmap_name: str, value: float, opacity: float, *, invert: bool = False
) -> tuple[float, float, float, float]:
    rgba = _matplotlib_rgba(
        cmap_name,
        np.array([np.clip(value, 0.0, 1.0)], dtype=float),
        invert=invert,
    )[0]
    return (
        float(rgba[0]) / 255.0,
        float(rgba[1]) / 255.0,
        float(rgba[2]) / 255.0,
        float(np.clip(opacity, 0.05, 1.0)),
    )


# Per-method 3-D blur, expressed as a fraction of the voxel size. Crisp for
# histogram; a light sub-voxel smoothing for the interpolating/smoothed methods
# (a voxel volume can't show a splat kernel's sub-voxel shape, so a small blur
# captures each method's "smoother than histogram" character). Fixed/precision
# use real nm sigmas instead (see _voxelize_volume).
_METHOD_VOXEL_BLUR = {
    "histogram": 0.0,
    "bilinear": 0.6,
    "bicubic": 0.9,
    "basic": 0.5,
}
# Anti-alias floor (voxels) for the precision volume: when the total-voxel cap
# forces a coarse voxel, the per-loc precision blur becomes sub-voxel and leaves
# a blocky histogram; this floor keeps it smooth (≈ the base render's default).
_MIN_VOXEL_BLUR = 0.75


def _precision_volume_3d(
    locs: np.ndarray,
    sigma_nm: np.ndarray,
    edges: list[np.ndarray],
    voxel_xyz: tuple[float, float, float],
    *,
    n_bins: int = 6,
) -> np.ndarray:
    """True per-localization precision 3-D volume.

    Each localization is an anisotropic 3-D Gaussian sized by its own
    ``sigma_nm`` (X/Y/Z). For speed this is done by **sigma binning**: group
    localizations into ``n_bins`` by their mean precision, histogram each group,
    blur it by that group's median sigma (in voxels), and sum — O(bins·voxels)
    instead of a per-localization stamp, visually indistinguishable from it.

    The per-bin blur is combined in quadrature with a **minimum anti-alias floor**
    (``_MIN_VOXEL_BLUR`` voxels): when the total-voxel cap forces a coarse voxel,
    the precision (nm) becomes sub-voxel and would leave a raw blocky histogram —
    the floor keeps the volume smooth (matching the base render), so it never
    looks pixelated even though it can't resolve below the voxel.
    """
    counts = tuple(len(e) - 1 for e in edges)
    total = int(np.prod(counts))
    flat = np.zeros(total, dtype=np.float64)
    if locs.shape[0] == 0:
        return flat.reshape(counts).astype(np.float32)

    sig = np.maximum(np.asarray(sigma_nm, dtype=np.float64), 1e-3)
    smean = np.cbrt(np.prod(sig, axis=1))
    lo, hi = np.percentile(smean, [1.0, 99.0])
    if hi <= lo:
        group = np.zeros(smean.shape[0], dtype=np.int64)
        bin_edges = np.array([lo, hi + 1e-6])
    else:
        bin_edges = np.linspace(lo, hi, n_bins + 1)
        group = np.clip(np.digitize(smean, bin_edges) - 1, 0, n_bins - 1)

    for g in range(len(bin_edges) - 1):
        sel = group == g
        if not np.any(sel):
            continue
        hist, _ = np.histogramdd(locs[sel], bins=edges)
        vol = hist.astype(np.float64)
        med = np.median(sig[sel], axis=0)
        sigma_px = tuple(
            float(np.hypot(med[i] / max(voxel_xyz[i], 1e-12), _MIN_VOXEL_BLUR))
            for i in range(3)
        )
        vol = gaussian_filter(vol, sigma=sigma_px, mode="constant")
        flat += vol.reshape(-1)
    return flat.reshape(counts).astype(np.float32)


def _voxelize_volume(
    locs: np.ndarray,
    edges: list[np.ndarray],
    voxel_xyz: tuple[float, float, float],
    *,
    render_method: str | None,
    sigma_nm_xyz: tuple[float, float, float],
    precision_sigma_nm: np.ndarray | None,
) -> np.ndarray:
    """Voxelise localizations into a scalar volume, reflecting *render_method*.

    ``render_method=None`` reproduces the original behavior (histogram + the
    ``sigma_nm_xyz``-with-0.75-voxel-fallback blur).
    """
    method = (render_method or "").lower()
    if method == "precision_gaussian" and precision_sigma_nm is not None:
        return _precision_volume_3d(locs, precision_sigma_nm, edges, voxel_xyz)

    hist, _ = np.histogramdd(locs, bins=edges)
    volume = hist.astype(np.float32, copy=False)
    if method == "fixed_gaussian":
        sigma_px = tuple(
            max(float(s) / max(v, 1e-12), 0.0) for s, v in zip(sigma_nm_xyz, voxel_xyz)
        )
    elif method in _METHOD_VOXEL_BLUR:
        frac = _METHOD_VOXEL_BLUR[method]
        sigma_px = (frac, frac, frac)
    else:
        # None / precision-without-sigma / unknown → original fallback behavior.
        sigma_px = tuple(
            max(float(s) / max(v, 1e-12), 0.0) if s > 0.0 else 0.75
            for s, v in zip(sigma_nm_xyz, voxel_xyz)
        )
    if max(sigma_px) >= 0.1:
        volume = gaussian_filter(volume, sigma=tuple(sigma_px), mode="constant")
    return volume


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
    render_method: str | None = None,
    precision_sigma_nm: np.ndarray | None = None,
    black_pct: float = 0.0,
    white_pct: float = 99.7,
    invert: bool = False,
) -> VolumePayload:
    """Voxelise localisation coordinates into a bounded RGBA volume.

    ``black_pct``/``white_pct`` are the percentiles of the non-zero voxel counts
    mapped to fully transparent / fully opaque — the volume's brightness/contrast.
    The default ``(0, 99.7)`` reproduces the historical auto-normalisation; the
    advanced view passes the 2-D view's contrast expressed as percentiles so the
    3-D visibility matches what's on screen.
    """
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
    voxel_xyz = tuple(float(edges[i][1] - edges[i][0]) for i in range(3))
    volume = _voxelize_volume(
        locs, edges, voxel_xyz,
        render_method=render_method,
        sigma_nm_xyz=sigma_nm_xyz,
        precision_sigma_nm=precision_sigma_nm,
    )

    norm, vmax = _normalize_volume(volume, black_pct, white_pct)

    rgba = _matplotlib_rgba(
        cmap_name, norm.ravel(), invert=invert
    ).reshape((*volume.shape, 4))
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


def lut_rgb(name: str) -> tuple[float, float, float]:
    """Representative 0..1 RGB for an overlay channel LUT name.

    Pure-colour LUTs (Red/Green/…) map to their colour; a named colormap is
    sampled near its bright end. Used to colour each overlay channel in the
    multi-channel volume composite.
    """
    from ..core.overlay import PURE_COLOR_RGB
    if name in PURE_COLOR_RGB:
        r, g, b = PURE_COLOR_RGB[name]
        return (r / 255.0, g / 255.0, b / 255.0)
    try:
        import matplotlib as mpl
        rgba = mpl.colormaps.get_cmap(name)(0.85)
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]))
    except Exception:
        return (1.0, 1.0, 1.0)


def _volume_grid(all_locs: np.ndarray, xy_voxel_nm: float, z_voxel_nm: float,
                 max_dim: int, max_voxels: int):
    """Shared voxel grid (edges/counts/voxel size) for a set of localisations."""
    lo = np.nanmin(all_locs, axis=0)
    hi = np.nanmax(all_locs, axis=0)
    span = hi - lo
    if float(span[2]) <= 1.0:
        raise ValueError("The data does not have a usable 3-D Z range.")
    pad = np.maximum(span * 0.02, 0.5)
    lo = lo - pad
    hi = hi + pad
    span = np.maximum(hi - lo, 1.0)
    xy_voxel = max(float(xy_voxel_nm), 0.001)
    z_voxel = max(float(z_voxel_nm), 0.001)
    voxel_req = np.array([xy_voxel, xy_voxel, z_voxel], dtype=np.float64)
    max_dim = max(int(max_dim), 8)
    max_voxels = max(int(max_voxels), 8)
    counts = np.maximum(np.ceil(span / voxel_req).astype(int), 2)
    scale = max(
        float(counts.max()) / float(max_dim),
        float(np.prod(counts, dtype=np.float64) / max_voxels) ** (1.0 / 3.0),
        1.0,
    )
    if scale > 1.0:
        voxel_req *= scale * 1.01
        counts = np.maximum(np.ceil(span / voxel_req).astype(int), 2)
    edges = [np.linspace(float(lo[i]), float(hi[i]), int(counts[i]) + 1, dtype=np.float64)
             for i in range(3)]
    voxel_xyz = tuple(float(edges[i][1] - edges[i][0]) for i in range(3))
    return edges, counts, voxel_xyz


def make_multichannel_volume_payload(
    channel_locs_nm: list,
    channel_rgb: list,
    *,
    xy_voxel_nm: float,
    z_voxel_nm: float,
    max_dim: int,
    max_voxels: int = _DEFAULT_MAX_VOXELS,
    opacity: float = 0.45,
    sigma_nm_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    black_pct: float = 0.0,
    white_pct: float = 99.7,
    channel_contrast_pct: list[tuple[float, float]] | None = None,
) -> VolumePayload:
    """Composite several overlay channels into one RGBA volume.

    Each channel's ``(N, 3)`` nm localisations are voxelised on a **shared** grid,
    normalised to its own percentile, and **additively coloured** by its LUT
    colour (so e.g. red + green overlap → yellow). Per-voxel alpha is the maximum
    channel response, so a voxel visible in any channel shows.
    """
    valid = []
    for locs, rgb in zip(channel_locs_nm, channel_rgb):
        arr = np.asarray(locs, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] > 0:
            valid.append((arr, np.asarray(rgb, dtype=np.float64).ravel()[:3]))
    if not valid:
        raise ValueError("No localisations pass the current filter.")

    all_locs = np.vstack([a for a, _ in valid])
    edges, counts, voxel_xyz = _volume_grid(
        all_locs, xy_voxel_nm, z_voxel_nm, max_dim, max_voxels)
    shape = (int(counts[0]), int(counts[1]), int(counts[2]))

    sigma_px = []
    for sigma_nm, step_nm in zip(sigma_nm_xyz, voxel_xyz):
        sigma_px.append(max(float(sigma_nm) / max(step_nm, 1e-12), 0.0)
                        if sigma_nm > 0.0 else 0.75)

    channel_norms: list[np.ndarray] = []
    scalar_sum = np.zeros(shape, dtype=np.float32)
    n_total = 0
    op = float(np.clip(opacity, 0.0, 1.0))
    for channel_index, (locs, _rgb) in enumerate(valid):
        hist, _ = np.histogramdd(locs, bins=edges)
        vol = hist.astype(np.float32, copy=False)
        if max(sigma_px) >= 0.1:
            vol = gaussian_filter(vol, sigma=tuple(sigma_px), mode="constant")
        if channel_contrast_pct is not None and channel_index < len(channel_contrast_pct):
            channel_black, channel_white = channel_contrast_pct[channel_index]
        else:
            channel_black, channel_white = black_pct, white_pct
        norm, _vhi = _normalize_volume(vol, channel_black, channel_white)
        channel_norms.append(np.rint(norm * 255.0).astype(np.uint8, copy=False))
        scalar_sum += vol
        n_total += int(locs.shape[0])

    rgba = _compose_multichannel_rgba(
        channel_norms, [rgb for _, rgb in valid], op
    )

    smax = float(scalar_sum.max()) if scalar_sum.size else 1.0
    norm_out = np.clip(scalar_sum / max(smax, 1e-12), 0.0, 1.0)
    return VolumePayload(
        scalar=np.ascontiguousarray(scalar_sum, dtype=np.float32),
        norm=np.ascontiguousarray(norm_out, dtype=np.float32),
        rgba=np.ascontiguousarray(rgba, dtype=np.uint8),
        origin_nm=(float(edges[0][0]), float(edges[1][0]), float(edges[2][0])),
        voxel_nm=voxel_xyz,
        counts=shape,
        n_locs=n_total,
        intensity_max=max(smax, 1e-12),
        channel_norms=tuple(np.ascontiguousarray(norm) for norm in channel_norms),
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
        render_method: str | None = None,
        region_bounds: tuple[float, float, float, float, float, float] | None = None,
        contrast_pct: tuple[float, float] | None = None,
        display_state: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._idx = int(dataset_idx)
        self._sigma_nm_xyz = sigma_nm_xyz
        # Volume brightness/contrast as (black, white) percentiles of the nonzero
        # voxels. Default (0, 99.7) = the historical auto-normalisation; the
        # advanced view passes its 2-D contrast so the 3-D visibility matches.
        self._black_pct, self._white_pct = contrast_pct or (0.0, 99.7)
        # Advanced view passes these: which 2-D method to reflect in 3-D, and a
        # 3-D box (xlo,xhi,ylo,yhi,zlo,zhi nm) to restrict the volume to the
        # currently focused region. Both None → the standard whole-data volume.
        self._render_method = render_method
        self._region_bounds = region_bounds
        self._display_state = display_state
        self._volume_channel_ids: tuple[int, ...] = ()
        self._overlay_transforms: dict[int, object] = {}
        # Cached GPU 3-D texture limit (see _gl_max_3d_texture); the Max dim
        # spinbox is constrained by it on first refresh, then user-adjustable.
        self._max_dim: int | None = None
        self._max_dim_initialised = False
        self._view = None
        self._volume_item = None
        self._mip_items: list = []
        self._surface_item = None
        self._overlay_items: list = []
        self._payload: VolumePayload | None = None
        self._lut_dialog = None
        self._lut_invert = False
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

        self.capture_spatial_state()
        self._build_ui()
        self._apply_display_state_to_controls(display_state)
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
        self._xy_voxel_spin.setValue(1.0)
        self._xy_voxel_spin.valueChanged.connect(self._schedule_rebuild)
        bar.addWidget(self._xy_voxel_spin)

        bar.addWidget(QLabel("Z voxel"))
        self._z_voxel_spin = QDoubleSpinBox()
        self._z_voxel_spin.setRange(0.1, 10_000.0)
        self._z_voxel_spin.setDecimals(1)
        self._z_voxel_spin.setSuffix(" nm")
        self._z_voxel_spin.setValue(1.0)
        self._z_voxel_spin.valueChanged.connect(self._schedule_rebuild)
        bar.addWidget(self._z_voxel_spin)

        bar.addWidget(QLabel("Max dim"))
        self._max_dim_spin = QSpinBox()
        self._max_dim_spin.setRange(32, 4096)      # narrowed to the GPU limit on first refresh
        self._max_dim_spin.setValue(1024)
        self._max_dim_spin.setToolTip(
            "Cap on the number of voxels along the LONGEST axis. Starts at 1024\n"
            "and is limited by the GPU's max 3-D texture size (up to 4096); raise\n"
            "for finer detail on an\n"
            "elongated or focused region, lower to save memory.\n"
            f"The TOTAL voxel count is also capped at {_DEFAULT_MAX_VOXELS:,} — for a\n"
            "big whole-field volume that usually binds first, so raising this may\n"
            "not change resolution (zoom into a region instead)."
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
        self._iso_spin.setToolTip(
            "Isosurface threshold on normalized voxel density (0.01-0.99).\n"
            "Enabled in Surface and ISO surface modes. Lower values include\n"
            "more low-density structure; higher values keep only dense cores."
        )
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

    def _display_channel_specs(self) -> list[dict]:
        if not self._display_state:
            return []
        return [
            spec for spec in self._display_state.get("channels", [])
            if isinstance(spec, dict)
        ]

    def _display_channel_map(self) -> dict[int, dict]:
        out = {}
        for spec in self._display_channel_specs():
            try:
                out[int(spec["dataset_idx"])] = spec
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def _display_visible_ids(self) -> tuple[int, ...]:
        return tuple(
            dataset_idx
            for dataset_idx, spec in self._display_channel_map().items()
            if spec.get("kind", "localizations") == "localizations"
            and bool(spec.get("visible", True))
        )

    def _display_contrast(self, dataset_idx: int) -> tuple[float, float] | None:
        spec = self._display_channel_map().get(int(dataset_idx))
        value = spec.get("contrast_pct") if spec else None
        if value is None:
            return None
        try:
            return float(value[0]), float(value[1])
        except (IndexError, TypeError, ValueError):
            return None

    def _apply_display_state_to_controls(self, display_state: dict | None) -> None:
        """Apply a 2-D snapshot to controls without starting a rebuild."""
        if display_state is None:
            return
        self._lut_invert = bool(display_state.get("invert", False))
        spec = self._display_channel_map().get(self._idx)
        if spec is None:
            return
        lut = str(spec.get("lut") or self._cmap_combo.currentText())
        if self._cmap_combo.findText(lut) < 0:
            self._cmap_combo.addItem(lut)
        self._cmap_combo.blockSignals(True)
        self._cmap_combo.setCurrentText(lut)
        self._cmap_combo.blockSignals(False)
        contrast = self._display_contrast(self._idx)
        if contrast is not None:
            self._black_pct, self._white_pct = contrast

    def capture_spatial_state(self) -> None:
        """Capture overlay transforms for an explicit 3-D refresh.

        Automatic display synchronization must not turn a later 2-D alignment
        edit into an implicit 3-D spatial refresh.
        """
        from ..core.overlay import overlay_members

        self._overlay_transforms = {}
        if not (0 <= self._idx < len(self._state.datasets)):
            return
        for dataset_idx, ds in overlay_members(self._state, self._idx):
            transform = ds.state.get("overlay_transform") or ds.state.get(
                "render_transform_2d"
            )
            if transform is not None:
                self._overlay_transforms[int(dataset_idx)] = deepcopy(transform)

    def _overlay_channels(self):
        """Visible overlay channels as ``[(locs_nm, rgb01), …]`` (display-space,
        each channel's transform applied), or ``None`` when the dataset is not a
        multi-channel overlay."""
        from ..core.overlay import (
            apply_display_transform_nm, dataset_group_id, overlay_members)

        if not (0 <= self._idx < len(self._state.datasets)):
            return None
        if not dataset_group_id(self._state.datasets[self._idx]):
            return None
        members = overlay_members(self._state, self._idx)
        if len(members) < 2:
            return None
        out = []
        display_map = self._display_channel_map()
        self._volume_channel_ids = ()
        channel_ids = []
        for dataset_idx, ds in members:
            spec = display_map.get(int(dataset_idx))
            if spec is not None:
                if spec.get("kind", "localizations") != "localizations":
                    continue
                if not bool(spec.get("visible", True)):
                    continue
            locs = _finite_filtered_locs(ds)
            if locs.shape[0] == 0:
                continue
            transform = self._overlay_transforms.get(int(dataset_idx))
            if transform is None:
                transform = ds.state.get("overlay_transform") or ds.state.get(
                    "render_transform_2d"
                )
            locs = apply_display_transform_nm(
                locs, transform
            )
            lut = (
                str(spec.get("lut")) if spec is not None and spec.get("lut")
                else str(ds.state.get("overlay_lut")
                         or ds.state.get("render_channel_lut") or "Gray")
            )
            rgb = lut_rgb(lut)
            out.append((np.ascontiguousarray(locs[:, :3], dtype=np.float64), rgb))
            channel_ids.append(int(dataset_idx))
        self._volume_channel_ids = tuple(channel_ids)
        return out or None

    def _gl_max_3d_texture(self) -> int:
        """The GPU's `GL_MAX_3D_TEXTURE_SIZE` — the hard per-axis voxel limit
        (a longer axis fails to upload → black window). Queried once from the
        live GL context; 2048 fallback (the OpenGL 4.3 guaranteed minimum, and
        the value on essentially every desktop GPU that runs this app)."""
        if self._max_dim is not None:
            return self._max_dim
        limit = 2048
        try:
            if self._view is not None:
                from OpenGL.GL import GL_MAX_3D_TEXTURE_SIZE, glGetIntegerv
                self._view.makeCurrent()
                try:
                    value = glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE)
                finally:
                    self._view.doneCurrent()
                gl_max = int(np.asarray(value).ravel()[0])
                if gl_max >= 256:
                    limit = gl_max
        except Exception:
            limit = 2048
        self._max_dim = int(limit)
        return self._max_dim

    def _recommended_max_dim(self) -> int:
        """Default Max dim: 1024 unless the GPU requires a lower value."""
        return int(min(self._gl_max_3d_texture(), 1024))

    def _single_channel_locs(self, ds):
        """Filtered finite native-XYZ locs, plus aligned per-loc precision sigma
        (N,3 nm) when the reflected method is the precision Gaussian, else None."""
        spec = self._display_channel_map().get(self._idx)
        if spec is not None and not bool(spec.get("visible", True)):
            return np.empty((0, 3), dtype=np.float64), None
        try:
            full = np.asarray(ds.loc_nm, dtype=np.float64)
        except Exception:
            return np.empty((0, 3), dtype=np.float64), None
        if full.ndim != 2 or full.shape[1] < 2:
            return np.empty((0, 3), dtype=np.float64), None
        if full.shape[1] == 2:
            full = np.column_stack([full, np.zeros(full.shape[0], dtype=np.float64)])
        n = full.shape[0]
        mask = np.asarray(ds.filter_mask, dtype=bool)
        if mask.shape[0] != n:
            mask = np.ones(n, dtype=bool)
        mask = mask & np.all(np.isfinite(full[:, :3]), axis=1)
        locs = np.ascontiguousarray(full[mask, :3], dtype=np.float64)
        if (self._render_method or "").lower() != "precision_gaussian":
            return locs, None
        try:
            from .precision_render import resolve_precision_xyz_nm
            sigma, _src = resolve_precision_xyz_nm(ds, n)   # (n,3) native nm, RIMF on z
            return locs, np.ascontiguousarray(sigma[mask], dtype=np.float64)
        except Exception:
            return locs, None

    def refresh_from_dataset(self) -> None:
        if self._idx < 0 or self._idx >= len(self._state.datasets):
            self._info_label.setText("No active dataset.")
            return
        ds = self._state.datasets[self._idx]
        channels = self._overlay_channels()
        precision_sigma = None
        if channels:
            self.setWindowTitle(
                f"3D Volume Preview - {ds.name} (+{len(channels) - 1} channel(s))")
            all_locs = np.vstack([c[0] for c in channels])
        else:
            self._volume_channel_ids = ()
            self.setWindowTitle(f"3D Volume Preview - {ds.name}")
            all_locs, precision_sigma = self._single_channel_locs(ds)
        # Restrict to the focused 3-D region (from the advanced 2-D view) if set.
        if self._region_bounds is not None and all_locs.shape[0] > 0:
            all_locs, precision_sigma = _clip_region(
                all_locs, precision_sigma, self._region_bounds
            )
        # Default to an isotropic 1 nm voxel (the total-voxel cap enlarges it
        # if the region is too big); only seed the spinboxes once.
        if not getattr(self, "_voxel_initialised", False):
            self._xy_voxel_spin.setValue(1.0)
            self._z_voxel_spin.setValue(1.0)
            self._voxel_initialised = True
        # Populate Max dim from the GPU limit once the GL context exists (widen
        # the range to the hard limit, default to the recommended value).
        if not getattr(self, "_max_dim_initialised", False) and self._view is not None:
            self._max_dim_initialised = True
            hard = self._gl_max_3d_texture()
            self._max_dim_spin.blockSignals(True)
            self._max_dim_spin.setRange(32, min(max(hard, 64), 4096))
            self._max_dim_spin.setValue(self._recommended_max_dim())
            self._max_dim_spin.blockSignals(False)

        if self._view is None:
            self._info_label.setText("3D OpenGL preview unavailable.")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if channels:
                channel_contrast_pct = [
                    self._display_contrast(dataset_idx)
                    or (self._black_pct, self._white_pct)
                    for dataset_idx in self._volume_channel_ids
                ]
                payload = make_multichannel_volume_payload(
                    [c[0] for c in channels], [c[1] for c in channels],
                    xy_voxel_nm=float(self._xy_voxel_spin.value()),
                    z_voxel_nm=float(self._z_voxel_spin.value()),
                    max_dim=int(self._max_dim_spin.value()),
                    max_voxels=_DEFAULT_MAX_VOXELS,
                    opacity=float(self._opacity_spin.value()),
                    sigma_nm_xyz=self._sigma_nm_xyz,
                    black_pct=self._black_pct,
                    white_pct=self._white_pct,
                    channel_contrast_pct=channel_contrast_pct,
                )
            else:
                contrast = self._display_contrast(self._idx)
                if contrast is not None:
                    self._black_pct, self._white_pct = contrast
                payload = make_volume_payload(
                    all_locs,
                    xy_voxel_nm=float(self._xy_voxel_spin.value()),
                    z_voxel_nm=float(self._z_voxel_spin.value()),
                    max_dim=int(self._max_dim_spin.value()),
                    max_voxels=_DEFAULT_MAX_VOXELS,
                    cmap_name=self._cmap_combo.currentText(),
                    opacity=float(self._opacity_spin.value()),
                    sigma_nm_xyz=self._sigma_nm_xyz,
                    render_method=self._render_method,
                    precision_sigma_nm=precision_sigma,
                    black_pct=self._black_pct,
                    white_pct=self._white_pct,
                    invert=self._lut_invert,
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
        ch_txt = f"  |  {len(channels)} channels" if channels else ""
        self._info_label.setText(
            f"{payload.n_locs:,} filtered locs{ch_txt}  |  volume {nx} x {ny} x {nz}  |  "
            f"voxel=({vx:.1f}, {vy:.1f}, {vz:.1f}) nm  |  "
            f"mode={self._mode_combo.currentText()}  |  intensity max~{payload.intensity_max:.3g}"
        )

    def _recompose_single_payload(self) -> bool:
        payload = self._payload
        if payload is None or self._volume_channel_ids:
            return False
        contrast = self._display_contrast(self._idx)
        black, white = contrast or (self._black_pct, self._white_pct)
        norm, vmax = _normalize_volume(payload.scalar, black, white)
        rgba = _matplotlib_rgba(
            self._cmap_combo.currentText(), norm.ravel(), invert=self._lut_invert
        ).reshape((*norm.shape, 4))
        alpha = (
            np.power(norm, 0.75)
            * np.clip(float(self._opacity_spin.value()), 0.0, 1.0)
            * 255.0
        ).astype(np.uint8)
        rgba[..., 3] = alpha
        rgba[norm <= 0.0, 3] = 0
        self._black_pct, self._white_pct = float(black), float(white)
        self._payload = VolumePayload(
            scalar=payload.scalar,
            norm=norm,
            rgba=np.ascontiguousarray(rgba, dtype=np.uint8),
            origin_nm=payload.origin_nm,
            voxel_nm=payload.voxel_nm,
            counts=payload.counts,
            n_locs=payload.n_locs,
            intensity_max=vmax,
        )
        self._set_render_item(self._payload)
        return True

    def _recompose_overlay_payload(self) -> bool:
        payload = self._payload
        if payload is None or not payload.channel_norms:
            return False
        norm_by_id = {
            dataset_idx: norm
            for dataset_idx, norm in zip(
                self._volume_channel_ids, payload.channel_norms
            )
        }
        visible_ids = self._display_visible_ids()
        if not visible_ids or any(dataset_idx not in norm_by_id for dataset_idx in visible_ids):
            return False
        specs = self._display_channel_map()
        norms = [norm_by_id[dataset_idx] for dataset_idx in visible_ids]
        rgbs = [lut_rgb(str(specs[dataset_idx].get("lut") or "Gray"))
                for dataset_idx in visible_ids]
        rgba = _compose_multichannel_rgba(
            norms, rgbs, float(self._opacity_spin.value())
        )
        self._payload = VolumePayload(
            scalar=payload.scalar,
            norm=payload.norm,
            rgba=rgba,
            origin_nm=payload.origin_nm,
            voxel_nm=payload.voxel_nm,
            counts=payload.counts,
            n_locs=payload.n_locs,
            intensity_max=payload.intensity_max,
            channel_norms=payload.channel_norms,
        )
        self._set_render_item(self._payload)
        return True

    def sync_from_2d(
        self, display_state: dict | None, *, render_method: str | None = None
    ) -> None:
        """Synchronize display state from the linked 2-D render.

        LUT and visibility changes use cached channel responses. B/C changes in
        an overlay, and any render-method change, are coalesced into one volume
        rebuild because they change the voxel scalar field. Camera/FOV and
        alignment are intentionally absent from this interface.
        """
        old_state = self._display_state
        old_visible_ids = (
            self._volume_channel_ids
            if self._volume_channel_ids
            else self._display_visible_ids()
        )
        old_map = self._display_channel_map()
        old_method = self._render_method
        self._display_state = display_state
        self._apply_display_state_to_controls(display_state)
        new_map = self._display_channel_map()
        new_visible_ids = self._display_visible_ids()
        if render_method is not None:
            self._render_method = render_method
        method_changed = render_method is not None and render_method != old_method
        if method_changed:
            self._schedule_rebuild()
            return
        if self._payload is None:
            return

        ids_changed = tuple(old_visible_ids) != tuple(new_visible_ids)
        ids_to_compare = set(old_map) | set(new_map)
        contrast_changed = any(
            old_map.get(dataset_idx, {}).get("contrast_pct")
            != new_map.get(dataset_idx, {}).get("contrast_pct")
            for dataset_idx in ids_to_compare
        )
        lut_changed = any(
            old_map.get(dataset_idx, {}).get("lut")
            != new_map.get(dataset_idx, {}).get("lut")
            for dataset_idx in ids_to_compare
        )
        if old_state is None:
            ids_changed = True

        if self._volume_channel_ids:
            if contrast_changed:
                self._schedule_rebuild()
            elif ids_changed or lut_changed:
                if not self._recompose_overlay_payload():
                    self._schedule_rebuild()
        elif contrast_changed or lut_changed:
            self._recompose_single_payload()

    def _schedule_rebuild(self, *_args) -> None:
        if not getattr(self, "_voxel_initialised", False):
            return
        self._rebuild_timer.start()

    def set_contrast_pct(self, black: float, white: float) -> None:
        """Seed the volume brightness/contrast (black/white percentiles) — used
        by the advanced view to match the 2-D contrast. The values are retained
        for LUT/Brightness-Contrast dialogs, not exposed as duplicate inline
        controls."""
        self._black_pct = float(black)
        self._white_pct = float(white)

    def _on_mode_changed(self, mode: str) -> None:
        self._iso_spin.setEnabled(mode in {"Surface", "ISO surface"})
        self._schedule_rebuild()

    def _lut_values(self) -> np.ndarray:
        payload = self._payload
        if payload is None:
            return np.empty(0, dtype=float)
        values = np.asarray(payload.scalar, dtype=float).ravel()
        return values[np.isfinite(values)]

    def open_lut_dialog(self) -> None:
        """Open the shared LUT editor for the 3-D volume view."""
        values = self._lut_values()
        if values.size == 0:
            self.refresh_from_dataset()
            values = self._lut_values()
        if values.size == 0:
            self._info_label.setText("LUT unavailable: no volume values to display.")
            return

        from .lut_dialog import LutDialog

        if self._lut_dialog is None:
            self._lut_dialog = LutDialog(
                on_levels_changed=self._on_lut_levels_changed,
                on_cmap_changed=self._on_lut_cmap_changed,
                on_invert_changed=self._on_lut_invert_changed,
                parent=self,
            )

        data_lo = float(values.min())
        data_hi = float(values.max())
        if data_hi <= data_lo:
            data_hi = data_lo + 1.0
        nonzero = values[values > 0.0]
        sample = nonzero if nonzero.size else values
        lo = float(np.percentile(sample, np.clip(self._black_pct, 0.0, 100.0)))
        hi = float(np.percentile(sample, np.clip(self._white_pct, 0.0, 100.0)))
        if hi <= lo:
            hi = lo + 1.0
        self._lut_dialog.load_image(
            pixels=values,
            data_lo=data_lo,
            data_hi=data_hi,
            lo=lo,
            hi=hi,
            cmap_name=self._cmap_combo.currentText(),
            invert=self._lut_invert,
        )
        self._lut_dialog.show()
        self._lut_dialog.raise_()
        self._lut_dialog.activateWindow()

    def _on_lut_levels_changed(self, lo: float, hi: float) -> None:
        values = self._lut_values()
        if values.size == 0:
            return
        sample = values[values > 0.0]
        if sample.size == 0:
            sample = values
        self._black_pct = float(np.mean(sample <= float(lo)) * 100.0)
        self._white_pct = float(np.mean(sample <= float(hi)) * 100.0)
        self._white_pct = max(self._white_pct, self._black_pct + 0.1)
        self._white_pct = min(self._white_pct, 100.0)
        self.refresh_from_dataset()

    def _on_lut_cmap_changed(self, name: str, invert: bool) -> None:
        self._lut_invert = bool(invert)
        if 0 <= self._idx < len(self._state.datasets):
            ds = self._state.datasets[self._idx]
            ds.state["render_channel_lut"] = name
            ds.state["overlay_lut"] = name
        if self._cmap_combo.findText(name) < 0:
            self._cmap_combo.addItem(name)
        self._cmap_combo.blockSignals(True)
        self._cmap_combo.setCurrentText(name)
        self._cmap_combo.blockSignals(False)
        self.refresh_from_dataset()

    def _on_lut_invert_changed(self, invert: bool) -> None:
        self._on_lut_cmap_changed(self._cmap_combo.currentText(), invert)

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
        color = _surface_color(
            self._cmap_combo.currentText(), level,
            float(self._opacity_spin.value()), invert=self._lut_invert,
        )
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
