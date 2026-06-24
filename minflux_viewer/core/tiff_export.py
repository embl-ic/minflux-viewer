"""Render localizations to a multipage (OME-)TIFF stack.

Reproduces the Z-slicing histogram export of the MATLAB
``resources/samples/interactive_render_MINFLUX.m`` script: the localization
volume is binned into voxels (XY = pixel size, Z = voxel depth); each Z slice is
the 2-D histogram of the localizations gated to that slice's depth. The bit
depth is chosen from the **global** maximum voxel count over the whole stack
(8 / 16 / 32-bit), so re-scanning the data is not needed after the fact.

Physical calibration is written as OME-TIFF ``PhysicalSizeX/Y/Z`` (nm) plus TIFF
resolution tags, so the viewer's TIFF reader (`core/tiff_source.py`) round-trips
the pixel/voxel size.

Pure NumPy + tifffile, no Qt — unit-tested in ``tests/test_tiff_export.py`` and
driven from the render window's "Export to TIFF…" action via a background
worker (`ui/tiff_export_dialog.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

UINT8_MAX = 255
UINT16_MAX = 65535
#: Single-file payloads above this many bytes need the BigTIFF format.
_BIGTIFF_THRESHOLD_BYTES = int(3.9 * 1024**3)
#: Reject grids whose per-axis bin count exceeds this (guards against a
#: pixel/voxel size far too small for the data extent → out-of-memory).
_MAX_AXIS_BINS = 100_000

#: Progress callback: ``progress(done_pages, total_pages, message)``.
ProgressFn = Callable[[int, int, str], None]


@dataclass
class TiffExportChannel:
    """One render channel to export: a display name and (N, 3) XYZ in nm."""

    name: str
    xyz: np.ndarray


@dataclass
class TiffExportResult:
    path: str
    axes: str
    shape: tuple[int, ...]
    dtype: str
    max_count: int
    n_channels: int
    n_slices: int


def build_edges(lo: float, hi: float, step: float) -> np.ndarray:
    """Histogram bin edges over ``[lo, hi]`` with bin width ``step`` (nm).

    Matches the MATLAB ``lo:step:hi+step`` convention: the number of bins is
    ``floor((hi - lo) / step) + 1`` so the maximum value always lands strictly
    inside the last bin (never on its right edge). A degenerate range collapses
    to a single bin.
    """
    step = float(step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be a positive, finite value")
    lo = float(lo)
    hi = float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.array([lo, lo + step], dtype=np.float64)
    nbins = int(np.floor((hi - lo) / step)) + 1
    return lo + np.arange(nbins + 1, dtype=np.float64) * step


def choose_dtype(max_count: int) -> type[np.unsignedinteger]:
    """Smallest unsigned integer dtype that holds ``max_count`` without clipping."""
    if max_count <= UINT8_MAX:
        return np.uint8
    if max_count <= UINT16_MAX:
        return np.uint16
    return np.uint32


def _bin_index(values: np.ndarray, edges: np.ndarray, nbins: int) -> np.ndarray:
    """0-based bin index for each value, clamped to ``[0, nbins - 1]``."""
    idx = np.searchsorted(edges, values, side="right") - 1
    return np.clip(idx, 0, nbins - 1).astype(np.int64, copy=False)


@dataclass
class _ChannelBins:
    """Per-channel digitized bin indices (in-plane + slice)."""

    name: str
    xb: np.ndarray
    yb: np.ndarray
    zb: np.ndarray


def _finite_xyz(xyz: np.ndarray) -> np.ndarray:
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.empty((0, 3), dtype=np.float64)
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.zeros(arr.shape[0], dtype=np.float64)])
    return arr[np.all(np.isfinite(arr[:, :3]), axis=1), :3]


def _resolve_span(rng, data_lo: float, data_hi: float) -> tuple[float, float]:
    """A user range ``(lo, hi)`` if given (sorted), else the data extent."""
    if rng is None:
        return float(data_lo), float(data_hi)
    lo, hi = float(rng[0]), float(rng[1])
    return (lo, hi) if hi >= lo else (hi, lo)


def export_render_to_tiff(
    channels: Sequence[TiffExportChannel],
    path: str | Path,
    *,
    pixel_size_nm: float,
    voxel_depth_nm: float,
    is_3d: bool,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    progress: ProgressFn | None = None,
) -> TiffExportResult:
    """Write the localization render of ``channels`` to a multipage OME-TIFF.

    All channels share one voxel grid so they stay aligned; the stack axes are
    ``YX`` / ``ZYX`` / ``CYX`` / ``CZYX`` depending on the channel count and
    whether the data is 3-D. ``x_range`` / ``y_range`` / ``z_range`` (nm) clip the
    exported region — localizations outside the box are dropped and the grid
    spans the range; when ``None`` the data extent on that axis is used. Pages
    are streamed one at a time (the full volume is never materialized), and
    ``progress`` — when given — is called once per written page.
    """
    import tifffile

    pixel_size_nm = float(pixel_size_nm)
    if not np.isfinite(pixel_size_nm) or pixel_size_nm <= 0.0:
        raise ValueError("pixel_size_nm must be a positive, finite value")

    finite = [(ch.name, _finite_xyz(ch.xyz)) for ch in channels]
    finite = [(name, xyz) for name, xyz in finite if xyz.shape[0] > 0]
    if not finite:
        raise ValueError("no finite localizations to export")

    all_xyz = np.vstack([xyz for _name, xyz in finite])
    data_mins = all_xyz.min(axis=0)
    data_maxs = all_xyz.max(axis=0)

    voxel_depth_nm = float(voxel_depth_nm)
    x_lo, x_hi = _resolve_span(x_range, data_mins[0], data_maxs[0])
    y_lo, y_hi = _resolve_span(y_range, data_mins[1], data_maxs[1])
    z_lo, z_hi = _resolve_span(z_range, data_mins[2], data_maxs[2])
    use_z = bool(is_3d) and voxel_depth_nm > 0.0 and (z_hi - z_lo) > 1.0

    # Clip each channel to the export box; drop channels left empty.
    clipped: list[tuple[str, np.ndarray]] = []
    for name, xyz in finite:
        m = (
            (xyz[:, 0] >= x_lo) & (xyz[:, 0] <= x_hi)
            & (xyz[:, 1] >= y_lo) & (xyz[:, 1] <= y_hi)
        )
        if use_z:
            m &= (xyz[:, 2] >= z_lo) & (xyz[:, 2] <= z_hi)
        sub = xyz[m]
        if sub.shape[0] > 0:
            clipped.append((name, sub))
    if not clipped:
        raise ValueError("no localizations fall inside the selected export range")
    finite = clipped

    x_edges = build_edges(x_lo, x_hi, pixel_size_nm)
    y_edges = build_edges(y_lo, y_hi, pixel_size_nm)
    z_edges = build_edges(z_lo, z_hi, voxel_depth_nm) if use_z else np.array(
        [z_lo, z_hi + 1.0], dtype=np.float64
    )
    nx = int(x_edges.size - 1)
    ny = int(y_edges.size - 1)
    nz = int(z_edges.size - 1) if use_z else 1

    if nx > _MAX_AXIS_BINS or ny > _MAX_AXIS_BINS or nz > _MAX_AXIS_BINS:
        raise ValueError(
            f"pixel/voxel size too small for the data extent "
            f"(image would be {nx} × {ny} × {nz} voxels); increase the pixel size."
        )

    # Digitize once per channel; reused for the max pre-pass and the write pass.
    binned: list[_ChannelBins] = []
    for name, xyz in finite:
        xb = _bin_index(xyz[:, 0], x_edges, nx)
        yb = _bin_index(xyz[:, 1], y_edges, ny)
        zb = _bin_index(xyz[:, 2], z_edges, nz) if use_z else np.zeros(xyz.shape[0], dtype=np.int64)
        binned.append(_ChannelBins(name=name, xb=xb, yb=yb, zb=zb))

    # Global max voxel count → bit depth (memory O(N), never builds the volume).
    max_count = 0
    for cb in binned:
        linear = (cb.xb * ny + cb.yb) * nz + cb.zb
        if linear.size:
            _uniq, counts = np.unique(linear, return_counts=True)
            max_count = max(max_count, int(counts.max()))
    dtype = choose_dtype(max_count)

    n_channels = len(binned)
    multi = n_channels > 1
    if multi:
        axes = "CZYX" if use_z else "CYX"
        shape: tuple[int, ...] = (n_channels, nz, ny, nx) if use_z else (n_channels, ny, nx)
    else:
        axes = "ZYX" if use_z else "YX"
        shape = (nz, ny, nx) if use_z else (ny, nx)

    total_pages = n_channels * nz

    def _slice_image(cb: _ChannelBins, k: int) -> np.ndarray:
        sel = cb.zb == k if use_z else slice(None)
        xb = cb.xb[sel]
        yb = cb.yb[sel]
        if xb.size == 0:
            return np.zeros((ny, nx), dtype=dtype)
        flat = np.bincount(xb * ny + yb, minlength=nx * ny)
        return flat.reshape(nx, ny).T.astype(dtype, copy=False)

    def _pages():
        done = 0
        for cb in binned:
            for k in range(nz):
                img = _slice_image(cb, k)
                done += 1
                if progress is not None:
                    progress(done, total_pages, f"channel '{cb.name}' slice {k + 1}/{nz}")
                yield img

    metadata: dict[str, object] = {
        "axes": axes,
        "PhysicalSizeX": pixel_size_nm,
        "PhysicalSizeXUnit": "nm",
        "PhysicalSizeY": pixel_size_nm,
        "PhysicalSizeYUnit": "nm",
    }
    if use_z:
        metadata["PhysicalSizeZ"] = voxel_depth_nm
        metadata["PhysicalSizeZUnit"] = "nm"
    if multi:
        metadata["Channel"] = {"Name": [cb.name for cb in binned]}

    itemsize = int(np.dtype(dtype).itemsize)
    bigtiff = (n_channels * nz * ny * nx * itemsize) > _BIGTIFF_THRESHOLD_BYTES

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Resolution in pixels per centimeter (1 cm = 1e7 nm) for ImageJ/Fiji.
    px_per_cm = 1.0e7 / pixel_size_nm
    with tifffile.TiffWriter(str(path), ome=True, bigtiff=bigtiff) as writer:
        writer.write(
            _pages(),
            shape=shape,
            dtype=dtype,
            photometric="minisblack",
            metadata=metadata,
            resolution=(px_per_cm, px_per_cm),
            resolutionunit="CENTIMETER",
        )

    return TiffExportResult(
        path=str(path),
        axes=axes,
        shape=shape,
        dtype=np.dtype(dtype).name,
        max_count=max_count,
        n_channels=n_channels,
        n_slices=nz,
    )
