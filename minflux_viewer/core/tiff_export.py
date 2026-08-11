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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

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


def write_ome_tiff(
    path: str | Path,
    data,
    *,
    axes: str,
    shape: tuple[int, ...],
    dtype,
    pixel_size_x_nm: float | None = None,
    pixel_size_y_nm: float | None = None,
    pixel_size_z_nm: float | None = None,
    channel_names: Sequence[str] | None = None,
    image_name: str | None = None,
    acquisition_date: str | None = None,
    description: str | None = None,
    time_interval: tuple[float, str] | None = None,
    source_metadata: dict[str, str] | None = None,
    roi=None,
) -> None:
    """Write *data* as an OME-TIFF with physical calibration in nm.

    *data* is anything ``tifffile`` accepts for one image series — an array or a
    generator of pages matching *shape*.  Pixel sizes go out as OME
    ``PhysicalSizeX/Y/Z`` **and** as TIFF resolution tags in pixels per
    centimetre, which is what lets :mod:`minflux_viewer.core.tiff_source` read
    the calibration back and what ImageJ/Fiji reads.

    *roi* — a :class:`~minflux_viewer.core.tiff_roi.TiffRoi` or raw ImageJ
    ``.roi`` bytes — is stored as the image's **active ROI** in the ImageJ
    metadata tag, so ImageJ/Fiji and our own viewer both restore it on open.

    This is the single TIFF-writing path: the localization render export and the
    OBF image-series export both go through it, so their output carries the same
    metadata and BigTIFF handling.
    """
    import tifffile

    from .tiff_roi import TIFF_BYTEORDER, roi_extratags

    metadata: dict[str, object] = {"axes": axes}
    for key, value in (("X", pixel_size_x_nm), ("Y", pixel_size_y_nm),
                       ("Z", pixel_size_z_nm)):
        if value:
            metadata[f"PhysicalSize{key}"] = float(value)
            metadata[f"PhysicalSize{key}Unit"] = "nm"
    if channel_names:
        metadata["Channel"] = {"Name": list(channel_names)}
    if image_name:
        metadata["Name"] = str(image_name)
    if acquisition_date:
        metadata["AcquisitionDate"] = str(acquisition_date)
    if description:
        metadata["Description"] = str(description)
    if time_interval and time_interval[0] is not None:
        metadata["TimeIncrement"] = float(time_interval[0])
        metadata["TimeIncrementUnit"] = str(time_interval[1] or "s")
    if source_metadata:
        from .tiff_source import SOURCE_METADATA_NAMESPACE

        metadata["MapAnnotation"] = {
            "Namespace": SOURCE_METADATA_NAMESPACE,
            "Value": {str(key): str(value) for key, value in source_metadata.items() if value},
        }

    itemsize = int(np.dtype(dtype).itemsize)
    bigtiff = int(np.prod(shape)) * itemsize > _BIGTIFF_THRESHOLD_BYTES

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_kwargs: dict[str, object] = {}
    if pixel_size_x_nm:
        # Resolution in pixels per centimeter (1 cm = 1e7 nm) for ImageJ/Fiji.
        px_per_cm = 1.0e7 / float(pixel_size_x_nm)
        py_per_cm = 1.0e7 / float(pixel_size_y_nm or pixel_size_x_nm)
        write_kwargs["resolution"] = (px_per_cm, py_per_cm)
        write_kwargs["resolutionunit"] = "CENTIMETER"
    extratags = roi_extratags(roi, TIFF_BYTEORDER)
    if extratags:
        write_kwargs["extratags"] = extratags
    # Byte order is pinned so the ImageJ ROI block (read with the file's order)
    # always matches what roi_extratags encoded.
    with tifffile.TiffWriter(str(path), ome=True, bigtiff=bigtiff,
                             byteorder=TIFF_BYTEORDER) as writer:
        writer.write(
            data,
            shape=shape,
            dtype=dtype,
            photometric="minisblack",
            metadata=metadata,
            **write_kwargs,
        )


#: ``roi=`` default for :func:`export_image_series_to_tiff`: take whatever ROI
#: the source carries. Distinct from ``roi=None``, which means "write no ROI" —
#: without the distinction, deleting a ROI and saving silently rewrote the
#: source's own ROI back into the file.
INHERIT_SOURCE_ROI = object()


def export_image_series_to_tiff(source, path: str | Path, *,
                                roi=INHERIT_SOURCE_ROI) -> TiffExportResult:
    """Write the **current series** of an image *source* to an OME-TIFF.

    *source* is anything with the reader interface
    (:class:`~minflux_viewer.core.obf_image_source.ObfImageSource` or
    :class:`~minflux_viewer.core.tiff_source.TiffImageSource`): a
    ``TiffMetadata`` plus ``read_plane``.  The stack is copied through
    unchanged — this is a format conversion, not a rendering — with the source's
    own pixel calibration carried into the OME metadata.

    *roi* is the active ROI to embed. Left at :data:`INHERIT_SOURCE_ROI` the
    source's own ``active_roi()`` is carried across, so a conversion does not
    silently drop the ROI it already had; pass ``roi=None`` to write **no** ROI.
    """
    meta = source.metadata
    if roi is INHERIT_SOURCE_ROI:
        getter = getattr(source, "active_roi", None)
        roi = getter() if callable(getter) else None
    axes = str(meta.axes)
    shape = tuple(int(v) for v in meta.shape)
    n_z = int(meta.axis_size("Z")) if "Z" in axes else 1
    page_axes = [axis for axis in axes if axis in "TCZ"]
    page_ranges = [range(int(meta.axis_size(axis))) for axis in page_axes]
    page_count = int(np.prod([len(values) for values in page_ranges])) if page_ranges else 1

    def _pages():
        indices_iter = product(*page_ranges) if page_ranges else [()]
        for indices in indices_iter:
            selected = dict(zip(page_axes, indices))
            yield np.asarray(source.read_plane(
                t=selected.get("T", 0),
                c=selected.get("C", 0),
                z=selected.get("Z", 0),
            ))

    first = np.asarray(source.read_plane())
    dtype = first.dtype
    # A single plane is written directly; tifffile wants the generator form only
    # when the series spans several pages.
    payload = _pages() if page_count > 1 else first
    source_metadata = _source_metadata_for_export(getattr(meta, "documents", ()))
    interval = None
    time_interval = getattr(meta, "time_interval", None)
    if time_interval is not None and time_interval.value is not None:
        interval = (float(time_interval.value), time_interval.unit or "s")
    write_ome_tiff(
        path, payload, axes=axes, shape=shape, dtype=dtype,
        pixel_size_x_nm=meta.pixel_size_x.nm,
        pixel_size_y_nm=meta.pixel_size_y.nm,
        pixel_size_z_nm=meta.pixel_size_z.nm,
        channel_names=list(meta.channel_names) or None,
        image_name=getattr(meta, "image_name", "") or None,
        acquisition_date=getattr(meta, "acquisition_date", "") or None,
        description=getattr(meta, "description", "") or None,
        time_interval=interval,
        source_metadata=source_metadata or None,
        roi=roi,
    )
    return TiffExportResult(
        path=str(path), axes=axes, shape=shape, dtype=np.dtype(dtype).name,
        max_count=0, n_channels=int(meta.axis_size("C")) if "C" in axes else 1,
        n_slices=n_z,
    )


def _source_metadata_for_export(documents) -> dict[str, str]:
    """Map reader documents to named OME MapAnnotation entries.

    The generated OME document describes the exported pixels. The original
    per-image OME and Imspector documents are retained as provenance rather
    than replacing that generated calibration block.
    """
    from .tiff_source import (
        SOURCE_IMSPECTOR_XML_KEY,
        SOURCE_MINFLUX_JSON_KEY,
        SOURCE_OBF_JSON_KEY,
        SOURCE_OME_XML_KEY,
    )

    by_name = {str(doc.name): str(doc.content) for doc in documents or () if doc.content}
    original_ome = by_name.get("Original OME-XML") or by_name.get("OME-XML")
    result: dict[str, str] = {}
    for key, value in (
        (SOURCE_OME_XML_KEY, original_ome),
        (SOURCE_IMSPECTOR_XML_KEY, by_name.get("Imspector XML")),
        (SOURCE_OBF_JSON_KEY, by_name.get("OBF stack metadata")),
        (SOURCE_MINFLUX_JSON_KEY, by_name.get("MINFLUX stack tag")),
    ):
        if value:
            result[key] = value
    return result


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

    write_ome_tiff(
        path, _pages(), axes=axes, shape=shape, dtype=dtype,
        pixel_size_x_nm=pixel_size_nm, pixel_size_y_nm=pixel_size_nm,
        pixel_size_z_nm=voxel_depth_nm if use_z else None,
        channel_names=[cb.name for cb in binned] if multi else None,
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
