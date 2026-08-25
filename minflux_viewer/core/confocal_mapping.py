"""Detect and map calibrated MSR confocal channels onto localizations.

Candidate discovery is deliberately conservative.  A scalar image stack is a
candidate only when it is not a known MINFLUX/generated stack and its complete
calibrated X/Y bounds match a selected dataset's acquisition-ROI union within
one percent on each axis.  The detector does not guess which fluorescent
channel is useful; it returns every geometric match for the user to choose.

Mapping is performed in the image's calibrated stage frame.  A 2-D mapping
uses either the image itself or a float64 Z sum; a 3-D mapping samples the
volume directly.  Out-of-bounds localizations receive NaN.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..msr.acquisition_roi import (
    AcquisitionRoi,
    group_by_dataset,
    read_acquisition_rois,
    union_bounds,
)
from ..msr.mfxdta import extract_did_label_map
from .obf_image_source import (
    ObfImageSource,
    axes_for_sizes,
    is_image_stack,
    scan_obf_stacks,
)

DEFAULT_GEOMETRY_TOLERANCE = 0.01

# Footer ``minflux.type`` is authoritative when present.  These name patterns
# cover older files whose generated stacks predate that footer tag.
_NON_CHANNEL_NAME = re.compile(
    r"(?:^mf\(|(?:^|[^a-z0-9])"
    r"(?:data|trace|density|histogram|population|localization|localisation|mask)"
    r"(?:[^a-z0-9]|$))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConfocalCandidateMatch:
    """One dataset whose acquisition ROI matches an image stack."""

    dataset_key: str
    did: str
    x_error_fraction: float
    y_error_fraction: float


@dataclass(frozen=True)
class ConfocalCandidate:
    """A calibrated scalar image stack that may be a fluorescent channel."""

    raw_index: int
    name: str
    shape: tuple[int, ...]
    axes: str
    dtype: str
    x_start_m: float
    y_start_m: float
    x_step_m: float
    y_step_m: float
    z_start_m: float | None
    z_step_m: float | None
    bounds_xy_m: tuple[tuple[float, float], tuple[float, float]]
    matches: tuple[ConfocalCandidateMatch, ...]

    @property
    def has_z(self) -> bool:
        return self.axes == "ZYX" and len(self.shape) == 3 and self.shape[0] > 1

    @property
    def matched_dataset_keys(self) -> tuple[str, ...]:
        return tuple(match.dataset_key for match in self.matches)


@dataclass(frozen=True)
class ConfocalMappingTransform:
    """Manual image-coordinate adjustment used for preview and sampling.

    Positive ``rotation_deg`` is visually counter-clockwise in the image's
    top-left coordinate system.  Translation is expressed in image pixels.
    """

    dx_pixels: float = 0.0
    dy_pixels: float = 0.0
    rotation_deg: float = 0.0


@dataclass(frozen=True)
class ConfocalMappingResult:
    attribute_name: str
    finite_count: int
    total_count: int
    raw_finite_count: int
    raw_total_count: int
    provenance: dict


def candidate_attribute_name(stack_name: str) -> str:
    """A compact editable default (``"Ch1 {12}"`` -> ``"Ch1"``)."""
    base = re.sub(r"\s*\{[^{}]*\}\s*$", "", str(stack_name)).strip()
    base = re.sub(r"[^0-9A-Za-z_]+", "_", base).strip("_") or "signal"
    if base[0].isdigit():
        base = f"signal_{base}"
    return base


def is_known_non_channel_stack(stack: Mapping) -> bool:
    """True for data/trace/density and other generated/non-scalar stacks."""
    if not is_image_stack(dict(stack)):
        return True
    if str(stack.get("minflux_type", "") or "").strip():
        return True
    axes = axes_for_sizes(tuple(int(v) for v in stack.get("sizes", ()) or ()))
    if axes not in {"YX", "ZYX"}:
        return True
    return bool(_NON_CHANNEL_NAME.search(str(stack.get("name", "") or "")))


def _finite_tuple(values) -> tuple[float, ...]:
    out: list[float] = []
    for value in () if values is None else values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ()
        if not np.isfinite(number):
            return ()
        out.append(number)
    return tuple(out)


def _stack_geometry(stack: Mapping) -> dict | None:
    """Calibrated array starts/steps and sorted XY bounds for one stack."""
    shape = tuple(int(v) for v in (stack.get("sizes") or ()))
    axes = axes_for_sizes(shape)
    if axes not in {"YX", "ZYX"} or any(v <= 0 for v in shape):
        return None
    ny, nx = shape[-2:]
    offset = _finite_tuple(stack.get("offset"))
    length = _finite_tuple(stack.get("length"))

    if len(offset) >= 2 and len(length) >= 2 and length[0] != 0.0 and length[1] != 0.0:
        x_start, y_start = offset[0], offset[1]
        x_step, y_step = length[0] / nx, length[1] / ny
    else:
        extent = stack.get("extent_m")
        try:
            (x_lo, x_hi), (y_lo, y_hi) = extent
            x_start, y_start = float(x_lo), float(y_lo)
            x_step, y_step = (float(x_hi) - x_start) / nx, (float(y_hi) - y_start) / ny
        except Exception:
            return None

    if not all(np.isfinite(v) and v != 0.0 for v in (x_step, y_step)):
        return None
    x_end = x_start + x_step * nx
    y_end = y_start + y_step * ny
    bounds = (
        (min(x_start, x_end), max(x_start, x_end)),
        (min(y_start, y_end), max(y_start, y_end)),
    )

    z_start: float | None = None
    z_step: float | None = None
    if axes == "ZYX":
        nz = shape[0]
        if len(offset) >= 3 and len(length) >= 3 and length[2] != 0.0:
            z_start, z_step = offset[2], length[2] / nz
        else:
            pixels = _finite_tuple(stack.get("pixel_m"))
            if len(pixels) >= 3 and pixels[-3] != 0.0:
                # Without a calibrated Z origin, a physical 3-D mapping would
                # be ambiguous.  Keep the stack as a valid 2-D projection
                # candidate, but advertise no usable Z calibration.
                z_step = pixels[-3]

    return {
        "shape": shape,
        "axes": axes,
        "x_start_m": x_start,
        "y_start_m": y_start,
        "x_step_m": x_step,
        "y_step_m": y_step,
        "z_start_m": z_start,
        "z_step_m": z_step,
        "bounds_xy_m": bounds,
    }


def geometry_match_error(
    image_bounds: tuple[tuple[float, float], tuple[float, float]],
    roi_box: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Return maximum normalized boundary error for X and Y.

    Both low and high image boundaries are compared, rather than span alone,
    so a same-sized image at a different stage position cannot pass.
    """
    rx, ry, rw, rh = (float(v) for v in roi_box)
    if rw <= 0.0 or rh <= 0.0:
        return math.inf, math.inf
    (ix0, ix1), (iy0, iy1) = image_bounds
    x_err = max(abs(ix0 - rx), abs(ix1 - (rx + rw))) / rw
    y_err = max(abs(iy0 - ry), abs(iy1 - (ry + rh))) / rh
    return float(x_err), float(y_err)


def _selected_dataset_ids(
    selected_datasets: Iterable[Mapping],
    roi_groups: Mapping[str, Sequence[AcquisitionRoi]],
    did_label_map: Mapping[str, str] | None,
) -> list[tuple[str, str]]:
    label_to_did = {
        str(label): str(did) for did, label in (did_label_map or {}).items() if str(label)
    }
    rows: list[tuple[str, str]] = []
    selected = list(selected_datasets)
    for item in selected:
        key = str(
            item.get("dataset_key")
            or item.get("display_name")
            or item.get("name")
            or item.get("did")
            or "dataset"
        )
        did = str(item.get("did") or "")
        if not did:
            did = label_to_did.get(key, "")
        if did in roi_groups:
            rows.append((key, did))

    # Old imported datasets did not persist their DID.  If there is only one
    # selected dataset and only one acquisition run in the file, association is
    # still unambiguous and can be recovered without a name guess.
    if not rows and len(selected) == 1 and len(roi_groups) == 1:
        item = selected[0]
        key = str(
            item.get("dataset_key")
            or item.get("display_name")
            or item.get("name")
            or item.get("did")
            or "dataset"
        )
        rows.append((key, next(iter(roi_groups))))
    return rows


def detect_confocal_candidates(
    stacks: Iterable[Mapping],
    rois: Iterable[AcquisitionRoi],
    selected_datasets: Iterable[Mapping],
    *,
    did_label_map: Mapping[str, str] | None = None,
    tolerance: float = DEFAULT_GEOMETRY_TOLERANCE,
) -> list[ConfocalCandidate]:
    """Filter scanned OBF stacks using type and calibrated ROI geometry."""
    tol = float(tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("Confocal candidate geometry tolerance must be non-negative")
    roi_groups = group_by_dataset(rois)
    selected = _selected_dataset_ids(selected_datasets, roi_groups, did_label_map)
    if not selected:
        return []

    roi_boxes = {did: union_bounds(roi_groups[did]) for _key, did in selected}
    candidates: list[ConfocalCandidate] = []
    for stack in stacks:
        if is_known_non_channel_stack(stack):
            continue
        geometry = _stack_geometry(stack)
        if geometry is None:
            continue
        matches: list[ConfocalCandidateMatch] = []
        for key, did in selected:
            box = roi_boxes.get(did)
            if box is None:
                continue
            x_err, y_err = geometry_match_error(geometry["bounds_xy_m"], box)
            if x_err <= tol and y_err <= tol:
                matches.append(ConfocalCandidateMatch(key, did, x_err, y_err))
        if not matches:
            continue
        candidates.append(
            ConfocalCandidate(
                raw_index=int(stack.get("raw_index", -1)),
                name=str(stack.get("name", "") or f"Series {len(candidates) + 1}"),
                shape=geometry["shape"],
                axes=geometry["axes"],
                dtype=str(stack.get("dtype", "") or ""),
                x_start_m=float(geometry["x_start_m"]),
                y_start_m=float(geometry["y_start_m"]),
                x_step_m=float(geometry["x_step_m"]),
                y_step_m=float(geometry["y_step_m"]),
                z_start_m=(None if geometry["z_start_m"] is None else float(geometry["z_start_m"])),
                z_step_m=(None if geometry["z_step_m"] is None else float(geometry["z_step_m"])),
                bounds_xy_m=geometry["bounds_xy_m"],
                matches=tuple(matches),
            )
        )
    return candidates


def discover_confocal_candidates(
    msr_path: str | Path,
    selected_datasets: Iterable[Mapping],
    *,
    tolerance: float = DEFAULT_GEOMETRY_TOLERANCE,
) -> list[ConfocalCandidate]:
    """Header-only candidate discovery for an MSR file."""
    path = Path(msr_path)
    return detect_confocal_candidates(
        scan_obf_stacks(path),
        read_acquisition_rois(path),
        selected_datasets,
        did_label_map=extract_did_label_map(path),
        tolerance=tolerance,
    )


def load_confocal_candidate_array(
    msr_path: str | Path,
    candidate: ConfocalCandidate,
) -> np.ndarray:
    """Read one chosen candidate stack and validate its recorded shape.

    Dispatches on the candidate's origin: an OBF stack (``raw_index >= 0``) is
    read from the ``.msr``; a standalone TIFF candidate (``raw_index == -1``,
    produced by :func:`candidates_from_tiff`) is read from the image file.
    """
    if int(candidate.raw_index) < 0:
        array = _read_tiff_array(msr_path)
    else:
        source = ObfImageSource(msr_path, raw_stack_index=candidate.raw_index)
        try:
            array = np.asarray(source.read_array())
        finally:
            source.close()
    if tuple(array.shape) != tuple(candidate.shape):
        raise ValueError(
            f"Image stack '{candidate.name}' changed shape: expected "
            f"{candidate.shape}, got {array.shape}"
        )
    return array


# ---------------------------------------------------------------------------
# Standalone TIFF as a confocal channel
# ---------------------------------------------------------------------------
def _read_tiff_array(tiff_path: str | Path, *, series_index: int = 0) -> np.ndarray:
    """The TIFF series as a plain ``YX`` / ``ZYX`` float-capable array."""
    from .tiff_source import TiffImageSource

    source = TiffImageSource(tiff_path, series_index=series_index)
    try:
        meta = source.metadata
        nz = meta.axis_size("Z")
        planes = [np.asarray(source.read_plane(z=z)) for z in range(max(1, nz))]
    finally:
        source.close()
    for plane in planes:
        if plane.ndim != 2:
            raise ValueError(
                "Confocal mapping needs a scalar (single-channel) image; this TIFF "
                f"plane is {plane.ndim}-D (RGB or multi-sample)."
            )
    return planes[0] if len(planes) == 1 else np.stack(planes, axis=0)


def _dataset_xy_extent_m(dataset) -> tuple[float, float, float, float] | None:
    """The dataset's raw ``(x0, x1, y0, y1)`` localization extent, in metres."""
    from .loader import attr_values_1d

    x = attr_values_1d(dataset, "loc_x")
    y = attr_values_1d(dataset, "loc_y")
    if x is None or y is None:
        return None
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return None
    return (
        float(x[finite].min()), float(x[finite].max()),
        float(y[finite].min()), float(y[finite].max()),
    )


def candidates_from_tiff(
    tiff_path: str | Path,
    dataset,
    *,
    series_index: int = 0,
) -> list[ConfocalCandidate]:
    """One :class:`ConfocalCandidate` for a standalone TIFF dropped on *dataset*.

    A TIFF carries a **pixel size** but, unlike an OBF stack, no stage origin —
    nothing in the file says where on the sample it was taken.  So the geometric
    ROI matching that :func:`detect_confocal_candidates` performs is impossible
    here and would be a fabrication.  Instead the image is **centred on the
    dataset's own XY extent**, which is the assumption the drop itself expresses
    ("this image is of this field"), and the mapping dialog's *manual* alignment
    is there to nudge it.  ``raw_index`` is ``-1`` to mark the candidate as
    file-backed rather than an OBF stack, and ``matches`` is empty because no
    acquisition ROI was matched.

    Returns ``[]`` when the TIFF has no usable pixel calibration or the placed
    image does not overlap the localizations at all.
    """
    from .tiff_source import TiffImageSource

    path = Path(tiff_path)
    source = TiffImageSource(path, series_index=series_index)
    try:
        meta = source.metadata
        nx = meta.axis_size("X")
        ny = meta.axis_size("Y")
        nz = meta.axis_size("Z")
        px_nm = meta.pixel_size_x.nm
        py_nm = meta.pixel_size_y.nm
        pz_nm = meta.pixel_size_z.nm
        dtype = meta.dtype
        name = meta.image_name or path.name
    finally:
        source.close()

    if not px_nm or not py_nm or nx <= 0 or ny <= 0:
        raise ValueError(
            f"'{path.name}' has no pixel-size calibration (OME PhysicalSize, "
            "ImageJ metadata or TIFF resolution tags), so it cannot be placed "
            "against localization coordinates."
        )

    extent = _dataset_xy_extent_m(dataset)
    if extent is None:
        raise ValueError("Dataset has no finite localization X/Y coordinates.")
    dx0, dx1, dy0, dy1 = extent

    x_step = float(px_nm) * 1e-9
    y_step = float(py_nm) * 1e-9
    width = x_step * nx
    height = y_step * ny
    x_start = 0.5 * (dx0 + dx1) - 0.5 * width       # centred on the data
    y_start = 0.5 * (dy0 + dy1) - 0.5 * height

    # Centring guarantees overlap for any non-degenerate image, but a zero-size
    # calibration would not — check rather than assume.
    if width <= 0.0 or height <= 0.0:
        return []

    shape: tuple[int, ...] = (nz, ny, nx) if nz > 1 else (ny, nx)
    return [
        ConfocalCandidate(
            raw_index=-1,
            name=str(name),
            shape=shape,
            axes="ZYX" if nz > 1 else "YX",
            dtype=str(dtype),
            x_start_m=x_start,
            y_start_m=y_start,
            x_step_m=x_step,
            y_step_m=y_step,
            z_start_m=None,
            z_step_m=(float(pz_nm) * 1e-9 if pz_nm else None),
            bounds_xy_m=((x_start, x_start + width), (y_start, y_start + height)),
            matches=(),
        )
    ]


def mapping_image(array: np.ndarray, dimension: str) -> np.ndarray:
    """Return the float64 2-D projection or 3-D volume used for mapping."""
    arr = np.asarray(array)
    dim = str(dimension).strip().upper()
    if dim == "2D":
        if arr.ndim == 2:
            return np.asarray(arr, dtype=np.float64)
        if arr.ndim == 3:
            # Explicit dtype prevents int16/uint16 accumulation overflow.
            return np.sum(arr, axis=0, dtype=np.float64)
        raise ValueError(f"2-D confocal mapping requires a YX or ZYX stack, got {arr.shape}")
    if dim == "3D":
        if arr.ndim != 3:
            raise ValueError(f"3-D confocal mapping requires a ZYX stack, got {arr.shape}")
        return np.asarray(arr, dtype=np.float64)
    raise ValueError(f"Unsupported confocal mapping dimension '{dimension}'")


def transform_pixel_coordinates(
    x_pixels,
    y_pixels,
    image_shape_yx: tuple[int, int],
    transform: ConfocalMappingTransform | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply manual rotation about image centre followed by translation."""
    x = np.asarray(x_pixels, dtype=np.float64)
    y = np.asarray(y_pixels, dtype=np.float64)
    tr = transform or ConfocalMappingTransform()
    ny, nx = (int(v) for v in image_shape_yx)
    cx, cy = (nx - 1.0) / 2.0, (ny - 1.0) / 2.0
    theta = math.radians(float(tr.rotation_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx, dy = x - cx, y - cy
    # Pixel Y increases downwards.  This matrix makes positive theta visually
    # counter-clockwise (rightward points move upward).
    tx = cos_t * dx + sin_t * dy + cx + float(tr.dx_pixels)
    ty = -sin_t * dx + cos_t * dy + cy + float(tr.dy_pixels)
    return tx, ty


def localization_pixel_coordinates(
    candidate: ConfocalCandidate,
    x_m,
    y_m,
    *,
    transform: ConfocalMappingTransform | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage coordinates to continuous image indices (pixel centres at integers)."""
    x = (np.asarray(x_m, dtype=np.float64) - candidate.x_start_m) / candidate.x_step_m - 0.5
    y = (np.asarray(y_m, dtype=np.float64) - candidate.y_start_m) / candidate.y_step_m - 0.5
    return transform_pixel_coordinates(x, y, candidate.shape[-2:], transform)


def sample_confocal_signal(
    image: np.ndarray,
    candidate: ConfocalCandidate,
    x_m,
    y_m,
    z_m=None,
    *,
    dimension: str = "2D",
    method: str = "bilinear",
    transform: ConfocalMappingTransform | None = None,
) -> np.ndarray:
    """Interpolate a chosen channel at localization coordinates."""
    from scipy.ndimage import map_coordinates

    arr = mapping_image(image, dimension)
    x = np.asarray(x_m, dtype=np.float64).ravel()
    y = np.asarray(y_m, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("Localization X and Y arrays must have the same shape")
    col, row = localization_pixel_coordinates(candidate, x, y, transform=transform)

    dim = str(dimension).strip().upper()
    key = str(method).strip().lower().replace("-", " ")
    if dim == "2D":
        orders = {
            "nearest": 0,
            "nearest neighbour": 0,
            "nearest neighbor": 0,
            "bilinear": 1,
            "bicubic": 3,
        }
        if key not in orders:
            raise ValueError(f"Unsupported 2-D interpolation method '{method}'")
        ny, nx = arr.shape
        finite = (
            np.isfinite(row)
            & np.isfinite(col)
            & (row >= -0.5)
            & (row <= ny - 0.5)
            & (col >= -0.5)
            & (col <= nx - 0.5)
        )
        # The calibrated bounds describe pixel edges, while scipy samples at
        # pixel centres.  Replicate the border half-pixels for positions still
        # inside those bounds; only positions outside the physical image get
        # NaN.
        coords = [np.clip(row, 0.0, ny - 1.0), np.clip(col, 0.0, nx - 1.0)]
        order = orders[key]
    else:
        orders = {"nearest": 0, "nearest neighbour": 0, "nearest neighbor": 0, "trilinear": 1}
        if key not in orders:
            raise ValueError(f"Unsupported 3-D interpolation method '{method}'")
        if candidate.z_start_m is None or candidate.z_step_m in (None, 0.0):
            raise ValueError(
                f"Image stack '{candidate.name}' has no calibrated Z origin/spacing; "
                "use 2-D projection mapping"
            )
        if z_m is None:
            raise ValueError("3-D confocal mapping requires localization Z coordinates")
        z = np.asarray(z_m, dtype=np.float64).ravel()
        if z.shape != x.shape:
            raise ValueError("Localization Z must have the same length as X and Y")
        plane = (z - candidate.z_start_m) / float(candidate.z_step_m) - 0.5
        nz, ny, nx = arr.shape
        finite = (
            np.isfinite(plane)
            & np.isfinite(row)
            & np.isfinite(col)
            & (plane >= -0.5)
            & (plane <= nz - 0.5)
            & (row >= -0.5)
            & (row <= ny - 0.5)
            & (col >= -0.5)
            & (col <= nx - 0.5)
        )
        coords = [
            np.clip(plane, 0.0, nz - 1.0),
            np.clip(row, 0.0, ny - 1.0),
            np.clip(col, 0.0, nx - 1.0),
        ]
        order = orders[key]

    out = np.full(x.shape, np.nan, dtype=np.float64)
    if np.any(finite):
        selected_coords = [coord[finite] for coord in coords]
        out[finite] = map_coordinates(
            arr,
            selected_coords,
            order=order,
            mode="constant",
            cval=np.nan,
            prefilter=order > 1,
        )
    return out


def _dataset_coordinates(dataset, store) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if store is dataset.attr:
        from .loader import attr_values_1d

        values = [attr_values_1d(dataset, name) for name in ("loc_x", "loc_y", "loc_z")]
    else:
        values = [store.get(name) for name in ("loc_x", "loc_y", "loc_z")]
    if values[0] is None or values[1] is None:
        raise ValueError("Dataset has no calibrated localization X/Y coordinates")
    x = np.asarray(values[0], dtype=np.float64).ravel()
    y = np.asarray(values[1], dtype=np.float64).ravel()
    z = np.zeros_like(x) if values[2] is None else np.asarray(values[2], dtype=np.float64).ravel()
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError("Dataset localization coordinates are not row-aligned")
    return x, y, z


def attach_confocal_signal(
    dataset,
    msr_path: str | Path,
    candidate: ConfocalCandidate,
    attribute_name: str,
    *,
    dimension: str = "2D",
    method: str = "bilinear",
    alignment: str = "automatic",
    transform: ConfocalMappingTransform | None = None,
    image: np.ndarray | None = None,
) -> ConfocalMappingResult:
    """Map a candidate channel and attach one user-visible numeric attribute.

    Values are stored on the materialized localization table and on the
    all-iteration raw table so iteration browsing remains coherent.
    """
    name = str(attribute_name).strip()
    if not name:
        raise ValueError("Confocal signal attribute name cannot be empty")
    if name in {"xnm", "ynm", "znm"}:
        raise ValueError(f"'{name}' is reserved for a calibrated coordinate view")
    if name in dataset.attr or name in dataset.mfx_raw:
        raise ValueError(f"Dataset already contains an attribute named '{name}'")

    source_image = (
        load_confocal_candidate_array(msr_path, candidate) if image is None else np.asarray(image)
    )
    x, y, z = _dataset_coordinates(dataset, dataset.attr)
    values = sample_confocal_signal(
        source_image,
        candidate,
        x,
        y,
        z,
        dimension=dimension,
        method=method,
        transform=transform,
    )

    raw_values = np.empty(0, dtype=np.float64)
    if len(dataset.mfx_raw):
        rx, ry, rz = _dataset_coordinates(dataset, dataset.mfx_raw)
        raw_values = sample_confocal_signal(
            source_image,
            candidate,
            rx,
            ry,
            rz,
            dimension=dimension,
            method=method,
            transform=transform,
        )

    tr = transform or ConfocalMappingTransform()
    provenance = {
        "kind": "confocal_image_signal",
        "source_path": str(Path(msr_path)),
        "stack_name": candidate.name,
        "raw_stack_index": int(candidate.raw_index),
        "stack_shape": list(candidate.shape),
        "stack_axes": candidate.axes,
        "stack_dtype": candidate.dtype,
        "bounds_xy_m": [list(candidate.bounds_xy_m[0]), list(candidate.bounds_xy_m[1])],
        "dimension": str(dimension).upper(),
        "method": str(method).lower(),
        "alignment": str(alignment).lower(),
        "transform": asdict(tr),
        "pixel_coordinate_convention": "calibrated bounds; pixel centres at integer indices",
        "out_of_bounds": "NaN",
        "z_coordinate": "raw un-Z-scaled localization z"
        if str(dimension).upper() == "3D"
        else None,
        "matched_datasets": [asdict(match) for match in candidate.matches],
    }
    meta = {
        "component": "mfx",
        "source": "derived from calibrated image",
        "description": f"{candidate.name} fluorescence sampled at each MINFLUX localization.",
        "unit": "image intensity",
        "user_visible": True,
        "confocal_mapping": provenance,
    }
    dataset.mfx.set_attr(name, values, meta=meta)
    if raw_values.size:
        dataset.mfx_raw[name] = raw_values
    if name not in dataset.prop.attr_names:
        dataset.prop.attr_names.append(name)
    mappings = dataset.metadata.setdefault("confocal_signal_mappings", {})
    mappings[name] = provenance
    return ConfocalMappingResult(
        attribute_name=name,
        finite_count=int(np.isfinite(values).sum()),
        total_count=int(values.size),
        raw_finite_count=int(np.isfinite(raw_values).sum()),
        raw_total_count=int(raw_values.size),
        provenance=provenance,
    )
