"""
Precision-aware, viewport-localized localization rendering.

Every localization contributes one unit-mass anisotropic Gaussian. The Gaussian
is integrated over pixel areas, so changing zoom or preview resolution changes
only raster sampling, not the scientific reconstruction model.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot
from scipy.special import ndtr

_DEFAULT_SIGMA_NM = 5.0
_MIN_SIGMA_NM = 1.0e-3

RENDER_METHOD_HISTOGRAM = "histogram"
RENDER_METHOD_BILINEAR = "bilinear"
RENDER_METHOD_BICUBIC = "bicubic"
RENDER_METHOD_BASIC = "basic"
RENDER_METHOD_FIXED_GAUSSIAN = "fixed_gaussian"
RENDER_METHOD_PRECISION_GAUSSIAN = "precision_gaussian"
RENDER_METHODS = {
    RENDER_METHOD_HISTOGRAM,
    RENDER_METHOD_BILINEAR,
    RENDER_METHOD_BICUBIC,
    RENDER_METHOD_BASIC,
    RENDER_METHOD_FIXED_GAUSSIAN,
    RENDER_METHOD_PRECISION_GAUSSIAN,
}


@dataclass
class PrecisionChannelData:
    """Filtered, transformed data for one localization channel."""

    dataset_idx: int
    x_nm: np.ndarray
    y_nm: np.ndarray
    depth_nm: np.ndarray
    sigma_x_nm: np.ndarray
    sigma_y_nm: np.ndarray
    sigma_depth_nm: np.ndarray
    grid: object
    source: str


@dataclass(frozen=True)
class PrecisionTileRequest:
    """One scalar precision tile requested from the background scheduler."""

    key: Hashable
    channel: PrecisionChannelData
    bounds: tuple[float, float, float, float]
    shape: tuple[int, int]
    depth_range: tuple[float, float] | None
    render_method: str = RENDER_METHOD_PRECISION_GAUSSIAN
    fixed_sigma_nm: float = _DEFAULT_SIGMA_NM


@dataclass(frozen=True)
class PrecisionTileResult:
    """Completed scalar precision tile emitted by a worker."""

    generation: int
    key: Hashable
    array: np.ndarray
    count: int


class ViewportScalarCache:
    """Small byte-limited LRU cache of viewport-sized scalar channel rasters."""

    def __init__(self, max_bytes: int = 64 * 1024 * 1024, max_items: int = 24) -> None:
        self.max_bytes = max(int(max_bytes), 0)
        self.max_items = max(int(max_items), 1)
        self._items: OrderedDict[Hashable, np.ndarray] = OrderedDict()
        self._bytes = 0

    def get(self, key: Hashable | None) -> np.ndarray | None:
        if key is None:
            return None
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: Hashable | None, value: np.ndarray) -> None:
        if key is None:
            return
        arr = np.asarray(value, dtype=np.float32)
        previous = self._items.pop(key, None)
        if previous is not None:
            self._bytes -= previous.nbytes
        self._items[key] = arr
        self._bytes += arr.nbytes
        while self._items and (
            len(self._items) > self.max_items or self._bytes > self.max_bytes
        ):
            _, evicted = self._items.popitem(last=False)
            self._bytes -= evicted.nbytes

    def remove_dataset(self, dataset_idx: int) -> None:
        for key in list(self._items):
            if isinstance(key, tuple) and len(key) > 1 and key[1] == dataset_idx:
                arr = self._items.pop(key)
                self._bytes -= arr.nbytes

    def clear(self) -> None:
        self._items.clear()
        self._bytes = 0

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._items)


def _component_value(ds, name: str):
    value = ds.attr.get(name)
    if value is None:
        value = ds.derived.get(name)
    return value


def _precision_scale_to_nm(ds, name: str, values: np.ndarray) -> float:
    """Resolve an explicit unit, then conservatively infer metres vs nanometres."""
    unit = None
    try:
        unit = str(ds.mfx.get_meta(name, {}).get("unit", "")).lower()
    except Exception:
        pass
    lname = name.lower()
    if unit == "m" or lname.endswith("_m"):
        return 1.0e9
    if unit in {"um", "µm"} or lname.endswith("_um"):
        return 1.0e3
    if unit == "nm" or lname.endswith("_nm") or lname.startswith("loc_precision"):
        return 1.0

    finite = np.abs(values[np.isfinite(values)])
    median = float(np.median(finite)) if finite.size else _DEFAULT_SIGMA_NM
    return 1.0e9 if median < 1.0e-3 else 1.0


def _row_precision(ds, names: tuple[str, ...], n_rows: int):
    for name in names:
        value = _component_value(ds, name)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64).ravel()
        if arr.size == 1:
            arr = np.full(n_rows, float(arr[0]), dtype=np.float64)
        if arr.size != n_rows:
            continue
        arr = arr * _precision_scale_to_nm(ds, name, arr)
        return arr, name
    return None, None


def _global_precision_xyz(ds, n_rows: int) -> tuple[np.ndarray, str]:
    cali = np.asarray(getattr(ds.cali, "loc_precision", []), dtype=np.float64).ravel()
    cali = cali[np.isfinite(cali) & (cali > 0.0)]
    per_loc_xy = _component_value(ds, "loc_precision_xy")
    if per_loc_xy is not None:
        values = np.asarray(per_loc_xy, dtype=np.float64).ravel()
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            lateral = float(np.median(values))
            z_value = _component_value(ds, "loc_precision_z")
            z_values = np.asarray(
                [] if z_value is None else z_value, dtype=np.float64
            ).ravel()
            z_values = z_values[np.isfinite(z_values) & (z_values > 0.0)]
            axial = float(np.median(z_values)) if z_values.size else lateral
            return np.array([lateral, lateral, axial]), "global per-localization median"
    if cali.size == n_rows and n_rows > 3:
        value = float(np.median(cali))
        return np.full(3, value), "global calibration median"
    if cali.size >= 3:
        return cali[:3], "global calibration"
    if cali.size == 2:
        return np.array([cali[0], cali[1], np.mean(cali)]), "global calibration"
    if cali.size:
        value = float(np.median(cali))
        return np.full(3, value), "global calibration"
    return np.full(3, _DEFAULT_SIGMA_NM), f"{_DEFAULT_SIGMA_NM:g} nm fallback"


def resolve_precision_xyz_nm(ds, n_rows: int) -> tuple[np.ndarray, str]:
    """Return ``(N,3)`` sigma in display nanometres and a source description.

    Source precedence is explicit per-localization precision, per-trace StdDev,
    dataset calibration, then a clearly reported 5 nm fallback.
    """
    fallback, fallback_source = _global_precision_xyz(ds, n_rows)
    sigma = np.broadcast_to(fallback, (n_rows, 3)).astype(np.float64, copy=True)
    sources: list[str] = []

    trace_sigma = _component_value(ds, "sigma_per_trace_nm")
    trace_ids = _component_value(ds, "sigma_trace_ids")
    tid = _component_value(ds, "tid")
    if trace_sigma is not None and trace_ids is not None and tid is not None:
        per_trace = np.asarray(trace_sigma, dtype=np.float64)
        ids = np.asarray(trace_ids).ravel()
        row_tid = np.asarray(tid).ravel()
        if per_trace.ndim == 2 and per_trace.shape[1] >= 2 and len(ids) == len(per_trace):
            if per_trace.shape[1] == 2:
                per_trace = np.column_stack([per_trace, np.mean(per_trace, axis=1)])
            order = np.argsort(ids, kind="stable")
            ids_sorted = ids[order]
            positions = np.searchsorted(ids_sorted, row_tid)
            valid = positions < ids_sorted.size
            valid[valid] &= ids_sorted[positions[valid]] == row_tid[valid]
            if np.any(valid):
                mapped = per_trace[order[positions[valid]], :3]
                good = np.all(np.isfinite(mapped) & (mapped > 0.0), axis=1)
                rows = np.flatnonzero(valid)[good]
                sigma[rows] = mapped[good]
                if rows.size:
                    sources.append("per-trace StdDev")

    x, x_name = _row_precision(
        ds, ("sx_nm", "sigma_x_nm", "precision_x_nm", "loc_precision_x", "lpx_nm", "lpx", "sx"),
        n_rows,
    )
    y, y_name = _row_precision(
        ds, ("sy_nm", "sigma_y_nm", "precision_y_nm", "loc_precision_y", "lpy_nm", "lpy", "sy"),
        n_rows,
    )
    lateral, lateral_name = _row_precision(
        ds, ("loc_precision_xy", "precision_xy_nm", "sigma_xy_nm"), n_rows
    )
    z, z_name = _row_precision(
        ds, ("sz_nm", "sigma_z_nm", "precision_z_nm", "loc_precision_z", "lpz_nm", "lpz", "sz"),
        n_rows,
    )
    if x is None:
        x = lateral
        x_name = lateral_name
    if y is None:
        y = lateral
        y_name = lateral_name

    explicit_names = []
    for axis, values, name in ((0, x, x_name), (1, y, y_name), (2, z, z_name)):
        if values is None:
            continue
        good = np.isfinite(values) & (values > 0.0)
        sigma[good, axis] = values[good]
        if np.any(good) and name:
            explicit_names.append(name)
    if explicit_names:
        sources.insert(0, "per-localization " + "/".join(dict.fromkeys(explicit_names)))

    sigma[~np.isfinite(sigma) | (sigma <= 0.0)] = _DEFAULT_SIGMA_NM
    np.maximum(sigma, _MIN_SIGMA_NM, out=sigma)
    rimf = abs(float(getattr(ds.cali, "RIMF", 1.0) or 1.0))
    sigma[:, 2] *= rimf

    if not sources:
        sources.append(fallback_source)
    elif np.any(np.all(np.isclose(sigma, fallback), axis=1)):
        sources.append(f"{fallback_source} for missing values")
    return sigma, "; ".join(sources)


def transform_precision_marginals(
    sigma_xyz_nm: np.ndarray, matrix_4x4: np.ndarray | None
) -> np.ndarray:
    """Apply the rigid transform's linear part to Gaussian marginal variances."""
    sigma = np.asarray(sigma_xyz_nm, dtype=np.float64)
    if matrix_4x4 is None:
        return sigma
    matrix = np.asarray(matrix_4x4, dtype=np.float64)
    if matrix.shape != (4, 4):
        return sigma
    variances = sigma * sigma
    transformed = variances @ (matrix[:3, :3] * matrix[:3, :3]).T
    return np.sqrt(np.maximum(transformed, _MIN_SIGMA_NM**2))


def render_precision_gaussians(
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    sigma_x_nm: np.ndarray,
    sigma_y_nm: np.ndarray,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    *,
    truncate_sigma: float = 4.0,
    cancelled: Callable[[], bool] | None = None,
    _max_stamp_elems: int = 1 << 21,
) -> np.ndarray:
    """Rasterize pixel-integrated anisotropic Gaussians with unit mass per loc.

    Vectorized: localizations are grouped by their integer pixel footprint
    ``(kh, kw)`` and each group is rasterized as batched array ops (one ``ndtr``
    over all edges, one ``bincount`` scatter-add) instead of a Python per-loc
    loop. The per-pixel math is identical to the scalar reference — each pixel's
    weight is the Gaussian CDF difference over its edges, normalized by the finite
    ±truncate support — so the result is tile-independent (a localization well
    inside the viewport contributes exactly one count at every sampling
    resolution; one at an image edge loses only its off-screen mass). Values are
    accumulated in float64, so this is marginally more accurate than the old
    per-loc float32 accumulation.
    """
    height, width = max(int(shape[0]), 1), max(int(shape[1]), 1)
    x0, x1, y0, y1 = (float(v) for v in bounds)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((height, width), dtype=np.float32)

    x = np.asarray(x_nm, dtype=np.float64).ravel()
    y = np.asarray(y_nm, dtype=np.float64).ravel()
    sx = np.maximum(np.asarray(sigma_x_nm, dtype=np.float64).ravel(), _MIN_SIGMA_NM)
    sy = np.maximum(np.asarray(sigma_y_nm, dtype=np.float64).ravel(), _MIN_SIGMA_NM)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(sx) & np.isfinite(sy)
    x, y, sx, sy = x[valid], y[valid], sx[valid], sy[valid]
    if not x.size:
        return np.zeros((height, width), dtype=np.float32)

    dx = (x1 - x0) / width
    dy = (y1 - y0) / height
    trunc = max(float(truncate_sigma), 1.0)

    # Integer pixel footprint per localization (floor/ceil of the ±trunc·sigma
    # support), identical to the scalar loop, so tile decomposition never changes
    # a pixel value.
    fc0 = np.floor((x - trunc * sx - x0) / dx).astype(np.int64)
    fc1 = np.ceil((x + trunc * sx - x0) / dx).astype(np.int64)
    fr0 = np.floor((y - trunc * sy - y0) / dy).astype(np.int64)
    fr1 = np.ceil((y + trunc * sy - y0) / dy).astype(np.int64)
    kw = fc1 - fc0
    kh = fr1 - fr0
    keep = (
        (kw > 0) & (kh > 0)
        & (fc1 > 0) & (fc0 < width)
        & (fr1 > 0) & (fr0 < height)
    )
    if not np.any(keep):
        return np.zeros((height, width), dtype=np.float32)
    x, y, sx, sy = x[keep], y[keep], sx[keep], sy[keep]
    fc0, fr0, kw, kh = fc0[keep], fr0[keep], kw[keep], kh[keep]

    acc = np.zeros(height * width, dtype=np.float64)

    # Group by identical footprint shape so a whole group shares one stamp shape.
    order = np.lexsort((kw, kh))
    kh_s, kw_s = kh[order], kw[order]
    boundaries = np.flatnonzero(np.diff(kh_s) | np.diff(kw_s)) + 1
    group_slices = np.concatenate(([0], boundaries, [order.size]))

    for gi in range(group_slices.size - 1):
        sel = order[group_slices[gi]:group_slices[gi + 1]]
        gh = int(kh[sel[0]])
        gw = int(kw[sel[0]])
        # Cap the batch so a group's (M, gh, gw) stamp stays within the budget.
        per = max(int(_max_stamp_elems // max(gh * gw, 1)), 1)
        col_off = np.arange(gw + 1, dtype=np.float64)
        row_off = np.arange(gh + 1, dtype=np.float64)
        acol = np.arange(gw, dtype=np.int64)
        arow = np.arange(gh, dtype=np.int64)
        for start in range(0, sel.size, per):
            if cancelled is not None and cancelled():
                return acc.reshape(height, width).astype(np.float32)
            grp = sel[start:start + per]
            cx = x[grp][:, None]
            cy = y[grp][:, None]
            sxc = sx[grp][:, None]
            syc = sy[grp][:, None]
            xe = x0 + (fc0[grp][:, None] + col_off[None, :]) * dx
            ye = y0 + (fr0[grp][:, None] + row_off[None, :]) * dy
            cdfx = ndtr((xe - cx) / sxc)
            cdfy = ndtr((ye - cy) / syc)
            wx = cdfx[:, 1:] - cdfx[:, :-1]
            wy = cdfy[:, 1:] - cdfy[:, :-1]
            nx = cdfx[:, -1] - cdfx[:, 0]
            ny = cdfy[:, -1] - cdfy[:, 0]
            np.divide(wx, nx[:, None], out=wx, where=nx[:, None] > 0.0)
            np.divide(wy, ny[:, None], out=wy, where=ny[:, None] > 0.0)
            stamp = wy[:, :, None] * wx[:, None, :]           # (M, gh, gw)
            rows = fr0[grp][:, None] + arow[None, :]          # (M, gh)
            cols = fc0[grp][:, None] + acol[None, :]          # (M, gw)
            mask = (
                ((rows >= 0) & (rows < height))[:, :, None]
                & ((cols >= 0) & (cols < width))[:, None, :]
            )
            lin = rows[:, :, None] * width + cols[:, None, :]
            acc += np.bincount(lin[mask], weights=stamp[mask], minlength=height * width)
    return acc.reshape(height, width).astype(np.float32)


def render_histogram(
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    """Render one count per localization into its containing image pixel."""
    height, width = max(int(shape[0]), 1), max(int(shape[1]), 1)
    x0, x1, y0, y1 = (float(value) for value in bounds)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((height, width), dtype=np.float32)
    x = np.asarray(x_nm, dtype=np.float64).ravel()
    y = np.asarray(y_nm, dtype=np.float64).ravel()
    finite = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= x0)
        & (x < x1)
        & (y >= y0)
        & (y < y1)
    )
    hist, _, _ = np.histogram2d(
        y[finite],
        x[finite],
        bins=(height, width),
        range=((y0, y1), (x0, x1)),
    )
    return hist.astype(np.float32, copy=False)


def render_gaussian_filtered(
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    sigma_x_nm: float,
    sigma_y_nm: float,
    *,
    truncate: float = 4.0,
) -> np.ndarray:
    """Fast **constant-sigma** Gaussian render: histogram → separable blur.

    A single ``scipy.ndimage.gaussian_filter`` costs O(pixels) — independent of
    the localization count **and** of sigma — versus the O(N·(4σ/px)²) cost of
    stamping one pixel-integrated Gaussian per localization. It snaps each
    localization to its pixel before blurring, so it loses exact sub-pixel
    placement (negligible once sigma ≳ 1 px). Only valid for one shared sigma
    (Fixed-Gaussian / Basic); the per-localization precision method still needs
    :func:`render_precision_gaussians`.

    Callers pass localizations from a ``≥ truncate·σ`` halo around the tile; this
    histograms into a **padded** grid and crops back, so the blur has no tile-edge
    seam.
    """
    from scipy.ndimage import gaussian_filter

    height, width = max(int(shape[0]), 1), max(int(shape[1]), 1)
    x0, x1, y0, y1 = (float(v) for v in bounds)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((height, width), dtype=np.float32)
    dx = (x1 - x0) / width
    dy = (y1 - y0) / height
    sx_px = max(float(sigma_x_nm), _MIN_SIGMA_NM) / dx
    sy_px = max(float(sigma_y_nm), _MIN_SIGMA_NM) / dy
    pad_x = int(np.ceil(max(truncate, 1.0) * sx_px))
    pad_y = int(np.ceil(max(truncate, 1.0) * sy_px))
    padded_bounds = (
        x0 - pad_x * dx,
        x1 + pad_x * dx,
        y0 - pad_y * dy,
        y1 + pad_y * dy,
    )
    hist = render_histogram(
        x_nm, y_nm, padded_bounds, (height + 2 * pad_y, width + 2 * pad_x)
    )
    if max(sx_px, sy_px) >= 0.3:
        hist = gaussian_filter(
            hist, sigma=(max(sy_px, 1e-6), max(sx_px, 1e-6)), mode="constant"
        )
    crop = hist[pad_y:pad_y + height, pad_x:pad_x + width]
    return np.ascontiguousarray(crop, dtype=np.float32)


def render_bilinear_histogram(
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    """Distribute each localization over the four nearest pixel centres.

    Callers may provide points up to one pixel outside the bounds. Their
    overlapping contribution makes adjacent cached tiles join without seams.
    """
    height, width = max(int(shape[0]), 1), max(int(shape[1]), 1)
    result = np.zeros((height, width), dtype=np.float32)
    x0, x1, y0, y1 = (float(value) for value in bounds)
    if x1 <= x0 or y1 <= y0:
        return result

    x = np.asarray(x_nm, dtype=np.float64).ravel()
    y = np.asarray(y_nm, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if not x.size:
        return result

    col = (x - x0) * width / (x1 - x0) - 0.5
    row = (y - y0) * height / (y1 - y0) - 0.5
    col0 = np.floor(col).astype(np.int64)
    row0 = np.floor(row).astype(np.int64)
    frac_x = col - col0
    frac_y = row - row0
    candidates = (
        (row0, col0, (1.0 - frac_y) * (1.0 - frac_x)),
        (row0, col0 + 1, (1.0 - frac_y) * frac_x),
        (row0 + 1, col0, frac_y * (1.0 - frac_x)),
        (row0 + 1, col0 + 1, frac_y * frac_x),
    )

    for rows, cols, weights in candidates:
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        np.add.at(result, (rows[valid], cols[valid]), weights[valid])
    return result


def _catmull_rom_taps(frac: np.ndarray) -> np.ndarray:
    """Catmull-Rom cubic weights (a=-0.5) for the 4 taps at offsets -1,0,1,2
    from the floor pixel, given the fractional position ``frac`` in [0, 1)."""
    a = -0.5

    def kernel(t: np.ndarray) -> np.ndarray:
        t = np.abs(t)
        w = np.zeros_like(t)
        near = t < 1.0
        far = (t >= 1.0) & (t < 2.0)
        w[near] = (a + 2.0) * t[near] ** 3 - (a + 3.0) * t[near] ** 2 + 1.0
        w[far] = a * t[far] ** 3 - 5.0 * a * t[far] ** 2 + 8.0 * a * t[far] - 4.0 * a
        return w

    # Distances from the sample to taps [floor-1, floor, floor+1, floor+2].
    return np.stack(
        [kernel(1.0 + frac), kernel(frac), kernel(1.0 - frac), kernel(2.0 - frac)],
        axis=1,
    )


def render_bicubic_histogram(
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    """Distribute each localization over its 4×4 pixel neighbourhood with a
    Catmull-Rom bicubic kernel — a smoother sub-pixel splat than bilinear.

    Still O(N) and PSF-free (no blur applied), ~a few× the cost of bilinear. The
    Catmull-Rom kernel has small negative lobes, so a handful of pixels around a
    bright localization can dip slightly negative; the result is clamped to ≥ 0
    (a tiny mass change) since a negative density is unphysical. Callers may pass
    points up to ~2 px outside the bounds so adjacent tiles join seamlessly.
    """
    height, width = max(int(shape[0]), 1), max(int(shape[1]), 1)
    x0, x1, y0, y1 = (float(value) for value in bounds)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((height, width), dtype=np.float32)

    x = np.asarray(x_nm, dtype=np.float64).ravel()
    y = np.asarray(y_nm, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if not x.size:
        return np.zeros((height, width), dtype=np.float32)

    col = (x - x0) * width / (x1 - x0) - 0.5
    row = (y - y0) * height / (y1 - y0) - 0.5
    col0 = np.floor(col).astype(np.int64)
    row0 = np.floor(row).astype(np.int64)
    wx = _catmull_rom_taps(col - col0)   # (N, 4) for cols col0-1 .. col0+2
    wy = _catmull_rom_taps(row - row0)

    acc = np.zeros(height * width, dtype=np.float64)
    for jx, off_x in enumerate((-1, 0, 1, 2)):
        cc = col0 + off_x
        col_ok = (cc >= 0) & (cc < width)
        for jy, off_y in enumerate((-1, 0, 1, 2)):
            rr = row0 + off_y
            mask = col_ok & (rr >= 0) & (rr < height)
            if not np.any(mask):
                continue
            weights = (wy[:, jy] * wx[:, jx])[mask]
            acc += np.bincount(
                rr[mask] * width + cc[mask], weights=weights, minlength=height * width
            )
    image = acc.reshape(height, width)
    np.maximum(image, 0.0, out=image)  # drop the kernel's negative ringing
    return image.astype(np.float32)


def render_advanced_tile(
    channel: PrecisionChannelData,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    depth_range: tuple[float, float] | None = None,
    *,
    render_method: str = RENDER_METHOD_PRECISION_GAUSSIAN,
    fixed_sigma_nm: float = _DEFAULT_SIGMA_NM,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, int]:
    """Render one advanced-view tile using the selected scientific method."""
    if channel.grid is None or channel.x_nm.size == 0:
        return np.zeros(shape, dtype=np.float32), 0
    if render_method not in RENDER_METHODS:
        raise ValueError(f"Unknown render method: {render_method}")

    x0, x1, y0, y1 = bounds
    height, width = max(int(shape[0]), 1), max(int(shape[1]), 1)
    dx = (x1 - x0) / width
    dy = (y1 - y0) / height
    fixed_sigma = max(float(fixed_sigma_nm), _MIN_SIGMA_NM)
    # Basic = production "smoothed histogram": a half-pixel anti-alias blur that
    # scales with zoom (constant in pixels), so its footprint stays ~1 px.
    basic_sigma_x = 0.5 * dx
    basic_sigma_y = 0.5 * dy
    if render_method == RENDER_METHOD_PRECISION_GAUSSIAN:
        halo_x = 4.0 * float(np.max(channel.sigma_x_nm))
        halo_y = 4.0 * float(np.max(channel.sigma_y_nm))
    elif render_method == RENDER_METHOD_FIXED_GAUSSIAN:
        halo_x = halo_y = 4.0 * fixed_sigma
    elif render_method == RENDER_METHOD_BASIC:
        halo_x = 4.0 * basic_sigma_x
        halo_y = 4.0 * basic_sigma_y
    elif render_method == RENDER_METHOD_BILINEAR:
        halo_x, halo_y = dx, dy
    elif render_method == RENDER_METHOD_BICUBIC:
        halo_x, halo_y = 2.0 * dx, 2.0 * dy  # 4×4 kernel reaches 2 px out
    else:
        halo_x = halo_y = 0.0

    indices = channel.grid.query(
        x0 - halo_x, x1 + halo_x, y0 - halo_y, y1 + halo_y
    )
    if indices.size == 0:
        return np.zeros(shape, dtype=np.float32), 0
    x = channel.x_nm[indices]
    y = channel.y_nm[indices]
    if render_method == RENDER_METHOD_HISTOGRAM:
        keep = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    else:
        keep = (
            (x + halo_x >= x0)
            & (x - halo_x <= x1)
            & (y + halo_y >= y0)
            & (y - halo_y <= y1)
        )
    if depth_range is not None:
        lo, hi = depth_range
        depth = channel.depth_nm[indices]
        keep &= (depth >= lo) & (depth <= hi)
    indices = indices[keep]
    x, y = x[keep], y[keep]
    if cancelled is not None and cancelled():
        return np.zeros(shape, dtype=np.float32), 0

    if render_method == RENDER_METHOD_HISTOGRAM:
        image = render_histogram(x, y, bounds, shape)
    elif render_method == RENDER_METHOD_BILINEAR:
        image = render_bilinear_histogram(x, y, bounds, shape)
    elif render_method == RENDER_METHOD_BICUBIC:
        image = render_bicubic_histogram(x, y, bounds, shape)
    elif render_method == RENDER_METHOD_BASIC:
        image = render_gaussian_filtered(
            x, y, bounds, shape, basic_sigma_x, basic_sigma_y
        )
    elif render_method == RENDER_METHOD_FIXED_GAUSSIAN:
        # Constant sigma → the fast histogram+filter path (100–1600× faster than
        # per-localization stamping, especially when zoomed in / large sigma).
        image = render_gaussian_filtered(
            x, y, bounds, shape, fixed_sigma, fixed_sigma
        )
    else:
        image = render_precision_gaussians(
            x,
            y,
            channel.sigma_x_nm[indices],
            channel.sigma_y_nm[indices],
            bounds,
            shape,
            cancelled=cancelled,
        )
    return image, int(x.size)


def render_precision_tile(
    channel: PrecisionChannelData,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    depth_range: tuple[float, float] | None = None,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, int]:
    """Render one tile with enough source halo to avoid Gaussian edge seams."""
    return render_advanced_tile(
        channel,
        bounds,
        shape,
        depth_range,
        render_method=RENDER_METHOD_PRECISION_GAUSSIAN,
        cancelled=cancelled,
    )


class _PrecisionTileSignals(QObject):
    result_ready = pyqtSignal(object)


class _PrecisionTileTask(QRunnable):
    def __init__(
        self,
        scheduler: PrecisionRenderScheduler,
        generation: int,
        request: PrecisionTileRequest,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.scheduler = scheduler
        self.generation = generation
        self.request = request
        self.signals = _PrecisionTileSignals()

    def _cancelled(self) -> bool:
        return self.scheduler.generation != self.generation

    def run(self) -> None:
        if self._cancelled():
            return
        array, count = render_advanced_tile(
            self.request.channel,
            self.request.bounds,
            self.request.shape,
            self.request.depth_range,
            render_method=self.request.render_method,
            fixed_sigma_nm=self.request.fixed_sigma_nm,
            cancelled=self._cancelled,
        )
        if not self._cancelled():
            self.signals.result_ready.emit(
                PrecisionTileResult(
                    generation=self.generation,
                    key=self.request.key,
                    array=array,
                    count=count,
                )
            )


class PrecisionRenderScheduler(QObject):
    """Run cancellable precision tiles on a small dedicated CPU thread pool."""

    result_ready = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None, max_threads: int = 4) -> None:
        super().__init__(parent)
        self._generation = 0
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(int(max_threads), 1))

    @property
    def generation(self) -> int:
        return self._generation

    def cancel(self) -> None:
        self._generation += 1
        self._pool.clear()

    def request(self, requests: list[PrecisionTileRequest]) -> int:
        self.cancel()
        generation = self._generation
        for request in requests:
            task = _PrecisionTileTask(self, generation, request)
            task.signals.result_ready.connect(self._forward_result)
            self._pool.start(task)
        return generation

    @pyqtSlot(object)
    def _forward_result(self, result: PrecisionTileResult) -> None:
        if result.generation == self._generation:
            self.result_ready.emit(result)
