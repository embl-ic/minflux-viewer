"""
minflux_viewer.analysis.conv_segmentation
==========================================
General **convolution-based segmentation** — a geometry-model generalisation of
:mod:`minflux_viewer.analysis.npc_segmentation`.

Pipeline (2-D):
    1. render the XY localizations into a ``pixel_size`` nm histogram image;
    2. build a convolutional **kernel** from a user-chosen *geometry model*
       (ring / disk / gaussian / …) at the same pixel size;
    3. FFT-convolve the histogram with the kernel (matched filter) to get a
       *response map* — high where the local point pattern resembles the model;
    4. **non-maximum suppression** (local maxima + greedy min-distance thinning)
       to extract the strongest, well-separated responses as detected centres.

Design notes
------------
* Geometry models live in a small **registry** (:data:`GEOMETRY_MODELS`) keyed by
  ``key``; each carries its own :class:`ParamSpec` list so the UI can build its
  parameter widgets generically and so new shapes drop in without touching the UI.
* ``zero_mean=True`` removes the kernel DC component → the convolution responds to
  the *contrast/shape* of the pattern rather than raw local density (true template
  matching / blob detection), which is usually what you want for segmentation.
* Kernels are **L2-normalised** (matched-filter SNR optimum); the response map is
  rescaled to its own max so the acceptance threshold is a density-independent
  fraction in ``[0, 1]``.

SMAP-aligned options (mirroring ``segmentNPC`` in J. Ries' SMAP):
* ``sqrt_stabilize`` — convolve the **square root** of the histogram, compressing
  bright aggregates that would otherwise dominate the response (SMAP convolves
  ``sqrt`` of the rendered image).
* ``dog`` — apply a **difference-of-Gaussians band-pass** to the response at the
  structure scale (σ ≈ 0.7·R and 2·R), suppressing slow background — a two-scale
  generalisation of the single-scale ``zero_mean`` trick.
* **ring validation** — for ring-like models, reject solid blobs by the
  inside/outside-ring localization-count ratios (SMAP's ``NPCsegmentCleanup``).

Pure NumPy/SciPy — Qt-free and unit-testable. Distances are in nm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

DEFAULT_PIXEL_NM = 4.0
DEFAULT_MIN_RESPONSE = 0.30        # fraction of the max response
DEFAULT_ZERO_MEAN = True
DEFAULT_DOG = False
DEFAULT_SQRT = False
DEFAULT_MAX_INSIDE = 0.5           # ring validation: max inside/ring loc-count ratio
DEFAULT_MAX_OUTSIDE = 2.0          # ring validation: max outside/ring loc-count ratio
_MIN_RING_LOCS = 6                 # ring band must hold at least this many locs


# --------------------------------------------------------------------------- #
# Geometry-model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParamSpec:
    """One numeric parameter of a geometry model (drives the UI spin box)."""

    name: str
    label: str
    default: float
    lo: float
    hi: float
    step: float = 1.0
    decimals: int = 1
    suffix: str = " nm"
    tooltip: str = ""


@dataclass(frozen=True)
class GeometryModel:
    """A named shape that turns its parameters into a 2-D convolution kernel."""

    key: str
    label: str
    description: str
    params: tuple[ParamSpec, ...]
    make_kernel: Callable[[dict, float], np.ndarray]
    #: characteristic size (nm) used for the default peak-separation / ROI box
    size: Callable[[dict], float] = field(default=lambda p: 0.0)
    #: ring-like models support inside/outside-ring validation
    ringlike: bool = False
    #: ``params -> (radius_nm, rim_nm)`` for ring validation (ring-like models only)
    ring_geom: Callable[[dict], tuple] | None = None

    def defaults(self) -> dict:
        return {ps.name: ps.default for ps in self.params}


def _coord_grid(extent_nm: float, pixel_size_nm: float):
    """Square (x, y) mesh in nm spanning ``[-extent, +extent]`` at the pixel size."""
    extent_nm = max(float(extent_nm), pixel_size_nm)
    n = np.arange(-extent_nm, extent_nm + pixel_size_nm / 2.0, pixel_size_nm)
    if n.size == 0:
        n = np.array([0.0])
    return np.meshgrid(n, n)


def _ring_kernel(p: dict, px: float) -> np.ndarray:
    """Donut: peaks along the circle of radius ``diameter/2`` (NPC-style)."""
    d = float(p["diameter_nm"])
    rim = max(float(p["rim_nm"]), px * 1e-3)
    xk, yk = _coord_grid(d, px)
    r0 = d / 2.0
    return np.exp(-np.abs(xk ** 2 + yk ** 2 - r0 ** 2) / (4.0 * rim ** 2))


def _disk_kernel(p: dict, px: float) -> np.ndarray:
    """Solid disk of the given diameter with a soft (tanh) edge."""
    d = float(p["diameter_nm"])
    edge = max(float(p.get("edge_nm", px)), px * 1e-3)
    r0 = d / 2.0
    xk, yk = _coord_grid(r0 + 3.0 * edge, px)
    r = np.hypot(xk, yk)
    return 0.5 * (1.0 - np.tanh((r - r0) / edge))


def _gaussian_kernel(p: dict, px: float) -> np.ndarray:
    """Isotropic Gaussian spot of the given full-width-half-maximum."""
    fwhm = max(float(p["fwhm_nm"]), px * 1e-3)
    sigma = fwhm / 2.3548200450309493
    xk, yk = _coord_grid(3.0 * sigma, px)
    return np.exp(-(xk ** 2 + yk ** 2) / (2.0 * sigma ** 2))


def _log_kernel(p: dict, px: float) -> np.ndarray:
    """Laplacian-of-Gaussian (Mexican-hat / Ricker) blob detector.

    Positive centre, negative surround, zero-crossing at the blob radius. Tuned to
    a bright blob of the given diameter (``sigma = diameter / (2·√2)`` puts the
    zero-crossing on the blob edge). Inherently zero-mean."""
    d = float(p["diameter_nm"])
    sigma = max(d / (2.0 * np.sqrt(2.0)), px * 1e-3)
    xk, yk = _coord_grid(3.5 * sigma, px)
    r2 = xk ** 2 + yk ** 2
    s2 = sigma ** 2
    return (1.0 - r2 / (2.0 * s2)) * np.exp(-r2 / (2.0 * s2))


GEOMETRY_MODELS: dict[str, GeometryModel] = {
    "ring": GeometryModel(
        key="ring",
        label="Ring (donut)",
        description="Hollow ring — detects ring-shaped structures such as NPCs. "
                    "Matched to a circle of the given diameter with a Gaussian rim.",
        params=(
            ParamSpec("diameter_nm", "Diameter", 100.0, 10.0, 5000.0, 1.0, 1),
            ParamSpec("rim_nm", "Rim size", 20.0, 1.0, 1000.0, 1.0, 1,
                      tooltip="Radial thickness (Gaussian width) of the ring."),
        ),
        make_kernel=_ring_kernel,
        size=lambda p: float(p["diameter_nm"]),
        ringlike=True,
        ring_geom=lambda p: (float(p["diameter_nm"]) / 2.0, float(p["rim_nm"])),
    ),
    "disk": GeometryModel(
        key="disk",
        label="Disk (solid)",
        description="Filled disk — detects compact, roughly circular clusters of "
                    "the given diameter.",
        params=(
            ParamSpec("diameter_nm", "Diameter", 80.0, 4.0, 5000.0, 1.0, 1),
            ParamSpec("edge_nm", "Edge softness", 8.0, 0.5, 500.0, 0.5, 1,
                      tooltip="Width of the soft tanh falloff at the disk rim."),
        ),
        make_kernel=_disk_kernel,
        size=lambda p: float(p["diameter_nm"]),
    ),
    "gaussian": GeometryModel(
        key="gaussian",
        label="Gaussian spot",
        description="Gaussian blob — generic spot/blob detector (matched filter).",
        params=(
            ParamSpec("fwhm_nm", "FWHM", 40.0, 2.0, 5000.0, 1.0, 1),
        ),
        make_kernel=_gaussian_kernel,
        size=lambda p: float(p["fwhm_nm"]),
    ),
    "log": GeometryModel(
        key="log",
        label="LoG (Laplacian of Gaussian)",
        description="Laplacian-of-Gaussian (Mexican-hat) blob detector — positive "
                    "centre, negative surround. Responds to bright blobs of the given "
                    "diameter and suppresses uniform background (inherently zero-mean).",
        params=(
            ParamSpec("diameter_nm", "Diameter", 60.0, 4.0, 5000.0, 1.0, 1),
        ),
        make_kernel=_log_kernel,
        size=lambda p: float(p["diameter_nm"]),
    ),
}


def get_model(model_key: str) -> GeometryModel:
    try:
        return GEOMETRY_MODELS[model_key]
    except KeyError as exc:
        raise ValueError(f"Unknown geometry model: {model_key!r}") from exc


def build_kernel(model_key: str, params: dict, pixel_size_nm: float, *,
                 zero_mean: bool = DEFAULT_ZERO_MEAN) -> np.ndarray:
    """Geometry model + params → finite, L2-normalised convolution kernel.

    With ``zero_mean`` the kernel DC component is removed first so the convolution
    measures local *contrast/shape* (template matching) rather than raw density.
    """
    model = get_model(model_key)
    h = np.asarray(model.make_kernel({**model.defaults(), **params},
                                     float(pixel_size_nm)), dtype=float)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    if zero_mean and h.size:
        h = h - h.mean()
    norm = float(np.sqrt((h ** 2).sum()))
    return h / norm if norm > 0 else h


# --------------------------------------------------------------------------- #
# Histogram + convolution + NMS
# --------------------------------------------------------------------------- #
def render_histogram_2d(x: np.ndarray, y: np.ndarray, pixel_size_nm: float):
    """XY localizations → counts image ``H`` (nx, ny) + bin edges (nm)."""
    px = float(pixel_size_nm)
    xedge = np.arange(x.min(), x.max() + px, px)
    yedge = np.arange(y.min(), y.max() + px, px)
    if xedge.size < 2:
        xedge = np.array([x.min(), x.min() + px])
    if yedge.size < 2:
        yedge = np.array([y.min(), y.min() + px])
    H, _, _ = np.histogram2d(x, y, bins=[xedge, yedge])
    return H, xedge, yedge


def stabilize_histogram(H, sqrt_stabilize: bool = DEFAULT_SQRT) -> np.ndarray:
    """Optionally √-compress the counts image (SMAP convolves ``sqrt`` of the
    rendered image), down-weighting bright aggregates before convolution."""
    H = np.asarray(H, dtype=float)
    return np.sqrt(np.clip(H, 0.0, None)) if sqrt_stabilize else H


def dog_bandpass(response, radius_px: float) -> np.ndarray:
    """Difference-of-Gaussians band-pass at the structure scale (SMAP's
    ``imfilter(imfD, gauss_small) - imfilter(imfD, gauss_large)``).

    ``radius_px`` is the structure radius in pixels; the two Gaussian widths are
    ``0.7·radius`` and ``2·radius`` (per ``segmentNPC``). Suppresses background
    slower than the structure while keeping the structure-scale signal.
    """
    from scipy.ndimage import gaussian_filter

    response = np.asarray(response, dtype=float)
    if response.size == 0:
        return response
    r = max(float(radius_px), 1.0)
    return gaussian_filter(response, max(0.7 * r, 1.0)) - gaussian_filter(response, max(2.0 * r, 2.0))


def non_max_suppression(response: np.ndarray, min_height: float,
                        min_distance_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima ``>= min_height`` thinned so none are closer than
    ``min_distance_px`` (strongest kept first).

    Returns ``(rc, values)`` where ``rc`` is (K, 2) integer (row, col) and
    ``values`` the response at each kept peak (descending).
    """
    from scipy.ndimage import maximum_filter

    response = np.asarray(response, dtype=float)
    if response.size == 0:
        return np.empty((0, 2), dtype=int), np.empty(0)
    local_max = maximum_filter(response, size=3, mode="constant", cval=-np.inf)
    rows, cols = np.nonzero((response == local_max) & (response >= min_height))
    if rows.size == 0:
        return np.empty((0, 2), dtype=int), np.empty(0)
    vals = response[rows, cols]
    order = np.argsort(vals)[::-1]
    rows, cols, vals = rows[order], cols[order], vals[order]
    coords = np.column_stack([rows, cols]).astype(float)
    min_d = max(float(min_distance_px), 0.0)
    taken = np.zeros(rows.size, dtype=bool)
    keep: list[int] = []
    for i in range(rows.size):
        if taken[i]:
            continue
        keep.append(i)
        if min_d > 0:
            d = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
            taken |= (d <= min_d) & (d > 0)
    keep_arr = np.asarray(keep, dtype=int)
    return np.column_stack([rows[keep_arr], cols[keep_arr]]), vals[keep_arr]


def convolve_response(H, kernel, *, dog_radius_px: float | None = None) -> np.ndarray:
    """FFT-convolve a histogram with a kernel; rescale the result to its own max.

    With ``dog_radius_px`` a difference-of-Gaussians band-pass is applied to the
    convolution before rescaling. Rescaling makes the downstream acceptance
    threshold a density-independent fraction in ``[0, 1]``. Returned as a plain
    (possibly all-zero) array.
    """
    from scipy.signal import fftconvolve

    H = np.asarray(H, dtype=float)
    if H.size == 0 or np.asarray(kernel).size == 0:
        return np.zeros_like(H)
    response = fftconvolve(H, np.asarray(kernel, dtype=float), mode="same")
    if dog_radius_px:
        response = dog_bandpass(response, dog_radius_px)
    rmax = float(response.max()) if response.size else 0.0
    return response / rmax if rmax > 0 else response


def detections_from_response(response, xedge, yedge, *, min_response: float,
                             min_distance_px: float,
                             max_detections: int | None = None):
    """NMS on a response map → ``(centers_nm (K, 2), peak_values)`` (descending).

    Cheap relative to the convolution, so the UI can re-run just this when the
    threshold / separation / cap change without rebuilding the response map.
    """
    response = np.asarray(response, dtype=float)
    xedge = np.asarray(xedge, dtype=float)
    yedge = np.asarray(yedge, dtype=float)
    peaks_rc, vals = non_max_suppression(response, float(min_response),
                                         max(float(min_distance_px), 1.0))
    if peaks_rc.shape[0] == 0 or xedge.size < 2 or yedge.size < 2:
        return np.empty((0, 2)), np.empty(0)
    if max_detections is not None and peaks_rc.shape[0] > max_detections:
        peaks_rc, vals = peaks_rc[:max_detections], vals[:max_detections]
    xcenter = 0.5 * (xedge[:-1] + xedge[1:])
    ycenter = 0.5 * (yedge[:-1] + yedge[1:])
    centers = np.column_stack([xcenter[peaks_rc[:, 0]], ycenter[peaks_rc[:, 1]]])
    return centers, vals


# --------------------------------------------------------------------------- #
# Ring validation (reject solid blobs) — SMAP NPCsegmentCleanup inside/outside
# --------------------------------------------------------------------------- #
def ring_inside_outside_ratios(xy, centers, radius_nm: float, rim_nm: float):
    """Per-detection localization-count ratios about a ring of radius ``R``.

    With ``half = rim/2``: the **ring** band is ``[R-half, R+half]``, the
    **inside** is the hole ``r < R-half`` and the **outside** is the shell
    ``(R+half, R+half+rim]``. Returns ``(inside_ratio, outside_ratio, n_ring)``
    arrays where the ratios are ``n_inside/n_ring`` and ``n_outside/n_ring``
    (NaN where the ring band is empty). A solid blob → high inside ratio.
    """
    xy = np.asarray(xy, dtype=float)
    xy = xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])] if xy.size else xy.reshape(0, 2)
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    K = centers.shape[0]
    inside = np.full(K, np.nan)
    outside = np.full(K, np.nan)
    n_ring = np.zeros(K, dtype=int)
    if K == 0 or xy.shape[0] == 0:
        return inside, outside, n_ring
    R = float(radius_nm)
    half = max(float(rim_nm) / 2.0, np.finfo(float).eps)
    inner = max(0.0, R - half)
    outer = R + half
    out2 = outer + float(rim_nm)
    inner2, outer2, out2_2 = inner ** 2, outer ** 2, out2 ** 2
    x, y = xy[:, 0], xy[:, 1]
    for k, (cx, cy) in enumerate(centers):
        dx, dy = x - cx, y - cy
        box = (np.abs(dx) <= out2) & (np.abs(dy) <= out2)
        if not box.any():
            continue
        r2 = dx[box] ** 2 + dy[box] ** 2
        nr = int(np.count_nonzero((r2 >= inner2) & (r2 <= outer2)))
        n_ring[k] = nr
        if nr == 0:
            continue
        ni = int(np.count_nonzero(r2 < inner2))
        no = int(np.count_nonzero((r2 > outer2) & (r2 <= out2_2)))
        inside[k] = ni / nr
        outside[k] = no / nr
    return inside, outside, n_ring


def ring_validation_mask(xy, centers, radius_nm: float, rim_nm: float, *,
                         max_inside: float = DEFAULT_MAX_INSIDE,
                         max_outside: float | None = DEFAULT_MAX_OUTSIDE,
                         min_ring_locs: int = _MIN_RING_LOCS):
    """Keep-mask for ring-like detections: enough ring locs and inside/outside
    ratios within bounds. Returns ``(keep, inside_ratio, outside_ratio)``."""
    inside, outside, n_ring = ring_inside_outside_ratios(xy, centers, radius_nm, rim_nm)
    keep = np.isfinite(inside) & (n_ring >= int(min_ring_locs)) & (inside <= float(max_inside))
    if max_outside is not None:
        keep &= np.isfinite(outside) & (outside <= float(max_outside))
    return keep, inside, outside


def segment_by_convolution(xy_nm, *, model_key: str, params: dict | None = None,
                           pixel_size_nm: float = DEFAULT_PIXEL_NM,
                           min_response: float = DEFAULT_MIN_RESPONSE,
                           min_distance_nm: float | None = None,
                           zero_mean: bool = DEFAULT_ZERO_MEAN,
                           sqrt_stabilize: bool = DEFAULT_SQRT,
                           dog: bool = DEFAULT_DOG,
                           ring_validation: dict | None = None,
                           max_detections: int | None = None) -> dict:
    """Detect structures of a given geometry by template-matched convolution.

    Parameters
    ----------
    xy_nm : (N, 2+) array
        Localization coordinates in nm; only the first two columns are used.
    model_key, params :
        Geometry model and its parameters (missing keys fall back to defaults).
    min_response :
        Acceptance threshold as a fraction of the max response, in ``[0, 1]``.
    min_distance_nm :
        Minimum centre-to-centre separation; defaults to the model's size.
    sqrt_stabilize :
        √-compress the histogram before convolution (down-weight aggregates).
    dog :
        Apply a difference-of-Gaussians band-pass to the response at the
        structure scale.
    ring_validation :
        For ring-like models, a dict ``{max_inside, max_outside, min_ring_locs}``
        to reject solid blobs by the inside/outside-ring loc-count ratios.

    Returns a dict with ``response_map`` (rescaled to max 1), ``xedge``/``yedge``,
    ``kernel``, ``centers`` (nm, (K, 2)), ``peak_values`` (descending) and, when
    ring validation runs, ``inside_ratio`` / ``outside_ratio`` per detection.
    """
    model = get_model(model_key)
    params = {**model.defaults(), **(params or {})}
    px = float(pixel_size_nm)

    xy = np.asarray(xy_nm, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        xy = xy.reshape(-1, 2) if xy.size else np.empty((0, 2))
    xy = xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])]

    kernel = build_kernel(model_key, params, px, zero_mean=zero_mean)
    out = {"response_map": np.zeros((0, 0)), "xedge": np.zeros(0), "yedge": np.zeros(0),
           "kernel": kernel, "centers": np.empty((0, 2)), "peak_values": np.empty(0),
           "inside_ratio": np.empty(0), "outside_ratio": np.empty(0),
           "model_key": model_key, "params": params, "pixel_size_nm": px}
    if xy.shape[0] < 3:
        return out

    H, xedge, yedge = render_histogram_2d(xy[:, 0], xy[:, 1], px)
    H = stabilize_histogram(H, sqrt_stabilize)
    dog_radius_px = (0.5 * model.size(params) / px) if dog else None
    response = convolve_response(H, kernel, dog_radius_px=dog_radius_px)
    out.update(response_map=response, xedge=xedge, yedge=yedge)

    size_nm = min_distance_nm if min_distance_nm is not None else model.size(params)
    centers, vals = detections_from_response(
        response, xedge, yedge, min_response=float(min_response),
        min_distance_px=float(size_nm) / px, max_detections=max_detections)

    if ring_validation is not None and model.ringlike and model.ring_geom and centers.shape[0]:
        radius_nm, rim_nm = model.ring_geom(params)
        keep, inside, outside = ring_validation_mask(
            xy, centers, radius_nm, rim_nm,
            max_inside=ring_validation.get("max_inside", DEFAULT_MAX_INSIDE),
            max_outside=ring_validation.get("max_outside", DEFAULT_MAX_OUTSIDE),
            min_ring_locs=ring_validation.get("min_ring_locs", _MIN_RING_LOCS))
        centers, vals = centers[keep], vals[keep]
        out.update(inside_ratio=inside[keep], outside_ratio=outside[keep])

    out.update(centers=centers, peak_values=vals)
    return out
