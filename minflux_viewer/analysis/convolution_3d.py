"""Region-focused XYZ matched-filter convolution, independent of the 2D tool.

Arrays and voxel tuples are always X,Y,Z (not image Z,Y,X). Display projections
are deliberately separate from these exact scientific grids. Templates are
axis-aligned; NPC two-ring detection assumes a ring normal along display Z.

The grid extends one template half-width (a halo) beyond the requested region on
every side, and localizations in that margin are binned too, so a structure at
the region's edge is scored on all of its data. Peaks are reported inside the
region only, and the response is scaled to the region's maximum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.fft import next_fast_len, set_workers
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from minflux_viewer.core.execution import cpu_threads

MODELS = {"gaussian": "Gaussian 3D", "shell": "Spherical shell",
          "npc": "NPC two rings (normal Z)"}

# Measured peak resident memory of volume_maps + volume_detections on Windows /
# SciPy pocketfft: 46-49 bytes per padded FFT voxel, image arrays included.
# The estimate keeps roughly 1.4x headroom over that.
_BYTES_PER_PADDED_VOXEL = 64
_BYTES_PER_IMAGE_VOXEL = 32
_BYTES_PER_KERNEL_VOXEL = 64
_BYTES_PER_POINT = 96


@dataclass(frozen=True)
class VolumeRequest:
    bounds_xyz: tuple[tuple[float, float], ...]
    voxel_xyz: tuple[float, float, float] = (4., 4., 4.)
    model: str = "npc"
    diameter_nm: float = 100.
    width_nm: float = 10.
    ring_separation_nm: float = 50.
    zero_mean: bool = True
    sqrt_counts: bool = True
    budget_bytes: int | None = None
    max_voxels: int = 8_000_000


@dataclass(frozen=True)
class VolumeEstimate:
    region_shape: tuple[int, ...]
    halo_voxels: tuple[int, ...]
    image_shape: tuple[int, ...]
    kernel_shape: tuple[int, ...]
    fft_shape: tuple[int, ...]
    host_bytes: int


def fft_shape(image_shape, kernel_shape) -> tuple[int, ...]:
    """The padded size SciPy's real FFT convolution uses for these shapes."""
    return tuple(next_fast_len(int(a) + int(b) - 1, real=True)
                 for a, b in zip(image_shape, kernel_shape))


def estimate_volume(request: VolumeRequest, points=0) -> VolumeEstimate:
    if request.model not in MODELS:
        raise ValueError("Unknown 3D convolution model")
    bounds, voxel = np.asarray(request.bounds_xyz), np.asarray(request.voxel_xyz)
    if (bounds.shape != (3, 2) or voxel.shape != (3,) or not np.isfinite(bounds).all()
            or not np.isfinite(voxel).all() or np.any(voxel <= 0)
            or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("Specify finite increasing XYZ bounds and positive XYZ voxel sizes")
    sizes = (request.diameter_nm, request.width_nm, request.ring_separation_nm)
    if not np.isfinite(sizes).all() or min(sizes) <= 0:
        raise ValueError("3D model dimensions must be finite and positive")
    radius, width = request.diameter_nm / 2., request.width_nm
    extent = ((4 * width,) * 3 if request.model == "gaussian" else
              (radius + 4 * width,) * 3 if request.model == "shell" else
              (radius + 4 * width, radius + 4 * width, request.ring_separation_nm / 2. + 4 * width))
    region = tuple(max(1, math.ceil(float((b[1]-b[0])/v))) for b, v in zip(bounds, voxel))
    halo = tuple(math.ceil(float(e/v)) for e, v in zip(extent, voxel))
    kernel = tuple(2 * h + 1 for h in halo)
    image = tuple(n + 2 * h for n, h in zip(region, halo))
    padded = fft_shape(image, kernel)
    host = (_BYTES_PER_PADDED_VOXEL * math.prod(padded) + _BYTES_PER_IMAGE_VOXEL * math.prod(image)
            + _BYTES_PER_KERNEL_VOXEL * math.prod(kernel) + _BYTES_PER_POINT * points)
    return VolumeEstimate(region, halo, image, kernel, padded, host)


def preflight_volume(request: VolumeRequest, points=0) -> VolumeEstimate:
    """Validate the request; ``budget_bytes`` is an optional caller-chosen cap.

    The viewer leaves it unset and submits ``host_bytes`` to the shared job
    admission (Preferences > Data) instead of keeping a second memory limit.
    """
    estimate = estimate_volume(request, points)
    if type(request.max_voxels) is not int or not 1 <= request.max_voxels <= 64_000_000:
        raise ValueError("3D voxel budget must be in [1, 64000000]")
    if math.prod(estimate.image_shape) > request.max_voxels:
        raise ValueError(f"3D region and its edge margin need {math.prod(estimate.image_shape):,} voxels; "
                         f"limit {request.max_voxels:,}. Reduce the region or increase the voxel size.")
    if request.budget_bytes is not None:
        if type(request.budget_bytes) is not int or request.budget_bytes < 1:
            raise ValueError("3D memory budget must be positive")
        if estimate.host_bytes > request.budget_bytes:
            raise ValueError(f"3D memory estimate {estimate.host_bytes / 1024**2:,.1f} MiB exceeds "
                             f"{request.budget_bytes / 1024**2:,.1f} MiB budget; reduce region or raise budget.")
    return estimate


def build_volume_kernel(request: VolumeRequest, shape):
    axes = [(np.arange(n) - n//2) * v for n, v in zip(shape, request.voxel_xyz)]
    x, y, z = np.meshgrid(*axes, indexing="ij", sparse=True)
    width, radius = request.width_nm, request.diameter_nm / 2.
    if request.model == "gaussian":
        kernel = np.exp(-(x*x + y*y + z*z) / (2 * width*width))
    elif request.model == "shell":
        kernel = np.exp(-((np.sqrt(x*x + y*y + z*z) - radius) / width)**2 / 2)
    else:
        radial = np.exp(-((np.sqrt(x*x + y*y) - radius) / width)**2 / 2)
        half = request.ring_separation_nm / 2.
        kernel = radial * (np.exp(-((z-half)/width)**2 / 2) + np.exp(-((z+half)/width)**2 / 2))
    if request.zero_mean:
        kernel -= kernel.mean()
    norm = np.linalg.norm(kernel.ravel())
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("3D kernel is degenerate at this voxel size")
    return kernel / norm


def volume_maps(xyz, request: VolumeRequest, *, report=lambda text: None):
    """The exact scientific volume, on the region plus its halo (CPU, float64).

    ``response`` and ``edges`` cover the requested region only and are what the
    viewer displays; detection reads the full grid so that a peak just inside
    the region is compared with its neighbours outside.
    """
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("3D convolution needs finite display XYZ coordinates")
    estimate = preflight_volume(request, len(xyz))
    report("Building exact XYZ histogram")
    voxel = np.asarray(request.voxel_xyz, dtype=float)
    lo = np.asarray([b[0] for b in request.bounds_xyz], dtype=float)
    halo = np.asarray(estimate.halo_voxels)
    full_edges = tuple(lo[a] + (np.arange(n + 1) - halo[a]) * voxel[a]
                       for a, n in enumerate(estimate.image_shape))
    inner = tuple(slice(h, h + n) for h, n in zip(estimate.halo_voxels, estimate.region_shape))
    inner_edges = tuple(edge[s.start:s.stop + 1] for edge, s in zip(full_edges, inner))
    region = np.asarray(request.bounds_xyz, dtype=float)
    selected_count = int(((xyz >= region[:, 0]) & (xyz <= region[:, 1])).all(axis=1).sum())
    if selected_count < 3:
        raise ValueError("The selected 3D region needs at least three localizations")
    keep = ((xyz >= [e[0] for e in full_edges]) & (xyz <= [e[-1] for e in full_edges])).all(axis=1)
    histogram, _ = np.histogramdd(xyz[keep], bins=full_edges)
    kernel = build_volume_kernel(request, estimate.kernel_shape)
    values = np.sqrt(histogram) if request.sqrt_counts else histogram
    del histogram
    values /= max(float(values.max()), 1.)
    report("3D FFT convolution")
    import scipy
    with cpu_threads() as workers, set_workers(workers):
        full = fftconvolve(values, kernel, mode="same")
    del values
    provenance = dict(backend="cpu-scipy", scipy=scipy.__version__, axes=[0, 1, 2], dtype="float64",
                      mode="same", halo_voxels=list(estimate.halo_voxels), fft_shape=list(estimate.fft_shape))
    report("Normalizing 3D response")
    maximum = float(full[inner].max())
    if maximum > 0:
        full /= maximum
    else:
        full[...] = 0.
    return dict(response=full[inner], edges=inner_edges, full_response=full, full_edges=full_edges,
                inner=inner, request=request, provenance=provenance,
                selected_count=selected_count, shape=estimate.region_shape)


def volume_detections(maps, *, min_response=.3, min_distance_nm=100., max_detections=500,
                      report=lambda text: None):
    """Physical-distance XYZ suppression on full resolution, never a projection.

    Local maxima are found and suppressed on the full grid (region plus halo),
    so a stronger structure just outside the region still suppresses its side
    lobes inside it; only accepted peaks inside the region are reported.
    """
    if (not np.isfinite((min_response, min_distance_nm)).all() or not 0 <= min_response <= 1
            or min_distance_nm <= 0 or type(max_detections) is not int or not 1 <= max_detections <= 10000):
        raise ValueError("Invalid 3D response threshold, separation or detection cap")
    report("Finding 3D local maxima")
    response = maps.get("full_response", maps["response"])
    edges = maps.get("full_edges", maps["edges"])
    inner = maps.get("inner", (slice(None),) * response.ndim)
    mask = (response >= min_response) & (response > 0) & (response == maximum_filter(response, size=3))
    indices = np.argwhere(mask)
    scores = response[tuple(indices.T)]
    order = np.argsort(-scores, kind="stable")
    indices, scores = indices[order], scores[order]
    inside = np.ones(len(indices), dtype=bool)
    for axis, part in enumerate(inner):
        lo, hi = part.start or 0, response.shape[axis] if part.stop is None else part.stop
        inside &= (indices[:, axis] >= lo) & (indices[:, axis] < hi)
    voxel = np.asarray(maps["request"].voxel_xyz)
    origin = np.asarray([edge[0] for edge in edges])
    centers = origin + (indices + .5) * voxel
    if not len(centers):
        return centers, scores
    tree = cKDTree(centers)
    suppressed = np.zeros(len(centers), dtype=bool)
    accepted = []
    for i, center in enumerate(centers):
        if i % 64 == 0:
            report("Suppressing 3D maxima")
        if suppressed[i]:
            continue
        if inside[i]:
            accepted.append(i)
            if len(accepted) >= max_detections:
                break
        suppressed[tree.query_ball_point(center, min_distance_nm)] = True
    return centers[accepted], scores[accepted]
