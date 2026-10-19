"""Region-focused XYZ matched-filter convolution, independent of the 2D tool.

Arrays and voxel tuples are always X,Y,Z (not image Z,Y,X). Display projections
are deliberately separate from these exact scientific grids. Templates are
axis-aligned; NPC two-ring detection assumes a ring normal along display Z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from minflux_viewer.core.cuda_fft import CudaFftSettings, cuda_fftconvolve
from minflux_viewer.core.cuda_fft_worker import fft_shape
from minflux_viewer.core.execution import cpu_threads

MODELS = {"gaussian": "Gaussian 3D", "shell": "Spherical shell",
          "npc": "NPC two rings (normal Z)"}


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
    budget_bytes: int = 512 * 1024**2
    max_voxels: int = 8_000_000


@dataclass(frozen=True)
class VolumeEstimate:
    image_shape: tuple[int, ...]
    kernel_shape: tuple[int, ...]
    fft_shape: tuple[int, ...]
    host_bytes: int


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
    image = tuple(max(1, math.ceil(float((b[1]-b[0])/v))) for b, v in zip(bounds, voxel))
    kernel = tuple(2 * math.ceil(float(e/v)) + 1 for e, v in zip(extent, voxel))
    padded = fft_shape(image, kernel)
    host = 192 * math.prod(padded) + 128 * math.prod(image) + 64 * math.prod(kernel) + 96 * points
    return VolumeEstimate(image, kernel, padded, host)


def preflight_volume(request: VolumeRequest, points=0) -> VolumeEstimate:
    estimate = estimate_volume(request, points)
    if type(request.max_voxels) is not int or not 1 <= request.max_voxels <= 64_000_000:
        raise ValueError("3D voxel budget must be in [1, 64000000]")
    if math.prod(estimate.image_shape) > request.max_voxels:
        raise ValueError(f"3D region requires {math.prod(estimate.image_shape):,} voxels; "
                         f"limit {request.max_voxels:,}. Reduce the region or explicitly change voxel size.")
    if type(request.budget_bytes) is not int or request.budget_bytes < 1:
        raise ValueError("3D host memory budget must be positive")
    if estimate.host_bytes > request.budget_bytes:
        raise ValueError(f"3D host estimate {estimate.host_bytes / 1024**2:,.1f} MiB exceeds "
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


def volume_maps(xyz, request: VolumeRequest, *, cuda: CudaFftSettings | None,
                report=lambda text: None):
    """Exact scientific volume; cuda=None is the explicit CPU test/reference path."""
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("3D convolution needs finite display XYZ coordinates")
    estimate = preflight_volume(request, len(xyz))
    if cuda is not None:
        cuda.preflight(estimate.image_shape, estimate.kernel_shape)
    report("Building exact XYZ histogram")
    bounds = np.asarray(request.bounds_xyz)
    keep = ((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1])).all(axis=1)
    selected = xyz[keep]
    if len(selected) < 3:
        raise ValueError("The selected 3D region needs at least three localizations")
    edges = tuple(lo + np.arange(n+1) * v for (lo, _), n, v in
                  zip(request.bounds_xyz, estimate.image_shape, request.voxel_xyz))
    histogram, _ = np.histogramdd(selected, bins=edges)
    kernel = build_volume_kernel(request, estimate.kernel_shape)
    values = np.sqrt(histogram) if request.sqrt_counts else histogram.copy()
    values /= max(float(values.max()), 1.)
    report("3D FFT convolution")
    if cuda is None:
        import scipy
        from scipy.fft import set_workers
        with cpu_threads() as workers, set_workers(workers):
            response = fftconvolve(values, kernel, mode="same")
        provenance = dict(backend="cpu-scipy-reference", scipy=scipy.__version__,
                          axes=[0, 1, 2], dtype="float64", mode="same", boundary="zero")
    else:
        response, provenance = cuda_fftconvolve(values, kernel, cuda,
                                              check_cancelled=lambda: report("3D CUDA FFT"))
    report("Normalizing 3D response")
    maximum = float(response.max())
    response = response / maximum if maximum > 0 else np.zeros_like(response)
    return dict(response=response, edges=edges, request=request, provenance=provenance,
                selected_count=len(selected), shape=estimate.image_shape)


def volume_detections(maps, *, min_response=.3, min_distance_nm=100., max_detections=500,
                      report=lambda text: None):
    """Physical-distance XYZ suppression on full resolution, never a projection."""
    if (not np.isfinite((min_response, min_distance_nm)).all() or not 0 <= min_response <= 1
            or min_distance_nm <= 0 or type(max_detections) is not int or not 1 <= max_detections <= 10000):
        raise ValueError("Invalid 3D response threshold, separation or detection cap")
    report("Finding 3D local maxima")
    response = maps["response"]
    mask = (response >= min_response) & (response > 0) & (response == maximum_filter(response, size=3))
    indices = np.argwhere(mask)
    scores = response[tuple(indices.T)]
    order = np.argsort(-scores, kind="stable")
    indices, scores = indices[order], scores[order]
    voxel = np.asarray(maps["request"].voxel_xyz)
    origin = np.asarray([edge[0] for edge in maps["edges"]])
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
        accepted.append(i)
        if len(accepted) >= max_detections:
            break
        suppressed[tree.query_ball_point(center, min_distance_nm)] = True
    return centers[accepted], scores[accepted]
