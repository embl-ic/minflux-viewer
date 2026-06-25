"""
minflux_viewer.analysis.conv_segmentation_3d
=============================================
**3-D convolution-based segmentation** — the volumetric counterpart of
:mod:`minflux_viewer.analysis.conv_segmentation`.

The data **and** the kernel are genuinely 3-D: the localizations are voxelised
with :func:`numpy.histogramdd` and convolved with a 3-D geometry kernel, so no
spatial information is lost along *z* (unlike running 2-D convolution on three
orthogonal max-projections, which collapses an axis before the kernel sees it).
Only the *viewing* of the resulting response volume is reduced to orthogonal
slices in the UI — the detection / peak-finding stays fully 3-D.

Pipeline (3-D):
    1. voxelise the XYZ localizations into a ``(pixel_xy, pixel_xy, pixel_z)`` grid;
    2. build a convolutional **kernel** from a user-chosen *geometry model*
       (spherical shell / ball / gaussian / LoG) at the same voxel sizes;
    3. FFT (overlap-add) convolve the volume with the kernel (matched filter) to get
       a *response volume* — high where the local point pattern resembles the model;
    4. **3-D non-maximum suppression** (27-neighbourhood local maxima + greedy
       physical-distance thinning) to extract the strongest, well-separated
       responses as detected centres ``(x, y, z)`` in nm.

Design mirrors the 2-D module so the two can later be merged behind one UI:
* geometry models live in a registry (:data:`GEOMETRY_MODELS_3D`) keyed by
  ``key``; each carries the shared :class:`conv_segmentation.ParamSpec` list;
* ``zero_mean`` removes the kernel DC component (template matching vs raw density);
* kernels are L2-normalised; the response is rescaled to its own max so the
  acceptance threshold is a density-independent fraction in ``[0, 1]``.

Anisotropic voxels are handled naturally: the kernel grid is sampled at the xy and
z pixel sizes but coordinates are physical nm, so ``r² = X² + Y² + Z²`` (nm²) is
correct regardless of the z/xy sampling ratio.

Pure NumPy/SciPy — Qt-free and unit-testable. Distances are in nm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# Reuse the 2-D parameter-spec dataclass so the UI builds widgets identically and
# the two tools can later share one dialog.
from .conv_segmentation import ParamSpec, stabilize_histogram  # re-exported

DEFAULT_PIXEL_NM = 8.0             # xy voxel size (nm)
DEFAULT_Z_PIXEL_NM = 16.0          # z voxel size (nm) — usually coarser than xy
DEFAULT_MIN_RESPONSE = 0.30        # fraction of the max response
DEFAULT_ZERO_MEAN = True
DEFAULT_DOG = False
DEFAULT_SQRT = False
DEFAULT_MAX_VOXELS = 8_000_000     # voxel-grid cap (matches the 3-D volume preview)

__all__ = [
    "ParamSpec", "stabilize_histogram",
    "GeometryModel3D", "GEOMETRY_MODELS_3D", "get_model",
    "build_kernel_3d", "render_histogram_3d", "convolve_response_3d",
    "dog_bandpass_3d", "non_max_suppression_3d", "detections_from_response_3d",
    "segment_by_convolution_3d", "estimate_voxel_count", "suggest_voxel_sizes",
    "DEFAULT_PIXEL_NM", "DEFAULT_Z_PIXEL_NM", "DEFAULT_MIN_RESPONSE",
    "DEFAULT_ZERO_MEAN", "DEFAULT_DOG", "DEFAULT_SQRT", "DEFAULT_MAX_VOXELS",
]


# --------------------------------------------------------------------------- #
# Geometry-model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeometryModel3D:
    """A named 3-D shape that turns its parameters into a 3-D convolution kernel."""

    key: str
    label: str
    description: str
    params: tuple[ParamSpec, ...]
    #: ``(params, pixel_xy_nm, pixel_z_nm) -> 3-D kernel``
    make_kernel: Callable[[dict, float, float], np.ndarray]
    #: characteristic size (nm) used for the default peak separation
    size: Callable[[dict], float] = field(default=lambda p: 0.0)

    def defaults(self) -> dict:
        return {ps.name: ps.default for ps in self.params}


def _coord_grid_3d(extent_xy_nm: float, extent_z_nm: float,
                   px_xy: float, px_z: float):
    """``(X, Y, Z)`` meshes in nm spanning ``[-extent, +extent]`` per axis at the
    respective voxel sizes (xy isotropic, z separate)."""
    extent_xy_nm = max(float(extent_xy_nm), px_xy)
    extent_z_nm = max(float(extent_z_nm), px_z)
    nxy = np.arange(-extent_xy_nm, extent_xy_nm + px_xy / 2.0, px_xy)
    nz = np.arange(-extent_z_nm, extent_z_nm + px_z / 2.0, px_z)
    if nxy.size == 0:
        nxy = np.array([0.0])
    if nz.size == 0:
        nz = np.array([0.0])
    return np.meshgrid(nxy, nxy, nz, indexing="ij")


def _shell_kernel(p: dict, px_xy: float, px_z: float) -> np.ndarray:
    """Hollow sphere: peaks on the spherical surface of radius ``diameter/2``
    (3-D analogue of the 2-D ring/donut)."""
    d = float(p["diameter_nm"])
    rim = max(float(p["rim_nm"]), px_xy * 1e-3)
    X, Y, Z = _coord_grid_3d(d, d, px_xy, px_z)
    r0 = d / 2.0
    return np.exp(-np.abs(X ** 2 + Y ** 2 + Z ** 2 - r0 ** 2) / (4.0 * rim ** 2))


def _ball_kernel(p: dict, px_xy: float, px_z: float) -> np.ndarray:
    """Solid sphere of the given diameter with a soft (tanh) edge (3-D disk)."""
    d = float(p["diameter_nm"])
    edge = max(float(p.get("edge_nm", px_xy)), px_xy * 1e-3)
    r0 = d / 2.0
    X, Y, Z = _coord_grid_3d(r0 + 3.0 * edge, r0 + 3.0 * edge, px_xy, px_z)
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    return 0.5 * (1.0 - np.tanh((r - r0) / edge))


def _gaussian_kernel(p: dict, px_xy: float, px_z: float) -> np.ndarray:
    """Isotropic 3-D Gaussian spot of the given full-width-half-maximum."""
    fwhm = max(float(p["fwhm_nm"]), px_xy * 1e-3)
    sigma = fwhm / 2.3548200450309493
    X, Y, Z = _coord_grid_3d(3.0 * sigma, 3.0 * sigma, px_xy, px_z)
    return np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2.0 * sigma ** 2))


def _log_kernel(p: dict, px_xy: float, px_z: float) -> np.ndarray:
    """3-D Laplacian-of-Gaussian (Mexican-hat) blob detector.

    Positive centre, negative surround. ``sigma = diameter / (2·√3)`` puts the
    zero-crossing (``r = √3·σ`` in 3-D) on the blob edge. Inherently zero-mean."""
    d = float(p["diameter_nm"])
    sigma = max(d / (2.0 * np.sqrt(3.0)), px_xy * 1e-3)
    X, Y, Z = _coord_grid_3d(3.5 * sigma, 3.5 * sigma, px_xy, px_z)
    r2 = X ** 2 + Y ** 2 + Z ** 2
    s2 = sigma ** 2
    return (3.0 - r2 / s2) * np.exp(-r2 / (2.0 * s2))


GEOMETRY_MODELS_3D: dict[str, GeometryModel3D] = {
    "shell": GeometryModel3D(
        key="shell",
        label="Spherical shell (hollow)",
        description="Hollow sphere — detects shell-shaped structures of the given "
                    "diameter (3-D analogue of the 2-D ring). Matched to a sphere "
                    "surface with a Gaussian rim.",
        params=(
            ParamSpec("diameter_nm", "Diameter", 100.0, 10.0, 5000.0, 1.0, 1),
            ParamSpec("rim_nm", "Rim size", 20.0, 1.0, 1000.0, 1.0, 1,
                      tooltip="Radial thickness (Gaussian width) of the shell."),
        ),
        make_kernel=_shell_kernel,
        size=lambda p: float(p["diameter_nm"]),
    ),
    "ball": GeometryModel3D(
        key="ball",
        label="Ball (solid sphere)",
        description="Filled sphere — detects compact, roughly spherical clusters "
                    "of the given diameter (3-D analogue of the 2-D disk).",
        params=(
            ParamSpec("diameter_nm", "Diameter", 80.0, 4.0, 5000.0, 1.0, 1),
            ParamSpec("edge_nm", "Edge softness", 8.0, 0.5, 500.0, 0.5, 1,
                      tooltip="Width of the soft tanh falloff at the ball surface."),
        ),
        make_kernel=_ball_kernel,
        size=lambda p: float(p["diameter_nm"]),
    ),
    "gaussian": GeometryModel3D(
        key="gaussian",
        label="Gaussian spot",
        description="3-D Gaussian blob — generic volumetric spot/blob detector "
                    "(matched filter).",
        params=(
            ParamSpec("fwhm_nm", "FWHM", 40.0, 2.0, 5000.0, 1.0, 1),
        ),
        make_kernel=_gaussian_kernel,
        size=lambda p: float(p["fwhm_nm"]),
    ),
    "log": GeometryModel3D(
        key="log",
        label="LoG (Laplacian of Gaussian)",
        description="3-D Laplacian-of-Gaussian (Mexican-hat) blob detector — "
                    "positive centre, negative surround. Responds to bright blobs of "
                    "the given diameter and suppresses uniform background "
                    "(inherently zero-mean).",
        params=(
            ParamSpec("diameter_nm", "Diameter", 60.0, 4.0, 5000.0, 1.0, 1),
        ),
        make_kernel=_log_kernel,
        size=lambda p: float(p["diameter_nm"]),
    ),
}


def get_model(model_key: str) -> GeometryModel3D:
    try:
        return GEOMETRY_MODELS_3D[model_key]
    except KeyError as exc:
        raise ValueError(f"Unknown 3-D geometry model: {model_key!r}") from exc


def build_kernel_3d(model_key: str, params: dict, pixel_size_nm: float,
                    z_pixel_size_nm: float, *,
                    zero_mean: bool = DEFAULT_ZERO_MEAN) -> np.ndarray:
    """Geometry model + params → finite, L2-normalised 3-D convolution kernel.

    With ``zero_mean`` the kernel DC component is removed first so the convolution
    measures local *contrast/shape* (template matching) rather than raw density.
    """
    model = get_model(model_key)
    h = np.asarray(model.make_kernel({**model.defaults(), **params},
                                     float(pixel_size_nm), float(z_pixel_size_nm)),
                   dtype=float)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    if zero_mean and h.size:
        h = h - h.mean()
    norm = float(np.sqrt((h ** 2).sum()))
    return h / norm if norm > 0 else h


# --------------------------------------------------------------------------- #
# Voxelise + convolve + NMS
# --------------------------------------------------------------------------- #
def estimate_voxel_count(xyz: np.ndarray, pixel_size_nm: float,
                         z_pixel_size_nm: float) -> int:
    """Number of voxels the grid would have for *xyz* at the given voxel sizes
    (cheap guard against an enormous volume before any allocation)."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[0] == 0 or xyz.shape[1] < 3:
        return 0
    px = float(pixel_size_nm)
    pz = float(z_pixel_size_nm)
    nx = int((np.nanmax(xyz[:, 0]) - np.nanmin(xyz[:, 0])) / px) + 2
    ny = int((np.nanmax(xyz[:, 1]) - np.nanmin(xyz[:, 1])) / px) + 2
    nz = int((np.nanmax(xyz[:, 2]) - np.nanmin(xyz[:, 2])) / pz) + 2
    return int(nx) * int(ny) * int(nz)


def suggest_voxel_sizes(xyz: np.ndarray, *, max_voxels: int = DEFAULT_MAX_VOXELS,
                        z_ratio: float = 2.0, round_to: float = 1.0,
                        min_nm: float = 1.0) -> tuple[float, float]:
    """Smallest ``(pixel_xy, pixel_z = z_ratio · pixel_xy)`` whose voxel grid fits
    within ``max_voxels`` for *xyz*.

    MINFLUX fields of view are often several µm wide, so the fixed default voxel
    size can blow past the cap; the UI uses this to pick a working default instead
    of refusing to render. Falls back to ``(min_nm, z_ratio·min_nm)`` when the data
    is degenerate.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[0] == 0 or xyz.shape[1] < 3:
        return float(min_nm), float(min_nm) * float(z_ratio)
    z_ratio = max(float(z_ratio), 1e-6)
    sx = float(np.ptp(xyz[:, 0]))
    sy = float(np.ptp(xyz[:, 1]))
    sz = float(np.ptp(xyz[:, 2]))
    # analytic seed from volume / (z_ratio · cap), then round up and verify the
    # exact (padded) voxel count so we never return an over-cap size
    seed = (max(sx * sy * sz, 0.0) / (z_ratio * max(int(max_voxels), 1))) ** (1.0 / 3.0)
    px = max(float(min_nm), float(np.ceil(max(seed, min_nm) / round_to) * round_to))
    while estimate_voxel_count(xyz, px, z_ratio * px) > max_voxels:
        px += round_to
    return float(px), float(z_ratio * px)


def render_histogram_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        pixel_size_nm: float, z_pixel_size_nm: float):
    """XYZ localizations → counts volume ``H`` (nx, ny, nz) + bin edges (nm)."""
    px = float(pixel_size_nm)
    pz = float(z_pixel_size_nm)
    xedge = np.arange(x.min(), x.max() + px, px)
    yedge = np.arange(y.min(), y.max() + px, px)
    zedge = np.arange(z.min(), z.max() + pz, pz)
    if xedge.size < 2:
        xedge = np.array([x.min(), x.min() + px])
    if yedge.size < 2:
        yedge = np.array([y.min(), y.min() + px])
    if zedge.size < 2:
        zedge = np.array([z.min(), z.min() + pz])
    H, _ = np.histogramdd(np.column_stack([x, y, z]), bins=[xedge, yedge, zedge])
    return H.astype(np.float32, copy=False), xedge, yedge, zedge


def dog_bandpass_3d(response: np.ndarray, radius_px) -> np.ndarray:
    """Difference-of-Gaussians band-pass at the structure scale (3-D).

    ``radius_px`` is the structure radius in voxels — a scalar or a per-axis
    sequence ``(rx, ry, rz)``. The two Gaussian widths are ``0.7·radius`` and
    ``2·radius``; suppresses background slower than the structure.
    """
    from scipy.ndimage import gaussian_filter

    response = np.asarray(response, dtype=float)
    if response.size == 0:
        return response
    r = np.atleast_1d(np.asarray(radius_px, dtype=float))
    small = np.maximum(0.7 * r, 1.0)
    large = np.maximum(2.0 * r, 2.0)
    small = small if small.size > 1 else float(small[0])
    large = large if large.size > 1 else float(large[0])
    return gaussian_filter(response, small) - gaussian_filter(response, large)


def convolve_response_3d(H, kernel, *, dog_radius_px=None) -> np.ndarray:
    """Overlap-add FFT-convolve a volume with a 3-D kernel; rescale to its own max.

    ``oaconvolve`` (overlap-add) bounds the transient memory for a small kernel
    over a large volume — important in 3-D. With ``dog_radius_px`` a 3-D
    difference-of-Gaussians band-pass is applied before rescaling. Rescaling makes
    the downstream acceptance threshold a density-independent fraction in ``[0, 1]``.
    """
    from scipy.signal import oaconvolve

    H = np.asarray(H, dtype=float)
    if H.size == 0 or np.asarray(kernel).size == 0:
        return np.zeros_like(H)
    response = oaconvolve(H, np.asarray(kernel, dtype=float), mode="same")
    if dog_radius_px is not None:
        response = dog_bandpass_3d(response, dog_radius_px)
    rmax = float(response.max()) if response.size else 0.0
    return response / rmax if rmax > 0 else response


def non_max_suppression_3d(response: np.ndarray, min_height: float,
                           min_distance_nm: float, voxel_nm) -> tuple[np.ndarray, np.ndarray]:
    """3-D local maxima ``>= min_height`` thinned so no two are closer than
    ``min_distance_nm`` in physical space (strongest kept first).

    ``voxel_nm`` is ``(vx, vy, vz)``; index distances are scaled to nm so the
    separation is correct for anisotropic voxels. Returns ``(ijk, values)`` where
    ``ijk`` is (K, 3) integer voxel indices and ``values`` the response at each
    kept peak (descending).
    """
    from scipy.ndimage import maximum_filter

    response = np.asarray(response, dtype=float)
    if response.ndim != 3 or response.size == 0:
        return np.empty((0, 3), dtype=int), np.empty(0)
    local_max = maximum_filter(response, size=3, mode="constant", cval=-np.inf)
    ii, jj, kk = np.nonzero((response == local_max) & (response >= min_height))
    if ii.size == 0:
        return np.empty((0, 3), dtype=int), np.empty(0)
    vals = response[ii, jj, kk]
    order = np.argsort(vals)[::-1]
    ii, jj, kk, vals = ii[order], jj[order], kk[order], vals[order]
    vx, vy, vz = (float(v) for v in voxel_nm)
    coords = np.column_stack([ii * vx, jj * vy, kk * vz]).astype(float)  # nm
    min_d = max(float(min_distance_nm), 0.0)
    taken = np.zeros(ii.size, dtype=bool)
    keep: list[int] = []
    for i in range(ii.size):
        if taken[i]:
            continue
        keep.append(i)
        if min_d > 0:
            d = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
            taken |= (d <= min_d) & (d > 0)
    keep_arr = np.asarray(keep, dtype=int)
    return (np.column_stack([ii[keep_arr], jj[keep_arr], kk[keep_arr]]),
            vals[keep_arr])


def detections_from_response_3d(response, xedge, yedge, zedge, *,
                                min_response: float, min_distance_nm: float,
                                max_detections: int | None = None):
    """3-D NMS on a response volume → ``(centers_nm (K, 3), peak_values)``
    (descending). Cheap relative to the convolution, so the UI can re-run just
    this when the threshold / separation / cap change.
    """
    response = np.asarray(response, dtype=float)
    xedge = np.asarray(xedge, dtype=float)
    yedge = np.asarray(yedge, dtype=float)
    zedge = np.asarray(zedge, dtype=float)
    if (response.ndim != 3 or response.size == 0
            or xedge.size < 2 or yedge.size < 2 or zedge.size < 2):
        return np.empty((0, 3)), np.empty(0)
    voxel_nm = (float(xedge[1] - xedge[0]), float(yedge[1] - yedge[0]),
                float(zedge[1] - zedge[0]))
    ijk, vals = non_max_suppression_3d(response, float(min_response),
                                       float(min_distance_nm), voxel_nm)
    if ijk.shape[0] == 0:
        return np.empty((0, 3)), np.empty(0)
    if max_detections is not None and ijk.shape[0] > max_detections:
        ijk, vals = ijk[:max_detections], vals[:max_detections]
    xc = 0.5 * (xedge[:-1] + xedge[1:])
    yc = 0.5 * (yedge[:-1] + yedge[1:])
    zc = 0.5 * (zedge[:-1] + zedge[1:])
    centers = np.column_stack([xc[ijk[:, 0]], yc[ijk[:, 1]], zc[ijk[:, 2]]])
    return centers, vals


def segment_by_convolution_3d(xyz_nm, *, model_key: str, params: dict | None = None,
                              pixel_size_nm: float = DEFAULT_PIXEL_NM,
                              z_pixel_size_nm: float = DEFAULT_Z_PIXEL_NM,
                              min_response: float = DEFAULT_MIN_RESPONSE,
                              min_distance_nm: float | None = None,
                              zero_mean: bool = DEFAULT_ZERO_MEAN,
                              sqrt_stabilize: bool = DEFAULT_SQRT,
                              dog: bool = DEFAULT_DOG,
                              max_detections: int | None = None) -> dict:
    """Detect 3-D structures of a given geometry by template-matched convolution.

    Parameters
    ----------
    xyz_nm : (N, 3+) array
        Localization coordinates in nm; only the first three columns are used.
    model_key, params :
        Geometry model and its parameters (missing keys fall back to defaults).
    pixel_size_nm, z_pixel_size_nm :
        Voxel sizes (xy isotropic, z separate).
    min_response :
        Acceptance threshold as a fraction of the max response, in ``[0, 1]``.
    min_distance_nm :
        Minimum centre-to-centre separation; defaults to the model's size.

    Returns a dict with ``response_map`` (3-D, rescaled to max 1),
    ``xedge``/``yedge``/``zedge``, ``kernel``, ``centers`` (nm, (K, 3)) and
    ``peak_values`` (descending).
    """
    model = get_model(model_key)
    params = {**model.defaults(), **(params or {})}
    px = float(pixel_size_nm)
    pz = float(z_pixel_size_nm)

    xyz = np.asarray(xyz_nm, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        xyz = xyz.reshape(-1, 3) if xyz.size else np.empty((0, 3))
    xyz = xyz[np.all(np.isfinite(xyz[:, :3]), axis=1)]

    kernel = build_kernel_3d(model_key, params, px, pz, zero_mean=zero_mean)
    out = {"response_map": np.zeros((0, 0, 0)),
           "xedge": np.zeros(0), "yedge": np.zeros(0), "zedge": np.zeros(0),
           "kernel": kernel, "centers": np.empty((0, 3)), "peak_values": np.empty(0),
           "model_key": model_key, "params": params,
           "pixel_size_nm": px, "z_pixel_size_nm": pz}
    if xyz.shape[0] < 3:
        return out

    H, xedge, yedge, zedge = render_histogram_3d(
        xyz[:, 0], xyz[:, 1], xyz[:, 2], px, pz)
    H = stabilize_histogram(H, sqrt_stabilize)
    if dog:
        rad = 0.5 * model.size(params)
        dog_radius_px = (rad / px, rad / px, rad / pz)
    else:
        dog_radius_px = None
    response = convolve_response_3d(H, kernel, dog_radius_px=dog_radius_px)
    out.update(response_map=response, xedge=xedge, yedge=yedge, zedge=zedge)

    size_nm = min_distance_nm if min_distance_nm is not None else model.size(params)
    centers, vals = detections_from_response_3d(
        response, xedge, yedge, zedge, min_response=float(min_response),
        min_distance_nm=float(size_nm), max_detections=max_detections)
    out.update(centers=centers, peak_values=vals)
    return out
