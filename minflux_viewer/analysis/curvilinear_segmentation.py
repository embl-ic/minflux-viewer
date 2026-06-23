"""
minflux_viewer.analysis.curvilinear_segmentation
=================================================
Segment **curvilinear structures** (microtubules, filaments, fibres) from a
rendered localization image, then trace their centre lines.

This is a pure-Python reproduction (and refinement) of Z. Huang's 2022 Fiji
``getLineSegments`` macro for MINFLUX microtubule-width measurement, whose
pipeline is:

    1. render the XY localizations to a histogram image;
    2. *(optional)* **directional filter** — convolve with a bank of oriented
       elongated (Gaussian) line kernels and keep the per-pixel max response,
       pre-enhancing filaments of any orientation;
    3. **Hessian ridge filter** — a multi-scale **Frangi** vesselness (or the
       simpler Sato ``-λ2`` tubeness), the Hessian-eigenvalue measure that the
       Fiji "Tubeness" plugin implements;
    4. **threshold** the ridge map (relative or Otsu) → binary mask;
    5. **skeletonize** (Zhang–Suen thinning) to 1-px centre lines;
    6. **split at junctions** and trace each branch into an ordered poly-line.

The traced centre lines come back in nm coordinates as poly-line vertex lists,
ready to become open-line ROIs.

Refinement over the original: the directional kernel bank is *optional* — a
proper multi-scale Frangi filter already detects ridges at every orientation and
width, so it is the recommended default; the directional pre-filter is offered
for parity / extra contrast on faint filaments.

Pure NumPy/SciPy — Qt-free and unit-testable. Distances are in nm.
"""

from __future__ import annotations

import numpy as np

from .conv_segmentation import render_histogram_2d, stabilize_histogram

DEFAULT_PIXEL_NM = 4.0
DEFAULT_SCALE_MIN_NM = 8.0
DEFAULT_SCALE_MAX_NM = 24.0
DEFAULT_N_SCALES = 4
DEFAULT_BETA = 0.5
DEFAULT_THRESHOLD = 0.10           # fraction of the max ridge response
DEFAULT_MIN_LENGTH_NM = 200.0
RIDGE_METHODS = ("frangi", "sato")
_MAX_THIN_ITERS = 200


# --------------------------------------------------------------------------- #
# Hessian ridge (Frangi / Sato)
# --------------------------------------------------------------------------- #
def hessian_eigenvalues(image, sigma_px: float):
    """γ-normalised Hessian eigenvalues at scale ``sigma_px`` (in pixels).

    Returns ``(lam1, lam2)`` sorted **by magnitude** per pixel (|lam1| ≤ |lam2|),
    so ``lam2`` is the across-ridge principal curvature (negative on a bright
    ridge). Second derivatives are scale-normalised by ``sigma²`` (Lindeberg) so
    responses are comparable across scales.
    """
    from scipy.ndimage import gaussian_filter

    image = np.asarray(image, dtype=float)
    s = max(float(sigma_px), 1e-3)
    g2 = s * s
    haa = gaussian_filter(image, s, order=(2, 0)) * g2
    hbb = gaussian_filter(image, s, order=(0, 2)) * g2
    hab = gaussian_filter(image, s, order=(1, 1)) * g2

    mid = 0.5 * (haa + hbb)
    d = np.sqrt(np.maximum(0.25 * (haa - hbb) ** 2 + hab ** 2, 0.0))
    e1 = mid - d
    e2 = mid + d
    swap = np.abs(e1) > np.abs(e2)
    lam1 = np.where(swap, e2, e1)      # smaller magnitude
    lam2 = np.where(swap, e1, e2)      # larger magnitude (across-ridge)
    return lam1, lam2


def _frangi_scale(image, sigma_px: float, beta: float, *, black_ridges: bool):
    lam1, lam2 = hessian_eigenvalues(image, sigma_px)
    # Bright ridges have lam2 < 0; black ridges lam2 > 0. Flip sign to normalise.
    ridge = lam2 < 0 if not black_ridges else lam2 > 0
    eps = np.finfo(float).eps
    rb = np.abs(lam1) / (np.abs(lam2) + eps)        # blobness
    s = np.sqrt(lam1 ** 2 + lam2 ** 2)              # structureness
    c = 0.5 * float(s.max()) if s.size and s.max() > 0 else 1.0
    v = np.exp(-(rb ** 2) / (2.0 * beta ** 2)) * (1.0 - np.exp(-(s ** 2) / (2.0 * c ** 2)))
    return np.where(ridge, v, 0.0)


def _sato_scale(image, sigma_px: float, *, black_ridges: bool):
    _lam1, lam2 = hessian_eigenvalues(image, sigma_px)
    return np.clip(-lam2, 0.0, None) if not black_ridges else np.clip(lam2, 0.0, None)


def ridge_filter(image, sigmas_px, *, method: str = "frangi", beta: float = DEFAULT_BETA,
                 black_ridges: bool = False) -> np.ndarray:
    """Multi-scale Hessian ridge filter; per-pixel max over scales, rescaled to
    its own max so the threshold is a density-independent fraction in ``[0, 1]``.

    ``method`` is ``"frangi"`` (Frangi 1998 vesselness) or ``"sato"`` (the simpler
    ``-λ2`` tubeness). ``black_ridges=False`` detects bright ridges (filaments on
    dark background, the MINFLUX-render case).
    """
    image = np.asarray(image, dtype=float)
    sigmas = [float(s) for s in np.atleast_1d(sigmas_px) if float(s) > 0]
    if image.size == 0 or not sigmas:
        return np.zeros_like(image)
    out = np.zeros_like(image)
    for s in sigmas:
        if method == "sato":
            v = _sato_scale(image, s, black_ridges=black_ridges)
        else:
            v = _frangi_scale(image, s, beta, black_ridges=black_ridges)
        out = np.maximum(out, v)
    m = float(out.max()) if out.size else 0.0
    return out / m if m > 0 else out


def scales_from_nm(scale_min_nm: float, scale_max_nm: float, n_scales: int,
                   pixel_size_nm: float) -> np.ndarray:
    """Geometric sequence of Hessian scales (σ, in pixels) spanning the requested
    physical width range. A single scale is used when ``n_scales <= 1``."""
    px = float(pixel_size_nm)
    lo = max(float(scale_min_nm), px) / px
    hi = max(float(scale_max_nm), float(scale_min_nm)) / px
    n = max(int(n_scales), 1)
    if n == 1 or hi <= lo:
        return np.array([lo])
    return np.geomspace(lo, hi, n)


# --------------------------------------------------------------------------- #
# Directional oriented-line filter (reproduces the original kernel bank)
# --------------------------------------------------------------------------- #
def oriented_line_kernel(length_px: int, width_px: float, angle_deg: float,
                         gaussian: bool = True) -> np.ndarray:
    """An oriented elongated bar kernel: a length×length patch that is non-zero
    over a rotated rectangle (``width`` across, ``length`` along ``angle``),
    weighted by a radial Gaussian (σ = length/6) when ``gaussian``. Sum-normalised.
    """
    L = max(int(length_px), 3)
    c = (L - 1) / 2.0
    yy, xx = np.mgrid[0:L, 0:L].astype(float)
    dx, dy = xx - c, yy - c
    th = np.deg2rad(angle_deg)
    along = dx * np.cos(th) + dy * np.sin(th)
    perp = -dx * np.sin(th) + dy * np.cos(th)
    mask = (np.abs(along) <= L / 2.0) & (np.abs(perp) <= width_px / 2.0)
    if gaussian:
        sigma = L / 6.0
        base = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))
    else:
        base = np.ones((L, L))
    k = np.where(mask, base, 0.0)
    s = k.sum()
    return k / s if s > 0 else k


def directional_filter(image, length_px: int, width_px: float, n_angles: int,
                       gaussian: bool = True) -> np.ndarray:
    """Convolve with a bank of ``n_angles`` oriented line kernels (0–180°) and
    keep the per-pixel maximum response — enhances filaments of any orientation."""
    from scipy.signal import fftconvolve

    image = np.asarray(image, dtype=float)
    if image.size == 0:
        return image
    n = max(int(n_angles), 1)
    out = np.full_like(image, -np.inf)
    for i in range(n):
        kernel = oriented_line_kernel(length_px, width_px, 180.0 * i / n, gaussian)
        out = np.maximum(out, fftconvolve(image, kernel, mode="same"))
    return out


# --------------------------------------------------------------------------- #
# Threshold + skeletonize + trace
# --------------------------------------------------------------------------- #
def otsu_threshold(values) -> float:
    """Otsu's threshold over the (finite) image values."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    counts, edges = np.histogram(v, bins=256)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    if total == 0:
        return float(v.min())
    w = np.cumsum(counts)
    mu = np.cumsum(counts * centers)
    mu_t = mu[-1]
    denom = w * (total - w)
    denom[denom == 0] = 1
    sigma_b2 = (mu_t * w - mu * total) ** 2 / (denom * total ** 2)
    return float(centers[int(np.argmax(sigma_b2))])


def zhang_suen_skeleton(binary, max_iters: int = _MAX_THIN_ITERS) -> np.ndarray:
    """Zhang–Suen morphological thinning → 1-pixel-wide skeleton (bool array)."""
    img = (np.asarray(binary) > 0).astype(np.uint8)
    if img.ndim != 2 or img.sum() == 0:
        return img.astype(bool)

    def _neighbours(padded):
        # P2..P9 clockwise from north (Zhang–Suen ordering)
        return [padded[0:-2, 1:-1], padded[0:-2, 2:], padded[1:-1, 2:],
                padded[2:, 2:], padded[2:, 1:-1], padded[2:, 0:-2],
                padded[1:-1, 0:-2], padded[0:-2, 0:-2]]

    for _ in range(int(max_iters)):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2, P3, P4, P5, P6, P7, P8, P9 = _neighbours(p)
            nb = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            transitions = sum(((a == 0) & (b == 1)).astype(np.uint8)
                              for a, b in zip(seq[:-1], seq[1:]))
            if step == 0:
                c1 = P2 * P4 * P6
                c2 = P4 * P6 * P8
            else:
                c1 = P2 * P4 * P8
                c2 = P2 * P6 * P8
            cond = (img == 1) & (nb >= 2) & (nb <= 6) & (transitions == 1) & (c1 == 0) & (c2 == 0)
            if cond.any():
                img[cond] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    p = np.pad(skel.astype(np.uint8), 1)
    total = np.zeros(skel.shape, dtype=np.uint8)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            total += p[1 + di:1 + di + skel.shape[0], 1 + dj:1 + dj + skel.shape[1]]
    return total * skel.astype(np.uint8)


def skeleton_to_paths(skel, min_length_px: int = 0) -> list[np.ndarray]:
    """Split a skeleton at junctions and trace each branch into an ordered path.

    Returns a list of ``(M, 2)`` arrays of ``(row, col)`` pixel coordinates; only
    paths with at least ``min_length_px`` pixels are kept.
    """
    from scipy.ndimage import label

    skel = np.asarray(skel) > 0
    if skel.sum() == 0:
        return []
    nbc = _neighbour_count(skel)
    branch = skel & (nbc >= 3)            # junction pixels
    simple = skel & ~branch               # remove junctions → simple arcs
    lbl, n = label(simple, structure=np.ones((3, 3)))
    paths: list[np.ndarray] = []
    for k in range(1, n + 1):
        rc = np.argwhere(lbl == k)
        if rc.shape[0] < max(int(min_length_px), 1):
            continue
        paths.append(_order_path(rc))
    return paths


def _order_path(rc: np.ndarray) -> np.ndarray:
    """Order the pixels of one simple arc by walking 8-neighbours from an end."""
    pts = [tuple(p) for p in rc]
    idx = {p: i for i, p in enumerate(pts)}
    adj: dict = {p: [] for p in pts}
    for p in pts:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                q = (p[0] + di, p[1] + dj)
                if q in idx:
                    adj[p].append(q)
    if not pts:
        return rc
    ends = [p for p in pts if len(adj[p]) <= 1]
    start = ends[0] if ends else pts[0]
    order = [start]
    visited = {start}
    cur = start
    while True:
        nxt = next((q for q in adj[cur] if q not in visited), None)
        if nxt is None:
            break
        order.append(nxt)
        visited.add(nxt)
        cur = nxt
    return np.array(order, dtype=float)


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def segment_curvilinear(xy_nm, *, pixel_size_nm: float = DEFAULT_PIXEL_NM,
                        scale_min_nm: float = DEFAULT_SCALE_MIN_NM,
                        scale_max_nm: float = DEFAULT_SCALE_MAX_NM,
                        n_scales: int = DEFAULT_N_SCALES,
                        method: str = "frangi", beta: float = DEFAULT_BETA,
                        threshold: float = DEFAULT_THRESHOLD, use_otsu: bool = False,
                        min_length_nm: float = DEFAULT_MIN_LENGTH_NM,
                        sqrt_stabilize: bool = False,
                        directional: dict | None = None) -> dict:
    """Detect curvilinear structures and trace their centre lines.

    Returns a dict with ``ridge_map`` (0–1), ``skeleton`` (bool), ``binary``,
    ``xedge``/``yedge`` and ``paths`` — a list of ``(M, 2)`` nm vertex arrays.
    """
    px = float(pixel_size_nm)
    xy = np.asarray(xy_nm, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        xy = xy.reshape(-1, 2) if xy.size else np.empty((0, 2))
    xy = xy[np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])]

    out = {"ridge_map": np.zeros((0, 0)), "skeleton": np.zeros((0, 0), dtype=bool),
           "binary": np.zeros((0, 0), dtype=bool), "image": np.zeros((0, 0)),
           "xedge": np.zeros(0), "yedge": np.zeros(0), "paths": [],
           "pixel_size_nm": px, "method": method}
    if xy.shape[0] < 3:
        return out

    H, xedge, yedge = render_histogram_2d(xy[:, 0], xy[:, 1], px)
    image = stabilize_histogram(H, sqrt_stabilize)
    if directional:
        image = directional_filter(
            image, int(directional.get("length_nm", 60.0) / px),
            max(directional.get("width_nm", 12.0) / px, 1.0),
            int(directional.get("n_angles", 18)),
            bool(directional.get("gaussian", True)))

    sigmas = scales_from_nm(scale_min_nm, scale_max_nm, n_scales, px)
    ridge = ridge_filter(image, sigmas, method=method, beta=beta)
    thr = otsu_threshold(ridge[ridge > 0]) if use_otsu else float(threshold)
    binary = ridge >= thr if thr > 0 else ridge > 0
    skel = zhang_suen_skeleton(binary)
    paths_rc = skeleton_to_paths(skel, int(max(min_length_nm / px, 1.0)))

    xcenter = 0.5 * (xedge[:-1] + xedge[1:])
    ycenter = 0.5 * (yedge[:-1] + yedge[1:])
    paths = [np.column_stack([xcenter[p[:, 0].astype(int)], ycenter[p[:, 1].astype(int)]])
             for p in paths_rc]
    out.update(ridge_map=ridge, skeleton=skel, binary=binary, image=H,
               xedge=xedge, yedge=yedge, paths=paths)
    return out


# --------------------------------------------------------------------------- #
# Width measurement — perpendicular intensity profile + FWHM
# --------------------------------------------------------------------------- #
def sample_image(image, xedge, yedge, xy_nm, pixel_size_nm: float) -> np.ndarray:
    """Bilinearly sample ``image`` (indexed ``[x_bin, y_bin]``) at nm positions.

    Positions outside the image read 0. ``xedge``/``yedge`` are the bin edges that
    produced ``image`` (e.g. from :func:`render_histogram_2d`)."""
    from scipy.ndimage import map_coordinates

    image = np.asarray(image, dtype=float)
    xy = np.asarray(xy_nm, dtype=float).reshape(-1, 2)
    if image.size == 0 or xy.shape[0] == 0:
        return np.zeros(xy.shape[0])
    px = float(pixel_size_nm)
    rows = (xy[:, 0] - float(xedge[0])) / px - 0.5      # axis 0 = x bins
    cols = (xy[:, 1] - float(yedge[0])) / px - 0.5      # axis 1 = y bins
    return map_coordinates(image, [rows, cols], order=1, mode="constant", cval=0.0)


def path_perpendiculars(path, window: int = 4) -> np.ndarray:
    """Unit vectors perpendicular to a poly-line's **local** tangent at each vertex.

    The tangent is a windowed central difference (``±window`` vertices) so it is
    stable against single-pixel skeleton jitter; the perpendicular is the tangent
    rotated 90°."""
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    n = path.shape[0]
    if n < 2:
        return np.zeros((n, 2))
    idx = np.arange(n)
    lo = np.clip(idx - window, 0, n - 1)
    hi = np.clip(idx + window, 0, n - 1)
    tang = path[hi] - path[lo]
    norm = np.hypot(tang[:, 0], tang[:, 1])
    norm[norm == 0] = 1.0
    tang /= norm[:, None]
    return np.column_stack([-tang[:, 1], tang[:, 0]])


def perpendicular_profile(image, xedge, yedge, paths, *, max_dist_nm: float,
                          step_nm: float, pixel_size_nm: float, window: int = 4):
    """Average image intensity vs perpendicular distance to each centre line.

    For every path vertex, the image is sampled along the local perpendicular at
    offsets ``[-max_dist, +max_dist]`` (step ``step_nm``) and averaged over all
    vertices. Returns ``(offsets_nm, combined_profile, per_path_profiles,
    vertex_counts)`` — the combined profile is the vertex-count-weighted mean.
    """
    step = max(float(step_nm), 1e-6)
    m = int(round(float(max_dist_nm) / step))
    offsets = np.arange(-m, m + 1) * step
    total_sum = np.zeros(offsets.size)
    total_n = 0
    per_path: list[np.ndarray] = []
    counts: list[int] = []
    for path in paths:
        path = np.asarray(path, dtype=float).reshape(-1, 2)
        nvert = path.shape[0]
        if nvert < 2:
            per_path.append(np.zeros(offsets.size))
            counts.append(nvert)
            continue
        perp = path_perpendiculars(path, window)
        # positions: (n_offsets, n_vertices, 2)
        pos = path[None, :, :] + offsets[:, None, None] * perp[None, :, :]
        vals = sample_image(image, xedge, yedge, pos.reshape(-1, 2),
                            pixel_size_nm).reshape(offsets.size, nvert)
        path_sum = vals.sum(axis=1)
        per_path.append(path_sum / nvert)
        total_sum += path_sum
        total_n += nvert
        counts.append(nvert)
    combined = total_sum / total_n if total_n else np.zeros(offsets.size)
    return offsets, combined, per_path, np.asarray(counts, dtype=int)


def fwhm_bounds(offsets, profile):
    """The two half-maximum crossing offsets around the profile peak, linearly
    interpolated. Returns ``(left, right, half_max)``; left/right are NaN if a
    side has no crossing."""
    offsets = np.asarray(offsets, dtype=float)
    profile = np.asarray(profile, dtype=float)
    n = profile.size
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    k = int(np.argmax(profile))
    pmax = float(profile[k])
    if pmax <= 0:
        return float("nan"), float("nan"), float("nan")
    half = pmax / 2.0
    left = float("nan")
    for i in range(k, 0, -1):
        if profile[i] <= half:                       # crossing between i and i+1
            denom = profile[i + 1] - profile[i]
            frac = 0.0 if denom == 0 else (half - profile[i]) / denom
            left = offsets[i] + frac * (offsets[i + 1] - offsets[i])
            break
    right = float("nan")
    for i in range(k, n - 1):
        if profile[i] <= half:                       # crossing between i-1 and i
            denom = profile[i - 1] - profile[i]
            frac = 0.0 if denom == 0 else (profile[i - 1] - half) / denom
            right = offsets[i - 1] + frac * (offsets[i] - offsets[i - 1])
            break
    return left, right, half


def fwhm(offsets, profile) -> float:
    """Full width at half maximum of a profile, with linear interpolation of the
    two half-max crossings around the peak. NaN if either side has no crossing."""
    left, right, _ = fwhm_bounds(offsets, profile)
    if not (np.isfinite(left) and np.isfinite(right)):
        return float("nan")
    return float(right - left)


def measure_filament_width(image, xedge, yedge, paths, *, max_dist_nm: float,
                           step_nm: float, pixel_size_nm: float) -> dict:
    """Perpendicular intensity profile + FWHM for a set of centre lines.

    Returns a dict with ``offsets`` (nm), combined ``profile`` and its ``fwhm``,
    plus ``per_path_profiles`` / ``per_path_fwhm`` and ``vertex_counts``.
    """
    offsets, combined, per_path, counts = perpendicular_profile(
        image, xedge, yedge, paths, max_dist_nm=max_dist_nm, step_nm=step_nm,
        pixel_size_nm=pixel_size_nm)
    per_fwhm = np.array([fwhm(offsets, p) for p in per_path])
    return {"offsets": offsets, "profile": combined, "fwhm": fwhm(offsets, combined),
            "per_path_profiles": per_path, "per_path_fwhm": per_fwhm,
            "vertex_counts": counts}
