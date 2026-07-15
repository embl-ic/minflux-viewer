"""
minflux_viewer.analysis.particle_average
=========================================
**Particle averaging** of detected objects (LocMoFit-style structure average).

Given a set of *particles* — small localization point clouds cropped from box
ROIs — this aligns them into a common frame and pools them into one averaged
super-particle. Two modes:

* **template-provided** — align every particle (2-D rigid: rotation + shift in
  XY) to a supplied geometry template image;
* **template-free** — seed a reference from one particle, then iterate
  *align-all → rebuild reference as the mean → repeat*, the classic reference-free
  2-D alignment.

Alignment is done on the **XY projection** (image cross-correlation: brute-force
rotation search + FFT translation). **Z is aligned translationally only** — each
particle's Z is centred on its own Z centroid — so the pooled cloud is 3-D-ready
while the fit stays 2-D.

Conventions
-----------
Images are indexed ``[x_bin, y_bin]`` (as :func:`render_points` produces them).
Rotation is applied in **point space** (exact, no image-rotation ambiguity); only
the translation comes from the image cross-correlation.

Pure NumPy/SciPy — Qt-free and unit-testable. Distances are in nm.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .geometry import rotation_a_to_b

DEFAULT_BOX_NM = 150.0
DEFAULT_PIXEL_NM = 3.0
DEFAULT_N_ANGLES = 36
DEFAULT_N_ITERS = 5


def _resolve_workers(n_workers, n_tasks: int) -> int:
    """Number of parallel alignment workers: ``None`` → all logical CPUs, capped to
    the task count (no point spawning more threads than particles). The per-particle
    alignment is FFT/histogram-bound and numpy/scipy release the GIL, so threads
    parallelise well without pickling the (potentially large) particle clouds."""
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    return max(1, min(int(n_workers), max(1, int(n_tasks))))


# --------------------------------------------------------------------------- #
# Axial tilt correction (per-particle PCA) — makes the image fitters 3-D-aware
# --------------------------------------------------------------------------- #


def estimate_tilt_normal(points):
    """Best-fit plane normal (PCA smallest-variance axis) + tilt vs +Z, degrees.

    Returns ``(+Z, 0.0)`` for planar / 2-D data (no out-of-plane spread) so the
    correction is a safe no-op there."""
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[0] < 3 or P.shape[1] < 3:
        return np.array([0.0, 0.0, 1.0]), 0.0
    C = P[:, :3] - P[:, :3].mean(axis=0)
    if not np.any(np.abs(C[:, 2]) > 1e-9):          # flat 2-D data → no tilt
        return np.array([0.0, 0.0, 1.0]), 0.0
    _w, V = np.linalg.eigh(C.T @ C)
    normal = V[:, 0]                                 # smallest eigenvalue
    if normal[2] < 0:
        normal = -normal
    tilt = float(np.degrees(np.arccos(np.clip(abs(float(normal[2])), 0.0, 1.0))))
    return normal, tilt


def flatten_tilt(points):
    """Rotate a particle so its best-fit-plane normal points along +Z (un-tilt).

    Returns ``(rotated_points, tilt_deg)``. The XYZ block is rotated about the
    particle centroid; any extra columns pass through. No-op for 2-D / planar data."""
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[1] < 3 or P.shape[0] < 3:
        return P, 0.0
    normal, tilt = estimate_tilt_normal(P)
    if tilt < 1e-9:
        return P, 0.0
    R = rotation_a_to_b(normal, [0.0, 0.0, 1.0])
    ctr = P[:, :3].mean(axis=0)
    out = P.copy()
    out[:, :3] = (R @ (P[:, :3] - ctr).T).T
    return out, tilt


def _particle_prep(points, correct_tilt: bool):
    """Center (+ optionally PCA-untilt) one particle **and** return the 4×4 matrix
    that maps its original (re-zeroed) XYZ into the prepared frame.

    The matrix lets the *same* preparation be replayed on another channel's
    localizations (multi-channel averaging)."""
    p = np.asarray(points, float)
    M = np.eye(4)
    work = p.copy()
    tilt = 0.0
    if correct_tilt and work.ndim == 2 and work.shape[1] >= 3 and work.shape[0] >= 3:
        normal, tilt = estimate_tilt_normal(work)
        if tilt >= 1e-9:
            R = rotation_a_to_b(normal, [0.0, 0.0, 1.0])
            C = work[:, :3].mean(axis=0)
            Mt = np.eye(4)
            Mt[:3, :3] = R
            Mt[:3, 3] = -R @ C                          # rotate about the centroid
            M = Mt @ M
            work[:, :3] = (R @ (work[:, :3] - C).T).T
    c = np.zeros(3)
    if work.shape[0]:
        c[0], c[1] = work[:, 0].mean(), work[:, 1].mean()
        if work.shape[1] > 2:
            c[2] = work[:, 2].mean()
    Mc = np.eye(4)
    Mc[:3, 3] = -c
    M = Mc @ M
    work[:, 0] -= c[0]
    work[:, 1] -= c[1]
    if work.shape[1] > 2:
        work[:, 2] -= c[2]
    return work, tilt, M


def _matrix_2d(theta_deg: float, dx_nm: float, dy_nm: float) -> np.ndarray:
    """4×4 matrix for a 2-D rigid transform (XY rotation about origin + shift; Z
    passes through) — the ``transform_points`` operation as a matrix."""
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    M = np.eye(4)
    M[0, 0], M[0, 1] = c, -s
    M[1, 0], M[1, 1] = s, c
    M[0, 3], M[1, 3] = float(dx_nm), float(dy_nm)
    return M


def apply_particle_transform(points, matrix) -> np.ndarray:
    """Apply a 4×4 particle transform to ``(M, ≥2)`` points → ``(M, 3)`` nm."""
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[0] == 0:
        return np.empty((0, 3))
    xyz = P[:, :3] if P.shape[1] >= 3 else np.column_stack([P[:, :2], np.zeros(P.shape[0])])
    h = np.column_stack([xyz, np.ones(xyz.shape[0])])
    return (np.asarray(matrix, float) @ h.T).T[:, :3]


def pool_transformed(particle_points, matrices) -> np.ndarray:
    """Pool a channel by applying each particle's 4×4 transform to its points."""
    pooled = []
    for pts, M in zip(particle_points, matrices):
        if pts is None or len(pts) == 0 or M is None:
            continue
        pooled.append(apply_particle_transform(pts, M))
    return np.vstack(pooled) if pooled else np.empty((0, 3))


def collapse_traces(points, tid) -> np.ndarray:
    """Collapse each MINFLUX trace (``tid``) to its centroid — **one point per
    molecule-blink** instead of one per localization.

    In MINFLUX the localizations sharing a ``tid`` are the iterative localizations
    of a *single emitter during one on-event*; they are highly correlated, so
    fitting/aligning on raw localizations over-counts long/bright molecules and
    biases the result toward them. Collapsing to trace centroids gives each
    molecule-blink one equal vote (denoised + far fewer points), which de-biases
    the average and speeds the fit. Falls back to the raw points when ``tid`` is
    missing, mis-sized or degenerate (≤1 group). Preserves the coordinate frame
    and dimensionality of *points*.
    """
    P = np.asarray(points, dtype=float)
    if tid is None or P.ndim != 2 or P.shape[0] == 0:
        return P
    t = np.asarray(tid).ravel()
    if t.size != P.shape[0]:
        return P
    uid, inv = np.unique(t, return_inverse=True)
    if uid.size <= 1 or uid.size == P.shape[0]:
        return P                                    # nothing to gain
    sums = np.zeros((uid.size, P.shape[1]), dtype=float)
    np.add.at(sums, inv, P)
    counts = np.bincount(inv, minlength=uid.size).astype(float)
    return sums / counts[:, None]


def _prepare_particles(particles, correct_tilt: bool):
    """Center (+ optionally PCA-untilt) each particle. Returns
    ``(centered, tilts, prep_matrices)``; ``prep_matrices[i]`` maps the original
    re-zeroed particle into the prepared frame (for multi-channel replay)."""
    prepared, tilts, mats = [], [], []
    for p in particles:
        work, t, M = _particle_prep(p, correct_tilt)
        prepared.append(work)
        mats.append(M)
        tilts.append(t)                 # per-particle (0.0 when not correcting)
    return prepared, tilts, mats


def image_particle_table(transforms, particles_c, tilts) -> list[dict]:
    """Per-particle alignment metrics for the image fitters (one row/particle).

    Columns mirror the NPC two-ring table's role: the fitted 2-D rigid transform
    (``angle``/``dx``/``dy``), per-particle PCA ``tilt`` (0 when not corrected),
    ``n_locs``, and the cross-correlation ``score`` (higher = better match to the
    reference/template). Image fitters pool every particle, so ``accepted`` is
    always True (kept for a uniform table across methods)."""
    rows = []
    for i, t in enumerate(transforms):
        theta, dx, dy, score = t
        n = int(np.asarray(particles_c[i]).shape[0]) if i < len(particles_c) else 0
        rows.append({"particle": i + 1, "n_locs": n, "angle": float(theta),
                     "dx": float(dx), "dy": float(dy),
                     "tilt": float(tilts[i]) if i < len(tilts) else 0.0,
                     "score": float(score), "accepted": True})
    return rows


def geometry_outline(kind: str, params: dict, n: int = 96):
    """Outline geometry for a template *kind* → ``(marker_points|None, [polyline])``.

    For overlaying the fitted template on a particle in the inspector: a ring
    circle polyline (``ring``/``disk``/``gaussian``) or a ring + N corner markers
    (``npc``), all at z = 0, in nm."""
    phi = np.linspace(0.0, 2 * np.pi, n)
    if kind == "npc":
        r = float(params.get("diameter_nm", 100.0)) / 2.0
        fold = max(int(round(float(params.get("symmetry", 8)))), 1)
        ang = 2 * np.pi * np.arange(fold) / fold
        corners = np.column_stack([r * np.cos(ang), r * np.sin(ang), np.zeros(fold)])
        circle = np.column_stack([r * np.cos(phi), r * np.sin(phi), np.zeros(n)])
        return corners, [circle]
    r = (float(params.get("fwhm_nm", 40.0)) if kind == "gaussian"
         else float(params.get("diameter_nm", 100.0))) / 2.0
    circle = np.column_stack([r * np.cos(phi), r * np.sin(phi), np.zeros(n)])
    return None, [circle]


# --------------------------------------------------------------------------- #
# Particle helpers
# --------------------------------------------------------------------------- #
def center_particle(points) -> np.ndarray:
    """Centre a particle: XY on its centroid, Z (col 2, if present) on its own
    centroid — the translational Z alignment."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return pts.reshape(-1, pts.shape[1] if pts.ndim == 2 else 3)
    out = pts.copy()
    out[:, 0] -= pts[:, 0].mean()
    out[:, 1] -= pts[:, 1].mean()
    if pts.shape[1] > 2:
        out[:, 2] -= pts[:, 2].mean()
    return out


def transform_points(points, theta_deg: float, dx_nm: float = 0.0, dy_nm: float = 0.0):
    """Rotate XY by ``theta_deg`` (about the origin) then translate. Other columns
    (e.g. Z) pass through unchanged."""
    pts = np.asarray(points, dtype=float).copy()
    if pts.shape[0] == 0:
        return pts
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    pts[:, 0] = c * x - s * y + dx_nm
    pts[:, 1] = s * x + c * y + dy_nm
    return pts


def render_points(points_xy, box_nm: float, pixel_nm: float,
                  sigma_nm: float | None = None) -> np.ndarray:
    """Render centred XY points into a square ``box_nm`` image at ``pixel_nm``
    (Gaussian-blurred histogram). Returns an ``[x_bin, y_bin]`` array."""
    n = max(int(round(box_nm / max(pixel_nm, 1e-6))), 8)
    half = box_nm / 2.0
    edges = np.linspace(-half, half, n + 1)
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros((n, n))
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[edges, edges])
    if sigma_nm:
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, max(sigma_nm / pixel_nm, 0.5))
    return H


def geometry_template_image(kind: str, params: dict, box_nm: float,
                            pixel_nm: float, sigma_nm: float | None = None) -> np.ndarray:
    """A geometry template rendered to a box-sized image.

    ``kind`` ∈ {``ring``, ``disk``, ``gaussian``, ``npc``}. ``npc`` is a **ring of
    N Gaussian corner blobs** (``symmetry``-fold, default 8) — it breaks the plain
    ring's continuous rotational symmetry down to Cn, so template alignment can
    **phase-lock** each particle to the nearest corner and the average shows the N
    discrete NPC sub-units instead of a smeared ring."""
    n = max(int(round(box_nm / max(pixel_nm, 1e-6))), 8)
    half = box_nm / 2.0
    c = np.linspace(-half + pixel_nm / 2, half - pixel_nm / 2, n)
    xx, yy = np.meshgrid(c, c, indexing="ij")
    r = np.hypot(xx, yy)
    if kind == "ring":
        r0 = float(params.get("diameter_nm", 100.0)) / 2.0
        rim = max(float(params.get("rim_nm", 20.0)), pixel_nm)
        img = np.exp(-((r - r0) ** 2) / (2.0 * rim ** 2))
    elif kind == "disk":
        r0 = float(params.get("diameter_nm", 80.0)) / 2.0
        edge = max(float(params.get("edge_nm", pixel_nm)), pixel_nm)
        img = 0.5 * (1.0 - np.tanh((r - r0) / edge))
    elif kind == "npc":                                  # ring of N corner blobs
        r0 = float(params.get("diameter_nm", 100.0)) / 2.0
        fold = max(int(round(float(params.get("symmetry", 8)))), 1)
        corner = max(float(params.get("corner_nm", 12.0)), pixel_nm)
        img = np.zeros_like(r)
        for k in range(fold):
            ang = 2.0 * np.pi * k / fold
            cx0, cy0 = r0 * np.cos(ang), r0 * np.sin(ang)
            img = img + np.exp(-(((xx - cx0) ** 2 + (yy - cy0) ** 2)) / (2.0 * corner ** 2))
    else:  # gaussian
        fwhm = max(float(params.get("fwhm_nm", 40.0)), pixel_nm)
        sigma = fwhm / 2.3548200450309493
        img = np.exp(-(r ** 2) / (2.0 * sigma ** 2))
    if sigma_nm:
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, max(sigma_nm / pixel_nm, 0.5))
    return img


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def _xcorr_shift_pre(source: np.ndarray, template_f: np.ndarray,
                     template_norm: float, shape):
    """Like :func:`_xcorr_shift` but with the template's ``rfft2`` and Frobenius
    norm **precomputed** — the template is constant across every particle and angle
    within one alignment pass, so its FFT/norm are computed once (not per angle)."""
    g = np.fft.rfft2(source)
    cc = np.fft.irfft2(template_f * np.conj(g), s=shape)
    peak = np.array(np.unravel_index(int(np.argmax(cc)), cc.shape), dtype=float)
    for i in range(peak.size):                       # wrap to [-N/2, N/2)
        if peak[i] > shape[i] / 2:
            peak[i] -= shape[i]
    denom = float(np.linalg.norm(source) * template_norm) or 1.0
    return peak, float(cc.max() / denom)


def _xcorr_shift(source: np.ndarray, template: np.ndarray):
    """Pixel shift to apply to *source* (its points translated by +shift) so it
    best matches *template*, plus a normalised correlation score."""
    return _xcorr_shift_pre(source, np.fft.rfft2(template),
                            float(np.linalg.norm(template)), source.shape)


def _align_one(pts, template_f, template_norm, shape, box_nm, pixel_nm, angles, sigma_nm):
    """Core rotation search for one particle against a precomputed template FFT.
    Returns ``(theta_deg, dx_nm, dy_nm, score)``.

    **Batched over all angles**: instead of a Python loop that renders + FFTs one
    rotation at a time (tens of thousands of tiny numpy calls with dominant Python
    overhead), all ``A`` rotations are rendered in a single ``histogramdd`` + one
    ``gaussian_filter`` + one batched ``rfft2``/``irfft2`` over the angle-stack. Same
    result as the per-angle search (points are still rotated in point space), an
    order of magnitude less overhead. Falls back to the per-angle loop only for the
    degenerate empty particle."""
    pts = np.asarray(pts, dtype=float)
    angles = np.asarray(angles, dtype=float)
    A = int(angles.size)
    n = int(shape[0])
    if pts.shape[0] == 0 or A == 0:
        return (float(angles[0]) if A else 0.0, 0.0, 0.0, 0.0)
    half = box_nm / 2.0
    edges = np.linspace(-half, half, n + 1)
    th = np.deg2rad(angles)
    cos, sin = np.cos(th)[:, None], np.sin(th)[:, None]        # (A, 1)
    x, y = pts[:, 0][None, :], pts[:, 1][None, :]              # (1, M)
    xr = cos * x - sin * y                                     # (A, M) rotated in point space
    yr = sin * x + cos * y
    # One (A, n, n) histogram stack (angle axis + the two spatial axes).
    a_idx = np.repeat(np.arange(A), pts.shape[0])
    H, _ = np.histogramdd(
        np.column_stack([a_idx, xr.ravel(), yr.ravel()]),
        bins=[np.arange(A + 1) - 0.5, edges, edges])          # (A, n, n)
    if sigma_nm:
        from scipy.ndimage import gaussian_filter
        s = max(sigma_nm / pixel_nm, 0.5)
        H = gaussian_filter(H, sigma=(0.0, s, s))             # blur only the spatial axes
    g = np.fft.rfft2(H, axes=(-2, -1))                        # batched source FFTs
    cc = np.fft.irfft2(template_f[None] * np.conj(g), s=shape, axes=(-2, -1))  # (A, n, n)
    flat = cc.reshape(A, -1)
    peak_lin = np.argmax(flat, axis=1)                        # first max per angle (C-order)
    peak_val = flat[np.arange(A), peak_lin]
    src_norm = np.sqrt(np.einsum("aij,aij->a", H, H))         # ‖source‖ per angle
    denom = src_norm * template_norm
    denom[denom == 0.0] = 1.0
    scores = peak_val / denom
    best_a = int(np.argmax(scores))                          # first angle achieving the max
    py, px = np.unravel_index(int(peak_lin[best_a]), shape)
    py = py - n if py > n / 2 else py                        # wrap shift to [-n/2, n/2)
    px = px - n if px > n / 2 else px
    return (float(angles[best_a]), float(py * pixel_nm), float(px * pixel_nm),
            float(scores[best_a]))


def align_particle(points_xy, template_img, *, box_nm: float, pixel_nm: float,
                   angles, sigma_nm: float | None = None):
    """Best 2-D rigid transform aligning a particle's XY to ``template_img``.

    Rotation is searched over ``angles`` (degrees) in point space; translation
    comes from the image cross-correlation at each angle. Returns
    ``(theta_deg, dx_nm, dy_nm, score)``."""
    template_f = np.fft.rfft2(template_img)
    template_norm = float(np.linalg.norm(template_img))
    return _align_one(np.asarray(points_xy, dtype=float), template_f, template_norm,
                      template_img.shape, box_nm, pixel_nm, angles, sigma_nm)


def _align_all(particles_c, template, *, box_nm, pixel_nm, angles, sigma_nm,
               executor, on_done=None):
    """Align every particle's XY to ``template`` (rotation search + FFT shift),
    optionally in parallel via *executor*. The template ``rfft2`` + norm are
    computed **once** here and shared across all particles. ``on_done()`` (if given)
    is called once per finished particle, in the calling thread (so it is safe for
    progress reporting). Returns a list of transforms in particle order."""
    template_f = np.fft.rfft2(template)
    template_norm = float(np.linalg.norm(template))
    shape = template.shape
    transforms: list = [None] * len(particles_c)
    if executor is None:
        for i, p in enumerate(particles_c):
            transforms[i] = _align_one(p[:, :2], template_f, template_norm, shape,
                                       box_nm, pixel_nm, angles, sigma_nm)
            if on_done is not None:
                on_done()
    else:
        futures = {
            executor.submit(_align_one, p[:, :2], template_f, template_norm, shape,
                            box_nm, pixel_nm, angles, sigma_nm): i
            for i, p in enumerate(particles_c)}
        for fut in as_completed(futures):
            transforms[futures[fut]] = fut.result()
            if on_done is not None:
                on_done()
    return transforms


def _aligned_image(particle_c, transform, box_nm, pixel_nm, sigma_nm):
    theta, dx, dy, _score = transform
    ap = transform_points(particle_c[:, :2], theta, dx, dy)
    return render_points(ap, box_nm, pixel_nm, sigma_nm)


# --------------------------------------------------------------------------- #
# Averaging
# --------------------------------------------------------------------------- #
def _pool(particles_c, transforms) -> np.ndarray:
    pooled = []
    for p, (theta, dx, dy, _s) in zip(particles_c, transforms):
        ap = p.copy()
        ap[:, :2] = transform_points(p[:, :2], theta, dx, dy)
        pooled.append(ap)
    return np.vstack(pooled) if pooled else np.empty((0, 3))


def average_template_provided(particles, template_img, *, box_nm=DEFAULT_BOX_NM,
                              pixel_nm=DEFAULT_PIXEL_NM, n_angles=DEFAULT_N_ANGLES,
                              sigma_nm: float | None = None, correct_tilt: bool = False,
                              progress=None, n_workers=None) -> dict:
    """Align every particle to a fixed template image and pool them.

    ``correct_tilt`` first PCA-untilts each particle (3-D data); ``progress(done,
    total)`` (optional) is called once per aligned particle. ``n_workers`` (``None``
    → all CPUs) parallelises the per-particle alignment across cores."""
    angles = np.linspace(0.0, 360.0, int(n_angles), endpoint=False)
    sigma_nm = pixel_nm if sigma_nm is None else sigma_nm
    particles_c, tilts, prep_mats = _prepare_particles(particles, correct_tilt)
    total = len(particles_c)
    done = [0]

    def _tick():
        done[0] += 1
        if progress is not None:
            progress(done[0], total)

    workers = _resolve_workers(n_workers, total)
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        transforms = _align_all(particles_c, template_img, box_nm=box_nm,
                                pixel_nm=pixel_nm, angles=angles, sigma_nm=sigma_nm,
                                executor=executor, on_done=_tick)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    pooled = _pool(particles_c, transforms)
    avg_img = render_points(pooled[:, :2], box_nm, pixel_nm, sigma_nm) if pooled.size else \
        np.zeros_like(template_img)
    scores = np.array([t[3] for t in transforms])
    mats4 = [_matrix_2d(t[0], t[1], t[2]) @ M for t, M in zip(transforms, prep_mats)]
    return {"points": pooled, "average_image": avg_img, "template": template_img,
            "transforms": transforms, "particle_transforms": mats4,
            "scores": scores, "n_particles": len(particles_c),
            "table": image_particle_table(transforms, particles_c, tilts),
            "box_nm": box_nm, "pixel_nm": pixel_nm, "mode": "template",
            "mean_tilt_deg": float(np.mean(tilts)) if tilts else 0.0}


def average_template_free(particles, *, box_nm=DEFAULT_BOX_NM, pixel_nm=DEFAULT_PIXEL_NM,
                          n_angles=DEFAULT_N_ANGLES, n_iter=DEFAULT_N_ITERS,
                          sigma_nm: float | None = None, seed: int = 0,
                          correct_tilt: bool = False, progress=None, n_workers=None) -> dict:
    """Reference-free 2-D averaging: seed the reference from one particle, then
    iterate *align-all → mean → repeat*.

    ``correct_tilt`` first PCA-untilts each particle (3-D data); ``progress(done,
    total)`` (optional) is called per particle-alignment across all iterations
    (``total = n_iter × n_particles``). ``n_workers`` (``None`` → all CPUs)
    parallelises the per-particle alignment within each iteration across cores;
    iterations stay sequential (each needs the previous mean)."""
    angles = np.linspace(0.0, 360.0, int(n_angles), endpoint=False)
    sigma_nm = pixel_nm if sigma_nm is None else sigma_nm
    particles_c, tilts, prep_mats = _prepare_particles(particles, correct_tilt)
    if not particles_c:
        return {"points": np.empty((0, 3)), "average_image": np.zeros((8, 8)),
                "template": np.zeros((8, 8)), "transforms": [], "particle_transforms": [],
                "scores": np.empty(0), "table": [], "n_particles": 0, "iterations": 0,
                "box_nm": box_nm, "pixel_nm": pixel_nm, "mode": "free"}
    # seed reference from the particle with the most localizations (most defined)
    seed_idx = int(np.argmax([p.shape[0] for p in particles_c]))
    template = render_points(particles_c[seed_idx][:, :2], box_nm, pixel_nm, sigma_nm)
    transforms = [(0.0, 0.0, 0.0, 0.0)] * len(particles_c)
    history = []
    n_it = max(int(n_iter), 1)
    total = n_it * len(particles_c)
    done = [0]

    def _tick():
        done[0] += 1
        if progress is not None:
            progress(done[0], total)

    workers = _resolve_workers(n_workers, len(particles_c))
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for _ in range(n_it):
            transforms = _align_all(particles_c, template, box_nm=box_nm,
                                    pixel_nm=pixel_nm, angles=angles, sigma_nm=sigma_nm,
                                    executor=executor, on_done=_tick)
            imgs = [_aligned_image(p, t, box_nm, pixel_nm, sigma_nm)
                    for p, t in zip(particles_c, transforms)]
            template = np.mean(imgs, axis=0)
            history.append(float(np.mean([t[3] for t in transforms])))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    pooled = _pool(particles_c, transforms)
    avg_img = render_points(pooled[:, :2], box_nm, pixel_nm, sigma_nm)
    mats4 = [_matrix_2d(t[0], t[1], t[2]) @ M for t, M in zip(transforms, prep_mats)]
    return {"points": pooled, "average_image": avg_img, "template": template,
            "transforms": transforms, "particle_transforms": mats4,
            "scores": np.array([t[3] for t in transforms]),
            "table": image_particle_table(transforms, particles_c, tilts),
            "n_particles": len(particles_c), "iterations": len(history),
            "score_history": history, "box_nm": box_nm, "pixel_nm": pixel_nm,
            "mode": "free", "mean_tilt_deg": float(np.mean(tilts)) if tilts else 0.0}


def average_particles(particles, *, mode="free", template_img=None, **kwargs) -> dict:
    """Dispatch to template-provided or template-free averaging."""
    if mode == "template":
        if template_img is None:
            raise ValueError("template mode requires template_img")
        return average_template_provided(particles, template_img, **kwargs)
    return average_template_free(particles, **kwargs)
