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

import numpy as np

DEFAULT_BOX_NM = 150.0
DEFAULT_PIXEL_NM = 3.0
DEFAULT_N_ANGLES = 36
DEFAULT_N_ITERS = 5


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
    """A geometry template rendered to a box-sized image (ring / disk / gaussian)."""
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
def _xcorr_shift(source: np.ndarray, template: np.ndarray):
    """Pixel shift to apply to *source* (its points translated by +shift) so it
    best matches *template*, plus a normalised correlation score."""
    shape = np.array(source.shape)
    f = np.fft.rfft2(template)
    g = np.fft.rfft2(source)
    cc = np.fft.irfft2(f * np.conj(g), s=source.shape)
    peak = np.array(np.unravel_index(int(np.argmax(cc)), cc.shape), dtype=float)
    for i in range(peak.size):                       # wrap to [-N/2, N/2)
        if peak[i] > shape[i] / 2:
            peak[i] -= shape[i]
    denom = float(np.linalg.norm(source) * np.linalg.norm(template)) or 1.0
    return peak, float(cc.max() / denom)


def align_particle(points_xy, template_img, *, box_nm: float, pixel_nm: float,
                   angles, sigma_nm: float | None = None):
    """Best 2-D rigid transform aligning a particle's XY to ``template_img``.

    Rotation is searched over ``angles`` (degrees) in point space; translation
    comes from the image cross-correlation at each angle. Returns
    ``(theta_deg, dx_nm, dy_nm, score)``."""
    pts = np.asarray(points_xy, dtype=float)
    best = (0.0, 0.0, 0.0, -np.inf)
    for a in angles:
        rp = transform_points(pts, a)
        img = render_points(rp, box_nm, pixel_nm, sigma_nm)
        shift, score = _xcorr_shift(img, template_img)
        if score > best[3]:
            best = (float(a), float(shift[0] * pixel_nm), float(shift[1] * pixel_nm), score)
    return best


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
                              sigma_nm: float | None = None, progress=None) -> dict:
    """Align every particle to a fixed template image and pool them.

    ``progress(done, total)`` (optional) is called once per aligned particle."""
    angles = np.linspace(0.0, 360.0, int(n_angles), endpoint=False)
    sigma_nm = pixel_nm if sigma_nm is None else sigma_nm
    particles_c = [center_particle(np.asarray(p, float)) for p in particles]
    transforms = []
    total = len(particles_c)
    for i, p in enumerate(particles_c, start=1):
        if progress is not None:
            progress(i, total)
        transforms.append(align_particle(p[:, :2], template_img, box_nm=box_nm,
                                         pixel_nm=pixel_nm, angles=angles, sigma_nm=sigma_nm))
    pooled = _pool(particles_c, transforms)
    avg_img = render_points(pooled[:, :2], box_nm, pixel_nm, sigma_nm) if pooled.size else \
        np.zeros_like(template_img)
    scores = np.array([t[3] for t in transforms])
    return {"points": pooled, "average_image": avg_img, "template": template_img,
            "transforms": transforms, "scores": scores, "n_particles": len(particles_c),
            "box_nm": box_nm, "pixel_nm": pixel_nm, "mode": "template"}


def average_template_free(particles, *, box_nm=DEFAULT_BOX_NM, pixel_nm=DEFAULT_PIXEL_NM,
                          n_angles=DEFAULT_N_ANGLES, n_iter=DEFAULT_N_ITERS,
                          sigma_nm: float | None = None, seed: int = 0, progress=None) -> dict:
    """Reference-free 2-D averaging: seed the reference from one particle, then
    iterate *align-all → mean → repeat*.

    ``progress(done, total)`` (optional) is called per particle-alignment across
    all iterations (``total = n_iter × n_particles``)."""
    angles = np.linspace(0.0, 360.0, int(n_angles), endpoint=False)
    sigma_nm = pixel_nm if sigma_nm is None else sigma_nm
    particles_c = [center_particle(np.asarray(p, float)) for p in particles]
    if not particles_c:
        return {"points": np.empty((0, 3)), "average_image": np.zeros((8, 8)),
                "template": np.zeros((8, 8)), "transforms": [], "scores": np.empty(0),
                "n_particles": 0, "iterations": 0, "box_nm": box_nm,
                "pixel_nm": pixel_nm, "mode": "free"}
    # seed reference from the particle with the most localizations (most defined)
    seed_idx = int(np.argmax([p.shape[0] for p in particles_c]))
    template = render_points(particles_c[seed_idx][:, :2], box_nm, pixel_nm, sigma_nm)
    transforms = [(0.0, 0.0, 0.0, 0.0)] * len(particles_c)
    history = []
    n_it = max(int(n_iter), 1)
    total = n_it * len(particles_c)
    step = 0
    for _ in range(n_it):
        transforms = []
        for p in particles_c:
            step += 1
            if progress is not None:
                progress(step, total)
            transforms.append(align_particle(p[:, :2], template, box_nm=box_nm,
                                             pixel_nm=pixel_nm, angles=angles, sigma_nm=sigma_nm))
        imgs = [_aligned_image(p, t, box_nm, pixel_nm, sigma_nm)
                for p, t in zip(particles_c, transforms)]
        new_template = np.mean(imgs, axis=0)
        history.append(float(np.mean([t[3] for t in transforms])))
        template = new_template
    pooled = _pool(particles_c, transforms)
    avg_img = render_points(pooled[:, :2], box_nm, pixel_nm, sigma_nm)
    return {"points": pooled, "average_image": avg_img, "template": template,
            "transforms": transforms, "scores": np.array([t[3] for t in transforms]),
            "n_particles": len(particles_c), "iterations": len(history),
            "score_history": history, "box_nm": box_nm, "pixel_nm": pixel_nm,
            "mode": "free"}


def average_particles(particles, *, mode="free", template_img=None, **kwargs) -> dict:
    """Dispatch to template-provided or template-free averaging."""
    if mode == "template":
        if template_img is None:
            raise ValueError("template mode requires template_img")
        return average_template_provided(particles, template_img, **kwargs)
    return average_template_free(particles, **kwargs)
