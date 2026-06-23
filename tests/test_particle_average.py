"""Particle averaging (analysis/particle_average.py).

Pure NumPy/SciPy — no Qt. Validates centering, the rigid transform conventions,
image cross-correlation alignment, and both averaging modes on synthetic
asymmetric particles.
"""

import numpy as np
import pytest

from minflux_viewer.analysis import particle_average as pa


def _L_shape():
    """An asymmetric 'L' point set (breaks rotational symmetry)."""
    return np.concatenate([
        np.column_stack([np.linspace(-40, 40, 60), np.zeros(60)]),
        np.column_stack([np.full(30, -40.0), np.linspace(0, 40, 30)]),
    ])


def _jitter(p, n=8, sigma=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.repeat(p, n, axis=0) + rng.normal(0, sigma, (p.shape[0] * n, 2))


def _sharpness(img):
    return float(img.max() / img.mean()) if img.mean() > 0 else 0.0


# --- helpers -------------------------------------------------------------------
def test_center_particle_zeroes_centroids():
    pts = np.array([[10.0, 20.0, 5.0], [12.0, 22.0, 7.0]])
    c = pa.center_particle(pts)
    assert np.allclose(c[:, :3].mean(axis=0), [0, 0, 0], atol=1e-9)


def test_transform_points_rotation_and_translation():
    pts = np.array([[1.0, 0.0]])
    r = pa.transform_points(pts, 90.0)
    assert np.allclose(r[0], [0.0, 1.0], atol=1e-9)
    t = pa.transform_points(pts, 0.0, 5.0, -3.0)
    assert np.allclose(t[0], [6.0, -3.0])


def test_transform_preserves_extra_columns():
    pts = np.array([[1.0, 0.0, 42.0]])
    r = pa.transform_points(pts, 90.0)
    assert r[0, 2] == 42.0           # Z untouched


def test_render_points_shape_and_peak():
    pts = np.zeros((100, 2))         # all at origin
    img = pa.render_points(pts, box_nm=60.0, pixel_nm=3.0)
    assert img.shape == (20, 20)
    assert np.unravel_index(int(np.argmax(img)), img.shape) == (10, 10)


# --- alignment -----------------------------------------------------------------
def test_xcorr_recovers_translation():
    P = _jitter(_L_shape())
    P -= P.mean(axis=0)
    templ = pa.render_points(P, 200, 3, sigma_nm=3)
    shifted = P + np.array([15.0, 0.0])
    _th, dx, dy, _s = pa.align_particle(shifted, templ, box_nm=200, pixel_nm=3,
                                        angles=[0.0], sigma_nm=3)
    assert abs(dx - (-15.0)) < 4.0   # undoes the +15 nm shift
    assert abs(dy) < 4.0


def test_align_recovers_rotation():
    P = _jitter(_L_shape())
    P -= P.mean(axis=0)
    templ = pa.render_points(P, 200, 3, sigma_nm=4)
    rotated = pa.transform_points(P, 50.0)
    angles = np.linspace(0, 360, 72, endpoint=False)
    theta, _dx, _dy, _s = pa.align_particle(rotated, templ, box_nm=200, pixel_nm=3,
                                            angles=angles, sigma_nm=4)
    # rotating the particle by -theta should bring it back ⇒ theta ≈ -50 (mod 360)
    err = (theta - (-50.0)) % 360.0
    assert min(err, 360 - err) < 12.0


# --- averaging -----------------------------------------------------------------
def _random_particles(k=12, seed=1, with_z=True):
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(k):
        p = pa.transform_points(_jitter(_L_shape(), seed=i),
                                rng.uniform(0, 360), *rng.uniform(-10, 10, 2))
        if with_z:
            p = np.column_stack([p, rng.normal(0, 5, p.shape[0])])
        parts.append(p)
    return parts


def test_template_free_sharper_than_unaligned():
    parts = _random_particles()
    res = pa.average_template_free(parts, box_nm=200, pixel_nm=3, n_angles=72,
                                   n_iter=6, sigma_nm=4)
    unaligned = pa.render_points(
        np.vstack([pa.center_particle(p)[:, :2] for p in parts]), 200, 3, sigma_nm=4)
    assert res["n_particles"] == len(parts)
    assert _sharpness(res["average_image"]) > 1.5 * _sharpness(unaligned)
    assert res["points"].shape[1] == 3            # Z carried through
    assert res["scores"].min() > 0.8


def test_template_free_converges():
    parts = _random_particles(k=10, seed=3)
    res = pa.average_template_free(parts, box_nm=200, pixel_nm=3, n_angles=48, n_iter=5)
    h = res["score_history"]
    assert len(h) == 5
    assert h[-1] >= h[0] - 1e-6                    # non-decreasing overall


def test_template_provided_aligns_to_ring():
    rng = np.random.default_rng(7)

    def _ring_particle(seed):
        r = np.random.default_rng(seed)
        th = r.uniform(0, 2 * np.pi, 200)
        rad = 50.0 + r.normal(0, 3, 200)
        xy = np.column_stack([rad * np.cos(th), rad * np.sin(th)])
        return pa.transform_points(xy, r.uniform(0, 360), *r.uniform(-8, 8, 2))

    parts = [_ring_particle(i) for i in range(8)]
    template = pa.geometry_template_image("ring", {"diameter_nm": 100.0, "rim_nm": 12.0},
                                          box_nm=160, pixel_nm=3, sigma_nm=3)
    res = pa.average_particles(parts, mode="template", template_img=template,
                               box_nm=160, pixel_nm=3, n_angles=36, sigma_nm=4)
    assert res["mode"] == "template"
    assert res["points"].shape[0] == sum(p.shape[0] for p in parts)
    # the pooled ring should match the ring template well
    pooled_img = res["average_image"]
    corr = np.corrcoef(pooled_img.ravel(), template.ravel())[0, 1]
    assert corr > 0.5


def test_empty_particles_safe():
    res = pa.average_template_free([], box_nm=150, pixel_nm=3)
    assert res["points"].shape == (0, 3)
    assert res["n_particles"] == 0


def test_geometry_template_image_shapes():
    for kind, params in (("ring", {"diameter_nm": 100, "rim_nm": 15}),
                         ("disk", {"diameter_nm": 80}), ("gaussian", {"fwhm_nm": 40})):
        img = pa.geometry_template_image(kind, params, box_nm=150, pixel_nm=3)
        assert img.shape == (50, 50)
        assert np.isfinite(img).all() and img.max() > 0
