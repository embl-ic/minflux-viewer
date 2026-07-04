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


def test_average_reports_progress():
    parts = _random_particles(k=6, seed=5)
    free_calls = []
    pa.average_particles(parts, mode="free", box_nm=200, pixel_nm=4, n_angles=24,
                         n_iter=3, progress=lambda d, t: free_calls.append((d, t)))
    assert free_calls[-1] == (3 * 6, 3 * 6)        # n_iter × n_particles
    tmpl = pa.geometry_template_image("ring", {"diameter_nm": 100, "rim_nm": 12},
                                      box_nm=160, pixel_nm=4)
    tmpl_calls = []
    pa.average_particles(parts, mode="template", template_img=tmpl, box_nm=160,
                         pixel_nm=4, n_angles=24,
                         progress=lambda d, t: tmpl_calls.append((d, t)))
    assert tmpl_calls[-1] == (6, 6)                # one per particle


def test_empty_particles_safe():
    res = pa.average_template_free([], box_nm=150, pixel_nm=3)
    assert res["points"].shape == (0, 3)
    assert res["n_particles"] == 0


# --- axial tilt (3-D) ----------------------------------------------------------
def _tilt_points(P, deg, axis="x"):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    R = (np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) if axis == "x"
         else np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))
    return (R @ np.asarray(P, float).T).T


def test_estimate_tilt_normal_planar_is_zero():
    rng = np.random.default_rng(0)
    P = np.column_stack([rng.normal(0, 40, 300), rng.normal(0, 40, 300), np.zeros(300)])
    normal, tilt = pa.estimate_tilt_normal(P)
    assert tilt < 1.0 and abs(abs(normal[2]) - 1.0) < 1e-6


def test_flatten_tilt_recovers_tilt_angle():
    rng = np.random.default_rng(1)
    disk = np.column_stack([rng.normal(0, 40, 500), rng.normal(0, 40, 500), rng.normal(0, 2, 500)])
    tilted = _tilt_points(disk, 20.0, axis="x")
    flat, tilt = pa.flatten_tilt(tilted)
    assert abs(tilt - 20.0) < 3.0
    assert np.std(flat[:, 2]) < 0.5 * np.std(tilted[:, 2])     # flattened in Z


def test_template_free_correct_tilt_flattens_3d():
    parts = []
    for i in range(8):
        r = np.random.default_rng(i)
        th = r.uniform(0, 2 * np.pi, 200)
        ring = np.column_stack([50 * np.cos(th), 50 * np.sin(th), r.normal(0, 2, 200)])
        ring = pa.transform_points(ring, r.uniform(0, 360))        # in-plane rotation
        ring = _tilt_points(ring, r.uniform(10, 25), axis="x")     # random out-of-plane tilt
        parts.append(ring)
    off = pa.average_template_free(parts, box_nm=160, pixel_nm=3, n_angles=36, n_iter=3,
                                   correct_tilt=False)
    on = pa.average_template_free(parts, box_nm=160, pixel_nm=3, n_angles=36, n_iter=3,
                                  correct_tilt=True)
    assert on["mean_tilt_deg"] > 5.0
    assert np.std(on["points"][:, 2]) < np.std(off["points"][:, 2])   # tilt removed


def test_particle_transforms_reproduce_pool_and_map_channels():
    """Multi-channel enabler: the exported per-particle 4×4 transforms reproduce the
    reference pool exactly and, applied to a second channel, preserve its offset."""
    parts = []
    for i in range(6):
        r = np.random.default_rng(i)
        a = np.arange(8) * 2 * np.pi / 8
        ring = np.column_stack([50 * np.cos(a), 50 * np.sin(a), np.zeros(8)])
        ring = pa.transform_points(ring, r.uniform(0, 360))
        ring = _tilt_points(ring, r.uniform(5, 20), axis="x")
        parts.append(ring + r.uniform(-20, 20, 3))
    # tilt-corrected: the exported transforms reproduce the pool exactly (they
    # encode the per-particle untilt + centring + 2-D alignment)
    res = pa.average_template_free(parts, box_nm=200, pixel_nm=4, n_angles=36, n_iter=3,
                                   correct_tilt=True)
    mats = res["particle_transforms"]
    assert len(mats) == len(parts)
    repro = pa.pool_transformed(parts, mats)
    np.testing.assert_allclose(np.sort(repro, axis=0), np.sort(res["points"], axis=0), atol=1e-6)

    # no-tilt replay: a second channel offset by pure +Z keeps that Z offset in the
    # averaged frame (the 2-D transform passes Z through) — the multi-channel promise
    res2 = pa.average_template_free(parts, box_nm=200, pixel_nm=4, n_angles=36, n_iter=3,
                                    correct_tilt=False)
    mats2 = res2["particle_transforms"]
    ch2 = [p + np.array([0.0, 0.0, 12.0]) for p in parts]
    pooled2 = pa.pool_transformed(ch2, mats2)
    assert pooled2.shape == res2["points"].shape
    assert abs(float(np.mean(pooled2[:, 2] - res2["points"][:, 2])) - 12.0) < 1e-6


def test_geometry_template_image_shapes():
    for kind, params in (("ring", {"diameter_nm": 100, "rim_nm": 15}),
                         ("disk", {"diameter_nm": 80}), ("gaussian", {"fwhm_nm": 40})):
        img = pa.geometry_template_image(kind, params, box_nm=150, pixel_nm=3)
        assert img.shape == (50, 50)
        assert np.isfinite(img).all() and img.max() > 0


# --- NPC 8-fold template model -------------------------------------------------
def _corner_ring(seed, radius=50.0, corners=8, per=25, csig=3.0, rot=None):
    """A ring of `corners` Gaussian blobs at a random rotation (2-D)."""
    r = np.random.default_rng(seed)
    rot = r.uniform(0, 2 * np.pi) if rot is None else rot
    pts = []
    for k in range(corners):
        a = rot + 2 * np.pi * k / corners
        cx, cy = radius * np.cos(a), radius * np.sin(a)
        pts.append(np.column_stack([r.normal(cx, csig, per), r.normal(cy, csig, per)]))
    return np.vstack(pts)


def _ring_profile_harmonic(img, box_nm=160.0, pixel_nm=2.0, radius=50.0, m=720):
    """Dominant angular Fourier harmonic of the template sampled on a ring."""
    half, n = box_nm / 2.0, img.shape[0]
    c = np.linspace(-half + pixel_nm / 2, half - pixel_nm / 2, n)
    ang = np.linspace(0, 2 * np.pi, m, endpoint=False)
    ix = np.abs(c[:, None] - radius * np.cos(ang)[None, :]).argmin(0)
    iy = np.abs(c[:, None] - radius * np.sin(ang)[None, :]).argmin(0)
    prof = img[ix, iy]
    return int(np.argmax(np.abs(np.fft.rfft(prof - prof.mean()))))


def _dominant_angular_harmonic(xy, bins=180):
    ang = np.arctan2(xy[:, 1], xy[:, 0])
    h, _ = np.histogram(ang, bins=bins, range=(-np.pi, np.pi))
    return int(np.argmax(np.abs(np.fft.rfft(h - h.mean()))[1:17]) + 1)


def test_geometry_template_image_npc_is_n_fold():
    for fold in (6, 8, 12):
        img = pa.geometry_template_image("npc", {"diameter_nm": 100.0, "corner_nm": 8.0,
                                                 "symmetry": fold}, box_nm=160, pixel_nm=2.0)
        assert img.shape == (80, 80) and img.max() > 0
        assert _ring_profile_harmonic(img) == fold        # sub-units break C∞ → Cn


def test_template_provided_npc_phase_locks_corners():
    parts = [_corner_ring(i) for i in range(12)]
    tmpl = pa.geometry_template_image("npc", {"diameter_nm": 100.0, "corner_nm": 10.0,
                                             "symmetry": 8}, box_nm=160, pixel_nm=2.0, sigma_nm=2.0)
    res = pa.average_particles(parts, mode="template", template_img=tmpl, box_nm=160,
                               pixel_nm=2.0, n_angles=72, sigma_nm=2.0)
    assert res["mode"] == "template"
    assert _dominant_angular_harmonic(res["points"]) == 8   # pooled corners cluster 8-fold


def test_image_particle_table_has_per_particle_rows():
    parts = _random_particles(k=5, seed=2)
    res = pa.average_template_free(parts, box_nm=180, pixel_nm=4, n_angles=24, n_iter=2)
    tbl = res["table"]
    assert len(tbl) == len(parts)
    assert {"particle", "n_locs", "angle", "dx", "dy", "tilt", "score", "accepted"} <= set(tbl[0])
    assert [r["particle"] for r in tbl] == list(range(1, len(parts) + 1))
    assert all(r["accepted"] for r in tbl)                 # image fitters pool all
    assert tbl[0]["n_locs"] == parts[0].shape[0]
    tmpl = pa.geometry_template_image("ring", {"diameter_nm": 100, "rim_nm": 12}, 160, 4)
    res2 = pa.average_particles(parts, mode="template", template_img=tmpl,
                                box_nm=160, pixel_nm=4, n_angles=24)
    assert len(res2["table"]) == len(parts)


def test_collapse_traces_centroids_and_fallback():
    # 3 traces (tids 0,1,2) with 4/2/1 locs → 3 centroids
    pts = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0],      # tid 0 centroid (1,1,0)
                    [10, 10, 5], [12, 10, 5],                         # tid 1 centroid (11,10,5)
                    [20, 20, 20]], float)                            # tid 2 (as-is)
    tid = np.array([0, 0, 0, 0, 1, 1, 2])
    c = pa.collapse_traces(pts, tid)
    assert c.shape == (3, 3)
    np.testing.assert_allclose(c[0], [1, 1, 0])
    np.testing.assert_allclose(c[1], [11, 10, 5])
    np.testing.assert_allclose(c[2], [20, 20, 20])
    # fallbacks: no tid, all-distinct, or degenerate → raw points unchanged
    np.testing.assert_array_equal(pa.collapse_traces(pts, None), pts)
    np.testing.assert_array_equal(pa.collapse_traces(pts, np.arange(7)), pts)
    np.testing.assert_array_equal(pa.collapse_traces(pts, np.zeros(7)), pts)


def test_geometry_outline_ring_and_npc():
    corners, curves = pa.geometry_outline("ring", {"diameter_nm": 100.0})
    assert corners is None and len(curves) == 1
    assert np.allclose(np.hypot(curves[0][:, 0], curves[0][:, 1]), 50.0, atol=1e-6)
    corners, curves = pa.geometry_outline("npc", {"diameter_nm": 80.0, "symmetry": 8})
    assert corners.shape == (8, 3) and len(curves) == 1
    assert np.allclose(np.hypot(corners[:, 0], corners[:, 1]), 40.0, atol=1e-6)  # radius=diam/2


def test_template_provided_npc_with_tilt_runs_3d():
    parts = []
    for i in range(8):
        p = _corner_ring(i)
        z = np.random.default_rng(i).normal(0, 2, p.shape[0])
        p = _tilt_points(np.column_stack([p, z]), np.random.default_rng(i + 99).uniform(10, 25))
        parts.append(p)
    tmpl = pa.geometry_template_image("npc", {"diameter_nm": 100.0, "corner_nm": 10.0,
                                             "symmetry": 8}, box_nm=160, pixel_nm=2.0, sigma_nm=2.0)
    res = pa.average_particles(parts, mode="template", template_img=tmpl, box_nm=160,
                               pixel_nm=2.0, n_angles=72, sigma_nm=2.0, correct_tilt=True)
    assert res["mean_tilt_deg"] > 5.0
    assert res["points"].shape[1] == 3
