"""Tests for the multi-channel overlay volume compositing (volume_window)."""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.analysis.voronoi_density import build_voronoi_3d_field
from minflux_viewer.ui.volume_window import (
    _clip_region,
    _compose_multichannel_rgba,
    _precision_volume_3d,
    lut_rgb,
    make_multichannel_volume_payload,
    make_volume_payload,
)


def _grid_edges(lo, hi, n):
    return [np.linspace(lo[i], hi[i], n[i] + 1) for i in range(3)]


def test_volume_render_methods_differ_and_conserve_mass():
    rng = np.random.default_rng(1)
    n = 8000
    locs = np.column_stack(
        [rng.normal(0, 150, n), rng.normal(0, 150, n), rng.normal(0, 50, n)]
    )
    sigma = np.column_stack(
        [rng.uniform(2, 5, n), rng.uniform(2, 5, n), rng.uniform(4, 9, n)]
    )
    common = dict(xy_voxel_nm=3.0, z_voxel_nm=3.0, max_dim=512, max_voxels=4_000_000)
    peaks = {}
    for method in ("histogram", "bilinear", "bicubic", "basic", "fixed_gaussian"):
        p = make_volume_payload(
            locs, render_method=method, sigma_nm_xyz=(5.0, 5.0, 5.0), **common
        )
        assert np.isclose(p.scalar.sum(), n, rtol=1e-4)   # mass-conserving
        peaks[method] = float(p.scalar.max())
    prec = make_volume_payload(
        locs, render_method="precision_gaussian", precision_sigma_nm=sigma, **common
    )
    assert np.isclose(prec.scalar.sum(), n, rtol=1e-4)
    # Histogram is crisp (highest peak); the smoothing methods spread it out.
    assert peaks["histogram"] > peaks["bilinear"]
    assert peaks["histogram"] > peaks["basic"]


def test_precision_volume_uses_per_localization_sigma():
    # Two identical points, one with tiny sigma (stays sharp) and one large
    # (spreads): the large-sigma point must occupy more voxels.
    edges = _grid_edges([-50, -50, -50], [50, 50, 50], [50, 50, 50])
    sharp = _precision_volume_3d(
        np.array([[0.0, 0.0, 0.0]]), np.array([[0.5, 0.5, 0.5]]), edges, (2.0, 2.0, 2.0)
    )
    broad = _precision_volume_3d(
        np.array([[0.0, 0.0, 0.0]]), np.array([[8.0, 8.0, 8.0]]), edges, (2.0, 2.0, 2.0)
    )
    assert np.count_nonzero(broad) > np.count_nonzero(sharp)


def test_volume_black_white_percentiles_control_contrast():
    rng = np.random.default_rng(3)
    n = 6000
    locs = np.column_stack(
        [rng.normal(0, 120, n), rng.normal(0, 120, n), rng.normal(0, 40, n)]
    )
    common = dict(xy_voxel_nm=4.0, z_voxel_nm=4.0, max_dim=512, max_voxels=2_000_000,
                  render_method="histogram")
    lo_white = make_volume_payload(locs, white_pct=50.0, **common)
    hi_white = make_volume_payload(locs, white_pct=99.7, **common)
    # A lower white percentile saturates sooner → brighter (higher mean norm)
    # and a lower reported intensity_max (the white point).
    assert lo_white.norm.mean() > hi_white.norm.mean()
    assert lo_white.intensity_max < hi_white.intensity_max
    # Raising the black point suppresses low voxels → fewer lit voxels.
    black = make_volume_payload(locs, black_pct=80.0, white_pct=99.7, **common)
    assert np.count_nonzero(black.norm) < np.count_nonzero(hi_white.norm)


def test_clip_region_keeps_aligned_sigma():
    locs = np.array([[0.0, 0, 0], [100, 0, 0], [0, 0, 100]])
    sigma = np.array([[1.0, 1, 1], [2, 2, 2], [3, 3, 3]])
    lc, sc = _clip_region(locs, sigma, (-10, 10, -10, 10, -10, 10))
    assert lc.shape[0] == 1 and np.allclose(lc[0], [0, 0, 0])
    assert np.allclose(sc[0], [1, 1, 1])
    lc2, sc2 = _clip_region(locs, None, (-10, 10, -10, 10, -10, 10))
    assert sc2 is None and lc2.shape[0] == 1


def test_lut_rgb_pure_colors():
    r = lut_rgb("Red")
    g = lut_rgb("Green")
    assert r[0] > 0.5 and r[1] < 0.3 and r[2] < 0.3
    assert g[1] > 0.5 and g[0] < 0.3 and g[2] < 0.3
    # Unknown name -> a finite fallback colour.
    assert all(0.0 <= c <= 1.0 for c in lut_rgb("definitely-not-a-cmap"))


def test_multichannel_composite_has_per_channel_colors():
    rng = np.random.default_rng(0)
    # Red cluster near x=0, green cluster near x=200 (no spatial overlap); both
    # span z so the grid has a usable 3-D Z range.
    red = rng.normal([0.0, 0.0, 0.0], [5.0, 5.0, 25.0], size=(3000, 3))
    green = rng.normal([200.0, 0.0, 0.0], [5.0, 5.0, 25.0], size=(3000, 3))
    payload = make_multichannel_volume_payload(
        [red, green], [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        xy_voxel_nm=5.0, z_voxel_nm=5.0, max_dim=128, opacity=1.0)

    rgba = payload.rgba.astype(float)
    R, G, A = rgba[..., 0], rgba[..., 1], rgba[..., 3]
    vis = A > 0
    assert payload.n_locs == 6000
    assert np.any((R > G + 20) & vis), "no red-dominant voxels"
    assert np.any((G > R + 20) & vis), "no green-dominant voxels"


def test_multichannel_overlap_is_yellow():
    rng = np.random.default_rng(1)
    pts = rng.normal([0.0, 0.0, 0.0], [5.0, 5.0, 25.0], size=(3000, 3))
    payload = make_multichannel_volume_payload(
        [pts, pts.copy()], [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        xy_voxel_nm=5.0, z_voxel_nm=5.0, max_dim=128, opacity=1.0)
    rgba = payload.rgba.astype(float)
    vis = rgba[..., 3] > 0
    R, G, B = rgba[..., 0], rgba[..., 1], rgba[..., 2]
    # Where both channels overlap, red + green add to yellow (R & G high, B low).
    assert np.any((R > 100) & (G > 100) & (B < 80) & vis), "no yellow overlap voxels"


def test_multichannel_caches_compact_norms_for_display_sync():
    rng = np.random.default_rng(11)
    red = rng.normal([0.0, 0.0, 0.0], [5.0, 5.0, 25.0], size=(1500, 3))
    green = rng.normal([160.0, 0.0, 0.0], [5.0, 5.0, 25.0], size=(1500, 3))
    payload = make_multichannel_volume_payload(
        [red, green], [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        xy_voxel_nm=5.0, z_voxel_nm=5.0, max_dim=96, opacity=1.0,
        channel_contrast_pct=[(0.0, 10.0), (0.0, 99.7)],
    )

    assert payload.channel_norms is not None
    assert len(payload.channel_norms) == 2
    assert all(norm.dtype == np.uint8 for norm in payload.channel_norms)
    assert all(norm.shape == payload.counts for norm in payload.channel_norms)
    # Recomposition from the cached responses changes channel contribution
    # without rebuilding the voxel grid.
    red_only = _compose_multichannel_rgba(
        [payload.channel_norms[0]], [(1.0, 0.0, 0.0)], 1.0
    )
    assert np.any(red_only[..., 0] > 0)
    assert np.all(red_only[..., 1] == 0)


def test_multichannel_requires_z_range():
    flat = np.random.default_rng(2).normal([0, 0, 0], [5, 5, 0.0], size=(500, 3))
    flat[:, 2] = 0.0                      # no Z extent
    with pytest.raises(ValueError):
        make_multichannel_volume_payload(
            [flat], [(1.0, 0.0, 0.0)], xy_voxel_nm=5.0, z_voxel_nm=5.0, max_dim=64)


def test_multichannel_empty_raises():
    with pytest.raises(ValueError):
        make_multichannel_volume_payload(
            [np.empty((0, 3))], [(1.0, 0.0, 0.0)],
            xy_voxel_nm=5.0, z_voxel_nm=5.0, max_dim=64)


def test_multichannel_voronoi_density_uses_one_field_per_channel():
    rng = np.random.default_rng(76)
    red = rng.normal([0.0, 0.0, 0.0], [5.0, 5.0, 15.0], size=(40, 3))
    green = rng.normal([80.0, 0.0, 0.0], [5.0, 5.0, 15.0], size=(40, 3))
    fields = [build_voronoi_3d_field(red), build_voronoi_3d_field(green)]
    payload = make_multichannel_volume_payload(
        [red, green], [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        xy_voxel_nm=8.0,
        z_voxel_nm=8.0,
        max_dim=32,
        max_voxels=32 ** 3,
        render_method="voronoi_density",
        voronoi_fields=fields,
        opacity=1.0,
    )
    assert payload.channel_norms is not None
    assert len(payload.channel_norms) == 2
    assert np.any(payload.rgba[..., 0] > payload.rgba[..., 1])
    assert np.any(payload.rgba[..., 1] > payload.rgba[..., 0])
