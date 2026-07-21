"""Tests for the multi-channel overlay volume compositing (volume_window)."""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.ui.volume_window import (
    lut_rgb,
    make_multichannel_volume_payload,
)


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
