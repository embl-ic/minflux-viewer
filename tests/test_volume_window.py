from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.ui.volume_window import make_volume_payload


def test_make_volume_payload_caps_shape_and_returns_rgba() -> None:
    rng = np.random.default_rng(42)
    locs = np.column_stack([
        rng.normal(0.0, 40.0, 500),
        rng.normal(0.0, 25.0, 500),
        rng.normal(0.0, 15.0, 500),
    ])

    payload = make_volume_payload(
        locs,
        xy_voxel_nm=1.0,
        z_voxel_nm=2.0,
        max_dim=64,
        max_voxels=64 * 64 * 64,
        cmap_name="hot",
        opacity=0.5,
    )

    assert payload.rgba.dtype == np.uint8
    assert payload.scalar.dtype == np.float32
    assert payload.norm.dtype == np.float32
    assert payload.rgba.shape == (*payload.counts, 4)
    assert payload.scalar.shape == payload.counts
    assert payload.norm.shape == payload.counts
    assert max(payload.counts) <= 64
    assert np.prod(payload.counts) <= 64 * 64 * 64
    assert payload.n_locs == 500
    assert payload.rgba[..., 3].max() > 0
    assert payload.voxel_nm[2] >= payload.voxel_nm[0]


def test_make_volume_payload_rejects_flat_z() -> None:
    locs = np.column_stack([
        np.linspace(0.0, 10.0, 20),
        np.linspace(0.0, 20.0, 20),
        np.zeros(20),
    ])

    with pytest.raises(ValueError, match="3-D Z range"):
        make_volume_payload(locs, xy_voxel_nm=1.0, z_voxel_nm=1.0, max_dim=64)


def test_make_volume_payload_allows_high_max_dim_but_caps_total_voxels() -> None:
    rng = np.random.default_rng(123)
    locs = np.column_stack([
        rng.uniform(0.0, 10_000.0, 1_000),
        rng.uniform(0.0, 8_000.0, 1_000),
        rng.uniform(0.0, 6_000.0, 1_000),
    ])

    payload = make_volume_payload(
        locs,
        xy_voxel_nm=1.0,
        z_voxel_nm=1.0,
        max_dim=1024,
        max_voxels=8_000_000,
    )

    assert max(payload.counts) <= 1024
    assert np.prod(payload.counts) <= 8_000_000
