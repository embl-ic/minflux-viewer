import numpy as np
import pytest

import minflux_viewer.analysis.local_density as local_density_module
from minflux_viewer.analysis.local_density import (
    _density_values,
    local_density_histogram_2d,
    local_density_histogram_nd,
    local_density_kdtree,
    local_density_voxel_radius_count,
)


def test_local_density_kdtree_counts_neighbours_within_radius():
    points = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0],
        [20.0, 0.0, 0.0],
    ])

    density = local_density_kdtree(points, radius_nm=5.0)

    np.testing.assert_array_equal(density, [1.0, 1.0, 0.0])


def test_local_density_kdtree_parallel_matches_serial():
    """workers=-1 + larger leafsize must give the exact serial neighbour count."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 1000.0, size=(4000, 3))
    serial = np.asarray(
        cKDTree(points).query_ball_point(points, r=80.0, return_length=True),
        dtype=float,
    ) - 1.0
    np.testing.assert_array_equal(local_density_kdtree(points, radius_nm=80.0), serial)


def test_local_density_histogram_2d_returns_one_value_per_point():
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [50.0, 50.0, 0.0],
    ])

    density = local_density_histogram_2d(points, voxel_size_nm=10.0, smooth_sigma=0.0)

    assert density.shape == (3,)
    assert density[0] == density[1] == 2.0
    assert density[2] == 1.0


def test_local_density_histogram_3d_counts_per_voxel():
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],     # same voxel as point 0
        [50.0, 50.0, 50.0],  # far away → its own voxel
    ])

    density = local_density_histogram_nd(
        points, dimensions=3, voxel_size_nm=10.0, smooth_sigma=0.0,
    )

    assert density.shape == (3,)
    assert density[0] == density[1] == 2.0
    assert density[2] == 1.0


def test_voxel_histogram_3d_supported_via_density_values():
    """3D histogram density no longer raises NotImplementedError."""
    rng = np.random.default_rng(1)
    points = rng.uniform(0.0, 500.0, size=(200, 3))
    values, label, _detail, _valid = _density_values(
        points, np.ones(len(points), bool),
        dimensions=3, radius_nm=50.0, method="voxel_histogram",
        voxel_size_nm=50.0, smooth_sigma=0.0,
    )
    assert label == "3D voxel histogram"
    assert np.isfinite(values).all() and values.min() >= 1.0


def test_histogram_density_voxel_grid_cap():
    points = np.array([[0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]])
    with pytest.raises(ValueError, match="voxel"):
        local_density_histogram_nd(points, dimensions=3, voxel_size_nm=1e-3)


def test_voxel_radius_count_matches_kdtree_on_grid_points_2d():
    points = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
    ])

    density = local_density_voxel_radius_count(
        points, dimensions=2, radius_nm=10.0, voxel_size_nm=10.0,
    )

    np.testing.assert_array_equal(density, local_density_kdtree(points[:, :2], 10.0))


def test_voxel_radius_count_uses_ball_in_3d():
    points = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [10.0, 10.0, 10.0],
    ])

    density = local_density_voxel_radius_count(
        points, dimensions=3, radius_nm=10.0, voxel_size_nm=10.0,
    )

    np.testing.assert_array_equal(density, [3.0, 1.0, 1.0, 1.0, 0.0])


def test_voxel_radius_count_supported_via_density_values():
    points = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ])
    values, label, detail, valid = _density_values(
        points, np.ones(len(points), bool),
        dimensions=2, radius_nm=10.0, method="voxel_radius",
        voxel_size_nm=10.0, smooth_sigma=0.0,
    )

    assert label == "2D voxel disk radius count"
    assert "radius=10" in detail
    np.testing.assert_array_equal(valid, [True, True])
    np.testing.assert_array_equal(values, [1.0, 1.0])


def test_auto_load_density_falls_back_to_voxel_radius_when_kdtree_work_is_too_large(
    monkeypatch,
):
    points = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [10.0, 10.0, 0.0],
    ])
    monkeypatch.setattr(local_density_module, "_AUTO_KDTREE_MAX_TOTAL_WORK", 1.0)

    values, label, detail, valid = _density_values(
        points,
        np.ones(len(points), dtype=bool),
        dimensions=2,
        radius_nm=20.0,
        method="auto_load",
        voxel_size_nm=10.0,
        smooth_sigma=0.0,
    )

    assert label == "2D voxel disk radius count"
    assert "auto fallback from KD-tree" in detail
    np.testing.assert_array_equal(valid, [True, True, True, True])
    assert np.isfinite(values).all()
