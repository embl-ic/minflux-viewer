import numpy as np
import pytest

from minflux_viewer.analysis.local_density import (
    _density_values,
    local_density_histogram_2d,
    local_density_histogram_nd,
    local_density_kdtree,
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


def test_histogram_3d_supported_via_density_values():
    """3D histogram density no longer raises NotImplementedError."""
    rng = np.random.default_rng(1)
    points = rng.uniform(0.0, 500.0, size=(200, 3))
    values, label, _detail, _valid = _density_values(
        points, np.ones(len(points), bool),
        dimensions=3, radius_nm=50.0, method="histogram_2d",
        voxel_size_nm=50.0, smooth_sigma=0.0,
    )
    assert label == "3D histogram"
    assert np.isfinite(values).all() and values.min() >= 1.0


def test_histogram_density_voxel_grid_cap():
    points = np.array([[0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]])
    with pytest.raises(ValueError, match="voxel"):
        local_density_histogram_nd(points, dimensions=3, voxel_size_nm=1e-3)
