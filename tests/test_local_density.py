import numpy as np

from minflux_viewer.analysis.local_density import (
    local_density_histogram_2d,
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
