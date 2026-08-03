from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import ConvexHull, Voronoi

from minflux_viewer.analysis.voronoi_density import (
    ProjectedVoronoiError,
    Voronoi3DError,
    build_projected_voronoi_field,
    build_voronoi_3d_field,
)


def _regular_grid(spacing: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(5, dtype=np.float64) * spacing
    x, y = np.meshgrid(axis, axis)
    return x.ravel(), y.ravel()


def _density_at(field, x: float, y: float) -> float:
    matches = np.all(np.isclose(field.points_nm, [x, y]), axis=1)
    assert np.count_nonzero(matches) == 1
    return float(field.density_per_nm2[matches][0])


def test_regular_grid_has_expected_interior_inverse_area() -> None:
    x, y = _regular_grid()

    field = build_projected_voronoi_field(x, y)

    assert field.source_count == 25
    assert field.unique_count == 25
    assert field.finite_cell_count == 9
    assert np.isclose(_density_at(field, 20.0, 20.0), 1.0 / 100.0)
    assert _density_at(field, 0.0, 0.0) == 0.0
    sampled = field.sample((19.5, 20.5, 19.5, 20.5), (1, 1))
    assert np.isclose(sampled[0, 0], 1.0 / 100.0)


def test_exact_duplicate_increases_cell_density_without_coordinate_rounding() -> None:
    x, y = _regular_grid()
    x = np.concatenate((x + 0.125, [20.125]))
    y = np.concatenate((y + 0.375, [20.375]))

    field = build_projected_voronoi_field(x, y)

    assert field.source_count == 26
    assert field.unique_count == 25
    assert np.any(np.isclose(field.points_nm[:, 0] % 1.0, 0.125))
    assert np.isclose(_density_at(field, 20.125, 20.375), 2.0 / 100.0)


def test_adjacent_samples_match_one_continuous_raster() -> None:
    x, y = _regular_grid()
    field = build_projected_voronoi_field(x, y)

    whole = field.sample((0.0, 40.0, 0.0, 40.0), (80, 80))
    left = field.sample((0.0, 20.0, 0.0, 40.0), (80, 40))
    right = field.sample((20.0, 40.0, 0.0, 40.0), (80, 40))

    assert np.allclose(np.hstack((left, right)), whole, atol=1e-10)


def test_sampling_outside_convex_hull_returns_zero() -> None:
    x, y = _regular_grid()
    field = build_projected_voronoi_field(x, y)

    image = field.sample((100.0, 110.0, 100.0, 110.0), (8, 8))

    assert not np.any(image)


def test_vectorized_dual_areas_match_explicit_voronoi_polygons() -> None:
    rng = np.random.default_rng(148)
    x = rng.uniform(-100.0, 100.0, 200)
    y = rng.uniform(-80.0, 80.0, 200)

    field = build_projected_voronoi_field(x, y)
    reference = Voronoi(field.points_nm)
    expected = np.zeros(field.unique_count, dtype=np.float64)
    for point_index, region_index in enumerate(reference.point_region):
        region = reference.regions[region_index]
        if len(region) < 3 or -1 in region:
            continue
        polygon = reference.vertices[region]
        area = 0.5 * abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
        expected[point_index] = 1.0 / area

    assert np.allclose(field.density_per_nm2, expected, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        ([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], "at least four"),
        ([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], "not all collinear"),
    ],
)
def test_invalid_projected_selections_fail_explicitly(x, y, message) -> None:
    with pytest.raises(ProjectedVoronoiError, match=message):
        build_projected_voronoi_field(np.asarray(x), np.asarray(y))


def test_unique_point_limit_fails_explicitly() -> None:
    x, y = _regular_grid()

    with pytest.raises(ProjectedVoronoiError, match="at most 8"):
        build_projected_voronoi_field(x, y, max_unique_points=8)


def test_3d_cell_volumes_match_explicit_voronoi_regions() -> None:
    rng = np.random.default_rng(72)
    points = rng.uniform(-100.0, 100.0, size=(36, 3))
    field = build_voronoi_3d_field(points)

    reference = Voronoi(points)
    expected = np.full(points.shape[0], np.nan, dtype=np.float64)
    for point_index, region_index in enumerate(reference.point_region):
        region = reference.regions[region_index]
        if len(region) > 3 and -1 not in region:
            expected[point_index] = 1.0 / ConvexHull(
                reference.vertices[region]
            ).volume

    # np.unique(..., axis=0) uses lexicographic order, which is also the order
    # produced by this explicit coordinate sort.
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    finite = np.isfinite(expected[order])
    assert np.count_nonzero(finite) == field.finite_cell_count
    np.testing.assert_allclose(
        field.density_per_nm3[finite], expected[order][finite], rtol=1e-9, atol=1e-12
    )


def test_3d_duplicate_multiplicity_scales_density() -> None:
    rng = np.random.default_rng(73)
    points = rng.uniform(-100.0, 100.0, size=(30, 3))
    base = build_voronoi_3d_field(points)
    sorted_order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    duplicate_index = int(sorted_order[np.flatnonzero(base.density_per_nm3 > 0.0)[0]])
    with_duplicate = build_voronoi_3d_field(
        np.concatenate((points, points[[duplicate_index]]), axis=0)
    )
    matches = np.all(np.isclose(base.points_nm, points[duplicate_index]), axis=1)
    duplicate_matches = np.all(
        np.isclose(with_duplicate.points_nm, points[duplicate_index]), axis=1
    )
    assert np.count_nonzero(matches) == 1
    assert np.count_nonzero(duplicate_matches) == 1
    base_value = base.density_per_nm3[matches][0]
    duplicate_value = with_duplicate.density_per_nm3[duplicate_matches][0]
    assert base_value > 0.0
    assert np.isclose(duplicate_value, 2.0 * base_value)


def test_3d_field_samples_bounded_volume() -> None:
    rng = np.random.default_rng(74)
    points = rng.uniform(-10.0, 10.0, size=(40, 3))
    field = build_voronoi_3d_field(points)
    volume = field.sample((-12.0, 12.0, -12.0, 12.0, -12.0, 12.0), (9, 8, 7))
    assert volume.shape == (9, 8, 7)
    assert volume.dtype == np.float32
    assert np.all(np.isfinite(volume))
    assert np.all(volume >= 0.0)


def test_3d_degenerate_selection_fails_explicitly() -> None:
    points = np.column_stack(
        (
            np.linspace(0.0, 10.0, 8),
            np.linspace(0.0, 20.0, 8),
            np.linspace(0.0, 30.0, 8),
        )
    )
    with pytest.raises(Voronoi3DError, match="span all three dimensions"):
        build_voronoi_3d_field(points)
