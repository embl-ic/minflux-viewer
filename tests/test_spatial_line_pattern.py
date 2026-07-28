"""Directed centerline straightening and repeating-pattern analysis."""

from __future__ import annotations

import numpy as np
import pytest

from minflux_viewer.analysis.spatial_line_pattern import (
    analyze_spatial_line_pattern,
    fit_centerline,
    project_to_centerline,
)


def test_projection_respects_line_direction_and_signed_side():
    points = np.array([[20.0, 10.0], [80.0, -10.0]])
    forward = fit_centerline(
        [[0.0, 0.0], [100.0, 0.0]],
        model="polyline",
        interpolation_step_nm=1.0,
    )
    indices, s_forward, u_forward = project_to_centerline(
        points,
        forward,
        half_width_nm=20.0,
    )
    assert np.array_equal(indices, [0, 1])
    assert np.allclose(s_forward, [20.0, 80.0], atol=0.2)
    assert np.allclose(u_forward, [10.0, -10.0], atol=0.2)

    reverse = fit_centerline(
        [[100.0, 0.0], [0.0, 0.0]],
        model="polyline",
        interpolation_step_nm=1.0,
    )
    _indices, s_reverse, u_reverse = project_to_centerline(
        points,
        reverse,
        half_width_nm=20.0,
    )
    assert np.allclose(s_reverse, [80.0, 20.0], atol=0.2)
    assert np.allclose(u_reverse, [-10.0, 10.0], atol=0.2)


def test_cubic_centerline_is_endpoint_anchored_and_uniform():
    source = np.array(
        [
            [0.0, 0.0],
            [30.0, 20.0],
            [60.0, -20.0],
            [100.0, 0.0],
        ]
    )
    centerline = fit_centerline(
        source,
        model="cubic",
        interpolation_step_nm=2.0,
        smoothing_nm=4.0,
    )
    assert np.allclose(centerline.points_nm[0], source[0], atol=1.0e-5)
    assert np.allclose(centerline.points_nm[-1], source[-1], atol=1.0e-5)
    spacing = np.diff(centerline.arc_nm)
    assert np.all(spacing > 0.0)
    assert np.all(spacing[:-1] <= 2.001)
    assert np.allclose(np.linalg.norm(centerline.tangent, axis=1), 1.0)
    assert np.allclose(
        np.einsum("ij,ij->i", centerline.tangent, centerline.normal),
        0.0,
        atol=1.0e-12,
    )


def test_repeating_clusters_and_alternating_sides_have_distinct_periods():
    rng = np.random.default_rng(9)
    cluster_s = 40.0 + 80.0 * np.arange(12)
    points = []
    for index, center in enumerate(cluster_s):
        n = 50
        points.append(
            np.column_stack(
                [
                    rng.normal(center, 2.0, n),
                    rng.normal(12.0 if index % 2 == 0 else -12.0, 1.5, n),
                ]
            )
        )
    localizations = np.vstack(points)
    result = analyze_spatial_line_pattern(
        localizations,
        [[0.0, 0.0], [960.0, 0.0]],
        centerline_model="polyline",
        interpolation_step_nm=1.0,
        half_width_nm=25.0,
        profile_bin_nm=4.0,
        transverse_bin_nm=2.0,
        profile_smoothing_nm=5.0,
        background_scale_nm=160.0,
        min_period_nm=50.0,
        max_period_nm=200.0,
        peak_prominence=0.10,
        peak_order=5,
    )

    assert result.n_used == localizations.shape[0]
    assert int(result.straightened_counts.sum()) == localizations.shape[0]
    assert abs(result.density_fft_period_nm - 80.0) < 5.0
    assert abs(result.density_autocorr_period_nm - 80.0) < 6.0
    assert abs(result.transverse_fft_period_nm - 160.0) < 12.0
    assert result.density_fft_snr > 2.0
    assert result.transverse_fft_snr > 2.0
    assert result.peak_positions_nm.size >= 10
    assert abs(np.median(result.peak_spacing_by_order_nm[0]) - 80.0) < 5.0


def test_flip_side_swaps_positive_and_negative_profiles():
    points = np.array(
        [
            [20.0, 5.0],
            [40.0, 6.0],
            [60.0, -5.0],
        ]
    )
    normal = analyze_spatial_line_pattern(
        points,
        [[0.0, 0.0], [100.0, 0.0]],
        centerline_model="polyline",
        half_width_nm=10.0,
        profile_bin_nm=100.0,
    )
    flipped = analyze_spatial_line_pattern(
        points,
        [[0.0, 0.0], [100.0, 0.0]],
        centerline_model="polyline",
        half_width_nm=10.0,
        profile_bin_nm=100.0,
        flip_side=True,
    )
    assert np.array_equal(normal.positive_profile, flipped.negative_profile)
    assert np.array_equal(normal.negative_profile, flipped.positive_profile)
    assert np.allclose(normal.point_u_nm, -flipped.point_u_nm)


def test_excessive_straightened_grid_is_rejected():
    with pytest.raises(ValueError, match="Straightened map would need"):
        analyze_spatial_line_pattern(
            np.zeros((0, 2)),
            [[0.0, 0.0], [10_000.0, 0.0]],
            centerline_model="polyline",
            interpolation_step_nm=10.0,
            half_width_nm=10_000.0,
            profile_bin_nm=0.1,
            transverse_bin_nm=0.1,
        )
