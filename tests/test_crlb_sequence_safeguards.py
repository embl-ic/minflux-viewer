"""Sequence selection and geometry limits of the centred CRLB model."""

from math import isfinite, isnan, sqrt
from types import SimpleNamespace

import pytest

from minflux_viewer.core.mfx_sequence import (
    PATTERN_SCALE_NM,
    PatternGeometry,
    assumed_pattern_geometry,
    crlb_geometry_for_dataset,
    detect_photon_iterations,
    pattern_geometry,
)


def _sequence(*steps):
    return {"Itr": [{"patGeoFactor": scale, "Mode": {"pattern": pattern}}
                    for scale, pattern in steps]}


@pytest.mark.parametrize("xy_scale,z_scale", [(0.11, 0.21), (0.21, 0.11)])
def test_final_iteration_is_selected_at_each_axes_own_minimum(xy_scale, z_scale):
    sequence = _sequence((0.8, "square"), (0.8, "zline2"),
                         (xy_scale, "square"), (z_scale, "zline2"))
    info = detect_photon_iterations(sequence=sequence, dims=3)
    assert info.lateral_iter == 2
    assert info.axial_iter == 3
    assert info.photon_iters == [2, 3]
    assert info.detail["min_lateral_geo_factor"] == xy_scale
    assert info.detail["min_axial_geo_factor"] == z_scale

    dataset = SimpleNamespace(metadata={"mfx_sequence": sequence},
                              prop=SimpleNamespace(num_itr=4, num_dim=3))
    _, xy, z = crlb_geometry_for_dataset(dataset)
    assert xy.L_nm == pytest.approx(xy_scale * PATTERN_SCALE_NM)
    assert z.L_nm == pytest.approx(z_scale * PATTERN_SCALE_NM)
    assert xy.supported and z.supported


def test_pooling_keeps_all_minimum_scale_iterations_but_crlb_uses_last_each():
    sequence = _sequence((0.8, "square"), (0.11, "square"),
                         (0.21, "zline2"), (0.11, "square"), (0.21, "zline2"))
    info = detect_photon_iterations(sequence=sequence, dims=3)
    assert info.photon_iters == [1, 2, 3, 4]
    assert (info.lateral_iter, info.axial_iter) == (3, 4)


def test_equal_scale_standard_sequence_and_volume_pooling_stay_unchanged():
    info = detect_photon_iterations(sequence=_sequence(
        (0.8, "square"), (0.8, "zline2"), (0.11, "square"), (0.11, "zline2")), dims=3)
    assert info.photon_iters == [2, 3]
    assert (info.lateral_iter, info.axial_iter) == (2, 3)
    info = detect_photon_iterations(sequence=_sequence(
        (0.8, "octahedron"), (0.28, "octahedron")), dims=3)
    assert info.photon_iters == [1]
    assert info.lateral_iter == info.axial_iter == 1


def test_axial_only_sequence_is_not_relabelled_as_lateral():
    info = detect_photon_iterations(sequence=_sequence((0.11, "zline2")), dims=3)
    assert info.photon_iters == [0]
    assert info.lateral_iter is None
    assert info.axial_iter == 0


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scales_do_not_win_final_iteration_selection(invalid):
    info = detect_photon_iterations(sequence=_sequence(
        (0.11, "square"), (invalid, "square")))
    assert info.photon_iters == [0]
    assert info.lateral_iter == 0


def test_multi_radius_zline_is_reported_but_not_used_as_a_two_point_model():
    sequence = _sequence((0.11, "square"), ([0.1, 0.5], "zline"))
    geometry = pattern_geometry(sequence, 1)
    assert geometry.L_nm == pytest.approx(0.1 * PATTERN_SCALE_NM)
    assert geometry.multi_radius
    assert not geometry.supported
    assert "multi-radius" in geometry.unsupported_reason
    assert all(isnan(value) for value in geometry.geometry_factors)
    assert not geometry.donut_envelope
    dataset = SimpleNamespace(metadata={"mfx_sequence": sequence},
                              prop=SimpleNamespace(num_itr=2, num_dim=3))
    _, xy, z = crlb_geometry_for_dataset(dataset)
    assert xy.supported
    assert not z.supported
    assert z.iteration == 1


def test_repeated_equal_radius_is_not_a_multi_radius_pattern():
    geometry = pattern_geometry(_sequence(([0.11, 0.11], "zline")), 0)
    assert geometry.supported
    assert not geometry.multi_radius
    assert geometry.geometry_factors[1] == 1.0


@pytest.mark.parametrize("pattern", ["custom_pattern", "zline3", ""])
def test_unsupported_recorded_pattern_never_assumes_standard_geometry(pattern):
    geometry = pattern_geometry(_sequence((0.11, pattern)), 0)
    assert not geometry.supported
    assert "unknown acquisition pattern" in geometry.unsupported_reason
    assert all(isnan(value) for value in geometry.geometry_factors)


def test_unrecorded_geometry_is_still_an_explicit_supported_assumption():
    geometry = assumed_pattern_geometry(4)
    assert geometry.supported
    assert geometry.unsupported_reason == ""
    assert "assumed" in geometry.source
    assert geometry.geometry_factors == pytest.approx((sqrt(2), 1))
    assert geometry.donut_envelope


@pytest.mark.parametrize("pattern", ["triangle", "square", "hexagon", "zline2", "octahedron"])
def test_supported_recorded_patterns_keep_their_geometry(pattern):
    geometry = pattern_geometry(_sequence((0.11, pattern)), 0)
    assert geometry.supported
    assert geometry.unsupported_reason == ""
    assert any(isfinite(value) for value in geometry.geometry_factors)


@pytest.mark.parametrize("extent", [float("nan"), float("inf"), 0.0, -1.0])
def test_direct_invalid_geometry_is_explicitly_unsupported(extent):
    geometry = PatternGeometry(L_nm=extent, pattern="square")
    assert not geometry.supported
    assert "finite and positive" in geometry.unsupported_reason
