"""XYZ numerical semantics, the region's edge margin and resource refusal."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import fftconvolve

from minflux_viewer.analysis import convolution_3d as conv
from minflux_viewer.core.inspection_raster import histogram_preview, peak_preview


def request():
    return conv.VolumeRequest(((-100., 100.), (-100., 100.), (-160., 160.)),
                               (4., 5., 2.), "gaussian", width_nm=6.)


def points():
    rng = np.random.default_rng(38)
    return np.vstack([rng.normal((30., -40., z), 2., size=(300, 3)) for z in (-80., 80.)])


def blob(center, n=300, sigma=4., seed=0):
    return np.random.default_rng(seed).normal(center, sigma, size=(n, 3))


def test_exact_anisotropic_xyz_grid_separates_overlapping_xy_objects():
    maps = conv.volume_maps(points(), request())
    centers, values = conv.volume_detections(maps, min_response=.5, min_distance_nm=100.)
    assert maps["response"].shape == (50, 40, 160)
    assert len(centers) == 2
    np.testing.assert_allclose(np.sort(centers[:, 2]), [-80., 80.], atol=2.)
    np.testing.assert_allclose(centers[:, :2], np.tile((30., -40.), (2, 1)), atol=2.5)
    assert np.max(values) == 1.
    np.testing.assert_array_equal(conv.volume_detections(maps, min_response=.5,
                                  min_distance_nm=100., max_detections=1)[0], centers[:1])


def test_cpu_convolution_is_scipy_on_the_region_plus_its_margin():
    req = request()
    estimate = conv.preflight_volume(req)
    assert estimate.halo_voxels == (6, 5, 12)          # 4 sigma / voxel, rounded up
    assert estimate.kernel_shape == tuple(2 * h + 1 for h in estimate.halo_voxels)
    assert estimate.image_shape == tuple(n + 2 * h for n, h in
                                         zip(estimate.region_shape, estimate.halo_voxels))
    maps = conv.volume_maps(points(), req)
    full_edges = maps["full_edges"]
    histogram, _ = np.histogramdd(points(), bins=full_edges)
    values = np.sqrt(histogram) / np.sqrt(histogram).max()
    expected = fftconvolve(values, conv.build_volume_kernel(req, estimate.kernel_shape), mode="same")
    expected /= expected[maps["inner"]].max()
    np.testing.assert_allclose(maps["full_response"], expected, rtol=1e-12, atol=1e-12)
    assert maps["provenance"]["backend"] == "cpu-scipy"
    for edge, full, h in zip(maps["edges"], full_edges, estimate.halo_voxels):
        np.testing.assert_array_equal(edge, full[h:len(full) - h])


@pytest.mark.parametrize("model", list(conv.MODELS))
def test_kernel_has_declared_xyz_shape_unit_l2_and_zero_mean(model):
    req = replace(request(), model=model)
    estimate = conv.preflight_volume(req)
    kernel = conv.build_volume_kernel(req, estimate.kernel_shape)
    assert kernel.shape == estimate.kernel_shape and kernel.ndim == 3
    assert np.linalg.norm(kernel.ravel()) == pytest.approx(1.)
    assert abs(kernel.sum()) < 1e-12
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1, ::-1], atol=1e-15)


@pytest.mark.parametrize("change", [dict(voxel_xyz=(0., 4., 4.)), dict(width_nm=-1.),
    dict(bounds_xyz=((0., 0.), (0., 1.), (0., 1.))), dict(model="missing"),
    dict(budget_bytes=1), dict(max_voxels=10)])
def test_refuse_invalid_or_oversized_volume_before_histogram_or_kernel(monkeypatch, change):
    monkeypatch.setattr(conv.np, "histogramdd", lambda *a, **kw: pytest.fail("allocated histogram"))
    monkeypatch.setattr(conv, "build_volume_kernel", lambda *a: pytest.fail("allocated kernel"))
    with pytest.raises(ValueError):
        conv.volume_maps(points(), replace(request(), **change))


def test_coordinates_beyond_the_margin_do_not_contribute():
    xyz = points()
    base = conv.volume_maps(xyz, request())
    augmented = conv.volume_maps(np.vstack((xyz, [[10000., 10000., 10000.]])), request())
    np.testing.assert_array_equal(base["full_response"], augmented["full_response"])
    assert augmented["selected_count"] == len(xyz)
    with pytest.raises(ValueError, match="three localizations"):
        conv.volume_maps(np.ones((2, 3)), request())


def test_an_object_on_the_region_edge_scores_like_one_in_the_middle():
    """Its localizations beyond the edge are read from the margin, not dropped."""
    req = conv.VolumeRequest(((-100., 100.), (-100., 100.), (-50., 50.)), (4., 4., 4.),
                              "gaussian", width_nm=6.)
    xyz = np.vstack([blob((0., 0., 0.), seed=1), blob((98., 0., 0.), seed=2)])
    centers, scores = conv.volume_detections(conv.volume_maps(xyz, req), min_response=.5,
                                             min_distance_nm=50.)
    assert len(centers) == 2
    edge = np.argmax(centers[:, 0])
    np.testing.assert_allclose(centers[edge], (98., 0., 0.), atol=3.)
    assert scores[edge] == pytest.approx(scores[1 - edge], rel=.1)


def test_an_object_centred_outside_the_region_is_not_reported():
    req = conv.VolumeRequest(((-100., 100.), (-100., 100.), (-50., 50.)), (4., 4., 4.),
                              "gaussian", width_nm=6.)
    xyz = np.vstack([blob((0., 0., 0.), seed=1), blob((108., 0., 0.), n=600, seed=3)])
    centers, _ = conv.volume_detections(conv.volume_maps(xyz, req), min_response=.2,
                                        min_distance_nm=30.)
    assert len(centers) == 1
    np.testing.assert_allclose(centers[0], (0., 0., 0.), atol=3.)


def test_simulated_npc_two_ring_template_finds_pore_centre_in_xyz():
    from minflux_viewer.core.simulate import simulate_localizations

    xyz, _, _ = simulate_localizations("npc", n_points=1000, locs_per_trace=4., precision_nm=1., dim=3,
                                       seed=38, params=dict(size_nm=200., n_pores=1, diameter_nm=100.,
                                                            ring_sep_nm=50., field_curvature=0., local_tilt_deg=0.))
    center = xyz.mean(axis=0)
    req = conv.VolumeRequest(tuple((float(c-100), float(c+100)) for c in center),
                              (4., 4., 2.), "npc", width_nm=6.)
    maps = conv.volume_maps(xyz, req)
    centers, _ = conv.volume_detections(maps, min_response=.7, min_distance_nm=100.)
    assert len(centers) == 1
    np.testing.assert_allclose(centers[0], center, atol=4.)


@pytest.mark.parametrize("kwargs", [dict(min_response=float("nan")), dict(min_response=-.1),
                                     dict(min_distance_nm=0.), dict(max_detections=0)])
def test_invalid_detection_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        conv.volume_detections({}, **kwargs)


def test_cancellation_propagates_between_volume_stages():
    stages = []
    def report(text):
        stages.append(text)
        if len(stages) == 2:
            raise InterruptedError("stop")
    with pytest.raises(InterruptedError):
        conv.volume_maps(points(), request(), report=report)
    assert len(stages) == 2


def test_display_only_rasters_are_bounded_and_do_not_mutate_scientific_values():
    xyz = points()
    maps = conv.volume_maps(xyz, request())
    before = maps["response"].copy()
    image = histogram_preview(xyz[:, :2], (-100., 100., -100., 100.), (320, 240))
    assert image.shape == (320, 240)
    projected = peak_preview(maps["response"].max(axis=2), max_dim=16)
    assert max(projected.shape) <= 16
    assert projected.max() == 1.
    np.testing.assert_array_equal(before, maps["response"])
    with pytest.raises(ValueError):
        histogram_preview(xyz[:, :2], (0., 1., 0., 1.), (100000, 100000))
