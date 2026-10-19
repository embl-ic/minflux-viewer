"""XYZ numerical semantics, resource refusal and optional real CUDA parity."""

import os
import sys
from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import fftconvolve

from minflux_viewer.analysis import convolution_3d as conv
from minflux_viewer.core import cuda_fft, cuda_fft_worker
from minflux_viewer.core.inspection_raster import histogram_preview, peak_preview


def request():
    return conv.VolumeRequest(((-100., 100.), (-100., 100.), (-160., 160.)),
                               (4., 5., 2.), "gaussian", width_nm=6.)


def points():
    rng = np.random.default_rng(38)
    return np.vstack([rng.normal((30., -40., z), 2., size=(300, 3)) for z in (-80., 80.)])


def test_exact_anisotropic_xyz_grid_separates_overlapping_xy_objects():
    maps = conv.volume_maps(points(), request(), cuda=None)
    centers, values = conv.volume_detections(maps, min_response=.5, min_distance_nm=100.)
    assert maps["response"].shape == (50, 40, 160)
    assert len(centers) == 2
    np.testing.assert_allclose(np.sort(centers[:, 2]), [-80., 80.], atol=2.)
    np.testing.assert_allclose(centers[:, :2], np.tile((30., -40.), (2, 1)), atol=2.5)
    assert np.max(values) == 1.
    np.testing.assert_array_equal(conv.volume_detections(maps, min_response=.5,
                                  min_distance_nm=100., max_detections=1)[0], centers[:1])


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
        conv.volume_maps(points(), replace(request(), **change), cuda=None)


def test_region_is_explicit_and_outside_coordinates_do_not_contribute():
    xyz = points()
    base = conv.volume_maps(xyz, request(), cuda=None)
    augmented = conv.volume_maps(np.vstack((xyz, [[10000., 10000., 10000.]])), request(), cuda=None)
    np.testing.assert_array_equal(base["response"], augmented["response"])
    assert augmented["selected_count"] == len(xyz)
    with pytest.raises(ValueError, match="three localizations"):
        conv.volume_maps(np.ones((2, 3)), request(), cuda=None)


def test_missing_runtime_is_not_a_cpu_fallback(monkeypatch):
    monkeypatch.setattr(conv, "fftconvolve", lambda *a, **kw: pytest.fail("CPU fallback"))
    with pytest.raises(ValueError, match="absolute path"):
        conv.volume_maps(points(), request(), cuda=cuda_fft.CudaFftSettings(""))


def test_simulated_npc_two_ring_template_finds_pore_centre_in_xyz():
    from minflux_viewer.core.simulate import simulate_localizations

    xyz, _, _ = simulate_localizations("npc", n_points=1000, locs_per_trace=4., precision_nm=1., dim=3,
                                       seed=38, params=dict(size_nm=200., n_pores=1, diameter_nm=100.,
                                                            ring_sep_nm=50., field_curvature=0., local_tilt_deg=0.))
    center = xyz.mean(axis=0)
    req = conv.VolumeRequest(tuple((float(c-100), float(c+100)) for c in center),
                              (4., 4., 2.), "npc", width_nm=6.)
    maps = conv.volume_maps(xyz, req, cuda=None)
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
        conv.volume_maps(points(), request(), cuda=None, report=report)
    assert len(stages) == 2


def test_display_only_rasters_are_bounded_and_do_not_mutate_scientific_values():
    xyz = points()
    maps = conv.volume_maps(xyz, request(), cuda=None)
    before = maps["response"].copy()
    image = histogram_preview(xyz[:, :2], (-100., 100., -100., 100.), (320, 240))
    assert image.shape == (320, 240)
    projected = peak_preview(maps["response"].max(axis=2), max_dim=16)
    assert max(projected.shape) <= 16
    assert projected.max() == 1.
    np.testing.assert_array_equal(before, maps["response"])
    with pytest.raises(ValueError):
        histogram_preview(xyz[:, :2], (0., 1., 0., 1.), (100000, 100000))


@pytest.mark.parametrize("shape,kshape", [((7, 9, 11), (3, 4, 2)), ((4, 6, 8), (8, 3, 10)),
                                        ((1, 7, 1), (2, 4, 3)), ((1, 1, 1), (1, 1, 1))])
def test_actual_cuda_xyz_fft_matches_signed_asymmetric_cpu_convolution(shape, kshape):
    python = os.environ.get("MFV_CUDA_PYTHON")
    if not python:
        pytest.skip("Set MFV_CUDA_PYTHON for real CUDA validation")
    rng = np.random.default_rng(98)
    image, kernel = rng.normal(size=shape), rng.normal(size=kshape)
    image.flat[0] += 20.
    result, provenance = cuda_fft.cuda_fftconvolve(image, kernel, cuda_fft.CudaFftSettings(python))
    np.testing.assert_allclose(result, fftconvolve(image, kernel, mode="same"), rtol=1e-11, atol=1e-11)
    assert provenance["axes"] == [0, 1, 2]
    assert provenance["fft_shape"] == list(cuda_fft_worker.fft_shape(shape, kshape))
    assert "cupy" not in sys.modules


def test_actual_cuda_volume_and_3d_centres_match_cpu_reference():
    python = os.environ.get("MFV_CUDA_PYTHON")
    if not python:
        pytest.skip("Set MFV_CUDA_PYTHON for real CUDA validation")
    cpu = conv.volume_maps(points(), request(), cuda=None)
    gpu = conv.volume_maps(points(), request(), cuda=cuda_fft.CudaFftSettings(python))
    np.testing.assert_allclose(cpu["response"], gpu["response"], rtol=1e-11, atol=1e-11)
    left, right = conv.volume_detections(cpu), conv.volume_detections(gpu)
    np.testing.assert_array_equal(left[0], right[0])
    np.testing.assert_allclose(left[1], right[1], rtol=1e-11, atol=1e-11)
