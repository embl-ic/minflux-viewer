"""Reject impossible convolution requests before grids or FFTs allocate."""

import numpy as np
import pytest

from minflux_viewer.analysis import conv_segmentation as cs


def test_extreme_valid_ring_settings_fail_before_meshgrid(monkeypatch):
    monkeypatch.setattr(cs.np, "meshgrid", lambda *a, **kw: pytest.fail("allocated grid"))
    with pytest.raises(ValueError, match="memory|budget"):
        cs.build_kernel("ring", {"diameter_nm": 5000., "rim_nm": 20.}, .5)


def test_full_preflight_includes_fft_padding_and_input_memory():
    xy = np.array([[0., 0.], [400., 500.], [10., 20.]])
    estimate = cs.preflight_convolution(xy, "ring", {}, 4., budget_bytes=10**9)
    assert estimate.fft_shape[0] > estimate.image_shape[0]
    assert estimate.fft_shape[1] > estimate.image_shape[1]
    assert estimate.bytes_required > 8 * np.prod(estimate.image_shape)
    with pytest.raises(ValueError, match="budget"):
        cs.preflight_convolution(xy, "ring", {}, 4., budget_bytes=1024)


@pytest.mark.parametrize("pixel", [0., -1., float("nan"), float("inf")])
def test_invalid_pixels_fail_before_allocation(pixel):
    with pytest.raises(ValueError):
        cs.build_kernel("gaussian", {}, pixel)
    with pytest.raises(ValueError):
        cs.histogram_maps(np.empty((0, 2)), pixel_size_nm=pixel)


def test_histogram_and_direct_fft_are_guarded(monkeypatch):
    monkeypatch.setattr(cs.np, "arange", lambda *a, **kw: pytest.fail("allocated edges"))
    with pytest.raises(ValueError, match="budget"):
        cs.render_histogram_2d(np.array([0., 1e9]), np.array([0., 1e9]), .5)
    import scipy.signal
    monkeypatch.setattr(scipy.signal, "fftconvolve", lambda *a, **kw: pytest.fail("started FFT"))
    with pytest.raises(ValueError, match="budget"):
        cs.convolve_response(np.zeros((50, 50)), np.ones((40, 40)), budget_bytes=1024)


def test_capped_nms_is_the_same_prefix_and_has_cancellation_checkpoints():
    response = np.random.default_rng(17).uniform(size=(50, 50))
    full, values = cs.non_max_suppression(response, .1, 3)
    capped, cvalues = cs.non_max_suppression(response, .1, 3, max_peaks=10)
    np.testing.assert_array_equal(capped, full[:10])
    np.testing.assert_array_equal(cvalues, values[:10])
    def stop():
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        cs.non_max_suppression(response, .1, 3, checkpoint=stop)


@pytest.mark.parametrize("key", list(cs.GEOMETRY_MODELS))
def test_registered_extents_bound_generated_grids(key):
    for pixel in (.5, 4., 200.):
        predicted = cs._kernel_shape(key, {}, pixel)
        actual = cs.build_kernel(key, {}, pixel).shape
        assert all(a <= bound for a, bound in zip(actual, predicted))


def test_histogram_preview_has_its_own_bound_without_relaxing_fft_budget():
    xy = np.array([[0., 0.], [4800., 4800.], [100., 150.]])
    required = cs.preflight_histogram(xy, 2.)
    assert 300 * 1024**2 < required < cs.DEFAULT_MEMORY_BUDGET
    estimate = cs.estimate_convolution(xy, "disk", {}, 2.)
    assert estimate.bytes_required > cs.DEFAULT_MEMORY_BUDGET
    with pytest.raises(ValueError, match="budget"):
        cs.preflight_convolution(xy, "disk", {}, 2.)
    with pytest.raises(ValueError, match="budget"):
        cs.preflight_histogram(xy, 2., budget_bytes=1024)


def test_data_only_maps_match_exact_histogram_and_never_start_fft(monkeypatch):
    monkeypatch.setattr(cs, "convolve_response", lambda *a, **kw: pytest.fail("started FFT"))
    xy = np.array([[0., 0.], [40., 50.], [10., 20.], [10., 20.]])
    maps = cs.histogram_maps(xy, pixel_size_nm=2.)
    histogram, xedge, yedge = cs.render_histogram_2d(xy[:, 0], xy[:, 1], 2.)
    np.testing.assert_array_equal(maps["histogram"], histogram)
    np.testing.assert_array_equal(maps["xedge"], xedge)
    np.testing.assert_array_equal(maps["yedge"], yedge)
    assert not {"response_map", "centers", "fft_provenance"}.intersection(maps)


def test_empty_data_preview_and_oversized_histogram_are_bounded():
    maps = cs.histogram_maps(np.empty((0, 2)), pixel_size_nm=4.)
    assert maps["histogram"].shape == (0, 0)
    with pytest.raises(ValueError, match="budget"):
        cs.histogram_maps(np.array([[0., 0.], [1e9, 1e9]]), pixel_size_nm=1.)
