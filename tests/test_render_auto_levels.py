"""Automatic display-level behavior for localization render rasters."""

from types import MethodType, SimpleNamespace

import numpy as np

from minflux_viewer.ui.render_window import (
    RenderWindow,
    localization_render_auto_levels,
)


def _sparse_smoothed_histogram() -> np.ndarray:
    """640k-pixel field with 135 isolated half-pixel Gaussian footprints."""
    image = np.zeros((800, 800), dtype=np.float32)
    footprint = np.asarray(
        [0.63]
        + [0.084] * 4
        + [0.011] * 4
        + [0.0002] * 8
        + [1.0e-7] * 8,
        dtype=np.float32,
    )
    image.ravel()[: 135 * footprint.size] = np.tile(footprint, 135)
    return image


def test_sparse_render_default_brightens_smoothed_localizations():
    image = _sparse_smoothed_histogram()

    levels = localization_render_auto_levels(image)

    assert levels is not None
    lo, hi = levels
    assert lo == 0.0
    assert np.isclose(hi, 0.084, rtol=1e-5)


def test_sparse_constant_histogram_keeps_signal_at_white_point():
    image = np.zeros((800, 800), dtype=np.float32)
    image.ravel()[:135] = 1.0

    assert localization_render_auto_levels(image) == (0.0, 1.0)


def test_passive_localization_auto_dispatches_to_sparse_render_levels():
    image = _sparse_smoothed_histogram()
    state = SimpleNamespace(
        _bc_auto_threshold=0,
        _render_mode="localizations",
    )

    levels = RenderWindow._compute_render_auto_levels(state, image)

    assert levels == localization_render_auto_levels(image)


def test_explicit_auto_threshold_remains_imagej_authoritative():
    image = _sparse_smoothed_histogram()
    state = SimpleNamespace(
        _bc_auto_threshold=5000,
        _render_mode="localizations",
    )
    state._compute_auto_levels = MethodType(RenderWindow._compute_auto_levels, state)

    _lo, hi = RenderWindow._compute_render_auto_levels(state, image)

    assert hi > 0.5


def test_fully_occupied_density_field_uses_robust_low_and_high_levels():
    image = np.linspace(0.1, 10.0, 10_000, dtype=np.float32).reshape(100, 100)

    lo, hi = localization_render_auto_levels(image)

    assert np.isclose(lo, np.percentile(image, 1.0))
    assert np.isclose(hi, np.percentile(image, 95.0))


def test_imagej_auto_failure_resets_escalation_counter():
    image = _sparse_smoothed_histogram()
    state = SimpleNamespace(_bc_auto_threshold=625)

    levels = RenderWindow._compute_auto_levels(
        state, image, advance_auto_threshold=True
    )

    assert levels == (0.0, float(image.max()))
    assert state._bc_auto_threshold == 0


def test_imagej_auto_restarts_after_failed_escalation():
    image = _sparse_smoothed_histogram()
    state = SimpleNamespace(_bc_auto_threshold=625)
    RenderWindow._compute_auto_levels(state, image, advance_auto_threshold=True)

    RenderWindow._compute_auto_levels(state, image, advance_auto_threshold=True)

    assert state._bc_auto_threshold == 5000
