"""Toned pure-colour LUT ramp (render view): black -> colour -> white so pixel
values are perceptually distinguishable, gray stays plain black -> white."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from minflux_viewer.ui.render_window import pure_color_ramp

_LUM = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def test_saturated_colour_spans_black_to_white():
    n = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    r = pure_color_ramp(n, (1.0, 0.0, 0.0))
    assert np.allclose(r[0], [0, 0, 0])          # low  -> black
    assert np.allclose(r[1], [1, 0, 0])          # mid  -> full hue
    assert np.allclose(r[2], [1, 1, 1])          # high -> white
    lum = r @ _LUM
    assert lum[0] < lum[1] < lum[2]              # lightness strictly increases
    assert np.isclose(lum[2], 1.0)              # reaches full-white lightness (unlike norm*hue)


def test_gray_is_plain_black_to_white_no_overshoot():
    n = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    g = pure_color_ramp(n, (1.0, 1.0, 1.0))
    assert np.allclose(g[0], [0, 0, 0])
    assert np.allclose(g[1], [0.5, 0.5, 0.5])    # mid = mid-gray, NOT white
    assert np.allclose(g[2], [1, 1, 1])


def test_monotonic_lightness_over_full_range():
    n = np.linspace(0, 1, 64, dtype=np.float32)
    for color in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)):
        lum = pure_color_ramp(n, color) @ _LUM
        assert np.all(np.diff(lum) >= -1e-6)     # non-decreasing lightness


def test_shape_preserved_and_clamped():
    img = pure_color_ramp(np.zeros((4, 5), dtype=np.float32), (0.0, 1.0, 0.0))
    assert img.shape == (4, 5, 3)
    out = pure_color_ramp(np.array([-0.5, 1.5]), (1.0, 0.0, 0.0))
    assert out.min() >= 0.0 and out.max() <= 1.0


# -- white background (inverted / subtractive ramp) ---------------------------
def test_white_bg_saturated_colour_white_to_colour_to_black():
    n = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    r = pure_color_ramp(n, (1.0, 0.0, 0.0), white_bg=True)
    assert np.allclose(r[0], [1, 1, 1])          # no signal -> white (page)
    assert np.allclose(r[1], [1, 0, 0])          # mid       -> hue
    assert np.allclose(r[2], [0, 0, 0])          # high      -> black
    lum = r @ _LUM
    assert lum[0] > lum[1] > lum[2]              # signal darkens the page


def test_white_bg_gray_is_inverted_grayscale():
    n = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    g = pure_color_ramp(n, (1.0, 1.0, 1.0), white_bg=True)
    assert np.allclose(g[0], [1, 1, 1])
    assert np.allclose(g[1], [0.5, 0.5, 0.5])
    assert np.allclose(g[2], [0, 0, 0])          # classic inverted-LUT render
