"""Toolbar icon normalization for light and dark Qt palettes."""

from __future__ import annotations

import sys

from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

from minflux_viewer import resource_path
from minflux_viewer.ui.main_window import _adaptive_toolbar_pixmap


def _app() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def test_monochrome_toolbar_icon_drops_white_matte_and_tints_linework():
    _app()
    pixmap = _adaptive_toolbar_pixmap(str(resource_path("icons", "angle.png")))
    assert pixmap is not None
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)

    # The white/near-white matte becomes (near-)transparent. angle.png's corner
    # is pure white → alpha 0; some source PNGs (e.g. color.png) have a slightly
    # off-white corner that maps to a small residual alpha, so assert "matte
    # substantially removed" rather than exactly 0.
    assert image.pixelColor(0, 0).alpha() < 40
    visible = [
        image.pixelColor(x, y)
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).alpha() > 0
    ]
    assert visible
    assert not all(color.red() == color.green() == color.blue() == 255 for color in visible)


def test_colored_toolbar_icon_keeps_artwork_but_drops_white_matte():
    _app()
    pixmap = _adaptive_toolbar_pixmap(str(resource_path("icons", "color.png")))
    assert pixmap is not None
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)

    # color.png's corner is near-white (~249) → the matte drops to a small
    # residual alpha, not exactly 0. What matters is it is no longer opaque.
    assert image.pixelColor(0, 0).alpha() < 40
    assert any(
        image.pixelColor(x, y).alpha() > 0
        and max(
            image.pixelColor(x, y).red(),
            image.pixelColor(x, y).green(),
            image.pixelColor(x, y).blue(),
        ) > 100
        for y in range(image.height())
        for x in range(image.width())
    )
